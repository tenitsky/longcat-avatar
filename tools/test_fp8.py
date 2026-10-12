"""The FP8 model setup.sh builds: which tensors become float8, that the values are
torch's own cast, and that the file is valid safetensors. python tools/test_fp8.py"""
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("make_fp8", ROOT / "tools/make_fp8.py")
make_fp8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_fp8)

try:
    from safetensors.torch import load_file
except ImportError:
    load_file = None


def write_bf16(path, tensors, metadata=None):
    """A minimal safetensors writer, so the test does not depend on the library."""
    header, blobs, offset = {}, [], 0
    for name, t in tensors.items():
        raw = t.to(torch.bfloat16).contiguous().view(torch.uint8).numpy().tobytes()
        header[name] = {"dtype": "BF16", "shape": list(t.shape),
                        "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    if metadata:
        header["__metadata__"] = metadata
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))


def read(path):
    header, base = make_fp8.read_header(path)
    data = path.read_bytes()
    dtypes = {"BF16": torch.bfloat16, "F8_E4M3": torch.float8_e4m3fn}
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        out[name] = torch.frombuffer(bytearray(data[base + start:base + end]),
                                     dtype=dtypes[info["dtype"]]).reshape(info["shape"])
    return header, out


class Fp8Tests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.dir = Path(directory.name)
        g = torch.Generator().manual_seed(0)
        # Names from Kijai's LongCat-Avatar-15_bf16.safetensors.
        self.tensors = {
            "blocks.0.self_attn.q.weight": torch.randn(8, 16, generator=g) * 0.05,
            "blocks.0.self_attn.q.bias": torch.randn(8, generator=g),
            "blocks.0.self_attn.norm_q.weight": torch.randn(4, generator=g),
            "blocks.0.ffn.w1.weight": torch.randn(16, 8, generator=g),
            "blocks.0.modulation.1.weight": torch.randn(12, 4, generator=g),
            "blocks.0.audio_modulation.1.weight": torch.randn(12, 4, generator=g),
            "blocks.0.audio_cross_attn.kv_linear.weight": torch.randn(8, 4, generator=g),
            "head.head.weight": torch.randn(4, 8, generator=g),
            "multitalk_audio_proj.proj1.weight": torch.randn(4, 8, generator=g),
            "patch_embedding.weight": torch.randn(4, 2, 1, 2, 2, generator=g),
            "text_embedding.0.weight": torch.randn(8, 8, generator=g),
            "time_embedding.mlp.0.weight": torch.randn(8, 4, generator=g),
        }
        # Out of float8 range: must saturate, not become NaN.
        self.tensors["blocks.0.ffn.w1.weight"][0, :2] = torch.tensor([1000.0, -1000.0])
        self.source = self.dir / "LongCat-Avatar-15_bf16.safetensors"
        write_bf16(self.source, self.tensors,
                   {"model_type": "LongCat-Avatar-1.5", "format": "pt"})
        self.dest = self.dir / "LongCat-Avatar-15_fp8_e4m3fn.safetensors"
        self.converted, self.total = make_fp8.convert(str(self.source), str(self.dest))

    def test_wrapper_rule_picks_the_large_linear_weights(self):
        header, _ = read(self.dest)
        fp8 = {k for k, v in header.items() if k != "__metadata__" and v["dtype"] == "F8_E4M3"}
        self.assertEqual(fp8, {"blocks.0.self_attn.q.weight", "blocks.0.ffn.w1.weight",
                               "blocks.0.audio_cross_attn.kv_linear.weight",
                               "head.head.weight"})
        self.assertEqual((self.converted, self.total), (4, len(self.tensors)))
        self.assertEqual(header["__metadata__"]["model_type"], "LongCat-Avatar-1.5")
        self.assertEqual(header["__metadata__"]["fp8_source"], self.source.name)

    def test_values_are_torchs_cast_and_kept_tensors_are_untouched(self):
        header, out = read(self.dest)
        for name, original in self.tensors.items():
            bf16 = original.to(torch.bfloat16)
            if header[name]["dtype"] == "F8_E4M3":
                expected = bf16.clamp(-448, 448).to(torch.float8_e4m3fn)
                self.assertTrue(torch.equal(out[name].view(torch.uint8),
                                            expected.view(torch.uint8)), name)
            else:
                self.assertTrue(torch.equal(out[name], bf16), name)
        w1 = out["blocks.0.ffn.w1.weight"].float()
        self.assertFalse(torch.isnan(w1).any())
        self.assertEqual(w1[0, :2].tolist(), [448.0, -448.0])

    def test_file_layout(self):
        size = struct.unpack("<Q", self.dest.read_bytes()[:8])[0]
        self.assertEqual(size % 8, 0)
        header, _ = read(self.dest)
        end = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
        self.assertEqual(8 + size + end, self.dest.stat().st_size)

    @unittest.skipIf(load_file is None, "safetensors not installed")
    def test_loads_with_safetensors(self):
        loaded = load_file(str(self.dest))
        self.assertEqual(loaded["blocks.0.ffn.w1.weight"].dtype, torch.float8_e4m3fn)
        self.assertEqual(loaded["blocks.0.self_attn.q.bias"].dtype, torch.bfloat16)
        self.assertEqual(set(loaded), set(self.tensors))


if __name__ == "__main__":
    unittest.main(verbosity=2)
