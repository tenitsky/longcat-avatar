#!/usr/bin/env python3
"""
Make the FP8 LongCat-Avatar model from Kijai's bf16 file. setup.sh runs this once.

    python tools/make_fp8.py SOURCE_bf16.safetensors DEST_fp8.safetensors

The result is what kijai/ComfyUI-WanVideoWrapper builds in memory when the bf16 file
is loaded with quantization fp8_e4m3fn: the large linear weights become float8_e4m3fn
and every tensor the wrapper keeps at base precision stays bf16. Built once, the FP8
workflows load ~17 GB instead of reading and converting 32 GB on every load.

Tensors are converted one at a time and streamed to disk, so memory use stays at the
size of the largest tensor (~90 MB). PyTorch is imported only when a tensor needs it.
"""
import json
import os
import struct
import sys

# WanVideoWrapper's params_to_keep plus keep_fp32 in nodes_model_loading.py at the
# pinned commit: a weight whose name contains any of these stays at base precision.
KEEP = ("norm", "bias", "time_in", "patch_embedding", "time_", "img_emb", "modulation",
        "text_embedding", "adapter", "add", "ref_conv", "audio_proj", "motion_encoder",
        "condition_embedding")
FP8_MAX = 448.0  # largest finite float8_e4m3fn; torch's cast turns overflow into NaN
PROGRESS_STEPS = 10


def read_header(path):
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
    return header, 8 + size


def to_fp8(name, info):
    return (info["dtype"] == "BF16" and name.endswith(".weight")
            and len(info["shape"]) >= 2 and not any(k in name for k in KEEP))


def convert(source, dest):
    header, base = read_header(source)
    metadata = header.pop("__metadata__", None) or {}
    names = sorted(header, key=lambda k: header[k]["data_offsets"][0])

    out, offset = {}, 0
    for name in names:
        info = header[name]
        size = info["data_offsets"][1] - info["data_offsets"][0]
        dtype = info["dtype"]
        if to_fp8(name, info):
            dtype, size = "F8_E4M3", size // 2
        out[name] = {"dtype": dtype, "shape": info["shape"], "data_offsets": [offset, offset + size]}
        offset += size
    out["__metadata__"] = {**{k: str(v) for k, v in metadata.items()},
                           "fp8_source": os.path.basename(source),
                           "fp8_method": "WanVideoWrapper fp8_e4m3fn weight cast"}
    encoded = json.dumps(out, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)

    torch = None
    converted = 0
    step = max(1, len(names) // PROGRESS_STEPS)
    with open(source, "rb") as src, open(dest, "wb") as dst:
        dst.write(struct.pack("<Q", len(encoded)))
        dst.write(encoded)
        for i, name in enumerate(names, 1):
            info = header[name]
            start, end = info["data_offsets"]
            src.seek(base + start)
            raw = src.read(end - start)
            if len(raw) != end - start:
                raise ValueError(f"{source} is truncated at {name}")
            if out[name]["dtype"] == "F8_E4M3":
                if torch is None:
                    import torch
                weight = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16)
                fp8 = weight.clamp_(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
                raw = fp8.view(torch.uint8).numpy().tobytes()
                converted += 1
            dst.write(raw)
            if i % step == 0 or i == len(names):
                print(f"  fp8: {i}/{len(names)} tensors", flush=True)
    return converted, len(names)


def main(argv):
    if len(argv) != 3:
        sys.exit(f"usage: {argv[0]} SOURCE_bf16.safetensors DEST_fp8.safetensors")
    converted, total = convert(argv[1], argv[2])
    print(f"  fp8: {converted} of {total} tensors stored as float8_e4m3fn, the rest kept.")


if __name__ == "__main__":
    main(sys.argv)
