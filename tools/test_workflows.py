"""CPU checks for every bundled graph: wiring, node schemas, settings, layout, and that
setup.sh downloads exactly the files the graphs load. python tools/test_workflows.py"""
import importlib.util
import json
from pathlib import Path
import re
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sys.modules.setdefault("folder_paths", types.ModuleType("folder_paths"))
builder = load("builder", ROOT / "tools/build_workflow.py")
lc = load("lc", ROOT / "custom_nodes/comfyui-longcat-longform/__init__.py")
SETUP = (ROOT / "setup.sh").read_text(encoding="utf-8")

# kijai/ComfyUI-WanVideoWrapper INPUT_TYPES / define_schema at builder.WANVIDEO_REF,
# in declaration order: (widgets, sockets). Re-transcribe when the pin moves.
WANVIDEO = {
    "WanVideoModelLoader": (
        ["model", "base_precision", "quantization", "load_device", "attention_mode",
         "rms_norm_function"],
        ["compile_args", "block_swap_args", "lora", "vram_management_args", "extra_model",
         "fantasytalking_model", "multitalk_model", "fantasyportrait_model"]),
    "WanVideoLoraSelect": (["lora", "strength", "low_mem_load", "merge_loras"],
                           ["prev_lora", "blocks"]),
    "WanVideoBlockSwap": (["blocks_to_swap", "offload_img_emb", "offload_txt_emb",
                           "use_non_blocking", "vace_blocks_to_swap", "prefetch_blocks",
                           "block_swap_debug"], []),
    "WanVideoVAELoader": (["model_name", "precision", "use_cpu_cache", "verbose"],
                          ["compile_args"]),
    "WanVideoTextEncodeCached": (["model_name", "precision", "positive_prompt",
                                  "negative_prompt", "quantization", "use_disk_cache",
                                  "device"], ["extender_args"]),
    "WanVideoEncode": (["enable_vae_tiling", "tile_x", "tile_y", "tile_stride_x",
                        "tile_stride_y", "noise_aug_strength", "latent_strength"],
                       ["vae", "image", "mask"]),
    "WanVideoDecode": (["enable_vae_tiling", "tile_x", "tile_y", "tile_stride_x",
                        "tile_stride_y", "normalization"], ["vae", "samples"]),
    "WhisperModelLoader": (["model", "base_precision", "load_device"], []),
    "LongCatAvatarWhisperEmbeds": (["normalize_loudness", "num_frames", "fps",
                                    "audio_scale", "audio_cfg_scale", "multi_audio_type"],
                                   ["whisper_model", "audio_1", "audio_2", "audio_3",
                                    "audio_4", "ref_target_masks"]),
    "WanVideoLongCatAvatarExtendEmbeds": (["num_frames", "overlap", "frames_processed",
                                           "if_not_enough_audio", "ref_frame_index",
                                           "ref_mask_frame_range"],
                                          ["prev_latents", "audio_embeds", "ref_latent",
                                           "samples", "prev_images", "vae"]),
    "WanVideoSchedulerv2": (["scheduler", "steps", "shift", "start_step", "end_step",
                             "enhance_hf"], ["sigmas"]),
    # seed also gets the frontend's control_after_generate widget.
    "WanVideoSamplerv2": (["cfg", "seed", "control_after_generate", "force_offload",
                           "add_noise_to_samples"],
                          ["model", "image_embeds", "scheduler", "text_embeds", "samples",
                           "extra_args"]),
}


class WorkflowTests(unittest.TestCase):
    def each(self):
        for w in builder.WORKFLOWS:
            wf = builder.build(w["variant"], w["batch"], w["precision"])
            yield w, wf, {n["id"]: n for n in wf["nodes"]}

    def sources(self, wf, nodes):
        links = {}
        for lid, src, slot, dst, port, kind in wf["links"]:
            out, inp = nodes[src]["outputs"][slot], nodes[dst]["inputs"][port]
            self.assertEqual(out["type"], kind)
            self.assertEqual(inp["type"], kind)
            self.assertEqual(inp["link"], lid)
            self.assertIn(lid, out["links"])
            links[(dst, inp["name"])] = (src, out["name"])
        return links

    def test_shipped_files_match_builder(self):
        shipped = {p.name for p in (ROOT / "workflows").glob("*.json")}
        self.assertEqual(shipped, {w["path"] for w in builder.WORKFLOWS})
        self.assertEqual(len({w["id"] for w in builder.WORKFLOWS}), len(builder.WORKFLOWS))
        for w, wf, _ in self.each():
            with self.subTest(workflow=w["path"]):
                path = ROOT / "workflows" / w["path"]
                self.assertEqual(json.loads(path.read_text(encoding="utf-8")), wf)

    def test_samples_ship_with_the_repo(self):
        shipped = {p.name for p in (ROOT / "samples").iterdir()}
        self.assertEqual(shipped, {builder.SAMPLE_PORTRAIT, builder.SAMPLE_VOICEOVER})
        self.assertIn('"$SCRIPT_DIR/samples/"*', SETUP)

    def test_current_workflows_are_not_in_setups_cleanup_list(self):
        # setup.sh deletes files matching these hashes; a current workflow must never
        # match, or it would be deleted and reinstalled on every boot.
        import hashlib
        retired = set(re.findall(r"^([0-9a-f]{64}) \S+\.json$", SETUP, re.M))
        self.assertGreater(len(retired), 0)
        for w, wf, _ in self.each():
            text = json.dumps(wf, indent=1, ensure_ascii=False) + "\n"  # as on the pod
            self.assertNotIn(hashlib.sha256(text.encode()).hexdigest(), retired, w["path"])

    def test_wanvideo_nodes_match_pinned_schema(self):
        for w, wf, nodes in self.each():
            for node in nodes.values():
                schema = WANVIDEO.get(node["type"])
                if schema is None:
                    continue
                with self.subTest(workflow=w["path"], node=node["type"]):
                    widgets, sockets = schema
                    self.assertEqual(len(node["widgets_values"]), len(widgets))
                    self.assertEqual([i["name"] for i in node["inputs"] if "widget" not in i],
                                     sockets)
                    for inp in node["inputs"]:
                        if "widget" in inp:
                            self.assertIn(inp["name"], widgets)
                    self.assertEqual(node["properties"]["ver"], builder.WANVIDEO_REF)

    def test_own_nodes_match_their_schema(self):
        for w, wf, nodes in self.each():
            for node in nodes.values():
                cls = lc.NODE_CLASS_MAPPINGS.get(node["type"])
                if cls is None:
                    continue
                with self.subTest(workflow=w["path"], node=node["type"]):
                    schema = cls.INPUT_TYPES()
                    inputs = {**schema.get("required", {}), **schema.get("optional", {})}
                    self.assertEqual(tuple(o["type"] for o in node["outputs"]), cls.RETURN_TYPES)
                    self.assertEqual(tuple(o["name"] for o in node["outputs"]), cls.RETURN_NAMES)
                    for inp in node["inputs"]:
                        self.assertIn(inp["name"], inputs)
                    widgets = [k for k, v in inputs.items()
                               if (isinstance(v[0], list) or v[0] in ("INT", "FLOAT",
                                                                      "BOOLEAN", "STRING"))
                               and not (len(v) > 1 and v[1].get("forceInput"))]
                    self.assertEqual(len(widgets), len(node["widgets_values"]))

    def test_wiring(self):
        for w, wf, nodes in self.each():
            with self.subTest(workflow=w["path"]):
                links = self.sources(wf, nodes)
                v = builder.VARIANTS[w["variant"]]
                expected = {
                    (55, "audio_1"): (20, "whisper_audio"),
                    (55, "num_frames"): (20, "whisper_frames"),
                    (56, "audio_embeds"): (55, "multitalk_embeds"),
                    (56, "prev_images"): (53, "prev_images"),
                    (56, "ref_latent"): (53, "ref_latent"),
                    (56, "prev_latents"): (51, "samples"),
                    (56, "vae"): (43, "vae"),
                    (53, "portrait"): (50, "IMAGE"),
                    (53, "portrait_latent"): (51, "samples"),
                    (53, "chunk_index"): (20, "chunk_index"),
                    (53, "opens_on_portrait"): (20, "opens_on_portrait"),
                    (57, "positive_prompt"): (30, "prompt"),
                    (60, "text_embeds"): (57, "text_embeds"),
                    (60, "image_embeds"): (56, "image_embeds"),
                    (47, "sigmas"): (46, "sigmas"),
                    (40, "lora"): (41, "lora"),
                    (40, "block_swap_args"): (42, "block_swap_args"),
                    (70, "images"): (61, "images"),
                    (70, "chunk_audio"): (20, "audio_chunk"),
                    (70, "trim_start"): (20, "trim_start"),
                    (70, "keep_frames"): (20, "keep_frames"),
                    (70, "chunk_index"): (20, "chunk_index"),
                    (70, "total_chunks"): (20, "total_chunks"),
                    (70, "original_audio"): (11, "audio" if w["batch"] else "AUDIO"),
                }
                for name in ("num_frames", "overlap", "frames_processed", "ref_frame_index",
                             "ref_mask_frame_range"):
                    expected[(56, name)] = (20, name)
                if w["batch"]:
                    for dst in (20, 53, 70):
                        expected[(dst, "session")] = (11, "name")
                    expected[(70, "filename")] = (11, "filename")
                else:
                    expected[(70, "session")] = (21, "name")
                    expected[(70, "filename")] = (21, "filename")
                for dest, source in expected.items():
                    self.assertEqual(links.get(dest), source, dest)
                # Every required socket of every node is connected.
                for node in nodes.values():
                    for inp in node["inputs"]:
                        if "widget" not in inp and "shape" not in inp:
                            self.assertIsNotNone(inp["link"], (node["type"], inp["name"]))
                # One portrait: no second image anywhere in the graph.
                self.assertEqual(sum(n["type"] == "LoadImage" for n in nodes.values()), 1)

    def test_variant_settings(self):
        for w, wf, nodes in self.each():
            v = builder.settings(w["variant"], w["precision"])
            by_type = {}
            for n in nodes.values():
                by_type.setdefault(n["type"], []).append(n)
            with self.subTest(workflow=w["path"]):
                # Return to portrait, with a restart from the portrait every 40-60 s,
                # in a real pause of 0.4 s or more.
                self.assertEqual(nodes[20]["widgets_values"],
                                 [v["anchor"], builder.RESTART_MIN, builder.RESTART_MAX,
                                  builder.MIN_PAUSE])
                self.assertEqual((builder.RESTART_MIN, builder.RESTART_MAX,
                                  builder.MIN_PAUSE), (40.0, 60.0, 0.4))
                self.assertEqual((builder.RESTART_MIN, builder.RESTART_MAX, builder.MIN_PAUSE),
                                 (lc.RESTART_MIN, lc.RESTART_MAX, lc.MIN_PAUSE))
                self.assertEqual(v["anchor"], lc.ANCHOR_WINDOW_END)
                self.assertEqual(nodes[13]["widgets_values"][0], v["canvas"])
                self.assertEqual(nodes[14]["widgets_values"][0], v["canvas"])
                self.assertEqual(v["canvas"] % 16, 0)
                for scale in by_type["ImageScale"]:
                    self.assertEqual(scale["widgets_values"][3], "center")
                # Upstream v1.5 distilled settings: bf16, 8 steps, cfg 1, audio cfg 1.
                # The precision is in the file name, the title and the output name.
                model = builder.MODEL_FP8 if w["precision"] == "fp8" else builder.MODEL
                quantization = "fp8_e4m3fn" if w["precision"] == "fp8" else "disabled"
                self.assertEqual(nodes[40]["widgets_values"][:3], [model, "bf16", quantization])
                self.assertTrue(w["path"].startswith(f"longcat_{w['precision']}_"))
                self.assertIn(w["precision"], nodes[40]["title"].lower())
                self.assertTrue(v["title"].startswith(f"LongCat {w['precision'].upper()} - "))
                # Names say what they are: precision, single or batch audio, canvas.
                kind = "batch_audio" if w["batch"] else "single_audio"
                self.assertTrue(w["path"].endswith(
                    f"_{kind}_{v['canvas']}x{v['canvas']}.json"), w["path"])
                self.assertNotIn("portrait", w["path"])
                self.assertTrue(v["suffix"].startswith(f"_{w['precision']}_{v['canvas']}"))
                self.assertEqual(nodes[41]["widgets_values"][:2], [builder.LORA, 1.0])
                self.assertEqual(nodes[47]["widgets_values"][:2], ["euler", 8])
                self.assertEqual(nodes[60]["widgets_values"][0], 1.0)
                self.assertEqual(nodes[55]["widgets_values"][0], False)  # Split normalises
                self.assertEqual(nodes[55]["widgets_values"][2:5], [25.0, 1.0, 1.0])
                # Block Swap: on for FP8 (smaller GPUs), bypassed for BF16 (80 GB GPUs).
                self.assertEqual(nodes[42]["mode"],
                                 0 if w["precision"] == "fp8" else builder.BYPASS)
                self.assertEqual(nodes[42]["widgets_values"][0], 20)
                suffix = (nodes[11]["widgets_values"][1] if w["batch"]
                          else nodes[21]["widgets_values"][2])
                self.assertEqual(suffix, v["suffix"])
                self.assertEqual(len(by_type["LoadImage"]), 1)
                # Every workflow opens on the shipped samples.
                for image in by_type["LoadImage"]:
                    self.assertEqual(image["widgets_values"][0], builder.SAMPLE_PORTRAIT)
                for audio in by_type.get("LoadAudio", []):
                    self.assertEqual(audio["widgets_values"][0], builder.SAMPLE_VOICEOVER)
                self.assertEqual(len(by_type.get("LoadAudio", [])), 0 if w["batch"] else 1)
        suffixes = [builder.settings(v, p)["suffix"]
                    for v in builder.VARIANTS for p in builder.PRECISIONS]
        self.assertEqual(len(set(suffixes)), len(suffixes), "sessions must not collide")
        self.assertEqual(len(builder.WORKFLOWS), len(builder.ORDER) * len(builder.PRECISIONS))
        # Single and batch, at 640 and 960, for BF16 and FP8.
        self.assertEqual(sorted((w["precision"], w["batch"], builder.VARIANTS[w["variant"]]["canvas"])
                                for w in builder.WORKFLOWS),
                         sorted((p, b, c) for p in ("bf16", "fp8") for b in (False, True)
                                for c in (640, 960)))

    def test_models_match_setup_downloads(self):
        downloads = re.findall(r'^download_kj "([^"]+)" (\w+)', SETUP, re.M)
        self.assertEqual(sorted((Path(p).name, d) for p, d in downloads),
                         sorted((n, d) for n, d, _ in builder.MODEL_FILES))
        self.assertEqual(sorted(p for p, _ in downloads),
                         sorted(p for _, _, p in builder.MODEL_FILES))
        self.assertIn(f'HF_REPO="Kijai/WanVideo_comfy"', SETUP)
        self.assertIn(builder.WANVIDEO_REF, SETUP)
        for w, wf, nodes in self.each():
            for node in nodes.values():
                for m in node["properties"].get("models", []):
                    self.assertIn(m["name"], node["widgets_values"])
                    self.assertEqual(m["url"], builder.MODEL_URL[m["name"]])
            if w["precision"] == "fp8":
                self.assertNotIn("models", nodes[40]["properties"])
        # The FP8 model is not downloaded: setup builds it from the BF16 file.
        self.assertIn(f'FP8_MODEL="$COMFYUI_PATH/models/diffusion_models/{builder.MODEL_FP8}"',
                      SETUP)
        self.assertIn(f'FP8_SOURCE="$COMFYUI_PATH/models/diffusion_models/{builder.MODEL}"',
                      SETUP)
        self.assertIn('"$SCRIPT_DIR/tools/make_fp8.py"', SETUP)

    def test_layout_is_spaced_grouped_and_expanded(self):
        title = 40  # LiteGraph draws the title bar above pos
        for w, wf, nodes in self.each():
            with self.subTest(workflow=w["path"]):
                boxes = {nid: (n["pos"][0], n["pos"][1] - title, n["pos"][0] + n["size"][0],
                               n["pos"][1] + n["size"][1]) for nid, n in nodes.items()}
                ids = sorted(boxes)
                for i, a in enumerate(ids):
                    self.assertNotEqual(nodes[a]["flags"].get("collapsed"), True)
                    for b in ids[i + 1:]:
                        ax0, ay0, ax1, ay1 = boxes[a]
                        bx0, by0, bx1, by1 = boxes[b]
                        self.assertFalse(ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1,
                                         f"{nodes[a]['type']} overlaps {nodes[b]['type']}")
                self.assertEqual([g["title"][:2] for g in wf["groups"]],
                                 [f"{i}." for i in range(1, 8)])
                for g in wf["groups"]:
                    x, y, gw, gh = g["bounding"]
                    inside = [n for n in nodes.values() if n["type"] != "MarkdownNote"
                              and x <= n["pos"][0] < x + gw and y <= n["pos"][1] < y + gh]
                    self.assertTrue(inside, g["title"])
                self.assertEqual(wf["last_link_id"], len(wf["links"]))


class StandaloneNotesTests(unittest.TestCase):
    """Every workflow stands alone: its notes and titles never mention another model,
    template or workflow."""

    def test_no_cross_references(self):
        pattern = re.compile(r"\b(FlashHead|Qwen|H3|MiniMax|InfiniteTalk|Lite)\b|\bworkflows? [0-9]|"
                             r"other (workflow|template)|avatar templates", re.I)
        for path in sorted((ROOT / "workflows").glob("*.json")):
            wf = json.loads(path.read_text(encoding="utf-8"))
            for node in wf["nodes"]:
                texts = [node.get("title") or ""]
                if node["type"] == "MarkdownNote":
                    texts += [v for v in node.get("widgets_values", []) if isinstance(v, str)]
                for text in texts:
                    hit = pattern.search(text)
                    self.assertIsNone(hit, f"{path.name} node {node['id']}: "
                                      f"{hit and hit.group(0)}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
