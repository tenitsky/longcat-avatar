#!/usr/bin/env python3
"""
Generate the LongCat-Video-Avatar workflow JSON files.

The graph is written from code rather than exported from the UI so every link, slot
and widget value is declared in one readable place and can be checked before a pod
ever loads it. Re-run after editing:

    python tools/build_workflow.py

Opening the result in ComfyUI and saving it again is fine - the UI adds sizes and
cosmetic properties, but the wiring stays the same.

Eight workflows ship: single audio and batch folder, at 640 x 640 and 960 x 960,
each in a BF16 and an FP8 version (PRECISIONS), which differ only in the model file
loaded and Block Swap. All use the return-to-portrait anchor and restarts from the
portrait at real pauses.

Model nodes come from kijai/ComfyUI-WanVideoWrapper (LongCat support lives there;
ComfyUI has no native LongCat video nodes). Their widget order follows that pack's
INPUT_TYPES at WANVIDEO_REF; tests/test_workflows.py checks it.

Layout: seven numbered stages from left to right, each a coloured group with a short
note underneath, and a "Start here" note on the far left. Nothing is collapsed.
"""

import json
import os

WORKFLOW_DIR = os.path.join(os.path.dirname(__file__), "..", "workflows")

WANVIDEO_REF = "088128b224242e110d3906c6750e9a3a348a659b"
WANVIDEO_CNR = "comfyui-wanvideowrapper"
KJ = "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/"
MODEL = "LongCat-Avatar-15_bf16.safetensors"
LORA = "LongCat-Avatar-15_dmd_distill_lora_rank128_bf16.safetensors"
TEXT_ENCODER = "umt5-xxl-enc-bf16.safetensors"
VAE = "Wan2_1_VAE_bf16.safetensors"
WHISPER = "whisper_large_v3_encoder_fp16.safetensors"
# (filename, ComfyUI models/ folder, path inside Kijai/WanVideo_comfy)
MODEL_FILES = [
    (MODEL, "diffusion_models", "LongCat/" + MODEL),
    (LORA, "loras", "LongCat/" + LORA),
    (TEXT_ENCODER, "text_encoders", TEXT_ENCODER),
    (VAE, "vae", VAE),
    (WHISPER, "audio_encoders", "HuMo/" + WHISPER),
]
MODEL_URL = {name: KJ + path for name, _, path in MODEL_FILES}
# Shipped in samples/ and copied into ComfyUI/input/ at setup: every workflow opens on
# them, so a first run needs no uploads.
SAMPLE_PORTRAIT = "sample_portrait.png"
SAMPLE_VOICEOVER = "sample_voiceover.mp3"
# Built from MODEL by tools/make_fp8.py at setup, not downloaded.
MODEL_FP8 = "LongCat-Avatar-15_fp8_e4m3fn.safetensors"

ANCHOR_IDENTITY = "identity (default)"
ANCHOR_WINDOW_END = "return to portrait"
# Restarts from the portrait against drift: one in every 20-40 s, in the middle of a
# real pause (0.4 s+ of near-silence) planned from the whole voiceover.
RESTART_MIN, RESTART_MAX = 20.0, 40.0
MIN_PAUSE = 0.4
WINDOW = 93

BYPASS = 4  # LiteGraph node mode: pass the input straight through

VARIANTS = {
    "640": dict(
        title="640 x 640",
        summary="640 x 640 (LongCat's 480p tier). " + "Every 3.2 s chunk continues from the one before, with the portrait as "
                "identity reference on each window's last frame, pulling every chunk "
                "back toward the portrait pose. Every 20-40 s, in the middle of a real "
                "pause in the speech (mouth shut), the render starts again from the "
                "portrait itself: a full refresh against the noise that otherwise grows "
                "from chunk to chunk.",
        canvas=640, anchor=ANCHOR_WINDOW_END),
    "960": dict(
        title="960 x 960 (high quality)",
        summary="960 x 960 (LongCat's 720p tier), high quality: about 2.3 times the "
                "rendering time of 640. " + "Every 3.2 s chunk continues from the one before, with the portrait as "
                "identity reference on each window's last frame, pulling every chunk "
                "back toward the portrait pose. Every 20-40 s, in the middle of a real "
                "pause in the speech (mouth shut), the render starts again from the "
                "portrait itself: a full refresh against the noise that otherwise grows "
                "from chunk to chunk.",
        canvas=960, anchor=ANCHOR_WINDOW_END),
}

# BF16 targets 80 GB GPUs, so Block Swap starts bypassed. FP8 targets smaller GPUs, so
# it starts on at 20 of 48 blocks.
PRECISIONS = {
    "bf16": dict(
        label="BF16", model=MODEL, quantization="disabled", block_swap=False,
        loader_title="LongCat-Avatar 1.5 (bf16, full precision)",
        summary="**BF16 model** (31.7 GB): full precision, the reference quality. Needs "
                "an 80 GB GPU, or Block Swap below that."),
    "fp8": dict(
        label="FP8", model=MODEL_FP8, quantization="fp8_e4m3fn", block_swap=True,
        loader_title="LongCat-Avatar 1.5 (fp8 e4m3fn)",
        summary="**FP8 model** (~17 GB), made from the BF16 file at setup: about half the "
                "VRAM for the model weights, at a small cost in precision. **Block Swap "
                "is on** (20 of 48 blocks) for 32-48 GB GPUs. Compare with the BF16 "
                "version of this workflow."),
}

# Numbered so the Workflows sidebar lists them in the order to use them: one voiceover,
# then a folder of them, at 640 and then 960.
ORDER = [
    ("640", False, "1_single_audio_640x640"),
    ("640", True, "2_batch_audio_640x640"),
    ("960", False, "3_single_audio_960x960"),
    ("960", True, "4_batch_audio_960x960"),
]
IDS = {
    "bf16": ["5d2e9f71-8a4c-4b3e-9f06-1c7b3e5a8d42", "ea9c44d6-5ded-4a95-bd4d-86675cdf87f9",
             "9dfa1065-d7b1-4af0-b2f3-91396d0c227d", "cbb88312-d0c1-4a97-b050-ca21ddf242d5"],
    "fp8": ["cdcc91ba-011a-4221-8ebe-078d7047e496", "589b2b49-fdab-45e7-bd58-818d78e63c71",
            "0c17be5a-9135-4615-8ba6-4b109ff2dbfc", "b2405b71-fd77-497a-a0d2-32c5956c59dc"],
}
WORKFLOWS = [dict(variant=variant, batch=batch, precision=precision,
                  path=f"longcat_{precision}_{name}.json", id=IDS[precision][i])
             for precision in PRECISIONS for i, (variant, batch, name) in enumerate(ORDER)]


def settings(variant, precision):
    """A variant's settings for one precision. The output suffix names both, so BF16
    and FP8 renders of the same voiceover never share (or resume) a session."""
    v, p = VARIANTS[variant], PRECISIONS[precision]
    return {**v, "precision": precision, "model": p["model"],
            "quantization": p["quantization"], "loader_title": p["loader_title"],
            "block_swap": p["block_swap"],
            "title": f"LongCat {p['label']} - {v['title']}",
            "label": p["label"], "size_title": v["title"],
            "summary": f"{v['summary']}\n\n{p['summary']}",
            "suffix": f"_{precision}_{v['canvas']}"}


# ------------------------------------------------------------------------ notes
NOTE_ABOUT = """## About LongCat-Video-Avatar 1.5

Meituan's audio-driven avatar model (MIT licence), built on LongCat-Video. It runs here through **kijai/ComfyUI-WanVideoWrapper**; ComfyUI has no native LongCat video nodes.

It generates **93-frame windows at 25 fps** (3.7 s). Long videos are **video continuation**: each window opens on the last 13 frames of the one before and adds 80 new frames.

**No end frame.** Conditioning frames always sit at the start of a window. Return to Portrait uses the one lever there is: the position of the identity reference inside the window.

**Restarts against drift.** Continuation alone drifts over long videos: small noise in the carried frames grows from chunk to chunk. So once in every 20-40 s, in the middle of a real pause in the speech, a chunk starts again from the portrait as the first one does.

Model card: https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.5"""


def note_inputs(v):
    return f"""## 1. Inputs

**Portrait:** one person facing the camera, face clearly visible. The first frame of the video, and the identity reference for every later chunk. Any size; it is center-cropped to the canvas.

**Canvas** (Width x Height), multiples of 16. LongCat has two tiers:

| Tier | 1:1 | 16:9 | 9:16 |
|---|---|---|---|
| 480p | 640 x 640 | 832 x 480 | 480 x 832 |
| 720p | 960 x 960 | 1280 x 768 | 768 x 1280 |

Larger canvases need more memory and time."""


NOTE_PROMPT = """## 2. Prompt (optional)

**Leave it empty** for a generic talking-head prompt: the person speaks to the camera, with small natural motion, a static camera and an unchanged background. The person and background themselves come from the portrait.

To steer it, type one or two sentences about the portrait, e.g. *A woman with brown hair in a navy blazer, in front of a plain grey wall.* They go in front of the generic prompt. Naming the background can help hold it in place on long videos.

There is no negative prompt: the distilled model runs at cfg 1, which does not use one. The text is encoded once and cached."""


def note_plan(v):
    anchor = {
        ANCHOR_IDENTITY: "**anchor: identity** (upstream default): the portrait is the "
                         "reference at frame index 10 of each window.",
        ANCHOR_WINDOW_END: "**anchor: return to portrait**: the reference sits on each "
                           "window's last frame with no mask, pulling every chunk back "
                           "toward the portrait.",
    }[v["anchor"]]
    return f"""## 3. Chunk plan

The voiceover is cut into **windows of 93 frames** (3.72 s at 25 fps). Chunk 1 keeps all 93; every later chunk keeps **80 new frames** after a 13-frame lead-in taken from the chunk before, so motion runs straight through every join.

**Each run renders the next unfinished chunk**, found from the files on disk. There is no counter to reset: a new voiceover starts at chunk 1 in its own folder, and an interrupted render continues where it stopped. Chunks must render **in order**.

**Restarts against drift.** Before the first chunk, the whole voiceover is scanned for **real pauses**: at least `min_pause_seconds` (0.4 s) of near-silence, mouth shut. Once in every **20-40 s** (`restart_min_seconds` / `restart_max_seconds`), a chunk opens on the **portrait** again, exactly as chunk 1 does, placed in the middle of the longest real pause in that range. With no real pause there, the next one within 20 s is used rather than cutting into speech. It costs a small jump in a silence and brings the face back to the portrait. The plan is listed in the log on the first chunk and saved as `restart_plan.txt` next to the chunks. Set `restart_max_seconds` to 0 to turn restarts off.

{anchor} You can switch the anchor here on any workflow; use a new `name_suffix` when you do.

Whisper gets the window's audio plus 2 s either side, after the whole track is normalised to -23 LUFS, so loudness never jumps between chunks."""


def note_models(v):
    if v["precision"] == "fp8":
        model = ("- **LongCat-Avatar 1.5, FP8** (`" + MODEL_FP8 + "`, ~17 GB). Setup builds "
                 "it once from the BF16 download: the large weights are stored as "
                 "float8_e4m3fn, everything else stays bf16, exactly as the loader's "
                 "`fp8_e4m3fn` option would convert the BF16 file. Computation still runs "
                 "at bf16 base precision, which LongCat requires.")
        low_vram = ("**Block Swap is on** (20 of 48 blocks): part of the model waits in "
                    "system RAM and moves to the GPU as each block runs, which fits 32-48 GB "
                    "GPUs. Still out of memory? Raise `blocks_to_swap`. On an 80 GB+ GPU, "
                    "select it and press Ctrl+B to bypass it for a little more speed.")
    else:
        model = ("- **LongCat-Avatar 1.5, BF16** (`" + MODEL + "`, 31.7 GB): full "
                 "precision. LongCat only runs at bf16 base precision.")
        low_vram = ("**Less than 80 GB of VRAM?** Use the **FP8** version of this "
                    "workflow, and/or select **Block Swap** and press Ctrl+B to enable it "
                    "(20 of 48 blocks).")
    return f"""## 4. Models - loaded automatically

Nothing to set here. Setup downloads every file on first boot (Kijai/WanVideo_comfy).

{model}
- **DMD distill LoRA**: 8 steps, cfg 1.
- **v1.5 Distill Sigmas** reproduces the upstream 8-step schedule. WanVideoWrapper's `longcat_distill_euler` is the v1.0 schedule.
- **umt5-xxl** text encoder, **Wan 2.1 VAE**, **Whisper large-v3** audio encoder.

{low_vram} Attention is `sdpa`; `sageattn` works if SageAttention 2 is installed (1.0.6 does not)."""


NOTE_GUIDE = """## 5. Portrait, opening and audio guidance

- The portrait is cropped to the canvas and encoded once.
- **Opening + Reference** gives chunk 1 the portrait as its single opening frame. Later chunks get the previous chunk's last 13 frames, re-encoded (the v1.5 way), plus a reference latent.
- **Whisper Embeds** turns the window's audio into lip-sync features; **Extend Embeds** combines them with the opening, the reference and its position.

Nothing to set here."""

NOTE_SAMPLING = """## 6. Sampling + decode

Fixed seed, 8 steps, cfg 1 and audio cfg 1, as upstream runs the distilled v1.5 model. Nothing to set here.

Out of memory? Enable Block Swap (stage 4), turn on `enable_vae_tiling` on the decode node, or use the FP8 version of this workflow."""


def note_output(batch):
    where = ("Each file's video is saved as `output/<file name>" if batch
             else "The final video is saved as `output/<audio name>")
    return f"""## 7. Save + stitch

Each chunk is saved to `output/longcat_longform/<name>/chunk_0000.mp4`, ... as it finishes, so you can check one early.

{where}<suffix>.mp4`, stitched with your **original** audio track.

**Resume:** run again; finished chunks are skipped. **Redo from a chunk:** delete that chunk's file **and every later one**, then run; each chunk continues from the one before it. **Render a finished video again:** change `name_suffix`, or delete its folder under `output/longcat_longform/`."""


def note_start(w, v):
    if w["batch"]:
        audio = ("2. **Audio From Folder** - put your voiceovers in `ComfyUI/input/batch_audio/` "
                 "(.wav, .mp3, .flac, .m4a, .aac, .ogg, .opus). Each file becomes its own "
                 "video, in name order. Upload them with Jupyter (port 8888) or "
                 "FileBrowser.")
        test = "4. Run with **batch count 1** to test the first chunk of the first file."
        run = ("5. Run again with a large batch count: roughly the **total seconds of all "
               "files / 3.2**, plus a margin. Each run renders the next unfinished chunk "
               "of the next unfinished file. When every file is done the queue is "
               "cleared, so extra runs cost nothing.")
        more = ("**Add files later:** drop them in the folder and run again; finished "
                "videos are skipped. The same portrait and prompt are used for every "
                "file.")
    else:
        audio = "2. **Load Voiceover** - your speech track, any length."
        test = "4. Run with **batch count 1** to test the first chunk."
        run = ("5. Run again with a batch count of at least **audio seconds / 3.2**, plus "
               "a margin. Each run renders the next unfinished chunk; extra runs are "
               "skipped in milliseconds.")
        more = ("**Next video:** load the new voiceover and run. It starts at chunk 1 in "
                "its own folder; there is nothing to reset.")
    kind = "Batch Audio (one video per voiceover)" if w["batch"] else "Single Audio"
    return f"""# LongCat {v['label']} - {kind} - {v['size_title']}

{v['summary']}

**Red nodes are the ones you set.** Everything else runs on its own. The stages run left to right, 1 to 7, each with a note underneath.

1. **Load Portrait** - one person facing the camera, face clearly visible.
{audio}
3. **Prompt** (optional) - leave empty, or describe the portrait in a sentence.
{test} Check `output/longcat_longform/<name>/chunk_0000.mp4`: lips, face, background.
{run}
6. The final video lands in `output/`, then the queue is cleared.

{more}

A 7-minute voiceover is about 130 chunks. If the batch count box stops at 100, click Run several times: the queue adds up.

Full instructions: https://github.com/tenitsky/longcat-avatar"""


# ------------------------------------------------------------------------ graph
class Graph:
    def __init__(self):
        self.nodes, self.links = [], []
        self._link_id = 0

    def node(self, nid, ntype, inputs=(), outputs=(), widgets=None, title=None,
             models=None, color=None, cnr="comfy-core", mode=0, size=(440, 100)):
        props = {"Node name for S&R": ntype}
        if cnr:
            props["cnr_id"] = cnr
            if cnr == WANVIDEO_CNR:
                props["aux_id"] = "kijai/ComfyUI-WanVideoWrapper"
                props["ver"] = WANVIDEO_REF
        if models:
            props["models"] = [{"name": n, "url": MODEL_URL[n], "directory": d}
                               for n, d in models]
        n = {
            "id": nid, "type": ntype, "pos": [0, 0], "size": list(size),
            "flags": {}, "order": len(self.nodes), "mode": mode,
            # (name, type, is_widget, optional)
            "inputs": [dict({"localized_name": nm, "name": nm, "type": tp, "link": None},
                            **({"widget": {"name": nm}} if wid else {}),
                            **({"shape": 7} if opt else {}))
                       for nm, tp, wid, opt in inputs],
            "outputs": [{"localized_name": nm, "name": nm, "type": tp, "links": []}
                        for nm, tp in outputs],
            "properties": props,
        }
        if title:
            n["title"] = title
        if widgets is not None:
            n["widgets_values"] = widgets
        if color:
            n["color"], n["bgcolor"] = color
        self.nodes.append(n)
        return nid

    def link(self, src, src_slot, dst, dst_name):
        s = next(n for n in self.nodes if n["id"] == src)
        d = next(n for n in self.nodes if n["id"] == dst)
        if isinstance(src_slot, str):
            src_slot = next(i for i, o in enumerate(s["outputs"]) if o["name"] == src_slot)
        out = s["outputs"][src_slot]
        slot = next(i for i, inp in enumerate(d["inputs"]) if inp["name"] == dst_name)
        inp = d["inputs"][slot]
        if inp["link"] is not None:
            raise ValueError(f"{d['type']}.{dst_name} already linked")
        if inp["type"] != out["type"]:
            raise ValueError(f"type mismatch {s['type']}.{out['name']} ({out['type']})"
                             f" -> {d['type']}.{dst_name} ({inp['type']})")
        self._link_id += 1
        lid = self._link_id
        inp["link"] = lid
        out["links"].append(lid)
        self.links.append([lid, src, src_slot, dst, slot, out["type"]])


def md(g, nid, title, text, size):
    g.node(nid, "MarkdownNote", widgets=[text], title=title, cnr=None,
           color=("#222", "#000"), size=size)


# Stage columns: (title, colour, node ids top to bottom, note ids below the group)
COLUMN_W, NODE_W, GAP_X, GAP_Y, TITLE_H, PAD = 520, 460, 140, 50, 120, 30
STAGE_COLORS = {"input": "#a1309b", "prompt": "#b06634", "plan": "#8a8a2c",
                "models": "#3f789e", "guide": "#4a7a4a", "sample": "#5a5a8a",
                "output": "#8a4a4a"}


def lay_out(g, stages, start_notes):
    """Place stages as spaced columns, each a titled group with notes under it."""
    nodes = {n["id"]: n for n in g.nodes}
    groups = []
    x = 0
    y = 0
    for nid in start_notes:
        nodes[nid]["pos"] = [x, y]
        y += nodes[nid]["size"][1] + GAP_Y
    x += COLUMN_W + GAP_X
    for gid, (title, color, ids, notes) in enumerate(stages, 1):
        y = TITLE_H
        for nid in ids:
            nodes[nid]["pos"] = [x + PAD, y]
            y += nodes[nid]["size"][1] + GAP_Y
        bottom = y - GAP_Y + PAD
        groups.append({"id": gid, "title": title,
                       "bounding": [x, 0, COLUMN_W, bottom],
                       "color": STAGE_COLORS[color], "font_size": 26, "flags": {}})
        y = bottom + 60
        for nid in notes:
            nodes[nid]["pos"] = [x, y]
            nodes[nid]["size"][0] = COLUMN_W
            y += nodes[nid]["size"][1] + GAP_Y
        x += COLUMN_W + GAP_X
    return groups


SPLIT_OUTPUTS = [("audio_chunk", "AUDIO"), ("whisper_audio", "AUDIO"),
                 ("whisper_frames", "INT"), ("num_frames", "INT"), ("overlap", "INT"),
                 ("frames_processed", "INT"), ("ref_frame_index", "INT"),
                 ("ref_mask_frame_range", "INT"), ("trim_start", "INT"),
                 ("keep_frames", "INT"), ("total_chunks", "INT"),
                 ("opens_on_portrait", "BOOLEAN"), ("chunk_index", "INT")]


def build(variant="640", batch=False, precision="bf16"):
    v = settings(variant, precision)
    w = next(w for w in WORKFLOWS if w["variant"] == variant and w["batch"] == batch
             and w["precision"] == precision)
    g = Graph()
    INPUT = ("#322", "#533")    # red: things you set
    W, I = True, False           # input is a widget / a plain socket
    side = v["canvas"]
    WV = WANVIDEO_CNR
    LF = "comfyui-longcat-longform"

    def size(h):
        return (NODE_W, h)

    # ---------------------------------------------------------------- notes
    md(g, 1, "Start here", note_start(w, v), (COLUMN_W, 900))
    md(g, 2, "About the model", NOTE_ABOUT, (COLUMN_W, 420))
    md(g, 3, "About stage 1", note_inputs(v), (COLUMN_W, 440))
    md(g, 4, "About stage 2", NOTE_PROMPT, (COLUMN_W, 420))
    md(g, 5, "About stage 3", note_plan(v), (COLUMN_W, 760))
    md(g, 6, "About stage 4", note_models(v), (COLUMN_W, 600))
    md(g, 7, "About stage 5", NOTE_GUIDE, (COLUMN_W, 380))
    md(g, 8, "About stage 6", NOTE_SAMPLING, (COLUMN_W, 260))
    md(g, 9, "About stage 7", note_output(batch), (COLUMN_W, 400))

    # ---------------------------------------------------------------- 1. inputs
    g.node(10, "LoadImage", outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")],
           widgets=[SAMPLE_PORTRAIT, "image"],
           title="Load Portrait (first frame + identity)", color=INPUT, size=size(500))
    if batch:
        g.node(11, "LongCatLongformAudioFolder",
               outputs=[("audio", "AUDIO"), ("name", "STRING"), ("filename", "STRING"),
                        ("file_number", "INT"), ("total_files", "INT")],
               widgets=["batch_audio", v["suffix"], True],
               title="Audio From Folder (one video per file)", color=INPUT, cnr=LF,
               size=size(190))
    else:
        g.node(11, "LoadAudio", outputs=[("AUDIO", "AUDIO")],
               widgets=[SAMPLE_VOICEOVER, None, None],
               title="Load Voiceover", color=INPUT, size=size(150))
        g.node(21, "LongCatLongformAudioName",
               outputs=[("name", "STRING"), ("filename", "STRING")],
               widgets=[".mp4", "", v["suffix"]],
               title="Output name (audio filename + name_suffix)", cnr=LF,
               size=size(130))
    g.node(13, "PrimitiveInt", outputs=[("INT", "INT")], widgets=[side, "fixed"],
           title="Width", size=size(90))
    g.node(14, "PrimitiveInt", outputs=[("INT", "INT")], widgets=[side, "fixed"],
           title="Height", size=size(90))

    # ---------------------------------------------------------------- 2. prompt
    g.node(30, "LongCatLongformPrompt", outputs=[("prompt", "STRING")], widgets=[""],
           title="Prompt (optional - leave empty or describe the portrait)",
           cnr=LF, size=size(200))
    g.node(57, "WanVideoTextEncodeCached",
           inputs=[("extender_args", "WANVIDEOPROMPTEXTENDER_ARGS", I, True),
                   ("positive_prompt", "STRING", W, False)],
           outputs=[("text_embeds", "WANVIDEOTEXTEMBEDS"),
                    ("negative_text_embeds", "WANVIDEOTEXTEMBEDS"),
                    ("positive_prompt", "STRING")],
           # No negative prompt: the distilled model samples at cfg 1, which skips it.
           widgets=[TEXT_ENCODER, "bf16", "", "", "disabled", True, "gpu"],
           title="Text encoder (umt5-xxl, cached)", models=[(TEXT_ENCODER, "text_encoders")],
           cnr=WV, size=size(300))

    # ---------------------------------------------------------------- 3. chunk plan
    g.node(20, "LongCatLongformSplit",
           inputs=[("audio", "AUDIO", I, False)]
           + ([("session", "STRING", I, True)] if batch else []),
           outputs=SPLIT_OUTPUTS, widgets=[v["anchor"], RESTART_MIN, RESTART_MAX, MIN_PAUSE],
           title="Split (next unfinished window)", cnr=LF, size=size(400))

    # ---------------------------------------------------------------- 4. models
    g.node(42, "WanVideoBlockSwap", outputs=[("block_swap_args", "BLOCKSWAPARGS")],
           widgets=[20, False, False, False, 0, 1, False],
           title=("Block Swap - ON (20 of 48 blocks); Ctrl+B to bypass on 80 GB+"
                  if v["block_swap"] else "Block Swap - Ctrl+B to enable below 80 GB VRAM"),
           cnr=WV, mode=0 if v["block_swap"] else BYPASS, size=size(250))
    g.node(41, "WanVideoLoraSelect",
           inputs=[("prev_lora", "WANVIDLORA", I, True),
                   ("blocks", "SELECTEDBLOCKS", I, True)],
           outputs=[("lora", "WANVIDLORA")], widgets=[LORA, 1.0, False, False],
           title="DMD distill LoRA (8 steps)", models=[(LORA, "loras")], cnr=WV,
           size=size(170))
    g.node(40, "WanVideoModelLoader",
           inputs=[("compile_args", "WANCOMPILEARGS", I, True),
                   ("block_swap_args", "BLOCKSWAPARGS", I, True),
                   ("lora", "WANVIDLORA", I, True),
                   ("vram_management_args", "VRAM_MANAGEMENTARGS", I, True),
                   ("extra_model", "VACEPATH", I, True),
                   ("fantasytalking_model", "FANTASYTALKINGMODEL", I, True),
                   ("multitalk_model", "MULTITALKMODEL", I, True),
                   ("fantasyportrait_model", "FANTASYPORTRAITMODEL", I, True)],
           outputs=[("model", "WANVIDEOMODEL")],
           widgets=[v["model"], "bf16", v["quantization"], "offload_device", "sdpa",
                    "default"],
           title=v["loader_title"],
           # The FP8 file is built at setup, so it has no download link.
           models=[(MODEL, "diffusion_models")] if v["model"] == MODEL else None,
           cnr=WV, size=size(330))
    g.node(43, "WanVideoVAELoader", inputs=[("compile_args", "WANCOMPILEARGS", I, True)],
           outputs=[("vae", "WANVAE")], widgets=[VAE, "bf16", False, False],
           title="Wan 2.1 VAE", models=[(VAE, "vae")], cnr=WV, size=size(170))
    g.node(45, "WhisperModelLoader", outputs=[("whisper_model", "WHISPERMODEL")],
           widgets=[WHISPER, "fp16", "main_device"], title="Whisper large-v3 encoder",
           models=[(WHISPER, "audio_encoders")], cnr=WV, size=size(130))
    g.node(46, "LongCatLongformSigmas", outputs=[("sigmas", "SIGMAS")], widgets=[7.0],
           title="v1.5 Distill Sigmas (8 steps, shift 7)", cnr=LF, size=size(90))
    g.node(47, "WanVideoSchedulerv2", inputs=[("sigmas", "SIGMAS", I, True)],
           outputs=[("scheduler", "WANVIDEOSCHEDULER")],
           widgets=["euler", 8, 7.0, 0, -1, False],
           title="Scheduler (euler, custom sigmas)", cnr=WV, size=size(220))

    # ---------------------------------------------------------------- 5. guidance
    g.node(50, "ImageScale",
           inputs=[("image", "IMAGE", I, False), ("width", "INT", W, False),
                   ("height", "INT", W, False)],
           outputs=[("IMAGE", "IMAGE")], widgets=["lanczos", side, side, "center"],
           title="Crop portrait to canvas", size=size(170))
    encode_inputs = [("vae", "WANVAE", I, False), ("image", "IMAGE", I, False),
                     ("mask", "MASK", I, True)]
    encode_widgets = [False, 272, 272, 144, 128, 0.0, 1.0]
    g.node(51, "WanVideoEncode", inputs=encode_inputs, outputs=[("samples", "LATENT")],
           widgets=list(encode_widgets), title="Encode portrait", cnr=WV, size=size(250))
    g.node(53, "LongCatLongformCarry",
           inputs=[("portrait", "IMAGE", I, False), ("portrait_latent", "LATENT", I, False)]
           + ([("session", "STRING", I, True)] if batch else [])
           + [("chunk_index", "INT", W, False), ("opens_on_portrait", "BOOLEAN", W, False)],
           outputs=[("prev_images", "IMAGE"), ("ref_latent", "LATENT")],
           widgets=[0, False], title="Opening + Reference", cnr=LF, size=size(190))
    g.node(55, "LongCatAvatarWhisperEmbeds",
           inputs=[("whisper_model", "WHISPERMODEL", I, False), ("audio_1", "AUDIO", I, False),
                   ("audio_2", "AUDIO", I, True), ("audio_3", "AUDIO", I, True),
                   ("audio_4", "AUDIO", I, True), ("ref_target_masks", "MASK", I, True),
                   ("num_frames", "INT", W, False)],
           outputs=[("multitalk_embeds", "MULTITALK_EMBEDS"), ("audio", "AUDIO"),
                    ("num_frames", "INT")],
           # normalize_loudness off: Split already normalised the whole track.
           widgets=[False, 194, 25.0, 1.0, 1.0, "para"],
           title="Whisper Embeds (window audio, 25 fps)", cnr=WV, size=size(300))
    g.node(56, "WanVideoLongCatAvatarExtendEmbeds",
           inputs=[("prev_latents", "LATENT", I, False),
                   ("audio_embeds", "MULTITALK_EMBEDS", I, False),
                   ("ref_latent", "LATENT", I, True), ("samples", "LATENT", I, True),
                   ("prev_images", "IMAGE", I, True), ("vae", "WANVAE", I, True),
                   ("num_frames", "INT", W, False), ("overlap", "INT", W, False),
                   ("frames_processed", "INT", W, False),
                   ("ref_frame_index", "INT", W, False),
                   ("ref_mask_frame_range", "INT", W, False)],
           outputs=[("image_embeds", "WANVIDIMAGE_EMBEDS"), ("samples_slice", "LATENT")],
           widgets=[WINDOW, 1, 1, "pad_with_start", 10, 3],
           title="LongCat Extend Embeds (opening + audio + reference)", cnr=WV,
           size=size(330))

    # ---------------------------------------------------------------- 6. sampling
    g.node(60, "WanVideoSamplerv2",
           inputs=[("model", "WANVIDEOMODEL", I, False),
                   ("image_embeds", "WANVIDIMAGE_EMBEDS", I, False),
                   ("scheduler", "WANVIDEOSCHEDULER", I, False),
                   ("text_embeds", "WANVIDEOTEXTEMBEDS", I, True),
                   ("samples", "LATENT", I, True),
                   ("extra_args", "WANVIDSAMPLEREXTRAARGS", I, True)],
           outputs=[("samples", "LATENT"), ("denoised_samples", "LATENT")],
           widgets=[1.0, 42, "fixed", True, False],
           title="Sampler (cfg 1, fixed seed)", cnr=WV, size=size(250))
    g.node(61, "WanVideoDecode",
           inputs=[("vae", "WANVAE", I, False), ("samples", "LATENT", I, False)],
           outputs=[("images", "IMAGE")], widgets=[False, 272, 272, 144, 128, "default"],
           title="Decode", cnr=WV, size=size(250))

    # ---------------------------------------------------------------- 7. output
    fallback = ["batch" + v["suffix"], "batch" + v["suffix"] + ".mp4"] if batch else \
        ["run1" + v["suffix"], "longcat_avatar" + v["suffix"] + ".mp4"]
    g.node(70, "LongCatLongformWrite",
           inputs=[("images", "IMAGE", I, False), ("original_audio", "AUDIO", I, False),
                   ("chunk_audio", "AUDIO", I, True), ("chunk_index", "INT", W, False),
                   ("total_chunks", "INT", W, False), ("keep_frames", "INT", W, False),
                   ("session", "STRING", W, False), ("filename", "STRING", W, False),
                   ("trim_start", "INT", W, True)],
           outputs=[("status", "STRING")],
           # In a batch, the folder node clears the queue once every file is done.
           widgets=[0, 1, WINDOW] + fallback + [not batch, 0],
           title="Write chunk + stitch final video", cnr=LF, size=size(340))

    # ---------------------------------------------------------------- wiring
    L = g.link
    L(11, 0, 20, "audio")
    L(11, 0, 70, "original_audio")

    L(30, 0, 57, "positive_prompt")

    L(42, 0, 40, "block_swap_args")
    L(41, 0, 40, "lora")
    L(46, 0, 47, "sigmas")

    L(10, 0, 50, "image")
    L(13, 0, 50, "width")
    L(14, 0, 50, "height")
    L(43, 0, 51, "vae")
    L(50, 0, 51, "image")

    L(50, 0, 53, "portrait")
    L(51, 0, 53, "portrait_latent")
    L(20, "chunk_index", 53, "chunk_index")
    L(20, "opens_on_portrait", 53, "opens_on_portrait")

    L(45, 0, 55, "whisper_model")
    L(20, "whisper_audio", 55, "audio_1")
    L(20, "whisper_frames", 55, "num_frames")

    L(51, 0, 56, "prev_latents")
    L(55, 0, 56, "audio_embeds")
    L(53, "ref_latent", 56, "ref_latent")
    L(53, "prev_images", 56, "prev_images")
    L(43, 0, 56, "vae")
    for name in ("num_frames", "overlap", "frames_processed", "ref_frame_index",
                 "ref_mask_frame_range"):
        L(20, name, 56, name)

    L(40, 0, 60, "model")
    L(56, 0, 60, "image_embeds")
    L(47, 0, 60, "scheduler")
    L(57, 0, 60, "text_embeds")
    L(60, 0, 61, "samples")
    L(43, 0, 61, "vae")

    L(61, 0, 70, "images")
    for src, dst in (("audio_chunk", "chunk_audio"), ("chunk_index", "chunk_index"),
                     ("total_chunks", "total_chunks"), ("keep_frames", "keep_frames"),
                     ("trim_start", "trim_start")):
        L(20, src, 70, dst)
    if batch:
        # The folder node names each file's session; Split and Carry take it
        # directly so all three agree on the folder.
        L(11, 1, 20, "session")
        L(11, 1, 53, "session")
        L(11, 1, 70, "session")
        L(11, 2, 70, "filename")
    else:
        L(21, 0, 70, "session")
        L(21, 1, 70, "filename")

    # ---------------------------------------------------------------- layout
    inputs = [10, 11] + ([] if batch else [21]) + [13, 14]
    guide = [50, 51, 53, 55, 56]
    stages = [
        ("1. Your inputs", "input", inputs, [3]),
        ("2. Prompt (optional)", "prompt", [30, 57], [4]),
        ("3. Chunk plan (automatic)", "plan", [20], [5]),
        ("4. Models (automatic)", "models", [42, 41, 40, 43, 45, 46, 47], [6]),
        ("5. Opening, reference + audio", "guide", guide, [7]),
        ("6. Sampling", "sample", [60, 61], [8]),
        ("7. Save + stitch", "output", [70], [9]),
    ]
    groups = lay_out(g, stages, [1, 2])

    return {
        "id": w["id"],
        "revision": 0,
        "last_node_id": max(n["id"] for n in g.nodes),
        "last_link_id": g._link_id,
        "nodes": g.nodes,
        "links": g.links,
        "groups": groups,
        "config": {},
        "extra": {"ds": {"scale": 0.55, "offset": [80, 80]}},
        "version": 0.4,
    }


if __name__ == "__main__":
    for w in WORKFLOWS:
        wf = build(w["variant"], w["batch"], w["precision"])
        path = os.path.join(WORKFLOW_DIR, w["path"])
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(wf, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"wrote {os.path.normpath(path)}: {len(wf['nodes'])} nodes, "
              f"{len(wf['links'])} links")
