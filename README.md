# LongCat-Video-Avatar 1.5: RunPod ComfyUI template (long-form talking avatar)

A one-shot template that installs **LongCat-Video-Avatar 1.5** (Meituan, MIT licence)
on RunPod with ComfyUI and JupyterLab. Its workflows turn **one portrait and a
voiceover of any length** (or a folder of voiceovers) into a talking-head video,
rendered in chunks and stitched with your original audio. It is the same approach as the MiniMax H3 template, rebuilt
around LongCat's video continuation.

ComfyUI has no native LongCat video nodes. The model runs through
[kijai/ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper),
pinned to a known commit, plus a small bundled node pack that plans the chunks.

## What LongCat can and cannot do

| | LongCat-Video-Avatar 1.5 | MiniMax H3 (for comparison) |
|---|---|---|
| Per generation | 93 frames at 25 fps (3.72 s) | up to 362 frames at 24 fps (15 s) |
| Long videos | **video continuation**: each window opens on the previous window's last 13 frames | independent chunks pinned to the portrait |
| First frame | yes (the portrait) | yes |
| **End frame** | **no**: conditioning frames always sit at the start of a window | yes |
| Identity | a reference latent inside every continuation window | first and last frame |
| Steps | 8 (DMD-distilled LoRA), cfg 1 | 4 to 20 |
| Licence | MIT | excludes the US, EU, UK and South Korea |

Because every chunk continues from the one before, **joins are seamless**: there is no
pose reset every few seconds. The price is that chunks must render **in order**, a
7-minute voiceover is about 130 chunks, and continuation **drifts** over long videos,
which the workflows counter with a restart from the portrait at a pause every
40-60 s.

### Pulling back to the portrait, and restarts against drift

LongCat has no end-frame input: its code takes conditioning latents only at the start
of a window. There is one lever: the identity **reference** has a time position inside
the window (`ref_frame_index`; upstream uses 10). The workflows move it to each
window's last frame (the Split node's `return to portrait` anchor), with no attention
mask, which draws every chunk back toward the portrait pose. In testing it kept the
face together best.

Continuation still drifts over long videos: small noise in the 13 carried frames grows
from chunk to chunk. So the workflows **restart from the portrait** once in every
**40-60 s**: the chunk starting there opens on the portrait itself, exactly as the first
chunk does, instead of on the previous frames. That is a full refresh of the face.

The restarts are **planned from the whole voiceover before rendering starts**, and only
go where the mouth is shut:

1. Every **real pause** is found first: at least `min_pause_seconds` (**0.4 s**) of
   near-silence, measured against the recording's own speech and noise levels. Short
   gaps between words do not count.
2. Each restart goes **in the middle of the longest real pause** 40-60 s after the
   previous one, so it is silent on both sides of the cut. The face drifts only slowly,
   so a restart about once a minute keeps it true with as few jumps as possible.
3. If that window has no real pause, the **next real pause within 20 s** is used rather
   than cutting into speech. Only when there is none at all does a shorter gap get it
   (flagged in the plan).

The plan (time, pause length, how it was chosen) is printed in the log on the first
chunk and saved as `output/longcat_longform/<name>/restart_plan.txt`. On the 71 s sample
voiceover, the single restart lands at 51.2 s, in the middle of a 0.44 s pause, the only
real pause between 40 and 60 s.

`restart_min_seconds`, `restart_max_seconds` and `min_pause_seconds` are on the **Split**
node; `restart_max_seconds = 0` turns restarts off.

## What boots

On first pod boot, `setup.sh`:

1. Works out which volume is attached: a **Network Volume** or a **Global Volume**
   (see [Storage](#storage-network-volume-or-global-volume)); refuses to install on the
   container disk alone
2. Copies the baked ComfyUI from `/opt/comfyui-baked` to `/workspace/runpod-slim/ComfyUI`
   (self-healing, skip-if-present). With a Global Volume it goes to the container disk
   instead, and its `models`, `input`, `output` and `user/default` folders link to the
   volume
3. Checks that ComfyUI has `comfy_api/latest` and that the image's PyTorch has a CUDA
   build that can use the GPU; stops if not, before any download
4. Installs the bundled `comfyui-longcat-longform` nodes and **ComfyUI-WanVideoWrapper
   at a pinned commit**, then installs the wrapper's Python packages into ComfyUI's
   persistent virtualenv with the image's PyTorch pinned (`/opt/comfyui-runtime-constraints.txt`)
5. Downloads the five model files (skip-if-exists, resumable)
6. Builds the **FP8 model** from the BF16 download (`tools/make_fp8.py`, once, a few
   minutes)
7. Copies the sample portrait and voiceover into `ComfyUI/input/` (never over a file
   of the same name) and installs the eight workflows (BF16 and FP8) into ComfyUI's
   **Workflows** sidebar
8. Hands over to `/start.sh` (ComfyUI on port **8188**, JupyterLab on **8888**)

On later boots every step finds its work done and is skipped, including the package
install (a marker records the wrapper's requirements hash). With a Global Volume, a
fresh container disk repeats steps 2 and 4 (a few minutes: copy ComfyUI, fetch the
wrapper, install its packages); the models are never downloaded again.

## Models pulled

All files come from [`Kijai/WanVideo_comfy`](https://huggingface.co/Kijai/WanVideo_comfy),
which is **ungated**, so no token is needed. They are Kijai's ComfyUI conversions of
[`meituan-longcat/LongCat-Video-Avatar-1.5`](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.5)
and the shared components the wrapper loads.

| File | Size | Destination (`ComfyUI/models/`) |
|---|---|---|
| `LongCat-Avatar-15_bf16.safetensors` (single-speaker model) | 31.7 GB | `diffusion_models/` |
| `LongCat-Avatar-15_dmd_distill_lora_rank128_bf16.safetensors` (8-step LoRA) | 1.3 GB | `loras/` |
| `umt5-xxl-enc-bf16.safetensors` (text encoder) | 11.4 GB | `text_encoders/` |
| `Wan2_1_VAE_bf16.safetensors` | 0.25 GB | `vae/` |
| `whisper_large_v3_encoder_fp16.safetensors` (v1.5 audio encoder) | 1.7 GB | `audio_encoders/` |
| `LongCat-Avatar-15_fp8_e4m3fn.safetensors` (**built at setup**, not downloaded) | ~17 GB | `diffusion_models/` |

The download is about **46 GB**; with the FP8 model, the models take about **63 GB**.
Use **100 GB of Network Volume storage** as a starting point for ComfyUI, its
environment, models and renders. A Global Volume grows by itself; it holds the models
(~63 GB) plus your inputs and renders.

### The FP8 model

There is no ready-made FP8 file of LongCat-Avatar 1.5 in the ComfyUI format (Kijai ships
bf16 only; the FP8 uploads elsewhere use Meituan's original layout). So setup builds one
from the BF16 file with [`tools/make_fp8.py`](tools/make_fp8.py), using the same rule
WanVideoWrapper applies when it loads the BF16 file with `quantization = fp8_e4m3fn`:
the large attention and feed-forward weights (673 of 1,896 tensors, 29.7 of 31.7 GB)
are stored as `float8_e4m3fn`; norms, biases, modulation, embeddings and the audio
projection stay bf16. Computation still runs at bf16.

FP8 halves the VRAM the model weights take (about 17 GB instead of 32 GB) and the data
read at load. It does not make sampling faster here: the weights are upcast to bf16 for
each layer. The cost is a small loss of precision; compare a BF16 and an FP8 render of
the same chunk to judge it.

## Storage: Network Volume or Global Volume

Attach **one** of these at `/workspace` when deploying. Setup detects which it is and
says so in the log (`Storage: network volume ...` or `Storage: global volume ...`).

| | Network Volume | Global Volume (RunPod beta) |
|---|---|---|
| Where pods can run | its datacenter only | any datacenter (GPU pods) |
| ComfyUI, venv, custom nodes | on the volume | on the container disk, set up again on each fresh container (a few minutes) |
| Models, inputs, outputs, saved workflows | on the volume | on the volume |
| Size | fixed, 100 GB recommended | grows by itself |
| Container disk | 5 GB is enough | **50 GB** (first-boot downloads are staged there) |
| Price | $0.07/GB/month | $0.09/GB/month plus a small per-request charge |

### Network Volume

Attach **one regional Network Volume at `/workspace`**. Setup installs everything
there; the container disk holds disposable system files.

| Data | Persistent location |
|---|---|
| ComfyUI and custom nodes | `/workspace/runpod-slim/ComfyUI/` |
| ComfyUI Python environment | `/workspace/runpod-slim/ComfyUI/.venv-cu128/` |
| Models | `ComfyUI/models/` |
| Uploaded portraits and audio | `ComfyUI/input/` |
| Saved sidebar workflows | `ComfyUI/user/default/workflows/` |
| Chunks, continuation tails and render workflow copies | `ComfyUI/output/longcat_longform/<session>/` |
| Final videos | `ComfyUI/output/` |
| Jupyter settings and IPython data | `/workspace/.jupyter/`, `/workspace/.local/share/jupyter/`, `/workspace/.ipython/` |
| Download caches | `/workspace/.cache/huggingface/` and `/workspace/.hf-cli/` |

The table's `ComfyUI/` paths are relative to `/workspace/runpod-slim/`. Setup verifies
that `/workspace` is a separate mounted volume and tests file replacement, symlinks and
file execution. A pod-local Volume disk is lost when its pod is deleted; select a
**Network Volume**. See [RunPod storage types](https://docs.runpod.io/pods/storage/types).

**Restart or replacement pod:** attach the same Network Volume at `/workspace` and use
this template. Open your workflow and queue again: completed chunks are skipped. The
chunk that was rendering may be rendered again after an interruption.

### Global Volume

A [Global Volume](https://docs.runpod.io/storage/globalvolume/overview) is not tied to
a datacenter: one copy of the models serves pods anywhere. It is object storage rather
than a full POSIX file system, with **no file locking, atomic rename, hard links or
permission bits**. A Python virtualenv, a git checkout and pip all depend on those, so
the earlier version of this template refused Global Volumes. Now setup splits the work:

| Data | Location |
|---|---|
| ComfyUI, custom nodes, Python environment | container disk: `/opt/longcat-local/ComfyUI/` |
| Models | volume: `runpod-slim/ComfyUI/models/` |
| Uploaded portraits and audio, `batch_audio/` | volume: `runpod-slim/ComfyUI/input/` |
| Saved sidebar workflows and ComfyUI settings | volume: `runpod-slim/ComfyUI/user/default/` |
| Chunks and final videos | volume: `runpod-slim/ComfyUI/output/` |
| Jupyter settings, IPython history, download caches, FileBrowser database | container disk: `/opt/longcat-local/` |

The volume paths are the same as on a Network Volume, so a Network Volume copied over
with RunPod's suggested `rsync -a /workspace/ /workspace-global/` is picked up as it
is: its models are reused and its old ComfyUI install is ignored. ComfyUI's
`models`, `input`, `output` and `user/default` folders are links into the volume.
In Jupyter, upload into `/workspace/runpod-slim/ComfyUI/input/` as usual (when the
volume is mounted at `/workspace-global`, it is linked as `/workspace/global-volume/`).

How setup handles the volume's limits:

- **Downloads** go to `/opt/longcat-local/downloads/` on the container disk, are checked
  there, then copied to their final name on the volume. Nothing is renamed or locked on
  the volume. The 31.7 GB BF16 model needs that much free container disk once, hence
  **50 GB**. With less, setup warns; a download that runs out of space fails with
  `No space left on device` and can be retried after raising the container disk.
- **The FP8 model** is built on the container disk and copied over the same way.
- **The image's `/start.sh`** fixes ComfyUI at `/workspace/runpod-slim/ComfyUI`. Setup
  runs a copy of it (`/opt/longcat-local/image-start.sh`) pointed at the container-disk
  install, and keeps FileBrowser's database there; the image file itself is not edited.

Things to know:

- **A fresh container disk takes a few minutes longer to boot.** A new pod, or a pod
  started again after a stop, gets a new container disk: ComfyUI is copied from the
  image, the wrapper fetched and its packages installed again. Models are not.
- **Custom nodes you add with ComfyUI-Manager do not persist.** They live on the
  container disk; add them to `setup.sh` if you need them every time.
- **Run one pod at a time on a fresh volume.** Let the first pod finish downloading
  (`Setup complete!`) before starting more: two pods writing the same file on a
  Global Volume overwrite each other. After that, several pods can read the models
  at once. Give each pod its own voiceovers so they don't write the same chunks.
- Model loading reads from the volume, which RunPod optimises for reads; the first load
  of a 32 GB model may take longer than from a Network Volume.
- Mounted at `/workspace-global` instead (for example next to a Network Volume, or with
  its mount path edited), setup finds it there. A Network Volume at `/workspace` takes
  priority; `LONGCAT_STORAGE=global` uses the Global Volume instead.

Global Volumes are in beta at RunPod; their behaviour may change. If setup misjudges
which volume you have, set `LONGCAT_STORAGE=network` or `LONGCAT_STORAGE=global`.

## RunPod public template settings

| Setting | Value |
|---|---|
| Template name | `LongCat Video Avatar - ComfyUI + JupyterLab` |
| Compute type | **NVIDIA / GPU** |
| Base image | **`runpod/comfyui:1.4.0-comfyuiv0.35.0-cuda12.8`** |
| Registry authentication | None; image is public |
| Container disk | **50 GB** (a Global Volume stages downloads there; a Network Volume needs only 5 GB) |
| Persistent storage in the public template editor | **Volume disk**, **0 GB** |
| Template volume mount path | **`/workspace`** |
| Volume at deployment | Each deployer attaches their own **Network Volume** (100 GB recommended) or **Global Volume** at `/workspace` |
| HTTP ports | **`8188,8888`** (ComfyUI, JupyterLab) |
| TCP ports | None required; expose `22` only if SSH is wanted |
| Environment variables | `JUPYTER_NO_AUTH=1`; `HF_TOKEN` optional |
| README tab | Paste [`TEMPLATE_README.md`](TEMPLATE_README.md) |

The **CUDA 12.8** image is used deliberately: unlike H3, nothing here depends on CUDA 13
kernels, and the 12.8 build runs on far more RunPod hosts (driver r570 and newer). Its
paths, venv name and Jupyter switch are the same as the CUDA 13.0 image's. If a host's
driver is too old for it, setup stops before downloading anything and says so.

`JUPYTER_NO_AUTH=1` runs Jupyter without a login: setup passes the image's own
`JUPYTER_DISABLE_AUTH=true` switch to `/start.sh`. For authenticated Jupyter, use
`JUPYTER_NO_AUTH=0` and set `JUPYTER_PASSWORD` at deployment; it becomes the token.
Do not override `COMFYUI_PATH`: the image starts ComfyUI at its fixed path.

**Container start command.** Paste this JSON into RunPod's container start command
field (also provided as [`runpod-start.json`](runpod-start.json); readable source:
[`bootstrap.sh`](bootstrap.sh)). It checks DNS for GitHub and Hugging Face first and
leaves working DNS unchanged. If two checks fail, it saves the original
`/etc/resolv.conf` under `/tmp/longcat-avatar-dns.*`, then tries the public resolvers
`1.1.1.1` and `8.8.8.8`; if resolution still fails, it restores the original and stops
with a networking error. Git clone failures are retried up to 10 times. It clones
`main` unless `TEMPLATE_REF` names another branch or tag: `TEMPLATE_REF=v1-network-volume`
runs the version from before Global Volume support.

```json
{
  "entrypoint": [
    "bash",
    "-lc"
  ],
  "cmd": [
    "set -e\ndns_ok() {\n  timeout 10 getent ahostsv4 github.com >/dev/null 2>&1 &&\n  timeout 10 getent ahostsv4 huggingface.co >/dev/null 2>&1\n}\nif ! dns_ok; then\n  echo 'DNS lookup failed; retrying in 3 seconds...'\n  sleep 3\n  if ! dns_ok; then\n    dns_backup=$(mktemp /tmp/longcat-avatar-dns.XXXXXX)\n    cp /etc/resolv.conf \"$dns_backup\"\n    echo \"Trying public DNS; original resolver saved at $dns_backup\"\n    if { printf 'nameserver 1.1.1.1\\nnameserver 8.8.8.8\\noptions timeout:2 attempts:2\\n'; } > /etc/resolv.conf && dns_ok; then\n      echo 'DNS recovered; continuing setup.'\n    else\n      cat \"$dns_backup\" > /etc/resolv.conf || true\n      echo 'FATAL: DNS still unavailable or resolver is read-only. Check RunPod networking or deploy on another host with the same Network Volume.' >&2\n      exit 1\n    fi\n  fi\nfi\nfor attempt in 1 2 3 4 5 6 7 8 9 10; do\n  repo=$(mktemp -d /tmp/longcat-avatar.XXXXXX)\n  if git clone --depth 1 --branch \"${TEMPLATE_REF:-main}\" https://github.com/tenitsky/longcat-avatar.git \"$repo\"; then\n    exec bash \"$repo/setup.sh\"\n  fi\n  sleep 5\ndone\necho 'FATAL: unable to clone longcat-avatar after 10 attempts' >&2\nexit 1\n"
  ]
}
```

For the first launch, check the logs for **`Network Volume workspace ready at
/workspace`** (or **`Global Volume ready at /workspace`**), then
`ComfyUI-WanVideoWrapper: 088128b`, the model downloads and
**`Setup complete!`**. If ComfyUI works but Jupyter stays on initializing, inspect
`/jupyter.log` from the pod's terminal.

## After it boots

1. Open ComfyUI (port 8188), open the **Workflows** sidebar and choose
   **`longcat_bf16_1_single_audio_640x640`** (80 GB GPUs) or
   **`longcat_fp8_1_single_audio_640x640`** (32-48 GB GPUs).
2. Set **Load Portrait** and **Load Voiceover**. The workflows open on the bundled
   samples (`sample_portrait.png`, `sample_voiceover.mp3`, 71 s), so the first run needs
   no uploads. The output is named after the audio file automatically.
3. **Prompt** is optional. Leave it empty for a generic talking-head prompt (speaking to
   the camera, small natural motion, static camera and background), or describe the
   portrait in a sentence, e.g. *A woman in a navy blazer, in front of a plain grey
   wall.* The person and background come from the portrait either way.
4. **Test one chunk first:** batch count 1, then check
   `output/longcat_longform/<name>/chunk_0000.mp4` for lip sync, identity and background.
5. Queue again with a batch count of at least **audio seconds ÷ 3.2**, plus a margin.
   Each queue item renders the next unfinished chunk, found from the files on disk;
   surplus items are skipped in milliseconds. If the batch count box stops at 100,
   click Run several times: the queue adds up.
6. The last chunk writes `output/<audio name>.mp4` and clears the queue.

**Next video:** load the new voiceover and queue. It starts at chunk 1 in its own
folder; there is no counter to reset. **Resume:** queue again. **Redo from a chunk:**
delete that chunk's mp4 **and every later one** (each chunk continues from the one
before it), then queue. **Render a finished video again:** change `name_suffix` on the
output-name node, or delete its folder under `output/longcat_longform/`.

> `setup.sh` skips existing workflows, so a workflow you edited on the pod is never
> overwritten on reboot. The flip side is that an updated workflow in this repo won't
> reach an existing volume until you delete the old file from
> `ComfyUI/user/default/workflows/` and restart. Exception: a file still identical to a
> version this repo shipped earlier (an old name, or an older version of a current one)
> is removed at boot and the current version installed; edited copies are kept.

### Workflows

| Workflow (`longcat_bf16_…` / `longcat_fp8_…`) | Audio | Canvas |
|---|---|---|
| `1_single_audio_640x640` | one voiceover | 640 × 640 |
| `2_batch_audio_640x640` | every voiceover in `input/batch_audio/` | 640 × 640 |
| `3_single_audio_960x960` | one voiceover | 960 × 960, high quality |
| `4_batch_audio_960x960` | every voiceover in `input/batch_audio/` | 960 × 960, high quality |

**BF16** loads the full-precision model with Block Swap off (80 GB GPUs); **FP8** loads
the FP8 model with Block Swap on, 20 of 48 blocks (32-48 GB GPUs). 960 × 960 takes about
2.3 times as long per chunk as 640. Every workflow pulls each chunk back toward the
portrait and restarts from it at a real pause every 40-60 s.

Videos are saved as `output/<audio name>_bf16_640.mp4` (or `_fp8_960`, …), so renders of
the same voiceover at another precision or size sit side by side and never resume each
other's chunks.

The graph is laid out in seven numbered, coloured stages from left to right (inputs,
prompt, chunk plan, models, opening and guidance, sampling, save), with a note under each
stage and a **Start here** note on the far left. Red nodes are the ones you set.

### Batch: a folder of voiceovers

1. Put the voiceovers in `ComfyUI/input/batch_audio/` (created at boot). Upload them with
   JupyterLab (port 8888). Accepted: `.wav`, `.mp3`, `.flac`, `.m4a`, `.aac`, `.ogg`,
   `.opus`.
2. Open a `batch_audio` workflow and set the portrait (and the optional prompt). The same
   portrait and prompt are used for every file.
3. Test with batch count 1, then queue a large batch count: roughly the total seconds of
   all files ÷ 3.2, plus a margin.

Each queue item takes the first file, in name order (`2` before `10`), whose video isn't
finished, and renders its next unfinished chunk; each file gets its own restart plan.
When every file is done, the **Audio From Folder** node clears the queue. Files added
later are picked up on the next run.

### How the workflow works

- **Chunk 1** opens on the portrait (one conditioning frame) and keeps all 93 frames.
  **Every later chunk** opens on the previous chunk's last 13 written frames, re-encoded
  through the VAE the way v1.5 does it, and keeps the 80 new frames after them. Nothing
  plays twice. The 13-frame tails are kept under `.carry/` for resume.
- **A restart** (every 40-60 s, in the middle of a real pause) is a chunk that opens on
  the portrait again and keeps up to a whole window, exactly like chunk 1. The chunk
  before it simply ends in the pause, however short that makes it.
- On every continuing chunk, the portrait is also the **identity reference**, on the
  window's last frame.
- **Audio.** The whole voiceover is normalised to -23 LUFS once, as upstream does, then
  each window gets its own 16 kHz slice with 2 s of context either side for Whisper.
  That keeps loudness constant across chunks and avoids the Whisper node's 400 s limit.
  Frame and sample grids line up exactly at 25 fps (640 samples per frame).
- **Sampling** reproduces upstream's distilled v1.5 settings: the DMD LoRA at strength 1,
  8 steps, cfg 1, audio cfg 1, and upstream's exact sigma schedule (shift 7). The
  wrapper's built-in `longcat_distill_euler` is the v1.0 schedule, so a small node
  supplies the v1.5 one.
- The final video uses your **original** audio track, encoded as AAC. Chunk positions
  share one 25 fps timeline, so there is no cumulative drift.

## GPU guidance

These are estimates from the model size. **They haven't been measured on a pod yet.**

| GPU | Notes |
|---|---|
| 80–96 GB (H100 / A100 80 GB / RTX PRO 6000) | **BF16** workflow as shipped (Block Swap off). FP8 works too; bypass its Block Swap (select it, Ctrl+B) for a little more speed |
| 48 GB (A6000 / L40S) | **FP8** workflow as shipped (Block Swap on, 20 blocks); VAE tiling if decode runs out of memory |
| 32 GB (RTX 5090) | **FP8** workflow as shipped; raise `blocks_to_swap` if a larger canvas runs out of memory; a pod with 64 GB+ of RAM |
| 24 GB | **FP8** workflow, `blocks_to_swap` 40+; slow |

The model must stay at **bf16 base precision** (a wrapper requirement for LongCat).
Attention is `sdpa`; `sageattn` works if SageAttention 2 is installed (1.0.6 does not).

## Template env vars (all optional)

| Var | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | *(empty)* | Not required (the repo is ungated). It raises Hugging Face rate limits |
| `JUPYTER_NO_AUTH` | `1` | Disables Jupyter login (anyone with the URL gets a shell). Set `0` and `JUPYTER_PASSWORD` for a login |
| `JUPYTER_PASSWORD` | *(empty)* | Jupyter token, used when `JUPYTER_NO_AUTH=0` |
| `WANVIDEO_WRAPPER_REF` | pinned commit | Test a newer ComfyUI-WanVideoWrapper commit. The workflows were built against the pin |
| `FILEBROWSER_PASSWORD` | `adminadmin12` | FileBrowser on port 8080 (user `admin`). **Change the default** |
| `COMFYUI_PATH` | `/workspace/runpod-slim/ComfyUI` | Leave unset; must match the base image's fixed startup path |
| `HF_HOME` | `/workspace/.cache/huggingface` | Keeps the HF cache on the volume, not the container disk (Global Volume: `/opt/longcat-local/.cache/huggingface`) |
| `LONGCAT_STORAGE` | `auto` | `network` or `global` overrides which volume setup thinks it has |
| `TEMPLATE_REF` | `main` | Branch or tag of this repo the start command clones; `v1-network-volume` is the version before Global Volume support |

ComfyUI runs from its own virtualenv (`$COMFYUI_PATH/.venv-cu128`; with a Global Volume
`/opt/longcat-local/ComfyUI/.venv-cu128`). If you install
anything by hand, use that interpreter with the image's pins, and never install `torch`
into it:

```bash
/workspace/runpod-slim/ComfyUI/.venv-cu128/bin/python -m pip install -c /opt/comfyui-runtime-constraints.txt <package>
```

## What has and hasn't been verified

Verified without a GPU: every WanVideoWrapper node's widget order and sockets against the
wrapper's source at the pinned commit; the chunk plan against upstream's segment
arithmetic; the v1.5 sigma schedule against upstream's pipeline code; audio slicing and
the Whisper frame count; continuation tails, trimming and stitching with real ffmpeg
(frame-exact across every join and restart); the restart planner on synthetic speech
(real pauses chosen over short gaps, cuts mid-pause) and on the sample voiceover;
resume, batch folders, and a full first boot and reboot
of `setup.sh` against an offline image fixture, on a Network Volume and on a simulated
Global Volume that refuses rename, hard links, symlinks and permission changes.

**Not yet verified on a real Global Volume:** how RunPod reports its mount type (setup
recognises it by its missing hard links), download and model-load speed from it, and
ffmpeg writing chunks to it.

**Not yet verified:** a render on a real GPU, so lip-sync quality, VRAM use and speed are
unknown, and the FP8 model's quality against BF16 is unmeasured. The FP8 conversion itself is tested on CPU: tensor
selection, values against PyTorch's own cast, and loading with `safetensors`.

## Repository layout

```
├── setup.sh                               # RunPod boot script
├── bootstrap.sh / runpod-start.json       # container start command (readable / paste-ready)
├── README.md
├── TEMPLATE_README.md                     # paste into the public template's README tab
├── custom_nodes/
│   └── comfyui-longcat-longform/          # Split / Opening + Reference / Write + Stitch / Name / Prompt / Sigmas / Audio From Folder
├── tools/
│   ├── build_workflow.py                  # generates all workflow JSONs; re-run after edits
│   ├── make_fp8.py                        # builds the FP8 model at setup (run by setup.sh)
│   ├── test_fp8.py                        # FP8 tensor selection, values and file format
│   ├── test_workflows.py                  # graphs vs. node schemas, settings, layout, downloads
│   ├── test_longform.py                   # planning, audio, continuation and ffmpeg stitching
│   ├── test_storage.py                    # volume detection, Network and Global Volume checks
│   ├── test_boot.py                       # full setup with image/network fixtures (Linux)
│   ├── test_jupyter.py                    # auth switch and real Jupyter Server smoke test
│   └── test_setup.py                      # Bash download and start-command checks
├── samples/                               # sample portrait + 71 s voiceover the workflows open on
└── workflows/                             # the eight workflows listed above
```

## References

- Model card, usage tips and licence: https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.5
- Inference code and technical report: https://github.com/meituan-longcat/LongCat-Video
- ComfyUI nodes: https://github.com/kijai/ComfyUI-WanVideoWrapper (`LongCat/nodes.py`)
- ComfyUI weights: https://huggingface.co/Kijai/WanVideo_comfy (`LongCat/`)
