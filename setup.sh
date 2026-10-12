#!/bin/bash
set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Starting LongCat Video Avatar Template Setup ==="
echo "Template revision: $(git -C "$SCRIPT_DIR" rev-parse --short HEAD 2>/dev/null || printf unknown)"

# Refuse a disposable workspace before doing installation work.
# The image's /start.sh fixes ComfyUI at this path.
export COMFYUI_PATH="${COMFYUI_PATH:-/workspace/runpod-slim/ComfyUI}"
if [ "$COMFYUI_PATH" != /workspace/runpod-slim/ComfyUI ]; then
  echo "FATAL: this image's /start.sh uses /workspace/runpod-slim/ComfyUI. Leave COMFYUI_PATH unset."
  exit 1
fi
python3 "$SCRIPT_DIR/custom_nodes/comfyui-longcat-longform/storage.py" preflight "$COMFYUI_PATH"

# Apply before the image starts Jupyter. The image's /start.sh runs JupyterLab
# without a token when JUPYTER_DISABLE_AUTH=true, uses JUPYTER_PASSWORD as the
# token when set, and otherwise does not start Jupyter at all. The exported value
# reaches /start.sh through the final exec. Jupyter is optional, so a start script
# without that switch is reported rather than stopping the pod.
if [ "${JUPYTER_NO_AUTH:-1}" = "1" ]; then
  export JUPYTER_DISABLE_AUTH=true
  echo "Jupyter authentication disabled: anyone with access to its URL can run commands."
  if ! grep -q 'JUPYTER_DISABLE_AUTH' /start.sh 2>/dev/null; then
    echo "WARNING: this image's /start.sh has no JUPYTER_DISABLE_AUTH switch, so JupyterLab"
    echo "         may not start. Use runpod/comfyui:1.4.0-comfyuiv0.35.0-cuda12.8, or set"
    echo "         JUPYTER_NO_AUTH=0 and JUPYTER_PASSWORD. ComfyUI is unaffected."
  fi
else
  unset JUPYTER_DISABLE_AUTH
  if [ -z "${JUPYTER_PASSWORD:-}" ]; then
    echo "NOTE: JUPYTER_NO_AUTH=0 without JUPYTER_PASSWORD: the image does not start JupyterLab."
  fi
fi

echo "=== Ensuring System Dependencies are Installed ==="
# The image already supplies these. Contact apt mirrors only if something is
# actually missing; an unrelated mirror outage must not break a complete image.
packages=()
command -v wget >/dev/null || packages+=(wget)
command -v git >/dev/null || packages+=(git)
command -v ffmpeg >/dev/null || packages+=(ffmpeg)
[ -s /etc/ssl/certs/ca-certificates.crt ] || packages+=(ca-certificates)
if [ "${#packages[@]}" -gt 0 ]; then
  apt-get update
  apt-get install -y --no-install-recommends "${packages[@]}"
fi

# Jupyter serves /workspace in the base image. Keep its settings and user data on
# that persistent volume too; /root belongs to the disposable container disk.
export JUPYTER_CONFIG_DIR=/workspace/.jupyter
export JUPYTER_DATA_DIR=/workspace/.local/share/jupyter
export IPYTHONDIR=/workspace/.ipython
mkdir -p "$JUPYTER_CONFIG_DIR" "$JUPYTER_DATA_DIR" "$IPYTHONDIR"

# Network Volumes can expose fixed permissions. Jupyter's cookie secrets and
# kernel connection files require private 0600 files; these are disposable
# runtime state, not notebooks or settings. Keep them on the container filesystem.
JUPYTER_RUNTIME_DIR="$(mktemp -d /tmp/longcat-jupyter-runtime.XXXXXX)"
export JUPYTER_RUNTIME_DIR
chmod 700 "$JUPYTER_RUNTIME_DIR"
python3 - <<'PY'
import os
from pathlib import Path
from jupyter_core.paths import secure_write

probe = Path(os.environ["JUPYTER_RUNTIME_DIR"]) / "secure-write-check"
try:
    with secure_write(str(probe)) as stream:
        stream.write("Jupyter runtime permissions verified")
finally:
    probe.unlink(missing_ok=True)
print(f"Jupyter runtime ready at {probe.parent}; notebooks and settings remain on /workspace.")
PY

# LongCat-Video-Avatar has no native ComfyUI nodes. It runs through
# kijai/ComfyUI-WanVideoWrapper, pinned here to the commit the bundled workflows
# were built against (LongCat-Avatar 1.5 support landed 2026-05-23). Override with
# WANVIDEO_WRAPPER_REF only to test a newer commit.
#   Base image: runpod/comfyui:1.4.0-comfyuiv0.35.0-cuda12.8
# This script deliberately does NOT update ComfyUI core: on these images
# `pip install -r requirements.txt` can reinstall torch and break the baked CUDA build.
WANVIDEO_WRAPPER_URL="https://github.com/kijai/ComfyUI-WanVideoWrapper.git"
WANVIDEO_WRAPPER_REF="${WANVIDEO_WRAPPER_REF:-088128b224242e110d3906c6750e9a3a348a659b}"

# Kijai/WanVideo_comfy is ungated, so no token is needed. Set HF_TOKEN anyway if you
# have one: anonymous requests are rate-limited (~120/h vs ~1000/h) and slower.
HF_TOKEN="${HF_TOKEN:-}"
[ -n "$HF_TOKEN" ] && export HF_TOKEN || echo "NOTE: HF_TOKEN not set - downloads may be slower and rate-limited."
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
# Kijai's ComfyUI conversions of LongCat-Avatar 1.5, its DMD LoRA, and the shared
# umt5 / Wan VAE / Whisper files the WanVideoWrapper nodes load.
HF_REPO="Kijai/WanVideo_comfy"

# Self-healing check prevents directory collisions and fixes broken folders
if [ ! -f "$COMFYUI_PATH/main.py" ]; then
  echo "First time setup: Copying baked ComfyUI to workspace..."
  if [ -e "$COMFYUI_PATH" ]; then
    echo "FATAL: existing ComfyUI directory has no main.py; refusing to delete user files."
    exit 1
  fi
  mkdir -p "$(dirname "$COMFYUI_PATH")"
  cp -r /opt/comfyui-baked "$COMFYUI_PATH"
fi

if [ ! -f "$COMFYUI_PATH/main.py" ]; then
  echo "FATAL: no ComfyUI at $COMFYUI_PATH and /opt/comfyui-baked did not provide one."
  echo "       Check the container image."
  exit 1
fi
echo "ComfyUI: $COMFYUI_PATH"

# Validation only needs the standard library. Use the image interpreter that
# already passed preflight, not a possibly stale persisted ComfyUI virtualenv.
PY="$(command -v python3)"
echo "Setup validator Python: $PY"

# The LongCat nodes in WanVideoWrapper are written against ComfyUI's comfy_api.latest
# schema; without it the pack fails to import and the workflows show red nodes.
if [ ! -d "$COMFYUI_PATH/comfy_api/latest" ]; then
  echo "FATAL: this ComfyUI has no comfy_api/latest, which WanVideoWrapper's LongCat"
  echo "         nodes need. Use the image runpod/comfyui:1.4.0-comfyuiv0.35.0-cuda12.8."
  exit 1
fi

# PyTorch comes from the image (the volume's venv only adds custom-node packages).
# The image also starts on hosts whose driver is too old for its CUDA build, where
# PyTorch cannot see the GPU - stop here, before 46 GB of downloads, rather than
# render on the CPU.
TORCH_PY="$(command -v python3.12 || command -v python3)"
if ! "$TORCH_PY" - <<'PY'
import sys
try:
    import torch
except Exception as exc:
    sys.exit(f"FATAL: cannot import the image's PyTorch ({exc}).")
build = torch.version.cuda or "none"
print(f"PyTorch {torch.__version__}, CUDA build {build}")
if build == "none":
    sys.exit("FATAL: the image's PyTorch has no CUDA build. Change the template's "
             "container image to runpod/comfyui:1.4.0-comfyuiv0.35.0-cuda12.8.")
if not torch.cuda.is_available():
    sys.exit(f"FATAL: CUDA {build} PyTorch cannot use this GPU: the host driver is too "
             "old for it. Redeploy on another GPU or datacenter; the same Network "
             "Volume can be reattached.")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY
then
  exit 1
fi

# 1. Custom nodes
echo "Installing custom nodes..."
mkdir -p "$COMFYUI_PATH/custom_nodes"
# The small LongCat Longform pack bundled in this repo (torch/numpy/torchaudio from
# the image, pyloudnorm from WanVideoWrapper's requirements, ffmpeg).
cp -r "$SCRIPT_DIR/custom_nodes/"* "$COMFYUI_PATH/custom_nodes/"

WRAPPER_DIR="$COMFYUI_PATH/custom_nodes/ComfyUI-WanVideoWrapper"
wrapper_at_ref() {
  [ "$(git -C "$WRAPPER_DIR" rev-parse HEAD 2>/dev/null)" = "$WANVIDEO_WRAPPER_REF" ]
}
if [ -d "$WRAPPER_DIR" ] && [ ! -d "$WRAPPER_DIR/.git" ]; then
  echo "FATAL: $WRAPPER_DIR exists but is not a git checkout; refusing to replace it."
  echo "       Move it aside and restart to install the pinned version."
  exit 1
fi
if ! wrapper_at_ref; then
  echo "Installing ComfyUI-WanVideoWrapper at ${WANVIDEO_WRAPPER_REF:0:12}..."
  if [ ! -d "$WRAPPER_DIR" ]; then
    git init -q "$WRAPPER_DIR"
    git -C "$WRAPPER_DIR" remote add origin "$WANVIDEO_WRAPPER_URL"
  elif [ -n "$(git -C "$WRAPPER_DIR" status --porcelain --untracked-files=no 2>/dev/null)" ]; then
    echo "FATAL: $WRAPPER_DIR has local changes; refusing to overwrite them."
    echo "       Commit or discard them, or set WANVIDEO_WRAPPER_REF to its current commit."
    exit 1
  fi
  fetched=0
  for attempt in 1 2 3; do
    if git -C "$WRAPPER_DIR" fetch -q --depth 1 "$WANVIDEO_WRAPPER_URL" "$WANVIDEO_WRAPPER_REF"; then
      fetched=1
      break
    fi
    sleep 5
  done
  if [ "$fetched" = 1 ]; then
    git -C "$WRAPPER_DIR" -c advice.detachedHead=false checkout -q --force FETCH_HEAD || true
  fi
fi
if ! wrapper_at_ref || [ ! -f "$WRAPPER_DIR/LongCat/nodes.py" ]; then
  echo "FATAL: ComfyUI-WanVideoWrapper is not at $WANVIDEO_WRAPPER_REF or has no"
  echo "         LongCat nodes. Check the log above for git errors and restart."
  exit 1
fi
echo "ComfyUI-WanVideoWrapper: $(git -C "$WRAPPER_DIR" rev-parse --short HEAD)"

# WanVideoWrapper's Python packages go into ComfyUI's persistent virtualenv, the
# interpreter that runs ComfyUI. A bare `pip` would install into the system Python,
# where ComfyUI never sees them. Create the venv exactly as the image's /start.sh
# does when it is missing, so start.sh then just activates it. PIP_CONSTRAINT pins
# torch/torchvision/torchaudio to the image's builds: nothing here may replace them.
VENV_DIR="$COMFYUI_PATH/${COMFYUI_VENV_NAME:-.venv-cu128}"
CONSTRAINTS=/opt/comfyui-runtime-constraints.txt
if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "Creating ComfyUI virtualenv $VENV_DIR (as the image's start.sh does)..."
  python3.12 -m venv --system-site-packages --without-pip "$VENV_DIR"
fi
VPY="$VENV_DIR/bin/python"
deps_ok() {
  "$VPY" - "$VENV_DIR" <<'PY'
import importlib, os, sys
venv = os.path.realpath(sys.argv[1])
for name in ("diffusers", "peft", "accelerate", "ftfy", "einops", "gguf",
             "pyloudnorm", "sentencepiece", "cv2", "torchaudio"):
    importlib.import_module(name)
import torch
if os.path.realpath(torch.__file__).startswith(venv + os.sep):
    sys.exit("PyTorch was installed inside the venv; it must come from the image.")
PY
}
REQS_HASH="$(sha256sum "$WRAPPER_DIR/requirements.txt" | cut -d' ' -f1)"
MARKER="$VENV_DIR/.longcat-avatar-wrapper-deps"
if [ "$(cat "$MARKER" 2>/dev/null)" = "$REQS_HASH" ] && deps_ok 2>/dev/null; then
  echo "WanVideoWrapper Python packages already installed."
else
  echo "Installing WanVideoWrapper Python packages into $VENV_DIR..."
  constraint_args=()
  if [ -f "$CONSTRAINTS" ]; then
    constraint_args=(-c "$CONSTRAINTS")
  else
    # Older images: pin to whatever PyTorch the image ships.
    CONSTRAINTS="$(mktemp /tmp/longcat-torch-pins.XXXXXX)"
    "$VPY" -c 'import importlib.metadata as m
for p in ("torch", "torchvision", "torchaudio"):
    try: print(f"{p}=={m.version(p)}")
    except m.PackageNotFoundError: pass' > "$CONSTRAINTS"
    constraint_args=(-c "$CONSTRAINTS")
  fi
  if ! "$VPY" -m pip install --no-cache-dir "${constraint_args[@]}" \
      -r "$WRAPPER_DIR/requirements.txt"; then
    echo "FATAL: installing WanVideoWrapper's requirements failed (see pip output above)."
    echo "         Restart the pod to retry."
    exit 1
  fi
  # opencv-python needs libGL, which slim images can lack.
  if ! "$VPY" -c "import cv2" 2>/dev/null && command -v apt-get >/dev/null; then
    echo "cv2 does not import; installing the system libraries OpenCV needs..."
    apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 || true
  fi
  if ! deps_ok; then
    echo "FATAL: WanVideoWrapper's packages do not import in $VENV_DIR (see above)."
    exit 1
  fi
  printf '%s\n' "$REQS_HASH" > "$MARKER"
fi

# 2. Model folders
echo "Preparing model directories..."
for d in diffusion_models loras text_encoders vae audio_encoders; do
  mkdir -p "$COMFYUI_PATH/models/$d"
done
# The batch workflows render every voiceover dropped in here, one video per file.
mkdir -p "$COMFYUI_PATH/input/batch_audio"

# -------------------------------------------------------------------
# Download helpers
# -------------------------------------------------------------------
# `hf` is only a downloader - ComfyUI never imports it - so it lives in its own venv.
# Upgrading huggingface_hub inside ComfyUI's environment breaks transformers' version
# pin, and ComfyUI then fails to start with "from transformers import CLIPTokenizer".
# Its Xet backend is much faster than wget on 30GB files but cannot resume a partial
# file, so wget stays as the fallback.
echo "Installing fast downloader (optional)..."
HF_VENV="/workspace/.hf-cli"
HF_BIN=""
if [ ! -x "$HF_VENV/bin/hf" ]; then
  if python3 -m venv "$HF_VENV" 2>/dev/null; then
    "$HF_VENV/bin/pip" install -q -U "huggingface_hub[hf-xet]" 2>/dev/null || true
  fi
fi
for cand in "$HF_VENV/bin/hf" "$HF_VENV/bin/huggingface-cli"; do
  [ -x "$cand" ] && { HF_BIN="$cand"; break; }
done
if [ -n "$HF_BIN" ]; then
  echo "Fast downloader: $HF_BIN (isolated venv)"
else
  echo "Fast downloader unavailable, using wget."
fi

# Check the safetensors header as well as a successful transfer exit status.
file_ok() {
  [ -f "$1" ] && [ "$(stat -c%s "$1")" -ge 1000000 ] || return 1
  "$PY" - "$1" <<'PY'
import json, os, struct, sys
try:
    with open(sys.argv[1], 'rb') as f:
        size = struct.unpack('<Q', f.read(8))[0]
        if not 2 <= size <= min(os.path.getsize(sys.argv[1]) - 8, 100_000_000):
            raise ValueError('invalid header length')
        header = json.loads(f.read(size))
        end = max(v['data_offsets'][1] for k, v in header.items() if k != '__metadata__')
        if end + size + 8 != os.path.getsize(sys.argv[1]):
            raise ValueError('incomplete tensor data')
except (OSError, ValueError, KeyError, TypeError, struct.error):
    sys.exit(1)
PY
}

hf_fast() {
  # $1 = destination file, $2 = path inside the repo
  local dest="$1" rpath="$2" tmp
  [ -n "$HF_BIN" ] || return 1
  # Stage beside the destination so the final move is a rename, not a 30GB copy.
  tmp="$(mktemp -d "$(dirname "$dest")/.hfdl.XXXXXX")" || return 1
  if HF_XET_HIGH_PERFORMANCE=1 "$HF_BIN" download "$HF_REPO" "$rpath" --local-dir "$tmp"; then
    file_ok "$tmp/$rpath" && mv -f "$tmp/$rpath" "$dest"
  fi
  rm -rf "$tmp"
  file_ok "$dest"
}

fetch_url() {
  # $1 = destination file, $2 = URL
  local dest="$1" url="$2" part="$1.part"
  if [ -n "$HF_TOKEN" ]; then
    wget -c -q --show-progress --tries=3 --read-timeout=120 \
      --header="Authorization: Bearer $HF_TOKEN" -O "$part" "$url" || return 1
  else
    wget -c -q --show-progress --tries=3 --read-timeout=120 -O "$part" "$url" || return 1
  fi
  file_ok "$part" && mv -f "$part" "$dest"
}

download_kj() {
  # $1 = path inside Kijai/WanVideo_comfy, $2 = ComfyUI models/ folder.
  # Files land flat in that folder under their own name, as the workflows expect.
  local rpath="$1" fname dest
  fname="$(basename "$rpath")"
  dest="$COMFYUI_PATH/models/$2/$fname"
  if file_ok "$dest"; then
    echo "$fname already exists, skipping."
    return 0
  fi
  echo "Downloading $fname ..."
  hf_fast "$dest" "$rpath" && { echo "$fname done."; return 0; }
  echo "  falling back to wget..."
  fetch_url "$dest" "https://huggingface.co/$HF_REPO/resolve/main/$rpath" || true
  if ! file_ok "$dest"; then
    echo "ERROR: could not download $fname. Existing files were retained for retry."
    return 1
  fi
  echo "$fname done."
}

# -------------------------------------------------------------------
# 3. Model downloads (all from Kijai/WanVideo_comfy, ~46 GB)
# -------------------------------------------------------------------
FAILED=0

# LongCat-Video-Avatar 1.5, single-speaker, bf16 (~31.7GB). LongCat only runs at bf16
# base precision; fp8 quantization on load is a loader option for smaller GPUs.
download_kj "LongCat/LongCat-Avatar-15_bf16.safetensors" diffusion_models || FAILED=1
# DMD2 step-distillation LoRA (~1.3GB): v1.5 is meant to run with it, 8 steps, cfg 1.
download_kj "LongCat/LongCat-Avatar-15_dmd_distill_lora_rank128_bf16.safetensors" loras || FAILED=1
# umt5-xxl text encoder (~11.4GB), Wan 2.1 VAE (~0.25GB), Whisper large-v3 encoder
# (~1.7GB, the v1.5 audio encoder).
download_kj "umt5-xxl-enc-bf16.safetensors" text_encoders || FAILED=1
download_kj "Wan2_1_VAE_bf16.safetensors" vae || FAILED=1
download_kj "HuMo/whisper_large_v3_encoder_fp16.safetensors" audio_encoders || FAILED=1

if [ "$FAILED" = "1" ]; then
  echo "FATAL: one or more model downloads failed (see ERROR lines above)."
  echo "         Restart the pod to retry - finished files are kept and skipped."
  exit 1
fi

# FP8 copy of the LongCat model (~17 GB) for the longcat_fp8_* workflows, built once
# from the BF16 download: the same float8_e4m3fn conversion WanVideoWrapper's
# fp8_e4m3fn option does on load, so FP8 workflows read half the data. Uses the image's
# PyTorch through ComfyUI's venv; streams one tensor at a time.
FP8_SOURCE="$COMFYUI_PATH/models/diffusion_models/LongCat-Avatar-15_bf16.safetensors"
FP8_MODEL="$COMFYUI_PATH/models/diffusion_models/LongCat-Avatar-15_fp8_e4m3fn.safetensors"
if file_ok "$FP8_MODEL"; then
  echo "$(basename "$FP8_MODEL") already exists, skipping."
else
  echo "Building the FP8 model from the BF16 download (a few minutes)..."
  if "$VPY" "$SCRIPT_DIR/tools/make_fp8.py" "$FP8_SOURCE" "$FP8_MODEL.part" \
      && file_ok "$FP8_MODEL.part"; then
    mv -f "$FP8_MODEL.part" "$FP8_MODEL"
    echo "FP8 model ready: $(basename "$FP8_MODEL")"
  else
    rm -f "$FP8_MODEL.part"
    echo "FATAL: building the FP8 model failed (see above). Restart the pod to retry;"
    echo "         the BF16 model and other downloads are kept."
    exit 1
  fi
fi

# 4. Install bundled workflows into ComfyUI's Workflows sidebar
echo "Installing workflows..."
WF_DEST="$COMFYUI_PATH/user/default/workflows"
mkdir -p "$WF_DEST"
# Workflows shipped under earlier names, and earlier versions of current ones. Each is
# removed only while it is byte-identical to that shipped version, so a copy edited on
# the pod is kept; a current name is then reinstalled at its new version below.
RETIRED_WORKFLOWS="
af1846410b415b004f083d3f6357450912f595db2c59f2df0343785aa4c08439 longcat_avatar_talking_head.json
3ec6605898ab35b1ca217a12804d5450bd15c315d54cf68a5d74f15abe5c7910 longcat_avatar_talking_head.json
862fca53113c2fc2bd9ec54ac35edb182b8a2e3bd1860b8276b1ea4d36132cb0 longcat_avatar_talking_head_draft.json
0649d070bb73c9192d13ddf63308639c7b4be1ddc25e6056ca1d217e37030547 longcat_avatar_talking_head_draft.json
c02021274fbe3bb5ec6aef35e18b838ff01d4677704c6d04e1c1e729051fa706 longcat_avatar_talking_head_batch.json
840769ad82021b53b1d07effc13019822a3440bc8b3ec097f738ecb13bfe51a1 longcat_avatar_talking_head_batch.json
7c9ca7ec8a1fbd3c093ba49b59fae2dd7ef81358333ad05884d8539979e76d32 longcat_avatar_talking_head_draft_batch.json
5e986c107dd281d98ba0672a24b3d9c5e9b19eb8e41935e9f8706880a09669e9 longcat_avatar_talking_head_draft_batch.json
1e6927c91855e43648110ef397398bdd11179e455b7460db7c7db230f9806b60 longcat_avatar_return_to_portrait.json
a5a57502909342733843872f796df2b101deee2b68aa6a3b1d5f213afe15094e longcat_avatar_return_to_portrait.json
95cf49958e8e4116eb8f09565c1cee9ff658ac63326ce2e39cce3779cab4399d longcat_avatar_start_to_end.json
ca1ad53684b8e5ae390fcdef8853861ef40b2f16b6e0734b97bceaa89c5ac8b1 longcat_avatar_start_to_end.json
f093027b65345532336bad2f06daa73f9886775c14e9f0b449a0baf49589dfd3 longcat_1_fast_640.json
5d5fcefee30fc615ac00eff2e1955861d3fff6798ab0786eabcdd7e33cdee0c1 longcat_2_high_quality_960.json
6de60850baca973addbdafaf9f91b5b46196f3d1dec9b20b10f2c75a920c4d8e longcat_3_fast_640_batch_folder.json
f58132330afb070245fb7b3f196fcd9639173496d40ba167c58c6149cb5b7dea longcat_4_high_quality_960_batch_folder.json
2af3f020d2886adaac0015430e969856d3f5dbbfe73adcbfe1d92ad778e0676b longcat_5_experimental_return_to_portrait_640.json
508bcb426816fe539712fb7a4a24bba6175fdc7aa3e217b74e6a46fcbfa7c782 longcat_6_experimental_start_to_end_image_640.json
d35d03a50b47cfa1448454fb8d61980eb5ef45b217eeb10be985f12a5478f5bc longcat_fp8_1_fast_640.json
c60666572973ccd8173cd78fcd1691d82eedd74b81b159118917dcedda11d32c longcat_fp8_2_high_quality_960.json
c5c7bd45f0d49ba866a5c2a5cd01c25ac7c6a6155004e95fa67c7e1657c656d6 longcat_fp8_3_fast_640_batch_folder.json
17ea1343f0fbec6e43e1bc4f0649283537d77e6b5f20e7a6d05d6099e410fda7 longcat_fp8_4_high_quality_960_batch_folder.json
d705d1e3e407a02ac8f57a027a8c2a86fc6b15a12a386679d4b65b81bdc0be60 longcat_fp8_5_experimental_return_to_portrait_640.json
80019aaf2240599dfa4ede2459d8f408498d143b5dcc3507200412b12e2f94b6 longcat_fp8_6_experimental_start_to_end_image_640.json
"
while read -r sum name; do
  [ -n "$name" ] && [ -f "$WF_DEST/$name" ] || continue
  if [ "$(sha256sum "$WF_DEST/$name" | cut -d' ' -f1)" = "$sum" ]; then
    rm -f "$WF_DEST/$name"
    echo "  removed outdated workflow: $name"
  fi
done <<< "$RETIRED_WORKFLOWS"
if [ -d "$SCRIPT_DIR/workflows" ]; then
  # Never overwrite a workflow already edited on the Network Volume.
  for workflow in "$SCRIPT_DIR/workflows/"*.json; do
    [ -e "$WF_DEST/$(basename "$workflow")" ] || cp "$workflow" "$WF_DEST/"
  done
  ls "$WF_DEST" | sed 's/^/  workflow: /'
else
  echo "No workflows directory found in repo, skipping."
fi

# 5. Start ComfyUI using the official RunPod entrypoint
echo "Setup complete! Handing over to start script..."
exec /start.sh
