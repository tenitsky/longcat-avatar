# LongCat Video Avatar - Long Talking Avatars

Turn a portrait and voiceover of any length into a talking-avatar video with LongCat-Video-Avatar 1.5 (MIT licence), stitched with your original audio.

## Storage

Attach your own volume at /workspace when deploying. Either kind works; setup detects which:

- **Network Volume** (**100 GB recommended**): everything stays on the volume. Reattach it in its datacenter when replacing a pod.
- **Global Volume** (beta, any datacenter): models, inputs, outputs and saved workflows stay on the volume; ComfyUI itself runs from the container disk and is set up again on each fresh container (a few minutes). Keep the container disk at **50 GB**: first-boot downloads are staged there. Let the first pod finish downloading before starting others.

## Connect

- **Port 8188:** ComfyUI
- **Port 8888:** JupyterLab

First boot downloads **46 GB** and builds a 17 GB FP8 copy of the model. Wait for "Setup complete!" in logs. Models are reused on later boots.

Jupyter has no login by default; anyone with its URL can run commands. For a login, set JUPYTER_NO_AUTH=0 and JUPYTER_PASSWORD.

## Generate a video

1. Open ComfyUI's Workflows sidebar.
2. Select **longcat_bf16_1_single_audio_640x640** (80 GB GPU) or the **fp8** one (32-48 GB). **3**/**4** are 960x960; **2**/**4** render every voiceover in input/batch_audio.
3. Load your portrait and voiceover (a 57 s sample is preloaded).
4. **Prompt** is optional: leave it empty, or describe the portrait in a sentence.
5. Test one chunk at batch count 1.
6. Queue roughly audio seconds divided by 3.2, plus a margin. Each item renders the next unfinished chunk, continuing from the one before.

The final video lands in ComfyUI/output. To resume, queue again.

**Against drift:** every chunk is pulled back toward the portrait, and every 40-60 s, in the middle of a real pause (mouth shut), the render restarts from the portrait itself. The cuts are planned from the voiceover up front.

## Faster on several GPUs or pods

One video can render on several GPUs at once: the render is split at the restarts into sections of about a minute, and each GPU renders its own. Use a pod with several GPUs, or several pods on the **same volume** (let the first one finish setup). Queue as usual, with a batch count of at least 2: chunk 1 renders on the first GPU alone, and the second item shares the render out ("3 sections shared across 3 workers" in the log). Each extra GPU then loads the model once, which takes a few minutes, before it starts rendering. The GPU you queued on stitches the video. Longer voiceovers gain most: about 3–4× faster for 7 minutes on 7–9 GPUs; a 2-minute one, about 2× on 3 GPUs. Check progress in /tmp/longcat-coordinator.log. LONGCAT_PARALLEL=0 turns it off.

## Hardware

**bf16** workflow: 80 GB or more VRAM, Block Swap off. **fp8** workflow: 32–48 GB, Block Swap already on (20 blocks). Not yet benchmarked on a live pod.

Full instructions:
https://github.com/tenitsky/longcat-avatar
