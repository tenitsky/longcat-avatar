# LongCat Video Avatar - Long Talking Avatars

Turn a portrait and voiceover of any length into a talking-avatar video with LongCat-Video-Avatar 1.5 (MIT licence), stitched with your original audio.

## Storage

Attach your own **Network Volume at /workspace** when deploying (**100 GB recommended**). The container disk is 5 GB and the Volume disk 0 GB.

Everything stays on the volume. Reattach it in its datacenter when replacing a pod.

## Connect

- **Port 8188:** ComfyUI
- **Port 8888:** JupyterLab

First boot downloads **46 GB** and builds a 17 GB FP8 copy of the model. Wait for "Setup complete!" in logs. Models are reused on later boots.

Jupyter has no login by default; anyone with its URL can run commands. For a login, set JUPYTER_NO_AUTH=0 and JUPYTER_PASSWORD.

## Generate a video

1. Open ComfyUI's Workflows sidebar.
2. Select **longcat_bf16_return_to_portrait_640** (80 GB GPU) or **longcat_fp8_return_to_portrait_640** (32-48 GB).
3. Load your portrait and voiceover (a 71 s sample is preloaded).
4. **Prompt** is optional: leave it empty, or describe the portrait in a sentence.
5. Test one chunk at batch count 1.
6. Queue roughly audio seconds divided by 3.2, plus a margin. Each item renders the next unfinished chunk, continuing from the one before.

The final video lands in ComfyUI/output. To resume, queue again.

**Against drift:** every chunk is pulled back toward the portrait, and every 30-40 s, at a pause, the render restarts from the portrait itself: a small jump at the pause, in exchange for a face that stays true.

## Hardware

**bf16** workflow: 80 GB or more VRAM, Block Swap off. **fp8** workflow: 32–48 GB, Block Swap already on (20 blocks). Not yet benchmarked on a live pod.

Full instructions:
https://github.com/tenitsky/longcat-avatar
