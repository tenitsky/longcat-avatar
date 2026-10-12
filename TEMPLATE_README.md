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
2. Select **longcat_bf16_1_fast_640** to test, then **longcat_bf16_2_high_quality_960** for the final video. On smaller GPUs use the **longcat_fp8_…** versions.
3. Load your portrait and voiceover.
4. **Prompt** is optional: leave it empty, or describe the portrait in a sentence.
5. Test one chunk at batch count 1.
6. Queue roughly audio seconds divided by 3.2, plus a margin. Each item renders the next unfinished chunk, continuing from the one before.

The final video lands in ComfyUI/output (e.g. talk_bf16_960.mp4). Next video: load its voiceover and queue. To resume, queue again.

**Batch folder:** workflows **3** (fast) and **4** (high quality) render every voiceover in ComfyUI/input/batch_audio/, one video each.

**Experimental:** LongCat has no end-frame input. Workflow **5** pulls each chunk back toward the portrait; **6** steers the ending toward a second image.

## Hardware

**bf16** workflows: 80 GB or more VRAM, Block Swap off. **fp8** workflows: 32–48 GB, Block Swap already on (20 blocks). Not yet benchmarked on a live pod.

Full instructions:
https://github.com/tenitsky/longcat-avatar
