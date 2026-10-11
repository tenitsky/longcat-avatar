# LongCat Video Avatar - Long Talking Avatars

Turn a portrait and voiceover of any length into a talking-avatar video with LongCat-Video-Avatar 1.5 (MIT licence), stitched with your original audio.

## Storage

Attach your own **Network Volume at /workspace** when deploying (**100 GB recommended**). The container disk is 5 GB and the Volume disk 0 GB.

ComfyUI, models, uploads, workflows and renders stay on the Network Volume. Reattach the same volume in its datacenter when replacing a pod.

## Connect

- **Port 8188:** ComfyUI
- **Port 8888:** JupyterLab

First boot downloads **46 GB**. Wait for "Setup complete!" in logs. Models are reused on later boots.

Jupyter has no login by default; anyone with its URL can run commands. For a login, set JUPYTER_NO_AUTH=0 and JUPYTER_PASSWORD.

## Generate a video

1. Open ComfyUI's Workflows sidebar.
2. Select **longcat_1_fast_640** to test, then **longcat_2_high_quality_960** for the final video.
3. Load your portrait and voiceover.
4. **Prompt** is optional: leave it empty, or describe the portrait in a sentence.
5. Test one chunk at batch count 1.
6. Queue roughly audio seconds divided by 3.2, plus a margin. Each item renders the next unfinished chunk, continuing from the one before.

The final video lands in ComfyUI/output. Next video: load its voiceover and queue. To resume, queue again.

**Batch folder:** workflows **3** (fast) and **4** (high quality) render every voiceover in ComfyUI/input/batch_audio/, one video each.

**Experimental:** LongCat has no end-frame input. Workflow **5** (return to portrait) pulls each chunk back toward the portrait; workflow **6** (start to end image) steers the final frames toward a second image.

## Hardware

Start with **80 GB or more VRAM**. On 48 GB, enable the Block Swap node. Not yet benchmarked on a live pod.

Full instructions:
https://github.com/tenitsky/longcat-avatar
