# LongCat Video Avatar - Long Talking Avatars

Turn a portrait and voiceover of any length into a talking-avatar video with LongCat-Video-Avatar 1.5 (MIT licence), stitched with your original audio.

## Storage

Attach your own **Network Volume at /workspace** when deploying (**100 GB recommended**). The container disk is 5 GB and the Volume disk 0 GB.

ComfyUI, models, uploads, workflows and renders stay on the Network Volume. Reattach the same volume in its datacenter when replacing a pod.

## Connect

- **Port 8188:** ComfyUI
- **Port 8888:** JupyterLab

First boot downloads **46 GB** and builds a 17 GB FP8 copy of the model. Wait for "Setup complete!" in logs. Models are reused on later boots.

Jupyter has no login by default; anyone with its URL can run commands. For a login, set JUPYTER_NO_AUTH=0 and JUPYTER_PASSWORD.

## Generate a video

1. Open ComfyUI's Workflows sidebar.
2. Select **longcat_bf16_1_fast_640** to test, then **longcat_bf16_2_high_quality_960** for the final video. Every workflow also has an **fp8** version (longcat_fp8_…): half the VRAM for the model.
3. Load your portrait and voiceover.
4. **Prompt** is optional: leave it empty, or describe the portrait in a sentence.
5. Test one chunk at batch count 1.
6. Queue roughly audio seconds divided by 3.2, plus a margin. Each item renders the next unfinished chunk, continuing from the one before.

The final video lands in ComfyUI/output, named after the audio plus precision and size (e.g. talk_bf16_960.mp4). Next video: load its voiceover and queue. To resume, queue again.

**Batch folder:** workflows **3** (fast) and **4** (high quality) render every voiceover in ComfyUI/input/batch_audio/, one video each.

**Experimental:** LongCat has no end-frame input. Workflow **5** (return to portrait) pulls each chunk back toward the portrait; workflow **6** (start to end image) steers the final frames toward a second image.

## Hardware

**80 GB or more VRAM** runs every workflow. On 32–48 GB, use the **fp8** workflows and enable the Block Swap node if needed. Not yet benchmarked on a live pod.

Full instructions:
https://github.com/tenitsky/longcat-avatar
