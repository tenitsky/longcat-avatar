# LongCat Longform

Seven nodes that turn one portrait and a long voiceover into a talking-head video with
LongCat-Video-Avatar 1.5, using ComfyUI's batch queue as the loop. They sit around
kijai/ComfyUI-WanVideoWrapper's LongCat nodes and are ported from the H3 Longform pack.

ComfyUI runs each node once per queue item, so a single node can't loop over chunks.
Instead, each queue item renders one 93-frame window, and these nodes carry the plan
from one item to the next through files on disk.

| Node | Does |
|---|---|
| **Split (next window)** | Plans the track in windows, picks the first unfinished chunk from disk, and outputs its audio (original for the chunk file, 16 kHz with context for Whisper) and every per-window setting the LongCat extend node needs |
| **Opening + Reference** | Chunk 0 and every restart: the portrait as a single opening frame, no reference. Other chunks: the previous chunk's last 13 frames, and the portrait as reference latent |
| **Write + Stitch** | Drops the lead-in, writes the chunk as mp4 and keeps its last 13 frames for the next window. On the last chunk it concatenates them all and muxes in the original track |
| **Name From Audio File** | Names the session and output after the Load Audio file |
| **Prompt (optional)** | A generic talking-head prompt, led by an optional `description` of the portrait |
| **v1.5 Distill Sigmas** | Upstream's exact 8-step DMD schedule (shift 7). The wrapper's `longcat_distill_euler` is the v1.0 one |
| **Audio From Folder (batch)** | Loads the first file in a folder (name order) whose video isn't finished. Wire `name` to Split, Opening and Write `session`, and `filename` to Write. Clears the queue when every file is done |

## Frame accounting

- A window is 93 frames at 25 fps. Chunk 0 opens on the portrait and keeps all 93.
- Every later window opens on the previous chunk's last 13 **written** frames, re-encoded
  through the VAE (`prev_images` + `vae` on the extend node, v1.5's behaviour), and keeps
  the 80 frames after them. The last chunk keeps what is left and is trimmed.
- So chunk `k > 0` covers frames `93 + 80 (k - 1)` onward, matching upstream's
  `audio_start_idx += num_frames - num_cond_frames`.

## Audio

- The whole track is downmixed, resampled to 16 kHz and measured once; one gain brings
  it to -23 LUFS, as upstream's `_loudness_norm` does on the full track. The Whisper
  node's own per-slice normalisation is switched off so the gain cannot jump between
  windows.
- Each window's Whisper slice adds 2 s of context on both sides (silence past the
  track's ends). `frames_processed` tells the extend node where the window starts in
  the slice; it subtracts `overlap` itself.
- 16 kHz / 25 fps is exactly 640 samples per frame. A 16-sample pad, and passing one
  extra frame as the Whisper node's `num_frames`, stop its float rounding from losing
  a frame.

## Reference (anchor) modes

The reference latent's RoPE time position is `ref_frame_index`, in latent frames;
frames within `ref_mask_frame_range` of it do not attend to it. The Split node sets both:

| anchor | Reference on every continuing chunk |
|---|---|
| return to portrait (default) | portrait at the last written frame's latent, mask 0 |
| identity | portrait at 10, mask 3 (upstream) |

Chunk 0 never has a reference: it is upstream's audio-image-to-video stage.

## Restarts against drift

Continuation drifts over long videos: small noise in the carried frames grows from
chunk to chunk. Before the first chunk, Split finds every real pause in the voiceover
(`min_pause_seconds`, 0.4 s, of near-silence against the recording's own floor and
speech level) and plans a restart in the middle of the longest real pause in every
`restart_min_seconds` to `restart_max_seconds` (20-40 s) window, or the next real pause
within 20 s, never in the last 3 s. The chunk starting there opens on the portrait,
exactly like chunk 0 (one conditioning frame, no reference, a whole window), and the
chunk before it ends in the pause. The plan is logged and saved as `restart_plan.txt` in
the session folder. Split's `opens_on_portrait` output tells Carry which chunks those
are. `restart_max_seconds = 0` turns restarts off.

## Resume and order

A chunk counts as done when its mp4 exists and, if the next chunk continues from it,
its tail exists.
Each queue item renders the first chunk that isn't done; the last is redone until a
stitch writes `.finished`. Chunks depend on their predecessors, so they always render
in order; there is no multi-pod split as in H3.

Delete a chunk **and every later one** to redo from it. Use a fresh session (change
`name_suffix`) when changing the canvas, audio, portrait or anchor; the Opening node
refuses a tail whose size doesn't match the canvas.

Intermediate files live in `output/longcat_longform/<session>/`.

Run local CPU checks with `python tools/test_longform.py` from the template root
(requires torch, torchaudio, numpy, pyloudnorm, ffmpeg and ffprobe).
