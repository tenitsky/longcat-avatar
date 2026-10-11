"""
LongCat Longform - turn one portrait + a long voiceover into a minutes-long talking-head
video with LongCat-Video-Avatar 1.5, using ComfyUI's own batch queue as the loop.

Same design as the H3 Longform pack this was ported from: ComfyUI's graph is a DAG, so
each queue item renders one window and these nodes carry the plan between them.

    Split  -> this window's audio (for Whisper and for the chunk file) and its settings
    Carry  -> the window's opening: the portrait, or the previous chunk's last frames
    Write  -> saves the chunk, and on the final one stitches everything together

How LongCat differs from H3:

  * Video continuation, not end anchors. LongCat-Video-Avatar generates 93-frame windows
    at 25 fps. The first window starts on the portrait; every later window starts on the
    previous chunk's last 13 frames (re-encoded, as v1.5 does) and generates 80 new ones,
    so motion carries across every join. The model cannot pin a frame at the end of a
    window: conditioning frames always sit at its start.
  * Identity reference. From the second window on, the portrait is also passed as a
    reference latent placed at a time position inside the window (ref_frame_index). The
    upstream default is position 10. The experimental anchors move it to the window's
    last frame, which is the closest LongCat gets to an end frame: a soft pull toward
    that image, not a pinned frame.
  * Audio. Whisper features are computed per window from a 16 kHz mono slice with two
    seconds of context either side, after normalising the WHOLE track to -23 LUFS, so
    the gain does not jump between windows and no 400-second cap applies.

The final mux always uses the untouched original track, never a reconstruction, so sync
is exact by construction.

Dependencies: torch, numpy and torchaudio from ComfyUI's environment, pyloudnorm (a
WanVideoWrapper requirement), and ffmpeg.
"""

import json
import os
import re
import shutil
import subprocess

import numpy as np
import torch

import folder_paths

try:
    # Returning this from a node blocks everything downstream, so a surplus queue
    # item finishes in milliseconds instead of sampling.
    from comfy_execution.graph_utils import ExecutionBlocker
except ImportError:  # pragma: no cover - older ComfyUI
    try:
        from comfy_execution.graph import ExecutionBlocker
    except ImportError:
        ExecutionBlocker = None

FPS = 25                  # LongCat-Video-Avatar 1.5 is trained at 25 fps
WINDOW = 93               # frames per generation (upstream num_frames)
OVERLAP = 13              # frames carried into the next window (num_cond_frames)
STEP = WINDOW - OVERLAP   # new frames per continuation window
REF_INDEX, REF_MASK = 10, 3   # upstream --ref_img_index / --mask_frame_range defaults
WHISPER_SR = 16000
SAMPLES_PER_FRAME = WHISPER_SR // FPS     # 640: frame and sample grids line up exactly
CONTEXT = 2 * FPS         # Whisper context either side of the window, in frames
# The Whisper node sizes its features with int(seconds * fps), which can land one frame
# short through float rounding. A few extra samples keep it on the intended count.
WHISPER_PAD = 16
TARGET_LUFS = -23.0
SESSION_ROOT = "longcat_longform"
LOG = "[LongCat Longform]"

ANCHOR_IDENTITY = "identity (default)"
ANCHOR_WINDOW_END = "return to portrait (experimental)"
ANCHOR_END_IMAGE = "end image (experimental)"
ANCHORS = [ANCHOR_IDENTITY, ANCHOR_WINDOW_END, ANCHOR_END_IMAGE]


# --------------------------------------------------------------------------- helpers
def plan_chunks(total, end_split=False):
    """Split [0, total) frames into (start_frame, keep_frames) spans.

    Chunk 0 keeps up to a whole window; every later chunk keeps up to STEP new frames
    after its OVERLAP lead-in. With end_split, a video that fits in one window is
    still made of two chunks, because only a continuation window can take a
    reference image (the end image).
    """
    if total <= 0:
        return []
    spans = [(0, min(WINDOW, total))]
    pos = spans[0][1]
    while pos < total:
        keep = min(STEP, total - pos)
        spans.append((pos, keep))
        pos += keep
    if end_split and len(spans) == 1 and total > OVERLAP:
        # The second window opens on the first chunk's last OVERLAP frames.
        first = max(OVERLAP, total - STEP)
        spans = [(0, first), (first, total - first)]
    return spans


def latent_index(frame):
    """Latent position of a window frame: frame 0 alone, then four frames per latent."""
    return 0 if frame <= 0 else (frame - 1) // 4 + 1


def ref_settings(anchor, trim, keep, is_last):
    """(ref_frame_index, ref_mask_frame_range) for one continuation window.

    The identity anchor uses upstream's defaults. The experimental anchors place the
    reference on the window's last written frame with no mask, so that frame is drawn
    toward the reference picture.
    """
    if anchor == ANCHOR_WINDOW_END or (anchor == ANCHOR_END_IMAGE and is_last):
        return latent_index(trim + keep - 1), 0
    return REF_INDEX, REF_MASK


def distill_sigmas(shift=7.0):
    """LongCat-Video-Avatar 1.5's 8-step DMD schedule, as its pipeline builds it.

    get_timesteps_sigmas(use_distill=True) for avatar-v1.5, then the shift its
    scheduler config applies (7.0), then the terminal zero. WanVideoWrapper's own
    longcat_distill_euler is the v1.0 schedule, which differs.
    """
    steps, train = 8, 1000
    picks = train - (torch.arange(1, steps + 1, dtype=torch.float32)
                     * (train // steps)).round().long()
    linear = torch.flip(torch.linspace(0, 1, train), [0])
    sigmas = torch.flip(linear[picks], [0]).float()
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    return torch.cat([sigmas, torch.zeros(1)])


def _session_dir(session):
    d = os.path.join(folder_paths.get_output_directory(), SESSION_ROOT, session)
    os.makedirs(d, exist_ok=True)
    return d


_track = {}   # one prepared track, reused while its chunks render


def _loudness_gain(mono16):
    """Gain that brings the whole track to -23 LUFS, as upstream's _loudness_norm."""
    try:
        import pyloudnorm
    except ImportError:
        print(f"{LOG} WARNING: pyloudnorm is missing, so the voiceover is not "
              f"loudness-normalised. Install WanVideoWrapper's requirements.")
        return 1.0
    loudness = pyloudnorm.Meter(WHISPER_SR).integrated_loudness(mono16.astype(np.float64))
    if not np.isfinite(loudness) or abs(loudness) > 100:
        return 1.0   # silence: upstream leaves it unchanged
    return float(10.0 ** ((TARGET_LUFS - loudness) / 20.0))


def prepared_track(audio):
    """The voiceover as 16 kHz mono float32 and its loudness gain, computed once.

    LoadAudio and the batch folder node hand every queue item the same tensor, so the
    resample and loudness pass run once per voiceover.
    """
    wav, sr = audio["waveform"], int(audio["sample_rate"])
    key = (wav.data_ptr(), tuple(wav.shape), sr, float(wav.reshape(-1)[::997].sum()))
    if key in _track:
        return _track[key]
    w = wav[0] if wav.dim() == 3 else wav
    mono = (w.mean(dim=0) if w.dim() == 2 else w).detach().cpu().float()
    if sr != WHISPER_SR:
        import torchaudio
        mono = torchaudio.functional.resample(mono, sr, WHISPER_SR)
    mono = mono.numpy()
    _track.clear()
    _track[key] = (mono, _loudness_gain(mono))
    return _track[key]


def whisper_slice(mono16, gain, first, frames):
    """Frames [first, first + frames) of the 16 kHz track, silence past either end."""
    a, b = first * SAMPLES_PER_FRAME, (first + frames) * SAMPLES_PER_FRAME
    out = np.zeros(b - a + WHISPER_PAD, dtype=np.float32)
    src = mono16[max(a, 0):min(b, len(mono16))]
    out[max(0, -a):max(0, -a) + len(src)] = src * gain
    return {"waveform": torch.from_numpy(out).reshape(1, 1, -1), "sample_rate": WHISPER_SR}


def _prompt_nodes(prompt, class_type):
    if not isinstance(prompt, dict):
        return
    for node in prompt.values():
        if isinstance(node, dict) and node.get("class_type") == class_type:
            yield node


def _audio_basename(prompt, namer_id=None, default="run1", source_title=None):
    """The LoadAudio selection, without directory or extension."""
    if not isinstance(prompt, dict):
        return default
    want = source_title.strip() if isinstance(source_title, str) else ""
    if source_title is None and namer_id is not None:
        n = prompt.get(str(namer_id)) or {}
        want = ((n.get("inputs") or {}).get("source_title") or "").strip()
    for node in _prompt_nodes(prompt, "LoadAudio"):
        if want and ((node.get("_meta") or {}).get("title") or "") != want:
            continue
        v = (node.get("inputs") or {}).get("audio")
        if isinstance(v, str) and v.strip():
            base = os.path.basename(v.strip().replace("\\", "/"))
            stem = os.path.splitext(base)[0].strip()
            if stem:
                return stem
    return default


def _session_from_prompt(prompt, default="run1"):
    """Read the session name off the Write node, so it is set in one place.

    Split and Carry need it but run before Write, so they cannot take the value as a
    link without creating a cycle.
    """
    for node in _prompt_nodes(prompt, "LongCatLongformWrite"):
        v = (node.get("inputs") or {}).get("session")
        if isinstance(v, str) and v.strip():
            return v.strip()
        # A widget converted to an input arrives as [node_id, slot]. Follow it when
        # it comes from the namer, otherwise Split would check the wrong folder.
        if isinstance(v, list) and len(v) == 2:
            src = prompt.get(str(v[0])) or {}
            if src.get("class_type") == "LongCatLongformAudioName":
                name_suffix = (src.get("inputs") or {}).get("name_suffix", "")
                return _audio_basename(prompt, v[0], default) + name_suffix
    return default


FINISHED = ".finished"          # written by Write once the stitch succeeds


def _tail_path(session, chunk_index):
    """Last written frames of a chunk, kept for the next window's lead-in."""
    return os.path.join(_session_dir(session), ".carry", f"tail_{chunk_index:04d}.npz")


def _read_tail(session, chunk_index):
    try:
        with np.load(_tail_path(session, chunk_index), allow_pickle=False) as saved:
            tail = saved["frames"]
        if (tail.ndim == 4 and tail.shape[0] >= OVERLAP
                and tail.shape[-1] == 3 and tail.dtype == np.uint8):
            return tail[-OVERLAP:]
    except (OSError, ValueError, KeyError, EOFError):
        pass
    return None


def next_chunk(session, count):
    """First chunk of a session that still needs rendering, or None when finished.

    A chunk is done when its mp4 exists and, unless it is the last, so does the tail
    the next window opens on. The last chunk is redone until a stitch has succeeded,
    so an interrupted stitch is retried rather than skipped.
    """
    sdir = _session_dir(session)
    for i in range(count):
        if not os.path.exists(os.path.join(sdir, f"chunk_{i:04d}.mp4")):
            return i
        if i < count - 1 and _read_tail(session, i) is None:
            return i
    if os.path.exists(os.path.join(sdir, FINISHED)):
        return None
    return count - 1


def _ffmpeg():
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError(
            "ffmpeg not found. On the RunPod template it is installed by setup.sh; "
            "otherwise: apt-get install -y ffmpeg")
    return exe


def _write_wav(audio, path, ff):
    wav = audio["waveform"]
    if wav.dim() == 3:
        wav = wav[0]
    arr = wav.detach().cpu().float().numpy().T  # [samples, channels]
    pcm = (arr.clip(-1, 1) * 32767).astype(np.int16)
    p = subprocess.Popen(
        [ff, "-y", "-v", "error", "-f", "s16le", "-ar",
         str(int(audio["sample_rate"])), "-ac", str(pcm.shape[1]),
         "-i", "-", path],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    _, err = p.communicate(pcm.tobytes())
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed writing audio:\n{err.decode()[-1500:]}")


def _clear_queue():
    try:
        from server import PromptServer
        PromptServer.instance.prompt_queue.wipe_queue()
        print(f"{LOG} pending queue cleared.")
    except Exception as e:
        print(f"{LOG} could not clear the queue ({e}); surplus items will be "
              f"skipped instead.")


# ----------------------------------------------------------------------------- nodes
class LongCatLongformSplit:
    """Pick the next unfinished window of a long track and slice its audio."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "anchor": (ANCHORS, {
                    "default": ANCHOR_IDENTITY,
                    "tooltip": "identity: the portrait is the identity reference at "
                               "upstream's position (frame index 10). return to portrait: "
                               "the reference sits on each window's last frame, pulling "
                               "every chunk back toward the portrait pose. end image: "
                               "like identity, but the final chunk is pulled toward the "
                               "End Image. The two experimental modes are soft; LongCat "
                               "cannot pin an end frame."}),
            },
            "optional": {
                # A socket, not a widget: the batch workflows wire the folder node's
                # name here. Unconnected, the name is read from the Write node.
                "session": ("STRING", {"forceInput": True}),
            },
            "hidden": {"prompt": "PROMPT"},
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "INT", "INT", "INT", "INT", "INT", "INT", "INT",
                    "INT", "INT", "BOOLEAN", "INT")
    RETURN_NAMES = ("audio_chunk", "whisper_audio", "whisper_frames", "num_frames",
                    "overlap", "frames_processed", "ref_frame_index",
                    "ref_mask_frame_range", "trim_start", "keep_frames", "total_chunks",
                    "use_end_image", "chunk_index")
    FUNCTION = "split"
    CATEGORY = "LongCat Longform"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # The chunk picked depends on files written by earlier queue items, and on
        # the session stored in the hidden prompt, not only on visible inputs.
        return float("nan")

    def split(self, audio, anchor=ANCHOR_IDENTITY, session=None, prompt=None):
        session = session or _session_from_prompt(prompt)
        wav, sr = audio["waveform"], int(audio["sample_rate"])
        total = int(round(wav.shape[-1] * FPS / sr))
        spans = plan_chunks(total, anchor == ANCHOR_END_IMAGE)
        if not spans:
            raise RuntimeError("The voiceover is too short to render.")

        # The browser-side counter keeps climbing through surplus queue items and
        # never learns that a render finished; the files on disk do.
        index = next_chunk(session, len(spans))
        if index is None:
            print(f"{LOG} '{session}' is finished ({len(spans)} chunks) - skipping. "
                  f"Change name_suffix, or delete output/{SESSION_ROOT}/{session}/, "
                  f"to render it again.")
            if ExecutionBlocker is None:
                raise RuntimeError(f"'{session}' is already finished.")
            return tuple(ExecutionBlocker(None) for _ in self.RETURN_TYPES)

        start, keep = spans[index]
        is_last = index == len(spans) - 1
        trim = 0 if index == 0 else OVERLAP
        window = start - trim                 # first frame of the generated window
        # Chunk 0 opens on the portrait, a single conditioning frame.
        overlap = 1 if index == 0 else OVERLAP
        ref_index, ref_mask = ref_settings(anchor, trim, keep, is_last)

        # Whisper sees two seconds either side; frames_processed tells the extend
        # node where the window starts in that slice (it subtracts the overlap).
        before = min(CONTEXT, window)
        frames = before + WINDOW + CONTEXT
        mono16, gain = prepared_track(audio)
        whisper = whisper_slice(mono16, gain, window - before, frames)

        a = int(round(start * sr / FPS))
        b = min(wav.shape[-1], int(round((start + keep) * sr / FPS)))
        chunk_audio = {"waveform": wav[..., a:b], "sample_rate": sr}

        if index == 0:
            print(f"{LOG} '{session}': {total / FPS:.1f}s, {len(spans)} chunks, "
                  f"anchor={anchor}, loudness gain {gain:.2f}")
        print(f"{LOG} chunk {index + 1}/{len(spans)}  {start / FPS:.2f}s "
              f"+{keep / FPS:.2f}s  (window from frame {window}, lead-in {trim}, "
              f"reference at latent {ref_index})")
        return (chunk_audio, whisper, frames + 1, WINDOW, overlap, before + overlap,
                ref_index, ref_mask, trim, keep, len(spans),
                anchor == ANCHOR_END_IMAGE and is_last, index)


class LongCatLongformCarry:
    """The window's opening frames and its identity reference.

    Chunk 0 opens on the portrait and has no reference, as in upstream's
    audio-image-to-video stage. Every later chunk opens on the previous chunk's last
    13 frames, with the portrait (or, on the final chunk, the end image) as reference.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "portrait": ("IMAGE",),
                "portrait_latent": ("LATENT",),
                "chunk_index": ("INT", {"default": 0, "min": 0, "max": 100000,
                                        "tooltip": "Wire from Split."}),
                "use_end_image": ("BOOLEAN", {"default": False,
                                              "tooltip": "Wire from Split."}),
            },
            "optional": {
                "end_latent": ("LATENT", {"tooltip": "Encoded End Image, used as the "
                                                     "final chunk's reference."}),
                # Unconnected, the session name is read from the Write node.
                "session": ("STRING", {"forceInput": True}),
            },
            "hidden": {"prompt": "PROMPT"},
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("prev_images", "ref_latent")
    FUNCTION = "pick"
    CATEGORY = "LongCat Longform"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # The tail file depends on the session and on files written by earlier queue
        # items. ComfyUI calls IS_CHANGED with an empty hidden prompt and without
        # linked values, so neither can be read here: always run.
        return float("nan")

    def pick(self, portrait, portrait_latent, chunk_index, use_end_image,
             end_latent=None, session=None, prompt=None):
        if chunk_index == 0:
            return (portrait[:1], None)
        tail = _read_tail(session or _session_from_prompt(prompt), chunk_index - 1)
        if tail is None:
            raise RuntimeError(
                f"Chunk {chunk_index} continues from the last {OVERLAP} frames of chunk "
                f"{chunk_index - 1}, which are missing. Queue again: chunks render in "
                f"order from the first unfinished one.")
        if tail.shape[1:3] != tuple(portrait.shape[1:3]):
            raise RuntimeError(
                f"Chunk {chunk_index - 1} was rendered at {tail.shape[2]}x{tail.shape[1]}, "
                f"but the canvas is now {portrait.shape[2]}x{portrait.shape[1]}. Change "
                f"name_suffix to start a fresh render at the new size.")
        reference = end_latent if use_end_image and end_latent is not None \
            else portrait_latent
        print(f"{LOG} chunk {chunk_index}: continuing from chunk {chunk_index - 1}, "
              f"reference: {'end image' if reference is end_latent else 'portrait'}")
        return (torch.from_numpy(tail.astype(np.float32) / 255.0), reference)


class LongCatLongformWrite:
    """Write this chunk, and on the final one stitch the whole render together.

    Takes decoded frames rather than a VIDEO so the lead-in can be dropped before
    anything is encoded.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "original_audio": ("AUDIO", {"tooltip": "The FULL track, not the "
                                                        "chunk - used for the final mux."}),
                "chunk_index": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "total_chunks": ("INT", {"default": 1, "min": 1, "max": 100000}),
                "keep_frames": ("INT", {"default": WINDOW, "min": 1, "max": 100000}),
                "session": ("STRING", {"default": "run1",
                                       "tooltip": "Folder under output/longcat_longform/. "
                                                  "Change it to start a fresh render."}),
                "filename": ("STRING", {"default": "longcat_longform_final.mp4"}),
            },
            "optional": {
                # Muxed into the per-chunk file only, so one chunk can be checked for
                # sync on its own. The finished render always uses original_audio.
                "chunk_audio": ("AUDIO", {"tooltip": "Connect Split's audio_chunk "
                                                     "to make chunks playable."}),
                "stop_when_done": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "After the final chunk, clear the pending queue. This "
                               "clears ALL pending items, including unrelated jobs."}),
                "trim_start": ("INT", {"default": 0, "min": 0, "max": 100000,
                                       "tooltip": "Wire from Split: the lead-in frames "
                                                  "to drop."}),
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "write"
    OUTPUT_NODE = True
    CATEGORY = "LongCat Longform"

    def write(self, images, original_audio, chunk_index, total_chunks, keep_frames,
              session, filename, chunk_audio=None, stop_when_done=True, trim_start=0,
              extra_pnginfo=None):
        ff = _ffmpeg()
        sdir = _session_dir(session)
        # Any chunk written makes the stitched video out of date until it is redone.
        finished = os.path.join(sdir, FINISHED)
        if os.path.exists(finished):
            os.remove(finished)

        # Drop the lead-in, then anything generated past the track's end. The chunk
        # ends up exactly as long as its audio slice.
        if images.shape[0] < trim_start + keep_frames:
            raise RuntimeError(
                f"chunk {chunk_index}: got {images.shape[0]} frames but need "
                f"{trim_start} + {keep_frames}. Is Split's num_frames wired to the "
                f"LongCat extend node?")
        frames = images[trim_start:trim_start + keep_frames]

        arr = (frames.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        n, h, w, _ = arr.shape

        # Keep this chunk's tail for the next window's lead-in, before anything else
        # can fail. Atomic rename, like the chunk itself.
        if chunk_index < total_chunks - 1:
            if n < OVERLAP:
                raise RuntimeError(f"chunk {chunk_index} keeps {n} frames, fewer than "
                                   f"the {OVERLAP} the next window opens on.")
            tail = _tail_path(session, chunk_index)
            os.makedirs(os.path.dirname(tail), exist_ok=True)
            tmp = tail[:-4] + ".writing.npz"
            np.savez_compressed(tmp, frames=arr[-OVERLAP:])
            os.replace(tmp, tail)

        chunk_mp4 = os.path.join(sdir, f"chunk_{chunk_index:04d}.mp4")
        # Temp name + atomic rename: the finished path either does not exist or is a
        # complete chunk, which is what lets its existence act as the manifest.
        chunk_tmp = os.path.join(sdir, f".writing_{chunk_index:04d}.mp4")
        cmd = [ff, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-"]
        # Frames go in on stdin, which carries one stream, so audio lands on disk
        # briefly and is deleted once muxed.
        #
        # No -shortest, here or in the final mux: the AAC stream comes out a few ms
        # short of the WAV, and -shortest then drops a video frame. One lost frame per
        # chunk shifts everything after it, so over a long render the picture drifts
        # ahead of the voice. The frames are already cut to the audio's exact length.
        chunk_wav = None
        if chunk_audio is not None:
            chunk_wav = os.path.join(sdir, f".chunk_{chunk_index:04d}.wav")
            _write_wav(chunk_audio, chunk_wav, ff)
            cmd += ["-i", chunk_wav, "-map", "0:v:0", "-map", "1:a:0",
                    "-c:a", "aac", "-b:a", "192k"]
        else:
            cmd += ["-an"]
        cmd += ["-c:v", "libx264", "-crf", "16", "-preset", "medium",
                "-pix_fmt", "yuv420p", chunk_tmp]
        try:
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            _, err = p.communicate(arr.tobytes())
            if p.returncode != 0:
                raise RuntimeError(f"ffmpeg failed writing chunk:\n{err.decode()[-1500:]}")
        finally:
            if chunk_wav and os.path.exists(chunk_wav):
                os.remove(chunk_wav)
        os.replace(chunk_tmp, chunk_mp4)
        workflow = (extra_pnginfo or {}).get("workflow")
        if isinstance(workflow, dict):
            workflow_tmp = os.path.join(sdir, f".workflow_{chunk_index:04d}.json.tmp")
            with open(workflow_tmp, "w", encoding="utf-8") as stream:
                json.dump(workflow, stream, ensure_ascii=False, indent=2)
            os.replace(workflow_tmp, os.path.join(sdir, "workflow.json"))

        msg = f"chunk {chunk_index + 1}/{total_chunks} written ({n} frames)"
        print(f"{LOG} {msg}")
        if chunk_index < total_chunks - 1:
            return (msg,)

        # Final chunk: stitch. A gap would shift everything after it out of sync.
        paths = [os.path.join(sdir, f"chunk_{i:04d}.mp4") for i in range(total_chunks)]
        missing = [i for i, p in enumerate(paths) if not os.path.exists(p)]
        if missing:
            raise RuntimeError(
                f"Cannot stitch: chunks {missing} are not on disk. Queue again to "
                f"render them in session '{session}'.")

        listfile = os.path.join(sdir, "concat.txt")
        with open(listfile, "w", encoding="utf-8") as f:
            for p in paths:
                # Relative, generated names avoid quoting user-supplied session
                # paths (apostrophes and Windows separators break concat syntax).
                f.write(f"file '{os.path.basename(p)}'\n")

        silent = os.path.join(sdir, "video_only.mp4")
        wav_path = os.path.join(sdir, ".original.wav")
        out = os.path.join(folder_paths.get_output_directory(), filename)
        try:
            subprocess.run([ff, "-y", "-v", "error", "-f", "concat", "-safe", "0",
                            "-i", listfile, "-c:v", "copy", "-an", silent], check=True)
            _write_wav(original_audio, wav_path, ff)
            subprocess.run([ff, "-y", "-v", "error", "-i", silent, "-i", wav_path,
                            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                            "-c:a", "aac", "-b:a", "192k", out],
                           check=True)
        finally:
            for tmp in (wav_path, silent, listfile):
                if os.path.exists(tmp):
                    os.remove(tmp)
        # Tails stay for a resumed render or a retry of the final chunk.

        with open(finished, "w", encoding="utf-8") as stream:
            stream.write(out + "\n")
        msg = f"FINISHED: {out}"
        print(f"{LOG} {msg}")
        if stop_when_done:
            _clear_queue()
        return (msg,)


DEFAULT_SUBJECT = "a person"
DEFAULT_BACKGROUND = "the background"
DEFAULT_DELIVERY = "a clear, calm, natural voice"


def _field(text, name):
    """A description with whitespace collapsed and an optional 'name:' tag removed.

    Multiline widgets hide their label once they hold text, so the bundled
    workflows start each value with its field name ("subject: a woman ...").
    """
    text = " ".join((text or "").split())
    if text.lower().startswith(name + ":"):
        text = text[len(name) + 1:].strip()
    return text


def talking_head_prompt(subject="", background="", delivery="", extra=""):
    """A descriptive single-shot prompt for a locked-off talking head.

    LongCat's guidance: long, concrete prompts with appearance, action and scene, and a
    clear speaking cue. The background is named so the model keeps it in place.
    """
    subject = _field(subject, "subject") or DEFAULT_SUBJECT
    background = _field(background, "background") or DEFAULT_BACKGROUND
    delivery = _field(delivery, "delivery") or DEFAULT_DELIVERY
    parts = [
        f"{subject[0].upper()}{subject[1:]} is speaking directly to the camera in "
        f"{delivery}, lips moving in sync with the speech, with natural blinking, small "
        "eyebrow movements and gentle head nods, while the shoulders and body stay in "
        "place.",
        f"Behind them, {background} stays the same throughout, with the same lighting.",
        "The camera is static on a tripod and the framing does not change.",
    ]
    extra = _field(extra, "extra")
    if extra:
        parts.append(extra)
    return " ".join(parts)


class LongCatLongformPrompt:
    """Build the prompt from a few plain descriptions of the portrait."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "subject": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Who is on screen, as seen in the portrait: e.g. 'a woman in "
                               "her thirties with shoulder-length brown hair, wearing a navy "
                               "blazer over a white shirt'."}),
                "background": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "What is behind them, concretely: e.g. 'a plain light-grey "
                               "wall with soft even lighting'."}),
                "delivery": ("STRING", {
                    "default": DEFAULT_DELIVERY,
                    "tooltip": "How they speak, completing 'speaking to the camera in ...'."}),
            },
            "optional": {
                "extra": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Optional extra sentence, e.g. 'She smiles briefly between "
                               "sentences.' Keep motion small."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "build"
    CATEGORY = "LongCat Longform"

    def build(self, subject, background, delivery, extra=""):
        return (talking_head_prompt(subject, background, delivery, extra),)


class LongCatLongformSigmas:
    """The 8-step distilled schedule LongCat-Video-Avatar 1.5 was trained for."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shift": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 20.0, "step": 0.1,
                                    "tooltip": "7.0 is the shift in v1.5's scheduler "
                                               "config."}),
            },
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "sigmas"
    CATEGORY = "LongCat Longform"

    def sigmas(self, shift=7.0):
        return (distill_sigmas(shift),)


AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus")
BATCH_SAMPLE_RATE = 48000
_decoded = {}   # one decoded file, reused while the batch renders its chunks


def _natural_key(name):
    """'voice 2.wav' before 'voice 10.wav'."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", name)]


def is_finished(session):
    """Whether Write has stitched this session's final video."""
    return os.path.exists(os.path.join(folder_paths.get_output_directory(),
                                       SESSION_ROOT, session, FINISHED))


def batch_folder(folder):
    """A folder name under ComfyUI's input directory, or an absolute path."""
    folder = (folder or "").strip() or "batch_audio"
    if os.path.isabs(folder):
        return folder
    return os.path.join(folder_paths.get_input_directory(), folder)


def batch_files(folder):
    path = batch_folder(folder)
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    files = sorted((f for f in os.listdir(path)
                    if f.lower().endswith(AUDIO_EXTENSIONS) and not f.startswith(".")
                    and os.path.isfile(os.path.join(path, f))), key=_natural_key)
    stems = {}
    for f in files:
        stems.setdefault(os.path.splitext(f)[0], []).append(f)
    clashes = [names for names in stems.values() if len(names) > 1]
    if clashes:
        # Each file's video and chunk folder are named after it.
        raise RuntimeError("Batch files must have different names before the extension: "
                           + "; ".join(", ".join(c) for c in clashes))
    return path, files


def _check_complete(path, decoded_seconds):
    """Refuse a file whose audio stops well short of the length its header states.

    A file cut short in transfer keeps its original header, so players, file
    properties and ffprobe still report the full length while only the first part
    decodes. Rendering it would quietly produce a short video. The margin is wide
    because a VBR file without a length header only has an estimated duration.
    """
    probe = shutil.which("ffprobe")
    if not probe:
        return
    p = subprocess.run([probe, "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", path], stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True)
    try:
        stated = float(p.stdout.strip())
    except ValueError:
        return
    if decoded_seconds < 0.8 * stated and stated - decoded_seconds > 10:
        raise RuntimeError(
            f"{os.path.basename(path)} is incomplete: its header says "
            f"{stated / 60:.1f} min, but the file only contains {decoded_seconds:.1f} s "
            f"of audio ({os.path.getsize(path):,} bytes). It was most likely cut short "
            f"while uploading. Upload it again (a zip is most reliable), then delete any "
            f"video already rendered from it.")


def _load_audio_file(path):
    """Decode any ffmpeg-readable file to 48 kHz stereo float."""
    key = (path, os.path.getmtime(path), os.path.getsize(path))
    if key in _decoded:
        return _decoded[key]
    p = subprocess.run([_ffmpeg(), "-v", "error", "-i", path, "-vn", "-f", "f32le",
                        "-acodec", "pcm_f32le", "-ac", "2",
                        "-ar", str(BATCH_SAMPLE_RATE), "-"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    data = np.frombuffer(p.stdout, dtype=np.float32)
    if p.returncode != 0 or not data.size:
        raise RuntimeError(f"Could not read the audio in {path}:\n"
                           f"{p.stderr.decode(errors='replace')[-1500:]}")
    waveform = torch.from_numpy(data.reshape(-1, 2).T.copy()).unsqueeze(0)
    _check_complete(path, waveform.shape[-1] / BATCH_SAMPLE_RATE)
    _decoded.clear()
    _decoded[key] = {"waveform": waveform, "sample_rate": BATCH_SAMPLE_RATE}
    return _decoded[key]


class LongCatLongformAudioFolder:
    """Feed every audio file in a folder through the render, one video per file.

    Each queue item takes the first file, in name order, whose video is not finished,
    and Split then renders that file's next unfinished chunk. When every file is
    done the pending queue is cleared, so a generous batch count is safe.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {
                    "default": "batch_audio",
                    "tooltip": "A folder inside ComfyUI/input/ (or an absolute path) "
                               "holding the voiceovers. Each file becomes its own "
                               "video, named after the file."}),
                "name_suffix": ("STRING", {
                    "default": "",
                    "tooltip": "Appended to every video and session name, e.g. _draft. "
                               "Change it to render the whole folder again."}),
                "stop_when_done": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When every file is finished, clear the pending queue. "
                               "This clears ALL pending items, including unrelated "
                               "jobs."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("audio", "name", "filename", "file_number", "total_files")
    FUNCTION = "next_file"
    CATEGORY = "LongCat Longform"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Which file is next depends on files written by earlier queue items.
        return float("nan")

    def next_file(self, folder, name_suffix="", stop_when_done=True):
        path, files = batch_files(folder)
        if not files:
            raise RuntimeError(f"No audio files in {path}. Add .wav, .mp3, .flac, .m4a, "
                               ".aac, .ogg or .opus files, then queue again.")
        for number, name in enumerate(files, 1):
            session = os.path.splitext(name)[0] + name_suffix
            if is_finished(session):
                continue
            print(f"{LOG} batch: file {number}/{len(files)} '{name}' -> "
                  f"output/{session}.mp4")
            return (_load_audio_file(os.path.join(path, name)), session,
                    session + ".mp4", number, len(files))

        print(f"{LOG} batch finished: all {len(files)} files in {path} are rendered.")
        if stop_when_done:
            _clear_queue()
        if ExecutionBlocker is None:
            raise RuntimeError("Every file in the batch folder is already rendered.")
        return tuple(ExecutionBlocker(None) for _ in self.RETURN_TYPES)


class LongCatLongformAudioName:
    """Turn the loaded audio's filename into strings for session and output name."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "suffix": ("STRING", {"default": ".mp4",
                                      "tooltip": "Appended to the filename output "
                                                 "only, normally .mp4."}),
            },
            "optional": {
                "source_title": ("STRING", {
                    "default": "",
                    "tooltip": "Title of the Load Audio node to read, if the graph "
                               "has more than one. Blank uses the first found."}),
                "name_suffix": ("STRING", {
                    "default": "",
                    "tooltip": "Appended to both the session and output name. "
                               "Use _draft to keep draft renders separate."}),
            },
            "hidden": {"prompt": "PROMPT"},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("name", "filename")
    FUNCTION = "derive"
    CATEGORY = "LongCat Longform"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # The name comes from the Load Audio node, which is not an input of this
        # one. ComfyUI calls IS_CHANGED with an empty hidden prompt, so a cached name
        # would outlive an audio swap and write a new voiceover's chunks into the
        # previous render's folder. Always run; it is a string.
        return float("nan")

    def derive(self, suffix, source_title="", prompt=None, name_suffix=""):
        name = _audio_basename(prompt, source_title=source_title) + name_suffix
        print(f"{LOG} run name from audio file: {name}")
        return (name, name + suffix)


NODE_CLASS_MAPPINGS = {
    "LongCatLongformSplit": LongCatLongformSplit,
    "LongCatLongformCarry": LongCatLongformCarry,
    "LongCatLongformWrite": LongCatLongformWrite,
    "LongCatLongformAudioName": LongCatLongformAudioName,
    "LongCatLongformPrompt": LongCatLongformPrompt,
    "LongCatLongformSigmas": LongCatLongformSigmas,
    "LongCatLongformAudioFolder": LongCatLongformAudioFolder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LongCatLongformSplit": "LongCat Longform: Split (next window)",
    "LongCatLongformCarry": "LongCat Longform: Opening + Reference",
    "LongCatLongformWrite": "LongCat Longform: Write + Stitch",
    "LongCatLongformAudioName": "LongCat Longform: Name From Audio File",
    "LongCatLongformPrompt": "LongCat Longform: Talking-Head Prompt",
    "LongCatLongformSigmas": "LongCat Longform: v1.5 Distill Sigmas",
    "LongCatLongformAudioFolder": "LongCat Longform: Audio From Folder (batch)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
