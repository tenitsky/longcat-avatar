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
    upstream default is position 10. The return-to-portrait anchor moves it to the
    window's last frame, which is the closest LongCat gets to an end frame: a soft pull
    toward that image, not a pinned frame.
  * Restarts against drift. Continuation alone drifts over long videos: small noise
    in the carried frames grows from chunk to chunk. So once in every 40-60 s a chunk
    opens on the portrait again instead of the previous frames, exactly as chunk 0
    does. The restarts are planned from the whole voiceover before rendering: each
    goes in the middle of the longest real pause (0.4 s+ of near-silence, mouth shut)
    in its window, or the next real pause within 20 s, never mid-word if a pause
    exists. Each is a small jump at a pause, for a face that keeps returning to the
    portrait.
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
    from . import parallel
except ImportError:  # loaded as a plain module, as the tests do
    import importlib.util as _importlib_util
    _spec = _importlib_util.spec_from_file_location(
        "longcat_longform_parallel",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "parallel.py"))
    parallel = _importlib_util.module_from_spec(_spec)
    _spec.loader.exec_module(parallel)

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
ANCHOR_WINDOW_END = "return to portrait"
ANCHORS = [ANCHOR_IDENTITY, ANCHOR_WINDOW_END]

# Restarts from the portrait, against drift: one in every RESTART_MIN-RESTART_MAX
# seconds, placed in the middle of a real pause in the speech (mouth shut), found in
# the whole voiceover before rendering starts.
RESTART_MIN, RESTART_MAX = 40.0, 60.0
MIN_PAUSE = 0.4           # seconds of near-silence that count as a real pause
PAUSE_SEARCH_EXTRA = 20.0  # look this much past RESTART_MAX for a real pause
RESTART_TAIL = 3 * FPS    # no restart in the last 3 s: not worth a jump


# --------------------------------------------------------------------------- helpers
def frame_loudness(mono16, total):
    """Loudness of every video frame in dB (RMS of its 640 samples at 16 kHz)."""
    rms = np.zeros(total, dtype=np.float64)
    usable = min(total, len(mono16) // SAMPLES_PER_FRAME)
    if usable:
        audio = mono16[:usable * SAMPLES_PER_FRAME].astype(np.float64)
        rms[:usable] = np.sqrt((audio.reshape(usable, SAMPLES_PER_FRAME) ** 2).mean(1))
    return 20 * np.log10(rms + 1e-9)


def find_pauses(mono16, total):
    """Every stretch of near-silence as (first_frame, end_frame), end exclusive.

    The threshold adapts to the recording: 30% of the way from its noise floor (5th
    percentile) to its speech level (95th), and at least 20 dB under the speech, so
    room tone counts as silence and soft syllables do not. A single louder frame
    inside a pause (a click, a breath) does not split it.
    """
    db = frame_loudness(mono16, total)
    if total == 0:
        return []
    floor, speech = np.percentile(db, 5), np.percentile(db, 95)
    threshold = min(floor + 0.3 * (speech - floor), speech - 20.0)
    quiet = db < threshold
    quiet[1:-1] |= quiet[:-2] & quiet[2:]          # bridge one-frame bumps
    pauses, start = [], None
    for i, q in enumerate(quiet):
        if q and start is None:
            start = i
        elif not q and start is not None:
            pauses.append((start, i))
            start = None
    if start is not None:
        pauses.append((start, total))
    return pauses


def plan_restarts(mono16, total, min_seconds=RESTART_MIN, max_seconds=RESTART_MAX,
                  min_pause=MIN_PAUSE, details=False):
    """Frames where the render starts again from the portrait, planned from the whole
    voiceover before any chunk renders.

    Walking from the start, each restart goes in the middle of the longest real pause
    (at least min_pause seconds of near-silence, mouth shut) between min_seconds and
    max_seconds after the previous restart. With no such pause in that window, the
    first real pause in the next PAUSE_SEARCH_EXTRA seconds is used instead of cutting
    into speech; only if there is none either does the longest shorter gap in the
    window get it, and as a last resort its quietest frame. max_seconds <= 0 turns
    restarts off. details=True returns (frame, pause_seconds, how) tuples.
    """
    if max_seconds <= 0 or total <= 0:
        return []
    lo, hi = sorted((max(1.0, float(min_seconds)), float(max_seconds)))
    lo_f, hi_f = int(round(lo * FPS)), int(round(hi * FPS))
    extra_f = int(round(PAUSE_SEARCH_EXTRA * FPS))
    need = max(1, int(round(float(min_pause) * FPS)))
    db = frame_loudness(mono16, total)
    pauses = find_pauses(mono16, total)
    end_limit = total - RESTART_TAIL
    plan, last = [], 0

    def mid(p):
        return (p[0] + p[1]) // 2

    while True:
        first, end = last + lo_f, min(last + hi_f, end_limit)
        if first >= end:
            break
        in_window = [p for p in pauses if first <= mid(p) <= end]
        real = [p for p in in_window if p[1] - p[0] >= need]
        later = [p for p in pauses if end < mid(p) <= min(last + hi_f + extra_f, end_limit)
                 and p[1] - p[0] >= need]
        if real:
            pick, how = max(real, key=lambda p: (p[1] - p[0], -db[p[0]:p[1]].mean())), "pause"
        elif later:
            pick, how = later[0], "pause after the window"
        elif in_window:
            pick, how = max(in_window, key=lambda p: p[1] - p[0]), "short gap"
        else:
            frame = first + int(np.argmin(db[first:end + 1]))
            pick, how = (frame, frame + 1), "no gap: quietest frame"
        last = mid(pick)
        plan.append((last, (pick[1] - pick[0]) / FPS, how))
    return plan if details else [f for f, _, _ in plan]


def plan_chunks(total, restarts=()):
    """Split [0, total) frames into (start_frame, keep_frames, opens_on_portrait) spans.

    The video is cut into segments at each restart frame. A segment's first chunk opens
    on the portrait and keeps up to a whole window, as chunk 0 does; every later chunk
    keeps up to STEP new frames after its OVERLAP lead-in from the chunk before.
    """
    if total <= 0:
        return []
    bounds = [0] + sorted(r for r in set(restarts) if 0 < r < total) + [total]
    spans = []
    for seg_start, seg_end in zip(bounds, bounds[1:]):
        keep = min(WINDOW, seg_end - seg_start)
        spans.append((seg_start, keep, True))
        pos = seg_start + keep
        while pos < seg_end:
            keep = min(STEP, seg_end - pos)
            spans.append((pos, keep, False))
            pos += keep
    return spans


def latent_index(frame):
    """Latent position of a window frame: frame 0 alone, then four frames per latent."""
    return 0 if frame <= 0 else (frame - 1) // 4 + 1


def ref_settings(anchor, trim, keep):
    """(ref_frame_index, ref_mask_frame_range) for one continuation window.

    The identity anchor uses upstream's defaults. Return to portrait places the
    reference on the window's last written frame with no mask, so every chunk ends
    drawn toward the portrait pose.
    """
    if anchor == ANCHOR_WINDOW_END:
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


FINISHED = parallel.FINISHED    # written once the stitch succeeds


def _write_setting(prompt, key):
    """A widget value of the Write node, or None when it is linked or missing."""
    for node in _prompt_nodes(prompt, "LongCatLongformWrite"):
        value = (node.get("inputs") or {}).get(key)
        if not isinstance(value, list):
            return value
    return None


def _filename_from_prompt(prompt, session):
    """The final video's name, as the Write node will get it."""
    for node in _prompt_nodes(prompt, "LongCatLongformWrite"):
        value = (node.get("inputs") or {}).get("filename")
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and len(value) == 2:
            src = prompt.get(str(value[0])) or {}
            if src.get("class_type") == "LongCatLongformAudioName":
                suffix = (src.get("inputs") or {}).get("suffix", ".mp4")
                return session + (suffix if isinstance(suffix, str) else ".mp4")
    return session + ".mp4"


def _pin(extra_pnginfo):
    """The session a coordinator queue item is pinned to, or None for a user's item."""
    if isinstance(extra_pnginfo, dict) and isinstance(extra_pnginfo.get(parallel.PIN_KEY), dict):
        return extra_pnginfo[parallel.PIN_KEY].get("session") or ""
    return None


def _load_job(sdir):
    try:
        return parallel.load_job(sdir)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"{parallel.job_path(sdir)} cannot be read ({exc}). Delete it "
                           f"to render this video on one GPU.") from exc


def _tail_path(session, chunk_index):
    """Last written frames of a chunk, kept for the next window's lead-in."""
    return parallel.tail_path(_session_dir(session), chunk_index)


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


def next_chunk(session, spans, allowed=None):
    """First chunk of a session that still needs rendering, or None when finished.

    A chunk is done when its mp4 exists and, if the next chunk continues from it, so
    does the tail that chunk opens on (a chunk before a restart needs none). The last
    chunk is redone until a stitch has succeeded, so an interrupted stitch is retried
    rather than skipped. With `allowed` (a shared render: this worker's chunks), only
    those are considered and None means they are all done.
    """
    sdir = _session_dir(session)
    count = len(spans)
    for i in (range(count) if allowed is None else sorted(allowed)):
        if not os.path.exists(parallel.chunk_path(sdir, i)):
            return i
        if i < count - 1 and not spans[i + 1][2] and _read_tail(session, i) is None:
            return i
    if allowed is not None or os.path.exists(os.path.join(sdir, FINISHED)):
        return None
    return count - 1


def share_render(session, spans, key, index, prompt=None, workflow=None):
    """Share a render out to the other live workers, or None to keep it on this one.

    Only from the second chunk on, so a one-chunk test stays on this GPU, and only
    while this worker's own coordinator is running (it feeds the queue that stitches).
    """
    if not parallel.enabled() or index is None or index == 0:
        return None
    sdir = _session_dir(session)
    me = parallel.worker_id()
    live = parallel.live_workers(os.path.dirname(sdir))
    if me not in live or len(live) < 2:
        return None
    secs = parallel.sections(spans)
    work = parallel.section_work(sdir, spans, secs)
    if sum(1 for w in work if w) < 2:
        return None
    current = next(s for s, (a, b) in enumerate(secs) if a <= index < b)
    assignment = parallel.assign(work, live, me, fixed={current: me})
    if len(assignment) < 2:
        return None
    job = parallel.make_job(session, me, key, spans, secs, assignment, prompt, workflow,
                            _filename_from_prompt(prompt, session),
                            _write_setting(prompt, "stop_when_done") is not False)
    parallel.save_job(sdir, job)
    owners = ", ".join(f"{w}: section(s) {', '.join(str(s + 1) for s in ss)}"
                       for w, ss in assignment.items())
    print(f"{LOG} '{session}': {len(secs)} sections shared across {len(assignment)} "
          f"workers ({owners}). This worker ({me}) leads and stitches.")
    return job


def next_shared_chunk(session, spans, job, audio, prompt=None, pinned=False):
    """This worker's next chunk of a shared render, or None when it has nothing to do.

    When every chunk is on disk, the leader stitches here: the queue item that
    rendered the last chunk may belong to another worker.
    """
    sdir = _session_dir(session)
    me = parallel.worker_id()
    if not pinned and job["leader"] != me:
        live = parallel.live_workers(os.path.dirname(sdir))
        if job["leader"] not in live:
            # Queued again after the leading pod went away: take over.
            parallel.reassign(sdir, job, live, leader=me)
            parallel.save_job(sdir, job)
            print(f"{LOG} '{session}': the previous leader is gone; {me} takes over.")
    index = next_chunk(session, spans, parallel.owned_chunks(job, me))
    if index is not None:
        return index
    left = next_chunk(session, spans, range(len(spans)))
    if left is not None:
        owners = sorted(w for w, ss in job["assignment"].items()
                        if any(a <= left < b for a, b in (job["sections"][s] for s in ss)))
        print(f"{LOG} '{session}': this worker's sections are done; waiting for "
              f"{', '.join(owners) or 'a worker'} (chunk {left + 1} is next).")
        return None
    if os.path.exists(os.path.join(sdir, FINISHED)):
        return None
    if job["leader"] != me:
        print(f"{LOG} '{session}': every chunk is done; {job['leader']} stitches.")
        return None
    out = stitch(session, len(spans), audio,
                 job.get("filename") or _filename_from_prompt(prompt, session))
    try:
        os.remove(parallel.job_path(sdir))
    except OSError:
        pass
    print(f"{LOG} FINISHED: {out} (rendered on {len(job['assignment'])} workers)")
    if job.get("stop_when_done", True):
        _clear_queue()
    return None


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


def stitch(session, total_chunks, original_audio, filename):
    """Join every chunk in order under the untouched original track; returns the path.

    A gap would shift everything after it out of sync, so a missing chunk stops it.
    """
    ff = _ffmpeg()
    sdir = _session_dir(session)
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
    with open(os.path.join(sdir, FINISHED), "w", encoding="utf-8") as stream:
        stream.write(out + "\n")
    return out


# ----------------------------------------------------------------------------- nodes
class LongCatLongformSplit:
    """Pick the next unfinished window of a long track and slice its audio."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "anchor": (ANCHORS, {
                    "default": ANCHOR_WINDOW_END,
                    "tooltip": "identity: the portrait is the identity reference at "
                               "upstream's position (frame index 10). return to portrait: "
                               "the reference sits on each window's last frame, pulling "
                               "every chunk back toward the portrait pose."}),
                "restart_min_seconds": ("FLOAT", {
                    "default": RESTART_MIN, "min": 1.0, "max": 600.0, "step": 1.0,
                    "tooltip": "Against drift, the render starts again from the "
                               "portrait once in every restart_min to restart_max "
                               "seconds, in the middle of the longest real pause in the "
                               "speech in that range (mouth shut), where the jump shows "
                               "least. Planned from the whole voiceover up front."}),
                "restart_max_seconds": ("FLOAT", {
                    "default": RESTART_MAX, "min": 0.0, "max": 600.0, "step": 1.0,
                    "tooltip": "Latest restart after the previous one; with no real "
                               "pause by then, the next one within 20 s is used instead "
                               "of cutting into speech. 0 turns restarts off."}),
                "min_pause_seconds": ("FLOAT", {
                    "default": MIN_PAUSE, "min": 0.1, "max": 3.0, "step": 0.05,
                    "tooltip": "How long a stretch of near-silence must be to count as a "
                               "real pause for a restart."}),
            },
            "optional": {
                # A socket, not a widget: the batch workflows wire the folder node's
                # name here. Unconnected, the name is read from the Write node.
                "session": ("STRING", {"forceInput": True}),
            },
            # extra_pnginfo carries the UI workflow, and a coordinator's pin to a session.
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "INT", "INT", "INT", "INT", "INT", "INT", "INT",
                    "INT", "INT", "BOOLEAN", "INT")
    RETURN_NAMES = ("audio_chunk", "whisper_audio", "whisper_frames", "num_frames",
                    "overlap", "frames_processed", "ref_frame_index",
                    "ref_mask_frame_range", "trim_start", "keep_frames", "total_chunks",
                    "opens_on_portrait", "chunk_index")
    FUNCTION = "split"
    CATEGORY = "LongCat Longform"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # The chunk picked depends on files written by earlier queue items, and on
        # the session stored in the hidden prompt, not only on visible inputs.
        return float("nan")

    def _skip(self):
        if ExecutionBlocker is None:
            raise RuntimeError("Nothing to render for this queue item.")
        return tuple(ExecutionBlocker(None) for _ in self.RETURN_TYPES)

    def split(self, audio, anchor=ANCHOR_WINDOW_END, restart_min_seconds=RESTART_MIN,
              restart_max_seconds=RESTART_MAX, min_pause_seconds=MIN_PAUSE, session=None,
              prompt=None, extra_pnginfo=None):
        session = session or _session_from_prompt(prompt)
        wav, sr = audio["waveform"], int(audio["sample_rate"])
        total = int(round(wav.shape[-1] * FPS / sr))
        mono16, gain = prepared_track(audio)
        plan = plan_restarts(mono16, total, restart_min_seconds, restart_max_seconds,
                             min_pause_seconds, details=True)
        restarts = [f for f, _, _ in plan]
        spans = plan_chunks(total, restarts)
        if not spans:
            raise RuntimeError("The voiceover is too short to render.")

        # Shared across several GPUs or pods (parallel.py), or rendered here alone.
        pinned = _pin(extra_pnginfo)
        key = parallel.plan_key(total, restarts)
        job = _load_job(_session_dir(session))
        if job is not None and job.get("plan") != key:
            print(f"{LOG} '{session}': ignoring the shared plan in job.json, made for "
                  f"different restart settings.")
            job = None
        if pinned is not None and (job is None or pinned != session):
            print(f"{LOG} coordinator item for '{pinned}': nothing to do here.")
            return self._skip()
        if job is None:
            # The browser-side counter keeps climbing through surplus queue items and
            # never learns that a render finished; the files on disk do.
            index = next_chunk(session, spans)
            workflow = (extra_pnginfo or {}).get("workflow") if isinstance(
                extra_pnginfo, dict) else None
            job = share_render(session, spans, key, index, prompt, workflow)
        if job is not None:
            index = next_shared_chunk(session, spans, job, audio, prompt,
                                      pinned=pinned is not None)
            if index is None:
                return self._skip()
        elif index is None:
            print(f"{LOG} '{session}' is finished ({len(spans)} chunks) - skipping. "
                  f"Change name_suffix, or delete output/{SESSION_ROOT}/{session}/, "
                  f"to render it again.")
            if ExecutionBlocker is None:
                raise RuntimeError(f"'{session}' is already finished.")
            return tuple(ExecutionBlocker(None) for _ in self.RETURN_TYPES)

        start, keep, on_portrait = spans[index]
        # A chunk that opens on the portrait (chunk 0 and every restart) has a single
        # conditioning frame and no lead-in; the rest open on the last OVERLAP frames.
        trim = 0 if on_portrait else OVERLAP
        window = start - trim                 # first frame of the generated window
        overlap = 1 if on_portrait else OVERLAP
        ref_index, ref_mask = ref_settings(anchor, trim, keep)

        # Whisper sees two seconds either side; frames_processed tells the extend
        # node where the window starts in that slice (it subtracts the overlap).
        before = min(CONTEXT, window)
        frames = before + WINDOW + CONTEXT
        whisper = whisper_slice(mono16, gain, window - before, frames)

        a = int(round(start * sr / FPS))
        b = min(wav.shape[-1], int(round((start + keep) * sr / FPS)))
        chunk_audio = {"waveform": wav[..., a:b], "sample_rate": sr}

        if index == 0:
            lines = [f"{f / FPS:7.2f}s  pause {length:.2f}s  ({how})"
                     for f, length, how in plan]
            print(f"{LOG} '{session}': {total / FPS:.1f}s, {len(spans)} chunks, "
                  f"anchor={anchor}, loudness gain {gain:.2f}, {len(plan)} restart(s) "
                  f"from the portrait:" + "".join(f"\n{LOG}   {l}" for l in lines))
            with open(os.path.join(_session_dir(session), "restart_plan.txt"), "w",
                      encoding="utf-8") as f:
                f.write("Restarts from the portrait (time in the voiceover, pause length, "
                        "how chosen):\n" + "".join(l + "\n" for l in lines))
        what = "opens on the portrait" if on_portrait else f"lead-in {trim}"
        print(f"{LOG} chunk {index + 1}/{len(spans)}  {start / FPS:.2f}s "
              f"+{keep / FPS:.2f}s  (window from frame {window}, {what}, "
              f"reference at latent {ref_index})")
        return (chunk_audio, whisper, frames + 1, WINDOW, overlap, before + overlap,
                ref_index, ref_mask, trim, keep, len(spans), on_portrait, index)


class LongCatLongformCarry:
    """The window's opening frames and its identity reference.

    Chunk 0, and every restart against drift, opens on the portrait with no reference,
    as in upstream's audio-image-to-video stage. Every other chunk opens on the
    previous chunk's last 13 frames, with the portrait as reference.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "portrait": ("IMAGE",),
                "portrait_latent": ("LATENT",),
                "chunk_index": ("INT", {"default": 0, "min": 0, "max": 100000,
                                        "tooltip": "Wire from Split."}),
                "opens_on_portrait": ("BOOLEAN", {"default": False,
                                                  "tooltip": "Wire from Split."}),
            },
            "optional": {
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

    def pick(self, portrait, portrait_latent, chunk_index, opens_on_portrait=False,
             session=None, prompt=None):
        if chunk_index == 0 or opens_on_portrait:
            if chunk_index:
                print(f"{LOG} chunk {chunk_index}: restarting from the portrait")
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
        print(f"{LOG} chunk {chunk_index}: continuing from chunk {chunk_index - 1}")
        return (torch.from_numpy(tail.astype(np.float32) / 255.0), portrait_latent)


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
        # can fail. Atomic rename, like the chunk itself. Only a segment's last chunk
        # can be shorter than the lead-in, and the chunk after it restarts from the
        # portrait, so it needs no tail.
        if chunk_index < total_chunks - 1 and n >= OVERLAP:
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

        # Final chunk: stitch. A shared render is stitched by its leader instead,
        # once every worker's sections are on disk.
        if _load_job(sdir) is not None:
            return (msg + "; the leader stitches the shared render",)
        out = stitch(session, total_chunks, original_audio, filename)
        msg = f"FINISHED: {out}"
        print(f"{LOG} {msg}")
        if stop_when_done:
            _clear_queue()
        return (msg,)


def talking_head_prompt(description=""):
    """A single-shot prompt for a locked-off talking head.

    Identity and background come from the portrait, so the fixed part only asks for
    speech, small natural motion and a static scene. An optional description of the
    portrait leads the prompt; LongCat follows concrete descriptions best.
    """
    description = " ".join((description or "").split())
    parts = [description] if description else []
    parts += [
        f"{'The' if description else 'A'} person is speaking directly to the camera in a "
        "clear, calm, natural voice, lips moving in sync with the speech, with natural "
        "blinking, small eyebrow movements and gentle head nods, while the shoulders and "
        "body stay in place.",
        "The background and lighting stay the same throughout.",
        "The camera is static on a tripod and the framing does not change.",
    ]
    return " ".join(parts)


class LongCatLongformPrompt:
    """The talking-head prompt, with an optional description of the portrait."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "description": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Optional. Leave empty for a generic talking-head prompt, "
                               "or describe the portrait: e.g. 'A woman with brown hair in "
                               "a navy blazer, in front of a plain grey wall.'"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "build"
    CATEGORY = "LongCat Longform"

    def build(self, description=""):
        return (talking_head_prompt(description),)


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


def waiting_on_others(session, worker=None):
    """True while a shared render needs nothing more from this worker for now: its own
    sections are done and the rest are with other workers (or another worker
    stitches). The batch folder moves on to the next file meanwhile."""
    sdir = os.path.join(folder_paths.get_output_directory(), SESSION_ROOT, session)
    try:
        job = parallel.load_job(sdir)
    except (OSError, ValueError):
        return False
    if not job:
        return False
    worker = worker or parallel.worker_id()
    if parallel.remaining(sdir, job["spans"], parallel.owned_chunks(job, worker)):
        return False
    return not (job["leader"] == worker and parallel.all_done(sdir, job["spans"]))


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
                    "tooltip": "Appended to every video and session name, e.g. _fast. "
                               "Change it to render the whole folder again."}),
                "stop_when_done": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When every file is finished, clear the pending queue. "
                               "This clears ALL pending items, including unrelated "
                               "jobs."}),
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("audio", "name", "filename", "file_number", "total_files")
    FUNCTION = "next_file"
    CATEGORY = "LongCat Longform"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Which file is next depends on files written by earlier queue items.
        return float("nan")

    def next_file(self, folder, name_suffix="", stop_when_done=True, extra_pnginfo=None):
        path, files = batch_files(folder)
        if not files:
            raise RuntimeError(f"No audio files in {path}. Add .wav, .mp3, .flac, .m4a, "
                               ".aac, .ogg or .opus files, then queue again.")
        # A coordinator item works on one file's shared render only.
        pinned = _pin(extra_pnginfo)
        waiting = []
        for number, name in enumerate(files, 1):
            session = os.path.splitext(name)[0] + name_suffix
            if pinned is not None and session != pinned:
                continue
            if is_finished(session):
                continue
            if pinned is None and waiting_on_others(session):
                waiting.append(session)
                continue
            print(f"{LOG} batch: file {number}/{len(files)} '{name}' -> "
                  f"output/{session}.mp4")
            return (_load_audio_file(os.path.join(path, name)), session,
                    session + ".mp4", number, len(files))

        if pinned is not None or waiting:
            print(f"{LOG} batch: nothing for this worker now; other workers are finishing "
                  f"{', '.join(waiting) or pinned}.")
            if ExecutionBlocker is None:
                raise RuntimeError("Nothing to render for this queue item.")
            return tuple(ExecutionBlocker(None) for _ in self.RETURN_TYPES)

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
                               "Use _fast to keep fast test renders separate."}),
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
    "LongCatLongformPrompt": "LongCat Longform: Prompt (optional)",
    "LongCatLongformSigmas": "LongCat Longform: v1.5 Distill Sigmas",
    "LongCatLongformAudioFolder": "LongCat Longform: Audio From Folder (batch)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
