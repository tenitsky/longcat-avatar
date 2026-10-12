"""CPU regression checks: python tools/test_longform.py (torch, torchaudio, numpy,
pyloudnorm, ffmpeg)."""
import contextlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import types
import unittest

import numpy as np
import torch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


class ExecutionBlocker:
    def __init__(self, value):
        self.value = value


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Only the ComfyUI host is stubbed; tensors, audio and media tools are real.
folder_paths = types.ModuleType("folder_paths")
sys.modules["folder_paths"] = folder_paths
sys.modules["comfy_execution"] = types.ModuleType("comfy_execution")
graph_utils = types.ModuleType("comfy_execution.graph_utils")
graph_utils.ExecutionBlocker = ExecutionBlocker
sys.modules["comfy_execution.graph_utils"] = graph_utils
lc = load("lc", ROOT / "custom_nodes/comfyui-longcat-longform/__init__.py")
builder = load("builder", ROOT / "tools/build_workflow.py")


def speech(sr, seconds, seed=0):
    """Noise bursts with gaps: something pyloudnorm can measure."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds)) / sr
    env = (np.sin(2 * np.pi * 1.3 * t) > -0.2).astype(np.float32)
    wave = (rng.standard_normal(t.shape).astype(np.float32) * 0.05 * env)
    return {"waveform": torch.from_numpy(wave).reshape(1, 1, -1), "sample_rate": sr}


class PlanTests(unittest.TestCase):
    def test_planner_tiles_the_track_in_windows(self):
        rng = random.Random(7)
        cases = [(t, []) for t in (1, 12, 13, 14, 50, 92, 93, 94, 173, 174)]
        for _ in range(300):
            total = rng.randrange(1, 25 * 900)
            restarts = sorted(rng.sample(range(1, total), min(total - 1, rng.randrange(0, 6))))
            cases.append((total, restarts))
        for total, restarts in cases:
            spans = lc.plan_chunks(total, restarts)
            self.assertTrue(spans[0][2], "chunk 0 opens on the portrait")
            self.assertEqual([s for s, _, on in spans if on], [0] + restarts)
            pos = 0
            for i, (start, keep, on_portrait) in enumerate(spans):
                self.assertEqual(start, pos)
                self.assertGreater(keep, 0)
                lead = 0 if on_portrait else lc.OVERLAP
                # Every window is 93 frames: lead-in plus kept frames fit inside it.
                self.assertLessEqual(lead + keep, lc.WINDOW)
                if i < len(spans) - 1 and not spans[i + 1][2]:
                    # The next window opens on this chunk's last 13 frames.
                    self.assertGreaterEqual(keep, lc.OVERLAP)
                pos += keep
            self.assertEqual(pos, total)

    def test_upstream_segment_arithmetic(self):
        # Upstream: segment 0 keeps 93 frames, each continuation adds 93 - 13 = 80.
        spans = lc.plan_chunks(93 + 80 * 4)
        self.assertEqual(spans, [(0, 93, True), (93, 80, False), (173, 80, False),
                                 (253, 80, False), (333, 80, False)])
        # A 7-minute voiceover.
        self.assertEqual(len(lc.plan_chunks(7 * 60 * 25)), 1 + math.ceil((10500 - 93) / 80))
        # A restart starts a new segment: a whole window again, then continuations.
        self.assertEqual(lc.plan_chunks(300, [200]),
                         [(0, 93, True), (93, 80, False), (173, 27, False),
                          (200, 93, True), (293, 7, False)])

    def test_restarts_land_on_pauses_30_to_40_seconds_apart(self):
        sr, seconds = 16000, 140
        rng = np.random.default_rng(3)
        wave = rng.standard_normal(sr * seconds).astype(np.float32) * 0.1
        pauses = [(33.0, 33.6), (38.0, 38.4), (71.2, 71.9), (108.0, 108.5)]
        for a, b in pauses:   # silent gaps the planner should pick
            wave[int(a * sr):int(b * sr)] *= 0.001
        total = seconds * 25
        restarts = lc.plan_restarts(wave, total)
        self.assertEqual(lc.RESTART_MIN, 30.0)
        self.assertEqual(lc.RESTART_MAX, 40.0)
        last = 0
        for r in restarts:
            self.assertTrue(30 * 25 <= r - last <= 40 * 25, (r, last))
            self.assertTrue(any(a * 25 <= r <= b * 25 for a, b in pauses), r / 25)
            last = r
        self.assertEqual(len(restarts), 3)  # 33, 71, 108 s; none in the last 3 s
        self.assertLess(restarts[-1], total - lc.RESTART_TAIL)
        self.assertEqual(lc.plan_restarts(wave, total, 30, 0), [])     # off
        self.assertEqual(lc.plan_restarts(wave, 25 * 25), [])          # under 30 s
        # No pause at all: still restarts, inside the window.
        loud = rng.standard_normal(sr * 90).astype(np.float32) * 0.1
        for r in lc.plan_restarts(loud, 90 * 25):
            self.assertTrue(30 * 25 <= r <= 40 * 25 or r > 40 * 25)

    def test_reference_positions(self):
        self.assertEqual(lc.ref_settings(lc.ANCHOR_IDENTITY, 13, 80), (10, 3))
        # Frame 92 of a window is its last latent (1 + 23 * 4 frames -> latent 23).
        self.assertEqual(lc.ref_settings(lc.ANCHOR_WINDOW_END, 13, 80), (23, 0))
        self.assertEqual(lc.ref_settings(lc.ANCHOR_WINDOW_END, 13, 20), (8, 0))
        self.assertEqual([lc.latent_index(f) for f in (0, 1, 4, 5, 8, 9, 92)],
                         [0, 1, 1, 2, 2, 3, 23])
        self.assertEqual(lc.ANCHORS, [lc.ANCHOR_IDENTITY, lc.ANCHOR_WINDOW_END])
        self.assertNotIn("experimental", " ".join(lc.ANCHORS))

    def test_distill_sigmas_match_upstream_v15(self):
        # LongCat-Video pipeline_longcat_video_avatar.get_timesteps_sigmas(use_distill,
        # avatar-v1.5), then FlowMatchEulerDiscreteScheduler(shift=7.0).set_timesteps.
        linear = [1 - j / 999 for j in range(1000)]
        raw = [linear[j] for j in (0, 125, 250, 375, 500, 625, 750, 875)]
        expected = [7 * s / (1 + 6 * s) for s in raw] + [0.0]
        got = lc.distill_sigmas().tolist()
        self.assertEqual(len(got), 9)
        for a, b in zip(got, expected):
            self.assertAlmostEqual(a, b, places=5)
        self.assertEqual(got[0], 1.0)
        self.assertEqual(lc.LongCatLongformSigmas().sigmas(7.0)[0].tolist(), got)

    def test_whisper_node_sees_the_intended_frame_count(self):
        # LongCatAvatarWhisperEmbeds: end_sample = int(num_frames / fps * sr), then
        # video_length = int(len(audio) / sr * fps). Split passes frames + 1.
        for frames in range(1, 3000):
            samples = frames * lc.SAMPLES_PER_FRAME + lc.WHISPER_PAD
            end_sample = int((frames + 1) / float(lc.FPS) * lc.WHISPER_SR)
            used = min(samples, end_sample)
            self.assertEqual(int(used / lc.WHISPER_SR * lc.FPS), frames, frames)


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="longcat-tests-")
        self.addCleanup(self.tmp.cleanup)
        folder_paths.get_output_directory = lambda: self.tmp.name
        folder_paths.get_input_directory = lambda: os.path.join(self.tmp.name, "input")
        lc._track.clear()

    def prompt(self, session="test"):
        return {"70": {"class_type": "LongCatLongformWrite", "inputs": {"session": session}}}

    def mark_done(self, session, upto):
        sdir = Path(lc._session_dir(session))
        for i in range(upto):
            (sdir / f"chunk_{i:04d}.mp4").write_bytes(b"done")
            tail = Path(lc._tail_path(session, i))
            tail.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(tail, frames=np.zeros((13, 4, 4, 3), np.uint8))

    def test_audio_windows_line_up_with_frames_and_extend_node(self):
        sr = 44100
        audio = speech(sr, 20)
        mono16, gain = lc.prepared_track(audio)
        self.assertGreater(gain, 1.0)   # quiet fixture is raised toward -23 LUFS
        # A restart every 5-8 s here, so a 20 s clip has some.
        spans = lc.plan_chunks(20 * 25, lc.plan_restarts(mono16, 20 * 25, 5, 8))
        self.assertGreater(sum(on for _, _, on in spans), 1)
        with contextlib.redirect_stdout(io.StringIO()):
            for index, (start, keep, on_portrait) in enumerate(spans):
                self.mark_done("test", index)
                out = lc.LongCatLongformSplit().split(audio, lc.ANCHOR_IDENTITY, 5, 8,
                                                      prompt=self.prompt())
                (chunk, whisper, wframes, nframes, overlap, processed, ref_index,
                 ref_mask, trim, kept, total, opens, picked) = out
                self.assertEqual((picked, kept, total), (index, keep, len(spans)))
                self.assertEqual(nframes, lc.WINDOW)
                self.assertEqual(opens, on_portrait)
                self.assertEqual(trim, 0 if on_portrait else 13)
                self.assertEqual(overlap, 1 if on_portrait else 13)
                self.assertEqual((ref_index, ref_mask), (10, 3))
                # The kept slice of the original track, sample-exact.
                a, b = round(start * sr / 25), min(audio["waveform"].shape[-1],
                                                   round((start + keep) * sr / 25))
                self.assertTrue(torch.equal(chunk["waveform"], audio["waveform"][..., a:b]))
                # Whisper slice: 16 kHz mono, whole frames plus the pad.
                w = whisper["waveform"]
                self.assertEqual(whisper["sample_rate"], 16000)
                self.assertEqual(tuple(w.shape[:2]), (1, 1))
                frames = (w.shape[-1] - lc.WHISPER_PAD) // 640
                self.assertEqual(w.shape[-1], frames * 640 + lc.WHISPER_PAD)
                self.assertEqual(wframes, frames + 1)
                # The extend node starts at (frames_processed - overlap): the window's
                # first frame, i.e. window start minus the slice's own start.
                window = start - trim
                slice_start = window - (processed - overlap)
                self.assertGreaterEqual(slice_start, 0)
                self.assertLessEqual(window - slice_start, lc.CONTEXT)
                # No padding needed in the extend node: the slice covers the window.
                self.assertGreaterEqual(frames, processed + lc.WINDOW)
                ref = mono16[slice_start * 640:(slice_start + frames) * 640] * gain
                np.testing.assert_allclose(w[0, 0, :len(ref)].numpy(), ref, atol=1e-6)
                self.assertTrue((w[0, 0, len(ref):] == 0).all())

    def test_return_to_portrait_is_the_default(self):
        audio = speech(16000, 10)
        with contextlib.redirect_stdout(io.StringIO()):
            self.mark_done("anchored", 1)
            out = lc.LongCatLongformSplit().split(audio, prompt=self.prompt("anchored"))
            self.assertEqual(out[6:8], (23, 0))
            self.assertFalse(out[11])   # no restart in a 10 s clip

    def test_loudness_gain_is_one_value_for_the_whole_track(self):
        loud = speech(48000, 12)
        loud["waveform"] = loud["waveform"] * 8
        mono16, gain = lc.prepared_track(loud)
        import pyloudnorm
        self.assertAlmostEqual(pyloudnorm.Meter(16000).integrated_loudness(
            (mono16 * gain).astype(np.float64)), -23.0, delta=0.05)
        silent = {"waveform": torch.zeros(1, 2, 48000), "sample_rate": 48000}
        self.assertEqual(lc.prepared_track(silent)[1], 1.0)

    def test_resume_cache_is_invalidated(self):
        # ComfyUI evaluates IS_CHANGED with the hidden prompt set to {} and linked
        # inputs omitted, so none of these nodes can see the session there.
        for cls, widgets in ((lc.LongCatLongformSplit, {}), (lc.LongCatLongformAudioFolder, {}),
                             (lc.LongCatLongformCarry, {}),
                             (lc.LongCatLongformAudioName, {"suffix": ".mp4"})):
            with self.subTest(node=cls.__name__):
                self.assertTrue(math.isnan(cls.IS_CHANGED(prompt={}, **widgets)))

    def test_audio_name_and_session_resolver_select_same_audio(self):
        prompt = self.prompt(["12", 0])
        prompt["10"] = {"class_type": "LoadAudio", "inputs": {"audio": "wrong.wav"},
                        "_meta": {"title": "Other audio"}}
        prompt["11"] = {"class_type": "LoadAudio", "inputs": {"audio": "uploads/selected.wav"},
                        "_meta": {"title": "Voiceover"}}
        prompt["12"] = {"class_type": "LongCatLongformAudioName",
                        "inputs": {"suffix": ".mp4", "source_title": "Voiceover",
                                   "name_suffix": "_fast"}}
        with contextlib.redirect_stdout(io.StringIO()):
            name, filename = lc.LongCatLongformAudioName().derive(
                ".mp4", "Voiceover", prompt, name_suffix="_fast")
        self.assertEqual((name, filename), ("selected_fast", "selected_fast.mp4"))
        self.assertEqual(lc._session_from_prompt(prompt), name)

    def test_prompt(self):
        generic = lc.talking_head_prompt("")
        self.assertTrue(generic.startswith("A person is speaking directly to the camera"))
        self.assertEqual(lc.talking_head_prompt("  \n "), generic)
        text = lc.talking_head_prompt("A man in a grey suit,\n  in front of a white wall.")
        self.assertTrue(text.startswith("A man in a grey suit, in front of a white wall. "
                                        "The person is speaking"))
        for w in builder.WORKFLOWS:
            node = next(n for n in builder.build(w["variant"], w["batch"], w["precision"])["nodes"]
                        if n["type"] == "LongCatLongformPrompt")
            self.assertEqual(node["widgets_values"], [""])
            out = lc.LongCatLongformPrompt().build(*node["widgets_values"])[0]
            self.assertEqual(out, generic)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg and ffprobe required")
class RenderTests(unittest.TestCase):
    """Drive Split -> Carry -> (fake sampler) -> Write the way the workflows wire them."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="longcat-render-")
        self.addCleanup(self.tmp.cleanup)
        folder_paths.get_output_directory = lambda: self.tmp.name
        folder_paths.get_input_directory = lambda: os.path.join(self.tmp.name, "input")
        lc._track.clear()
        self.portrait = torch.full((1, 16, 16, 3), 0.5)
        self.portrait_latent = {"samples": torch.zeros(1, 16, 1, 2, 2)}

    def probe(self, path, *args):
        return json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", *args, "-of", "json", str(path)], text=True))

    def frames(self, path):
        info = self.probe(path, "-count_frames", "-select_streams", "v:0",
                          "-show_entries", "stream=nb_read_frames")
        return int(info["streams"][0]["nb_read_frames"])

    def item(self, audio, session, filename, anchor=lc.ANCHOR_IDENTITY, restart=(30, 40),
             batch_session=None):
        """One queue item. The 'sampler' paints each window frame with its global index
        so continuity and trimming can be checked in the written files."""
        prompt = {"70": {"class_type": "LongCatLongformWrite", "inputs": {"session": session}}}
        out = lc.LongCatLongformSplit().split(audio, anchor, *restart,
                                              session=batch_session, prompt=prompt)
        if isinstance(out[0], ExecutionBlocker):
            return None
        (chunk, _, _, nframes, _, _, _, _, trim, keep, total, opens, index) = out
        prev, ref = lc.LongCatLongformCarry().pick(
            self.portrait, self.portrait_latent, index, opens,
            session=batch_session, prompt=prompt)
        if opens:
            # Chunk 0 and every restart: the portrait alone, no reference. The fake
            # sampler then paints from this chunk's own start frame.
            self.assertEqual(prev.shape[0], 1)
            self.assertIsNone(ref)
            mono16, _ = lc.prepared_track(audio)
            frames_total = int(round(audio["waveform"].shape[-1] * 25
                                     / audio["sample_rate"]))
            spans = lc.plan_chunks(frames_total, lc.plan_restarts(mono16, frames_total,
                                                                  *restart))
            window_start = spans[index][0]
        else:
            self.assertEqual(prev.shape[0], lc.OVERLAP)
            self.assertIs(ref, self.portrait_latent)
            # The lead-in is the previous chunk's last 13 written frames.
            window_start = int(round(prev[0, 0, 0, 0].item() * 255))
        values = torch.arange(window_start, window_start + nframes).float() % 256 / 255
        images = values.reshape(-1, 1, 1, 1).expand(-1, 16, 16, 3).contiguous()
        # The batch workflows wire the folder node's name into Write's session too.
        lc.LongCatLongformWrite().write(images, audio, index, total, keep,
                                        batch_session or session, filename, chunk, False,
                                        trim)
        return index, total, opens

    def test_continuation_render_stitch_and_resume(self):
        sr, seconds = 44100, 13
        audio = speech(sr, seconds)
        with contextlib.redirect_stdout(io.StringIO()):
            done = []
            while True:
                r = self.item(audio, "speaker's take", "final.mp4")
                if r is None:
                    break
                done.append(r)
                self.assertLess(len(done), 20)
        total = seconds * 25
        spans = lc.plan_chunks(total)
        self.assertEqual([d[0] for d in done], list(range(len(spans))))
        sdir = Path(lc._session_dir("speaker's take"))
        for i, (_, keep, _) in enumerate(spans):
            self.assertEqual(self.frames(sdir / f"chunk_{i:04d}.mp4"), keep)
        final = Path(self.tmp.name) / "final.mp4"
        info = self.probe(final, "-count_frames", "-show_entries",
                          "stream=codec_type,nb_read_frames,duration")
        video = next(s for s in info["streams"] if s["codec_type"] == "video")
        self.assertEqual(int(video["nb_read_frames"]), total)
        self.assertAlmostEqual(float(video["duration"]), seconds, delta=1 / 25)
        times = [float(f["best_effort_timestamp_time"]) for f in self.probe(
            final, "-select_streams", "v:0", "-show_entries",
            "frame=best_effort_timestamp_time")["frames"]]
        self.assertLess(max(abs(t - i / 25) for i, t in enumerate(times)), 0.001)
        # Continuity: frame g of the video is global frame g of the generated windows,
        # across every join (no lead-in frame plays twice, none is skipped).
        raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(final), "-f",
                                       "rawvideo", "-pix_fmt", "rgb24", "-"])
        got = np.frombuffer(raw, np.uint8).reshape(total, -1)[:, 0].astype(int)
        diff = np.abs(got - np.arange(total) % 256)
        self.assertLessEqual(int(np.minimum(diff, 256 - diff).max()), 3)
        # Finished: further items are skipped; a missing tail re-renders that chunk.
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(self.item(audio, "speaker's take", "final.mp4"))
            Path(lc._tail_path("speaker's take", 1)).unlink()
            self.assertEqual(self.item(audio, "speaker's take", "final.mp4")[0], 1)
            # Then the stitch reruns because writing a chunk cleared .finished.
            self.assertEqual(self.item(audio, "speaker's take", "final.mp4")[0],
                             len(spans) - 1)
            self.assertIsNone(self.item(audio, "speaker's take", "final.mp4"))

    def test_missing_tail_explains_order(self):
        with self.assertRaisesRegex(RuntimeError, "chunks render in order"):
            lc.LongCatLongformCarry().pick(self.portrait, self.portrait_latent, 3, False,
                                           prompt={})

    def test_canvas_change_is_refused(self):
        tail = Path(lc._tail_path("run1", 0))
        tail.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(tail, frames=np.zeros((13, 8, 8, 3), np.uint8))
        with self.assertRaisesRegex(RuntimeError, "fresh render at the new size"):
            lc.LongCatLongformCarry().pick(self.portrait, self.portrait_latent, 1, False,
                                           prompt={})

    def test_restarts_open_on_the_portrait_and_stitch_exactly(self):
        # 75 s with restarts on: two segments restart from the portrait, every frame
        # still lands once, in order, and the video is exactly as long as the audio.
        sr, seconds = 16000, 75
        audio = speech(sr, seconds)
        seen = []
        with contextlib.redirect_stdout(io.StringIO()):
            while True:
                r = self.item(audio, "refresh", "refresh.mp4", lc.ANCHOR_WINDOW_END)
                if r is None:
                    break
                seen.append(r)
                self.assertLess(len(seen), 40)
        mono16, _ = lc.prepared_track(audio)
        restarts = lc.plan_restarts(mono16, seconds * 25)
        self.assertEqual(len(restarts), 2)
        spans = lc.plan_chunks(seconds * 25, restarts)
        self.assertEqual([o for _, _, o in seen], [on for _, _, on in spans])
        final = Path(self.tmp.name) / "refresh.mp4"
        self.assertEqual(self.frames(final), seconds * 25)
        raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(final), "-f",
                                       "rawvideo", "-pix_fmt", "rgb24", "-"])
        got = np.frombuffer(raw, np.uint8).reshape(seconds * 25, -1)[:, 0].astype(int)
        diff = np.abs(got - np.arange(seconds * 25) % 256)
        self.assertLessEqual(int(np.minimum(diff, 256 - diff).max()), 3)

    def test_batch_folder_renders_each_file_once_in_name_order(self):
        folder = Path(self.tmp.name) / "input" / "batch_audio"
        folder.mkdir(parents=True)
        for name, seconds in (("talk 10.mp3", 3), ("talk 2.wav", 7), ("talk 1.flac", 2)):
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                            f"anoisesrc=d={seconds}:a=0.05", "-ac", "1",
                            str(folder / name)], check=True)
        (folder / "notes.txt").write_text("not audio")
        source = lc.LongCatLongformAudioFolder()
        rendered = []
        with contextlib.redirect_stdout(io.StringIO()):
            while True:
                picked = source.next_file("batch_audio", "_fast", True)
                if isinstance(picked[0], ExecutionBlocker):
                    break
                audio, session, filename, number, count = picked
                self.assertEqual(count, 3)
                r = self.item(audio, "wrong-session", filename, batch_session=session)
                rendered.append((session, r[0]))
                self.assertLess(len(rendered), 20, "batch never finished")
        sessions = list(dict.fromkeys(s for s, _ in rendered))
        self.assertEqual(sessions, ["talk 1_fast", "talk 2_fast", "talk 10_fast"])
        for session in sessions:
            chunks = [i for s, i in rendered if s == session]
            self.assertEqual(chunks, list(range(len(chunks))))
            self.assertTrue((Path(self.tmp.name) / f"{session}.mp4").exists())
        self.assertGreater(len([1 for s, _ in rendered if s == "talk 2_fast"]), 1)
        self.assertFalse((Path(self.tmp.name) / lc.SESSION_ROOT / "wrong-session").exists())
        shutil.copy(folder / "talk 1.flac", folder / "talk 1.wav")
        with self.assertRaisesRegex(RuntimeError, "different names"):
            source.next_file("batch_audio", "_fast", True)

    def test_batch_folder_refuses_a_file_cut_short_in_transfer(self):
        folder = Path(self.tmp.name) / "input" / "batch_audio"
        folder.mkdir(parents=True)
        full = Path(self.tmp.name) / "full.mp3"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                        "sine=frequency=300:duration=480", "-ac", "2", "-c:a", "libmp3lame",
                        "-b:a", "160k", str(full)], check=True)
        (folder / "lesson.mp3").write_bytes(full.read_bytes()[:1048576])
        with contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(RuntimeError, r"lesson\.mp3 is incomplete"):
            lc.LongCatLongformAudioFolder().next_file("batch_audio")

    def test_batch_folder_is_created_and_explains_when_empty(self):
        with self.assertRaisesRegex(RuntimeError, "No audio files"):
            lc.LongCatLongformAudioFolder().next_file("batch_audio")
        self.assertTrue((Path(self.tmp.name) / "input" / "batch_audio").is_dir())


if __name__ == "__main__":
    unittest.main(verbosity=2)
