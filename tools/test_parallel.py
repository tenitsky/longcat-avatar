"""Shared renders and the coordinator, without ComfyUI or a GPU: python tools/test_parallel.py.

The coordinator talks to a fake ComfyUI API; files are real. The full render across
three workers (real Split / Carry / Write and ffmpeg) is in test_longform.py.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "custom_nodes/comfyui-longcat-longform"
sys.path.insert(0, str(PACK))
sys.dont_write_bytecode = True
import parallel  # noqa: E402

spec = importlib.util.spec_from_file_location("coordinator", PACK / "coordinator.py")
coordinator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coordinator)

# Two sections of two chunks each: chunk 2 restarts from the portrait.
SPANS = [[0, 93, True], [93, 80, False], [173, 93, True], [266, 80, False]]
NOW = 1_000_000.0


class FakeAPI:
    def __init__(self):
        self.status = {}        # port -> (running, pending); missing = not answering
        self.submitted = []
        self.accept = True

    def queue(self, port):
        return self.status.get(port)

    def submit(self, port, prompt, extra_pnginfo):
        self.submitted.append((port, prompt, extra_pnginfo))
        if self.accept:
            self.status[port] = (1, 0)
        return self.accept


class Base(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="longcat-parallel-")
        self.addCleanup(temp.cleanup)
        self.comfy = Path(temp.name) / "ComfyUI"
        self.root = self.comfy / "output" / "longcat_longform"
        self.sdir = self.root / "talk"
        self.sdir.mkdir(parents=True)

    def done(self, *chunks):
        for i in chunks:
            Path(parallel.chunk_path(self.sdir, i)).write_bytes(b"mp4")
            tail = Path(parallel.tail_path(self.sdir, i))
            tail.parent.mkdir(exist_ok=True)
            tail.write_bytes(b"npz")

    def job(self, assignment, leader="podA-gpu0"):
        secs = parallel.sections(SPANS)
        job = parallel.make_job("talk", leader, "346:173", SPANS, secs, assignment,
                                prompt={"1": {"class_type": "LoadAudio"}},
                                workflow={"nodes": []}, filename="talk.mp4")
        parallel.save_job(self.sdir, job)
        return job


class PlanTests(Base):
    def test_sections_start_at_every_restart(self):
        self.assertEqual(parallel.sections(SPANS), [[0, 2], [2, 4]])
        self.assertEqual(parallel.sections([[0, 93, True]]), [[0, 1]])

    def test_assign_longest_first_and_keeps_fixed(self):
        work = [100, 300, 200, 50, 0]
        plan = parallel.assign(work, ["b", "a", "c"], "a")
        self.assertEqual(sorted(s for ss in plan.values() for s in ss), [0, 1, 2, 3])
        self.assertEqual(plan["a"], [1])   # leader first, longest section
        self.assertEqual(plan["b"], [2])
        self.assertEqual(plan["c"], [0, 3])
        fixed = parallel.assign(work, ["a", "b"], "a", fixed={0: "a"})
        self.assertIn(0, fixed["a"])
        self.assertEqual(sorted(s for ss in fixed.values() for s in ss), [0, 1, 2, 3])
        # A finished section is nobody's work.
        self.assertNotIn(4, [s for ss in plan.values() for s in ss])

    def test_chunk_needs_its_tail_only_when_the_next_one_continues(self):
        Path(parallel.chunk_path(self.sdir, 0)).write_bytes(b"mp4")
        Path(parallel.chunk_path(self.sdir, 1)).write_bytes(b"mp4")
        self.assertFalse(parallel.chunk_done(self.sdir, SPANS, 0))   # chunk 1 continues
        self.assertTrue(parallel.chunk_done(self.sdir, SPANS, 1))    # chunk 2 restarts
        self.assertEqual(parallel.section_work(self.sdir, SPANS, [[0, 2], [2, 4]]), [93, 173])

    def test_workers_go_stale_and_unreadable_heartbeats_are_skipped(self):
        parallel.write_heartbeat(self.root, "a", now=NOW)
        parallel.write_heartbeat(self.root, "b", now=NOW - parallel.STALE_SECONDS - 1)
        (self.root / parallel.WORKERS_DIR / "c.json").write_text('{"worker": "c", "ti')
        self.assertEqual(parallel.live_workers(self.root, NOW), ["a"])
        self.assertEqual(parallel.live_workers(self.root / "missing", NOW), [])

    def test_job_round_trip_and_owned_chunks(self):
        job = self.job({"podA-gpu0": [0], "podB-gpu0": [1]})
        self.assertEqual(parallel.load_job(self.sdir), json.loads(json.dumps(job)))
        self.assertEqual(parallel.owned_chunks(job, "podB-gpu0"), [2, 3])
        self.assertEqual(parallel.owned_chunks(job, "nobody"), [])

    def test_reassign_moves_only_silent_workers_sections(self):
        job = self.job({"podA-gpu0": [0], "podB-gpu0": [1]})
        self.assertFalse(parallel.reassign(self.sdir, job, ["podA-gpu0", "podB-gpu0"]))
        self.assertTrue(parallel.reassign(self.sdir, job, ["podA-gpu0", "podC-gpu0"]))
        self.assertEqual(job["assignment"], {"podA-gpu0": [0], "podC-gpu0": [1]})
        # A new leader takes over a silent one, keeping live owners' sections.
        self.assertTrue(parallel.reassign(self.sdir, job, ["podC-gpu0", "podD-gpu0"],
                                          leader="podD-gpu0"))
        self.assertEqual(job["leader"], "podD-gpu0")
        self.assertEqual(job["assignment"]["podC-gpu0"], [1])
        self.assertEqual(job["assignment"]["podD-gpu0"], [0])

    def test_switch_and_worker_name(self):
        with patch.dict(os.environ, {"LONGCAT_PARALLEL": "0", "LONGCAT_WORKER_ID": "p-gpu1"}):
            self.assertFalse(parallel.enabled())
            self.assertEqual(parallel.worker_id(), "p-gpu1")
        with patch.dict(os.environ, {"LONGCAT_PARALLEL": "1", "LONGCAT_WORKER_ID": ""}):
            self.assertTrue(parallel.enabled())
            self.assertEqual(parallel.worker_id(), "local")


class CoordinatorTests(Base):
    def coordinator(self, pod, gpus=1, api=None):
        self.api = api or FakeAPI()
        self.clock = [NOW]
        return coordinator.Coordinator(str(self.comfy), pod, gpus, api=self.api,
                                       clock=lambda: self.clock[0], launch=False)

    def test_idle_without_shared_renders_but_registers(self):
        c = self.coordinator("podB")
        self.api.status[8188] = (0, 0)
        self.assertEqual(c.tick(), [])
        self.assertEqual(parallel.live_workers(self.root, NOW), ["podB-gpu0"])
        # A ComfyUI that does not answer is not registered.
        c = self.coordinator("podZ")
        c.tick()
        self.assertNotIn("podZ-gpu0", parallel.live_workers(self.root, NOW))

    def test_feeds_a_helper_its_sections_pinned_to_the_session(self):
        self.job({"podA-gpu0": [0], "podB-gpu0": [1]})
        c = self.coordinator("podB")
        self.api.status[8188] = (0, 0)
        self.assertEqual(c.tick(), [("talk", "podB-gpu0")])
        port, prompt, extra = self.api.submitted[0]
        self.assertEqual(port, 8188)
        self.assertEqual(prompt, {"1": {"class_type": "LoadAudio"}})
        self.assertEqual(extra[parallel.PIN_KEY], {"session": "talk"})
        self.assertEqual(extra["workflow"], {"nodes": []})
        # Busy: nothing more until the item is done.
        self.assertEqual(c.tick(), [])
        # Its sections done: nothing more at all (the leader stitches).
        self.api.status[8188] = (0, 0)
        self.done(2, 3)
        self.assertEqual(c.tick(), [])

    def test_leader_gets_an_item_to_stitch_and_finished_renders_are_left(self):
        self.job({"podA-gpu0": [0], "podB-gpu0": [1]})
        parallel.write_heartbeat(self.root, "podB-gpu0", now=NOW)
        self.done(0, 1)
        c = self.coordinator("podA")
        self.api.status[8188] = (0, 0)
        self.assertEqual(c.tick(), [])             # waiting for podB's section
        self.done(2, 3)
        self.assertEqual(c.tick(), [("talk", "podA-gpu0")])
        (self.sdir / parallel.FINISHED).write_text("talk.mp4")
        self.api.status[8188] = (0, 0)
        self.assertEqual(c.tick(), [])

    def test_backs_off_when_items_make_no_progress(self):
        self.job({"podA-gpu0": [0], "podB-gpu0": [1]})
        c = self.coordinator("podB")
        sent = 0
        for _ in range(coordinator.STUCK_AFTER + 3):
            self.api.status[8188] = (0, 0)       # each item fails at once
            sent += len(c.tick())
        self.assertEqual(sent, coordinator.STUCK_AFTER)
        self.clock[0] += coordinator.BACKOFF + 1
        self.api.status[8188] = (0, 0)
        self.assertEqual(len(c.tick()), 1)

    def test_leader_hands_a_silent_workers_sections_to_live_ones(self):
        self.job({"podA-gpu0": [0], "podB-gpu0": [1]})
        parallel.write_heartbeat(self.root, "podB-gpu0", now=NOW - parallel.STALE_SECONDS - 5)
        self.done(0, 1)
        c = self.coordinator("podA")
        self.api.status[8188] = (0, 0)
        self.assertEqual(c.tick(), [("talk", "podA-gpu0")])
        self.assertEqual(parallel.load_job(self.sdir)["assignment"], {"podA-gpu0": [1]})

    def test_every_gpu_on_a_pod_is_a_worker(self):
        self.job({"podA-gpu0": [0], "podA-gpu1": [1]})
        c = self.coordinator("podA", gpus=2)
        self.api.status.update({8188: (0, 0), 8189: (0, 0)})
        self.assertEqual(sorted(c.tick()), [("talk", "podA-gpu0"), ("talk", "podA-gpu1")])
        self.assertEqual(sorted(p for p, _, _ in self.api.submitted), [8188, 8189])

    def test_extra_comfyui_args_drop_listen_and_port(self):
        args = self.comfy / "comfyui_args.txt"
        args.write_text("# comment\n--listen 0.0.0.0\n--reserve-vram 1\n--port=9000\n"
                        "--lowvram\n--listen\n--fast\n")
        self.assertEqual(coordinator.extra_args(args),
                         ["--reserve-vram", "1", "--lowvram", "--fast"])
        self.assertEqual(coordinator.extra_args(self.comfy / "missing.txt"), [])

    def test_switched_off(self):
        with patch.dict(os.environ, {"LONGCAT_PARALLEL": "0"}), \
                patch.object(coordinator.Coordinator, "tick") as tick:
            coordinator.main(["--comfy", str(self.comfy), "--gpus", "1", "--once"])
            tick.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
