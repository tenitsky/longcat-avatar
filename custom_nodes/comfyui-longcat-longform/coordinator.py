#!/usr/bin/env python3
"""LongCat coordinator: share long renders across every GPU and pod on the volume.

setup.sh starts this in the background on every pod. With one GPU and no other pod on
the volume it has nothing to do, and renders work exactly as without it. Otherwise:

  * on a pod with several GPUs it starts one more ComfyUI per extra GPU (ports 8189,
    8190, ... on 127.0.0.1), so each GPU is a worker;
  * every HEARTBEAT_SECONDS it writes a heartbeat for each local ComfyUI that answers,
    so the Split node on any pod can count the workers;
  * when a render has been shared out (the Split node writes <session>/job.json), it
    keeps each local worker's queue fed with that render, pinned to the session, until
    the worker's sections are done; the leader gets one more item to stitch;
  * for renders led from this pod, it hands the sections of a worker that went silent
    to the live ones.

It uses only ComfyUI's own HTTP API and the files described in parallel.py.

    python3 coordinator.py --comfy /path/to/ComfyUI --python /path/to/venv/python
"""
import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parallel  # noqa: E402

BASE_PORT = 8188
INTERVAL = 10              # seconds between passes
STUCK_AFTER = 3            # items in a row without progress before backing off
BACKOFF = 300              # seconds to wait after that
RELAUNCH_GAP = 60          # seconds between restarts of a crashed extra ComfyUI
CLIENT_ID = "longcat-coordinator"


def log(message):
    print(f"[LongCat Coordinator] {time.strftime('%H:%M:%S')} {message}", flush=True)


def gpu_count():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True,
                             timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return 1
    return max(1, sum(1 for line in out.splitlines() if line.startswith("GPU ")))


class ComfyAPI:
    """The two ComfyUI endpoints the coordinator needs."""

    def __init__(self, timeout=10):
        self.timeout = timeout

    def queue(self, port):
        """(running, pending) item counts, or None while ComfyUI is not answering."""
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/queue",
                                        timeout=self.timeout) as response:
                data = json.load(response)
        except (OSError, ValueError, urllib.error.URLError):
            return None
        return len(data.get("queue_running", [])), len(data.get("queue_pending", []))

    def submit(self, port, prompt, extra_pnginfo):
        body = json.dumps({"prompt": prompt, "client_id": CLIENT_ID,
                           "extra_data": {"extra_pnginfo": extra_pnginfo}}).encode()
        request = urllib.request.Request(f"http://127.0.0.1:{port}/prompt", data=body,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return 200 <= response.status < 300
        except urllib.error.HTTPError as exc:
            log(f"ComfyUI on port {port} refused a queue item: "
                f"{exc.read().decode(errors='replace')[:500]}")
        except (OSError, urllib.error.URLError) as exc:
            log(f"ComfyUI on port {port} did not take a queue item: {exc}")
        return False


def extra_args(args_file):
    """The user's comfyui_args.txt flags, without the listen address and port."""
    try:
        with open(args_file, encoding="utf-8") as stream:
            tokens = shlex.split(" ".join(l for l in stream.read().splitlines()
                                          if l.strip() and not l.lstrip().startswith("#")))
    except (OSError, ValueError):
        return []
    kept, skip = [], False
    for token in tokens:
        if skip and not token.startswith("-"):
            skip = False
            continue
        skip = token.split("=")[0] in ("--listen", "--port")
        if not skip:
            kept.append(token)
    return kept


class Worker:
    def __init__(self, worker_id, gpu, port):
        self.id, self.gpu, self.port = worker_id, gpu, port
        self.process = None
        self.launched = 0.0


class Coordinator:
    def __init__(self, comfy, pod, gpus=1, python=None, args_file=None, api=None,
                 clock=time.time, launch=True, log_dir="/tmp"):
        self.comfy, self.pod, self.python = comfy, pod, python
        self.root = os.path.join(comfy, "output", "longcat_longform")
        self.api, self.clock = api or ComfyAPI(), clock
        self.args_file, self.launch, self.log_dir = args_file, launch, log_dir
        self.workers = [Worker(f"{pod}-gpu{g}", g, BASE_PORT + g) for g in range(gpus)]
        self.beats = {}
        self.progress = {}   # (session, worker) -> (chunks left, items in a row, wait until)

    # ------------------------------------------------------------- extra ComfyUIs
    def start_extras(self):
        """One ComfyUI per extra GPU, once the main one (GPU 0) is answering."""
        if not self.launch or len(self.workers) < 2 or self.api.queue(BASE_PORT) is None:
            return
        for worker in self.workers[1:]:
            running = worker.process is not None and worker.process.poll() is None
            if running or self.clock() - worker.launched < RELAUNCH_GAP:
                continue
            worker.launched = self.clock()
            temp = os.path.join(self.log_dir, f"comfyui-gpu{worker.gpu}")
            command = [self.python or sys.executable, "main.py", "--listen", "127.0.0.1",
                       "--port", str(worker.port), "--cuda-device", str(worker.gpu),
                       "--temp-directory", temp] + extra_args(self.args_file)
            env = {**os.environ, "LONGCAT_WORKER_ID": worker.id, "PYTHONUNBUFFERED": "1"}
            out = open(os.path.join(self.log_dir, f"comfyui-gpu{worker.gpu}.log"), "a")
            worker.process = subprocess.Popen(command, cwd=self.comfy, env=env, stdout=out,
                                              stderr=subprocess.STDOUT,
                                              start_new_session=True)
            log(f"started ComfyUI for GPU {worker.gpu} on port {worker.port} "
                f"({worker.id}; log {out.name})")

    # ----------------------------------------------------------------- one pass
    def tick(self):
        now = self.clock()
        self.start_extras()
        idle = {}
        for worker in self.workers:
            status = self.api.queue(worker.port)
            if status is None:
                continue
            idle[worker.id] = status == (0, 0)
            if now - self.beats.get(worker.id, 0) >= parallel.HEARTBEAT_SECONDS:
                parallel.write_heartbeat(self.root, worker.id,
                                         {"pod": self.pod, "gpu": worker.gpu,
                                          "port": worker.port}, now)
                self.beats[worker.id] = now
        live = parallel.live_workers(self.root, now)
        submitted = []
        for sdir, job in self.jobs():
            if job["leader"] in idle and parallel.reassign(sdir, job, live):
                parallel.save_job(sdir, job)
                log(f"'{job['session']}': sections handed to live workers: "
                    f"{job['assignment']}")
            for worker in self.workers:
                if not idle.get(worker.id) or not self.wants_item(sdir, job, worker.id, now):
                    continue
                pin = {parallel.PIN_KEY: {"session": job["session"]}}
                if job.get("workflow"):
                    pin["workflow"] = job["workflow"]
                if self.api.submit(worker.port, job["prompt"], pin):
                    idle[worker.id] = False
                    submitted.append((job["session"], worker.id))
        return submitted

    def jobs(self):
        try:
            names = sorted(n for n in os.listdir(self.root) if not n.startswith("."))
        except FileNotFoundError:
            return
        for name in names:
            sdir = os.path.join(self.root, name)
            try:
                job = parallel.load_job(sdir)
            except (OSError, ValueError):
                continue
            if job and job.get("prompt") and not parallel.is_finished(sdir):
                yield sdir, job

    def wants_item(self, sdir, job, worker, now):
        """Whether `worker` has chunks of this render left, or stitches it now. A worker
        whose items stop making progress (an error in the workflow) is left alone for
        BACKOFF seconds instead of being fed failing items."""
        left = len(parallel.remaining(sdir, job["spans"], parallel.owned_chunks(job, worker)))
        if not left and not (job["leader"] == worker and parallel.all_done(sdir, job["spans"])):
            return False
        key = (job["session"], worker)
        before, streak, wait = self.progress.get(key, (None, 0, 0))
        if now < wait:
            return False
        streak = streak + 1 if left == before else 1
        if streak > STUCK_AFTER:
            log(f"'{job['session']}': {worker} made no progress in {STUCK_AFTER} items; "
                f"retrying in {BACKOFF // 60} min. Check the ComfyUI log.")
            self.progress[key] = (left, 0, now + BACKOFF)
            return False
        self.progress[key] = (left, streak, 0)
        return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--comfy", required=True, help="ComfyUI directory")
    parser.add_argument("--python", help="ComfyUI's interpreter, for extra GPUs")
    parser.add_argument("--pod", default=os.environ.get("RUNPOD_POD_ID") or platform.node())
    parser.add_argument("--args-file", help="comfyui_args.txt, passed to extra ComfyUIs")
    parser.add_argument("--gpus", type=int, help="GPU count (default: nvidia-smi)")
    parser.add_argument("--once", action="store_true", help="one pass, then exit")
    args = parser.parse_args(argv)
    if not parallel.enabled():
        log("LONGCAT_PARALLEL=0: not sharing renders.")
        return
    coordinator = Coordinator(args.comfy, args.pod, args.gpus or gpu_count(), args.python,
                              args.args_file)
    log(f"pod {args.pod}: {len(coordinator.workers)} GPU(s), watching {coordinator.root}")
    while True:
        try:
            for session, worker in coordinator.tick():
                log(f"'{session}': queued an item on {worker}")
        except Exception as exc:  # keep helping; one bad pass must not stop the pod
            log(f"pass failed: {exc!r}")
        if args.once:
            return
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
