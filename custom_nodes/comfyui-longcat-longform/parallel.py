"""Share one long render across several GPUs or pods. Standard library only.

A voiceover is rendered in chunks that continue from each other, so the chunks of one
video must render in order. But every restart from the portrait (one in every 40-60 s)
starts a chunk that depends on nothing before it, so a video is a series of independent
sections. Several workers (one ComfyUI per GPU, on one pod or many) can each render
their own sections at the same time, and the worker the render was queued on (the
leader) stitches the video once every section is on the shared volume.

State lives next to the renders, under output/longcat_longform/:

    .workers/<worker>.json   heartbeat of each ComfyUI that can help, written every
                             HEARTBEAT_SECONDS by the coordinator on its pod
    <session>/job.json       how a session is shared out; written only by its leader

Every file has a single writer and none is renamed or locked, so this also works on
a RunPod Global Volume (object storage: no locks, no atomic rename). Readers retry a
file caught mid-write.
"""
import json
import os
import time

WORKERS_DIR = ".workers"
JOB_FILE = "job.json"
FINISHED = ".finished"
PIN_KEY = "longcat_job"       # extra_pnginfo key: a queue item pinned to one session
HEARTBEAT_SECONDS = 20
# A Global Volume can show one pod's writes to another late, so a worker counts as
# gone only after a generous silence; reassigning a live worker's section would render
# it twice.
STALE_SECONDS = 300


def enabled():
    """LONGCAT_PARALLEL=0 keeps every render on the GPU it was queued on."""
    return os.environ.get("LONGCAT_PARALLEL", "1").strip() != "0"


def worker_id():
    """This ComfyUI's name; setup and the coordinator set it per GPU."""
    return os.environ.get("LONGCAT_WORKER_ID", "").strip() or "local"


def chunk_path(sdir, index):
    return os.path.join(sdir, f"chunk_{index:04d}.mp4")


def tail_path(sdir, index):
    """Last written frames of a chunk, kept for the next window's lead-in."""
    return os.path.join(sdir, ".carry", f"tail_{index:04d}.npz")


def read_json(path, attempts=3, delay=0.3):
    """A JSON file, or None if it is missing. A file caught mid-write is retried."""
    for attempt in range(attempts):
        try:
            with open(path, encoding="utf-8") as stream:
                return json.load(stream)
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            if attempt == attempts - 1:
                raise
            time.sleep(delay)
    return None


def write_json(path, data):
    """Written in place: a Global Volume has no atomic rename. Readers retry."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=1)


# ------------------------------------------------------------------------- workers
def write_heartbeat(root, worker, info=None, now=None):
    write_json(os.path.join(root, WORKERS_DIR, f"{worker}.json"),
               {**(info or {}), "worker": worker, "time": time.time() if now is None else now})


def live_workers(root, now=None):
    """Workers whose heartbeat is recent, in name order."""
    now = time.time() if now is None else now
    directory = os.path.join(root, WORKERS_DIR)
    try:
        names = sorted(n for n in os.listdir(directory) if n.endswith(".json"))
    except FileNotFoundError:
        return []
    live = []
    for name in names:
        try:
            beat = read_json(os.path.join(directory, name), attempts=1)
        except (OSError, ValueError):
            continue
        if beat and now - float(beat.get("time", 0)) <= STALE_SECONDS:
            live.append(beat.get("worker") or name[:-5])
    return live


# -------------------------------------------------------------------- plan and jobs
def plan_key(total, restarts):
    """Identifies a chunk plan: the same voiceover with the same restart settings."""
    return f"{int(total)}:" + ",".join(str(int(r)) for r in restarts)


def sections(spans):
    """[first_chunk, end_chunk) ranges: a new section starts at every chunk that opens
    on the portrait (chunk 0 and each restart)."""
    starts = [i for i, span in enumerate(spans) if span[2]] or [0]
    return [[a, b] for a, b in zip(starts, starts[1:] + [len(spans)])]


def chunk_done(sdir, spans, index):
    """A chunk is done when its file exists and, if the next chunk continues from it,
    so does its tail. The Split node also checks the tail's contents."""
    if not os.path.exists(chunk_path(sdir, index)):
        return False
    if index < len(spans) - 1 and not spans[index + 1][2]:
        return os.path.exists(tail_path(sdir, index))
    return True


def remaining(sdir, spans, chunks):
    """The chunks in `chunks` that still need rendering."""
    return [i for i in chunks if not chunk_done(sdir, spans, i)]


def section_work(sdir, spans, secs):
    """Frames still to render in each section."""
    return [sum(spans[i][1] for i in remaining(sdir, spans, range(a, b))) for a, b in secs]


def assign(work, workers, leader, fixed=None):
    """Hand out sections with work left: each to the worker with the least frames so far,
    longest first. `fixed` maps sections to a worker up front (the leader keeps the
    section it is already rendering). Returns {worker: [section, ...]}."""
    order = [leader] + [w for w in workers if w != leader]
    load = {w: 0 for w in order}
    plan = {w: [] for w in order}
    for section, worker in (fixed or {}).items():
        if work[section] > 0:
            plan.setdefault(worker, []).append(section)
            load[worker] = load.get(worker, 0) + work[section]
    free = sorted((s for s in range(len(work)) if work[s] > 0 and s not in (fixed or {})),
                  key=lambda s: (-work[s], s))
    for section in free:
        worker = min(order, key=lambda w: (load[w], order.index(w)))
        plan[worker].append(section)
        load[worker] += work[section]
    return {w: sorted(s) for w, s in plan.items() if s}


def make_job(session, leader, key, spans, secs, assignment, prompt=None, workflow=None,
             filename=None, stop_when_done=True):
    return {"session": session, "leader": leader, "plan": key,
            "spans": [list(map(int, s[:2])) + [bool(s[2])] for s in spans],
            "sections": secs, "assignment": assignment, "prompt": prompt,
            "workflow": workflow, "filename": filename,
            "stop_when_done": bool(stop_when_done), "updated": time.time()}


def job_path(sdir):
    return os.path.join(sdir, JOB_FILE)


def load_job(sdir):
    return read_json(job_path(sdir))


def save_job(sdir, job):
    job["updated"] = time.time()
    write_json(job_path(sdir), job)


def owned_chunks(job, worker):
    """Chunk indices of the sections assigned to `worker`."""
    secs = job["sections"]
    return [i for s in job["assignment"].get(worker, []) for i in range(*secs[s])]


def reassign(sdir, job, live, leader=None):
    """Give the unfinished sections of workers that have gone silent (and any section
    nobody owns) to live workers; live owners keep theirs. `leader` takes over from a
    silent leader. Returns True if the job changed."""
    leader = leader or job["leader"]
    alive = [leader] + [w for w in live if w != leader]
    work = section_work(sdir, job["spans"], job["sections"])
    kept = {s: w for w, ss in job["assignment"].items() if w in alive for s in ss}
    plan = assign(work, alive, leader, fixed=kept)
    before = {w: sorted(s for s in ss if work[s]) for w, ss in job["assignment"].items()}
    changed = plan != {w: ss for w, ss in before.items() if ss} or leader != job["leader"]
    job["assignment"], job["leader"] = plan, leader
    return changed


def all_done(sdir, spans):
    return not remaining(sdir, spans, range(len(spans)))


def is_finished(sdir):
    return os.path.exists(os.path.join(sdir, FINISHED))
