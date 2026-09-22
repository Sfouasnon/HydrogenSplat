"""Project folder and manifest (strategy §3).

One folder per capture; everything the run touches lives inside it::

    <project>/
      manifest.json        state machine + metrics + tool versions + calibration profile id
      source/              clip + .md5 + probe.json
      select/frames/       VID_NNN_FFFF_2x1.jpg, selection.json, contact.jpg
      solve/               images/{L,R}, sparse/rig, sfm_report.json, per_image.json, coverage.json
      scale/               scale_report.json (hs scale: the board's corners, factor and plane)
      train/dataset/       undistorted PINHOLE COLMAP set + rig.npz;  train/exports/export_NNNNN.ply
      move/                <name>.json, <name>_aim_check.jpg
      render/<name>/       frame_%04d.png (deleted after encode unless kept), <name>_1920.mp4, <name>_1080x1350.mp4
      logs/                <stage>.log — child stdout+stderr verbatim

manifest.json holds, per stage: status (pending / running / done / failed / stale), started /
finished times, the exact argv, metrics, checks and artifacts. Re-running a stage marks every
downstream stage ``stale``; the app offers to re-run them. Re-running deletes only that
stage's own folder.

``.hs.lock`` at the project root holds the pid and stage of the one run that may touch the
project. Recording a pid per stage tells a later reader that a crash happened; it never
stopped a second process from wiping a folder the first was still writing into. begin()
takes the lock, finish() releases it, and a lock whose pid is dead is stale and reclaimed.
"""
import hashlib
import json
import os
import shutil
import sys
import time

from . import events

MANIFEST_VERSION = 1

# Stage order for stale propagation. "prune" hangs off train and is optional; "render"
# depends on train (the ply) and move (the path). "scale" (hs scale: a ChArUco board's metric
# scale applied to the solve) sits between solve and train but is optional: train REQUIRES
# solve only, a stereo solve is metric already, and a project that never runs scale keeps it
# pending. A manifest written before scale existed gains a pending entry on load (__init__).
STAGES = ["ingest", "select", "solve", "scale", "train", "move", "prune", "render", "views"]
# "exposure" and "masks" are not in STAGES — they are operations on the solve output rather
# than steps in the chain — but they write into train/dataset, which `solve` deletes and
# rewrites. Without them here a re-solve leaves both claiming `done` while their outputs are
# gone, and the next train runs on unnormalised, unmasked images with a manifest that says
# otherwise. Being outside STAGES means they never block a stage; it must not mean they can
# lie about being current.
# scale rewrites rig.npz and the sparse model in new units — a similarity, so no pixel and no
# projection changes: exposure and masks stay current, and everything measured in mm goes stale.
DOWNSTREAM = {
    "ingest": ["select", "solve", "scale", "exposure", "masks", "train", "move", "prune", "render", "views"],
    "select": ["solve", "scale", "exposure", "masks", "train", "move", "prune", "render", "views"],
    "solve": ["scale", "exposure", "masks", "train", "move", "prune", "render", "views"],
    "scale": ["train", "move", "prune", "render", "views"],
    "train": ["prune", "render", "views"],
    "move": ["render"],
    "prune": ["render"],
    "render": [],
    "views": [],
}
STAGE_DIR = {
    "ingest": "source", "select": "select", "solve": "solve", "scale": "scale", "train": "train",
    "move": "move", "prune": "prune", "render": "render", "views": "views",
}
# hard prerequisites checked before a stage starts
REQUIRES = {
    "ingest": [], "select": ["ingest"], "solve": ["select"], "scale": ["solve"], "train": ["solve"],
    "move": ["solve"], "prune": ["train"], "render": ["train", "move"], "views": ["solve", "train"],
    # not in STAGES: operations on the solve output that do not join the state machine
    "masks": ["solve"], "exposure": ["solve"], "archive": ["solve"], "grade": ["render"],
    "merge": ["train"],
    # post-hoc operations on a trained model; --ply stands in for train (the stage passes satisfied)
    "split": ["train"],
}


def md5_file(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def dir_digest(root, exts):
    """md5 of (relative path, size, content md5) for every matching file, and the file count.
    The same digest identifies a training image set in an archive manifest and in the train
    stage's dataset fingerprint, so the two can be compared."""
    h, n = hashlib.md5(), 0
    if not os.path.isdir(root):
        return None, 0
    for dirpath, _, files in os.walk(root):
        for f in sorted(files):
            if not f.lower().endswith(exts):
                continue
            p = os.path.join(dirpath, f)
            rel = os.path.relpath(p, root)
            h.update(rel.encode()); h.update(str(os.path.getsize(p)).encode())
            h.update(md5_file(p).encode())
            n += 1
    return h.hexdigest(), n


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def parse_iso(s):
    """Inverse of now_iso(); epoch seconds, or None."""
    try:
        return time.mktime(time.strptime(s, "%Y-%m-%dT%H:%M:%S%z"))
    except (TypeError, ValueError):
        return None


LOCK_FILE = ".hs.lock"
# source.kind values that go through the frames (monocolmap) route
FRAME_KINDS = ("array", "mono")


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError, TypeError):
        return False
    return True


class Project:
    def __init__(self, root, create=False):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.manifest_path = os.path.join(self.root, "manifest.json")
        if os.path.exists(self.manifest_path):
            self.m = json.load(open(self.manifest_path))
        elif create:
            os.makedirs(self.root, exist_ok=True)
            self.m = {
                "version": MANIFEST_VERSION,
                "name": os.path.basename(self.root),
                "created": now_iso(),
                "profile_id": None,
                "tools": {},
                "source": {},
                "stages": {s: {"status": "pending"} for s in STAGES},
            }
            self.save()
        else:
            raise events.StageError(f"no manifest.json in {self.root}",
                                    hint="run `hs ingest --project DIR --clip CLIP` first")
        self.m.setdefault("stages", {})
        for s in STAGES:
            self.m["stages"].setdefault(s, {"status": "pending"})
        self.reconcile()

    def reconcile(self):
        """A stage left `running` by a crash, a ^C or a closed lid is not running any more.

        Every begin() records its pid; if that process is gone, the stage failed and should
        say so rather than blocking the next stage with "solve is running". Train is the one
        that matters — 55 minutes, and the app offers resume-or-restart from this state."""
        changed = False
        for name, st in self.m["stages"].items():
            if st.get("status") != "running":
                continue
            pid = st.get("pid")
            if pid == os.getpid() or (pid and _pid_alive(pid)):
                continue
            st["status"] = "failed"
            st["error"] = f"interrupted (process {pid} is no longer running)"
            st["finished"] = now_iso()
            st.pop("pid", None)
            changed = True
        if changed:
            self.save()

    # ------------------------------------------------------------------ paths
    def path(self, *parts):
        return os.path.join(self.root, *parts)

    def rel(self, path):
        """Project-relative path for artifact events."""
        return os.path.relpath(os.path.abspath(path), self.root)

    def stage_dir(self, stage):
        return self.path(STAGE_DIR[stage])

    def log_path(self, stage):
        return self.path("logs", f"{stage}.log")

    @property
    def clip(self):
        c = self.m.get("source", {}).get("clip")
        return self.path(c) if c else None

    @property
    def source_kind(self):
        """"stereo" (a Hydrogen 2x1 clip — manifests from before arrays have no kind at all),
        "array" (one frame from each of several cameras) or "mono" (frames from one camera)."""
        return self.m.get("source", {}).get("kind") or "stereo"

    @property
    def frames_route(self):
        """True for a source ingested as frames (array or mono): no clip, select already done,
        solve through monocolmap.py. What differs between the two is only where the number of
        cameras matters — e.g. hs exposure's median refusal, which is about an array."""
        return self.source_kind in FRAME_KINDS

    @property
    def frames_dir(self):
        return self.path("select", "frames")

    @property
    def dataset_dir(self):
        return self.path("train", "dataset")

    @property
    def rig_npz(self):
        return self.path("train", "dataset", "rig.npz")

    @property
    def exports_dir(self):
        return self.path("train", "exports")

    # ------------------------------------------------------------------ manifest
    def save(self):
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.m, f, indent=1, default=events._default)
        os.replace(tmp, self.manifest_path)

    def stage(self, name):
        """Stages outside STAGES (exposure) record here too, without joining the state machine."""
        return self.m["stages"].setdefault(name, {"status": "pending"})

    def status(self, name):
        return self.stage(name).get("status", "pending")

    def require(self, stage, satisfied=()):
        """Raise unless every prerequisite stage is done (stale counts as done, with a
        warning — the user may knowingly re-render on an old solve). A prerequisite named in
        ``satisfied`` is taken as met by other evidence — hs render passes "move" when the
        move json already exists, since the app's keyframe editor writes move/<name>.json
        itself and never runs `hs move`, so stages.move stays pending on those projects."""
        for pre in REQUIRES[stage]:
            if pre in satisfied:
                continue
            st = self.status(pre)
            if st == "done":
                continue
            if st == "stale":
                events.check(stage, f"upstream_{pre}_fresh", False,
                             value=f"{pre} is stale (an earlier stage re-ran); results may not correspond")
                continue
            raise events.StageError(f"stage '{pre}' is {st}; '{stage}' needs it done",
                                    hint=f"run `hs {pre} --project {self.root}` first")

    # ------------------------------------------------------------------ lock
    @property
    def lock_path(self):
        return self.path(LOCK_FILE)

    def lock_holder(self):
        """{"pid", "stage", "started"} of the run holding the lock, or None."""
        try:
            with open(self.lock_path) as f:
                info = json.load(f)
        except (OSError, ValueError):
            return None
        return info if isinstance(info, dict) else None

    def acquire(self, stage):
        """Take the project lock for this process, or raise if another live process holds it.

        Every operation that writes into the project — the STAGES chain and the operations
        outside it (exposure, masks, archive) — must hold this before touching a folder,
        because begin() wipes the stage folder and solve rewrites train/dataset: a second
        `hs` on the same project would otherwise erase work the first is still producing.
        Re-entrant for the holding process (selftest runs several stages in one process)."""
        os.makedirs(self.root, exist_ok=True)
        info = {"pid": os.getpid(), "stage": stage, "started": now_iso()}
        for _ in range(2):
            try:
                fd = os.open(self.lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            except FileExistsError:
                holder = self.lock_holder()
                pid = holder.get("pid") if holder else None
                if pid == os.getpid():
                    with open(self.lock_path, "w") as f:
                        json.dump(info, f)
                    return
                if pid and _pid_alive(pid):
                    raise events.StageError(
                        f"project is busy: '{holder.get('stage')}' is running as pid {pid} "
                        f"(since {holder.get('started')})",
                        hint=f"wait for it, or if that process is not an hs run remove {self.lock_path}")
                # stale (crashed / unreadable) lock: reclaim it
                try:
                    os.remove(self.lock_path)
                except OSError:
                    pass
                continue
            with os.fdopen(fd, "w") as f:
                json.dump(info, f)
            return
        raise events.StageError(f"could not take the project lock at {self.lock_path}")

    def release(self):
        """Drop the lock if this process holds it. Safe to call when it does not."""
        holder = self.lock_holder()
        if holder and holder.get("pid") == os.getpid():
            try:
                os.remove(self.lock_path)
            except OSError:
                pass

    def begin(self, stage, argv, clean=True):
        """Mark a stage running, wipe its folder (only its own), mark downstream stale.
        Takes the project lock first; a second process gets a StageError, not a wiped folder."""
        self.acquire(stage)
        d = self.stage_dir(stage)
        if clean and os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
        os.makedirs(self.path("logs"), exist_ok=True)
        prev = self.m["stages"].get(stage, {})
        self.m["stages"][stage] = {
            "status": "running", "started": now_iso(), "finished": None,
            "argv": list(argv), "pid": os.getpid(),
            "metrics": {}, "checks": [], "artifacts": [],
        }
        # move and render hold several named results per project; keep the earlier ones
        if not clean and "runs" in prev:
            self.m["stages"][stage]["runs"] = prev["runs"]
        for ds in DOWNSTREAM[stage]:
            if self.status(ds) in ("done", "failed", "running"):
                self.m["stages"][ds]["status"] = "stale"
        self.save()

    def finish(self, stage, ok=True, error=None):
        st = self.m["stages"][stage]
        st["status"] = "done" if ok else "failed"
        st["finished"] = now_iso()
        st.pop("pid", None)
        if error:
            st["error"] = error
        self.save()
        self.release()

    # recorders that also emit the event
    def metric(self, stage, name, value, **extra):
        self.stage(stage).setdefault("metrics", {})[name] = value
        events.metric(stage, name, value, **extra)

    def check(self, stage, name, ok, value=None, needs_human=False, **extra):
        rec = {"name": name, "ok": bool(ok)}
        if value is not None:
            rec["value"] = value
        if needs_human:
            rec["needs_human"] = True
        self.stage(stage).setdefault("checks", []).append(rec)
        events.check(stage, name, ok, value, needs_human, **extra)
        return bool(ok)

    def artifact(self, stage, path, kind):
        rel = self.rel(path)
        self.stage(stage).setdefault("artifacts", []).append({"path": rel, "kind": kind})
        events.artifact(stage, rel, kind)

    def record_run(self, stage, name, **extra):
        """Snapshot this run's metrics/checks/artifacts under stages.<stage>.runs.<name>
        (move and render produce several named results per project)."""
        st = self.m["stages"][stage]
        rec = {"metrics": dict(st.get("metrics", {})), "checks": list(st.get("checks", [])),
               "artifacts": list(st.get("artifacts", [])), "finished": now_iso()}
        rec.update(extra)
        st.setdefault("runs", {})[name] = rec
        self.save()

    def record_tool(self, name, info):
        self.m.setdefault("tools", {})[name] = info

    def all_checks_ok(self, stage):
        return all(c["ok"] for c in self.stage(stage).get("checks", []))


def tool_versions():
    """Python-side tool versions recorded into every manifest a stage touches."""
    info = {"python": sys.version.split()[0], "python_exe": sys.executable}
    for mod in ("numpy", "cv2", "pycolmap"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:
            info[mod] = None
    return info
