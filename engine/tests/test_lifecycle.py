#!/usr/bin/env python3
"""Regression tests for the project lifecycle guards (review 2026-09-14, the five High items).

Standard library unittest plus numpy / cv2, so they run wherever `hs` imports:

    cd engine && python3 -m unittest tests.test_lifecycle -v
    cd engine && python3 tests/test_lifecycle.py

No Brush: `hs train` runs against tests/fakebin/brush (HS_PYTHON picks the interpreter).
Every test builds its own project in a temporary folder and never touches fixtures/.
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from argparse import Namespace

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(HERE)
sys.path.insert(0, ENGINE)

import numpy as np  # noqa: E402

from hs import events, keepawake, runner  # noqa: E402
from hs.project import LOCK_FILE, Project, md5_file, now_iso  # noqa: E402
from hs.stages import archive, exposure, grade, ingest, phone, render, replay, train  # noqa: E402
from hs import framing  # noqa: E402

FAKEBIN = os.path.join(HERE, "fakebin")


def write_ply(path, n=10, seed=0):
    rng = np.random.default_rng(seed)
    props = ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2"]
    hdr = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n
           + "".join(f"property float {p}\n" for p in props) + "end_header\n")
    with open(path, "wb") as f:
        f.write(hdr.encode() + rng.normal(size=(n, len(props))).astype("<f4").tobytes())


def write_jpg(path, level, size=(32, 24)):
    import cv2
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = np.full((size[1], size[0], 3), level, np.uint8)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])


def write_rig(path, seed):
    np.savez(path, names=np.array(["cap000_L"]), K=np.eye(3)[None], R=np.eye(3)[None],
             t=np.zeros((1, 3)), pts=np.random.default_rng(seed).normal(size=(50, 3)), w=32, h=24)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hs-test-")
        self.root = os.path.join(self.tmp, "proj")
        self.out = io.StringIO()            # swallow the JSON-lines event stream
        self._redir = contextlib.redirect_stdout(self.out)
        self._redir.__enter__()

    def tearDown(self):
        self._redir.__exit__(None, None, None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def solved_project(self, n_frames=1):
        """A project whose solve is done: dataset with sparse/, rig.npz and two views."""
        pj = Project(self.root, create=True)
        pj.m["stages"]["ingest"] = {"status": "done"}
        pj.m["stages"]["select"] = {"status": "done"}
        pj.m["stages"]["solve"] = {"status": "done", "metrics": {"num_frames": n_frames},
                                   "started": now_iso(), "finished": now_iso()}
        os.makedirs(os.path.join(pj.dataset_dir, "sparse"))
        for f, body in (("cameras.txt", "# cameras\n"), ("images.txt", "# images\n")):
            open(os.path.join(pj.dataset_dir, "sparse", f), "w").write(body)
        write_rig(pj.rig_npz, seed=1)
        write_jpg(os.path.join(pj.dataset_dir, "images", "L", "cap000_L.jpg"), 60)
        write_jpg(os.path.join(pj.dataset_dir, "images", "L", "cap001_L.jpg"), 120)
        pj.save()
        return pj

    def trained_project(self, total=40):
        """solved_project plus a train stage recorded the way hs train records it."""
        pj = self.solved_project()
        os.makedirs(pj.exports_dir)
        final = os.path.join(pj.exports_dir, f"export_{total:05d}.ply")
        write_ply(final, seed=7)
        pj.m["stages"]["train"] = {
            "status": "done", "started": now_iso(), "finished": now_iso(),
            "metrics": {"final_export": pj.rel(final), "final_export_md5": md5_file(final),
                        "dataset_fingerprint": {"rig_npz_md5": md5_file(pj.rig_npz)}},
            "checks": [], "artifacts": []}
        pj.save()
        return pj, final


# ------------------------------------------------------------------ 1. train resume
class TrainResume(Base):
    def train_args(self, **kw):
        base = dict(brush=os.path.join(FAKEBIN, "brush"), total_train_iters=self.TOTAL, growth_stop_iter=30,
                    refine_every=10, split_at_screen_size=None, export_every=10, resume_from=None,
                    start_iter=None, no_caffeinate=True, brush_args="", exclude="", no_masks=False,
                    min_scale_factor=None)
        base.update(kw)
        return Namespace(**base)

    TOTAL = 40

    @classmethod
    def exp(cls, it):
        # Brush (and the fake) zero-pad {iter} to the digit count of --total-train-iters:
        # 40 iters -> export_40.ply, 40000 -> export_02500.ply. Name exports the same way here.
        return f"export_{it:0{len(str(cls.TOTAL))}d}.ply"

    def setUp(self):
        super().setUp()
        os.environ["HS_PYTHON"] = sys.executable
        os.environ["HS_FAKE_REFINE_STOP"] = "0"
        os.environ["HS_FAKE_QUIET_TAIL"] = "0"

    def test_resume_keeps_its_checkpoint_in_exports(self):
        pj = self.solved_project()
        os.makedirs(pj.exports_dir)
        for it in (10, 20, 30):
            write_ply(os.path.join(pj.exports_dir, self.exp(it)), seed=it)
        ckpt = os.path.join(pj.exports_dir, self.exp(20))
        ckpt_md5 = md5_file(ckpt)
        train.run(self.train_args(resume_from=ckpt), pj)
        self.assertTrue(os.path.exists(ckpt), "the checkpoint was deleted by the run that resumed from it")
        self.assertEqual(md5_file(ckpt), ckpt_md5)
        self.assertTrue(os.path.exists(os.path.join(pj.exports_dir, self.exp(10))), "history before the checkpoint kept")
        self.assertTrue(os.path.exists(os.path.join(pj.exports_dir, self.exp(40))), "final export written")
        tm = pj.stage("train")["metrics"]
        self.assertEqual(tm["resume_from_md5"], ckpt_md5)
        self.assertEqual(tm["start_iter"], 20)
        self.assertEqual(tm["dataset_fingerprint"]["rig_npz_md5"], md5_file(pj.rig_npz))
        self.assertEqual(tm["dataset_fingerprint"]["images"]["count"], 2)
        self.assertFalse(os.path.exists(os.path.join(pj.dataset_dir, "init.ply")), "init.ply removed after the run")
        self.assertEqual(pj.status("train"), "done")
        self.assertFalse(os.path.exists(pj.lock_path), "lock released")

    def test_resume_rejects_a_non_ply_before_touching_exports(self):
        pj = self.solved_project()
        os.makedirs(pj.exports_dir)
        keep = os.path.join(pj.exports_dir, self.exp(10))
        write_ply(keep)
        bogus = os.path.join(self.tmp, "notaply.ply")
        open(bogus, "w").write("hello\n")
        with self.assertRaises(events.StageError):
            train.run(self.train_args(resume_from=bogus), pj)
        self.assertTrue(os.path.exists(keep), "a rejected resume must not have cleared train/exports")
        self.assertEqual(pj.status("train"), "pending")

    def test_fresh_run_still_clears_exports(self):
        pj = self.solved_project()
        os.makedirs(pj.exports_dir)
        stale = os.path.join(pj.exports_dir, self.exp(99))
        write_ply(stale)
        train.run(self.train_args(), pj)
        self.assertFalse(os.path.exists(stale))
        self.assertTrue(os.path.exists(os.path.join(pj.exports_dir, self.exp(40))))


class TrainView(TrainResume):
    """--exclude / --no-masks hand brush a symlink view and never touch the dataset."""

    def masked_project(self):
        pj = self.solved_project()
        for eye in ("L", "R"):
            for c in ("cap000", "cap001"):
                write_jpg(os.path.join(pj.dataset_dir, "images", eye, f"{c}.jpg"), 90)
                d = os.path.join(pj.dataset_dir, "masks", eye)
                os.makedirs(d, exist_ok=True)
                open(os.path.join(d, f"{c}.png"), "wb").close()
        return pj

    def argv_root(self, pj):
        argv = pj.stage("train")["brush_argv"]
        return argv[argv.index(os.path.join(FAKEBIN, "brush")) + 1]

    def test_exclude_and_no_masks(self):
        pj = self.masked_project()
        before = sorted(os.path.relpath(os.path.join(r, f), pj.dataset_dir)
                        for r, _d, fs in os.walk(pj.dataset_dir) for f in fs)
        train.run(self.train_args(exclude="L/cap001,cap000_R", no_masks=True), pj)
        view = pj.path("train", "view")
        self.assertEqual(self.argv_root(pj), view)
        imgs = sorted(os.path.relpath(os.path.join(r, f), view)
                      for r, _d, fs in os.walk(os.path.join(view, "images")) for f in fs)
        self.assertNotIn("images/L/cap001.jpg", imgs)
        self.assertNotIn("images/R/cap000.jpg", imgs)
        self.assertIn("images/L/cap000.jpg", imgs)
        self.assertFalse(os.path.exists(os.path.join(view, "masks")), "--no-masks left masks in the view")
        self.assertTrue(os.path.islink(os.path.join(view, "sparse")))
        after = sorted(os.path.relpath(os.path.join(r, f), pj.dataset_dir)
                       for r, _d, fs in os.walk(pj.dataset_dir) for f in fs)
        self.assertEqual(before, after, "the dataset itself must not change")
        tm = pj.stage("train")["metrics"]
        self.assertEqual(tm["excluded_views"], ["L/cap001", "R/cap000"])
        self.assertFalse(tm["masks_used"])
        checks = {c["name"]: c for c in pj.stage("train")["checks"]}
        self.assertTrue(checks["view_count_matches"]["ok"], checks["view_count_matches"])

    def test_masks_kept_minus_excluded(self):
        pj = self.masked_project()
        train.run(self.train_args(exclude="R/cap001"), pj)
        view = pj.path("train", "view")
        self.assertTrue(os.path.exists(os.path.join(view, "masks", "L", "cap001.png")))
        self.assertFalse(os.path.exists(os.path.join(view, "masks", "R", "cap001.png")))
        self.assertTrue(pj.stage("train")["metrics"]["masks_used"])

    def test_unknown_view_is_refused_before_anything_runs(self):
        pj = self.masked_project()
        with self.assertRaises(events.StageError):
            train.run(self.train_args(exclude="L/cap099"), pj)
        with self.assertRaises(events.StageError):
            train.run(self.train_args(exclude="left064"), pj)

    def test_plain_run_trains_on_the_dataset_and_drops_a_stale_view(self):
        pj = self.masked_project()
        train.run(self.train_args(exclude="L/cap001"), pj)
        pj.m["stages"]["train"]["status"] = "pending"
        pj.save()
        train.run(self.train_args(), pj)
        self.assertEqual(self.argv_root(pj), pj.dataset_dir)
        self.assertFalse(os.path.exists(pj.path("train", "view")))


# ------------------------------------------------------------------ 2. exposure dry-run
class ExposureDryRun(Base):
    def args(self, **kw):
        base = dict(mode="rgb", restore=False, dry_run=False)
        base.update(kw)
        return Namespace(**base)

    def images(self, pj):
        d = os.path.join(pj.dataset_dir, "images", "L")
        return {f: md5_file(os.path.join(d, f)) for f in sorted(os.listdir(d))}

    def test_dry_run_with_backup_changes_nothing(self):
        pj = self.solved_project()
        pj.m["stages"]["train"] = {"status": "done"}
        pj.save()
        exposure.run(self.args(), pj)                       # a real run: creates the backup
        self.assertEqual(pj.status("exposure"), "done")
        self.assertEqual(pj.status("train"), "stale")
        pj.m["stages"]["train"]["status"] = "done"         # pretend a retrain happened
        pj.save()
        before = self.images(pj)
        self.assertTrue(os.path.isdir(pj.path("solve", "exposure_backup")))
        exposure.run(self.args(dry_run=True), pj)
        self.assertEqual(self.images(pj), before, "dry run rewrote the training images")
        self.assertEqual(pj.status("train"), "done", "dry run marked train stale")
        self.assertEqual(pj.status("exposure"), "done")
        dr = pj.stage("exposure")["dry_run"]
        self.assertEqual(dr["metrics"]["views"], 2)
        self.assertEqual(dr["metrics"]["measured"], "solve/exposure_backup")
        self.assertGreater(dr["metrics"]["luma_spread_before"], 1.5, "measured the originals, not the corrected set")

    def test_dry_run_without_backup_writes_nothing(self):
        pj = self.solved_project()
        pj.m["stages"]["train"] = {"status": "done"}
        pj.save()
        before = self.images(pj)
        exposure.run(self.args(dry_run=True), pj)
        self.assertEqual(self.images(pj), before)
        self.assertFalse(os.path.isdir(pj.path("solve", "exposure_backup")))
        self.assertFalse(os.path.exists(os.path.join(pj.dataset_dir, "exposure.json")))
        self.assertEqual(pj.status("exposure"), "pending")
        self.assertEqual(pj.status("train"), "done")
        self.assertFalse(os.path.exists(pj.lock_path))

    def test_dry_run_and_restore_together_is_an_error(self):
        pj = self.solved_project()
        with self.assertRaises(events.StageError):
            exposure.run(self.args(dry_run=True, restore=True), pj)


# ------------------------------------------------------------------ 3. archive name reuse
class ArchiveReuse(Base):
    def args(self, **kw):
        base = dict(name="keep", ply=None, link=False, no_images=False, force=False)
        base.update(kw)
        return Namespace(**base)

    def test_existing_name_is_refused_and_untouched(self):
        pj, final = self.trained_project()
        archive.run(self.args(), pj)
        out = pj.path("archive", "keep")
        man1 = json.load(open(os.path.join(out, "manifest.json")))
        self.assertEqual(man1["ply"]["md5"], md5_file(os.path.join(out, os.path.basename(final))))
        # a new solve + a new model under the same archive name
        write_rig(pj.rig_npz, seed=2)
        write_ply(final, seed=99)
        pj.m["stages"]["train"]["metrics"]["final_export_md5"] = md5_file(final)
        pj.save()
        with self.assertRaises(events.StageError):
            archive.run(self.args(), pj)
        man2 = json.load(open(os.path.join(out, "manifest.json")))
        self.assertEqual(man1, man2, "a refused archive must leave the old one exactly as it was")
        self.assertEqual(man2["ply"]["md5"], md5_file(os.path.join(out, os.path.basename(final))))
        self.assertFalse(os.path.exists(pj.path("archive", ".keep.building")))
        self.assertFalse(os.path.exists(pj.lock_path))

    def test_force_replaces_the_whole_archive_consistently(self):
        pj, final = self.trained_project()
        archive.run(self.args(), pj)
        out = pj.path("archive", "keep")
        open(os.path.join(out, "leftover.txt"), "w").write("from the first archive\n")
        write_rig(pj.rig_npz, seed=2)
        write_ply(final, seed=99)
        pj.m["stages"]["train"]["metrics"]["final_export_md5"] = md5_file(final)
        pj.save()
        archive.run(self.args(force=True), pj)
        man = json.load(open(os.path.join(out, "manifest.json")))
        arch_ply = os.path.join(out, os.path.basename(final))
        self.assertEqual(man["ply"]["md5"], md5_file(arch_ply))
        self.assertEqual(md5_file(arch_ply), md5_file(final), "the archived ply is the current source")
        self.assertEqual(man["rig_npz_md5"], md5_file(os.path.join(out, "rig.npz")))
        self.assertEqual(man["rig_npz_md5"], md5_file(pj.rig_npz))
        self.assertFalse(os.path.exists(os.path.join(out, "leftover.txt")), "--force replaces, it does not merge")


# ------------------------------------------------------------------ 4. render lineage
class RenderLineage(Base):
    def move_file(self, pj):
        os.makedirs(pj.path("move"), exist_ok=True)
        mv = pj.path("move", "boom.json")
        json.dump({"fps": 30, "frames": []}, open(mv, "w"))
        return mv

    def args(self, ply=None):
        return Namespace(ply=ply, allow_mismatch=False)

    def test_model_from_an_earlier_solve_is_refused(self):
        pj, final = self.trained_project()
        write_rig(pj.rig_npz, seed=2)                 # a re-solve: new frame, model survives
        mv = self.move_file(pj)                        # freshly built move passes the mtime check
        with self.assertRaises(events.StageError):
            render.guards(self.args(), pj, final, mv)
        checks = {c["name"]: c for c in pj.stage("render")["checks"]}
        self.assertTrue(checks["ply_md5_matches"]["ok"], "the old identity guard is exactly what lets this through")
        self.assertTrue(checks["move_newer_than_solve"]["ok"])
        self.assertFalse(checks["model_matches_solve"]["ok"])

    def test_model_trained_on_this_solve_passes(self):
        pj, final = self.trained_project()
        mv = self.move_file(pj)
        render.guards(self.args(), pj, final, mv)
        checks = {c["name"]: c for c in pj.stage("render")["checks"]}
        self.assertTrue(all(c["ok"] for c in checks.values()), checks)

    def test_pruned_ply_must_descend_from_trains_export(self):
        pj, final = self.trained_project()
        mv = self.move_file(pj)
        os.makedirs(pj.path("prune"))
        pruned = pj.path("prune", "export_00040_pruned_r03.ply")
        write_ply(pruned, seed=3)
        pj.m["stages"]["prune"] = {"status": "done", "metrics": {
            "output_ply": pj.rel(pruned), "input_ply_md5": "0" * 32}}   # pruned from some other model
        pj.save()
        with self.assertRaises(events.StageError):
            render.guards(self.args(ply=pruned), pj, pruned, mv)
        pj.m["stages"]["prune"]["metrics"]["input_ply_md5"] = md5_file(final)
        pj.m["stages"]["render"]["checks"] = []
        pj.save()
        render.guards(self.args(ply=pruned), pj, pruned, mv)
        checks = {c["name"]: c for c in pj.stage("render")["checks"]}
        self.assertTrue(checks["model_matches_solve"]["ok"])

    def test_legacy_prune_older_than_train_is_refused(self):
        # 2026-09-13_coins: a prune with no input md5, run before the current train, was
        # assumed current and rendered in another solve's frame.
        pj, final = self.trained_project()
        mv = self.move_file(pj)
        os.makedirs(pj.path("prune"))
        pruned = pj.path("prune", "export_00040_pruned_r03.ply")
        write_ply(pruned, seed=3)
        t_fin = pj.m["stages"]["train"]["finished"]
        pj.m["stages"]["prune"] = {"status": "done", "finished": "2000-01-01T00:00:00+0000",
                                   "metrics": {"output_ply": pj.rel(pruned)}}
        pj.save()
        with self.assertRaises(events.StageError):
            render.guards(self.args(ply=pruned), pj, pruned, mv)
        self.assertFalse({c["name"]: c for c in pj.stage("render")["checks"]}["ply_md5_matches"]["ok"])
        pj.m["stages"]["prune"]["finished"] = t_fin          # same second as train: accepted
        pj.m["stages"]["render"]["checks"] = []
        pj.save()
        render.guards(self.args(ply=pruned), pj, pruned, mv)
        self.assertTrue({c["name"]: c for c in pj.stage("render")["checks"]}["ply_md5_matches"]["ok"])

    def test_legacy_model_falls_back_to_timestamps(self):
        pj, final = self.trained_project()
        del pj.m["stages"]["train"]["metrics"]["dataset_fingerprint"]
        pj.save()
        mv = self.move_file(pj)
        # rig.npz older than the training run: accepted, and says it is a timestamp-only verdict
        old = time.time() - 3600
        os.utime(pj.rig_npz, (old, old))
        render.guards(self.args(), pj, final, mv)
        checks = {c["name"]: c for c in pj.stage("render")["checks"]}
        self.assertTrue(checks["model_matches_solve"]["ok"])
        self.assertIn("legacy", checks["model_matches_solve"]["value"])
        # rig.npz newer than the training run: refused
        pj.m["stages"]["train"]["started"] = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(time.time() - 7200))
        pj.m["stages"]["render"]["checks"] = []
        pj.save()
        new = time.time()
        os.utime(pj.rig_npz, (new, new))
        with self.assertRaises(events.StageError):
            render.guards(self.args(), pj, final, mv)


# ------------------------------------------------------------------ 5. project lock
class ProjectLock(Base):
    def test_second_process_cannot_begin_while_first_runs(self):
        a = Project(self.root, create=True)
        a.begin("select", argv=["hs", "select"])
        marker = a.path("select", "in-progress.txt")
        open(marker, "w").write("first run's output\n")
        b = Project(self.root)
        holder = b.lock_holder()
        holder["pid"] = os.getpid() + 100000               # stand in for another live process
        json.dump(holder, open(b.lock_path, "w"))
        # a foreign pid that is "alive": monkeypatch liveness so the test is deterministic
        from hs import project as pmod
        orig = pmod._pid_alive
        pmod._pid_alive = lambda pid: True
        try:
            with self.assertRaises(events.StageError):
                b.begin("select", argv=["hs", "select"])
        finally:
            pmod._pid_alive = orig
        self.assertTrue(os.path.exists(marker), "the second begin() wiped the folder the first run was writing")

    def test_lock_released_on_finish_and_stale_lock_reclaimed(self):
        a = Project(self.root, create=True)
        a.begin("select", argv=[])
        self.assertTrue(os.path.exists(a.lock_path))
        a.finish("select", ok=True)
        self.assertFalse(os.path.exists(a.lock_path))
        # a crashed run leaves a lock whose pid is gone
        json.dump({"pid": 2 ** 22 + 12345, "stage": "solve", "started": now_iso()}, open(a.lock_path, "w"))
        b = Project(self.root)
        b.begin("select", argv=[])
        self.assertEqual(b.lock_holder()["pid"], os.getpid())
        b.finish("select", ok=False, error="x")
        self.assertFalse(os.path.exists(os.path.join(self.root, LOCK_FILE)))

    def test_same_process_is_reentrant(self):
        a = Project(self.root, create=True)
        a.begin("ingest", argv=[])
        a.begin("select", argv=[])            # selftest runs stages back to back in one process
        self.assertEqual(a.lock_holder()["stage"], "select")


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ------------------------------------------------------------------ 6. keep awake / sleep detection
class FakeClocks:
    """Stands in for keepawake.clocks(): sleep() advances only the through-sleep clock."""
    def __init__(self):
        self.through = self.awake = 1000.0

    def __call__(self):
        return self.through, self.awake

    def run(self, s):
        self.through += s
        self.awake += s

    def sleep(self, s):
        self.through += s


class KeepAwake(Base):
    def setUp(self):
        super().setUp()
        self._real = keepawake.clocks
        self.clk = FakeClocks()
        keepawake.clocks = self.clk

    def tearDown(self):
        keepawake.clocks = self._real
        super().tearDown()

    def test_sleepwatch_counts_only_real_gaps(self):
        w = keepawake.SleepWatch()
        self.clk.run(3)
        self.assertEqual(w.poll(), 0.0)
        self.clk.run(30); self.clk.sleep(965)
        self.assertEqual(w.poll(), 965.0)
        self.clk.run(10); self.clk.sleep(5)          # under SLEEP_GAP_S: jitter, not sleep
        self.assertEqual(w.poll(), 0.0)
        self.assertEqual(w.summary(), {"slept_s": 965.0, "sleeps": 1, "wall_s": 1013.0})

    def test_runner_reports_a_sleep_while_the_child_runs(self):
        slept = []

        def tick():
            if not slept:
                self.clk.sleep(600)
                slept.append(1)

        log = os.path.join(self.tmp, "x.log")
        with contextlib.redirect_stdout(io.StringIO()) as out:
            res = runner.run([sys.executable, "-c", "import time; time.sleep(0.6)"], "train",
                             log_path=log, tick=tick, tick_interval=0.1)
        self.assertEqual(res.returncode, 0)
        self.assertEqual(round(res.slept_s), 600)
        self.assertEqual(len(res.sleeps), 1)
        evs = [json.loads(l) for l in out.getvalue().splitlines()]
        self.assertTrue(any(e.get("name") == "sleep_detected" and e["value"] == 600.0 for e in evs))
        with open(log) as f:
            text = f.read()
        self.assertIn("the machine slept 10.0 min", text)
        self.assertIn("slept 600s (1x)", text)

    def test_disabled_by_flag_or_env(self):
        self.assertTrue(keepawake.disabled(Namespace(no_caffeinate=True)))
        self.assertFalse(keepawake.disabled(Namespace()))
        os.environ["HS_NO_CAFFEINATE"] = "1"
        try:
            self.assertTrue(keepawake.disabled(Namespace()))
            with keepawake.hold("train", off=keepawake.disabled(Namespace())) as h:
                self.assertIsNone(h.proc)
        finally:
            del os.environ["HS_NO_CAFFEINATE"]


class TrainKeepAwake(TrainResume):
    def test_brush_is_not_wrapped_and_the_run_records_sleep(self):
        pj = self.solved_project()
        with contextlib.redirect_stdout(io.StringIO()):
            train.run(self.train_args(), pj)
        st = pj.stage("train")
        self.assertNotIn("caffeinate", st["brush_argv"])
        self.assertIn("slept_s", st["metrics"])
        self.assertIn("wall_s", st["metrics"])
        chk = {c["name"]: c for c in st["checks"]}
        self.assertTrue(chk["no_sleep_during_run"]["ok"])


# ------------------------------------------------------------------ 7. phone listing, pull, replay (M1)
class PhoneIngest(Base):
    def setUp(self):
        super().setUp()
        os.environ["HS_PYTHON"] = sys.executable
        self.clip = os.path.join(self.tmp, "VID_20260915_145235_2x1.h4v")
        with open(self.clip, "wb") as f:
            f.write(os.urandom(300_000))
        os.environ["HS_FAKE_ADB_CLIP"] = self.clip
        os.environ.pop("HS_FAKE_ADB_MD5", None)
        self.adb = os.path.join(FAKEBIN, "adb")

    def events(self):
        return [json.loads(l) for l in self.out.getvalue().splitlines() if l.startswith("{")]

    def ingest_args(self, **kw):
        base = dict(clip=None, link=False, profile=None, ffprobe=os.path.join(FAKEBIN, "ffprobe"),
                    phone="FAKE01", remote="/sdcard/DCIM/Camera/" + os.path.basename(self.clip), adb=self.adb)
        base.update(kw)
        return Namespace(**base)

    def test_list_devices_and_clips(self):
        phone.run(Namespace(adb=self.adb, serial=None))
        ev = self.events()
        devs = next(e["value"] for e in ev if e.get("name") == "devices")
        self.assertEqual(devs[0]["serial"], "FAKE01")
        self.assertEqual(devs[0]["model"], "H1A1000")
        clips = next(e for e in ev if e.get("name") == "clips")
        self.assertEqual(clips["serial"], "FAKE01")
        self.assertEqual([c["name"] for c in clips["value"]], [os.path.basename(self.clip)])
        self.assertEqual(clips["value"][0]["bytes"], 300_000)

    def test_pull_lands_in_source_with_matching_md5(self):
        pj = Project(self.root, create=True)
        ingest.run(self.ingest_args(), pj)
        dst = pj.path("source", os.path.basename(self.clip))
        self.assertEqual(md5_file(dst), md5_file(self.clip))
        st = pj.stage("ingest")
        self.assertEqual(st["status"], "done")
        chk = {c["name"]: c for c in st["checks"]}
        self.assertTrue(chk["pull_matches_phone"]["ok"])
        self.assertTrue(chk["clip_is_2x1_video"]["ok"])
        self.assertEqual(pj.m["source"]["original_path"], "adb:FAKE01:" + self.ingest_args().remote)
        prog = [e for e in self.events() if e["ev"] == "progress" and e.get("step") == "pull"]
        self.assertEqual(prog[-1]["done"], 300_000)
        self.assertEqual(prog[-1]["total"], 300_000)

    def test_md5_mismatch_is_refused(self):
        os.environ["HS_FAKE_ADB_MD5"] = "0" * 32
        pj = Project(self.root, create=True)
        with self.assertRaises(events.StageError):
            ingest.run(self.ingest_args(), pj)
        chk = {c["name"]: c for c in pj.stage("ingest")["checks"]}
        self.assertFalse(chk["pull_matches_phone"]["ok"])

    def test_missing_remote_and_both_sources_are_refused(self):
        pj = Project(self.root, create=True)
        with self.assertRaises(events.StageError):
            ingest.run(self.ingest_args(remote="/sdcard/DCIM/Camera/nope.h4v"), pj)
        pj.release()
        with self.assertRaises(events.StageError):
            ingest.run(self.ingest_args(clip=self.clip), pj)

    def test_replay_strips_t_and_can_fail(self):
        f = os.path.join(self.tmp, "ev.jsonl")
        with open(f, "w") as fh:
            fh.write('{"t":0,"ev":"start","stage":"train"}\n{"t":0.2,"ev":"progress","stage":"train","done":5}\n')
        replay.run(Namespace(file=f, speed=100.0, gap=0.0, max_wait=0.01, fail_at=None))
        ev = self.events()
        self.assertEqual(ev, [{"ev": "start", "stage": "train"}, {"ev": "progress", "stage": "train", "done": 5}])
        with self.assertRaises(events.StageError):
            replay.run(Namespace(file=f, speed=100.0, gap=0.0, max_wait=0.01, fail_at=1))


# ------------------------------------------------------------------ 8. framing + grade
@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs ffmpeg")
class Grade(Base):
    W, H, N, FPS = 320, 180, 10, 10

    def graded_project(self, track=True):
        import cv2
        pj = self.solved_project()
        os.makedirs(pj.path("render"))
        w = cv2.VideoWriter(pj.path("render", "arc_src.avi"), cv2.VideoWriter_fourcc(*"MJPG"), self.FPS, (self.W, self.H))
        ramp = np.tile(np.arange(self.H, dtype=np.uint8)[:, None], (1, self.W))
        for _ in range(self.N):
            w.write(cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR))
        w.release()
        subprocess_run(["ffmpeg", "-loglevel", "error", "-y", "-i", pj.path("render", "arc_src.avi"),
                        "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", pj.path("render", "arc_1920.mp4")])
        if track:
            os.makedirs(pj.path("move"), exist_ok=True)
            # crown row climbs 0..90 px; depth 1000 mm, fy 1000 -> headroom 25.4 mm = 25.4 px
            rec = {"aspect": 4.0, "headroom_mm": 25.4, "native": [self.W, self.H], "fy": 1000.0, "fps": self.FPS,
                   "crown_row": [25.4 + 10 * k for k in range(self.N)], "depth_mm": [1000.0] * self.N}
            json.dump(rec, open(pj.path("move", "arc_frame.json"), "w"))
        return pj

    def args(self, **kw):
        base = dict(move="arc", lift=None, gamma=None, gain=None, lift_rgb=None, gamma_rgb=None, gain_rgb=None,
                    sharpen=0.0, aspect=4.0, headroom_mm=None, still=None, graded=False, reset=False,
                    ffmpeg="ffmpeg", crf=0)
        base.update(kw)
        return Namespace(**base)

    def test_lut_matches_ffmpeg_lutrgb(self):
        import cv2
        g = np.tile(np.arange(256, dtype=np.uint8), (2, 1))
        src, dst = os.path.join(self.tmp, "g.png"), os.path.join(self.tmp, "o.png")
        cv2.imwrite(src, np.dstack([g, g, g]))
        s = dict(grade.DEFAULTS, lift=0.03, gamma=1.2, gain=0.9, gain_rgb=[1.05, 1, 0.9], lift_rgb=[0, 0.01, -0.02], sharpen=0)
        subprocess_run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-vf", ",".join(grade.grade_filter(s)), dst])
        o = cv2.imread(dst)
        for c, bgr in ((0, 2), (1, 1), (2, 0)):
            np.testing.assert_array_equal(o[0, :, bgr], grade.lut(*grade.channel_params(s)[c]))

    def test_crop_follows_the_crown(self):
        pj = self.graded_project()
        grade.run(self.args(), pj)
        out = pj.path("render", "arc_graded.mp4")
        raw = subprocess_run(["ffmpeg", "-loglevel", "error", "-i", out, "-f", "rawvideo", "-pix_fmt", "gray", "-"])
        ch = framing.even(self.W / 4.0)
        frames = np.frombuffer(raw, np.uint8).reshape(-1, ch, self.W)
        self.assertEqual(len(frames), self.N)
        tops = [int(f[:, 5].astype(int).min()) for f in frames]    # ramp value = source row
        # crown_row - headroom = 10k, smoothed with edge padding; must rise monotonically and cover the range
        self.assertTrue(all(b >= a for a, b in zip(tops, tops[1:])), tops)
        self.assertLessEqual(abs(tops[0] - 0), 12)
        self.assertLessEqual(abs(tops[-1] - 90), 12)
        saved = json.load(open(pj.path("grade", "arc.json")))
        self.assertEqual(saved["aspect"], 4.0)

    def test_still_and_saved_settings_are_reused(self):
        pj = self.graded_project(track=False)
        grade.run(self.args(gain=1.2, still=None), pj)
        grade.run(self.args(still=3, graded=True), pj)
        ev = [json.loads(l) for l in self.out.getvalue().splitlines() if l.startswith("{")]
        crop = [e["value"] for e in ev if e.get("name") == "crop"][-1]
        self.assertEqual(crop["mode"], "centred")
        self.assertTrue(os.path.exists(pj.path("grade", "arc_still_graded.png")))
        self.assertEqual(json.load(open(pj.path("grade", "arc.json")))["gain"], 1.2)
        s = grade.settings(self.args(), json.load(open(pj.path("grade", "arc.json"))))
        self.assertEqual(s["gain"], 1.2)
        self.assertEqual(grade.settings(self.args(reset=True), {"gain": 1.2})["gain"], 1.0)


def subprocess_run(argv):
    import subprocess
    return subprocess.run(argv, check=True, capture_output=True).stdout


# ------------------------------------------------------------------ 9. brush config recorded
class BrushConfig(TrainResume):
    def test_min_scale_factor_follows_the_binary(self):
        pj = self.solved_project()
        # the fake brush prints no --help flags: an old binary -> not passed, recorded as 0
        train.run(self.train_args(), pj)
        st = pj.stage("train")
        self.assertNotIn("--min-scale-factor", st["brush_argv"])
        self.assertEqual(st["metrics"]["brush_config"]["min_scale_factor"], 0.0)
        with self.assertRaises(events.StageError):
            train.run(self.train_args(min_scale_factor=0.03), pj)

    def test_new_binary_gets_it_explicitly(self):
        pj = self.solved_project()
        real = train.brush_info
        train.brush_info = lambda b: {**real(b), "flags": ["--min-scale-factor", "--total-train-iters"]}
        try:
            train.run(self.train_args(), pj)
            argv = pj.stage("train")["brush_argv"]
            self.assertEqual(argv[argv.index("--min-scale-factor") + 1], "0.1")
            train.run(self.train_args(min_scale_factor=0.0), pj)
            argv = pj.stage("train")["brush_argv"]
            self.assertEqual(argv[argv.index("--min-scale-factor") + 1], "0.0")
            self.assertEqual(pj.stage("train")["metrics"]["brush_config"]["min_scale_factor"], 0.0)
        finally:
            train.brush_info = real


# ------------------------------------------------------------------ array source (R3D / frames)
class ArrayIngest(Base):
    """hs ingest --frames / --r3d: one frame per camera, select marked done, mono rig layout."""

    def setUp(self):
        super().setUp()
        os.environ["HS_PYTHON"] = sys.executable
        self.redline = os.path.join(FAKEBIN, "REDline")

    def events(self):
        return [json.loads(l) for l in self.out.getvalue().splitlines() if l.startswith("{")]

    def args(self, **kw):
        base = dict(clip=None, phone=None, remote=None, adb="adb", link=False, profile=None, ffprobe="ffprobe",
                    frames=None, r3d=None, take=None, redline=self.redline, res=1)
        base.update(kw)
        return Namespace(**base)

    def frames_dir(self, cams=("GA", "GB", "HA", "HB"), size=(64, 36)):
        d = os.path.join(self.tmp, "frames")
        os.makedirs(d)
        for i, c in enumerate(cams):
            write_jpg(os.path.join(d, c + ".jpg"), 40 + 30 * i, size)
        return d

    def rdm_tree(self, take="067", cams=("GA", "GB", "HA")):
        root = os.path.join(self.tmp, "RED_Footage")
        for cam in cams:
            for tk in (take, "068"):
                clip = f"{cam[0]}007_{cam[1]}{tk}_0403XX"
                d = os.path.join(root, cam, f"{cam[0]}007_ZZZZZZ.RDM", clip + ".RDC")
                os.makedirs(d)
                with open(os.path.join(d, clip + "_001.R3D"), "wb") as f:
                    f.write(bytes([ord(cam[1]) if tk == take else 1]) + b"\0" * 64)
        return root

    def test_frames_route_marks_select_done_and_records_cameras(self):
        pj = Project(self.root, create=True)
        ingest.run(self.args(frames=self.frames_dir()), pj)
        self.assertEqual(pj.m["source"]["kind"], "array")
        self.assertEqual([c["camera"] for c in pj.m["source"]["cameras"]], ["GA", "GB", "HA", "HB"])
        self.assertEqual(pj.status("ingest"), "done")
        self.assertEqual(pj.status("select"), "done")
        self.assertEqual(sorted(os.listdir(pj.frames_dir)), ["GA.jpg", "GB.jpg", "HA.jpg", "HB.jpg"])
        self.assertTrue(os.path.islink(os.path.join(pj.frames_dir, "GA.jpg")))
        chk = {c["name"]: c for c in pj.stage("ingest")["checks"]}
        self.assertTrue(chk["one_frame_size"]["ok"])
        self.assertIsNone(pj.m["profile_id"])
        self.assertIsNone(pj.lock_holder())

    def test_mixed_sizes_and_too_few_cameras_are_refused(self):
        d = self.frames_dir(cams=("GA", "GB", "HA"))
        write_jpg(os.path.join(d, "HB.jpg"), 90, (32, 18))
        pj = Project(self.root, create=True)
        with self.assertRaises(events.StageError):
            ingest.run(self.args(frames=d), pj)
        chk = {c["name"]: c for c in pj.stage("ingest")["checks"]}
        self.assertFalse(chk["one_frame_size"]["ok"])
        self.assertIn("HB", chk["one_frame_size"]["value"])
        pj.release()
        shutil.rmtree(d)
        d = self.frames_dir(cams=("GA", "GB"))
        with self.assertRaises(events.StageError):
            ingest.run(self.args(frames=d), pj)
        self.assertNotEqual(pj.status("ingest"), "done")      # cli.py turns the raise into "failed"
        self.assertEqual(pj.status("select"), "pending")

    def test_r3d_route_transcodes_the_take_through_redline(self):
        root = self.rdm_tree()
        pj = Project(self.root, create=True)
        ingest.run(self.args(r3d=root, take="67"), pj)
        cams = pj.m["source"]["cameras"]
        self.assertEqual([c["camera"] for c in cams], ["GA", "GB", "HA"])
        self.assertTrue(all(c["file"].endswith(".png") for c in cams))
        self.assertTrue(all("_A067_" in c["origin"] or "_B067_" in c["origin"] for c in cams))
        self.assertEqual((cams[0]["width"], cams[0]["height"]), (64, 36))
        import cv2
        im = cv2.imread(pj.path("source", "frames", "GB.png"))
        self.assertEqual(im.dtype, np.uint8)
        self.assertEqual(int(im[0, -1, 1]), ord("B"))       # 16-bit grey level survived the 8-bit conversion
        self.assertEqual(pj.m["tools"]["redline"]["path"], self.redline)
        prog = [e for e in self.events() if e["ev"] == "progress" and e.get("step") == "transcode"]
        self.assertEqual((prog[-1]["done"], prog[-1]["total"]), (3, 3))
        self.assertEqual(pj.m["source"]["take"], "67")

    def test_r3d_needs_a_take_and_a_clip_for_it(self):
        root = self.rdm_tree()
        pj = Project(self.root, create=True)
        with self.assertRaises(events.StageError):
            ingest.run(self.args(r3d=root), pj)
        pj.release()
        with self.assertRaises(events.StageError):
            ingest.run(self.args(r3d=root, take="099"), pj)

    def test_select_is_refused_on_an_array_project(self):
        from hs.stages import select
        pj = Project(self.root, create=True)
        ingest.run(self.args(frames=self.frames_dir()), pj)
        with self.assertRaises(events.StageError):
            select.run(Namespace(), pj)

    def test_exclude_accepts_camera_ids(self):
        self.assertEqual(train.parse_exclude("GA, HB_L, L/cap004, cap005_R"), {"L/GA", "L/HB", "L/cap004", "R/cap005"})


class MonoRig(Base):
    """rig.npz with stereo=False: every consumer takes one view per capture."""

    def mono_rig(self, path, n=4):
        rng = np.random.default_rng(3)
        R = np.tile(np.eye(3), (n, 1, 1))
        C = np.stack([np.array([700.0 * i, 0.0, 0.0]) for i in range(n)])
        t = np.einsum("nij,nj->ni", R, -C)
        K = np.tile(np.array([[5000.0, 0, 1920.0], [0, 5000.0, 1080.0], [0, 0, 1]]), (n, 1, 1))
        pts = rng.normal(size=(200, 3)) * 100 + np.array([1000.0, 0, 3000.0])
        np.savez(path, names=np.array([f"{c}_L" for c in ("GA", "GB", "HA", "HB")[:n]]), K=K, R=R, t=t, C=C,
                 pts=pts, wh=np.tile([3840, 2160], (n, 1)), w=3840, h=2160, s_mm=1.0,
                 photos=np.array(["GA", "GB", "HA", "HB"][:n]), stereo=False)

    def test_helpers_and_coverage(self):
        from hs import coverage, rig
        p = os.path.join(self.tmp, "rig.npz")
        self.mono_rig(p)
        G, names, L = rig.load(p)
        self.assertFalse(rig.is_stereo(G))
        self.assertEqual(L.tolist(), [0, 1, 2, 3])
        self.assertIsNone(rig.right_index(G, 0))
        self.assertEqual(rig.capture_name("GA_L"), "GA")
        self.assertEqual(rig.capture_name("cap004_R"), "cap004")
        cov = coverage.compute(p)
        self.assertEqual(cov["n_captures"], 4)
        self.assertFalse(cov["stereo"])
        self.assertEqual([c["name"] for c in cov["captures"]], ["GA", "GB", "HA", "HB"])
        self.assertTrue(all(c["lr_separation_mm"] is None for c in cov["captures"]))
        # the stereo default is untouched: a file without the key is still pairs
        write_rig(os.path.join(self.tmp, "old.npz"), seed=1)
        G2 = np.load(os.path.join(self.tmp, "old.npz"), allow_pickle=True)
        self.assertTrue(rig.is_stereo(G2))
        self.assertEqual(rig.right_index(G2, 0), 1)

    def test_views_path_is_one_frame_per_camera_and_refuses_the_right_eye(self):
        from hs.stages import views
        pj = self.solved_project()
        self.mono_rig(pj.rig_npz)
        out = os.path.join(self.tmp, "path.json")
        v, subj, fx = views.build_path(pj, [0, 3], out, "L")
        self.assertEqual([x["view"] for x in v], ["GA_L", "HB_L"])
        self.assertEqual([x["image"] for x in v], ["GA.jpg", "HB.jpg"])
        self.assertEqual(len(json.load(open(out))["frames"]), 2)
        with self.assertRaises(events.StageError):
            views.build_path(pj, [0], out, "R")
