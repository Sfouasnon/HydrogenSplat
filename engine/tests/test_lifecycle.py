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

from hs import events  # noqa: E402
from hs.project import LOCK_FILE, Project, md5_file, now_iso  # noqa: E402
from hs.stages import archive, exposure, render, train  # noqa: E402

FAKEBIN = os.path.join(HERE, "fakebin")


def write_ply(path, n=10, seed=0):
    rng = np.random.default_rng(seed)
    props = ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2"]
    hdr = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n
           + "".join(f"property float {p}\n" for p in props) + "end_header\n")
    with open(path, "wb") as f:
        f.write(hdr.encode() + rng.normal(size=(n, len(props))).astype("<f4").tobytes())


def write_jpg(path, level):
    import cv2
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = np.full((24, 32, 3), level, np.uint8)
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
                    start_iter=None, no_caffeinate=True, brush_args="")
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
