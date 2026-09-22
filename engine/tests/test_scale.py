"""hs scale: the board's scale applied to a mono/array solve through monocolmap's own writer.

A synthetic project: a mono rig.npz and a COLMAP text model in "solve units" (true scale k),
training images of a ChArUco board rendered by exact homography. After `hs scale` the model,
rig.npz and coverage.json must be in millimetres, solve's scene_scaled check must say so, and
exactly the stages measured in mm must be stale.
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from argparse import Namespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from hs import board as B, events  # noqa: E402
from hs.project import DOWNSTREAM, REQUIRES, STAGES, Project, now_iso  # noqa: E402
from hs.stages import scale  # noqa: E402
import board_synth as S  # noqa: E402

BOARD = "7,5,30,22"
NAMES = ["sel000-00010", "sel001-00020", "sel002-00030", "sel003-00040", "sel004-00050"]


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hs-scale-")
        self.root = os.path.join(self.tmp, "proj")
        self.out = io.StringIO()
        self._redir = contextlib.redirect_stdout(self.out)
        self._redir.__enter__()

    def tearDown(self):
        self._redir.__exit__(None, None, None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def events(self):
        return [json.loads(l) for l in self.out.getvalue().splitlines() if l.startswith("{")]

    def project(self, k=0.0425, kind="mono"):
        self.scene = sc = S.Scene(B.parse_spec(BOARD), n_views=len(NAMES), k=k)
        pj = Project(self.root, create=True)
        pj.m["source"] = {"kind": kind}
        for s in ("ingest", "select"):
            pj.m["stages"][s] = {"status": "done"}
        pj.m["stages"]["solve"] = {"status": "done", "started": now_iso(), "finished": now_iso(),
                                   "metrics": {"scale_to_m": None},
                                   "checks": [{"name": "scene_scaled", "ok": False, "needs_human": True}]}
        pts = sc.corners_world_mm() / k
        S.write_mono_dataset(pj.dataset_dir, NAMES, sc.K, sc.R, sc.t_units, sc.size, pts, sc.imgs)
        pj.save()
        return pj

    def args(self, **kw):
        base = dict(board=BOARD, legacy_board=False, eye="L", min_views=3, dry_run=False)
        base.update(kw)
        return Namespace(**base)


class ScaleStage(Base):
    def test_applies_the_board_scale_everywhere_it_lives(self):
        import pycolmap
        pj = self.project()
        for s in ("train", "move", "prune", "render", "views", "exposure", "masks"):
            pj.m["stages"][s] = {"status": "done"}
        pj.save()
        C_before = np.load(pj.rig_npz)["C"].copy()
        scale.run(self.args(), pj)

        self.assertEqual(pj.status("scale"), "done")
        m = pj.stage("scale")["metrics"]
        k = self.scene.k
        err = abs(m["scale_factor"] / k - 1)
        self.assertLess(err, 0.005, m)
        # rig.npz is now millimetres: the camera centres are the true ones
        G = np.load(pj.rig_npz, allow_pickle=True)
        self.assertFalse(bool(G["stereo"]))
        self.assertEqual([str(n) for n in G["names"]], [n + "_L" for n in NAMES])
        np.testing.assert_allclose(G["C"], self.scene.C_mm, rtol=0.005, atol=0.5)
        np.testing.assert_allclose(G["C"], C_before * m["scale_factor"], rtol=1e-6)
        self.assertEqual(float(G["s_mm"]), 1.0)
        # ...and the sparse points (the board's corners here) are where the board really is, in mm
        truth = self.scene.corners_world_mm()
        nn = np.linalg.norm(G["pts"][:, None, :] - truth[None], axis=2).min(axis=1)
        self.assertEqual(len(G["pts"]), len(truth))
        self.assertLess(nn.max(), 0.5, nn)
        # the COLMAP model the trainer reads moved with it (metres there)
        rec = pycolmap.Reconstruction(os.path.join(pj.dataset_dir, "sparse"))
        Cm = np.array([rec.images[i].projection_center() for i in sorted(rec.images)]) * 1000.0
        np.testing.assert_allclose(Cm, G["C"], rtol=1e-5, atol=1e-3)
        self.assertTrue(os.path.exists(os.path.join(pj.dataset_dir, "sparse", "points3D.ply")))
        self.assertTrue(os.path.exists(os.path.join(pj.dataset_dir, "sparse", "images.bin")))
        # coverage.json in mm
        cov = json.load(open(pj.path("solve", "coverage.json")))
        d = cov["distance_range_mm"]
        self.assertTrue(400 < d[0] <= d[1] < 700, d)      # 520 mm from the board, as rendered
        # solve's own scene_scaled record — what hs masks and the app read — now passes, from the board
        ss = [c for c in pj.stage("solve")["checks"] if c["name"] == "scene_scaled"]
        self.assertEqual(len(ss), 1)
        self.assertTrue(ss[0]["ok"])
        self.assertEqual(ss[0]["source"], "board")
        self.assertAlmostEqual(pj.stage("solve")["metrics"]["scale_to_m"], m["scale_factor"], places=8)
        self.assertEqual(pj.m["scale"]["source"], "board")
        # stale: everything measured in mm; exposure and masks (pixels) are still current
        for s in ("train", "move", "prune", "render", "views"):
            self.assertEqual(pj.status(s), "stale", s)
        for s in ("exposure", "masks"):
            self.assertEqual(pj.status(s), "done", s)
        # the board plane is reported against the current up
        self.assertIn("board_normal_to_up_deg", m)
        n = np.array(m["board_normal_world"])
        self.assertLess(B.angle_deg(n, self.scene.up), 1.0)
        rep = json.load(open(pj.path("scale", "scale_report.json")))
        self.assertEqual(rep["board"], "7,5,30,22,DICT_5X5_100")
        self.assertFalse(os.path.exists(pj.lock_path))
        # a second run measures the now-metric solve: a factor of one
        pj.m["stages"]["train"]["status"] = "done"
        scale.run(self.args(), pj)
        self.assertLess(abs(pj.stage("scale")["metrics"]["scale_factor"] - 1.0), 0.002)
        self.assertAlmostEqual(pj.m["scale"]["scale_to_m"], m["scale_factor"], delta=0.002 * m["scale_factor"])

    def test_dry_run_writes_nothing(self):
        pj = self.project()
        pj.m["stages"]["train"] = {"status": "done"}
        pj.save()
        before = open(pj.rig_npz, "rb").read()
        scale.run(self.args(dry_run=True), pj)
        self.assertEqual(open(pj.rig_npz, "rb").read(), before)
        self.assertEqual(pj.status("scale"), "pending")
        self.assertEqual(pj.status("train"), "done")
        dr = pj.stage("scale")["dry_run"]
        self.assertLess(abs(dr["metrics"]["scale_factor"] / self.scene.k - 1), 0.005)
        self.assertFalse(os.path.exists(pj.path("scale", "scale_report.json")))

    def test_stereo_is_metric_already(self):
        pj = self.project(k=1.0, kind=None)
        pj.m["source"] = {}
        G = dict(np.load(pj.rig_npz, allow_pickle=True))
        G.pop("stereo")                                  # a rigcolmap.py rig.npz has no key: stereo
        np.savez(pj.rig_npz, **G)
        pj.m["stages"]["solve"]["metrics"]["profile_baseline_mm"] = 10.595
        pj.save()
        with self.assertRaises(events.StageError) as e:
            scale.run(self.args(), pj)
        self.assertIn("baseline", str(e.exception))
        self.assertIn("--dry-run", e.exception.hint)
        # measuring is allowed and says what the baseline is worth
        scale.run(self.args(dry_run=True), pj)
        m = pj.stage("scale")["dry_run"]["metrics"]
        self.assertLess(abs(m["scale_factor"] - 1.0), 0.005)
        self.assertAlmostEqual(m["implied_baseline_mm"], 10.595 * m["scale_factor"], places=3)

    def test_wrong_board_is_a_clear_error(self):
        pj = self.project()
        with self.assertRaises(events.StageError) as e:
            scale.run(self.args(board="7,5,30,22,DICT_4X4_50"), pj)
        self.assertIn("--board", e.exception.hint)
        self.assertEqual(pj.status("scale"), "pending")
        with self.assertRaises(events.StageError):
            scale.run(self.args(board="7,5,30"), pj)

    def test_a_wild_spread_is_refused_and_changes_nothing(self):
        from unittest import mock
        pj = self.project()
        before = open(pj.rig_npz, "rb").read()
        wild = {"scale": 0.04, "mad": 0.008, "mad_rel": 0.2, "n_pairs": 276, "n_corners": 24}
        with mock.patch.object(B, "scale_from_pairs", return_value=wild):
            with self.assertRaises(events.StageError) as e:
                scale.run(self.args(), pj)
        self.assertIn("not applied", str(e.exception))
        self.assertEqual(open(pj.rig_npz, "rb").read(), before)
        self.assertEqual(pj.status("scale"), "pending")

    def test_needs_a_solve(self):
        pj = Project(self.root, create=True)
        with self.assertRaises(events.StageError):
            scale.run(self.args(), pj)


class StateMachine(unittest.TestCase):
    def test_scale_sits_between_solve_and_train_and_is_optional(self):
        self.assertEqual(STAGES.index("scale"), STAGES.index("solve") + 1)
        self.assertEqual(STAGES.index("train"), STAGES.index("scale") + 1)
        self.assertEqual(REQUIRES["scale"], ["solve"])
        self.assertNotIn("scale", REQUIRES["train"])
        self.assertEqual(set(DOWNSTREAM["scale"]), {"train", "move", "prune", "render", "views"})
        for up in ("ingest", "select", "solve"):
            self.assertIn("scale", DOWNSTREAM[up])

    def test_an_old_manifest_without_scale_loads_and_trains(self):
        tmp = tempfile.mkdtemp(prefix="hs-scale-old-")
        try:
            stages = {s: {"status": "done"} for s in ("ingest", "select", "solve", "train", "move")}
            json.dump({"version": 1, "name": "old", "source": {"kind": "array"}, "stages": stages},
                      open(os.path.join(tmp, "manifest.json"), "w"))
            pj = Project(tmp)
            self.assertEqual(pj.status("scale"), "pending")
            pj.require("train")                          # no raise: scale is not a prerequisite
            pj.begin("solve", argv=["hs", "solve"])      # a re-solve marks a pending scale nothing...
            self.assertEqual(pj.status("scale"), "pending")
            pj.finish("solve")
            pj.m["stages"]["scale"]["status"] = "done"   # ...and a done one stale
            pj.begin("solve", argv=["hs", "solve"])
            self.assertEqual(pj.status("scale"), "stale")
            pj.finish("solve")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_cli_knows_the_stage(self):
        from hs import cli
        a = cli.build_parser().parse_args(["scale", "-p", "/x", "--board", "7,5,40,30", "--dry-run"])
        self.assertEqual((a.cmd, a.project, a.board, a.dry_run, a.min_views, a.eye), ("scale", "/x", "7,5,40,30", True, 3, "L"))
        self.assertIs(cli.PROJECT_STAGES["scale"], scale)
        a = cli.build_parser().parse_args(["solve", "-p", "/x", "--board", "7,5,40,30"])
        self.assertEqual(a.board, "7,5,40,30")


if __name__ == "__main__":
    unittest.main()
