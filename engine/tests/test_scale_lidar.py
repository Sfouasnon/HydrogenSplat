"""hs scale --lidar: a phone LiDAR scan as the second measurement source of `hs scale`.

Synthetic projects from lidar_synth.py: the room scan (metres, +Y up) and a solve of it — 12,000
sparse points over 60 % of the room with 3 mm noise, ten cameras — in a random Sim(3) frame,
written as a mono rig.npz + COLMAP text model (board_synth.write_mono_dataset, what monocolmap.py
leaves) or as a stereo rig.npz. A mono project must come out in millimetres through the same
apply path the board uses (scene_scaled source "lidar"); a stereo one must report its scale
ratio and never be changed; a scan with nothing in common with the solve must fail with exit 1.
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
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from hs import events, lidar as L  # noqa: E402
from hs.project import Project, now_iso  # noqa: E402
from hs.stages import scale  # noqa: E402
import lidar_synth as S  # noqa: E402


def args(**kw):
    base = dict(board=None, legacy_board=False, eye="L", min_views=3, dry_run=False, lidar=None, units=None,
                scan_up="y", pairs=None, init=None, apply=False, trust_scan=False, factor=None, note=None, min_inlier=0.5,
                max_rms_mm=30.0, inlier_mm=50.0, min_scale_sensitivity=0.1, icp_iters=100, seed=0)
    base.update(kw)
    return Namespace(**base)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hs-scale-lidar-")
        self.root = os.path.join(self.tmp, "proj")
        self.out = io.StringIO()
        self._redir = contextlib.redirect_stdout(self.out)
        self._redir.__enter__()

    def tearDown(self):
        self._redir.__exit__(None, None, None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def events(self):
        return [json.loads(l) for l in self.out.getvalue().splitlines() if l.startswith("{")]

    def checks(self):
        return {e["name"]: e for e in self.events() if e["ev"] == "check"}

    def metrics(self):
        return {e["name"]: e["value"] for e in self.events() if e["ev"] == "metric"}

    def project(self, kind="mono", k=1.21, seed=21, baseline_mm=10.6, sparse=False, **scene_kw):
        self.scene = sc = S.Scene(k=k, seed=seed, **scene_kw)
        pj = Project(self.root, create=True)
        pj.m["source"] = {"kind": "mono"} if kind == "mono" else {}
        for s in ("ingest", "select"):
            pj.m["stages"][s] = {"status": "done"}
        pj.m["stages"]["solve"] = {"status": "done", "started": now_iso(), "finished": now_iso(),
                                   "metrics": {"scale_to_m": None},
                                   "checks": [{"name": "scene_scaled", "ok": False, "needs_human": True}]}
        if kind == "mono":
            self.names = sc.write_mono_rig(pj.dataset_dir)
        else:
            self.names = sc.write_stereo_rig(pj.dataset_dir, baseline_mm=baseline_mm, sparse=sparse)
            # the solve carries the profile's baseline: in a solve k times too big that is baseline * k
            pj.m["stages"]["solve"]["metrics"]["profile_baseline_mm"] = baseline_mm * k
        pj.save()
        self.scan = os.path.join(self.tmp, "scan.ply")
        sc.write_scan_ply(self.scan)
        return pj

    def report(self, pj):
        return json.load(open(pj.path("scale", "lidar_report.json")))


class MonoApplies(Base):
    def test_applies_the_scan_scale_through_the_board_path(self):
        import pycolmap
        pj = self.project()
        for s in ("train", "move", "prune", "render", "views", "exposure", "masks"):
            pj.m["stages"][s] = {"status": "done"}
        pj.save()
        C_before = np.load(pj.rig_npz)["C"].copy()
        scale.run(args(lidar=self.scan), pj)

        self.assertEqual(pj.status("scale"), "done")
        m = pj.stage("scale")["metrics"]
        k = self.scene.k
        self.assertLess(abs(m["scale_factor"] * k - 1), 0.001, m["scale_factor"] * k)
        self.assertLessEqual(m["lidar_rms_mm"], 5.0)
        self.assertGreater(m["lidar_inlier_fraction"], 0.99)
        # rig.npz is millimetres now: camera spacings are the true ones
        G = np.load(pj.rig_npz, allow_pickle=True)
        self.assertFalse(bool(G["stereo"]))
        self.assertEqual(float(G["s_mm"]), 1.0)
        D = np.linalg.norm(G["C"][:, None] - G["C"][None], axis=2)
        Dt = np.linalg.norm(self.scene.C_true[:, None] - self.scene.C_true[None], axis=2)
        np.testing.assert_allclose(D, Dt, rtol=0.001, atol=0.5)
        np.testing.assert_allclose(G["C"], C_before * m["scale_factor"], rtol=1e-6)
        rec = pycolmap.Reconstruction(os.path.join(pj.dataset_dir, "sparse"))
        Cm = np.array([rec.images[i].projection_center() for i in sorted(rec.images)]) * 1000.0
        np.testing.assert_allclose(Cm, G["C"], rtol=1e-5, atol=1e-3)
        # the same bookkeeping as a board: scene_scaled from "lidar", the unit chain, stale downstream
        ss = [c for c in pj.stage("solve")["checks"] if c["name"] == "scene_scaled"]
        self.assertEqual(len(ss), 1)
        self.assertTrue(ss[0]["ok"])
        self.assertEqual(ss[0]["source"], "lidar")
        self.assertEqual(pj.stage("solve")["metrics"]["scale_source"], "lidar")
        self.assertAlmostEqual(pj.stage("solve")["metrics"]["scale_to_m"], m["scale_factor"], places=8)
        self.assertEqual(pj.m["scale"]["source"], "lidar")
        for s in ("train", "move", "prune", "render", "views"):
            self.assertEqual(pj.status(s), "stale", s)
        for s in ("exposure", "masks"):
            self.assertEqual(pj.status(s), "done", s)
        self.assertFalse(os.path.exists(pj.lock_path))
        # checks: all four pass on this scene
        c = {x["name"]: x for x in pj.stage("scale")["checks"]}
        for name in ("lidar_aligned", "lidar_geometry_constrains", "lidar_covers_captures", "lidar_up_agrees",
                     "lidar_units_known", "scene_scaled"):
            self.assertTrue(c[name]["ok"], c[name])
        self.assertGreater(m["lidar_scale_sensitivity"], 0.15)
        self.assertNotIn("lidar_scale_agrees", c)
        # the report: everything the app needs to draw the scan in the solve
        rep = self.report(pj)
        self.assertTrue(rep["applied"] and rep["aligned"])
        self.assertEqual(rep["init"]["method"], "auto")
        self.assertEqual(rep["subsample"]["scan_register"], 5000)
        self.assertTrue(rep["icp"]["rms_mm_per_iteration"])
        self.assertEqual(len(rep["coverage"]["captures"]), len(self.names))
        self.assertLess(rep["up"]["angle_deg"], 10.0)
        gp = rep["ground_plane"]
        self.assertLess(L.angle_deg(gp["normal_solve"], rep["up"]["up_world"]), 0.5)
        self.assertAlmostEqual(gp["camera_height_mm"]["median"], 1000.0, delta=10.0)
        # scan file coordinates -> solve mm lands on the (now scaled) sparse points
        M = np.array(rep["transform"]["scan_file_to_solve"])
        X = self.scene.scan_m @ M[:3, :3].T + M[:3, 3]
        d, _ = L.NN(X).query(G["pts"])
        self.assertLess(np.median(d), 5.0)
        # ...and so does the viewer's point cloud
        v = L.load_scan(pj.path("scale", "lidar_aligned.ply"), units="mm")
        self.assertIsNotNone(v.colors)
        self.assertLessEqual(len(v.points), scale.LIDAR_VIEW_POINTS)
        d, _ = L.NN(v.points).query(G["pts"])
        self.assertLess(np.median(d), 25.0)
        # events: ICP progress with a total, both artifacts
        ev = self.events()
        prog = [e for e in ev if e["ev"] == "progress" and e.get("step") == "lidar_icp"]
        self.assertTrue(prog and all("total" in e for e in prog))
        arts = {e["path"] for e in ev if e["ev"] == "artifact"}
        self.assertTrue({"scale/lidar_report.json", "scale/lidar_aligned.ply"} <= arts)

        # a second run measures the now-metric solve: a factor of one
        pj.m["stages"]["train"]["status"] = "done"
        scale.run(args(lidar=self.scan), pj)
        self.assertLess(abs(pj.stage("scale")["metrics"]["scale_factor"] - 1.0), 0.001)

    def test_dry_run_measures_and_changes_nothing(self):
        pj = self.project(k=0.78, seed=22)
        pj.m["stages"]["train"] = {"status": "done"}
        pj.save()
        before = open(pj.rig_npz, "rb").read()
        scale.run(args(lidar=self.scan, dry_run=True), pj)
        self.assertEqual(open(pj.rig_npz, "rb").read(), before)
        self.assertEqual(pj.status("scale"), "pending")
        self.assertEqual(pj.status("train"), "done")
        rec = pj.stage("scale")["lidar_check"]
        self.assertFalse(rec["applied"])
        self.assertTrue(rec["aligned"])
        self.assertLess(abs(rec["metrics"]["scale_factor"] * self.scene.k - 1), 0.001)
        rep = self.report(pj)
        self.assertFalse(rep["applied"])
        self.assertIn("unscaled", rep["transform"]["frame"])
        # in the current frame the transform carries the scale: scan mm -> solve units
        M = np.array(rep["transform"]["scan_mm_to_solve"])
        self.assertAlmostEqual(np.linalg.norm(M[:3, 0]), self.scene.k, delta=0.001 * self.scene.k)
        self.assertFalse(os.path.exists(pj.lock_path))

    def test_pairs_and_a_millimetre_file(self):
        pj = self.project(k=1.33, seed=23)
        sc = self.scene
        self.scan = os.path.join(self.tmp, "scan_mm.ply")
        sc.write_scan_ply(self.scan, units="mm")
        rng = np.random.default_rng(1)
        picks = [np.argmin(np.linalg.norm(sc.scan_m - q, axis=1)) for q in ([2.6, 0.5, 1.8], [1.3, 0.6, 2.8], [0.0, 2.3, 0.4])]
        scan_mm = sc.scan_m[picks] * 1000 + rng.normal(0, 10, (3, 3))
        solve = sc.to_solve(sc.scan_m[picks] * 1000) + rng.normal(0, 10 * sc.k, (3, 3))
        text = ";".join(",".join(f"{v:.2f}" for v in a) + "=" + ",".join(f"{v:.3f}" for v in b)
                        for a, b in zip(scan_mm, solve))
        scale.run(args(lidar=self.scan, pairs=text, dry_run=True), pj)
        rec = pj.stage("scale")["lidar_check"]
        self.assertEqual((rec["metrics"]["scan_units"], rec["metrics"]["lidar_init"]), ("mm", "pairs"))
        self.assertLess(abs(rec["metrics"]["scale_factor"] * sc.k - 1), 0.001)
        rep = self.report(pj)
        self.assertEqual(rep["init"]["pairs"], 3)
        self.assertNotIn("lidar_register_s", rec["metrics"])

    def test_z_up_scan(self):
        pj = self.project(k=0.95, seed=24)
        sc = self.scene
        Rz = S.rot([1, 0, 0], 90.0)                       # ARKit's +Y becomes +Z
        np.testing.assert_allclose(Rz @ [0, 1.0, 0], [0, 0, 1.0], atol=1e-12)
        S.write_ply(self.scan, sc.scan_m @ Rz.T, sc.colors)
        scale.run(args(lidar=self.scan, scan_up="z", dry_run=True), pj)
        c = self.checks()
        self.assertTrue(c["lidar_up_agrees"]["ok"], c["lidar_up_agrees"])
        rep = self.report(pj)
        self.assertLess(L.angle_deg(rep["up"]["up_world"], sc.Rg @ [0, 1.0, 0]), 0.1)
        self.assertEqual((rep["scan_up"], rep["scan_up_source"]), ("z", "flag"))

    def test_up_axis_is_detected(self):
        # Scaniverse writes +Z up, Polycam +Y: with --scan-up auto (the default) the floor decides
        pj = self.project(k=0.95, seed=25)
        sc = self.scene
        for axis, Rz in (("z", S.rot([1, 0, 0], 90.0)), ("y", np.eye(3))):
            S.write_ply(self.scan, sc.scan_m @ Rz.T, sc.colors)
            scale.run(args(lidar=self.scan, scan_up="auto", dry_run=True), pj)
            rep = self.report(pj)
            self.assertEqual((rep["scan_up"], rep["scan_up_source"]), (axis, "detected"), axis)
            self.assertTrue(self.checks()["lidar_up_agrees"]["ok"])
            self.assertLess(L.angle_deg(rep["up"]["up_world"], sc.Rg @ [0, 1.0, 0]), 0.1)
        m = pj.stage("scale")["lidar_check"]["metrics"]
        self.assertEqual((m["scan_up"], m["scan_up_source"]), ("y", "detected"))


class Stereo(Base):
    def test_reports_the_ratio_and_changes_nothing(self):
        pj = self.project(kind="stereo", k=1.0, seed=31)
        for s in ("train", "move"):
            pj.m["stages"][s] = {"status": "done"}
        pj.save()
        before = open(pj.rig_npz, "rb").read()
        scale.run(args(lidar=self.scan), pj)
        self.assertEqual(open(pj.rig_npz, "rb").read(), before)
        self.assertEqual(pj.status("scale"), "pending")
        self.assertEqual(pj.status("train"), "done")
        m = pj.stage("scale")["lidar_check"]["metrics"]
        self.assertAlmostEqual(m["scale_ratio"], 1.0, delta=0.001)
        self.assertAlmostEqual(m["implied_baseline_mm"], 10.6, delta=0.02)
        self.assertNotIn("scale_factor", m)
        c = self.checks()
        self.assertTrue(c["lidar_scale_agrees"]["ok"])
        self.assertTrue(c["lidar_aligned"]["ok"])
        self.assertNotIn("scene_scaled", c)
        rep = self.report(pj)
        self.assertFalse(rep["applied"])
        self.assertEqual(rep["mode"], "sim3")                        # the scale is solved for on stereo too
        self.assertAlmostEqual(rep["solve_to_scan"]["s"], 1.0, delta=0.001)
        M = np.array(rep["transform"]["scan_mm_to_solve"])
        np.testing.assert_allclose(M[:3, :3] @ M[:3, :3].T, np.eye(3) / rep["solve_to_scan"]["s"] ** 2, atol=1e-6)
        # --apply alone is refused, as for a board; --trust-scan is the way in (tested below)
        with self.assertRaises(events.StageError) as e:
            scale.run(args(lidar=self.scan, apply=True), pj)
        self.assertIn("baseline", str(e.exception))
        self.assertIn("--trust-scan", e.exception.hint)

    def test_a_wrong_baseline_is_flagged_and_its_size_implied(self):
        # the solve is 3 % too big (the profile's baseline is 3 % long); capture 0 stood outside the scan
        pj = self.project(kind="stereo", k=1.03, seed=32)
        G = dict(np.load(pj.rig_npz, allow_pickle=True))
        for v in (0, 1):
            C = G["C"][v] + self.scene.to_solve(np.array([[0, 0, 0], [0, 0, 9000.0]]))[1] - self.scene.to_solve(np.zeros((1, 3)))[0]
            G["C"][v] = C
            G["t"][v] = -G["R"][v] @ C
        np.savez(pj.rig_npz, **G)
        scale.run(args(lidar=self.scan), pj)
        m = pj.stage("scale")["lidar_check"]["metrics"]
        self.assertAlmostEqual(m["scale_ratio"], 1 / 1.03, delta=0.001)
        self.assertAlmostEqual(m["implied_baseline_mm"], 10.6, delta=0.02)      # the true baseline
        c = self.checks()
        self.assertFalse(c["lidar_scale_agrees"]["ok"])
        self.assertTrue(c["lidar_scale_agrees"]["needs_human"])
        self.assertTrue(c["lidar_aligned"]["ok"])
        self.assertFalse(c["lidar_covers_captures"]["ok"])
        self.assertIn("cap000", c["lidar_covers_captures"]["value"])
        self.assertNotIn("cap001", c["lidar_covers_captures"]["value"])

    def test_a_baseline_far_off_is_still_found_and_trust_scan_applies_it(self):
        # CirclesSculpture: the realigned H1 solve came out 5.8x small. The old SE(3) search could
        # only fail there; the Sim(3) one must find the ratio, and --trust-scan must apply it
        # through the rig route's own writer (rig.npz stays interleaved L,R, the eyes scaled too).
        pj = self.project(kind="stereo", k=1 / 5.8, seed=33, sparse=True)
        for st in ("train", "views"):
            pj.m["stages"][st] = {"status": "done"}
        pj.save()
        G0 = dict(np.load(pj.rig_npz, allow_pickle=True))
        b0 = float(np.linalg.norm(G0["C"][0] - G0["C"][1]))
        scale.run(args(lidar=self.scan), pj)                       # measure: ratio reported, nothing applied
        m = pj.stage("scale")["lidar_check"]["metrics"]
        self.assertAlmostEqual(m["scale_ratio"], 5.8, delta=0.03)
        self.assertAlmostEqual(m["implied_baseline_mm"], 10.6, delta=0.06)      # the true baseline
        c = self.checks()
        self.assertTrue(c["lidar_aligned"]["ok"], c["lidar_aligned"])
        self.assertFalse(c["lidar_scale_agrees"]["ok"])
        self.assertEqual(pj.status("train"), "done")
        scale.run(args(lidar=self.scan, trust_scan=True), pj)      # apply
        self.assertEqual(pj.status("scale"), "done")
        self.assertEqual(pj.status("train"), "stale")
        G1 = np.load(pj.rig_npz, allow_pickle=True)
        self.assertEqual(list(G1["names"]), list(G0["names"]))
        self.assertTrue(bool(G1["stereo"]) if "stereo" in G1.files else True)
        b1 = float(np.linalg.norm(G1["C"][0] - G1["C"][1]))
        self.assertAlmostEqual(b1 / b0, m["scale_ratio"], delta=0.02)
        s1 = pj.stage("solve")
        self.assertEqual(s1["metrics"]["scale_source"], "lidar")
        self.assertTrue([x for x in s1["checks"] if x["name"] == "scene_scaled"][0]["ok"])
        # aligned once more, the scan and the solve now agree
        scale.run(args(lidar=self.scan, dry_run=True), pj)
        self.assertAlmostEqual(pj.stage("scale")["lidar_check"]["metrics"]["scale_ratio"], 1.0, delta=0.01)


class Factor(Base):
    def test_a_known_factor_is_applied_and_stereo_needs_trust(self):
        pj = self.project(kind="stereo", k=1 / 5.8, seed=34, sparse=True)
        G0 = dict(np.load(pj.rig_npz, allow_pickle=True))
        with self.assertRaises(events.StageError) as e:
            scale.run(args(factor=5.8), pj)
        self.assertIn("--trust-scan", e.exception.hint)
        with self.assertRaises(events.StageError):
            scale.run(args(factor=5.8, trust_scan=True, dry_run=True), pj)
        with self.assertRaises(events.StageError):
            scale.run(args(factor=-1.0, trust_scan=True), pj)
        scale.run(args(factor=5.8, trust_scan=True, note="ring silhouette fit"), pj)
        self.assertEqual(pj.status("scale"), "done")
        G1 = np.load(pj.rig_npz, allow_pickle=True)
        np.testing.assert_allclose(G1["C"], G0["C"] * 5.8, rtol=1e-6)
        srt = lambda X: X[np.lexsort(X.T[::-1])]                  # points3D come back in the map's order
        np.testing.assert_allclose(srt(G1["pts"]), srt(G0["pts"] * 5.8), rtol=1e-5)
        self.assertEqual(pj.m["scale"]["source"], "manual")
        self.assertEqual(pj.stage("solve")["metrics"]["scale_source"], "manual")
        self.assertAlmostEqual(pj.stage("scale")["metrics"]["implied_baseline_mm"], 10.6, delta=0.05)
        # the scan now agrees with the solve
        scale.run(args(lidar=self.scan, dry_run=True), pj)
        self.assertAlmostEqual(pj.stage("scale")["lidar_check"]["metrics"]["scale_ratio"], 1.0, delta=0.01)
        # mono: no trust needed
        pj = self.project(kind="mono", k=1.21, seed=35)
        scale.run(args(factor=1 / 1.21), pj)
        self.assertEqual(pj.status("scale"), "done")


class Refusals(Base):
    def test_no_overlap_fails_with_exit_1_and_a_hint(self):
        from hs import cli
        pj = self.project(k=1.1, seed=41)
        G = dict(np.load(pj.rig_npz, allow_pickle=True))
        G["pts"] = S.blob_cloud()                        # the solve is of something else entirely
        np.savez(pj.rig_npz, **G)
        before = open(pj.rig_npz, "rb").read()
        with mock.patch.dict(os.environ, {"HS_NO_CAFFEINATE": "1"}):
            code = cli.main(["scale", "-p", self.root, "--lidar", self.scan])
        self.assertEqual(code, 1)
        ev = self.events()
        err = [e for e in ev if e["ev"] == "error"]
        self.assertEqual(len(err), 1)
        self.assertIn("did not align", err[0]["message"])
        self.assertIn("--pairs", err[0]["hint"])
        self.assertIn("--units", err[0]["hint"])
        self.assertEqual(ev[-1], {"ev": "done", "stage": "scale", "exit": 1})
        self.assertFalse(self.checks()["lidar_aligned"]["ok"])
        pj = Project(self.root)
        self.assertEqual(pj.status("scale"), "pending")
        self.assertFalse(pj.stage("scale")["lidar_check"]["aligned"])
        self.assertEqual(open(pj.rig_npz, "rb").read(), before)
        rep = self.report(pj)
        self.assertFalse(rep["aligned"])
        self.assertFalse(rep["applied"])
        self.assertTrue(rep["reasons"])
        self.assertFalse(os.path.exists(pj.lock_path))

    def test_a_bare_corner_fixes_no_scale_and_is_not_applied(self):
        # the cameras saw only the 30 % of the room nearest its corner: floor and two walls, which
        # fit the scan at any scale about the corner (and turned 120 degrees about it)
        pj = self.project(k=0.9, seed=43, covered=0.3)
        before = open(pj.rig_npz, "rb").read()
        with self.assertRaises(events.StageError) as e:
            scale.run(args(lidar=self.scan), pj)
        self.assertRegex(str(e.exception), "scale|ambiguous")
        self.assertEqual(open(pj.rig_npz, "rb").read(), before)
        self.assertEqual(pj.status("scale"), "pending")
        rec = pj.stage("scale")["lidar_check"]
        self.assertFalse(rec["applied"])
        self.assertFalse(os.path.exists(pj.lock_path))

    def test_argument_errors(self):
        pj = self.project(k=1.1, seed=42)
        with self.assertRaises(events.StageError) as e:
            scale.run(args(lidar=self.scan, board="7,5,40,30"), pj)
        self.assertIn("not both", str(e.exception))
        for bad in ("1,2,3=4,5,6", "1,2,3=4,5,6;1,2=3,4,5;0,0,1=1,1,1", "1,2,3;4,5,6;7,8,9",
                    "0,0,0=0,0,0;1,1,1=1,1,1;2,2,2=2,2,2"):
            with self.assertRaises(events.StageError, msg=bad):
                scale.run(args(lidar=self.scan, pairs=bad), pj)
        with self.assertRaises(events.StageError) as e:
            scale.run(args(lidar=self.scan, init="pairs"), pj)
        self.assertIn("--pairs", str(e.exception))
        with self.assertRaises(events.StageError):
            scale.run(args(lidar=self.scan, apply=True, dry_run=True), pj)
        with self.assertRaises(events.StageError) as e:
            scale.run(args(lidar=os.path.join(self.tmp, "nope.e57")), pj)
        self.assertIn("cannot read the scan", str(e.exception))
        self.assertFalse(os.path.exists(pj.lock_path) and Project(self.root).lock_holder()["pid"] != os.getpid())
        self.assertEqual(pj.status("scale"), "pending")

    def test_parse_pairs(self):
        S_, P = scale.parse_pairs("1,2,3=10,20,30; 4 5 6 = 40,50,60\n0,1,0=0,10,0")
        np.testing.assert_allclose(S_, [[1, 2, 3], [4, 5, 6], [0, 1, 0]])
        np.testing.assert_allclose(P, [[10, 20, 30], [40, 50, 60], [0, 10, 0]])

    def test_cli_knows_the_flags(self):
        from hs import cli
        a = cli.build_parser().parse_args(["scale", "-p", "/x", "--lidar", "s.ply", "--units", "cm", "--scan-up", "z",
                                           "--pairs", "1,2,3=4,5,6", "--init", "pairs", "--apply",
                                           "--min-inlier", "0.4", "--max-rms-mm", "20", "--inlier-mm", "40"])
        self.assertFalse(a.trust_scan)
        self.assertIsNone(a.factor)
        self.assertEqual((a.lidar, a.units, a.scan_up, a.pairs, a.init, a.apply, a.min_inlier, a.max_rms_mm, a.inlier_mm),
                         ("s.ply", "cm", "z", "1,2,3=4,5,6", "pairs", True, 0.4, 20.0, 40.0))
        self.assertIsNone(a.board)
        a = cli.build_parser().parse_args(["scale", "-p", "/x", "--board", "7,5,40,30"])
        self.assertIsNone(a.lidar)


if __name__ == "__main__":
    unittest.main()
