"""hs scale --lidar --init silhouette, --init-points, and hs train --init lidar.

The synthetic yard (lidar_synth.YardScene): an 8 x 8 m lawn with a box and a ball pressed together
on it (the subject) and a thin post; the solve holds only lawn points, 5 mm noise, in a random
Sim(3) frame — the Circles situation, where the subject is missing from the sparse cloud and the
geometric registration can only fail. 24 cameras orbit it at 1.5 m; their masks are the
subject's points projected and closed. The fit must find the scale from the silhouettes alone.
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

from hs import events, initsplats, lidar as L, silhouette  # noqa: E402
from hs.project import Project, md5_file, now_iso  # noqa: E402
from hs.splatweights import Splats  # noqa: E402
from hs.stages import scale, train  # noqa: E402
import lidar_synth as S  # noqa: E402

FAKEBIN = os.path.join(HERE, "fakebin")


def args(**kw):
    base = dict(board=None, legacy_board=False, eye="L", min_views=3, dry_run=False, lidar=None, units=None,
                scan_up="y", pairs=None, init="silhouette", apply=False, trust_scan=False, factor=None, note=None,
                min_inlier=0.5, max_rms_mm=30.0, inlier_mm=50.0, min_scale_sensitivity=0.1, icp_iters=100, seed=0,
                subject_above_mm=150.0, silhouette_views=24, silhouette_tilt=False, silhouette_icp="vertical",
                min_iou=0.35, init_points=False, init_points_max=400000, init_opacity=0.3)
    base.update(kw)
    return Namespace(**base)


def pose_error(scene, T):
    """(scale ratio error, rotation deg, camera position error mm) of a solve -> scan fit against truth."""
    s, R, t = T
    ds = s * scene.k - 1.0
    ang = L.rotation_angle_deg(np.asarray(R), scene.Rg.T)
    C, _R, _t = scene.cameras_solve()
    err = np.linalg.norm(L.apply(T, C) - scene.C_true, axis=1)
    return ds, ang, float(err.max())


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hs-silhouette-")
        self.root = os.path.join(self.tmp, "proj")
        self.out = io.StringIO()
        self._redir = contextlib.redirect_stdout(self.out)
        self._redir.__enter__()

    def tearDown(self):
        self._redir.__exit__(None, None, None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def events(self):
        return [json.loads(l) for l in self.out.getvalue().splitlines() if l.startswith("{")]

    def metrics(self):
        return {e["name"]: e["value"] for e in self.events() if e["ev"] == "metric"}

    def checks(self):
        return {e["name"]: e for e in self.events() if e["ev"] == "check"}

    def project(self, kind="stereo", k=0.2, seed=3, masks=True, **kw):
        self.scene = sc = S.YardScene(k=k, seed=seed, **kw)
        pj = Project(self.root, create=True)
        pj.m["source"] = {"kind": "mono"} if kind == "mono" else {}
        for s in ("ingest", "select"):
            pj.m["stages"][s] = {"status": "done"}
        pj.m["stages"]["solve"] = {"status": "done", "started": now_iso(), "finished": now_iso(),
                                   "metrics": {"scale_to_m": None},
                                   "checks": [{"name": "scene_scaled", "ok": False, "needs_human": True}]}
        if kind == "mono":
            self.names = sc.write_mono_rig(pj.dataset_dir)
            stems = self.names
        else:
            self.names = sc.write_stereo_rig(pj.dataset_dir, sparse=True)
            pj.m["stages"]["solve"]["metrics"]["profile_baseline_mm"] = 10.6 * k
            stems = [n[:-2] for n in self.names[::2]]
        if masks:
            sc.write_masks(pj.dataset_dir, names=stems)
        pj.save()
        self.scan = os.path.join(self.tmp, "scan.ply")
        sc.write_scan_ply(self.scan)
        return pj


class Subject(unittest.TestCase):
    def test_scan_subject_is_the_box_ball_and_column(self):
        P, _c, lab = S.yard(seed=3)
        sub = silhouette.scan_subject(P * 1000.0, [0, 1.0, 0])
        want = int(np.isin(lab, (1, 2, 4)).sum())
        # the box, ball and column above the 150 mm cut, nothing of the lawn or the post
        idx = sub["index"]
        self.assertTrue(np.isin(lab[idx], [1, 2, 4]).all())
        self.assertGreater(len(idx), 0.6 * want)
        self.assertEqual(sub["cut_mm"], 150.0)            # upright things stop at the first cut
        self.assertLessEqual(len(sub["fit"]), 6000)


class SilhouetteStereo(Base):
    def test_recovers_the_scale_the_sparse_cloud_cannot(self):
        pj = self.project()
        scale.run(args(lidar=self.scan, init_points=True), pj)
        m = self.metrics()
        c = self.checks()
        rep = json.load(open(pj.path("scale", "lidar_report.json")))
        T = (rep["solve_to_scan"]["s"], np.array(rep["solve_to_scan"]["R"]), np.array(rep["solve_to_scan"]["t"]))
        ds, ang, cam = pose_error(self.scene, T)
        self.assertLess(abs(ds), 0.03, (T[0], 1 / self.scene.k))
        self.assertLess(ang, 3.0)
        self.assertLess(cam, 50.0)                        # every camera within 50 mm of where it stood
        self.assertEqual(m["lidar_init"], "silhouette")
        self.assertAlmostEqual(m["silhouette_scale"], m["scale_ratio"], places=5)   # the scale is the silhouettes'
        self.assertTrue(c["lidar_silhouette_fits"]["ok"], c["lidar_silhouette_fits"])
        self.assertTrue(c["lidar_aligned"]["ok"], c["lidar_aligned"])
        self.assertNotIn("lidar_geometry_constrains", c)
        self.assertTrue(c["lidar_ground_agrees_with_silhouettes"]["ok"], c["lidar_ground_agrees_with_silhouettes"])
        self.assertGreaterEqual(m["silhouette_views_used"], 20)
        self.assertEqual(m["silhouette_views_offered"], 24)
        self.assertGreater(m["silhouette_iou_mean"], 0.6)
        # the stereo solve is measured, not changed; the record says so
        self.assertEqual(pj.status("scale"), "pending")
        lc = pj.stage("scale")["lidar_check"]
        self.assertTrue(lc["aligned"])
        self.assertEqual(rep["init"]["method"], "silhouette")
        self.assertEqual(len(rep["init"]["iou"]), 24)
        # the human check: three overlays
        ov = sorted(f for f in os.listdir(pj.path("scale")) if f.startswith("silhouette_overlay_"))
        self.assertEqual(len(ov), 3, ov)
        self.assertEqual(lc["overlays"], ["scale/" + f for f in ov])
        # camera height above the lawn: 1.5 m
        self.assertLess(abs(m["silhouette_camera_height_mm"][1] - 1500.0), 60.0)
        self.assertLess(abs(m["silhouette_walk_radius_mm"][1] - 3000.0), 150.0)
        # --init-points: written in the dataset's current units (solve units / 1000), recorded
        self.assertEqual(lc["init_ply"], "scale/lidar_init.ply")
        self.assertEqual(lc["init_ply_md5"], md5_file(pj.path("scale", "lidar_init.ply")))
        self.assertEqual(lc["init_rig_md5"], md5_file(pj.rig_npz))
        sp = Splats(pj.path("scale", "lidar_init.ply"))
        self.assertEqual(sp.n, m["init_points_scan"] + m["init_points_sparse"])
        self.assertEqual(m["init_points_scan"], len(self.scene.scan_m))          # under --init-points-max: all
        # the scan's lawn lies on the solve's lawn: every sparse point has an init splat within 3 x noise
        d, _ = L.NN(sp.xyz_mm[:m["init_points_scan"]]).query(np.asarray(self.scene.pts))
        self.assertLess(float(np.median(d)), 20.0 * self.scene.k + 20.0)

    def test_refuses_without_masks(self):
        pj = self.project(masks=False)
        with self.assertRaises(events.StageError) as cm:
            scale.run(args(lidar=self.scan), pj)
        self.assertIn("hs masks", cm.exception.hint)
        self.assertFalse(os.path.exists(pj.lock_path))

    def test_refuses_pairs_with_silhouette(self):
        pj = self.project(masks=False)
        with self.assertRaises(events.StageError):
            scale.run(args(lidar=self.scan, pairs="0,0,0=1,1,1;1,0,0=2,1,1;0,1,0=1,2,1"), pj)


class SilhouetteMonoApplied(Base):
    def test_applies_and_writes_the_init_in_metres(self):
        pj = self.project(kind="mono", k=0.37, seed=5)
        scale.run(args(lidar=self.scan, init_points=True, init_points_max=20000), pj)
        self.assertEqual(pj.status("scale"), "done")
        m = pj.stage("scale")["metrics"]
        self.assertLess(abs(m["scale_factor"] * self.scene.k - 1), 0.03, m["scale_factor"] * self.scene.k)
        self.assertEqual(pj.m["scale"]["init_ply"], "scale/lidar_init.ply")
        self.assertEqual(pj.m["scale"]["init_rig_md5"], md5_file(pj.rig_npz))
        # applied: rig.npz is mm, the dataset metres; the init's scan splats sit on the true lawn / 1000
        G = np.load(pj.rig_npz, allow_pickle=True)
        sp = Splats(pj.path("scale", "lidar_init.ply"))
        self.assertLessEqual(abs(m["init_points_scan"] - 20000), 2000)
        d, _ = L.NN(sp.xyz_mm[:m["init_points_scan"]]).query(G["pts"])
        self.assertLess(float(np.median(d)), 30.0)
        # scales are the knn distances in metres, clamped
        sc = sp.scale_mm[:, 0] / 1000.0
        self.assertTrue((sc >= 0.002 - 1e-6).all() and (sc <= 0.1 + 1e-6).all())


class InitPly(unittest.TestCase):
    def test_brush_layout_round_trips(self):
        tmp = tempfile.mkdtemp(prefix="hs-initply-")
        try:
            rng = np.random.default_rng(0)
            xyz = rng.normal(size=(500, 3))
            rgb = rng.integers(0, 256, (500, 3)).astype(np.uint8)
            p = initsplats.write(os.path.join(tmp, "i.ply"), initsplats.splats(xyz, rgb, opacity=0.3), comments=["t"])
            # a plain parse: header lines, property order exactly Brush's export
            raw = open(p, "rb").read()
            k = raw.index(b"end_header\n") + len(b"end_header\n")
            head = raw[:k].decode().splitlines()
            self.assertEqual(head[:5], ["ply", "format binary_little_endian 1.0", "comment Exported from Brush",
                                        "comment SH degree: 3", "comment t"])
            props = [ln.split()[2] for ln in head if ln.startswith("property")]
            want = (["x", "y", "z", "scale_0", "scale_1", "scale_2", "opacity", "rot_0", "rot_1", "rot_2", "rot_3",
                     "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(45)])
            self.assertEqual(props, want)
            self.assertTrue(all(ln.startswith("property float ") for ln in head if ln.startswith("property")))
            self.assertIn("element vertex 500", head)
            body = np.frombuffer(raw[k:], "<f4").reshape(500, 59)
            np.testing.assert_allclose(body[:, :3], xyz, rtol=1e-6, atol=1e-6)
            self.assertTrue((body[:, 14:] == 0).all())                         # f_rest
            np.testing.assert_allclose(body[:, 6], np.log(0.3 / 0.7), rtol=1e-6)
            np.testing.assert_allclose(body[:, 7:11], np.tile([1, 0, 0, 0], (500, 1)))
            # the engine's own splat reader: positions (mm), opacity, colour back
            sp = Splats(p)
            np.testing.assert_allclose(sp.xyz_mm, xyz * 1000.0, rtol=1e-5, atol=1e-3)
            np.testing.assert_allclose(sp.opacity, 0.3, rtol=1e-5)
            np.testing.assert_allclose(sp.rgb * 255.0, rgb, atol=0.01)
            # the scan loader reads it as a splat PLY too
            scan = L.load_scan(p, units="m")
            self.assertTrue(scan.meta["gaussian_splat"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TrainInitLidar(Base):
    TOTAL = 40

    def train_args(self, **kw):
        base = dict(brush=os.path.join(FAKEBIN, "brush"), total_train_iters=self.TOTAL, growth_stop_iter=30,
                    refine_every=10, split_at_screen_size=None, export_every=10, resume_from=None,
                    start_iter=None, no_caffeinate=True, brush_args="", exclude="", no_masks=False,
                    min_scale_factor=None, layer=None, alpha_mode=None, init="lidar")
        base.update(kw)
        return Namespace(**base)

    def setUp(self):
        super().setUp()
        os.environ["HS_PYTHON"] = sys.executable
        os.environ["HS_FAKE_REFINE_STOP"] = "0"
        os.environ["HS_FAKE_QUIET_TAIL"] = "0"

    def trainable(self, record="lidar_check"):
        """A solved project with a LiDAR init recorded (as hs scale --init-points leaves it)."""
        pj = Project(self.root, create=True)
        for s in ("ingest", "select"):
            pj.m["stages"][s] = {"status": "done"}
        pj.m["stages"]["solve"] = {"status": "done", "metrics": {"num_frames": 1}, "started": now_iso(),
                                   "finished": now_iso()}
        os.makedirs(os.path.join(pj.dataset_dir, "sparse"))
        for f in ("cameras.txt", "images.txt"):
            open(os.path.join(pj.dataset_dir, "sparse", f), "w").write("# x\n")
        np.savez(pj.rig_npz, names=np.array(["cap000_L"]), K=np.eye(3)[None], R=np.eye(3)[None],
                 t=np.zeros((1, 3)), pts=np.zeros((50, 3)), w=32, h=24)
        import cv2
        os.makedirs(os.path.join(pj.dataset_dir, "images", "L"))
        cv2.imwrite(os.path.join(pj.dataset_dir, "images", "L", "cap000.jpg"), np.zeros((24, 32, 3), np.uint8))
        p = initsplats.write(pj.path("scale", "lidar_init.ply"),
                             initsplats.splats(np.random.default_rng(1).normal(size=(64, 3))))
        rec = {"init_ply": "scale/lidar_init.ply", "init_ply_md5": md5_file(p), "init_rig_md5": md5_file(pj.rig_npz),
               "at": now_iso()}
        if record == "scale":
            pj.m["scale"] = dict(rec, source="lidar")
        else:
            pj.m["stages"]["scale"] = {"status": "pending", "lidar_check": rec}
        pj.save()
        return pj, p

    def test_stages_the_scan_init_and_removes_it(self):
        pj, p = self.trainable()
        md5 = md5_file(p)
        train.run(self.train_args(), pj)
        self.assertEqual(pj.status("train"), "done")
        tm = pj.stage("train")["metrics"]
        self.assertEqual(tm["init"], "lidar")
        self.assertEqual(tm["init_ply_md5"], md5)
        self.assertEqual(tm["init_splats"], 64)
        self.assertEqual(tm["dataset_fingerprint"]["init"], "lidar")
        self.assertEqual(tm["dataset_fingerprint"]["init_md5"], md5)
        log = open(pj.log_path("train")).read()
        self.assertIn(f"fake: init.ply md5 {md5}", log)                        # brush saw it
        self.assertFalse(os.path.exists(os.path.join(pj.dataset_dir, "init.ply")), "init.ply removed after")
        self.assertTrue(os.path.exists(p), "the source is kept")

    def test_applied_record_works_too(self):
        pj, p = self.trainable(record="scale")
        train.run(self.train_args(), pj)
        self.assertEqual(pj.stage("train")["metrics"]["init"], "lidar")

    def test_default_run_records_sparse_and_stages_nothing(self):
        pj, _p = self.trainable()
        train.run(self.train_args(init="sparse"), pj)
        tm = pj.stage("train")["metrics"]
        self.assertEqual(tm["init"], "sparse")
        self.assertNotIn("init_md5", tm["dataset_fingerprint"])
        self.assertNotIn("fake: init.ply", open(pj.log_path("train")).read())

    def test_refuses_with_resume_from(self):
        pj, p = self.trainable()
        with self.assertRaises(events.StageError):
            train.run(self.train_args(resume_from=p), pj)
        self.assertEqual(pj.status("train"), "pending")

    def test_refuses_a_stale_init(self):
        pj, p = self.trainable()
        os.makedirs(pj.exports_dir)
        keep = os.path.join(pj.exports_dir, "export_10.ply")
        initsplats.write(keep, initsplats.splats(np.zeros((3, 3)) + [[0, 0, 0], [1, 0, 0], [0, 1, 0]]))
        # a re-solve since: the init was written for another rig.npz
        np.savez(pj.rig_npz, names=np.array(["cap000_L"]), K=np.eye(3)[None], R=np.eye(3)[None],
                 t=np.ones((1, 3)), pts=np.zeros((50, 3)), w=32, h=24)
        with self.assertRaises(events.StageError) as cm:
            train.run(self.train_args(), pj)
        self.assertIn("another rig.npz", str(cm.exception))
        self.assertTrue(os.path.exists(keep), "refused before train/exports was touched")
        self.assertEqual(pj.status("train"), "pending")

    def test_refuses_without_a_record(self):
        pj, p = self.trainable()
        del pj.m["stages"]["scale"]["lidar_check"]
        pj.save()
        with self.assertRaises(events.StageError) as cm:
            train.run(self.train_args(), pj)
        self.assertIn("--init-points", cm.exception.hint)


if __name__ == "__main__":
    unittest.main()
