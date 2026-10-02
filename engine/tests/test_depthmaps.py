"""hs/depthmaps.py and `hs train --depth-weight`.

The scene is a ball of radius 100 mm on a table, cameras on two rings around it 500 mm from its
centre. The ball's depth along any camera ray is known in closed form, so every depth the module
writes can be checked against the truth. The interesting scan is the half one: only the side of
the ball facing +x was scanned (nobody walked behind it, as with the Stormtrooper helmet), so a
camera on the -x side looks into the scanned shell from behind. Its true depth there is the ball's
unscanned near side; the scan only offers the far side, 100 to 200 mm too deep. Those pixels must
come out empty, not wrong.
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

from hs import depthmaps as D, events, initsplats  # noqa: E402
from hs.project import Project, md5_file, now_iso  # noqa: E402
from hs.stages import train  # noqa: E402

FAKEBIN = os.path.join(HERE, "fakebin")
RADIUS = 100.0
DIST = 500.0
W, H, F = 320, 240, 300.0
K = np.array([[F, 0, W / 2.0], [0, F, H / 2.0], [0, 0, 1.0]])
DEPTH_FLAGS = "--depth-loss-weight=,--depth-loss-tolerance=,--depth-loss-every=,--depth-unit="


def fibonacci_sphere(n, radius=RADIUS):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return radius * np.stack([np.cos(th) * np.sin(phi), np.cos(phi), np.sin(th) * np.sin(phi)], 1)


def table(n=20000, half=450.0, y=RADIUS):
    """Camera 'down' is +y: the table is the plane y = +RADIUS the ball rests on."""
    rng = np.random.default_rng(5)
    xz = rng.uniform(-half, half, (n, 2))
    return np.stack([xz[:, 0], np.full(n, y), xz[:, 1]], 1)


def cameras(n_ring=10):
    """Two rings looking at the ball's centre. -> [(R, t, C)], Xc = X @ R.T + t."""
    out = []
    for height, phase in ((-150.0, 0.0), (-320.0, 0.5)):          # above the table (y is down)
        rad = np.sqrt(DIST ** 2 - height ** 2)
        for k in range(n_ring):
            a = 2 * np.pi * (k + phase) / n_ring
            C = np.array([rad * np.cos(a), height, rad * np.sin(a)])
            fwd = -C / np.linalg.norm(C)
            right = np.cross(fwd, [0, 1.0, 0])
            right /= np.linalg.norm(right)
            down = np.cross(fwd, right)
            R = np.stack([right, down, fwd])
            out.append((R, -R @ C, C))
    return out


def true_depth(R, C):
    """The ball's depth (camera z) at every pixel centre; 0 where the ray misses it."""
    v, u = np.mgrid[0:H, 0:W]
    d_cam = np.stack([(u + 0.5 - W / 2.0) / F, (v + 0.5 - H / 2.0) / F, np.ones((H, W))], -1)
    d = d_cam @ R                                   # world directions, per unit of camera z
    b = d @ C
    a = (d * d).sum(-1)
    disc = b * b - a * (C @ C - RADIUS ** 2)
    with np.errstate(invalid="ignore"):
        z = (-b - np.sqrt(disc)) / a
    return np.where(disc > 0, z, 0.0)


def rig(cams):
    n = len(cams)
    return dict(names=np.array([f"cap{i:03d}_L" for i in range(n)]), K=np.tile(K, (n, 1, 1)),
                R=np.array([c[0] for c in cams]), t=np.array([c[1] for c in cams]),
                C=np.array([c[2] for c in cams]), pts=np.zeros((50, 3)),
                wh=np.tile([W, H], (n, 1)), w=W, h=H, stereo=False)


def write_dataset(dataset, cams):
    """images/L and masks/L for every camera: the mask is the ball's true silhouette."""
    import cv2
    for top in ("images", "masks"):
        os.makedirs(os.path.join(dataset, top, "L"), exist_ok=True)
    for i, (R, _t, C) in enumerate(cams):
        m = (true_depth(R, C) > 0).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(dataset, "masks", "L", f"cap{i:03d}.png"), m)
        cv2.imwrite(os.path.join(dataset, "images", "L", f"cap{i:03d}.jpg"), np.dstack([m // 2] * 3))


class RenderTest(unittest.TestCase):
    def test_a_full_scan_gives_the_true_depth(self):
        pts = fibonacci_sphere(60000)
        R, t, C = cameras()[3]
        depth, info = D.render(pts, K, R, t, (W, H))
        truth = true_depth(R, C)
        ok = depth > 0
        self.assertGreater(ok.sum(), 0.7 * (truth > 0).sum())
        # a limb pixel can hold a scan point while its centre ray misses the ball: one ring, no more
        import cv2
        ring = cv2.dilate((truth > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        self.assertFalse((ok & ~ring).any(), "no depth outside the ball")
        self.assertLess(int((ok & (truth == 0)).sum()), 0.02 * ok.sum())
        ok &= truth > 0
        err = depth[ok] - truth[ok]
        self.assertLess(abs(float(np.median(err))), 0.5)
        self.assertLess(float(np.percentile(np.abs(err), 95)), 2.5)
        # the far side is 2 * sqrt(R^2 - r^2) behind: none of it comes through
        self.assertLess(float(err.max()), 15.0)
        self.assertEqual(info["size"], [W, H])

    def test_the_map_is_capped_at_the_photographs_size_and_scaled_below_it(self):
        self.assertEqual(D.map_size((W, H), 512), (W, H))
        self.assertEqual(D.map_size((2137, 3807), 512), (287, 512))
        pts = fibonacci_sphere(60000)
        R, t, C = cameras()[0]
        small, _ = D.render(pts, K, R, t, (W, H), long_edge=160)
        self.assertEqual(small.shape, (120, 160))
        truth = true_depth(R, C)[::2, ::2]           # close enough: the same rays to within a pixel
        ok = (small > 0) & (truth > 0)
        self.assertGreater(ok.sum(), 500)
        self.assertLess(abs(float(np.median(small[ok] - truth[ok]))), 2.0)

    def test_png_round_trip_and_units(self):
        self.assertEqual(D.choose_unit_mm(600.0), 0.1)
        self.assertEqual(D.choose_unit_mm(6500.0), 0.1)
        self.assertEqual(D.choose_unit_mm(6501.0), 0.2)
        self.assertEqual(D.choose_unit_mm(30000.0), 0.5)
        d = np.array([[0.0, 0.04, 123.456], [599.99, 6499.9, 0.0]], np.float32)
        tmp = tempfile.mkdtemp(prefix="hs-depth-")
        try:
            p = os.path.join(tmp, "L", "x.png")
            D.write_png16(p, d, 0.1)
            back = D.read_png16(p, 0.1)
            self.assertEqual(back[0, 0], 0.0)
            self.assertGreater(back[0, 1], 0.0, "a measured pixel never rounds to 'no measurement'")
            np.testing.assert_allclose(back[d > 0.05], d[d > 0.05], atol=0.051)
            with self.assertRaises(ValueError):
                D.write_png16(p, np.array([[7000.0]], np.float32), 0.1)
        finally:
            shutil.rmtree(tmp)

    def test_view_key(self):
        self.assertEqual(D.view_key("cap004_R"), "R/cap004")
        self.assertEqual(D.view_key("sel010-00085_L"), "L/sel010-00085")


class BuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="hs-depth-")
        cls.cams = cameras()
        cls.dataset = os.path.join(cls.tmp, "dataset")
        write_dataset(cls.dataset, cls.cams)
        cls.G = os.path.join(cls.tmp, "rig.npz")
        np.savez(cls.G, **rig(cls.cams))
        ball = fibonacci_sphere(60000)
        cls.half = np.concatenate([ball[ball[:, 0] > -20.0], table()])     # the +x side, and the table
        cls.out = os.path.join(cls.tmp, "depth")
        G = np.load(cls.G, allow_pickle=True)
        cls.names = [str(n) for n in G["names"]]
        cls.exclude = {"L/cap004"}
        cls.rep = D.build(cls.half, G, cls.names, cls.out, layer="subject",
                          masks_dir=os.path.join(cls.dataset, "masks"),
                          images_dir=os.path.join(cls.dataset, "images"), exclude=cls.exclude)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_the_subject_is_the_ball_not_the_table(self):
        s = self.rep["subject"]
        n_ball = int((np.linalg.norm(self.half, axis=1) < RADIUS + 1).sum())
        self.assertGreater(s["points"], 0.95 * n_ball)
        self.assertLess(s["points"], 1.05 * n_ball)
        self.assertEqual(self.rep["scan_points"], len(self.half))

    def test_nothing_is_written_too_deep(self):
        """Every depth in every view is the ball's near surface. The far side seen from behind
        would be 100-200 mm deeper."""
        worst, n = 0.0, 0
        for i, (R, _t, C) in enumerate(self.cams):
            p = os.path.join(self.out, "L", f"cap{i:03d}.png")
            if not os.path.exists(p):
                continue
            d = D.read_png16(p, self.rep["unit_mm"])
            truth = true_depth(R, C)
            ok = d > 0
            self.assertFalse((ok & (truth == 0)).any())
            err = d[ok] - truth[ok]
            worst = max(worst, float(np.abs(err).max()))
            n += int(ok.sum())
            # a view that only catches the scan's edge is all grazing surface, steep per pixel
            sliver = ok.sum() < 0.3 * (truth > 0).sum()
            self.assertLess(float(np.percentile(np.abs(err), 95)), 6.0 if sliver else 3.0, f"cap{i:03d}")
        self.assertGreater(n, 20000)
        self.assertLess(worst, 15.0)

    def test_the_hull_is_what_keeps_the_far_side_out(self):
        """The same scan rendered into a rear view without the hull test: wrong by the ball's width."""
        ball = self.half[np.linalg.norm(self.half, axis=1) < RADIUS + 1]
        i = int(np.argmin([c[2][0] for c in self.cams]))            # the camera furthest to -x
        R, t, C = self.cams[i]
        naive, _ = D.render(ball, K, R, t, (W, H))
        truth = true_depth(R, C)
        ok = (naive > 0) & (truth > 0)
        self.assertGreater(float((naive[ok] - truth[ok]).max()), 100.0)
        row = next(r for r in self.rep["per_view"] if r["view"] == f"L/cap{i:03d}")
        self.assertLess(row["region_covered"], 0.15, "the rear view is left (almost) empty instead")

    def test_coverage_follows_what_was_scanned(self):
        rows = {r["view"]: r for r in self.rep["per_view"]}
        for i, (_R, _t, C) in enumerate(self.cams):
            key = f"L/cap{i:03d}"
            if key in self.exclude:
                self.assertNotIn(key, rows)
                self.assertFalse(os.path.exists(os.path.join(self.out, key + ".png")))
            elif C[0] > 350:
                self.assertGreater(rows[key]["region_covered"], 0.7, key)
                self.assertTrue(rows[key]["written"])
            elif C[0] < -350:
                self.assertLess(rows[key]["region_covered"], 0.3, key)       # the top of the ball is in both halves
        self.assertEqual(self.rep["views"], len(self.cams) - 1)
        self.assertEqual(self.rep["excluded"], 1)
        self.assertEqual(self.rep["views_with_depth"],
                         len([f for f in os.listdir(os.path.join(self.out, "L")) if f.endswith(".png")]))
        self.assertGreater(self.rep["coverage"], 0.3)
        self.assertLess(self.rep["coverage"], 0.75, "half a scan cannot cover the whole orbit")
        self.assertGreater(self.rep["views_under_half"], 3)
        self.assertTrue(os.path.exists(os.path.join(self.out, self.rep["sheet"])))

    def test_a_scan_of_something_else_is_refused(self):
        G = np.load(self.G, allow_pickle=True)
        with self.assertRaises(ValueError) as cm:
            D.build(table(), G, self.names, os.path.join(self.tmp, "none"), layer="subject",
                    masks_dir=os.path.join(self.dataset, "masks"))
        self.assertIn("does not hold the subject", str(cm.exception))

    def test_full_and_background_layers(self):
        G = np.load(self.G, allow_pickle=True)
        full_scan = np.concatenate([fibonacci_sphere(60000), table()])
        out = os.path.join(self.tmp, "full")
        rep = D.build(full_scan, G, self.names, out, layer="full")
        self.assertEqual(rep["views_with_depth"], len(self.cams))
        R, _t, C = self.cams[2]
        d = D.read_png16(os.path.join(out, "L", "cap002.png"), rep["unit_mm"])
        truth = true_depth(R, C)
        ok = (d > 0) & (truth > 0)
        self.assertGreater(ok.sum(), 0.6 * (truth > 0).sum())
        self.assertLess(float(np.percentile(np.abs(d[ok] - truth[ok]), 95)), 3.0)
        self.assertGreater(int(((d > 0) & (truth == 0)).sum()), 2000, "the table has depth too")

        out = os.path.join(self.tmp, "background")
        rep = D.build(full_scan, G, self.names, out, layer="background",
                      masks_dir=os.path.join(self.dataset, "masks"))
        d = D.read_png16(os.path.join(out, "L", "cap002.png"), rep["unit_mm"])
        self.assertFalse(((d > 0) & (truth > 0)).any(), "nothing inside the subject's silhouette")
        self.assertGreater(int((d > 0).sum()), 2000)


class TrainDepth(unittest.TestCase):
    TOTAL = 1000

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hs-depth-")
        self.root = os.path.join(self.tmp, "proj")
        self.out = io.StringIO()
        self._redir = contextlib.redirect_stdout(self.out)
        self._redir.__enter__()
        self._env = {k: os.environ.get(k) for k in ("HS_PYTHON", "HS_FAKE_REFINE_STOP", "HS_FAKE_QUIET_TAIL",
                                                    "HS_FAKE_BRUSH_FLAGS")}
        os.environ.update(HS_PYTHON=sys.executable, HS_FAKE_REFINE_STOP="0", HS_FAKE_QUIET_TAIL="0",
                          HS_FAKE_BRUSH_FLAGS=DEPTH_FLAGS)

    def tearDown(self):
        self._redir.__exit__(None, None, None)
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(self.tmp, ignore_errors=True)

    def train_args(self, **kw):
        base = dict(brush=os.path.join(FAKEBIN, "brush"), total_train_iters=self.TOTAL, growth_stop_iter=700,
                    refine_every=100, split_at_screen_size=None, export_every=500, resume_from=None,
                    start_iter=None, no_caffeinate=True, brush_args="", exclude="", no_masks=False,
                    min_scale_factor=None, layer="subject", alpha_mode=None, init="sparse",
                    depth_weight=0.2, depth_tolerance=0.01, depth_res=512, depth_every=1)
        base.update(kw)
        return Namespace(**base)

    def project(self, sparse_rows=300):
        """A solved project whose scale stage recorded a LiDAR init: the half-scanned ball in the
        dataset's metres, then `sparse_rows` of the solve's own points (which are not the scan)."""
        pj = Project(self.root, create=True)
        for s in ("ingest", "select"):
            pj.m["stages"][s] = {"status": "done"}
        cams = cameras()
        pj.m["stages"]["solve"] = {"status": "done", "metrics": {"num_frames": len(cams)}, "started": now_iso(),
                                   "finished": now_iso()}
        os.makedirs(os.path.join(pj.dataset_dir, "sparse"))
        for f in ("cameras.txt", "images.txt"):
            open(os.path.join(pj.dataset_dir, "sparse", f), "w").write("# x\n")
        np.savez(pj.rig_npz, **rig(cams))
        write_dataset(pj.dataset_dir, cams)
        ball = fibonacci_sphere(60000)
        scan = np.concatenate([ball[ball[:, 0] > -20.0], table()])
        stray = np.random.default_rng(2).normal(0, 30.0, (sparse_rows, 3))      # inside the ball: not a surface
        p = initsplats.write(pj.path("scale", "lidar_init.ply"),
                             initsplats.splats(np.concatenate([scan, stray]) / 1000.0))
        rec = {"init_ply": "scale/lidar_init.ply", "init_ply_md5": md5_file(p), "init_rig_md5": md5_file(pj.rig_npz),
               "init_points_scan": len(scan), "init_points_sparse": sparse_rows, "at": now_iso()}
        pj.m["scale"] = dict(rec, source="lidar")
        pj.save()
        self.n_scan = len(scan)
        return pj, cams

    def test_depth_maps_reach_brush(self):
        pj, cams = self.project()
        train.run(self.train_args(exclude="L/cap004"), pj)
        self.assertEqual(pj.status("train"), "done")
        st = pj.stage("train")
        tm = st["metrics"]
        rep = json.load(open(pj.path("train", "depth", "depth_report.json")))
        self.assertEqual(rep["scan_points"], self.n_scan, "the sparse rows of the init file are left out")
        self.assertEqual(rep["layer"], "subject")
        # the files, and the view Brush was pointed at
        maps = sorted(os.listdir(pj.path("train", "depth", "L")))
        self.assertEqual(len(maps), rep["views_with_depth"])
        self.assertNotIn("cap004.png", maps)
        links = sorted(os.listdir(pj.path("train", "view", "depths", "L")))
        self.assertEqual(links, maps)
        self.assertTrue(all(os.path.islink(pj.path("train", "view", "depths", "L", f)) for f in links))
        self.assertEqual(sorted(os.listdir(pj.path("train", "view"))), ["depths", "images", "masks", "sparse"])
        argv = st["brush_argv"]
        self.assertEqual(argv[1], pj.path("train", "view"))
        for flag, value in (("--depth-loss-weight", "0.2"), ("--depth-loss-tolerance", "0.01"),
                            ("--depth-loss-every", "1"), ("--depth-unit", "0.0001")):
            self.assertEqual(argv[argv.index(flag) + 1], value, flag)
        # what was recorded
        d = tm["depth"]
        self.assertEqual((d["weight"], d["unit"], d["views_with_depth"]), (0.2, 0.0001, len(maps)))
        self.assertEqual(d["views"], len(cams) - 1)
        self.assertEqual(tm["brush_config"]["depth_loss_weight"], 0.2)
        fp = tm["dataset_fingerprint"]["depth"]
        self.assertEqual(fp["count"], len(maps))
        self.assertEqual(fp["source_md5"], md5_file(pj.path("scale", "lidar_init.ply")))
        checks = {c["name"]: c for c in st["checks"]}
        self.assertTrue(checks["depth_maps_loaded"]["ok"], checks["depth_maps_loaded"])
        self.assertTrue(checks["depth_error_held"]["ok"], checks["depth_error_held"])
        self.assertFalse(checks["depth_reference_coverage"]["ok"], "half a scan: flagged, not hidden")
        self.assertIn("under half", checks["depth_reference_coverage"]["value"])
        self.assertTrue(checks["view_count_matches"]["ok"])
        self.assertEqual(tm["depth_error_curve"][0][0], 1)
        self.assertLess(tm["depth_rel_error_final"], tm["depth_error_curve"][0][1])
        log = open(pj.log_path("train")).read()
        self.assertIn(f"Depth maps for {len(maps)} of {len(cams) - 1} training views", log)

    def test_off_by_default_and_cleans_up(self):
        pj, _cams = self.project()
        train.run(self.train_args(), pj)
        self.assertTrue(os.path.isdir(pj.path("train", "depth")))
        train.run(self.train_args(depth_weight=0.0), pj)
        st = pj.stage("train")
        self.assertNotIn("--depth-loss-weight", st["brush_argv"])
        self.assertFalse(os.path.exists(pj.path("train", "depth")), "an earlier run's maps do not linger")
        self.assertFalse(os.path.exists(pj.path("train", "view")), "nor the view that linked them")
        self.assertNotIn("depth", st["metrics"])
        self.assertNotIn("depth", st["metrics"]["dataset_fingerprint"])
        self.assertEqual(st["metrics"]["brush_config"]["depth_loss_weight"], 0.0)

    def test_refuses_a_brush_without_the_flag(self):
        pj, _cams = self.project()
        os.environ["HS_FAKE_BRUSH_FLAGS"] = ""
        with self.assertRaises(events.StageError) as cm:
            train.run(self.train_args(), pj)
        self.assertIn("depth-loss", cm.exception.hint)
        self.assertEqual(pj.status("train"), "pending")
        self.assertFalse(os.path.exists(pj.path("train", "depth")))

    def test_refuses_without_a_registered_scan(self):
        pj, _cams = self.project()
        del pj.m["scale"]
        pj.save()
        with self.assertRaises(events.StageError) as cm:
            train.run(self.train_args(), pj)
        self.assertIn("--depth-weight", str(cm.exception))
        self.assertIn("--init-points", cm.exception.hint)

    def test_refuses_a_scan_registered_to_another_solve(self):
        pj, cams = self.project()
        r = rig(cams)
        r["t"] = r["t"] + 1.0
        np.savez(pj.rig_npz, **r)
        with self.assertRaises(events.StageError) as cm:
            train.run(self.train_args(), pj)
        self.assertIn("another rig.npz", str(cm.exception))

    def test_with_the_scan_as_init_too(self):
        pj, _cams = self.project()
        train.run(self.train_args(init="lidar", alpha_mode=None), pj)
        tm = pj.stage("train")["metrics"]
        self.assertEqual(tm["init"], "lidar")
        self.assertEqual(tm["depth"]["source_md5"], tm["init_ply_md5"])
        self.assertIn("fake: init.ply md5", open(pj.log_path("train")).read())


if __name__ == "__main__":
    unittest.main()
