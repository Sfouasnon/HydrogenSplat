"""hs masks must not care what units the solve is in.

A mono solve with no reference in frame has no metres. 2026-09-20_GreetingCard asked for a
0.12 m radius, got 120 scene units, and the card sat 1,000-1,500 out: "only 0 points". The fix
fits the radius to the camera orbit, so the same scene at any scale gets the same masks. These
tests build one synthetic scene, solve it at x1 and at x13, and check exactly that.
"""
import os
import sys
import tempfile
import unittest
from argparse import Namespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from hs import events  # noqa: E402
from hs.project import Project, now_iso  # noqa: E402
from hs.stages import masks  # noqa: E402

W, H, FX = 640, 480, 800.0
N_CAM, RING, SPHERE = 12, 300.0, 40.0          # mm-units at x1


def look_at(C):
    f = -C / np.linalg.norm(C)
    down = np.array([0.0, 1.0, 0.0])
    y = down - down.dot(f) * f
    y /= np.linalg.norm(y)
    x = np.cross(y, f)
    return np.stack([x, y, f])


def build(root, k):
    """A sphere of splats at the origin, cameras on a ring around it, everything x k."""
    import cv2
    pj = Project(root, create=True)
    for s in ("ingest", "select", "solve"):
        pj.m["stages"][s] = {"status": "done", "finished": now_iso(), "checks": [], "metrics": {}}
    pj.m["stages"]["solve"]["checks"] = [{"name": "scene_scaled", "ok": k == 1}]
    rng = np.random.default_rng(1)
    v = rng.normal(size=(20000, 3))
    surf = v / np.linalg.norm(v, axis=1, keepdims=True) * SPHERE          # mm-units
    names, Ks, Rs, ts = [], [], [], []
    img_dir = os.path.join(pj.dataset_dir, "images", "L")
    os.makedirs(img_dir, exist_ok=True)
    for i in range(N_CAM):
        a = 2 * np.pi * i / N_CAM
        C = np.array([RING * np.cos(a), -60.0, RING * np.sin(a)])
        R = look_at(C)
        names.append(f"cap{i:03d}_L")
        Ks.append([[FX, 0, W / 2], [0, FX, H / 2], [0, 0, 1]])
        Rs.append(R)
        ts.append(-R @ (C * k))
        cv2.imwrite(os.path.join(img_dir, f"cap{i:03d}.jpg"), np.full((H, W, 3), 128, np.uint8))
    np.savez(pj.rig_npz, names=np.array(names), K=np.array(Ks), R=np.array(Rs), t=np.array(ts),
             wh=np.array([[W, H]] * N_CAM), pts=surf[::20] * k, stereo=False)
    props = ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2"]
    arr = np.zeros((len(surf), len(props)), "<f4")
    arr[:, :3] = surf * k / 1000.0                                         # ply is in "metres"
    arr[:, 3] = 3.0                                                        # sigmoid -> 0.95
    arr[:, 4:7] = np.log(2.0 * k / 1000.0)
    ply = os.path.join(root, "model.ply")
    hdr = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % len(arr)
           + "".join(f"property float {p}\n" for p in props) + "end_header\n")
    open(ply, "wb").write(hdr.encode() + arr.tobytes())
    pj.save()
    return pj, ply


def args(ply, **kw):
    a = dict(radius=None, radius_scale=1.0, ply=ply, min_opacity=0.1, margin_mm=None,
             margin_frac=masks.MARGIN_FRAC, close_px=25, keep_largest=True, preview=0,
             max_points=60000, from_points=False, point_mm=None, method="geometry",
             select_frac=masks.SELECT_FRAC, feather_px=masks.FEATHER_PX, grow_px=0)
    a.update(kw)
    return Namespace(**a)


def load_masks(pj):
    import cv2
    d = os.path.join(pj.dataset_dir, "masks", "L")
    return {f: cv2.imread(os.path.join(d, f), cv2.IMREAD_GRAYSCALE) > 127 for f in sorted(os.listdir(d))}


class ScaleFree(unittest.TestCase):
    def setUp(self):
        self.t1, self.t13 = tempfile.TemporaryDirectory(), tempfile.TemporaryDirectory()
        self.addCleanup(self.t1.cleanup)
        self.addCleanup(self.t13.cleanup)

    def test_same_scene_same_masks_at_any_scale(self):
        p1, ply1 = build(self.t1.name, 1)
        p13, ply13 = build(self.t13.name, 13)
        masks.run(args(ply1), p1)
        masks.run(args(ply13), p13)
        m1, m13 = load_masks(p1), load_masks(p13)
        self.assertEqual(sorted(m1), sorted(m13))
        for f in m1:
            a, b = m1[f], m13[f]
            iou = (a & b).sum() / max((a | b).sum(), 1)
            self.assertGreater(iou, 0.98, f"{f}: IoU {iou:.3f} between x1 and x13")
            self.assertGreater(a.mean(), 0.01, f"{f}: empty silhouette")
        r1 = p1.stage("masks")["metrics"]["radius_m"]
        r13 = p13.stage("masks")["metrics"]["radius_m"]
        self.assertAlmostEqual(r13 / r1, 13.0, places=3)
        # 0.7 * 300 * (640/2) / 800 = 84 units at x1
        self.assertAlmostEqual(r1 * 1000, 0.7 * RING * np.hypot(1, 60 / RING) * (W / 2) / FX, delta=1.0)
        self.assertFalse(p13.stage("masks")["metrics"]["scene_scaled"])

    def test_a_metric_radius_is_what_broke(self):
        # the old default, 0.12 m, on the same scene at x13: the sphere sits 520 units out
        p13, ply13 = build(self.t13.name, 13)
        with self.assertRaises(events.StageError) as e:
            masks.run(args(ply13, radius=0.12), p13)
        self.assertIn("the radius out", str(e.exception))
        self.assertIn("--radius-scale", e.exception.hint)

    def test_radius_scale_tightens(self):
        p1, ply1 = build(self.t1.name, 1)
        masks.run(args(ply1, radius_scale=1.0), p1)
        full = sum(m.sum() for m in load_masks(p1).values())
        masks.run(args(ply1, radius_scale=0.5), p1)       # 42 units: inside the 40-unit sphere's shell
        tight = sum(m.sum() for m in load_masks(p1).values())
        self.assertLessEqual(tight, full)

    def test_sampling_is_deterministic(self):
        p1, ply1 = build(self.t1.name, 1)
        masks.run(args(ply1, max_points=3000), p1)
        a = load_masks(p1)
        masks.run(args(ply1, max_points=3000), p1)
        b = load_masks(p1)
        for f in a:
            self.assertTrue((a[f] == b[f]).all(), f"{f} changed between identical runs")
        self.assertEqual(p1.stage("masks")["metrics"]["points_used"], 3000)


FAKE_SEGMENT = os.path.join(HERE, "fake_segment.py")


class Vision(unittest.TestCase):
    """--method vision: Vision finds objects, the geometric mask decides which one is the subject."""

    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.env = {k: os.environ.get(k) for k in ("HS_SEGMENT_BIN", "HS_FAKE_SEGMENT")}
        # The fake is a Python script that imports cv2. Run directly, its `#!/usr/bin/env python3`
        # picks whatever python3 is first on PATH — on the Mac the system one, which has no cv2 —
        # so wrap it in a shell stub that runs it with the interpreter running these tests.
        stub = os.path.join(self.t.name, "hs-segment-fake")
        with open(stub, "w") as f:
            f.write(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE_SEGMENT}" "$@"\n')
        os.chmod(stub, 0o755)
        os.environ["HS_SEGMENT_BIN"] = stub
        self.addCleanup(self.restore)
        self.pj, self.ply = build(self.t.name, 1)

    def restore(self):
        for k, v in self.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def disc(self, r=90):
        import cv2
        d = np.zeros((H, W), np.uint8)
        cv2.circle(d, (W // 2, H // 2), r, 255, -1)
        return d > 0

    def test_the_instance_inside_the_region_is_kept_and_the_clutter_dropped(self):
        masks.run(args(self.ply, method="vision"), self.pj)
        disc = self.disc()
        for f, m in load_masks(self.pj).items():
            iou = (m & disc).sum() / (m | disc).sum()
            self.assertGreater(iou, 0.97, f"{f}: IoU {iou:.3f} with the subject instance")
            self.assertFalse(m[0:60, 0:80].any(), f"{f}: the corner block (not the subject) got in")
        met = self.pj.stage("masks")["metrics"]
        self.assertEqual(met["method"], "vision")
        self.assertEqual(met["vision_instances_median"], 2)
        self.assertEqual(met["vision_selected_median"], 1)
        self.assertEqual(met["vision_fell_back"], 0)
        self.assertLess(met["coverage_median"], met["prior_coverage_median"], "the object is tighter than the region")

    def test_the_edge_is_hard_with_an_anti_aliased_rim(self):
        import cv2
        masks.run(args(self.ply, method="vision", feather_px=1.0), self.pj)
        m = cv2.imread(os.path.join(self.pj.dataset_dir, "masks", "L", "cap000.png"), cv2.IMREAD_GRAYSCALE)
        grey = ((m > 0) & (m < 255)).mean()
        rim = self.disc(92) & ~self.disc(88)
        self.assertGreater(grey, 0, "no anti-aliasing at all")
        self.assertLess(grey, 1.5 * rim.mean(), "the soft zone is wider than a couple of px")
        masks.run(args(self.ply, method="vision", feather_px=0), self.pj)
        m = cv2.imread(os.path.join(self.pj.dataset_dir, "masks", "L", "cap000.png"), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(set(np.unique(m)) - {0, 255}, set(), "feather 0 must be binary")

    def test_nothing_found_falls_back_to_the_region_and_says_so(self):
        os.environ["HS_FAKE_SEGMENT"] = "none"
        masks.run(args(self.ply, method="vision"), self.pj)
        st = self.pj.stage("masks")
        self.assertEqual(st["metrics"]["vision_fell_back"], N_CAM)
        chk = {c["name"]: c for c in st["checks"]}["vision_found_the_subject"]
        self.assertFalse(chk["ok"])
        self.assertIn("fell back", chk["value"])
        self.assertTrue(all(m.mean() > 0.01 for m in load_masks(self.pj).values()), "fallback masks empty")

    def test_per_image_errors_fall_back_but_a_crash_stops_the_stage(self):
        os.environ["HS_FAKE_SEGMENT"] = "error"
        masks.run(args(self.ply, method="vision"), self.pj)
        self.assertIn("vision error", self.pj.stage("masks")["metrics"]["vision_fell_back_views"][0])
        os.environ["HS_FAKE_SEGMENT"] = "crash"
        with self.assertRaises(events.StageError) as e:
            masks.run(args(self.ply, method="vision"), self.pj)
        self.assertIn("exited 3", str(e.exception))
        self.assertIn("--method geometry", e.exception.hint)

    def test_no_helper_off_macos_names_the_way_out(self):
        os.environ.pop("HS_SEGMENT_BIN")
        if sys.platform == "darwin":
            self.skipTest("macOS builds the real helper")
        with self.assertRaises(events.StageError) as e:
            masks.run(args(self.ply, method="vision"), self.pj)
        self.assertIn("--method geometry", e.exception.hint)


class FillHoles(unittest.TestCase):
    def test_a_subject_in_the_corner_does_not_fill_the_frame(self):
        m = np.zeros((100, 160), np.uint8)
        m[0:60, 0:70] = 255                      # the subject covers the top-left corner
        m[20:40, 20:40] = 0                      # with a hole in it
        out = masks.fill_holes(m)
        self.assertTrue((out[20:40, 20:40] == 255).all(), "the enclosed hole was not filled")
        self.assertTrue((out[:, 100:] == 0).all(), "background beyond the subject was filled")
        self.assertAlmostEqual((out > 0).mean(), 60 * 70 / (100 * 160), places=6)

    def test_subject_touching_every_border_keeps_its_open_gaps(self):
        m = np.full((50, 50), 255, np.uint8)
        m[10:20, 0:30] = 0                       # a notch open to the left border: background, not a hole
        out = masks.fill_holes(m)
        self.assertTrue((out[10:20, 0:30] == 0).all())

    def test_empty_stays_empty(self):
        self.assertEqual(int(masks.fill_holes(np.zeros((30, 30), np.uint8)).sum()), 0)


if __name__ == "__main__":
    unittest.main()
