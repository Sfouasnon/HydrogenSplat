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


class MisplacedCamera(unittest.TestCase):
    def test_a_camera_with_the_subject_behind_it_gets_an_empty_mask_not_a_crash(self):
        # CirclesSculpture 2026-09-27: 12 cameras the solve placed kilometres away had no subject
        # point in front of them; the margin's median depth was NaN and the stage died
        with tempfile.TemporaryDirectory() as d:
            pj, ply = build(d, 1)
            G = dict(np.load(pj.rig_npz, allow_pickle=True))
            R0 = np.asarray(G["R"][0])
            G["R"][0] = np.diag([-1.0, 1.0, -1.0]) @ R0          # turn camera 0 round: the sphere is behind it
            G["t"][0] = np.diag([-1.0, 1.0, -1.0]) @ np.asarray(G["t"][0])
            np.savez(pj.rig_npz, **G)
            masks.run(args(ply), pj)
            m = load_masks(pj)
            self.assertFalse(m["cap000.png"].any())
            self.assertTrue(all(v.any() for k, v in m.items() if k != "cap000.png"))
            st = pj.stage("masks")
            self.assertEqual(st["metrics"]["views_subject_not_in_front"], 1)
            self.assertEqual(st["metrics"]["views_subject_not_in_front_names"], ["cap000_L"])
            c = {x["name"]: x for x in st["checks"]}
            self.assertFalse(c["subject_in_front_of_every_camera"]["ok"])
            self.assertTrue(c["every_view_has_a_silhouette"]["ok"])      # the unseen view is not "nearly empty"
            self.assertEqual(st["metrics"]["views"], N_CAM - 1)


class ExcludeHighlights(unittest.TestCase):
    """--exclude-highlights: neutral glints inside the subject are cut out, colour and paint are not."""

    def paint(self, pj, base, glint=True):
        import cv2
        d = os.path.join(pj.dataset_dir, "images", "L")
        for f in os.listdir(d):
            im = np.full((H, W, 3), base, np.uint8)
            if glint:
                cv2.circle(im, (W // 2, H // 2), 4, (252, 252, 252), -1)        # a neutral glint at the centre
                im[H // 2 - 3:H // 2 + 3, W // 2 + 20:W // 2 + 26] = (0, 0, 255)  # bright red: not a glint
            cv2.imwrite(os.path.join(d, f), im, [cv2.IMWRITE_JPEG_QUALITY, 97])

    def test_glints_cut_colour_and_paint_kept(self):
        with tempfile.TemporaryDirectory() as d:
            pj, ply = build(d, 1)
            self.paint(pj, 225)
            masks.run(args(ply, exclude_highlights=240, highlight_grow_px=2), pj)
            for f, m in load_masks(pj).items():
                self.assertFalse(m[H // 2 - 3:H // 2 + 4, W // 2 - 3:W // 2 + 4].any(), f"{f}: glint left in")
                self.assertTrue(m[H // 2, W // 2 + 23], f"{f}: the red patch was cut")
                self.assertTrue(m[H // 2 + 15, W // 2], f"{f}: paint beside the glint was cut")
            st = pj.stage("masks")
            self.assertEqual(st["metrics"]["exclude_highlights"], 240)
            self.assertLess(st["metrics"]["highlights_share_of_subject_median"], 0.05)
            c = {x["name"]: x for x in st["checks"]}
            self.assertTrue(c["highlights_are_glints_not_paint"]["ok"])

    def test_paint_over_the_line_fails_the_check(self):
        with tempfile.TemporaryDirectory() as d:
            pj, ply = build(d, 1)
            self.paint(pj, 245, glint=False)                  # SDR whites that clip: the paint passes 240
            masks.run(args(ply, exclude_highlights=240, highlight_grow_px=0), pj)
            c = {x["name"]: x for x in pj.stage("masks")["checks"]}
            self.assertFalse(c["highlights_are_glints_not_paint"]["ok"])
            self.assertIn("that is surface", c["highlights_are_glints_not_paint"]["value"])

    def test_off_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            pj, ply = build(d, 1)
            self.paint(pj, 225)
            masks.run(args(ply), pj)
            self.assertTrue(all(m[H // 2, W // 2] for m in load_masks(pj).values()))
            self.assertIsNone(pj.stage("masks")["metrics"]["exclude_highlights"])


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

    def test_a_glossy_subject_is_kept_when_it_covers_its_fragmentary_prior(self):
        """IMG_2525: the SfM points sat on the faceplate only, so the prior is a fragment of the
        helmet and the helmet is mostly outside it. It covers the prior, so it is the subject; a
        block beside it that covers none of the prior is not. --contain-frac 0 restores the old rule."""
        import argparse
        import cv2
        d = self.t.name
        big = np.zeros((H, W), np.uint8)
        cv2.circle(big, (W // 2, H // 2), 100, 255, -1)            # the helmet
        clutter = np.zeros((H, W), np.uint8)
        clutter[0:60, 0:80] = 255                                   # a box beside it
        cv2.imwrite(os.path.join(d, "1.png"), big)
        cv2.imwrite(os.path.join(d, "2.png"), clutter)
        prior = np.zeros((H, W), np.uint8)
        cv2.circle(prior, (W // 2 + 30, H // 2 + 20), 40, 255, -1)  # the faceplate: ~16% of the helmet
        seg = {"dir": d, "instances": 2}
        a = argparse.Namespace(select_frac=0.5, contain_frac=0.6, grow_px=0, feather_px=0)
        m, n, k, why = masks.vision_mask(seg, prior, a)
        self.assertIsNone(why)
        self.assertEqual((n, k), (2, 1))
        self.assertGreater(((m > 0) & (big > 0)).sum() / (big > 0).sum(), 0.99)
        self.assertFalse(m[0:60, 0:80].any())
        a.contain_frac = 0
        m, n, k, why = masks.vision_mask(seg, prior, a)
        self.assertEqual(k, 0)
        self.assertEqual(why, "nothing inside the geometric mask")

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


class SnapEdge(Vision):
    """--snap-edge: the boundary goes where the photograph's edge is, not where Vision's 50 % level is.

    2026-09-28_Stormtrooper_iPhone: Vision's matte sat 2-6 px outside the helmet, by a different
    amount per view, and the trainer's answer to a rim that is subject in one view and background in
    the next was a half-opaque ramp 6-8 px wide (claude/edge-profile-mask-bias-2026-10-04.md). Here
    the photograph holds a disc of radius 86 and the fake Vision instance is a disc of radius 90."""

    PHOTO_R = 86

    def paint(self, colour=(200, 180, 160)):
        import cv2
        d = os.path.join(self.pj.dataset_dir, "images", "L")
        for f in os.listdir(d):
            im = np.full((H, W, 3), 128, np.uint8)
            cv2.circle(im, (W // 2, H // 2), self.PHOTO_R, colour, -1)
            cv2.imwrite(os.path.join(d, f), im, [cv2.IMWRITE_JPEG_QUALITY, 95])

    def test_the_estimator_reads_the_bias_and_nothing_on_a_flat_photograph(self):
        import cv2
        from hs import edgesnap
        im = np.full((H, W, 3), 128, np.uint8)
        cv2.circle(im, (W // 2, H // 2), self.PHOTO_R, (200, 180, 160), -1)
        for bias in (0, 2, 4, -3):
            r = edgesnap.measure(im, self.disc(self.PHOTO_R + bias))
            self.assertIsNotNone(r, f"bias {bias}: unmeasured")
            # the step sits between the last background sample and the first subject sample, so a
            # boundary exactly on the edge reads -0.5 and a mask k px generous reads k - 0.5
            self.assertAlmostEqual(r["offset"], bias - 0.5, delta=0.6, msg=f"bias {bias}: read {r['offset']}")
            self.assertEqual(edgesnap.pixels(r["offset"], 8), bias, f"bias {bias}: would move by {edgesnap.pixels(r['offset'], 6)}")
        self.assertIsNone(edgesnap.measure(np.full((H, W, 3), 128, np.uint8), self.disc(90)), "a flat photograph has no edge to find")
        self.assertEqual(edgesnap.pixels(9.5, 8), 8, "the move is capped")

    def test_snap_moves_the_mask_onto_the_photograph(self):
        self.paint()
        masks.run(args(self.ply, method="vision"), self.pj)
        for f, m in load_masks(self.pj).items():
            iou = (m & self.disc(90)).sum() / (m | self.disc(90)).sum()
            self.assertGreater(iou, 0.97, f"{f}: without --snap-edge the mask is Vision's (IoU {iou:.3f} with r=90)")
        masks.run(args(self.ply, method="vision", snap_edge=True, snap_fallback_px=3, snap_max_px=8), self.pj)
        truth = self.disc(self.PHOTO_R)
        for f, m in load_masks(self.pj).items():
            iou = (m & truth).sum() / (m | truth).sum()
            self.assertGreater(iou, 0.97, f"{f}: IoU {iou:.3f} with the photograph's disc (r={self.PHOTO_R})")
        met = self.pj.stage("masks")["metrics"]
        self.assertTrue(met["snap_edge"])
        self.assertEqual(met["snap_measured_views"], N_CAM)
        self.assertEqual(met["snap_fell_back_views"], 0)
        self.assertAlmostEqual(met["snap_offset_px"]["median"], 3.5, delta=0.6)
        self.assertEqual(met["snap_moved_px"]["median"], 4)
        self.assertLessEqual(abs(met["snap_residual_px"]["median"]), 1.0)
        chk = {c["name"]: c for c in self.pj.stage("masks")["checks"]}
        self.assertTrue(chk["masks_sit_on_the_photographs_edge"]["ok"], chk["masks_sit_on_the_photographs_edge"]["value"])
        self.assertTrue(os.path.exists(self.pj.path("masks_snap.json")))

    def test_a_view_with_no_edge_falls_back_and_is_counted(self):
        """Flat grey photographs (the default build): nothing to measure, every view falls back to
        --snap-fallback-px, and the check says so rather than passing."""
        masks.run(args(self.ply, method="vision", snap_edge=True, snap_fallback_px=2, snap_max_px=8), self.pj)
        met = self.pj.stage("masks")["metrics"]
        self.assertEqual(met["snap_measured_views"], 0)
        self.assertEqual(met["snap_fell_back_views"], N_CAM)
        truth = self.disc(90 - 2)
        for f, m in load_masks(self.pj).items():
            iou = (m & truth).sum() / (m | truth).sum()
            self.assertGreater(iou, 0.97, f"{f}: IoU {iou:.3f} with r=88 (Vision's 90 eroded by the fallback 2)")
        chk = {c["name"]: c for c in self.pj.stage("masks")["checks"]}
        self.assertFalse(chk["masks_sit_on_the_photographs_edge"]["ok"])
        self.assertIn("fell back", chk["masks_sit_on_the_photographs_edge"]["value"])

    def test_off_by_default_and_the_rim_stays_feathered(self):
        import cv2
        self.paint()
        masks.run(args(self.ply, method="vision", snap_edge=True, feather_px=1.0), self.pj)
        m = cv2.imread(os.path.join(self.pj.dataset_dir, "masks", "L", "cap000.png"), cv2.IMREAD_GRAYSCALE)
        self.assertGreater(((m > 0) & (m < 255)).mean(), 0, "snapping must not lose the anti-aliased rim")
        masks.run(args(self.ply, method="vision"), self.pj)
        self.assertFalse(self.pj.stage("masks")["metrics"]["snap_edge"])
        self.assertNotIn("snap_measured_views", self.pj.stage("masks")["metrics"])


if __name__ == "__main__":
    unittest.main()
