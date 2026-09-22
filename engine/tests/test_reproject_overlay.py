"""Reproject overlay: recover a known shift between photograph and render, and tell a solve
fault (sparse points off their keypoints) from a training fault (uniform render shift)."""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(HERE)
sys.path.insert(0, ENGINE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ENGINE, "tools"))

from hs import overlay  # noqa: E402

import reproject_overlay  # noqa: E402
from test_cameras import write_rig  # noqa: E402

W, H = 1913, 1073          # write_rig's L canvas


def texture(w=W, h=H, seed=3):
    import cv2
    rng = np.random.default_rng(seed)
    base = rng.normal(128, 45, (h // 10, w // 10)).astype(np.float32)
    img = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(img, 0, 255).astype(np.uint8)


def shifted(img, dx, dy):
    import cv2
    return cv2.warpAffine(img, np.float32([[1, 0, dx], [0, 1, dy]]), (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


class Shift(unittest.TestCase):
    def test_known_5px_shift_recovered_within_0_3px(self):
        gt = texture()
        for d in ((5.0, 0.0), (0.0, -5.0), (3.0, 4.0), (5.0, 5.0)):
            dx, dy, resp = overlay.phase_shift(gt, shifted(gt, *d))
            self.assertLess(abs(dx - d[0]), 0.3, (d, dx, dy))
            self.assertLess(abs(dy - d[1]), 0.3, (d, dx, dy))
            self.assertGreater(resp, 0.5)

    def test_fractional_shift_recovered(self):
        # the integer back-shift makes whole-pixel shifts exact; the fine pass must carry the rest
        gt = texture()
        for d in ((5.4, -2.6), (0.3, 0.7), (-12.25, 6.5)):
            dx, dy, _ = overlay.phase_shift(gt, shifted(gt, *d))
            self.assertLess(abs(dx - d[0]), 0.1, (d, dx, dy))
            self.assertLess(abs(dy - d[1]), 0.1, (d, dx, dy))

    def test_quadrants_agree_on_a_uniform_shift(self):
        gt = texture()
        q = overlay.quadrant_shifts(gt, shifted(gt, 5, 0))
        self.assertEqual(set(q), {"TL", "TR", "BL", "BR"})
        for k, (dx, dy, _) in q.items():
            self.assertLess(abs(dx - 5) + abs(dy), 0.3, (k, dx, dy))

    def test_colour_and_size_mismatch_are_handled(self):
        import cv2
        gt = cv2.cvtColor(texture(), cv2.COLOR_GRAY2BGR)
        dx, dy, _ = overlay.phase_shift(gt, shifted(gt, 5, 0))
        self.assertLess(abs(dx - 5), 0.3)

    def test_diagnosis(self):
        self.assertTrue(overlay.diagnose(6.0, (5, 0), 0.1).startswith("solve"))
        self.assertTrue(overlay.diagnose(0.4, (5, 0), 0.1).startswith("uniform"))
        self.assertTrue(overlay.diagnose(0.4, (0.2, 0.1), 0.2).startswith("registered"))
        self.assertTrue(overlay.diagnose(0.4, (2, 0), 3.0).startswith("non-uniform"))
        self.assertTrue(overlay.diagnose(None, None, 0).startswith("no render"))


class Tool(unittest.TestCase):
    """engine/tools/reproject_overlay.py on a synthetic project."""

    def setUp(self):
        import cv2
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.tmp.name, "proj")
        ds = os.path.join(self.root, "train", "dataset")
        os.makedirs(os.path.join(ds, "images", "L"))
        self.rig = os.path.join(ds, "rig.npz")
        self.pts = write_rig(self.rig, n=6)
        self.G = np.load(self.rig, allow_pickle=True)
        self.v = 4                                   # cap002_L
        self.gt = cv2.cvtColor(texture(), cv2.COLOR_GRAY2BGR)
        cv2.imwrite(os.path.join(ds, "images", "L", "cap002.jpg"), self.gt, [cv2.IMWRITE_JPEG_QUALITY, 98])
        self.ds = ds

    def tearDown(self):
        self.tmp.cleanup()

    def write_sparse(self, offset=(0.0, 0.0), n=150):
        """images.txt / points3D.txt: cap002 observes n points at their exact projection + offset."""
        sp = os.path.join(self.ds, "sparse")
        os.makedirs(sp, exist_ok=True)
        uv, z = overlay.project(self.G["K"][self.v], self.G["R"][self.v], self.G["t"][self.v], self.pts[:n])
        with open(os.path.join(sp, "points3D.txt"), "w") as f:
            f.write("# POINT3D_ID X Y Z R G B ERROR TRACK[]\n")
            for i, X in enumerate(self.pts):
                m = X / 1000.0
                f.write(f"{i + 1} {m[0]} {m[1]} {m[2]} 128 128 128 0.5 5 {i}\n")
        with open(os.path.join(sp, "images.txt"), "w") as f:
            f.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n")
            f.write("5 1 0 0 0 0 0 0 1 L/cap001.jpg\n10 10 -1\n")
            f.write("6 1 0 0 0 0 0 0 1 L/cap002.jpg\n")
            f.write(" ".join(f"{u + offset[0]} {v + offset[1]} {i + 1}" for i, (u, v) in enumerate(uv)) + " 3 3 -1\n")

    def keep_views_frame(self, img):
        """What `hs views --keep-frames` leaves: views/views.json + views/frames/frame_NNNN.png."""
        import cv2
        os.makedirs(os.path.join(self.root, "views", "frames"))
        frames = [{"c2w": reproject_overlay.c2w(self.G, v).tolist()} for v in (0, self.v)]
        json.dump({"width": W, "height": H, "frames": frames}, open(os.path.join(self.root, "views", "views.json"), "w"))
        for i in range(2):
            cv2.imwrite(os.path.join(self.root, "views", "frames", f"frame_{i:04d}.png"), img if i == 1 else img * 0)

    def run_tool(self, *extra):
        return reproject_overlay.main(["-p", self.root, "--capture", "cap002"] + list(extra))

    def test_uniform_shift_with_good_sparse_points(self):
        self.write_sparse()
        self.keep_views_frame(shifted(self.gt, 5, 0))
        rep = self.run_tool()
        out = os.path.join(self.root, "views", "overlay_cap002_L")
        for f in ("overlay_gt.jpg", "overlay_render.jpg", "report.json"):
            self.assertTrue(os.path.exists(os.path.join(out, f)), f)
        self.assertEqual(rep["points_source"], "sparse_track")
        self.assertEqual(rep["n_points"], 150, "only the points this image observes")
        self.assertLess(rep["sparse_residual_px"]["median"], 0.01)
        self.assertLess(abs(rep["shift_px"][0] - 5) + abs(rep["shift_px"][1]), 0.3)
        self.assertEqual(rep["render_source"], "views/frames")
        self.assertTrue(rep["diagnosis"].startswith("uniform"), rep["diagnosis"])
        saved = json.load(open(os.path.join(out, "report.json")))
        for k in ("n_points", "shift_px", "quadrant_shifts", "response"):
            self.assertIn(k, saved)

    def test_misplaced_sparse_points_are_a_solve_fault(self):
        self.write_sparse(offset=(6.0, 0.0))
        self.keep_views_frame(self.gt)
        rep = self.run_tool()
        self.assertGreater(rep["sparse_residual_px"]["median"], 5.9)
        self.assertTrue(rep["diagnosis"].startswith("solve"), rep["diagnosis"])

    def test_no_sparse_no_render_uses_rig_points(self):
        rep = self.run_tool("--out", os.path.join(self.tmp.name, "o"))
        self.assertEqual(rep["points_source"], "rig_pts")
        self.assertEqual(rep["n_points"], len(self.pts), "every cloud point is in front of and inside this camera")
        self.assertNotIn("shift_px", rep)
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "o", "overlay_render.jpg")))

    def test_fresh_render_with_ply(self):
        os.environ["HS_PYTHON"] = sys.executable     # the fake renderer's interpreter
        ply = os.path.join(self.tmp.name, "m.ply")
        open(ply, "w").write("ply\nformat ascii 1.0\nelement vertex 3\nend_header\n")
        rep = self.run_tool("--ply", ply, "--render-bin", os.path.join(HERE, "fakebin", "brush-path-render"),
                            "--out", os.path.join(self.tmp.name, "f"))
        self.assertEqual(rep["render_source"], "brush-path-render")
        path = json.load(open(os.path.join(self.tmp.name, "f", "pose.json")))
        self.assertEqual((path["width"], path["height"], len(path["frames"])), (W, H, 1))
        self.assertIn("shift_px", rep)


class ViewsOverlay(unittest.TestCase):
    """hs views --captures holdout --overlay, end to end against the stand-in renderer."""

    def test_views_scores_the_holdouts_and_writes_overlays(self):
        import cv2
        from argparse import Namespace
        from hs import coverage, events
        from hs.project import Project
        from hs.stages import views
        from hs.stages.cameras import holdout
        os.environ["HS_PYTHON"] = sys.executable
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "proj")
            pj = Project(root, create=True)
            ds = pj.dataset_dir
            os.makedirs(os.path.join(ds, "images", "L"))
            write_rig(pj.rig_npz, n=8)
            coverage.write(pj.rig_npz, pj.path("solve", "coverage.json"))
            for i in range(8):
                cv2.imwrite(os.path.join(ds, "images", "L", f"cap{i:03d}.jpg"), cv2.cvtColor(texture(seed=i), cv2.COLOR_GRAY2BGR))
            ply = os.path.join(tmp, "m.ply")
            open(ply, "w").write("ply\nformat ascii 1.0\nelement vertex 3\nend_header\n")
            for s in ("ingest", "select", "solve", "train"):
                pj.m["stages"][s]["status"] = "done"
            pj.save()
            events.set_sink(None)
            h = holdout(Namespace(holdout=3, method="fps", seed=0, write=True), root)
            a = Namespace(captures="holdout", ply=ply, name="views", eye="L", subject_mm=None, displaced_px=4.0,
                          patch_step=32, max_displaced_fraction=0.1, keep_frames=False, overlay=True,
                          mask_region=None, edge_px=8, render_bin=os.path.join(HERE, "fakebin", "brush-path-render"))
            views.run(a, Project(root))
            rep = json.load(open(pj.path("views", "views_report.json")))
            self.assertEqual(sorted(r["capture"] for r in rep["views"]), h["captures"])
            for r in rep["views"]:
                self.assertIn("diagnosis", r["overlay"])
                self.assertTrue(os.path.exists(pj.path("views", "views_overlay", r["view"], "overlay_render.jpg")))


if __name__ == "__main__":
    unittest.main()
