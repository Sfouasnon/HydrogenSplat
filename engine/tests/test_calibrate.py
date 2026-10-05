"""hs calibrate (stages/calibrate.py) and the lens profile store (hs/lens.py).

Synthetic throughout: the board picture the stage writes goes back through the detector and its
corners land where the whole-pixel squares put them; a camera with a known K and k1 k2 p1 p2
photographs the generated board at 25 poses (one remap from the board image through the exact
homography and the lens model, so the only approximation is the interpolation) and the fit gets
fx back within 1 % and k1 within 20 %; the store saves and finds by key; `hs solve --lens none`
adds nothing to monocolmap's argv.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from hs import board as B, events, lens  # noqa: E402
from hs.stages import calibrate as C, solve  # noqa: E402
import board_synth as S  # noqa: E402

SPEC = B.parse_spec("7,5,30,22")


def render_view(gen, M, spec, K, dist, R, t, size, noise=1.5, seed=0, background=170):
    """The board (image `gen`, board-mm -> gen-px homography M) seen by a camera with K and
    OpenCV distortion `dist` at pose (R, t), as one remap: distorted pixel -> ideal pixel
    (undistortPoints) -> board plane (the pose's homography, inverted) -> gen pixel."""
    import cv2
    w, h = size
    Hv = K @ np.column_stack([R[:, 0], R[:, 1], t])          # board (x, y, 0) mm -> ideal pixels
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    pts = np.stack([uu.ravel(), vv.ravel()], axis=1).reshape(-1, 1, 2)
    ideal = cv2.undistortPoints(pts, K, np.asarray(dist, np.float64), P=K).reshape(-1, 2)
    T = M @ np.linalg.inv(Hv)                                 # ideal px -> gen px
    q = np.hstack([ideal, np.ones((len(ideal), 1))]) @ T.T
    mapx = (q[:, 0] / q[:, 2]).reshape(h, w).astype(np.float32)
    mapy = (q[:, 1] / q[:, 2]).reshape(h, w).astype(np.float32)
    img = cv2.remap(gen.astype(np.float32), mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                    borderValue=float(background))
    rng = np.random.default_rng(seed)
    return np.clip(np.round(img + rng.normal(0, noise, img.shape)), 0, 255).astype(np.uint8)


def poses(spec, n=25, seed=3):
    """Board poses that walk the board over the whole frame with tilts of up to ~35 degrees."""
    rng = np.random.default_rng(seed)
    bw, bh = spec.sx * spec.square_mm, spec.sy * spec.square_mm
    out = []
    for i in range(n):
        # a grid walk over the frame (5 x 5) plus jitter, so the corners of the frame get corners
        gx, gy = (i % 5) / 4.0 - 0.5, (i // 5) / 4.0 - 0.5
        yaw, pitch, roll = rng.uniform(-35, 35), rng.uniform(-35, 35), rng.uniform(-25, 25)
        R = S.rot([0, 0, 1], roll) @ S.rot([1, 0, 0], pitch) @ S.rot([0, 1, 0], yaw)
        z = rng.uniform(650, 1000)
        # the frame at depth z spans +-0.58 z across and +-0.33 z down for f 1100 / 1280x720
        centre = np.array([gx * 2 * 0.58 * z * 0.85, gy * 2 * 0.33 * z * 0.85, z])
        t = centre - R @ np.array([bw / 2, bh / 2, 0.0])
        out.append((R, t))
    return out


class BoardPicture(unittest.TestCase):
    def test_print_board_round_trips_through_detection(self):
        img, info = C.render_board(SPEC, 300 / 25.4)
        ids, xy = B.Detector(SPEC).detect(img)
        self.assertEqual(len(ids), (SPEC.sx - 1) * (SPEC.sy - 1))
        s, (ox, oy) = info["square_px"], info["origin"]
        # a corner sits between two pixels: in OpenCV's pixel-centre coordinates that is k*s - 0.5
        want = np.array([[(k % (SPEC.sx - 1) + 1) * s + ox - 0.5, (k // (SPEC.sx - 1) + 1) * s + oy - 0.5] for k in ids], float)
        self.assertLess(np.abs(xy - want).max(), 0.3)
        self.assertAlmostEqual(info["square_px"] / info["px_per_mm"], SPEC.square_mm, places=9)   # exact at the file's dpi

    def test_png_carries_dpi_and_pdf_has_two_pages(self):
        import cv2
        img, info = C.render_board(SPEC, 300 / 25.4)
        with tempfile.TemporaryDirectory() as d:
            png = os.path.join(d, "b.png")
            C.write_png(png, img, info["px_per_mm"] * 25.4)
            raw = open(png, "rb").read()
            self.assertIn(b"pHYs", raw[:64])
            self.assertEqual(cv2.imread(png, 0).shape, img.shape)
            pdf = os.path.join(d, "b.pdf")
            self.assertEqual(C.write_pdf(pdf, img, info["px_per_mm"], spec=SPEC), ["Letter", "A4"])
            raw = open(pdf, "rb").read()
            self.assertTrue(raw.startswith(b"%PDF-1.4"))
            self.assertEqual(raw.count(b"/Type /Page "), 2)
            self.assertTrue(raw.rstrip().endswith(b"%%EOF"))
            # a 7x5 board of 60 mm squares is 440 mm wide: no page takes it, and the hint says what would fit
            big = B.parse_spec("7,5,60,45")
            img2, info2 = C.render_board(big, 300 / 25.4)
            with self.assertRaises(events.StageError) as e:
                C.write_pdf(os.path.join(d, "big.pdf"), img2, info2["px_per_mm"], spec=big)
            self.assertIn("--square-mm 37", e.exception.hint)

    def test_screen_board_is_sized_for_the_screen(self):
        ppmm = C.screen_px_per_mm(15.0, 1728, 1117)
        self.assertAlmostEqual(ppmm * 25.4, 137.17, places=1)               # a 15" 1728x1117 screen is 137 ppi
        spec = B.parse_spec("7,5,35,26")
        img, info = C.render_screen(spec, 15.0, 1728, 1117)
        self.assertEqual(img.shape, (1117, 1728))
        self.assertEqual(info["square_px"], int(round(35 * ppmm)))
        self.assertTrue(info["fits"])
        ids, _ = B.Detector(spec).detect(img)
        self.assertEqual(len(ids), 24)
        # too big for the screen: shrunk, and said so
        img, info = C.render_screen(B.parse_spec("7,5,80,60"), 15.0, 1728, 1117)
        self.assertFalse(info["fits"])
        self.assertLess(info["square_mm_actual"], 80)


SIZE = (1280, 720)
K_TRUE = np.array([[1100.0, 0, 648.0], [0, 1100.0, 352.0], [0, 0, 1.0]])
DIST_TRUE = np.array([-0.22, 0.06, 0.0008, -0.0005, 0.0])
_FRAMES = []


def synthetic_frames():
    """25 views of the board through K_TRUE / DIST_TRUE, rendered once per test run."""
    if not _FRAMES:
        gen, M = S.generated(SPEC, px_per_mm=6.0)
        _FRAMES.extend((f"f{i:03d}", render_view(gen, M, SPEC, K_TRUE, DIST_TRUE, R, t, SIZE, seed=i))
                       for i, (R, t) in enumerate(poses(SPEC)))
    return _FRAMES


class Fit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.size, cls.K, cls.dist = SIZE, K_TRUE, DIST_TRUE
        cls.frames = synthetic_frames()

    def test_recovers_fx_and_k1(self):
        res = C.calibrate(self.frames, SPEC, min_corners=12)
        self.assertGreaterEqual(res["frames_used"], 15, [r["corners"] for r in res["frames"]])
        self.assertLess(abs(res["K"][0, 0] / self.K[0, 0] - 1), 0.01, res["K"])
        self.assertLess(abs(res["K"][1, 1] / self.K[1, 1] - 1), 0.01, res["K"])
        self.assertLess(abs(res["dist"][0] / self.dist[0] - 1), 0.20, res["dist"])
        self.assertLess(res["rms_px"], 0.5)
        self.assertGreaterEqual(res["coverage"]["share"], 0.7, res["coverage"])
        self.assertEqual(res["image_size"], list(self.size))

    def test_blurred_frame_is_dropped_not_fitted(self):
        import cv2
        name, im = self.frames[3]
        smeared = cv2.blur(im, (25, 1))                      # a pan: corners still found, badly
        frames = list(self.frames)
        frames[3] = ("smeared", smeared)
        res = C.calibrate(frames, SPEC, min_corners=12)
        row = next(r for r in res["frames"] if r["name"] == "smeared")
        if row["corners"] >= 12:                             # found: then it must have been dropped
            self.assertFalse(row["used"], row)
        self.assertLess(abs(res["K"][0, 0] / self.K[0, 0] - 1), 0.01)

    def test_too_few_frames_is_a_value_error(self):
        with self.assertRaises(ValueError):
            C.calibrate(self.frames[:2], SPEC, min_corners=12)

    def test_coverage_share_counts_cells(self):
        share, seen = C.coverage_share([np.array([[1.0, 1.0], [1279.0, 719.0]])], (1280, 720))
        self.assertEqual(seen, {(0, 0), (5, 3)})
        self.assertAlmostEqual(share, 2 / 24)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs ffmpeg")
class ClipToProfile(unittest.TestCase):
    """The whole door: a clip (each pose held for 3 frames, tagged like an iPhone) -> every 3rd
    frame decoded -> the profile in the store under the camera's key."""

    def test_clip_becomes_a_stored_profile(self):
        import cv2
        import subprocess
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"HS_LENSES": os.path.join(d, "lenses")}):
            fr = os.path.join(d, "fr")
            os.makedirs(fr)
            for i, (_n, im) in enumerate(synthetic_frames()):
                for k in range(3):
                    cv2.imwrite(os.path.join(fr, f"{i * 3 + k:05d}.png"), im)
            clip = os.path.join(d, "board.mov")
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", "30", "-i", os.path.join(fr, "%05d.png"),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
                            "-metadata", "com.apple.quicktime.make=Apple", "-metadata", "com.apple.quicktime.model=iPhone 15 Pro",
                            "-movflags", "use_metadata_tags", clip], check=True)
            a = argparse.Namespace(squares="7x5", square_mm=30, marker_mm=22, dict="DICT_5X5_100", board_image=None,
                                   board_pdf=None, board_screen=None, screen_in=None, screen_px=None, clip=clip, frames=None,
                                   every=3, max_frames=60, min_corners=12, make=None, model=None, lens_name=None,
                                   keep_frames=False, out=os.path.join(d, "p.json"), ffmpeg="ffmpeg", ffprobe="ffprobe")
            prof = C.run(a, None)
            self.assertEqual(prof["key"], "apple_iphone-15-pro_default_1280x720")
            self.assertEqual(prof["camera"]["model"], "iPhone 15 Pro")
            self.assertGreaterEqual(prof["frames_used"], 15)
            self.assertLess(abs(prof["K"][0][0] / K_TRUE[0, 0] - 1), 0.01)
            self.assertLess(abs(prof["dist"][0] / DIST_TRUE[0] - 1), 0.20)
            found = lens.find({"make": "Apple", "model": "iPhone 15 Pro", "width": 1280, "height": 720})
            self.assertIsNotNone(found)
            self.assertTrue(os.path.exists(os.path.join(d, "p.json")))


class Store(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.env = mock.patch.dict(os.environ, {"HS_LENSES": self.d})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        shutil.rmtree(self.d, ignore_errors=True)

    def test_key_is_safe_and_sized(self):
        meta = {"make": "Apple", "model": "iPhone 15 Pro", "lens": "iPhone 15 Pro back camera 6.765mm f/1.78",
                "width": 3840, "height": 2160}
        self.assertEqual(lens.key_for(meta), "apple_iphone-15-pro_iphone-15-pro-back-camera-6-765mm-f-1-78_3840x2160")
        self.assertEqual(lens.key_for({"width": 1920, "height": 1080}), "unknown_unknown_default_1920x1080")

    def test_save_then_find_by_key(self):
        prof = {"camera": {"make": "RED", "model": "KOMODO-X", "lens": "Sigma 24mm"}, "image_size": [3840, 2160],
                "K": [[3000, 0, 1920], [0, 3000, 1080], [0, 0, 1]], "dist": [-0.1, 0.02, 0, 0, 0], "rms_px": 0.3}
        p = lens.save(prof)
        self.assertEqual(os.path.dirname(p), self.d)
        self.assertEqual(os.path.basename(p), "red_komodo-x_sigma-24mm_3840x2160.json")
        found = lens.find({"make": "RED", "model": "KOMODO-X", "lens": "Sigma 24mm", "width": 3840, "height": 2160})
        self.assertIsNotNone(found)
        self.assertEqual(found[1]["rms_px"], 0.3)
        self.assertIn("date", found[1])
        self.assertIsNone(lens.find({"make": "RED", "model": "KOMODO-X", "lens": "Sigma 24mm", "width": 1920, "height": 1080}))
        self.assertEqual(lens.colmap_params(found[1]), "3000,3000,1920,1080,-0.1,0.02,0,0")
        self.assertEqual(len(lens.list_profiles()), 1)

    def test_meta_from_probe_reads_apple_tags(self):
        probe = {"format": {"tags": {"com.apple.quicktime.make": "Apple", "com.apple.quicktime.model": "iPhone 15 Pro",
                                     "com.apple.quicktime.software": "17.5"}},
                 "streams": [{"codec_type": "video", "width": 3840, "height": 2160}]}
        m = lens.meta_from_probe(probe)
        self.assertEqual((m["make"], m["model"], m["lens"], m["width"], m["height"]), ("Apple", "iPhone 15 Pro", None, 3840, 2160))
        self.assertEqual(lens.key_for(m), "apple_iphone-15-pro_default_3840x2160")


class SolveLens(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.env = mock.patch.dict(os.environ, {"HS_LENSES": os.path.join(self.d, "lenses")})
        self.env.start()
        self.pj = _FakeProject(self.d, {"kind": "mono", "probe": {"width": 1280, "height": 720}})

    def tearDown(self):
        self.env.stop()
        shutil.rmtree(self.d, ignore_errors=True)

    def args(self, **kw):
        base = dict(lens="auto", focal_px=None, fix_intrinsics=False, intrinsics=None)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_none_adds_nothing(self):
        self.assertEqual(solve._lens_prior(self.args(lens="none"), self.pj), ([], "none"))
        self.assertEqual(self.pj.metrics.get("lens_profile"), "none")

    def test_auto_without_a_profile_adds_nothing(self):
        argv, info = solve._lens_prior(self.args(), self.pj)
        self.assertEqual(argv, [])
        self.assertEqual(self.pj.metrics["lens_profile"], "none")

    def test_auto_finds_the_matching_profile(self):
        p = lens.save({"camera": dict(lens.UNKNOWN), "image_size": [1280, 720], "rms_px": 0.21,
                       "K": [[1100, 0, 640], [0, 1100, 360], [0, 0, 1]], "dist": [-0.2, 0.05, 0.001, -0.0005, 0.0]})
        argv, info = solve._lens_prior(self.args(), self.pj)
        self.assertEqual(argv, ["--camera-params", "1100,1100,640,360,-0.2,0.05,0.001,-0.0005"])
        self.assertEqual(info["path"], p)
        self.assertEqual(self.pj.metrics["lens_profile"]["rms_px"], 0.21)

    def test_explicit_file_must_match_the_size(self):
        p = lens.save({"camera": dict(lens.UNKNOWN), "image_size": [3840, 2160], "rms_px": 0.2,
                       "K": [[3000, 0, 1920], [0, 3000, 1080], [0, 0, 1]], "dist": [0, 0, 0, 0, 0]},
                      os.path.join(self.d, "p.json"))
        with self.assertRaises(events.StageError) as e:
            solve._lens_prior(self.args(lens=p), self.pj)
        self.assertIn("3840x2160", str(e.exception))

    def test_focal_px_beats_auto(self):
        lens.save({"camera": dict(lens.UNKNOWN), "image_size": [1280, 720], "rms_px": 0.2,
                   "K": [[1100, 0, 640], [0, 1100, 360], [0, 0, 1]], "dist": [0, 0, 0, 0, 0]})
        argv, info = solve._lens_prior(self.args(focal_px=1000.0), self.pj)
        self.assertEqual(argv, [])

    def test_solve_parser_has_lens(self):
        ap = argparse.ArgumentParser()
        solve.add_parser(ap.add_subparsers(dest="cmd"))
        self.assertEqual(ap.parse_args(["solve"]).lens, "auto")
        self.assertEqual(ap.parse_args(["solve", "--lens", "none"]).lens, "none")


class _FakeProject:
    """Just what _lens_prior touches: the manifest's source and a metric recorder."""

    def __init__(self, root, source):
        self.root = root
        self.m = {"source": source}
        self.metrics = {}

    def path(self, *parts):
        return os.path.join(self.root, *parts)

    def metric(self, stage, name, value, **extra):
        self.metrics[name] = value


if __name__ == "__main__":
    unittest.main()
