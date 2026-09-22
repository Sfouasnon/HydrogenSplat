"""hs/board.py: ChArUco detection, DLT triangulation, pair-median scale and the board plane.

Two ways in: pure geometry (corners in 3D with a known pitch, projected into synthetic cameras
with pixel noise) and the detector path (OpenCV's own generated board image warped into each
view by the exact homography, then cv2.aruco.CharucoDetector). Both must recover the scale to
0.5 % and the plane normal to 1 degree.
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from hs import board as B  # noqa: E402
import board_synth as S  # noqa: E402

SPEC = "7,5,30,22"


class Spec(unittest.TestCase):
    def test_parse(self):
        s = B.parse_spec("7,5,40,30")
        self.assertEqual((s.sx, s.sy, s.square_mm, s.marker_mm, s.dictionary), (7, 5, 40.0, 30.0, "DICT_5X5_100"))
        self.assertEqual(B.parse_spec("7,5,40,30,4x4_50").dictionary, "DICT_4X4_50")
        self.assertEqual(B.parse_spec(" 7 , 5 , 40 , 30 , DICT_6X6_250 ").text(), "7,5,40,30,DICT_6X6_250")

    def test_refusals(self):
        for bad in ("7,5,40", "7,5,30,40", "a,5,40,30", "7,5,40,30,DICT_NOPE", "1,1,40,30",
                    "20,20,40,30,DICT_4X4_50"):      # 200 markers do not fit a 50-marker dictionary
            with self.assertRaises(ValueError, msg=bad):
                B.parse_spec(bad)

    def test_corner_ids_follow_opencv(self):
        obj = B.corner_points(B.parse_spec("7,5,40,30"))
        self.assertEqual(obj.shape, (24, 3))
        np.testing.assert_allclose(obj[0], [40, 40, 0])
        np.testing.assert_allclose(obj[5], [240, 40, 0])
        np.testing.assert_allclose(obj[6], [40, 80, 0])


class Geometry(unittest.TestCase):
    def test_dlt_is_exact_without_noise(self):
        sc = S.Scene(B.parse_spec(SPEC), n_views=3)
        X = np.array([10.0, -20.0, 35.0])
        xy = [B.project(X[None], sc.K, sc.R[v], sc.t_mm[v])[0][0] for v in range(3)]
        np.testing.assert_allclose(B.triangulate_point([sc.K] * 3, sc.R, sc.t_mm, xy), X, atol=1e-6)

    def test_scale_and_plane_from_noisy_projections(self):
        spec = B.parse_spec(SPEC)
        sc = S.Scene(spec, n_views=6, k=0.0425)          # an unscaled solve 23.5x too big
        Xw = sc.corners_world_mm()
        rng = np.random.default_rng(4)
        obs = {}
        for v in range(6):
            uv, _ = B.project(Xw, sc.K, sc.R[v], sc.t_mm[v])
            uv += rng.normal(0, 0.3, uv.shape)
            for cid, p in enumerate(uv):
                obs.setdefault(cid, []).append((v, p))
        ids, X, res = B.triangulate(obs, sc.Ks(), sc.R, sc.t_units, min_views=3)
        self.assertEqual(len(ids), 24)
        s = B.scale_from_pairs(ids, X, B.corner_points(spec))
        self.assertLess(abs(s["scale"] / sc.k - 1), 0.005, s)
        self.assertEqual(s["n_pairs"], 24 * 23 // 2)
        c, n, rms = B.fit_plane(X)
        n = B.orient_towards(n, c, sc.C_units.mean(axis=0))
        self.assertLess(B.angle_deg(n, sc.up), 1.0)
        self.assertLess(max(B.per_view_rms(res).values()), 1.0)
        # the Umeyama similarity agrees with the pair median
        s_um, _, _ = B.umeyama(B.corner_points(spec)[ids], X)
        self.assertLess(abs((1 / s_um) / s["scale"] - 1), 0.002)

    def test_one_wild_corner_does_not_move_the_median(self):
        spec = B.parse_spec(SPEC)
        obj = B.corner_points(spec)
        X = obj * 0.5
        X[7] += [30.0, 0, 0]                             # a misdetection triangulated 60 mm off
        s = B.scale_from_pairs(np.arange(24), X, obj)
        self.assertLess(abs(s["scale"] / 2.0 - 1), 0.005, s)

    def test_min_views_filters(self):
        sc = S.Scene(B.parse_spec(SPEC), n_views=3)
        obs = {0: [(0, (1.0, 2.0)), (1, (1.0, 2.0))], 1: [(0, (1.0, 2.0)), (0, (1.0, 2.0)), (1, (1.0, 2.0))]}
        ids, X, _ = B.triangulate(obs, sc.Ks(), sc.R, sc.t_mm, min_views=3)
        self.assertEqual(len(ids), 0, "two distinct views are not three, however many observations")


class DetectorPath(unittest.TestCase):
    def test_generated_board_rendered_into_views(self):
        spec = B.parse_spec(SPEC)
        sc = S.Scene(spec, n_views=5, k=3.7, noise=2.0)
        det = B.Detector(spec)
        obs = {}
        for v, img in enumerate(sc.imgs):
            ids, xy = det.detect(img)
            self.assertGreaterEqual(len(ids), 20, f"view {v}: {len(ids)} corners")
            # the detector's corners land where the true geometry puts them
            truth, _ = B.project(sc.corners_world_mm()[ids], sc.K, sc.R[v], sc.t_mm[v])
            self.assertLess(np.median(np.linalg.norm(truth - xy, axis=1)), 0.5)
            for c, p in zip(ids, xy):
                obs.setdefault(int(c), []).append((v, p))
        ids, X, res = B.triangulate(obs, sc.Ks(), sc.R, sc.t_units, min_views=3)
        s = B.scale_from_pairs(ids, X, B.corner_points(spec))
        err = abs(s["scale"] / sc.k - 1)
        self.assertLess(err, 0.005, s)
        self.assertLess(s["mad_rel"], 0.005)
        c, n, _ = B.fit_plane(X)
        n = B.orient_towards(n, c, sc.C_units.mean(axis=0))
        self.assertLess(B.angle_deg(n, sc.up), 1.0)
        self.assertLess(max(B.per_view_rms(res).values()), 0.5)

    def test_nothing_found_is_empty_not_an_error(self):
        det = B.Detector(B.parse_spec(SPEC))
        ids, xy = det.detect(np.full((200, 300, 3), 128, np.uint8))
        self.assertEqual((len(ids), xy.shape), (0, (0, 2)))

    def test_white_samples_are_white_paper(self):
        spec = B.parse_spec(SPEC)
        gen, M = S.generated(spec)
        self.assertEqual(len(B.white_cells(spec)), (spec.sx * spec.sy) // 2)
        mask = B.white_mask(spec, M, gen.shape)
        vals = gen[mask]
        # 17 cells x a 2.4 mm band around a 23.6 mm square, at 8 px/mm
        self.assertGreater(len(vals), 17 * 4 * 23.6 * 2.4 * 64 * 0.8)
        self.assertTrue((vals == 255).all(), f"{(vals != 255).sum()} of {len(vals)} samples are not white")
        img = np.repeat(gen[:, :, None], 3, axis=2)
        w, n, clipped = B.sample_white(img, mask, clip_level=256)
        np.testing.assert_allclose(w, [255, 255, 255])
        self.assertEqual(n, len(vals))
        # and the band really is the whole margin's middle: a looser inset reaches the black
        self.assertFalse((gen[B.white_mask(spec, M, gen.shape, inset=-0.3)] == 255).all())


if __name__ == "__main__":
    unittest.main()
