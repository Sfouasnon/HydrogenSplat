"""hs.splatweights: the renderer's forward weights, on a coarse grid, in numpy.

Both hs split and hs prune --score are sums over these weights, so what is tested here is that
they mean what a renderer means by them: per cell, the weights and the remaining transmittance
account for exactly one unit of light; nearer splats occlude farther ones; a splat's footprint
has the size EWA projection says it has; and a splat smaller than a cell contributes its area,
not all-or-nothing depending on where its centre falls.

HS_TIMING=1 also runs the size the brief asks about (200k splats x 30 views at 1920x1080, cell 8)
and prints the time.
"""
import os
import sys
import tempfile
import time
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import splat_scene as ss  # noqa: E402
from hs import splatweights as sw  # noqa: E402
from hs.stages.merge import read_header  # noqa: E402

K = np.array([[ss.FX, 0, ss.W / 2], [0, ss.FX, ss.H / 2], [0, 0, 1.0]])
R0, T0 = np.eye(3), np.array([0.0, 0.0, 400.0])      # camera 400 mm in front of the origin, looking +z


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def splats(self, rows, name="m.ply"):
        p = os.path.join(self.tmp.name, name)
        ss.write_ply(p, rows)
        return sw.Splats(p)

    def one(self, xyz, scale, opacity, rgb=(0.5, 0.5, 0.5)):
        return ss.splat_rows(np.array([xyz], float), np.array([scale], float), opacity, np.array([rgb], float))


class Conservation(Tmp):
    def test_weights_and_transmittance_account_for_all_the_light(self):
        rows, _lab = ss.card_and_room()
        sp = self.splats(rows)
        for cell in (1, 4, 8):
            for R, t in ss.cameras()[:4]:
                vw = sw.view_weights(sp, K, R, t, (ss.W, ss.H), cell=cell, max_cells=1 << 16)
                tot = np.bincount(vw.cell, weights=vw.w, minlength=vw.T.size)
                self.assertLessEqual(tot.max(), 1.0 + 1e-6, f"cell {cell}: a cell received more than all the light")
                both = tot + vw.T.ravel()
                # the dropped tail (w < MIN_WEIGHT, and whatever lies past T < 1e-4) is all that is missing
                self.assertLessEqual(both.max(), 1.0 + 1e-6)
                self.assertGreater(both.min(), 0.995)
                self.assertTrue(np.all(np.diff(vw.cell) >= 0), "triples must come grouped by cell")

    def test_nearer_splat_occludes(self):
        near = self.one([0, 0, -50], [5, 5, 5], 0.6, (1, 0, 0))
        far = self.one([0, 0, 50], [5, 5, 5], 0.8, (0, 0, 1))
        sp = self.splats(np.concatenate([far, near]))     # file order must not matter
        vw = sw.view_weights(sp, K, R0, T0, (ss.W, ss.H), cell=1)
        c = (ss.H // 2) * ss.W + ss.W // 2
        at = {int(s): float(w) for s, w, cc in zip(vw.splat, vw.w, vw.cell) if cc == c}
        self.assertAlmostEqual(at[1], 0.6, delta=0.02)                  # near: alpha * 1
        self.assertAlmostEqual(at[0], 0.8 * (1 - at[1]), delta=0.02)   # far: alpha * T
        self.assertAlmostEqual(float(vw.T.ravel()[c]), (1 - at[1]) * (1 - 0.8), delta=0.02)
        r, g, b = vw.rgb.reshape(-1, 3)[c]
        self.assertGreater(r, b)
        self.assertAlmostEqual(g, 0.0, places=5)


class Footprint(Tmp):
    def test_ewa_footprint_has_the_projected_size(self):
        # isotropic 4 mm at 400 mm under fx 300 -> sigma 3 px, plus the 0.3 px^2 dilation
        sp = self.splats(self.one([0, 0, 0], [4, 4, 4], 0.5))
        vw = sw.view_weights(sp, K, R0, T0, (ss.W, ss.H), cell=1)
        yy, xx = np.divmod(vw.cell.astype(float), ss.W)
        w = vw.w.astype(float)
        mx = (w * xx).sum() / w.sum()
        var = (w * (xx - mx) ** 2).sum() / w.sum()
        self.assertAlmostEqual(mx + 0.5, ss.W / 2, delta=0.05)          # pixel centres sit at i + 0.5
        self.assertAlmostEqual(var, 9.0 + 0.3, delta=0.4)
        # the centre sits on a pixel corner, so the brightest pixel is half a pixel off in x and y
        self.assertAlmostEqual(float(w.max()), 0.5 * np.exp(-0.5 * 0.5 / 9.3), delta=0.005)

    def test_rotation_turns_the_footprint(self):
        rows = self.one([0, 0, 0], [8, 1, 1], 0.5)
        col = {p: i for i, p in enumerate(ss.PROPS)}
        c45 = np.cos(np.pi / 8)
        rows[0, col["rot_0"]:col["rot_3"] + 1] = [c45, 0, 0, np.sin(np.pi / 8)]   # 45 deg about z
        sp = self.splats(rows)
        vw = sw.view_weights(sp, K, R0, T0, (ss.W, ss.H), cell=1)
        yy, xx = np.divmod(vw.cell.astype(float), ss.W)
        w = vw.w.astype(float)
        cov = np.cov(np.stack([xx, yy]), aweights=w)
        self.assertGreater(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]), 0.9, "the long axis should lie on the diagonal")

    def test_a_sub_cell_splat_contributes_its_area_wherever_it_lands(self):
        # 1 mm at 400 mm = 0.75 px: at cell 8 the centre-sampled Gaussian would hit or miss
        totals = []
        for dx in np.linspace(0, 8, 9) * 400 / ss.FX:           # slide the centre across one cell
            sp = self.splats(self.one([dx, 0, 0], [1, 1, 1], 0.9))
            fine = sw.view_weights(sp, K, R0, T0, (ss.W, ss.H), cell=1).w.sum()
            coarse = sw.view_weights(sp, K, R0, T0, (ss.W, ss.H), cell=8).w.sum()
            totals.append(coarse * 64 / fine)
        self.assertLess(max(totals) / min(totals), 1.05, f"coverage swings with sub-cell position: {totals}")
        self.assertAlmostEqual(float(np.median(totals)), 1.0, delta=0.05)

    def test_oversized_footprints_are_capped_and_counted(self):
        sp = self.splats(self.one([0, 0, 0], [60, 60, 60], 0.9))
        vw = sw.view_weights(sp, K, R0, T0, (ss.W, ss.H), cell=4, max_cells=64)
        self.assertEqual(vw.clipped, 1)
        self.assertLessEqual(len(np.unique(vw.cell)), 64)

    def test_live_splats_only_and_nothing_behind_the_camera(self):
        rows = np.concatenate([self.one([0, 0, 0], [5, 5, 5], 0.04), self.one([0, 0, -600], [5, 5, 5], 0.9)])
        vw = sw.view_weights(self.splats(rows), K, R0, T0, (ss.W, ss.H), cell=4)
        self.assertEqual(len(vw.w), 0)
        self.assertTrue(np.all(vw.T == 1.0))


class Ply(Tmp):
    def test_subset_and_extra_property_round_trip(self):
        rows, _ = ss.card_and_room()
        sp = self.splats(rows)
        out = os.path.join(self.tmp.name, "lab.ply")
        p = np.linspace(0, 1, sp.n)
        sw.write_ply(out, sp, extra=("subject_p", p))
        _h, n, props, _o = read_header(out)
        self.assertEqual(n, sp.n)
        self.assertEqual([x for x, _t in props][-1], "subject_p")
        back = sw.Splats(out)
        np.testing.assert_allclose(back.arr["subject_p"], p.astype("<f4"))
        np.testing.assert_array_equal(back.arr["x"], sp.arr["x"])
        sub = os.path.join(self.tmp.name, "sub.ply")
        sw.write_ply(sub, sp, keep=p > 0.5)
        self.assertEqual(read_header(sub)[1], int((p > 0.5).sum()))
        self.assertEqual(open(sub, "rb").read()[:200].count(b"subject_p"), 0)

    def test_view_keys_follow_train_exclude(self):
        self.assertEqual(sw.view_key("cap064_L"), "L/cap064")
        self.assertEqual(sw.view_key("cap064_R"), "R/cap064")
        self.assertEqual(sw.view_key("GA_L"), "L/GA")
        self.assertEqual(sw.training_views(["cap000_L", "cap000_R", "cap001_L"], {"R/cap000"}), [0, 2])


@unittest.skipUnless(os.environ.get("HS_TIMING"), "set HS_TIMING=1 for the 200k x 30 view timing")
class Timing(Tmp):
    def test_200k_splats_30_views_1080p_cell8(self):
        rng = np.random.default_rng(0)
        n = 200000
        xyz = rng.normal(0, 1, (n, 3)) * [150, 100, 150]
        sc = np.exp(rng.normal(np.log(3.0), 0.7, (n, 3)))
        rows = ss.splat_rows(xyz, sc, rng.uniform(0.05, 1, n), rng.uniform(0, 1, (n, 3)))
        rows[:, -4:] = rng.normal(size=(n, 4))
        sp = self.splats(rows)
        Kb = np.array([[1500, 0, 960], [0, 1500, 540], [0, 0, 1.0]])
        G = {"K": [], "R": [], "t": [], "wh": []}
        for i in range(30):
            az = np.radians(-60 + 120 * i / 29)
            C = 700 * np.array([np.sin(az), 0.1, -np.cos(az)])
            R = ss.look_at(C)
            G["K"].append(Kb); G["R"].append(R); G["t"].append(-R @ C); G["wh"].append((1920, 1080))
        from hs.stages.split import run_views
        tot = [0]

        def per_view(v, vw):
            tot[0] += len(vw.w)
        t0 = time.time()
        run_views("timing", sp, G, list(range(30)), 8, sw.MAX_CELLS_PER_SPLAT, per_view)
        dt = time.time() - t0
        sys.stderr.write(f"\n[timing] 200k splats x 30 views 1920x1080 cell 8: {dt:.1f} s, {tot[0]:,} triples\n")
        self.assertLess(dt, 180.0)


if __name__ == "__main__":
    unittest.main()
