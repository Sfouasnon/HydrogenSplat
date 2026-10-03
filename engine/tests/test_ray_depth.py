"""tools/ray_depth.py: thickness and depth error of a splat model along camera rays.

The ball of test_depthmaps again (radius 100 mm, 20 cameras, LiDAR depth maps of the whole
ball). Two models of it:

  thin    40,000 opaque 3 mm splats on the ball's surface: what a good model looks like
  smoky   the same shell at a thirtieth of the opacity, over an opaque shell 40 mm further in: the
          glossy-helmet failure, most of what a ray sees sitting behind the surface

The tool has to tell them apart by thickness, and against the depth maps by depth error and by
the share of the weight behind the surface.
"""
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))

from hs import depthmaps as D, initsplats  # noqa: E402
import ray_depth  # noqa: E402
import test_depthmaps as T  # noqa: E402


class RayDepth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json
        cls.tmp = tempfile.mkdtemp(prefix="hs-rays-")
        cls.project = os.path.join(cls.tmp, "proj")
        dataset = os.path.join(cls.project, "train", "dataset")
        cams = T.cameras()
        T.write_dataset(dataset, cams)
        np.savez(os.path.join(dataset, "rig.npz"), **T.rig(cams))
        G = np.load(os.path.join(dataset, "rig.npz"), allow_pickle=True)
        names = [str(n) for n in G["names"]]
        ball = T.fibonacci_sphere(60000)
        depth = os.path.join(cls.project, "train", "depth")
        rep = D.build(np.concatenate([ball, T.table()]), G, names, depth, layer="subject",
                      masks_dir=os.path.join(dataset, "masks"))
        with open(os.path.join(depth, "depth_report.json"), "w") as f:
            json.dump(rep, f)
        shell = T.fibonacci_sphere(40000) / 1000.0
        cls.thin = initsplats.write(os.path.join(cls.tmp, "thin", "export.ply"),
                                    initsplats.splats(shell, opacity=0.99, scale=np.full(len(shell), 0.003)))
        inner = T.fibonacci_sphere(20000, radius=60.0) / 1000.0
        rows = np.concatenate([initsplats.splats(shell, opacity=0.03, scale=np.full(len(shell), 0.003)),
                               initsplats.splats(inner, opacity=0.99, scale=np.full(len(inner), 0.003))])
        cls.smoky = initsplats.write(os.path.join(cls.tmp, "smoky", "export.ply"), rows)
        views = ray_depth.pick_views(cls.project, 8)
        cls.views = views
        cls.res = {name: ray_depth.measure(p, cls.project, views)
                   for name, p in (("thin", cls.thin), ("smoky", cls.smoky))}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_views_are_spread_over_the_ones_with_a_depth_map(self):
        self.assertEqual(len(self.views), 8)
        self.assertEqual(len({k for k, _i in self.views}), 8)
        self.assertEqual(len(ray_depth.pick_views(self.project, 0)), 20)
        self.assertEqual(len(ray_depth.pick_views(self.project, 0, exclude={"L/cap003"})), 19)

    def test_a_surface_is_thin_and_where_the_scan_says(self):
        g = self.res["thin"]["groups"]["covered frontal"]
        self.assertGreater(g["rays"], 1500)
        self.assertLess(g["thickness_median_mm"], 3.0)
        self.assertLess(g["thicker_than_10mm"], 0.02)
        self.assertLess(g["rel_spread_mean"], 0.003)
        # not zero: depth is the splats' centre depth, as in Brush, and on a slope a 3 mm splat's
        # centre is a few millimetres nearer or further than the point the ray meets it
        self.assertLess(g["rel_depth_error_mean"], 0.01)
        # ...and nearer on balance, by a little under one splat radius: the nearer centres are
        # composited first and take the weight. A model whose surface is exactly right reads
        # about 0.9 sigma in front of the scan (helmet run B, 2 mm splats: -1.5 mm).
        self.assertLess(g["depth_offset_median_mm"], -1.0)
        self.assertGreater(g["depth_offset_median_mm"], -4.0)
        self.assertLess(g["weight_behind_10mm"], 0.02)
        self.assertLess(g["weight_front_5mm"], 0.1)

    def test_smoke_is_thick_and_behind_the_surface(self):
        g = self.res["smoky"]["groups"]["covered frontal"]
        self.assertGreater(g["thickness_median_mm"], 25.0)
        self.assertGreater(g["thicker_than_20mm"], 0.8)
        self.assertGreater(g["rel_spread_mean"], 0.02)
        self.assertGreater(g["rel_depth_error_mean"], 0.03)
        self.assertGreater(g["depth_offset_median_mm"], 15.0)
        self.assertGreater(g["weight_behind_20mm"], 0.4)
        self.assertLess(g["weight_front_5mm"], 0.02)
        # the limb is thick in any model: the frontal rows exist to leave it out
        thin = self.res["thin"]["groups"]
        self.assertGreater(thin["covered"]["thickness_mean_mm"] + 1e-9, thin["covered frontal"]["thickness_mean_mm"])

    def test_the_spread_is_the_standard_deviation_brush_logs(self):
        """A veil of weight a at the surface over an opaque shell d behind: std = d sqrt(a (1 - a)),
        and the mean sits (1 - a) d behind the surface. Both follow from the reported offset."""
        g = self.res["smoky"]["groups"]["covered frontal"]
        d = 40.0
        behind = g["weight_behind_20mm"]                     # 1 - a
        self.assertAlmostEqual(g["depth_offset_median_mm"], behind * d, delta=6.0)
        std = d * np.sqrt(behind * (1 - behind))
        self.assertAlmostEqual(g["rel_spread_mean"], std / (T.DIST - T.RADIUS + behind * d), delta=0.012)

    def test_table(self):
        text = ray_depth.table([self.res["thin"], self.res["smoky"]], ["thin", "smoky"])
        self.assertIn("covered frontal", text)
        self.assertIn("thickness, median mm", text)
        self.assertIn("weight > 20 mm behind", text)
        row = next(ln for ln in text.splitlines() if "thickness, median mm" in ln)
        a, b = (float(x) for x in row.split()[-2:])
        self.assertLess(a, b)


if __name__ == "__main__":
    unittest.main()
