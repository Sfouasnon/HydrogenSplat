"""monocolmap.py --intrinsics staged / --rematch and the camera prior written into the database.

IMG_2525 (2026-09-28, iPhone 4K portrait): with the focal refined from the first pair, OPENCV's eight
intrinsics ran away on two views (fx 2796, fy 3737, k2 -1.68) and no third image registered: 2 of 267.
Held at a prior while mapping and refined once over the whole model: 216 of 267, fx 3168 / fy 3134.

Run: cd engine && ../.venv/bin/python -W ignore -m unittest tests.test_monocolmap_intrinsics
"""
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    import pycolmap
except ImportError:          # pragma: no cover
    pycolmap = None

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "hs", "monocolmap.py")


@unittest.skipIf(pycolmap is None, "needs pycolmap")
class CameraPrior(unittest.TestCase):
    def test_prior_replaces_the_extraction_guess(self):
        from hs import monocolmap
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "database.db")
            h = pycolmap.Database.open(db)
            cam = pycolmap.Camera.create_from_model_name(1, "OPENCV", 4608.0, 2160, 3840)
            cam.params = [4608.0, 4608.0, 1080.0, 1920.0, 0.3, -1.6, 0.06, -0.01]
            h.write_camera(cam)
            h.close()
            self.assertEqual(monocolmap.set_camera_prior(db, 3340.0), 1)
            h = pycolmap.Database.open(db)
            c = h.read_all_cameras()[0]
            h.close()
            np.testing.assert_allclose(c.params, [3340, 3340, 1080, 1920, 0, 0, 0, 0])
            self.assertTrue(c.has_prior_focal_length)


class Cli(unittest.TestCase):
    def test_staged_needs_a_prior(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "images", "L"))
            open(os.path.join(d, "database.db"), "w").close()
            r = subprocess.run([sys.executable, SCRIPT, "sfm", d, "--intrinsics", "staged"],
                               capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("--focal-px", r.stderr)


if __name__ == "__main__":
    unittest.main()
