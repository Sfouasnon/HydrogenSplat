"""hs views --inside-masks / --outside-masks: score a layer only where it is meant to exist.

A subject layer trained with --alpha-mode transparent is empty outside its silhouette on
purpose. Scored over the whole crop, every empty pixel reads as an error — on coins that is
what put the alpha-matched model at 11.10 dB against 19.41. These tests build exactly that
situation from synthetic images and check the region fixes it without changing anything when
no region is given.
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from hs.stages.views import compare  # noqa: E402

H, W = 480, 640
UV = (W / 2, H / 2)
HALF = 200


def scene(seed=3):
    """Textured grey image: phase correlation needs structure to lock on to."""
    rng = np.random.default_rng(seed)
    base = rng.normal(128, 40, (H // 8, W // 8)).astype(np.float32)
    import cv2
    img = cv2.resize(base, (W, H), interpolation=cv2.INTER_CUBIC)
    return np.clip(img + rng.normal(0, 3, (H, W)), 0, 255).astype(np.float32)


def disc(radius=110):
    yy, xx = np.mgrid[:H, :W]
    return np.hypot(xx - UV[0], yy - UV[1]) <= radius


class Region(unittest.TestCase):
    def setUp(self):
        self.src = scene()
        self.sil = disc()

    def test_subject_layer_is_charged_for_the_room_without_a_region(self):
        ren = np.where(self.sil, self.src, 0.0).astype(np.float32)   # empty outside, like arm B
        whole = compare(self.src, ren, UV, HALF, 4.0)
        inside = compare(self.src, ren, UV, HALF, 4.0, region=self.sil)
        self.assertLess(whole["psnr_db"], 15.0, "whole-crop PSNR should punish the empty room")
        self.assertGreater(inside["psnr_db"], 60.0, "inside the silhouette the render is exact")
        self.assertEqual(inside["displaced_fraction"], 0.0)
        self.assertGreater(inside["correlation"], 0.999)

    def test_patches_outside_the_region_are_not_scored(self):
        ren = self.src.copy()
        whole = compare(self.src, ren, UV, HALF, 4.0)
        inside = compare(self.src, ren, UV, HALF, 4.0, region=self.sil)
        self.assertLess(inside["patches"], whole["patches"])
        self.assertGreater(inside["patches"], 0)
        self.assertAlmostEqual(inside["region_fraction"], self.sil[40:440, 120:520].mean(), places=2)

    def test_a_displaced_subject_still_reads_as_displaced(self):
        # the region hides the empty room, not real errors: shift the subject 8 px
        import cv2
        moved = cv2.warpAffine(self.src, np.float32([[1, 0, 8], [0, 1, 0]]), (W, H))
        ren = np.where(self.sil, moved, 0.0).astype(np.float32)
        inside = compare(self.src, ren, UV, HALF, 4.0, region=self.sil)
        self.assertGreater(inside["displaced_fraction"], 0.5)
        self.assertGreater(inside["bulk_shift_px"], 6.0)

    def test_background_layer_scored_outside(self):
        ren = np.where(self.sil, 0.0, self.src).astype(np.float32)   # empty where the subject was
        outside = compare(self.src, ren, UV, HALF, 4.0, region=~self.sil)
        self.assertGreater(outside["psnr_db"], 60.0)
        self.assertEqual(outside["displaced_fraction"], 0.0)

    def test_no_region_is_unchanged(self):
        ren = self.src + 5.0
        r = compare(self.src, ren, UV, HALF, 4.0)
        self.assertIsNone(r["region_fraction"])
        self.assertAlmostEqual(r["psnr_db"], round(10 * np.log10(255.0 ** 2 / 25.0), 2), places=2)

    def test_region_that_misses_the_crop_scores_nothing(self):
        far = np.zeros((H, W), bool)
        far[0:5, 0:5] = True
        self.assertIsNone(compare(self.src, self.src, UV, HALF, 4.0, region=far))

    def test_a_short_silhouette_is_charged_to_the_edge_not_the_interior(self):
        # the GreetingCard case: exact inside, black in a 4 px ring just inside the mask
        import cv2
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        short = cv2.erode(self.sil.astype(np.uint8), k).astype(bool)
        ren = np.where(short, self.src, 0.0).astype(np.float32)
        r = compare(self.src, ren, UV, HALF, 4.0, region=self.sil)
        self.assertGreater(r["psnr_interior_db"], 60.0, "the interior is exact")
        self.assertLess(r["psnr_edge_db"], 15.0, "the ring is black")
        self.assertLess(r["psnr_db"], 30.0, "and the whole-region number hides which is which")
        self.assertGreater(r["edge_error_share"], 0.99)

    def test_no_region_reports_no_split(self):
        r = compare(self.src, self.src + 5.0, UV, HALF, 4.0)
        self.assertIsNone(r["psnr_interior_db"])
        self.assertIsNone(r["edge_error_share"])


if __name__ == "__main__":
    unittest.main()
