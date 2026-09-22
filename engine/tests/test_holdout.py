"""Hold-out selection: choose the captures to leave out by where the cameras are, not by index.

"Every 10th from 5" left the 09-16 head's +90…+135 band without a single hold-out, so the part
of the orbit the model was worst at was never scored. The synthetic orbit here has the same
shape of problem: a dense stretch, a gap, and a sparse stretch. Index spacing spends most of
its hold-outs where the captures are dense; farthest-point sampling spreads them over space.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(HERE)
sys.path.insert(0, ENGINE)
sys.path.insert(0, HERE)

from hs import coverage, events  # noqa: E402
from hs.stages import train, views  # noqa: E402

from test_cameras import write_rig  # noqa: E402


def orbit():
    """80 captures over az −150…−30 (1.5° apart), nothing over −30…+30, 16 over +30…+150."""
    az = np.r_[np.linspace(-150, -30, 80), np.linspace(30, 150, 16)]
    C = np.stack([600 * np.sin(np.radians(az)), np.zeros_like(az), 600 * np.cos(np.radians(az))], 1)
    return az, C


class Select(unittest.TestCase):
    def test_fps_spreads_across_bands_and_interval_does_not(self):
        az, C = orbit()
        fps = coverage.select_holdout(C, 8, "fps", azimuth=az)
        itv = coverage.select_holdout(C, 8, "interval", azimuth=az)
        s_fps = coverage.holdout_stats(C, fps, azimuth=az)
        s_itv = coverage.holdout_stats(C, itv, azimuth=az)
        n_bands = len(s_fps["bands"])
        self.assertEqual(n_bands, 8, "the orbit touches eight 45° bands")
        self.assertEqual(s_fps["empty_bands"], [], "fps leaves no covered band without a hold-out")
        self.assertGreaterEqual(len(s_itv["empty_bands"]), 3, "index spacing starves the sparse side")
        self.assertEqual(sum(az[i] > 0 for i in itv), 1, "7 of 8 interval picks land on the dense side")
        self.assertGreaterEqual(sum(az[i] > 0 for i in fps), 3)
        # spread: the closest two fps hold-outs are further apart than the closest two by index
        self.assertGreater(s_fps["nn_holdout_mm"]["min"], s_itv["nn_holdout_mm"]["min"])

    def test_fps_starts_at_the_capture_farthest_from_the_centroid(self):
        az, C = orbit()
        start = int(np.argmax(np.linalg.norm(C - C.mean(0), axis=1)))
        self.assertIn(start, coverage.select_holdout(C, 3, "fps"))
        self.assertEqual(coverage.select_holdout(C, 1, "fps"), [start])

    def test_interval_is_the_old_every_nth(self):
        C = np.random.default_rng(0).normal(size=(160, 3))
        self.assertEqual(coverage.select_holdout(C, 16, "interval"), list(range(5, 160, 10)))
        self.assertEqual(coverage.select_holdout(C, 16, "interval", seed=2)[:2], [7, 17])

    def test_azimuth_one_per_band_and_fills_the_gap(self):
        az, C = orbit()
        ch = coverage.select_holdout(C, 6, "azimuth", azimuth=az)
        self.assertEqual(len(ch), 6, "a band the camera never entered is filled from what is left")
        self.assertEqual(len(set(ch)), 6)
        with self.assertRaises(ValueError):
            coverage.select_holdout(C, 4, "azimuth")          # no azimuths given

    def test_bad_counts_are_refused(self):
        _, C = orbit()
        for n in (0, len(C)):
            with self.assertRaises(ValueError):
                coverage.select_holdout(C, n, "fps")

    def test_stats_measure_out_of_sample_distance(self):
        az, C = orbit()
        s = coverage.holdout_stats(C, [0, 95], azimuth=az)
        step = float(np.linalg.norm(C[1] - C[0]))
        self.assertAlmostEqual(s["holdout_to_rest_mm"]["min"], round(step, 2), places=2)
        self.assertAlmostEqual(s["nn_holdout_mm"]["min"], round(float(np.linalg.norm(C[0] - C[95])), 2), places=2)


class HoldoutFile(unittest.TestCase):
    """`hs cameras --holdout N --write` -> solve/holdout.json -> train --exclude @holdout and
    views --captures holdout name the same captures."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.tmp.name, "proj")
        os.makedirs(os.path.join(self.root, "train", "dataset"))
        self.rig = os.path.join(self.root, "train", "dataset", "rig.npz")
        json.dump({"stages": {}}, open(os.path.join(self.root, "manifest.json"), "w"))

    def tearDown(self):
        self.tmp.cleanup()

    def hs(self, *args):
        return subprocess.run([sys.executable, "-m", "hs", "cameras", "-p", self.root] + list(args),
                              cwd=ENGINE, capture_output=True, text=True)

    def test_write_then_train_and_views_read_it(self):
        write_rig(self.rig, n=12)
        coverage.write(self.rig, os.path.join(self.root, "solve", "coverage.json"))
        r = self.hs("--holdout", "3")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertFalse(os.path.exists(os.path.join(self.root, "solve", "holdout.json")), "no --write, no file")
        self.assertIn("hold-out fps: 3 of 12", r.stderr)
        evs = [json.loads(l) for l in r.stdout.splitlines()]
        self.assertTrue(any(e.get("name") == "holdout_nn_median_mm" for e in evs))

        r = self.hs("--holdout", "3", "--write")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        h = json.load(open(os.path.join(self.root, "solve", "holdout.json")))
        self.assertEqual((h["method"], h["n"], len(h["captures"])), ("fps", 3, 3))
        self.assertEqual(h["names"], [f"cap{i:03d}" for i in h["captures"]])
        self.assertTrue(h["stereo"])
        for k in ("nn_holdout_mm", "holdout_to_rest_mm", "bands"):
            self.assertIn(k, h["stats"])

        # train: both eyes of every hold-out, mixable with a name
        ex = train.parse_exclude("@holdout, L/cap099", root=self.root)
        want = {f"{e}/{n}" for n in h["names"] for e in "LR"} | {"L/cap099"}
        self.assertEqual(ex, want)
        # views: the same captures, as indices
        cov = json.load(open(os.path.join(self.root, "solve", "coverage.json")))
        self.assertEqual(views.parse_captures("holdout", self.root, cov), h["captures"])
        self.assertEqual(views.parse_captures("5,7", self.root, cov), [5, 7])

    def test_mono_rig_holds_out_one_view_per_camera(self):
        write_rig(self.rig, n=6, stereo=False)
        r = self.hs("--holdout", "2", "--method", "interval", "--write")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)     # no coverage.json: computed from rig.npz
        h = json.load(open(os.path.join(self.root, "solve", "holdout.json")))
        self.assertFalse(h["stereo"])
        self.assertEqual(train.parse_exclude("@holdout", root=self.root), {f"L/{n}" for n in h["names"]})

    def test_missing_file_says_how_to_make_it(self):
        with self.assertRaises(events.StageError) as cm:
            train.parse_exclude("@holdout", root=self.root)
        self.assertIn("hs cameras", str(cm.exception))
        with self.assertRaises(events.StageError):
            views.parse_captures("holdout", self.root, {"captures": [], "n_captures": 0})
        with self.assertRaises(events.StageError):
            train.parse_exclude("@everything", root=self.root)

    def test_plain_exclude_is_unchanged(self):
        self.assertEqual(train.parse_exclude("L/cap064, cap069_R, GA"), {"L/cap064", "R/cap069", "L/GA"})


if __name__ == "__main__":
    unittest.main()
