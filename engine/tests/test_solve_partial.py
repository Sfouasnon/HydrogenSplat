"""hs solve: the per-image checks survive an image with no observations, unregistered captures
are reported as runs, camera-centre outliers are flagged, and the new flags parse."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hs.stages import solve  # noqa: E402


class Recorder:
    def __init__(self):
        self.metrics, self.checks = {}, []

    def metric(self, stage, name, value, **_):
        self.metrics[name] = value

    def check(self, stage, name, ok, value=None, needs_human=False, **_):
        self.checks.append((name, bool(ok), value, needs_human))
        return bool(ok)


class Runs(unittest.TestCase):
    def test_runs_compress_consecutive_captures(self):
        names = ["cap245", "cap246", "cap247", "cap257", "cap294", "cap295"]
        self.assertEqual(solve._runs(names), "cap245–247, cap257, cap294–295")

    def test_runs_on_names_without_numbers(self):
        self.assertEqual(solve._runs(["GA", "GB"]), "GA, GB")


class PerImage(unittest.TestCase):
    def test_none_mean_does_not_crash_and_is_reported(self):
        rec = Recorder()
        per = {"images": [{"name": "L/cap000.jpg", "n_obs": 1500, "mean_reproj_px": 1.7},
                          {"name": "L/cap256.jpg", "n_obs": 0, "mean_reproj_px": None},
                          {"name": "R/cap274.jpg", "n_obs": 142, "mean_reproj_px": 2.475}]}
        solve._per_image_checks(rec, per)
        self.assertEqual(rec.metrics["worst_image_reproj_px"], 2.475)
        self.assertEqual(rec.metrics["images_without_observations"], ["L/cap256.jpg"])
        names = {c[0]: c for c in rec.checks}
        self.assertTrue(names["no_image_over_3px"][1])
        self.assertFalse(names["registered_images_have_observations"][1])
        self.assertTrue(names["registered_images_have_observations"][3])   # needs a human

    def test_all_observed_passes(self):
        rec = Recorder()
        solve._per_image_checks(rec, {"images": [{"name": "L/a.jpg", "n_obs": 10, "mean_reproj_px": 1.0}]})
        self.assertTrue(dict((c[0], c[1]) for c in rec.checks)["registered_images_have_observations"])


class Outliers(unittest.TestCase):
    def test_far_camera_is_named(self):
        rng = np.random.default_rng(0)
        C = rng.normal(0, 300, (40, 3))
        C[7] = [50000.0, 0.0, 0.0]          # one capture placed 50 m away in a 30 cm orbit
        names = [f"cap{i:03d}_L" for i in range(40)]
        rec = Recorder()
        solve._outlier_check(rec, {"names": np.array(names), "C": C})
        self.assertEqual(rec.metrics["outlier_captures"], ["cap007"])
        chk = [c for c in rec.checks if c[0] == "no_outlier_camera_positions"][0]
        self.assertFalse(chk[1]); self.assertTrue(chk[3])

    def test_tight_orbit_passes(self):
        rng = np.random.default_rng(1)
        rec = Recorder()
        solve._outlier_check(rec, {"names": np.array([f"c{i}_L" for i in range(30)]), "C": rng.normal(0, 300, (30, 3))})
        self.assertTrue([c for c in rec.checks if c[0] == "no_outlier_camera_positions"][0][1])
        self.assertNotIn("outlier_captures", rec.metrics)


class Flags(unittest.TestCase):
    def test_new_flags_parse(self):
        import argparse
        sub = argparse.ArgumentParser().add_subparsers()
        solve.add_parser(sub)
        a = sub.choices["solve"].parse_args(["--export-only", "--allow-partial"])
        self.assertTrue(a.export_only and a.allow_partial)
        b = sub.choices["solve"].parse_args([])
        self.assertFalse(b.export_only or b.allow_partial)


if __name__ == "__main__":
    unittest.main()
