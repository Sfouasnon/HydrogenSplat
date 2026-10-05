"""hs views' summary and next_steps: words from the numbers, nothing the engine did not measure."""
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from hs.project import Project  # noqa: E402
from hs.stages import views  # noqa: E402

# the Stormtrooper A-holdout numbers, rounded
METRICS = {"psnr_median": 28.91, "psnr_interior_median": 31.105, "psnr_edge_median": 14.065,
           "retained_edge_energy_norm_median": 0.828, "displaced_fraction_median": 0.013, "displaced_fraction_worst": 0.078,
           "by_azimuth": [{"azimuth_deg": [-135, -90], "views": 2, "displaced_median": 0.043},
                          {"azimuth_deg": [0, 45], "views": 3, "displaced_median": 0.011}],
           "views_soft": []}
OK = [{"name": "model_registers_to_photographs", "ok": True}, {"name": "no_view_much_softer_than_achievable", "ok": True},
      {"name": "every_azimuth_band_registers", "ok": True}, {"name": "subject_registers_better_than_its_surroundings", "ok": True}]


class Summary(unittest.TestCase):
    def test_lines_from_a_good_run_with_no_previous(self):
        S = views.summarize(METRICS, OK)
        self.assertEqual(len(S["summary"]), 4)
        self.assertEqual(S["summary"][0], "The model matches the photographs at 31.1 dB inside the outline.")
        self.assertIn("14.1 dB", S["summary"][1])
        self.assertIn("the weak part", S["summary"][1])
        self.assertEqual(S["summary"][2], "The model keeps 83% of the photograph's sharpness.")
        self.assertEqual(S["summary"][3], "1.3% of the model sits out of place in the typical view (worst view 8%).")
        # the weak edge is the one next step on a run whose checks all pass
        self.assertEqual(len(S["next_steps"]), 1)
        self.assertIn("The outline is the weak part", S["next_steps"][0])

    def test_against_the_last_run(self):
        same = views.summarize(METRICS, OK, prev={"psnr_interior_median": 31.0})
        self.assertIn("the same as the last run within noise (31.0 dB)", same["summary"][0])
        better = views.summarize(METRICS, OK, prev={"psnr_interior_median": 29.8})
        self.assertIn("better than the last run (29.8 dB, up 1.3 dB)", better["summary"][0])
        worse = views.summarize(METRICS, OK, prev={"psnr_interior_median": 33.0})
        self.assertIn("worse than the last run (33.0 dB, down 1.9 dB)", worse["summary"][0])
        # a whole-crop run compares whole-crop numbers, and says nothing about an outline
        m = {k: v for k, v in METRICS.items() if k not in ("psnr_interior_median", "psnr_edge_median")}
        S = views.summarize(m, OK, prev={"psnr_median": 28.9, "psnr_interior_median": 20.0})
        self.assertEqual(S["summary"][0], "The model matches the photographs at 28.9 dB, the same as the last run within noise (28.9 dB).")
        self.assertEqual(len(S["summary"]), 3)
        self.assertEqual(S["next_steps"], [])

    def test_a_close_edge_is_not_the_weak_part(self):
        m = dict(METRICS, psnr_edge_median=24.0)
        S = views.summarize(m, OK)
        self.assertEqual(S["summary"][1], "The edge of the outline scores 24.0 dB, close to the interior.")
        self.assertEqual(S["next_steps"], [])

    def test_next_steps_name_what_to_do(self):
        failed = [{"name": "model_registers_to_photographs", "ok": False, "value": "worst view x"},
                  {"name": "every_azimuth_band_registers", "ok": False},
                  {"name": "no_view_much_softer_than_achievable", "ok": False},
                  {"name": "subject_registers_better_than_its_surroundings", "ok": False}]
        m = dict(METRICS, views_soft=["sel010_L", "sel011_L"], retained_edge_energy_norm_median=1.05)
        S = views.summarize(m, failed, solve_metrics={"elevation_range_deg": [-12.6, 14.0], "coverage_gaps": ["behind", {"name": "below"}]},
                            solve_checks=[{"name": "mean_reproj_ok", "ok": False, "value": "2.4 px (want ≤ 1.8)"}],
                            exposure_checks=[{"name": "exposure_consistent", "ok": False,
                                              "value": "The subject's brightness swings 1.2 stops across the frames; match every frame to one reference before training."}])
        self.assertEqual(S["summary"][2], "The model is sharper than the photographs: the frames carry motion blur.")
        steps = S["next_steps"]
        self.assertTrue(steps[0].startswith("Shoot more frames around azimuth -135° to -90° (4% out of place"), steps[0])
        self.assertIn("Shoot from above: the highest frame looks from only 14° up", steps[1])
        self.assertEqual(steps[2], "Shoot from behind: no frame covers it.")
        self.assertEqual(steps[3], "Shoot from below: no frame covers it.")
        self.assertTrue(steps[4].startswith("Match the exposure in the Look step: The subject's brightness swings 1.2 stops"))
        self.assertTrue(steps[5].startswith("Calibrate the lens: the cameras agree to 2.4 px"))
        self.assertTrue(steps[6].startswith("The outline is the weak part"))
        self.assertIn("sel010_L, sel011_L", steps[7])
        self.assertIn("The subject itself is out of place", steps[8])
        self.assertEqual(len(steps), 9)

    def test_exposure_drift_without_a_check_still_counts(self):
        S = views.summarize(METRICS, OK, exposure_metrics={"drift_stops": 0.9})
        self.assertTrue(any(s.startswith("Match the exposure in the Look step: the frames swing 0.9 stops") for s in S["next_steps"]))
        S = views.summarize(METRICS, OK, exposure_metrics={"drift_stops": 0.2})
        self.assertFalse(any("exposure" in s for s in S["next_steps"]))


class PreviousReport(unittest.TestCase):
    def test_medians_of_the_last_report_of_the_same_name(self):
        tmp = tempfile.mkdtemp(prefix="hs-views-")
        try:
            pj = Project(os.path.join(tmp, "p"), create=True)
            os.makedirs(pj.path("views"))
            rows = [{"psnr_db": 20.0, "psnr_interior_db": 30.0, "psnr_edge_db": 10.0, "retained_edge_energy_norm": 0.8, "displaced_fraction": 0.01},
                    {"psnr_db": 22.0, "psnr_interior_db": 32.0, "psnr_edge_db": None, "retained_edge_energy_norm": 0.9, "displaced_fraction": 0.03},
                    {"psnr_db": 24.0, "psnr_interior_db": 34.0, "psnr_edge_db": 12.0, "retained_edge_energy_norm": 1.0, "displaced_fraction": 0.05}]
            json.dump({"views": rows}, open(pj.path("views", "A_report.json"), "w"))
            prev = views.previous_report(pj, "A")
            self.assertEqual(prev["psnr_median"], 22.0)
            self.assertEqual(prev["psnr_interior_median"], 32.0)
            self.assertEqual(prev["psnr_edge_median"], 11.0)
            self.assertEqual(prev["views"], 3)
            self.assertIsNone(views.previous_report(pj, "B"), "another name is another series")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
