"""hs/frame_quality.py — the per-pick report behind the app's contact sheet. Pure; no clip.

Run: cd engine && ../.venv/bin/python -W ignore -m unittest tests.test_frame_quality
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hs import frame_quality as fq  # noqa: E402


def clip(n=60, picks=(0, 10, 20, 30, 40, 50), sharp=1000.0, std=40.0, luma=0.2, edit=None):
    """A selection.json with a flat trace; edit(k, row) may change frame k's trace row."""
    tr = {"frame": [], "sharp": [], "std": [], "luma": [], "clip": [], "dark": [], "residual": []}
    for k in range(n):
        row = {"sharp": sharp, "std": std, "luma": luma, "clip": 0.0, "dark": 0.0}
        if edit:
            edit(k, row)
        tr["frame"].append(k)
        for key in ("sharp", "std", "luma", "clip", "dark"):
            tr[key].append(row[key])
        tr["residual"].append(None)
    sel = []
    for i, f in enumerate(picks):
        s = {"sel": i, "frame": f, "residual": 1.6, "sharpness": tr["sharp"][f], "clip": tr["clip"][f]}
        if i:
            s["gap"] = f - picks[i - 1]
        sel.append(s)
    return {"fps": 30.0, "frames_total": n, "params": {"max_clip": 0.02, "max_gap": 90},
            "selected": sel, "trace": tr}


def flags(q, frame):
    return next(f for f in q["frames"] if f["frame"] == frame)["flags"]


class TestFrameQuality(unittest.TestCase):
    def test_a_clean_clip_flags_nothing(self):
        q = fq.analyse(clip())
        self.assertEqual(q["flagged"], 0)
        self.assertEqual(q["frames"][0]["ev"], 0.0)
        self.assertEqual(q["frames"][0]["focus_rel"], 1.0)

    def test_darker_is_not_softer(self):
        # half the gain: Laplacian variance and grey variance both drop to a quarter
        def dim(k, row):
            if k == 20:
                row.update(sharp=250.0, std=20.0, luma=0.2 * 0.5 ** 2.2)
        q = fq.analyse(clip(edit=dim))
        self.assertEqual(flags(q, 20), ["exposure"])
        f = next(f for f in q["frames"] if f["frame"] == 20)
        self.assertAlmostEqual(f["sharp_rel"], 0.25, places=3)   # raw Laplacian would call it soft
        self.assertAlmostEqual(f["focus_rel"], 1.0, places=3)    # focus does not
        self.assertAlmostEqual(f["ev"], -2.2, places=3)

    def test_blur_is_soft_and_points_at_the_sharp_frame_in_its_interval(self):
        def blur(k, row):
            if k == 30:
                row["sharp"] = 100.0
            if k == 33:
                row["sharp"] = 1200.0
        q = fq.analyse(clip(edit=blur))
        self.assertEqual(flags(q, 30), ["soft", "sharper_nearby"])
        near = next(f for f in q["frames"] if f["frame"] == 30)["sharper_nearby"]
        self.assertEqual((near["frame"], near["offset"]), (33, 3))
        self.assertAlmostEqual(near["gain"], 12.0, places=2)

    def test_nearby_stays_inside_the_picks_own_interval(self):
        # pick 30 owns 26..35 (halfway to 20 and to 40); a sharp frame at 36 belongs to pick 40
        def blur(k, row):
            if k == 36:
                row["sharp"] = 5000.0
        q = fq.analyse(clip(edit=blur))
        self.assertEqual(fq.neighbourhood([0, 10, 20, 30, 40, 50], 3, 0, 59), (26, 35))
        self.assertNotIn("sharper_nearby", flags(q, 30))
        self.assertIn("sharper_nearby", flags(q, 40))

    def test_clipping_everywhere_is_a_warning_not_a_flag_on_every_frame(self):
        # a white subject: every frame 8% clipped, one frame 20%
        def hot(k, row):
            row["clip"] = 0.20 if k == 30 else 0.08
        q = fq.analyse(clip(edit=hot))
        self.assertEqual([f["frame"] for f in q["frames"] if "clipped" in f["flags"]], [30])
        self.assertTrue(any("every pick clips" in w for w in q["warnings"]))
        # and a sharper frame is still found when every frame clips about the same
        def hot_blur(k, row):
            row["clip"] = 0.08
            if k == 20:
                row["sharp"] = 300.0
        self.assertIn("sharper_nearby", flags(fq.analyse(clip(edit=hot_blur)), 20))
        # a normal clip with one hot frame: flagged, no warning
        def one(k, row):
            if k == 30:
                row["clip"] = 0.05
        q = fq.analyse(clip(edit=one))
        self.assertEqual(flags(q, 30), ["clipped"])
        self.assertEqual(q["warnings"], [])

    def test_a_soft_stretch_with_nothing_better_is_named(self):
        def blur(k, row):
            if 26 <= k <= 35:
                row["sharp"] = 100.0
        q = fq.analyse(clip(edit=blur))
        self.assertEqual(flags(q, 30), ["soft"])
        self.assertTrue(any("nothing sharper" in w and "30" in w for w in q["warnings"]))

    def test_nearby_ignores_clipped_frames(self):
        def hot(k, row):
            if k == 32:
                row.update(sharp=5000.0, clip=0.05)
        self.assertEqual(flags(fq.analyse(clip(edit=hot)), 30), [])

    def test_eye_offset_is_judged_against_the_clips_own_offset(self):
        picks = (0, 10, 20, 30, 40, 50)
        # the right sensor runs a constant 0.3 stops hot: normal, not a flag
        measured = {f: {"file": f"VID_{i:03d}_{f:04d}_2x1.jpg", "sharp_R": 1000.0, "std_R": 40.0,
                        "luma_R": 0.2 * 2 ** 0.3, "clip_R": 0.0, "noise": 1.0}
                    for i, f in enumerate(picks)}
        q = fq.analyse(clip(), measured)
        self.assertEqual(q["flagged"], 0)
        self.assertAlmostEqual(q["medians"]["eye_ev"], 0.3, places=3)
        # one frame's right eye a further half stop off, and another's right eye blurred
        measured[20]["luma_R"] = 0.2 * 2 ** 0.8
        measured[40]["sharp_R"] = 100.0
        q = fq.analyse(clip(), measured)
        self.assertEqual(flags(q, 20), ["eye_exposure"])
        self.assertEqual(flags(q, 40), ["right_soft"])

    def test_noisy_is_relative_to_the_set(self):
        picks = (0, 10, 20, 30, 40, 50)
        measured = {f: {"noise": 1.0} for f in picks}
        measured[50]["noise"] = 2.0
        self.assertEqual(flags(fq.analyse(clip(), measured), 50), ["noisy"])

    def test_max_gap_and_lost_tracking_are_named(self):
        s = clip(n=200, picks=(0, 10, 100))
        s["params"]["max_gap"] = 90
        s["selected"][1]["residual"] = -1.0
        q = fq.analyse(s)
        self.assertIn("untracked", flags(q, 10))
        self.assertIn("stood_still", flags(q, 100))

    def test_dry_run_has_no_eye_columns_and_no_files(self):
        q = fq.analyse(clip())
        f = q["frames"][0]
        self.assertIsNone(f["file"])
        self.assertIsNone(f["eye_ev"])
        self.assertFalse(q["measured_eyes"])

    def test_every_flag_has_text_for_the_app(self):
        codes = {"soft", "sharper_nearby", "exposure", "eye_exposure", "right_soft", "clipped",
                 "crushed", "noisy", "stood_still", "untracked"}
        self.assertEqual(set(fq.FLAG_TEXT), codes)


if __name__ == "__main__":
    unittest.main()
