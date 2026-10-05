"""hs select says, before the solve, whether the picks chain.

2026-10-05, a 60 m walk through a garden on the Hydrogen: 211 picks, a 19-minute solve, 162
registered. The selector had already measured it: 48 of the picks carried under 10 tracked points
from the pick before. On the four clips with the measurement the share of weak picks and the share
of frames the solver lost run together (0.8 % and none, 11 % and 12 %, 23 % and 23 %, 30 % and 36 %).
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from hs.stages import select  # noqa: E402


class Recorder:
    def __init__(self):
        self.metrics, self.checks = {}, {}

    def metric(self, stage, name, value):
        self.metrics[name] = value

    def check(self, stage, name, ok, value=None, needs_human=False):
        self.checks[name] = {"ok": ok, "value": value, "needs_human": needs_human}


def picks(tracked):
    return [{"sel": i, "tracked": t} for i, t in enumerate(tracked)]


class WeakLinks(unittest.TestCase):
    def test_a_clip_whose_picks_chain(self):
        pj = Recorder()
        select.weak_links(pj, picks([None] + [120] * 98 + [4]))        # the first pick has no pick before it
        self.assertEqual(pj.metrics, {"weak_links": 1, "weak_link_share": round(1 / 99, 4)})
        c = pj.checks["picks_overlap_enough"]
        self.assertTrue(c["ok"])
        self.assertFalse(c["needs_human"])
        self.assertIn("1 of 99 picks", c["value"])

    def test_a_clip_that_will_lose_frames_says_how_many_and_what_to_do(self):
        pj = Recorder()
        select.weak_links(pj, picks([None] + [150] * 80 + [3, 1, 7, 9, 0] * 4))
        self.assertEqual(pj.metrics["weak_links"], 20)
        c = pj.checks["picks_overlap_enough"]
        self.assertFalse(c["ok"])
        self.assertTrue(c["needs_human"])
        self.assertIn("20 of 100 picks", c["value"])
        self.assertIn("expect the solver to lose about that many frames", c["value"])
        self.assertIn("smaller gap", c["value"])

    def test_the_threshold_is_under_ten_points_and_five_percent_of_picks(self):
        pj = Recorder()
        select.weak_links(pj, picks([10] * 95 + [9] * 5))
        self.assertEqual(pj.metrics["weak_links"], 5)
        self.assertFalse(pj.checks["picks_overlap_enough"]["ok"], "5 % is over the line")
        pj = Recorder()
        select.weak_links(pj, picks([10] * 96 + [9] * 4))
        self.assertTrue(pj.checks["picks_overlap_enough"]["ok"])

    def test_nothing_is_said_when_no_tracking_was_measured(self):
        pj = Recorder()
        select.weak_links(pj, [{"sel": 0, "tracked": None}, {"sel": 1}])
        self.assertEqual((pj.metrics, pj.checks), ({}, {}))


if __name__ == "__main__":
    unittest.main()
