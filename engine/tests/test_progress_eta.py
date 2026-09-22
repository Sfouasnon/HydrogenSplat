"""events.progress fills eta_s itself when a stage gives a total and no ETA (events.auto_eta).

Driven by a fake clock, so the numbers are exact: the ETA is the mean rate since the step's
first sample applied to what is left.
"""
import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from hs import events  # noqa: E402


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def monotonic(self):
        return self.t

    def time(self):
        return self.t

    def strftime(self, *a):
        return "2026-09-22T00:00:00"


class AutoEta(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.got = []
        events._last_progress.clear()
        events.reset_eta()
        self.p = mock.patch.object(events, "time", self.clock)
        self.p.start()
        events.set_sink(self.got.append)
        self.out = io.StringIO()
        self.r = redirect_stdout(self.out)
        self.r.__enter__()

    def tearDown(self):
        self.r.__exit__(None, None, None)
        events.set_sink(None)
        self.p.stop()
        events._last_progress.clear()
        events.reset_eta()

    def at(self, t, *a, **k):
        self.clock.t = 1000.0 + t
        events.progress(*a, **k)
        return self.got[-1] if self.got and self.got[-1].get("ev") == "progress" else None

    def test_mean_rate_since_first_sample(self):
        self.at(0, "views", 0, 100, step="render")
        ev = self.at(1, "views", 10, 100, step="render")
        self.assertNotIn("eta_s", ev, "no ETA before AUTO_ETA_MIN_S")
        ev = self.at(4, "views", 20, 100, step="render")
        self.assertEqual(ev["eta_s"], 16)            # 20 done in 4 s -> 80 left at 5/s
        ev = self.at(10, "views", 40, 100, step="render")
        self.assertEqual(ev["eta_s"], 15)            # 40 in 10 s -> 60 left at 4/s
        ev = self.at(20, "views", 100, 100, step="render")
        self.assertEqual(ev["eta_s"], 0)

    def test_first_step_after_start_is_anchored_at_start(self):
        self.clock.t = 1000.0
        events.start("exposure", "measure")
        ev = self.at(5, "exposure", 1, 11, step="measure")   # i + 1 after item 0 took 5 s
        self.assertEqual(ev["eta_s"], 50)

    def test_later_step_is_not_anchored(self):
        self.clock.t = 1000.0
        events.start("solve", "sfm")
        self.at(100, "solve", 50, 100, step="features")
        # matching begins at t=300 with block 0: must not inherit features' 100 s
        self.at(300, "solve", 0, 289, step="matching", force=True)
        ev = self.at(310, "solve", 10, 289, step="matching", force=True)
        self.assertEqual(ev["eta_s"], 279)          # 1 block/s

    def test_start_resets(self):
        self.at(0, "masks", 0, 10, step="project")
        self.at(5, "masks", 5, 10, step="project")
        self.clock.t = 1100.0
        events.start("masks", "project")
        # a fresh run of the step: anchored at the new start, not at t=0
        ev = self.at(104, "masks", 2, 10, step="project", force=True)
        self.assertEqual(ev["eta_s"], 16)

    def test_explicit_eta_wins_and_no_total_no_eta(self):
        self.at(0, "train", 0, 100, step="train")
        ev = self.at(5, "train", 50, 100, step="train", eta_s=999)
        self.assertEqual(ev["eta_s"], 999)
        self.at(0, "select", 0, None, step="select", force=True)
        ev = self.at(9, "select", 5, None, step="select", force=True)
        self.assertNotIn("eta_s", ev)

    def test_samples_count_even_when_rate_limited(self):
        # the first sample is taken by a call the 0.25 s rate limit swallows
        self.at(0, "split", 0, 10, step="weights")
        self.at(0.1, "split", 1, 10, step="weights")          # dropped by the rate limit
        n = len(self.got)
        ev = self.at(4, "split", 5, 10, step="weights")
        self.assertEqual(len(self.got), n + 1)
        self.assertEqual(ev["eta_s"], 4)                       # from the t=0 sample: 5 in 4 s

    def test_done_going_backwards_restarts(self):
        self.at(0, "render", 0, 100, step="encode")
        self.at(10, "render", 50, 100, step="encode")
        self.at(20, "render", 0, 100, step="encode", force=True)   # second encode pass
        ev = self.at(24, "render", 20, 100, step="encode", force=True)
        self.assertEqual(ev["eta_s"], 16)

    def test_non_numeric_is_ignored(self):
        ev = self.at(0, "x", "a", "b", step="s", force=True)
        self.assertNotIn("eta_s", ev)

    def test_stdout_lines_carry_eta(self):
        self.at(0, "views", 0, 10, step="render")
        self.at(5, "views", 5, 10, step="render")
        last = self.out.getvalue().strip().splitlines()[-1]
        self.assertIn('"eta_s":5', last)


if __name__ == "__main__":
    unittest.main()


class StageEta(unittest.TestCase):
    """`rest_s` turns a step ETA into the whole stage's remaining time (`stage_eta_s`)."""

    def test_stage_eta_is_step_eta_plus_rest(self):
        import io, json
        from hs import events
        buf = io.StringIO()
        old = events._out if hasattr(events, "_out") else None
        evs = []
        orig = events._emit
        events._emit = lambda ev: evs.append(ev)
        try:
            events._last_progress.clear()
            events.progress("solve", 10, 100, step="features", eta_s=90.0, rest_s=3600.0, force=True)
            events.progress("solve", 20, 100, step="features", eta_s=80.0, force=True)
        finally:
            events._emit = orig
        self.assertEqual(evs[0]["eta_s"], 90)
        self.assertEqual(evs[0]["stage_eta_s"], 3690)
        self.assertNotIn("stage_eta_s", evs[1])


class SolveRest(unittest.TestCase):
    def test_rest_after_sums_the_later_phases(self):
        from hs.stages.solve import SfmParser
        from hs import timing
        p = SfmParser(n_img=300, stereo=True)
        sec = timing.phase_seconds(p.model, 300, 300 * 299 // 2)
        self.assertAlmostEqual(p.rest_after("features"), sec["matching"] + sec["mapping"] + sec["export"])
        self.assertAlmostEqual(p.rest_after("mapping"), sec["export"])
        self.assertEqual(p.rest_after("export"), 0.0)
        p.num_pairs = 1000
        sec2 = timing.phase_seconds(p.model, 300, 1000)
        self.assertAlmostEqual(p.rest_after("matching"), sec2["mapping"] + sec2["export"])
