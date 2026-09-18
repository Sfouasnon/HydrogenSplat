"""Bands and the phone guide's orientation maths.

The point of engine/hs/bands.py is that the scorer and the phone agree about what a band is,
so the first test here is the one that would catch them drifting apart: bands.azimuth_band_lo
against the binning views.azimuth_table actually does.
"""
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "capture"))

import numpy as np                                                            # noqa: E402

from hs import bands                                                          # noqa: E402

import orbit_guide                                                            # noqa: E402

CAPTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "capture")


def qmul(a, b):
    """(w, x, y, z) Hamilton product."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def axis_q(axis, deg):
    h = math.radians(deg) / 2.0
    s, c = math.sin(h), math.cos(h)
    return {"x": (c, s, 0, 0), "y": (c, 0, s, 0), "z": (c, 0, 0, s)}[axis]


def rotvec(q):
    """Android sensors report (x, y, z); w is implied. Feed the guide what the phone would."""
    w, x, y, z = q
    return [x, y, z] if w >= 0 else [-x, -y, -z]


class TestBands(unittest.TestCase):
    def test_azimuth_binning_matches_the_scorer(self):
        # views.azimuth_table bins with int(np.floor(az / band) * band); bands.py must agree
        for az in list(np.arange(-179.9, 180.0, 7.3)) + [-180.0, -45.0, 0.0, 44.999, 45.0]:
            self.assertEqual(bands.azimuth_band_lo(az),
                             int(np.floor(bands.wrap180(az) / 45) * 45), msg=f"az {az}")

    def test_wrap180(self):
        for a, want in [(0, 0), (180, 180), (-180, 180), (181, -179), (540, 180), (-190, 170)]:
            self.assertAlmostEqual(bands.wrap180(a), want, places=6, msg=f"{a}")

    def test_elevation_rings_are_half_open_and_gapless(self):
        self.assertEqual(bands.elevation_band(-25.0), "low")
        self.assertEqual(bands.elevation_band(4.999), "low")
        self.assertEqual(bands.elevation_band(5.0), "mid")
        self.assertEqual(bands.elevation_band(19.999), "mid")
        self.assertEqual(bands.elevation_band(20.0), "high")
        self.assertIsNone(bands.elevation_band(-25.1))
        self.assertIsNone(bands.elevation_band(45.0))

    def test_grid_needs_dwell_and_refuses_smeared_frames(self):
        g = bands.Grid(dwell=1.0, slow_deg_s=25.0)
        self.assertIsNone(g.mark(10.0, 10.0, 0.5))               # halfway
        self.assertEqual(g.mark(10.0, 10.0, 0.6), (0, "mid"))    # completes, reports once
        self.assertIsNone(g.mark(10.0, 10.0, 0.6))               # already covered
        self.assertIn((0, "mid"), g.covered())

        fast = bands.Grid(dwell=1.0, slow_deg_s=25.0)
        self.assertIsNone(fast.mark(10.0, 10.0, 2.0, rate_deg_s=40.0))
        self.assertEqual(fast.covered(), set())

    def test_out_of_ring_elevation_marks_nothing(self):
        g = bands.Grid(dwell=0.1)
        self.assertIsNone(g.mark(0.0, 80.0, 5.0))
        self.assertEqual(g.covered(), set())

    def _grid_missing_only(self, *cells):
        g = bands.Grid(dwell=0.1)
        for lo, ring in bands.all_cells():
            if (lo, ring) not in set(cells):
                g.mark(bands.band_centre(lo), bands.ring_centre(ring), 1.0)
        return g

    def test_a_ring_change_costs_sixty_degrees_of_turn(self):
        # standing in (0, mid) at az 20: the cell above is 2.5° away in azimuth, so the rule is
        # "prefer the turn while it is under 60°, otherwise change ring"
        near = self._grid_missing_only((45, "mid"), (0, "high"))    # turn of 47.5°
        self.assertEqual(near.cue(20.0, 10.0), ("right", (45, "mid")))
        far = self._grid_missing_only((90, "mid"), (0, "high"))     # turn of 92.5°
        self.assertEqual(far.cue(20.0, 10.0), ("raise", (0, "high")))

    def test_cue_turns_the_short_way_round(self):
        g = self._grid_missing_only((-180, "mid"))
        self.assertEqual(g.cue(170.0, 10.0)[0], "right",
                         "az 170 to band centre -157.5 is 32° to the right, not 328° left")

    def test_cue_holds_inside_the_target_cell_then_reports_done(self):
        g = bands.Grid(dwell=10.0)
        for lo, ring in bands.all_cells():
            if (lo, ring) != (0, "mid"):
                g.mark(bands.band_centre(lo), bands.ring_centre(ring), 20.0)
        self.assertEqual(g.cue(20.0, 10.0)[0], "hold")
        g.mark(20.0, 10.0, 20.0)
        self.assertEqual(g.cue(20.0, 10.0), ("done", None))


class TestOrientation(unittest.TestCase):
    def test_elevation_from_gravity(self):
        # phone upright, rear camera level: gravity down the screen
        self.assertAlmostEqual(orbit_guide.elevation_from_gravity([0, -9.81, 0]), 0.0, places=4)
        # screen up, camera pointing at the floor: the camera is directly above the subject
        self.assertAlmostEqual(orbit_guide.elevation_from_gravity([0, 0, -9.81]), 90.0, places=4)
        # screen down, camera at the ceiling
        self.assertAlmostEqual(orbit_guide.elevation_from_gravity([0, 0, 9.81]), -90.0, places=4)
        # tilted 30° down from upright
        g = [0, -9.81 * math.cos(math.radians(30)), -9.81 * math.sin(math.radians(30))]
        self.assertAlmostEqual(orbit_guide.elevation_from_gravity(g), 30.0, places=4)
        self.assertIsNone(orbit_guide.elevation_from_gravity([0, 0, 0]))

    def test_yaw_is_zero_upright_and_increases_as_you_turn_right(self):
        upright = axis_q("x", 90)                      # stand the phone up from face-down
        self.assertAlmostEqual(orbit_guide.yaw_from_quaternion(rotvec(upright)), 0.0, places=4)
        for turn in (-150, -90, -30, 30, 90, 150):
            # a right-hand turn of `turn` is a rotation of −turn about world up
            q = qmul(axis_q("z", -turn), upright)
            got = orbit_guide.yaw_from_quaternion(rotvec(q))
            self.assertAlmostEqual(bands.wrap180(got - turn), 0.0, places=3, msg=f"{turn}°")

    def test_orbit_azimuth_is_the_far_side(self):
        self.assertAlmostEqual(orbit_guide.orbit_azimuth(0.0), 180.0, places=6)
        self.assertAlmostEqual(orbit_guide.orbit_azimuth(90.0), -90.0, places=6)


class TestStream(unittest.TestCase):
    def test_decode_back_to_back_pretty_objects_split_mid_token(self):
        text = ('{\n "Gravity": {\n  "values": [\n   0.0,\n   -9.8,\n   0.4\n  ]\n }\n}'
                '{\n "Game Rotation Vector": {\n  "values": [0.1, 0.2, 0.3]\n }\n}')
        chunks = [text[i:i + 7] for i in range(0, len(text), 7)]   # arbitrary split points
        got = list(orbit_guide.decode_stream(chunks))
        self.assertEqual(len(got), 2)
        self.assertEqual(got[0]["Gravity"]["values"][1], -9.8)
        self.assertEqual(got[1]["Game Rotation Vector"]["values"], [0.1, 0.2, 0.3])

    def test_pick_matches_sensor_names_loosely(self):
        s = {"LSM6DSM Gravity -Wakeup Secondary": {"values": [1, 2, 3]},
             "Game Rotation Vector": {"values": [4, 5, 6]}}
        self.assertEqual(orbit_guide.pick(s, orbit_guide.GRAVITY), [1, 2, 3])
        self.assertEqual(orbit_guide.pick(s, orbit_guide.ROTATION), [4, 5, 6])
        self.assertIsNone(orbit_guide.pick(s, "barometer"))


class TestReplay(unittest.TestCase):
    """An orbit flown only at mid elevation must come back with low and high still open."""

    def _track(self, path, ring_el, step_deg=5.0, dt=0.25):
        with open(path, "w") as fh:
            fh.write(json.dumps({"started": 0, "band": 45, "dwell": 1.5}) + "\n")
            t, az = 0.0, -180.0
            while az < 180.0:
                fh.write(json.dumps({"t": round(t, 3), "az": round(az, 2), "el": ring_el}) + "\n")
                t, az = t + dt, az + step_deg

    def test_a_mid_only_orbit_reports_the_two_empty_rings(self):
        with tempfile.TemporaryDirectory() as d:
            track = os.path.join(d, "mid.jsonl")
            self._track(track, ring_el=12.0)
            r = subprocess.run([sys.executable, os.path.join(CAPTURE, "orbit_guide.py"),
                                "--replay", track, "--quiet"],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("8/24 cells covered", r.stdout)
            self.assertIn("low   still open", r.stdout)
            self.assertIn("high  still open", r.stdout)
            self.assertNotIn("mid   still open", r.stdout)

    def test_bursty_arrival_does_not_read_as_a_sprint(self):
        """Samples delivered several at a time share an arrival clock to within microseconds.

        Dividing the angle moved by that interval reported hundreds of degrees a second for a
        phone held still, every cell looked smeared, and a real 20-second run covered nothing.
        """
        def bursts():
            clock, az, el = 1000.0, 10.0, 10.0
            for burst in range(40):                     # 40 bursts of 5, 0.1 s apart
                for i in range(5):
                    az += 0.05                          # a hand not quite steady
                    yield {"az": az, "el": el}, clock + i * 2e-4
                clock += 0.1

        grid = bands.Grid(dwell=1.5, slow_deg_s=25.0)
        report, stats = orbit_guide.run(bursts(), grid,
                                        orbit_guide.Voice(enabled=False, vibrate=False), quiet=True)
        covered = [r for r in report if r["covered"]]
        self.assertEqual([(r["azimuth_deg"][0], r["ring"]) for r in covered], [(0, "mid")])
        self.assertLess(orbit_guide.median(stats["rates"]), 25.0,
                        "a hand jittering 0.05° between samples is not a 250°/s sprint")
        self.assertEqual(stats["samples"], 200)

    def test_stats_separate_the_three_ways_to_cover_nothing(self):
        def track(el, turn, n=60, dt=0.25):
            clock, az = 0.0, 0.0
            for _ in range(n):
                yield {"az": az, "el": el}, clock
                clock, az = clock + dt, bands.wrap180(az + turn * dt)

        def go(gen):
            g = bands.Grid(dwell=1.5, slow_deg_s=25.0)
            return orbit_guide.run(gen, g, orbit_guide.Voice(enabled=False, vibrate=False),
                                   quiet=True)[1]

        held_wrong = go(track(el=80.0, turn=0.0))        # phone flat, outside every ring
        self.assertGreater(held_wrong["out_of_band_s"], 10.0)
        self.assertEqual(held_wrong["too_fast_s"], 0.0)

        sprinted = go(track(el=10.0, turn=90.0))         # in band, far too quick
        self.assertEqual(sprinted["out_of_band_s"], 0.0)
        self.assertGreater(sprinted["too_fast_s"], 10.0)

        stalled = go(track(el=10.0, turn=0.0, n=3))      # stream died after three samples
        self.assertEqual(stalled["samples"], 3)
        self.assertLess(stalled["elapsed_s"], 1.0)

    def test_a_sprinted_orbit_covers_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            track = os.path.join(d, "fast.jsonl")
            self._track(track, ring_el=12.0, step_deg=30.0, dt=0.1)   # 300°/s
            r = subprocess.run([sys.executable, os.path.join(CAPTURE, "orbit_guide.py"),
                                "--replay", track, "--quiet"],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("0/24 cells covered", r.stdout)


if __name__ == "__main__":
    unittest.main()
