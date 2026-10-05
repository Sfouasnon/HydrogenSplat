"""hs move --preset walk: a move for a place, built from the path the camera walked.

The capture here is the shape of a first walk through a garden: in along a path with the camera
weaving left and right, a turn on the spot at the far end, part of the way back out — and then a
stretch the solve placed five metres from where the camera was a second earlier (a tear), which
is what a sequential match does when a whip pan leaves nothing in common between two frames.
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

from hs import walk_path as W  # noqa: E402

UP = np.array([0.0, -1.0, 0.0])                 # image y is down


def _look(yaw_deg, pitch_deg=0.0):
    """World -> camera rotation for a level camera heading yaw (0 = +x, 90 = +z), pitched up."""
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    f = np.array([np.cos(y) * np.cos(p), -np.sin(p), np.sin(y) * np.cos(p)])
    right = np.cross(-UP, f)
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(f, right), f])


def garden(root, timed=True, drop=range(10, 15)):
    """-> (rig.npz, quality.json or None, facts). Millimetres, one capture a second."""
    rng = np.random.default_rng(4)
    caps = []                                    # (centre, yaw)
    for i in range(30):                          # in: 14.5 m, weaving
        caps.append(([500.0 * i, -300.0 + 40 * np.sin(i), 60 * np.sin(i * 1.3)], 35 * np.sin(i * 0.9)))
    for i in range(8):                           # turn on the spot
        caps.append(([14500 + 40 * np.cos(i), -300.0, 40 * np.sin(i)], 180 * (i + 1) / 9))
    for i in range(12):                          # back out, 6 m
        caps.append(([14500 - 500.0 * (i + 1), -500.0, 300.0], 180.0))
    for i in range(20):                          # the torn stretch: in again, higher, from x = 2 m
        caps.append(([2000 + 550.0 * i, -1000.0, 200.0], 5 * np.sin(i * 0.5)))
    names, K, R, t, C = [], [], [], [], []
    keep = [k for k in range(len(caps)) if k not in drop]
    for k in keep:
        c, yaw = caps[k]
        Rwc = _look(yaw, -3.0)
        for e, off in (("L", 0.0), ("R", 10.64)):
            cc = np.array(c) + Rwc[0] * off
            names.append(f"cap{k:03d}_{e}")
            K.append([[1500.0, 0, 956.0], [0, 1500.0, 536.0], [0, 0, 1]])
            R.append(Rwc); C.append(cc); t.append(-Rwc @ cc)
    pts = np.column_stack([rng.uniform(-2000, 18000, 4000), rng.uniform(-2500, 0, 4000),
                           rng.choice([-1, 1], 4000) * rng.uniform(2000, 6000, 4000)])
    rig = os.path.join(root, "rig.npz")
    np.savez(rig, names=np.array(names), K=np.array(K), R=np.array(R), t=np.array(t), C=np.array(C), pts=pts,
             wh=np.array([[1913, 1073]] * len(names)), w=1913, h=1073, s_mm=1.0)
    q = None
    if timed:
        q = os.path.join(root, "quality.json")
        with open(q, "w") as f:
            json.dump({"frames": [{"sel": k, "t_s": float(k)} for k in range(len(caps))]}, f)
    return rig, q, {"in": (0, 29), "turn": (30, 37), "back": (38, 49), "torn": (50, 69)}


def frames(path):
    with open(path) as f:
        m = json.load(f)
    F = np.array([f["c2w"] for f in m["frames"]])
    return m, F[:, :3, 3] * 1000.0, F[:, :3, :3]


class Runs(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)

    def test_a_step_nobody_could_walk_ends_the_run(self):
        rig, q, facts = garden(self.t.name)
        T = W.capture_table(rig, q)
        runs, tears, pace = W.find_runs(T["num"], T["t"], T["C"])
        self.assertEqual([(x["after"], x["before"]) for x in tears], [(49, 50)])
        self.assertEqual([(int(T["num"][a]), int(T["num"][b])) for a, b in runs], [(0, 49), (50, 69)])
        self.assertAlmostEqual(pace, 500.0, delta=60.0)
        # frames that were not placed are a hole in the run, not the end of it: 2.5 m in 6 s is a walk
        self.assertFalse(any(x["after"] == 9 for x in tears))

    def test_without_times_the_same_tear_is_found_from_the_steps(self):
        rig, _q, _f = garden(self.t.name, timed=False)
        T = W.capture_table(rig, None)
        self.assertIsNone(T["t"])
        _runs, tears, _pace = W.find_runs(T["num"], T["t"], T["C"])
        self.assertEqual([(x["after"], x["before"]) for x in tears], [(49, 50)])

    def test_in_and_back_out_are_two_legs_and_the_turn_is_neither(self):
        rig, q, facts = garden(self.t.name)
        T = W.capture_table(rig, q)
        runs, _t, _p = W.find_runs(T["num"], T["t"], T["C"])
        legs = W.find_legs(T["num"], T["t"], T["C"], runs)
        big = [g for g in legs if g["net_mm"] > 3000]
        self.assertEqual(len(big), 3, [(g["first"], g["last"]) for g in legs])
        walk_in, back, torn = big
        self.assertLessEqual(walk_in["first"], 2)
        self.assertTrue(27 <= walk_in["last"] <= 33, walk_in)
        self.assertTrue(34 <= back["first"] <= 40 and back["last"] == 49, back)
        self.assertEqual((torn["first"], torn["last"]), (50, 69))
        self.assertLess(walk_in["placed_share"], 0.9)          # five of its frames were never placed
        self.assertEqual(torn["placed_share"], 1.0)


class Build(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.rig, self.q, self.facts = garden(self.t.name)
        self.out = os.path.join(self.t.name, "walk.json")

    def test_the_default_follows_the_leg_that_travels_furthest_through_placed_frames(self):
        rep = W.build(self.rig, self.out, UP, self.q)
        # in: 14.5 m with 5 of its 30 picks missing scores 14.5 * 25/30 = 12.1; the torn leg 10.45 * 1 — in wins,
        # and it stops where the walking stops, not three frames into the turn
        self.assertEqual(rep["run"], [0, 29])
        self.assertEqual(len(rep["gaps"]), 1)
        self.assertEqual((rep["gaps"][0]["after"], rep["gaps"][0]["before"], rep["gaps"][0]["missing"]), (9, 15, 5))

    def test_every_frame_is_level_on_the_path_and_looks_where_it_is_going(self):
        rep = W.build(self.rig, self.out, UP, self.q, captures=(50, 69))
        m, P, R = frames(self.out)
        self.assertEqual(len(P), rep["frames"])
        self.assertEqual(rep["frames"], int(round(19.0 * 30)) + 1)       # the leg took 19 s; so does the move
        np.testing.assert_allclose(np.einsum("nij,nkj->nik", R, R), np.tile(np.eye(3), (len(R), 1, 1)), atol=1e-9)
        self.assertTrue(np.all(np.linalg.det(R) > 0.999))
        np.testing.assert_allclose(R[:, :, 0] @ UP, 0.0, atol=1e-9)       # camera right has no up in it: level
        self.assertTrue(np.all(R[:, :, 1] @ UP < -0.9))                   # camera down points down
        travel = np.diff(P, axis=0)
        speed = np.linalg.norm(travel, axis=1) * 30
        mid = slice(60, -60)
        self.assertLess(speed[0], 0.05 * speed[mid].mean())               # starts and stops at rest
        self.assertLess(speed[-1], 0.05 * speed[mid].mean())
        self.assertLess(speed[mid].std() / speed[mid].mean(), 0.02)       # and is even in between
        heading = travel[mid] / np.linalg.norm(travel[mid], axis=1, keepdims=True)
        self.assertTrue(np.all((R[:-1][mid][:, :, 2] * heading).sum(1) > 0.99))
        np.testing.assert_allclose(np.degrees(np.arcsin(R[:, :, 2] @ UP)), -3.0, atol=0.2)   # pitched as it was shot
        self.assertLess(rep["off_path_max_mm"], 30.0)                     # a straight leg: on the line
        np.testing.assert_allclose(P[0], [2000, -1000, 200], atol=1.0)    # it starts on the first camera
        np.testing.assert_allclose(P[-1], [2000 + 550 * 19, -1000, 200], atol=1.0)
        self.assertEqual((m["width"], m["height"], m["fps"]), (1913, 1073, 30.0))

    def test_the_weave_is_averaged_out_of_the_path_and_out_of_the_look(self):
        rep = W.build(self.rig, self.out, UP, self.q, captures=(0, 29), steady_s=1.5)
        _m, P, R = frames(self.out)
        self.assertLess(np.ptp(P[:, 2]), 0.6 * 120.0)                     # the cameras swing 120 mm side to side
        yaw = np.degrees(np.arctan2(R[:, 2, 2], R[:, 0, 2]))
        self.assertLess(np.abs(yaw).max(), 6.0)                           # they pan 35 deg each way; the walk looks ahead
        self.assertLess(rep["peak_pan_deg_s"], 10.0)

    def test_as_shot_looks_where_the_camera_looked(self):
        W.build(self.rig, self.out, UP, self.q, captures=(38, 49), look="as-shot")
        _m, P, R = frames(self.out)
        self.assertTrue(np.all(R[:, 0, 2] < -0.99))                       # the way back was shot facing -x
        self.assertLess(P[-1][0], P[0][0])

    def test_reversed_it_travels_the_other_way_and_still_looks_ahead(self):
        W.build(self.rig, self.out, UP, self.q, captures=(50, 69), reverse=True)
        _m, P, R = frames(self.out)
        self.assertLess(P[-1][0], P[0][0])
        self.assertTrue(np.all(R[100:-100, 0, 2] < -0.99))

    def test_a_range_across_the_tear_is_refused_and_says_where_it_is(self):
        with self.assertRaisesRegex(ValueError, "tear in the solve between cap049 and cap050"):
            W.build(self.rig, self.out, UP, self.q, captures=(40, 60))
        with self.assertRaisesRegex(ValueError, "hold 2 placed frames"):
            W.build(self.rig, self.out, UP, self.q, captures=(10, 16))

    def test_the_frame_count_can_be_set(self):
        rep = W.build(self.rig, self.out, UP, self.q, captures=(50, 69), frames=240)
        self.assertEqual((rep["frames"], len(frames(self.out)[1])), (240, 240))


class Stage(unittest.TestCase):
    """Through `hs move`, as the app and Terminal call it."""

    def test_preset_walk_writes_the_move_its_checks_and_the_map(self):
        with tempfile.TemporaryDirectory() as root:
            for d in ("train/dataset", "select", "solve", "logs"):
                os.makedirs(os.path.join(root, d))
            rig, q, _f = garden(root)
            os.replace(rig, os.path.join(root, "train", "dataset", "rig.npz"))
            os.replace(q, os.path.join(root, "select", "quality.json"))
            with open(os.path.join(root, "manifest.json"), "w") as f:
                json.dump({"version": 1, "name": "garden",
                           "stages": {s: {"status": "done"} for s in ("ingest", "select", "solve")}}, f)
            r = subprocess.run([sys.executable, "-m", "hs", "-p", root, "move", "--preset", "walk"],
                               cwd=ENGINE, capture_output=True, text=True)
            evs = [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")]
            self.assertEqual(r.returncode, 0, r.stdout[-1500:] + r.stderr[-1500:])
            self.assertTrue(os.path.exists(os.path.join(root, "move", "walk.json")))
            checks = {e["name"]: e for e in evs if e.get("ev") == "check"}
            self.assertFalse(checks["solve_has_no_tear"]["ok"])
            self.assertIn("cap049 to cap050", checks["solve_has_no_tear"]["value"])
            self.assertFalse(checks["walk_crosses_no_gap"]["ok"])
            self.assertIn("after cap009 (5 frames)", checks["walk_crosses_no_gap"]["value"])
            self.assertTrue(checks["walk_stays_on_the_path"]["ok"])
            self.assertTrue(checks["path_checked_on_the_map"]["needs_human"])
            self.assertTrue(os.path.exists(os.path.join(root, "move", "walk_plan.jpg")))
            with open(os.path.join(root, "manifest.json")) as f:
                met = json.load(f)["stages"]["move"]["metrics"]
            self.assertEqual(met["walk.follows"], [0, 29])
            self.assertIn([50, 69], [g["captures"] for g in met["walk.legs"]])
            self.assertEqual(met["walk.tears"][0]["between"], [49, 50])
            self.assertNotIn("aim_in_frame", checks)


if __name__ == "__main__":
    unittest.main()
