"""hs prune --clear-path: nothing sits where the camera itself went.

A model of a walk-through puts faint splats on the line the camera travelled; a move along that
line flies through them. The test model has a wall of scenery 2 m to one side of the path, a
cloud of floaters hugging the path, and a clump sitting in the gap a tear in the solve leaves
between two stretches -- which is not somewhere the camera went and must be left alone.
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

from hs import clearpath  # noqa: E402
from test_walk import garden  # noqa: E402

PROPS = ["x", "y", "z", "scale_0", "scale_1", "scale_2", "opacity", "rot_0", "rot_1", "rot_2", "rot_3",
         "f_dc_0", "f_dc_1", "f_dc_2"]


def write_model(path, xyz_m, opacity_logit):
    n = len(xyz_m)
    arr = np.zeros((n, len(PROPS)), "<f4")
    arr[:, 0:3] = xyz_m
    arr[:, 3:6] = np.log(0.01)
    arr[:, 6] = opacity_logit
    arr[:, 7] = 1.0
    arr[:, 11:14] = np.arange(n)[:, None]            # each row can be told apart afterwards
    head = "ply\nformat binary_little_endian 1.0\ncomment made for a test\n" + f"element vertex {n}\n" \
           + "".join(f"property float {p}\n" for p in PROPS) + "end_header\n"
    with open(path, "wb") as f:
        f.write(head.encode("ascii") + arr.tobytes())


def read_model(path):
    with open(path, "rb") as f:
        raw = f.read()
    k = raw.index(b"end_header\n") + 11
    n = int([l for l in raw[:k].decode().split("\n") if l.startswith("element vertex")][0].split()[-1])
    return np.frombuffer(raw[k:], "<f4").reshape(n, len(PROPS)), raw[:k].decode()


def scene():
    """(xyz in metres, labels). The garden rig walks x = 0..14.5 m at y = -0.3 m, z about 0."""
    rng = np.random.default_rng(2)
    wall = np.column_stack([rng.uniform(0, 14, 3000), rng.uniform(-2, 0, 3000), np.full(3000, 2.0)])
    hug = np.column_stack([rng.uniform(1, 13, 400), -0.3 + rng.normal(0, 0.03, 400), rng.normal(0, 0.03, 400)])
    # the solve's tear: cap049 at x = 8.5 m (y -0.5, z 0.3) to cap050 at x = 2 m (y -1.0, z 0.2).
    # A clump high above the middle of that jump is 1.6 m from every real camera position.
    gap = np.array([5.25, -2.4, 1.2]) + rng.normal(0, 0.02, (50, 3))
    return np.concatenate([wall, hug, gap]), np.r_[np.zeros(3000), np.ones(400), np.full(50, 2)]


class Line(unittest.TestCase):
    def test_the_line_stops_at_a_tear(self):
        with tempfile.TemporaryDirectory() as root:
            rig, q, _f = garden(root)
            a, b, step, tears = clearpath.walked_line(rig, q)
            self.assertEqual([(t["after"], t["before"]) for t in tears], [(49, 50)])
            self.assertEqual(len(a), 65 - 1 - 1)            # 65 placed frames, one segment fewer, one tear
            # the longest segment is the 3 m hole where five frames were not placed (the camera did walk
            # it); the 6.5 m jump across the tear is not on the line
            self.assertLess(np.linalg.norm(b - a, axis=1).max(), 3100.0)
            self.assertAlmostEqual(step, 500.0, delta=60.0)

    def test_distance_is_to_the_segment_not_the_infinite_line(self):
        a = np.array([[0.0, 0, 0]]); b = np.array([[10.0, 0, 0]])
        d = clearpath.distance_to_line(np.array([[5.0, 3, 0], [-4.0, 3, 0], [12.0, 0, 0]]), a, b)
        np.testing.assert_allclose(d, [3.0, 5.0, 2.0], atol=1e-5)


class Clear(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.rig, self.q, _f = garden(self.t.name)
        self.xyz, self.label = scene()
        self.ply = os.path.join(self.t.name, "model.ply")
        write_model(self.ply, self.xyz, np.where(self.label == 0, 3.0, -2.5))

    def test_what_hugs_the_path_goes_and_nothing_else_does(self):
        a, b, _step, _t = clearpath.walked_line(self.rig, self.q)
        out, only = os.path.join(self.t.name, "o.ply"), os.path.join(self.t.name, "r.ply")
        rep = clearpath.clear(self.ply, out, a, b, 300.0, removed_out=only)
        kept, head = read_model(out)
        gone, _h = read_model(only)
        ids_kept, ids_gone = kept[:, 11].astype(int), gone[:, 11].astype(int)
        self.assertEqual(rep["removed"], 400)
        self.assertTrue(np.all(self.label[ids_gone] == 1))
        self.assertEqual(sorted(np.unique(self.label[ids_kept])), [0, 2])     # scenery, and the clump in the tear's gap
        self.assertEqual((rep["splats_in"], rep["splats_out"], len(kept)), (3450, 3050, 3050))
        self.assertIn("element vertex 3050\n", head)
        self.assertIn("comment made for a test\n", head)
        np.testing.assert_array_equal(kept[:, :3], self.xyz[ids_kept].astype("<f4"))  # rows are copied, not rewritten
        self.assertGreater(rep["opacity_mass_kept"], 0.98)                    # the floaters were faint
        self.assertAlmostEqual(rep["removed_median_opacity"], 1 / (1 + np.exp(2.5)), places=4)

    def test_through_hs_prune_with_the_radius_taken_from_the_capture(self):
        root = os.path.join(self.t.name, "proj")
        for d in ("train/dataset", "select", "solve", "train/exports"):
            os.makedirs(os.path.join(root, d))
        os.replace(self.rig, os.path.join(root, "train", "dataset", "rig.npz"))
        os.replace(self.q, os.path.join(root, "select", "quality.json"))
        os.replace(self.ply, os.path.join(root, "train", "exports", "export_20000.ply"))
        with open(os.path.join(root, "manifest.json"), "w") as f:      # a run that was stopped part-way
            json.dump({"version": 1, "name": "garden", "stages": {
                "ingest": {"status": "done"}, "select": {"status": "done"}, "solve": {"status": "done"},
                "train": {"status": "failed", "error": "interrupted"}}}, f)
        r = subprocess.run([sys.executable, "-m", "hs", "-p", root, "prune", "--clear-path"],
                           cwd=ENGINE, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout[-1500:] + r.stderr[-1500:])
        with open(os.path.join(root, "manifest.json")) as f:
            met = json.load(f)["stages"]["prune"]["metrics"]
        self.assertAlmostEqual(met["clear_path_radius_mm"], 1.3 * 500.0, delta=80.0)
        self.assertEqual(met["output_ply"], "prune/export_20000_clearpath.ply")
        self.assertEqual((met["splats_in"], met["on_the_path"]), (3450, 400))
        kept, _h = read_model(os.path.join(root, "prune", "export_20000_clearpath.ply"))
        self.assertEqual(len(kept), 3050)
        self.assertTrue(os.path.exists(os.path.join(root, "prune", "export_20000_path_only.ply")))
        checks = {json.loads(l)["name"]: json.loads(l) for l in r.stdout.splitlines()
                  if l.startswith("{") and json.loads(l).get("ev") == "check"}
        self.assertFalse(checks["path_clearing_is_small"]["ok"])           # 400 of 3,450 is 11.6 %: it asks to be looked at
        self.assertTrue(checks["path_clearing_is_small"]["needs_human"])


if __name__ == "__main__":
    unittest.main()
