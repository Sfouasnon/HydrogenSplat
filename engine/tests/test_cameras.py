"""cameras.json: the poses the app's viewer stands at must be the poses rig.npz holds.

The check that matters is a reprojection: a world point pushed through the exported c2w and
pinhole must land on the pixel rig.npz's own R, t, K put it on. A transposed rotation, a
missed mm->m or a swapped cx/cy all pass a "the file has the right keys" test and all fail
this one.
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

from hs import cameras  # noqa: E402
from hs.project import md5_file  # noqa: E402


def _rot(axis, deg):
    a = np.radians(deg)
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    Kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * Kx + (1 - np.cos(a)) * Kx @ Kx


def write_rig(path, stereo=True, n=6, seed=0):
    """A ring of cameras 600 mm from a point cloud at the origin, each looking at it."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(0, 40.0, (400, 3))
    names, K, R, t, C, WH = [], [], [], [], [], []
    for i in range(n):
        az = -50 + 100 * i / (n - 1)
        centre = _rot([0, 1, 0], az) @ np.array([0.0, -60.0, -600.0])
        eyes = (("L", 0.0), ("R", 10.64)) if stereo else (("L", 0.0),)
        fwd = -centre / np.linalg.norm(centre)
        right = np.cross([0.0, 1.0, 0.0], fwd); right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        Rwc = np.stack([right, down, fwd])            # rows: camera axes in world = world->camera
        for e, off in eyes:
            c = centre + right * off
            names.append((f"cap{i:03d}" if stereo else "GHIJKL"[i] + "A") + "_" + e)
            K.append([[1500.0 + i, 0, 955.5 + i], [0, 1498.0, 540.25 - i], [0, 0, 1]])
            WH.append([1913 - (4 if e == "R" else 0), 1073 - (2 if e == "R" else 0)])
            R.append(Rwc); C.append(c); t.append(-Rwc @ c)
    kw = dict(names=np.array(names), K=np.array(K), R=np.array(R), t=np.array(t), C=np.array(C), pts=pts,
              wh=np.array(WH, int), w=WH[0][0], h=WH[0][1], s_mm=1.0)
    if not stereo:
        kw["stereo"] = False
    np.savez(path, **kw)
    return pts


class CamerasJson(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.rig = os.path.join(self.tmp.name, "rig.npz")

    def tearDown(self):
        self.tmp.cleanup()

    def test_reprojection_matches_rig(self):
        pts = write_rig(self.rig)
        t = cameras.compute(self.rig)
        G = np.load(self.rig, allow_pickle=True)
        self.assertEqual(len(t["views"]), len(G["names"]))
        worst = 0.0
        for i, v in enumerate(t["views"]):
            self.assertEqual(v["name"], str(G["names"][i]))
            # the rig's own projection, mm
            Xc = (G["R"][i] @ pts.T).T + G["t"][i]
            uv_rig = (G["K"][i] @ (Xc / Xc[:, 2:3]).T).T[:, :2]
            # the exported one, metres
            w2c = np.linalg.inv(np.array(v["c2w"]))
            Xm = (w2c[:3, :3] @ (pts / 1000.0).T).T + w2c[:3, 3]
            self.assertTrue((Xm[:, 2] > 0).all(), "points must be in front of the camera (z forward)")
            uv = np.stack([v["fx"] * Xm[:, 0] / Xm[:, 2] + v["cx"], v["fy"] * Xm[:, 1] / Xm[:, 2] + v["cy"]], 1)
            worst = max(worst, float(np.abs(uv - uv_rig).max()))
        self.assertLess(worst, 1e-6, f"reprojection differs from rig.npz by {worst} px")

    def test_units_sizes_and_frame(self):
        write_rig(self.rig)
        t = cameras.compute(self.rig)
        v0, v1 = t["views"][0], t["views"][1]
        self.assertAlmostEqual(np.linalg.norm(np.array(v0["c2w"])[:3, 3]), 0.6030, places=3)   # metres, not mm
        self.assertAlmostEqual(np.linalg.norm(np.array(v0["c2w"])[:3, 3] - np.array(v1["c2w"])[:3, 3]), 0.01064, places=6)
        self.assertEqual((v0["w"], v0["h"], v1["w"], v1["h"]), (1913, 1073, 1909, 1071))      # per view, not shared
        self.assertEqual((v0["capture"], v0["eye"], v1["eye"]), ("cap000", "L", "R"))
        self.assertTrue(t["stereo"])
        self.assertEqual(t["rig_npz_md5"], md5_file(self.rig))
        self.assertLess(np.linalg.norm(t["subject_m"]), 0.02)                                   # cloud sits at the origin
        self.assertAlmostEqual(np.linalg.norm(t["up_world"]), 1.0, places=5)
        self.assertIsNotNone(v0["azimuth_deg"]); self.assertEqual(v0["azimuth_deg"], v1["azimuth_deg"])
        R = np.array(v0["c2w"])[:3, :3]
        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=9)

    def test_mono_array(self):
        write_rig(self.rig, stereo=False)
        t = cameras.compute(self.rig)
        self.assertFalse(t["stereo"])
        self.assertEqual([v["capture"] for v in t["views"]][:2], ["GA", "HA"])
        self.assertTrue(all(v["eye"] == "L" for v in t["views"]))

    def test_cli_writes_without_touching_manifest_or_lock(self):
        root = os.path.join(self.tmp.name, "proj")
        os.makedirs(os.path.join(root, "train", "dataset")); os.makedirs(os.path.join(root, "archive", "base"))
        write_rig(os.path.join(root, "train", "dataset", "rig.npz"), seed=1)
        write_rig(os.path.join(root, "archive", "base", "rig.npz"), seed=2)
        man = os.path.join(root, "manifest.json")
        json.dump({"stages": {}}, open(man, "w"))
        with open(os.path.join(root, ".hs.lock"), "w") as f:        # a train run holds the lock
            json.dump({"pid": os.getpid(), "stage": "train", "started": "x"}, f)
        before = (md5_file(man), os.path.getmtime(man))
        for extra, name, src in (([], "cameras_current.json", "train/dataset/rig.npz"),
                                 (["--archive", "base"], "cameras_base.json", "archive/base/rig.npz")):
            r = subprocess.run([sys.executable, "-m", "hs", "cameras", "-p", root] + extra,
                               cwd=ENGINE, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            t = json.load(open(os.path.join(root, "viewer", name)))
            self.assertEqual(t["rig_npz_md5"], md5_file(os.path.join(root, src)))
        self.assertEqual(before, (md5_file(man), os.path.getmtime(man)))
        self.assertTrue(os.path.exists(os.path.join(root, ".hs.lock")))
        self.assertEqual(os.listdir(os.path.join(root, "archive", "base")), ["rig.npz"])       # nothing added to the archive
        r = subprocess.run([sys.executable, "-m", "hs", "cameras", "-p", root, "--archive", "nope"],
                           cwd=ENGINE, capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)


if __name__ == "__main__":
    unittest.main()
