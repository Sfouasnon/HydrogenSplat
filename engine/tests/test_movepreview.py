"""hs movepreview: the viewer's live compile of an .hsmove script.

It must write the same path `hs move --script` would (same compiler, same rig.npz), carry the
track and captures the move panel plots, and never touch the pipeline: no lock, no manifest
write — the panel recompiles on every edit, including while train holds the lock.
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

from hs import movescript  # noqa: E402
from test_cameras import write_rig  # noqa: E402


def hs(*args):
    r = subprocess.run([sys.executable, "-m", "hs", *args], cwd=ENGINE, capture_output=True, text=True)
    evs = [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")]
    return r.returncode, evs


class MovePreview(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        os.makedirs(os.path.join(self.root, "train", "dataset"))
        os.makedirs(os.path.join(self.root, "move"))
        self.rig = os.path.join(self.root, "train", "dataset", "rig.npz")
        write_rig(self.rig, stereo=True, n=60)       # 60 captures 1.7 deg apart: a continuous band
        self.manifest = os.path.join(self.root, "manifest.json")
        with open(self.manifest, "w") as f:
            json.dump({"stages": {}}, f)
        # the band's elevation, measured the way movescript measures it
        G = np.load(self.rig)
        L = list(range(0, len(G["names"]), 2))
        down = np.mean([G["R"][v].T @ np.array([0.0, 1.0, 0.0]) for v in L], axis=0)
        up = -down / np.linalg.norm(down)
        V = G["C"][L] - np.median(G["pts"], axis=0)
        self.el = float(np.median(np.degrees(np.arcsin(V @ up / np.linalg.norm(V, axis=1)))))

    def tearDown(self):
        self.tmp.cleanup()

    def script(self, name, text):
        p = os.path.join(self.root, "move", name + ".hsmove")
        with open(p, "w") as f:
            f.write(text)
        return p

    def test_compiles_what_hs_move_would_and_leaves_the_pipeline_alone(self):
        s = self.script("a", f"start az 0 el {self.el:+.2f}\nhold 0.2s\narc left 20 speed 2\n")
        lock = os.path.join(self.root, ".hs.lock")
        with open(lock, "w") as f:
            f.write('{"pid": 1, "stage": "train"}')
        before = open(self.manifest).read()
        code, evs = hs("movepreview", "-p", self.root, "--script", s)
        self.assertEqual(code, 0, evs)
        self.assertEqual(open(self.manifest).read(), before)
        self.assertTrue(os.path.exists(lock))
        self.assertFalse(os.path.exists(os.path.join(self.root, "move", "a.json")))

        rep = json.load(open(os.path.join(self.root, "viewer", "move_a.report.json")))
        move = json.load(open(os.path.join(self.root, "viewer", "move_a.json")))
        self.assertEqual(rep["frames"], len(move["frames"]))
        self.assertEqual(len(rep["track"]), rep["frames"])
        self.assertEqual(len(rep["captures"]), 60)
        self.assertEqual(rep["capture_names"][0], "cap000")
        self.assertLessEqual(rep["hull_mm"][1], 25.0)
        self.assertAlmostEqual(rep["track"][0][1], self.el, places=1)
        self.assertAlmostEqual(rep["track"][-1][0] - rep["track"][0][0], -20.0, delta=0.5)
        # same compiler, same rig: identical to what the stage would write
        ref = os.path.join(self.root, "ref.json")
        movescript.build(self.rig, open(s).read(), ref)
        self.assertEqual(json.load(open(ref))["frames"], move["frames"])

    def test_script_error_names_the_line_and_writes_nothing(self):
        s = self.script("bad", f"start az 0 el {self.el:+.2f}\nboom to el 0\nwobble 3\n")
        code, evs = hs("movepreview", "-p", self.root, "--script", s)
        self.assertEqual(code, 1)
        err = [e for e in evs if e["ev"] == "error"][0]["message"]
        self.assertTrue(err.startswith("line 3:"), err)
        self.assertFalse(os.path.exists(os.path.join(self.root, "viewer", "move_bad.json")))

    def test_clamped_cue_reports_json_clean(self):
        # the arc runs off the end of the capture: clamped, and the report must still be valid JSON
        s = self.script("far", f"start az 0 el {self.el:+.2f}\narc right 170 speed 5\n")
        code, evs = hs("movepreview", "-p", self.root, "--script", s)
        self.assertEqual(code, 0, evs)
        raw = open(os.path.join(self.root, "viewer", "move_far.report.json")).read()
        self.assertNotIn("NaN", raw)
        rep = json.loads(raw)
        self.assertTrue(rep["cues"][0]["clamped"])
        self.assertLess(rep["cues"][0]["got"], 170)

    def test_locate_round_trips_every_capture(self):
        """A capture's own centre, turned into az / el / dolly, is a reachable start that points
        back at that centre — the check that caught a non-orthogonal azimuth basis (a camera's
        own az/el pointed 260 mm away from it on a 215 deg orbit)."""
        fr = movescript.frame(self.rig)
        reach = movescript.Reach(fr["CL"], fr["subject"], fr["up"], fr["ref"], fr["right"])
        self.assertAlmostEqual(float(fr["ref"] @ fr["right"]), 0.0, places=9)
        for c in fr["CL"]:
            d = movescript.locate(self.rig, c)
            self.assertTrue(d["reachable"], d)
            back = fr["subject"] + d["r_mm"] * reach.direction(d["az"], d["el"])
            self.assertLess(float(np.linalg.norm(back - c)), 1e-6)

    def test_wide_orbit_far_away_is_reachable(self):
        """Cameras 1.4 m out on a 300 deg orbit: every centre must locate as reachable (the old
        fixed 120-600 mm radius search called them all out of reach)."""
        far = os.path.join(self.root, "far.npz")
        rng = np.random.default_rng(1)
        names, R, C = [], [], []
        for i in range(40):
            az = np.radians(-150 + 300 * i / 39)
            c = np.array([1400 * np.sin(az), -150.0, -1400 * np.cos(az)])
            fwd = -c / np.linalg.norm(c)
            right = np.cross([0.0, 1.0, 0.0], fwd); right /= np.linalg.norm(right)
            down = np.cross(fwd, right)
            names.append(f"cap{i:03d}_L"); R.append(np.stack([right, down, fwd])); C.append(c)
        np.savez(far, names=np.array(names), R=np.array(R), C=np.array(C), t=-np.einsum("nij,nj->ni", np.array(R), np.array(C)),
                 K=np.tile(np.array([[1500.0, 0, 960], [0, 1500.0, 540], [0, 0, 1]]), (40, 1, 1)),
                 pts=rng.normal(0, 40.0, (300, 3)), w=1920, h=1080, stereo=False)
        for c in C:
            self.assertTrue(movescript.locate(far, c)["reachable"])

    def test_hull_line_sets_the_limit(self):
        s = self.script("h", f"start az 0 el {self.el:+.2f}\nhull 40\narc left 10 speed 2\n")
        code, evs = hs("movepreview", "-p", self.root, "--script", s)
        self.assertEqual(code, 0, evs)
        rep = json.load(open(os.path.join(self.root, "viewer", "move_h.report.json")))
        self.assertEqual(rep["hull_limit_mm"], 40.0)
        bad = self.script("hb", f"start az 0 el {self.el:+.2f}\nhull 900\narc left 10\n")
        code, evs = hs("movepreview", "-p", self.root, "--script", bad)
        self.assertEqual(code, 1)

    def test_frame_and_locate_need_no_script(self):
        code, evs = hs("movepreview", "-p", self.root, "--frame")
        self.assertEqual(code, 0, evs)
        t = json.load(open(os.path.join(self.root, "viewer", "move_frame.json")))
        self.assertEqual(len(t["captures"]), 60)
        c = np.load(self.rig)["C"][0]
        code, evs = hs("movepreview", "-p", self.root, "--locate=" + ",".join(f"{x:.3f}" for x in c))
        self.assertEqual(code, 0, evs)
        d = [e for e in evs if e.get("name") == "locate"][0]["value"]
        self.assertTrue(d["reachable"])
        self.assertEqual(d["nearest_capture"], "cap000")


if __name__ == "__main__":
    unittest.main()
