"""The Subject step's first verdict: is masking worth it?

hs solve ends by projecting the subject's sparse points (the same selection hs masks' geometric
prior makes) into every view and taking the convex hull's share of the frame. A sphere of radius
40 seen from a ring at 300 with fx 800 on a 640x480 frame projects to a disc of radius about
800 x 40 / 300 = 107 px: pi x 107^2 / (640 x 480) = 0.117 of the frame. The hull of a sparse sample
of that disc sits a little inside it.
"""
import contextlib
import io
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from hs.project import Project, now_iso  # noqa: E402
from hs.stages import solve  # noqa: E402

W, H, FX = 640, 480, 800.0
N_CAM, RING = 12, 300.0


def look_at(C):
    f = -C / np.linalg.norm(C)
    down = np.array([0.0, 1.0, 0.0])
    y = down - down.dot(f) * f
    y /= np.linalg.norm(y)
    x = np.cross(y, f)
    return np.stack([x, y, f])


def scene(root, sphere_mm, n_pts=400, room_pts=0, h=H):
    """Sparse points on a sphere at the origin (plus, optionally, room points far outside the fitted
    radius), cameras on a ring. -> (Project, rig as np.load gives it)."""
    pj = Project(root, create=True)
    for s in ("ingest", "select", "solve"):
        pj.m["stages"][s] = {"status": "done", "finished": now_iso(), "checks": [], "metrics": {}}
    rng = np.random.default_rng(3)
    v = rng.normal(size=(n_pts, 3))
    pts = v / np.linalg.norm(v, axis=1, keepdims=True) * sphere_mm
    if room_pts:
        pts = np.vstack([pts, rng.uniform(-900, 900, size=(room_pts, 3)) + np.array([0, 0, -600.0])])
    names, Ks, Rs, ts = [], [], [], []
    for i in range(N_CAM):
        a = 2 * np.pi * i / N_CAM
        C = np.array([RING * np.cos(a), -60.0, RING * np.sin(a)])
        R = look_at(C)
        names.append(f"cap{i:03d}_L")
        Ks.append([[FX, 0, W / 2], [0, FX, h / 2], [0, 0, 1]])
        Rs.append(R)
        ts.append(-R @ C)
    os.makedirs(os.path.dirname(pj.rig_npz), exist_ok=True)
    np.savez(pj.rig_npz, names=np.array(names), K=np.array(Ks), R=np.array(Rs), t=np.array(ts),
             wh=np.array([[W, h]] * N_CAM), pts=pts, stereo=False)
    pj.save()
    return pj, np.load(pj.rig_npz, allow_pickle=True)


class SubjectShare(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hs-share-")
        self._redir = contextlib.redirect_stdout(io.StringIO())
        self._redir.__enter__()

    def tearDown(self):
        self._redir.__exit__(None, None, None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_a_small_subject_in_a_room_recommends_masking(self):
        pj, G = scene(os.path.join(self.tmp, "small"), sphere_mm=40.0, room_pts=60)
        share = solve.subject_share(pj, G)
        self.assertIsNotNone(share)
        self.assertGreater(share, 0.08)
        self.assertLess(share, 0.125)
        m = pj.stage("solve")["metrics"]
        self.assertEqual(m["subject_share_of_frame"], round(share, 4))
        self.assertAlmostEqual(m["room_share_estimate"], 1.0 - share, places=3)
        self.assertEqual(m["subject_share_views"], N_CAM)
        c = {x["name"]: x for x in pj.stage("solve")["checks"]}
        self.assertTrue(c["masking_worth_it"]["ok"])
        self.assertTrue(c["masking_worth_it"]["needs_human"])
        self.assertIn("masking recommended", c["masking_worth_it"]["value"])
        self.assertIn("9 of 10 splats", c["masking_worth_it"]["value"])
        self.assertIn("share of pixels", c["masking_worth_it"]["value"])

    def test_room_points_do_not_count_as_subject(self):
        pj_a, G_a = scene(os.path.join(self.tmp, "a"), sphere_mm=40.0, room_pts=0)
        pj_b, G_b = scene(os.path.join(self.tmp, "b"), sphere_mm=40.0, room_pts=200)
        share_a, _ = solve.subject_share_of_frame(G_a)
        share_b, _ = solve.subject_share_of_frame(G_b)
        # the fitted radius (0.7 x 300 x 320 / 800 = 84) takes the sphere and leaves the room out
        self.assertAlmostEqual(share_a, share_b, delta=0.01)

    def test_a_subject_that_fills_the_frame_makes_masks_optional(self):
        # the selection is the fitted radius's (0.7 of the half-width at the ring: 84 here), so the
        # share has a ceiling of pi x 0.49 x W / (4 H): 0.51 on 4:3, 0.62 on 16:10. A sphere of 80
        # at 300 on a 640x400 frame projects to r ~ 221 px, 0.6 of the frame
        pj, G = scene(os.path.join(self.tmp, "big"), sphere_mm=80.0, h=400)
        share = solve.subject_share(pj, G)
        self.assertGreater(share, 0.5)
        c = {x["name"]: x for x in pj.stage("solve")["checks"]}
        self.assertIn("masks optional", c["masking_worth_it"]["value"])

    def test_too_few_points_is_no_verdict_not_a_crash(self):
        pj, G = scene(os.path.join(self.tmp, "few"), sphere_mm=40.0, n_pts=3)
        self.assertIsNone(solve.subject_share(pj, G))
        self.assertNotIn("subject_share_of_frame", pj.stage("solve")["metrics"])


if __name__ == "__main__":
    unittest.main()
