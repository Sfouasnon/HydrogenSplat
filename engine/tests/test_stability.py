"""hs stability: a steady move scores high, a frame that pops is named as the worst.

Synthetic only: a textured square translating 2 px/frame over a smooth background. The flow
explains that motion, so the warped neighbour matches (> 40 dB). Jump the square 6 px on one
frame, or brighten it, and that frame is the one the flow cannot explain. DIS is the backend
exercised here; RAFT needs torch, which CI does not install.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(HERE)
sys.path.insert(0, ENGINE)
sys.path.insert(0, HERE)

from hs import events  # noqa: E402
from hs.stages import stability  # noqa: E402

H, W, SQ, N = 240, 320, 80, 20
POP = 10


def sequence(pop_shift=None, pop_gain=None, n=N):
    import cv2
    rng = np.random.default_rng(1)
    tex = np.clip(cv2.resize(rng.uniform(0, 1, (12, 12, 3)).astype(np.float32), (SQ, SQ),
                             interpolation=cv2.INTER_CUBIC), 0, 1)
    bg = cv2.resize(rng.uniform(0.2, 0.5, (6, 8, 3)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    out = []
    for i in range(n):
        x = 40 + 2 * i + (pop_shift if (pop_shift and i == POP) else 0)
        im = bg.copy()
        im[80:80 + SQ, x:x + SQ] = np.clip(tex * (pop_gain if (pop_gain and i == POP) else 1.0), 0, 1)
        out.append((im * 255).round().astype(np.uint8))
    return out


def write_frames(d, frames):
    import cv2
    os.makedirs(d, exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(os.path.join(d, f"frame_{i:04d}.png"), f)
    return d


class Measure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="hs-stab-")
        cls.quiet = open(os.devnull, "w")
        cls._stdout, sys.stdout = sys.stdout, cls.quiet       # swallow the event stream
        cls.res = {}
        for key, kw in (("stable", {}), ("jump", {"pop_shift": 6}), ("flash", {"pop_gain": 1.35})):
            d = write_frames(os.path.join(cls.tmp, key), sequence(**kw))
            cls.res[key] = stability.measure(frames=d, ks=(1, 7), out_dir=os.path.join(cls.tmp, key + "_out"), name=key)
        sys.stdout = cls._stdout
        cls.quiet.close()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_stable_sequence_scores_above_40_db(self):
        s = self.res["stable"]["1"]
        self.assertEqual(s["pairs"], N - 1)
        self.assertGreater(s["warped_psnr_median"], 40.0)
        self.assertGreater(s["warped_psnr_min"], 40.0)
        self.assertLess(s["popping_pixel_fraction_p95"], 0.01)
        self.assertGreater(self.res["stable"]["7"]["warped_psnr_median"], 40.0)
        self.assertEqual(self.res["stable"]["7"]["pairs"], N - 7)

    def test_a_jumped_frame_is_the_worst_at_k1(self):
        s = self.res["jump"]["1"]
        self.assertEqual(s["worst_frames"][0]["frame"], POP)
        self.assertEqual(s["worst_frames"][0]["pairs"], [[POP - 1, POP], [POP, POP + 1]])
        self.assertEqual((s["worst_pairs"][0]["from"], s["worst_pairs"][0]["to"]), (POP - 1, POP))
        self.assertLess(s["worst_pairs"][0]["warped_psnr"], self.res["stable"]["1"]["warped_psnr_min"])

    def test_a_flash_is_popping_pixels(self):
        s = self.res["flash"]["1"]
        self.assertEqual(s["worst_frames"][0]["frame"], POP)
        self.assertGreater(s["worst_frames"][0]["popping_fraction"], 0.02)
        self.assertLess(s["worst_frames"][0]["warped_psnr"], 35.0)

    def test_outputs(self):
        meta = self.res["stable"]["meta"]
        self.assertEqual((meta["frames"], meta["size"], meta["backend"]), (N, [W, H], "dis"))
        for p in meta["outputs"].values():
            self.assertTrue(os.path.getsize(p) > 0, p)
        rows = open(meta["outputs"]["csv"]).read().splitlines()
        self.assertEqual(len(rows), 1 + (N - 1) + (N - 7))
        j = json.load(open(meta["outputs"]["json"]))
        self.assertEqual(set(j), {"1", "7", "meta"})
        for k in ("warped_psnr_median", "warped_mse_median", "popping_pixel_fraction_p95", "worst_frames"):
            self.assertIn(k, j["1"])

    def test_torch_is_not_imported_by_dis(self):
        self.assertNotIn("torch", sys.modules)


class Units(unittest.TestCase):
    def test_identical_frames_are_perfect(self):
        f = sequence(n=1)[0]
        m = stability.pair_metrics(f, f, stability.flow_backend("dis"))
        self.assertGreater(m["warped_psnr"], 60)
        self.assertEqual(m["popping_fraction"], 0.0)
        self.assertGreater(m["valid_fraction"], 0.99)

    def test_stride_and_k_parsing(self):
        self.assertEqual(stability.parse_k("7, 1,1"), [1, 7])
        for bad in ("0", "x", ""):
            with self.assertRaises(events.StageError):
                stability.parse_k(bad)

    def test_backends_report_without_importing_torch(self):
        b = stability.available_backends()
        self.assertTrue(b["dis"][0])
        self.assertIn("raft", b)
        self.assertNotIn("torch", sys.modules)

    @unittest.skipIf(__import__("importlib.util").util.find_spec("torch") is not None, "torch is installed")
    def test_raft_without_torch_says_how_to_get_it(self):
        with self.assertRaises(events.StageError) as cm:
            stability.flow_backend("raft")
        self.assertIn("metrics", cm.exception.hint)


class Cli(unittest.TestCase):
    def test_free_stage_frames_and_stride(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = write_frames(os.path.join(tmp, "boom"), sequence(n=9))
            out = os.path.join(tmp, "o")
            r = subprocess.run([sys.executable, "-m", "hs", "stability", "--frames", d, "--k", "1,2",
                                "--stride", "2", "--out", out], cwd=ENGINE, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            evs = [json.loads(l) for l in r.stdout.splitlines()]
            names = {e.get("name") for e in evs if e["ev"] == "metric"}
            self.assertIn("warped_psnr_median_k1", names)
            self.assertIn("popping_pixel_fraction_p95_k2", names)
            s = json.load(open(os.path.join(out, "boom_stability.json")))
            self.assertEqual(s["1"]["pairs"], 4)            # t = 0, 2, 4, 6
            self.assertEqual(s["2"]["pairs"], 4)            # t = 0, 2, 4, 6 (t + 2 <= 8)
            self.assertEqual(evs[-1], {"ev": "done", "stage": "stability", "exit": 0})

    def test_tools_reports_flow_backends(self):
        r = subprocess.run([sys.executable, "-m", "hs", "tools", "--brush", "/nonexistent", "--render-bin", "/nonexistent"],
                           cwd=ENGINE, capture_output=True, text=True)
        checks = {e["name"]: e for e in map(json.loads, r.stdout.splitlines()) if e["ev"] == "check"}
        self.assertTrue(checks["flow_dis"]["ok"])
        self.assertIn("flow_raft", checks)

    def test_video_input(self):
        import cv2
        with tempfile.TemporaryDirectory() as tmp:
            vid = os.path.join(tmp, "clip.avi")
            wr = cv2.VideoWriter(vid, cv2.VideoWriter_fourcc(*"MJPG"), 30, (W, H))
            if not wr.isOpened():
                self.skipTest("this cv2 build cannot write MJPG")
            for f in sequence(n=8):
                wr.write(f)
            wr.release()
            quiet = open(os.devnull, "w")
            old, sys.stdout = sys.stdout, quiet
            try:
                s = stability.measure(video=vid, ks=(1,), out_dir=os.path.join(tmp, "o"))
            finally:
                sys.stdout = old
                quiet.close()
            self.assertEqual(s["meta"]["frames"], 8)
            self.assertEqual(s["1"]["pairs"], 7)
            self.assertGreater(s["1"]["warped_psnr_median"], 30.0)   # MJPG costs a few dB


@unittest.skipUnless(shutil.which("ffmpeg"), "hs render encodes with ffmpeg")
class RenderHook(unittest.TestCase):
    """hs render --stability runs the measure on the PNGs before they are deleted."""

    def test_render_stability_records_metrics_and_drops_frames(self):
        from test_lifecycle import Base
        from hs.stages import render

        class T(Base):
            def runTest(self):
                pass
        t = T()
        t.setUp()
        try:
            os.environ["HS_PYTHON"] = sys.executable
            pj, final = t.trained_project()
            os.makedirs(pj.path("move"))
            json.dump({"fps": 30, "width": 320, "height": 240, "frames": [{"c2w": np.eye(4).tolist()}] * 12},
                      open(pj.path("move", "boom.json"), "w"))
            a = Namespace(move="boom", ply=None, name=None, width=320, keep_frames=False, no_crop=True, crf=17,
                          render_bin=os.path.join(HERE, "fakebin", "brush-path-render"), ffmpeg="ffmpeg",
                          allow_mismatch=False, stability=True, stability_k="1", stability_backend="dis")
            render.run(a, pj)
            m = pj.stage("render")["metrics"]
            self.assertIn("stability_warped_psnr_median_k1", m)
            self.assertIn("stability_worst_frames_k1", m)
            self.assertTrue(os.path.exists(pj.path("render", "boom_stability", "boom_stability.json")))
            self.assertFalse(os.path.exists(pj.path("render", "boom")), "frames still deleted without --keep-frames")
        finally:
            t.tearDown()


if __name__ == "__main__":
    unittest.main()
