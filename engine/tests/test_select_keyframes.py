"""select_frames.py --keyframes / --highlight-knee, and what hs select reports about them.

The parsing and the knee table are pure; the selection tests write a tiny 2x1 H.264 Baseline
clip with a known GOP (ffmpeg -g 10, 60 frames of bands sliding at different speeds, so the
parallax residual rises) and run the script on it. They skip without ffmpeg/ffprobe.

Run: cd engine && ../.venv/bin/python -W ignore -m unittest tests.test_select_keyframes
"""
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hs import frame_quality, select_frames as sf  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "hs", "select_frames.py")
FAKE_FFPROBE = os.path.join(HERE, "fakebin", "ffprobe")

# ffprobe -v error -select_streams v:0 -show_entries frame=pict_type -of csv=p=0, as ffmpeg 6.1
# prints it for a GOP-10 Baseline clip: the first I-frame carries side data, hence "I,"
CAPTURED = "I,\nP\nP\nP\nP\nP\nP\nP\nP\nP\nI\nP\nP\nP\nP\nP\nP\nP\nP\nP\nI\nP\nP\n\n"


def have_tools():
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        return False
    enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return "libx264" in enc


def make_clip(path, n=60, gop=10, eye=(320, 180), mono=False, hot=()):
    """Six horizontal bands sliding at different speeds (depth no homography absorbs); a
    frame in `hot` gets its top third burnt out to 255."""
    import cv2
    w, h = eye
    rng = np.random.default_rng(7)
    speeds = [1.0, 3.0, 0.5, 2.5, 1.5, 3.5]
    bh = h // len(speeds)
    tex = [cv2.GaussianBlur(rng.integers(0, 255, (bh, w + int(s * n) + 8), dtype=np.uint8), (0, 0), 1.5)
           for s in speeds]
    W = w if mono else 2 * w
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{h}",
                          "-r", "30", "-i", "-", "-c:v", "libx264", "-profile:v", "baseline", "-g", str(gop),
                          "-keyint_min", str(gop), "-sc_threshold", "0", "-pix_fmt", "yuv420p", path],
                         stdin=subprocess.PIPE)
    for t in range(n):
        img = np.zeros((h, w), np.uint8)
        for b, s in enumerate(speeds):
            x = int(round(s * t))
            img[b * bh:(b + 1) * bh] = tex[b][:, x:x + w]
        if t in hot:
            img[: h // 3] = 255
        bgr = cv2.merge([img, img, img])
        p.stdin.write((bgr if mono else np.hstack([bgr, bgr])).tobytes())
    p.stdin.close()
    assert p.wait() == 0


def run_select(clip, out, *extra, env=None):
    # --residual 0.8: the crossing must land well before the next keyframe on every platform (with
    # 1.5 it fell on frame 20 or 21 depending on the libx264 / OpenCV build — the Mac read frame 30
    # as the first candidate after pick 10, the container frame 20)
    argv = [sys.executable, SCRIPT, clip, "-o", out, "--work-width", "160", "--residual", "0.8"] + [str(x) for x in extra]
    r = subprocess.run(argv, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise AssertionError(f"select_frames.py failed:\n{r.stdout}\n{r.stderr}")
    return json.load(open(os.path.join(out, "selection.json"))), r.stdout


class PictTypeParsing(unittest.TestCase):
    def test_captured_sample(self):
        types = sf.parse_pict_types(CAPTURED)
        self.assertEqual(len(types), 23)                      # the blank line is not a frame
        self.assertEqual(sf.keyframe_indices(types), [0, 10, 20])
        self.assertEqual(sf.gop_median([0, 10, 20]), 10.0)
        info = sf.keyframe_info(types)
        self.assertEqual((info["total"], info["gop_median"], info["probe_frames"]), (3, 10.0, 23))

    def test_only_I_is_a_keyframe(self):
        self.assertEqual(sf.keyframe_indices(sf.parse_pict_types("I\nP\nB\nSI\nI,\n")), [0, 4])
        self.assertIsNone(sf.gop_median([5]))
        # -show_entries frame=key_frame,pict_type prints "1,I": the picture type still decides
        self.assertEqual(sf.parse_pict_types("1,I,\n0,P\n"), ["I", "P"])

    def test_anything_else_is_an_error_not_a_clip_without_keyframes(self):
        fake = json.dumps({"streams": [{"codec_type": "video"}]})   # what the stand-in ffprobe prints
        with self.assertRaises(ValueError):
            sf.parse_pict_types(fake)
        with self.assertRaises(ValueError):
            sf.parse_pict_types("1\n0\n0\n")                         # frame=key_frame, not pict_type

    def test_argv(self):
        self.assertEqual(sf.ffprobe_argv("ffprobe", "c.mp4"),
                         ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pict_type",
                          "-of", "csv=p=0", "c.mp4"])

    def test_thresholds_match_the_quality_report(self):
        self.assertEqual(sf.SOFT_REL, frame_quality.SOFT_REL)
        self.assertEqual(sf.CRUSH, frame_quality.CRUSH)


class HighlightKnee(unittest.TestCase):
    def test_table(self):
        for k in (0.5, 0.8, 0.85, 0.9, 0.99):
            lut = sf.knee_lut(k)
            self.assertEqual((lut.dtype, lut.shape), (np.uint8, (256,)))
            self.assertTrue(np.all(np.diff(lut.astype(int)) >= 0), f"K={k} not monotonic")
            codes = np.arange(256)
            below = sf.srgb_to_linear(codes / 255.0) <= k
            self.assertTrue(np.array_equal(lut[below], codes[below]), f"K={k} touches values below the knee")
            self.assertLessEqual(int(lut[255]), 255)
            self.assertGreater(int(lut[255]), k * 255)
            self.assertTrue(np.all(lut <= codes), "a knee only ever lowers a value")
        self.assertEqual(int(sf.knee_lut(0.85)[255]), 249)       # the number the docs quote
        self.assertLess(int(sf.knee_lut(0.85)[250]), 250)          # 250+ leaves the clip band

    def test_same_curve_on_every_channel(self):
        lut = sf.knee_lut(0.85)
        img = np.zeros((2, 3, 3), np.uint8)
        img[..., 0], img[..., 1], img[..., 2] = 255, 240, 100
        img[1, 2] = (60, 250, 255)
        out = sf.apply_knee(img, lut)
        self.assertTrue(np.array_equal(out, lut[img]))
        self.assertEqual(out[0, 0].tolist(), [lut[255], lut[240], 100])
        self.assertIs(sf.apply_knee(img, None), img)

    def test_out_of_range(self):
        for k in (0.0, 1.0, -0.2, 1.5):
            with self.assertRaises(ValueError):
                sf.knee_lut(k)


class QualityReportInKeyframeMode(unittest.TestCase):
    """frame_quality on a hand-made keyframe selection: no clip needed."""

    def selection(self):
        n = 200
        tr = {"frame": list(range(n)), "sharp": [1000.0] * n, "std": [40.0] * n, "luma": [0.2] * n,
              "clip": [0.0] * n, "dark": [0.0] * n, "residual": [None] * n}
        tr["sharp"][25] = 5000.0                   # a P-frame with a far higher Laplacian (artefacts)
        tr["sharp"][30] = 900.0
        sel = [{"sel": 0, "frame": 0, "residual": 0.0, "sharpness": 1000.0, "clip": 0.0, "keyframe": True},
               {"sel": 1, "frame": 30, "residual": 1.6, "sharpness": 900.0, "clip": 0.0, "gap": 30,
                "keyframe": True, "crossing": 12, "trigger": "parallax"},
               # a parallax pick whose gap overshot max-gap waiting for the I-frame: not "stood still"
               {"sel": 2, "frame": 150, "residual": 1.7, "sharpness": 1000.0, "clip": 0.0, "gap": 120,
                "keyframe": True, "crossing": 118, "trigger": "parallax"},
               {"sel": 3, "frame": 180, "residual": 0.2, "sharpness": 1000.0, "clip": 0.0, "gap": 30,
                "keyframe": True, "crossing": 179, "trigger": "max-gap", "quality_fallback": "none of 2 …"}]
        return {"fps": 30.0, "frames_total": n, "selected": sel, "trace": tr,
                "params": {"max_clip": 0.02, "max_gap": 90, "keyframes": True, "mode": "keyframes"},
                "keyframes": {"frames": list(range(0, n, 30)), "total": 7, "gop_median": 30.0}}

    def test_flags_and_summary(self):
        q = frame_quality.analyse(self.selection())
        by = {f["frame"]: f for f in q["frames"]}
        self.assertIsNone(by[30]["sharper_nearby"])           # frame 25 is a P-frame: not offered
        self.assertNotIn("stood_still", by[150]["flags"])
        self.assertIn("stood_still", by[180]["flags"])
        self.assertEqual(by[180]["quality_fallback"], "none of 2 …")
        self.assertTrue(by[0]["keyframe"])
        k = q["keyframes"]
        self.assertEqual((k["mode"], k["picks"], k["picks_keyframes"], k["keyframes_total"], k["gop_median"],
                          k["quality_fallbacks"]), ("keyframes", 4, 4, 7, 30.0, 1))

    def test_default_mode_keeps_the_old_rules(self):
        s = self.selection()
        s["params"] = {"max_clip": 0.02, "max_gap": 90}
        for p in s["selected"]:
            for key in ("trigger", "crossing"):
                p.pop(key, None)
        q = frame_quality.analyse(s)
        by = {f["frame"]: f for f in q["frames"]}
        self.assertEqual(by[30]["sharper_nearby"]["frame"], 25)
        self.assertIn("stood_still", by[150]["flags"])        # gap 120 >= max-gap, as before
        self.assertEqual(q["keyframes"]["mode"], "parallax")


@unittest.skipUnless(have_tools(), "needs ffmpeg with libx264 and ffprobe")
class KeyframeSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="hs-kf-")
        cls.clip = os.path.join(cls.tmp, "clip.mp4")
        make_clip(cls.clip)
        cls.hot = os.path.join(cls.tmp, "hot.mp4")
        make_clip(cls.hot, hot=(20,))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def out(self, name):
        return os.path.join(self.tmp, name)

    def test_real_ffprobe_finds_the_gop(self):
        r = subprocess.run(sf.ffprobe_argv("ffprobe", self.clip), capture_output=True, text=True, check=True)
        self.assertEqual(sf.keyframe_indices(sf.parse_pict_types(r.stdout)), list(range(0, 60, 10)))

    def check_keyframe_picks(self, sel, min_gap=6, residual=0.8, max_gap=90):
        kf = sel["keyframes"]["frames"]
        self.assertEqual(kf, list(range(0, 60, 10)))
        self.assertEqual(sel["params"]["mode"], "keyframes")
        self.assertTrue(sel["params"]["keyframes"])
        tr = dict(zip(sel["trace"]["frame"], sel["trace"]["residual"]))
        picks = sel["selected"]
        self.assertGreaterEqual(len(picks), 3)
        self.assertEqual(picks[0]["frame"], 0)                     # the first pick is a keyframe too
        for prev, s in zip(picks, picks[1:]):
            self.assertIn(s["frame"], kf)
            self.assertTrue(s["keyframe"])
            c = s["crossing"]
            # the parallax rule decided WHEN: the crossing respects min-gap and is a real crossing
            self.assertGreaterEqual(c - prev["frame"], min_gap)
            if s["trigger"] == "parallax":
                self.assertGreaterEqual(tr[c], residual)
            elif s["trigger"] == "max-gap":
                self.assertGreaterEqual(c - prev["frame"], max_gap)
            # ...and the pick is a keyframe at or after it, the first one unless an earlier one failed
            self.assertGreaterEqual(s["frame"], c)
            first_after = min(f for f in kf if f >= c)
            self.assertEqual(s["candidates"][0][0], first_after)
            self.assertEqual(s["candidates"][-1][0], s["frame"])
            self.assertEqual(s["gap"], s["frame"] - prev["frame"])
        self.assertTrue(picks[0]["keyframe"])

    def test_every_pick_is_a_keyframe_and_parallax_still_decides_when(self):
        sel, log = run_select(self.clip, self.out("kf"), "--keyframes")
        self.check_keyframe_picks(sel)
        self.assertNotIn("quality_fallback", json.dumps(sel["selected"]))
        # the trace still covers every frame read, not only the keyframes
        self.assertEqual(sel["trace"]["frame"], list(range(60)))
        self.assertTrue(os.path.exists(os.path.join(self.out("kf"), "VID_000_0000_2x1.jpg")))
        self.assertIn("keyframe", log)

    def test_a_failing_keyframe_passes_the_pick_to_the_next(self):
        sel, _ = run_select(self.hot, self.out("hot"), "--keyframes", "--dry-run")
        self.check_keyframe_picks(sel)
        frames = [s["frame"] for s in sel["selected"]]
        self.assertNotIn(20, frames)
        skip = next(s for s in sel["selected"][1:] if s["candidates"][0][0] == 20)
        self.assertEqual([c[0] for c in skip["candidates"]], [20, 30])
        self.assertEqual(skip["frame"], 30)

    def test_none_passes_takes_the_least_bad_and_says_why(self):
        sel, _ = run_select(self.hot, self.out("hot1"), "--keyframes", "--search-keyframes", "1", "--dry-run")
        s = next(s for s in sel["selected"] if s["frame"] == 20)
        self.assertTrue(s["keyframe"])
        self.assertIn("clip", s["quality_fallback"])
        self.assertIn("least bad", s["quality_fallback"])

    def test_default_mode_marks_keyframes_but_does_not_depend_on_them(self):
        sel, _ = run_select(self.clip, self.out("def"), "--dry-run")
        self.assertEqual(sel["params"]["mode"], "parallax")
        self.assertEqual(sel["keyframes"]["frames"], list(range(0, 60, 10)))
        for s in sel["selected"]:
            self.assertEqual(s["keyframe"], s["frame"] % 10 == 0)
        # with a stand-in ffprobe that cannot list frame types: same picks, keyframe unknown
        env = dict(os.environ, HS_FFPROBE=FAKE_FFPROBE, HS_PYTHON=sys.executable)
        blind, _ = run_select(self.clip, self.out("def-blind"), "--dry-run", env=env)
        self.assertIsNone(blind["keyframes"])
        self.assertTrue(all(s["keyframe"] is None for s in blind["selected"]))
        strip = lambda ss: [{k: v for k, v in s.items() if k != "keyframe"} for s in ss]  # noqa: E731
        self.assertEqual(strip(sel["selected"]), strip(blind["selected"]))
        self.assertEqual(sel["trace"], blind["trace"])

    def test_missing_ffprobe_is_a_clear_error(self):
        argv = [sys.executable, SCRIPT, self.clip, "-o", self.out("noprobe"), "--keyframes",
                "--ffprobe", os.path.join(self.tmp, "no-such-ffprobe")]
        r = subprocess.run(argv, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("ffprobe", r.stderr)
        self.assertIn("brew install ffmpeg", r.stderr)

    def test_mono_keyframes(self):
        mono = os.path.join(self.tmp, "mono.mp4")
        make_clip(mono, mono=True)
        sel, _ = run_select(mono, self.out("mono"), "--keyframes", "--mono")
        self.check_keyframe_picks(sel)
        self.assertTrue(os.path.exists(os.path.join(self.out("mono"), "sel000-00000.jpg")))

    def test_knee_changes_the_written_pixels_not_the_picks(self):
        import cv2
        plain, _ = run_select(self.hot, self.out("plain"), "--keyframes", "--search-keyframes", "1")
        knee, _ = run_select(self.hot, self.out("knee"), "--keyframes", "--search-keyframes", "1",
                             "--highlight-knee", "0.85")
        self.assertEqual(plain["selected"], knee["selected"])
        self.assertEqual(knee["params"]["highlight_knee"], 0.85)
        s = next(s for s in knee["selected"] if s["frame"] == 20)
        name = f"VID_{s['sel']:03d}_0020_2x1.jpg"
        a = cv2.imread(os.path.join(self.out("plain"), name))
        b = cv2.imread(os.path.join(self.out("knee"), name))
        top = slice(5, 50)                                          # inside the burnt-out third
        pa, pb = int(np.median(a[top])), int(np.median(b[top]))
        self.assertGreaterEqual(pa, 250)                           # burnt out in the source (JPEG: ~251-255)
        self.assertLess(pb, 250)                                   # off the clip band after the knee
        self.assertLessEqual(abs(pb - int(sf.knee_lut(0.85)[pa])), 2)
        # both eyes get the same curve
        half = b.shape[1] // 2
        self.assertLessEqual(abs(int(np.median(b[top, :half])) - int(np.median(b[top, half:]))), 1)


@unittest.skipUnless(have_tools(), "needs ffmpeg with libx264 and ffprobe")
class SelectStageReport(unittest.TestCase):
    """hs select end to end on the synthetic clip: quality.json and the checks."""

    def setUp(self):
        from hs.project import Project
        self.tmp = tempfile.mkdtemp(prefix="hs-kfstage-")
        self.pj = Project(os.path.join(self.tmp, "proj"), create=True)
        os.makedirs(self.pj.path("source"))
        make_clip(self.pj.path("source", "clip.mp4"))
        self.pj.m["source"] = {"clip": "source/clip.mp4", "probe": {"nb_frames": 60}}
        self.pj.m["stages"]["ingest"] = {"status": "done"}
        self.pj.save()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_stage(self, **kw):
        from hs.project import Project
        from hs.stages import select
        a = Namespace(residual=1.5, min_gap=6, max_gap=90, search=4, max_clip=0.02, work_width=160,
                      start=0, end=-1, dry_run=False, keyframes=False, search_keyframes=2, min_sharp_rel=0.6,
                      highlight_knee=None, ffprobe="ffprobe")
        for k, v in kw.items():
            setattr(a, k, v)
        with contextlib.redirect_stdout(io.StringIO()):
            select.run(a, self.pj)
        pj = Project(self.pj.root)
        st = pj.stage("select")
        return st, {c["name"]: c for c in st["checks"]}, json.load(open(pj.path("select", "quality.json")))

    def test_keyframes_and_knee_are_reported(self):
        st, checks, q = self.run_stage(keyframes=True, highlight_knee=0.85)
        self.assertEqual(st["status"], "done")
        self.assertTrue(checks["keyframes_used"]["ok"])
        self.assertIn("GOP 10", checks["keyframes_used"]["value"])
        self.assertEqual(checks["highlight_knee_applied"]["value"], f"K=0.85 on {q['keyframes']['picks']} frames")
        self.assertTrue(checks["median_gap_in_range"]["ok"])        # judged in GOPs in keyframe mode
        m = st["metrics"]
        self.assertEqual((m["keyframes_total"], m["gop_median"], m["highlight_knee"]), (6, 10.0, 0.85))
        self.assertEqual(m["keyframes_picked"], m["frames_selected"])
        self.assertEqual(m["selection_mode"], "keyframes")
        k = q["keyframes"]
        self.assertEqual((k["mode"], k["keyframes_total"], k["gop_median"]), ("keyframes", 6, 10.0))
        self.assertEqual(k["picks_keyframes"], k["picks"])
        self.assertTrue(all(f["keyframe"] for f in q["frames"]))
        # a P-frame is never offered as the "sharper frame nearby" in keyframe mode
        for f in q["frames"]:
            if f["sharper_nearby"]:
                self.assertEqual(f["sharper_nearby"]["frame"] % 10, 0)

    def test_default_run_counts_keyframes_without_the_new_checks(self):
        st, checks, q = self.run_stage()
        self.assertNotIn("keyframes_used", checks)
        self.assertNotIn("highlight_knee_applied", checks)
        self.assertEqual(q["keyframes"]["mode"], "parallax")
        self.assertEqual(q["keyframes"]["keyframes_total"], 6)
        self.assertEqual(st["metrics"]["keyframes_picked"],
                         sum(1 for f in q["frames"] if f["frame"] % 10 == 0))

    def test_keyframes_without_ffprobe_fails_before_running(self):
        from hs import events
        with self.assertRaises(events.StageError) as e:
            self.run_stage(keyframes=True, ffprobe=os.path.join(self.tmp, "no-ffprobe"))
        self.assertIn("ffprobe", str(e.exception))


if __name__ == "__main__":
    unittest.main()
