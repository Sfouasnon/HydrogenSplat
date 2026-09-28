"""select_frames.py --hdr: HLG clips (iPhone "HDR Video") decoded by ffmpeg, one fixed curve.

The probe parsing and the curve are pure. The clip tests write a tiny 10-bit ProRes tagged
BT.2020 / HLG (bands sliding at different speeds, as test_select_keyframes does, plus a patch at a
known HLG signal level) and run the script on it; they skip without ffmpeg/ffprobe or prores_ks.

Run: cd engine && ../.venv/bin/python -W ignore -m unittest tests.test_select_hdr
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hs import select_frames as sf  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "hs", "select_frames.py")

# ffprobe -v error -select_streams v:0 -show_streams -of json on IMG_2525.MOV (iPhone 17 Pro Max,
# iOS 26.6.1, 4K30 HDR, portrait), trimmed to the fields that matter
IPHONE_HLG = json.dumps({"streams": [{
    "codec_name": "hevc", "width": 3840, "height": 2160, "pix_fmt": "yuv420p10le",
    "color_space": "bt2020nc", "color_transfer": "arib-std-b67", "color_primaries": "bt2020",
    "avg_frame_rate": "1466400/48881", "nb_frames": "2444", "tags": {"rotate": "90"},
    "side_data_list": [{"side_data_type": "DOVI configuration record", "dv_profile": 8},
                       {"side_data_type": "Display Matrix", "rotation": -90}]}]})


def have_tools():
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        return False
    enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return "prores_ks" in enc


def make_hlg_clip(path, n=40, size=(320, 180), patch_signal=0.97, rotate=None):
    """Bands sliding at different speeds, as 16-bit RGB HLG signal, encoded 10-bit and tagged
    BT.2020 / HLG. The top-left 40x40 is a neutral patch held at `patch_signal` (E')."""
    import cv2
    w, h = size
    rng = np.random.default_rng(3)
    speeds = [1.0, 3.0, 0.5, 2.5, 1.5, 3.5]
    bh = h // len(speeds)
    tex = [cv2.GaussianBlur(rng.integers(0, 256, (bh, w + int(s * n) + 8)).astype(np.float32), (0, 0), 1.5)
           for s in speeds]
    argv = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{w}x{h}",
            "-r", "30", "-i", "-", "-vf", "scale=out_color_matrix=bt2020:out_range=tv,format=yuv422p10le",
            "-c:v", "prores_ks", "-profile:v", "3", "-colorspace", "bt2020nc", "-color_primaries", "bt2020",
            "-color_trc", "arib-std-b67", "-color_range", "tv"]
    final, path = path, (path + ".tmp.mov" if rotate is not None else path)
    p = subprocess.Popen(argv + [path], stdin=subprocess.PIPE)
    for t in range(n):
        img = np.zeros((h, w), np.float32)
        for b, s in enumerate(speeds):
            x = int(round(s * t))
            img[b * bh:(b + 1) * bh] = tex[b][:, x:x + w]
        sig = 0.15 + 0.6 * img / 255.0                   # E' 0.15..0.75: the texture below diffuse white
        sig[:40, :40] = patch_signal
        rgb = np.repeat(np.round(sig * 65535).astype("<u2")[:, :, None], 3, axis=2)
        p.stdin.write(rgb.tobytes())
    p.stdin.close()
    assert p.wait() == 0
    if rotate is not None:
        # a display matrix on a stream copy: ffmpeg >= 6.1 takes -display_rotation on the input (counter-
        # clockwise degrees); older ones the legacy clockwise rotate tag
        full = subprocess.run(["ffmpeg", "-hide_banner", "-h", "full"], capture_output=True, text=True).stdout
        rot = (["-display_rotation", str(-rotate), "-i", path] if "display_rotation" in full
               else ["-i", path, "-metadata:s:v:0", f"rotate={rotate}"])
        subprocess.run(["ffmpeg", "-v", "error", "-y"] + rot + ["-c", "copy", final], check=True)
        os.remove(path)


def run(clip, out, *extra):
    argv = [sys.executable, SCRIPT, clip, "-o", out, "--mono", "--work-width", "160", "--residual", "0.8"] + list(extra)
    r = subprocess.run(argv, capture_output=True, text=True)
    if r.returncode != 0:
        raise AssertionError(f"select_frames.py failed:\n{r.stdout}\n{r.stderr}")
    return json.load(open(os.path.join(out, "selection.json"))), r.stdout


class Probe(unittest.TestCase):
    def test_iphone_portrait_hlg(self):
        p = sf.parse_probe(IPHONE_HLG)
        self.assertEqual((p["width"], p["height"]), (2160, 3840))     # displayed: portrait
        self.assertEqual(p["rotation"], -90.0)
        self.assertEqual(sf.hdr_kind(p), "hlg")
        self.assertAlmostEqual(p["fps"], 29.9997, places=3)
        self.assertEqual(p["nb_frames"], 2444)

    def test_legacy_rotate_tag_alone(self):
        s = json.loads(IPHONE_HLG)
        del s["streams"][0]["side_data_list"]
        p = sf.parse_probe(json.dumps(s))
        self.assertEqual((p["width"], p["height"]), (2160, 3840))
        self.assertEqual(p["rotation"], -90.0)

    def test_sdr_and_garbage(self):
        s = json.loads(IPHONE_HLG)["streams"][0]
        s.update(color_transfer="bt709", tags={}, side_data_list=[])
        p = sf.parse_probe(json.dumps({"streams": [s]}))
        self.assertEqual((p["width"], p["height"]), (3840, 2160))
        self.assertIsNone(sf.hdr_kind(p))
        self.assertEqual(sf.hdr_kind(sf.parse_probe(json.dumps({"streams": [dict(s, color_transfer="smpte2084")]}))), "pq")
        self.assertIsNone(sf.parse_probe("I,\nP\n"))                 # the stand-in ffprobe's output
        self.assertIsNone(sf.hdr_kind(None))


class Curve(unittest.TestCase):
    def test_oetf_round_trip(self):
        e = np.linspace(0, 1, 1001)
        np.testing.assert_allclose(sf.hlg_inv_oetf(sf.hlg_oetf(e)), e, atol=1e-9)
        self.assertAlmostEqual(float(sf.hlg_oetf(1.0 / 12.0)), 0.5, places=6)
        self.assertAlmostEqual(float(sf.hlg_oetf(1.0)), 1.0, places=6)

    def test_neutrals_are_the_signal(self):
        """A neutral keeps its HLG code: the gamut matrix leaves grey alone (D65 both sides)."""
        codes = np.arange(0, 1024, 7) / 1023.0
        rgb = np.repeat(np.round(codes * 65535).astype(np.uint16)[None, :, None], 3, axis=2)
        out = sf.hlg_rgb16_to_bgr8(rgb)[0]
        np.testing.assert_allclose(out[:, 1], np.round(codes * 255), atol=1)
        self.assertTrue((np.diff(out[:, 1].astype(int)) >= 0).all())
        self.assertTrue((np.abs(out[:, 0].astype(int) - out[:, 2]) <= 1).all())

    def test_gamut_and_channel_order(self):
        """BT.2020 red is outside BT.709: it clips to full red, and lands in the R of BGR."""
        px = np.array([[[65535, 0, 0], [0, 65535, 0]]], np.uint16)       # RGB: 2020 red, 2020 green
        out = sf.hlg_rgb16_to_bgr8(px)[0]
        self.assertEqual(out[0].tolist(), [0, 0, 255])
        self.assertEqual(out[1][1], 255)
        self.assertEqual(out[1][0], 0)
        self.assertEqual(out[1][2], 0)

    def test_gray8_of_both_kinds(self):
        self.assertEqual(int(sf.gray8(np.full((2, 2, 3), 65535, np.uint16))[0, 0]), 255)
        self.assertEqual(int(sf.gray8(np.full((2, 2, 3), 257 * 100, np.uint16))[0, 0]), 100)
        self.assertEqual(int(sf.gray8(np.full((2, 2, 3), 100, np.uint8))[0, 0]), 100)


@unittest.skipUnless(have_tools(), "needs ffmpeg + ffprobe with prores_ks")
class Clip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_auto_decodes_hlg_and_keeps_the_top_end(self):
        clip = os.path.join(self.tmp, "hlg.mov")
        make_hlg_clip(clip, patch_signal=0.97)
        sel, out = run(clip, os.path.join(self.tmp, "o"))
        self.assertEqual(sel["decoder"]["path"], "ffmpeg")
        self.assertEqual(sel["decoder"]["hdr"], "hlg")
        self.assertIn("HLG HDR decoded by ffmpeg", out)
        self.assertGreaterEqual(len(sel["selected"]), 3)
        import cv2
        first = sorted(f for f in os.listdir(os.path.join(self.tmp, "o")) if f.endswith(".jpg"))[0]
        im = cv2.imread(os.path.join(self.tmp, "o", first))
        self.assertEqual(im.shape[:2], (180, 320))
        patch = float(np.median(im[5:35, 5:35]))
        self.assertTrue(243 <= patch <= 251, patch)                  # E' 0.97 -> 247, not clipped to 255
        self.assertEqual(sel["frames_total"], 40)

    def test_off_is_opencv(self):
        clip = os.path.join(self.tmp, "hlg.mov")
        make_hlg_clip(clip)
        sel, out = run(clip, os.path.join(self.tmp, "o"), "--hdr", "off")
        self.assertEqual(sel["decoder"]["path"], "opencv")
        self.assertIn("warning: this clip is HLG", out)

    def test_rotation_is_applied(self):
        clip = os.path.join(self.tmp, "rot.mov")
        make_hlg_clip(clip, rotate=90)
        pr = sf.parse_probe(subprocess.run(sf.probe_argv("ffprobe", clip), capture_output=True, text=True).stdout)
        if not pr["rotation"]:
            self.skipTest("this ffmpeg does not write the rotate tag")
        sel, _ = run(clip, os.path.join(self.tmp, "o"), "--residual", "0.3")
        self.assertEqual(sel["decoder"]["size"], [180, 320])
        import cv2
        first = sorted(f for f in os.listdir(os.path.join(self.tmp, "o")) if f.endswith(".jpg"))[0]
        self.assertEqual(cv2.imread(os.path.join(self.tmp, "o", first)).shape[:2], (320, 180))

    def test_end_stops_cleanly(self):
        clip = os.path.join(self.tmp, "hlg.mov")
        make_hlg_clip(clip)
        sel, _ = run(clip, os.path.join(self.tmp, "o"), "--end", "12")
        self.assertEqual(sel["frames_total"], 14)      # frame 13 is read, then the loop stops (as the OpenCV path)


if __name__ == "__main__":
    unittest.main()
