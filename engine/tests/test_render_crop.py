"""hs render's 4:5 crop, for frames that lie either way.

The crop was written for the Hydrogen's 16:9 frames: fit the height to 1350, cut the sides. An
iPhone clip shot upright renders 2400 x 4276; fitted by height it is 758 px wide and ffmpeg
refuses to cut 1080 out of it ("Invalid too big or non positive size for width '1080'"), after
the frames and the 1920 mp4 were already made (Stormtrooper push-in, 2026-10-02).
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

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from hs import events  # noqa: E402
from hs.project import Project  # noqa: E402
from hs.stages import render  # noqa: E402


class Filter(unittest.TestCase):
    def test_a_wide_frame_is_fitted_by_height(self):
        for wh in ((2400, 1350), (1920, 1080), (1000, 1000), (1080, 1350)):
            self.assertEqual(render.crop_4x5_filter(*wh), "scale=-2:1350,crop=1080:1350", wh)

    def test_an_upright_frame_is_fitted_by_width(self):
        for wh in ((2400, 4276), (2137, 3807), (1080, 1352), (1080, 1920)):
            self.assertEqual(render.crop_4x5_filter(*wh), "scale=1080:-2,crop=1080:1350", wh)

    def test_the_scaled_frame_always_covers_the_crop(self):
        """What ffmpeg checks: after the scale, at least 1080 x 1350 is there to cut."""
        rng = np.random.default_rng(0)
        for w, h in rng.integers(200, 5000, (500, 2)):
            vf = render.crop_4x5_filter(int(w), int(h))
            if vf.startswith("scale=-2:1350"):
                sw, sh = w * 1350.0 / h, 1350
            else:
                sw, sh = 1080, h * 1080.0 / w
            # -2 rounds to the nearest even number: never more than one pixel under
            self.assertGreaterEqual(2 * round(sw / 2), 1080, (w, h))
            self.assertGreaterEqual(2 * round(sh / 2), 1350, (w, h))

    def test_png_size(self):
        import cv2
        tmp = tempfile.mkdtemp(prefix="hs-crop-")
        try:
            p = os.path.join(tmp, "f.png")
            cv2.imwrite(p, np.zeros((428, 240, 3), np.uint8))
            self.assertEqual(render.png_size(p), (240, 428))
            with open(p, "wb") as f:
                f.write(b"not a png at all, just text")
            with self.assertRaises(events.StageError):
                render.png_size(p)
        finally:
            shutil.rmtree(tmp)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs ffmpeg")
class Encode(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hs-crop-")
        self._redir = contextlib.redirect_stdout(io.StringIO())
        self._redir.__enter__()
        self.pj = Project(os.path.join(self.tmp, "proj"), create=True)

    def tearDown(self):
        self._redir.__exit__(None, None, None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def encode(self, w, h):
        import cv2
        frames = os.path.join(self.tmp, f"frames_{w}x{h}")
        os.makedirs(frames)
        yy, xx = np.mgrid[0:h, 0:w]
        for i in range(3):
            # a mark at the centre and a gradient, so a stretch or an off-centre crop would show
            img = np.dstack([xx * 255 // w, yy * 255 // h, np.full_like(xx, 40 * i)]).astype(np.uint8)
            cv2.circle(img, (w // 2, h // 2), min(w, h) // 8, (255, 255, 255), -1)
            cv2.imwrite(os.path.join(frames, f"frame_{i:04d}.png"), img)
        out = os.path.join(self.tmp, f"out_{w}x{h}.mp4")
        os.makedirs(os.path.dirname(self.pj.log_path("render")), exist_ok=True)
        fw, fh = render.png_size(os.path.join(frames, "frame_0000.png"))
        render._ffmpeg("ffmpeg", 30.0, os.path.join(frames, "frame_%04d.png"), render.crop_4x5_filter(fw, fh),
                       17, out, self.pj, 3, "encode_4x5")
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height,nb_frames", "-of", "json", out], capture_output=True, text=True)
        s = json.loads(r.stdout)["streams"][0]
        png = os.path.join(self.tmp, f"first_{w}x{h}.png")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", out, "-frames:v", "1", png], check=True)
        return s, cv2.imread(png)

    def check(self, w, h):
        s, img = self.encode(w, h)
        self.assertEqual((s["width"], s["height"], int(s["nb_frames"])), (1080, 1350, 3))
        # the white disc is still a disc (no stretch) and still in the middle (centred crop)
        ys, xs = np.nonzero(img.min(axis=2) > 200)
        self.assertLess(abs(xs.mean() - 540), 6)
        self.assertLess(abs(ys.mean() - 675), 6)
        self.assertLess(abs((xs.max() - xs.min()) / float(ys.max() - ys.min()) - 1.0), 0.04)

    def test_an_upright_iphone_frame_encodes(self):
        self.check(240, 428)             # 2400 x 4276 at a tenth

    def test_a_16_9_frame_still_encodes(self):
        self.check(480, 270)

    def test_the_old_filter_fails_on_an_upright_frame(self):
        """The failure, kept: fitting an upright frame by height leaves nothing 1080 wide to cut."""
        import cv2
        frames = os.path.join(self.tmp, "frames_old")
        os.makedirs(frames)
        cv2.imwrite(os.path.join(frames, "frame_0000.png"), np.zeros((428, 240, 3), np.uint8))
        os.makedirs(os.path.dirname(self.pj.log_path("render")), exist_ok=True)
        with self.assertRaises(events.StageError):
            render._ffmpeg("ffmpeg", 30.0, os.path.join(frames, "frame_%04d.png"), "scale=-2:1350,crop=1080:1350",
                           17, os.path.join(self.tmp, "old.mp4"), self.pj, 1, "encode_4x5")


if __name__ == "__main__":
    unittest.main()
