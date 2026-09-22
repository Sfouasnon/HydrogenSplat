"""hs exposure --reference board: match every view on the white paper of a ChArUco board.

Synthetic views of the board (board_synth.Scene) each get their own per-channel gain in linear
light — an array whose cameras disagree in exposure and white balance. The stage must recover
every view's correction to 1 % and leave the board's white the same in every view.
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from argparse import Namespace
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from hs import board as B, events  # noqa: E402
from hs.project import Project, md5_file, now_iso  # noqa: E402
from hs.stages import exposure  # noqa: E402
import board_synth as S  # noqa: E402

BOARD = "7,5,30,22"
CAMS = ["GA", "GB", "GC", "HA", "HB"]
# per-view gains in linear light, B G R: exposure and white balance both differ
GAINS = np.array([[1.00, 1.00, 1.00], [0.80, 0.85, 0.95], [1.20, 1.10, 1.05], [0.90, 1.00, 1.15], [1.10, 0.95, 0.85]])


def lut():
    return exposure._srgb_to_linear_lut()


def apply_gain(img, g):
    lin = lut()[img] * np.asarray(g)[None, None, :]
    return exposure._linear_to_srgb(lin)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hs-expboard-")
        self.root = os.path.join(self.tmp, "proj")
        self.out = io.StringIO()
        self._redir = contextlib.redirect_stdout(self.out)
        self._redir.__enter__()

    def tearDown(self):
        self._redir.__exit__(None, None, None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def project(self, kind="array", board_in=None):
        import cv2
        sc = S.Scene(B.parse_spec(BOARD), n_views=len(CAMS), white=200, black=35, background=120,
                     noise=1.5, size=(1200, 900), f=1300.0)
        pj = Project(self.root, create=True)
        pj.m["source"] = {"kind": kind}
        for s in ("ingest", "select"):
            pj.m["stages"][s] = {"status": "done"}
        pj.m["stages"]["solve"] = {"status": "done", "started": now_iso(), "finished": now_iso()}
        pj.m["stages"]["train"] = {"status": "done"}
        d = os.path.join(pj.dataset_dir, "images", "L")
        os.makedirs(d)
        for i, c in enumerate(CAMS):
            img = sc.exposed(i, GAINS[i])
            if board_in is not None and i not in board_in:
                img = np.full_like(img, 120)
            cv2.imwrite(os.path.join(d, c + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        pj.save()
        return pj

    def args(self, **kw):
        base = dict(mode="rgb", restore=False, dry_run=False, force=False, reference="board",
                    reference_view=None, board=BOARD, legacy_board=False)
        base.update(kw)
        return Namespace(**base)

    def report(self, pj):
        rep = json.load(open(os.path.join(pj.dataset_dir, "exposure.json")))
        return rep, {v["image"]: np.array(v["gain_bgr"]) for v in rep["views"]}


class BoardReference(Base):
    def test_recovers_every_views_gain_on_an_array(self):
        pj = self.project()
        originals = {c: md5_file(os.path.join(pj.dataset_dir, "images", "L", c + ".jpg")) for c in CAMS}
        exposure.run(self.args(), pj)                   # no --force: a shared target is allowed on an array
        self.assertEqual(pj.status("exposure"), "done")
        self.assertEqual(pj.status("train"), "stale")
        rep, g = self.report(pj)
        ref = rep["reference"]
        self.assertTrue(ref.startswith("board:L/"), ref)
        ri = CAMS.index(ref.split("/")[-1])
        worst = 0.0
        for i, c in enumerate(CAMS):
            want = GAINS[ri] / GAINS[i]
            err = np.abs(g[c + ".jpg"] / want - 1).max()
            worst = max(worst, err)
            self.assertLess(err, 0.01, f"{c}: got {g[c + '.jpg']}, want {want}")
        np.testing.assert_allclose(g[CAMS[ri] + ".jpg"], [1, 1, 1])
        m = pj.stage("exposure")["metrics"]
        tm = m["exposure_target"]
        self.assertEqual(tm["target"], "board")
        self.assertEqual(tm["views_with_target"], len(CAMS))
        self.assertEqual(set(tm["gains"]), {f"L/{c}.jpg" for c in CAMS})
        self.assertGreater(tm["white_rms_before"], 0.05)
        self.assertLess(tm["white_rms_after"], 0.01)
        self.assertLess(m["white_rms_after"], m["white_rms_before"])
        checks = {c["name"]: c for c in pj.stage("exposure")["checks"]}
        self.assertTrue(checks["views_share_one_white"]["ok"], checks["views_share_one_white"])
        self.assertTrue(checks["board_seen_in_every_view"]["ok"])
        # the stage's backup / --restore pipeline is the same one the median mode uses
        pj.release()
        exposure.run(self.args(restore=True), pj)
        for c in CAMS:
            self.assertEqual(md5_file(os.path.join(pj.dataset_dir, "images", "L", c + ".jpg")), originals[c])
        self.worst = worst

    def test_named_reference_view_keeps_its_exposure(self):
        pj = self.project()
        exposure.run(self.args(reference_view="GC"), pj)
        rep, g = self.report(pj)
        self.assertEqual(rep["reference"], "board:L/GC")
        np.testing.assert_allclose(g["GC.jpg"], [1, 1, 1])
        np.testing.assert_allclose(g["GA.jpg"], GAINS[2] / GAINS[0], rtol=0.01)
        pj.release()
        with self.assertRaises(events.StageError):
            exposure.run(self.args(reference_view="ZZ"), pj)

    def test_the_board_defaults_to_the_one_hs_scale_used(self):
        pj = self.project()
        pj.m["scale"] = {"board": BOARD + ",DICT_5X5_100"}
        pj.save()
        exposure.run(self.args(board=None, dry_run=True), pj)
        dr = pj.stage("exposure")["dry_run"]["metrics"]
        self.assertEqual(dr["exposure_target"]["views_with_target"], len(CAMS))
        self.assertEqual(pj.status("exposure"), "pending")          # a dry run changes nothing
        self.assertFalse(os.path.exists(os.path.join(pj.dataset_dir, "exposure.json")))

    def test_a_view_without_the_board_is_left_alone_and_named(self):
        pj = self.project(board_in={0, 1, 2, 4})
        exposure.run(self.args(), pj)
        rep, g = self.report(pj)
        np.testing.assert_allclose(g["HA.jpg"], [1, 1, 1])
        checks = {c["name"]: c for c in pj.stage("exposure")["checks"]}
        self.assertFalse(checks["board_seen_in_every_view"]["ok"])
        self.assertTrue(checks["board_seen_in_every_view"].get("needs_human"))
        self.assertIn("L/HA.jpg", checks["board_seen_in_every_view"]["value"])

    def test_median_is_still_refused_on_an_array(self):
        pj = self.project()
        with self.assertRaises(events.StageError) as e:
            exposure.run(self.args(reference="median"), pj)
        self.assertIn("--reference board", e.exception.hint)

    def test_checker_needs_mcc(self):
        pj = self.project()
        with mock.patch.object(exposure, "checker_available", return_value=False):
            with self.assertRaises(events.StageError) as e:
                exposure.run(self.args(reference="checker"), pj)
        self.assertIn("cv2.mcc", str(e.exception))
        self.assertEqual(pj.status("exposure"), "pending")

    def test_checker_that_finds_no_chart_says_so(self):
        if not exposure.checker_available():
            self.skipTest("no cv2.mcc in this OpenCV")
        pj = self.project()      # a ChArUco board, no Macbeth: mcc's false positive must be rejected
        with self.assertRaises(events.StageError) as e:
            exposure.run(self.args(reference="checker", dry_run=True), pj)
        self.assertIn("checker was not measured", str(e.exception))


# X-Rite ColorChecker Classic, sRGB code values (approx.), row by row
MACBETH = np.array([
    [115, 82, 68], [194, 150, 130], [98, 122, 157], [87, 108, 67], [133, 128, 177], [103, 189, 170],
    [214, 126, 44], [80, 91, 166], [193, 90, 99], [94, 60, 108], [157, 188, 64], [224, 163, 46],
    [56, 61, 150], [70, 148, 73], [175, 54, 60], [231, 199, 31], [187, 86, 149], [8, 133, 161],
    [243, 243, 242], [200, 200, 200], [160, 160, 160], [122, 122, 121], [85, 85, 85], [52, 52, 52]])


def macbeth_image(shift=(0, 0)):
    import cv2
    P, G = 60, 12
    img = np.full((560, 760, 3), 150, np.uint8)
    x0, y0 = 120 + shift[0], 110 + shift[1]
    cv2.rectangle(img, (x0, y0), (x0 + 6 * (P + G) + G, y0 + 4 * (P + G) + G), (20, 20, 20), -1)
    for k, rgb in enumerate(MACBETH):
        r, c = divmod(k, 6)
        x, y = x0 + G + c * (P + G), y0 + G + r * (P + G)
        cv2.rectangle(img, (x, y), (x + P, y + P), tuple(int(v) for v in rgb[::-1]), -1)
    # sensor noise: a flat synthetic patch would otherwise quantise to one code value, ~1 % of
    # linear light near white, which no real photograph of a chart does
    rng = np.random.default_rng(shift[0])
    return np.clip(img + rng.normal(0, 2.0, img.shape), 0, 255).astype(np.uint8)


class CheckerReference(Base):
    def test_recovers_gains_from_the_white_patch(self):
        import cv2
        if not exposure.checker_available():
            self.skipTest("no cv2.mcc in this OpenCV")
        pj = Project(self.root, create=True)
        pj.m["source"] = {"kind": "array"}
        for s in ("ingest", "select", "solve"):
            pj.m["stages"][s] = {"status": "done"}
        d = os.path.join(pj.dataset_dir, "images", "L")
        os.makedirs(d)
        base = 0.8                                     # keep the white patch off the clip after the gains
        for i, c in enumerate(CAMS):
            cv2.imwrite(os.path.join(d, c + ".jpg"), apply_gain(macbeth_image((7 * i, -5 * i)), GAINS[i] * base),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        pj.save()
        exposure.run(self.args(reference="checker", reference_view="GA"), pj)
        rep, g = self.report(pj)
        self.assertEqual(rep["reference"], "checker:L/GA")
        for i, c in enumerate(CAMS):
            np.testing.assert_allclose(g[c + ".jpg"], GAINS[0] / GAINS[i], rtol=0.01, err_msg=c)
        self.assertLess(pj.stage("exposure")["metrics"]["white_rms_after"], 0.01)


if __name__ == "__main__":
    unittest.main()
