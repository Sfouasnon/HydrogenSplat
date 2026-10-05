"""hs exposure --analyze and its remedies (the Look step, docs/ui-rebuild.md).

Synthetic views: a flat grey subject in the central box (or under a mask), a darker surround, a
brightness that walks frame to frame by a known number of stops, and clipped blobs of a known size.
The numbers the stage reports must come back as built, and --apply global_drop / shoulder must
move the pixels exactly as advertised.
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

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from hs import events  # noqa: E402
from hs.project import Project, now_iso  # noqa: E402
from hs.stages import exposure  # noqa: E402

W, H = 240, 180


def code(lin):
    """linear light -> sRGB code"""
    lin = np.clip(lin, 0, 1)
    s = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
    return int(round(float(s) * 255))


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hs-exposure-")
        self.out = io.StringIO()
        self._redir = contextlib.redirect_stdout(self.out)
        self._redir.__enter__()

    def tearDown(self):
        self._redir.__exit__(None, None, None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def project(self, stops, clip_blob_px=0, clip_whole=False, masks=False, kind=None, name="proj"):
        """n views whose subject luma is 0.18 x 2^s for s in `stops`, the surround 0.03 x 2^s (the whole
        frame walks, as a camera's auto-exposure makes it). clip_blob_px: a square of 255s this wide in
        the subject; clip_whole: half the subject at 255."""
        import cv2
        pj = Project(os.path.join(self.tmp, name), create=True)
        for s in ("ingest", "select", "solve"):
            pj.m["stages"][s] = {"status": "done", "finished": now_iso(), "checks": [], "metrics": {}}
        pj.m["stages"]["train"] = {"status": "done"}
        if kind:
            pj.m["project"] = {"subject_kind": kind}
        d = os.path.join(pj.dataset_dir, "images", "L")
        os.makedirs(d)
        for i, s in enumerate(stops):
            im = np.full((H, W, 3), code(0.03 * 2.0 ** s), np.uint8)
            y0, y1, x0, x1 = H // 4, 3 * H // 4, W // 4, 3 * W // 4
            im[y0:y1, x0:x1] = code(0.18 * 2.0 ** s)
            if clip_whole:
                im[y0:(y0 + y1) // 2, x0:x1] = 255
            elif clip_blob_px:
                im[H // 2:H // 2 + clip_blob_px, W // 2:W // 2 + clip_blob_px] = 255
            cv2.imwrite(os.path.join(d, f"cap{i:03d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 97])
            if masks:
                md = os.path.join(pj.dataset_dir, "masks", "L")
                os.makedirs(md, exist_ok=True)
                m = np.zeros((H, W), np.uint8)
                m[y0:y1, x0:x1] = 255
                cv2.imwrite(os.path.join(md, f"cap{i:03d}.png"), m)
        pj.save()
        return pj

    @staticmethod
    def args(**kw):
        base = dict(mode="rgb", reference="median", restore=False, dry_run=False, force=False,
                    analyze=False, apply=None, stops=exposure.DROP_STOPS, knee=exposure.SHOULDER_KNEE, subject=None)
        base.update(kw)
        return Namespace(**base)

    def mean_luma(self, pj, name):
        import cv2
        lut = exposure._srgb_to_linear_lut()
        im = cv2.imread(os.path.join(pj.dataset_dir, "images", "L", name))
        return float((lut[im[H // 4:3 * H // 4, W // 4:3 * W // 4]] @ exposure.LUMA_BGR).mean())


class Analyze(Base):
    def test_known_drift_and_small_blobs_recommend_match(self):
        # 12 views walking 0 .. 1.1 stops: p95/p5 of the means is about 1.0 stop
        stops = np.linspace(0.0, 1.1, 12)
        pj = self.project(stops, clip_blob_px=6)
        exposure.run(self.args(analyze=True), pj)
        m = pj.stage("exposure")["metrics"]
        self.assertAlmostEqual(m["drift_stops"], 1.0, delta=0.12)
        self.assertEqual(m["clipped_where"], "highlights")
        self.assertLess(m["clipped_share"], 0.05)
        self.assertGreater(m["clipped_share"], 0.0)
        self.assertEqual(m["recommendation"], "match")
        self.assertEqual(len(m["brightness_by_frame"]), 12)
        self.assertEqual(m["brightness_by_frame"][0]["view"], "L/cap000")
        self.assertEqual(m["brightness_by_frame"][0]["region"], "centre")
        # the per-frame stops are relative to the median view and follow the walk
        rel = [r["stops"] for r in m["brightness_by_frame"]]
        self.assertAlmostEqual(rel[-1] - rel[0], 1.1, delta=0.1)
        c = {x["name"]: x for x in pj.stage("exposure")["checks"]}
        self.assertFalse(c["exposure_consistent"]["ok"])
        self.assertIn("match every frame", c["exposure_consistent"]["value"])
        self.assertIn("stops", c["exposure_consistent"]["value"])
        # a measurement: no status, no lock, no pixels
        self.assertEqual(pj.status("exposure"), "pending")
        self.assertEqual(pj.status("train"), "done")
        self.assertFalse(os.path.exists(pj.lock_path))
        self.assertFalse(os.path.isdir(pj.path("solve", "exposure_backup")))
        rep = json.load(open(pj.path("exposure", "analysis.json")))
        self.assertEqual(rep["recommendation"], "match")
        self.assertEqual(len(rep["brightness_by_frame"]), 12)

    def test_steady_exposure_with_nothing_clipped_is_consistent(self):
        pj = self.project([0.0, 0.05, -0.05, 0.02, 0.0, 0.03])
        exposure.run(self.args(analyze=True), pj)
        m = pj.stage("exposure")["metrics"]
        self.assertLess(m["drift_stops"], 0.2)
        self.assertEqual(m["clipped_where"], "none")
        self.assertEqual(m["recommendation"], "none")
        c = {x["name"]: x for x in pj.stage("exposure")["checks"]}
        self.assertTrue(c["exposure_consistent"]["ok"])
        self.assertIn("no exposure work needed", c["exposure_consistent"]["value"])

    def test_whole_subject_clipped_recommends_a_global_drop(self):
        pj = self.project([0.0] * 6, clip_whole=True)
        exposure.run(self.args(analyze=True), pj)
        m = pj.stage("exposure")["metrics"]
        self.assertEqual(m["clipped_where"], "whole")
        self.assertGreater(m["clipped_share"], 0.4)
        self.assertEqual(m["recommendation"], "global_drop")
        c = {x["name"]: x for x in pj.stage("exposure")["checks"]}
        self.assertFalse(c["exposure_consistent"]["ok"])
        self.assertIn("drop the exposure", c["exposure_consistent"]["value"])

    def test_highlights_on_a_glossy_subject_recommend_the_shoulder(self):
        pj = self.project([0.0] * 6, clip_blob_px=6, kind="glossy")
        exposure.run(self.args(analyze=True), pj)
        m = pj.stage("exposure")["metrics"]
        self.assertEqual(m["clipped_where"], "highlights")
        self.assertEqual(m["recommendation"], "shoulder")
        c = {x["name"]: x for x in pj.stage("exposure")["checks"]}
        self.assertTrue(c["exposure_consistent"]["ok"], "highlights alone do not fail the check")
        # a matte subject with the same glints gets no recommendation
        pj2 = self.project([0.0] * 6, clip_blob_px=6, kind="matte", name="matte")
        exposure.run(self.args(analyze=True), pj2)
        self.assertEqual(pj2.stage("exposure")["metrics"]["recommendation"], "none")
        # --subject overrides the manifest
        exposure.run(self.args(analyze=True, subject="bright"), pj2)
        self.assertEqual(pj2.stage("exposure")["metrics"]["recommendation"], "shoulder")
        self.assertEqual(len([c for c in pj2.stage("exposure")["checks"] if c["name"] == "exposure_consistent"]), 1,
                         "a second analysis replaces its check, not appends")

    def test_masks_pick_the_subject_pixels(self):
        # the subject is clipped only outside the masked region: with masks the share is 0, without it is not
        import cv2
        pj = self.project([0.0] * 4, masks=True)
        for f in os.listdir(os.path.join(pj.dataset_dir, "images", "L")):
            p = os.path.join(pj.dataset_dir, "images", "L", f)
            im = cv2.imread(p)
            im[:H // 8, :] = 255                                   # a blown strip at the top: not the subject
            cv2.imwrite(p, im, [cv2.IMWRITE_JPEG_QUALITY, 97])
        exposure.run(self.args(analyze=True), pj)
        m = pj.stage("exposure")["metrics"]
        self.assertEqual(m["clipped_where"], "none")
        self.assertEqual(m["analysis"]["regions"], {"mask": 4, "centre": 0})
        self.assertEqual(m["brightness_by_frame"][0]["region"], "mask")


class Apply(Base):
    def test_global_drop_darkens_every_view_by_the_stops_asked(self):
        pj = self.project([0.0, 0.1, -0.1, 0.05])
        before = {f: self.mean_luma(pj, f) for f in ("cap000.jpg", "cap001.jpg")}
        exposure.run(self.args(apply="global_drop", stops=1.0), pj)
        for f, b in before.items():
            self.assertAlmostEqual(np.log2(b / self.mean_luma(pj, f)), 1.0, delta=0.06, msg=f)
        st = pj.stage("exposure")
        self.assertEqual(st["status"], "done")
        self.assertEqual(st["metrics"]["applied"], ["global_drop"])
        self.assertEqual(st["metrics"]["global_drop_stops"], 1.0)
        self.assertAlmostEqual(st["metrics"]["global_drop_measured_stops"], 1.0, delta=0.06)
        c = {x["name"]: x for x in st["checks"]}
        self.assertTrue(c["global_drop_landed"]["ok"])
        self.assertIn("exposure_consistent", c, "an apply ends with the analysis of its result")
        self.assertEqual(pj.status("train"), "stale")
        self.assertTrue(os.path.isdir(pj.path("solve", "exposure_backup")), "originals kept")
        self.assertTrue(os.path.exists(pj.path("exposure", "analysis.json")))
        # a second run starts from the originals, so two drops of 1 stop are 1 stop, not 2
        pj.m["stages"]["train"]["status"] = "done"
        pj.save()
        exposure.run(self.args(apply="global_drop", stops=1.0), pj)
        for f, b in before.items():
            self.assertAlmostEqual(np.log2(b / self.mean_luma(pj, f)), 1.0, delta=0.06, msg=f)
        # and --restore puts them back
        exposure.run(self.args(restore=True), pj)
        for f, b in before.items():
            self.assertAlmostEqual(self.mean_luma(pj, f), b, delta=0.002)

    def test_shoulder_table_is_identity_below_the_knee_and_tames_255(self):
        lut = exposure.shoulder_lut(0.85)
        k = int(round(0.85 * 255))
        self.assertTrue(np.array_equal(lut[:k + 1], np.arange(k + 1)))
        self.assertTrue(np.all(np.diff(lut.astype(int)) >= 0), "order kept")
        self.assertLess(int(lut[255]), 250, "a clipped code no longer reads as clipped")
        self.assertGreater(int(lut[255]), k)

    def test_shoulder_rewrites_only_the_highlights(self):
        import cv2
        pj = self.project([0.0] * 3, clip_blob_px=10)
        p = os.path.join(pj.dataset_dir, "images", "L", "cap000.jpg")
        before = cv2.imread(p)
        exposure.run(self.args(apply="shoulder"), pj)
        after = cv2.imread(p)
        self.assertLess(int(after[H // 2 + 2, W // 2 + 2].max()), 250, "the blob was rolled off")
        mid = (H // 2 - 20, W // 2 - 20)
        self.assertLessEqual(abs(int(after[mid].mean()) - int(before[mid].mean())), 2, "the paint is untouched")
        st = pj.stage("exposure")
        self.assertEqual(st["metrics"]["applied"], ["shoulder"])
        self.assertEqual(st["metrics"]["shoulder_knee"], 0.85)
        self.assertEqual(st["metrics"]["clipped_where"], "none", "the analysis of the result sees no clipping")

    def test_match_then_shoulder_is_one_rewrite(self):
        pj = self.project(np.linspace(0.0, 1.0, 6), clip_blob_px=8)
        exposure.run(self.args(apply="match,shoulder"), pj)
        st = pj.stage("exposure")
        self.assertEqual(st["metrics"]["applied"], ["match", "shoulder"])
        self.assertEqual(st["metrics"]["reference"], "median")
        self.assertLess(st["metrics"]["drift_stops"], exposure.DRIFT_STOPS_OK, "matched: the drift is gone")
        c = {x["name"]: x for x in st["checks"]}
        self.assertTrue(c["views_share_one_exposure"]["ok"])
        self.assertTrue(c["exposure_consistent"]["ok"])

    def test_bad_steps_are_refused(self):
        pj = self.project([0.0] * 3)
        with self.assertRaises(events.StageError):
            exposure.run(self.args(apply="brighten"), pj)
        with self.assertRaises(events.StageError):
            exposure.run(self.args(apply="shoulder", knee=1.5), pj)
        self.assertFalse(os.path.exists(pj.lock_path))


if __name__ == "__main__":
    unittest.main()
