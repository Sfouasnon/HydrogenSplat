"""source.kind: "mono" (one camera's frames) beside "array" (one frame per camera).

Both take the frames route (select done at ingest, solve through monocolmap.py); they differ
only where the number of cameras matters — hs exposure's median refusal. Old manifests (no kind:
a Hydrogen clip; mono data tagged array) keep working as they did.
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

from hs import events  # noqa: E402
from hs.project import Project, now_iso  # noqa: E402
from hs.stages import exposure, ingest, select, solve  # noqa: E402


def write_jpg(path, level, size=(64, 36)):
    import cv2
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, np.full((size[1], size[0], 3), level, np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95])


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hs-kind-")
        self.root = os.path.join(self.tmp, "proj")
        self.out = io.StringIO()
        self._redir = contextlib.redirect_stdout(self.out)
        self._redir.__enter__()

    def tearDown(self):
        self._redir.__exit__(None, None, None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def args(self, **kw):
        base = dict(clip=None, phone=None, remote=None, adb="adb", link=False, profile=None, ffprobe="ffprobe",
                    frames=None, r3d=None, take=None, redline="REDline", res=1, kind="auto")
        base.update(kw)
        return Namespace(**base)

    def folder(self, names, sub=None):
        d = os.path.join(self.tmp, "src", *( [sub] if sub else []))
        for i, n in enumerate(names):
            write_jpg(os.path.join(d, n + ".jpg"), 40 + 20 * i)
        return os.path.join(self.tmp, "src")

    def ingest(self, **kw):
        pj = Project(self.root, create=True)
        ingest.run(self.args(**kw), pj)
        return pj


class Ingest(Base):
    def test_select_frames_mono_picks_are_mono(self):
        pj = self.ingest(frames=self.folder(["sel000-00012", "sel001-00031", "sel002-00050", "sel003-00077"]))
        self.assertEqual(pj.m["source"]["kind"], "mono")
        self.assertIn("selNNN", pj.m["source"]["kind_why"])
        self.assertEqual(pj.stage("ingest")["metrics"]["source_kind"], "mono")
        self.assertEqual(pj.status("select"), "done")
        self.assertIn("mono source", pj.stage("select")["metrics"]["note"])
        chk = {c["name"] for c in pj.stage("ingest")["checks"]}
        self.assertIn("enough_frames", chk)
        self.assertTrue(pj.frames_route)

    def test_a_numbered_sequence_is_mono(self):
        pj = self.ingest(frames=self.folder([f"frame-{i:04d}" for i in (1, 7, 13, 19)]))
        self.assertEqual(pj.m["source"]["kind"], "mono")

    def test_selection_json_says_mono(self):
        src = self.folder(["pickA", "pickB", "pickC"])
        json.dump({"mono": True, "selected": []}, open(os.path.join(src, "selection.json"), "w"))
        pj = self.ingest(frames=src)
        self.assertEqual(pj.m["source"]["kind"], "mono")
        self.assertEqual(sorted(os.listdir(pj.frames_dir)), ["pickA.jpg", "pickB.jpg", "pickC.jpg"])

    def test_a_single_subfolder_is_looked_into(self):
        pj = self.ingest(frames=self.folder(["sel000-00012", "sel001-00031", "sel002-00050"], sub="picks"))
        self.assertEqual(pj.m["source"]["kind"], "mono")
        self.assertEqual(len(os.listdir(pj.frames_dir)), 3)

    def test_camera_named_flat_folder_is_still_an_array(self):
        pj = self.ingest(frames=self.folder(["GA", "GB", "HA", "HB"]))
        self.assertEqual(pj.m["source"]["kind"], "array")
        self.assertIn("enough_cameras", {c["name"] for c in pj.stage("ingest")["checks"]})
        pj2 = Project(os.path.join(self.tmp, "p2"), create=True)
        shutil.rmtree(os.path.join(self.tmp, "src"))
        ingest.run(self.args(frames=self.folder(["cam0", "cam1", "cam2"])), pj2)
        self.assertEqual(pj2.m["source"]["kind"], "array")

    def test_per_camera_folders_are_an_array(self):
        src = os.path.join(self.tmp, "src")
        for i, cam in enumerate(("GA", "GB", "HA")):
            write_jpg(os.path.join(src, cam, "A001_C003_0001.jpg"), 50 + 30 * i)
        pj = self.ingest(frames=src)
        self.assertEqual(pj.m["source"]["kind"], "array")
        self.assertEqual([c["camera"] for c in pj.m["source"]["cameras"]], ["GA", "GB", "HA"])
        self.assertEqual(sorted(os.listdir(pj.frames_dir)), ["GA.jpg", "GB.jpg", "HA.jpg"])
        # --kind cannot call per-camera folders one camera
        pj2 = Project(os.path.join(self.tmp, "p2"), create=True)
        with self.assertRaises(events.StageError):
            ingest.run(self.args(frames=src, kind="mono"), pj2)

    def test_a_camera_folder_with_several_frames_is_refused(self):
        src = os.path.join(self.tmp, "src")
        for cam in ("GA", "GB"):
            for f in ("f1", "f2"):
                write_jpg(os.path.join(src, cam, f + ".jpg"), 80)
        pj = Project(self.root, create=True)
        with self.assertRaises(events.StageError) as e:
            ingest.run(self.args(frames=src), pj)
        self.assertIn("one frame per camera", str(e.exception))

    def test_kind_overrides_the_guess_for_a_flat_folder(self):
        pj = self.ingest(frames=self.folder(["GA", "GB", "HA"]), kind="mono")
        self.assertEqual(pj.m["source"]["kind"], "mono")
        self.assertEqual(pj.m["source"]["kind_why"], "--kind mono")


class Consumers(Base):
    def solved(self, kind):
        pj = Project(self.root, create=True)
        if kind is not None:
            pj.m["source"] = {"kind": kind}
        for s in ("ingest", "select"):
            pj.m["stages"][s] = {"status": "done"}
        pj.m["stages"]["solve"] = {"status": "done", "started": now_iso(), "finished": now_iso()}
        write_jpg(os.path.join(pj.dataset_dir, "images", "L", "sel000-00001.jpg"), 60)
        write_jpg(os.path.join(pj.dataset_dir, "images", "L", "sel001-00009.jpg"), 90)
        pj.save()
        return pj

    def xargs(self, **kw):
        base = dict(mode="rgb", restore=False, dry_run=False, force=False, reference="median",
                    reference_view=None, board=None, legacy_board=False)
        base.update(kw)
        return Namespace(**base)

    def test_old_manifests(self):
        self.assertEqual(self.solved(None).source_kind, "stereo")
        self.assertFalse(self.solved(None).frames_route)
        shutil.rmtree(self.root)
        self.assertTrue(self.solved("array").frames_route)        # includes mono data tagged array

    def test_select_is_refused_on_mono(self):
        pj = self.solved("mono")
        with self.assertRaises(events.StageError) as e:
            select.run(Namespace(), pj)
        self.assertIn("mono", str(e.exception))

    def test_solve_routes_mono_through_monocolmap(self):
        pj = self.solved("mono")
        a = Namespace(board=None)
        with mock.patch.object(solve, "run_array") as ra, mock.patch.object(solve, "_run_stereo") as rs:
            solve.run(a, pj)
        ra.assert_called_once()
        rs.assert_not_called()
        shutil.rmtree(self.root)
        pj = self.solved(None)
        with mock.patch.object(solve, "run_array") as ra, mock.patch.object(solve, "_run_stereo") as rs:
            solve.run(a, pj)
        rs.assert_called_once()
        ra.assert_not_called()

    def test_exposure_median_is_an_array_refusal_only(self):
        # mono = one camera orbiting the subject, the case the median match was built for
        pj = self.solved("mono")
        exposure.run(self.xargs(), pj)
        self.assertEqual(pj.status("exposure"), "done")
        shutil.rmtree(self.root)
        pj = self.solved("array")
        with self.assertRaises(events.StageError):
            exposure.run(self.xargs(), pj)

    def test_exposure_board_is_allowed_on_mono_and_array(self):
        # allowed = gets past the refusal; this project has no board, so it then fails on that
        for kind in ("mono", "array"):
            pj = self.solved(kind)
            with self.assertRaises(events.StageError) as e:
                exposure.run(self.xargs(reference="board", board="7,5,30,22"), pj)
            self.assertIn("board was not measured", str(e.exception))
            pj.release()
            shutil.rmtree(self.root)


if __name__ == "__main__":
    unittest.main()
