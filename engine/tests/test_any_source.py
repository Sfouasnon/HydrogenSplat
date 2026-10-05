"""Any of the sources the engine takes can be brought in without knowing which it is.

Until 2026-10-04 the app's New Project page took a Hydrogen clip and nothing else; a phone orbit,
a folder of photographs and a RED array went through Terminal, and a folder straight off a camera
was refused for the underscores in IMG_0001.JPG. `hs source PATH` now says what a path is, and
`hs ingest` takes it: --clip decides Hydrogen or one ordinary camera from the container's own
tags, a one-camera clip is picked by `hs select`, file names become view names, HEIC goes through
sips, hidden files are skipped.
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
sys.path.insert(0, HERE)

from hs import events, frame_quality, sourceprobe  # noqa: E402
from hs.project import Project  # noqa: E402
from hs.stages import ingest, select, source  # noqa: E402

import test_select_keyframes as KF  # noqa: E402

FAKEBIN = os.path.join(HERE, "fakebin")


def write_jpg(path, level, size=(64, 48)):
    import cv2
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, np.full((size[1], size[0], 3), level, np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95])


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hs-anysource-")
        self.root = os.path.join(self.tmp, "proj")
        self.out = io.StringIO()
        self._redir = contextlib.redirect_stdout(self.out)
        self._redir.__enter__()
        self._env = {k: os.environ.get(k) for k in ("HS_PYTHON", "HS_SIPS", "HS_REDLINE")}
        os.environ.update(HS_PYTHON=sys.executable, HS_SIPS=os.path.join(FAKEBIN, "sips"),
                          HS_REDLINE=os.path.join(FAKEBIN, "REDline"))

    def tearDown(self):
        self._redir.__exit__(None, None, None)
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(self.tmp, ignore_errors=True)

    def args(self, **kw):
        base = dict(clip=None, phone=None, remote=None, adb="adb", link=False, profile=None, ffprobe="ffprobe",
                    frames=None, r3d=None, take=None, redline=os.path.join(FAKEBIN, "REDline"), res=1, kind="auto")
        base.update(kw)
        return Namespace(**base)

    def ingest(self, **kw):
        pj = Project(self.root, create=True)
        ingest.run(self.args(**kw), pj)
        return pj

    def stills(self, names, ext=".JPG"):
        d = os.path.join(self.tmp, "card", "DCIM")
        for i, n in enumerate(names):
            write_jpg(os.path.join(d, n + ext), 40 + 10 * i)
        return d

    def rdm_tree(self, cams=("GA", "GB", "HA"), takes=("067", "068")):
        root = os.path.join(self.tmp, "RED_Footage")
        for cam in cams:
            for tk in takes:
                clip = f"{cam[0]}007_{cam[1]}{tk}_0403XX"
                d = os.path.join(root, cam, f"{cam[0]}007_ZZZZZZ.RDM", clip + ".RDC")
                os.makedirs(d)
                with open(os.path.join(d, clip + "_001.R3D"), "wb") as f:
                    f.write(bytes([ord(cam[1])]) + b"\0" * 64)
        return root


class Names(unittest.TestCase):
    def test_a_file_name_becomes_a_view_name(self):
        self.assertEqual(sourceprobe.safe_stem("IMG_0001"), "IMG-0001")
        self.assertEqual(sourceprobe.safe_stem("DSC 0042 (1)"), "DSC-0042-1")
        self.assertEqual(sourceprobe.safe_stem("Stormtrooper.final_v2"), "Stormtrooper-final-v2")
        self.assertEqual(sourceprobe.safe_stem("sel012-00345"), "sel012-00345", "already a view name: untouched")
        self.assertEqual(sourceprobe.safe_stem("___"), "frame")

    def test_two_files_that_come_out_the_same_stay_two_views(self):
        self.assertEqual(sourceprobe.safe_names(["IMG_0001", "IMG-0001", "IMG 0001", "GA"]),
                         ["IMG-0001", "IMG-0001-2", "IMG-0001-3", "GA"])
        self.assertEqual(sourceprobe.safe_names(["a_b", "A-B"]), ["a-b", "A-B-2"], "the Mac's disk does not tell case apart")

    def test_hidden_files(self):
        self.assertTrue(sourceprobe.hidden("._IMG_0001.JPG"))
        self.assertTrue(sourceprobe.hidden(".DS_Store"))
        self.assertFalse(sourceprobe.hidden("IMG_0001.JPG"))


class VideoFacts(unittest.TestCase):
    """What the container says about a clip, read the way the selector will decode it."""

    def probe(self, **stream):
        s = {"index": 0, "codec_type": "video", "codec_name": "hevc", "width": 3840, "height": 2160,
             "avg_frame_rate": "30/1", "r_frame_rate": "30/1", "nb_frames": "900", "pix_fmt": "yuv420p10le"}
        s.update(stream)
        return {"streams": [{"index": 1, "codec_type": "audio"}, s], "format": {"duration": "30.0", "tags": {}}}

    def test_a_portrait_iphone_clip_reads_as_it_is_shown(self):
        f, problems = sourceprobe.video_facts(self.probe(
            color_transfer="arib-std-b67", side_data_list=[{"side_data_type": "Display Matrix", "rotation": -90}]))
        self.assertEqual(problems, [])
        self.assertEqual((f["width"], f["height"], f["rotation"], f["hdr"]), (2160, 3840, -90.0, "hlg"))
        self.assertEqual((f["fps"], f["nb_frames"], f["duration_s"], f["codec"]), (30.0, 900, 30.0, "hevc"))
        self.assertEqual(sourceprobe.video_line(f), "2160×3840 · hevc · 30 fps · 900 frames · 30.0 s · HLG HDR")

    def test_pq_is_refused_and_says_what_to_record_instead(self):
        _f, problems = sourceprobe.video_facts(self.probe(color_transfer="smpte2084"))
        self.assertEqual(len(problems), 1)
        self.assertIn("PQ", problems[0])
        self.assertIn("HLG", problems[0])

    def test_no_frame_count_in_the_container(self):
        f, problems = sourceprobe.video_facts(self.probe(nb_frames=None))
        self.assertEqual((f["nb_frames"], problems), (900, []), "duration x rate")

    def test_too_short_and_no_picture(self):
        _f, problems = sourceprobe.video_facts(self.probe(nb_frames="12"))
        self.assertIn("too short", problems[0])
        f, problems = sourceprobe.video_facts({"streams": [{"codec_type": "audio"}], "format": {}})
        self.assertIsNone(f)
        self.assertEqual(problems, ["no video stream in this file"])

    def test_cover_art_is_not_the_picture(self):
        p = self.probe()
        p["streams"].insert(0, {"codec_type": "video", "codec_name": "mjpeg", "width": 600, "height": 600,
                                "disposition": {"attached_pic": 1}})
        f, _ = sourceprobe.video_facts(p)
        self.assertEqual(f["codec"], "hevc")

    def test_which_camera_wrote_it(self):
        leia = {"format": {"tags": {"comment": "leia3d_layout=2x1;leia3d_width_per_view=1920;"}}}
        self.assertEqual(sourceprobe.leia_tags(leia), {"leia3d_layout": "2x1", "leia3d_width_per_view": 1920})
        self.assertEqual(sourceprobe.leia_tags({"format": {"tags": {"comment": "shot on a phone; a=b"}}}), {})
        self.assertEqual(sourceprobe.leia_tags({}), {})

    def test_the_folder_is_dated_by_the_recording(self):
        self.assertEqual(sourceprobe.recorded_date("/nowhere", {"format": {"tags": {"creation_time": "2026-09-28T19:02:11.000000Z"}}})[:8],
                         "2026-09-")
        with tempfile.NamedTemporaryFile() as f:
            os.utime(f.name, (1790000000, 1790000000))
            self.assertRegex(sourceprobe.recorded_date(f.name), r"^2026-09-2\d$")


class Stills(Base):
    def test_names_off_a_camera_are_taken_and_renamed(self):
        d = self.stills([f"IMG_{i:04d}" for i in (1, 2, 3, 5, 8)])
        open(os.path.join(d, "._IMG_0001.JPG"), "wb").write(b"resource fork")     # a card's hidden twins
        open(os.path.join(d, ".DS_Store"), "wb").write(b"x")
        pj = self.ingest(frames=d)
        src = pj.m["source"]
        self.assertEqual(src["kind"], "mono")
        self.assertEqual(src["kind_why"], "one numbered sequence (IMG-0001 … IMG-0008)")
        self.assertEqual([c["camera"] for c in src["cameras"]], [f"IMG-{i:04d}" for i in (1, 2, 3, 5, 8)])
        self.assertEqual(sorted(os.listdir(pj.frames_dir)), [f"IMG-{i:04d}.JPG" for i in (1, 2, 3, 5, 8)])
        self.assertTrue(src["cameras"][0]["origin"].endswith("IMG_0001.JPG"), "where it came from keeps its own name")
        st = pj.stage("ingest")
        self.assertEqual(st["metrics"]["frames_renamed"], 5)
        c = {x["name"]: x for x in st["checks"]}
        self.assertTrue(c["names_made_safe"]["ok"])
        self.assertIn("IMG_0001 → IMG-0001", c["names_made_safe"]["value"])
        self.assertEqual(sorted(os.listdir(d)), sorted([".DS_Store", "._IMG_0001.JPG"] + [f"IMG_{i:04d}.JPG" for i in (1, 2, 3, 5, 8)]),
                         "the originals are not touched")
        self.assertEqual(pj.status("select"), "done")

    def test_names_that_were_fine_are_left_alone(self):
        pj = self.ingest(frames=self.stills(["GA", "GB", "HA"]))
        self.assertEqual(pj.m["source"]["kind"], "array")
        self.assertNotIn("frames_renamed", pj.stage("ingest")["metrics"])
        self.assertNotIn("names_made_safe", [c["name"] for c in pj.stage("ingest")["checks"]])

    def test_iphone_photographs_are_converted(self):
        d = os.path.join(self.tmp, "airdrop")
        os.makedirs(d)
        for i in range(4):
            open(os.path.join(d, f"IMG_{4100 + i}.HEIC"), "wb").write(bytes([60 + 20 * i]))
        pj = self.ingest(frames=d)
        self.assertEqual(sorted(os.listdir(pj.path("source", "frames"))), [f"IMG-{4100 + i}.jpg" for i in range(4)])
        self.assertEqual(pj.stage("ingest")["metrics"]["heic_converted"], 4)
        self.assertEqual((pj.m["source"]["kind"], pj.m["source"]["probe"]["width"]), ("mono", 64))

    def test_heic_without_sips_and_a_file_sips_cannot_read(self):
        d = os.path.join(self.tmp, "airdrop")
        os.makedirs(d)
        for i in range(3):
            open(os.path.join(d, f"IMG_{4100 + i}.HEIC"), "wb").write(b"FAIL" if i == 2 else bytes([90]))
        with self.assertRaisesRegex(events.StageError, "sips could not convert IMG_4102.HEIC"):
            self.ingest(frames=d)
        os.environ["HS_SIPS"] = os.path.join(self.tmp, "no-sips")
        shutil.rmtree(self.root)
        with self.assertRaisesRegex(events.StageError, "is HEIC, and sips") as e:
            self.ingest(frames=d)
        self.assertIn("JPEG", e.exception.hint)

    def test_photographs_of_two_sizes_are_still_refused(self):
        d = self.stills([f"IMG_{i:04d}" for i in (1, 2, 3)])
        write_jpg(os.path.join(d, "IMG_0004.JPG"), 90, (48, 64))          # one shot held upright
        pj = Project(self.root, create=True)
        with self.assertRaises(events.StageError):
            ingest.run(self.args(frames=d), pj)
        c = {x["name"]: x for x in pj.stage("ingest")["checks"]}
        self.assertFalse(c["one_frame_size"]["ok"])
        self.assertIn("IMG-0004", c["one_frame_size"]["value"])


class Probe(Base):
    """hs source PATH: what the New Project page shows for a dropped file or folder."""

    def test_a_folder_of_photographs(self):
        d = self.stills([f"DSC_{i:04d}" for i in range(1, 8)])
        open(os.path.join(d, "IMG_9000.HEIC"), "wb").write(bytes([70]))
        r = source.probe(d)
        self.assertEqual((r["kind"], r["accepted"], r["title"]), ("stills", True, "Photographs"))
        self.assertEqual(r["ingest"], ["--frames", d])
        self.assertEqual(r["stills"]["count"], 8)
        self.assertEqual((r["stills"]["renamed"], r["stills"]["heic"], r["stills"]["size"]), (8, 1, [64, 48]))
        self.assertEqual(r["stills"]["kind"], "array", "two prefixes: not one numbered sequence")
        self.assertTrue(any("DSC_0001 → DSC-0001" in n for n in r["notes"]))
        self.assertTrue(any("HEIC" in n for n in r["notes"]))
        self.assertEqual(r["stem"], "DCIM")
        self.assertRegex(r["date"], r"^\d{4}-\d\d-\d\d$")
        self.assertEqual(os.listdir(self.tmp), ["card"], "a probe writes nothing")

    def test_one_photograph_stands_for_its_folder(self):
        d = self.stills([f"IMG_{i:04d}" for i in range(1, 6)])
        r = source.probe(os.path.join(d, "IMG_0003.JPG"))
        self.assertEqual((r["kind"], r["stills"]["count"], r["stills"]["kind"]), ("stills", 5, "mono"))
        self.assertEqual(r["ingest"], ["--frames", d])
        self.assertTrue(any("whole folder" in n for n in r["notes"]))

    def test_too_few_and_none(self):
        r = source.probe(self.stills(["IMG_0001", "IMG_0002"]))
        self.assertEqual((r["kind"], r["accepted"]), ("stills", False))
        self.assertIn("at least three", r["problems"][0])
        empty = os.path.join(self.tmp, "empty")
        os.makedirs(empty)
        r = source.probe(empty)
        self.assertEqual((r["kind"], r["accepted"]), ("unknown", False))
        r = source.probe(os.path.join(self.tmp, "nowhere"))
        self.assertEqual((r["kind"], r["accepted"]), ("unknown", False))
        self.assertIn("no such file or folder", r["problems"][0])

    def test_a_red_folder_lists_its_takes(self):
        root = self.rdm_tree()
        r = source.probe(root)
        self.assertEqual((r["kind"], r["accepted"]), ("r3d", True))
        self.assertEqual([(t["take"], t["cameras"], t["count"], t["problem"]) for t in r["r3d"]["takes"]],
                         [("067", ["GA", "GB", "HA"], 3, None), ("068", ["GA", "GB", "HA"], 3, None)])
        self.assertRegex(r["r3d"]["takes"][0]["date"], r"^\d{4}-\d\d-\d\d$")
        self.assertEqual(r["ingest"], ["--r3d", root, "--take", "067"])
        self.assertEqual(r["r3d"]["redline"], os.path.join(FAKEBIN, "REDline"))
        self.assertEqual(r["stem"], "array067")

    def test_one_clip_of_the_array_finds_the_array_and_its_take(self):
        root = self.rdm_tree()
        clip = os.path.join(root, "GB", "G007_ZZZZZZ.RDM", "G007_B068_0403XX.RDC", "G007_B068_0403XX_001.R3D")
        for dropped in (clip, os.path.dirname(clip), os.path.dirname(os.path.dirname(clip))):
            r = source.probe(dropped)
            self.assertEqual((r["kind"], r["path"]), ("r3d", root), dropped)
            self.assertEqual(r["r3d"]["take"], "068" if "B068" in dropped else "067", dropped)
            self.assertEqual(len(r["r3d"]["takes"]), 2)

    def test_a_red_folder_without_redline_or_with_too_few_cameras(self):
        root = self.rdm_tree(cams=("GA", "GB"))
        r = source.probe(root)
        self.assertFalse(r["accepted"])
        self.assertIn("three cameras", " ".join(r["problems"]))
        root2 = os.path.join(self.tmp, "second")
        shutil.copytree(self.rdm_tree(cams=("HA", "HB", "HC"), takes=("070",)), root2)
        os.environ["HS_REDLINE"] = os.path.join(self.tmp, "no-REDline")
        r = source.probe(root2)
        self.assertFalse(r["accepted"])
        self.assertIn("REDline", " ".join(r["problems"]))

    def test_a_take_with_too_few_cameras_is_not_offered_as_ingestable(self):
        root = self.rdm_tree()
        for tk in ("068",):                                        # take 068: only GA and GB
            shutil.rmtree(os.path.join(root, "HA", "H007_ZZZZZZ.RDM", f"H007_A{tk}_0403XX.RDC"))
        clip = os.path.join(root, "GB", "G007_ZZZZZZ.RDM", "G007_B068_0403XX.RDC", "G007_B068_0403XX_001.R3D")
        r = source.probe(clip)
        self.assertEqual((r["path"], r["r3d"]["take"], r["accepted"], r["ingest"]), (root, "068", False, None))
        self.assertIn("three cameras", r["problems"][0])
        self.assertIn("take 067 has 3", r["problems"][0])
        self.assertIn(f"The array is read from {root}.", r["notes"])
        self.assertTrue(source.probe(root)["accepted"], "the folder itself proposes the take that can be used")

    def test_two_clips_for_one_camera_spoil_that_take_only(self):
        root = self.rdm_tree()
        twin = os.path.join(root, "GA", "G007_ZZZZZZ.RDM", "G007_A067_0403YY.RDC")
        os.makedirs(twin)
        open(os.path.join(twin, "G007_A067_0403YY_001.R3D"), "wb").write(b"A")
        r = source.probe(root)
        by = {t["take"]: t for t in r["r3d"]["takes"]}
        self.assertIn("two clips for camera GA take 067", by["067"]["problem"])
        self.assertIsNone(by["068"]["problem"])
        self.assertEqual((r["r3d"]["take"], r["accepted"]), ("068", True))

    def test_the_climb_stops_at_an_array_and_does_not_sweep_in_the_next_shoot(self):
        a = self.rdm_tree()
        day1 = os.path.join(self.tmp, "Shoots", "2026-04-03")
        os.makedirs(os.path.dirname(day1))
        shutil.move(a, day1)
        day2 = os.path.join(self.tmp, "Shoots", "2026-04-04")
        shutil.copytree(day1, day2)
        clip = os.path.join(day1, "GB", "G007_ZZZZZZ.RDM", "G007_B067_0403XX.RDC", "G007_B067_0403XX_001.R3D")
        r = source.probe(clip)
        self.assertEqual(r["path"], day1, "three cameras found: no need to look further up")
        self.assertEqual(r["r3d"]["takes"][0]["cameras"], ["GA", "GB", "HA"])

    def test_brackets_in_a_folder_name_and_a_link_to_the_root(self):
        top = os.path.join(self.tmp, "Shoot [day 2]")
        os.makedirs(top)
        root = os.path.join(top, "RED_Footage")
        shutil.move(self.rdm_tree(), root)
        os.symlink("/", os.path.join(root, "everything"))           # a walk that followed it would not come back
        r = source.probe(root)
        self.assertEqual((r["kind"], r["accepted"], len(r["r3d"]["takes"])), ("r3d", True, 2))
        pj = self.ingest(r3d=root, take="067")                       # ...and ingest finds what the probe found
        self.assertEqual([c["camera"] for c in pj.m["source"]["cameras"]], ["GA", "GB", "HA"])

    def test_a_hydrogen_clip_by_its_tags(self):
        clip = os.path.join(self.tmp, "VID_20260915_145235_2x1.h4v")
        open(clip, "wb").write(b"\0" * 64)
        r = source.probe(clip, os.path.join(FAKEBIN, "ffprobe"))        # the stand-in says: Holocam 2x1
        self.assertEqual((r["kind"], r["title"]), ("stereo", "Hydrogen One 3D clip"))
        self.assertEqual(r["ingest"], ["--clip", clip])
        self.assertIn("two eyes side by side", r["summary"])

    def test_the_cli_emits_one_source_metric(self):
        from hs import cli
        d = self.stills([f"IMG_{i:04d}" for i in range(1, 5)])
        self.assertEqual(cli.main(["source", d]), 0)
        evs = [json.loads(x) for x in self.out.getvalue().splitlines() if x.startswith("{")]
        m = [e for e in evs if e["ev"] == "metric" and e["name"] == "source"]
        self.assertEqual(len(m), 1)
        self.assertEqual((m[0]["stage"], m[0]["value"]["kind"], m[0]["value"]["accepted"]), ("source", "stills", True))
        self.assertEqual(evs[-1], {"ev": "done", "stage": "source", "exit": 0})


class LookBelow(Base):
    """A folder given as footage is looked into: the clip may be several folders down.

    2026-10-05: a folder holding a Hydrogen clip read "no photographs here", and the clip itself had
    to be picked. A folder is the natural thing to drop."""

    FF = os.path.join(FAKEBIN, "ffprobe")                # the stand-in: every file is a Holocam 2x1 clip

    def clip(self, *parts):
        p = os.path.join(self.tmp, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").write(b"\0" * 64)
        return p

    def photos(self, n, *parts):
        d = os.path.join(self.tmp, *parts)
        for i in range(n):
            write_jpg(os.path.join(d, f"IMG-{i:04d}.jpg"), 40 + i)
        return d

    def test_one_clip_three_folders_down_is_taken_as_if_it_had_been_given(self):
        c = self.clip("Shoot", "day1", "phone", "DCIM", "VID_20260915_145235_2x1.h4v")
        top = os.path.join(self.tmp, "Shoot")
        r = source.probe(top, self.FF)
        self.assertEqual((r["kind"], r["accepted"], r["ingest"]), ("stereo", True, ["--clip", c]))
        self.assertEqual(r["found_in"], top)
        self.assertEqual(r["notes"][0], "found inside Shoot: " + os.path.join("day1", "phone", "DCIM", os.path.basename(c)))

    def test_a_clip_lying_in_the_folder_itself(self):
        c = self.clip("Shoot", "VID_20260915_145235_2x1.h4v")
        r = source.probe(os.path.join(self.tmp, "Shoot"), self.FF)
        self.assertEqual((r["kind"], r["ingest"]), ("stereo", ["--clip", c]))

    def test_several_finds_are_listed_for_a_choice(self):
        a = self.clip("Shoot", "a", "one.mov")
        b = self.clip("Shoot", "b", "two.mp4")
        ph = self.photos(9, "Shoot", "c", "stills")
        r = source.probe(os.path.join(self.tmp, "Shoot"), self.FF)
        self.assertEqual((r["kind"], r["accepted"]), ("unknown", False))
        self.assertEqual([(c["path"], c["kind"]) for c in r["candidates"]], [(a, "video"), (b, "video"), (ph, "stills")])
        self.assertEqual(r["candidates"][2]["rel"], os.path.join("c", "stills"))
        self.assertEqual(r["candidates_more"], 0)
        self.assertIn("choose one", r["problems"][0])

    def test_no_deeper_than_four_folders_and_not_into_hidden_ones(self):
        self.clip("Shoot", "1", "2", "3", "4", "5", "deep.mov")
        self.clip("Shoot", ".Trashes", "gone.mov")
        r = source.probe(os.path.join(self.tmp, "Shoot"), self.FF)
        self.assertEqual(r["kind"], "unknown")
        self.assertIn("no clip and no folder of photographs in Shoot or up to 4 folders below it", r["problems"][0])
        self.assertNotIn("candidates", r)

    def test_a_stray_picture_does_not_hide_the_clip(self):
        write_jpg(os.path.join(self.tmp, "Shoot", "notes", "cover.jpg"), 90)
        c = self.clip("Shoot", "clips", "VID_20260915_145235_2x1.h4v")
        r = source.probe(os.path.join(self.tmp, "Shoot"), self.FF)
        self.assertEqual((r["kind"], r["ingest"]), ("stereo", ["--clip", c]))

    def test_photographs_that_were_meant_are_not_replaced_by_a_clip_below(self):
        d = self.photos(9, "Set")
        self.clip("Set", "behind-the-scenes", "bts.mov")
        r = source.probe(d, self.FF)
        self.assertEqual((r["kind"], r["stills"]["count"]), ("stills", 9))
        few = self.photos(2, "Two")                         # too few, and nothing below: its own message stands
        r = source.probe(few, self.FF)
        self.assertEqual((r["kind"], r["accepted"]), ("stills", False))
        self.assertIn("at least three", r["problems"][0])

    def test_a_project_given_as_footage_gives_its_own_source(self):
        proj = os.path.join(self.tmp, "2026-09-22_Garden")
        c = self.clip("2026-09-22_Garden", "source", "VID_20260922_112802_2x1.h4v")
        self.clip("2026-09-22_Garden", "render", "orbit_1920.mp4")
        write_jpg(os.path.join(proj, "select", "contact.jpg"), 90)
        self.photos(12, "2026-09-22_Garden", "train", "dataset", "images", "L")
        json.dump({"version": 1, "stages": {"ingest": {"status": "done"}}}, open(os.path.join(proj, "manifest.json"), "w"))
        r = source.probe(proj, self.FF)
        self.assertEqual((r["kind"], r["ingest"]), ("stereo", ["--clip", c]))
        self.assertEqual(r["notes"][0], "found inside 2026-09-22_Garden: " + os.path.join("source", os.path.basename(c)))


@unittest.skipUnless(KF.have_tools(), "needs ffmpeg with libx264 and ffprobe")
class OneCameraClip(Base):
    """An ordinary video: ingested by its tags (none), picked by hs select with --mono."""

    def setUp(self):
        super().setUp()
        self.clip = os.path.join(self.tmp, "IMG_2525.mp4")
        KF.make_clip(self.clip, mono=True)

    def select_args(self, **kw):
        a = Namespace(residual=0.8, min_gap=6, max_gap=90, search=4, max_clip=0.02, work_width=160,
                      start=0, end=-1, dry_run=False, keyframes=False, search_keyframes=2, min_sharp_rel=0.6,
                      highlight_knee=None, ffprobe="ffprobe")
        for k, v in kw.items():
            setattr(a, k, v)
        return a

    def test_the_probe_says_one_camera(self):
        r = source.probe(self.clip)
        self.assertEqual((r["kind"], r["accepted"], r["title"]), ("video", True, "Video from one camera"))
        self.assertEqual(r["ingest"], ["--clip", self.clip])
        self.assertEqual((r["video"]["width"], r["video"]["height"], r["video"]["nb_frames"]), (320, 180, 60))
        self.assertEqual(r["stem"], "IMG-2525")

    def test_ingest_then_select_then_the_frames_route(self):
        pj = self.ingest(clip=self.clip)
        src = pj.m["source"]
        self.assertEqual((src["kind"], src["clip"]), ("mono", "source/IMG_2525.mp4"))
        self.assertEqual((src["probe"]["width"], src["probe"]["nb_frames"]), (320, 60))
        self.assertIsNone(pj.m["profile_id"])
        self.assertEqual(pj.status("select"), "pending", "the picks are made in the project, not before it")
        self.assertTrue(pj.frames_route)
        self.assertTrue({c["name"]: c for c in pj.stage("ingest")["checks"]}["clip_is_video"]["ok"])
        self.assertTrue(os.path.exists(pj.path("source", "IMG_2525.mp4.md5")))

        select.run(self.select_args(), pj)
        pj = Project(self.root)
        self.assertEqual(pj.status("select"), "done")
        picks = sorted(f for f in os.listdir(pj.frames_dir) if f.endswith(".jpg"))
        self.assertGreaterEqual(len(picks), 3)
        self.assertTrue(all(sourceprobe.safe_stem(os.path.splitext(f)[0]) == os.path.splitext(f)[0] for f in picks))
        self.assertRegex(picks[0], r"^sel000-\d{5}\.jpg$")
        q = json.load(open(pj.path("select", "quality.json")))
        self.assertEqual((q["eyes"], q["measured_eyes"], len(q["frames"])), (1, True, len(picks)))
        f = q["frames"][0]
        self.assertEqual((f["file"], f["sharp_R"], f["eye_ev"]), (picks[0], None, None))
        self.assertTrue(os.path.exists(pj.path("select", f["thumb"])))
        self.assertNotIn("right_soft", q["flag_counts"])
        self.assertNotIn("eye_exposure", q["flag_counts"])
        self.assertEqual(q["exposure_reference"]["cap"], os.path.splitext(q["frames"][[x["sel"] for x in q["frames"]].index(
            q["exposure_reference"]["sel"])]["file"])[0], "a one-camera capture is named after its file")
        self.assertTrue(os.path.exists(pj.path("select", "contact.jpg")))
        c = {x["name"]: x for x in pj.stage("select")["checks"]}
        self.assertIn("one camera", c["frame_count_in_range"]["value"])

    def test_frames_ingested_already_picked_still_have_nothing_to_select(self):
        d = os.path.join(self.tmp, "picks")
        for i, n in enumerate(["sel000-00000", "sel001-00009", "sel002-00020"]):
            write_jpg(os.path.join(d, n + ".jpg"), 50 + 20 * i)
        pj = self.ingest(frames=d)
        with self.assertRaisesRegex(events.StageError, "already picked"):
            select.run(self.select_args(), pj)

    def test_one_picture_is_not_a_clip(self):
        import cv2
        for name in ("plate.bmp", "IMG_0001.dng"):                 # ffprobe calls either a video stream
            p = os.path.join(self.tmp, name)
            cv2.imwrite(p if name.endswith(".bmp") else p + ".tif", np.full((48, 64, 3), 90, np.uint8))
            if not name.endswith(".bmp"):
                os.rename(p + ".tif", p)
            r = source.probe(p)
            self.assertEqual((r["kind"], r["accepted"]), ("unknown", False), name)
            self.assertIn("single picture", r["problems"][0])
        pj = Project(self.root, create=True)
        with self.assertRaisesRegex(events.StageError, "single picture"):
            ingest.run(self.args(clip=os.path.join(self.tmp, "plate.bmp")), pj)
        self.assertFalse({c["name"]: c for c in pj.stage("ingest")["checks"]}["clip_is_video"]["ok"])

    def test_a_file_ffprobe_cannot_read_is_the_projects_record_not_an_empty_folder(self):
        junk = os.path.join(self.tmp, "notes.mov")
        open(junk, "wb").write(b"not a movie")
        pj = Project(self.root, create=True)
        with self.assertRaisesRegex(events.StageError, "ffprobe cannot read notes.mov"):
            ingest.run(self.args(clip=junk), pj)
        self.assertFalse({c["name"]: c for c in pj.stage("ingest")["checks"]}["clip_is_video"]["ok"])
        self.assertEqual(pj.status("ingest"), "running", "cli.py turns the raise into failed")

    def test_the_golden_tests_ingest_call_still_works(self):
        # hs selftest gives ingest a clip and nothing about a phone
        with self.assertRaisesRegex(events.StageError, "clip not found"):
            ingest.run(Namespace(clip=os.path.join(self.tmp, "absent.h4v"), link=True, profile=None, ffprobe="ffprobe"),
                       Project(self.root, create=True))

    def test_a_hydrogen_clip_stripped_of_its_tags_is_not_read_as_one_wide_picture(self):
        wide = os.path.join(self.tmp, "VID_20260915_145235_2x1.mp4")
        KF.make_clip(wide)                                         # 640x180, two eyes, no leia tags
        r = source.probe(wide)
        self.assertEqual((r["kind"], r["accepted"]), ("video", False))
        with self.assertRaisesRegex(events.StageError, "leia3d tags"):
            self.ingest(clip=wide)
        self.assertEqual(os.listdir(Project(self.root).path("source")) if os.path.isdir(Project(self.root).path("source")) else [], [],
                         "nothing is copied before the clip is accepted")


class HandPickedReference(unittest.TestCase):
    """Exposure ▸ "A frame I pick" sends capNNN; one camera's views are named after the picks."""

    def ref(self, text, views):
        from hs.stages import exposure
        med = {v: np.array([i + 1.0] * 3) for i, v in enumerate(views)}
        target, cap, why = exposure._reference(Namespace(reference=text), None, views, med, np.array(list(med.values())))
        return cap, float(target[0]), why

    def test_pick_n_is_found_by_its_number_or_its_name(self):
        views = [("L", "sel000-00000.jpg"), ("L", "sel001-00009.jpg"), ("L", "sel002-00018.jpg")]
        self.assertEqual(self.ref("cap002", views), ("sel002-00018", 3.0, "chosen by hand"))
        self.assertEqual(self.ref("1", views), ("sel001-00009", 2.0, "chosen by hand"))
        self.assertEqual(self.ref("sel002-00018", views), ("sel002-00018", 3.0, "chosen by hand"))
        with self.assertRaisesRegex(events.StageError, "no left-eye view cap007"):
            self.ref("cap007", views)

    def test_a_stereo_capture_is_found_as_before(self):
        views = [("L", "cap000.jpg"), ("R", "cap000.jpg"), ("L", "cap001.jpg"), ("R", "cap001.jpg")]
        self.assertEqual(self.ref("cap001", views)[:2], ("cap001", 3.0))
        self.assertEqual(self.ref("1", views)[0], "cap001")


class ReferenceName(unittest.TestCase):
    def test_a_one_camera_reference_is_named_after_its_file(self):
        frames = [{"sel": i, "frame": 9 * i, "file": f"sel{i:03d}-{9 * i:05d}.jpg", "ev": 0.0, "clip": 0.01 * (3 - i),
                   "flags": []} for i in range(3)]
        self.assertEqual(frame_quality.pick_reference(frames)["cap"], "cap002")
        self.assertEqual(frame_quality.pick_reference(frames, mono=True)["cap"], "sel002-00018")
        for f in frames:
            f["file"] = None                                         # a dry run measured no files
        self.assertEqual(frame_quality.pick_reference(frames, mono=True)["cap"], "sel002-00018")


if __name__ == "__main__":
    unittest.main()
