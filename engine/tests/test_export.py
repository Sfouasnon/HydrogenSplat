#!/usr/bin/env python3
"""hs export against the fake splat-transform (tests/fakebin/splat-transform).

Asserts the argv the tool received (the real v3.5.1 syntax: [GLOBAL] input [ACTIONS] output),
the deliver/<name>/ layout, md5s, the opacity filter, the GPU -> cpu retry, and that the shot
sheet carries the archive's splat count. Every test builds its own project in a temp folder.
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

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(HERE)
sys.path.insert(0, ENGINE)

import numpy as np  # noqa: E402

from hs import cli  # noqa: E402
from hs.project import Project, md5_file, now_iso  # noqa: E402
from hs.stages import archive, export, tools  # noqa: E402

FAKE = os.path.join(HERE, "fakebin", "splat-transform")
PROPS = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(9)] + \
        ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
N = 137


def write_ply(path, n=N, seed=0):
    rng = np.random.default_rng(seed)
    arr = rng.normal(size=(n, len(PROPS))).astype("<f4")
    arr[:, PROPS.index("opacity")] = rng.normal(0, 2, n)
    hdr = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n
           + "".join(f"property float {p}\n" for p in PROPS) + "end_header\n")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(hdr.encode() + arr.tobytes())
    return arr


def n_opaque(arr, thr):
    return int((1 / (1 + np.exp(-arr[:, PROPS.index("opacity")].astype(np.float64))) >= thr).sum())


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hs-export-")
        self.root = os.path.join(self.tmp, "proj")
        self.log = os.path.join(self.tmp, "st_argv.jsonl")
        self.env = {k: os.environ.get(k) for k in ("HS_PYTHON", "HS_FAKE_ST_LOG", "HS_FAKE_ST_NO_GPU",
                                                   "HS_NO_CAFFEINATE", "HS_SPLAT_TRANSFORM")}
        os.environ.update({"HS_PYTHON": sys.executable, "HS_FAKE_ST_LOG": self.log, "HS_NO_CAFFEINATE": "1"})
        os.environ.pop("HS_FAKE_ST_NO_GPU", None)
        os.environ.pop("HS_SPLAT_TRANSFORM", None)

    def tearDown(self):
        for k, v in self.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(self.tmp, ignore_errors=True)

    def hs(self, *args):
        """Run `hs` in-process; -> (exit code, [events])."""
        out = io.StringIO()
        old = sys.argv
        sys.argv = ["hs"] + list(args)
        try:
            with contextlib.redirect_stdout(out):
                code = cli.main(list(args))
        finally:
            sys.argv = old
        evs = [json.loads(l) for l in out.getvalue().splitlines() if l.startswith("{")]
        return code, evs

    def argvs(self):
        if not os.path.exists(self.log):
            return []
        # the fake logs its arguments; the program is FAKE whenever a test passes it
        return [[FAKE] + json.loads(l) for l in open(self.log) if l.strip()]

    def conversions(self):
        return [a for a in self.argvs() if "--version" not in a]

    def trained(self, n=N, layer="subject"):
        pj = Project(self.root, create=True)
        for s in ("ingest", "select", "solve"):
            pj.m["stages"][s] = {"status": "done", "metrics": {}}
        pj.m["stages"]["solve"]["metrics"] = {"num_images": 12, "mean_reproj_px": 0.512, "num_points": 5375}
        pj.m["source"] = {"clip": "source/VID_1_2x1.h4v", "md5": "c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1"}
        os.makedirs(os.path.join(pj.dataset_dir, "sparse"))
        for f in ("cameras.txt", "images.txt"):
            open(os.path.join(pj.dataset_dir, "sparse", f), "w").write("# x\n")
        np.savez(pj.rig_npz, names=np.array(["cap000_L"]), K=np.eye(3)[None])
        final = os.path.join(pj.exports_dir, "export_00040.ply")
        arr = write_ply(final, n)
        pj.m["stages"]["train"] = {
            "status": "done", "started": now_iso(), "finished": now_iso(),
            "argv": ["hs", "train", "-p", self.root, "--layer", layer],
            "metrics": {"final_export": pj.rel(final), "final_export_md5": md5_file(final), "final_splats": n,
                        "layer": layer, "masks_used": True,
                        "brush_config": {"commit": "abc1234", "branch": "hs-fork", "dirty": False, "layer": layer},
                        "dataset_fingerprint": {"rig_npz_md5": md5_file(pj.rig_npz)}},
            "checks": [], "artifacts": []}
        pj.save()
        return pj, final, arr


class Export(Base):
    def test_default_ply_all_formats(self):
        pj, final, _ = self.trained()
        code, evs = self.hs("export", "-p", self.root, "--splat-transform", FAKE)
        self.assertEqual(code, 0, [e for e in evs if e["ev"] == "error"])
        d = os.path.join(self.root, "deliver", "export_00040")
        self.assertEqual(sorted(os.listdir(d)), sorted(f"export_00040.{x}" for x in ("ply", "spz", "sog", "html"))
                         + ["manifest.json"])
        self.assertEqual(md5_file(os.path.join(d, "export_00040.ply")), md5_file(final))
        # one splat-transform call per non-ply format, in the real syntax, no filter
        calls = self.conversions()
        self.assertEqual(calls, [[FAKE, "-w", final, os.path.join(self.root, "deliver", ".export_00040.building",
                                                                    f"export_00040.{x}")] for x in ("spz", "sog", "html")])
        man = json.load(open(os.path.join(d, "manifest.json")))
        self.assertEqual(man["source"]["md5"], md5_file(final))
        self.assertEqual(man["source"]["splats"], N)
        self.assertEqual(man["splat_transform"]["version"], "splat-transform v0.0.0-fake (fake)")
        for rec in man["files"]:
            self.assertEqual(rec["md5"], md5_file(os.path.join(d, rec["path"])))
        arts = sorted(e["path"] for e in evs if e["ev"] == "artifact")
        self.assertEqual(arts, sorted(f"deliver/export_00040/{f}" for f in os.listdir(d)))
        pj = Project(self.root)
        self.assertEqual(pj.status("export"), "done")
        self.assertIn("export_00040", pj.stage("export")["runs"])
        self.assertFalse(os.path.exists(pj.lock_path))

    def test_archive_with_shot_sheet(self):
        pj, final, _ = self.trained()
        with contextlib.redirect_stdout(io.StringIO()):
            archive.run(Namespace(name="masked", ply=None, link=False, no_images=True, force=False), pj)
        aply = "archive/masked/export_00040.ply"
        os.makedirs(pj.path("views"))
        json.dump({"ply": aply, "views": [{"psnr_db": 30.0, "displaced_fraction": 0.02},
                                          {"psnr_db": 32.0, "displaced_fraction": 0.04},
                                          {"psnr_db": 34.0, "displaced_fraction": 0.30}]},
                  open(pj.path("views", "holdout_report.json"), "w"))
        json.dump({"ply": "train/exports/export_99999.ply", "views": [{"psnr_db": 20.0}]},
                  open(pj.path("views", "old_report.json"), "w"))
        os.makedirs(pj.path("move"))
        json.dump({"fps": 30, "width": 1920, "height": 1080, "frames": [{"c2w": []}] * 90},
                  open(pj.path("move", "arc.json"), "w"))
        json.dump({"schema": 1, "fps": 30, "keys": [{"t": 0.0, "eye": [0, 0, 0], "ease": True},
                                                    {"t": 3.0, "eye": [1, 0, 0], "ease": True}]},
                  open(pj.path("move", "arc.keys.json"), "w"))
        os.makedirs(pj.path("grade"))
        json.dump({"lift": 0.02, "gamma": 1.1, "gain": 1.0, "sharpen": 0.35, "aspect": 2.35, "move": "arc_masked",
                   "output": "render/arc_masked_graded.mp4"}, open(pj.path("grade", "arc_masked.json"), "w"))
        pj = Project(self.root)
        pj.m["stages"]["render"]["runs"] = {"arc_masked": {"move": "arc", "ply": aply, "finished": now_iso()}}
        pj.save()

        code, evs = self.hs("export", "-p", self.root, "--archive", "masked", "--formats", "ply,spz",
                            "--shot-sheet", "--splat-transform", FAKE)
        self.assertEqual(code, 0, [e for e in evs if e["ev"] == "error"])
        d = pj.path("deliver", "masked")
        self.assertEqual(sorted(os.listdir(d)), ["manifest.json", "masked.ply", "masked.spz",
                                                 "shot-sheet.html", "shot-sheet.md"])
        self.assertEqual(self.conversions()[0][2], pj.path(aply))
        md = open(os.path.join(d, "shot-sheet.md")).read()
        html = open(os.path.join(d, "shot-sheet.html")).read()
        man = json.load(open(os.path.join(d, "manifest.json")))
        for text in (md, html):
            self.assertIn(f"{N:,}", text)                              # the archive's splat count
            self.assertIn("abc1234", text)                             # Brush commit
            self.assertIn("masked", text)
            self.assertIn("c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1", text)    # clip md5
            self.assertIn("0.512", text)                               # solve reprojection
            self.assertIn("arc", text)
            self.assertIn("32.00", text)                               # median PSNR of this model's report
            self.assertIn("4.0%", text)                                # median displaced
            self.assertIn("1.1", text)                                 # grade gamma
            for rec in man["files"]:
                self.assertIn(rec["md5"], text)
        self.assertIn("2 (app keyframes", md)
        self.assertIn("| holdout | archive/masked/export_00040.ply | this model |", md)
        self.assertIn("another model", md)
        self.assertEqual(sum(1 for e in evs if e["ev"] == "artifact" and e["kind"] == "shot-sheet"), 2)
        self.assertTrue(any(e.get("name") == "archive_ply_md5_matches" and e["ok"] for e in evs if e["ev"] == "check"))

    def test_min_opacity_and_subject(self):
        pj, final, arr = self.trained()
        sub = pj.path("split", "1", "subject.ply")
        sarr = write_ply(sub, 60, seed=3)
        code, evs = self.hs("export", "-p", self.root, "--min-opacity", "0.5", "--subject", "split/1/subject.ply",
                            "--formats", "ply,sog", "--name", "look", "--splat-transform", FAKE)
        self.assertEqual(code, 0, [e for e in evs if e["ev"] == "error"])
        d = pj.path("deliver", "look")
        self.assertEqual(sorted(f for f in os.listdir(d) if f != "manifest.json"),
                         ["look.ply", "look.sog", "look_subject.ply", "look_subject.sog"])
        calls = self.conversions()
        self.assertEqual(len(calls), 4)
        for c in calls:
            self.assertEqual(c[:2], [FAKE, "-w"])
            self.assertEqual(c[3:5], ["-V", "opacity,gte,0.5"])       # the filter action follows the input
        self.assertEqual([c[2] for c in calls], [final, final, sub, sub])
        man = json.load(open(os.path.join(d, "manifest.json")))
        by = {r["path"]: r for r in man["files"]}
        self.assertEqual(by["look.ply"]["splats"], n_opaque(arr, 0.5))
        self.assertEqual(by["look_subject.ply"]["splats"], n_opaque(sarr, 0.5))
        self.assertLess(by["look.ply"]["splats"], N)
        self.assertEqual(man["subject"]["ply"], "split/1/subject.ply")

    def test_gpu_failure_retries_on_cpu(self):
        self.trained()
        os.environ["HS_FAKE_ST_NO_GPU"] = "1"
        code, evs = self.hs("export", "-p", self.root, "--formats", "spz,sog", "--splat-transform", FAKE)
        self.assertEqual(code, 0, [e for e in evs if e["ev"] == "error"])
        calls = self.conversions()
        self.assertEqual(len(calls), 3)                                # spz, sog (fails), sog -g cpu
        self.assertNotIn("-g", calls[1])
        self.assertEqual(calls[2][1:4], ["-w", "-g", "cpu"])
        self.assertTrue(any(e["ev"] == "check" and e["name"] == "model_sog_on_gpu" and not e["ok"] for e in evs))
        # an explicit --gpu is respected: no silent retry
        code, evs = self.hs("export", "-p", self.root, "--formats", "sog", "--gpu", "0", "--splat-transform", FAKE)
        self.assertEqual(code, 1)
        self.assertFalse(os.path.exists(os.path.join(self.root, "deliver", ".export_00040.building")))

    def test_requires_train_unless_ply_given(self):
        Project(self.root, create=True)
        code, evs = self.hs("export", "-p", self.root)
        self.assertEqual(code, 1)
        self.assertIn("train", next(e for e in evs if e["ev"] == "error")["message"])
        ply = os.path.join(self.tmp, "loose.ply")
        write_ply(ply, 20)
        # ply only: no splat-transform needed, even a missing one is never looked at
        code, evs = self.hs("export", "-p", self.root, "--ply", ply, "--formats", "ply",
                            "--splat-transform", "/nonexistent/splat-transform", "--shot-sheet")
        self.assertEqual(code, 0, [e for e in evs if e["ev"] == "error"])
        md = open(os.path.join(self.root, "deliver", "loose", "shot-sheet.md")).read()
        self.assertIn("not archived", md)
        self.assertIn("not recorded", md)
        self.assertEqual(self.conversions(), [])

    def test_missing_tool_and_bad_args(self):
        pj, final, _ = self.trained()
        code, evs = self.hs("export", "-p", self.root, "--splat-transform", "/nonexistent/st")
        self.assertEqual(code, 1)
        self.assertIn("splat-transform not found", next(e for e in evs if e["ev"] == "error")["message"])
        self.assertEqual(self.hs("export", "-p", self.root, "--formats", "ply,glb")[0], 1)
        self.assertEqual(self.hs("export", "-p", self.root, "--min-opacity", "1.5")[0], 1)
        self.assertNotEqual(Project(self.root).status("export"), "running")

    def test_archive_md5_mismatch_refused_and_rerun_replaces_whole(self):
        pj, final, _ = self.trained()
        with contextlib.redirect_stdout(io.StringIO()):
            archive.run(Namespace(name="m", ply=None, link=False, no_images=True, force=False), pj)
        code, _ = self.hs("export", "-p", self.root, "--archive", "m", "--formats", "ply")
        self.assertEqual(code, 0)
        stray = pj.path("deliver", "m", "stray.txt")
        open(stray, "w").write("left from an older export")
        code, _ = self.hs("export", "-p", self.root, "--archive", "m", "--formats", "ply")
        self.assertEqual(code, 0)
        self.assertFalse(os.path.exists(stray))
        with open(pj.path("archive", "m", "export_00040.ply"), "ab") as f:
            f.write(b"\0")
        code, evs = self.hs("export", "-p", self.root, "--archive", "m", "--formats", "ply")
        self.assertEqual(code, 1)
        self.assertIn("archive manifest", next(e for e in evs if e["ev"] == "error")["message"])
        self.assertFalse(os.path.exists(pj.path("deliver", ".m.building")))
        self.assertEqual(Project(self.root).status("export"), "failed")


class Locate(Base):
    def test_explicit_path_env_and_npx(self):
        self.assertEqual(export.splat_transform_argv(FAKE), [FAKE])
        path = os.environ.get("PATH", "")
        try:
            fake_npx_dir = os.path.join(self.tmp, "bin")
            os.makedirs(fake_npx_dir)
            npx = os.path.join(fake_npx_dir, "npx")
            open(npx, "w").write("#!/bin/sh\nexit 0\n")
            os.chmod(npx, 0o755)
            os.environ["PATH"] = fake_npx_dir
            self.assertEqual(export.splat_transform_argv(None), [npx, "-y", "--", "@playcanvas/splat-transform"])
            os.environ["PATH"] = os.path.join(self.tmp, "empty")
            with self.assertRaises(Exception) as cm:
                export.splat_transform_argv(None)
            self.assertIn("node", str(cm.exception))
        finally:
            os.environ["PATH"] = path

    def test_env_var_reaches_parser(self):
        os.environ["HS_SPLAT_TRANSFORM"] = FAKE
        a = cli.build_parser().parse_args(["export", "-p", self.root])
        self.assertEqual(a.splat_transform, FAKE)

    def test_transform_argv(self):
        self.assertEqual(export.transform_argv(["st"], "a.ply", "b.sog"), ["st", "-w", "a.ply", "b.sog"])
        self.assertEqual(export.transform_argv(["npx", "-y", "--", "@playcanvas/splat-transform"], "a.ply", "b.html",
                                               0.05, "cpu"),
                         ["npx", "-y", "--", "@playcanvas/splat-transform", "-w", "-g", "cpu", "a.ply",
                          "-V", "opacity,gte,0.05", "b.html"])

    def test_tools_reports_splat_transform(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            info = tools.run(Namespace(brush="/nonexistent", render_bin="/nonexistent", splat_transform=FAKE))
        evs = [json.loads(l) for l in out.getvalue().splitlines() if l.startswith("{")]
        names = {e["name"]: e for e in evs if e["ev"] == "check"}
        for k in ("node", "npx", "splat-transform"):
            self.assertIn(k, names)
        self.assertTrue(names["splat-transform"]["ok"])
        self.assertIn("v0.0.0-fake", names["splat-transform"]["value"])
        self.assertEqual(info["splat-transform"]["version"], "splat-transform v0.0.0-fake (fake)")


if __name__ == "__main__":
    unittest.main()
