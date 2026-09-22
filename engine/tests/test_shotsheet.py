#!/usr/bin/env python3
"""hs.shotsheet: a pure function of manifest dicts. Missing pieces say "not recorded"."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from hs.shotsheet import NR, sections, shot_sheet  # noqa: E402

EXPORT = {"name": "head", "exported": "2026-09-21T12:00:00+0000", "project": "/p/2026-09-15_head",
          "source": {"ply": "archive/head/export_40000.ply", "md5": "a" * 32, "splats": 1034567, "archive": "head"},
          "min_opacity": 0.05, "splat_transform": {"argv": ["npx", "-y", "--", "@playcanvas/splat-transform"],
                                                   "version": "splat-transform v3.5.1 (c23730c)"},
          "files": [{"path": "head.ply", "role": "model", "format": "ply", "md5": "b" * 32, "bytes": 9000, "splats": 998000},
                    {"path": "head.sog", "role": "model", "format": "sog", "md5": "c" * 32, "bytes": 900}]}
ARCHIVE = {"name": "head", "rig_npz_md5": "r" * 32, "ply": {"source": "train/exports/export_40000.ply", "md5": "a" * 32},
           "source": {"kind": "array", "frames": "source/frames", "md5": "f" * 32, "original_path": "~/RED/067"},
           "stages": {"solve": {"status": "done", "metrics": {"mean_reproj_px": 0.51, "num_images": 12}},
                      "train": {"status": "done", "argv": ["hs", "train", "--layer", "full"],
                                "metrics": {"final_splats": 1034567, "final_export_md5": "a" * 32,
                                            "layer": "full", "brush_config": {"commit": "dd5ea36", "dirty": True}}}}}


class ShotSheet(unittest.TestCase):
    def test_everything_missing_degrades(self):
        md, html = shot_sheet({}, {})
        for sec in ("Source", "Solve", "Move", "Hold-out scores", "Grade", "Files"):
            self.assertIn(f"## {sec}\n", md)
        self.assertGreaterEqual(md.count(NR), 6)
        self.assertIn("not archived", md)
        self.assertIn("<!doctype html>", html)
        self.assertIn(NR, html)
        # rubbish in the optional inputs is tolerated, not trusted
        md, _ = shot_sheet(None, {"files": ["x", None]}, archive="junk", reports=[None, {"report": 3}], move=[],
                           grade="x")
        self.assertIn("## Hold-out scores\n\n_no views/*_report.json in the project_\n\nnot recorded", md)

    def test_full_record(self):
        project = {"name": "2026-09-15_head", "profile_id": "hydrogen-2x1",
                   "source": {"clip": "source/VID.h4v", "md5": "9" * 32},       # the archive's source wins
                   "tools": {"brush": {"git_commit": "ffff000"}}}
        reports = [{"name": "views", "relation": "this model",
                    "report": {"ply": "archive/head/export_40000.ply",
                               "views": [{"psnr_db": 28.0, "displaced_fraction": 0.1, "retained_edge_energy_norm": 0.6},
                                         {"psnr_db": 30.0, "displaced_fraction": 0.2, "retained_edge_energy_norm": 0.7}]}}]
        move = {"name": "boom", "path": {"fps": 30, "frames": [{}] * 60}, "keys": None,
                "run": {"preset": "boom", "info": {"keys": [50, 55, 57]}}, "render": "boom_head"}
        grade = {"lift": 0.02, "gamma": 1.1, "gain_rgb": [1.03, 1, 0.97], "move": "boom_head"}
        md, html = shot_sheet(project, EXPORT, archive=ARCHIVE, reports=reports, move=move, grade=grade)
        for text in (md, html):
            self.assertIn("1,034,567", text)                 # splats, header and train record
            self.assertIn("array", text)
            self.assertIn("f" * 32, text)                    # frames digest from the archive, not the clip
            self.assertNotIn("9" * 32, text)
            self.assertIn("dd5ea36", text)
            self.assertIn("dirty tree", text)
            self.assertIn("hs train --layer full", text)
            self.assertIn("2.00 s", text)                    # 60 frames at 30 fps
            self.assertIn("50, 55, 57", text)
            self.assertIn("29.00", text)                     # median PSNR
            self.assertIn("15.0%", text)                     # median displaced
            self.assertIn("0.650", text)
            self.assertIn("1.03, 1, 0.97", text)
            self.assertIn("splat-transform v3.5.1", text)
            for f in EXPORT["files"]:
                self.assertIn(f["md5"], text)
        self.assertIn("- **min opacity**: 0.05", md)
        self.assertNotIn("may describe a different model", md)

    def test_training_record_for_another_ply_is_flagged(self):
        ex = dict(EXPORT, source=dict(EXPORT["source"], md5="d" * 32))
        md, _ = shot_sheet({"stages": ARCHIVE["stages"]}, ex)
        self.assertIn("may describe a different model", md)

    def test_move_without_record(self):
        md, _ = shot_sheet({}, EXPORT, move={"why": "no render of this ply and 3 moves in move/; pass --move"})
        self.assertIn("pass --move", md)
        secs = dict((s[0], s[2]) for s in sections({}, EXPORT, move={"name": "arc", "path": {}}))
        self.assertIn(("keyframes", NR), secs["Move"])

    def test_html_escapes(self):
        _, html = shot_sheet({"name": "<script>alert(1)</script>"}, {"name": "a&b"})
        self.assertNotIn("<script>alert", html)
        self.assertIn("a&amp;b", html)


if __name__ == "__main__":
    unittest.main()
