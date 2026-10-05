"""hs train --recipe KIND --effort quick|standard|final (docs/ui-rebuild.md's recipe table).

resolve_recipe() is the whole mapping; it is checked row by row against the table here, with
the project's state (masks there or not, scan there or not) given explicitly. One run through the
fake Brush then checks that what the table says reaches brush_argv and the manifest.
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
from hs.stages import train  # noqa: E402

FAKEBIN = os.path.join(HERE, "fakebin")
MASKED = ("matte", "glossy", "bright", "person")


def args(**kw):
    """A parser-shaped Namespace: every recipe-affected flag unset, as argparse leaves it."""
    base = dict(brush=os.path.join(FAKEBIN, "brush"), total_train_iters=None, growth_stop_iter=None, refine_every=None,
                split_at_screen_size=None, export_every=10, resume_from=None, start_iter=None, no_caffeinate=True,
                brush_args="", exclude="", no_masks=False, allow_unreviewed_masks=True, min_scale_factor=None,
                layer=None, alpha_mode=None, init=None, depth_weight=None, depth_spread_weight=None,
                depth_tolerance=0.01, depth_res=512, depth_every=1, depth_spread_tolerance=0.005,
                depth_spread_from=None, recipe=None, effort=None)
    base.update(kw)
    return Namespace(**base)


class Table(unittest.TestCase):
    """resolve_recipe against the two tables; pj is only read for the source's long edge."""

    def resolve(self, a, have_masks=True, have_scan=True, long_edge=4032):
        return train.resolve_recipe(a, None, have_masks=have_masks, have_scan=have_scan, long_edge=long_edge)

    def test_masked_kinds_with_a_scan(self):
        for kind in MASKED:
            a = args(recipe=kind)
            rec = self.resolve(a)
            self.assertEqual((a.layer, a.alpha_mode, a.init), ("subject", "transparent", "lidar"), kind)
            self.assertEqual((a.depth_weight, a.depth_spread_weight), (0.2, 0.2), kind)
            self.assertEqual(rec["sh_degree"], 3)
            self.assertEqual(rec["recipe"], kind)
            self.assertEqual(rec["overridden"], [])
            # no effort: the plain defaults, and nothing about the resolution
            self.assertEqual((a.total_train_iters, a.growth_stop_iter, a.refine_every), (40000, 30000, 130))
            self.assertEqual(a.brush_args, "")
            self.assertIsNone(rec["max_resolution"])

    def test_without_a_scan_only_glossy_keeps_the_spread_term(self):
        for kind in MASKED:
            a = args(recipe=kind)
            self.resolve(a, have_scan=False)
            self.assertEqual(a.init, "sparse", kind)
            self.assertEqual(a.depth_weight, 0.0, kind)
            self.assertEqual(a.depth_spread_weight, 0.2 if kind == "glossy" else 0.0, kind)

    def test_scene_is_the_full_layer(self):
        a = args(recipe="scene")
        rec = self.resolve(a)
        self.assertEqual(a.layer, "full")
        self.assertIsNone(a.alpha_mode)
        self.assertEqual(a.init, "lidar")                  # lidar if present, as the table says
        self.assertEqual((a.depth_weight, a.depth_spread_weight), (0.0, 0.0))
        self.assertNotIn("masks_missing", rec)

    def test_a_masked_kind_without_masks_falls_back_to_full_and_says_so(self):
        a = args(recipe="matte")
        rec = self.resolve(a, have_masks=False)
        self.assertEqual(a.layer, "full")
        self.assertIsNone(a.alpha_mode)
        self.assertTrue(rec["masks_missing"])

    def test_efforts(self):
        for effort, (res, iters, growth, refine) in train.EFFORTS.items():
            a = args(effort=effort)
            rec = self.resolve(a, long_edge=4032)
            self.assertEqual((a.total_train_iters, a.growth_stop_iter, a.refine_every), (iters, growth, refine), effort)
            want = res if res is not None else 3840
            self.assertEqual(a.brush_args, f"--max-resolution {want}", effort)
            self.assertEqual(rec["max_resolution"], want)
            self.assertEqual(rec["settings"]["max_resolution"]["source"], "effort")
        self.assertEqual(train.EFFORTS["quick"], (1920, 20000, 15000, 130))
        self.assertEqual(train.EFFORTS["standard"], (1920, 40000, 30000, 130))

    def test_final_uses_the_source_long_edge_up_to_3840(self):
        a = args(effort="final")
        self.resolve(a, long_edge=2160)
        self.assertEqual(a.brush_args, "--max-resolution 2160")
        a = args(effort="final")
        self.resolve(a, long_edge=8192)
        self.assertEqual(a.brush_args, "--max-resolution 3840")

    def test_explicit_flags_win_over_the_recipe(self):
        a = args(recipe="glossy", effort="quick", total_train_iters=5000, init="sparse", layer="full",
                 depth_spread_weight=0.05, brush_args="--max-resolution 1024 --sh-degree 2")
        rec = self.resolve(a)
        self.assertEqual(a.total_train_iters, 5000)
        self.assertEqual(a.growth_stop_iter, 15000)          # not given: the effort's
        self.assertEqual(a.init, "sparse")
        self.assertEqual(a.layer, "full")
        self.assertEqual(a.depth_spread_weight, 0.05)
        self.assertEqual(a.depth_weight, 0.2)                 # not given: the recipe's
        self.assertEqual(a.brush_args, "--max-resolution 1024 --sh-degree 2", "the user's brush-args are kept as they are")
        self.assertEqual(rec["max_resolution"], 1024)
        for name in ("total_train_iters", "init", "layer", "depth_spread_weight", "max_resolution"):
            self.assertIn(name, rec["overridden"], name)
        self.assertEqual(rec["settings"]["growth_stop_iter"]["source"], "effort")
        self.assertEqual(rec["settings"]["total_train_iters"]["source"], "flag")

    def test_no_masks_flag_beats_a_masked_recipe(self):
        a = args(recipe="person", no_masks=True)
        self.resolve(a)
        self.assertEqual(a.layer, "full")
        self.assertIsNone(a.alpha_mode)

    def test_user_brush_args_are_merged_not_replaced(self):
        a = args(effort="standard", brush_args="--sh-degree 2")
        self.resolve(a)
        self.assertEqual(a.brush_args, "--sh-degree 2 --max-resolution 1920")

    def test_a_resume_keeps_its_own_initial_splats(self):
        a = args(recipe="matte", resume_from="/x/export_00100.ply")
        self.resolve(a)
        self.assertEqual(a.init, "sparse")

    def test_nothing_given_is_the_old_defaults(self):
        a = args()
        rec = self.resolve(a)
        self.assertEqual((a.total_train_iters, a.growth_stop_iter, a.refine_every, a.init), (40000, 30000, 130, "sparse"))
        self.assertEqual((a.depth_weight, a.depth_spread_weight), (0.0, 0.0))
        self.assertIsNone(rec["recipe"])
        self.assertIsNone(rec["effort"])
        self.assertEqual(a.brush_args, "")

    def test_unknown_words_are_refused(self):
        with self.assertRaises(events.StageError):
            self.resolve(args(recipe="shiny"))
        with self.assertRaises(events.StageError):
            self.resolve(args(effort="forever"))


class Run(unittest.TestCase):
    """The recipe through hs train with the fake Brush: the flags reach brush_argv and the manifest."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hs-recipe-")
        self._redir = contextlib.redirect_stdout(io.StringIO())
        self._redir.__enter__()
        os.environ["HS_PYTHON"] = sys.executable
        os.environ["HS_FAKE_REFINE_STOP"] = "0"
        os.environ["HS_FAKE_QUIET_TAIL"] = "0"

    def tearDown(self):
        self._redir.__exit__(None, None, None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def project(self, masks):
        import cv2
        pj = Project(os.path.join(self.tmp, "proj"), create=True)
        for s in ("ingest", "select"):
            pj.m["stages"][s] = {"status": "done"}
        pj.m["stages"]["solve"] = {"status": "done", "metrics": {"num_frames": 2, "view_size_L": [2160, 3840]},
                                   "started": now_iso(), "finished": now_iso()}
        os.makedirs(os.path.join(pj.dataset_dir, "sparse"))
        open(os.path.join(pj.dataset_dir, "sparse", "cameras.txt"), "w").write("# cameras\n")
        np.savez(pj.rig_npz, names=np.array(["cap000_L", "cap001_L"]), K=np.stack([np.eye(3)] * 2),
                 R=np.stack([np.eye(3)] * 2), t=np.zeros((2, 3)), pts=np.random.default_rng(1).normal(size=(50, 3)), w=32, h=24)
        for c in ("cap000", "cap001"):
            p = os.path.join(pj.dataset_dir, "images", "L", f"{c}.jpg")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            cv2.imwrite(p, np.full((24, 32, 3), 90, np.uint8))
            if masks:
                m = os.path.join(pj.dataset_dir, "masks", "L", f"{c}.png")
                os.makedirs(os.path.dirname(m), exist_ok=True)
                cv2.imwrite(m, np.full((24, 32), 255, np.uint8))
        pj.save()
        return pj

    def test_quick_glossy_run_records_the_recipe(self):
        os.environ["HS_FAKE_BRUSH_FLAGS"] = "--alpha-mode=,--sh-degree=,--depth-spread-weight=,--depth-spread-from-iter=,--depth-loss-weight="
        try:
            pj = self.project(masks=True)
            a = args(recipe="glossy", effort="quick", total_train_iters=40, growth_stop_iter=30, refine_every=10)
            train.run(a, pj)
        finally:
            os.environ.pop("HS_FAKE_BRUSH_FLAGS", None)
        st = pj.stage("train")
        self.assertEqual(st["status"], "done")
        m = st["metrics"]
        self.assertEqual(m["recipe"], "glossy")
        self.assertEqual(m["effort"], "quick")
        self.assertEqual(m["layer"], "subject")
        self.assertEqual(m["init"], "sparse")                         # no scan on this project
        self.assertEqual(m["brush_config"]["recipe"], "glossy")
        self.assertEqual(m["brush_config"]["effort"], "quick")
        self.assertEqual(m["brush_config"]["max_resolution"], 1920)
        self.assertEqual(m["brush_config"]["alpha_mode"], "transparent")
        self.assertEqual(m["brush_config"]["depth_spread_weight"], 0.2)   # glossy keeps it without a scan
        self.assertEqual(m["brush_config"]["depth_loss_weight"], 0.0)
        self.assertEqual(m["recipe_settings"]["overridden"], ["total_train_iters", "growth_stop_iter", "refine_every"])
        argv = st["brush_argv"]
        self.assertIn("--max-resolution", argv)
        self.assertEqual(argv[argv.index("--max-resolution") + 1], "1920")
        self.assertEqual(argv[argv.index("--total-train-iters") + 1], "40")
        self.assertEqual(argv[argv.index("--alpha-mode") + 1], "transparent")
        self.assertEqual(argv[argv.index("--sh-degree") + 1], "3")
        self.assertEqual(argv[argv.index("--depth-spread-weight") + 1], "0.2")
        self.assertEqual(json.load(open(pj.manifest_path))["stages"]["train"]["metrics"]["effort"], "quick")

    def test_final_without_masks_trains_the_full_scene_and_flags_it(self):
        pj = self.project(masks=False)
        a = args(recipe="matte", effort="final", total_train_iters=40, growth_stop_iter=30, refine_every=10)
        train.run(a, pj)
        st = pj.stage("train")
        self.assertEqual(st["metrics"]["layer"], "full")
        self.assertEqual(st["metrics"]["brush_config"]["max_resolution"], 3840)   # view_size_L 2160x3840
        c = {x["name"]: x for x in st["checks"]}
        self.assertFalse(c["recipe_masks_missing"]["ok"])
        self.assertTrue(c["recipe_masks_missing"]["needs_human"])
        argv = st["brush_argv"]
        self.assertEqual(argv[argv.index("--max-resolution") + 1], "3840")
        self.assertNotIn("--sh-degree", argv, "the fake binary without the flag is not handed it")


if __name__ == "__main__":
    unittest.main()
