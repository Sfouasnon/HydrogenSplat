"""hs split: subject / background labels for a trained model's splats, lifted from the 2D masks.

The scene (splat_scene.py) is a white card in front of a wall, twelve cameras, masks rendered
from the card alone. The labels come out of FlashSplat's closed form p = sum(w M) / sum(w) over
the training views, so these tests check what that promises and what the stage adds around it:
both clusters recovered, a splat that straddles the silhouette left honestly ambiguous before
the KNN refine, the hold-outs never used, and the layers written as plys hs render can take.
"""
import json
import os
import sys
import tempfile
import unittest
from argparse import Namespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import splat_scene as ss  # noqa: E402
from hs import events, splatweights as sw  # noqa: E402
from hs.cli import build_parser  # noqa: E402
from hs.stages import split  # noqa: E402
from hs.stages.merge import read_header  # noqa: E402


def args(**kw):
    base = dict(ply=None, masks=None, exclude=None, name=None, cell=4, bias=0.0, refine="knn", k=16,
                iters=4, colour_scale=0.2, max_cells_per_splat=sw.MAX_CELLS_PER_SPLAT, jobs=None)
    base.update(kw)
    return Namespace(**base)


class SplitBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        rows, lab = ss.card_and_room()
        # the straddler: a splat centred on the card's right edge, a little in front of it, that
        # the masks were NOT drawn from -- half its footprint falls inside the silhouette
        edge = ss.splat_rows(np.array([[42.0, 0.0, -2.0]]), np.array([[6.0, 6.0, 0.3]]), 0.9,
                             np.array([[0.9, 0.9, 0.9]]))
        self.lab = lab
        self.n = len(lab)
        self.pj, self.ply = ss.build_project(self.tmp.name, np.concatenate([rows, edge]),
                                             np.concatenate([lab, [0]]), truth_rows=rows)

    def p_of(self, name="export_01000"):
        return sw.Splats(self.pj.path("split", name, "full_labelled.ply")).arr["subject_p"].astype(float)


class Split(SplitBase):
    def test_recovers_card_and_room(self):
        split.run(args(), self.pj)
        p = self.p_of()[:self.n]
        card, room = p[self.lab == 1], p[self.lab == 0]
        self.assertGreaterEqual((card >= 0.5).mean(), 0.99, f"card labels: {np.sort(card)[:5]}")
        self.assertGreaterEqual((room < 0.5).mean(), 0.99, f"room labels: {np.sort(room)[-5:]}")

    def test_the_straddler_is_ambiguous_before_refine(self):
        split.run(args(refine="none"), self.pj)
        p = self.p_of()
        self.assertTrue(0.2 < p[-1] < 0.8, f"straddler p = {p[-1]:.3f}")
        rep = json.load(open(self.pj.path("split", "export_01000", "report.json")))
        self.assertGreaterEqual(rep["ambiguous"]["before_refine"], 1)
        self.assertEqual(rep["ambiguous"]["before_refine"], rep["ambiguous"]["after_refine"])

    def test_refine_resolves_the_band_and_the_check_says_so(self):
        split.run(args(), self.pj)
        rep = json.load(open(self.pj.path("split", "export_01000", "report.json")))
        self.assertLess(rep["ambiguous"]["after_refine"], rep["ambiguous"]["before_refine"])
        checks = {c["name"]: c for c in self.pj.stage("split")["checks"]}
        self.assertTrue(checks["labels_bimodal"]["ok"])
        self.assertTrue(checks["masks_cover_training_views"]["ok"])

    def test_layers_are_written_and_add_up(self):
        split.run(args(), self.pj)
        d = self.pj.path("split", "export_01000")
        n_all = read_header(os.path.join(d, "full_labelled.ply"))[1]
        n_sub = read_header(os.path.join(d, "subject.ply"))[1]
        n_bg = read_header(os.path.join(d, "background.ply"))[1]
        self.assertEqual(n_all, self.n + 1)
        self.assertEqual(n_sub + n_bg, n_all)
        self.assertEqual(n_sub, int((self.p_of() >= 0.5).sum()))
        # the layers keep the source's properties exactly; only the labelled copy gains subject_p
        src = [x for x, _t in read_header(self.ply)[2]]
        self.assertEqual([x for x, _t in read_header(os.path.join(d, "subject.ply"))[2]], src)
        self.assertEqual([x for x, _t in read_header(os.path.join(d, "full_labelled.ply"))[2]], src + ["subject_p"])
        rep = json.load(open(os.path.join(d, "report.json")))
        for k in ("counts", "p_histogram", "ambiguous", "views_used", "mask_source", "seconds"):
            self.assertIn(k, rep)
        self.assertEqual(sum(rep["p_histogram"]["counts"]), n_all)
        man = json.load(open(os.path.join(d, "manifest.json")))
        self.assertEqual(man["rig_npz_md5"], rep["rig_npz_md5"])
        arts = {a["path"] for a in self.pj.stage("split")["artifacts"]}
        for f in ("full_labelled.ply", "subject.ply", "background.ply", "report.json"):
            self.assertIn(os.path.join("split", "export_01000", f), arts)
        self.assertEqual(self.pj.stage("split")["status"], "done")

    def test_split_changes_no_other_stage(self):
        before = {k: v.get("status") for k, v in self.pj.m["stages"].items()}
        split.run(args(), self.pj)
        after = {k: v.get("status") for k, v in self.pj.m["stages"].items() if k != "split"}
        self.assertEqual(after, {k: v for k, v in before.items() if k != "split"})


class HoldOuts(SplitBase):
    def test_default_is_what_the_model_was_trained_without(self):
        self.pj.m["stages"]["train"]["metrics"]["dataset_fingerprint"]["excluded_views"] = ["L/cap003", "L/cap007"]
        self.pj.save()
        split.run(args(), self.pj)
        rep = json.load(open(self.pj.path("split", "export_01000", "report.json")))
        self.assertNotIn("cap003_L", rep["views_used"])
        self.assertNotIn("cap007_L", rep["views_used"])
        self.assertEqual(len(rep["views_used"]), 10)
        self.assertEqual(rep["exclude_source"], "train stage")

    def test_explicit_list_and_holdout_file(self):
        split.run(args(exclude="L/cap000,cap001_L", name="a"), self.pj)
        rep = json.load(open(self.pj.path("split", "a", "report.json")))
        self.assertEqual(sorted(rep["excluded_views"]), ["L/cap000", "L/cap001"])
        self.assertEqual(len(rep["views_used"]), 10)
        os.makedirs(self.pj.path("solve"), exist_ok=True)
        json.dump({"method": "fps", "n": 2, "names": ["cap004", "cap009"], "captures": [4, 9],
                   "exclude": ["L/cap004", "R/cap004", "L/cap009", "R/cap009"], "stereo": True},
                  open(self.pj.path("solve", "holdout.json"), "w"))
        split.run(args(exclude="@holdout", name="b"), self.pj)
        rep = json.load(open(self.pj.path("split", "b", "report.json")))
        self.assertIn("L/cap004", rep["excluded_views"])
        self.assertNotIn("cap009_L", rep["views_used"])

    def test_archive_manifest_names_its_holdouts(self):
        import shutil
        d = self.pj.path("archive", "base")
        os.makedirs(d)
        shutil.copy(self.ply, os.path.join(d, "export_01000.ply"))
        json.dump({"train_dataset_fingerprint": {"excluded_views": ["L/cap011"]}},
                  open(os.path.join(d, "manifest.json"), "w"))
        split.run(args(ply="archive/base/export_01000.ply"), self.pj)
        rep = json.load(open(self.pj.path("split", "base", "report.json")))   # named after the archive
        self.assertEqual(rep["excluded_views"], ["L/cap011"])
        self.assertEqual(rep["exclude_source"], "archive manifest")


class Refusals(SplitBase):
    def test_masks_made_by_the_other_method_are_refused(self):
        with self.assertRaises(events.StageError):
            split.run(args(masks="vision"), self.pj)
        self.pj.release()
        split.run(args(masks="region", name="r"), self.pj)          # geometry masks answer to 'region'

    def test_a_mask_folder_can_be_given(self):
        import shutil
        shutil.copytree(self.pj.path("train", "dataset", "masks"), self.pj.path("mymasks"))
        os.remove(self.pj.path("mymasks", "L", "cap005.png"))
        split.run(args(masks="mymasks"), self.pj)
        rep = json.load(open(self.pj.path("split", "export_01000", "report.json")))
        self.assertEqual(rep["training_views_without_mask"], ["cap005_L"])
        self.assertEqual(len(rep["views_used"]), 11)

    def test_needs_train_or_a_ply(self):
        self.pj.m["stages"]["train"]["status"] = "pending"
        self.pj.save()
        with self.assertRaises(events.StageError):
            split.run(args(), self.pj)
        split.run(args(ply=self.ply), self.pj)

    def test_registered_on_the_cli(self):
        a = build_parser().parse_args(["split", "-p", "/x", "--ply", "m.ply", "--exclude", "@holdout",
                                       "--masks", "vision", "--refine", "none", "--cell", "8"])
        self.assertEqual((a.cmd, a.project, a.masks, a.refine, a.cell), ("split", "/x", "vision", "none", 8))


class Refine(unittest.TestCase):
    def test_an_ambiguous_splat_takes_its_neighbours_label(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, (200, 3))
        b = rng.normal(0, 1, (200, 3)) + [10, 0, 0]
        xyz = np.vstack([a, b, [[0.2, 0, 0]], [[9.8, 0, 0]]])
        rgb = np.full((len(xyz), 3), 0.5)
        p = np.concatenate([np.ones(200), np.zeros(200), [0.55], [np.nan]])   # one ambiguous, one unseen
        q, n = split.refine_knn(p, xyz, rgb, k=8, iters=4)
        self.assertEqual(n, 2)
        self.assertGreater(q[400], 0.9)
        self.assertLess(q[401], 0.1)
        np.testing.assert_array_equal(q[:400], p[:400])                   # confident splats never move


if __name__ == "__main__":
    unittest.main()
