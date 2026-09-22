"""hs prune --score / --floaters: photometric per-splat scores and the floater prune they drive.

Scene (splat_scene.py): the white card before the wall, photographed without two extra splats
the model carries. One is the floater -- dark, a quarter opaque, 15 mm in front of the card, so
it lies over the card in all twelve views and darkens it. The other is faint (7 % opaque) and
stands alone where the photographs show something bright the model never grew: it paints its
patch badly (high blame) and hardly at all (low importance), but it is the only thing painting
it (top contributor). The rule is low importance AND top contributor nowhere AND high blame:
the floater goes, the faint splat stays -- Mini-Splatting's guard is the only thing saving it,
and the test checks that too.

The test passes --max-blame 0.1: on this scene the floater's blame is 0.18, under the default
0.25 (a 25 %-opaque floater over a card cannot push a cell's linear error much further before it
becomes that cell's top contributor itself). The default is the brief's; tune it on real data.
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
from hs import splatweights as sw  # noqa: E402
from hs.cli import build_parser  # noqa: E402
from hs.project import md5_file, now_iso  # noqa: E402
from hs.stages import prune, render  # noqa: E402
from hs.stages.merge import read_header  # noqa: E402

MAX_BLAME = 0.1


def args(**kw):
    base = dict(ply=None, score=False, floaters=False, name=None, exclude=None, cell=4, max_cells_per_splat=None,
                min_importance_quantile=0.02, max_blame=MAX_BLAME, jobs=None)
    base.update(kw)
    return Namespace(**base)


class ScoreBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        rows, lab = ss.card_and_room()
        self.n = len(rows)
        floater = ss.splat_rows(np.array([[0.0, 0.0, -15.0]]), np.array([[3.0, 3.0, 3.0]]), 0.25,
                                np.array([[0.02, 0.02, 0.02]]))
        faint = ss.splat_rows(np.array([[340.0, 0.0, 250.0]]), np.array([[3.0, 3.0, 3.0]]), 0.07,
                              np.array([[0.6, 0.6, 0.6]]))
        bright = ss.splat_rows(np.array([[340.0, 0.0, 250.0]]), np.array([[6.0, 6.0, 6.0]]), 0.95,
                               np.array([[1.0, 1.0, 1.0]]))
        self.FAINT, self.FLOATER = self.n, self.n + 1
        self.pj, self.ply = ss.build_project(self.tmp.name, np.concatenate([rows, faint, floater]),
                                             np.concatenate([lab, [0, 0]]),
                                             truth_rows=np.concatenate([rows, bright]), masks=False)

    def scores(self, name="export_01000"):
        return np.load(self.pj.path("prune", f"{name}_scores.npz"))


class Score(ScoreBase):
    def test_score_alone_is_a_report(self):
        self.pj.m["stages"]["prune"] = {"status": "done", "finished": now_iso(),
                                        "metrics": {"output_ply": "prune/x_pruned_r03.ply", "input_ply_md5": "abc"}}
        self.pj.m["stages"]["render"] = {"status": "done"}
        self.pj.save()
        prune.run(args(score=True), self.pj)
        for f in ("export_01000_scores.npz", "export_01000_report.json"):
            self.assertTrue(os.path.exists(self.pj.path("prune", f)), f)
        self.assertFalse([f for f in os.listdir(self.pj.path("prune")) if f.endswith(".ply")], "--score pruned")
        st = self.pj.stage("prune")
        self.assertEqual(st["status"], "done")
        self.assertEqual(st["metrics"]["output_ply"], "prune/x_pruned_r03.ply", "the last prune's record was overwritten")
        self.assertEqual(self.pj.status("render"), "done", "a report must not stale the render")
        self.assertIn("export_01000_score", st["runs"])
        self.assertIn("would_flag_floaters", st["runs"]["export_01000_score"]["metrics"])
        self.assertFalse(os.path.exists(self.pj.lock_path))

    def test_scores_mean_what_they_say(self):
        prune.run(args(score=True), self.pj)
        z = self.scores()
        imp, blame, top, seen = z["importance"], z["blame"], z["top_contributor"], z["views_seen"]
        self.assertEqual(len(imp), self.n + 2)
        self.assertEqual(int(seen[self.FLOATER]), 12, "the floater lies over the card in every view")
        # importance: the card is what the photographs are mostly of; each card splat outweighs the floater
        card = np.arange(289)
        self.assertLess(imp[self.FLOATER], np.median(imp[card]))
        self.assertLess(imp[self.FAINT], imp[self.FLOATER])
        # blame: the two wrong splats stand out from a model that otherwise matches the photos
        live_seen = seen > 0
        self.assertGreater(blame[self.FLOATER], 10 * np.median(blame[live_seen]))
        self.assertGreater(blame[self.FAINT], 10 * np.median(blame[live_seen]))
        self.assertTrue(top[self.FAINT])
        self.assertFalse(top[self.FLOATER], "a quarter-opaque floater over an opaque card is never on top")
        rep = json.load(open(self.pj.path("prune", "export_01000_report.json")))
        for k in ("importance_log10", "blame", "views_seen"):
            self.assertIn(k, rep["histograms"])
        self.assertEqual(rep["would_flag"]["splats"], 1)


class Floaters(ScoreBase):
    def test_the_floater_goes_and_the_faint_top_contributor_stays(self):
        prune.run(args(score=True), self.pj)
        prune.run(args(floaters=True, name="export_01000"), self.pj)
        z = self.scores()
        flag, thr = prune.flag_floaters({k: z[k] for k in z.files}, z["live"], 0.02, MAX_BLAME)
        self.assertTrue(flag[self.FLOATER])
        self.assertFalse(flag[self.FAINT])
        # ...and only the guard kept it: low importance and high blame, as a floater would be
        self.assertLessEqual(z["importance"][self.FAINT], thr)
        self.assertGreater(z["blame"][self.FAINT], MAX_BLAME)
        self.assertTrue(z["top_contributor"][self.FAINT])
        self.assertEqual(int(flag.sum()), 1)

        out = self.pj.path("prune", "export_01000_nofloat.ply")
        only = self.pj.path("prune", "export_01000_floaters_only.ply")
        self.assertEqual(read_header(out)[1], self.n + 1)
        self.assertEqual(read_header(only)[1], 1)
        got = sw.Splats(only)
        self.assertAlmostEqual(float(got.arr["z"][0]), -0.015, places=5)
        m = self.pj.stage("prune")["metrics"]
        self.assertEqual(m["floaters"], 1)
        self.assertEqual(m["scores"], "reused prune/export_01000_scores.npz")
        self.assertGreater(m["opacity_mass_kept"], 0.999)
        self.assertLess(m["opacity_mass_kept"], 1.0)
        self.assertEqual(self.pj.stage("prune")["status"], "done")
        # both files survive: --floaters does not wipe prune/
        self.assertTrue(os.path.exists(self.pj.path("prune", "export_01000_scores.npz")))

    def test_render_takes_the_nofloat_model_as_prunes_output(self):
        prune.run(args(score=True, floaters=True), self.pj)
        out = self.pj.path("prune", "export_01000_nofloat.ply")
        self.pj.m["stages"]["train"]["metrics"]["final_export_md5"] = md5_file(self.ply)
        _rig, how, ok = render.model_lineage(self.pj, out, md5_file(out), explicit=True)
        self.assertTrue(ok, how)
        self.assertEqual(how, "pruned variant of train's export")

    def test_stale_scores_are_recomputed(self):
        prune.run(args(score=True), self.pj)
        z = dict(np.load(self.pj.path("prune", "export_01000_scores.npz")))
        z["ply_md5"] = np.array("not-this-model")
        np.savez(self.pj.path("prune", "export_01000_scores.npz"), **z)
        prune.run(args(floaters=True), self.pj)
        self.assertNotIn("scores", self.pj.stage("prune")["metrics"])
        self.assertEqual(self.pj.stage("prune")["metrics"]["floaters"], 1)

    def test_cli_flags(self):
        a = build_parser().parse_args(["prune", "-p", "/x", "--score", "--floaters", "--name", "m",
                                       "--min-importance-quantile", "0.05", "--max-blame", "0.3"])
        self.assertTrue(a.score and a.floaters)
        self.assertEqual((a.name, a.min_importance_quantile, a.max_blame, a.radius), ("m", 0.05, 0.3, 0.3))
        g = build_parser().parse_args(["prune", "-p", "/x", "--radius", "0.5"])
        self.assertFalse(g.score or g.floaters)


if __name__ == "__main__":
    unittest.main()
