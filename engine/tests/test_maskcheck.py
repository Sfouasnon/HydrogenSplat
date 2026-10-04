"""The masks are checked against each other before anything is trained on them.

2026-09-28_Stormtrooper_iPhone: 36 of 243 masks left part of the helmet out, the trainer painted
those parts black, and they were found in four rounds, three of them after a 2 h 40 min train.
`hs/maskcheck.py` finds such masks from the masks and the poses alone; `hs/maskreview.py` keeps
the decisions and is what `hs train` asks before it starts.

One synthetic scene: an ellipsoid, 40 cameras on two rings, every mask the exact silhouette. Then
one mask at a time is spoiled the way the helmet's were.
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from argparse import Namespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from hs import events, maskcheck, maskreview  # noqa: E402
from hs.project import Project, md5_file, now_iso  # noqa: E402
from hs.stages import masks as masks_stage  # noqa: E402
from hs.stages import train  # noqa: E402

import test_lifecycle as LC  # noqa: E402
import test_masks as TM  # noqa: E402

W, H, FX = 640, 480, 800.0
AXES = np.array([40.0, 55.0, 30.0])                 # mm
N = 40
BITE, HOLE, EDGE, FRAGMENT = 5, 11, 8, 20           # the views that get spoiled


def scene():
    """-> (views, exact masks, sparse points). View EDGE looks past the subject: the frame cuts it."""
    import cv2
    rng = np.random.default_rng(3)
    v = rng.normal(size=(30000, 3))
    surf = v / np.linalg.norm(v, axis=1, keepdims=True) * AXES
    views, masks = [], []
    for i in range(N):
        a = 2 * np.pi * i / N
        C = np.array([300 * np.cos(a), (-60.0, -150.0)[i % 2], 300 * np.sin(a)])
        R = TM.look_at(C)
        t = -R @ C
        cx = W / 2 + (230.0 if i == EDGE else 0.0)
        K = np.array([[FX, 0, cx], [0, FX, H / 2], [0, 0, 1.0]])
        Xc = surf @ R.T + t
        uv = Xc[:, :2] / Xc[:, 2:3] * FX + [cx, H / 2]
        m = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(m, cv2.convexHull(uv.astype(np.float32)).astype(np.int32), 255)
        views.append((K, R, t, (W, H)))
        masks.append(m)
    return views, masks, surf[::20]


def spoil(masks, which):
    """Copies of the masks with the named views spoiled. -> list"""
    import cv2
    m = [x.copy() for x in masks]
    if BITE in which:                                # a bite out of the silhouette
        ys, xs = np.nonzero(m[BITE])
        cv2.circle(m[BITE], (int(xs.max()), int(ys.mean())), 30, 0, -1)
    if HOLE in which:                                # a hole inside the subject
        cv2.circle(m[HOLE], (W // 2, H // 2), 18, 0, -1)
    if EDGE in which:                                # a bite where the frame cuts the subject
        cv2.circle(m[EDGE], (W - 1, H // 2), 30, 0, -1)
    if FRAGMENT in which:                            # what the rough region looked like on the helmet
        m[FRAGMENT][:] = 0
        cv2.circle(m[FRAGMENT], (W // 2 + 60, H // 2 - 70), 14, 255, -1)
    return m


def iou(a, b):
    a, b = a > 127, b > 127
    return float((a & b).sum()) / float((a | b).sum())


class Check(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.views, cls.masks, cls.pts = scene()

    def flagged(self, rep):
        return {r["index"]: r for r in rep["views"] if r["reasons"]}

    def repair(self, rep, masks, i, **kw):
        S, _core = rep["work"][i]
        return maskcheck.repair(masks[i], S, subject_uv=maskcheck.subject_uv(rep["subject"], rep["prepared"][i]), **kw)

    def test_exact_masks_flag_nothing(self):
        rep = maskcheck.check(self.views, self.masks, self.pts)
        self.assertEqual(self.flagged(rep), {})
        self.assertIsNone(rep["note"])
        self.assertEqual(rep["method"]["voters"], N)
        self.assertTrue(all(r["agreement"] > 0.97 for r in rep["views"]))

    def test_a_bite_is_found_in_that_view_only_and_the_repair_is_approximate(self):
        m = spoil(self.masks, {BITE})
        rep = maskcheck.check(self.views, m, self.pts)
        fl = self.flagged(rep)
        self.assertEqual(list(fl), [BITE], "one bad mask must not put the others in doubt")
        self.assertEqual(fl[BITE]["reasons"], ["piece_outside"])
        self.assertGreater(fl[BITE]["piece_share"], 0.02)
        self.assertIn("leaves out", maskcheck.why(fl[BITE]))
        out, kind, approx, whole = self.repair(rep, m, BITE)
        self.assertEqual((kind, approx, whole), ("hull", True, False))
        self.assertLess(iou(m[BITE], self.masks[BITE]), 0.97)
        self.assertGreater(iou(out, self.masks[BITE]), 0.99)

    def test_a_hole_is_filled_and_that_repair_is_exact(self):
        m = spoil(self.masks, {HOLE})
        rep = maskcheck.check(self.views, m, self.pts)
        self.assertEqual(list(self.flagged(rep)), [HOLE])
        out, kind, approx, whole = self.repair(rep, m, HOLE)
        self.assertEqual((kind, approx, whole), ("fill", False, False))
        self.assertGreater(iou(out, self.masks[HOLE]), 0.999)

    def test_a_piece_closed_by_the_frame_edge_is_filled_but_wants_a_look(self):
        # no outline is invented, but that the piece is subject is the hull's word alone
        self.assertTrue(self.masks[EDGE][:, -1].any(), "the scene must cut this view at the frame edge")
        m = spoil(self.masks, {EDGE})
        rep = maskcheck.check(self.views, m, self.pts)
        self.assertEqual(list(self.flagged(rep)), [EDGE])
        out, kind, approx, _whole = self.repair(rep, m, EDGE)
        self.assertEqual((kind, approx), ("fill", True))
        self.assertGreater(iou(out, self.masks[EDGE]), 0.99)

    def test_a_fragment_is_flagged_and_has_no_vote(self):
        m = spoil(self.masks, {FRAGMENT})
        rep = maskcheck.check(self.views, m, self.pts)
        fl = self.flagged(rep)
        self.assertEqual(list(fl), [FRAGMENT])
        self.assertEqual(fl[FRAGMENT]["reasons"], ["holds_subject"])
        self.assertLess(fl[FRAGMENT]["agreement"], 0.2)
        self.assertEqual(rep["method"]["voters"], N - 1)
        self.assertIn("holds only", maskcheck.why(fl[FRAGMENT]))

    def test_a_fall_back_is_flagged_whatever_it_looks_like(self):
        rep = maskcheck.check(self.views, self.masks, self.pts, fell_back={3: "no foreground found"})
        fl = self.flagged(rep)
        self.assertEqual(list(fl), [3])
        self.assertEqual(fl[3]["reasons"], ["fell_back"])
        self.assertEqual(rep["method"]["voters"], N - 1, "the rough region is not a vote on the subject")

    def test_a_fall_back_without_an_object_gets_the_whole_hull_marked_as_such(self):
        m = spoil(self.masks, {FRAGMENT})
        rep = maskcheck.check(self.views, m, self.pts, fell_back={FRAGMENT: "no foreground found"})
        self.assertEqual(self.flagged(rep)[FRAGMENT]["reasons"], ["fell_back", "holds_subject"])
        out, kind, approx, whole = self.repair(rep, m, FRAGMENT, fell_back=True)
        self.assertEqual((kind, approx, whole), ("hull", True, True))
        self.assertGreater(iou(out, self.masks[FRAGMENT]), 0.93)

    def test_visions_object_is_taken_when_the_hull_confirms_it(self):
        import cv2
        m = spoil(self.masks, {FRAGMENT})
        rep = maskcheck.check(self.views, m, self.pts, fell_back={FRAGMENT: "nothing inside the geometric mask"})
        obj = cv2.GaussianBlur(self.masks[FRAGMENT], (0, 0), 3)          # Vision's masks are soft
        clutter = np.zeros((H, W), np.uint8)
        clutter[0:60, 0:80] = 255
        out, kind, approx, whole = self.repair(rep, m, FRAGMENT, fell_back=True, instances=[obj, clutter])
        self.assertEqual((kind, approx, whole), ("reselect", False, False))
        self.assertGreater(iou(out, self.masks[FRAGMENT]), 0.99)
        self.assertFalse(out[0:60, 0:80].any(), "the box beside the subject was taken along")

    def test_an_object_that_is_not_the_subject_is_not_taken(self):
        m = spoil(self.masks, {FRAGMENT})
        rep = maskcheck.check(self.views, m, self.pts, fell_back={FRAGMENT: "nothing inside the geometric mask"})
        clutter = np.zeros((H, W), np.uint8)
        clutter[0:200, 0:260] = 255
        _out, kind, approx, whole = self.repair(rep, m, FRAGMENT, fell_back=True, instances=[clutter])
        self.assertEqual((kind, approx, whole), ("hull", True, True))

    def test_without_points_on_the_subject_it_says_so_and_judges_nothing(self):
        m = spoil(self.masks, {BITE})
        rep = maskcheck.check(self.views, m, np.zeros((0, 3)), fell_back={3: "no foreground found"})
        self.assertIn("too few", rep["note"])
        self.assertEqual(list(self.flagged(rep)), [3], "a fall-back is still a fall-back")

    def test_too_few_masks_to_vote(self):
        keep = list(range(6))
        rep = maskcheck.check([self.views[i] for i in keep], [self.masks[i] for i in keep], self.pts)
        self.assertIn("too few to vote", rep["note"])
        self.assertEqual(self.flagged(rep), {})


# --------------------------------------------------------------------------- the project side
class Quiet(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hs-maskcheck-")
        self.root = os.path.join(self.tmp, "proj")
        self.out = io.StringIO()
        self._redir = contextlib.redirect_stdout(self.out)
        self._redir.__enter__()

    def tearDown(self):
        self._redir.__exit__(None, None, None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def events(self):
        return [json.loads(x) for x in self.out.getvalue().splitlines() if x.startswith("{")]


SCENE = None


def project(root, spoiled, method="geometry"):
    """A solved project whose masks are the scene's, with `spoiled` views spoiled."""
    import cv2
    global SCENE
    if SCENE is None:
        SCENE = scene()
    views, exact, pts = SCENE
    pj = Project(root, create=True)
    for s in ("ingest", "select", "solve"):
        pj.m["stages"][s] = {"status": "done", "finished": now_iso(), "checks": [], "metrics": {}}
    pj.m["stages"]["masks"] = {"status": "done", "started": now_iso(), "finished": now_iso(), "checks": [],
                               "metrics": {"method": method}, "artifacts": [], "argv": []}
    pj.m["stages"]["train"] = {"status": "done", "metrics": {}, "checks": [], "artifacts": []}
    for sub in ("images", "masks"):
        os.makedirs(os.path.join(pj.dataset_dir, sub, "L"))
    ms = spoil(exact, spoiled)
    for i in range(N):
        cv2.imwrite(os.path.join(pj.dataset_dir, "images", "L", f"cap{i:03d}.jpg"), np.full((H, W, 3), 128, np.uint8))
        cv2.imwrite(os.path.join(pj.dataset_dir, "masks", "L", f"cap{i:03d}.png"), ms[i])
    np.savez(pj.rig_npz, names=np.array([f"cap{i:03d}_L" for i in range(N)]),
             K=np.array([v[0] for v in views]), R=np.array([v[1] for v in views]),
             t=np.array([v[2] for v in views]), wh=np.array([[W, H]] * N), pts=pts, stereo=False)
    pj.save()
    return pj


def key(i):
    return f"L/cap{i:03d}"


def mask_path(pj, i):
    return os.path.join(pj.dataset_dir, "masks", "L", f"cap{i:03d}.png")


def read(pj, i):
    import cv2
    return cv2.imread(mask_path(pj, i), cv2.IMREAD_GRAYSCALE)


def stage(pj, **kw):
    """`hs masks --check-only` / `--decide` as the CLI runs them."""
    return masks_stage.run(Namespace(check_only=kw.get("check_only", False), decide=kw.get("decide")), pj)


class Review(Quiet):
    def test_check_only_writes_the_report_and_marks_nothing_stale(self):
        pj = project(self.root, {BITE, HOLE})
        stage(pj, check_only=True)
        rv = json.load(open(pj.path("masks_review", "review.json")))
        self.assertEqual(rv["version"], 1)
        self.assertEqual(rv["views"], N)
        self.assertEqual([f["view"] for f in rv["flagged"]], [key(BITE), key(HOLE)], "worst first")
        self.assertEqual(rv["summary"], {"flagged": 2, "undecided": 2, "repairable": 2, "exact": 1,
                                         "repair": 0, "exclude": 0, "keep": 0})
        self.assertEqual(rv["masks_digest"], maskreview.digest(pj))
        for f in rv["flagged"]:
            self.assertEqual(f["mask_md5"], md5_file(pj.path("train", "dataset", "masks", f["view"] + ".png")))
            self.assertIsNone(f["decision"])
            for p in (f["preview"], f["repair"]["preview"], f["repair"]["mask"]):
                self.assertTrue(os.path.exists(pj.path(p)), p)
        by = {f["view"]: f for f in rv["flagged"]}
        self.assertEqual((by[key(HOLE)]["repair"]["kind"], by[key(HOLE)]["repair"]["approximate"]), ("fill", False))
        self.assertEqual((by[key(BITE)]["repair"]["kind"], by[key(BITE)]["repair"]["approximate"]), ("hull", True))
        pj = Project(self.root)
        st = pj.stage("masks")
        self.assertEqual(st["status"], "done")
        self.assertEqual(pj.status("train"), "done", "a check changes no mask: nothing downstream is stale")
        self.assertEqual(st["metrics"]["mask_review"]["undecided"], 2)
        self.assertEqual(st["metrics"]["mask_review"]["report"], "masks_review/review.json")
        c = {x["name"]: x for x in st["checks"]}["masks_reviewed"]
        self.assertFalse(c["ok"])
        self.assertTrue(c["needs_human"])
        self.assertIsNone(pj.lock_holder(), "the lock was left behind")

    def test_exact_masks_pass_and_say_so(self):
        pj = project(self.root, set())
        rv = stage(pj, check_only=True)
        self.assertEqual(rv["flagged"], [])
        c = {x["name"]: x for x in Project(self.root).stage("masks")["checks"]}["masks_reviewed"]
        self.assertTrue(c["ok"])
        self.assertFalse(c.get("needs_human"))
        self.assertIn(f"all {N} masks agree", c["value"])

    def test_decisions_are_applied_one_at_a_time_in_the_order_given(self):
        pj = project(self.root, {BITE, HOLE, EDGE})
        stage(pj, check_only=True)
        before = {i: md5_file(mask_path(pj, i)) for i in (BITE, HOLE, EDGE)}
        # the exact repair goes in; what is then still undecided is excluded — not the one just repaired
        rv = stage(pj, decide=["@exact=repair,@undecided=exclude"])
        by = {f["view"]: f for f in rv["flagged"]}
        self.assertEqual({k: f["decision"] for k, f in by.items()},
                         {key(HOLE): "repair", key(BITE): "exclude", key(EDGE): "exclude"})
        self.assertEqual(rv["summary"]["undecided"], 0)
        self.assertNotEqual(md5_file(mask_path(pj, HOLE)), before[HOLE])
        self.assertEqual(md5_file(mask_path(pj, BITE)), before[BITE], "exclude changes no file")
        self.assertGreater(iou(read(pj, HOLE), SCENE[1][HOLE]), 0.999)
        self.assertEqual(md5_file(pj.path("masks_review", "original", key(HOLE) + ".png")), before[HOLE])
        self.assertEqual(by[key(HOLE)]["mask_md5"], md5_file(mask_path(pj, HOLE)))
        self.assertEqual(rv["masks_digest"], maskreview.digest(pj))
        pj = Project(self.root)
        self.assertEqual(pj.status("train"), "stale", "a mask changed: the model trained on the old one is stale")
        self.assertTrue({x["name"]: x for x in pj.stage("masks")["checks"]}["masks_reviewed"]["ok"])
        left_out, info = maskreview.gate(pj, set())
        self.assertEqual(left_out, {key(BITE), key(EDGE)})
        self.assertEqual((info["repair"], info["exclude"], info["undecided"]), (1, 2, 0))

    def test_undo_puts_the_mask_back_as_it_was_built(self):
        pj = project(self.root, {HOLE})
        stage(pj, check_only=True)
        before = md5_file(mask_path(pj, HOLE))
        stage(pj, decide=[f"{key(HOLE)}=repair"])
        self.assertNotEqual(md5_file(mask_path(pj, HOLE)), before)
        rv = stage(pj, decide=[f"{key(HOLE)}=undo"])
        self.assertEqual(md5_file(mask_path(pj, HOLE)), before)
        f = rv["flagged"][0]
        self.assertEqual((f["decision"], f["mask_md5"]), (None, before))
        self.assertFalse(os.path.exists(pj.path("masks_review", "original")), "no empty folders left behind")
        self.assertEqual(rv["masks_digest"], maskreview.digest(pj))
        stage(pj, decide=[f"{key(HOLE)}=exclude"])
        rv = stage(pj, decide=["@decided=undo"])              # start over
        self.assertEqual(rv["summary"]["undecided"], 1)
        # and a repaired view can be changed to something else: the original comes back first
        stage(pj, decide=[f"{key(HOLE)}=repair"])
        rv = stage(pj, decide=[f"{key(HOLE)}=keep"])
        self.assertEqual(md5_file(mask_path(pj, HOLE)), before)
        self.assertEqual(rv["flagged"][0]["decision"], "keep")

    def test_the_report_is_json_the_app_can_read(self):
        pj = project(self.root, {BITE})
        rv = stage(pj, check_only=True)
        rv["flagged"][0]["agreement"] = float("nan")
        rv["method"]["voxel_mm"] = np.float32(1.5)
        maskreview.save(pj, rv)
        text = open(pj.path("masks_review", "review.json")).read()
        self.assertNotIn("NaN", text)
        back = json.loads(text, parse_constant=lambda c: self.fail(f"{c} in review.json"))
        self.assertIsNone(back["flagged"][0]["agreement"])
        self.assertEqual(back["method"]["voxel_mm"], 1.5)

    def test_what_decide_refuses(self):
        pj = project(self.root, {BITE})
        with self.assertRaisesRegex(events.StageError, "no mask review"):
            stage(pj, decide=[f"{key(BITE)}=keep"])
        stage(pj, check_only=True)
        for text, msg in ((f"{key(2)}=keep", "not a flagged view"), ("@all=keep", "unknown list"),
                          (f"{key(BITE)}=mend", "not one of"), (key(BITE), "not VIEW=CHOICE"), ("", "at least one")):
            with self.assertRaisesRegex(events.StageError, msg):
                stage(pj, decide=[text])
        # refused as a whole: the first half of a command is not applied when the second half is wrong
        before = md5_file(mask_path(pj, BITE))
        with self.assertRaisesRegex(events.StageError, "not a flagged view"):
            stage(pj, decide=[f"{key(BITE)}=repair,{key(2)}=keep"])
        self.assertEqual(md5_file(mask_path(pj, BITE)), before)
        self.assertIsNone(maskreview.load(pj)["flagged"][0]["decision"])
        rv = json.load(open(pj.path("masks_review", "review.json")))
        rv["flagged"][0]["repair"] = None
        json.dump(rv, open(pj.path("masks_review", "review.json"), "w"))
        with self.assertRaisesRegex(events.StageError, "no repair for"):
            stage(pj, decide=[f"{key(BITE)}=repair"])
        rv = stage(pj, decide=["@undecided=repair"])          # a list: what cannot be repaired stays undecided
        self.assertIsNone(rv["flagged"][0]["decision"])
        self.assertIsNone(Project(self.root).lock_holder())

    def test_a_decide_that_stops_half_way_leaves_what_it_did_on_record(self):
        pj = project(self.root, {BITE, HOLE})
        stage(pj, check_only=True)
        os.remove(pj.path("masks_review", key(BITE) + "_repaired.png"))
        with self.assertRaisesRegex(events.StageError, "repaired mask of L/cap005 is missing"):
            stage(pj, decide=[f"{key(HOLE)}=repair,{key(BITE)}=repair"])
        rv = maskreview.load(pj)
        by = {f["view"]: f for f in rv["flagged"]}
        self.assertEqual((by[key(HOLE)]["decision"], by[key(BITE)]["decision"]), ("repair", None))
        self.assertEqual(by[key(HOLE)]["mask_md5"], md5_file(mask_path(pj, HOLE)))
        self.assertEqual(rv["masks_digest"], maskreview.digest(pj))
        pj = Project(self.root)
        self.assertEqual(pj.status("train"), "stale")
        self.assertIsNone(pj.lock_holder())
        was = md5_file(pj.path("masks_review", "original", key(HOLE) + ".png"))
        rv = stage(pj, decide=[f"{key(HOLE)}=undo"])                       # and it can be taken back
        self.assertEqual(md5_file(mask_path(pj, HOLE)), was)
        self.assertIsNone({f["view"]: f for f in rv["flagged"]}[key(HOLE)]["decision"])

    def test_a_decision_belongs_to_the_mask_file_not_to_the_view_name(self):
        import cv2
        pj = project(self.root, {BITE, HOLE})
        stage(pj, check_only=True)
        stage(pj, decide=[f"{key(BITE)}=exclude,{key(HOLE)}=keep"])
        rv = stage(pj, check_only=True)                       # the same files: the decisions stand
        self.assertEqual({f["view"]: f["decision"] for f in rv["flagged"]},
                         {key(BITE): "exclude", key(HOLE): "keep"})
        m = read(pj, HOLE)
        cv2.circle(m, (W // 2, H // 2), 22, 0, -1)            # rebuilt: still a hole, another file
        cv2.imwrite(mask_path(pj, HOLE), m)
        rv = stage(pj, check_only=True)
        self.assertEqual({f["view"]: f["decision"] for f in rv["flagged"]},
                         {key(BITE): "exclude", key(HOLE): None})

    def test_a_repair_that_is_installed_stays_on_record_through_a_new_check(self):
        pj = project(self.root, {HOLE, BITE})
        stage(pj, check_only=True)
        stage(pj, decide=[f"{key(HOLE)}=repair"])
        rv = stage(pj, check_only=True)
        by = {f["view"]: f for f in rv["flagged"]}
        self.assertEqual(by[key(HOLE)]["decision"], "repair")
        self.assertEqual(rv["flagged"][-1]["view"], key(HOLE), "decided long ago: listed last")
        self.assertEqual(rv["summary"]["undecided"], 1)
        rv = stage(pj, decide=[f"{key(HOLE)}=undo"])          # and it can still be undone
        self.assertIsNone({f["view"]: f for f in rv["flagged"]}[key(HOLE)]["decision"])

    def test_the_repaired_mask_is_finished_the_way_the_build_finished_its_own(self):
        pj = project(self.root, {HOLE}, method="vision")
        pj.m["stages"]["masks"]["argv"] = ["hs", "masks", "-p", self.root, "--method", "vision", "--feather-px", "2"]
        pj.save()
        self.assertEqual(maskreview.build_settings(pj)["feather_px"], 2.0)
        stage(pj, check_only=True)
        stage(pj, decide=[f"{key(HOLE)}=repair"])
        m = read(pj, HOLE)
        self.assertTrue(((m > 0) & (m < 255)).any(), "a Vision build feathers its masks; so must the repair")
        self.assertGreater(iou(m, SCENE[1][HOLE]), 0.999)


class VisionCache(Quiet):
    """Vision's own objects are still on disk after a build (masks_vision/<i>/<k>.png, images.txt
    in the order of the folders). A view that fell back is given Vision's object when the hull
    confirms it — but only when those files belong to the build the masks came from."""

    def cache(self, pj, age_s=0.0):
        import cv2
        root = pj.path("masks_vision")
        os.makedirs(root)
        lines = []
        for i in range(N):
            os.makedirs(os.path.join(root, str(i)))
            cv2.imwrite(os.path.join(root, str(i), "1.png"), cv2.GaussianBlur(SCENE[1][i], (0, 0), 3))
            lines.append(os.path.join(pj.dataset_dir, "images", "L", f"cap{i:03d}.jpg"))
        lst = os.path.join(root, "images.txt")
        open(lst, "w").write("\n".join(lines) + "\n")
        t = time.time() - age_s
        os.utime(lst, (t, t))

    def test_the_object_vision_found_replaces_the_rough_region(self):
        pj = project(self.root, {FRAGMENT}, method="vision")
        pj.m["stages"]["masks"]["metrics"]["vision_fell_back_views"] = [f"cap{FRAGMENT:03d}_L: nothing inside the geometric mask"]
        pj.save()
        self.cache(pj)
        rv = stage(pj, check_only=True)
        f = rv["flagged"][0]
        self.assertEqual(f["reasons"], ["fell_back", "holds_subject"], "the build's record names the fall-backs")
        self.assertEqual(f["fell_back"], "nothing inside the geometric mask")
        self.assertEqual((f["repair"]["kind"], f["repair"]["approximate"]), ("reselect", False))
        self.assertEqual(rv["summary"]["exact"], 1)
        stage(pj, decide=["@exact=repair"])
        self.assertGreater(iou(read(pj, FRAGMENT), SCENE[1][FRAGMENT]), 0.99)
        rv = stage(pj, check_only=True)                       # the repaired mask is a mask like any other
        self.assertEqual(rv["method"]["voters"], N, "an installed repair votes; it is no longer the rough region")
        self.assertEqual([(f["view"], f["decision"]) for f in rv["flagged"]], [(key(FRAGMENT), "repair")])

    def test_objects_from_an_older_build_are_not_used(self):
        pj = project(self.root, {FRAGMENT}, method="vision")
        self.cache(pj, age_s=3600.0)
        rv = stage(pj, check_only=True)
        self.assertEqual(rv["flagged"][0]["repair"]["kind"], "hull")
        self.assertTrue(rv["flagged"][0]["repair"]["approximate"])


class BuildEndsWithTheCheck(Quiet):
    def test_a_build_writes_the_review(self):
        pj, ply = TM.build(self.root, 1)
        masks_stage.run(TM.args(ply), pj)
        rv = maskreview.load(pj)
        self.assertEqual((rv["views"], rv["flagged"]), (TM.N_CAM, []))
        st = Project(self.root).stage("masks")
        self.assertEqual(st["status"], "done")
        self.assertTrue({x["name"]: x for x in st["checks"]}["masks_reviewed"]["ok"])
        self.assertEqual(maskreview.gate(pj, set())[0], set())

    def test_no_check_leaves_it_to_train_to_ask(self):
        pj, ply = TM.build(self.root, 1)
        masks_stage.run(TM.args(ply, no_check=True), pj)
        self.assertIsNone(maskreview.load(pj))
        with self.assertRaisesRegex(events.StageError, "not been checked"):
            maskreview.gate(pj, set())

    def test_a_rebuild_does_not_inherit_the_old_review(self):
        pj, ply = TM.build(self.root, 1)
        masks_stage.run(TM.args(ply), pj)
        os.makedirs(pj.path("masks_review", "original", "L"), exist_ok=True)
        open(pj.path("masks_review", "original", "L", "cap000.png"), "wb").close()
        masks_stage.run(TM.args(ply, no_check=True), pj)
        self.assertIsNone(maskreview.load(pj), "a review of masks that no longer exist")
        self.assertFalse(os.path.exists(pj.path("masks_review", "original")))


class TrainGate(LC.Base):
    """hs train with masks in use does not start on masks nobody has looked at."""
    TOTAL = 40

    def setUp(self):
        super().setUp()
        os.environ.update(HS_PYTHON=sys.executable, HS_FAKE_REFINE_STOP="0", HS_FAKE_QUIET_TAIL="0")

    def args(self, **kw):
        kw.setdefault("allow_unreviewed_masks", False)
        return LC.TrainResume.train_args(self, **kw)

    def reviewed(self, flagged=()):
        pj = LC.TrainView.masked_project(self)
        rv = {"version": 1, "created": now_iso(), "views": 4, "masks_digest": maskreview.digest(pj), "method": {},
              "note": None,
              "flagged": [{"view": v, "reasons": ["piece_outside"], "score": 0.1, "why": "", "repair": None,
                           "mask_md5": "", "decision": d, "decided": None} for v, d in flagged]}
        maskreview.save(pj, rv)
        return pj

    def test_masks_never_checked(self):
        pj = LC.TrainView.masked_project(self)
        with self.assertRaisesRegex(events.StageError, "not been checked") as e:
            train.run(self.args(), pj)
        self.assertIn("--check-only", e.exception.hint)
        self.assertIn("--allow-unreviewed-masks", e.exception.hint)
        train.run(self.args(allow_unreviewed_masks=True), pj)
        self.assertEqual(pj.stage("train")["metrics"]["mask_review"],
                         {"reviewed": False, "allowed_unreviewed": True, "excluded_by_review": []})

    def test_without_masks_nobody_asks(self):
        pj = LC.TrainView.masked_project(self)
        train.run(self.args(no_masks=True), pj)
        self.assertEqual(pj.status("train"), "done")
        self.assertNotIn("mask_review", pj.stage("train")["metrics"])

    def test_an_undecided_view_stops_the_run_unless_it_is_left_out_anyway(self):
        pj = self.reviewed([("L/cap001", None), ("R/cap000", "keep")])
        with self.assertRaisesRegex(events.StageError, r"1 flagged mask\(s\) have no decision: L/cap001") as e:
            train.run(self.args(), pj)
        self.assertIn("--decide", e.exception.hint)
        train.run(self.args(exclude="L/cap001"), pj)
        self.assertEqual(pj.status("train"), "done")

    def test_views_decided_exclude_are_left_out_of_the_run(self):
        pj = self.reviewed([("L/cap001", "exclude"), ("R/cap000", "keep")])
        train.run(self.args(), pj)
        view = pj.path("train", "view")
        self.assertFalse(os.path.exists(os.path.join(view, "images", "L", "cap001.jpg")))
        self.assertFalse(os.path.exists(os.path.join(view, "masks", "L", "cap001.png")))
        self.assertTrue(os.path.exists(os.path.join(view, "masks", "R", "cap000.png")), "keep trains on it")
        tm = pj.stage("train")["metrics"]
        self.assertEqual(tm["excluded_views"], ["L/cap001"])
        self.assertEqual(tm["mask_review"]["excluded_by_review"], ["L/cap001"])
        self.assertEqual((tm["mask_review"]["flagged"], tm["mask_review"]["undecided"]), (2, 0))

    def test_masks_changed_after_the_review(self):
        pj = self.reviewed()
        train.run(self.args(), pj)                                   # reviewed, nothing flagged: runs
        open(os.path.join(pj.dataset_dir, "masks", "L", "cap000.png"), "wb").write(b"x")
        with self.assertRaisesRegex(events.StageError, "not the ones that were reviewed"):
            train.run(self.args(), pj)
        train.run(self.args(allow_unreviewed_masks=True), pj)
        self.assertTrue(pj.stage("train")["metrics"]["mask_review"]["masks_changed"])


if __name__ == "__main__":
    unittest.main()
