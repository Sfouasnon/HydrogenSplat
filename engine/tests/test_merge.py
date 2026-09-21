"""hs merge: a subject model and a background model become one archive.

The merge is a concatenation, which is only sound because both layers were trained against the
same solve. These tests hold that line: the frames must agree, the PLY property lists must be
identical (no silent SH padding), and the merged file must be byte-for-byte the two bodies under
one rewritten header — because a splat that loses or gains a property is a splat that renders
wrong rather than failing.
"""
import json
import os
import sys
import tempfile
import unittest
from argparse import Namespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(HERE)
sys.path.insert(0, ENGINE)

from hs import events  # noqa: E402
from hs.project import Project, md5_file, now_iso  # noqa: E402
from hs.stages import merge  # noqa: E402

BASE = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
TAIL = ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]


def props(sh_rest=45):
    return BASE + [f"f_rest_{i}" for i in range(sh_rest)] + TAIL


def write_ply(path, n, seed=0, sh_rest=45):
    p = props(sh_rest)
    rng = np.random.default_rng(seed)
    arr = rng.normal(0, 1, (n, len(p))).astype("<f4")
    hdr = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n
           + "".join(f"property float {x}\n" for x in p) + "end_header\n")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(hdr.encode())
        f.write(arr.tobytes())
    return arr


class MergeBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        pj = Project(self.root, create=True)
        for s in ("ingest", "select", "solve"):
            pj.m["stages"][s] = {"status": "done"}
        pj.m["stages"]["train"] = {"status": "done", "metrics": {}, "finished": now_iso()}
        os.makedirs(os.path.join(pj.dataset_dir), exist_ok=True)
        np.savez(pj.rig_npz, names=np.array(["cap000_L"]))
        pj.save()
        self.pj = pj

    def archive(self, name, n, seed=0, sh_rest=45, rig_md5="same-frame", layer=None):
        """An archive as `hs archive` leaves it: a ply, a rig, and a manifest that names the frame."""
        d = self.pj.path("archive", name)
        arr = write_ply(os.path.join(d, "export_40000.ply"), n, seed=seed, sh_rest=sh_rest)
        np.savez(os.path.join(d, "rig.npz"), names=np.array(["cap000_L"]))
        man = {"name": name, "rig_npz_md5": rig_md5,
               "ply": {"md5": md5_file(os.path.join(d, "export_40000.ply"))},
               "train_dataset_fingerprint": {"layer": layer} if layer else None}
        json.dump(man, open(os.path.join(d, "manifest.json"), "w"))
        return d, arr

    def args(self, **kw):
        base = dict(models="", name="merged", force=False)
        base.update(kw)
        return Namespace(**base)

    def body(self, ply, sh_rest=45):
        with open(ply, "rb") as f:
            raw = f.read()
        k = raw.find(b"end_header\n") + len(b"end_header\n")
        return np.frombuffer(raw[k:], "<f4").reshape(-1, len(props(sh_rest)))


class Merge(MergeBase):
    def test_two_layers_become_one_archive(self):
        _, a = self.archive("subject-a", 40, seed=1, layer="subject")
        _, b = self.archive("background-a", 60, seed=2, layer="background")
        merge.run(self.args(models="subject-a,background-a", name="both"), self.pj)

        out = self.pj.path("archive", "both", "merged.ply")
        self.assertTrue(os.path.exists(out))
        got = self.body(out)
        self.assertEqual(got.shape[0], 100, "the merged model is not the sum of its parts")
        # the splats must survive untouched, in order: subject first, then background
        np.testing.assert_array_equal(got[:40], a)
        np.testing.assert_array_equal(got[40:], b)

    def test_it_is_an_archive_the_rest_of_hs_already_understands(self):
        self.archive("subject-a", 10, seed=1, layer="subject")
        self.archive("background-a", 10, seed=2, layer="background")
        merge.run(self.args(models="subject-a,background-a", name="both"), self.pj)

        d = self.pj.path("archive", "both")
        # hs render's lineage guard reads this manifest; the viewer needs the rig beside the ply
        man = json.load(open(os.path.join(d, "manifest.json")))
        self.assertTrue(os.path.exists(os.path.join(d, "rig.npz")))
        self.assertIsNotNone(man["rig_npz_md5"])
        self.assertEqual(man["ply"]["md5"], md5_file(os.path.join(d, "merged.ply")))
        self.assertEqual([m["layer"] for m in man["merged_from"]], ["subject", "background"])
        self.assertEqual([m["splats"] for m in man["merged_from"]], [10, 10])

        checks = {c["name"]: c for c in self.pj.stage("merge")["checks"]}
        self.assertTrue(checks["sources_share_a_frame"]["ok"])
        self.assertTrue(checks["properties_match"]["ok"])
        self.assertTrue(checks["splat_count_is_the_sum"]["ok"])
        self.assertTrue(checks["layers_are_complementary"]["ok"])
        self.assertEqual(self.pj.stage("merge")["metrics"]["splats"], 20)
        self.assertEqual(self.pj.stage("merge")["status"], "done")

    def test_a_ply_path_works_as_well_as_an_archive_name(self):
        d, _ = self.archive("subject-a", 10, seed=1, layer="subject")
        self.archive("background-a", 10, seed=2, layer="background")
        rel = os.path.relpath(os.path.join(d, "export_40000.ply"), self.pj.root)
        merge.run(self.args(models=f"{rel},background-a", name="both"), self.pj)
        self.assertEqual(self.pj.stage("merge")["metrics"]["splats"], 20)

    # ---- the refusals
    def test_different_solves_are_refused(self):
        self.archive("subject-a", 10, seed=1, rig_md5="frame-one", layer="subject")
        self.archive("background-a", 10, seed=2, rig_md5="frame-two", layer="background")
        with self.assertRaises(events.StageError) as e:
            merge.run(self.args(models="subject-a,background-a", name="both"), self.pj)
        self.assertIn("world frames", e.exception.hint)
        self.assertFalse(os.path.exists(self.pj.path("archive", "both")), "a refused merge left a folder")

    def test_different_solves_can_be_forced(self):
        self.archive("subject-a", 10, seed=1, rig_md5="frame-one")
        self.archive("background-a", 10, seed=2, rig_md5="frame-two")
        merge.run(self.args(models="subject-a,background-a", name="both", force=True), self.pj)
        checks = {c["name"]: c for c in self.pj.stage("merge")["checks"]}
        self.assertFalse(checks["sources_share_a_frame"]["ok"], "forcing must still record the disagreement")

    def test_mismatched_sh_degree_is_refused_not_padded(self):
        self.archive("subject-a", 10, seed=1, sh_rest=45, layer="subject")
        self.archive("background-a", 10, seed=2, sh_rest=9, layer="background")
        with self.assertRaises(events.StageError) as e:
            merge.run(self.args(models="subject-a,background-a", name="both"), self.pj)
        self.assertIn("different PLY properties", str(e.exception))
        self.assertIn("SH degree", e.exception.hint)
        self.assertFalse(os.path.exists(self.pj.path("archive", "both")))

    def test_one_model_is_not_a_merge(self):
        self.archive("subject-a", 10)
        with self.assertRaises(events.StageError):
            merge.run(self.args(models="subject-a", name="both"), self.pj)

    def test_it_will_not_quietly_replace_an_archive(self):
        self.archive("subject-a", 10, seed=1)
        self.archive("background-a", 10, seed=2)
        self.archive("both", 5, seed=3)
        with self.assertRaises(events.StageError) as e:
            merge.run(self.args(models="subject-a,background-a", name="both"), self.pj)
        self.assertIn("already exists", str(e.exception))
        self.assertEqual(self.body(self.pj.path("archive", "both", "export_40000.ply")).shape[0], 5)

    def test_two_subjects_merge_but_are_flagged(self):
        self.archive("subject-a", 10, seed=1, layer="subject")
        self.archive("subject-b", 10, seed=2, layer="subject")
        merge.run(self.args(models="subject-a,subject-b", name="both"), self.pj)
        checks = {c["name"]: c for c in self.pj.stage("merge")["checks"]}
        self.assertFalse(checks["layers_are_complementary"]["ok"])
        self.assertTrue(checks["layers_are_complementary"].get("needs_human"))

    def test_a_truncated_source_fails_instead_of_writing_a_short_model(self):
        d, _ = self.archive("subject-a", 40, seed=1)
        self.archive("background-a", 10, seed=2)
        p = os.path.join(d, "export_40000.ply")
        with open(p, "rb") as f:
            raw = f.read()
        with open(p, "wb") as f:
            f.write(raw[:-400])                      # lose the last few splats
        with self.assertRaises(events.StageError) as e:
            merge.run(self.args(models="subject-a,background-a", name="both"), self.pj)
        self.assertIn("truncated", e.exception.hint or str(e.exception))
        self.assertFalse(os.path.exists(self.pj.path("archive", "both")))


if __name__ == "__main__":
    unittest.main()
