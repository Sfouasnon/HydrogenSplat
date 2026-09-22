"""The solve cost model, the timing store, `hs solve --estimate`, and the solve pieces around
them that can run without COLMAP: argument parsing, the matcher argv, the log parser, and
the vocabulary-tree fetch (against a file:// URL).

HS_TIMING_FILE points every test at a temp store; nothing here touches the real one.
"""
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(HERE)
sys.path.insert(0, ENGINE)

from hs import cli, events, pairs, timing  # noqa: E402
from hs.stages import solve, tools  # noqa: E402

PHASE_KEYS = {"features", "matching", "mapping", "export", "total"}


class TempStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = os.path.join(self.tmp.name, "hs", "timing.json")
        self.env = mock.patch.dict(os.environ, {"HS_TIMING_FILE": self.store})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()


class Store(TempStore):
    def test_path_override_and_platform_default(self):
        self.assertEqual(timing.store_path(), self.store)
        with mock.patch.dict(os.environ, {}, clear=False):
            del os.environ["HS_TIMING_FILE"]
            with mock.patch.object(timing.sys, "platform", "darwin"):
                self.assertTrue(timing.store_path().endswith("Library/Application Support/HydrogenSplat/timing.json"))
            with mock.patch.object(timing.sys, "platform", "linux"):
                self.assertTrue(timing.store_path().endswith(".hydrogensplat/timing.json"))

    def test_missing_and_corrupt_are_empty(self):
        self.assertEqual(timing.load()["solve"], [])
        os.makedirs(os.path.dirname(self.store))
        open(self.store, "w").write("{not json")
        self.assertEqual(timing.load()["solve"], [])
        open(self.store, "w").write("[1, 2]")
        self.assertEqual(timing.load()["solve"], [])

    def test_append_round_trip(self):
        timing.append_solve_run({"images": 100, "pairs": 4950, "features_s": 60.0})
        timing.append_solve_run({"images": 200, "pairs": 19900, "matching_s": 1990.0})
        runs = timing.load()["solve"]
        self.assertEqual([r["images"] for r in runs], [100, 200])
        self.assertIn("at", runs[0])
        self.assertEqual(timing.calibration_label(timing.model()), "measured on this Mac (2 runs)")

    def test_unwritable_store_is_not_fatal(self):
        os.makedirs(self.store)            # a directory where the file should be
        self.assertIsNone(timing.append_solve_run({"images": 1}))


class Model(unittest.TestCase):
    def test_defaults(self):
        m = timing.fit([])
        self.assertEqual(m["runs"], 0)
        self.assertEqual(m["measured"], [])
        self.assertEqual(timing.calibration_label(m), "defaults")
        self.assertEqual(m["matching"], 0.10)
        self.assertEqual(m["mapping_p"], 1.5)

    def test_one_run_sets_measured_phases_only(self):
        m = timing.fit([{"images": 200, "pairs": 10000, "features_s": 100.0, "matching_s": 500.0}])
        self.assertAlmostEqual(m["features"], 0.5)
        self.assertAlmostEqual(m["matching"], 0.05)
        self.assertEqual(m["export"], timing.DEFAULTS["export"])
        self.assertEqual(m["measured"], ["features", "matching"])
        self.assertEqual(timing.calibration_label(m), "measured on this Mac (1 run)")

    def test_least_squares_through_origin(self):
        runs = [{"images": 100, "export_s": 20.0}, {"images": 300, "export_s": 66.0}]
        # k = (100*20 + 300*66) / (100^2 + 300^2) = 21800 / 100000
        self.assertAlmostEqual(timing.fit(runs)["export"], 0.218)

    def test_mapping_exponent_fitted_from_three_runs(self):
        runs = [{"images": n, "mapping_s": 0.2 * n ** 1.8} for n in (100, 300, 844)]
        m = timing.fit(runs)
        self.assertAlmostEqual(m["mapping_p"], 1.8, places=6)
        self.assertAlmostEqual(m["mapping_c"], 0.2, places=6)
        # two runs: exponent stays at the default, c is fitted for it
        m2 = timing.fit(runs[:2])
        self.assertEqual(m2["mapping_p"], 1.5)
        self.assertGreater(m2["mapping_c"], 0)
        # clamped when the data says something implausible
        m3 = timing.fit([{"images": n, "mapping_s": 1e-3 * n ** 4} for n in (100, 200, 400)])
        self.assertEqual(m3["mapping_p"], timing.EXPONENT_RANGE[1])

    def test_bad_rows_ignored(self):
        m = timing.fit([{"images": 0, "features_s": 5}, {"images": "x"}, {"pairs": 10, "matching_s": -1}, {}])
        self.assertEqual(m["runs"], 0)
        self.assertEqual(m, {**timing.fit([]), "runs": 0})


class Estimate(unittest.TestCase):
    def test_shape(self):
        e = timing.estimate(150, 2)
        self.assertEqual(set(e), {"captures", "images_per_capture", "auto", "calibration", "runs", "matchers"})
        self.assertEqual(set(e["matchers"]), {"exhaustive", "sequential"})
        for name, m in e["matchers"].items():
            self.assertEqual(m["images"], 300)
            self.assertEqual(set(m["seconds"]), PHASE_KEYS)
            self.assertTrue(all(isinstance(v, int) for v in m["seconds"].values()))
            parts = sum(m["seconds"][k] for k in ("features", "matching", "mapping", "export"))
            self.assertLessEqual(abs(parts - m["seconds"]["total"]), 2)
        self.assertEqual(e["matchers"]["sequential"]["overlap"], 15)
        self.assertEqual(e["matchers"]["sequential"]["loop_stride"], 8)
        self.assertEqual(e["auto"], "sequential")
        self.assertEqual(timing.estimate(40, 2)["auto"], "exhaustive")
        json.dumps(e)

    def test_default_numbers(self):
        e = timing.estimate(422, 2)
        ex, sq = e["matchers"]["exhaustive"], e["matchers"]["sequential"]
        self.assertEqual((ex["pairs"], sq["pairs"]), (355746, 30566))
        self.assertEqual(ex["seconds"]["matching"], 35575)      # 0.10 s/pair
        self.assertEqual(ex["seconds"]["mapping"], 12260)       # 0.5 * 844^1.5
        self.assertEqual(e["calibration"], "defaults")

    def test_monotonic(self):
        prev = None
        for n in (10, 40, 61, 100, 150, 300, 422):
            e = timing.estimate(n, 2)
            if prev:
                for k in ("exhaustive", "sequential"):
                    self.assertGreater(e["matchers"][k]["pairs"], prev["matchers"][k]["pairs"])
                    self.assertGreater(e["matchers"][k]["seconds"]["total"], prev["matchers"][k]["seconds"]["total"])
            self.assertLessEqual(e["matchers"]["sequential"]["pairs"], e["matchers"]["exhaustive"]["pairs"])
            prev = e
        self.assertLess(timing.estimate(150, 1)["matchers"]["exhaustive"]["seconds"]["total"],
                        timing.estimate(150, 2)["matchers"]["exhaustive"]["seconds"]["total"])

    def test_measured_model_changes_the_numbers(self):
        m = timing.fit([{"images": 844, "pairs": 355746, "matching_s": 355746 * 0.05}])
        e = timing.estimate(422, 2, m)
        self.assertEqual(e["matchers"]["exhaustive"]["seconds"]["matching"], round(355746 * 0.05))
        self.assertEqual(e["calibration"], "measured on this Mac (1 run)")
        self.assertEqual(e["runs"], 1)

    def test_mapping_eta(self):
        self.assertIsNone(timing.mapping_eta(0, 0, 0))
        self.assertEqual(timing.mapping_eta(50, 100, 100, 200), 0.0)
        self.assertEqual(timing.mapping_eta(30, 2, 100, 200), 170)          # early: the model
        # a quarter in: the run's own curve, elapsed * ((n / k)^p - 1)
        self.assertAlmostEqual(timing.mapping_eta(100, 25, 100, 200), 100 * (4 ** 1.5 - 1))
        # past the model with too few registered: extrapolate rather than claim 0
        self.assertAlmostEqual(timing.mapping_eta(300, 5, 100, 200, 1.0), 300 * 19)
        self.assertIsNone(timing.mapping_eta(10, 0, 100, None))

    def test_fmt_duration(self):
        self.assertEqual(timing.fmt_duration(45), "45 s")
        self.assertEqual(timing.fmt_duration(35 * 60), "35 min")
        self.assertEqual(timing.fmt_duration(4 * 3600 + 40 * 60), "4 h 40 min")


def run_cli(*argv):
    out = io.StringIO()
    with redirect_stdout(out):
        code = cli.main(list(argv))
    return code, [json.loads(l) for l in out.getvalue().splitlines() if l.strip()]


class SolveEstimateCli(TempStore):
    def make_project(self, kind=None, n=7, ext=".jpg"):
        root = os.path.join(self.tmp.name, f"p_{kind}")
        os.makedirs(os.path.join(root, "select", "frames"))
        src = {"kind": kind} if kind else {}
        json.dump({"version": 1, "source": src, "stages": {"solve": {"status": "running", "pid": 2 ** 22 + 12345}}},
                  open(os.path.join(root, "manifest.json"), "w"))
        for i in range(n):
            open(os.path.join(root, "select", "frames", f"f{i:04d}{ext}"), "w").close()
        open(os.path.join(root, "select", "frames", "selection.json"), "w").close()
        return root

    def test_captures_without_project(self):
        code, evs = run_cli("solve", "--estimate", "--captures", "150")
        self.assertEqual(code, 0)
        self.assertEqual(len(evs), 1, "one estimate event, nothing else")
        self.assertEqual(evs[0]["ev"], "estimate")
        self.assertEqual(evs[0]["stage"], "solve")
        self.assertEqual(evs[0]["matchers"]["exhaustive"]["images"], 300)
        self.assertFalse(os.path.exists(self.store), "--estimate writes nothing")

    def test_project_read_only(self):
        root = self.make_project(n=7)
        man = os.path.join(root, "manifest.json")
        before = (open(man).read(), os.path.getmtime(man))
        code, evs = run_cli("solve", "-p", root, "--estimate", "--overlap", "2", "--loop-stride", "3")
        self.assertEqual(code, 0)
        self.assertEqual([e["ev"] for e in evs], ["estimate"])
        e = evs[0]
        self.assertEqual((e["captures"], e["images_per_capture"]), (7, 2))
        self.assertEqual(e["matchers"]["sequential"]["pairs"], 63)   # as matched by COLMAP on 7 captures
        self.assertEqual((open(man).read(), os.path.getmtime(man)), before, "a running stage is not reconciled")
        self.assertEqual(sorted(os.listdir(root)), ["manifest.json", "select"], "no logs/, no lock")

    def test_mono_project_one_image_per_capture(self):
        root = self.make_project(kind="mono", n=90, ext=".png")
        code, evs = run_cli("solve", "-p", root, "--estimate")
        self.assertEqual(code, 0)
        self.assertEqual((evs[0]["captures"], evs[0]["images_per_capture"]), (90, 1))
        self.assertEqual(evs[0]["auto"], "sequential")
        code, evs = run_cli("solve", "-p", root, "--estimate", "--captures", "300")
        self.assertEqual((evs[0]["captures"], evs[0]["images_per_capture"]), (300, 1))

    def test_errors(self):
        code, evs = run_cli("solve", "--estimate")
        self.assertEqual(code, 1)
        self.assertEqual([e["ev"] for e in evs], ["error", "done"])
        root = self.make_project(n=0)
        code, evs = run_cli("solve", "-p", root, "--estimate")
        self.assertEqual(code, 1)
        self.assertIn("--captures", evs[0]["hint"])

    def test_calibration_from_store(self):
        timing.append_solve_run({"images": 300, "pairs": 9282, "matching_s": 464.1})
        code, evs = run_cli("solve", "--estimate", "--captures", "150")
        self.assertEqual(evs[0]["calibration"], "measured on this Mac (1 run)")
        self.assertEqual(evs[0]["matchers"]["sequential"]["seconds"]["matching"], 464)

    def test_subprocess_entry_point(self):
        r = subprocess.run([sys.executable, "-m", "hs", "solve", "--estimate", "--captures", "422"],
                           capture_output=True, text=True, cwd=ENGINE,
                           env={**os.environ, "PYTHONPATH": ENGINE})
        self.assertEqual(r.returncode, 0, r.stderr)
        lines = r.stdout.strip().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["matchers"]["exhaustive"]["pairs"], 355746)


class SolveArgs(unittest.TestCase):
    def parse(self, *argv):
        return cli.build_parser().parse_args(["solve", "-p", "P", *argv])

    def test_defaults(self):
        a = self.parse()
        self.assertEqual((a.matcher, a.overlap, a.loop_stride, a.loop), ("auto", 15, 8, "stride"))
        self.assertFalse(a.estimate)
        self.assertIsNone(a.captures)

    def test_choices(self):
        for m in ("exhaustive", "sequential", "auto"):
            self.assertEqual(self.parse("--matcher", m).matcher, m)
        with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()), mock.patch("sys.stderr", io.StringIO()):
            self.parse("--matcher", "vocab")

    def test_matcher_argv(self):
        a = self.parse()
        self.assertEqual(solve._matcher_argv(a, 12, None), ("exhaustive", ["--matcher", "exhaustive"]))
        m, argv = solve._matcher_argv(a, 422, None)
        self.assertEqual(m, "sequential")
        self.assertEqual(argv, ["--matcher", "sequential", "--overlap", 15, "--loop-stride", 8, "--loop", "stride"])
        a = self.parse("--matcher", "exhaustive")
        self.assertEqual(solve._matcher_argv(a, 422, None)[0], "exhaustive")

    def test_vocab_needs_a_tree(self):
        with tempfile.TemporaryDirectory() as d:
            missing = os.path.join(d, "none.bin")
            a = self.parse("--matcher", "sequential", "--loop", "vocab", "--vocab-tree", missing)
            with self.assertRaises(events.StageError):
                solve._matcher_argv(a, 100, None)
            open(missing, "wb").write(b"x")
            _, argv = solve._matcher_argv(a, 100, None)
            self.assertEqual(argv[-4:], ["--vocab-tree", missing, "--vocab-neighbors", 20])


class SfmLogParser(unittest.TestCase):
    """The lines rigcolmap.py / monocolmap.py and pycolmap print, as hs solve reads them."""

    def setUp(self):
        self.got = []
        events.set_sink(self.got.append)
        events._last_progress.clear()
        events.reset_eta()
        self.out = redirect_stdout(io.StringIO())
        self.out.__enter__()

    def tearDown(self):
        self.out.__exit__(None, None, None)
        events.set_sink(None)

    def prog(self, step):
        return [e for e in self.got if e["ev"] == "progress" and e.get("step") == step]

    def test_sequential_run(self):
        p = solve.SfmParser(14, stereo=True)
        lines = [
            "I20260922 13:29:20.1 1 feature_extraction.cc:1] Processed file [1/14]",
            "  features per image: min 2000, median 2500, max 3000",
            "timing: features 2.6 s",
            "sequential matching 14 images (63 pairs; overlap 2, loop stride 3)...",
            "I20260922 13:29:24.826298 140003017488064 pairing.cc:934] Processing block [1/1]",
            "timing: matching 3.3 s",
            "mapping (rig fixed, intrinsics fixed)...",
            "I20260922 13:29:30.1 1 incremental_pipeline.cc:1] Registering image #3 (3)",
            "timing: mapping 2.5 s",
            "  eye L: 100 observations, rms 0.500 px, median 0.400 px",
        ]
        for l in lines:
            p(l)
        self.assertEqual((p.matcher, p.num_pairs), ("sequential", 63))
        self.assertEqual(p.timings, {"features": 2.6, "matching": 3.3, "mapping": 2.5})
        self.assertEqual(self.prog("matching")[0]["total"], 1)
        self.assertIn("63 pairs", self.prog("matching")[0]["detail"])
        mapping = self.prog("mapping")
        self.assertEqual(mapping[0]["done"], 0)
        self.assertIn("eta_s", mapping[0], "mapping ETA comes from the cost model")
        self.assertEqual(p.metrics["featstat"], ("2000", "2500", "3000"))
        self.assertIn("eye_L", p.metrics)

    def test_exhaustive_blocks(self):
        p = solve.SfmParser(844, stereo=False)
        p("exhaustive matching 844 images (355746 pairs)...")
        p("I0 0 pairing.cc:212] Processing block [2/17, 1/17]")
        ev = self.prog("matching")[-1]
        self.assertEqual((ev["done"], ev["total"]), (17, 289))
        self.assertEqual(p.num_pairs, 355746)
        p("  eye L: 100 observations, rms 0.500 px, median 0.400 px")
        self.assertNotIn("eye_L", p.metrics, "mono ignores the stereo report lines")


class VocabTree(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.got = []
        events.set_sink(self.got.append)
        self.out = redirect_stdout(io.StringIO())
        self.out.__enter__()

    def tearDown(self):
        self.out.__exit__(None, None, None)
        events.set_sink(None)
        self.tmp.cleanup()

    def test_fetch_checks_sha_and_status_line(self):
        src = os.path.join(self.tmp.name, "tree.bin")
        open(src, "wb").write(b"vocab" * 1000)
        sha = hashlib.sha256(b"vocab" * 1000).hexdigest()
        dest = os.path.join(self.tmp.name, "cache", "t.bin")
        url = "file://" + src
        with self.assertRaises(events.StageError):
            tools.fetch_vocab_tree(url, dest=dest, sha256="0" * 64)
        self.assertFalse(os.path.exists(dest))
        self.assertFalse(os.path.exists(dest + ".part"))
        self.assertEqual(tools.fetch_vocab_tree(url, dest=dest, sha256=sha), dest)
        self.assertEqual(open(dest, "rb").read(), b"vocab" * 1000)
        with mock.patch.dict(os.environ, {"HS_VOCAB_TREE": dest}):
            self.assertEqual(tools.vocab_tree_path(), dest)
            info = {}
            tools.vocab_tree_status(info)
            self.assertEqual(info["vocab_tree"], dest)
        with mock.patch.dict(os.environ, {"HS_VOCAB_TREE": dest + ".missing"}):
            info = {}
            tools.vocab_tree_status(info)
            self.assertIsNone(info["vocab_tree"])
        line = [e for e in self.got if e.get("name") == "vocab_tree"][-1]
        self.assertTrue(line["ok"], "optional: never a failing check")
        self.assertIn("--fetch-vocab-tree", line["value"])

    def test_default_url_is_colmaps_faiss_tree(self):
        self.assertTrue(tools.VOCAB_TREE_URL.startswith("https://github.com/colmap/colmap/releases/download/"))
        self.assertTrue(tools.VOCAB_TREE_URL.endswith("vocab_tree_faiss_flickr100K_words256K.bin"))
        self.assertEqual(len(tools.VOCAB_TREE_SHA256), 64)
        with mock.patch.object(tools.sys, "platform", "darwin"), mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HS_VOCAB_TREE", None)
            self.assertIn("Library/Caches/HydrogenSplat", tools.vocab_tree_path())


if __name__ == "__main__":
    unittest.main()
