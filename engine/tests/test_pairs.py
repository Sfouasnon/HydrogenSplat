"""The sequential pair list hs solve hands to COLMAP (hs/pairs.py) and the matcher choice.

No COLMAP here: the list is a pure function of the capture order, and these check what it
must contain (window, rig mates, loop pass), what it must not (duplicates, self pairs, far
pairs outside the loop pass) and its exact size, since the estimate quotes that number.
"""
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(HERE)
sys.path.insert(0, ENGINE)

from hs import pairs  # noqa: E402


def caps(n):
    return [f"cap{i:03d}" for i in range(n)]


def cap_index(name):
    return int(name.split("/")[1][3:6])


class SequentialPairs(unittest.TestCase):
    def test_window_count_no_loop(self):
        n, w = 10, 3
        got = pairs.sequential_pairs(caps(n), overlap=w, loop_stride=0)
        # n rig mates + 4 eye combinations per capture pair within the window
        want = n + 4 * sum(min(w, n - 1 - i) for i in range(n))
        self.assertEqual(len(got), want)
        self.assertEqual(len(got), 106)

    def test_rig_mates_always(self):
        got = set(pairs.sequential_pairs(caps(12), overlap=0, loop_stride=0))
        self.assertEqual(got, {(f"L/cap{i:03d}.jpg", f"R/cap{i:03d}.jpg") for i in range(12)})

    def test_every_eye_combination_in_window(self):
        got = set(pairs.sequential_pairs(caps(5), overlap=1, loop_stride=0))
        for a, b in (("L", "L"), ("L", "R"), ("R", "L"), ("R", "R")):
            x, y = f"{a}/cap001.jpg", f"{b}/cap002.jpg"
            self.assertIn(tuple(sorted((x, y))), got, (x, y))

    def test_unique_sorted_canonical_no_self(self):
        got = pairs.sequential_pairs(caps(40), overlap=15, loop_stride=8)
        self.assertEqual(got, sorted(set(got)))
        for a, b in got:
            self.assertLess(a, b)

    def test_far_pairs_only_between_stride_captures(self):
        n, w, s = 60, 4, 8
        got = pairs.sequential_pairs(caps(n), overlap=w, loop_stride=s)
        for a, b in got:
            i, j = cap_index(a), cap_index(b)
            if abs(i - j) > w:
                self.assertTrue(i % s == 0 and j % s == 0, (a, b))

    def test_loop_pass_pairs_all_stride_captures(self):
        n, s = 50, 8
        got = set(pairs.sequential_pairs(caps(n), overlap=2, loop_stride=s))
        stride = list(range(0, n, s))
        for i in stride:
            for j in stride:
                if i < j:
                    for ea in "LR":
                        for eb in "LR":
                            p = tuple(sorted((f"{ea}/cap{i:03d}.jpg", f"{eb}/cap{j:03d}.jpg")))
                            self.assertIn(p, got)
        without = set(pairs.sequential_pairs(caps(n), overlap=2, loop_stride=s, loop=False))
        self.assertEqual(without, set(pairs.sequential_pairs(caps(n), overlap=2, loop_stride=0)))
        self.assertLess(len(without), len(got))

    def test_count_with_loop(self):
        n, w, s = 100, 15, 8
        stride = list(range(0, n, s))
        window = n + 4 * sum(min(w, n - 1 - i) for i in range(n))
        extra = sum(4 for a in stride for b in stride if a < b and b - a > w)
        self.assertEqual(len(pairs.sequential_pairs(caps(n), overlap=w, loop_stride=s)), window + extra)

    def test_subset_of_exhaustive_and_equal_when_window_covers(self):
        n = 9
        allp = {tuple(sorted((f"{ea}/{c}.jpg", f"{eb}/{d}.jpg")))
                for i, c in enumerate(caps(n)) for j, d in enumerate(caps(n))
                for ea in "LR" for eb in "LR" if (i, ea) != (j, eb)}
        self.assertEqual(len(allp), pairs.exhaustive_pair_count(2 * n))
        self.assertTrue(set(pairs.sequential_pairs(caps(n), overlap=3)) <= allp)
        self.assertEqual(set(pairs.sequential_pairs(caps(n), overlap=n)), allp)

    def test_mono(self):
        got = pairs.sequential_pairs(["GA", "GB", "GC", "GD"], eyes=("L",), overlap=1, loop_stride=0)
        self.assertEqual(got, [("L/GA.jpg", "L/GB.jpg"), ("L/GB.jpg", "L/GC.jpg"), ("L/GC.jpg", "L/GD.jpg")])

    def test_counts_match_the_list(self):
        for n, e in ((1, 2), (7, 2), (61, 2), (150, 2), (61, 1), (200, 1)):
            self.assertEqual(pairs.pair_count("sequential", n, e),
                             len(pairs.sequential_pairs(caps(n), eyes=("L", "R")[:e])))
            self.assertEqual(pairs.pair_count("exhaustive", n, e), pairs.exhaustive_pair_count(n * e))

    def test_circles_sculpture_numbers(self):
        # 422 stereo captures: the 2026-09-22 match that ran 355,746 pairs
        self.assertEqual(pairs.pair_count("exhaustive", 422, 2), 355746)
        self.assertEqual(pairs.pair_count("sequential", 422, 2), 30566)
        self.assertEqual(pairs.pair_count("auto", 422, 2), 30566)

    def test_write_pair_list(self):
        with tempfile.TemporaryDirectory() as d:
            p = pairs.write_pair_list(pairs.sequential_pairs(caps(3), overlap=1, loop_stride=0),
                                      os.path.join(d, "sub", "pairs.txt"))
            lines = open(p).read().splitlines()
        self.assertIn("L/cap000.jpg R/cap000.jpg", lines)
        self.assertTrue(all(len(l.split(" ")) == 2 for l in lines))


class Matcher(unittest.TestCase):
    def test_auto_threshold(self):
        self.assertEqual(pairs.resolve_matcher("auto", 60), "exhaustive")
        self.assertEqual(pairs.resolve_matcher("auto", 61), "sequential")
        self.assertEqual(pairs.resolve_matcher("auto", 12), "exhaustive")
        self.assertEqual(pairs.resolve_matcher("exhaustive", 400), "exhaustive")
        self.assertEqual(pairs.resolve_matcher("sequential", 5), "sequential")
        with self.assertRaises(ValueError):
            pairs.resolve_matcher("vocab", 10)

    def test_scripts_keep_exhaustive_default(self):
        for script in ("rigcolmap.py", "monocolmap.py"):
            r = subprocess.run([sys.executable, os.path.join(ENGINE, "hs", script), "sfm", "--help"],
                               capture_output=True, text=True)
            if r.returncode != 0 and "pycolmap" in (r.stdout + r.stderr):
                self.skipTest("pycolmap not installed")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("--matcher {exhaustive,sequential,auto}", r.stdout)
            self.assertIn("exhaustive (the default", r.stdout)
            bad = subprocess.run([sys.executable, os.path.join(ENGINE, "hs", script), "sfm", "w", "--matcher", "bogus"],
                                 capture_output=True, text=True)
            self.assertEqual(bad.returncode, 2)


class ExhaustiveBlocks(unittest.TestCase):
    """pycolmap 4.2.0 logs 'Processing block [i/ni, j/nj]' for every one of the ni×nj blocks
    (checked on synthetic images: 14 images, block size 4 → all 16 lines, [2/4, 1/4] included),
    so the matching total is ni·nj, not the ni·(ni+1)/2 of i ≤ j. This replays COLMAP's
    ExhaustivePairGenerator dedup rule to show why: every block holds ~half its pairs and
    together they are each pair exactly once."""

    @staticmethod
    def colmap_blocks(n, bs):
        out = []
        for s1 in range(0, n, bs):
            for s2 in range(0, n, bs):
                blk = []
                for i1 in range(s1, min(n, s1 + bs)):
                    for i2 in range(s2, min(n, s2 + bs)):
                        b1, b2 = i1 % bs, i2 % bs
                        if (i1 > i2 and b1 <= b2) or (i1 < i2 and b1 < b2):
                            blk.append((min(i1, i2), max(i1, i2)))
                out.append(blk)
        return out

    def test_all_blocks_run_and_partition_pairs(self):
        for n, bs in ((14, 4), (844, 50), (100, 50)):
            blocks = self.colmap_blocks(n, bs)
            nb = -(-n // bs)
            self.assertEqual(len(blocks), nb * nb)
            self.assertEqual(pairs.exhaustive_blocks(n, bs), nb * nb)
            flat = [p for b in blocks for p in b]
            self.assertEqual(len(flat), len(set(flat)))
            self.assertEqual(len(flat), n * (n - 1) // 2)
            lower = [b for k, b in enumerate(blocks) if k // nb > k % nb]
            self.assertTrue(all(lower), "blocks below the diagonal hold pairs too")

    def test_progress_runs_to_ni_nj(self):
        ni = nj = 17
        seen = [pairs.exhaustive_block_progress(i, ni, j, nj) for i in range(1, ni + 1) for j in range(1, nj + 1)]
        self.assertEqual([d for d, _ in seen], list(range(ni * nj)))
        self.assertTrue(all(t == 289 for _, t in seen))
        self.assertEqual(pairs.exhaustive_block_progress(2, 17, 1, 17), (17, 289))


if __name__ == "__main__":
    unittest.main()
