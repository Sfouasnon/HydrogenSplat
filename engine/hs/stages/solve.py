"""hs solve — rigcolmap.py prep → sfm --float-rig → export, plus per_image.json and
coverage.json (strategy §4.4). The calibrated intrinsics from the profile go into the COLMAP
rig config through the same npz path the by-hand run used; --refine-intrinsics is not
exposed (it collapses the reconstruction without --float-rig and buys ~0.03 px with it).

Checks: registered frames == selected frames; mean reprojection ≤ 1.8 px; no image > 3 px;
L–R separation equals the profile baseline. A partial registration fails the stage with the
remedies in order (re-select with --max-gap 45; trim the tail with --end) — never silently
proceed to train on a partial solve.

Matching (hs/pairs.py): ``--matcher auto`` (the default here; the scripts' own default stays
``exhaustive``) runs sequential above 60 captures — a window of ``--overlap`` captures, every
rig mate and a ``--loop-stride`` loop pass — and exhaustive below. The one that ran is the
``matcher`` metric, with ``num_pairs`` and the phase times ``features_s``, ``matching_s``,
``mapping_s``, ``export_s``; a completed solve appends them to the timing store (timing.py),
which is what ``hs solve --estimate`` and the mapping ETA are fitted from.

``hs solve --estimate [-p P] [--captures N] [--eyes 1|2]`` takes no lock and writes nothing:
one ``estimate`` event on stdout (shape in timing.py) with both matchers' pairs and seconds.
"""
import json
import os
import re
import shutil
import sys
import time

import numpy as np

from .. import calib, coverage, events, pairs, runner, timing

STAGE = "solve"

RE_FEAT = re.compile(r"Processed file \[(\d+)/(\d+)\]")
RE_MATCH = re.compile(r"Processing block \[(\d+)/(\d+), (\d+)/(\d+)\]")
RE_REG = re.compile(r"Registering image #(\d+) \((\d+)\)")
RE_FEATSTAT = re.compile(r"features per image: min (\d+), median (\d+), max (\d+)")
RE_RECON = re.compile(r"reconstruction: (\d+) images in (\d+) posed frames, (\d+) points")
RE_REPROJ = re.compile(r"reprojection error: mean ([\d.]+) px, median ([\d.]+) px")
RE_EYE = re.compile(r"eye (\w): (\d+) observations, rms ([\d.]+) px, median ([\d.]+) px")
RE_RIG = re.compile(r"rig sensor_from_rig: \|t\| = ([\d.]+) mm \(calibrated ([\d.]+) mm\)")
RE_BA = re.compile(r"BA-refined sensor_from_rig before rescale: \|t\| ([\d.]+), rotation vs calibrated ([\d.]+) deg")
RE_STEP = re.compile(r"step between consecutive captures: (\d+)-(\d+) mm \(median (\d+)\), path length (\d+) mm")
RE_DEPTH = re.compile(r"scene depth from path centre: p5 (\d+) mm, median (\d+) mm, p95 (\d+) mm")
RE_UNDIST = re.compile(r"Undistorting image \[(\d+)/(\d+)\]")
RE_IMPORT = re.compile(r"Processing block \[(\d+)/(\d+)\]")          # match_image_pairs (sequential)
RE_MATCHING = re.compile(r"(exhaustive|sequential) matching (\d+) images \((\d+) pairs")
RE_TIMING = re.compile(r"^timing: (features|matching|mapping) ([\d.]+) s")
RE_MAPSTART = re.compile(r"^mapping \(")
FRAME_EXTS_STEREO = (".jpg", ".jpeg", ".png")                         # rigcolmap.py prep
FRAME_EXTS_MONO = (".png", ".jpg", ".jpeg", ".tif", ".tiff")           # monocolmap.py IMAGE_EXTS


def add_parser(sub):
    p = sub.add_parser("solve", help="split eyes, rig-aware COLMAP SfM, export the training set (rigcolmap.py)")
    p.add_argument("--profile", default=None, help="override the profile chosen at ingest")
    p.add_argument("--peak-threshold", type=float, default=0.0025)
    p.add_argument("--features", type=int, default=16384)
    p.add_argument("--masks", default=None, help="mask dir mirroring images/ (wired, unverified)")
    p.add_argument("--baseline-mm", type=float, default=None,
                   help="override the profile's L-R separation (direction kept); the calibration's "
                        "own best joint fit for this rig is 11.71 mm against the 10.595 it ships")
    p.add_argument("--no-float-rig", action="store_true", help="NOT recommended: keep sensor_from_rig fixed")
    p.add_argument("--reuse-matches", action="store_true", help="keep an existing database.db (skip features/matching)")
    p.add_argument("--allow-partial", action="store_true",
                   help="export the registered captures when some did not register (the check fails and needs a "
                        "human; the unregistered captures are listed; the coverage will have a hole)")
    p.add_argument("--export-only", action="store_true",
                   help="skip prep and sfm: export solve/sparse/rig from an earlier run (with --allow-partial "
                        "when that run stopped at a partial registration)")
    p.add_argument("--max-reproj", type=float, default=1.8, help="check threshold, px")
    m = p.add_argument_group("matching (hs/pairs.py)")
    m.add_argument("--matcher", choices=pairs.MATCHER_CHOICES, default="auto",
                   help=f"auto (default): sequential above {pairs.AUTO_SEQUENTIAL_ABOVE} captures, else exhaustive. "
                        "sequential suits orbits and walk-arounds; exhaustive is for captures that revisit a "
                        "view from far apart in time where no loop pass would pair the visits")
    m.add_argument("--overlap", type=int, default=pairs.DEFAULT_OVERLAP,
                   help="sequential: captures matched on each side (both eyes)")
    m.add_argument("--loop-stride", type=int, default=pairs.DEFAULT_LOOP_STRIDE,
                   help="sequential: every Nth capture matched against every other Nth (0 = no loop pass)")
    m.add_argument("--loop", choices=("stride", "vocab"), default="stride",
                   help="sequential loop pass: stride (needs nothing) or vocab (a vocabulary tree: "
                        "hs tools --fetch-vocab-tree)")
    m.add_argument("--vocab-tree", default=None, help="vocabulary tree file (default HS_VOCAB_TREE, else the cache)")
    m.add_argument("--vocab-neighbors", type=int, default=20, help="--loop vocab: images retrieved per image")
    e = p.add_argument_group("estimate (no lock, no writes)")
    e.add_argument("--estimate", action="store_true",
                   help="print one `estimate` event: images, pairs and seconds per phase for both matchers")
    e.add_argument("--captures", type=int, default=None,
                   help="--estimate for this many captures instead of the project's frames (no project needed)")
    e.add_argument("--eyes", type=int, choices=(1, 2), default=None,
                   help="--estimate: images per capture (default: 2 on a Hydrogen project or without one, "
                        "1 on an array/mono project)")
    g = p.add_argument_group("array source (monocolmap.py; ignored on a Hydrogen clip)")
    g.add_argument("--scale-pair", default=None, metavar="CAM1,CAM2,MM",
                   help="metric scale from the measured distance between two camera centres, e.g. GA,GB,700")
    g.add_argument("--scale", type=float, default=None, help="multiply the reconstruction by this to get metres")
    g.add_argument("--focal-px", type=float, default=None,
                   help="focal length prior in pixels = lens mm / sensor width mm x image width")
    g.add_argument("--fix-intrinsics", action="store_true", help="keep --focal-px; do not refine focal/distortion")
    g.add_argument("--board", default=None, metavar="SX,SY,SQUARE_MM,MARKER_MM[,DICT]",
                   help="after the solve, run `hs scale --board` with this ChArUco board (on a Hydrogen "
                        "clip: `hs scale --dry-run`, which measures the baseline instead)")
    g.add_argument("--legacy-board", action="store_true", help="with --board: pre-4.6 OpenCV board layout")
    return p


def count_captures(root):
    """(captures, images per capture) for a project, read-only: select/frames as prep will see
    it. Reads manifest.json directly (Project() would reconcile and could write)."""
    root = os.path.abspath(os.path.expanduser(root))
    try:
        m = json.load(open(os.path.join(root, "manifest.json")))
    except (OSError, ValueError):
        raise events.StageError(f"no readable manifest.json in {root}", hint="--captures N estimates without a project")
    kind = (m.get("source") or {}).get("kind") or "stereo"
    mono = kind in ("array", "mono")
    frames = os.path.join(root, "select", "frames")
    exts = FRAME_EXTS_MONO if mono else FRAME_EXTS_STEREO
    n = len([f for f in os.listdir(frames) if f.lower().endswith(exts)]) if os.path.isdir(frames) else 0
    if n == 0:
        raise events.StageError(f"no frames in {frames}",
                                hint="run hs select first, or pass --captures N (e.g. the frame count the Frames page plans)")
    return n, 1 if mono else 2


def estimate(a):
    """hs solve --estimate: one `estimate` event on stdout (timing.py). No lock, no writes, no
    events file — cli.py calls this before it opens the project."""
    eyes = a.eyes
    if a.captures is not None:
        n = int(a.captures)
        if n < 1:
            raise events.StageError("--captures must be at least 1")
        if eyes is None and a.project:
            try:
                eyes = count_captures(a.project)[1]
            except events.StageError:
                eyes = None
    else:
        if not a.project:
            raise events.StageError("hs solve --estimate needs --project DIR or --captures N")
        n, eyes_p = count_captures(a.project)
        eyes = eyes or eyes_p
    est = timing.estimate(n, eyes or 2, timing.model(), overlap=a.overlap, loop_stride=a.loop_stride)
    events._emit({"ev": "estimate", "stage": STAGE, **est})
    return est


def _matcher_argv(a, n_caps, pj):
    """--matcher etc. for the script, resolved here so the metric says what actually ran."""
    matcher = pairs.resolve_matcher(a.matcher, n_caps)
    argv = ["--matcher", matcher]
    if matcher == "sequential":
        argv += ["--overlap", a.overlap, "--loop-stride", a.loop_stride, "--loop", a.loop]
        if a.loop == "vocab":
            from .tools import vocab_tree_path
            tree = a.vocab_tree or vocab_tree_path()
            if not os.path.isfile(tree):
                raise events.StageError(f"--loop vocab: no vocabulary tree at {tree}",
                                        hint="hs tools --fetch-vocab-tree, or --loop stride (the default)")
            argv += ["--vocab-tree", tree, "--vocab-neighbors", a.vocab_neighbors]
    return matcher, argv


class SfmParser:
    """on_line for rigcolmap.py / monocolmap.py sfm: progress for features, matching (both
    matchers' block lines) and mapping (ETA from the cost model — incremental mapping is
    superlinear, so a linear ETA would promise too early), the phase times, and the report
    lines (``metrics``, keyed as before)."""

    def __init__(self, n_img, stereo, model=None):
        self.n_img, self.stereo = n_img, stereo
        self.model = model or timing.fit([])
        self.step = "features"
        self.reg = 0
        self.metrics = {}
        self.timings = {}
        self.matcher = None
        self.num_pairs = None
        self.t_map0 = None

    PHASES = ("features", "matching", "mapping", "export")

    def rest_after(self, step):
        """Seconds the phases after `step` are expected to take (cost model; the pair count once
        the matcher has said it, else the exhaustive count as the upper bound) -> stage_eta_s."""
        try:
            n_pairs = self.num_pairs if self.num_pairs else self.n_img * (self.n_img - 1) // 2
            sec = timing.phase_seconds(self.model, self.n_img, n_pairs)
            later = self.PHASES[self.PHASES.index(step) + 1:]
            return float(sum(sec[ph] for ph in later))
        except (ValueError, KeyError, TypeError):
            return None

    def mapping_eta(self, now=None):
        if self.t_map0 is None:
            return None
        el = (time.monotonic() if now is None else now) - self.t_map0
        m = self.model
        predicted = m["mapping_c"] * self.n_img ** m["mapping_p"]
        return timing.mapping_eta(el, self.reg, self.n_img, predicted, m["mapping_p"])

    def __call__(self, line):
        m = RE_FEAT.search(line)
        if m:
            self.step = "features"
            events.progress(STAGE, int(m.group(1)), int(m.group(2)), step="features",
                            rest_s=self.rest_after("features"))
            return
        m = RE_MATCH.search(line)
        if m:
            self.step = "matching"
            i, ni, j, nj = (int(x) for x in m.groups())
            done, total = pairs.exhaustive_block_progress(i, ni, j, nj)
            events.progress(STAGE, done, total, step="matching", detail=f"block {i}/{ni},{j}/{nj}",
                            rest_s=self.rest_after("matching"))
            return
        m = RE_IMPORT.search(line)
        if m:
            self.step = "matching"
            k, nk = int(m.group(1)), int(m.group(2))
            events.progress(STAGE, k - 1, nk, step="matching",
                            detail=f"block {k}/{nk}" + (f" of {self.num_pairs} pairs" if self.num_pairs else ""),
                            rest_s=self.rest_after("matching"))
            return
        m = RE_MATCHING.search(line)
        if m:
            self.matcher, self.num_pairs = m.group(1), int(m.group(3))
            return
        m = RE_TIMING.search(line)
        if m:
            self.timings[m.group(1)] = float(m.group(2))
            return
        if RE_MAPSTART.search(line):
            self.step = "mapping"
            self.t_map0 = time.monotonic()
            events.progress(STAGE, 0, self.n_img, step="mapping", eta_s=self.mapping_eta(),
                            detail="images registered", force=True, rest_s=self.rest_after("mapping"))
            return
        m = RE_REG.search(line)
        if m:
            self.step = "mapping"
            if self.t_map0 is None:
                self.t_map0 = time.monotonic()
            self.reg = int(m.group(2))
            events.progress(STAGE, self.reg, self.n_img, step="mapping", eta_s=self.mapping_eta(), rest_s=self.rest_after("mapping"),
                            detail="images registered")
            return
        keys = (("featstat", RE_FEATSTAT), ("recon", RE_RECON), ("reproj", RE_REPROJ),
                ("rig", RE_RIG), ("ba", RE_BA), ("step", RE_STEP), ("depth", RE_DEPTH))
        for key, rx in keys:
            if not self.stereo and key in ("rig", "ba", "step"):
                continue
            m = rx.search(line)
            if m:
                self.metrics[key] = m.groups()
                return
        if self.stereo:
            m = RE_EYE.search(line)
            if m:
                self.metrics["eye_" + m.group(1)] = m.groups()[1:]


def _record_matching(pj, parser, matcher, reused):
    pj.metric(STAGE, "matcher", "reused" if reused else (parser.matcher or matcher))
    if parser.num_pairs is not None:
        pj.metric(STAGE, "num_pairs", parser.num_pairs)
    for ph in ("features", "matching", "mapping"):
        if ph in parser.timings:
            pj.metric(STAGE, f"{ph}_s", round(parser.timings[ph], 1))


def _append_timing(pj, parser, n_caps, n_img, export_s, reused):
    """A completed solve's phase times -> the timing store. Never fatal."""
    run = {"project": os.path.basename(pj.root), "route": pj.source_kind, "captures": n_caps,
           "images": n_img, "matcher": "reused" if reused else parser.matcher}
    if parser.num_pairs is not None and not reused:
        run["pairs"] = parser.num_pairs
    for ph in ("features", "matching", "mapping"):
        if ph in parser.timings and not (reused and ph != "mapping"):
            run[f"{ph}_s"] = round(parser.timings[ph], 1)
    if export_s is not None:
        run["export_s"] = round(export_s, 1)
    try:
        path = timing.append_solve_run(run)
    except Exception as e:  # noqa: BLE001 — the store is a convenience, never a failure
        events.log(STAGE, f"[hs] timing store not written: {e!r}")
        return
    if path:
        events.log(STAGE, f"[hs] timing appended to {path}")


def _load_model():
    try:
        return timing.model()
    except Exception:  # noqa: BLE001
        return timing.fit([])


def run(a, pj):
    pj.require(STAGE)
    if pj.frames_route:                 # array or mono: one shared camera through monocolmap.py
        run_array(a, pj)
        if getattr(a, "board", None):
            _scale_after(a, pj, dry_run=False)
        return
    _run_stereo(a, pj)
    if getattr(a, "board", None):
        # a stereo solve is metric already; the board measures how right the baseline is
        _scale_after(a, pj, dry_run=True)


def _scale_after(a, pj, dry_run):
    """`hs solve --board ...` = hs solve, then hs scale with the same board."""
    from argparse import Namespace
    from . import scale
    try:
        scale.run(Namespace(board=a.board, legacy_board=getattr(a, "legacy_board", False), eye="L",
                            min_views=3, dry_run=dry_run), pj)
    except events.StageError:
        if pj.status(scale.STAGE) == "running":
            pj.finish(scale.STAGE, ok=False, error="failed after hs solve --board")
        raise


def _run_stereo(a, pj):
    frames = pj.frames_dir
    n_sel = len([f for f in os.listdir(frames) if f.lower().endswith(".jpg")]) if os.path.isdir(frames) else 0
    if n_sel == 0:
        raise events.StageError("no selected frames", hint="hs select --project ...")
    prof_ref = a.profile or pj.m.get("profile_path") or pj.m.get("profile_id")
    if not prof_ref:
        raise events.StageError("project has no calibration profile", hint="re-run hs ingest")
    prof_path, prof = calib.load_profile(prof_ref)
    work = pj.stage_dir(STAGE)
    export_only = bool(getattr(a, "export_only", False))
    if export_only:
        for need in ("sparse/rig", "captures.json", "sfm_report.json", "images"):
            if not os.path.exists(os.path.join(work, need)):
                raise events.StageError(f"--export-only: no {pj.rel(os.path.join(work, need))} from an earlier solve",
                                        hint="run hs solve without --export-only")
        # keep the earlier run's numbers: begin() replaces the stage entry, and there is no sfm log to re-read
        prev = dict(pj.stage(STAGE).get("metrics") or {})
    keep_db = None
    if a.reuse_matches and not export_only and os.path.exists(os.path.join(work, "database.db")):
        keep_db = os.path.join(pj.root, "database.db.keep")
        shutil.move(os.path.join(work, "database.db"), keep_db)
    pj.begin(STAGE, argv=sys.argv, clean=not export_only)
    reused = bool(keep_db)
    if keep_db:
        shutil.move(keep_db, os.path.join(work, "database.db"))
    # the train dataset is written by export; it belongs to train/ but is produced here
    if os.path.isdir(pj.dataset_dir):
        shutil.rmtree(pj.dataset_dir)
    os.makedirs(pj.path("train"), exist_ok=True)

    calib_npz = calib.profile_to_npz(prof, os.path.join(work, "calib.npz"), a.baseline_mm)
    json.dump(prof, open(os.path.join(work, "profile.json"), "w"), indent=1)
    pj.m["profile_id"] = prof.get("profile_id")
    pj.m["profile_path"] = prof_path
    baseline = a.baseline_mm or calib.baseline_mm(prof)
    if a.baseline_mm:
        pj.metric(STAGE, "baseline_override_mm", a.baseline_mm)
    pj.metric(STAGE, "profile_baseline_mm", round(baseline, 4))
    pj.metric(STAGE, "source_kind", pj.source_kind)
    log = pj.log_path(STAGE)

    if export_only:
        caps = json.load(open(os.path.join(work, "captures.json")))["captures"]
        n_img = 2 * len(caps)
        pj.metric(STAGE, "captures", len(caps))
        pj.metric(STAGE, "export_only", True)
        parser = SfmParser(n_img, stereo=True, model=_load_model())
        parser.matcher = prev.get("matcher")
        parser.num_pairs = prev.get("num_pairs")
        for ph in ("features", "matching", "mapping"):
            if isinstance(prev.get(f"{ph}_s"), (int, float)):
                parser.timings[ph] = float(prev[f"{ph}_s"])
        for k in ("matcher", "num_pairs", "features_s", "matching_s", "mapping_s", "features_per_image_median",
                  "median_reproj_px", "eye_L_rms_px", "eye_R_rms_px", "ba_rig_rotation_shift_deg",
                  "rig_baseline_mm", "path_length_mm", "capture_step_median_mm",
                  "scene_depth_p5_mm", "scene_depth_median_mm", "scene_depth_p95_mm"):
            if k in prev:
                pj.metric(STAGE, k, prev[k])
        events.log(STAGE, f"[hs] --export-only: exporting solve/sparse/rig from the earlier run ({prev.get('matcher', '?')} matcher)")
        rep = json.load(open(os.path.join(work, "sfm_report.json")))
        rep_path = os.path.join(work, "sfm_report.json")
        mm = {}
    else:
        # ---- prep
        events.start(STAGE, "prep")
        runner.run(runner.python_argv("rigcolmap.py", "prep", frames, "-o", work), STAGE, log_path=log)
        caps = json.load(open(os.path.join(work, "captures.json")))["captures"]
        pj.metric(STAGE, "captures", len(caps))
        n_img = 2 * len(caps)
        matcher, m_argv = _matcher_argv(a, len(caps), pj)

        # ---- sfm
        events.start(STAGE, "sfm")
        argv = runner.python_argv("rigcolmap.py", "sfm", work, "--calib", calib_npz,
                                  "--peak-threshold", a.peak_threshold, "--features", a.features, *m_argv)
        if not a.no_float_rig:
            argv.append("--float-rig")
        if a.masks:
            argv += ["--masks", a.masks]
        parser = SfmParser(n_img, stereo=True, model=_load_model())

        runner.run(argv, STAGE, log_path=log, on_line=parser)
        _record_matching(pj, parser, matcher, reused)
        rep_path = os.path.join(work, "sfm_report.json")
        if not os.path.exists(rep_path):
            raise events.StageError("sfm wrote no sfm_report.json", hint="see logs/solve.log")
        rep = json.load(open(rep_path))
        mm = parser.metrics
    if "featstat" in mm:
        pj.metric(STAGE, "features_per_image_median", int(mm["featstat"][1]))
    pj.metric(STAGE, "num_images", rep["num_images"])
    pj.metric(STAGE, "num_frames", rep["num_frames"])
    pj.metric(STAGE, "num_points", rep["num_points"])
    pj.metric(STAGE, "mean_reproj_px", round(float(rep["mean_reproj_px"]), 4))
    if "reproj" in mm:
        pj.metric(STAGE, "median_reproj_px", float(mm["reproj"][1]))
    for eye in ("L", "R"):
        if "eye_" + eye in mm:
            pj.metric(STAGE, f"eye_{eye}_rms_px", float(mm["eye_" + eye][1]))
    if "ba" in mm:
        pj.metric(STAGE, "ba_rig_rotation_shift_deg", float(mm["ba"][1]))
    if "rig" in mm:
        pj.metric(STAGE, "rig_baseline_mm", float(mm["rig"][0]))
    if "step" in mm:
        pj.metric(STAGE, "path_length_mm", int(mm["step"][3]))
        pj.metric(STAGE, "capture_step_median_mm", int(mm["step"][2]))
    if "depth" in mm:
        pj.metric(STAGE, "scene_depth_p5_mm", int(mm["depth"][0]))
        pj.metric(STAGE, "scene_depth_median_mm", int(mm["depth"][1]))
        pj.metric(STAGE, "scene_depth_p95_mm", int(mm["depth"][2]))
    pj.artifact(STAGE, rep_path, "json")

    all_reg = rep["num_frames"] == len(caps)
    pj.check(STAGE, "all_frames_registered", all_reg, value=f"{rep['num_frames']}/{len(caps)}")
    pj.check(STAGE, "mean_reproj_ok", float(rep["mean_reproj_px"]) <= a.max_reproj,
             value=f"{float(rep['mean_reproj_px']):.3f} px (want ≤ {a.max_reproj}; rig6 1.394)")

    # ---- per-image table (needs the reconstruction; pycolmap is already a dependency)
    events.start(STAGE, "per_image")
    per = per_image_table(os.path.join(work, "sparse", "rig"))
    per_path = os.path.join(work, "per_image.json")
    json.dump(per, open(per_path, "w"), indent=1)
    pj.artifact(STAGE, per_path, "json")
    _per_image_checks(pj, per)

    if not all_reg:
        missing = sorted(set(c["capture"] for c in caps) - set(r["name"].split("/")[1].split(".")[0] for r in per["images"]))
        pj.metric(STAGE, "unregistered_captures", missing)
        runs = _runs(missing)
        if not getattr(a, "allow_partial", False):
            pj.finish(STAGE, ok=False, error="partial registration")
            raise events.StageError(
                f"only {rep['num_frames']} of {len(caps)} frames registered — the chain broke "
                f"(unregistered: {runs})",
                hint="a block in the middle of the clip is a passage COLMAP could not see (sky, a blown "
                     "wall, motion blur): `hs solve --export-only --allow-partial` trains on the rest and "
                     "leaves a hole in the coverage; a block at the end: `hs select --end N`; a chain of "
                     "single misses: `hs select --max-gap 45`. Then read solve/per_image.json.")
        pj.check(STAGE, "partial_solve_accepted", False, needs_human=True,
                 value=f"{rep['num_frames']}/{len(caps)} registered, exporting without {len(missing)} "
                       f"captures ({runs}); the coverage has a hole there")

    # ---- export
    events.start(STAGE, "export")
    ex = runner.run(runner.python_argv("rigcolmap.py", "export", os.path.join(work, "sparse", "rig"),
                                       "--images", os.path.join(work, "images"), "-o", pj.dataset_dir),
                    STAGE, log_path=log,
                    on_line=lambda l: (lambda m: m and events.progress(STAGE, int(m.group(1)), int(m.group(2)), step="export", rest_s=0.0))(RE_UNDIST.search(l)))
    pj.metric(STAGE, "export_s", round(ex.elapsed, 1))
    if not os.path.exists(pj.rig_npz):
        raise events.StageError("export wrote no rig.npz")
    # rig.npz's w/h drive the aim check's in-frame bounds and every path's output canvas, and
    # the two eyes are undistorted to different sizes — so they must be the LEFT views' size.
    # This caught write_rig_npz handing over the right eye's canvas (off by 4x2 px).
    import cv2
    G = np.load(pj.rig_npz, allow_pickle=True)
    l_dir = os.path.join(pj.dataset_dir, "images", "L")
    l_first = sorted(f for f in os.listdir(l_dir) if f.lower().endswith(".jpg"))[0]
    im = cv2.imread(os.path.join(l_dir, l_first))
    rig_wh, img_wh = (int(G["w"]), int(G["h"])), (im.shape[1], im.shape[0])
    pj.metric(STAGE, "view_size_L", list(img_wh))
    if "wh" in G.files and len(G["wh"]) > 1:
        pj.metric(STAGE, "view_size_R", [int(x) for x in G["wh"][1]])
    pj.check(STAGE, "rig_size_matches_L_views", rig_wh == img_wh,
             value=f"rig.npz {rig_wh[0]}x{rig_wh[1]} vs {l_first} {img_wh[0]}x{img_wh[1]}")
    pj.artifact(STAGE, pj.dataset_dir, "dataset")
    pj.artifact(STAGE, pj.rig_npz, "rig")
    ply = os.path.join(pj.dataset_dir, "sparse", "points3D.ply")
    if os.path.exists(ply):
        pj.artifact(STAGE, ply, "pointcloud")

    # ---- coverage (in the frame the path builders use)
    events.start(STAGE, "coverage")
    cov = coverage.write(pj.rig_npz, os.path.join(work, "coverage.json"))
    pj.artifact(STAGE, os.path.join(work, "coverage.json"), "json")
    pj.metric(STAGE, "subject_mm", cov["subject_mm"])
    pj.metric(STAGE, "azimuth_range_deg", cov["azimuth_range_deg"])
    pj.metric(STAGE, "elevation_range_deg", cov["elevation_range_deg"])
    pj.metric(STAGE, "distance_range_mm", cov["distance_range_mm"])
    seps = np.array([c["lr_separation_mm"] for c in cov["captures"]])
    dev = float(np.abs(seps - baseline).max())
    pj.metric(STAGE, "lr_separation_max_dev_mm", round(dev, 4))
    pj.check(STAGE, "rig_constraint_held", dev < 0.01,
             value=f"L–R separation {seps.min():.3f}–{seps.max():.3f} mm vs profile {baseline:.3f} mm")
    _outlier_check(pj, G)
    # the stage completed; a failed quality check stays visible in the manifest and the
    # event stream rather than blocking (partial registration raised above — that one blocks)
    pj.finish(STAGE, ok=True)
    _append_timing(pj, parser, len(caps), n_img, ex.elapsed, reused)


def _runs(names):
    """'cap245–255, cap257–269, cap294–318' from a list of capture names, for a message."""
    import re
    nums = []
    for n in names:
        m = re.search(r"(\d+)$", str(n))
        if m:
            nums.append((int(m.group(1)), str(n)[:m.start()]))
    if not nums:
        return ", ".join(str(n) for n in names[:12]) + ("…" if len(names) > 12 else "")
    nums.sort()
    out, start, prev, pre = [], nums[0][0], nums[0][0], nums[0][1]
    for k, p in nums[1:]:
        if k == prev + 1:
            prev = k
            continue
        out.append(f"{pre}{start}" if start == prev else f"{pre}{start}–{prev}")
        start = prev = k
        pre = p
    out.append(f"{pre}{start}" if start == prev else f"{pre}{start}–{prev}")
    return ", ".join(out)


def _per_image_checks(pj, per):
    """worst image (over the images that have observations) and the registered-but-empty ones:
    an image COLMAP posed with no surviving 3D observation has a pose nothing supports. The
    solve on 2026-09-22 had one (L/cap256, 0 observations) and `max()` over a None crashed the
    stage after two hours of work."""
    rows = per.get("images") or []
    scored = [r for r in rows if isinstance(r.get("mean_reproj_px"), (int, float))]
    empty = [r["name"] for r in rows if r.get("n_obs", 0) == 0]
    if scored:
        worst = max(scored, key=lambda r: r["mean_reproj_px"])
        pj.metric(STAGE, "worst_image_reproj_px", worst["mean_reproj_px"])
        pj.check(STAGE, "no_image_over_3px", worst["mean_reproj_px"] <= 3.0,
                 value=f"worst {worst['name']} {worst['mean_reproj_px']:.2f} px")
    if empty:
        pj.metric(STAGE, "images_without_observations", empty)
    pj.check(STAGE, "registered_images_have_observations", not empty, needs_human=bool(empty),
             value=("every registered image has 3D observations" if not empty else
                    f"{len(empty)} registered with none: {', '.join(empty[:8])}{'…' if len(empty) > 8 else ''} "
                    f"— exclude them from train (hs train --exclude) or re-select around them"))


def _outlier_check(pj, G):
    """Camera centres far from everyone else: a capture COLMAP placed somewhere impossible
    drags the path length and the scene depth (24.6 km and 78 m on the 09-22 solve) and would
    be a hole in a move's hull. Median distance from the median centre, 10x = an outlier."""
    try:
        names = [str(x) for x in G["names"]]
        C = np.asarray(G["C"], float)
    except Exception:
        return
    if len(C) < 4:
        return
    d = np.linalg.norm(C - np.median(C, axis=0), axis=1)
    med = float(np.median(d))
    far = [names[i] for i in np.flatnonzero(d > 10.0 * max(med, 1e-9))]
    caps = sorted({n[:-2] if n[-2:] in ("_L", "_R") else n for n in far})
    if caps:
        pj.metric(STAGE, "outlier_captures", caps)
    pj.check(STAGE, "no_outlier_camera_positions", not caps, needs_human=bool(caps),
             value=(f"all camera centres within 10x the median distance ({med:.0f} mm) of the middle" if not caps else
                    f"{len(caps)} capture(s) placed >10x the median distance ({med:.0f} mm) from the rest: "
                    f"{_runs(caps)} — exclude them from train and moves"))


def run_array(a, pj):
    """Array source: monocolmap.py prep -> sfm (one shared camera) -> export. Same outputs,
    metrics and checks as the Hydrogen path minus everything that is about a stereo rig."""
    frames = pj.frames_dir
    n_cam = len([f for f in os.listdir(frames)]) if os.path.isdir(frames) else 0
    if n_cam == 0:
        raise events.StageError("no frames", hint="hs ingest --frames DIR / --r3d DIR --take NNN")
    work = pj.stage_dir(STAGE)
    keep_db = None
    if a.reuse_matches and os.path.exists(os.path.join(work, "database.db")):
        keep_db = os.path.join(pj.root, "database.db.keep")
        shutil.move(os.path.join(work, "database.db"), keep_db)
    pj.begin(STAGE, argv=sys.argv)
    reused = bool(keep_db)
    if keep_db:
        shutil.move(keep_db, os.path.join(work, "database.db"))
    if os.path.isdir(pj.dataset_dir):
        shutil.rmtree(pj.dataset_dir)
    os.makedirs(pj.path("train"), exist_ok=True)
    pj.metric(STAGE, "source_kind", pj.source_kind)
    log = pj.log_path(STAGE)

    events.start(STAGE, "prep")
    runner.run(runner.python_argv("monocolmap.py", "prep", frames, "-o", work), STAGE, log_path=log)
    caps = json.load(open(os.path.join(work, "captures.json")))["captures"]
    pj.metric(STAGE, "captures", len(caps))
    n_img = len(caps)
    matcher, m_argv = _matcher_argv(a, len(caps), pj)

    events.start(STAGE, "sfm")
    argv = runner.python_argv("monocolmap.py", "sfm", work,
                              "--peak-threshold", a.peak_threshold, "--features", a.features, *m_argv)
    if a.masks:
        argv += ["--masks", a.masks]
    if a.focal_px:
        argv += ["--focal-px", a.focal_px]
    if a.fix_intrinsics:
        argv.append("--fix-intrinsics")
    if a.scale_pair:
        argv += ["--scale-pair", a.scale_pair]
    elif a.scale:
        argv += ["--scale", a.scale]
    parser = SfmParser(n_img, stereo=False, model=_load_model())

    runner.run(argv, STAGE, log_path=log, on_line=parser)
    _record_matching(pj, parser, matcher, reused)
    rep_path = os.path.join(work, "sfm_report.json")
    if not os.path.exists(rep_path):
        raise events.StageError("sfm wrote no sfm_report.json", hint="see logs/solve.log")
    rep = json.load(open(rep_path))
    mm = parser.metrics
    if "featstat" in mm:
        pj.metric(STAGE, "features_per_image_median", int(mm["featstat"][1]))
    pj.metric(STAGE, "num_images", rep["num_images"])
    pj.metric(STAGE, "num_frames", rep["num_frames"])
    pj.metric(STAGE, "num_points", rep["num_points"])
    pj.metric(STAGE, "mean_reproj_px", round(float(rep["mean_reproj_px"]), 4))
    if "reproj" in mm:
        pj.metric(STAGE, "median_reproj_px", float(mm["reproj"][1]))
    cam = rep.get("camera", {})
    if cam:
        pj.metric(STAGE, "camera_params", cam)
        fx = cam["params"][0]
        pj.metric(STAGE, "focal_px", round(fx, 1))
        pj.metric(STAGE, "hfov_deg", round(float(2 * np.degrees(np.arctan(cam["width"] / (2 * fx)))), 2))
    if "depth" in mm:
        pj.metric(STAGE, "scene_depth_p5_mm", int(mm["depth"][0]))
        pj.metric(STAGE, "scene_depth_median_mm", int(mm["depth"][1]))
        pj.metric(STAGE, "scene_depth_p95_mm", int(mm["depth"][2]))
    pj.artifact(STAGE, rep_path, "json")
    scale = rep.get("scale_to_m")
    pj.metric(STAGE, "scale_to_m", scale)
    pj.check(STAGE, "scene_scaled", scale is not None, needs_human=scale is None,
             value=(f"x{scale:.6f} from {a.scale_pair or a.scale}" if scale is not None
                    else "units are arbitrary: `hs scale --board SX,SY,SQ,MK` if a ChArUco board is in view, "
                         "else re-solve with --scale-pair CAM1,CAM2,MM (a measured camera spacing) or --scale S"))

    all_reg = rep["num_frames"] == len(caps)
    pj.check(STAGE, "all_frames_registered", all_reg, value=f"{rep['num_frames']}/{len(caps)}")
    pj.check(STAGE, "mean_reproj_ok", float(rep["mean_reproj_px"]) <= a.max_reproj,
             value=f"{float(rep['mean_reproj_px']):.3f} px (want ≤ {a.max_reproj})")

    events.start(STAGE, "per_image")
    per = per_image_table(os.path.join(work, "sparse", "rig"))
    per_path = os.path.join(work, "per_image.json")
    json.dump(per, open(per_path, "w"), indent=1)
    pj.artifact(STAGE, per_path, "json")
    _per_image_checks(pj, per)
    if not all_reg:
        pj.finish(STAGE, ok=False, error="partial registration")
        got = set(os.path.splitext(os.path.basename(r["name"]))[0] for r in per["images"])
        missing = sorted(set(c["capture"] for c in caps) - got)
        raise events.StageError(f"only {rep['num_frames']} of {len(caps)} cameras registered "
                                f"(unregistered: {', '.join(missing)})",
                                hint="look at solve/per_image.json; a camera that sees too little of what the "
                                     "others see cannot be placed. Do not train on a partial solve.")

    events.start(STAGE, "export")
    ex = runner.run(runner.python_argv("monocolmap.py", "export", os.path.join(work, "sparse", "rig"),
                                       "--images", os.path.join(work, "images"), "-o", pj.dataset_dir),
                    STAGE, log_path=log,
                    on_line=lambda l: (lambda m: m and events.progress(STAGE, int(m.group(1)), int(m.group(2)), step="export", rest_s=0.0))(RE_UNDIST.search(l)))
    pj.metric(STAGE, "export_s", round(ex.elapsed, 1))
    if not os.path.exists(pj.rig_npz):
        raise events.StageError("export wrote no rig.npz")
    import cv2
    G = np.load(pj.rig_npz, allow_pickle=True)
    l_dir = os.path.join(pj.dataset_dir, "images", "L")
    l_first = sorted(f for f in os.listdir(l_dir) if f.lower().endswith(".jpg"))[0]
    im = cv2.imread(os.path.join(l_dir, l_first))
    rig_wh, img_wh = (int(G["w"]), int(G["h"])), (im.shape[1], im.shape[0])
    pj.metric(STAGE, "view_size_L", list(img_wh))
    pj.check(STAGE, "rig_size_matches_L_views", rig_wh == img_wh,
             value=f"rig.npz {rig_wh[0]}x{rig_wh[1]} vs {l_first} {img_wh[0]}x{img_wh[1]}")
    pj.artifact(STAGE, pj.dataset_dir, "dataset")
    pj.artifact(STAGE, pj.rig_npz, "rig")
    ply = os.path.join(pj.dataset_dir, "sparse", "points3D.ply")
    if os.path.exists(ply):
        pj.artifact(STAGE, ply, "pointcloud")

    events.start(STAGE, "coverage")
    cov = coverage.write(pj.rig_npz, os.path.join(work, "coverage.json"))
    pj.artifact(STAGE, os.path.join(work, "coverage.json"), "json")
    pj.metric(STAGE, "subject_mm", cov["subject_mm"])
    pj.metric(STAGE, "azimuth_range_deg", cov["azimuth_range_deg"])
    pj.metric(STAGE, "elevation_range_deg", cov["elevation_range_deg"])
    pj.metric(STAGE, "distance_range_mm", cov["distance_range_mm"])
    pj.finish(STAGE, ok=True)
    _append_timing(pj, parser, len(caps), n_img, ex.elapsed, reused)


def per_image_table(recon_dir):
    """Per-image mean reprojection error and observation count, ordered by name."""
    import pycolmap
    rec = pycolmap.Reconstruction(recon_dir)
    rows = []
    for im in rec.images.values():
        if not im.has_pose:
            continue
        obs, xyz = [], []
        for p2 in im.points2D:
            if p2.has_point3D():
                obs.append(p2.xy)
                xyz.append(rec.point3D(p2.point3D_id).xyz)
        if not obs:
            rows.append({"name": im.name, "n_obs": 0, "mean_reproj_px": None})
            continue
        loc = im.cam_from_world() * np.asarray(xyz, float)
        ok = loc[:, 2] > 0
        pr = np.asarray(rec.camera(im.camera_id).img_from_cam(loc[ok]), float)
        e = np.linalg.norm(np.asarray(obs)[ok] - pr, axis=1)
        rows.append({"name": im.name, "n_obs": int(ok.sum()),
                     "mean_reproj_px": round(float(e.mean()), 4), "median_reproj_px": round(float(np.median(e)), 4)})
    rows.sort(key=lambda r: r["name"])
    return {"images": rows}
