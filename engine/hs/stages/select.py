"""hs select — frame selection by parallax (strategy §4.3) = select_frames.py, unchanged.

Defaults are the calibrated ones (residual 1.5 px at 480 wide, min gap 6, max 90, search 4,
clip < 2%). Adds: the checks — frame count in 30–120, median gap 6–15, fraction of max-gap
picks < 15% — and select/quality.json: every pick's sharpness (Laplacian variance), exposure off
the set's median in stops, both eyes, noise, clipping, the sharpest frame in its interval, and
report-only flags (hs/frame_quality.py). select/thumbs/ holds a left-eye thumbnail per pick for
the app; select/contact.jpg is the same data drawn on one sheet.

--keyframes (opt-in): only H.264 I-frames are candidates — on the Hydrogen's Baseline GOP-30
stream the P-frames' Laplacian is inflated by compression artefacts, and the I-frame is the
cleanest picture in each GOP (select_frames.py has the why). The median-gap check then reads
in GOPs, and keyframes_used checks every pick is one. --highlight-knee K: a soft highlight knee
on the written frames, here because every later stage reads these photographs; not exposure
matching (hs exposure) and not a look (hs grade). quality.json's "keyframes" block and the
keyframe metrics say how many picks are I-frames in either mode (when ffprobe could list them).
"""
import json
import os
import re
import sys

import numpy as np

from .. import events, frame_measure, frame_quality, runner
from .ingest import find_ffprobe

STAGE = "select"
SEL_RE = re.compile(r"^\s*sel\s+(\d+)\s+frame\s+(\d+)\s+gap\s+(\d+)\s+residual\s+([-\d.]+) px\s+sharp\s+([\d.]+)\s+clip\s+([\d.]+)%\s+\[(\w[\w-]*)\]")


def add_parser(sub):
    p = sub.add_parser("select", help="select frames by parallax (select_frames.py)")
    p.add_argument("--residual", type=float, default=1.5, help="median homography residual, px at work width")
    p.add_argument("--min-gap", type=int, default=6)
    p.add_argument("--max-gap", type=int, default=90)
    p.add_argument("--search", type=int, default=4)
    p.add_argument("--max-clip", type=float, default=0.02)
    p.add_argument("--work-width", type=int, default=480)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1)
    p.add_argument("--dry-run", action="store_true", help="selection.json only (the threshold slider path)")
    p.add_argument("--keyframes", action="store_true",
                   help="only H.264 I-frames are candidates (the frames without accumulated compression artefacts)")
    p.add_argument("--search-keyframes", type=int, default=2,
                   help="--keyframes: keyframes looked at per pick, from the first at or after the parallax crossing")
    p.add_argument("--min-sharp-rel", type=float, default=frame_quality.SOFT_REL,
                   help="--keyframes: a keyframe's focus must be at least this fraction of the picks' running median")
    p.add_argument("--highlight-knee", type=float, default=None, metavar="K",
                   help="soft highlight knee at linear K (0 < K < 1) on the written frames; off by default")
    p.add_argument("--ffprobe", default=os.environ.get("HS_FFPROBE", "ffprobe"),
                   help="ffprobe, for the keyframe indices (as hs ingest)")
    return p


def run(a, pj):
    pj.require(STAGE)
    if pj.frames_route:
        what = ("an array project has one frame per camera" if pj.source_kind == "array"
                else "a mono project was ingested as frames already picked (select_frames.py --mono)")
        raise events.StageError(f"{what}; ingest already marked select done",
                                hint="hs solve --project ...")
    clip = pj.clip
    if not clip or not os.path.exists(clip):
        raise events.StageError("no source clip in the project", hint="hs ingest --clip ...")
    n_total = int(pj.m.get("source", {}).get("probe", {}).get("nb_frames") or 0)
    frames_dir = pj.frames_dir
    argv = runner.python_argv("select_frames.py", clip, "-o", frames_dir,
                              "--residual", a.residual, "--min-gap", a.min_gap, "--max-gap", a.max_gap,
                              "--search", a.search, "--max-clip", a.max_clip, "--work-width", a.work_width,
                              "--start", a.start, "--end", a.end)
    if a.dry_run:
        argv.append("--dry-run")
    # new flags only when set, so a default run's command line is what it always was
    # (getattr: selftest builds its Namespace by hand)
    keyframes = bool(getattr(a, "keyframes", False))
    knee = getattr(a, "highlight_knee", None)
    ffprobe_bin = getattr(a, "ffprobe", None) or os.environ.get("HS_FFPROBE", "ffprobe")
    if knee is not None and not 0.0 < knee < 1.0:
        raise events.StageError(f"--highlight-knee must be between 0 and 1, got {knee}",
                                hint="0.85 compresses the top of the range; 0.9 is gentler")
    if keyframes:
        exe = find_ffprobe(ffprobe_bin)
        if exe is None:
            raise events.StageError(f"--keyframes needs ffprobe to find the I-frames; not found ({ffprobe_bin})",
                                    hint="brew install ffmpeg")
        argv += ["--keyframes", "--search-keyframes", str(getattr(a, "search_keyframes", 2)),
                 "--min-sharp-rel", str(getattr(a, "min_sharp_rel", frame_quality.SOFT_REL)), "--ffprobe", exe]
    elif ffprobe_bin != os.environ.get("HS_FFPROBE", "ffprobe"):
        argv += ["--ffprobe", ffprobe_bin]
    if knee is not None:
        argv += ["--highlight-knee", str(knee)]
    pj.begin(STAGE, argv=sys.argv, clean=not a.dry_run)
    os.makedirs(frames_dir, exist_ok=True)
    events.start(STAGE, "select")
    picks = []

    def on_line(line):
        m = SEL_RE.match(line)
        if m:
            k, f, gap = int(m.group(1)), int(m.group(2)), int(m.group(3))
            picks.append(f)
            events.progress(STAGE, done=f, total=n_total or None, detail=f"{k + 1} frames selected",
                            step="select")

    runner.run(argv, STAGE, log_path=pj.log_path(STAGE), on_line=on_line)
    sel_path = os.path.join(frames_dir, "selection.json")
    if not os.path.exists(sel_path):
        raise events.StageError("select_frames.py wrote no selection.json")
    sel = json.load(open(sel_path))
    selected = sel["selected"]
    n = len(selected)
    gaps = [s["gap"] for s in selected if "gap" in s]
    med_gap = float(np.median(gaps)) if gaps else 0.0
    # a keyframe pick names what made it due; its gap includes the wait for the next I-frame
    n_maxgap = sum(1 for s in selected if "gap" in s
                   and (s["trigger"] == "max-gap" if "trigger" in s else s["gap"] >= a.max_gap))
    frac_maxgap = n_maxgap / max(1, len(gaps))
    kfs = frame_quality.keyframe_summary(sel)
    gop = kfs["gop_median"]
    events.progress(STAGE, done=sel["frames_total"], total=sel["frames_total"], detail=f"{n} frames selected",
                    step="select", force=True)

    pj.metric(STAGE, "frames_total", sel["frames_total"])
    pj.metric(STAGE, "frames_selected", n)
    pj.metric(STAGE, "gap_min", int(min(gaps)) if gaps else 0)
    pj.metric(STAGE, "gap_max", int(max(gaps)) if gaps else 0)
    pj.metric(STAGE, "gap_median", med_gap)
    pj.metric(STAGE, "maxgap_fraction", round(frac_maxgap, 4))
    pj.metric(STAGE, "residual_px", a.residual)
    pj.metric(STAGE, "clip_fraction_median", round(float(np.median([s["clip"] for s in selected])), 4))
    pj.metric(STAGE, "sharpness_median", round(float(np.median([s["sharpness"] for s in selected])), 1))
    pj.artifact(STAGE, sel_path, "json")

    pj.metric(STAGE, "selection_mode", kfs["mode"])
    if kfs["picks_known"]:
        pj.metric(STAGE, "keyframes_picked", kfs["picks_keyframes"])
    if kfs["keyframes_total"] is not None:
        pj.metric(STAGE, "keyframes_total", kfs["keyframes_total"])
    if gop is not None:
        pj.metric(STAGE, "gop_median", gop)
    if keyframes:
        pj.metric(STAGE, "quality_fallbacks", kfs["quality_fallbacks"])
    if knee is not None:
        pj.metric(STAGE, "highlight_knee", knee)

    pj.check(STAGE, "frame_count_in_range", 30 <= n <= 120, value=f"{n} (want 30–120; rig6 65)")
    if keyframes and gop:
        # every gap is a whole number of GOPs; more than two in the median means the parallax
        # rule is routinely waiting past a keyframe it could have used — or the keyframes are sparse
        pj.check(STAGE, "median_gap_in_range", med_gap <= 2 * gop,
                 value=f"{med_gap:g} (keyframes: want at most 2 GOPs = {2 * gop:g}; the parallax rule alone wants 6–15)")
    else:
        pj.check(STAGE, "median_gap_in_range", 6 <= med_gap <= 15, value=f"{med_gap:g} (want 6–15)")
    pj.check(STAGE, "maxgap_fraction_low", frac_maxgap < 0.15,
             value=f"{100 * frac_maxgap:.1f}% of picks hit max-gap {a.max_gap} (want < 15%; high = phone stood still)")
    if keyframes:
        k_ok = kfs["picks_keyframes"] == n
        pj.check(STAGE, "keyframes_used", k_ok,
                 value=f"{kfs['picks_keyframes']} of {n} picks are I-frames ({kfs['keyframes_total']} in the clip, "
                       f"GOP {gop if gop is None else format(gop, 'g')})"
                       + (f"; {kfs['quality_fallbacks']} took the least bad keyframe" if kfs["quality_fallbacks"] else ""))
    elif kfs["picks_known"]:
        events.log(STAGE, f"[hs] {kfs['picks_keyframes']} of {n} picks happen to be I-frames "
                          f"({kfs['keyframes_total']} in the clip); --keyframes takes only those")
    if knee is not None:
        pj.check(STAGE, "highlight_knee_applied", True,
                 value=f"K={knee:g} on {0 if a.dry_run else n} frames" + (" (dry run: none written)" if a.dry_run else ""))

    select_dir = os.path.dirname(frames_dir)
    measured = {}
    if not a.dry_run:
        events.start(STAGE, "measure")
        try:
            measured = frame_measure.measure(frames_dir, os.path.join(select_dir, "thumbs"),
                                             work_width=a.work_width)
        except Exception as e:
            events.log(STAGE, f"[hs] measuring the written frames failed: {e!r}")
    quality = frame_quality.analyse(sel, measured)
    q_path = os.path.join(select_dir, "quality.json")
    with open(q_path, "w") as fh:
        json.dump(quality, fh, indent=1)
    pj.artifact(STAGE, q_path, "json")
    # report, don't judge: the flag thresholds are uncalibrated (frame_quality.py)
    pj.metric(STAGE, "frames_flagged", quality["flagged"])
    for name, count in sorted(quality["flag_counts"].items()):
        pj.metric(STAGE, f"flag_{name}", count)
    med = quality["medians"]
    if med["noise"] is not None:
        pj.metric(STAGE, "noise_median", med["noise"])
    if med["eye_ev"] is not None:
        pj.metric(STAGE, "eye_ev_median", med["eye_ev"])

    if not a.dry_run:
        events.start(STAGE, "contact")
        try:
            contact = frame_measure.contact_sheet(frames_dir, quality, os.path.join(select_dir, "contact.jpg"))
            if contact:
                pj.artifact(STAGE, contact, "image")
        except Exception as e:
            events.log(STAGE, f"[hs] contact sheet failed: {e!r}")
    pj.finish(STAGE, ok=True)
