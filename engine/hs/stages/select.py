"""hs select — frame selection by parallax (strategy §4.3) = select_frames.py, unchanged.

Defaults are the calibrated ones (residual 1.5 px at 480 wide, min gap 6, max 90, search 4,
clip < 2%). Adds: a contact sheet, per-frame noise estimate, and the checks — frame count in
30–120, median gap 6–15, fraction of max-gap picks < 15%.
"""
import json
import os
import re
import sys

import numpy as np

from .. import events, runner

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
    return p


def run(a, pj):
    pj.require(STAGE)
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
    n_maxgap = sum(1 for s in selected if "gap" in s and s["gap"] >= a.max_gap)
    frac_maxgap = n_maxgap / max(1, len(gaps))
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

    pj.check(STAGE, "frame_count_in_range", 30 <= n <= 120, value=f"{n} (want 30–120; rig6 65)")
    pj.check(STAGE, "median_gap_in_range", 6 <= med_gap <= 15, value=f"{med_gap:g} (want 6–15)")
    pj.check(STAGE, "maxgap_fraction_low", frac_maxgap < 0.15,
             value=f"{100 * frac_maxgap:.1f}% of picks hit max-gap {a.max_gap} (want < 15%; high = phone stood still)")

    if not a.dry_run:
        events.start(STAGE, "contact")
        try:
            contact, noise = contact_sheet_and_noise(frames_dir, selected)
            if contact:
                pj.artifact(STAGE, contact, "image")
            if noise is not None:
                pj.metric(STAGE, "noise_median", noise)
                # rig6 reference is not yet measured with this estimator; report, don't judge
        except Exception as e:
            events.log(STAGE, f"[hs] contact sheet failed: {e!r}")
    pj.finish(STAGE, ok=True)


def contact_sheet_and_noise(frames_dir, selected, thumb_w=240, per_row=8):
    import cv2
    files = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith(".jpg"))
    if not files:
        return None, None
    by_frame = {s["frame"]: s for s in selected}
    thumbs, noises = [], []
    for f in files:
        im = cv2.imread(os.path.join(frames_dir, f), cv2.IMREAD_COLOR)
        if im is None:
            continue
        L = im[:, : im.shape[1] // 2]
        # noise: median |Laplacian| where the image is flat (low gradient)
        g = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY)
        gs = cv2.resize(g, (960, 540), interpolation=cv2.INTER_AREA)
        gx = cv2.Sobel(gs, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gs, cv2.CV_32F, 0, 1, ksize=3)
        flat = (np.abs(gx) + np.abs(gy)) < 8.0
        lap = np.abs(cv2.Laplacian(gs, cv2.CV_32F, ksize=1))
        if flat.sum() > 1000:
            noises.append(float(lap[flat].mean()))
        th = cv2.resize(L, (thumb_w, int(round(thumb_w * L.shape[0] / L.shape[1]))), interpolation=cv2.INTER_AREA)
        m = re.match(r"VID_(\d+)_(\d+)_2x1", f)
        label = f"{int(m.group(1))}:{int(m.group(2))}" if m else f
        s = by_frame.get(int(m.group(2))) if m else None
        if s:
            label += f" r{s['residual']:.1f} c{100 * s['clip']:.0f}%"
        cv2.rectangle(th, (0, 0), (th.shape[1], 16), (0, 0, 0), -1)
        cv2.putText(th, label, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(th)
    if not thumbs:
        return None, None
    h, w = thumbs[0].shape[:2]
    rows = (len(thumbs) + per_row - 1) // per_row
    sheet = np.zeros((rows * h, per_row * w, 3), np.uint8)
    for i, th in enumerate(thumbs):
        r, c = divmod(i, per_row)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = th
    out = os.path.join(os.path.dirname(frames_dir), "contact.jpg")
    cv2.imwrite(out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out, (round(float(np.median(noises)), 3) if noises else None)
