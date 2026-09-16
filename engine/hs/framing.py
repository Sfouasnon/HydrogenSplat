"""Framing a person: keep the crown of the head a fixed distance below the top of a cropped frame.

``move/subject.json`` holds the subject's crown in solve coordinates (mm)::

    {"crown_mm": [x, y, z], "note": "..."}

With it, ``hs move --headroom-mm 25.4`` solves the aim height so that, after a centred-width
crop to ``--frame-aspect`` (default 2.35), the crown plus the headroom fits in the crop on every
frame, as low as that allows (the ceiling above a person is the least-observed part of a room
capture). The aim moves only along the solve's up axis, straight below the crown, so the subject
stays centred left-right. The per-frame crown row and depth are written to
``move/<name>_frame.json``; ``hs grade`` turns them into a crop that follows the head, so the
headroom stays the same while the virtual camera pushes in and out.

Moves are OpenCV camera-to-world matrices in metres; the solve (rig.npz, subject.json) is in mm.
"""
import json
import os
import subprocess

import numpy as np

from . import events

MM_PER_M = 1000.0
FRAME_FILE = "{name}_frame.json"


def load_subject(pj):
    p = pj.path("move", "subject.json")
    if not os.path.exists(p):
        return None
    try:
        return np.array(json.load(open(p))["crown_mm"], float)
    except (ValueError, KeyError, TypeError):
        raise events.StageError(f"{p} has no crown_mm [x, y, z]")


def up_axis(pj):
    cov = json.load(open(pj.path("solve", "coverage.json")))
    up = np.array(cov["up_world"], float)
    return up / np.linalg.norm(up)


def crown_track(move, crown_mm):
    """Per frame: (row of the crown in native px, depth in mm, fy) for a loaded move json."""
    K = np.array(move["K"], float)
    rows, depths = [], []
    for f in move["frames"]:
        c2w = np.array(f["c2w"], float)
        X = c2w[:3, :3].T @ (crown_mm - c2w[:3, 3] * MM_PER_M)
        rows.append(float((K @ X)[1] / X[2]))
        depths.append(float(X[2]))
    return np.array(rows), np.array(depths), float(K[1, 1])


def crop_rows(rows, depths, fy, headroom_mm):
    """Top row of a crop that leaves headroom_mm above the crown, native px (unclamped)."""
    return rows - fy * headroom_mm / np.maximum(depths, 1.0)


def room(move, aspect):
    """How far down (native px) a centred full-width crop of this aspect can start."""
    return move["height"] - move["width"] / aspect


def solve_aim(build, crown_mm, up, aspect, headroom_mm, lo=0.0, hi=1500.0, iters=14):
    """Bisection on t (mm below the crown along -up). build(aim_xyz) -> move dict.

    Aiming lower tilts the camera down and lifts the crown in frame. The chosen t is the
    lowest crop start that still fits on every frame: max over frames of the crop row equals
    the room the aspect leaves (so the ceiling is cut as much as possible), and no frame
    needs the crop to start above row 0."""
    best = None
    for _ in range(iters):
        t = (lo + hi) / 2.0
        move = build(crown_mm - t * up)
        rows, depths, fy = crown_track(move, crown_mm)
        tops = crop_rows(rows, depths, fy, headroom_mm)
        if tops.max() > room(move, aspect):
            lo = t          # crown too low in frame: aim lower
        else:
            hi = t
            best = t
    if best is None:
        raise events.StageError("no aim keeps the head inside the crop",
                                hint="the move is too close to the subject for this aspect; use a wider aspect or a farther window")
    move = build(crown_mm - best * up)
    rows, depths, fy = crown_track(move, crown_mm)
    tops = crop_rows(rows, depths, fy, headroom_mm)
    if tops.min() < 0:
        events.check("move", "headroom_fits_every_frame", False,
                     value=f"crown needs the crop to start {-tops.min():.0f} px above the frame on its highest frame")
    return best, move, rows, depths, fy


def builder(argv_without_aim, out):
    """A build(aim) callable that reruns the path builder quietly with --aim-point."""
    def build(aim):
        a = ",".join(f"{x:.1f}" for x in aim)
        r = subprocess.run(argv_without_aim + [f"--aim-point={a}"], capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out):
            raise events.StageError("path builder failed while framing", hint=(r.stderr or r.stdout)[-300:])
        return json.load(open(out))
    return build


def write_track(pj, name, move, rows, depths, fy, aspect, headroom_mm, aim, t):
    rec = {"aspect": aspect, "headroom_mm": headroom_mm, "native": [move["width"], move["height"]],
           "fy": fy, "fps": move.get("fps", 30.0), "aim_mm": [round(float(x), 1) for x in aim],
           "aim_below_crown_mm": round(t, 1),
           "crown_row": [round(r, 2) for r in rows], "depth_mm": [round(d, 1) for d in depths]}
    p = pj.path("move", FRAME_FILE.format(name=name))
    with open(p, "w") as f:
        json.dump(rec, f)
    return p


def load_track(pj, name):
    p = pj.path("move", FRAME_FILE.format(name=name))
    return json.load(open(p)) if os.path.exists(p) else None


def crop_track_px(track, out_w, out_h, aspect, headroom_mm=None, smooth=15):
    """Crop top row per frame in output pixels for a video of out_w x out_h."""
    hr = track["headroom_mm"] if headroom_mm is None else headroom_mm
    nw, nh = track["native"]
    sy = out_h / float(nh)
    tops = crop_rows(np.array(track["crown_row"]), np.array(track["depth_mm"]), track["fy"], hr) * sy
    if smooth > 1 and len(tops) > smooth:
        k = np.ones(smooth) / smooth
        pad = np.pad(tops, (smooth // 2, smooth - 1 - smooth // 2), mode="edge")
        tops = np.convolve(pad, k, mode="valid")
    ch = even(out_w / aspect)
    return np.clip(np.round(tops), 0, out_h - ch).astype(int), ch


def even(x):
    return int(round(x / 2.0)) * 2
