"""Coverage table: where every capture sits about the subject (strategy §4.4, §4.6).

Computed from ``rig.npz`` — the same file and the same frame the path builders use — so a
key chosen from this table is exactly a capture index spline_path.py / key_path.py accept.

Definitions (all in the SfM frame, mm):
  subject   = median of points3D                         (the --aim-median point)
  up        = −mean over left views of the camera +y axis in world  (what the builders call
              "down", negated) — handheld video has no world vertical, so the average camera
              orientation defines it
  ref       = mean horizontal direction from the subject to the cameras   (azimuth 0)
  right     = mean camera +x axis, projected horizontal  (azimuth +)
  elevation = angle of (camera − subject) above the horizontal plane, degrees
  azimuth   = signed angle of the horizontal component from ref towards right, degrees
  distance  = |camera − subject|, mm

On the rig6 selftest: el −12° … +26° with the low passes at ≈ −7° (spec §1 agrees); az
−43° … +41°. The spec's table quoted az −26° … +50° for the same clip — its azimuth zero was
about 10° to the right of this one (the convention was never written down; the span
matches). Azimuth here is only used relatively (right third, nearest zero, leftmost).
"""
import json
import os

import numpy as np

from . import rig


def load_rig(rig_npz):
    return rig.load(rig_npz)


def compute(rig_npz):
    G, names, L = load_rig(rig_npz)
    stereo = rig.is_stereo(G)
    C = G["C"].astype(np.float64)
    R = G["R"].astype(np.float64)
    pts = G["pts"].astype(np.float64)
    subject = np.median(pts, axis=0)

    down = np.mean([R[v].T @ np.array([0.0, 1.0, 0.0]) for v in L], axis=0)
    up = -down / np.linalg.norm(down)
    xr = np.mean([R[v].T @ np.array([1.0, 0.0, 0.0]) for v in L], axis=0)

    CL = C[L]
    V = CL - subject
    dist = np.linalg.norm(V, axis=1)
    elev = np.degrees(np.arcsin(np.clip(V @ up / dist, -1, 1)))
    H = V - np.outer(V @ up, up)
    ref = H.mean(axis=0)
    ref -= (ref @ up) * up
    ref /= np.linalg.norm(ref)
    right = xr - (xr @ up) * up
    right /= np.linalg.norm(right)
    az = np.degrees(np.arctan2(H @ right, H @ ref))

    caps = []
    for i, v in enumerate(L):
        caps.append({"capture": i, "name": rig.capture_name(names[v]),
                     "azimuth_deg": round(float(az[i]), 2), "elevation_deg": round(float(elev[i]), 2),
                     "distance_mm": round(float(dist[i]), 1),
                     # a mono rig has no pair to separate; None keeps the column, not a fake zero
                     "lr_separation_mm": round(float(np.linalg.norm(C[v] - C[v + 1])), 4) if stereo else None})
    table = {
        "note": __doc__.split("\n\n")[1].strip(),
        "subject_mm": [round(float(x), 3) for x in subject],
        "up_world": [round(float(x), 6) for x in up],
        "azimuth_ref_world": [round(float(x), 6) for x in ref],
        "n_captures": len(caps),
        "stereo": stereo,
        "azimuth_range_deg": [round(float(az.min()), 2), round(float(az.max()), 2)],
        "elevation_range_deg": [round(float(elev.min()), 2), round(float(elev.max()), 2)],
        "distance_range_mm": [round(float(dist.min()), 1), round(float(dist.max()), 1)],
        "captures": caps,
    }
    return table


def write(rig_npz, out_path):
    t = compute(rig_npz)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    json.dump(t, open(out_path, "w"), indent=1)
    return t


# ---------------------------------------------------------------- presets (strategy §4.6)

def _longest_run(ok):
    best, cur = (0, 0), None
    for i, o in enumerate(list(ok) + [False]):
        if o and cur is None:
            cur = i
        elif not o and cur is not None:
            if i - cur > best[1] - best[0]:
                best = (cur, i)
            cur = None
    return best


def sweep_window(table, tol_deg=6.0, step_break=2.5, rig_npz=None):
    """Longest contiguous run of captures within ±tol of the median elevation, then — when
    the camera centres are available — the longest piece of that run not crossing a
    capture-to-capture step larger than step_break × the run's median step. A fast step is
    where the smoothed spline cuts a corner and the hull number climbs (rig6 selftest: the
    64 mm step between captures 2 and 3 took the sweep to 25 mm; breaking there gives 3:34
    at 17 mm). Returns ((lo, hi), median_elevation) with lo:hi half-open for
    spline_path.py --captures."""
    el = np.array([c["elevation_deg"] for c in table["captures"]])
    med = float(np.median(el))
    lo, hi = _longest_run(np.abs(el - med) <= tol_deg)
    if rig_npz and hi - lo >= 6:
        G, names, L = load_rig(rig_npz)
        CL = G["C"].astype(np.float64)[L]
        step = np.linalg.norm(np.diff(CL[lo:hi], axis=0), axis=1)     # step[i] = cap lo+i -> lo+i+1
        limit = step_break * float(np.median(step))
        # captures joined by a slow-enough step stay together; a fast step breaks the run
        pieces, cur = [], lo
        for i, s in enumerate(step):
            if s > limit:
                pieces.append((cur, lo + i + 1))
                cur = lo + i + 1
        pieces.append((cur, hi))
        plo, phi = max(pieces, key=lambda p: p[1] - p[0])
        if phi - plo >= 4:
            lo, hi = plo, phi
    return (lo, hi), med


def boom_keys(table, low_tol_deg=6.0, high_min_deg=8.0):
    """Boom–slide–settle keys from the coverage table.

    low   = captures within low_tol of the low-pass median elevation (the median elevation
            of the captures below the overall median)
    start = highest-elevation capture in the right third of the azimuth range
    boom  = walk forward in capture order from start to the first low capture
    slide = every other low capture from there in capture order, then the leftmost low
    settle= the low capture nearest azimuth 0
    Returns (keys, info). If the clip has no high pass (max elevation < high_min) the
    result is a low-only slide and info["fallback"] says so. rig6: 51,56,58,60,62,64,16,62.
    """
    caps = table["captures"]
    az = np.array([c["azimuth_deg"] for c in caps])
    el = np.array([c["elevation_deg"] for c in caps])
    n = len(caps)
    overall_med = float(np.median(el))
    low_pool = el[el <= overall_med]
    low_med = float(np.median(low_pool)) if len(low_pool) else overall_med
    low = np.abs(el - low_med) <= low_tol_deg
    low_idx = [i for i in range(n) if low[i]]
    info = {"low_pass_median_el": round(low_med, 2), "n_low": int(low.sum()), "fallback": None}
    if len(low_idx) < 3:
        raise ValueError(f"only {len(low_idx)} captures within ±{low_tol_deg}° of the low-pass elevation")

    leftmost_low = min(low_idx, key=lambda i: az[i])
    settle = min(low_idx, key=lambda i: abs(az[i]))

    if el.max() < high_min_deg:
        # no high pass: slide along the low pass from its right end to its left end
        info["fallback"] = f"no high pass (max elevation {el.max():.1f}° < {high_min_deg}°): low-only slide"
        rightmost_low = max(low_idx, key=lambda i: az[i])
        seq = [i for i in low_idx if i >= rightmost_low][::2]
        keys = [rightmost_low] + [i for i in seq if i != rightmost_low]
    else:
        a_lo, a_hi = az.min(), az.max()
        right_third = [i for i in range(n) if az[i] >= a_lo + 2.0 * (a_hi - a_lo) / 3.0]
        start = max(right_third, key=lambda i: el[i])
        after = [i for i in low_idx if i > start]
        if not after:
            after = [i for i in low_idx if i < start]  # boom backwards if nothing follows
            first_low = after[-1] if after else low_idx[0]
            slide = [i for i in low_idx if i <= first_low][::-1][::2]
        else:
            first_low = after[0]
            slide = after[::2]
        keys = [start] + slide
    if keys[-1] != leftmost_low:
        keys.append(leftmost_low)
    if keys[-1] != settle:
        keys.append(settle)
    # collapse accidental repeats of adjacent keys
    out = []
    for k in keys:
        if not out or out[-1] != k:
            out.append(int(k))
    info["start"] = out[0]
    info["settle"] = int(settle)
    info["leftmost_low"] = int(leftmost_low)
    return out, info
