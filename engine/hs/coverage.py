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


HOLDOUT_JSON = os.path.join("solve", "holdout.json")


def load_holdout(root):
    """solve/holdout.json of a project (written by `hs cameras --holdout N --write`), or raise
    ValueError saying how to make one. `hs train --exclude @holdout` and
    `hs views --captures holdout` both read it, so the views scored are the views left out."""
    p = os.path.join(root, HOLDOUT_JSON)
    if not os.path.exists(p):
        raise ValueError(f"no {HOLDOUT_JSON} in {root}: run `hs cameras -p {root} --holdout N --write` first")
    h = json.load(open(p))
    if not h.get("captures") or not h.get("names"):
        raise ValueError(f"{p} names no captures")
    return h


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


# ---------------------------------------------------------------- hold-out selection

HOLDOUT_METHODS = ("fps", "interval", "azimuth")


def _fps(C, n, start, pool=None, chosen=()):
    """Farthest-point sampling over rows of C: each pick is the capture farthest from every
    capture already picked. ``pool`` limits the candidates, ``chosen`` seeds the picked set."""
    pool = list(range(len(C))) if pool is None else list(pool)
    picked = list(chosen)
    if not picked:
        picked.append(int(start))
    d = np.full(len(C), np.inf)
    for p in picked:
        d = np.minimum(d, np.linalg.norm(C - C[p], axis=1))
    cand = np.array([i for i in pool if i not in picked], int)
    while len(picked) < len(chosen) + n and len(cand):
        # ties go to the lower index, so a run is reproducible to the capture
        j = int(cand[np.argmax(d[cand])])
        picked.append(j)
        d = np.minimum(d, np.linalg.norm(C - C[j], axis=1))
        cand = cand[cand != j]
    return picked[len(chosen):] if chosen else picked


def select_holdout(centres, n, method="fps", azimuth=None, seed=0):
    """Capture indices to hold out of training, sorted.

    ``centres`` (M, 3) camera centres of the reference views in capture order (rig.npz C[L]);
    ``azimuth`` (M,) degrees from the coverage table, needed by ``azimuth``.

      fps       farthest-point sampling over the camera centres, seeded at the capture farthest
                from their centroid: spatially uniform, so a dense stretch of the orbit cannot
                take most of the hold-outs and a sparse one none (NeRF Director's selection).
      interval  every (M // n)-th capture starting half a step in -- the hand-picked baselines'
                cap005, cap015, ... -- kept for comparison. ``seed`` shifts the phase.
      azimuth   n equal-width bands over the covered azimuth range, one capture per band (the
                one nearest the band centre); a band the camera never entered gets nothing, and
                the shortfall is filled by farthest-point sampling over what is left.
    """
    C = np.asarray(centres, float)
    M = len(C)
    if n < 1 or n >= M:
        raise ValueError(f"cannot hold out {n} of {M} captures (need 1 <= n < {M})")
    if method == "fps":
        start = int(np.argmax(np.linalg.norm(C - C.mean(axis=0), axis=1)))
        out = _fps(C, n, start)
    elif method == "interval":
        step = max(M // n, 1)
        start = (step // 2 + int(seed)) % step
        out = list(range(start, M, step))[:n]
    elif method == "azimuth":
        if azimuth is None:
            raise ValueError("azimuth selection needs the coverage table's azimuths")
        az = np.asarray(azimuth, float)
        lo, hi = float(az.min()), float(az.max())
        width = (hi - lo) / n or 1.0
        out = []
        for b in range(n):
            b_lo = lo + b * width
            inside = np.where((az >= b_lo) & ((az < b_lo + width) | (b == n - 1)))[0]
            inside = [int(i) for i in inside if int(i) not in out]
            if inside:
                centre = b_lo + width / 2
                out.append(min(inside, key=lambda i: (abs(az[i] - centre), i)))
        if len(out) < n:
            out += _fps(C, n - len(out), None, chosen=out)
    else:
        raise ValueError(f"unknown hold-out method {method!r} (one of {', '.join(HOLDOUT_METHODS)})")
    return sorted(int(i) for i in out)


def holdout_stats(centres, chosen, azimuth=None, band=None):
    """How well a hold-out set samples the capture, as numbers.

    nn_holdout_mm      each hold-out's distance to its nearest other hold-out: a small minimum
                       means two hold-outs are spent on the same place
    holdout_to_rest_mm each hold-out's distance to its nearest TRAINING capture: how far out of
                       sample it is (0 would be a duplicate of a training view)
    bands              per azimuth band (hs.bands.AZIMUTH_BAND, the band hs views scores):
                       captures in the band, hold-outs in it
    empty_bands        bands that have captures but no hold-out
    """
    from .bands import AZIMUTH_BAND, azimuth_band_lo
    band = band or AZIMUTH_BAND
    C = np.asarray(centres, float)
    chosen = sorted(int(i) for i in chosen)
    rest = [i for i in range(len(C)) if i not in chosen]

    def summary(d):
        d = np.asarray(d, float)
        return ({"min": round(float(d.min()), 2), "median": round(float(np.median(d)), 2)}
                if len(d) else {"min": None, "median": None})

    D = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=2)
    nn_h = [min(D[i, j] for j in chosen if j != i) for i in chosen] if len(chosen) > 1 else []
    to_rest = [min(D[i, j] for j in rest) for i in chosen] if rest else []
    out = {"n": len(chosen), "of": len(C),
           "nn_holdout_mm": summary(nn_h), "holdout_to_rest_mm": summary(to_rest)}
    if azimuth is not None:
        az = np.asarray(azimuth, float)
        per = {}
        for i, a in enumerate(az):
            lo = azimuth_band_lo(a, band)
            rec = per.setdefault(lo, {"azimuth_deg": [lo, lo + band], "captures": 0, "holdouts": 0})
            rec["captures"] += 1
            rec["holdouts"] += int(i in chosen)
        out["bands"] = [per[k] for k in sorted(per)]
        out["empty_bands"] = [per[k]["azimuth_deg"] for k in sorted(per) if not per[k]["holdouts"]]
    return out
