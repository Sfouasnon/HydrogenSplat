"""walk — a camera move for a place, not a subject: go where the camera went.

Every other move in hs looks at one point (the subject, or an anchor) from a shell of cameras
around it. A walk through a garden or a room has no such point: the camera travelled a line and
looked along it. This builds the move the capture already contains — one stretch of that line,
with the hand motion averaged out, the horizon levelled and the speed evened — so every frame
stands where a real camera stood, or between two that did.

What it does, in order:
  1. Cuts the placed captures into RUNS the camera could have walked. A run ends at a TEAR: a
     step between two consecutive placed frames several times faster than the capture's own pace.
     Nobody moves like that; it means the solve placed the two stretches without tying one to
     the other, and a path across it would cross a gap that is not there in the world.
     The pace is the capture's own (median speed), not metres per second, because a solve's
     scale is only as good as what fixed it.
  2. Finds the LEGS inside each run: stretches where the camera kept going one way (in, or back
     out), not the standing look-around at either end. Picks the leg that travels furthest
     through frames that were placed, or --captures A:B by capture number.
  3. Averages the hand out of the positions (a Gaussian over `steady_s` seconds of the walk).
  4. Looks `ahead` (where the path goes next, pitched as the real camera was) or `as-shot`
     (where the real camera looked, averaged the same way). Level horizon either way.
  5. Resamples at an even speed with a soft start and stop.

The report says how far the path strays from the line the real cameras walked, and which
stretches of it no frame was placed in. The move file is the same one every other move writes.
"""
import json
import os

import numpy as np

from . import rig as rigmod

TEAR_FACTOR = 4.0          # a step this many times the capture's median pace ends a run
MIN_RUN = 4                # captures


def _gauss(x, t, sigma):
    """Gaussian average of rows x sampled at times t (uneven), at those same times. Ends are
    held by reflecting the data about each end, so the path still starts and stops on a camera."""
    if sigma <= 0 or len(x) < 3:
        return x.copy()
    lo = 2 * x[:1] - x[1:][::-1]
    hi = 2 * x[-1:] - x[:-1][::-1]
    xx = np.concatenate([lo, x, hi])
    tt = np.concatenate([2 * t[0] - t[1:][::-1], t, 2 * t[-1] - t[:-1][::-1]])
    w = np.exp(-0.5 * ((t[:, None] - tt[None, :]) / sigma) ** 2)
    return (w @ xx) / w.sum(1, keepdims=True)


def capture_table(rig_npz, quality_json=None):
    """Left-eye captures in order: number, time (s, or None), centre (mm), R (world -> camera)."""
    G, names, L = rigmod.load(rig_npz)
    num = np.array([int("".join(ch for ch in rigmod.capture_name(names[i]) if ch.isdigit())) for i in L])
    t = None
    if quality_json and os.path.exists(quality_json):
        with open(quality_json) as f:
            q = json.load(f)
        by = {int(f["sel"]): float(f["t_s"]) for f in q.get("frames", []) if f.get("t_s") is not None}
        if all(int(n) in by for n in num):
            t = np.array([by[int(n)] for n in num])
            if np.any(np.diff(t) <= 0):
                t = None
    return {"G": G, "names": names, "L": L, "num": num, "t": t,
            "C": G["C"].astype(np.float64)[L], "R": G["R"].astype(np.float64)[L]}


def find_runs(num, t, C, tear_factor=TEAR_FACTOR):
    """-> (runs, tears, pace). A run is (first, last) indices into the table, inclusive."""
    step = np.linalg.norm(np.diff(C, axis=0), axis=1)
    dt = np.diff(t) if t is not None else np.diff(num).astype(float)
    pace = step / np.maximum(dt, 1e-9)
    typical = float(np.median(pace))
    # a tear is fast for this capture AND a real distance (not a short step over a short interval)
    torn = (pace > tear_factor * typical) & (step > 2.0 * float(np.median(step)))
    tears = [{"after": int(num[i]), "before": int(num[i + 1]), "step_mm": float(step[i]),
              "dt": float(dt[i]), "times_typical": float(pace[i] / typical)} for i in np.flatnonzero(torn)]
    cuts = [0] + [int(i) + 1 for i in np.flatnonzero(torn)] + [len(C)]
    runs = [(a, b - 1) for a, b in zip(cuts[:-1], cuts[1:]) if b - a >= MIN_RUN]
    return runs, tears, typical


STRAIGHT_MIN = 0.5         # a leg's net travel is at least this share of its path


def _clock(num, t, C, a, b):
    """Seconds from the first capture of [a, b]: the picks' own times, or a nominal half second each."""
    if t is not None:
        return t[a:b + 1] - t[a]
    return (num[a:b + 1] - num[a]) * 0.5


STILL = 0.4                # of the capture's pace: slower than this at either end of a leg is standing, and is cut off


def find_legs(num, t, C, runs, pace=None, steady_s=1.5):
    """-> legs, in capture order. A leg is the stretch [i, j] of a run that goes somewhere:
    it maximises (net travel)^2 / (path length) over the hand-smoothed path, so a walk in wins
    over the same walk with the look-around at its start attached, and a there-and-back is two legs."""
    out = []
    if pace is None:
        pace = find_runs(num, t, C)[2]

    def split(a, b):
        if b - a + 1 < MIN_RUN:
            return
        P = _gauss(C[a:b + 1], _clock(num, t, C, a, b), steady_s)
        cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        m = len(P)
        net = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
        path = np.abs(cum[:, None] - cum[None, :])
        i, j = np.triu_indices(m, MIN_RUN - 1)
        ok = (path[i, j] > 0) & (net[i, j] >= STRAIGHT_MIN * path[i, j])
        if not ok.any():
            return
        score = np.where(ok, net[i, j] ** 2 / np.maximum(path[i, j], 1e-9), -1.0)
        k = int(np.argmax(score))
        i, j = int(i[k]), int(j[k])
        # the look-around before setting off and the turn at the far end move the camera a little,
        # so they ride along on the leg's ends; a walk starts where the walking does
        clock = _clock(num, t, C, a, b)
        v = np.diff(cum) / np.maximum(np.diff(clock), 1e-9)
        while j - i + 1 > MIN_RUN and v[i] < STILL * pace:
            i += 1
        while j - i + 1 > MIN_RUN and v[j - 1] < STILL * pace:
            j -= 1
        if net[i, j] < STRAIGHT_MIN * path[i, j] or path[i, j] <= 0:
            return
        score = {k: net[i, j] ** 2 / path[i, j]}
        picks = int(num[a + j] - num[a + i] + 1)
        out.append({"i": a + i, "j": a + j, "first": int(num[a + i]), "last": int(num[a + j]),
                    "captures": j - i + 1, "placed_share": (j - i + 1) / picks,
                    "net_mm": float(net[i, j]), "path_mm": float(path[i, j]),
                    "seconds": float(t[a + j] - t[a + i]) if t is not None else None,
                    "score": float(score[k]) * (j - i + 1) / picks})
        split(a, a + i - 1)
        split(a + j + 1, b)

    for a, b in runs:
        split(a, b)
    return sorted(out, key=lambda g: g["i"])


def _level(fwd, up):
    """Camera-to-world rotation looking along fwd with the horizon level to up (OpenCV axes)."""
    f = fwd / np.linalg.norm(fwd)
    right = np.cross(-up, f)                    # down x forward
    n = np.linalg.norm(right)
    if n < 1e-6:
        raise ValueError("the camera looks straight up or down here; a level frame is undefined")
    right /= n
    down = np.cross(f, right)
    return np.stack([right, down, f], 1)


def _to_polyline(P, line):
    """Distance from each point of P to the polyline through `line`."""
    a, b = line[:-1], line[1:]
    ab = b - a
    den = np.maximum((ab * ab).sum(1), 1e-12)
    out = np.empty(len(P))
    for i, p in enumerate(P):
        s = np.clip(((p - a) * ab).sum(1) / den, 0.0, 1.0)
        out[i] = np.sqrt((((a + ab * s[:, None]) - p) ** 2).sum(1).min())
    return out


def build(rig_npz, out, up, quality_json=None, captures=None, look="ahead", reverse=False,
          steady_s=1.0, frames=None, fps=30.0, look_ahead_s=1.5, ease_s=1.0):
    """Write the move json and return the report. `up` is world up (unit). `captures` is
    (first, last) capture NUMBERS, inclusive. Raises ValueError with a sentence a person can act on."""
    T = capture_table(rig_npz, quality_json)
    num, t, C, R, G = T["num"], T["t"], T["C"], T["R"], T["G"]
    if len(C) < MIN_RUN:
        raise ValueError(f"a walk needs at least {MIN_RUN} placed frames; this solve has {len(C)}")
    up = np.asarray(up, float)
    up = up / np.linalg.norm(up)
    timed = t is not None
    runs, tears, typical = find_runs(num, t, C)
    if not runs:
        raise ValueError("no stretch of four or more consecutive placed frames to walk along")
    legs = find_legs(num, t, C, runs, pace=typical)

    def travel(r):
        return float(np.linalg.norm(np.diff(C[r[0]:r[1] + 1], axis=0), axis=1).sum())

    if captures:
        a, b = (int(x) for x in captures)
        inside = np.flatnonzero((num >= min(a, b)) & (num <= max(a, b)))
        if len(inside) < MIN_RUN:
            raise ValueError(f"captures {a}:{b} hold {len(inside)} placed frames; a walk needs {MIN_RUN}")
        run = (int(inside[0]), int(inside[-1]))
        crossed = [x for x in tears if num[run[0]] <= x["after"] and x["before"] <= num[run[1]]]
        if crossed:
            x = crossed[0]
            raise ValueError(f"captures {a}:{b} cross a tear in the solve between cap{x['after']:03d} and "
                             f"cap{x['before']:03d} ({x['step_mm'] / 1000:.1f} m in one step, "
                             f"{x['times_typical']:.0f} times this capture's pace); pick a range on one side of it")
    else:
        if not legs:
            raise ValueError("the camera never travels in this capture (every stretch doubles back on itself); "
                             "name the frames to follow with --captures A:B")
        best = max(legs, key=lambda g: g["score"])
        run = (best["i"], best["j"])
    i0, i1 = run
    c, r, n = C[i0:i1 + 1], R[i0:i1 + 1], num[i0:i1 + 1]
    tt = _clock(num, t, C, i0, i1).astype(float)
    duration = float(tt[-1])

    pos = _gauss(c, tt, steady_s)
    fwd_real = _gauss(r[:, 2, :], tt, steady_s * 1.5)
    fwd_real /= np.linalg.norm(fwd_real, axis=1, keepdims=True)

    # dense path, by arc length
    k = 20
    td = np.linspace(0.0, duration, (len(c) - 1) * k + 1)
    dense = np.stack([np.interp(td, tt, pos[:, j]) for j in range(3)], 1)
    dense = _gauss(dense, td, min(steady_s, duration / 8) / 2)          # round the corners linear interpolation leaves
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    total = float(s[-1])
    if total < 1e-6:
        raise ValueError("the chosen frames do not move; there is no path to walk")

    n_frames = int(frames) if frames else max(2, int(round(duration * fps)) + 1)
    u = np.linspace(0.0, 1.0, n_frames)
    # even speed with a soft start and stop over ease_s seconds at each end
    e = min(0.45, (ease_s * fps) / max(n_frames - 1, 1))
    if e > 0:
        v = np.clip(np.minimum(u, 1 - u) / e, 0.0, 1.0)
        v = v * v * (3 - 2 * v)
        prog = np.r_[0.0, np.cumsum((v[1:] + v[:-1]) / 2)]
        prog /= prog[-1]
    else:
        prog = u
    if reverse:
        prog = 1.0 - prog
    sf = prog * total
    P = np.stack([np.interp(sf, s, dense[:, j]) for j in range(3)], 1)
    tf = np.interp(sf, s, td)                                           # the capture time each frame stands at

    pitch_real = np.arcsin(np.clip(fwd_real @ up, -1.0, 1.0))
    pitch = np.interp(tf, tt, pitch_real)
    c2w = []
    if look == "as-shot":
        F = np.stack([np.interp(tf, tt, fwd_real[:, j]) for j in range(3)], 1)
    else:
        ahead_mm = total / max(duration, 1e-9) * look_ahead_s
        direction = -1.0 if reverse else 1.0
        sa = sf + direction * ahead_mm
        A = np.stack([np.interp(np.clip(sa, 0, total), s, dense[:, j]) for j in range(3)], 1)
        # past the end of the path, keep going along its last direction
        tan0, tan1 = dense[1] - dense[0], dense[-1] - dense[-2]
        tan0, tan1 = tan0 / np.linalg.norm(tan0), tan1 / np.linalg.norm(tan1)
        A += np.outer(np.clip(sa - total, 0, None), tan1) + np.outer(np.clip(sa, None, 0), tan0)
        H = A - P
        H -= np.outer(H @ up, up)                                       # heading only; pitch comes from the real camera
        bad = np.linalg.norm(H, axis=1) < 1e-6
        if bad.any():
            H[bad] = np.stack([np.interp(tf[bad], tt, fwd_real[:, j]) for j in range(3)], 1)
            H[bad] -= np.outer(H[bad] @ up, up)
        H /= np.linalg.norm(H, axis=1, keepdims=True)
        H = _gauss(H, np.arange(len(H)) / fps, 0.8)                     # no snap where the path kinks
        H /= np.linalg.norm(H, axis=1, keepdims=True)
        F = H * np.cos(pitch)[:, None] + np.outer(np.sin(pitch), up)
    for p, f in zip(P, F):
        M = np.eye(4)
        M[:3, :3] = _level(f, up)
        M[:3, 3] = p / 1000.0
        c2w.append(M)

    Fw = np.array([m[:3, 2] for m in c2w])
    pan = np.degrees(np.arccos(np.clip((Fw[1:] * Fw[:-1]).sum(1), -1.0, 1.0))) * fps
    off = _to_polyline(P, c) if len(c) > 1 else np.zeros(len(P))
    speed = np.linalg.norm(np.diff(P, axis=0), axis=1) * fps
    # stretches of the run no frame was placed in
    gaps = [{"after": int(n[i]), "before": int(n[i + 1]), "missing": int(n[i + 1] - n[i] - 1),
             "length_mm": float(np.linalg.norm(c[i + 1] - c[i]))} for i in range(len(n) - 1) if n[i + 1] - n[i] > 1]
    # how far the scene is from the cameras of this run: the yardstick for "off the path"
    pts = G["pts"].astype(np.float64) if "pts" in G.files else np.zeros((0, 3))
    near = med = None
    if len(pts):
        zs = []
        for j in range(0, len(c), max(1, len(c) // 12)):
            z = (pts - c[j]) @ r[j][2]
            z = z[z > 0]
            if len(z) >= 20:
                zs.append(np.percentile(z, [10, 50]))
        if zs:
            near, med = (float(x) for x in np.median(np.array(zs), axis=0))

    mid = int(T["L"][i0 + (i1 - i0) // 2])
    K = G["K"].astype(np.float64)[mid]
    W, H_ = int(G["w"]), int(G["h"])
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "note": "OpenCV camera convention (x right, y down, z forward), metres; a walk along the captured "
                    f"path, cap{int(n[0]):03d} to cap{int(n[-1]):03d}{' reversed' if reverse else ''}, looking {look}",
            "width": W, "height": H_, "K": K.tolist(), "fps": fps, "frames": [{"c2w": m.tolist()} for m in c2w],
        }, f, indent=1)
    return {
        "frames": n_frames, "fps": fps, "duration_s": (n_frames - 1) / fps, "timed": timed,
        "run": [int(n[0]), int(n[-1])], "run_captures": int(len(n)), "reverse": bool(reverse), "look": look,
        "runs": [{"first": int(num[a]), "last": int(num[b]), "captures": int(b - a + 1), "travel_mm": travel((a, b))}
                 for a, b in runs],
        "legs": [{k: v for k, v in g.items() if k not in ("i", "j")} for g in legs],
        "tears": tears, "gaps": gaps,
        "path_length_mm": total, "capture_path_mm": travel(run), "capture_seconds": duration if timed else None,
        "peak_speed_mm_s": float(speed.max()) if len(speed) else 0.0,
        "mean_speed_mm_s": float(speed.mean()) if len(speed) else 0.0,
        "peak_pan_deg_s": float(pan.max()) if len(pan) else 0.0,
        "off_path_max_mm": float(off.max()), "off_path_median_mm": float(np.median(off)),
        "scene_near_mm": near, "scene_median_mm": med,
        "positions_mm": P, "run_centres_mm": c, "all_centres_mm": C, "points_mm": pts, "up": up,
        "lens_from": T["names"][mid],
    }


def plan_image(rep, out_jpg, size=1400):
    """A map from above: the sparse points, every placed camera, the run and the path. This is
    what a person checks before rendering — that the path goes where they think it does."""
    import cv2
    up = rep["up"]
    C, c, P, pts = rep["all_centres_mm"], rep["run_centres_mm"], rep["positions_mm"], rep["points_mm"]
    flat = C - C.mean(0)
    flat = flat - np.outer(flat @ up, up)
    e1 = np.linalg.svd(flat, full_matrices=False)[2][0]
    e2 = np.cross(up, e1)
    o = C.mean(0)

    def xy(X):
        return np.stack([(X - o) @ e1, (X - o) @ e2], 1)
    cc, rc, pp = xy(C), xy(c), xy(P)
    lo, hi = cc.min(0), cc.max(0)
    pad = 0.25 * float((hi - lo).max())
    lo, hi = lo - pad, hi + pad
    sc = (size - 40) / float((hi - lo).max())
    Wd, Hd = int((hi - lo)[0] * sc) + 40, int((hi - lo)[1] * sc) + 40

    def px(Q):
        q = (Q - lo) * sc + 20
        return np.stack([q[:, 0], Hd - q[:, 1]], 1).astype(np.int32)
    img = np.full((Hd, Wd, 3), 250, np.uint8)
    if len(pts):
        q = xy(pts[:: max(1, len(pts) // 60000)])
        q = px(q[(q >= lo).all(1) & (q <= hi).all(1)])
        img[q[:, 1].clip(0, Hd - 1), q[:, 0].clip(0, Wd - 1)] = (200, 200, 200)
    for q in px(cc):
        cv2.circle(img, (int(q[0]), int(q[1])), 3, (150, 150, 150), -1, cv2.LINE_AA)
    for q in px(rc):
        cv2.circle(img, (int(q[0]), int(q[1])), 4, (60, 60, 60), -1, cv2.LINE_AA)
    cv2.polylines(img, [px(pp)], False, (40, 110, 230), 3, cv2.LINE_AA)
    a, b = px(pp[:1])[0], px(pp[-1:])[0]
    cv2.circle(img, (int(a[0]), int(a[1])), 9, (60, 160, 60), -1, cv2.LINE_AA)
    cv2.circle(img, (int(b[0]), int(b[1])), 9, (40, 40, 200), -1, cv2.LINE_AA)
    bar = 10 ** np.floor(np.log10(max((hi - lo).max() / 4, 1.0)))
    cv2.line(img, (20, Hd - 18), (20 + int(bar * sc), Hd - 18), (0, 0, 0), 2)
    cv2.putText(img, f"{bar / 1000:g} m (solve units)   green: start   red: end   dark dots: the frames this walk follows",
                (20, Hd - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(out_jpg, img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out_jpg
