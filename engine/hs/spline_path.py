#!/usr/bin/env python3
"""
spline_path.py — virtual camera path that follows the trajectory the camera actually
travelled, for captures that are not a single planar arc.

arc_path.py fits one circle in one plane. That works for a deliberate orbit; it fails on
handheld video, where the path may be several loops at different heights. On one 55-capture
clip the circle fit returned a 115 mm radius with a 32 mm mean residual and a nonsensical
353.8 deg "arc" — it was wrapping a small circle around a path that is not one.

This instead fits a Catmull-Rom spline through the left-eye camera centres in capture order,
optionally smoothing out handheld jitter, resamples it by arc length so the motion is even,
and oscillates over a sub-segment. Every point on the path is therefore a place the camera
genuinely was (or within a few mm of one), which keeps the render inside the captured hull —
the only part of novel-view synthesis that is interpolation rather than invention.

Usage:
  python3 spline_path.py rig.npz -o DIR/sweep_path.json [--colmap DIR]
        [--frames 120] [--captures A:B] [--center C --window W]
        [--smooth 3] [--span 0.8] [--fps 30] [--aim object|path]
"""
import argparse, json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rig  # noqa: E402


def read_ply_xyz(path):
    with open(path, "rb") as f:
        hdr = b""
        while b"end_header\n" not in hdr:
            c = f.read(1)
            if not c:
                return None
            hdr += c
        head = hdr.decode("ascii", "replace")
        n = int([l for l in head.split("\n") if l.startswith("element vertex")][0].split()[-1])
        stride = 0
        for p in [l for l in head.split("\n") if l.startswith("property")]:
            stride += 4 if p.split()[1] in ("float", "float32", "int", "uint") else 1
        raw = np.frombuffer(f.read(n * stride), dtype=np.uint8).reshape(n, stride)
        return raw[:, :12].copy().view("<f4").reshape(n, 3).astype(np.float64)


def catmull_rom(P, u):
    """Sample a centripetal-ish Catmull-Rom spline through control points P at parameter u
    (0 .. len(P)-1). Endpoints are duplicated so the curve starts and ends on the data."""
    n = len(P)
    i = np.clip(np.floor(u).astype(int), 0, n - 2)
    t = (u - i)[:, None]
    idx = lambda k: P[np.clip(k, 0, n - 1)]
    p0, p1, p2, p3 = idx(i - 1), idx(i), idx(i + 1), idx(i + 2)
    return 0.5 * ((2 * p1) + (-p0 + p2) * t
                  + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rig")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--colmap", default=None, help="colmap dir, for the object cloud and full-res K")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--captures", default=None, help="capture index range to use, e.g. 10:34")
    ap.add_argument("--center", type=int, default=None, help="centre capture index of the window")
    ap.add_argument("--window", type=int, default=None, help="number of captures in the window")
    ap.add_argument("--smooth", type=int, default=3, help="moving-average window over camera centres (1 = off)")
    ap.add_argument("--span", type=float, default=0.8, help="fraction of the selected segment to sweep")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--aim", choices=["object", "path"], default="object",
                    help="object: every frame looks at the object centre. path: keep the captured orientation, interpolated.")
    ap.add_argument("--aim-point", default=None,
                    help="x,y,z in mm — aim here instead of finding the subject automatically")
    ap.add_argument("--aim-median", action="store_true",
                    help="use median(points) as the subject, the old behaviour; correct only "
                         "for a masked object capture, wrong for an unmasked room")
    a = ap.parse_args()

    G = np.load(a.rig, allow_pickle=True)
    names = [str(x) for x in G["names"]]
    C = G["C"].astype(np.float64)
    R = G["R"].astype(np.float64)
    Kw = G["K"].astype(np.float64)
    L = rig.left_indices(G, names)
    CL = C[L]
    nC = len(CL)
    if nC < 4:
        sys.exit("need at least 4 captures for a spline")

    # ---- window selection
    if a.captures:
        lo, hi = (int(x) for x in a.captures.split(":"))
    elif a.center is not None or a.window is not None:
        w = a.window or max(8, nC // 3)
        c = a.center if a.center is not None else nC // 2
        lo, hi = max(0, c - w // 2), min(nC, c + w // 2)
    else:
        lo, hi = 0, nC
    lo, hi = max(0, lo), min(nC, hi)
    if hi - lo < 4:
        sys.exit(f"window {lo}:{hi} has fewer than 4 captures")
    seg = CL[lo:hi]
    print(f"{nC} captures; using {lo}:{hi} ({len(seg)} captures)")

    # ---- smooth out handheld jitter, and report how far that moves the path off the data
    if a.smooth > 1:
        k = a.smooth
        pad = np.vstack([seg[:1]] * (k // 2) + [seg] + [seg[-1:]] * (k // 2))
        sm = np.stack([np.convolve(pad[:, d], np.ones(k) / k, mode="valid") for d in range(3)], 1)
        dev = np.linalg.norm(sm - seg, axis=1)
        print(f"smoothing window {k}: path moves {dev.mean():.1f} mm from the camera centres on average, {dev.max():.1f} mm max")
        seg = sm

    # ---- resample by arc length so the virtual camera moves at constant speed
    u = np.linspace(0, len(seg) - 1, 2000)
    dense = catmull_rom(seg, u)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    total = d[-1]
    print(f"path length {total:.0f} mm, straight-line span {np.linalg.norm(seg[-1]-seg[0]):.0f} mm")

    # ---- object centre
    #
    # Default is the densest cell of a coarse grid over points3D; --aim-median uses the
    # median instead. On the captures checked so far (rig5, rig6) BOTH land on the subject
    # (2026-09-13: density -> left pauldron, median -> chest plate, 27 mm apart), and the
    # median is the better-centred of the two. The earlier "median aims into the room"
    # diagnosis was wrong — that failure was a path from a different reconstruction being
    # rendered against the wrong .ply. Neither heuristic is trusted blind: the projection
    # check below is what decides, and it must be read every time.
    P = None
    if a.colmap:
        p = os.path.join(a.colmap, "sparse", "0", "points3D.ply")
        if not os.path.exists(p):
            p = os.path.join(a.colmap, "sparse", "points3D.ply")
        if os.path.exists(p):
            Q = read_ply_xyz(p)
            if Q is not None and len(Q):
                P = Q * 1000.0
    if P is None:
        P = G["pts"].astype(np.float64)

    if a.aim_point:
        obj = np.array([float(v) for v in a.aim_point.split(",")])
        print(f"object centre (mm), given: {np.round(obj,1)}")
    elif a.aim_median:
        obj = np.median(P, axis=0)
        print(f"object centre (mm), median of {len(P)} points: {np.round(obj,1)}")
    else:
        # note the local names: lo/hi/k/d are all live variables in this function
        # (capture window, smoothing width, arc-length table) — do not reuse them here
        plo, phi = np.percentile(P, [1, 99], axis=0)
        Hgrid, edges = np.histogramdd(P, bins=32, range=list(zip(plo, phi)))
        cell = np.unravel_index(np.argmax(Hgrid), Hgrid.shape)
        obj = np.array([(edges[ax][cell[ax]] + edges[ax][cell[ax] + 1]) / 2 for ax in range(3)])
        med = np.median(P, axis=0)
        print(f"object centre (mm), densest cell of {len(P)} points: {np.round(obj,1)}"
              f"   [median would be {np.round(med,1)}, {np.linalg.norm(obj-med):.0f} mm away]")

    # ---- sanity-check the aim point by projecting it into a real view.
    # This check exists because three different aim points were shipped before anyone looked
    # at where they actually land. The opacity-weighted centre of a trained splat model put
    # the aim 1936 px across a 1913 px image — off frame — while the SfM median was on the
    # subject's chest. A number that looks plausible in world coordinates tells you nothing.
    vmid = int(L[lo + (hi - lo) // 2])
    Xc = R[vmid] @ obj + G["t"].astype(np.float64)[vmid]
    uv = Kw[vmid] @ Xc
    W0c, H0c = int(G["w"]), int(G["h"])
    if Xc[2] <= 0:
        print(f"  !! aim point is BEHIND view {vmid} ({names[vmid]}) — the path will not "
              f"look at it. Pass --aim-point, or --aim-median.")
    else:
        uv = uv[:2] / uv[2]
        inside = 0 <= uv[0] < W0c and 0 <= uv[1] < H0c
        print(f"  aim check: projects to ({uv[0]:.0f}, {uv[1]:.0f}) in view {vmid} "
              f"({names[vmid]}, {W0c}x{H0c}) at {Xc[2]:.0f} mm — "
              f"{'in frame' if inside else 'OUT OF FRAME, the path is aimed at nothing'}")
        if not inside:
            print("     fix this before rendering: --aim-point x,y,z in mm, or --aim-median")

    # ---- camera 'down' in world (OpenCV: +y is down), averaged over the window
    down = np.mean([R[v].T @ np.array([0.0, 1.0, 0.0]) for v in L[lo:hi]], axis=0)
    down /= np.linalg.norm(down)

    # ---- intrinsics from the capture nearest the middle of the window
    icen = int(L[lo + (hi - lo) // 2])
    if a.colmap and os.path.exists(os.path.join(a.colmap, "sparse", "0", "cameras.txt")):
        rows = [l.split() for l in open(os.path.join(a.colmap, "sparse", "0", "cameras.txt")) if not l.startswith("#")]
        r = rows[icen]
        W0, H0 = int(r[2]), int(r[3])
        K = [[float(r[4]), 0.0, float(r[6])], [0.0, float(r[5]), float(r[7])], [0.0, 0.0, 1.0]]
    else:
        W0, H0 = int(G["w"]), int(G["h"])
        K = Kw[icen].tolist()
    print(f"intrinsics from view {icen} ({names[icen]}) at {W0}x{H0}")

    # ---- sweep: sinusoidal traverse of the middle `span` of the segment, looping cleanly
    mid, half = total / 2.0, total / 2.0 * a.span
    frames = []
    dists = []
    for i in range(a.frames):
        s = mid + half * math.sin(2 * math.pi * i / a.frames)
        pos = np.array([np.interp(s, d, dense[:, k]) for k in range(3)])
        if a.aim == "object":
            fwd = obj - pos
        else:
            ahead = np.array([np.interp(min(s + 5.0, total), d, dense[:, k]) for k in range(3)])
            fwd = obj - pos if np.linalg.norm(ahead - pos) < 1e-6 else (obj - pos)
        fwd /= np.linalg.norm(fwd)
        right = np.cross(down, fwd)
        nr = np.linalg.norm(right)
        if nr < 1e-6:
            sys.exit("degenerate up vector")
        right /= nr
        dn = np.cross(fwd, right)
        c2w = np.eye(4)
        c2w[:3, 0] = right; c2w[:3, 1] = dn; c2w[:3, 2] = fwd
        c2w[:3, 3] = pos / 1000.0
        frames.append({"c2w": c2w.tolist()})
        dists.append(np.linalg.norm(pos - obj))

    # ---- how far the virtual path strays from the real cameras (should be ~0: interpolation, not extrapolation)
    vp = np.array([np.array(f["c2w"])[:3, 3] * 1000.0 for f in frames])
    near = np.min(np.linalg.norm(vp[:, None, :] - CL[None, :, :], axis=2), axis=1)
    print(f"virtual cameras sit {near.min():.0f}-{near.max():.0f} mm from the nearest real camera (median {np.median(near):.0f})")
    print(f"virtual camera distance to object: {min(dists):.0f}-{max(dists):.0f} mm")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    json.dump({
        "note": "OpenCV camera convention (x right, y down, z forward), metres; spline through the "
                "captured camera centres, resampled by arc length; K is the mid-window camera's pinhole",
        "width": W0, "height": H0, "K": K, "fps": a.fps, "frames": frames,
    }, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: {a.frames} frames sweeping {2*half:.0f} mm of a {total:.0f} mm path")


if __name__ == "__main__":
    main()
