#!/usr/bin/env python3
"""
key_path.py — one-way keyframed camera move through chosen real camera positions.

spline_path.py oscillates over a contiguous window of captures. This instead takes an
ordered list of capture indices as keyframes, fits a Catmull-Rom spline through those camera
centres, resamples by arc length, and eases in and out (smoothstep) so the move starts and
ends at rest. Keyframes need not be contiguous in capture order — that is the point: a
handheld clip visits the same region several times at different heights, and a move can
stitch those visits together. Between keyframes that are far apart the path leaves the
captured hull; the report prints how far the virtual camera strays from the nearest real
camera so that can be judged, and the aim point is projected into a real view exactly as
spline_path.py does.

Usage:
  python3 key_path.py rig.npz -o DIR/move.json --keys 52,54,56,58,60,62,64,14,16,18,20,22
        [--frames 300] [--fps 30] [--aim-median | --aim-point x,y,z] [--ease 1.0]
        [--hold 0.0]

--ease  0 = constant speed, 1 = full smoothstep ease in/out (default 1).
--hold  seconds held still at each end, inside the frame count (default 0).
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spline_path import catmull_rom  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rig")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--keys", required=True, help="ordered capture indices, e.g. 52,54,56,64,14,16,22")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--aim-point", default=None, help="x,y,z in mm")
    ap.add_argument("--aim-median", action="store_true", help="median of points3D (default: densest cell)")
    ap.add_argument("--ease", type=float, default=1.0)
    ap.add_argument("--hold", type=float, default=0.0)
    ap.add_argument("--eye", choices=["L", "R", "mid"], default="L",
                    help="which eye's centre to key on (mid = midpoint of the pair)")
    a = ap.parse_args()

    G = np.load(a.rig, allow_pickle=True)
    names = [str(x) for x in G["names"]]
    C = G["C"].astype(np.float64)
    R = G["R"].astype(np.float64)
    Kw = G["K"].astype(np.float64)
    L = np.arange(0, len(names), 2)
    nC = len(L)

    keys = [int(k) for k in a.keys.split(",")]
    bad = [k for k in keys if not 0 <= k < nC]
    if bad:
        sys.exit(f"keys out of range 0..{nC-1}: {bad}")
    if len(keys) < 2:
        sys.exit("need at least 2 keys")

    def centre(k):
        if a.eye == "L":
            return C[L[k]]
        if a.eye == "R":
            return C[L[k] + 1]
        return 0.5 * (C[L[k]] + C[L[k] + 1])

    P = np.array([centre(k) for k in keys])
    seglen = np.linalg.norm(np.diff(P, axis=0), axis=1)
    print(f"{nC} captures; {len(keys)} keys: {keys}")
    print("key-to-key distances (mm): " + " ".join(f"{d:.0f}" for d in seglen))

    # ---- spline through the keys, resampled by arc length
    u = np.linspace(0, len(P) - 1, 4000)
    dense = catmull_rom(P, u)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    total = d[-1]
    print(f"path length {total:.0f} mm, straight-line span {np.linalg.norm(P[-1]-P[0]):.0f} mm")

    # ---- aim point
    Pts = G["pts"].astype(np.float64)
    if a.aim_point:
        obj = np.array([float(v) for v in a.aim_point.split(",")])
        print(f"object centre (mm), given: {np.round(obj,1)}")
    elif a.aim_median:
        obj = np.median(Pts, axis=0)
        print(f"object centre (mm), median of {len(Pts)} points: {np.round(obj,1)}")
    else:
        plo, phi = np.percentile(Pts, [1, 99], axis=0)
        Hg, edges = np.histogramdd(Pts, bins=32, range=list(zip(plo, phi)))
        cell = np.unravel_index(np.argmax(Hg), Hg.shape)
        obj = np.array([(edges[ax][cell[ax]] + edges[ax][cell[ax] + 1]) / 2 for ax in range(3)])
        print(f"object centre (mm), densest cell of {len(Pts)} points: {np.round(obj,1)}")

    # ---- aim check: project into the real view at the middle key
    vmid = int(L[keys[len(keys) // 2]])
    Xc = R[vmid] @ obj + G["t"].astype(np.float64)[vmid]
    W0, H0 = int(G["w"]), int(G["h"])
    if Xc[2] <= 0:
        print(f"  !! aim point is BEHIND view {vmid} ({names[vmid]})")
    else:
        uv = Kw[vmid] @ Xc
        uv = uv[:2] / uv[2]
        inside = 0 <= uv[0] < W0 and 0 <= uv[1] < H0
        print(f"  aim check: projects to ({uv[0]:.0f}, {uv[1]:.0f}) in view {vmid} ({names[vmid]}, {W0}x{H0}) "
              f"at {Xc[2]:.0f} mm — {'in frame' if inside else 'OUT OF FRAME, the path is aimed at nothing'}")
        if not inside:
            print("     fix this before rendering: --aim-point x,y,z in mm, or --aim-median")

    # ---- camera 'down' in world, averaged over the keyed views
    down = np.mean([R[L[k]].T @ np.array([0.0, 1.0, 0.0]) for k in keys], axis=0)
    down /= np.linalg.norm(down)

    # ---- intrinsics from the middle key
    K = Kw[vmid].tolist()
    print(f"intrinsics from view {vmid} ({names[vmid]}) at {W0}x{H0}")

    # ---- timing: hold, ease, hold
    n = a.frames
    hold = int(round(a.hold * a.fps))
    move = max(2, n - 2 * hold)
    s_lin = np.clip((np.arange(n) - hold) / (move - 1), 0, 1)
    s_ease = s_lin * s_lin * (3 - 2 * s_lin)
    s = (1 - a.ease) * s_lin + a.ease * s_ease
    step = np.diff(s) * total * a.fps
    print(f"{n} frames at {a.fps:g} fps = {n/a.fps:.1f} s; hold {hold} frames each end; "
          f"peak speed {step.max():.0f} mm/s, mean {step.mean():.0f} mm/s")

    frames, near, dists = [], [], []
    CL = C[L]
    for si in s:
        pos = np.array([np.interp(si * total, d, dense[:, k]) for k in range(3)])
        fwd = obj - pos
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
        near.append(np.min(np.linalg.norm(CL - pos, axis=1)))
        dists.append(np.linalg.norm(pos - obj))
    near = np.array(near)
    print(f"virtual cameras sit {near.min():.0f}-{near.max():.0f} mm from the nearest real camera "
          f"(median {np.median(near):.0f}); worst at frame {int(np.argmax(near))}")
    print(f"virtual camera distance to object: {min(dists):.0f}-{max(dists):.0f} mm")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    json.dump({
        "note": "OpenCV camera convention (x right, y down, z forward), metres; keyframed one-way "
                "move through real camera centres, eased; K is the middle key's pinhole",
        "width": W0, "height": H0, "K": K, "fps": a.fps, "frames": frames,
    }, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: {n} frames, keys {keys}")


if __name__ == "__main__":
    main()
