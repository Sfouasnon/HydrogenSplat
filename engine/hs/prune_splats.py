#!/usr/bin/env python3
"""
prune_splats.py — cut background, near-invisible and oversized splats out of a Brush .ply.

On an unmasked room capture most of the model is not the subject. On Mando_Capture4.ply,
199,181 splats, only about a third sat near the subject and survived basic quality filters;
the rest was room. Background splats are not free: they are sorted and blended every frame,
and the large ones sweep across the image during a parallax move, which is what reads as
swimming.

Criteria, in the order they are applied:
  opacity     sigmoid(opacity) below the threshold contributes almost nothing but still costs
              a sort slot. 37% of that file was under 0.1.
  anisotropy  max_scale/min_scale. Needle splats are the ones that streak.
  max scale   absolute world-space size. That file's largest was 2.72 units across, sitting
              7.6 units from a subject 0.34 units wide.
  radius      distance from the subject centre, found by an opacity-weighted density grid
              unless you give --center.

--report-only prints the ladder without writing, so you can pick thresholds by looking at
how much opacity mass each step actually costs.

Usage:
  python3 prune_splats.py in.ply out.ply [--radius 0.5] [--min-opacity 0.05]
                         [--max-aniso 50] [--max-scale 0.2] [--center x,y,z] [--report-only]
"""
import argparse, sys
import numpy as np


def read_ply(path):
    raw = open(path, "rb").read()
    k = raw.find(b"end_header\n")
    if k < 0:
        sys.exit("not a binary ply with an end_header line")
    head = raw[:k].decode("ascii", "replace")
    body = raw[k + len(b"end_header\n"):]
    if "binary_little_endian" not in head:
        sys.exit("only binary_little_endian is supported")
    props = [l.split()[-1] for l in head.split("\n") if l.startswith("property float")]
    n = int([l for l in head.split("\n") if l.startswith("element vertex")][0].split()[-1])
    if len(props) * 4 * n != len(body):
        sys.exit(f"expected {len(props)} float properties x {n} vertices, "
                 f"got {len(body)} bytes — non-float properties are not supported")
    return head, np.frombuffer(body, dtype="<f4").reshape(n, len(props)), props, n


def write_ply(path, head, arr, n_keep):
    out = []
    for l in head.split("\n"):
        out.append(f"element vertex {n_keep}" if l.startswith("element vertex") else l)
    with open(path, "wb") as f:
        f.write(("\n".join(out) + "\nend_header\n").encode("ascii"))
        f.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())


def find_center(xyz, A):
    """Opacity-weighted densest cell of a coarse grid. Weighting by opacity matters —
    unweighted, a haze of near-transparent background splats can outvote the subject."""
    lo, hi = np.percentile(xyz, [0.5, 99.5], axis=0)
    H, edges = np.histogramdd(xyz, bins=40, range=list(zip(lo, hi)), weights=A)
    j = np.unravel_index(np.argmax(H), H.shape)
    return np.array([(edges[k][j[k]] + edges[k][j[k] + 1]) / 2 for k in range(3)])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inp")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--center", default=None, help="x,y,z — default: opacity-weighted densest cell")
    ap.add_argument("--radius", type=float, default=None, help="keep within this of the centre")
    ap.add_argument("--min-opacity", type=float, default=0.05)
    ap.add_argument("--max-aniso", type=float, default=50.0)
    ap.add_argument("--max-scale", type=float, default=None)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if not a.report_only and not a.out:
        sys.exit("give an output path, or pass --report-only")

    head, arr, props, n = read_ply(a.inp)
    ix = {p: i for i, p in enumerate(props)}
    xyz = arr[:, [ix["x"], ix["y"], ix["z"]]].astype(np.float64)
    S = np.exp(arr[:, [ix["scale_0"], ix["scale_1"], ix["scale_2"]]].astype(np.float64))
    A = 1.0 / (1.0 + np.exp(-arr[:, ix["opacity"]].astype(np.float64)))
    mass = A.sum()

    ctr = np.array([float(v) for v in a.center.split(",")]) if a.center else find_center(xyz, A)
    d = np.linalg.norm(xyz - ctr, axis=1)
    aniso = S.max(1) / np.maximum(S.min(1), 1e-9)
    print(f"{n} splats, subject centre {np.round(ctr, 4)}"
          f"{'' if a.center else ' (opacity-weighted densest cell)'}")
    # DO NOT feed this to spline_path --aim-point without checking it first. On the rig5 room
    # model the opacity-weighted densest cell landed at [95.6 -50.4 164.2] mm, which projects
    # to (1936, 72) in a 1913-wide training image — off frame, and nowhere near the subject.
    # The trained model piles opacity into bright background (a window, a lit wall) as readily
    # as into the subject. The SfM cloud's median was correct on that capture.
    # Verify by projecting the candidate into a training view before building a path with it.
    print(f"  densest-cell centre in mm: {ctr[0]*1000:.1f},{ctr[1]*1000:.1f},{ctr[2]*1000:.1f}"
          f"   (verify by projection before using as --aim-point)")

    def step(keep, label):
        print(f"  {label:34s} {keep.sum():8d} splats ({100*keep.mean():5.1f}%)   "
              f"opacity mass kept {100*A[keep].sum()/mass:5.1f}%")
        return keep

    keep = step(np.ones(n, bool), "all")
    if a.min_opacity:
        keep = step(keep & (A > a.min_opacity), f"opacity > {a.min_opacity}")
    if a.max_aniso:
        keep = step(keep & (aniso < a.max_aniso), f"anisotropy < {a.max_aniso:g}")
    if a.max_scale:
        keep = step(keep & (S.max(1) < a.max_scale), f"max scale < {a.max_scale:g}")
    if a.radius:
        keep = step(keep & (d < a.radius), f"within {a.radius:g} of the centre")

    if a.report_only:
        print("\n  radius ladder (with the quality filters already applied):")
        q = np.ones(n, bool)
        if a.min_opacity: q &= A > a.min_opacity
        if a.max_aniso:   q &= aniso < a.max_aniso
        if a.max_scale:   q &= S.max(1) < a.max_scale
        for r in (0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0):
            m = q & (d < r)
            ext = np.ptp(xyz[m], axis=0) if m.sum() else np.zeros(3)
            print(f"    r < {r:<5g} {m.sum():8d} splats ({100*m.mean():5.1f}%)  "
                  f"mass {100*A[m].sum()/mass:5.1f}%  extent {np.round(ext, 3)}")
        return

    sel = arr[keep]
    write_ply(a.out, head, sel, int(keep.sum()))
    ext = np.ptp(xyz[keep], axis=0)
    print(f"\nwrote {a.out}: {keep.sum()} splats ({100*keep.mean():.1f}% of the input), "
          f"extent {np.round(ext, 3)}")


if __name__ == "__main__":
    main()
