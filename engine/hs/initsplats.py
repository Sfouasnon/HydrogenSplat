"""Initial splats from points: a Brush-layout Gaussian PLY for `hs train --init lidar`.

Brush takes its initial splats from whatever .ply sits in the dataset folder (``init.ply`` wins,
stages/train.py). Normally that is the SfM cloud; `hs scale --lidar ... --init-points` writes the
LiDAR scan instead (scale/lidar_init.ply), in the training set's units (COLMAP metres = rig.npz
mm / 1000), plus the solve's own sparse points where the scan does not reach, so the trees and
the far lawn keep their usual start.

The layout is Brush's own export, property for property (a real header, 2026-09):

    x y z scale_0 scale_1 scale_2 opacity rot_0 rot_1 rot_2 rot_3 f_dc_0 f_dc_1 f_dc_2
    f_rest_0 ... f_rest_44            all float, binary little-endian, SH degree 3

with the comments ``Exported from Brush`` and ``SH degree: 3`` Brush writes (kept so its reader
sees the file it expects; a third comment says who really wrote it). Per point: f_dc =
(rgb / 255 - 0.5) / SH_C0, f_rest = 0, opacity = logit(opacity), rot = (1, 0, 0, 0), and one
isotropic scale = log(mean distance to its 3 nearest neighbours), clamped.
"""
import os

import numpy as np

SH_C0 = 0.28209479177387814
SH_REST = 45                                  # degree 3: 3 x 15 coefficients
BRUSH_PROPS = (["x", "y", "z", "scale_0", "scale_1", "scale_2", "opacity", "rot_0", "rot_1", "rot_2", "rot_3",
                "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(SH_REST)])
BRUSH_COMMENTS = ("Exported from Brush", "SH degree: 3")
DTYPE = np.dtype([(p, "<f4") for p in BRUSH_PROPS])


def knn_scale(xyz, k=3, lo=0.002, hi=0.1):
    """Per point: the mean distance to its k nearest other points, clamped to [lo, hi] (units of xyz)."""
    from .lidar import NN
    P = np.asarray(xyz, np.float64)
    if len(P) < 2:
        return np.full(len(P), lo)
    d, _ = NN(P).knn(P, min(k + 1, len(P)))
    return np.clip(d[:, 1:].mean(axis=1), lo, hi)


def splats(xyz, rgb=None, opacity=0.3, scale=None):
    """Rows in Brush's layout. xyz (n, 3) training-set units; rgb (n, 3) uint8 or None (mid-grey);
    scale (n,) linear, or None for knn_scale. -> structured array of DTYPE."""
    xyz = np.asarray(xyz, np.float64)
    n = len(xyz)
    out = np.zeros(n, DTYPE)
    out["x"], out["y"], out["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    sc = knn_scale(xyz) if scale is None else np.asarray(scale, np.float64)
    ls = np.log(np.maximum(sc, 1e-12))
    out["scale_0"] = out["scale_1"] = out["scale_2"] = ls
    op = float(np.clip(opacity, 1e-4, 1 - 1e-4))
    out["opacity"] = np.log(op / (1 - op))
    out["rot_0"] = 1.0
    if rgb is not None:
        dc = (np.asarray(rgb, np.float64) / 255.0 - 0.5) / SH_C0
        out["f_dc_0"], out["f_dc_1"], out["f_dc_2"] = dc[:, 0], dc[:, 1], dc[:, 2]
    return out


def write(path, rows, comments=()):
    """A binary little-endian PLY of `rows` (DTYPE) with Brush's header, atomically."""
    rows = np.asarray(rows, DTYPE)
    head = ["ply", "format binary_little_endian 1.0"] + [f"comment {c}" for c in BRUSH_COMMENTS + tuple(comments)]
    head += [f"element vertex {len(rows)}"] + [f"property float {p}" for p in BRUSH_PROPS] + ["end_header"]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".partial"
    with open(tmp, "wb") as f:
        f.write(("\n".join(head) + "\n").encode("ascii"))
        f.write(np.ascontiguousarray(rows).tobytes())
    os.replace(tmp, path)
    return path
