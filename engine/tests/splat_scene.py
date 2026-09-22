"""A synthetic splat scene shared by test_splatweights / test_split / test_prune_score.

A "card": ~300 opaque white splats on a plane through the origin, facing the cameras. A "room":
~300 splats on a wall 250 mm behind it, wider than any view's share of it, in muted colours.
Twelve pinhole cameras on a spherical cap 400 mm out, all looking at the origin. The photographs
are the scene rendered by splatweights at cell 1 (i.e. per pixel), the masks are the card alone
rendered the same way and thresholded -- so the ground truth is exact, and anything a test adds
to the PLY afterwards (a floater) is by construction absent from the photographs.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from hs import splatweights as sw  # noqa: E402
from hs.project import Project, now_iso  # noqa: E402

W, H, FX = 320, 240, 300.0
PROPS = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
         + [f"f_rest_{i}" for i in range(9)]
         + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def dc_of(rgb):
    return (np.asarray(rgb, float) - 0.5) / sw.SH_C0


def look_at(C):
    f = -C / np.linalg.norm(C)
    down = np.array([0.0, 1.0, 0.0])
    y = down - down.dot(f) * f
    y /= np.linalg.norm(y)
    x = np.cross(y, f)
    return np.stack([x, y, f])


def cameras(n=12, dist=400.0):
    """World->camera (R, t) in mm, cameras at z<0 looking toward +z at the origin."""
    out = []
    for i in range(n):
        az = np.radians(-40 + 80 * (i % 6) / 5.0)
        el = np.radians(-18 if i < 6 else 18)
        C = dist * np.array([np.sin(az) * np.cos(el), np.sin(el), -np.cos(az) * np.cos(el)])
        R = look_at(C)
        out.append((R, -R @ C))
    return out


def splat_rows(xyz_mm, scale_mm, opacity, rgb):
    """Rows in the 3DGS PLY layout: metres, log scales, logit opacity, identity rotation."""
    n = len(xyz_mm)
    arr = np.zeros((n, len(PROPS)), "<f4")
    col = {p: i for i, p in enumerate(PROPS)}
    arr[:, col["x"]:col["z"] + 1] = np.asarray(xyz_mm) / 1000.0
    arr[:, col["f_dc_0"]:col["f_dc_2"] + 1] = dc_of(rgb)
    arr[:, col["opacity"]] = logit(np.broadcast_to(opacity, (n,)))
    arr[:, col["scale_0"]:col["scale_2"] + 1] = np.log(np.asarray(scale_mm, float) / 1000.0)
    arr[:, col["rot_0"]] = 1.0
    return arr


def write_ply(path, arr):
    hdr = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % len(arr)
           + "".join(f"property float {x}\n" for x in PROPS) + "end_header\n")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(hdr.encode())
        f.write(np.ascontiguousarray(arr, "<f4").tobytes())


def card_and_room(seed=0):
    """-> (rows, labels) with labels 1 = card, 0 = room."""
    rng = np.random.default_rng(seed)
    g = np.linspace(-40, 40, 17)
    cx, cy = np.meshgrid(g, g)
    card = np.stack([cx.ravel(), cy.ravel(), np.zeros(cx.size)], 1)
    card[:, :2] += rng.uniform(-0.8, 0.8, (len(card), 2))
    card_rows = splat_rows(card, np.tile([3.5, 3.5, 0.3], (len(card), 1)), 0.99,
                           np.tile([0.95, 0.93, 0.9], (len(card), 1)))
    gx, gy = np.meshgrid(np.linspace(-300, 300, 20), np.linspace(-220, 220, 15))
    room = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, 250.0)], 1)
    room[:, :2] += rng.uniform(-4, 4, (len(room), 2))
    room_rgb = rng.uniform(0.15, 0.55, (len(room), 3))
    room_rows = splat_rows(room, np.tile([16.0, 16.0, 1.0], (len(room), 1)), 0.97, room_rgb)
    rows = np.concatenate([card_rows, room_rows])
    labels = np.concatenate([np.ones(len(card_rows), int), np.zeros(len(room_rows), int)])
    return rows, labels


def render(sp, R, t, only=None, cell=1):
    K = np.array([[FX, 0, W / 2], [0, FX, H / 2], [0, 0, 1.0]])
    return sw.view_weights(sp, K, R, t, (W, H), cell=cell, only=only)


def build_project(root, rows, labels, truth_rows=None, masks=True, stereo=False):
    """A project with solve and train done, train/dataset/{images,masks}/L and rig.npz, and the
    model at train/exports/export_01000.ply. The photographs show `truth_rows` (default: rows)."""
    import cv2
    pj = Project(root, create=True)
    for s in ("ingest", "select", "solve"):
        pj.m["stages"][s] = {"status": "done", "finished": now_iso(), "checks": [], "metrics": {}}
    ply = pj.path("train", "exports", "export_01000.ply")
    write_ply(ply, rows)
    truth = pj.path("scratch_truth.ply")
    write_ply(truth, rows if truth_rows is None else truth_rows)
    tsp = sw.Splats(truth)
    card_only = np.zeros(tsp.n, bool)
    card_only[:int(labels.sum())] = True      # the card rows come first in every scene built here
    cams = cameras()
    names, Ks, Rs, ts = [], [], [], []
    K = np.array([[FX, 0, W / 2], [0, FX, H / 2], [0, 0, 1.0]])
    for i, (R, t) in enumerate(cams):
        nm = f"cap{i:03d}"
        names.append(nm + "_L")
        Ks.append(K); Rs.append(R); ts.append(t)
        vw = render(tsp, R, t)
        img = np.clip(vw.rgb, 0, 1)                          # black background
        os.makedirs(pj.path("train", "dataset", "images", "L"), exist_ok=True)
        cv2.imwrite(pj.path("train", "dataset", "images", "L", nm + ".png"),
                    np.round(img[:, :, ::-1] * 255).astype(np.uint8))
        if masks:
            m = render(tsp, R, t, only=card_only)
            os.makedirs(pj.path("train", "dataset", "masks", "L"), exist_ok=True)
            cv2.imwrite(pj.path("train", "dataset", "masks", "L", nm + ".png"),
                        np.where(1.0 - m.T >= 0.5, 255, 0).astype(np.uint8))
    os.remove(truth)
    np.savez(pj.rig_npz, names=np.array(names), K=np.array(Ks), R=np.array(Rs), t=np.array(ts),
             C=np.array([-R.T @ t for R, t in cams]), wh=np.array([[W, H]] * len(names)),
             pts=np.zeros((10, 3)), s_mm=1.0, stereo=stereo)
    pj.m["stages"]["train"] = {"status": "done", "finished": now_iso(), "checks": [],
                               "metrics": {"final_export": pj.rel(ply),
                                           "dataset_fingerprint": {"excluded_views": []}}}
    pj.m["stages"]["masks"] = {"status": "done", "metrics": {"method": "geometry"}}
    pj.save()
    return pj, ply
