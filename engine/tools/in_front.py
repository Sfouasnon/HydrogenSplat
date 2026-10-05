"""How much of a move's picture is drawn by splats floating in front of the surface.

    python3 engine/tools/in_front.py PROJECT --ply A.ply [--ply B.ply] [--move walk] [--frames 60,130,260,390,470]
                                     [--views cap191_L,cap040_L] [--out DIR]

For each frame of the move (or training view) and each model, the numpy forward pass
(hs/splatweights.py, 8 px cells, base colour only) gives every cell its splats front to back with
their blend weights. "The surface" of a cell is the depth at which half its weight has been
reached; "in front" is the weight drawn by splats at less than half that depth. The table prints,
per frame: the mean in-front weight over the picture, the share of cells where it is over 30 %,
the picture's mean opacity, and how much of the in-front weight comes from splats within
0.3 m / 1 m of the line the real camera walked (hs/clearpath.py).

It overstates floaters where the scene itself is layered (a near plant's ragged edge over a far
hedge is honestly in front), so read differences between models on the same frames, not the
level. It is a coarse preview, not a Brush render: no view-dependent colour, 239 cells across.

The view is rendered in vertical strips so an 8-million-splat model fits in 3 GB.
First used 2026-10-05 on the garden walk: 16.8 % in front 2 s into the walk as trained, 6.9 %
without the splats within 0.3 m of the walked line.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hs import clearpath, splatweights as SW  # noqa: E402
from hs.stages.merge import read_header  # noqa: E402

CELL, NSTRIP, CHUNK = 8, 10, 500_000


class Model:
    """The columns the measurement needs, read in chunks; rows are fetched per strip from the file."""

    def __init__(self, path, line):
        self.path = path
        head, self.n, props, off = read_header(path)
        names = [p for p, _t in props]
        if any(t not in ("float", "float32") for _p, t in props):
            raise SystemExit(f"{path}: not an all-float Gaussian PLY")
        self.mm = np.memmap(path, dtype="<f4", mode="r", offset=off, shape=(self.n, len(names)))
        self.col = [names.index(k) for k in ("x", "y", "z", "scale_0", "scale_1", "scale_2", "opacity",
                                             "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2")]
        self.xyz = np.empty((self.n, 3), np.float32)
        self.op = np.empty(self.n, np.float32)
        self.smax = np.empty(self.n, np.float32)
        self.dline = np.empty(self.n, np.float32)
        a, b = line
        for s in range(0, self.n, CHUNK):
            blk = np.asarray(self.mm[s:s + CHUNK, :])[:, self.col[:7]]
            self.xyz[s:s + CHUNK] = blk[:, :3]
            self.smax[s:s + CHUNK] = np.exp(blk[:, 3:6]).max(1)
            self.op[s:s + CHUNK] = 1.0 / (1.0 + np.exp(-blk[:, 6]))
            self.dline[s:s + CHUNK] = clearpath.distance_to_line(blk[:, :3], a / 1000.0, b / 1000.0)

    def mini(self, idx):
        blk = np.asarray(self.mm[idx, :])[:, self.col].astype(np.float64)
        sp = object.__new__(SW.Splats)
        sp.n = len(blk)
        sp.xyz_mm = blk[:, 0:3] * 1000.0
        sp.scale_mm = np.exp(blk[:, 3:6]) * 1000.0
        sp.opacity = 1.0 / (1.0 + np.exp(-blk[:, 6]))
        q = blk[:, 7:11]
        sp.quat = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        sp.rgb = np.clip(0.5 + SW.SH_C0 * blk[:, 11:14], 0.0, 1.0)
        sp.live = sp.opacity >= SW.MIN_OPACITY
        sp._cov = None
        return sp


def measure(M, K, R, t_mm, wh):
    """-> (stats, picture (gh, gw, 3), in-front map (gh, gw))."""
    W, H = int(wh[0]), int(wh[1])
    K = np.asarray(K, np.float64)
    Xc = M.xyz @ np.asarray(R, np.float32).T + (np.asarray(t_mm) / 1000.0).astype(np.float32)
    z = Xc[:, 2]
    with np.errstate(all="ignore"):
        u = K[0, 0] * Xc[:, 0] / z + K[0, 2]
        v = K[1, 1] * Xc[:, 1] / z + K[1, 2]
        r = 3 * M.smax * K[0, 0] / z + 2 * CELL
    base = (z > 0.02) & (v > -r) & (v < H + r) & (M.op >= SW.MIN_OPACITY)
    edges = [int(round(W * i / NSTRIP / CELL)) * CELL for i in range(NSTRIP)] + [W]
    imgs, fws, alphas = [], [], []
    tot = n03 = n10 = 0.0
    for u0, u1 in zip(edges[:-1], edges[1:]):
        gh, gw = SW.grid_shape(u1 - u0, H, CELL)
        idx = np.flatnonzero(base & (u + r > u0) & (u - r < u1))
        if len(idx) == 0:
            imgs.append(np.zeros((gh, gw, 3))); fws.append(np.zeros((gh, gw))); alphas.append(np.zeros((gh, gw)))
            continue
        Ks = K.copy()
        Ks[0, 2] -= u0
        vw = SW.view_weights(M.mini(idx), Ks, R, t_mm, (u1 - u0, H), cell=CELL)
        imgs.append(np.clip(vw.rgb, 0, 1)); alphas.append(1.0 - vw.T)
        if len(vw.w) == 0:
            fws.append(np.zeros((gh, gw)))
            continue
        gid = idx[vw.splat]
        zt = z[gid].astype(np.float64)
        cell, w = vw.cell, vw.w.astype(np.float64)
        newseg = np.r_[True, cell[1:] != cell[:-1]]
        starts = np.flatnonzero(newseg)
        segid = np.cumsum(newseg) - 1
        cw = np.cumsum(w)
        before = (cw - w) - (cw - w)[starts][segid]
        half = before + w >= 0.5 * np.add.reduceat(w, starts)[segid]
        pos = np.flatnonzero(half)
        first = pos[np.r_[True, segid[pos][1:] != segid[pos][:-1]]]
        z50 = np.full(len(starts), np.nan)
        z50[segid[first]] = zt[first]
        front = zt < 0.5 * z50[segid]
        fws.append(np.bincount(cell[front], weights=w[front], minlength=gh * gw).reshape(gh, gw))
        wf, dl = w[front], M.dline[gid[front]]
        tot += wf.sum(); n03 += wf[dl < 0.3].sum(); n10 += wf[dl < 1.0].sum()
    img, fw, alpha = np.hstack(imgs), np.hstack(fws), np.hstack(alphas)
    return {"in_front": float(fw.mean()), "cells_over_30": float((fw > 0.3).mean()), "opacity": float(alpha.mean()),
            "from_03": float(n03 / tot) if tot else 0.0, "from_10": float(n10 / tot) if tot else 0.0}, img, fw


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("project")
    ap.add_argument("--ply", action="append", required=True, help="a model (project-relative or a path); repeat to compare")
    ap.add_argument("--move", default="walk")
    ap.add_argument("--frames", default="60,130,260,390,470")
    ap.add_argument("--views", default="", help="training views to measure as well, e.g. cap191_L,cap040_L")
    ap.add_argument("--out", default=None, help="write picture + in-front map per frame and model here")
    a = ap.parse_args(argv)
    P = a.project
    G = np.load(os.path.join(P, "train", "dataset", "rig.npz"), allow_pickle=True)
    names = [str(x) for x in G["names"]]
    la, lb, _step, _tears = clearpath.walked_line(os.path.join(P, "train", "dataset", "rig.npz"),
                                                  os.path.join(P, "select", "quality.json"))
    with open(os.path.join(P, "move", a.move + ".json")) as f:
        mv = json.load(f)
    shots = []
    for k in [int(x) for x in a.frames.split(",") if x.strip()]:
        c2w = np.array(mv["frames"][k]["c2w"], float)
        Rm = c2w[:3, :3].T
        shots.append((f"{a.move} {k}", np.array(mv["K"], float), Rm, -Rm @ (c2w[:3, 3] * 1000.0), (mv["width"], mv["height"])))
    for nm in [x for x in a.views.split(",") if x.strip()]:
        i = names.index(nm)
        shots.append((nm, G["K"][i], G["R"][i], G["t"][i], (int(G["w"]), int(G["h"]))))
    rows = []
    for ply in a.ply:
        path = ply if os.path.exists(ply) else os.path.join(P, ply)
        t0 = time.time()
        M = Model(path, (la, lb))
        label = os.path.relpath(path, P) if os.path.abspath(path).startswith(os.path.abspath(P)) else path
        print(f"{label}: {M.n:,} splats, read in {time.time() - t0:.0f} s", flush=True)
        for tag, K, Rm, t, wh in shots:
            st, img, fw = measure(M, K, Rm, t, wh)
            rows.append((label, tag, st))
            print(f"  {tag:14s} in front {100 * st['in_front']:5.1f} %   cells over 30 %: {100 * st['cells_over_30']:5.1f} %   "
                  f"opacity {st['opacity']:.3f}   of it within 0.3 m / 1 m of the walked line: "
                  f"{100 * st['from_03']:3.0f} % / {100 * st['from_10']:3.0f} %", flush=True)
            if a.out:
                import cv2
                os.makedirs(a.out, exist_ok=True)
                up = lambda x: cv2.resize(x, (x.shape[1] * 3, x.shape[0] * 3), interpolation=cv2.INTER_NEAREST)
                heat = cv2.applyColorMap(np.clip(fw * 510, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
                stem = (os.path.splitext(os.path.basename(path))[0] + "_" + tag).replace(" ", "_")
                cv2.imwrite(os.path.join(a.out, stem + ".jpg"),
                            np.vstack([up((img[..., ::-1] * 255).astype(np.uint8)), up(heat)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
        del M
    return rows


if __name__ == "__main__":
    main()
