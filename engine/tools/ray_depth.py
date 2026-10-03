#!/usr/bin/env python3
"""What a camera sees of a splat model in depth: where along each ray the blend weight sits.

Hold-out PSNR does not see a smoky surface — a model that draws a glossy helmet as 40 mm of haze
reproduces the photographs on the capture path to 31 dB. This does. For a sample of the training
views it composites the splats along rays the way Brush's depth pass does (each splat's alpha at
the ray from its projected 2D covariance, front to back by centre depth) and reports, per ray:

  thickness   the depth span that holds the 10th to the 90th percentile of the blend weight.
              A surface drawn by millimetre splats is 3-5 mm; smoke is tens of millimetres.
  spread      std(depth) / mean(depth) over the weights: the number Brush logs as
              "mean relative depth spread", and what --depth-spread-weight penalises.

and, where train/depth holds a LiDAR depth map for the view (hs train --depth-weight):

  depth error      mean |ln(expected depth / LiDAR depth)|: Brush's "mean relative depth error"
  weight behind    share of the blend weight more than 10 / 20 / 40 mm behind the LiDAR surface
  weight in front  share more than 5 mm in front of it

Rays are split into "covered" (the view's depth map has a value there) and "uncovered" (inside
the subject mask, no value: what the scan did not reach, plus edges the maps leave out). The
"frontal" rows keep rays within 45 degrees of the direction from the subject's centre, which
drops the limb, where any surface is thick along the ray. That test assumes a roughly convex
subject.

  python3 engine/tools/ray_depth.py Projects/P --ply Projects/P/archive/C/export_40000.ply
  python3 engine/tools/ray_depth.py Projects/P --ply A.ply --ply B.ply --views 60

Run C against B on the Stormtrooper helmet (2026-10-02) is in the project notes: the depth loss
halved the depth error where the scan was and left the thickness alone.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # engine/, so `hs` imports without an install

from hs import depthmaps as D, rig as riglib  # noqa: E402
from hs.splatweights import Splats  # noqa: E402

STEP = 4            # one ray every STEP pixels of the depth-map-sized image
LONG_EDGE = 512
MIN_ALPHA = 0.5     # a ray counts where the model is at least this opaque, as in Brush
FRONTAL_COS = 0.7
ALPHA_MAX = 0.999   # Brush's cap on one splat's alpha (kernels/rasterize.rs)
T_STOP = 1.0e-4     # ...and the transmittance at which it stops compositing a ray


def load(path):
    """-> (xyz mm, opacity, 3x3 covariance mm^2) of the splats that are not dead."""
    sp = Splats(path)
    live = sp.opacity > 1 / 255.0
    w, x, y, z = sp.quat[live].T
    Rq = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                   2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                   2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)
    M = Rq * sp.scale_mm[live][:, None, :]
    return sp.xyz_mm[live], sp.opacity[live].astype(np.float32), (M @ M.transpose(0, 2, 1)).astype(np.float32)


def composite(xyz, opac, cov, K, R, t, wh, long_edge=LONG_EDGE, step=STEP):
    """Blend weights along a grid of rays. -> dict(Wg, Hg, pix, z, w, first, grp, full)

    One ray per `step` pixels of the long_edge-sized image, through pixel centres. Entries are
    (ray, splat) pairs sorted by ray then depth; `w` is the splat's blend weight on that ray."""
    W, H = D.map_size(wh, long_edge)
    sx, sy = W / float(wh[0]), H / float(wh[1])
    Xc = xyz @ R.T + t
    z = Xc[:, 2]
    fx, fy, cx, cy = K[0, 0] * sx, K[1, 1] * sy, K[0, 2] * sx, K[1, 2] * sy
    ok = z > 1.0
    zs = np.where(ok, z, 1.0)
    u = np.where(ok, fx * Xc[:, 0] / zs + cx, -99.0)
    v = np.where(ok, fy * Xc[:, 1] / zs + cy, -99.0)
    ok &= (u > -8 * step) & (u < W + 8 * step) & (v > -8 * step) & (v < H + 8 * step)
    idx = np.flatnonzero(ok)
    Xc, z, u, v, o = Xc[idx], z[idx], u[idx], v[idx], opac[idx].astype(np.float64)
    Sc = np.einsum("ij,njk,lk->nil", R, cov[idx].astype(np.float64), R)
    a, b, c, d = fx / z, -fx * Xc[:, 0] / z ** 2, fy / z, -fy * Xc[:, 1] / z ** 2
    # the colour pass widens every footprint by 0.3 px^2 of its own (full) resolution
    S00 = a * a * Sc[:, 0, 0] + 2 * a * b * Sc[:, 0, 2] + b * b * Sc[:, 2, 2] + 0.3 * sx * sx
    S01 = a * c * Sc[:, 0, 1] + a * d * Sc[:, 0, 2] + b * c * Sc[:, 1, 2] + b * d * Sc[:, 2, 2]
    S11 = c * c * Sc[:, 1, 1] + 2 * c * d * Sc[:, 1, 2] + d * d * Sc[:, 2, 2] + 0.3 * sy * sy
    del Sc
    det = np.maximum(S00 * S11 - S01 ** 2, 1e-12)
    kk = np.clip(np.ceil(3.0 * np.sqrt(np.maximum(S00, S11)) / step), 1, 8).astype(int)
    g0x, g0y = np.rint((u - 0.5) / step).astype(int), np.rint((v - 0.5) / step).astype(int)
    Wg, Hg = (W + step - 1) // step, (H + step - 1) // step
    pix, zz, al = [], [], []
    for k in range(1, 9):
        m = np.flatnonzero(kk == k)
        if not len(m):
            continue
        um, vm, om, zm = u[m], v[m], o[m], z[m].astype(np.float32)
        i00, i01, i11 = S11[m] / det[m], -S01[m] / det[m], S00[m] / det[m]
        for dy in range(-k, k + 1):
            gy = g0y[m] + dy
            ey = gy * step + 0.5 - vm
            for dx in range(-k, k + 1):
                gx = g0x[m] + dx
                ex = gx * step + 0.5 - um
                alpha = np.minimum(om * np.exp(-0.5 * (i00 * ex * ex + 2 * i01 * ex * ey + i11 * ey * ey)), ALPHA_MAX)
                keep = (alpha > 1 / 255.0) & (gx >= 0) & (gx < Wg) & (gy >= 0) & (gy < Hg)
                if keep.any():
                    pix.append((gy[keep] * Wg + gx[keep]).astype(np.int32))
                    zz.append(zm[keep])
                    al.append(alpha[keep].astype(np.float32))
    if not pix:
        e = np.zeros(0)
        return dict(Wg=Wg, Hg=Hg, pix=e.astype(np.int32), z=e, w=e, first=e.astype(int), grp=e.astype(int), full=(W, H))
    pix, zz, al = np.concatenate(pix), np.concatenate(zz), np.concatenate(al)
    order = np.lexsort((zz, pix))
    pix, zz, al = pix[order], zz[order], al[order]
    lt = np.log1p(-al.astype(np.float64))
    cs = np.cumsum(lt)
    newg = np.r_[True, pix[1:] != pix[:-1]]
    first = np.flatnonzero(newg)
    grp = np.cumsum(newg) - 1
    before = cs - lt - (cs - lt)[first][grp]                # log transmittance in front of each entry
    # Brush stops a ray at the splat that would take the transmittance to T_STOP: it and
    # everything behind it are not drawn. Without this the far side of the subject, seen
    # through the last 0.01 % of a solid surface, would count as depth spread.
    live = (before + lt) > np.log(T_STOP)
    pix, zz, al, before = pix[live], zz[live], al[live], before[live]
    newg = np.r_[True, pix[1:] != pix[:-1]] if len(pix) else np.zeros(0, bool)
    first = np.flatnonzero(newg)
    grp = np.cumsum(newg) - 1
    w = np.exp(before) * al
    return dict(Wg=Wg, Hg=Hg, pix=pix, z=zz.astype(np.float64), w=w, first=first, grp=grp, full=(W, H))


def ray_stats(r):
    """Per ray: alpha, mean depth, std, and the 10 / 50 / 90 % weight quantile depths."""
    n = r["Wg"] * r["Hg"]
    A = np.bincount(r["pix"], weights=r["w"], minlength=n)
    safe = np.maximum(A, 1e-9)
    mean = np.bincount(r["pix"], weights=r["w"] * r["z"], minlength=n) / safe
    var = np.bincount(r["pix"], weights=r["w"] * r["z"] ** 2, minlength=n) / safe - mean ** 2
    cw = np.cumsum(r["w"])
    cw = cw - (cw - r["w"])[r["first"]][r["grp"]] if len(cw) else cw
    Apx = A[r["pix"]]

    def quant(q):
        hit = (cw >= q * Apx) & (cw - r["w"] < q * Apx)
        out = np.zeros(n)
        out[r["pix"][hit]] = r["z"][hit]
        return out
    return dict(alpha=A, mean=mean, std=np.sqrt(np.maximum(var, 0.0)), q10=quant(0.1), q50=quant(0.5), q90=quant(0.9))


def measure(ply, project, views, step=STEP, centre=None, budget_s=None):
    """-> {"groups": {name: stats}, "views": n} for one model over `views` ([(key, index)])."""
    import cv2
    G, names, _ = riglib.load(os.path.join(project, "train", "dataset", "rig.npz"))
    depth_dir = os.path.join(project, "train", "depth")
    unit = None
    rep_path = os.path.join(depth_dir, "depth_report.json")
    if os.path.exists(rep_path):
        with open(rep_path) as f:
            unit = json.load(f)["unit_mm"]
    xyz, opac, cov = load(ply)
    if centre is None:
        centre = np.median(xyz[opac > 0.5], axis=0) if (opac > 0.5).any() else np.median(xyz, axis=0)
    acc = {}
    t0 = time.time()
    done = 0
    for key, i in views:
        if budget_s and time.time() - t0 > budget_s:
            break
        wh = (int(G["wh"][i][0]), int(G["wh"][i][1])) if "wh" in G.files else (int(G["w"]), int(G["h"]))
        K, R, t = G["K"][i], G["R"][i], G["t"][i]
        r = composite(xyz, opac, cov, K, R, t, wh, step=step)
        s = ray_stats(r)
        Wg, Hg = r["Wg"], r["Hg"]
        n = Wg * Hg
        Wf, Hf = r["full"]
        sx, sy = Wf / float(wh[0]), Hf / float(wh[1])
        ref = np.zeros(n)
        p = os.path.join(depth_dir, key + ".png")
        if unit and os.path.exists(p):
            ref = D.read_png16(p, unit)[::step, ::step].ravel().astype(np.float64)
        m = cv2.imread(os.path.join(project, "train", "dataset", "masks", key + ".png"), cv2.IMREAD_GRAYSCALE)
        inside = np.ones(n, bool) if m is None else \
            (cv2.resize(m, (Wf, Hf), interpolation=cv2.INTER_AREA) >= 250)[::step, ::step].ravel()
        gy, gx = np.divmod(np.arange(n), Wg)
        dc = np.stack([(gx * step + 0.5 - K[0, 2] * sx) / (K[0, 0] * sx),
                       (gy * step + 0.5 - K[1, 2] * sy) / (K[1, 1] * sy), np.ones(n)], 1)
        pw = (dc * s["q50"][:, None] - t) @ R                    # world point at the median depth
        dw = dc @ R
        dw /= np.linalg.norm(dw, axis=1)[:, None]
        nr = pw - centre
        nr /= np.maximum(np.linalg.norm(nr, axis=1), 1e-9)[:, None]
        frontal = np.abs((dw * nr).sum(1)) > FRONTAL_COS
        solid = s["alpha"] > MIN_ALPHA
        zr = r["z"] - ref[r["pix"]]
        safe = np.maximum(s["alpha"], 1e-9)

        def share(cond):
            return np.bincount(r["pix"], weights=r["w"] * cond, minlength=n) / safe
        behind = {k: share(zr > k) for k in (10, 20, 40)}
        front5 = share(zr < -5)
        for name, sel in (("covered", solid & (ref > 0)), ("uncovered", solid & inside & (ref <= 0))):
            for tag, sub in ((name, sel), (name + " frontal", sel & frontal)):
                a = acc.setdefault(tag, {k: [] for k in ("thick", "spread", "err", "signed", "b10", "b20", "b40", "f5")})
                a["thick"].append((s["q90"] - s["q10"])[sub])
                a["spread"].append((s["std"] / np.maximum(s["mean"], 1e-9))[sub])
                if name == "covered":
                    a["err"].append(np.abs(np.log(np.maximum(s["mean"][sub], 1e-9) / ref[sub])))
                    a["signed"].append((s["mean"] - ref)[sub])
                    a["b10"].append(behind[10][sub]); a["b20"].append(behind[20][sub]); a["b40"].append(behind[40][sub])
                    a["f5"].append(front5[sub])
        done += 1
    out = {}
    for tag, a in acc.items():
        th = np.concatenate(a["thick"]) if a["thick"] else np.zeros(0)
        if not len(th):
            continue
        g = {"rays": int(len(th)), "thickness_median_mm": round(float(np.median(th)), 2),
             "thickness_mean_mm": round(float(th.mean()), 2),
             "thicker_than_10mm": round(float((th > 10).mean()), 4), "thicker_than_20mm": round(float((th > 20).mean()), 4),
             "thicker_than_40mm": round(float((th > 40).mean()), 4),
             "rel_spread_mean": round(float(np.concatenate(a["spread"]).mean()), 5)}
        if a["err"] and sum(len(x) for x in a["err"]):
            g.update({"rel_depth_error_mean": round(float(np.concatenate(a["err"]).mean()), 5),
                      "depth_offset_median_mm": round(float(np.median(np.concatenate(a["signed"]))), 2),
                      "weight_front_5mm": round(float(np.concatenate(a["f5"]).mean()), 4),
                      "weight_behind_10mm": round(float(np.concatenate(a["b10"]).mean()), 4),
                      "weight_behind_20mm": round(float(np.concatenate(a["b20"]).mean()), 4),
                      "weight_behind_40mm": round(float(np.concatenate(a["b40"]).mean()), 4)})
        out[tag] = g
    all_spread = [x for tag in ("covered", "uncovered") for x in acc.get(tag, {}).get("spread", [])]
    if all_spread:
        out["all solid rays"] = {"rays": int(sum(len(x) for x in all_spread)),
                                 "rel_spread_mean": round(float(np.concatenate(all_spread).mean()), 5)}
    return {"ply": ply, "splats": int(len(xyz)), "views": done, "groups": out,
            "centre_mm": [round(float(x), 1) for x in centre]}


def pick_views(project, n_views, exclude=()):
    """Evenly spaced training views: those with a depth map when train/depth exists, else all."""
    G, names, _ = riglib.load(os.path.join(project, "train", "dataset", "rig.npz"))
    keys = [(D.view_key(nm), i) for i, nm in enumerate(names)]
    rep_path = os.path.join(project, "train", "depth", "depth_report.json")
    if os.path.exists(rep_path):
        with open(rep_path) as f:
            written = {r["view"] for r in json.load(f)["per_view"] if r.get("written")}
        keys = [k for k in keys if k[0] in written]
    keys = [k for k in keys if k[0] not in exclude]
    if n_views and len(keys) > n_views:
        pick = np.linspace(0, len(keys) - 1, n_views).round().astype(int)
        keys = [keys[j] for j in pick]
    return keys


ROWS = [("rays", "rays", "{:,}"), ("thickness_median_mm", "thickness, median mm", "{:.2f}"),
        ("thickness_mean_mm", "thickness, mean mm", "{:.2f}"), ("thicker_than_10mm", "rays thicker than 10 mm", "{:.1%}"),
        ("thicker_than_20mm", "rays thicker than 20 mm", "{:.1%}"), ("thicker_than_40mm", "rays thicker than 40 mm", "{:.1%}"),
        ("rel_spread_mean", "relative spread, mean", "{:.4f}"), ("rel_depth_error_mean", "relative depth error, mean", "{:.4f}"),
        ("depth_offset_median_mm", "depth offset, median mm", "{:+.2f}"), ("weight_front_5mm", "weight > 5 mm in front", "{:.1%}"),
        ("weight_behind_10mm", "weight > 10 mm behind", "{:.1%}"), ("weight_behind_20mm", "weight > 20 mm behind", "{:.1%}"),
        ("weight_behind_40mm", "weight > 40 mm behind", "{:.1%}")]


def table(results, labels):
    lines = []
    tags = [t for t in ("covered frontal", "uncovered frontal", "covered", "uncovered", "all solid rays")
            if any(t in r["groups"] for r in results)]
    for tag in tags:
        lines.append(f"\n{tag}")
        lines.append(f"  {'':30s}" + "".join(f"{lb:>16s}" for lb in labels))
        for key, title, fmt in ROWS:
            vals = [r["groups"].get(tag, {}).get(key) for r in results]
            if all(v is None for v in vals):
                continue
            lines.append(f"  {title:30s}" + "".join(f"{(fmt.format(v) if v is not None else '-'):>16s}" for v in vals))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project")
    ap.add_argument("--ply", action="append", required=True, help="a model to measure; give it again to compare")
    ap.add_argument("--views", type=int, default=60, help="training views sampled, evenly (0 = all; default 60)")
    ap.add_argument("--step", type=int, default=STEP, help="one ray every N pixels of the 512-px image (default 4)")
    ap.add_argument("--out", default=None, help="write the numbers as JSON here")
    a = ap.parse_args(argv)
    views = pick_views(a.project, a.views)
    results, labels, centre = [], [], None
    for ply in a.ply:
        t0 = time.time()
        res = measure(ply, a.project, views, step=a.step, centre=centre)
        centre = np.asarray(res["centre_mm"])          # every model is cut by the first one's centre
        results.append(res)
        d = os.path.basename(os.path.dirname(os.path.abspath(ply)))
        labels.append(d[:15] if d not in ("exports", "") else os.path.basename(ply)[:15])
        print(f"{ply}: {res['splats']:,} live splats, {res['views']} views, {time.time() - t0:.0f} s", file=sys.stderr)
    print(table(results, labels))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
