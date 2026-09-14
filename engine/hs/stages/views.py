"""hs views — render the trained model from real capture poses and compare it with the
photographs those poses came from.

Every other number in this pipeline is the solver grading its own homework: the reprojection
residual measures how well the SfM fits the features it chose, and the hull check measures
how far a virtual camera strays from a real one. Neither can tell you the model is wrong.
This one can, because the reference is the photograph.

For each chosen capture it renders the .ply from that capture's exact pose (its own K and
canvas, so the render lands pixel-aligned on the training image) and reports three things
about a subject-centred crop:

  retained edge energy   render's edge response / the source's, per unit contrast.
                         Blur in the capture, or a model too soft to reproduce it, lowers it.
  agreement              PSNR and correlation against the photograph.
  displacement           phase correlation on 64 px patches: how far the model puts
                         structure from where the photograph puts it, and what fraction of
                         textured patches are off by more than --displaced-px.

The pair separates two failure modes that look alike in a rendered move: blur costs edge
energy and leaves displacement alone, misregistration displaces patches. On the rig6 golden
train the two azimuth extremes come out at 3% of patches displaced (az −42°) against 24%
(az +43°) — an 8× separation — while retained edge energy differs by only 1.24× (61% against
49%). Both degrade on the bad side, but displacement is what distinguishes it, and 10.4 px
at 238 mm is 1.46 mm of world error against 0.23 mm on the good side. That is what identified
the soft right side of rig6_boom as thin coverage and weak registration rather than the
motion blur it resembled — the source frames at the two extremes are equally sharp.

Note the highest view (az −7°, el +25°) scores best of all at 69% and 1%, so a fast pass is
not automatically a bad one. What failed at az +43° was density: 5 captures within 8° of it
spread over 18° of elevation, against 9 within an 8° band at az −42°.

The displaced fraction is diluted by whatever background falls inside the crop, so compare it
between views of one capture, and between captures shot the same way — not against a number
from a differently framed run.
"""
import json
import os
import shutil
import sys

import numpy as np

from .. import events, runner
from .prune import final_export
from .render import DEFAULT_RENDER, RE_FRAME, RE_LOADED, RE_PATH

STAGE = "views"
PATCH, PATCH_STEP, PATCH_MIN_STD, PATCH_MIN_RESP = 64, 32, 12.0, 0.25


def add_parser(sub):
    p = sub.add_parser("views", help="render from real capture poses and grade the model against the photographs")
    p.add_argument("--captures", default=None,
                   help="capture indices, e.g. 5,15,55 (default: the azimuth extremes, the centre and the highest view)")
    p.add_argument("--ply", default=None, help="default: the train stage's final export")
    p.add_argument("--name", default="views")
    p.add_argument("--subject-mm", type=float, default=None,
                   help="world extent the comparison crop covers (default: measured from the point cloud)")
    p.add_argument("--displaced-px", type=float, default=4.0, help="a patch this far off counts as displaced")
    p.add_argument("--patch-step", type=int, default=PATCH_STEP,
                   help=f"patch stride in px (default {PATCH_STEP}; 8 or 16 samples far denser, "
                        "which the displaced-population direction needs -- patches overlap, so the "
                        "extra samples are correlated and do not buy sqrt(n) on the mean)")
    p.add_argument("--max-displaced-fraction", type=float, default=0.10,
                   help="check threshold (rig6 golden: 0.03 on the well-covered side, 0.24 on the thin one)")
    p.add_argument("--keep-frames", action="store_true", help="keep the raw renders as well as the comparisons")
    p.add_argument("--render-bin", default=os.environ.get("HS_PATH_RENDER", DEFAULT_RENDER))
    return p


def auto_captures(cov, n):
    """The views that actually discriminate: both azimuth extremes, the centre, the highest."""
    caps = cov["captures"]
    pick = [max(caps, key=lambda c: c["azimuth_deg"])["capture"],
            min(caps, key=lambda c: c["azimuth_deg"])["capture"],
            min(caps, key=lambda c: abs(c["azimuth_deg"]))["capture"],
            max(caps, key=lambda c: c["elevation_deg"])["capture"]]
    out = []
    for c in pick:
        if c not in out and 0 <= c < n:
            out.append(int(c))
    return out


def subject_extent_mm(pts, subject):
    """Diameter of the inner half of the cloud about the median — the subject, without a
    hard-coded size (rig6: 89 mm, against a bust measuring 86 x 93 mm)."""
    d = np.linalg.norm(pts - subject, axis=1)
    inner = d[d <= np.median(d)]
    return float(2.0 * np.percentile(inner, 95)) if len(inner) else 100.0


def build_path(pj, captures, out_path):
    """One frame per capture at that capture's own left-eye pose, K and canvas."""
    G = np.load(pj.rig_npz, allow_pickle=True)
    names = [str(x) for x in G["names"]]
    L = np.arange(0, len(names), 2)
    K, R, t, C = (G[k].astype(float) for k in ("K", "R", "t", "C"))
    pts = G["pts"].astype(float)
    subject = np.median(pts, axis=0)
    bad = [c for c in captures if not 0 <= c < len(L)]
    if bad:
        raise events.StageError(f"captures out of range 0..{len(L) - 1}: {bad}")
    # the L views share one K and canvas; wh is per view for anything that needs the R eye's
    wh = G["wh"] if "wh" in G.files else None
    w, h = (int(wh[L[0]][0]), int(wh[L[0]][1])) if wh is not None else (int(G["w"]), int(G["h"]))
    frames, views = [], []
    for c in captures:
        v = int(L[c])
        c2w = np.eye(4)
        c2w[:3, :3] = R[v].T
        c2w[:3, 3] = C[v] / 1000.0
        frames.append({"c2w": c2w.tolist()})
        Xc = R[v] @ subject + t[v]
        uv = (K[v] @ Xc)[:2] / Xc[2]
        views.append({"capture": c, "view": names[v], "image": names[v][:-2] + ".jpg",
                      "subject_px": [float(uv[0]), float(uv[1])], "depth_mm": float(Xc[2])})
    json.dump({"note": "hs views: the model rendered from real capture poses, for A/B against the photographs",
               "width": w, "height": h, "K": K[int(L[0])].tolist(), "fps": 30.0, "frames": frames},
              open(out_path, "w"), indent=1)
    return views, subject_extent_mm(pts, subject), float(K[int(L[0])][0, 0])


def compare(src, ren, uv, half, displaced_px, step=PATCH_STEP):
    """Edge energy, agreement and displacement on one subject-centred crop."""
    import cv2
    h, w = src.shape
    x0, x1 = max(0, int(uv[0] - half)), min(w, int(uv[0] + half))
    y0, y1 = max(0, int(uv[1] - half)), min(h, int(uv[1] + half))
    s, r = src[y0:y1, x0:x1], ren[y0:y1, x0:x1]
    if min(s.shape) < PATCH * 2:
        return None

    def sharp(a):
        contrast = np.percentile(a, 95) - np.percentile(a, 5)
        return float(np.percentile(np.abs(cv2.Laplacian(a, cv2.CV_32F)), 90) / max(contrast, 1e-6))

    mse = float(((s - r) ** 2).mean())
    win = cv2.createHanningWindow((PATCH, PATCH), cv2.CV_32F)
    mags = []
    for yy in range(0, s.shape[0] - PATCH, step):
        for xx in range(0, s.shape[1] - PATCH, step):
            a, b = s[yy:yy + PATCH, xx:xx + PATCH], r[yy:yy + PATCH, xx:xx + PATCH]
            if a.std() < PATCH_MIN_STD:
                continue                                  # flat patch: phase correlation is noise
            (dx, dy), resp = cv2.phaseCorrelate(a.copy(), b.copy(), win)
            if resp < PATCH_MIN_RESP:
                continue
            mags.append((xx + PATCH // 2, yy + PATCH // 2, float(np.hypot(dx, dy)), float(dx), float(dy)))
    m = np.array([p[2] for p in mags]) if mags else np.zeros(0)

    # A view can fail two different ways and the fraction over threshold cannot tell them
    # apart: the whole crop can sit shifted (one bad frame pose), or the bulk can register
    # while a subpopulation flies off (the bimodal residual). Detrend to separate them --
    # but detrend on the MEDIAN vector, not the mean. When two populations sit at 0 and at
    # some offset, the mean lands between them and subtracting it moves *both* off zero;
    # that reads as the view getting worse. The component-wise median rides with whichever
    # population holds the majority, so the registered bulk collapses and the displaced
    # minority is left standing on its own.
    vec = np.array([[p[3], p[4]] for p in mags], dtype=np.float64) if mags else np.zeros((0, 2))
    med_vec = np.median(vec, axis=0) if len(vec) else np.zeros(2)
    mean_vec = vec.mean(axis=0) if len(vec) else np.zeros(2)
    bulk = float(np.hypot(*med_vec))
    # Directional agreement, not uniformity: 1.0 means every patch moves the same way, which
    # a two-population field along one axis also satisfies. Read it next to the detrended
    # fraction -- high coherence with a high detrended fraction is two clusters, not a shift.
    coherence = float(np.hypot(*mean_vec)) / float(m.mean()) if len(m) and m.mean() else None
    det = np.hypot(*(vec - med_vec).T) if len(vec) else np.zeros(0)

    # Where does the displaced population actually go? If those patches all travel the same
    # way, the displacement is a second image-formation mode rather than scattered geometry
    # error -- and its bearing is then comparable against the stereo baseline in the image.
    # Select on the DETRENDED magnitude, not the raw one. Selecting raw and then subtracting
    # the median picks every patch when the whole view is shifted, leaving near-zero residuals
    # whose bearing is pure noise -- which is exactly what the uniform-shift harness case shows.
    resid = (vec - med_vec) if len(vec) else np.zeros((0, 2))
    off = resid[det > displaced_px] if len(resid) else np.zeros((0, 2))
    off_mean = off.mean(axis=0) if len(off) else np.zeros(2)   # already detrended
    off_mag = np.hypot(*off.T) if len(off) else np.zeros(0)
    off_coh = float(np.hypot(*off_mean)) / float(off_mag.mean()) if len(off) and off_mag.mean() else None
    off_deg = float(np.degrees(np.arctan2(off_mean[1], off_mean[0]))) if len(off) else None
    ss, rs = sharp(s), sharp(r)
    return {
        "src_sharpness": round(ss, 4), "render_sharpness": round(rs, 4),
        "retained_edge_energy": round(rs / ss, 3) if ss else None,
        "psnr_db": round(10 * np.log10(255.0 ** 2 / max(mse, 1e-9)), 2),
        "correlation": round(float(np.corrcoef(s.ravel(), r.ravel())[0, 1]), 4),
        "patches": len(m),
        "displacement_median_px": round(float(np.median(m)), 2) if len(m) else None,
        "displacement_p90_px": round(float(np.percentile(m, 90)), 2) if len(m) else None,
        "displaced_fraction": round(float((m > displaced_px).mean()), 3) if len(m) else None,
        "bulk_shift_px": round(bulk, 2) if len(m) else None,
        "bulk_shift_dir_px": [round(float(med_vec[0]), 2), round(float(med_vec[1]), 2)] if len(m) else None,
        "registered_fraction": round(float((m < 1.0).mean()), 3) if len(m) else None,
        "displaced_dir_px": [round(float(off_mean[0]), 2), round(float(off_mean[1]), 2)] if len(off) else None,
        "displaced_dir_deg": round(off_deg, 1) if off_deg is not None else None,
        "displaced_dir_coherence": round(off_coh, 3) if off_coh is not None else None,
        "displaced_mean_px": round(float(off_mag.mean()), 2) if len(off) else None,
        "shift_coherence": round(coherence, 3) if coherence is not None else None,
        "displaced_fraction_detrended": round(float((det > displaced_px).mean()), 3) if len(det) else None,
        "displacement_p90_detrended_px": round(float(np.percentile(det, 90)), 2) if len(det) else None,
        "_patches": mags, "_box": (x0, y0, x1, y1),
    }


def comparison_image(src_bgr, ren_bgr, res, out_path, label, displaced_px):
    """Photograph, model, difference, and where the displaced patches are."""
    import cv2
    x0, y0, x1, y1 = res["_box"]
    S, R = src_bgr[y0:y1, x0:x1], ren_bgr[y0:y1, x0:x1]
    D = cv2.applyColorMap(np.clip(cv2.absdiff(cv2.cvtColor(S, cv2.COLOR_BGR2GRAY),
                                              cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)) * 4, 0, 255).astype(np.uint8),
                          cv2.COLORMAP_INFERNO)
    M = S.copy()
    mvx, mvy = res.get("bulk_shift_dir_px") or (0.0, 0.0)
    for x, y, mag, dx, dy in res["_patches"]:
        col = (0, 200, 0) if mag < displaced_px / 2 else ((0, 200, 255) if mag < displaced_px else (0, 0, 255))
        cv2.circle(M, (x, y), 4, col, -1)
        rx, ry = dx - mvx, dy - mvy          # what is left once the bulk shift is removed
        if np.hypot(rx, ry) > displaced_px / 2:
            cv2.arrowedLine(M, (x, y), (int(x + rx * 3), int(y + ry * 3)), (255, 255, 255), 1,
                            cv2.LINE_AA, tipLength=0.3)
    tiles = []
    for a, t in ((S, "photograph"), (R, "model, same pose"), (D, "difference x4"),
                 (M, f"patches (red > {displaced_px:g} px)")):
        a = cv2.copyMakeBorder(a, 24, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        cv2.putText(a, t, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(a)
    sheet = np.hstack(tiles)
    sheet = cv2.copyMakeBorder(sheet, 26, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(sheet, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


def run(a, pj):
    import cv2
    pj.require(STAGE)
    ply = os.path.abspath(a.ply) if a.ply else final_export(pj)
    if not ply or not os.path.exists(ply):
        raise events.StageError("no .ply to grade", hint="hs train first, or --ply PATH")
    rbin = runner.which(a.render_bin)
    if not rbin:
        raise events.StageError(f"brush-path-render not found at {a.render_bin}",
                                hint="cargo build --release -p brush-path-render, or --render-bin PATH")
    cov_path = pj.path("solve", "coverage.json")
    if not os.path.exists(cov_path):
        raise events.StageError("no solve/coverage.json", hint="hs solve --project ...")
    cov = json.load(open(cov_path))
    by_cap = {c["capture"]: c for c in cov["captures"]}

    pj.begin(STAGE, argv=sys.argv)
    captures = ([int(c) for c in a.captures.split(",")] if a.captures
                else auto_captures(cov, cov["n_captures"]))
    events.start(STAGE, "path")
    path_json = pj.path("views", f"{a.name}.json")
    views, subj_mm, fx = build_path(pj, captures, path_json)
    subj_mm = a.subject_mm or subj_mm
    pj.metric(STAGE, "captures", captures)
    pj.metric(STAGE, "subject_extent_mm", round(subj_mm, 1))
    pj.metric(STAGE, "ply", pj.rel(ply) if ply.startswith(pj.root) else ply)

    events.start(STAGE, "render")
    out_dir = pj.path("views", "frames")
    os.makedirs(out_dir, exist_ok=True)
    st = {"n": 0}

    def on_line(line):
        m = RE_PATH.search(line)
        if m:
            st["n"] = int(m.group(1))
            return
        m = RE_LOADED.search(line)
        if m:
            pj.metric(STAGE, "splats", int(m.group(1)))
            return
        m = RE_FRAME.search(line)
        if m:
            events.progress(STAGE, int(m.group(1)), st["n"] or len(views), step="render")

    runner.run([rbin, ply, "--path", path_json, "-o", out_dir], STAGE, log_path=pj.log_path(STAGE),
               on_line=on_line, env={"RUST_LOG": os.environ.get("RUST_LOG", "warn")})

    events.start(STAGE, "compare")
    report, worst = [], None
    for i, v in enumerate(views):
        frame = os.path.join(out_dir, f"frame_{i:04d}.png")
        img = os.path.join(pj.dataset_dir, "images", "L", v["image"])
        if not (os.path.exists(frame) and os.path.exists(img)):
            events.log(STAGE, f"[hs] missing {frame if not os.path.exists(frame) else img}")
            continue
        src_bgr, ren_bgr = cv2.imread(img), cv2.imread(frame)
        half = int(0.5 * fx * subj_mm / v["depth_mm"])
        res = compare(cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32),
                      cv2.cvtColor(ren_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32),
                      v["subject_px"], half, a.displaced_px, a.patch_step)
        if res is None:
            continue
        c = by_cap.get(v["capture"], {})
        label = (f"{v['view']}  az {c.get('azimuth_deg', 0):+.0f} el {c.get('elevation_deg', 0):+.0f}  "
                 f"{v['depth_mm']:.0f} mm  |  edge energy kept {100 * (res['retained_edge_energy'] or 0):.0f}%  "
                 f"PSNR {res['psnr_db']:.1f} dB  displaced {100 * (res['displaced_fraction'] or 0):.0f}% "
                 f"(p90 {res['displacement_p90_px'] or 0:.1f} px = {(res['displacement_p90_px'] or 0) * v['depth_mm'] / fx:.2f} mm)"
                 f"  |  bulk {res['bulk_shift_px'] or 0:.1f} px coh {res['shift_coherence'] or 0:.2f}"
                 f"  detrended {100 * (res['displaced_fraction_detrended'] or 0):.0f}%"
                 f"  registered {100 * (res['registered_fraction'] or 0):.0f}%"
                 f"  displaced pop {res['displaced_mean_px'] or 0:.1f} px at {res['displaced_dir_deg'] or 0:+.0f} deg"
                 f" coh {res['displaced_dir_coherence'] or 0:.2f}")
        cmp_path = pj.path("views", f"{a.name}_{v['view']}.jpg")
        comparison_image(src_bgr, ren_bgr, res, cmp_path, label, a.displaced_px)
        pj.artifact(STAGE, cmp_path, "image")
        row = {k: val for k, val in res.items() if not k.startswith("_")}
        row.update({"capture": v["capture"], "view": v["view"], "depth_mm": round(v["depth_mm"], 1),
                    "azimuth_deg": c.get("azimuth_deg"), "elevation_deg": c.get("elevation_deg"),
                    "displacement_p90_mm": round((res["displacement_p90_px"] or 0) * v["depth_mm"] / fx, 3)})
        report.append(row)
        events.metric(STAGE, "view", row["view"], displaced_fraction=row["displaced_fraction"],
                      displaced_fraction_detrended=row["displaced_fraction_detrended"],
                      bulk_shift_px=row["bulk_shift_px"], shift_coherence=row["shift_coherence"],
                      registered_fraction=row["registered_fraction"],
                      displaced_dir_deg=row["displaced_dir_deg"],
                      displaced_dir_coherence=row["displaced_dir_coherence"],
                      retained_edge_energy=row["retained_edge_energy"], psnr_db=row["psnr_db"])
        if worst is None or (row["displaced_fraction"] or 0) > (worst["displaced_fraction"] or 0):
            worst = row

    if not report:
        raise events.StageError("no view could be compared", hint="see logs/views.log")
    rp = pj.path("views", f"{a.name}_report.json")
    json.dump({"ply": pj.rel(ply) if ply.startswith(pj.root) else ply, "subject_extent_mm": round(subj_mm, 1),
               "displaced_px": a.displaced_px, "views": report}, open(rp, "w"), indent=1)
    pj.artifact(STAGE, rp, "json")

    frac = [r["displaced_fraction"] or 0 for r in report]
    keep = [r["retained_edge_energy"] or 0 for r in report]
    pj.metric(STAGE, "displaced_fraction_worst", max(frac))
    pj.metric(STAGE, "displaced_fraction_median", round(float(np.median(frac)), 3))
    pj.metric(STAGE, "retained_edge_energy_median", round(float(np.median(keep)), 3))
    pj.check(STAGE, "model_registers_to_photographs", max(frac) <= a.max_displaced_fraction,
             value=f"worst view {worst['view']} (az {worst['azimuth_deg']:+.0f}) has "
                   f"{100 * (worst['displaced_fraction'] or 0):.0f}% of patches over {a.displaced_px:g} px "
                   f"(want ≤ {100 * a.max_displaced_fraction:.0f}%; rig6 golden: 3% at az −42, 24% at az +43)",
             needs_human=True)
    pj.check(STAGE, "edge_energy_consistent_across_views",
             (max(keep) - min(keep)) <= 0.25 if len(keep) > 1 else True,
             value=f"kept {100 * min(keep):.0f}–{100 * max(keep):.0f}% across {len(keep)} views "
                   f"(a spread here means some views are blurrier than others; rig6 golden: 49–69%)")
    if not a.keep_frames:
        shutil.rmtree(out_dir, ignore_errors=True)
    pj.finish(STAGE, ok=True)
