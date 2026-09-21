"""hs masks — a per-view silhouette of the subject, for Brush's mask channel.

Why: on this rig the subject is a small object in a whole room, and the trainer spends most
of its capacity explaining the room. Worse, when the subject is transparent the optimizer can
explain the room *through* it — the ghost rings on the coin set are the straight-ray model
fitting refracted correspondences it cannot bend a ray to reach
(3DGS_4DGS_Challenging_Materials_Guide.docx §1: "duplicate or ghosted objects through glass →
mask glass and retrain background"). A mask is the one lever that guide recommends which Brush
actually has: its dataset loader looks for a `masks/` tree mirroring `images/`, matches by file
stem, and takes white as keep.

What Brush does with it then depends on the alpha mode, and its default is not what this stage
used to claim here. A `masks/` folder on its own selects `AlphaMode::Masked`, and brush-train's
train.rs computes `do_alpha_match = has_alpha && !masked_alpha && match_alpha_weight > 0.0` — so
with masks the alpha L1 is OFF, and everything outside the silhouette is merely *unsupervised*
rather than pushed empty. That is exactly the coins `exposure-masks` result: best subject,
shredded room. To get an empty outside, train with `hs train --alpha-mode transparent`, which
premultiplies the ground truth and turns the L1 on rendered alpha back on
(`--match-alpha-weight`, default 0.1 in the current fork).

The same files serve the other half: `hs train --layer background` reads them through Brush's
`--invert-masks` and trains the room without the subject. Nothing is written twice.

Two methods (--method). `vision` (default): Apple Vision's foreground instance masks, with the
geometric silhouette below deciding which instance is the subject — an instance is kept when at
least --select-frac of it lies inside the geometric mask — then a hard edge with a 1-2 px
anti-aliased feather (--feather-px). Why: the geometric mask alone is a region, not an object. On
2026-09-20_GreetingCard it covered 45% of every frame and swept in the backdrop, wall, table and
box beside the card; a subject layer trained on it scored 2.5 dB under the full model on held-out
views, all of it on that clutter (claude/subject-layer-holdout-2026-09-21.md). A view where Vision
finds nothing, or nothing inside the geometric mask, falls back to the geometric mask and is
counted, never silently.

`geometry`: no segmentation model. The solve already knows where the subject is: take the splats
(or SfM points) within `--radius` of the subject centre, project them into every view with that
view's own K, R, t, splat each as a disc the size of its own projected footprint, close the
gaps, fill, and dilate by a world-space margin so the silhouette errs outward. A mask that is a
little too generous costs some background supervision; a mask that clips the subject removes
real observations, so the margin is deliberately one-sided.

Scale-free by default. Nothing here needs the solve to be in metres: the radius is fitted to
the camera orbit (the subject spans FRAME_FILL of the frame's half-width at the median camera
distance — 109 mm on coins, where hand-picked masks used 100), and the margin and point footprint
are fractions of it. So an unscaled mono solve gets the same masks a metric one would. An explicit
--radius (metres) still overrides, for scaled scenes where a measured size is known.

Writes train/dataset/masks/{L,R}/capNNN.png next to images/{L,R}/capNNN.jpg, plus a preview
sheet. Marks train stale: the dataset changed.
"""
import os
import sys

import numpy as np

from .. import events
from ..project import now_iso

STAGE = "masks"


FRAME_FILL = 0.7          # subject radius, as a share of the frame half-width at the median camera distance
MARGIN_FRAC = 0.05        # outward dilation, as a share of the radius (coins: 6 mm on 100)
SELECT_FRAC = 0.5         # vision: share of an instance that must lie inside the geometric mask
FEATHER_PX = 1.0          # vision: anti-aliased edge; the research report's "hard alpha + 1-2 px AA"
POINT_FRAC = 0.02         # --from-points disc footprint, as a share of the radius (coins: 2 mm on 100)


def add_parser(sub):
    p = sub.add_parser("masks", help="per-view subject silhouettes for Brush's mask channel")
    p.add_argument("--method", choices=("vision", "geometry"), default="vision",
                   help="vision (default): Apple Vision object masks, the geometric silhouette picks the "
                        "subject; geometry: the projected silhouette alone")
    p.add_argument("--select-frac", type=float, default=SELECT_FRAC,
                   help="vision: keep an instance when at least this share of it lies inside the geometric mask")
    p.add_argument("--feather-px", type=float, default=FEATHER_PX,
                   help="vision: Gaussian sigma of the anti-aliased edge (0 = hard binary)")
    p.add_argument("--grow-px", type=int, default=0,
                   help="vision: dilate the object mask outward by this many px before feathering")
    p.add_argument("--radius", type=float, default=None,
                   help="metres about the subject centre; default: fitted to the camera orbit (works unscaled)")
    p.add_argument("--radius-scale", type=float, default=1.0,
                   help="multiply the fitted radius: 0.8 tighter, 1.3 looser (ignored with --radius)")
    p.add_argument("--ply", default=None, help="default: the prune output if present, else the final export")
    p.add_argument("--min-opacity", type=float, default=0.1)
    p.add_argument("--margin-mm", type=float, default=None,
                   help="dilate the silhouette outward by this much world space; default: --margin-frac of the radius")
    p.add_argument("--margin-frac", type=float, default=MARGIN_FRAC, help="outward dilation as a share of the radius")
    p.add_argument("--close-px", type=int, default=25, help="morphological close, to bridge gaps between splats")
    p.add_argument("--keep-largest", action="store_true", default=True,
                   help="keep only the largest connected silhouette (default); --no-keep-largest to disable")
    p.add_argument("--no-keep-largest", dest="keep_largest", action="store_false")
    p.add_argument("--preview", type=int, default=6, help="views in the preview sheet")
    p.add_argument("--max-points", type=int, default=60000,
                   help="draw at most this many splats per view (a seeded sample); silhouettes close the gaps")
    p.add_argument("--from-points", action="store_true", help="use the sparse SfM points instead of a .ply")
    p.add_argument("--point-mm", type=float, default=None,
                   help="--from-points: world footprint drawn per SfM point; default: 2%% of the radius")
    return p


def fill_holes(m):
    """Fill the silhouette's interior holes: flood the background in from outside, and whatever the
    flood cannot reach is enclosed.

    The flood starts in a one-pixel border of guaranteed background. Seeding at pixel (0, 0) of the
    mask itself — as this did until 2026-09-20 — fails whenever the subject covers that corner: the
    flood reaches nothing, everything reads as a hole, and the whole frame becomes subject.
    (GreetingCard sel060, a close-up of the card's back: 100% mask with the couch in it.)"""
    import cv2
    h, w = m.shape
    ff = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 255)
    return m | cv2.bitwise_not(ff)[1:-1, 1:-1]


def vision_mask(seg, prior, a):
    """Pick the subject out of Vision's instances. -> (mask uint8, instances, selected, why_fell_back).

    `prior` is the geometric silhouette. Each instance's hard area (soft >= 0.5) is tested against
    it; instances mostly inside are the subject, the rest (a box beside it, a hand) are dropped.
    The union of the chosen soft masks is thresholded to a hard edge, its holes filled, optionally
    grown, and feathered by a Gaussian of --feather-px: an opaque subject gets a hard alpha with an
    anti-aliased rim, not a wide soft matte."""
    import cv2
    h, w = prior.shape
    if "error" in seg:
        return prior, 0, 0, "vision error: " + str(seg["error"])[:80]
    pri = prior > 0
    chosen = []
    for k in range(1, int(seg.get("instances", 0)) + 1):
        sm = cv2.imread(os.path.join(seg["dir"], f"{k}.png"), cv2.IMREAD_GRAYSCALE)
        if sm is None:
            continue
        if sm.shape != (h, w):
            sm = cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR)
        hard = sm >= 128
        area = int(hard.sum())
        if area and (hard & pri).sum() / area >= a.select_frac:
            chosen.append(sm)
    n = int(seg.get("instances", 0))
    if not chosen:
        return prior, n, 0, ("nothing inside the geometric mask" if n else "no foreground found")
    m = np.where(np.maximum.reduce(chosen) >= 128, 255, 0).astype(np.uint8)
    m = fill_holes(m)
    if a.grow_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.grow_px + 1, 2 * a.grow_px + 1))
        m = cv2.dilate(m, k)
    if a.feather_px > 0:
        m = cv2.GaussianBlur(m, (0, 0), a.feather_px)
    return m, n, len(chosen), None


def auto_radius(G, subject):
    """The subject's radius in the solve's own units, from the camera orbit alone.

    You frame what you shoot: at the median camera-to-centre distance d, a lens with focal fx over
    an image W wide sees a half-width of d*(W/2)/fx, and the subject fills FRAME_FILL of it. Every
    term is in the solve's units or in pixels, so this holds whether or not the solve is metric."""
    K = G["K"].astype(np.float64)
    R = G["R"].astype(np.float64)
    t = G["t"].astype(np.float64)
    C = -np.einsum("nji,nj->ni", R, t)                    # camera centres, -R^T t
    d = float(np.median(np.linalg.norm(C - subject, axis=1)))
    W = float(np.median(G["wh"][:, 0])) if "wh" in G.files else float(G["w"])
    fx = float(np.median(K[:, 0, 0]))
    return FRAME_FILL * d * (W / 2.0) / fx


def read_ply_cloud(path):
    """x,y,z, sigmoid(opacity), exp(max scale) from a binary_little_endian Gaussian .ply."""
    with open(path, "rb") as f:
        raw = f.read()
    k = raw.find(b"end_header\n")
    head = raw[:k].decode("ascii", "replace")
    n = int([l for l in head.split("\n") if l.startswith("element vertex")][0].split()[2])
    # not every Gaussian .ply is all-float: a COLMAP points3D.ply carries uchar colours, and
    # reading those as float32 silently shifts every row
    tmap = {"float": "f4", "float32": "f4", "double": "f8", "uchar": "u1", "uint8": "u1",
            "char": "i1", "short": "i2", "ushort": "u2", "int": "i4", "uint": "u4"}
    fields = []
    for line in head.split("\n"):
        w = line.split()
        if len(w) == 3 and w[0] == "property":
            fields.append((w[2], tmap.get(w[1], "f4")))
    arr = np.frombuffer(raw[k + len(b"end_header\n"):], dtype=np.dtype(fields), count=n)
    have = arr.dtype.names
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], 1).astype(np.float64)
    opa = (1.0 / (1.0 + np.exp(-arr["opacity"].astype(np.float64)))) if "opacity" in have else np.ones(n)
    sc = [k_ for k_ in ("scale_0", "scale_1", "scale_2") if k_ in have]
    scale = np.exp(np.stack([arr[k_] for k_ in sc], 1).astype(np.float64)).max(1) if sc else np.full(n, 0.002)
    return xyz, opa, scale


def run(a, pj):
    import cv2
    pj.require(STAGE)
    rig = pj.rig_npz
    if not os.path.exists(rig):
        raise events.StageError("no train/dataset/rig.npz", hint="hs solve first")
    pj.acquire(STAGE)
    st = pj.stage(STAGE)
    st.update({"status": "running", "started": now_iso(), "finished": None, "argv": list(sys.argv),
               "metrics": {}, "checks": [], "artifacts": [], "error": None})   # a past failure must not outlive a success
    pj.save()

    G = np.load(rig, allow_pickle=True)
    names = [str(x) for x in G["names"]]
    K, R, t = (G[k].astype(np.float64) for k in ("K", "R", "t"))
    wh = G["wh"] if "wh" in G.files else None
    pts = G["pts"].astype(np.float64)
    subject = np.median(pts, axis=0)                      # mm, same centre prune and views use
    fitted = auto_radius(G, subject)
    if a.radius is not None:
        r_units, r_src = a.radius * 1000.0, "given"
    else:
        r_units, r_src = fitted * a.radius_scale, "fitted to the camera orbit"
    margin_units = a.margin_mm if a.margin_mm is not None else a.margin_frac * r_units
    point_units = a.point_mm if a.point_mm is not None else POINT_FRAC * r_units
    scaled = not any(c.get("name") == "scene_scaled" and not c.get("ok")
                     for c in pj.stage("solve").get("checks", []))

    events.start(STAGE, "cloud")
    if a.from_points:
        xyz_mm, opa, scale_m = pts, np.ones(len(pts)), np.full(len(pts), point_units / 1000.0)
        src = "sparse SfM points"
    else:
        ply = a.ply
        if not ply and pj.status("prune") == "done":
            # a pruned cloud makes a tighter silhouette, but only while it belongs to this
            # solve; a stale prune folder is geometry from a frame that no longer exists
            pdir = pj.path("prune")
            cands = sorted(f for f in os.listdir(pdir) if f.endswith(".ply")) if os.path.isdir(pdir) else []
            ply = os.path.join(pdir, cands[-1]) if cands else None
        if not ply:
            from .prune import final_export
            ply = final_export(pj)
        if not ply or not os.path.exists(ply):
            raise events.StageError("no .ply to build silhouettes from",
                                    hint="hs train (or hs prune) first, or pass --from-points")
        xyz, opa, scale_m = read_ply_cloud(ply)
        xyz_mm = xyz * 1000.0
        src = pj.rel(ply) if ply.startswith(pj.root) else ply
    dist = np.linalg.norm(xyz_mm - subject, axis=1)
    keep = (opa >= a.min_opacity) & (dist <= r_units)
    idx = np.flatnonzero(keep)
    available = int(len(idx))
    if len(idx) > a.max_points:
        # a seeded sample, so the same inputs give the same masks; the close and fill steps bridge it
        idx = np.sort(np.random.default_rng(0).choice(idx, a.max_points, replace=False))
    P, S = xyz_mm[idx], scale_m[idx]
    if len(P) < 100:
        dv = np.sort(dist[opa >= a.min_opacity])
        where = (f"the nearest splats sit {dv[0] / r_units:.2f}x, the 1,000th {dv[min(999, len(dv) - 1)] / r_units:.2f}x "
                 f"and the 20,000th {dv[min(19999, len(dv) - 1)] / r_units:.2f}x the radius out" if len(dv) else "no splats at all")
        raise events.StageError(f"only {len(P)} points inside the radius ({r_src}); {where}",
                                hint="raise --radius-scale past those multiples, or lower --min-opacity")
    pj.metric(STAGE, "source", src)
    pj.metric(STAGE, "points_used", int(len(P)))
    pj.metric(STAGE, "points_available", available)
    pj.metric(STAGE, "radius_m", round(r_units / 1000.0, 5))
    pj.metric(STAGE, "radius_source", r_src)
    pj.metric(STAGE, "radius_scale", a.radius_scale if a.radius is None else None)
    pj.metric(STAGE, "fitted_radius_m", round(fitted / 1000.0, 5))
    pj.metric(STAGE, "margin_mm", round(margin_units, 3))
    pj.metric(STAGE, "scene_scaled", scaled)   # False: every _m/_mm above is in the solve's own units

    method = getattr(a, "method", "geometry")
    pj.metric(STAGE, "method", method)
    segs = {}
    if method == "vision":
        from .. import segment
        events.start(STAGE, "segment")
        todo = []
        for v, name in enumerate(names):
            img = os.path.join(pj.dataset_dir, "images", name[-1], name[:-2] + ".jpg")
            if os.path.exists(img):
                todo.append((v, img))
        out = segment.run(STAGE, [p for _v, p in todo], pj.path("masks_vision"),
                          progress=lambda d, n: events.progress(STAGE, d, n, step="segment"))
        segs = {v: r for (v, _p), r in zip(todo, out)}

    events.start(STAGE, "project")
    mdir = os.path.join(pj.dataset_dir, "masks")
    cover, previews, empty = [], [], []
    prior_cover, n_inst, n_sel, fell_back = [], [], [], []
    for v, name in enumerate(names):
        eye = name[-1]
        cap = name[:-2]
        img = os.path.join(pj.dataset_dir, "images", eye, cap + ".jpg")
        if not os.path.exists(img):
            continue
        w, h = (int(wh[v][0]), int(wh[v][1])) if wh is not None else (int(G["w"]), int(G["h"]))
        Xc = (R[v] @ P.T).T + t[v]
        z = Xc[:, 2]
        ok = z > 1.0
        uv = (K[v] @ Xc[ok].T).T
        uv = uv[:, :2] / uv[:, 2:3]
        zz = z[ok]
        fx = K[v][0, 0]
        rad = np.clip(fx * S[ok] * 1000.0 / zz, 1.5, 40.0)   # the splat's own footprint, in px
        m = np.zeros((h, w), np.uint8)
        for (u, vv), rr in zip(uv, rad):
            if -50 <= u < w + 50 and -50 <= vv < h + 50:
                cv2.circle(m, (int(u), int(vv)), int(rr), 255, -1)
        if a.close_px > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (a.close_px, a.close_px))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        m = fill_holes(m)
        if a.keep_largest:
            # the subject is one object; detached blobs are haze and table caught by the radius
            nlab, lab, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
            if nlab > 1:
                big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                m = np.where(lab == big, 255, 0).astype(np.uint8)
        margin_px = int(round(fx * margin_units / float(np.median(zz))))   # world -> px at the subject
        if margin_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin_px + 1, 2 * margin_px + 1))
            m = cv2.dilate(m, k)
        prior = m
        if method == "vision":
            m, ni, ns, why = vision_mask(segs.get(v, {"error": "not segmented"}), prior, a)
            prior_cover.append(float((prior > 0).mean()))
            n_inst.append(ni)
            n_sel.append(ns)
            if why:
                fell_back.append((name, why))
        frac = float((m >= 128).mean())
        cover.append(frac)
        if frac < 0.01:
            empty.append(name)
        os.makedirs(os.path.join(mdir, eye), exist_ok=True)
        cv2.imwrite(os.path.join(mdir, eye, cap + ".png"), m)
        if eye == "L" and len(previews) < a.preview and v % max(1, len(names) // (2 * a.preview)) == 0:
            im = cv2.imread(img)
            mb = np.where(m >= 128, 255, 0).astype(np.uint8)
            if prior is not m:     # the geometric region, in magenta, under the object mask
                pe = cv2.dilate(prior, np.ones((3, 3), np.uint8)) - cv2.erode(prior, np.ones((3, 3), np.uint8))
                im[pe > 0] = (255, 0, 255)
            edge = cv2.dilate(mb, np.ones((3, 3), np.uint8)) - cv2.erode(mb, np.ones((3, 3), np.uint8))
            im[edge > 0] = (0, 255, 255)
            im = (im * (0.35 + 0.65 * (mb[:, :, None] > 0))).astype(np.uint8)
            previews.append(cv2.resize(im, (im.shape[1] // 3, im.shape[0] // 3)))
        events.progress(STAGE, v + 1, len(names), step="project")

    cover = np.array(cover)
    pj.metric(STAGE, "views", int(len(cover)))
    pj.metric(STAGE, "coverage_median", round(float(np.median(cover)), 4))
    pj.metric(STAGE, "coverage_min", round(float(cover.min()), 4))
    pj.metric(STAGE, "coverage_max", round(float(cover.max()), 4))
    pj.artifact(STAGE, mdir, "masks")
    if previews:
        sheet = np.hstack(previews[:a.preview])
        sp = pj.path("select", "masks_preview.jpg") if os.path.isdir(pj.path("select")) else pj.path("masks_preview.jpg")
        cv2.imwrite(sp, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        pj.artifact(STAGE, sp, "image")

    if method == "vision":
        pj.metric(STAGE, "prior_coverage_median", round(float(np.median(prior_cover)), 4))
        pj.metric(STAGE, "vision_instances_median", float(np.median(n_inst)))
        pj.metric(STAGE, "vision_selected_median", float(np.median(n_sel)))
        pj.metric(STAGE, "vision_fell_back", len(fell_back))
        if fell_back:
            pj.metric(STAGE, "vision_fell_back_views", [f"{n}: {w}" for n, w in fell_back[:12]])
        pj.check(STAGE, "vision_found_the_subject", len(fell_back) <= 0.1 * len(cover), needs_human=True,
                 value=(f"object mask in {len(cover) - len(fell_back)} of {len(cover)} views; "
                        f"{len(fell_back)} fell back to the geometric mask"
                        + (f" (e.g. {fell_back[0][0]}: {fell_back[0][1]})" if fell_back else "")
                        + f"; subject {100 * np.median(cover):.1f}% of frame vs region "
                          f"{100 * np.median(prior_cover):.1f}%"))
    pj.check(STAGE, "every_view_has_a_silhouette", not empty,
             value="all views covered" if not empty else f"{len(empty)} views nearly empty: {empty[:5]}")
    pj.check(STAGE, "coverage_sane", bool(0.02 <= np.median(cover) <= 0.75),
             value=f"subject covers {100 * np.median(cover):.1f}% of frame (median; {100 * cover.min():.1f}–{100 * cover.max():.1f}%)",
             needs_human=True)
    for s in ("train", "prune", "render", "views"):
        if pj.status(s) in ("done", "failed", "running"):
            pj.m["stages"][s]["status"] = "stale"
    st["status"] = "done"
    st["finished"] = now_iso()
    pj.save()
    pj.release()
