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

No segmentation model is needed. The solve already knows where the subject is: take the splats
(or SfM points) within `--radius` of the subject centre, project them into every view with that
view's own K, R, t, splat each as a disc the size of its own projected footprint, close the
gaps, fill, and dilate by a world-space margin so the silhouette errs outward. A mask that is a
little too generous costs some background supervision; a mask that clips the subject removes
real observations, so the margin is deliberately one-sided.

Writes train/dataset/masks/{L,R}/capNNN.png next to images/{L,R}/capNNN.jpg, plus a preview
sheet. Marks train stale: the dataset changed.
"""
import os
import sys

import numpy as np

from .. import events
from ..project import now_iso

STAGE = "masks"


def add_parser(sub):
    p = sub.add_parser("masks", help="per-view subject silhouettes for Brush's mask channel")
    p.add_argument("--radius", type=float, default=0.12, help="metres about the subject centre")
    p.add_argument("--ply", default=None, help="default: the prune output if present, else the final export")
    p.add_argument("--min-opacity", type=float, default=0.1)
    p.add_argument("--margin-mm", type=float, default=5.0, help="dilate the silhouette outward by this much world space")
    p.add_argument("--close-px", type=int, default=25, help="morphological close, to bridge gaps between splats")
    p.add_argument("--keep-largest", action="store_true", default=True,
                   help="keep only the largest connected silhouette (default); --no-keep-largest to disable")
    p.add_argument("--no-keep-largest", dest="keep_largest", action="store_false")
    p.add_argument("--preview", type=int, default=6, help="views in the preview sheet")
    p.add_argument("--from-points", action="store_true", help="use the sparse SfM points instead of a .ply")
    p.add_argument("--point-mm", type=float, default=2.0,
                   help="--from-points: world footprint drawn per SfM point. The sparse cloud is thin and\n                        scattered, so a large disc inflates the silhouette; the close step bridges the gaps")
    return p


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
               "metrics": {}, "checks": [], "artifacts": []})
    pj.save()

    G = np.load(rig, allow_pickle=True)
    names = [str(x) for x in G["names"]]
    K, R, t = (G[k].astype(np.float64) for k in ("K", "R", "t"))
    wh = G["wh"] if "wh" in G.files else None
    pts = G["pts"].astype(np.float64)
    subject = np.median(pts, axis=0)                      # mm, same centre prune and views use

    events.start(STAGE, "cloud")
    if a.from_points:
        xyz_mm, opa, scale_m = pts, np.ones(len(pts)), np.full(len(pts), a.point_mm / 1000.0)
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
    keep = (opa >= a.min_opacity) & (np.linalg.norm(xyz_mm - subject, axis=1) <= a.radius * 1000.0)
    P, S = xyz_mm[keep], scale_m[keep]
    if len(P) < 100:
        raise events.StageError(f"only {len(P)} points inside {a.radius:g} m of the subject",
                                hint="raise --radius, or lower --min-opacity")
    pj.metric(STAGE, "source", src)
    pj.metric(STAGE, "points_used", int(len(P)))
    pj.metric(STAGE, "radius_m", a.radius)

    events.start(STAGE, "project")
    mdir = os.path.join(pj.dataset_dir, "masks")
    cover, previews, empty = [], [], []
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
        # fill interior holes: flood from the border, invert what the flood could not reach
        ff = m.copy()
        cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
        m = m | cv2.bitwise_not(ff)
        if a.keep_largest:
            # the subject is one object; detached blobs are haze and table caught by the radius
            nlab, lab, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
            if nlab > 1:
                big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                m = np.where(lab == big, 255, 0).astype(np.uint8)
        margin_px = int(round(fx * a.margin_mm / float(np.median(zz))))   # mm -> px at the subject
        if margin_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin_px + 1, 2 * margin_px + 1))
            m = cv2.dilate(m, k)
        frac = float((m > 0).mean())
        cover.append(frac)
        if frac < 0.01:
            empty.append(name)
        os.makedirs(os.path.join(mdir, eye), exist_ok=True)
        cv2.imwrite(os.path.join(mdir, eye, cap + ".png"), m)
        if eye == "L" and len(previews) < a.preview and v % max(1, len(names) // (2 * a.preview)) == 0:
            im = cv2.imread(img)
            edge = cv2.dilate(m, np.ones((3, 3), np.uint8)) - cv2.erode(m, np.ones((3, 3), np.uint8))
            im[edge > 0] = (0, 255, 255)
            im = (im * (0.35 + 0.65 * (m[:, :, None] > 0))).astype(np.uint8)
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
