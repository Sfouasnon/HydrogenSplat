"""Reproject overlay: put the solve's own points on the photograph, and on the model's render of
the same pose, and measure how far the render sits from the photograph.

Two different faults look the same in a render that is "off":

  the solve is wrong        the sparse points this image observed do not land on the features
                            they were triangulated from -- the pose (or the points) are bad
                            before any splat is trained. Seen here as a large keypoint-to-
                            projection residual on the photograph.
  training moved the model  the sparse points sit on their features, but the render is shifted
                            against the photograph by a near-uniform amount -- the splats (or the
                            pose Brush used) disagree with a solve that is itself fine.

The shift is phase correlation on grey (cv2.phaseCorrelate, Hanning window): first on both
images downsampled x4, so a shift of tens of pixels is still inside the correlation peak, then
at full resolution on the render moved back by that estimate, which adds the sub-pixel residual.
The same is done per quadrant; four quadrant shifts that agree with the global one are a
uniform shift, quadrants that disagree are local geometry (hs views' patch map says where).

Sign: ``shift_px = [dx, dy]`` is where the render's content sits relative to the photograph's
(+x right, +y down): a render drawn 5 px right of the photograph reads [+5, 0].

Used by ``engine/tools/reproject_overlay.py`` (one capture, any render) and ``hs views --overlay``
(every scored view, on the renders views already made).
"""
import json
import os

import numpy as np

# a solve whose sparse points miss their own keypoints by more than this (median px) is suspect
SPARSE_RESIDUAL_PX = 2.0
# a global shift above this, with quadrants agreeing, is a training / pose-in-training fault
SHIFT_PX = 1.0


def _grey(img):
    import cv2
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return g.astype(np.float32)


def _pc(a, b):
    import cv2
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
    return float(dx), float(dy), float(resp)


def phase_shift(gt, ren, factor=4):
    """(dx, dy, response): where ``ren`` sits relative to ``gt``, coarse x``factor`` then refined."""
    import cv2
    a, b = _grey(gt), _grey(ren)
    if b.shape != a.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    cx = cy = 0.0
    if factor > 1 and min(a.shape) >= 32 * factor:
        size = (a.shape[1] // factor, a.shape[0] // factor)
        sa = cv2.resize(a, size, interpolation=cv2.INTER_AREA)
        sb = cv2.resize(b, size, interpolation=cv2.INTER_AREA)
        cx, cy, _ = _pc(sa, sb)
        cx, cy = cx * factor, cy * factor
    # Move the render back by the coarse estimate ROUNDED to whole pixels, so no interpolation
    # blurs it; the fine pass then measures a residual under a pixel. Shifting back by the
    # fractional estimate instead cost ~0.2 px: phaseCorrelate's sub-pixel peak fit is biased
    # towards zero, and a bilinear resample of the render is not the render shifted.
    ix, iy = int(round(cx)), int(round(cy))
    back = cv2.warpAffine(b, np.float32([[1, 0, -ix], [0, 1, -iy]]), (a.shape[1], a.shape[0]),
                          flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    fx, fy, resp = _pc(a, back)
    return ix + fx, iy + fy, resp


def quadrant_shifts(gt, ren, factor=4):
    """{'TL': [dx, dy, response], 'TR': ..., 'BL': ..., 'BR': ...}"""
    a, b = _grey(gt), _grey(ren)
    h, w = a.shape
    out = {}
    for key, (ys, xs) in {"TL": (slice(0, h // 2), slice(0, w // 2)), "TR": (slice(0, h // 2), slice(w // 2, w)),
                          "BL": (slice(h // 2, h), slice(0, w // 2)), "BR": (slice(h // 2, h), slice(w // 2, w))}.items():
        dx, dy, r = phase_shift(a[ys, xs], b[ys, xs], factor)
        out[key] = [round(dx, 3), round(dy, 3), round(r, 3)]
    return out


def project(K, R, t, X):
    """world (N,3) mm -> (uv (N,2), z (N,)) with rig.npz's Xc = X @ R.T + t."""
    Xc = np.asarray(X, float) @ np.asarray(R, float).T + np.asarray(t, float)
    z = Xc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0][0] * Xc[:, 0] / z + K[0][2]
        v = K[1][1] * Xc[:, 1] / z + K[1][2]
    return np.stack([u, v], 1), z


def _lines(p):
    return [l.rstrip("\n") for l in open(p) if not l.startswith("#")]


def sparse_observations(sparse_dir, image_name):
    """(keypoints (n,2) px, xyz (n,3) mm) for the points ``image_name`` (e.g. 'L/cap004.jpg')
    observes in a COLMAP model, or None when there is no model or no such image.

    Reads images.txt / points3D.txt (hs solve writes both beside the .bin); falls back to
    pycolmap for a binary-only model. The dataset's sparse model is in metres, rig.npz in mm."""
    if not os.path.isdir(sparse_dir):
        return None
    it, pt = os.path.join(sparse_dir, "images.txt"), os.path.join(sparse_dir, "points3D.txt")
    if os.path.exists(it) and os.path.exists(pt):
        xyz = {}
        for l in _lines(pt):
            f = l.split()
            if len(f) >= 4:
                xyz[int(f[0])] = [float(f[1]), float(f[2]), float(f[3])]
        il = _lines(it)
        k = 0
        while k < len(il):
            h = il[k].split()
            if not h:
                k += 1
                continue
            if len(h) >= 10 and h[9] == image_name:
                obs = np.array(il[k + 1].split(), float).reshape(-1, 3) if k + 1 < len(il) and il[k + 1].strip() else np.zeros((0, 3))
                keep = [(u, v, int(p)) for u, v, p in obs if int(p) >= 0 and int(p) in xyz]
                if not keep:
                    return np.zeros((0, 2)), np.zeros((0, 3))
                return (np.array([[u, v] for u, v, _ in keep]),
                        np.array([xyz[p] for _, _, p in keep]) * 1000.0)
            k += 2
        return None
    try:
        import pycolmap
        rec = pycolmap.Reconstruction(sparse_dir)
    except Exception:
        return None
    for im in rec.images.values():
        if im.name == image_name:
            kp, X = [], []
            for p2 in im.points2D:
                if p2.has_point3D():
                    kp.append(np.asarray(p2.xy))
                    X.append(np.asarray(rec.points3D[p2.point3D_id].xyz))
            return np.array(kp).reshape(-1, 2), np.array(X).reshape(-1, 3) * 1000.0
    return None


def points_for_view(G, v, dataset_dir, eye):
    """What to draw for view index ``v`` of rig.npz ``G``.

    -> {"uv": (n,2) projected through rig.npz, "keypoints": (n,2) or None, "source": str}
    With a sparse model that has tracks, only the points this image observes (and their
    keypoints, so the residual can be drawn); otherwise every rig.npz point in front of the
    camera and inside the canvas."""
    from . import rig
    names = [str(x) for x in G["names"]]
    K, R, t = G["K"][v].astype(float), G["R"][v].astype(float), G["t"][v].astype(float)
    w, h = (int(G["wh"][v][0]), int(G["wh"][v][1])) if "wh" in G.files else (int(G["w"]), int(G["h"]))
    cap = rig.capture_name(names[v])
    obs = None
    for ext in (".jpg", ".jpeg", ".png"):
        obs = sparse_observations(os.path.join(dataset_dir, "sparse"), f"{eye}/{cap}{ext}")
        if obs is not None:
            break
    if obs is not None and len(obs[0]):
        uv, _ = project(K, R, t, obs[1])
        return {"uv": uv, "keypoints": obs[0], "source": "sparse_track"}
    uv, z = project(K, R, t, G["pts"].astype(float))
    ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return {"uv": uv[ok], "keypoints": None, "source": "rig_pts"}


def _draw(img, uv, keypoints=None, max_points=6000):
    import cv2
    out = img.copy()
    n = len(uv)
    idx = np.arange(n) if n <= max_points else np.random.default_rng(0).choice(n, max_points, replace=False)
    r = max(2, int(round(min(img.shape[:2]) / 400)))
    for i in idx:
        p = (int(round(uv[i][0])), int(round(uv[i][1])))
        if keypoints is not None:
            q = (int(round(keypoints[i][0])), int(round(keypoints[i][1])))
            cv2.line(out, q, p, (0, 0, 255), 1, cv2.LINE_AA)            # keypoint -> reprojection
        cv2.circle(out, p, r, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def _label(img, text):
    import cv2
    out = cv2.copyMakeBorder(img, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(out, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def diagnose(sparse_px, shift, spread):
    if sparse_px is not None and sparse_px > SPARSE_RESIDUAL_PX:
        return (f"solve: the sparse points miss their own keypoints by {sparse_px:.1f} px (median) -- "
                "the pose or the points are wrong before training")
    if shift is None:
        return "no render to compare" + ("" if sparse_px is None else f"; sparse points land within {sparse_px:.2f} px")
    mag = float(np.hypot(*shift))
    if mag <= SHIFT_PX and spread <= SHIFT_PX:
        return f"registered: render within {mag:.2f} px of the photograph"
    if spread <= max(0.5, 0.3 * mag):
        return (f"uniform {mag:.2f} px shift with well-placed sparse points: the splats or the pose used in "
                "training disagree with the solve")
    return (f"non-uniform: quadrants disagree by {spread:.2f} px (global {mag:.2f} px) -- local geometry, "
            "see hs views' patch map")


def make(gt_bgr, ren_bgr, pts, out_dir, label=""):
    """Write overlay_gt.jpg, overlay_render.jpg (when there is a render) and report.json.
    ``pts`` is points_for_view()'s dict. Returns the report."""
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    uv, kp = pts["uv"], pts.get("keypoints")
    rep = {"n_points": int(len(uv)), "points_source": pts.get("source")}
    sparse_px = None
    if kp is not None and len(kp):
        res = np.linalg.norm(uv - kp, axis=1)
        sparse_px = float(np.median(res))
        rep["sparse_residual_px"] = {"median": round(sparse_px, 3), "p90": round(float(np.percentile(res, 90)), 3)}
    gt_img = _draw(gt_bgr, uv, kp)
    cv2.imwrite(os.path.join(out_dir, "overlay_gt.jpg"),
                _label(gt_img, f"{label}  photograph: {len(uv)} points ({pts.get('source')}), green = reprojected"
                               + (", red = to its keypoint" if kp is not None else "")),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    shift, spread = None, 0.0
    if ren_bgr is not None:
        if ren_bgr.shape[:2] != gt_bgr.shape[:2]:
            ren_bgr = cv2.resize(ren_bgr, (gt_bgr.shape[1], gt_bgr.shape[0]), interpolation=cv2.INTER_AREA)
        dx, dy, resp = phase_shift(gt_bgr, ren_bgr)
        q = quadrant_shifts(gt_bgr, ren_bgr)
        shift = (dx, dy)
        spread = float(max(np.hypot(v[0] - dx, v[1] - dy) for v in q.values()))
        rep.update({"shift_px": [round(dx, 3), round(dy, 3)], "shift_mag_px": round(float(np.hypot(dx, dy)), 3),
                    "response": round(resp, 3), "quadrant_shifts": q, "quadrant_spread_px": round(spread, 3)})
        cv2.imwrite(os.path.join(out_dir, "overlay_render.jpg"),
                    _label(_draw(ren_bgr, uv), f"{label}  render, same points: shift {dx:+.2f},{dy:+.2f} px "
                                               f"(response {resp:.2f})"),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    rep["diagnosis"] = diagnose(sparse_px, shift, spread)
    json.dump(rep, open(os.path.join(out_dir, "report.json"), "w"), indent=1)
    return rep
