"""Silhouette registration: a LiDAR scan's subject fitted to the per-view subject masks.

`hs scale --lidar SCAN --init silhouette` (stages/scale.py) is the caller. It exists for the scenes
the geometric registration cannot do: a glossy, uniform or thin subject leaves (almost) no points
in the SfM cloud, so the overlap between the solve and the scan is the floor or a lawn — a plane,
which fixes no scale and no horizontal position. The subject is still in every photograph, and
`hs masks` has its outline. So the scan's subject points are projected through the solve's own
cameras (rig.npz K, R, t) and the similarity scan -> solve is chosen to make the projections
cover the masks.

Found on CirclesSculpture (2026-09-23; H1 stereo, 373 captures at 3-5 m from a 2 m glossy red
ring on a lawn): the sparse cloud was the lawn and the trees, the registration was honestly
ambiguous, and this fit gave the scale — 5.81 (the solve 5.81x too small), yaw 84 deg, mean IoU
0.49 over 14 views.

Model (all lengths in mm on the scan side, the solve's units on the other):

    X_solve = a * (X_scan - c) @ R.T + g + b,   R = Rbase @ Rot(up_scan, yaw) [@ tilts],

where c is the scan subject's centroid, g the solve's subject guess, Rbase the rotation taking
the scan's up axis onto the solve's up (coverage's mean-camera up): gravity is locked, so the
parameters are yaw, log a and b (three), plus two small tilts with ``tilt=True``. Off by default:
on Circles a free tilt overfitted the masks (IoU 0.50 against 0.49) and tipped the ground so the
cameras went through the lawn. ``s = 1 / a`` is the solve -> scan scale, the number `hs scale`
applies.

Cost: 1 - mean over views of IoU(projected points rasterised at 1/4 resolution and dilated 2 px,
mask at 1/4 resolution). Search: a coarse grid over yaw (4 deg) x scale (12 log-spaced values)
with b = 0, then Nelder-Mead from the best, then a second Nelder-Mead over the views with
IoU >= 0.3 (a view whose pose or mask is wrong otherwise drags the fit). Deterministic.

Scan subject (`scan_subject`): the scan's ground plane (lidar.ground_plane along its up axis),
the points more than `above_mm` above it, and of those the largest 26-connected cluster on a
100 mm voxel grid (scipy.ndimage.label) — on Circles the ring on its plinth.
"""
import os

import numpy as np

from . import lidar as lidarlib

DS = 4                     # masks and projections at 1/4 resolution
DILATE_PX = 2
KEEP_IOU = 0.3             # second pass: views at least this good after the first
YAW_STEP_DEG = 4.0
SCALE_GRID_METRIC = np.geomspace(0.5, 10.0, 12)   # a nominally metric solve (stereo, or scaled): off by 0.5-10x
SCALE_GRID_PRIOR = np.geomspace(0.25, 4.0, 12)    # an unscaled solve: about the apparent-size prior


# ------------------------------------------------------------------------ rotations

def rot(axis, deg):
    """Rodrigues: rotation by `deg` about `axis`."""
    a = np.radians(deg)
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    Kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * Kx + (1 - np.cos(a)) * Kx @ Kx


def rot_a_to_b(a, b):
    """The shortest rotation taking unit vector a onto unit vector b."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b)
    s, c = np.linalg.norm(v), float(a @ b)
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        p = np.cross(a, [1.0, 0, 0])                  # opposite: half a turn about any perpendicular
        if np.linalg.norm(p) < 1e-6:
            p = np.cross(a, [0, 1.0, 0])
        return rot(p, 180.0)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


# ------------------------------------------------------------------------ scan subject

RISE_LEVELS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)   # x above_mm: the cuts `scan_subject` tries
RISE_STABLE = 0.85         # ...stopping where the next cut keeps this share of the footprint


def _largest_cluster(P, h, cut, voxel_mm):
    """Largest 26-connected cluster of the points more than `cut` above the ground, on a voxel
    grid. -> (indices into P, voxel used, n clusters, top-5 sizes, footprint in voxel columns)."""
    from scipy import ndimage
    above = np.flatnonzero(h > cut)
    if len(above) < 50:
        return None
    A = P[above]
    v = float(voxel_mm)
    lo = A.min(axis=0)
    while True:                                    # a scan of a field: coarsen rather than allocate gigabytes
        ijk = np.floor((A - lo) / v).astype(np.int64)
        shape = ijk.max(axis=0) + 1
        if np.prod(shape.astype(np.float64)) <= 64e6:
            break
        v *= 1.5
    occ = np.zeros(shape, bool)
    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    lab, n_lab = ndimage.label(occ, structure=np.ones((3, 3, 3), bool))
    pl = lab[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    counts = np.bincount(pl, minlength=n_lab + 1)
    counts[0] = 0
    best = int(np.argmax(counts))
    sel = pl == best
    # footprint: occupied voxel columns seen from above (the grid axis closest to up is dropped)
    return {"idx": above[sel], "voxel_mm": v, "clusters": int(n_lab), "above_points": int(len(above)),
            "top": sorted(counts[1:].tolist(), reverse=True)[:5], "ijk": ijk[sel]}


def scan_subject(points, up, above_mm=150.0, voxel_mm=100.0, n_fit=6000, seed=0, ground=None, rise=True):
    """The scan's subject: the largest connected cluster above the ground plane.

    points (N, 3) mm; up: the scan's gravity axis (unit, file frame). The ground is
    lidar.ground_plane along `up`; the points more than `above_mm` above it are clustered on a
    `voxel_mm` grid (26-connectivity) and the largest cluster is the subject.

    `rise`: a plinth, a slab or a lawn that rolls by more than `above_mm` joins the subject at
    the first cut (Circles at 150 mm: a 6.2 x 5.7 m "subject" of lawn humps and plinth around a
    2.9 m ring; the fit on it came out 6.45 at IoU 0.37). So the cut is raised through
    RISE_LEVELS x above_mm while the cluster's footprint (occupied voxel columns seen from
    above) keeps shrinking, and stops at the first cut whose next one keeps >= RISE_STABLE of it
    — what is left is standing up, not lying on the ground (Circles: 2154, 1492, 307, 219, 217
    columns at 150, 225, 300, 375, 450 mm: the cut is 375 mm, the ring alone). A cut never goes
    past half the cluster's height. A box or a ball stops at the first cut.

    -> dict {"points" (the cluster's points), "fit" (a seeded voxel subsample of about n_fit),
    "ground" (the plane), "cut_mm", "levels" [{cut_mm, points, footprint}], "centroid_mm",
    "extent_mm" (along the cluster's principal axes, largest first), "height_mm" (its top above
    the ground), "clusters", ...} or raise ValueError saying why."""
    P = np.asarray(points, np.float64)
    up = np.asarray(up, np.float64) / np.linalg.norm(up)
    g = ground or lidarlib.ground_plane(P, up, seed=seed)
    if g is None:
        raise ValueError("the scan has no ground plane under it (too few points in its lowest band)")
    n, p0 = np.asarray(g["normal"]), np.asarray(g["point_mm"])
    h = (P - p0) @ n
    drop = int(np.argmax(np.abs(up)))                  # the voxel axis nearest to up
    levels, chosen = [], None
    for k in (RISE_LEVELS if rise else (1.0,)):
        cut = float(above_mm) * k
        cl = _largest_cluster(P, h, cut, voxel_mm)
        if cl is None:
            break
        cols = np.delete(cl["ijk"], drop, axis=1)
        fp = int(len(np.unique(cols[:, 0] * (cols[:, 1].max() + 1) + cols[:, 1])))
        top_h = float(np.percentile(h[cl["idx"]], 99.5))
        if levels and cut > 0.5 * top_h:
            break
        rec = {"cut_mm": round(cut, 1), "points": int(len(cl["idx"])), "footprint": fp}
        if levels and fp >= RISE_STABLE * levels[-1]["footprint"]:
            break                                       # stable: the previous cut is the subject
        levels.append(rec)
        chosen = cl
    if chosen is None:
        raise ValueError(f"fewer than 50 scan points stand more than {above_mm:g} mm above its ground")
    S = P[chosen["idx"]]
    fit = S[lidarlib.subsample(S, n_fit, seed=seed)]
    c = S.mean(axis=0)
    _, _, Vt = np.linalg.svd(S - c, full_matrices=False)
    proj = (S - c) @ Vt.T
    ext = np.percentile(proj, 99.5, axis=0) - np.percentile(proj, 0.5, axis=0)
    return {"points": S, "index": chosen["idx"], "fit": fit, "ground": g, "centroid_mm": c.tolist(),
            "cut_mm": levels[-1]["cut_mm"], "levels": levels,
            "extent_mm": [round(float(x), 1) for x in sorted(ext, reverse=True)],
            "height_mm": round(float(np.percentile(h[chosen["idx"]], 99.5)), 1), "voxel_mm": chosen["voxel_mm"],
            "clusters": chosen["clusters"], "largest_clusters_points": chosen["top"],
            "above_points": chosen["above_points"], "above_mm": float(above_mm)}


# ------------------------------------------------------------------------ views

class View:
    """One view: its mask at 1/DS resolution (bool) and the 3x4 projection into that raster."""

    def __init__(self, name, K, R, t, wh, mask_full):
        import cv2
        self.name = name
        mh, mw = mask_full.shape[:2]
        self.mask = cv2.resize((mask_full > 127).astype(np.uint8), (mw // DS, mh // DS),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
        h, w = self.mask.shape
        fx, fy = w / float(wh[0]), h / float(wh[1])    # image pixels -> raster pixels
        S = np.diag([fx, fy, 1.0])
        self.P = S @ np.asarray(K, np.float64) @ np.hstack([np.asarray(R, np.float64), np.asarray(t, np.float64)[:, None]])
        self.R, self.t = np.asarray(R, np.float64), np.asarray(t, np.float64)
        self.C = -self.R.T @ self.t
        self.f = float(np.sqrt(fx * fy * K[0][0] * K[1][1]))    # focal length in raster pixels
        self.mask_px = int(self.mask.sum())

    def raster(self, X):
        """X (n, 3) solve units -> bool raster of the projected points, dilated DILATE_PX."""
        import cv2
        uvw = X @ self.P[:, :3].T + self.P[:, 3]
        z = uvw[:, 2]
        front = z > 1e-6 * max(1.0, float(np.abs(z).max()))
        u = uvw[front, 0] / z[front]
        v = uvw[front, 1] / z[front]
        h, w = self.mask.shape
        ins = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        img = np.zeros((h, w), np.uint8)
        img[v[ins].astype(np.int64), u[ins].astype(np.int64)] = 1
        if DILATE_PX:
            img = cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=DILATE_PX)
        return img.astype(bool)

    def iou(self, X):
        r = self.raster(X)
        inter = np.count_nonzero(r & self.mask)
        union = np.count_nonzero(r | self.mask)
        return inter / union if union else 0.0


# ------------------------------------------------------------------------ the fit

class Model:
    """The parametrisation X_solve = a (X - c) R^T + g + d * q (module docstring). p = (yaw_deg,
    log a, qx, qy, qz[, tilt1_deg, tilt2_deg]); the translation q is in units of `d`, the median
    camera-to-subject distance, so one simplex step means the same on any solve."""

    def __init__(self, scan_up, solve_up, centroid, subject, d, tilt=False):
        self.u = np.asarray(scan_up, float) / np.linalg.norm(scan_up)
        self.base = rot_a_to_b(self.u, solve_up)
        e = np.cross(self.u, [1.0, 0, 0])
        if np.linalg.norm(e) < 1e-6:
            e = np.cross(self.u, [0, 1.0, 0])
        self.e1 = e / np.linalg.norm(e)                  # tilt axes, perpendicular to the scan's up
        self.e2 = np.cross(self.u, self.e1)
        self.c = np.asarray(centroid, float)
        self.g = np.asarray(subject, float)
        self.d = float(d)
        self.tilt = bool(tilt)

    def parts(self, p):
        p = np.asarray(p, float)
        R = self.base @ rot(self.u, p[0])
        if self.tilt:
            R = R @ rot(self.e1, p[5]) @ rot(self.e2, p[6])
        a = float(np.exp(p[1]))
        return a, R, self.g + self.d * p[2:5]

    def apply(self, p, X):
        a, R, b = self.parts(p)
        return a * (np.asarray(X, float) - self.c) @ R.T + b

    def scan_to_solve(self, p):
        """(a, R, c') with X_solve = a R X_scan + c'."""
        a, R, b = self.parts(p)
        return a, R, b - a * R @ self.c

    def solve_to_scan(self, p):
        """(s, R, t) of the stage's convention y_scan = s R x_solve + t."""
        a, R, c = self.scan_to_solve(p)
        return lidarlib.invert((a, R, c))


def size_prior(views, extent_mm, subject, d_views):
    """Apparent-size estimate of the solve -> scan scale: per view, the mask's bounding box against
    the subject's two largest extents at the camera's distance to the subject guess. Rough (a
    ring's mask is its outline, the box its diameter), which is all a grid centre needs.
    -> (median s, per-view list)."""
    e = sorted(extent_mm, reverse=True)
    size_mm = float(np.sqrt(max(e[0], 1.0) * max(e[1], 1.0)))
    out = []
    for vw, dist in zip(views, d_views):
        ys, xs = np.nonzero(vw.mask)
        if len(xs) < 4 or dist <= 0:
            continue
        size_px = np.sqrt((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
        a = size_px / vw.f * dist / size_mm           # solve units per mm
        out.append(1.0 / a)
    return (float(np.median(out)) if out else None), out


def fit(subject_pts, views, scan_up, solve_up, subject, metric=True, tilt=False, progress=None,
        yaw_step=YAW_STEP_DEG, keep_iou=KEEP_IOU, maxiter=2500):
    """Fit the scan subject to the view masks. subject_pts (n, 3) scan mm; views: [View];
    scan_up: the scan's up (file frame); solve_up: coverage's up_world; subject: the solve's
    subject guess (solve units); metric: the solve's units are nominally mm (stereo or already
    scaled): the scale grid is 0.5-10x, else it is 0.25-4x about the apparent-size prior.

    -> {"s", "R", "t" (solve -> scan), "scale_solve_to_scan", "yaw_deg", "tilt_deg", "params",
    "iou" {name: IoU at the result, every view}, "views_used" [names], "iou_mean" (over the used),
    "grid" {best yaw, scale, cost}, "prior", "evaluations", "passes"}."""
    from scipy.optimize import minimize
    X = np.asarray(subject_pts, np.float64)
    c = X.mean(axis=0)
    subject = np.asarray(subject, np.float64)
    d_views = [float(np.linalg.norm(v.C - subject)) for v in views]
    d = float(np.median(d_views)) or 1.0
    model = Model(scan_up, solve_up, c, subject, d, tilt=tilt)
    ext = np.ptp(X - c, axis=0)
    prior, prior_views = size_prior(views, sorted(ext, reverse=True), subject, d_views)
    grid_s = SCALE_GRID_METRIC if metric or not prior else prior * SCALE_GRID_PRIOR
    n_evals = [0]

    def ious(p, sel):
        n_evals[0] += 1
        Y = model.apply(p, X)
        return np.array([views[k].iou(Y) for k in sel])

    def cost(p, sel):
        return 1.0 - float(ious(p, sel).mean())

    allv = list(range(len(views)))
    extra = [0.0, 0.0] if tilt else []
    yaws = np.arange(0.0, 360.0, yaw_step)
    best = None
    total = len(yaws) * len(grid_s)
    k = 0
    for yaw in yaws:
        for sc in grid_s:
            k += 1
            cst = cost([yaw, -np.log(sc), 0, 0, 0] + extra, allv)
            if best is None or cst < best[0]:
                best = (cst, float(yaw), float(sc))
        if progress is not None:
            progress(k, total, "grid")
    p0 = np.array([best[1], -np.log(best[2]), 0, 0, 0] + extra)

    def simplex(p, steps):
        p = np.asarray(p, float)
        return np.array([p] + [p + dv for dv in np.diag(steps)])

    tsteps = [1.0, 1.0] if tilt else []
    res = minimize(cost, p0, args=(allv,), method="Nelder-Mead",
                   options=dict(initial_simplex=simplex(p0, [5, 0.1, 0.07, 0.07, 0.07] + tsteps),
                                xatol=1e-3, fatol=1e-5, maxiter=maxiter))
    if progress is not None:
        progress(1, 2, "refine")
    io1 = ious(res.x, allv)
    sel = [k for k in allv if io1[k] >= keep_iou] or allv
    res2 = minimize(cost, res.x, args=(sel,), method="Nelder-Mead",
                    options=dict(initial_simplex=simplex(res.x, [3, 0.05, 0.04, 0.04, 0.04] + [0.5] * len(tsteps)),
                                 xatol=1e-4, fatol=1e-6, maxiter=maxiter))
    if progress is not None:
        progress(2, 2, "refine")
    p = res2.x
    io = ious(p, allv)
    s, R, t = model.solve_to_scan(p)
    return {"s": float(s), "R": np.asarray(R), "t": np.asarray(t), "scale_solve_to_scan": float(s),
            "yaw_deg": float(p[0] % 360.0), "tilt_deg": [float(x) for x in p[5:7]] if tilt else None,
            "params": [float(x) for x in p], "model": model,
            "iou": {views[k].name: round(float(io[k]), 4) for k in allv},
            "iou_first_pass": {views[k].name: round(float(io1[k]), 4) for k in allv},
            "views_used": [views[k].name for k in sel],
            "iou_mean": float(np.mean(io[sel])) if sel else 0.0,
            "grid": {"yaw_deg": best[1], "scale": best[2], "iou_mean": round(1.0 - best[0], 4),
                     "scales": [round(float(x), 4) for x in grid_s], "yaw_step_deg": yaw_step},
            "prior": {"scale": prior, "per_view": [round(float(x), 4) for x in prior_views], "used": not metric},
            "subject_distance": d, "evaluations": int(n_evals[0]),
            "passes": [{"iterations": int(res.nit), "cost": float(res.fun)},
                       {"iterations": int(res2.nit), "cost": float(res2.fun), "views": len(sel)}]}


def ious_at(views, X_solve):
    """{name: IoU} of points already in the solve frame."""
    return {v.name: round(float(v.iou(X_solve)), 4) for v in views}


# ------------------------------------------------------------------------ the human check

def write_overlay(path, image_path, view, X_solve, iou=None, X_before=None, iou_before=None, max_side=1600):
    """The photograph with the projected subject points in cyan, the mask's outline in magenta and,
    given `X_before` (the silhouette fit's pose, before ICP), those points in yellow underneath."""
    import cv2
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return False
    H, W = img.shape[:2]
    h, w = view.mask.shape
    sx, sy = W / w, H / h                               # raster -> photograph

    def dots(X):
        uvw = np.asarray(X, float) @ view.P[:, :3].T + view.P[:, 3]
        z = uvw[:, 2]
        front = z > 0
        u = uvw[front, 0] / z[front] * sx
        v = uvw[front, 1] / z[front] * sy
        ins = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        d = np.zeros((H, W), np.uint8)
        d[v[ins].astype(np.int64), u[ins].astype(np.int64)] = 255
        return cv2.dilate(d, np.ones((3, 3), np.uint8)) > 0

    out = img.copy()
    if X_before is not None:
        out[dots(X_before)] = (0, 220, 255)             # BGR yellow: the silhouettes' own pose
    out[dots(X_solve)] = (255, 255, 0)                  # BGR cyan: the reported pose
    m = cv2.resize(view.mask.astype(np.uint8) * 255, (W, H), interpolation=cv2.INTER_NEAREST)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (255, 0, 255), 2)
    label = view.name + (f"  IoU {iou:.2f}" if iou is not None else "") + \
        (f" (fit {iou_before:.2f})" if iou_before is not None else "")
    cv2.putText(out, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(out, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2, cv2.LINE_AA)
    k = max_side / max(H, W)
    if k < 1:
        out = cv2.resize(out, (int(W * k), int(H * k)), interpolation=cv2.INTER_AREA)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return bool(cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, 88]))
