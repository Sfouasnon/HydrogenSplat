"""Per-view depth maps from a registered scan: the reference for Brush's depth loss.

`hs train --depth-weight W` (stages/train.py) is the caller. Photometric loss alone explains a
glossy or textureless surface as a translucent veil with opaque content behind it — the reflected
room drawn as a virtual image inside the object (Stormtrooper helmet, 2026-10-01: 31 % of the
opacity more than 10 mm behind the surface, the shell itself a third opaque). A LiDAR scan knows
where the surface is. Each training view gets the scan's depth as that camera sees it, and Brush
(`--depth-loss-weight`, fork branch depth-loss) penalises the rendered expected depth for
leaving it.

Depth is distance along the optical axis (camera z), the same quantity Brush renders. The maps
are small on purpose: a phone scan carries about a point per millimetre, and the depth pass in
Brush renders at the map's own resolution.

Rendering a point cloud as a depth map (`render`):

  1. project every point through the view's K, R, t (rig.npz, mm);
  1b. what the scan cannot know is hidden: with masks the caller first keeps only the points the
     subject's visual hull says this camera sees (`hull_visible`: not behind the subject's
     volume, and facing the camera); without them, points whose normal (`oriented_normals`)
     faces away from the camera are dropped here;
  2. hidden points: a coarse z-buffer (cells of ``COARSE`` px) eroded by one cell is the front
     surface's envelope; points well behind it are the far side of the object seen through the
     gaps between near points, and are dropped. "Well behind" allows for a surface tilted up to
     76 degrees across the envelope's window (``SLOPE_TAN``), so steep flanks survive;
  3. splat the rest: a pixel takes the mean of the front-layer points that land in it, and a
     pixel no point landed in takes the mean of its neighbours within a square footprint sized to
     the scan's point spacing in this view (means, so a slanted surface is not pulled towards
     the camera the way a z-buffer minimum would);
  4. drop pixels on a depth discontinuity (3x3 range above ``EDGE_TOL`` of the depth: silhouettes
     and grazing surfaces, where one pixel mixes two depths), the outermost ring of what is left,
     and patches of fewer than ``MIN_PATCH_PX`` pixels (stray points);
  5. keep only the layer's region when masks are in use (fully inside the subject mask for a
     subject layer, fully outside for a background layer).

`build` does all of it for a training set and writes the files and the report.

0 means "no measurement" throughout. Files are 16-bit PNG, counts of ``unit`` dataset units
(dataset units = rig.npz mm / 1000), named and laid out like the masks so Brush finds them.
"""
import os

import numpy as np

LONG_EDGE = 512          # depth map long edge, px (capped at the photograph's own)
COARSE = 4               # occlusion envelope cell, in depth-map px
OCCLUSION_TOL = 0.03     # a point this far (relative) behind the envelope is hidden, at least
SLOPE_TAN = 4.0          # ...and far enough that a surface tilted 76 deg across the envelope window is kept
EDGE_TOL = 0.05          # 3x3 depth range above this share of the depth: a discontinuity
MAX_FOOTPRINT = 4        # z-buffer footprint radius cap, px
MIN_PATCH_PX = 64        # a connected patch of depth smaller than this is a stray point, not a surface
NEAR_MM = 1.0            # points closer than this to the camera plane are ignored
SPACING_VOXEL_MM = 10.0  # voxel used to estimate the scan's point spacing
FACING_COS = 0.17        # a point whose normal is more than 80 deg off the line of sight is not seen
NORMAL_K = 32            # neighbours for the PCA normals
VOTE_EDGE = 480          # mask long edge for the subject vote, px
VOTE_MIN = 0.7           # a scan point may be the subject's if it is inside the mask in this share of its views
VOTE_SEEN = 0.3          # ...and in frame in at least this share of all views
HULL_RES = 128           # visual-hull grid: voxels along the longest side of the subject's box
HULL_VIEWS = 96          # ...voted by at most this many views
HULL_STEPS = (0.12, 0.24, 0.36)   # line-of-sight samples in front of a point, in subject extents
HULL_SLACK = 0.08        # subject_points: a point within this of the subject's own vote share belongs to it
HULL_VIEW_MIN = 0.9      # a view carves the hull only if its mask holds this share of the subject's points
HULL_SOLID = 0.02        # a voxel within this of the subject's own share (in the carving views) is solid
HULL_NORMAL_STEPS = (2.0, 4.0, 8.0)   # hull_normals: probe distances either side of a point, voxels
HULL_NORMAL_MARGIN = 0.05             # ...and the vote difference that settles which side is outside
HULL_NEAR_FROM = 2.0     # unsettled points: the line of sight is marched from this many voxels out
MASK_INSIDE = 250        # an area-averaged mask value at or above this is "fully inside"
MASK_OUTSIDE = 5
UNITS_MM = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)   # count sizes tried, mm
MAX_COUNT = 65000


def point_spacing_mm(points, voxel=SPACING_VOXEL_MM):
    """Typical distance between neighbouring scan points on the surface, mm.

    The surface area is about (occupied voxels) x voxel^2, so the spacing is
    voxel * sqrt(occupied / points). Good to tens of percent, which is all the footprint needs."""
    pts = np.asarray(points, np.float64)
    if len(pts) < 2:
        return float(voxel)
    q = np.floor(pts / voxel).astype(np.int64)
    occupied = len(np.unique(q, axis=0))
    return float(voxel * np.sqrt(occupied / len(pts)))


def vote_views(views, masks):
    """Shrink the masks once for `subject_votes`. views: [(K, R, t, wh)], masks: uint8, any size.
    -> [(fx, fy, cx, cy, R, t, W, H, mask)] with the intrinsics already in the small mask's pixels."""
    import cv2
    out = []
    for (K, R, t, wh), mask in zip(views, masks):
        W, H = map_size(wh, VOTE_EDGE)
        m = cv2.resize(mask, (W, H), interpolation=cv2.INTER_AREA) if mask.shape[:2] != (H, W) else mask
        sx, sy = W / float(wh[0]), H / float(wh[1])
        out.append((K[0][0] * sx, K[1][1] * sy, K[0][2] * sx, K[1][2] * sy,
                    np.asarray(R, np.float64), np.asarray(t, np.float64), W, H, m >= 128))
    return out


def subject_votes(points, prepared, with_seen=False):
    """Share of its views in which each point lands inside the subject mask (NaN: in no view).
    With `with_seen`, also the number of views that had the point in frame.

    `prepared` comes from `vote_views`. The subject is what the masks outline in every
    photograph, so a point of the subject is inside nearly all of them — this is membership of
    the visual hull; the table the subject stands on and the wall behind it fail it."""
    P = np.asarray(points, np.float64)
    inside = np.zeros(len(P))
    seen = np.zeros(len(P))
    for fx, fy, cx, cy, R, t, W, H, m in prepared:
        Xc = P @ R.T + t
        z = Xc[:, 2]
        ok = z > NEAR_MM
        zz = np.where(ok, z, 1.0)
        u = np.floor(fx * Xc[:, 0] / zz + cx).astype(np.int64)
        v = np.floor(fy * Xc[:, 1] / zz + cy).astype(np.int64)
        ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
        seen[ok] += 1
        inside[ok] += m[v[ok], u[ok]]
    share = np.where(seen > 0, inside / np.maximum(seen, 1), np.nan)
    return (share, seen) if with_seen else share


def subject_points(points, prepared):
    """Which scan points are the subject's. -> (bool (N,), info)

    A point of the subject is inside the mask in nearly every view that has it in frame, and
    it is in frame in most views — the photographs are of the subject. "Nearly every" is
    measured, not assumed: the level is the median share among the points that pass
    ``VOTE_MIN``, and a point within ``HULL_SLACK`` of it counts (on the helmet the level was
    0.97, not 1: seventeen close-up masks were fragments). Points seen by fewer than
    ``VOTE_SEEN`` of the views are the far end of the table, in a handful of frames and inside
    the mask in those by chance."""
    share, seen = subject_votes(points, prepared, with_seen=True)
    with np.errstate(invalid="ignore"):
        rough = share >= VOTE_MIN
    level = float(np.median(share[rough])) if rough.any() else 1.0
    thr = max(VOTE_MIN, level - HULL_SLACK)
    with np.errstate(invalid="ignore"):
        subject = (share >= thr) & (seen >= VOTE_SEEN * len(prepared))
    return subject, {"level": round(level, 4), "threshold": round(thr, 4), "points": int(subject.sum()),
                     "of": int(len(share)), "views": len(prepared)}


def view_agreement(points, prepared):
    """Per view: the share of `points` in its frame that land inside its mask (NaN: none in frame).

    Given the subject's own points this grades the masks: a good mask holds nearly all of them,
    a fragment (a close-up where the segmenter outlined one panel of a helmet that fills the
    frame) holds few. The hull is carved by every mask it is given, so a fragment would carve
    the subject away."""
    P = np.asarray(points, np.float64)
    out = np.full(len(prepared), np.nan)
    for j, (fx, fy, cx, cy, R, t, W, H, m) in enumerate(prepared):
        Xc = P @ R.T + t
        z = Xc[:, 2]
        ok = z > NEAR_MM
        zz = np.where(ok, z, 1.0)
        u = np.floor(fx * Xc[:, 0] / zz + cx).astype(np.int64)
        v = np.floor(fy * Xc[:, 1] / zz + cy).astype(np.int64)
        ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if ok.any():
            out[j] = float(m[v[ok], u[ok]].mean())
    return out


def oriented_normals(points, cam_centres, k=NORMAL_K):
    """Unit normals of the scan, signed towards the nearest capture camera.

    The fallback for a scene without masks (see `hull_grid` for the one with). PCA gives the
    normal's line (lidar.estimate_normals); the nearest camera stood on the side the surface
    faces — outside an object, inside a room. That is right for what the cameras looked at
    squarely and wrong for a surface they only grazed: on the Stormtrooper helmet 21 % of the
    normals came out pointing inwards, the top of the dome among them (every camera sat below
    its tangent plane). `render` drops the points that face away from the view."""
    from . import lidar as lidarlib
    P = np.asarray(points, np.float64)
    n, _curv = lidarlib.estimate_normals(P, k=min(k, len(P)))
    C = np.asarray(cam_centres, np.float64)
    _, ci = lidarlib.NN(C).query(P)
    n[np.einsum("ij,ij->i", n, C[ci] - P) < 0] *= -1.0
    return n


def hull_grid(prepared, points, res=HULL_RES, max_views=HULL_VIEWS):
    """The subject's visual hull on a voxel grid: each voxel's share of views inside the mask.

    A scan is a shell seen from outside, and it has gaps — the back of a helmet the scanner
    never walked behind. Through such a gap a camera on the far side looks straight at the
    front of the helmet and the table beyond it, which in the photograph are hidden behind the
    helmet's own back. Those points are real, they face the camera, and their depth is 200 mm
    too deep for that pixel. The scan cannot say they are hidden; the masks can: the subject
    fills its silhouette in every view, so its visual hull is solid where the scan is empty.
    `hull_occluded` asks whether the line of sight to a point runs through that solid first.

    Only the views whose mask holds the subject carve it (`view_agreement` of at least
    ``HULL_VIEW_MIN``), at most `max_views` of them, evenly spaced. A hull is an intersection:
    a voxel is solid when it is inside (nearly) every mask, so the level is the share the
    subject's own points reach in those views and the slack is small. A looser hull is a fatter
    one — at "inside 90 % of the masks" the helmet's hull stood 12 mm off its surface.

    The grid spans the points (2nd-98th percentile box, padded by a quarter of its size) at
    `res` voxels along its longest side.
    -> {"lo", "voxel", "votes" (X, Y, Z) float32, "level", "views", "views_rejected"}"""
    P = np.asarray(points, np.float64)
    agree = view_agreement(P, prepared)
    good = [v for v, a in zip(prepared, agree) if np.isfinite(a) and a >= HULL_VIEW_MIN]
    rejected = len(prepared) - len(good)
    if not good:
        good, rejected = list(prepared), 0
    lo, hi = np.percentile(P, [2, 98], axis=0)
    pad = 0.25 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    voxel = float((hi - lo).max() / res)
    dims = np.maximum(1, np.ceil((hi - lo) / voxel).astype(int))
    step = max(1, int(np.ceil(len(good) / float(max_views))))
    use = good[::step]
    ax = [lo[i] + (np.arange(dims[i]) + 0.5) * voxel for i in range(3)]
    votes = np.zeros(tuple(dims), np.float32)
    for ix in range(dims[0]):                      # one slab at a time: bounded memory
        yy, zz = np.meshgrid(ax[1], ax[2], indexing="ij")
        slab = np.stack([np.full(yy.size, ax[0][ix]), yy.ravel(), zz.ravel()], 1)
        votes[ix] = np.nan_to_num(subject_votes(slab, use), nan=0.0).reshape(dims[1], dims[2])
    grid = {"lo": lo, "voxel": voxel, "votes": votes, "views": len(use), "views_rejected": rejected}
    grid["level"] = float(np.median(hull_votes(grid, P)))
    return grid


def hull_votes(grid, points):
    """The grid's vote share at each point (nearest voxel; 0 outside the grid)."""
    P = np.asarray(points, np.float64)
    idx = np.floor((P - grid["lo"]) / grid["voxel"]).astype(np.int64)
    dims = np.array(grid["votes"].shape)
    ok = ((idx >= 0) & (idx < dims)).all(axis=1)
    out = np.zeros(len(P), np.float32)
    i = idx[ok]
    out[ok] = grid["votes"][i[:, 0], i[:, 1], i[:, 2]]
    return out


def hull_normals(grid, points, k=NORMAL_K):
    """Unit normals of the subject's points, signed outwards by the hull. -> (normals, settled)

    PCA gives the normal's line. Which way is out is read off the hull: its vote share falls
    going out of the subject and holds going in, so the two are probed a few voxels either side
    of the point (``HULL_NORMAL_STEPS``, the nearest probe that tells them apart wins). A point
    where no probe does — the floor of a recess the hull has filled in, both sides solid; a fin
    thinner than the probe, both sides empty — is not `settled` and its normal means nothing.

    Why normals at all, when `hull_occluded` already asks the hull what is hidden: a point on
    the far side of the subject, close to the edge of what was scanned, is seen from behind
    along a line that only clips the subject — a chord shorter than the first occlusion sample.
    It faces away from the camera by a few degrees, and is 20-100 mm too deep for its pixel
    (ball test, 2026-10-01: 343 of a rear view's 733 depth pixels). Facing is what removes it."""
    from . import lidar as lidarlib
    P = np.asarray(points, np.float64)
    n, _curv = lidarlib.estimate_normals(P, k=min(k, len(P)))
    n = np.array(n, np.float64)
    settled = np.zeros(len(P), bool)
    for s in HULL_NORMAL_STEPS:
        d = s * grid["voxel"]
        diff = hull_votes(grid, P - d * n) - hull_votes(grid, P + d * n)     # > 0: +n leaves the subject
        new = ~settled & (np.abs(diff) > HULL_NORMAL_MARGIN)
        n[new & (diff < 0)] *= -1.0
        settled |= new
    return n, settled


def hull_visible(grid, points, normals, settled, cam_centre, extent):
    """Which of the subject's points a camera at `cam_centre` sees, as far as the scan and the
    hull can tell: not behind the subject's volume (`hull_occluded`), and facing the camera
    (`hull_normals`). A point whose normal could not be signed is kept only if its line of sight
    is clear of the hull all the way from ``HULL_NEAR_FROM`` voxels out — which drops the floor
    of a recess along with the far side; no depth there is the safe answer."""
    P = np.asarray(points, np.float64)
    to_cam = np.asarray(cam_centre, np.float64) - P
    dist = np.linalg.norm(to_cam, axis=1)
    unit = to_cam / np.maximum(dist, 1e-9)[:, None]
    visible = ~hull_occluded(grid, P, cam_centre, extent)
    facing = np.einsum("ij,ij->i", normals, unit) > FACING_COS
    visible &= facing | ~settled
    loose = np.flatnonzero(visible & ~settled)
    if len(loose):
        lo, hi = HULL_NEAR_FROM * grid["voxel"], HULL_STEPS[0] * extent
        steps = np.arange(lo, max(hi, lo + grid["voxel"]), grid["voxel"])
        visible[loose] = ~hull_occluded(grid, P[loose], cam_centre, 1.0, steps=steps)
    return visible


def hull_occluded(grid, points, cam_centre, extent, steps=HULL_STEPS):
    """True for the points the subject's own volume hides from a camera at `cam_centre`.

    The line of sight is sampled at ``HULL_STEPS`` x `extent` in front of the point; a sample
    inside the hull (vote share within ``HULL_SOLID`` of the subject's own level) is solid
    subject between the camera and the point. The first sample sits far enough out (12 % of the
    extent) that a point down a recess — an eye socket, the gap between two tubes — still sees
    daylight: the hull fills recesses in, since no silhouette shows them."""
    P = np.asarray(points, np.float64)
    to_cam = np.asarray(cam_centre, np.float64) - P
    dist = np.linalg.norm(to_cam, axis=1)
    unit = to_cam / np.maximum(dist, 1e-9)[:, None]
    thr = grid["level"] - HULL_SOLID
    hidden = np.zeros(len(P), bool)
    for f in steps:
        step = np.minimum(f * extent, 0.9 * dist)
        hidden |= hull_votes(grid, P + unit * step[:, None]) >= thr
    return hidden


def extent_of(points):
    """Size of a point set: the diagonal of its 2nd-98th percentile box."""
    lo, hi = np.percentile(np.asarray(points, np.float64), [2, 98], axis=0)
    return float(np.linalg.norm(hi - lo))


def map_size(wh, long_edge=LONG_EDGE):
    """Depth map (W, H) for a photograph of size wh: the long edge at most `long_edge`."""
    w, h = int(wh[0]), int(wh[1])
    s = min(1.0, float(long_edge) / max(w, h))
    return max(1, int(round(w * s))), max(1, int(round(h * s)))


def _zbuffer(flat, idx, z):
    np.minimum.at(flat, idx, z)


def render(points, K, R, t, wh, long_edge=LONG_EDGE, spacing_mm=None, mask=None, region=None, normals=None):
    """The scan as view (K, R, t) sees it. -> (depth float32 [H, W] in the points' units, 0 = none; info)

    points   (N, 3) in the rig's frame (mm), Xc = X @ R.T + t
    wh       the photograph's (width, height) that K refers to
    mask     the view's subject mask (uint8, any size) or None
    region   'inside' | 'outside' | None: which side of the mask to keep
    normals  (N, 3) outward unit normals (`oriented_normals`) or None; with them, points facing
             away from the camera are not rendered
    """
    import cv2
    W, H = map_size(wh, long_edge)
    sx, sy = W / float(wh[0]), H / float(wh[1])
    K = np.asarray(K, np.float64)
    R = np.asarray(R, np.float64)
    points = np.asarray(points, np.float64)
    Xc = points @ R.T + np.asarray(t, np.float64)
    z = Xc[:, 2]
    front = z > NEAR_MM
    info = {"size": [W, H], "points_in_front": int(front.sum())}
    if normals is not None:
        # line of sight from the point back to the camera, in the camera frame: -Xc
        nc = np.asarray(normals, np.float64) @ R.T
        facing = -np.einsum("ij,ij->i", nc, Xc) > FACING_COS * np.linalg.norm(Xc, axis=1)
        info["points_facing"] = int((front & facing).sum())
        front &= facing
    Xc, z = Xc[front], z[front]
    depth = np.zeros((H, W), np.float32)
    if not len(z):
        info.update({"points_in_view": 0, "footprint_px": 0, "valid_px": 0})
        return depth, info
    u = (K[0, 0] * Xc[:, 0] / z + K[0, 2]) * sx
    v = (K[1, 1] * Xc[:, 1] / z + K[1, 2]) * sy
    ui, vi = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    inside = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui, vi, z = ui[inside], vi[inside], z[inside]
    info["points_in_view"] = int(inside.sum())
    if not len(z):
        info.update({"footprint_px": 0, "valid_px": 0})
        return depth, info

    # 2. hidden points
    Wc, Hc = -(-W // COARSE), -(-H // COARSE)
    coarse = np.full(Wc * Hc, np.inf)
    cidx = (vi // COARSE) * Wc + (ui // COARSE)
    _zbuffer(coarse, cidx, z)
    # erode = 3x3 minimum; +inf where a cell is empty, so empty neighbours do not pull it down
    env = cv2.erode(coarse.reshape(Hc, Wc), np.ones((3, 3), np.uint8),
                    borderType=cv2.BORDER_CONSTANT, borderValue=float("inf")).ravel()
    f = 0.5 * (K[0, 0] * sx + K[1, 1] * sy)
    # the window is 3 cells wide: 3 * COARSE px, i.e. 3 * COARSE * z / f across the surface
    tol = max(OCCLUSION_TOL, SLOPE_TAN * 3 * COARSE / f)
    visible = z <= env[cidx] * (1.0 + tol)
    ui, vi, z = ui[visible], vi[visible], z[visible]
    info["points_visible"] = int(visible.sum())

    # 3. z-buffer, footprint from the point spacing as this view sees it
    if spacing_mm is None:
        spacing_mm = point_spacing_mm(points)
    spacing_px = spacing_mm * f / float(np.median(z))
    r = int(np.clip(np.ceil(0.75 * spacing_px), 1, MAX_FOOTPRINT))
    info["spacing_px"], info["footprint_px"] = round(float(spacing_px), 3), r
    # The pixel's own points give its depth; the footprint only fills the pixels no point landed
    # in. (A footprint minimum everywhere would pull every slanted surface towards the camera by
    # up to r px of slope.) Both are means over the front layer, not minima: `near` is the
    # nearest point within the footprint, and anything well behind it is the far side showing
    # through a gap.
    pidx = vi * W + ui
    near = np.full(W * H, np.inf)
    for dy in range(-r, r + 1):
        yy = vi + dy
        oky = (yy >= 0) & (yy < H)
        for dx in range(-r, r + 1):
            xx = ui + dx
            ok = oky & (xx >= 0) & (xx < W)
            _zbuffer(near, yy[ok] * W + xx[ok], z[ok])
    front_layer = z <= near[pidx] * (1.0 + max(OCCLUSION_TOL, SLOPE_TAN * (2 * r + 1) / f))
    zsum = np.bincount(pidx[front_layer], weights=z[front_layer], minlength=W * H)
    cnt = np.bincount(pidx[front_layer], minlength=W * H).astype(np.float64)
    has_own = (cnt > 0).reshape(H, W)
    own = np.where(has_own, (zsum / np.maximum(cnt, 1)).reshape(H, W), 0.0)
    k = (2 * r + 1, 2 * r + 1)
    box = dict(ddepth=-1, ksize=k, normalize=False, borderType=cv2.BORDER_CONSTANT)
    nsum = cv2.boxFilter(own, **box)
    ncnt = cv2.boxFilter(has_own.astype(np.float64), **box)
    filled = ~has_own & (ncnt > 0.5)
    info["filled_px"] = int(filled.sum())
    zb = np.where(has_own, own, np.where(filled, nsum / np.maximum(ncnt, 1.0), np.inf))
    valid = np.isfinite(zb)

    # 4. discontinuities and the border ring
    k3 = np.ones((3, 3), np.uint8)
    zmin = cv2.erode(zb, k3, borderType=cv2.BORDER_CONSTANT, borderValue=float("inf"))
    zmax = cv2.dilate(np.where(valid, zb, 0.0), k3, borderType=cv2.BORDER_CONSTANT, borderValue=0.0)
    smooth = valid & ((zmax - zmin) <= EDGE_TOL * np.where(valid, zb, 1.0))
    info["discontinuity_px"] = int(valid.sum() - smooth.sum())
    # the ring comes off the outline of the measured region, not off every pinhole inside it
    outline = cv2.morphologyEx(smooth.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    keep = smooth & (cv2.erode(outline, k3, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0)
    # a few stray points make a speck with a depth of its own and nothing around it to contradict it
    n_lab, lab, stats, _cent = cv2.connectedComponentsWithStats(keep.astype(np.uint8), connectivity=8)
    if n_lab > 1:
        small = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] < MIN_PATCH_PX)
        small = small[small > 0]
        if len(small):
            speck = np.isin(lab, small)
            info["speck_px"] = int(speck.sum())
            keep &= ~speck

    # 5. the layer's region
    if mask is not None and region in ("inside", "outside"):
        m = cv2.resize(mask, (W, H), interpolation=cv2.INTER_AREA) if mask.shape[:2] != (H, W) else mask
        sel = (m >= MASK_INSIDE) if region == "inside" else (m <= MASK_OUTSIDE)
        info["region_px"] = int(sel.sum())
        info["region_covered"] = round(float((keep & sel).sum() / sel.sum()), 4) if sel.any() else None
        keep &= sel
    depth[keep] = zb[keep].astype(np.float32)
    info["valid_px"] = int(keep.sum())
    info["valid_fraction"] = round(float(keep.mean()), 4)
    if keep.any():
        d = depth[keep]
        info["depth_min"], info["depth_median"], info["depth_max"] = (
            round(float(d.min()), 2), round(float(np.median(d)), 2), round(float(d.max()), 2))
    return depth, info


def choose_unit_mm(max_depth_mm):
    """The smallest count size from UNITS_MM that holds `max_depth_mm` in 16 bits."""
    for u in UNITS_MM:
        if max_depth_mm / u <= MAX_COUNT:
            return u
    return float(np.ceil(max_depth_mm / MAX_COUNT))


def write_png16(path, depth_mm, unit_mm):
    """depth_mm [H, W] (0 = none) -> 16-bit PNG of counts of unit_mm. A measured pixel never rounds to 0."""
    import cv2
    counts = np.rint(np.asarray(depth_mm, np.float64) / unit_mm)
    counts[(depth_mm > 0) & (counts < 1)] = 1
    if counts.max(initial=0) > 65535:
        raise ValueError(f"depth {depth_mm.max():.0f} mm does not fit 16 bits at {unit_mm} mm per count")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not cv2.imwrite(path, counts.astype(np.uint16)):
        raise OSError(f"could not write {path}")


def read_png16(path, unit_mm):
    import cv2
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is None:
        raise OSError(f"could not read {path}")
    return a.astype(np.float32) * float(unit_mm)


def colourise(depth, lo=None, hi=None):
    """Depth map -> BGR picture (turbo, near = warm), unmeasured pixels dark grey."""
    import cv2
    valid = depth > 0
    out = np.full(depth.shape + (3,), 40, np.uint8)
    if valid.any():
        lo = float(depth[valid].min()) if lo is None else lo
        hi = float(depth[valid].max()) if hi is None else hi
        n = np.clip((hi - depth) / max(hi - lo, 1e-9), 0, 1)
        col = cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        out[valid] = col[valid]
    return out


def contact_sheet(rows, path, tile_h=360):
    """rows: [(label, photograph BGR, depth [H, W])] -> a sheet of photograph | depth pairs."""
    import cv2
    tiles = []
    for label, photo, depth in rows:
        s = tile_h / photo.shape[0]
        ph = cv2.resize(photo, (max(1, int(round(photo.shape[1] * s))), tile_h), interpolation=cv2.INTER_AREA)
        dc = cv2.resize(colourise(depth), (ph.shape[1], tile_h), interpolation=cv2.INTER_NEAREST)
        # the photograph shows through where there is depth, so a misregistered scan is visible
        blend = np.where((dc != 40).any(axis=2, keepdims=True), (0.45 * ph + 0.55 * dc).astype(np.uint8), ph // 3)
        cv2.putText(ph, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.hstack([ph, blend]))
    if not tiles:
        return None
    w = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, 0, 0, w - t.shape[1], cv2.BORDER_CONSTANT) for t in tiles]
    per_row = 3
    grid = [np.hstack(tiles[i:i + per_row] + [np.zeros_like(tiles[0])] * (per_row - len(tiles[i:i + per_row])))
            for i in range(0, len(tiles), per_row)]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, np.vstack(grid), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return path


MIN_VIEW_PX = 100        # a view's map is written only if it holds at least this many measured pixels
MIN_SUBJECT_POINTS = 500 # fewer scan points than this on the subject: the scan does not hold it
SHEET_VIEWS = 6          # photograph | depth pairs on the contact sheet


def view_key(name):
    """``cap004_R`` -> ``R/cap004``: the view as --exclude and the dataset's folders spell it."""
    return f"{name[-1]}/{name[:-2]}" if name.endswith(("_L", "_R")) else f"L/{name}"


def _view_wh(G, i):
    if "wh" in G.files:
        return int(G["wh"][i][0]), int(G["wh"][i][1])
    return int(G["w"]), int(G["h"])


def build(points, G, names, out_dir, layer="full", masks_dir=None, images_dir=None, exclude=(),
          long_edge=LONG_EDGE, progress=None):
    """One depth map per training view, written under `out_dir` the way the masks are laid out.

    points      (N, 3) scan points in the rig's frame, mm
    G, names    the rig.npz and its view names
    layer       'subject' (needs `masks_dir`: the scan's subject points, hidden where the visual
                hull says so, kept inside the mask), 'background' (everything, kept outside the
                mask) or 'full' (everything, no masks: back faces removed by `oriented_normals`)
    exclude     view keys (``L/cap064``) that are not trained: no map, and no say in the hull
    progress    called as progress(done, total)

    -> report: the settings, per-view coverage, the totals, and "unit_mm" (one count of the PNGs).
    Raises ValueError when the scan cannot serve (no subject points, no view sees it)."""
    import cv2
    P = np.asarray(points, np.float64)
    if layer not in ("full", "subject", "background"):
        raise ValueError(f"unknown layer {layer}")
    use_masks = layer in ("subject", "background")
    if use_masks and not (masks_dir and os.path.isdir(masks_dir)):
        raise ValueError(f"layer {layer} needs the masks folder")
    views = []
    for i, name in enumerate(names):
        key = view_key(name)
        if key in exclude:
            continue
        views.append((i, key, np.asarray(G["K"][i], np.float64), np.asarray(G["R"][i], np.float64),
                      np.asarray(G["t"][i], np.float64), _view_wh(G, i)))
    if not views:
        raise ValueError("no training views")
    report = {"layer": layer, "long_edge": int(long_edge), "scan_points": int(len(P)),
              "views": len(views), "excluded": len(names) - len(views)}

    # the masks, read once: shrunk for the vote and to the map's own size for the region
    small, vote_in = {}, []
    if use_masks:
        for i, key, K, R, t, wh in views:
            m = cv2.imread(os.path.join(masks_dir, key + ".png"), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            W, H = map_size(wh, long_edge)
            small[key] = cv2.resize(m, (W, H), interpolation=cv2.INTER_AREA)
            # the vote's own size now, so 240 full-size masks are never held at once
            vote_in.append(((K, R, t, wh), cv2.resize(m, map_size(wh, VOTE_EDGE), interpolation=cv2.INTER_AREA)))
        report["views_with_mask"] = len(small)
        if not small:
            raise ValueError(f"no mask found for any training view under {masks_dir}")

    normals = settled = grid = None
    extent = 0.0
    X = P
    if layer == "subject":
        prepared = vote_views([v for v, _m in vote_in], [m for _v, m in vote_in])
        del vote_in
        subject, sinfo = subject_points(P, prepared)
        report["subject"] = sinfo
        if sinfo["points"] < MIN_SUBJECT_POINTS:
            raise ValueError(f"only {sinfo['points']} of the scan's {len(P)} points fall inside the subject masks "
                             f"(threshold {sinfo['threshold']} of the views): the scan does not hold the subject, "
                             "or it is not registered to this solve")
        X = P[subject]
        # a stray point that the masks happen to hold (a speck of the table seen in few views)
        # is not the subject: it has no neighbours among the points that are
        from . import lidar as lidarlib
        near = lidarlib.denoise(X)
        report["subject"]["isolated_dropped"] = int((~near).sum())
        X = X[near]
        grid = hull_grid(prepared, X)
        extent = extent_of(X)
        normals, settled = hull_normals(grid, X)
        report["hull"] = {"voxel_mm": round(grid["voxel"], 3), "views": grid["views"],
                          "views_rejected": grid["views_rejected"], "level": round(grid["level"], 4),
                          "extent_mm": round(extent, 1), "normals_settled": round(float(settled.mean()), 4)}
    elif layer == "full":
        C = np.array([-(R.T @ t) for _i, _k, _K, R, t, _wh in views])
        normals = oriented_normals(P, C)
    spacing = point_spacing_mm(X)
    report["spacing_mm"] = round(spacing, 3)

    # one count size for every file: no depth can exceed the furthest scan corner from any camera
    lo, hi = X.min(axis=0), X.max(axis=0)
    corners = np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])])
    far = max(float(np.linalg.norm(corners - (-(R.T @ t)), axis=1).max()) for _i, _k, _K, R, t, _wh in views)
    unit = choose_unit_mm(far)
    report["unit_mm"] = unit

    rows, written, region_px, covered_px, valid_px = [], [], 0, 0, 0
    for n, (i, key, K, R, t, wh) in enumerate(views):
        mask = small.get(key) if use_masks else None
        row = {"view": key}
        if use_masks and mask is None:
            row["skipped"] = "no mask"
            rows.append(row)
            continue
        if layer == "subject":
            seen = hull_visible(grid, X, normals, settled, -(R.T @ t), extent)
            row["points_seen"] = int(seen.sum())
            depth, info = render(X[seen], K, R, t, wh, long_edge, spacing, mask=mask, region="inside")
        elif layer == "background":
            depth, info = render(X, K, R, t, wh, long_edge, spacing, mask=mask, region="outside")
        else:
            depth, info = render(X, K, R, t, wh, long_edge, spacing, normals=normals)
        for k in ("size", "valid_px", "region_px", "region_covered", "depth_min", "depth_median", "depth_max",
                  "footprint_px"):
            if k in info:
                row[k] = info[k]
        px = info.get("region_px", depth.size)
        region_px += px
        if info["valid_px"] >= MIN_VIEW_PX:
            write_png16(os.path.join(out_dir, key + ".png"), depth, unit)
            written.append((key, i))
            covered_px += info["valid_px"]
            valid_px += info["valid_px"]
            row["written"] = True
        else:
            row["written"] = False
        rows.append(row)
        if progress:
            progress(n + 1, len(views))
    cov = sorted((r.get("region_covered") if use_masks else r["valid_px"] / float(np.prod(r["size"])))
                 for r in rows if "size" in r and (not use_masks or r.get("region_covered") is not None))
    report.update({
        "views_with_depth": len(written),
        "valid_px": int(valid_px),
        "region_px": int(region_px),
        "coverage": round(covered_px / region_px, 4) if region_px else 0.0,
        "coverage_percentiles": ({str(q): round(float(np.percentile(cov, q)), 4) for q in (10, 25, 50, 75, 90)}
                                 if cov else None),
        "views_under_half": int(sum(c < 0.5 for c in cov)),
        "per_view": rows,
    })
    if not written:
        raise ValueError("the scan gave no depth in any training view: it does not cover what the cameras saw")

    if images_dir:
        step = max(1, len(written) // SHEET_VIEWS)
        sheet = []
        for key, _i in written[step // 2::step][:SHEET_VIEWS]:
            photo = next((cv2.imread(os.path.join(images_dir, key + e)) for e in (".jpg", ".jpeg", ".png")
                          if os.path.exists(os.path.join(images_dir, key + e))), None)
            if photo is not None:
                sheet.append((key, photo, read_png16(os.path.join(out_dir, key + ".png"), unit)))
        p = contact_sheet(sheet, os.path.join(out_dir, "depth_sheet.jpg"))
        if p:
            report["sheet"] = os.path.basename(p)
    return report
