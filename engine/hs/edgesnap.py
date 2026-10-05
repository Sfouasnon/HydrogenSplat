"""Where is the photograph's edge, relative to the mask's?

Apple Vision's soft matte is generous: on 2026-09-28_Stormtrooper_iPhone its 50 % level sits
2–6 px outside the helmet (median +3.5 px over 29 views, no view negative), and the threshold
that would land on the photograph's edge differs from view to view (128 on one, 250 on another),
so no fixed threshold or fixed erosion puts every mask in the right place. A boundary that is
helmet in one view's mask and background in the next leaves the optimiser one answer for the rim
— a half-opaque splat — and that is G's soft 6–8 px silhouette ramp
(claude/edge-profile-mask-bias-2026-10-04.md).

`measure` finds the offset per view: from sampled boundary pixels of the hard mask, walk the
inward normal −WALK_PX…+WALK_PX px through the photograph and take the position of the largest
colour step (Σ|ΔBGR| between neighbouring samples). Boundary points with no real step anywhere on
the walk (a white dome against a white wall) are dropped; the median over the rest is the view's
offset, positive when the photograph's edge is inside the mask. `snap` then erodes (or dilates)
the hard mask by that many whole pixels.

Limits, stated: the largest step within ±WALK_PX px is the photograph's edge where the edge has
contrast; a dark trim band just inside the silhouette can also win, which biases the estimate
inward by its width. The per-view median tolerates that on a minority of the boundary. A view
whose boundary has contrast almost nowhere returns None and the caller falls back.
"""
import numpy as np

WALK_PX = 12         # half-length of the walk along the normal, px (the helmet's worst views were 7-8 px out)
MIN_STEP = 45        # Σ|ΔBGR| a boundary point needs somewhere on its walk to count (8-bit units)
SAMPLE = 3000        # boundary pixels sampled per view (seeded), fewer if the boundary is shorter
MIN_POINTS = 50      # usable points below which the view is unmeasured ...
MIN_SHARE = 0.05     # ... and the share of the sampled boundary they must make up (57 points of 3,000 is noise)
NORMAL_BLUR = 2.0    # sigma of the blur on the signed distance before taking its gradient


def _crop_box(hard, pad):
    ys, xs = np.nonzero(hard)
    if not len(ys):
        return None
    h, w = hard.shape
    return (max(0, ys.min() - pad), min(h, ys.max() + pad + 1), max(0, xs.min() - pad), min(w, xs.max() + pad + 1))


def measure(photo_bgr, hard, seed=0, walk_px=WALK_PX, min_step=MIN_STEP):
    """Offset of the photograph's edge from the mask's boundary, px, + = inside the mask.

    `photo_bgr`: uint8 HxWx3. `hard`: bool HxW. Returns {"offset": median, "p10", "p90",
    "points": usable boundary points, "sampled": boundary points tried, "boundary": boundary
    pixels in all} or None when fewer than MIN_POINTS boundary points see a colour step."""
    import cv2
    hard = hard.astype(bool)
    box = _crop_box(hard, walk_px + 4)
    if box is None:
        return None
    y0, y1, x0, x1 = box
    m = hard[y0:y1, x0:x1]
    I = photo_bgr[y0:y1, x0:x1].astype(np.float32)
    H, W = m.shape
    din = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    dout = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5)
    sd = np.where(m, din, -dout).astype(np.float32)
    sdf = cv2.GaussianBlur(sd, (0, 0), NORMAL_BLUR)
    gy, gx = np.gradient(sdf)
    g = np.hypot(gx, gy) + 1e-6
    by, bx = np.nonzero((sd > 0) & (sd <= 1.0))
    boundary = int(len(by))
    if boundary < MIN_POINTS:
        return None
    if boundary > SAMPLE:
        idx = np.sort(np.random.default_rng(seed).choice(boundary, SAMPLE, replace=False))
        by, bx = by[idx], bx[idx]
    nx, ny = gx[by, bx] / g[by, bx], gy[by, bx] / g[by, bx]        # unit normal, pointing inward
    t = np.arange(-walk_px, walk_px + 1, dtype=np.float32)
    xs = (bx[:, None] + t[None, :] * nx[:, None]).astype(np.float32)
    ys = (by[:, None] + t[None, :] * ny[:, None]).astype(np.float32)
    inside = ((xs >= 1) & (xs < W - 1) & (ys >= 1) & (ys < H - 1)).all(1)
    prof = cv2.remap(I, xs, ys, cv2.INTER_LINEAR)                 # (n, 2*walk+1, 3)
    step = np.abs(np.diff(prof, axis=1)).sum(2)                   # (n, 2*walk)
    keep = inside & (step.max(1) > min_step)
    n = int(keep.sum())
    if n < MIN_POINTS or n < MIN_SHARE * len(by):
        return None
    e = t[step[keep].argmax(1)] + 0.5                             # the step sits between two samples
    return {"offset": float(np.median(e)), "p10": float(np.percentile(e, 10)),
            "p90": float(np.percentile(e, 90)), "points": n, "sampled": int(len(by)), "boundary": boundary}


def pixels(offset, max_px):
    """Whole pixels to erode (+) or dilate (−) for a measured offset, within ±max_px. The offset
    is a half-integer by construction (the step sits between two samples), so round half up
    rather than to even: 3.5 → 4 and 2.5 → 3, never 2.5 → 2."""
    return int(np.clip(int(np.floor(offset + 0.5)), -int(max_px), int(max_px)))


def snap(hard, px):
    """Erode (px > 0) or dilate (px < 0) a hard mask by whole pixels. -> uint8 0/255."""
    import cv2
    m = np.where(hard.astype(bool), 255, 0).astype(np.uint8)
    if px == 0:
        return m
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(px) + 1, 2 * abs(px) + 1))
    return cv2.erode(m, k) if px > 0 else cv2.dilate(m, k)
