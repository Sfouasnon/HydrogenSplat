"""Do the subject masks agree with each other? A check that needs no trained model.

Why. In `--alpha-mode transparent` everything outside a view's mask is supervised as empty. A mask
that leaves part of the subject out therefore teaches the model that this part is empty *from that
direction*; opacity cannot depend on the view, so the model keeps the surface and paints it black
from there. On 2026-09-28_Stormtrooper_iPhone 36 of 243 masks did this — 17 where Vision found no
object and the rough region was used, 4 that stop at the brow and leave the dome out, 15 with a
bite out of the jaw or the chin — and they were found in four rounds, three of them after a
2 h 40 min train each (claude/run-e-…, run-f-…, mask-check-before-training-2026-10-04.md).

What is known before training: every view's mask and its pose. Two tests come out of that.

A. *Does the mask hold the subject?* A sparse SfM point that lands inside the mask in nearly all
   of its views is a point of the subject. A mask that holds under ``AGREE_MIN`` of those points
   (of the ones in its frame) has lost the subject: a fragment.

B. *Does the mask agree with the others?* The masks that pass A vote a visual hull: a voxel is
   solid for view i when at least ``HULL_THRESHOLD`` of the *other* views that have it in frame
   have it inside their mask. That hull, projected into view i, is the subject as the other views
   see it. What falls outside mask i is either a thin rim (the hull and a mask never agree to the
   pixel) or a compact piece — a bite. Rim and piece are told apart by a morphological opening.

The vote is deliberately not unanimous. A unanimous hull is carved by every bad mask, so the bite
of one view would vanish from everyone's hull; at 0.97 it takes more than 3 % of the views sharing
a bite to hide it. The price is the same mechanism read the other way: a region that only a few
views can carve (the air in front of a face, which only strict profiles see) survives, and the
profile views are then flagged for a piece that is air. So the check does not decide. It chooses
what a person looks at; `hs masks --decide` records what they said.

Measured on the helmet (243 views, 9 s): A flags 19, B flags 29 more, 48 in all; 34 of the 36
known bad masks are among them, the two it misses leave out 0.2-0.3 % of the helmet. Of the 14
flagged that were not known, the ones looked at leave out the dark opening under the helmet or are
air in front of the face. The thresholds were chosen on that one project.

Repairs. For a flagged view the engine prepares, when it can, a mask to use instead:
  reselect  the view fell back to the rough region because Vision's object failed the old test
            (the SfM points sit on the textured faceplate only, so the object was "outside" its
            own prior). Tested against the hull instead, the object is kept;
  fill      the missing piece is bounded by the mask itself (a hole), or by the mask and the frame
            edge: it is filled, and no new outline is drawn;
  hull      the missing piece is on the silhouette: it is added with the hull's own outline, which
            is good to about a pixel of the 480 px check image (8 px of a 3840 px frame) on
            average and four at worst.
A repair is offered only if the repaired mask then passes this same check.

Which repairs can be trusted without looking. Two: Vision's own object (a real segmentation that
the hull confirms), and a hole closed inside the mask. Every other repair rests on the voted hull
being right about where the subject is, and the hull is too large exactly where a bite would be —
near the silhouette, where few views are tangent. Those are marked ``approximate``: on the helmet
they include the real bites out of the chin and the dome, and also two pieces of air in front of
the face. A person tells them apart in a second from the picture; the engine cannot.
"""
import os

import numpy as np

from . import depthmaps

CHECK_EDGE = depthmaps.VOTE_EDGE   # 480 px: the masks are shrunk to this for every test here
AGREE_MIN = 0.9          # A: a mask holding under this share of the subject's points is flagged
SUBJECT_SHARE = 0.9      # a sparse point inside the mask in this share of its views is the subject's
SUBJECT_SEEN = 0.3       # ...and in frame in this share of all views
MIN_SUBJECT_POINTS = 50  # fewer than this: no box to vote in, the check cannot run
MIN_AGREE_POINTS = 20    # A is measured only on a view with at least this many subject points in frame
HULL_THRESHOLD = 0.97    # B: share of the other views that must have a voxel inside their mask
GRID_RES = 128           # voxels along the longest side of the box
MIN_SEEN = 0.3           # a voxel votes only if in frame in this share of the voting views
OPEN_FRAC = 0.023        # opening that separates rim from piece, as a share of the long edge (11 px at 480)
MIN_PIECE = 0.003        # B: flag when the piece is at least this share of the subject's area in the view
MIN_PIECE_PX = 16        # ...and at least this many check pixels
MIN_HULL_PX = 400        # a view with less of the subject in frame than this is not judged
BOX_PAD = 0.2            # the voting box: the subject's points, padded by this share of its size
BOX_GROW = 1.5           # ...and grown by this factor while solid voxels still touch its faces
RESELECT_INSIDE = 0.85   # reselect: share of a Vision object that must lie inside the hull
RESELECT_GAIN = 0.02     # ...and it must cover this much more of the hull than the mask in use
ENCLOSED_MIN = 0.95      # fill: share of the piece's outline that runs along the mask (or the frame edge)
SPILL_MAX = 0.03         # a repair may put at most this share of the subject's area outside the hull
HULL_GROW_PX = 1         # hull pieces are grown by this (the voted hull sits ~1 px inside the masks)
PREVIEW_H = 540          # preview pictures: height of each of the two panels


def _open_kernel(shape, frac=OPEN_FRAC):
    import cv2
    k = 2 * int(round(frac * max(shape) / 2.0)) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(k, 3), max(k, 3)))


def _project(P, pr):
    fx, fy, cx, cy, R, t, W, H, _m = pr
    Xc = P @ R.T + t
    z = Xc[:, 2]
    ok = z > depthmaps.NEAR_MM
    zz = np.where(ok, z, 1.0)
    u = np.floor(fx * Xc[:, 0] / zz + cx).astype(np.int64)
    v = np.floor(fy * Xc[:, 1] / zz + cy).astype(np.int64)
    ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u, v, z, ok


def subject_points(points, prepared):
    """The sparse points that are the subject's: inside the mask in nearly all their views."""
    P = np.asarray(points, np.float64)
    if not len(P):
        return P.reshape(0, 3)
    share, seen = depthmaps.subject_votes(P, prepared, with_seen=True)
    keep = np.nan_to_num(share, nan=0.0) >= SUBJECT_SHARE
    keep &= seen >= SUBJECT_SEEN * len(prepared)
    return P[keep]


def agreement(sub, prepared):
    """Per view: the share of the subject's points in its frame that land inside its mask; NaN
    when fewer than ``MIN_AGREE_POINTS`` are in frame (three points are not a measurement)."""
    out = np.full(len(prepared), np.nan)
    for j, pr in enumerate(prepared):
        u, v, _z, ok = _project(sub, pr)
        if int(ok.sum()) >= MIN_AGREE_POINTS:
            out[j] = float(pr[8][v[ok], u[ok]].mean())
    return out


def vote(prepared, voters, sub, res=GRID_RES):
    """The voxels that could be solid, with their votes among `voters` (bool per view).
    -> {"P": (n, 3) centres, "k": inside counts, "n": in-frame counts, "voxel", "voters", "box"}"""
    use = [prepared[i] for i in np.flatnonzero(voters)]
    lo, hi = np.percentile(sub, [1, 99], axis=0)
    pad = BOX_PAD * (hi - lo)
    lo, hi = lo - pad, hi + pad
    need = MIN_SEEN * len(use)
    for _grow in range(3):
        voxel = float((hi - lo).max() / res)
        dims = np.maximum(1, np.ceil((hi - lo) / voxel).astype(int))
        ax = [lo[i] + (np.arange(dims[i]) + 0.5) * voxel for i in range(3)]
        k = np.zeros(tuple(dims), np.int32)
        n = np.zeros(tuple(dims), np.int32)
        yy, zz = np.meshgrid(ax[1], ax[2], indexing="ij")
        slab = np.stack([np.zeros(yy.size), yy.ravel(), zz.ravel()], 1)
        for ix in range(dims[0]):                 # one slab at a time: bounded memory
            slab[:, 0] = ax[0][ix]
            ki = np.zeros(len(slab), np.int32)
            ni = np.zeros(len(slab), np.int32)
            for pr in use:
                u, v, _z, ok = _project(slab, pr)
                ni[ok] += 1
                ki[ok] += pr[8][v[ok], u[ok]]
            k[ix] = ki.reshape(dims[1], dims[2])
            n[ix] = ni.reshape(dims[1], dims[2])
        solid = (n >= need) & (k >= HULL_THRESHOLD * n)
        # the subject must end inside the box; a hull that reaches a face has been cut off by it
        touches = (solid[0].any() or solid[-1].any() or solid[:, 0].any() or solid[:, -1].any()
                   or solid[:, :, 0].any() or solid[:, :, -1].any())
        if not touches:
            break
        c, half = (lo + hi) / 2.0, (hi - lo) / 2.0 * BOX_GROW
        lo, hi = c - half, c + half
    # keep what could be solid once any one view's vote is taken out
    cand = (n >= need) & (k >= HULL_THRESHOLD * n - 1.0)
    idx = np.argwhere(cand)
    P = np.stack([ax[0][idx[:, 0]], ax[1][idx[:, 1]], ax[2][idx[:, 2]]], 1) if len(idx) else np.zeros((0, 3))
    return {"P": P, "k": k[cand].astype(np.int32), "n": n[cand].astype(np.int32), "voxel": voxel,
            "voters": int(len(use)), "need": need, "box": (lo, hi), "cut_off": bool(touches)}


def silhouette(hull, pr, is_voter):
    """The hull as view `pr` would see it, voted without that view. -> bool (H, W)"""
    import cv2
    fx, _fy, _cx, _cy, _R, _t, W, H, m = pr
    S = np.zeros((H, W), np.uint8)
    if not len(hull["P"]):
        return S.astype(bool)
    u, v, z, ok = _project(hull["P"], pr)
    k, n = hull["k"].copy(), hull["n"].copy()
    if is_voter:                                  # leave this view's own vote out
        own = np.zeros(len(k), np.int32)
        own[ok] = m[v[ok], u[ok]]
        k = k - own
        n = n - ok.astype(np.int32)
    solid = ok & (n >= hull["need"] - 1) & (k >= HULL_THRESHOLD * np.maximum(n, 1))
    if not solid.any():
        return S.astype(bool)
    S[v[solid], u[solid]] = 1
    # voxel centres leave gaps where a voxel is wider than a pixel: close by its footprint
    r = int(np.ceil(hull["voxel"] * fx / max(float(z[solid].min()), 50.0))) + 1
    S = cv2.morphologyEx(S, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    return S > 0


def piece(S, mask):
    """What the hull has that the mask lacks, without the thin rim. -> (bool (H, W), share of S)"""
    import cv2
    out = (S & ~mask).astype(np.uint8)
    core = cv2.morphologyEx(out, cv2.MORPH_OPEN, _open_kernel(S.shape)) > 0
    area = int(S.sum())
    return core, (float(core.sum()) / area if area else 0.0)


def check(views, masks, points, fell_back=None, progress=None):
    """Grade every mask. views: [(K, R, t, (w, h))], masks: uint8 arrays (any size), points: the
    sparse cloud in the rig's units, fell_back: {index: reason} from the build.

    -> {"views": [per-view dict], "note": str or None, "method": {...}, "work": {index: (S, piece)}}
    Per view: index, agreement (None: not measured), piece_share, piece_px, hull_px, reasons, score."""
    fell_back = fell_back or {}
    prep = depthmaps.vote_views(views, masks)
    nv = len(prep)
    method = {"agree_min": AGREE_MIN, "hull_threshold": HULL_THRESHOLD, "open_frac": OPEN_FRAC,
              "min_piece": MIN_PIECE, "check_edge": CHECK_EDGE}
    recs = [{"index": i, "agreement": None, "piece_share": None, "piece_px": 0, "hull_px": 0,
             "reasons": (["fell_back"] if i in fell_back else []), "score": 0.0} for i in range(nv)]
    sub = subject_points(points, prep)
    method["subject_points"] = int(len(sub))
    if len(sub) < MIN_SUBJECT_POINTS:
        for r in recs:
            r["score"] = 1.0 if r["reasons"] else 0.0
        return {"views": recs, "method": method, "work": {}, "prepared": prep, "subject": sub,
                "note": (f"only {len(sub)} sparse points lie on the subject in the masks' own judgement, "
                         f"too few to check the masks against each other (needs {MIN_SUBJECT_POINTS})")}
    agree = agreement(sub, prep)
    voters = np.array([bool(np.isfinite(a) and a >= AGREE_MIN) for a in agree])
    for i in fell_back:                           # the rough region is not a vote on the subject
        if 0 <= i < nv:
            voters[i] = False
    for i, a in enumerate(agree):
        if np.isfinite(a):
            recs[i]["agreement"] = round(float(a), 4)
            if a < AGREE_MIN:
                recs[i]["reasons"].append("holds_subject")
    note = None
    work = {}
    if voters.sum() < 8:
        note = (f"only {int(voters.sum())} masks hold the subject; that is too few to vote a hull, "
                "so only the first test ran")
        hull = None
    else:
        hull = vote(prep, voters, sub)
        method["voxel_mm"] = round(hull["voxel"], 3)
        method["voters"] = hull["voters"]
        if hull["cut_off"]:
            note = "the subject's hull still reaches the edge of the voting box after growing it twice"
    if hull is not None:
        for i in range(nv):
            S = silhouette(hull, prep[i], bool(voters[i]))
            hp = int(S.sum())
            recs[i]["hull_px"] = hp
            if hp >= MIN_HULL_PX:
                core, share = piece(S, prep[i][8])
                recs[i]["piece_share"] = round(share, 4)
                recs[i]["piece_px"] = int(core.sum())
                if share >= MIN_PIECE and core.sum() >= MIN_PIECE_PX:
                    if "fell_back" not in recs[i]["reasons"] and "holds_subject" not in recs[i]["reasons"]:
                        recs[i]["reasons"].append("piece_outside")
                    work[i] = (S, core)
                elif recs[i]["reasons"]:
                    work[i] = (S, core)
            if progress:
                progress(i + 1, nv)
    for r in recs:
        if not r["reasons"]:
            continue
        a = r["agreement"]
        r["score"] = round(max(r["piece_share"] or 0.0, (1.0 - a) if (a is not None and a < AGREE_MIN) else 0.0,
                               1e-4), 4)
    return {"views": recs, "note": note, "method": method, "work": work, "prepared": prep, "subject": sub,
            "voters": voters, "hull": hull}


def why(rec, fell_reason=None):
    """One clause for a person: what is wrong with this mask."""
    a, p = rec.get("agreement"), rec.get("piece_share")
    left = f"leaves out {100 * p:.1f}% of the subject" if p else None
    if "fell_back" in rec["reasons"]:
        s = "Vision found no object it could accept here, so the rough region was used"
        return s + (f"; it {left}" if left else "")
    if "holds_subject" in rec["reasons"]:
        s = f"holds only {100 * a:.0f}% of the subject's own points"
        return s + (f" and {left}" if left else "")
    return (left or "disagrees with the other views") + " that the other views agree on"


# --------------------------------------------------------------------------- repairs
def _fill_holes(m):
    import cv2
    h, w = m.shape
    ff = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 255)
    return m | cv2.bitwise_not(ff)[1:-1, 1:-1]


def _small(hard, shape):
    import cv2
    H, W = shape
    if hard.shape[:2] == (H, W):
        return hard > 0
    return cv2.resize((hard > 0).astype(np.uint8) * 255, (W, H), interpolation=cv2.INTER_AREA) >= 128


def _up(small, shape):
    """A check-size region at the mask's own size, with its stair-steps rounded off."""
    import cv2
    H, W = shape
    s = max(H / float(small.shape[0]), 1.0)
    f = cv2.resize(small.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    if s > 1.5:
        f = cv2.GaussianBlur(f, (0, 0), 0.6 * s)
    return f >= 0.5


def outline_shares(region, base):
    """Whose line bounds the region? -> (share of its outline along the mask, along the frame
    edge). What is left of 1 runs through neither: that stretch would be the hull's own outline."""
    import cv2
    pad = cv2.copyMakeBorder(region.astype(np.uint8), 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
    wall = cv2.copyMakeBorder(base.astype(np.uint8), 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
    frame = np.ones(pad.shape, bool)
    frame[2:-2, 2:-2] = False
    ring = (cv2.dilate(pad, np.ones((3, 3), np.uint8)) > 0) & (pad == 0)
    n = max(int(ring.sum()), 1)
    return float((ring & (wall > 0)).sum()) / n, float((ring & frame).sum()) / n


def reselect(instances, S):
    """Vision's objects tested against the hull instead of the rough region. `instances`: soft
    masks (uint8, any size). -> hard uint8 mask at the first kept instance's size, or None."""
    import cv2
    grown = cv2.dilate(S.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    chosen = None
    for sm in instances:
        hard = sm >= 128
        if not hard.any():
            continue
        small = _small(hard, S.shape)
        inside = float((small & grown).sum()) / max(int(small.sum()), 1)
        if inside < RESELECT_INSIDE:
            continue
        if chosen is None:
            chosen = hard
        else:
            if hard.shape != chosen.shape:
                hard = cv2.resize(hard.astype(np.uint8), (chosen.shape[1], chosen.shape[0]),
                                  interpolation=cv2.INTER_NEAREST) > 0
            chosen = chosen | hard
    if chosen is None:
        return None
    return _fill_holes(np.where(chosen, 255, 0).astype(np.uint8))


def repair(mask_full, S, instances=None, fell_back=False, subject_uv=None):
    """A mask to use instead of `mask_full` (uint8, the mask on disk) given the hull silhouette `S`
    (bool, check size). `instances`: Vision's soft masks for this view when they are still on
    disk. `subject_uv`: (u, v) of the subject's sparse points in this view at check size, to grade
    the result. -> (hard uint8 mask at mask_full's size, kind, approximate, whole) or None; `whole`
    says the mask in use contributed nothing (the result is Vision's object or the hull alone).

    The result is hard (0/255) and filled; the caller feathers it the way the build did."""
    import cv2
    full = mask_full.shape[:2]
    cur = np.where(mask_full >= 128, 255, 0).astype(np.uint8)
    cur_small = _small(cur, S.shape)
    area = max(int(S.sum()), 1)
    kind, base = None, cur
    if instances:
        got = reselect(instances, S)
        if got is not None:
            if got.shape != full:
                got = cv2.resize(got, (full[1], full[0]), interpolation=cv2.INTER_NEAREST)
            gain = (float((_small(got, S.shape) & S).sum()) - float((cur_small & S).sum())) / area
            if fell_back or gain >= RESELECT_GAIN:
                kind, base = "reselect", got
    if fell_back and kind is None:
        base = np.zeros(full, np.uint8)           # the rough region is not a mask of the subject: start empty
    base_small = _small(base, S.shape)
    core, share = piece(S, base_small)
    approximate = False
    out = base
    if share >= MIN_PIECE and core.sum() >= MIN_PIECE_PX:
        # the opening left a seam between the piece and the mask: take the rim beside the piece too
        near = cv2.dilate(core.astype(np.uint8), _open_kernel(S.shape)) > 0
        add = core | (S & ~base_small & near)
        by_mask, by_frame = outline_shares(add, base_small)
        if kind is None and not fell_back and by_mask + by_frame >= ENCLOSED_MIN:
            kind = "fill"
            approximate = by_mask < ENCLOSED_MIN  # closed by the frame edge: no outline is invented,
            #                                       but that the piece is subject is the hull's word alone
        else:
            kind = kind or "hull"
            approximate = True
            if HULL_GROW_PX > 0:
                k = 2 * HULL_GROW_PX + 1
                add = (cv2.dilate(add.astype(np.uint8), np.ones((k, k), np.uint8)) > 0) & ~base_small
        out = np.where(_up(add, full), 255, base).astype(np.uint8)
        s = max(full[0] / float(S.shape[0]), 1.0)
        kk = 2 * int(round(1.5 * s)) + 1
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)))
        out = _fill_holes(out)
    if kind is None:
        return None
    # the repair must pass the test that flagged the view
    out_small = _small(out, S.shape)
    _core2, share2 = piece(S, out_small)
    if share2 >= MIN_PIECE and _core2.sum() >= MIN_PIECE_PX:
        return None
    grown = cv2.dilate(S.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    spill = float((out_small & ~grown).sum()) / area
    old_spill = float((cur_small & ~grown).sum()) / area
    if spill > max(SPILL_MAX, old_spill):
        return None
    if subject_uv is not None and len(subject_uv[0]) >= 20:
        if float(out_small[subject_uv[1], subject_uv[0]].mean()) < AGREE_MIN:
            return None
    if float((out_small != cur_small).sum()) / area < MIN_PIECE / 2:
        return None                               # nothing was changed
    return out, kind, approximate, bool(fell_back and kind == "hull")


def subject_uv(sub, pr):
    u, v, _z, ok = _project(np.asarray(sub, np.float64), pr)
    return u[ok], v[ok]


# --------------------------------------------------------------------------- pictures
RED, BLUE, YELLOW, GREEN = (0, 0, 255), (255, 160, 0), (0, 255, 255), (80, 220, 80)


def _outline(img, region, colour, thick):
    import cv2
    cs, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cs, -1, colour, thick)


def preview(photo, mask, S, core, repaired=None):
    """Two panels side by side: the whole frame, and the piece in question enlarged.
    photo: BGR at any size; mask / S / core / repaired: bool at any sizes. Red is the mask in use,
    blue the hull, yellow the piece; with `repaired`, green is the repaired mask."""
    import cv2
    H, W = photo.shape[:2]

    def at(a):
        return cv2.resize(a.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)

    t = max(2, int(round(H / 400.0)))
    img = photo.copy()
    m, s, c = at(mask), at(S), at(core)
    if repaired is None:
        _outline(img, s, BLUE, t)
        _outline(img, m, RED, t)
        _outline(img, c, YELLOW, t)
    else:
        _outline(img, m, RED, t)
        _outline(img, at(repaired), GREEN, t)
    whole = cv2.resize(img, (max(1, int(round(W * PREVIEW_H / float(H)))), PREVIEW_H), interpolation=cv2.INTER_AREA)
    ys, xs = np.nonzero(c)
    if not len(ys):
        return whole
    r = int(max(0.14 * max(H, W), 0.75 * max(ys.max() - ys.min(), xs.max() - xs.min()) + 0.04 * max(H, W)))
    r = max(8, min(r, W // 2, H // 2))
    cy, cx = int(ys.mean()), int(xs.mean())
    y0, x0 = min(max(cy - r, 0), H - 2 * r), min(max(cx - r, 0), W - 2 * r)
    zoom = cv2.resize(img[y0:y0 + 2 * r, x0:x0 + 2 * r], (PREVIEW_H, PREVIEW_H), interpolation=cv2.INTER_AREA)
    gap = np.full((PREVIEW_H, 6, 3), 24, np.uint8)
    return np.hstack([whole, gap, zoom])
