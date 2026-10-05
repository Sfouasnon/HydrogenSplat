"""clearpath — take out the splats that sit where the camera itself went.

A walk-through's cameras travelled a line through the scene. Whatever a trained model has put ON
that line is not scenery: the camera was there, so the space was empty. Those splats are what the
optimiser used to explain differences between neighbouring frames (exposure, a leaf that moved),
and from the training cameras they are behind the lens or off to the side. A move that follows the
path flies straight through them: they are the haze and the specks nearest the lens.

On the first garden walk (2026-10-05, 8.2 million splats, stopped at 20,000 steps) the splats
within 0.3 m of the walked line were 1.0 % of the model and carried 72 % of the weight drawn from
less than half way to the surface in a frame 2 s into the walk; removing them took that weight
from 16.8 % of the picture to 6.9 % and left the picture's opacity where it was (0.993 -> 0.991).
One model, coarse numpy renders; not yet scored against hold-outs.

The line is the same one `hs move --preset walk` follows: consecutive placed frames joined inside
each RUN, never across a tear (hs/walk_path.py), since the stretch between two torn frames is not
somewhere the camera went. The default radius is 1.3 times the median step between consecutive
placed frames -- relative to the capture, because a solve's scale is only as good as what fixed it.
"""
import os

import numpy as np

from . import walk_path

AUTO_STEPS = 1.3           # default radius, in median steps between consecutive placed frames
CHUNK = 250_000


def walked_line(rig_npz, quality_json=None):
    """-> (a, b, median_step_mm, tears): the segments of the walked line, mm, (m, 3) each."""
    T = walk_path.capture_table(rig_npz, quality_json)
    C = T["C"]
    if len(C) < 2:
        raise ValueError("fewer than two placed frames: there is no path")
    runs, tears, _pace = walk_path.find_runs(T["num"], T["t"], C)
    spans = runs if runs else [(0, len(C) - 1)]
    a = np.concatenate([C[i0:i1] for i0, i1 in spans])
    b = np.concatenate([C[i0 + 1:i1 + 1] for i0, i1 in spans])
    step = float(np.median(np.linalg.norm(np.diff(C, axis=0), axis=1)))
    return a, b, step, tears


def distance_to_line(xyz, a, b):
    """Distance from each row of xyz to the nearest of the segments a[k] -> b[k] (same units)."""
    xyz = np.asarray(xyz, np.float32)
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    ab = b - a
    den = np.maximum((ab * ab).sum(1), 1e-12)
    best = np.full(len(xyz), np.inf, np.float32)
    for k in range(len(a)):
        v = xyz - a[k]
        s = np.clip((v @ ab[k]) / den[k], 0.0, 1.0)
        v -= s[:, None] * ab[k]
        np.minimum(best, (v * v).sum(1), out=best)
    return np.sqrt(best)


def rewrite(ply_in, ply_out, drop_of, removed_out=None, progress=None):
    """Copy ply_in to ply_out without the splats `drop_of(xyz)` marks (xyz: one chunk's centres,
    metres, float32 -> boolean array). Reads and writes in chunks, so an 8-million-splat model
    needs no more memory than a chunk; ply_out may be ply_in itself (it is written beside and
    renamed into place). Rows are copied byte for byte.
    -> {"splats_in", "removed", "splats_out", "opacity_mass_kept", "removed_median_opacity"}."""
    from .splatweights import TMAP
    from .stages.merge import read_header
    head, n, props, off = read_header(ply_in)
    dt = np.dtype([(name, TMAP[typ]) for name, typ in props])
    rows = np.memmap(ply_in, dtype=dt, mode="r", offset=off, shape=(n,))
    drop = np.zeros(n, bool)
    op_all = op_drop = 0.0
    ops = []
    for s in range(0, n, CHUNK):
        blk = rows[s:s + CHUNK]
        xyz = np.stack([blk["x"], blk["y"], blk["z"]], 1).astype(np.float32)
        d = np.asarray(drop_of(xyz), bool)
        drop[s:s + CHUNK] = d
        o = 1.0 / (1.0 + np.exp(-blk["opacity"].astype(np.float64)))
        op_all += float(o.sum()); op_drop += float(o[d].sum())
        ops.append(o[d])
        if progress:
            progress(min(s + CHUNK, n), n)

    def write(path, keep):
        m = int(keep.sum())
        lines = [f"element vertex {m}" if (ln.split()[:2] == ["element", "vertex"]) else ln
                 for ln in head.decode("ascii").split("\n")]
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".partial"
        with open(tmp, "wb") as f:
            f.write("\n".join(lines).encode("ascii"))
            for s in range(0, n, CHUNK):
                k = keep[s:s + CHUNK]
                if k.any():
                    f.write(np.ascontiguousarray(rows[s:s + CHUNK][k]).tobytes())
        return tmp, m

    # both files are written whole before either is put in place: ply_out may be the file being read
    made = [(ply_out,) + write(ply_out, ~drop)]
    if removed_out:
        made.append((removed_out,) + write(removed_out, drop))
    del rows
    for path, tmp, _m in made:
        os.replace(tmp, path)
    ops = np.concatenate(ops) if ops else np.zeros(0)
    return {"splats_in": int(n), "removed": int(drop.sum()), "splats_out": made[0][2],
            "opacity_mass_kept": 1.0 - op_drop / max(op_all, 1e-12),
            "removed_median_opacity": float(np.median(ops)) if len(ops) else None}


def clear(ply_in, ply_out, a_mm, b_mm, radius_mm, removed_out=None, progress=None):
    """Copy ply_in to ply_out without the splats whose centre is within radius_mm of the line."""
    a, b, r = a_mm / 1000.0, b_mm / 1000.0, radius_mm / 1000.0       # the PLY is in metres
    return rewrite(ply_in, ply_out, lambda xyz: distance_to_line(xyz, a, b) < r,
                   removed_out=removed_out, progress=progress)


# --------------------------------------------------------------------------- erased by hand
def boxes_from(spec):
    """The boxes of an erase file, checked. Each is what the app's viewer saw when a box was
    dragged: ``vp`` (4 rows of 4: projection x view, world metres -> clip, w = distance in front of
    the camera), the box in normalised device coordinates (x right, y up, -1..1) and how far in
    front of the camera it reaches (``near``..``far``, metres)."""
    out = []
    for k, b in enumerate(spec.get("boxes") or []):
        try:
            vp = np.array(b["vp"], np.float32)
            box = tuple(float(b[key]) for key in ("x0", "x1", "y0", "y1"))
            near, far = float(b.get("near", 0.0)), float(b["far"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"box {k + 1} of the erase file is not readable ({e})")
        if vp.shape != (4, 4) or not np.isfinite(vp).all():
            raise ValueError(f"box {k + 1} of the erase file has no 4 x 4 view")
        if not (box[0] < box[1] and box[2] < box[3] and far > near >= 0):
            raise ValueError(f"box {k + 1} of the erase file is empty")
        out.append((vp, box, near, far))
    if not out:
        raise ValueError("the erase file holds no boxes")
    return out


def inside_boxes(xyz, boxes):
    """Splat centres (metres) inside any box: in front of that camera, within its rectangle on
    screen, and no further from the camera than the box reaches. The same test the viewer ran."""
    xyz = np.asarray(xyz, np.float32)
    h = np.concatenate([xyz, np.ones((len(xyz), 1), np.float32)], 1)
    hit = np.zeros(len(xyz), bool)
    for vp, (x0, x1, y0, y1), near, far in boxes:
        c = h @ vp.T
        w = c[:, 3]
        ok = w > 1e-6
        with np.errstate(divide="ignore", invalid="ignore"):
            x, y = c[:, 0] / w, c[:, 1] / w
        hit |= ok & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1) & (w >= near) & (w <= far)
    return hit
