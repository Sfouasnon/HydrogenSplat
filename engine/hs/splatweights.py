"""splatweights — which splat paints which part of which photograph, in numpy.

The shared machinery under `hs split` (subject / background labels lifted from 2D masks) and
`hs prune --score` (importance, top-contributor guard, photometric blame). For one view it
answers: for every coarse image cell, which splats did the renderer composite there, front to
back, and with what weight ``w = alpha * T``. That is the quantity both FlashSplat's closed-form
label uplift and LightGaussian's importance are sums over.

What it models, per view (the standard 3DGS forward pass, on a coarse grid):

  * live splats only (sigmoid(opacity) >= MIN_OPACITY), centres ``Xc = X @ R.T + t`` in the
    rig's mm (PLY positions are metres, so x1000), culled behind a near plane;
  * the EWA footprint ``Sigma2D = J W Sigma3D W^T J^T`` with ``Sigma3D = R S S^T R^T`` from the
    PLY's log scales and wxyz quaternion, the pinhole Jacobian with 3DGS's 1.3x-FOV clamp, plus
    the 0.3 px dilation every 3DGS renderer adds;
  * the Gaussian evaluated at CELL centres (``--cell`` px) inside its bounding box, opacity-
    weighted, alpha clamped to 0.99 and dropped under 1/255 (of the pixel alpha; the box is the
    opacity-aware radius where that happens, capped at 3 sigma), then per cell sorted front to back
    by the splat's view depth and composited with a cumulative log-transmittance -- one lexsort-
    free int64 argsort per view, no Python loop over cells or splats.

One decision the brief left open, and why. Evaluating the pixel-level Gaussian only at cell
centres makes a splat smaller than a cell hit or miss depending on where its centre lands --
the fine detail and the thin floaters are exactly those splats. So the footprint is prefiltered
to the cell, the way a renderer run at 1/cell resolution would do it: the pixel covariance
(dilated) is expressed in cell units and widened by CELL_FILTER (a quarter of a cell squared,
i.e. sigma >= 0.5 cell), and the peak alpha is scaled by ``sqrt(det(before) / det(after))``
(Mip-Splatting's 2D-filter compensation), so the splat's integral is unchanged. A large splat is
unchanged (the factor tends to 1); a sub-cell splat contributes its area-weighted share of the
cells it lands in instead of all or nothing. Why a quarter and not a cell-wide box's 1/12: at
1/12 a sub-cell splat sampled at cell centres still swings between 0.49x and 0.90x of its true
coverage with sub-cell position (tests/test_splatweights.py); at 1/4 the sampled sum matches the
integral to about 1 %. With ``cell=1`` the filter and the compensation vanish and this is the
plain pixel renderer. The bounding box is 3 sigma
(``extent``), not 2: a 2-sigma cut drops 13.5 % of a 2D Gaussian's mass, 3 sigma drops 1.1 %.

Colour is the DC term only (``0.5 + SH_C0 * f_dc``), i.e. the view-independent colour; the
composited cell colour is what the renderer would show with the SH bands switched off. The
photometric comparison in `hs prune --score` is therefore coarse in the same way.

Output per view (ViewWeights): sparse triples ``(splat, cell, w)`` sorted by cell then depth,
per-cell ``T`` (final transmittance) and ``rgb`` (composited DC colour, background excluded).
"""
import os
from collections import namedtuple

import numpy as np

from . import events

SH_C0 = 0.28209479177387814
MIN_OPACITY = 0.05          # "live": what every downstream consumer treats as a real splat
ALPHA_MIN = 1.0 / 255.0     # the renderer's own cut
ALPHA_MAX = 0.99
T_MIN = 1e-4                # the renderer stops compositing a pixel past this transmittance
DILATION_PX = 0.3
CELL_FILTER = 0.25          # cell^2: prefilter variance added at cell > 1 (module docstring)
EXTENT_SIGMA = 3.0
MAX_CELLS_PER_SPLAT = 1024  # 32x32 cells = 256 px at cell 8; bigger footprints are clipped (and counted)
MIN_WEIGHT = 1e-4           # triples under this are dropped from the output (they are still composited)
CHUNK_SAMPLES = 1 << 22     # cell evaluations per vectorised chunk (~4M; bounds peak memory)

TMAP = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1",
        "char": "i1", "int8": "i1", "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
        "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4"}

ViewWeights = namedtuple("ViewWeights", "splat cell w T rgb shape clipped")


# --------------------------------------------------------------------------- PLY
class Splats:
    """A Gaussian PLY in memory: the raw structured rows (for writing subsets back out) and the
    derived quantities the rasteriser needs, in the rig's mm."""

    def __init__(self, path):
        from .stages.merge import read_header
        self.path = path
        self.head, self.n, self.props, off = read_header(path)
        try:
            dt = np.dtype([(name, TMAP[typ]) for name, typ in self.props])
        except KeyError as e:
            raise events.StageError(f"{os.path.basename(path)}: unknown PLY property type {e}")
        with open(path, "rb") as f:
            f.seek(off)
            body = f.read(self.n * dt.itemsize)
        if len(body) < self.n * dt.itemsize:
            raise events.StageError(f"{os.path.basename(path)} is truncated: {len(body)} of "
                                    f"{self.n * dt.itemsize} body bytes")
        self.arr = np.frombuffer(body, dtype=dt, count=self.n)
        have = set(dt.names)
        need = {"x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"}
        missing = sorted(need - have)
        if missing:
            raise events.StageError(f"{os.path.basename(path)} is not a Gaussian splat PLY: no {', '.join(missing)}")
        a = self.arr
        self.xyz_mm = np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64) * 1000.0
        self.opacity = 1.0 / (1.0 + np.exp(-a["opacity"].astype(np.float64)))
        self.scale_mm = np.exp(np.stack([a["scale_0"], a["scale_1"], a["scale_2"]], 1).astype(np.float64)) * 1000.0
        q = np.stack([a["rot_0"], a["rot_1"], a["rot_2"], a["rot_3"]], 1).astype(np.float64)
        q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        self.quat = q
        if {"f_dc_0", "f_dc_1", "f_dc_2"} <= have:
            dc = np.stack([a["f_dc_0"], a["f_dc_1"], a["f_dc_2"]], 1).astype(np.float64)
            self.rgb = np.clip(0.5 + SH_C0 * dc, 0.0, 1.0)
        else:
            self.rgb = np.full((self.n, 3), 0.5)
        self.live = self.opacity >= MIN_OPACITY
        self._cov = None

    @property
    def cov_mm(self):
        """(n, 3, 3) world covariance in mm^2, Sigma = R S S^T R^T."""
        if self._cov is None:
            w, x, y, z = self.quat.T
            R = np.empty((self.n, 3, 3))
            R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
            R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
            R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
            M = R * self.scale_mm[:, None, :]
            self._cov = M @ M.transpose(0, 2, 1)
        return self._cov


def write_ply(path, sp, keep=None, extra=None):
    """Write `sp`'s rows (all, or the boolean/index subset `keep`) with the original header and
    property layout. `extra` = (name, float values for every row) appends one float property."""
    rows = sp.arr if keep is None else sp.arr[keep]
    head = sp.head.decode("ascii")
    lines = head.split("\n")
    out = []
    for ln in lines:
        w = ln.split()
        if len(w) == 3 and w[0] == "element" and w[1] == "vertex":
            ln = f"element vertex {len(rows)}"
        if ln == "end_header" and extra is not None:
            out.append(f"property float {extra[0]}")
        out.append(ln)
    head = "\n".join(out)
    if extra is not None:
        vals = np.asarray(extra[1], dtype="<f4")
        vals = vals if keep is None else vals[keep]
        dt = np.dtype(rows.dtype.descr + [(extra[0], "<f4")])
        full = np.empty(len(rows), dtype=dt)
        for name in rows.dtype.names:
            full[name] = rows[name]
        full[extra[0]] = vals
        rows = full
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".partial"
    with open(tmp, "wb") as f:
        f.write(head.encode("ascii"))
        f.write(np.ascontiguousarray(rows).tobytes())
    os.replace(tmp, path)


# --------------------------------------------------------------------------- one view
def grid_shape(w, h, cell):
    return int(np.ceil(h / cell)), int(np.ceil(w / cell))


def view_weights(sp, K, R, t, wh, cell=8, max_cells=MAX_CELLS_PER_SPLAT, extent=EXTENT_SIGMA,
                 min_weight=MIN_WEIGHT, only=None):
    """Composite every live splat into one view's cell grid. -> ViewWeights.

    `K, R, t` are the view's intrinsics and world->camera pose (rig.npz, mm); `wh` its canvas.
    `only`: optional boolean mask over splats further restricting what is rendered (tests use
    it to render a single cluster). Triples come back sorted by cell, then front to back."""
    W, H = int(wh[0]), int(wh[1])
    gh, gw = grid_shape(W, H, cell)
    ncell = gh * gw
    live = sp.live if only is None else (sp.live & only)
    ids = np.flatnonzero(live)
    K = np.asarray(K, np.float64); R = np.asarray(R, np.float64); t = np.asarray(t, np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    Xc = sp.xyz_mm[ids] @ R.T + t
    z = Xc[:, 2]
    near = 1e-3 * max(float(np.median(np.abs(z))) if len(z) else 1.0, 1e-9)
    front = z > near
    ids, Xc, z = ids[front], Xc[front], z[front]
    x, y = Xc[:, 0], Xc[:, 1]
    # the Jacobian is evaluated at a clamped position, as in 3DGS, so a splat far outside the
    # frustum does not get an exploding footprint
    limx, limy = 1.3 * 0.5 * W / fx, 1.3 * 0.5 * H / fy
    tx = np.clip(x / z, -limx, limx) * z
    ty = np.clip(y / z, -limy, limy) * z
    j00, j02 = fx / z, -fx * tx / (z * z)
    j11, j12 = fy / z, -fy * ty / (z * z)
    S = sp.cov_mm[ids]
    Sc = R @ S @ R.T                                          # camera-frame covariance, (n,3,3)
    a = j00 * j00 * Sc[:, 0, 0] + 2 * j00 * j02 * Sc[:, 0, 2] + j02 * j02 * Sc[:, 2, 2]
    b = (j00 * j11 * Sc[:, 0, 1] + j00 * j12 * Sc[:, 0, 2] + j02 * j11 * Sc[:, 1, 2]
         + j02 * j12 * Sc[:, 2, 2])
    c = j11 * j11 * Sc[:, 1, 1] + 2 * j11 * j12 * Sc[:, 1, 2] + j12 * j12 * Sc[:, 2, 2]
    # pixel footprint with the renderer's dilation, then in cell units, prefiltered by the cell box
    c2 = float(cell) * float(cell)
    a_d, b_d, c_d = (a + DILATION_PX) / c2, b / c2, (c + DILATION_PX) / c2
    box = (1.0 - 1.0 / c2) * CELL_FILTER                       # 0 at cell 1
    a_f, b_f, c_f = a_d + box, b_d, c_d + box
    det_d = np.maximum(a_d * c_d - b_d * b_d, 1e-30)
    det_f = np.maximum(a_f * c_f - b_f * b_f, 1e-30)
    comp = np.sqrt(det_d / det_f)                             # 1 at cell 1 and for large splats
    peak = sp.opacity[ids] * comp
    # the 1/255 cut is the renderer's, on the PIXEL alpha; the prefiltered cell alpha of a small
    # splat is that alpha spread thinner, so its cut scales by the same factor (otherwise a sub-
    # cell splat would lose its tails to a threshold no pixel of it ever fell under)
    amin = ALPHA_MIN * comp
    A, B, C = c_f / det_f, -b_f / det_f, a_f / det_f           # conic (inverse covariance)
    uc = (fx * x / z + cx) / cell                              # centre, cell units (cell j spans [j, j+1))
    vc = (fy * y / z + cy) / cell
    # axis-aligned bbox of the ellipse, out to `extent` sigma or to where alpha falls under 1/255,
    # whichever is nearer (the opacity-aware radius Brush uses): a faint splat needs a small box
    ext = np.minimum(extent, np.sqrt(2.0 * np.log(np.maximum(peak / amin, 1.0))))
    rx, ry = ext * np.sqrt(a_f), ext * np.sqrt(c_f)
    j0 = np.maximum(np.ceil(uc - rx - 0.5), 0)
    j1 = np.minimum(np.floor(uc + rx - 0.5), gw - 1)
    i0 = np.maximum(np.ceil(vc - ry - 0.5), 0)
    i1 = np.minimum(np.floor(vc + ry - 0.5), gh - 1)
    ok = (j1 >= j0) & (i1 >= i0) & (peak >= amin) & np.isfinite(uc) & np.isfinite(vc)
    bw = np.where(ok, j1 - j0 + 1, 0).astype(np.int64)
    bh = np.where(ok, i1 - i0 + 1, 0).astype(np.int64)
    # cap the footprint: a splat wider than max_cells keeps a window of that many cells about its
    # centre (clamped into the image). Counted, because it under-weights exactly those splats.
    over = bw * bh > max_cells
    clipped = int(over.sum())
    if clipped:
        f = np.sqrt(max_cells / (bw[over] * bh[over]).astype(np.float64))
        nbw = np.maximum(1, np.floor(bw[over] * f)).astype(np.int64)
        nbh = np.maximum(1, np.floor(bh[over] * f)).astype(np.int64)
        j0[over] = np.clip(np.round(uc[over] - 0.5 - (nbw - 1) / 2.0), 0, gw - nbw)
        i0[over] = np.clip(np.round(vc[over] - 0.5 - (nbh - 1) / 2.0), 0, gh - nbh)
        bw[over], bh[over] = nbw, nbh
    j0 = j0.astype(np.int64); i0 = i0.astype(np.int64)

    keep = np.flatnonzero(bw * bh > 0)
    counts = (bw * bh)[keep]
    # depth rank per splat: the per-cell sort key is (cell, depth), packed into one int64
    drank = np.empty(len(ids), np.int64)
    drank[np.argsort(z, kind="stable")] = np.arange(len(ids))

    parts_s, parts_c, parts_a, parts_k = [], [], [], []
    A32, B32, C32, peak32, amin32 = (v.astype(np.float32) for v in (A, B, C, peak, amin))
    cc = np.cumsum(counts)
    start = 0
    while start < len(keep):
        done_before = int(cc[start - 1]) if start else 0
        stop = max(start + 1, int(np.searchsorted(cc, done_before + CHUNK_SAMPLES, side="right")))
        sel = keep[start:stop]
        cnt = counts[start:stop]
        start = stop
        rep = np.repeat(np.arange(len(sel)), cnt)
        first = np.cumsum(cnt) - cnt
        local = np.arange(int(cnt.sum()), dtype=np.int64) - np.repeat(first, cnt)
        s = sel[rep]
        oy = local // bw[s]
        ox = local - oy * bw[s]
        jj = j0[s] + ox
        ii = i0[s] + oy
        dx = (jj + 0.5 - uc[s]).astype(np.float32)
        dy = (ii + 0.5 - vc[s]).astype(np.float32)
        power = -0.5 * (A32[s] * dx * dx + C32[s] * dy * dy) - B32[s] * dx * dy
        alpha = np.minimum(ALPHA_MAX, peak32[s] * np.exp(np.minimum(power, 0.0)))
        m = alpha >= amin32[s]
        s, jj, ii, alpha = s[m], jj[m], ii[m], alpha[m]
        cellid = ii * gw + jj
        parts_s.append(s.astype(np.int32))
        parts_c.append(cellid.astype(np.int32))
        parts_a.append(alpha.astype(np.float64))
        parts_k.append(cellid * len(ids) + drank[s])

    T = np.ones(ncell, np.float64)
    rgb = np.zeros((ncell, 3), np.float64)
    if not parts_s:
        return ViewWeights(np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.float32),
                           T.reshape(gh, gw).astype(np.float32), rgb.reshape(gh, gw, 3).astype(np.float32),
                           (gh, gw), clipped)
    s = np.concatenate(parts_s); cellid = np.concatenate(parts_c)
    alpha = np.concatenate(parts_a); key = np.concatenate(parts_k)
    del parts_s, parts_c, parts_a, parts_k
    order = np.argsort(key)          # keys are unique per (cell, splat): no stable sort needed
    s, cellid, alpha = s[order], cellid[order], alpha[order]
    del key, order
    # front-to-back: T_before = exp(sum of log(1-alpha) over earlier splats in the same cell)
    la = np.log1p(-alpha)
    cs = np.cumsum(la)
    newseg = np.empty(len(cellid), bool)
    newseg[0] = True
    newseg[1:] = cellid[1:] != cellid[:-1]
    starts = np.flatnonzero(newseg)
    segid = np.cumsum(newseg) - 1
    base = (cs - la)[starts]                                   # exclusive sum at each segment start
    excl = cs - la - base[segid]
    Tb = np.exp(excl)
    w = np.where(Tb >= T_MIN, alpha * Tb, 0.0)
    ends = np.append(starts[1:], len(cellid)) - 1
    ucell = cellid[starts]
    T[ucell] = np.maximum(np.exp(cs[ends] - base), 0.0)
    gid = ids[s]
    col = sp.rgb[ids].T.copy()                                 # (3, n_visible), gathered once per channel
    for ch in range(3):
        rgb[:, ch] = np.bincount(cellid, weights=w * col[ch][s], minlength=ncell)
    m = w >= min_weight
    return ViewWeights(gid[m].astype(np.int32), cellid[m], w[m].astype(np.float32),
                       T.reshape(gh, gw).astype(np.float32), rgb.reshape(gh, gw, 3).astype(np.float32),
                       (gh, gw), clipped)


def cell_max_mask(vw):
    """Boolean over vw's triples: this triple holds its cell's largest weight (ties all count).
    Relies on the triples being grouped by cell, which view_weights guarantees."""
    if len(vw.w) == 0:
        return np.zeros(0, bool)
    newseg = np.empty(len(vw.cell), bool)
    newseg[0] = True
    newseg[1:] = vw.cell[1:] != vw.cell[:-1]
    starts = np.flatnonzero(newseg)
    segid = np.cumsum(newseg) - 1
    mx = np.maximum.reduceat(vw.w, starts)
    return vw.w >= mx[segid]


# --------------------------------------------------------------------------- images
def cell_mean(img, cell):
    """Block mean over `cell` x `cell` pixels; edge cells average only the pixels they hold."""
    img = np.asarray(img, np.float64)
    h, w = img.shape[:2]
    gh, gw = grid_shape(w, h, cell)
    pad = [(0, gh * cell - h), (0, gw * cell - w)] + [(0, 0)] * (img.ndim - 2)
    ones = np.pad(np.ones((h, w)), pad[:2])
    im = np.pad(img, pad)
    sh = (gh, cell, gw, cell) + img.shape[2:]
    s = im.reshape(sh).sum(axis=(1, 3))
    n = ones.reshape(gh, cell, gw, cell).sum(axis=(1, 3))
    return s / (n[..., None] if img.ndim == 3 else n)


def srgb_to_linear(x):
    x = np.clip(np.asarray(x, np.float64), 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


# --------------------------------------------------------------------------- rig + views
def load_rig(path):
    if not os.path.exists(path):
        raise events.StageError(f"no rig at {path}", hint="hs solve first")
    G = np.load(path, allow_pickle=True)
    names = [str(x) for x in G["names"]]
    wh = G["wh"] if "wh" in G.files else np.array([[int(G["w"]), int(G["h"])]] * len(names))
    return {"names": names, "K": G["K"].astype(np.float64), "R": G["R"].astype(np.float64),
            "t": G["t"].astype(np.float64), "wh": np.asarray(wh).astype(int)}


def view_key(name):
    """rig.npz view name -> train's exclude key: ``cap064_L`` -> ``L/cap064``."""
    if len(name) > 2 and name[-2] in "_-" and name[-1] in "LR":
        return f"{name[-1]}/{name[:-2]}"
    return f"L/{name}"


def image_path(dataset, name, top="images"):
    """The undistorted training image (or with top='masks', its mask) for a rig view name."""
    eye, stem = view_key(name).split("/", 1)
    exts = (".png",) if top == "masks" else (".jpg", ".jpeg", ".png")
    for ext in exts:
        p = os.path.join(dataset, top, eye, stem + ext)
        if os.path.exists(p):
            return p
    return None


def model_exclude(pj, ply):
    """What the model was trained without, from its own record: an archive manifest beside the
    ply, else the train stage's fingerprint when the ply is train's export. -> (set | None, source)."""
    import json
    man = os.path.join(os.path.dirname(os.path.abspath(ply)), "manifest.json")
    if os.path.exists(man):
        try:
            m = json.load(open(man))
            fp = (m or {}).get("train_dataset_fingerprint") or {}
            if "excluded_views" in fp:
                return set(fp["excluded_views"] or []), "archive manifest"
        except (OSError, ValueError):
            pass
    fp = (pj.stage("train").get("metrics") or {}).get("dataset_fingerprint") or {}
    exp = (pj.stage("train").get("metrics") or {}).get("final_export")
    if "excluded_views" in fp and (not exp or os.path.dirname(os.path.abspath(pj.path(exp))) == os.path.dirname(os.path.abspath(ply))):
        return set(fp["excluded_views"] or []), "train stage"
    return None, None


def resolve_exclude(pj, ply, text, names):
    """--exclude NAMES | @holdout | @FILE.json | (omitted: the model's own record). -> (set, source).

    Bare names in a hold-out file are captures, so both eyes go; a typed list follows
    `hs train --exclude` exactly (a bare id there is an array camera, L/<id>)."""
    import json
    from .stages.train import parse_exclude
    text = (text or "").strip()
    if not text:
        ex, src = model_exclude(pj, ply)
        return (ex or set()), (src or "none (no record of the model's hold-outs)")
    if text.startswith("@") and "," not in text and text != "@holdout":
        ref = text[1:]
        path = pj.path("solve", "holdout.json") if ref == "holdout" else (ref if os.path.isabs(ref) else pj.path(ref))
        if not os.path.exists(path):
            raise events.StageError(f"--exclude {text}: no {pj.rel(path) if path.startswith(pj.root) else path}",
                                    hint="hs cameras --holdout N --write, or list the views: --exclude L/cap064,R/cap064")
        h = json.load(open(path))
        out = set()
        for nm in h.get("names") or []:
            nm = os.path.splitext(str(nm))[0]
            if "/" in nm or nm[-2:] in ("_L", "_R"):
                out |= parse_exclude(nm)
            else:
                out |= {f"L/{nm}", f"R/{nm}"}
        return out, pj.rel(path) if path.startswith(pj.root) else path
    # @holdout, and mixes like @holdout,L/cap099, follow hs train's own parser (both eyes per capture)
    return parse_exclude(text, pj.root), ("solve/holdout.json + --exclude" if "@holdout" in text else "--exclude")


def training_views(names, exclude):
    return [v for v, nm in enumerate(names) if view_key(nm) not in exclude]
