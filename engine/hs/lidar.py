"""LiDAR scan: load a phone scan (Polycam / Scaniverse PLY, OBJ, XYZ/CSV), register it to a
solve, and read metric scale, up and the ground plane off it.

Pure functions over numpy arrays, like hs/board.py; `hs scale --lidar` (stages/scale.py) is the
caller. Nearest neighbours are cv2.flann kd-trees (`NN`), so the dependencies stay numpy + cv2.

Frames and units. Everything this module returns is in millimetres. A scan is read in its own
file units and multiplied into mm by `load_scan` (declared by --units, else by a header
comment, else inferred from its extent). The solve is rig.npz's frame: mm on a stereo rig,
arbitrary units on an unscaled mono/array one. Registration runs **solve -> scan**
(``y = s R x + t``, x solve, y scan mm): the solve's sparse points are a subset of what the scan
saw, so every solve point has a surface under it and the scan's extra rooms do not have to be
explained away; ``s`` is then exactly the factor that turns the solve's units into mm.

Registration.
  global_register  RANSAC over congruent point triples. A triple x1, x2, x3 is sampled in the
                   solve (well separated, not collinear, on locally planar surface, rare normal
                   directions preferred); its first edge is looked up in a hash of the scan's
                   point pairs keyed by three scale-invariant angles (each point's PCA normal to
                   the edge, and normal to normal), so the candidate (y1, y2) come from a table
                   instead of an O(N^2) scan; the edge fixes the scale (or, with the scale known,
                   must match the distance), the third point must land on the scan with matching
                   distance ratios and normal, Umeyama on the triple is the hypothesis; a
                   distance grid prefilters, a short trimmed ICP refines the best few, and they
                   are ranked by truncated-quadratic cost. Failure is reported, not hidden:
                   ``aligned`` is False when the best inlier fraction is under ``min_inlier``,
                   when a different pose fits nearly as well (``ambiguous``), or when the
                   overlap's shape cannot fix the pose or the scale (``constraints``).
  icp              point-to-point, trimmed (the best ``trim`` of the correspondences each
                   iteration, plus any within ``keep_within``), Sim(3) or SE(3) step (Umeyama),
                   the scale bounded about its initial value so a partial overlap cannot shrink
                   the solve onto one wall.

Clean the solve with ``denoise`` *before* subsampling it: a voxel subsample keeps every stray.
"""
import io
import os
import re
import time
from dataclasses import dataclass, field

import numpy as np

UNIT_MM = {"m": 1000.0, "cm": 10.0, "mm": 1.0}


# ======================================================================== loading

@dataclass
class Scan:
    points: np.ndarray                      # (N, 3) float64, mm
    colors: np.ndarray = None               # (N, 3) uint8 RGB, or None
    normals: np.ndarray = None              # (N, 3) float64 as stored in the file, or None
    meta: dict = field(default_factory=dict)


_PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2",
              "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
              "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}
_UNIT_WORDS = {"m": "m", "meter": "m", "meters": "m", "metre": "m", "metres": "m",
               "cm": "cm", "centimeter": "cm", "centimeters": "cm", "centimetre": "cm", "centimetres": "cm",
               "mm": "mm", "millimeter": "mm", "millimeters": "mm", "millimetre": "mm", "millimetres": "mm"}
_UNIT_RE = re.compile(r"\bunits?\b\s*[:=]?\s*\(?\s*([a-z]+)", re.I)
SH_C0 = 0.28209479177387814


def _declared_units(texts):
    """A unit named in header comments ("comment units: meters", "x (mm)"), or None."""
    for s in texts:
        m = _UNIT_RE.search(s)
        if m and m.group(1).lower() in _UNIT_WORDS:
            return _UNIT_WORDS[m.group(1).lower()]
    return None


def _read_ply_header(f):
    lines = []
    first = f.readline()
    if first.strip() != b"ply":
        raise ValueError("not a PLY file (no 'ply' magic on the first line)")
    while True:
        ln = f.readline()
        if not ln:
            raise ValueError("PLY header has no end_header")
        s = ln.decode("ascii", "replace").strip()
        if s == "end_header":
            break
        lines.append(s)
    fmt, elements, comments = None, [], []
    for s in lines:
        w = s.split()
        if not w:
            continue
        if w[0] == "format":
            fmt = w[1] if len(w) > 1 else None
        elif w[0] in ("comment", "obj_info"):
            comments.append(s.split(None, 1)[1] if len(w) > 1 else "")
        elif w[0] == "element" and len(w) >= 3:
            elements.append({"name": w[1], "count": int(w[2]), "props": []})
        elif w[0] == "property" and elements:
            if len(w) >= 5 and w[1] == "list":
                elements[-1]["props"].append({"name": w[4], "list": (w[2], w[3])})
            elif len(w) >= 3:
                elements[-1]["props"].append({"name": w[2], "type": w[1]})
    if fmt not in ("ascii", "binary_little_endian", "binary_big_endian"):
        raise ValueError(f"unknown PLY format {fmt!r}")
    return fmt, elements, comments, f.tell()


def _ply_dtype(props, endian):
    try:
        return np.dtype([(p["name"], endian + _PLY_TYPES[p["type"]]) for p in props])
    except KeyError as e:
        raise ValueError(f"unknown PLY property type {e}")


def _skip_binary_element(f, el, endian):
    """Move past one element of a binary PLY (only needed for elements before 'vertex')."""
    if not any("list" in p for p in el["props"]):
        f.seek(_ply_dtype(el["props"], endian).itemsize * el["count"], 1)
        return
    for _ in range(el["count"]):                      # rare: a list element ahead of the vertices
        for p in el["props"]:
            if "list" in p:
                ct = np.dtype(endian + _PLY_TYPES[p["list"][0]])
                n = int(np.frombuffer(f.read(ct.itemsize), ct)[0])
                f.seek(n * np.dtype(_PLY_TYPES[p["list"][1]]).itemsize, 1)
            else:
                f.seek(np.dtype(_PLY_TYPES[p["type"]]).itemsize, 1)


def _pick(names, *cands):
    low = {n.lower(): n for n in names}
    for c in cands:
        if c in low:
            return low[c]
    return None


def _columns(arr_get, names):
    """x/y/z, colour and normal columns out of named vertex properties."""
    xs = [_pick(names, a) for a in ("x", "y", "z")]
    if None in xs:
        alt = [_pick(names, a) for a in ("px", "py", "pz")]
        if None in alt:
            raise ValueError(f"no x, y, z vertex properties (have: {', '.join(names)})")
        xs = alt
    pts = np.stack([np.asarray(arr_get(n), np.float64) for n in xs], 1)
    rgb, colour_from = None, None
    for trip in (("red", "green", "blue"), ("r", "g", "b"), ("diffuse_red", "diffuse_green", "diffuse_blue")):
        cols = [_pick(names, c) for c in trip]
        if None not in cols:
            rgb = np.stack([np.asarray(arr_get(c), np.float64) for c in cols], 1)
            colour_from = "/".join(cols)
            break
    if rgb is None:
        dc = [_pick(names, c) for c in ("f_dc_0", "f_dc_1", "f_dc_2")]
        if None not in dc:                          # a Gaussian-splat PLY: the DC term is the colour
            rgb = np.clip(0.5 + SH_C0 * np.stack([np.asarray(arr_get(c), np.float64) for c in dc], 1), 0, 1)
            colour_from = "f_dc (splat SH DC)"
    nrm = None
    for trip in (("nx", "ny", "nz"), ("normal_x", "normal_y", "normal_z")):
        cols = [_pick(names, c) for c in trip]
        if None not in cols:
            nrm = np.stack([np.asarray(arr_get(c), np.float64) for c in cols], 1)
            break
    return pts, rgb, colour_from, nrm


def _to_uint8(rgb):
    if rgb is None:
        return None
    rgb = np.asarray(rgb, np.float64)
    finite = rgb[np.isfinite(rgb)]
    top = float(finite.max()) if finite.size else 0.0
    if top <= 1.0:
        rgb = rgb * 255.0
    elif top > 255.0:                               # 16-bit colour
        rgb = rgb / 257.0
    return np.clip(np.round(np.nan_to_num(rgb)), 0, 255).astype(np.uint8)


def _load_ply(path):
    with open(path, "rb") as f:
        fmt, elements, comments, off = _read_ply_header(f)
        vi = next((i for i, e in enumerate(elements) if e["name"] == "vertex"), None)
        if vi is None:
            raise ValueError("PLY has no 'vertex' element")
        vel = elements[vi]
        if any("list" in p for p in vel["props"]):
            raise ValueError("PLY vertex element carries a list property; not a point cloud or mesh")
        names = [p["name"] for p in vel["props"]]
        n = vel["count"]
        if fmt == "ascii":
            text = f.read().decode("ascii", "replace")
            rows = text.split("\n")
            skip = sum(e["count"] for e in elements[:vi])
            # elements before the vertices are one line per row in ascii
            block = rows[skip:skip + n]
            if len(block) < n:
                raise ValueError(f"PLY is truncated: {len(block)} of {n} vertex lines")
            vals = np.array(" ".join(block).split(), dtype=np.float64)
            if vals.size != n * len(names):
                raise ValueError(f"PLY ascii vertex block has {vals.size} values, want {n} x {len(names)}")
            vals = vals.reshape(n, len(names))
            get = (lambda name: vals[:, names.index(name)])
        else:
            endian = "<" if fmt == "binary_little_endian" else ">"
            for e in elements[:vi]:
                _skip_binary_element(f, e, endian)
            dt = _ply_dtype(vel["props"], endian)
            body = f.read(n * dt.itemsize)
            if len(body) < n * dt.itemsize:
                raise ValueError(f"PLY is truncated: {len(body)} of {n * dt.itemsize} vertex bytes")
            arr = np.frombuffer(body, dtype=dt, count=n)
            get = (lambda name: arr[name])
    pts, rgb, colour_from, nrm = _columns(get, names)
    faces = sum(e["count"] for e in elements if e["name"] in ("face", "tristrips"))
    gaussian = {"opacity", "scale_0", "rot_0"} <= {x.lower() for x in names}
    meta = {"format": f"ply-{fmt}", "vertex_properties": names,
            "elements": {e["name"]: e["count"] for e in elements},
            "faces": int(faces), "comments": comments[:20], "colour_from": colour_from,
            "gaussian_splat": bool(gaussian)}
    return pts, rgb, nrm, meta, _declared_units(comments)


def _load_obj(path):
    vs, faces, vn = [], 0, 0
    with open(path, "r", errors="replace") as f:
        comments = []
        for ln in f:
            if ln.startswith("v "):
                vs.append(ln[2:])
            elif ln.startswith("f "):
                faces += 1
            elif ln.startswith("vn "):
                vn += 1
            elif ln.startswith("#") and len(comments) < 20:
                comments.append(ln[1:].strip())
    if not vs:
        raise ValueError("OBJ has no 'v' lines")
    widths = np.array([len(s.split()) for s in vs])
    w = int(np.bincount(widths).argmax())
    keep = [s for s, k in zip(vs, widths) if k == w]
    vals = np.array(" ".join(keep).split(), dtype=np.float64).reshape(-1, w)
    pts = vals[:, :3]
    rgb = vals[:, 3:6] if w >= 6 else None          # "v x y z r g b" (MeshLab / Polycam vertex colour)
    meta = {"format": "obj", "faces": int(faces), "obj_vertex_width": w, "obj_vn_lines": vn,
            "comments": comments, "colour_from": "v r g b" if rgb is not None else None,
            "skipped_vertex_lines": int(len(vs) - len(keep))}
    return pts, rgb, None, meta, _declared_units(comments)


def _is_number(tok):
    try:
        float(tok)
        return True
    except ValueError:
        return False


def _load_text_points(path):
    """XYZ / CSV / TXT / PTS: one point per line, delimiter and header guessed."""
    with open(path, "r", errors="replace") as f:
        raw = [ln.strip() for ln in f]
    # '#' lines are comments; a CloudCompare header "//X,Y,Z,R,G,B" is kept as the header row
    lines = [ln for ln in raw if ln and not ln.startswith("#")]
    if not lines:
        raise ValueError("no data lines")
    sample = lines[1] if len(lines) > 1 else lines[0]
    delim = "," if "," in sample else (";" if ";" in sample else ("\t" if "\t" in sample else None))

    def split(ln):
        return [t.strip() for t in (ln.split(delim) if delim else ln.split()) if t.strip() != ""]

    header, comments = None, [ln for ln in raw[:5] if ln.startswith("#")]
    first = split(lines[0].lstrip("/"))
    if not all(_is_number(t) for t in first):
        header = [re.sub(r"[^a-z_]", "", t.lower().split("(")[0].strip()) for t in first]
        comments.append(lines[0])
        lines = lines[1:]
    elif len(first) == 1 and len(lines) > 1:            # PTS: a point count, then the points
        lines = lines[1:]
    widths = np.array([len(split(ln)) for ln in lines[:2000]])
    w = int(np.bincount(widths).argmax())
    if w < 3:
        raise ValueError(f"lines have {w} numeric columns; want at least x y z")
    good = [ln for ln in lines if len(split(ln)) == w] if (widths != w).any() else lines
    body = "\n".join(",".join(split(ln)) for ln in good)
    vals = np.loadtxt(io.StringIO(body), delimiter=",", ndmin=2, dtype=np.float64)
    rgb = nrm = None
    how = None
    if header and len(header) == w:
        names = header
        get = (lambda nm: vals[:, names.index(nm)])
        try:
            pts, rgb, how, nrm = _columns(get, names)
        except ValueError:
            pts = vals[:, :3]
    else:
        pts = vals[:, :3]
        triples = [vals[:, i:i + 3] for i in range(3, w - 2, 3)] if w in (6, 9) else []
        if w in (7, 10):                                  # x y z intensity r g b [nx ny nz] (PTS)
            triples = [vals[:, 4:7]] + ([vals[:, 7:10]] if w == 10 else [])
        for tri in triples:
            norms = np.linalg.norm(tri, axis=1)
            if nrm is None and np.median(np.abs(norms - 1.0)) < 0.05:
                nrm = tri
            elif rgb is None and tri.min() >= 0 and tri.max() <= 255:
                rgb, how = tri, "columns"
    meta = {"format": os.path.splitext(path)[1].lower().lstrip(".") or "xyz", "faces": 0,
            "columns": w, "header": header, "delimiter": delim or "whitespace",
            "colour_from": how, "skipped_lines": int(len(lines) - len(good))}
    return pts, rgb, nrm, meta, _declared_units(comments)


def infer_units(extent):
    """(units, confident) from the largest robust bounding-box side, in file units.

    A phone scan of a room or a set is 2-20 m across: 2-20 in metres, 2,000-20,000 in mm.
    Anything up to 150 reads as metres (a 150 m scan is beyond a phone; 150 mm is not a room),
    anything above as millimetres. Centimetres are never guessed — say --units cm."""
    e = float(extent)
    if not np.isfinite(e) or e <= 0:
        return "m", False
    if e <= 150.0:
        return "m", 0.3 <= e <= 60.0
    return "mm", 1500.0 <= e <= 60000.0


def load_scan(path, units=None):
    """A scan file -> Scan with points in mm (float64), colours (uint8) and normals if the file
    has them, and meta: format, vertex / face counts, properties, bbox extent, units and where
    they came from (flag | header | extent). ValueError with the reason on anything unreadable."""
    ext = os.path.splitext(path)[1].lower()
    if not os.path.exists(path):
        raise ValueError(f"no such file: {path}")
    if ext == ".ply":
        pts, rgb, nrm, meta, declared = _load_ply(path)
    elif ext == ".obj":
        pts, rgb, nrm, meta, declared = _load_obj(path)
    elif ext in (".xyz", ".csv", ".txt", ".pts"):
        pts, rgb, nrm, meta, declared = _load_text_points(path)
    else:
        raise ValueError(f"unsupported scan format {ext or '(none)'}: export PLY (point cloud or mesh), "
                         "OBJ, or XYZ/CSV from the scanning app")
    n_file = len(pts)
    good = np.isfinite(pts).all(axis=1)
    pts = pts[good]
    rgb = _to_uint8(rgb[good]) if rgb is not None else None
    nrm = nrm[good] if nrm is not None else None
    if len(pts) < 10:
        raise ValueError(f"only {len(pts)} finite points in the scan")
    lo, hi = np.percentile(pts, [0.5, 99.5], axis=0)
    ext_file = float((hi - lo).max())
    if units:
        u, src, confident = units, "flag", True
    elif declared:
        u, src, confident = declared, "header", True
    else:
        src = "extent"
        u, confident = infer_units(ext_file)
    k = UNIT_MM[u]
    pts = pts * k
    meta.update({
        "path": os.path.abspath(path), "file_bytes": os.path.getsize(path),
        "vertices": int(n_file), "points": int(len(pts)), "dropped_nonfinite": int(n_file - len(pts)),
        "units": u, "units_source": src, "units_confident": bool(confident),
        "declared_units": declared, "to_mm": k,
        "extent_file_units": round(ext_file, 6), "extent_mm": [round(float(x), 2) for x in (hi - lo) * k],
        "bbox_mm": [[round(float(x), 2) for x in pts.min(0)], [round(float(x), 2) for x in pts.max(0)]],
        "has_colour": rgb is not None, "has_normals": nrm is not None,
        "is_mesh": bool(meta.get("faces")),
    })
    return Scan(points=pts, colors=rgb, normals=nrm, meta=meta)


def write_points_ply(path, xyz, rgb=None, comments=()):
    """Binary little-endian PLY of float x, y, z (+ uchar red, green, blue)."""
    xyz = np.asarray(xyz, np.float64)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if rgb is not None:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    arr = np.empty(len(xyz), dtype=fields)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if rgb is not None:
        arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    head = ["ply", "format binary_little_endian 1.0"] + [f"comment {c}" for c in comments]
    head += [f"element vertex {len(arr)}"] + [f"property {'float' if t == '<f4' else 'uchar'} {n}" for n, t in fields]
    head += ["end_header"]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".partial"
    with open(tmp, "wb") as f:
        f.write(("\n".join(head) + "\n").encode("ascii"))
        f.write(arr.tobytes())
    os.replace(tmp, path)


# ======================================================================== neighbours

class NN:
    """Exact-ish nearest neighbours on a fixed point set (cv2.flann kd-tree). Coordinates are
    centred before going to float32, so a scan in a far-off frame loses no precision."""

    def __init__(self, pts):
        import cv2
        pts = np.asarray(pts, np.float64)
        self.n = len(pts)
        self.origin = pts.mean(axis=0) if len(pts) else np.zeros(3)
        self.data = np.ascontiguousarray(pts - self.origin, np.float32)
        # the kd-tree's split dimensions are drawn from OpenCV's process RNG: seed it so the
        # same clouds give the same neighbours on every machine (the Mac and the container
        # disagreed by 0.4 % of scale on the noisy synthetic room before this)
        cv2.setRNGSeed(0)
        self.index = cv2.flann_Index(self.data, dict(algorithm=1, trees=1))

    def knn(self, q, k=1, chunk=200000):
        q = np.ascontiguousarray(np.asarray(q, np.float64).reshape(-1, 3) - self.origin, np.float32)
        k = int(min(k, self.n))
        idx = np.empty((len(q), k), np.int64)
        d2 = np.empty((len(q), k), np.float64)
        for i in range(0, len(q), chunk):
            ii, dd = self.index.knnSearch(q[i:i + chunk], k, params=dict(checks=-1))
            idx[i:i + chunk] = ii
            d2[i:i + chunk] = dd
        return np.sqrt(np.maximum(d2, 0.0)), idx

    def query(self, q):
        d, i = self.knn(q, 1)
        return d[:, 0], i[:, 0]


def spacing(points, index=None, sample=2000, seed=0):
    """Median distance from a point to its nearest other point."""
    P = np.asarray(points, np.float64)
    if len(P) < 2:
        return 0.0
    idx = index or NN(P)
    rng = np.random.default_rng(seed)
    q = P[rng.choice(len(P), min(sample, len(P)), replace=False)]
    d, _ = idx.knn(q, 2)
    return float(np.median(d[:, 1]))


def estimate_normals(points, k=16, ref=None, index=None):
    """PCA normals: the smallest-variance direction of each point's k nearest neighbours in `ref`
    (default the points themselves). -> (normals (N, 3) unit, sign arbitrary; curvature (N,) =
    smallest eigenvalue / sum, ~0 on a plane, 1/3 in a blob)."""
    P = np.asarray(points, np.float64)
    ref = P if ref is None else np.asarray(ref, np.float64)
    idx = index or NN(ref)
    _, nb = idx.knn(P, k)
    Q = ref[nb]                                             # (N, k, 3)
    Q = Q - Q.mean(axis=1, keepdims=True)
    C = np.einsum("nki,nkj->nij", Q, Q) / max(k, 1)
    w, V = np.linalg.eigh(C)
    n = V[:, :, 0]
    curv = w[:, 0] / np.maximum(w.sum(axis=1), 1e-30)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30), curv


def denoise(points, k=8, ratio=3.0, index=None):
    """Statistical outlier removal: keep the points whose mean distance to their k nearest
    neighbours is within `ratio` x the median of that distance. -> boolean mask.

    Needed before a voxel subsample of an SfM cloud: every isolated outlier keeps a voxel of its
    own while a surface's hundreds share theirs, so 20 % scattered outliers become ~half of a
    5,000-point subsample."""
    P = np.asarray(points, np.float64)
    if len(P) <= k + 1:
        return np.ones(len(P), bool)
    idx = index or NN(P)
    d, _ = idx.knn(P, k + 1)
    m = d[:, 1:].mean(axis=1)
    return m <= ratio * float(np.median(m))


def constraints(P, N):
    """How well point-to-plane correspondences pin a similarity down, at a solution.

    P (n, 3): aligned points (the solve's inliers, in the scan's frame); N (n, 3): the scan's
    normals at their matches. The residual n.(y - x) moves with a small rotation w, translation
    d and log-scale g as n.d + (p x n).w + g (n.p), p about the centroid; the 7 x 7 information
    matrix of those columns (rotation and scale in units of the cloud's RMS radius, so every
    column is dimensionless) says what the geometry can observe.

    -> {"scale_sensitivity": sqrt of the Schur complement of the scale in it — the RMS normal
    displacement per unit log-scale, over the RMS radius, that no rotation or translation can
    absorb (0 for points on planes through one point: floor and two walls scale about their
    corner), "pose_min_eig": the smallest eigenvalue of the 6 x 6 pose block (0 when a direction
    of motion slides along the geometry), "radius_mm", "n"}."""
    P = np.asarray(P, np.float64)
    N = np.asarray(N, np.float64)
    if len(P) < 10:
        return {"scale_sensitivity": 0.0, "pose_min_eig": 0.0, "radius_mm": 0.0, "n": int(len(P))}
    p = P - P.mean(axis=0)
    Lr = float(np.sqrt(np.mean((p ** 2).sum(axis=1)))) or 1.0
    J = np.hstack([N, np.cross(p, N) / Lr, ((N * p).sum(axis=1) / Lr)[:, None]])
    H = J.T @ J / len(J)
    A, b, c = H[:6, :6], H[:6, 6], H[6, 6]
    try:
        schur = float(c - b @ np.linalg.solve(A, b))
    except np.linalg.LinAlgError:
        schur = 0.0
    return {"scale_sensitivity": float(np.sqrt(max(schur, 0.0))),
            "pose_min_eig": float(np.linalg.eigvalsh(A)[0]), "radius_mm": Lr, "n": int(len(P))}


def subsample(points, n, method="voxel", seed=0):
    """Indices of about `n` points, deterministic for a seed.

    voxel: the voxel size is searched so the occupied voxels number ~n, one point per voxel (a
    seeded random pick inside it), then trimmed to n at random — an even spread over the surface
    that does not follow the scanner's density. random: n uniform indices."""
    P = np.asarray(points, np.float64)
    N = len(P)
    if N <= n:
        return np.arange(N)
    rng = np.random.default_rng(seed)
    if method == "random":
        return np.sort(rng.choice(N, n, replace=False))
    if method != "voxel":
        raise ValueError(f"unknown subsample method {method!r}")
    perm = rng.permutation(N)
    Pp = P[perm]
    lo = Pp.min(axis=0)
    ext = float((Pp.max(axis=0) - lo).max()) or 1.0

    def cells(v):
        g = np.floor((Pp - lo) / v).astype(np.int64)
        key = (g[:, 0] * 2097152 + g[:, 1]) * 2097152 + g[:, 2]
        _, first = np.unique(key, return_index=True)
        return first

    a, b = ext / 1e5, ext                        # a: about one point per cell, b: one cell
    best = None
    for _ in range(24):
        v = np.sqrt(a * b)
        first = cells(v)
        if len(first) >= n:
            a, best = v, first
            if len(first) <= 1.1 * n:
                break
        else:
            b = v
    if best is None:
        best = cells(a)
    pick = perm[best]
    if len(pick) > n:
        pick = rng.choice(pick, n, replace=False)
    return np.sort(pick)


# ======================================================================== similarity fits

def umeyama(src, dst, with_scale=True, weights=None):
    """Least-squares dst ~ s R src + t. -> (s, R, t, residuals (N,)). s = 1 without scale."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    w = np.ones(len(src)) if weights is None else np.asarray(weights, np.float64)
    w = w / w.sum()
    ms, md = w @ src, w @ dst
    a, b = src - ms, dst - md
    U, S, Vt = np.linalg.svd((b * w[:, None]).T @ a)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    var = float(w @ (a ** 2).sum(axis=1))
    s = float(np.trace(np.diag(S) @ D) / var) if with_scale and var > 0 else 1.0
    t = md - s * R @ ms
    res = np.linalg.norm(src @ R.T * s + t - dst, axis=1)
    return s, R, t, res


def _umeyama_batch(A, B, with_scale=True):
    """Umeyama on H small point sets at once: A, B (H, m, 3). -> s (H,), R (H, 3, 3), t (H, 3)."""
    ma, mb = A.mean(axis=1), B.mean(axis=1)
    a, b = A - ma[:, None], B - mb[:, None]
    M = np.einsum("hki,hkj->hij", b, a) / A.shape[1]
    U, S, Vt = np.linalg.svd(M)
    d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    d[d == 0] = 1.0
    D = np.ones((len(A), 3))
    D[:, 2] = d
    R = np.einsum("hij,hj,hjk->hik", U, D, Vt)
    var = (a ** 2).sum(axis=(1, 2)) / A.shape[1]
    s = (S * D).sum(axis=1) / np.maximum(var, 1e-30) if with_scale else np.ones(len(A))
    t = mb - s[:, None] * np.einsum("hij,hj->hi", R, ma)
    return s, R, t


def apply(T, X):
    s, R, t = T
    return s * np.asarray(X, np.float64) @ np.asarray(R).T + np.asarray(t)


def invert(T):
    """(s, R, t) of y = s R x + t -> the same for x in terms of y."""
    s, R, t = T
    Ri = np.asarray(R).T
    return 1.0 / s, Ri, -(Ri @ np.asarray(t)) / s


def rotation_angle_deg(Ra, Rb):
    c = (np.trace(np.asarray(Ra) @ np.asarray(Rb).T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def angle_deg(a, b):
    a = np.asarray(a, np.float64) / np.linalg.norm(a)
    b = np.asarray(b, np.float64) / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))


# ======================================================================== ICP

def _kept(d, keep_n, keep_within):
    keep = np.argpartition(d, keep_n - 1)[:keep_n] if keep_n < len(d) else np.arange(len(d))
    if keep_within is not None:
        keep = np.union1d(keep, np.flatnonzero(d <= keep_within))
    return keep


def icp(src, dst, init=None, max_iter=100, trim=0.7, with_scale=True, tol=1e-5, index=None,
        inlier_dist=None, scale_bounds=1.25, callback=None, keep_within=None):
    """Trimmed point-to-point ICP: dst ~ s R src + t.

    Each iteration: nearest dst point for every transformed src point, keep the best `trim`
    of those correspondences (and, with keep_within, every other one closer than that), then
    Umeyama (Sim(3) with_scale, else SE(3)) on the kept ones. The scale is kept within
    [s0 / scale_bounds, s0 * scale_bounds] of the initial one, so an overlap that is only
    partial cannot shrink the source into a corner. Stops when the trimmed RMS changes by less
    than `tol` (relative) or after max_iter.

    Why keep_within: a room is nearly scale-degenerate — scaling about the corner where floor
    and walls meet leaves every wall and floor point on its plane — and what pins the scale is
    the furniture. A few percent off, the furniture's points are the worst 30 % and a pure
    trim throws away exactly the correspondences that would pull the scale back.

    -> {"s", "R", "t", "rms" (trimmed, at the returned transform), "rms_history" (per iteration,
    at the transform that iteration started from), "iterations", "converged", "inlier_fraction"
    (share of src within inlier_dist, when given), "median"}."""
    src = np.asarray(src, np.float64)
    idx = index or NN(dst)
    dst = np.asarray(dst, np.float64) if dst is not None else None
    s, R, t = init if init is not None else (1.0, np.eye(3), np.zeros(3))
    s0 = float(s)
    R, t = np.asarray(R, np.float64), np.asarray(t, np.float64)
    keep_n = max(3, int(np.ceil(trim * len(src))))
    hist, converged, it = [], False, 0
    ref = _Ref(idx, dst)
    lo_c, hi_c = src.min(axis=0), src.max(axis=0)
    corners = np.array([[x, y, z] for x in (lo_c[0], hi_c[0]) for y in (lo_c[1], hi_c[1]) for z in (lo_c[2], hi_c[2])])
    size = float(np.linalg.norm(hi_c - lo_c)) or 1.0
    for it in range(1, max_iter + 1):
        Y = s * src @ R.T + t
        d, j = idx.query(Y)
        rms = float(np.sqrt(np.mean(d[_kept(d, keep_n, None)] ** 2)))
        hist.append(rms)
        if callback is not None:
            callback(it, max_iter, rms)
        keep = _kept(d, keep_n, keep_within)
        s2, R2, t2, _ = umeyama(src[keep], ref.points(j[keep]), with_scale)
        if with_scale:
            lo, hi = s0 / scale_bounds, s0 * scale_bounds
            if not lo <= s2 <= hi:              # clamp, and re-fit t for the clamped scale
                s2 = float(np.clip(s2, lo, hi))
                t2 = ref.points(j[keep]).mean(0) - s2 * R2 @ src[keep].mean(0)
        # converged when the step moves the source's bounding box by < tol of its size (the RMS
        # alone can stall while the scale is still being pulled: the planes do not care)
        move = np.abs(s2 * corners @ R2.T + t2 - (s * corners @ R.T + t)).max()
        s, R, t = s2, R2, t2
        if move <= tol * s * size:
            converged = True
            break
    Y = s * src @ R.T + t
    d, _ = idx.query(Y)
    keep = _kept(d, keep_n, None)
    out = {"s": float(s), "R": R, "t": t, "rms": float(np.sqrt(np.mean(d[keep] ** 2))),
           "rms_history": hist, "iterations": it, "converged": converged,
           "median": float(np.median(d)), "distances": d}
    if inlier_dist is not None:
        out["inlier_fraction"] = float(np.mean(d <= inlier_dist))
    return out


class _Ref:
    """dst coordinates by index: from the array when there is one, else the NN's own copy."""

    def __init__(self, idx, dst):
        self.idx, self.dst = idx, dst

    def points(self, j):
        if self.dst is not None:
            return self.dst[j]
        return self.idx.data[j].astype(np.float64) + self.idx.origin


# ======================================================================== global registration

def _pair_features(P1, N1, P2, N2):
    """Scale-invariant angles of point pairs with unsigned normals, in degrees:
    a1 = angle(normal line 1, segment), a2 = angle(normal line 2, segment), a3 = angle between
    the normal lines; all in [0, 90]. Also the length and unit direction."""
    V = P2 - P1
    L = np.linalg.norm(V, axis=-1)
    u = V / np.maximum(L, 1e-30)[..., None]
    a1 = np.degrees(np.arccos(np.clip(np.abs((N1 * u).sum(-1)), 0, 1)))
    a2 = np.degrees(np.arccos(np.clip(np.abs((N2 * u).sum(-1)), 0, 1)))
    a3 = np.degrees(np.arccos(np.clip(np.abs((N1 * N2).sum(-1)), 0, 1)))
    return L, u, a1, a2, a3


class _PairIndex:
    """Every pair of a few hundred to ~1,500 scan points, hashed by (a1, a2, a3) with a1 <= a2."""

    def __init__(self, P, N, bin_deg, min_len):
        n = len(P)
        i, j = np.triu_indices(n, k=1)
        L, _u, a1, a2, a3 = _pair_features(P[i], N[i], P[j], N[j])
        ok = L > min_len
        i, j, L, a1, a2, a3 = i[ok], j[ok], L[ok], a1[ok], a2[ok], a3[ok]
        sw = a1 > a2                                          # canonical order: a1 <= a2
        i, j = np.where(sw, j, i), np.where(sw, i, j)
        a1, a2 = np.where(sw, a2, a1), np.where(sw, a1, a2)
        self.bin = float(bin_deg)
        self.nb = int(np.ceil(90.0 / bin_deg)) + 1
        key = self._key(a1, a2, a3)
        order = np.argsort(key, kind="stable")
        self.key = key[order]
        self.i, self.j, self.L = i[order], j[order], L[order]
        self.a = np.stack([a1[order], a2[order], a3[order]], 1)
        self.n_pairs = len(order)

    def _key(self, a1, a2, a3):
        q = lambda a: np.clip((np.asarray(a) // self.bin).astype(np.int64), 0, self.nb - 1)
        return (q(a1) * self.nb + q(a2)) * self.nb + q(a3)

    def lookup(self, f, tol):
        ranges = [np.arange(max(0, int((x - tol) // self.bin)), min(self.nb - 1, int((x + tol) // self.bin)) + 1)
                  for x in f]
        keys = ((ranges[0][:, None, None] * self.nb + ranges[1][None, :, None]) * self.nb
                + ranges[2][None, None, :]).ravel()
        lo = np.searchsorted(self.key, keys, "left")
        hi = np.searchsorted(self.key, keys, "right")
        if not (hi > lo).any():
            return np.zeros(0, np.int64)
        c = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi) if b > a])
        close = (np.abs(self.a[c] - np.asarray(f)[None]) <= tol).all(axis=1)
        return c[close]


def _frames(e1, n1):
    """Right-handed frames (..., 3, 3) with columns e1, n1 made orthogonal to e1, and their cross."""
    e2 = n1 - (n1 * e1).sum(-1, keepdims=True) * e1
    e2 = e2 / np.maximum(np.linalg.norm(e2, axis=-1, keepdims=True), 1e-30)
    e3 = np.cross(e1, e2)
    return np.stack([e1, e2, e3], axis=-1)


class DistanceGrid:
    """Approximate distance to a point set, looked up per query in O(1): a voxel grid of cell
    `cell` holding, for every cell within `levels` cells of a point, how many cells away the
    nearest occupied one is (Chebyshev). Used to prefilter thousands of hypotheses per sample,
    where exact kd-tree queries for points far off the surface would dominate the run time."""

    def __init__(self, pts, cell, levels=4, max_cells=40_000_000):
        pts = np.asarray(pts, np.float64)
        pad = (levels + 1) * cell
        lo, hi = pts.min(axis=0) - pad, pts.max(axis=0) + pad
        shape = np.ceil((hi - lo) / cell).astype(int) + 1
        while np.prod(shape.astype(np.float64)) > max_cells:
            cell *= 1.25
            pad = (levels + 1) * cell
            lo, hi = pts.min(axis=0) - pad, pts.max(axis=0) + pad
            shape = np.ceil((hi - lo) / cell).astype(int) + 1
        self.lo, self.cell, self.shape, self.levels = lo, float(cell), shape, levels
        far = levels + 1
        g = np.full(shape, far, np.uint8)
        ijk = np.floor((pts - lo) / cell).astype(int)
        g[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = 0
        occ = g == 0
        for lev in range(1, levels + 1):
            grown = occ.copy()
            for ax in range(3):                         # separable 3x3x3 dilation
                a = grown.copy()
                sl_lo = [slice(None)] * 3
                sl_hi = [slice(None)] * 3
                sl_lo[ax], sl_hi[ax] = slice(0, -1), slice(1, None)
                a[tuple(sl_lo)] |= grown[tuple(sl_hi)]
                a[tuple(sl_hi)] |= grown[tuple(sl_lo)]
                grown = a
            g[grown & (g == far)] = lev
            occ = grown
        self.g = g

    def dist(self, q):
        """Nominal distance from each query to the set: (L + 0.5) cells for a cell whose nearest
        occupied cell is L cells away (Chebyshev), inf beyond `levels`. An estimate for ranking,
        not a bound: the exact distances come from the kd-tree afterwards."""
        ijk = np.floor((np.asarray(q, np.float64) - self.lo) / self.cell).astype(np.int64)
        inside = ((ijk >= 0) & (ijk < self.shape)).all(axis=-1)
        out = np.full(ijk.shape[:-1], np.inf)
        v = self.g[tuple(ijk[inside].T)].astype(np.float64)
        v[v > self.levels] = np.inf
        out[inside] = (v + 0.5) * self.cell
        return out


def _rarity(normals, bin_deg=15.0):
    """Sampling weights (sum 1) inversely proportional to how common each point's (unsigned)
    normal direction is: every direction bin gets the same total weight."""
    n = np.asarray(normals, np.float64)
    n = n * np.where(n[np.arange(len(n)), np.abs(n).argmax(axis=1)] < 0, -1.0, 1.0)[:, None]
    az = np.degrees(np.arctan2(n[:, 1], n[:, 0])) % 360.0
    el = np.degrees(np.arcsin(np.clip(n[:, 2], -1, 1))) + 90.0
    key = (el // bin_deg).astype(np.int64) * 1000 + (az // bin_deg).astype(np.int64)
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]
    return w / w.sum()


def _msac(d, tau):
    """Truncated quadratic cost in [0, 1]: mean(min(d, tau)^2) / tau^2. Lower is better. Unlike a
    count it still ranks two poses that put most points on the planes: the one that also puts
    the furniture on the furniture wins."""
    tau = np.asarray(tau, np.float64)
    return float(np.mean(np.minimum(d, tau) ** 2) / np.maximum(tau, 1e-30) ** 2)


def global_register(src, dst, with_scale=True, *, seed=0, max_samples=60, time_budget_s=20.0,
                    min_inlier=0.5, n_index=1200, angle_tol_deg=7.0, n_score=400, max_candidates=20000,
                    scale_tol=0.06, dst_ref=None, src_ref=None, min_samples=15, confirm=2,
                    n_refine=12, n_final=4, ambiguity=1.5, min_scale_sensitivity=0.1, min_pose_eig=0.01,
                    progress=None, trace=None):
    """Find dst ~ s R src + t with no initial guess (see the module docstring).

    src, dst: (N, 3) subsamples (a few thousand each) — dst in mm, src in any unit. dst_ref /
    src_ref: denser clouds for the normals, and dst_ref also for the final refinement (default
    the subsamples). With with_scale=False the scale is 1 and pair lengths must agree within
    scale_tol. `progress(done, total)` is called once per sample.

    Each sample: a src triple -> candidate scan pairs from the angle hash -> a pose per candidate
    and normal sign -> the third point must land on the scan with the triple's distance ratios ->
    Umeyama on the triple -> a grid prefilter counts the score subsample's inliers -> the best
    `n_refine` get a short trimmed ICP and are ranked by truncated-quadratic cost. Stops once
    `min_samples` are done and the lowest-cost pose has been found from `confirm` different
    triples, else at max_samples / time_budget_s. (No stop on the inlier fraction alone: in a
    room a pose 18 % too small about the corner, or turned 120 degrees about it, still puts 90 %
    of the points on the floor and walls.) The best three are then refined against dst_ref and
    the lowest-cost one returned.

    -> {"aligned", "s", "R", "t", "inlier_fraction" (share of src within `tau_used` of the scan
    after the final ICP), "rms" (trimmed, mm), "cost", "tau", "tau_used", "samples",
    "hypotheses", "confirmations", "runner_up_cost", "seconds", "dst_spacing", "reason"}."""
    t0 = time.monotonic()
    rng = np.random.default_rng(seed)
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    out = {"aligned": False, "s": 1.0, "R": np.eye(3), "t": np.zeros(3), "inlier_fraction": 0.0,
           "rms": float("inf"), "samples": 0, "hypotheses": 0, "seconds": 0.0, "reason": None}
    if len(src) < 20 or len(dst) < 20:
        out["reason"] = f"too few points (src {len(src)}, dst {len(dst)})"
        return out
    keep = denoise(src)
    n_removed = int((~keep).sum())
    if keep.sum() >= 20:
        src = src[keep]
    dnn = NN(dst)
    h_d = spacing(dst, dnn, seed=seed)
    h_s = spacing(src, seed=seed)
    tau = 1.5 * h_d

    def tau_at(s):
        # the inlier distance is capped by the src's own spacing seen at scale s: otherwise a
        # hypothesis that shrinks the solve into a heap on one wall has every point "on" the scan
        s = np.asarray(s, np.float64)
        return np.minimum(tau, 1.5 * s * h_s) if with_scale else np.full(s.shape, tau)

    ref_nn = NN(dst_ref) if dst_ref is not None else None
    nd, cd = estimate_normals(dst, 16, ref=dst_ref, index=ref_nn)
    ns, cs = estimate_normals(src, 16, ref=src_ref)
    flat_d = np.flatnonzero(cd < 0.05)
    if len(flat_d) < 50:
        flat_d = np.argsort(cd)[:max(50, len(cd) // 2)]
    flat_s = np.flatnonzero(cs < 0.05)
    if len(flat_s) < 30:
        flat_s = np.argsort(cs)[:max(30, len(cs) // 2)]
    # rare normals first: a room is mostly floor and walls, three planes at right angles that
    # fit a corner turned 120 degrees as well as the right one; the furniture decides
    w_d = _rarity(nd[flat_d])
    w_s = _rarity(ns[flat_s])
    Q = rng.choice(flat_d, min(n_index, len(flat_d)), replace=False, p=w_d)
    pidx = _PairIndex(dst[Q], nd[Q], angle_tol_deg, 3.0 * h_d)
    grid = DistanceGrid(dst, tau / 3.0, levels=3)
    lo_s, hi_s = np.percentile(src, [2, 98], axis=0)
    diam = float(np.linalg.norm(hi_s - lo_s))
    Xs = src[rng.choice(len(src), min(n_score, len(src)), replace=False)]
    refine_ids = rng.choice(len(src), min(500, len(src)), replace=False)
    cos_n = np.cos(np.radians(2.0 * angle_tol_deg))
    Ps = src[flat_s]
    best = []                                           # refined hypotheses, lowest cost first
    n_hyp = 0
    samples = 0
    for samples in range(1, max_samples + 1):
        if progress is not None:
            progress(samples, max_samples)
        # ---- a well-separated, non-degenerate src triple on locally planar surface
        tri = None
        for _try in range(200):
            k1 = int(rng.choice(len(flat_s), p=w_s))
            d1 = np.linalg.norm(Ps - Ps[k1], axis=1)
            far = np.flatnonzero((d1 >= 0.25 * diam) & (d1 <= 0.9 * diam))
            if not len(far):
                continue
            k2 = int(rng.choice(far, p=w_s[far] / w_s[far].sum()))
            i1, i2 = int(flat_s[k1]), int(flat_s[k2])
            L, u, f1, f2, f3 = _pair_features(src[i1], ns[i1], src[i2], ns[i2])
            if min(f1, f2) < 12.0 or (f1 > 80.0 and f2 > 80.0 and f3 < 10.0):
                continue                               # degenerate frame / two points of one plane
            if f1 > f2:
                i1, i2 = i2, i1
                f1, f2 = f2, f1
            x1, x2 = src[i1], src[i2]
            a = np.linalg.norm(Ps - x1, axis=1)
            b = np.linalg.norm(Ps - x2, axis=1)
            cr = np.linalg.norm(np.cross(Ps - x1, x2 - x1), axis=1) / (L * np.maximum(a, 1e-30))
            ok3 = np.flatnonzero((a >= 0.25 * L) & (b >= 0.25 * L) & (cr >= 0.3) & (a <= 1.5 * L) & (b <= 1.5 * L))
            if not len(ok3):
                continue
            k3 = int(rng.choice(ok3, p=w_s[ok3] / w_s[ok3].sum()))
            tri = (i1, i2, int(flat_s[k3]), L, (f1, f2, f3))
            break
        if tri is None:
            continue
        i1, i2, i3, L, f = tri
        x1, x2, x3 = src[i1], src[i2], src[i3]
        # ---- candidate scan pairs with the same three angles (and length, when scale is known)
        cand = pidx.lookup(f, angle_tol_deg)
        if abs(f[0] - f[1]) <= angle_tol_deg:            # symmetric within tolerance: both orders
            cand_sw = pidx.lookup((f[1], f[0], f[2]), angle_tol_deg)
        else:
            cand_sw = np.zeros(0, np.int64)
        I = np.concatenate([pidx.i[cand], pidx.j[cand_sw]])
        J = np.concatenate([pidx.j[cand], pidx.i[cand_sw]])
        DL = np.concatenate([pidx.L[cand], pidx.L[cand_sw]])
        if not with_scale:
            ok = np.abs(DL - L) <= scale_tol * L + 2.0 * h_d
            I, J, DL = I[ok], J[ok], DL[ok]
        n_cand = len(I)
        if len(I) > max_candidates:
            k = rng.choice(len(I), max_candidates, replace=False)
            I, J, DL = I[k], J[k], DL[k]
        if not len(I):
            continue
        # ---- a pose per candidate and normal sign, from the edge and the first normal
        Y1, Y2 = dst[Q[I]], dst[Q[J]]
        M1, M2 = nd[Q[I]], nd[Q[J]]
        E = _frames((x2 - x1) / L, ns[i1])                        # (3, 3)
        F = _frames((Y2 - Y1) / DL[:, None], M1)                  # (C, 3, 3)
        Fm = F * np.array([1.0, -1.0, -1.0])                      # the other normal sign
        R = np.concatenate([F @ E.T, Fm @ E.T])
        Y1, Y2, M2, DL = (np.concatenate([a, a]) for a in (Y1, Y2, M2, DL))
        ok = np.abs(np.einsum("hij,j,hi->h", R, ns[i2], M2)) >= cos_n
        R, Y1, Y2, DL = R[ok], Y1[ok], Y2[ok], DL[ok]
        if not len(R):
            continue
        s = DL / L if with_scale else np.ones(len(R))
        mid_x = (x1 + x2) / 2
        t = (Y1 + Y2) / 2 - s[:, None] * np.einsum("hij,j->hi", R, mid_x)
        # ---- the third point: lands on the scan, with the triple's distance ratios
        y3p = s[:, None] * np.einsum("hij,j->hi", R, x3) + t
        e3, j3 = dnn.query(y3p)
        Y3 = dst[j3]
        r13 = np.linalg.norm(x3 - x1) / L
        r23 = np.linalg.norm(x3 - x2) / L
        q13 = np.linalg.norm(Y3 - Y1, axis=1) / DL
        q23 = np.linalg.norm(Y3 - Y2, axis=1) / DL
        tol_r = 0.08 + 2.0 * h_d / np.maximum(DL, 1e-30)
        ok = (np.abs(q13 - r13) <= tol_r) & (np.abs(q23 - r23) <= tol_r) & (e3 <= np.maximum(3 * h_d, 0.1 * DL))
        # ...and its normal agrees with the scan's there
        ok &= np.abs(np.einsum("hij,j,hi->h", R, ns[i3], nd[j3])) >= cos_n
        if not ok.any():
            continue
        A = np.broadcast_to(np.stack([x1, x2, x3])[None], (int(ok.sum()), 3, 3))
        B = np.stack([Y1[ok], Y2[ok], Y3[ok]], axis=1)
        s_h, R_h, t_h = _umeyama_batch(A, B, with_scale)
        if not with_scale:
            s_h = np.ones(len(R_h))
        n_hyp += len(R_h)
        # ---- prefilter on the grid, then a short trimmed ICP on the best few, ranked by cost
        cnt = np.empty(len(R_h), np.int64)
        for c0 in range(0, len(R_h), 2000):
            sl = slice(c0, c0 + 2000)
            Yh = s_h[sl, None, None] * np.einsum("hij,nj->hni", R_h[sl], Xs) + t_h[sl, None, :]
            cnt[sl] = (grid.dist(Yh) <= tau_at(s_h[sl])[:, None]).sum(axis=1)
        if trace is not None:
            trace.append({"sample": samples, "candidates": int(n_cand), "hypotheses": int(len(R_h)),
                          "s": s_h, "R": R_h, "t": t_h, "count": cnt, "t_s": time.monotonic() - t0})
        for h in np.argsort(-cnt)[:n_refine]:
            if cnt[h] < 0.1 * len(Xs):
                break
            r = icp(src[refine_ids], None, (float(s_h[h]), R_h[h], t_h[h]), max_iter=8, trim=0.7,
                    with_scale=with_scale, index=dnn, scale_bounds=1.25, keep_within=tau)
            ta = float(tau_at(r["s"]))
            hyp = {"cost": _msac(r["distances"], ta), "frac": float(np.mean(r["distances"] <= ta)),
                   "s": r["s"], "R": r["R"], "t": r["t"], "hits": 1, "sample": samples}
            for b in best:                       # the same pose again: from this sample, or another
                if (abs(b["s"] / hyp["s"] - 1) < 0.03 and rotation_angle_deg(b["R"], hyp["R"]) < 3.0
                        and np.linalg.norm(b["t"] - hyp["t"]) < 3 * tau + 0.03 * np.linalg.norm(hyp["t"])):
                    if b["sample"] != samples:
                        b["hits"] += 1
                        b["sample"] = samples
                    if hyp["cost"] < b["cost"]:
                        b.update({k: hyp[k] for k in ("cost", "frac", "s", "R", "t")})
                    break
            else:
                best.append(hyp)
        best.sort(key=lambda b: b["cost"])
        best = best[:8]
        if best and samples >= min_samples and best[0]["hits"] >= confirm:
            break
        if samples >= 3 * min_samples and any(b["hits"] >= confirm for b in best[:3]):
            break                       # the final refinement against the dense scan decides
        if time.monotonic() - t0 > time_budget_s:
            break
    out.update({"samples": samples, "hypotheses": int(n_hyp), "tau": tau, "dst_spacing": h_d,
                "src_spacing": h_s, "index_pairs": pidx.n_pairs, "src_outliers_removed": n_removed})
    if not best:
        out["reason"] = "no hypothesis survived the triple test"
        out["seconds"] = round(time.monotonic() - t0, 3)
        return out
    # ---- the best few, refined against the densest scan we were given, lowest cost wins
    final = []
    fin = src[rng.choice(len(src), min(2000, len(src)), replace=False)]
    for b in best[:n_final]:
        r = icp(fin, None, (b["s"], b["R"], b["t"]), max_iter=60, trim=0.7, with_scale=with_scale,
                index=ref_nn if ref_nn is not None else dnn, scale_bounds=1.1, keep_within=tau)
        ta = float(tau_at(r["s"]))
        r.update({"cost": _msac(r["distances"], ta), "inlier_fraction": float(np.mean(r["distances"] <= ta)),
                  "hits": b["hits"], "tau_used": ta})
        final.append(r)
    final.sort(key=lambda r: r["cost"])
    r = final[0]
    # a different pose that fits almost as well means the geometry cannot tell them apart (a bare
    # corner is symmetric under 120-degree turns and under scaling about its apex): say so
    alts = []
    for q in final[1:]:
        rot = rotation_angle_deg(q["R"], r["R"])
        sr = q["s"] / r["s"]
        dt = float(np.linalg.norm(q["t"] - r["t"]))
        # within 3 degrees / 3 % it is the same basin, not yet converged (ICP creeps along a room's scale)
        if rot > 3.0 or abs(sr - 1) > 0.03 or dt > 5 * tau:
            alts.append({"cost": q["cost"], "cost_ratio": q["cost"] / max(r["cost"], 1e-12),
                         "rotation_deg": rot, "scale_ratio": sr, "translation_mm": dt,
                         "inlier_fraction": q["inlier_fraction"]})
    ambiguous = [x for x in alts if x["cost_ratio"] < ambiguity]
    # what the winner's geometry can observe (see `constraints`)
    rn = ref_nn if ref_nn is not None else dnn
    ref_pts = dst_ref if dst_ref is not None else dst
    Yw = apply((r["s"], r["R"], r["t"]), fin)
    dw, jw = rn.query(Yw)
    inl = dw <= r["tau_used"]
    if inl.sum() >= 10:
        nw, _ = estimate_normals(ref_pts[jw[inl]], 16, ref=ref_pts, index=rn)
        con = constraints(Yw[inl], nw)
    else:
        con = constraints(Yw[:0], Yw[:0])
    weak_scale = with_scale and con["scale_sensitivity"] < min_scale_sensitivity
    weak_pose = con["pose_min_eig"] < min_pose_eig
    out.update({"s": r["s"], "R": r["R"], "t": r["t"], "inlier_fraction": r["inlier_fraction"],
                "rms": r["rms"], "cost": r["cost"], "tau_used": r["tau_used"],
                "aligned": bool(r["inlier_fraction"] >= min_inlier and not ambiguous and not weak_scale
                                and not weak_pose),
                "confirmations": int(r["hits"]), "alternatives": alts, "ambiguous": bool(ambiguous),
                "runner_up_cost": alts[0]["cost"] if alts else None, "constraints": con,
                "seconds": round(time.monotonic() - t0, 3)})
    if r["inlier_fraction"] < min_inlier:
        out["reason"] = (f"best inlier fraction {r['inlier_fraction']:.2f} < {min_inlier} "
                         f"(within {r['tau_used']:.0f} mm)")
    elif ambiguous:
        x = ambiguous[0]
        out["reason"] = (f"ambiguous: another pose ({x['rotation_deg']:.0f} deg away, scale x{x['scale_ratio']:.3f}) "
                         f"fits almost as well (cost {x['cost_ratio']:.2f}x the best; want >= {ambiguity:g}x)")
    elif weak_pose:
        out["reason"] = (f"the overlap does not pin the pose down (smallest pose eigenvalue "
                         f"{con['pose_min_eig']:.4f} < {min_pose_eig:g}): the solve sees too few surfaces of the scan")
    elif weak_scale:
        out["reason"] = (f"the overlap does not pin the scale down (sensitivity {con['scale_sensitivity']:.3f} < "
                         f"{min_scale_sensitivity:g}): planes through one corner fit at any scale")
    return out


# ======================================================================== up, ground, coverage

SCAN_UP = {"y": np.array([0.0, 1.0, 0.0]), "z": np.array([0.0, 0.0, 1.0])}
DETECT_UP_MAX_TILT_DEG = 20.0     # detect_up: the lowest band along the true up is a floor, not a wall


def detect_up(points, band_mm=300.0, thresh_mm=15.0, seed=0, min_extent_ratio=1.15):
    """Which of the file's axes is gravity.

    ARKit's world frame is +Y up and the parser assumed every phone export kept it; Scaniverse's
    PLY (2026-09-23, CirclesSculpture) is +Z up. Two cues, both needed: a scanned world is wider
    than it is tall (a room 5 x 4 x 2.6 m, a lawn 11 x 9 x 2.6), so the up axis has the smallest
    robust extent; and the lowest band along the up axis is a floor, a plane whose normal lies
    along that axis (`ground_plane`). The extent alone would be fooled by a stairwell; the plane
    alone by a wall facing the other axis (its normal lies along it just as a floor's does — the
    synthetic room is ambiguous on that cue). -> (axis or None, {"candidates": {axis: {"extent_mm",
    "tilt_to_up_deg", "inlier_share", ...}}, "reason"}); None means the scan does not say (extents
    within `min_extent_ratio`, or no floor under the shorter axis) and the caller should ask for
    --scan-up."""
    P = np.asarray(points, np.float64)
    lo, hi = np.percentile(P, [2, 98], axis=0)
    ext = hi - lo
    cands = {}
    for ax, u in SCAN_UP.items():
        g = ground_plane(P, u, band_mm=band_mm, thresh_mm=thresh_mm, seed=seed)
        c = {"extent_mm": round(float(ext @ u), 1)}
        if g:
            c.update({k: (round(float(g[k]), 4) if isinstance(g[k], float) else g[k])
                      for k in ("tilt_to_up_deg", "inlier_share", "inliers", "band_points", "rms_mm")})
        cands[ax] = c
    short = min(cands, key=lambda ax: cands[ax]["extent_mm"])
    other = [ax for ax in cands if ax != short][0]
    ratio = cands[other]["extent_mm"] / max(cands[short]["extent_mm"], 1e-9)
    if ratio < min_extent_ratio:
        return None, {"candidates": cands, "reason": f"the scan is about as tall along {short.upper()} as along "
                                                     f"{other.upper()} (extents within x{min_extent_ratio:g})"}
    tilt = cands[short].get("tilt_to_up_deg")
    if tilt is None or tilt > DETECT_UP_MAX_TILT_DEG:
        return None, {"candidates": cands, "reason": f"the shortest axis {short.upper()} has no floor under it "
                                                     f"(lowest band tilted {tilt if tilt is not None else 'n/a'} deg, "
                                                     f"want <= {DETECT_UP_MAX_TILT_DEG:g})"}
    return short, {"candidates": cands, "reason": f"{short.upper()} is the scan's shortest axis (x{ratio:.2f}) and "
                                                  f"its lowest band is a plane {tilt:.1f} deg from it"}


def up_axis(R_scan_to_solve, scan_up="y", current_up=None):
    """The scan's gravity up (ARKit / Polycam export +Y; Scaniverse's PLY and other Z-up exports
    +Z — `detect_up` tells them apart) in the solve frame, and its angle in degrees to the
    solve's current up (None without one)."""
    u = np.asarray(R_scan_to_solve, np.float64) @ SCAN_UP[scan_up]
    u = u / np.linalg.norm(u)
    return u, (angle_deg(u, current_up) if current_up is not None else None)


def ground_plane(points, up, band_mm=150.0, thresh_mm=10.0, iters=300, seed=0, max_points=20000):
    """RANSAC plane through the lowest band of `points` along `up` (the floor under a scan).

    The band is everything within band_mm above the 0.5th percentile of height (not the minimum:
    a scan always has a few points under the floor). -> dict (normal oriented along up, a point on
    it, height of that point along up, inliers, rms, tilt to up) or None when the band is thin."""
    P = np.asarray(points, np.float64)
    up = np.asarray(up, np.float64) / np.linalg.norm(up)
    h = P @ up
    h0 = float(np.percentile(h, 0.5))
    band = P[h <= h0 + band_mm]
    if len(band) < 30:
        return None
    rng = np.random.default_rng(seed)
    if len(band) > max_points:
        band = band[rng.choice(len(band), max_points, replace=False)]
    best_n, best_c, best_in = None, None, -1
    for _ in range(iters):
        a, b, c = band[rng.choice(len(band), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        cnt = int((np.abs((band - a) @ n) <= thresh_mm).sum())
        if cnt > best_in:
            best_n, best_c, best_in = n, a, cnt
    if best_n is None:
        return None
    inl = band[np.abs((band - best_c) @ best_n) <= thresh_mm]
    c = inl.mean(axis=0)
    _, _, Vt = np.linalg.svd(inl - c, full_matrices=False)
    n = Vt[-1]
    if n @ up < 0:
        n = -n
    res = (inl - c) @ n
    return {"normal": n.tolist(), "point_mm": c.tolist(), "height_mm": float(c @ up),
            "inliers": int(len(inl)), "band_points": int(len(band)), "inlier_share": float(len(inl) / len(band)),
            "rms_mm": float(np.sqrt(np.mean(res ** 2))), "tilt_to_up_deg": angle_deg(n, up)}


def coverage_check(C, scan, pts=None, K=None, R=None, t=None, wh=None, names=None, cam_max_mm=3000.0,
                   pts_max_mm=50.0, near_k=300, min_pts=5, index=None):
    """Per capture: does the scan cover it?

    C (M, 3) camera centres and `scan` (N, 3) points in one frame (the solve's, mm). For each
    capture: the distance from its centre to the nearest scan point, and the median distance to
    the scan of the sparse points it sees (in its frustum when K, R, t, wh are given, else its
    `near_k` nearest sparse points). ok = centre within cam_max_mm and median within pts_max_mm
    (a capture that sees fewer than min_pts points is judged on its centre alone, and says so)."""
    C = np.asarray(C, np.float64)
    nn = index or NN(scan)
    dC, _ = nn.query(C)
    dP = None
    if pts is not None and len(pts):
        pts = np.asarray(pts, np.float64)
        dP, _ = nn.query(pts)
    rows = []
    for i in range(len(C)):
        row = {"capture": names[i] if names is not None else i, "camera_to_scan_mm": round(float(dC[i]), 1)}
        med, n_used = None, 0
        if dP is not None:
            if K is not None and R is not None and t is not None:
                Xc = pts @ np.asarray(R[i]).T + np.asarray(t[i])
                z = Xc[:, 2]
                front = z > 1e-6
                uv = Xc[:, :2] / np.where(front, z, 1.0)[:, None] @ np.asarray(K[i])[:2, :2].T + np.asarray(K[i])[:2, 2]
                w, h = (wh[i] if wh is not None else (np.inf, np.inf))
                vis = np.flatnonzero(front & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h))
            else:
                vis = np.argsort(np.linalg.norm(pts - C[i], axis=1))[:near_k]
            n_used = int(len(vis))
            if n_used >= min_pts:
                med = float(np.median(dP[vis]))
        row["points_seen"] = n_used
        row["points_median_to_scan_mm"] = round(med, 2) if med is not None else None
        row["camera_ok"] = bool(dC[i] <= cam_max_mm)
        row["points_ok"] = None if med is None else bool(med <= pts_max_mm)
        row["ok"] = bool(row["camera_ok"] and row["points_ok"] is not False)
        rows.append(row)
    return rows
