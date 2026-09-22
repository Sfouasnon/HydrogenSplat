"""ChArUco board: detection, triangulation from rig.npz poses, metric scale, board plane, and
the white-square sample `hs exposure --reference board` matches views on.

Pure functions over numpy arrays and cv2 images, so every step is testable on synthetic data
without a project. `hs scale` (stages/scale.py) and `hs exposure` (stages/exposure.py) are the
two callers.

Board spec, as on the command line: ``SX,SY,SQUARE_MM,MARKER_MM[,DICT]`` — squares across,
squares down, the printed square and marker side in millimetres (measure the print; a printer
that scales to fit changes both), and the ArUco dictionary (default ``DICT_5X5_100``). This is
OpenCV's own ``CharucoBoard((SX, SY), square, marker, dict)``, so a board made with
``CharucoBoard.generateImage`` or calib.io's generator for the same numbers is the same board.
OpenCV >= 4.6 lays out even-row boards differently from older releases; ``legacy=True`` selects
the old layout for a board printed from an old generator.

Frames. Corner ids and board coordinates are OpenCV's: corner k of an SX x SY board sits at
``((k % (SX-1)) + 1, (k // (SX-1)) + 1) * square`` mm on the board plane, z = 0. The world is
rig.npz's: ``Xc = X @ R[v].T + t[v]``, pixel ``u = fx * Xc / z + cx`` on the undistorted
PINHOLE images, in whatever unit rig.npz is in (mm once scaled, arbitrary before).
"""
from dataclasses import dataclass

import numpy as np

DEFAULT_DICT = "DICT_5X5_100"


@dataclass(frozen=True)
class BoardSpec:
    sx: int
    sy: int
    square_mm: float
    marker_mm: float
    dictionary: str = DEFAULT_DICT
    legacy: bool = False

    def text(self):
        return f"{self.sx},{self.sy},{self.square_mm:g},{self.marker_mm:g},{self.dictionary}"


def parse_spec(s, legacy=False):
    """``"7,5,40,30[,DICT_5X5_100]"`` -> BoardSpec. ValueError with the reason otherwise."""
    import cv2
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if len(parts) not in (4, 5):
        raise ValueError(f"board {s!r}: want SX,SY,SQUARE_MM,MARKER_MM[,DICT]")
    try:
        sx, sy = int(parts[0]), int(parts[1])
        sq, mk = float(parts[2]), float(parts[3])
    except ValueError:
        raise ValueError(f"board {s!r}: SX,SY are whole numbers of squares, SQUARE_MM,MARKER_MM are millimetres")
    if sx < 2 or sy < 2 or (sx < 3 and sy < 3):
        raise ValueError(f"board {s!r}: at least 3x2 squares (a board needs interior corners)")
    if not (0 < mk < sq):
        raise ValueError(f"board {s!r}: the marker ({mk:g} mm) must be smaller than the square ({sq:g} mm)")
    name = parts[4].upper() if len(parts) == 5 else DEFAULT_DICT
    if not name.startswith("DICT_"):
        name = "DICT_" + name
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"board {s!r}: unknown ArUco dictionary {name} (e.g. DICT_5X5_100, DICT_4X4_50)")
    spec = BoardSpec(sx, sy, sq, mk, name, bool(legacy))
    n_markers = (sx * sy) // 2          # one marker per white square
    size = int(name.rsplit("_", 1)[1]) if name.rsplit("_", 1)[1].isdigit() else None
    if size is not None and n_markers > size:
        raise ValueError(f"board {s!r}: {sx}x{sy} needs {n_markers} markers, {name} has {size}")
    return spec


def make_board(spec):
    import cv2
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.dictionary))
    b = cv2.aruco.CharucoBoard((spec.sx, spec.sy), float(spec.square_mm), float(spec.marker_mm), d)
    if spec.legacy:
        if not hasattr(b, "setLegacyPattern"):
            raise ValueError("this OpenCV has no legacy ChArUco layout (setLegacyPattern needs >= 4.8)")
        b.setLegacyPattern(True)
    return b


def corner_points(spec):
    """(N, 3) board-plane coordinates in mm of every interior corner, indexed by corner id."""
    return np.asarray(make_board(spec).getChessboardCorners(), np.float64).reshape(-1, 3)


class Detector:
    """One CharucoDetector per board, reused across views (construction is not free)."""

    def __init__(self, spec):
        import cv2
        self.spec = spec
        self.board = make_board(spec)
        self.det = cv2.aruco.CharucoDetector(self.board)

    def detect(self, img, min_corners=4):
        """-> (ids (n,), xy (n, 2) float64 pixels). Empty arrays when fewer than min_corners."""
        import cv2
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cc, ids = self.det.detectBoard(gray)[:2]
        if ids is None or cc is None or len(np.asarray(ids).ravel()) < min_corners:
            return np.zeros(0, int), np.zeros((0, 2))
        return np.asarray(ids, int).ravel(), np.asarray(cc, np.float64).reshape(-1, 2)


# ------------------------------------------------------------------ geometry

def project(X, K, R, t):
    """World points (M, 3) -> pixels (M, 2) and depth (M,) in one view."""
    Xc = np.asarray(X, np.float64) @ np.asarray(R, np.float64).T + np.asarray(t, np.float64)
    z = Xc[:, 2]
    uv = (Xc[:, :2] / z[:, None]) @ np.asarray(K, np.float64)[:2, :2].T + np.asarray(K, np.float64)[:2, 2]
    return uv, z


def triangulate_point(Ks, Rs, ts, xys):
    """Linear DLT from >= 2 views. Pixels are normalised by K first so the system is well
    conditioned whatever the focal length. -> (3,) world point."""
    A = []
    for K, R, t, xy in zip(Ks, Rs, ts, xys):
        n = np.linalg.solve(np.asarray(K, np.float64), np.array([xy[0], xy[1], 1.0]))
        P = np.hstack([np.asarray(R, np.float64), np.asarray(t, np.float64).reshape(3, 1)])
        A.append(n[0] * P[2] - P[0])
        A.append(n[1] * P[2] - P[1])
    A = np.asarray(A)
    # equilibrate rows: views far away otherwise weigh less than views close up
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def triangulate(observations, K, R, t, min_views=3):
    """observations: {corner_id: [(view, (u, v)), ...]}. Every corner seen in >= min_views views
    is triangulated. -> (ids (n,), X (n, 3), residuals [(view, id, err_px)])."""
    ids, pts, res = [], [], []
    for cid in sorted(observations):
        obs = observations[cid]
        if len({v for v, _ in obs}) < min_views:
            continue
        vs = [v for v, _ in obs]
        X = triangulate_point([K[v] for v in vs], [R[v] for v in vs], [t[v] for v in vs], [xy for _, xy in obs])
        errs = []
        ok = True
        for v, xy in obs:
            uv, z = project(X[None], K[v], R[v], t[v])
            if z[0] <= 0:
                ok = False
                break
            errs.append((v, cid, float(np.linalg.norm(uv[0] - np.asarray(xy)))))
        if not ok:
            continue
        ids.append(cid)
        pts.append(X)
        res.extend(errs)
    return np.asarray(ids, int), np.asarray(pts, np.float64).reshape(-1, 3), res


def per_view_rms(residuals):
    """{view: rms px} from triangulate()'s residual list."""
    by = {}
    for v, _cid, e in residuals:
        by.setdefault(v, []).append(e)
    return {v: float(np.sqrt(np.mean(np.square(e)))) for v, e in by.items()}


def scale_from_pairs(ids, X, obj_mm):
    """Metric scale as the median over every corner pair of known_mm / reconstructed distance.

    -> {"scale", "mad", "mad_rel", "n_pairs", "n_corners"}. The median is what makes a single
    misdetected corner harmless; MAD (median absolute deviation of the ratios) is the spread."""
    ids = np.asarray(ids, int)
    X = np.asarray(X, np.float64)
    if len(ids) < 2:
        raise ValueError("need at least two triangulated corners for a distance")
    O = np.asarray(obj_mm, np.float64)[ids]
    i, j = np.triu_indices(len(ids), k=1)
    d_rec = np.linalg.norm(X[i] - X[j], axis=1)
    d_mm = np.linalg.norm(O[i] - O[j], axis=1)
    keep = d_rec > 0
    r = d_mm[keep] / d_rec[keep]
    s = float(np.median(r))
    mad = float(np.median(np.abs(r - s)))
    return {"scale": s, "mad": mad, "mad_rel": mad / s if s else float("inf"),
            "n_pairs": int(keep.sum()), "n_corners": int(len(ids))}


def fit_plane(X):
    """Least-squares plane through (n, 3) points -> (centroid, unit normal, rms out-of-plane)."""
    X = np.asarray(X, np.float64)
    c = X.mean(axis=0)
    _, sv, Vt = np.linalg.svd(X - c)
    n = Vt[-1] / np.linalg.norm(Vt[-1])
    rms = float(sv[-1] / np.sqrt(len(X))) if len(X) else float("nan")
    return c, n, rms


def orient_towards(n, origin, towards):
    """Flip normal n so it points from `origin` towards `towards` (the cameras): a board lying
    on the table then has its normal pointing up."""
    n = np.asarray(n, np.float64)
    return n if float(n @ (np.asarray(towards) - np.asarray(origin))) >= 0 else -n


def angle_deg(a, b):
    a = np.asarray(a, np.float64) / np.linalg.norm(a)
    b = np.asarray(b, np.float64) / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))


def umeyama(src, dst):
    """Similarity dst ~ s * R @ src + t (least squares). -> (s, R, t). Used as a cross-check
    on the pair-median scale: the two agree when the corners are clean. (hs/lidar.py holds the
    general one — weights, SE(3), residuals; this is it with the scale on.)"""
    from .lidar import umeyama as _umeyama
    s, R, t, _res = _umeyama(src, dst, with_scale=True)
    return s, R, t


# ------------------------------------------------------------------ the white squares

def white_cells(spec, inset=0.2):
    """[(centre_mm (2,), lo, hi)] per white square: the band of white paper about each marker.

    In a ChArUco board the white squares are the ones that carry a marker, so the interior of
    a white cell is mostly marker, black and white bits. What is reliably white is the margin
    between the marker's edge and the square's: the ring from marker/2 to square/2 about the
    cell centre. `inset` keeps that share of the ring's width away from both edges (blur, JPEG
    ringing, the black squares' bleed), so with 40 mm squares and 30 mm markers the sample is the
    band from 16 mm to 19 mm across — the middle 60 % of the 5 mm margin."""
    b = make_board(spec)
    half_sq, half_mk = spec.square_mm / 2.0, spec.marker_mm / 2.0
    w = half_sq - half_mk
    lo, hi = half_mk + inset * w, half_sq - inset * w
    return [(np.asarray(c, np.float64).reshape(-1, 3)[:, :2].mean(axis=0), lo, hi) for c in b.getObjPoints()]


def white_mask(spec, H, shape, inset=0.2):
    """Boolean image mask of every pixel on the white paper (white_cells) under homography H
    (board mm -> image px). Rasterised on the board plane at about twice the image's own pixel
    density and warped in with nearest-neighbour, so every image pixel is counted once."""
    import cv2
    h, w = shape[:2]
    c = np.array([[spec.sx * spec.square_mm / 2, spec.sy * spec.square_mm / 2]])
    p = map_points(H, np.vstack([c, c + [1.0, 0.0], c + [0.0, 1.0]]))
    px_per_mm = max(np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[0]))
    r = float(np.clip(2.0 * px_per_mm, 2.0, 20.0))
    W, Hh = int(np.ceil(spec.sx * spec.square_mm * r)), int(np.ceil(spec.sy * spec.square_mm * r))
    board = np.zeros((Hh, W), np.uint8)
    for ctr, lo, hi in white_cells(spec, inset):
        o0 = np.round((ctr - hi) * r).astype(int)
        o1 = np.round((ctr + hi) * r).astype(int)
        i0 = np.round((ctr - lo) * r).astype(int)
        i1 = np.round((ctr + lo) * r).astype(int)
        board[o0[1]:o1[1], o0[0]:o1[0]] = 1
        board[i0[1]:i1[1], i0[0]:i1[0]] = 0
    # raster pixel (i, j) is centred on board mm ((i + 0.5) / r, (j + 0.5) / r)
    S = np.array([[r, 0.0, -0.5], [0.0, r, -0.5], [0.0, 0.0, 1.0]])
    out = cv2.warpPerspective(board, np.asarray(H, np.float64) @ np.linalg.inv(S), (w, h),
                              flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out.astype(bool)


def board_homography(ids, xy, obj_mm):
    """Board plane (mm) -> image homography from detected corners (>= 4). None if degenerate."""
    import cv2
    if len(ids) < 4:
        return None
    src = np.asarray(obj_mm, np.float64)[np.asarray(ids, int)][:, :2]
    H, _ = cv2.findHomography(src, np.asarray(xy, np.float64), 0)
    return H


def map_points(H, pts):
    p = np.hstack([np.asarray(pts, np.float64), np.ones((len(pts), 1))]) @ np.asarray(H).T
    return p[:, :2] / p[:, 2:3]


def sample_white(img, mask, lut=None, clip_level=250):
    """Mean BGR of the pixels under `mask` (white_mask) in one image, in linear light when `lut`
    (256 -> linear) is given.

    img: (h, w, 3) uint8. -> (mean_bgr (3,), n_used, clipped_fraction). Clipped pixels (any
    channel >= clip_level) are counted and dropped, because a clipped white carries no exposure
    information. Only the masked pixels are decoded, so a 4K frame costs nothing extra."""
    raw = img[mask].reshape(-1, 3)
    if not len(raw):
        return None, 0, 0.0
    clipped = raw.max(axis=1) >= clip_level
    frac = float(clipped.mean())
    raw = raw[~clipped]
    if not len(raw):
        return None, 0, frac
    vals = lut[raw] if lut is not None else raw.astype(np.float64)
    return vals.mean(axis=0), int(len(raw)), frac
