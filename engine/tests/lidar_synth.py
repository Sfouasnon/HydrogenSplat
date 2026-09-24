"""Synthetic LiDAR scenes for test_lidar / test_scale_lidar.

A room as a phone scan sees it, in metres with +Y up (ARKit's frame): a 5 x 4 m floor, two
2.6 m walls meeting in a corner, a 0.6 x 0.5 x 0.45 m box turned 30 degrees, a 0.3 m ball on the
floor and a 1.2 m column — points drawn uniformly on each surface, coloured per surface.

The "solve" is what photogrammetry of the same room would give: a random subset of the scan's
points inside the part the cameras covered (the lowest 60 % in x — 40 % of the scan is outside
it), 3 mm Gaussian noise per axis, then moved into an arbitrary frame by a random Sim(3)
``solve = k R X_mm + t`` (k 0.7-1.4 for a mono solve, 1 for a metric stereo one). Cameras stand
at eye height on an arc inside the covered part, looking at the ball and the box, and go
through the same Sim(3), so rig.npz can be written in either the mono or the stereo layout.
"""
import os

import numpy as np

AREA_PTS = 6000            # points per square metre: ~10 mm mean spacing, 5 mm to the nearest neighbour


def rot(axis, deg):
    a = np.radians(deg)
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    Kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * Kx + (1 - np.cos(a)) * Kx @ Kx


def random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _rect(rng, o, u, v, density):
    """Uniform points on the parallelogram o + a u + b v, a, b in [0, 1]."""
    area = np.linalg.norm(np.cross(u, v))
    n = int(area * density)
    ab = rng.random((n, 2))
    return o + ab[:, :1] * u + ab[:, 1:] * v


def room(density=AREA_PTS, seed=0):
    """-> (points (N, 3) metres, colours (N, 3) uint8, labels (N,) surface id)."""
    rng = np.random.default_rng(seed)
    parts = []
    X, Z, H = 5.0, 4.0, 2.6
    parts.append(_rect(rng, np.zeros(3), np.array([X, 0, 0]), np.array([0, 0, Z]), density))          # floor
    parts.append(_rect(rng, np.zeros(3), np.array([X, 0, 0]), np.array([0, H, 0]), density))          # wall z=0
    parts.append(_rect(rng, np.zeros(3), np.array([0, 0, Z]), np.array([0, H, 0]), density))          # wall x=0
    # box: 0.6 (x) x 0.5 (y) x 0.45 (z), turned 30 degrees about +Y, standing on the floor
    Rb = rot([0, 1, 0], 30.0)
    c = np.array([2.6, 0.0, 1.8])
    ex, ey, ez = Rb @ [0.6, 0, 0], np.array([0, 0.5, 0]), Rb @ [0, 0, 0.45]
    o = c - ex / 2 - ez / 2
    box = [_rect(rng, o + ey, ex, ez, density),                     # top
           _rect(rng, o, ex, ey, density), _rect(rng, o + ez, ex, ey, density),
           _rect(rng, o, ez, ey, density), _rect(rng, o + ex, ez, ey, density)]
    parts.append(np.vstack(box))
    # ball r 0.3 on the floor
    n = int(4 * np.pi * 0.09 * density)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    ball = np.array([1.3, 0.3, 2.8]) + 0.3 * d
    parts.append(ball[ball[:, 1] > 0.01])
    # column r 0.15, 1.2 m tall
    n = int(2 * np.pi * 0.15 * 1.2 * density)
    th = rng.random(n) * 2 * np.pi
    parts.append(np.stack([3.8 + 0.15 * np.cos(th), rng.random(n) * 1.2, 1.0 + 0.15 * np.sin(th)], 1))
    cols = [(150, 140, 120), (200, 200, 190), (190, 200, 210), (180, 60, 40), (40, 90, 180), (60, 160, 70)]
    P = np.vstack(parts)
    lab = np.concatenate([np.full(len(p), i) for i, p in enumerate(parts)])
    C = np.array(cols, float)[lab] + rng.normal(0, 6, (len(P), 3))
    return P, np.clip(C, 0, 255).astype(np.uint8), lab


class Scene:
    """The room scan (metres) and a photogrammetric solve of it in an arbitrary Sim(3) frame."""

    def __init__(self, k=1.0, seed=1, n_solve=12000, noise_mm=3.0, covered=0.6, n_cams=10,
                 density=AREA_PTS, outliers=0.0, cam_height=1.0, target_height=0.75):
        rng = np.random.default_rng(seed)
        self.scan_m, self.colors, self.labels = room(density, seed=seed)
        X = self.scan_m * 1000.0
        cut = np.quantile(X[:, 0], covered)
        self.cut_mm = float(cut)
        cov = np.flatnonzero(X[:, 0] <= cut)
        pick = rng.choice(cov, min(n_solve, len(cov)), replace=False)
        Xs = X[pick] + rng.normal(0, noise_mm, (len(pick), 3))
        if outliers:
            n_o = int(outliers * len(Xs))
            lo, hi = X.min(0), X.max(0)
            Xs[:n_o] = lo + rng.random((n_o, 3)) * (hi - lo)
        self.k = float(k)
        self.Rg = random_rotation(rng)
        self.tg = rng.uniform(-2000, 2000, 3)
        self.pts_true_mm = Xs
        self.pts = self.to_solve(Xs)
        # cameras on an arc at cam_height, looking at a point between the ball and the box (the
        # defaults pitch them ~8 degrees down; coverage's mean-camera up leans by about that much)
        target = np.array([1.9, target_height, 2.3]) * 1000
        self.C_true, self.Rc_true = [], []
        for az in np.linspace(-60, 60, n_cams):
            a = np.radians(az + 200.0)
            C = target + np.array([1700 * np.cos(a), 1000.0 * (cam_height - target_height), 1700 * np.sin(a)])
            C[0] = np.clip(C[0], 250, cut - 100)
            C[2] = np.clip(C[2], 250, 3750)
            fwd = (target - C) / np.linalg.norm(target - C)
            right = np.cross(fwd, [0, 1.0, 0])      # OpenCV x right, y down, z forward; +Y is up
            right = right / np.linalg.norm(right)
            down = np.cross(fwd, right)
            R = np.stack([right, down, fwd])
            self.C_true.append(C)
            self.Rc_true.append(R)
        self.C_true, self.Rc_true = np.array(self.C_true), np.array(self.Rc_true)
        self.K = np.array([[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1]])
        self.size = (1920, 1080)

    def to_solve(self, X_mm):
        return self.k * np.asarray(X_mm) @ self.Rg.T + self.tg

    def cameras_solve(self):
        """(C, R, t) of every camera in the solve frame."""
        C = self.to_solve(self.C_true)
        R = self.Rc_true @ self.Rg.T
        t = -np.einsum("nij,nj->ni", R, C)
        return C, R, t

    # ------------------------------------------------------------------ files
    def write_scan_ply(self, path, units="m", binary=True, colours=True, normals=False, faces=None):
        P = self.scan_m * {"m": 1.0, "mm": 1000.0, "cm": 100.0}[units]
        write_ply(path, P, self.colors if colours else None, binary=binary, faces=faces,
                  normals=np.tile([0, 1.0, 0], (len(P), 1)) if normals else None)

    def write_mono_rig(self, dataset, names=None):
        """train/dataset as monocolmap.py leaves it (board_synth.write_mono_dataset)."""
        import board_synth as BS
        C, R, t = self.cameras_solve()
        names = names or [f"sel{i:03d}-{10 * i:05d}" for i in range(len(C))]
        BS.write_mono_dataset(dataset, names, self.K, R, t, self.size, self.pts)
        return names

    def write_stereo_rig(self, dataset, baseline_mm=10.6, sparse=False):
        """A rigcolmap.py-style stereo rig.npz: capNNN_L, capNNN_R interleaved, no stereo key;
        with `sparse`, the text model beside it that an apply transforms and rewrites."""
        C, R, t = self.cameras_solve()
        names, Ks, Rs, ts, Cs = [], [], [], [], []
        for i in range(len(C)):
            for eye, off in (("L", 0.0), ("R", baseline_mm * self.k)):
                Ce = C[i] + off * R[i][0]            # the right eye sits along the camera's +x
                names.append(f"cap{i:03d}_{eye}")
                Ks.append(self.K)
                Rs.append(R[i])
                ts.append(-R[i] @ Ce)
                Cs.append(Ce)
        os.makedirs(dataset, exist_ok=True)
        w, h = self.size
        np.savez(os.path.join(dataset, "rig.npz"), names=np.array(names), K=np.array(Ks), R=np.array(Rs),
                 t=np.array(ts), C=np.array(Cs), pts=self.pts, wh=np.tile([w, h], (len(names), 1)),
                 w=w, h=h, s_mm=1.0)
        if sparse:
            # the text sparse model `hs scale` transforms and rigcolmap.write_rig_npz rewrites:
            # two PINHOLE cameras, images L/capNNN.jpg and R/capNNN.jpg, COLMAP units = mm / 1000
            from board_synth import quat_wxyz
            sp = os.path.join(dataset, "sparse")
            os.makedirs(sp, exist_ok=True)
            with open(os.path.join(sp, "cameras.txt"), "w") as f:
                for cid in (1, 2):
                    f.write(f"{cid} PINHOLE {w} {h} {float(self.K[0, 0])!r} {float(self.K[1, 1])!r} "
                            f"{float(self.K[0, 2])!r} {float(self.K[1, 2])!r}\n")
            pts_m = np.asarray(self.pts, float) / 1000.0
            with open(os.path.join(sp, "images.txt"), "w") as f:
                for i, n in enumerate(names):
                    cap, eye = n.split("_")
                    q = quat_wxyz(Rs[i])
                    tv = np.asarray(ts[i], float) / 1000.0
                    f.write(f"{i + 1} {float(q[0])!r} {float(q[1])!r} {float(q[2])!r} {float(q[3])!r} "
                            f"{float(tv[0])!r} {float(tv[1])!r} {float(tv[2])!r} {1 if eye == 'L' else 2} {eye}/{cap}.jpg\n")
                    f.write(" ".join(f"{10.0 + j} {20.0 + j} {j + 1}" for j in range(len(pts_m))) + "\n")
            with open(os.path.join(sp, "points3D.txt"), "w") as f:
                for j, p in enumerate(pts_m):
                    track = " ".join(f"{i + 1} {j}" for i in range(len(names)))
                    f.write(f"{j + 1} {float(p[0])!r} {float(p[1])!r} {float(p[2])!r} 128 128 128 0.5 {track}\n")
        return names


def yard(density=1500, seed=0):
    """An outdoor scene as `hs scale --init silhouette` meets it (Circles in miniature), metres, +Y
    up: an 8 x 8 m lawn, and on it the subject — the box of `room` with the ball pressed against
    one side and a 1.2 m column against another, one connected thing (box and ball alone are
    nearly the same silhouette turned half round: the fit found IoU 0.85 there against 0.82 at the
    truth) — plus a thin 0.8 m post 2.6 m away (a distractor, smaller than the subject).
    -> (points, colours, labels: 0 lawn, 1 box, 2 ball, 3 post, 4 column)."""
    rng = np.random.default_rng(seed)
    parts = [_rect(rng, np.array([-4.0, 0, -4.0]), np.array([8.0, 0, 0]), np.array([0, 0, 8.0]), density)]
    Rb = rot([0, 1, 0], 30.0)
    ex, ey, ez = Rb @ [0.6, 0, 0], np.array([0, 0.5, 0]), Rb @ [0, 0, 0.45]
    o = -ex / 2 - ez / 2
    parts.append(np.vstack([_rect(rng, o + ey, ex, ez, 4 * density),
                            _rect(rng, o, ex, ey, 4 * density), _rect(rng, o + ez, ex, ey, 4 * density),
                            _rect(rng, o, ez, ey, 4 * density), _rect(rng, o + ex, ez, ey, 4 * density)]))
    n = int(4 * np.pi * 0.09 * 4 * density)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    side = Rb @ np.array([0.3 + 0.28, 0, 0])                 # just touching the box's +x face
    ball = np.array([side[0], 0.3, side[2]]) + 0.3 * d
    parts.append(ball[ball[:, 1] > 0.01])
    n = int(2 * np.pi * 0.08 * 0.8 * 4 * density)
    th = rng.random(n) * 2 * np.pi
    parts.append(np.stack([-1.8 + 0.08 * np.cos(th), rng.random(n) * 0.8, 1.9 + 0.08 * np.sin(th)], 1))
    n = int(2 * np.pi * 0.12 * 1.2 * 4 * density)
    th = rng.random(n) * 2 * np.pi
    cc = Rb @ np.array([0, 0, -(0.225 + 0.11)])               # against the box's -z face
    parts.append(np.stack([cc[0] + 0.12 * np.cos(th), rng.random(n) * 1.2, cc[2] + 0.12 * np.sin(th)], 1))
    cols = [(70, 120, 50), (180, 60, 40), (40, 90, 180), (200, 200, 200), (200, 170, 40)]
    P = np.vstack(parts)
    lab = np.concatenate([np.full(len(p), i) for i, p in enumerate(parts)])
    C = np.array(cols, float)[lab] + rng.normal(0, 6, (len(P), 3))
    return P, np.clip(C, 0, 255).astype(np.uint8), lab


class YardScene(Scene):
    """`yard` and a solve of it that holds the lawn but not the subject (glossy, as the Circles
    ring was): n_solve lawn points within 4.5 m of the subject, noise_mm per axis, in a random
    Sim(3) frame `solve = k R X_mm + t`. Cameras on a full orbit of radius_mm at cam_height_m,
    looking at the subject's middle; `write_masks` renders the subject's silhouette in each."""

    def __init__(self, k=0.2, seed=3, n_solve=8000, noise_mm=5.0, n_cams=24, radius_mm=3000.0,
                 cam_height_m=1.5, density=1500):
        rng = np.random.default_rng(seed)
        self.scan_m, self.colors, self.labels = yard(density, seed=seed)
        X = self.scan_m * 1000.0
        lawn = np.flatnonzero((self.labels == 0) & (np.linalg.norm(X[:, [0, 2]], axis=1) <= 4500))
        pick = rng.choice(lawn, min(n_solve, len(lawn)), replace=False)
        Xs = X[pick] + rng.normal(0, noise_mm, (len(pick), 3))
        self.k = float(k)
        self.Rg = random_rotation(rng)
        self.tg = rng.uniform(-2000, 2000, 3)
        self.pts_true_mm = Xs
        self.pts = self.to_solve(Xs)
        target = np.array([0.1, 0.3, 0.0]) * 1000
        self.C_true, self.Rc_true = [], []
        for az in np.linspace(0, 360, n_cams, endpoint=False):
            a = np.radians(az)
            C = np.array([radius_mm * np.cos(a), 1000.0 * cam_height_m, radius_mm * np.sin(a)])
            fwd = (target - C) / np.linalg.norm(target - C)
            right = np.cross(fwd, [0, 1.0, 0])
            right = right / np.linalg.norm(right)
            down = np.cross(fwd, right)
            self.C_true.append(C)
            self.Rc_true.append(np.stack([right, down, fwd]))
        self.C_true, self.Rc_true = np.array(self.C_true), np.array(self.Rc_true)
        self.K = np.array([[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1]])
        self.size = (1920, 1080)
        self.cut_mm = None

    def subject_mm(self):
        return self.scan_m[np.isin(self.labels, (1, 2, 4))] * 1000.0

    def write_masks(self, dataset, eye="L", names=None, images=True):
        """masks/{eye}/capNNN.png: the subject's points projected through each true camera, dilated
        and closed into a solid silhouette (white = subject); images/{eye}/capNNN.jpg beside them
        (the lawn green, the silhouette red: enough for the overlays)."""
        import cv2
        X = self.subject_mm()
        w, h = self.size
        out = []
        for i, (C, R) in enumerate(zip(self.C_true, self.Rc_true)):
            name = names[i] if names else f"cap{i:03d}"
            xc = (X - C) @ R.T
            uv = xc @ self.K.T
            u, v = uv[:, 0] / uv[:, 2], uv[:, 1] / uv[:, 2]
            ok = (xc[:, 2] > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            m = np.zeros((h, w), np.uint8)
            m[v[ok].astype(int), u[ok].astype(int)] = 255
            m = cv2.dilate(m, np.ones((5, 5), np.uint8))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
            for top, img in (("masks", m), ("images", None)):
                d = os.path.join(dataset, top, eye)
                os.makedirs(d, exist_ok=True)
                if top == "masks":
                    cv2.imwrite(os.path.join(d, name + ".png"), img)
                elif images:
                    photo = np.full((h, w, 3), (60, 130, 70), np.uint8)
                    photo[m > 0] = (50, 60, 190)
                    cv2.imwrite(os.path.join(d, name + ".jpg"), photo, [cv2.IMWRITE_JPEG_QUALITY, 80])
            out.append(name)
        return out


def write_ply(path, P, rgb=None, binary=True, faces=None, normals=None, comments=(), extra_elements=()):
    """A scanner-style PLY: vertex x y z (float) [nx ny nz] [red green blue (uchar)] and optional
    triangle faces (list uchar int vertex_indices)."""
    P = np.asarray(P, float)
    props = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if normals is not None:
        props += [("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
    if rgb is not None:
        props += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    head = ["ply", f"format {'binary_little_endian' if binary else 'ascii'} 1.0"]
    head += [f"comment {c}" for c in comments]
    for name, count, lines in extra_elements:
        head += [f"element {name} {count}"] + lines
    head += [f"element vertex {len(P)}"]
    head += [f"property {'float' if t == 'f4' else 'uchar'} {n}" for n, t in props]
    if faces is not None:
        head += [f"element face {len(faces)}", "property list uchar int vertex_indices"]
    head += ["end_header"]
    cols = [P]
    if normals is not None:
        cols.append(np.asarray(normals, float))
    if rgb is not None:
        cols.append(np.asarray(rgb, float))
    with open(path, "wb") as f:
        f.write(("\n".join(head) + "\n").encode("ascii"))
        if binary:
            arr = np.empty(len(P), dtype=[(n, "<" + t if t == "f4" else t) for n, t in props])
            k = 0
            for block in cols:
                for c in range(block.shape[1]):
                    arr[props[k][0]] = block[:, c]
                    k += 1
            f.write(arr.tobytes())
            if faces is not None:
                fa = np.empty(len(faces), dtype=[("n", "u1"), ("i", "<i4", (3,))])
                fa["n"] = 3
                fa["i"] = faces
                f.write(fa.tobytes())
        else:
            M = np.hstack(cols)
            ncol = P.shape[1] + (3 if normals is not None else 0)
            for row in M:
                f.write((" ".join(f"{v:.6f}" for v in row[:ncol]) + (" " + " ".join(str(int(v)) for v in row[ncol:]) if rgb is not None else "") + "\n").encode())
            if faces is not None:
                for tri in faces:
                    f.write(("3 " + " ".join(str(int(i)) for i in tri) + "\n").encode())


def blob_cloud(n=8000, seed=5):
    """Something that is not the room: a few fuzzy volumetric blobs (a tree, a crowd), mm."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(-1500, 1500, (6, 3))
    P = centres[rng.integers(0, len(centres), n)] + rng.normal(0, 250, (n, 3))
    return P
