"""Synthetic ChArUco scenes for test_board / test_scale / test_exposure_board.

A board lies on a tilted plane in a world measured in millimetres; a ring of pinhole cameras
looks down at it. Each view is the board image OpenCV generates (CharucoBoard.generateImage)
warped in by the exact plane-to-image homography, so the detector runs on a real picture of
the board and the ground truth is known to the micron. The poses handed to the code under test
are in "solve units" = mm / k, i.e. an unscaled reconstruction whose true scale factor is k.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(HERE)


def rot(axis, deg):
    a = np.radians(deg)
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    Kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * Kx + (1 - np.cos(a)) * Kx @ Kx


def generated(spec, px_per_mm=8.0):
    """The generated board image and the board-mm -> image-pixel homography, fitted from the
    detector's own corners (independent of OpenCV's layout conventions)."""
    import cv2
    from hs import board as B
    det = B.Detector(spec)
    W = int(spec.sx * spec.square_mm * px_per_mm) + 80
    H = int(spec.sy * spec.square_mm * px_per_mm) + 80
    gen = det.board.generateImage((W, H), marginSize=40)
    ids, xy = det.detect(gen)
    obj = B.corner_points(spec)
    M, _ = cv2.findHomography(obj[ids][:, :2], xy, 0)
    return gen, M


class Scene:
    """Board pose, cameras and rendered views."""

    def __init__(self, spec, k=3.7, n_views=5, size=(1024, 768), f=900.0, dist_mm=520.0,
                 tilt_deg=12.0, seed=0, noise=2.0, white=255, black=0, background=180,
                 elev=(50.0, 58.0), az_span=(-45.0, 50.0), px_per_mm=8.0):
        import cv2
        self.spec, self.k, self.size = spec, float(k), size
        gen, M = generated(spec, px_per_mm)
        gen = (black + (white - black) * (gen.astype(np.float64) / 255.0))
        self.gen, self.M = gen, M
        Rb = rot([1.0, 0.3, 0.0], tilt_deg)
        self.e1, self.e2 = Rb[:, 0], Rb[:, 1]
        # the printed side faces -(e1 x e2): board x right / y down, seen from the front
        self.up = -Rb[:, 2]
        mid = np.array([spec.sx * spec.square_mm / 2, spec.sy * spec.square_mm / 2])
        self.centre = np.array([60.0, -25.0, 40.0])
        self.o = self.centre - (mid[0] * self.e1 + mid[1] * self.e2)
        self.K = np.array([[f, 0, size[0] / 2.0], [0, f, size[1] / 2.0], [0, 0, 1.0]])
        rng = np.random.default_rng(seed)
        x0 = self.e1
        y0 = np.cross(self.up, x0)
        self.R, self.t_mm, self.C_mm, self.imgs, self.clean = [], [], [], [], []
        for i, az in enumerate(np.linspace(az_span[0], az_span[1], n_views)):
            el = elev[i % 2]
            d = np.array([np.cos(np.radians(el)) * np.sin(np.radians(az)),
                          np.cos(np.radians(el)) * np.cos(np.radians(az)), np.sin(np.radians(el))])
            C = self.centre + dist_mm * (d[0] * x0 + d[1] * y0 + d[2] * self.up)
            fwd = (self.centre - C) / np.linalg.norm(self.centre - C)
            right = np.cross(fwd, self.up)
            right /= np.linalg.norm(right)
            down = np.cross(fwd, right)
            R = np.stack([right, down, fwd])
            t = -R @ C
            Hv = self.K @ np.column_stack([R @ self.e1, R @ self.e2, R @ self.o + t])
            img = cv2.warpPerspective(self.gen, Hv @ np.linalg.inv(M), size, flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=float(background))
            self.clean.append(img)                      # float code values, before noise
            img = np.clip(np.round(img + rng.normal(0, noise, img.shape)), 0, 255)
            self.imgs.append(np.repeat(img[:, :, None], 3, axis=2).astype(np.uint8))
            self.R.append(R)
            self.t_mm.append(t)
            self.C_mm.append(C)
        self.R = np.array(self.R)
        self.t_mm = np.array(self.t_mm)
        self.C_mm = np.array(self.C_mm)

    @property
    def t_units(self):
        return self.t_mm / self.k

    @property
    def C_units(self):
        return self.C_mm / self.k

    def exposed(self, v, gain_bgr, noise=1.5, seed=0):
        """View v photographed with a per-channel linear gain: the gain applies to the light, and
        the sensor's noise and the 8-bit quantisation come after it, once — as in a camera."""
        s = np.clip(self.clean[v] / 255.0, 0, 1)[:, :, None]
        lin = np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4) * np.asarray(gain_bgr)[None, None, :]
        lin = np.clip(lin, 0, 1)
        enc = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055) * 255.0
        rng = np.random.default_rng(seed + 101 * v)
        return np.clip(np.round(enc + rng.normal(0, noise, enc.shape)), 0, 255).astype(np.uint8)

    def Ks(self):
        return np.repeat(self.K[None], len(self.R), axis=0)

    def corners_world_mm(self):
        from hs import board as B
        obj = B.corner_points(self.spec)
        return self.o + np.outer(obj[:, 0], self.e1) + np.outer(obj[:, 1], self.e2)


def quat_wxyz(R):
    """Rotation matrix -> COLMAP's (qw, qx, qy, qz)."""
    R = np.asarray(R, float)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def write_mono_dataset(dataset, names, K, R, t_units, size, pts_units, imgs=None):
    """train/dataset as monocolmap.py export leaves it: images/L/<name>.jpg, a text sparse model
    in COLMAP units (= rig units / 1000, as write_rig_npz multiplies by 1000) and a mono rig.npz."""
    import cv2
    os.makedirs(os.path.join(dataset, "images", "L"), exist_ok=True)
    sp = os.path.join(dataset, "sparse")
    os.makedirs(sp, exist_ok=True)
    w, h = size
    with open(os.path.join(sp, "cameras.txt"), "w") as f:
        f.write(f"1 PINHOLE {w} {h} {float(K[0, 0])!r} {float(K[1, 1])!r} {float(K[0, 2])!r} {float(K[1, 2])!r}\n")
    pts_m = np.asarray(pts_units, float) / 1000.0
    with open(os.path.join(sp, "images.txt"), "w") as f:
        for i, n in enumerate(names):
            q = quat_wxyz(R[i])
            tv = np.asarray(t_units[i], float) / 1000.0
            f.write(f"{i + 1} {float(q[0])!r} {float(q[1])!r} {float(q[2])!r} {float(q[3])!r} {float(tv[0])!r} {float(tv[1])!r} {float(tv[2])!r} 1 L/{n}.jpg\n")
            f.write(" ".join(f"{10.0 + j} {20.0 + j} {j + 1}" for j in range(len(pts_m))) + "\n")
    with open(os.path.join(sp, "points3D.txt"), "w") as f:
        for j, p in enumerate(pts_m):
            track = " ".join(f"{i + 1} {j}" for i in range(len(names)))
            f.write(f"{j + 1} {float(p[0])!r} {float(p[1])!r} {float(p[2])!r} 128 128 128 0.5 {track}\n")
    if imgs is not None:
        for n, im in zip(names, imgs):
            cv2.imwrite(os.path.join(dataset, "images", "L", n + ".jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    C = -np.einsum("nji,nj->ni", np.asarray(R), np.asarray(t_units))
    np.savez(os.path.join(dataset, "rig.npz"), names=np.array([n + "_L" for n in names]),
             K=np.repeat(np.asarray(K)[None], len(names), axis=0), R=np.asarray(R), t=np.asarray(t_units),
             C=C, pts=np.asarray(pts_units, float), wh=np.tile([w, h], (len(names), 1)), w=w, h=h,
             s_mm=1.0, photos=np.array(names), stereo=False)
