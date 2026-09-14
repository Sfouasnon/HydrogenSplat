#!/usr/bin/env python3
"""Is the second population a second depth layer in the photographs themselves?

No model, no render, no training. For one capture: warp the right eye into the left through
the homography of the plane at the subject's depth, refine every feature's correspondence by
subpixel phase correlation, triangulate it with the calibrated stereo pair, and look at the
distribution of depths. A single surface gives one cluster. A reflection or a transmitted
image seen through an interface gives a second, at a depth where nothing physical is.

Then the same back-cluster is transformed to world coordinates in every capture that shows
it. A stationary planar reflector produces a virtual scene that is projectively consistent,
so its world position agrees across captures; a curved or refractive interface produces a
view-dependent one that slides along each viewing ray.

    python3 engine/tests/stereo_layers.py <project> [captures]
"""
import json
import os
import sys

import numpy as np
import cv2

WIN, MIN_STD, MIN_RESP = 48, 10.0, 0.20


def rig_relative(sparse_dir):
    """R,t of sensor 2 in sensor 1's frame, from the COLMAP rig record (mm)."""
    for line in open(os.path.join(sparse_dir, "rigs.txt")):
        if line.startswith("#") or not line.strip():
            continue
        w = line.split()
        q = np.array([float(x) for x in w[-7:-3]])
        t = np.array([float(x) for x in w[-3:]]) * 1000.0
        qw, qx, qy, qz = q
        R = np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                      [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                      [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
        return R, t
    raise SystemExit("no rig record")


def analyse(P, cap, G, names, Rrel, trel, subj):
    K, R, t = (G[k].astype(float) for k in ("K", "R", "t"))
    vL, vR = 2 * cap, 2 * cap + 1
    Xc = R[vL] @ subj + t[vL]
    Z = float(Xc[2])
    uv = (K[vL] @ Xc)[:2] / Xc[2]
    fx = K[vL][0, 0]
    half = int(0.5 * fx * 94.5 / Z)
    L = cv2.imread(f"{P}/train/dataset/images/L/{names[vL][:-2]}.jpg", 0)
    Rimg = cv2.imread(f"{P}/train/dataset/images/R/{names[vR][:-2]}.jpg", 0)
    if L is None or Rimg is None:
        return None
    H = K[vR] @ (Rrel + np.outer(trel, [0, 0, 1.0]) / Z) @ np.linalg.inv(K[vL])
    Rw = cv2.warpPerspective(Rimg, np.linalg.inv(H), (L.shape[1], L.shape[0]), flags=cv2.INTER_LINEAR)
    x0, x1 = max(0, int(uv[0] - half)), min(L.shape[1], int(uv[0] + half))
    y0, y1 = max(0, int(uv[1] - half)), min(L.shape[0], int(uv[1] + half))
    mask = np.zeros(L.shape, np.uint8)
    mask[y0:y1, x0:x1] = 255
    pts = cv2.goodFeaturesToTrack(L, maxCorners=1200, qualityLevel=0.01, minDistance=8, mask=mask)
    if pts is None:
        return None
    Lf, Rwf = L.astype(np.float32), Rw.astype(np.float32)
    win = cv2.createHanningWindow((WIN, WIN), cv2.CV_32F)
    PL = K[vL] @ np.hstack([np.eye(3), np.zeros((3, 1))])
    PR = K[vR] @ np.hstack([Rrel, trel.reshape(3, 1)])
    rows = []
    h2 = WIN // 2
    for p in pts[:, 0, :]:
        u, v = float(p[0]), float(p[1])
        iu, iv = int(round(u)), int(round(v))
        if iu - h2 < 0 or iv - h2 < 0 or iu + h2 >= L.shape[1] or iv + h2 >= L.shape[0]:
            continue
        a = Lf[iv - h2:iv + h2, iu - h2:iu + h2]
        b = Rwf[iv - h2:iv + h2, iu - h2:iu + h2]
        if a.std() < MIN_STD or b.std() < MIN_STD:
            continue
        (dx, dy), resp = cv2.phaseCorrelate(a.copy(), b.copy(), win)
        if resp < MIN_RESP:
            continue
        q = H @ np.array([u + dx, v + dy, 1.0])          # matching pixel in the right eye
        q = q[:2] / q[2]
        X = cv2.triangulatePoints(PL, PR, np.array([[u], [v]]), q.reshape(2, 1))
        X = (X[:3] / X[3]).ravel()
        rows.append((u, v, dx, dy, float(X[2]), resp))
    if not rows:
        return None
    A = np.array(rows)
    return {"cap": cap, "plane_mm": Z, "uv": uv, "box": (x0, y0, x1, y1), "rows": A,
            "R_L": R[vL], "t_L": t[vL]}


def main(project, caps):
    G = np.load(f"{project}/train/dataset/rig.npz", allow_pickle=True)
    names = [str(x) for x in G["names"]]
    subj = np.median(G["pts"].astype(float), 0)
    Rrel, trel = rig_relative(f"{project}/train/dataset/sparse")
    print(f"rig: |t| = {np.linalg.norm(trel):.3f} mm, relative rotation "
          f"{np.degrees(np.linalg.norm(cv2.Rodrigues(Rrel)[0])):.3f} deg\n")
    print(f"{'cap':>6} {'plane':>7} {'n':>5} {'front cluster':>22} {'back cluster':>22} {'offset':>9}")
    out = {}
    for c in caps:
        r = analyse(project, c, G, names, Rrel, trel, subj)
        if r is None:
            continue
        z = r["rows"][:, 4]
        keep = (z > 100) & (z < 2000)
        z = z[keep]
        if len(z) < 20:
            continue
        # two-cluster split by 1-D k-means on depth
        lo, hi = np.percentile(z, 20), np.percentile(z, 80)
        for _ in range(40):
            m = (z < (lo + hi) / 2)
            if m.sum() == 0 or (~m).sum() == 0:
                break
            lo, hi = z[m].mean(), z[~m].mean()
        front, back = z[z < (lo + hi) / 2], z[z >= (lo + hi) / 2]
        sep = np.median(back) - np.median(front) if len(back) else float("nan")
        print(f"cap{c:03d} {r['plane_mm']:7.0f} {len(z):5d} "
              f"{np.median(front):8.0f} mm x{len(front):4d} ({100*len(front)/len(z):4.0f}%) "
              f"{np.median(back):8.0f} mm x{len(back):4d} ({100*len(back)/len(z):4.0f}%) "
              f"{sep:+8.0f} mm")
        r["rows"] = r["rows"][keep]
        out[c] = (r, front, back, (lo + hi) / 2)
    # world position of the back cluster, per capture
    print("\nback-cluster centroid in world coordinates (mm) — a stable virtual layer agrees across captures:")
    W = []
    for c, (r, front, back, thr) in out.items():
        A = r["rows"]
        sel = A[:, 4] >= thr
        if sel.sum() < 10:
            continue
        K = G["K"].astype(float)[2 * c]
        uvz = A[sel][:, [0, 1, 4]]
        Xc = np.stack([(uvz[:, 0] - K[0, 2]) / K[0, 0] * uvz[:, 2],
                       (uvz[:, 1] - K[1, 2]) / K[1, 1] * uvz[:, 2], uvz[:, 2]], 1)
        Xw = (r["R_L"].T @ (Xc - r["t_L"]).T).T
        m = np.median(Xw, 0)
        W.append(m)
        print(f"  cap{c:03d}  n={int(sel.sum()):4d}  centroid {np.round(m,1)}  "
              f"distance from subject centre {np.linalg.norm(m-subj):6.1f} mm")
    if len(W) > 1:
        W = np.array(W)
        print(f"\n  spread of those centroids: {np.round(W.std(0),1)} mm (sd per axis), "
              f"{np.linalg.norm(W.std(0)):.1f} mm overall")


if __name__ == "__main__":
    proj = sys.argv[1] if len(sys.argv) > 1 else "Projects/2026-09-13_coins"
    caps = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [3, 5, 7, 9, 21, 40, 50]
    main(proj, caps)
