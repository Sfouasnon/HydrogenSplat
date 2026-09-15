#!/usr/bin/env python3
"""Independent per-eye PnP against the solved sparse map — does each capture obey the rigid rig?

The rig solve gives every capture ONE pose shared by both eyes. If L and R were not exposed at
the same instant (or one image is warped), that shared pose is a compromise and the sparse
reprojection error can still look fine. This re-poses every image on its own:

  * 2D-3D pairs come from the image's own observations, but only for points that at least
    MIN_OTHER observations from OTHER captures also see, so a point this frame created
    does not vote for this frame's pose.
  * cv2 PnP RANSAC + LM refine, intrinsics fixed (the dataset is PINHOLE, undistorted).

Per capture it reports
  e_rig_L / e_rig_R  median px between the eye's independent pose and the pose the OTHER eye's
                     independent pose implies through the solved sensor_from_rig
  e_ba_L  / e_ba_R   median px between the eye's independent pose and the BA (rig) pose
  drot / dt          the independently observed R-from-L transform minus the rig's, deg / mm
  row_slope_L        px of L x-residual (BA pose) per 1000 rows — a rolling-shutter hint
and flags captures with a robust threshold (see --warn / --hard).

  python3 engine/tools/rig_consistency.py Projects/2026-09-13_coins
"""
import argparse, json, os, sys
import numpy as np
import cv2


def qmat(qw, qx, qy, qz):
    q = np.array([qw, qx, qy, qz], float); q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def T(R, t):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t; return M


def lines(p):
    return [l.rstrip("\n") for l in open(p) if not l.startswith("#")]


def load(sp):
    cams = {}
    for l in lines(os.path.join(sp, "cameras.txt")):
        if not l.strip(): continue
        f = l.split(); assert f[1] == "PINHOLE", f[1]
        fx, fy, cx, cy = map(float, f[4:8])
        cams[int(f[0])] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    rl = [l for l in lines(os.path.join(sp, "rigs.txt")) if l.strip()]
    f = rl[0].split()
    # RIG_ID NUM REF_TYPE REF_ID  then (TYPE ID HAS_POSE [7])
    ref_cam = int(f[3]); rest = f[4:]; S = {}
    i = 0
    while i < len(rest):
        cid, has = int(rest[i+1]), int(rest[i+2])
        if has:
            v = list(map(float, rest[i+3:i+10])); S[cid] = T(qmat(*v[:4]), v[4:7]); i += 10
        else:
            i += 3
    imgs = {}
    il = lines(os.path.join(sp, "images.txt"))
    il = [l for l in il]
    k = 0
    while k < len(il):
        h = il[k].split()
        if not h: k += 1; continue
        pts = il[k+1].split() if k + 1 < len(il) else []
        v = list(map(float, h[1:8]))
        a = np.array(pts, float).reshape(-1, 3) if pts else np.zeros((0, 3))
        imgs[int(h[0])] = dict(T=T(qmat(*v[:4]), v[4:7]), cam=int(h[8]), name=h[9], obs=a)
        k += 2
    P = {}
    for l in lines(os.path.join(sp, "points3D.txt")):
        if not l.strip(): continue
        f = l.split()
        tr = np.array(f[8:], int).reshape(-1, 2)[:, 0]
        P[int(f[0])] = (np.array(list(map(float, f[1:4]))), tr)
    return cams, ref_cam, S, imgs, P


def proj(K, Tcw, X):
    Xc = X @ Tcw[:3, :3].T + Tcw[:3, 3]
    uv = Xc @ K.T
    return uv[:, :2] / uv[:, 2:3], Xc[:, 2]


def pnp(K, X, x, thr):
    if len(X) < 12: return None
    ok, rv, tv, inl = cv2.solvePnPRansac(X, x, K, None, iterationsCount=5000,
                                         reprojectionError=thr, confidence=0.9999,
                                         flags=cv2.SOLVEPNP_SQPNP)
    if not ok or inl is None or len(inl) < 12: return None
    inl = inl.ravel()
    rv, tv = cv2.solvePnPRefineLM(X[inl], x[inl], K, None, rv, tv)
    R, _ = cv2.Rodrigues(rv)
    Tcw = T(R, tv.ravel())
    r = np.linalg.norm(proj(K, Tcw, X[inl])[0] - x[inl], axis=1)
    return dict(T=Tcw, inl=inl, rms=float(np.sqrt(np.mean(r**2))))


def rdeg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("--min-other", type=int, default=2, help="other-capture observations a point needs")
    ap.add_argument("--thr", type=float, default=3.0, help="RANSAC px")
    ap.add_argument("--warn", type=float, default=2.0)
    ap.add_argument("--hard", type=float, default=4.0)
    a = ap.parse_args()
    cv2.setRNGSeed(0)                       # RANSAC is random; a flag must not flip between runs
    sp = os.path.join(a.project, "train/dataset/sparse")
    cams, ref_cam, S, imgs, P = load(sp)
    other = [c for c in S if c != ref_cam]; assert len(other) == 1
    S_RL = S[other[0]]                       # R_from_L (rig frame == ref camera)
    S_LR = np.linalg.inv(S_RL)
    cap_of = {i: os.path.splitext(os.path.basename(d["name"]))[0] for i, d in imgs.items()}
    eye_of = {i: d["name"].split("/")[0] for i, d in imgs.items()}
    by_cap = {}
    for i in imgs: by_cap.setdefault(cap_of[i], {})[eye_of[i]] = i

    rows = []
    for cap in sorted(by_cap):
        ids = by_cap[cap]
        if set(ids) != {"L", "R"}: continue
        sol = {}
        for e, iid in ids.items():
            d = imgs[iid]; K = cams[d["cam"]]
            X, x = [], []
            for u, v, pid in d["obs"]:
                pid = int(pid)
                if pid < 0 or pid not in P: continue
                xyz, tr = P[pid]
                n_other = sum(1 for t in tr if cap_of.get(int(t)) != cap)
                if n_other >= a.min_other:
                    X.append(xyz); x.append((u, v))
            X = np.array(X, float); x = np.array(x, float)
            s = pnp(K, X, x, a.thr)
            sol[e] = dict(s=s, K=K, X=X, x=x, Tba=d["T"], n=len(X))
        L, R = sol["L"], sol["R"]
        row = dict(capture=cap, n_L=L["n"], n_R=R["n"])
        if L["s"] is None or R["s"] is None:
            row["status"] = "pnp_failed"; rows.append(row); continue
        TL, TR = L["s"]["T"], R["s"]["T"]
        XL = L["X"][L["s"]["inl"]]; XR = R["X"][R["s"]["inl"]]
        TL_from_R = S_LR @ TR; TR_from_L = S_RL @ TL
        med = lambda K, A, B, X: float(np.median(np.linalg.norm(proj(K, A, X)[0] - proj(K, B, X)[0], axis=1)))
        E = TR @ np.linalg.inv(TL); D = E @ np.linalg.inv(S_RL)
        # rolling-shutter hint: L x-residual under BA pose vs image row
        xb, _ = proj(L["K"], L["Tba"], XL); res = (L["x"][L["s"]["inl"]] - xb)[:, 0]
        yy = L["x"][L["s"]["inl"]][:, 1]
        slope = float(np.polyfit(yy, res, 1)[0] * 1000) if len(yy) > 20 else None
        # direction of the L correction: independent vs BA, image-space median vector
        dv = np.median(proj(L["K"], TL, XL)[0] - proj(L["K"], L["Tba"], XL)[0], axis=0)
        row.update(inl_L=len(L["s"]["inl"]), inl_R=len(R["s"]["inl"]),
                   rms_L=round(L["s"]["rms"], 2), rms_R=round(R["s"]["rms"], 2),
                   e_rig_L=round(med(L["K"], TL, TL_from_R, XL), 2),
                   e_rig_R=round(med(R["K"], TR, TR_from_L, XR), 2),
                   e_ba_L=round(med(L["K"], TL, L["Tba"], XL), 2),
                   e_ba_R=round(med(R["K"], TR, R["Tba"], XR), 2),
                   dvec_L=[round(float(dv[0]), 2), round(float(dv[1]), 2)],
                   drot_deg=round(rdeg(D[:3, :3]), 3),
                   dt_mm=round(float(np.linalg.norm(D[:3, 3])) * 1000, 2),
                   row_slope_L=None if slope is None else round(slope, 2),
                   status="ok")
        rows.append(row)

    ok = [r for r in rows if r["status"] == "ok"]
    e = np.array([max(r["e_rig_L"], r["e_rig_R"]) for r in ok])
    m = float(np.median(e)); sig = float(1.4826 * np.median(np.abs(e - m)))
    for r in ok:
        v = max(r["e_rig_L"], r["e_rig_R"])
        r["flag"] = ("HARD" if (v > a.hard or v > m + 5 * sig) else
                     "warn" if v > max(a.warn, m + 3.5 * sig) else "")
        if min(r["inl_L"], r["inl_R"]) < 75: r["flag"] = (r["flag"] + " lowsupport").strip()
    hdr = f"{'cap':8}{'inlL':>6}{'inlR':>6}{'rmsL':>6}{'rmsR':>6}{'eRigL':>7}{'eRigR':>7}{'eBaL':>7}{'eBaR':>7}  {'dL(px)':>13}{'dRot°':>7}{'dT mm':>7}{'rowSl':>7}  flag"
    print(hdr)
    for r in rows:
        if r["status"] != "ok":
            print(f"{r['capture']:8} {r['status']}  (n_L {r['n_L']}, n_R {r['n_R']})"); continue
        print(f"{r['capture']:8}{r['inl_L']:6d}{r['inl_R']:6d}{r['rms_L']:6.2f}{r['rms_R']:6.2f}"
              f"{r['e_rig_L']:7.2f}{r['e_rig_R']:7.2f}{r['e_ba_L']:7.2f}{r['e_ba_R']:7.2f}  "
              f"{r['dvec_L'][0]:+6.2f},{r['dvec_L'][1]:+6.2f}{r['drot_deg']:7.3f}{r['dt_mm']:7.2f}"
              f"{(r['row_slope_L'] or 0):7.2f}  {r['flag']}")
    print(f"\ne_rig (worse eye): median {m:.2f} px, MAD-sigma {sig:.2f} px; "
          f"warn > {max(a.warn, m + 3.5*sig):.2f}, hard > {min(a.hard, m + 5*sig) if False else a.hard:.2f} or > {m + 5*sig:.2f}")
    out = os.path.join(a.project, "views", "rig_consistency.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(dict(median=m, mad_sigma=sig, rows=rows), open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
