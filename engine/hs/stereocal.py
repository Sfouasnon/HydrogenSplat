#!/usr/bin/env python3
"""
stereocal.py — per-eye intrinsics + stereo extrinsics of the RED Hydrogen One _2x1 capture
from ChArUco captures (both eyes see the board in every _2x1 file).

Usage: python3 stereocal.py IMG_*_2x1.jpg --square 15.45 --marker 12.36 -o h1_stereo.npz
Board: SplatVizLive 8x11 ChArUco, DICT_4X4_100 (nominal 100/80 mm; on-screen size passed in).
"""
import argparse, glob, json, sys
import cv2, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photos", nargs="+"); ap.add_argument("-o", "--out", default="h1_stereo.npz")
    ap.add_argument("--square", type=float, required=True); ap.add_argument("--marker", type=float, required=True)
    ap.add_argument("--min-corners", type=int, default=20)
    ap.add_argument("--f-init", type=float, default=None, help="initial focal guess in px (default: nominal for a 4056-wide sensor crop)")
    a = ap.parse_args()
    ar = cv2.aruco; dic = ar.getPredefinedDictionary(ar.DICT_4X4_100)
    board = ar.CharucoBoard((8, 11), a.square, a.marker, dic); det = ar.CharucoDetector(board)
    obj = {"L": [], "R": []}; img = {"L": [], "R": []}; ids = {"L": [], "R": []}; used = []
    for p in sorted(a.photos):
        im = cv2.imread(p); H, W = im.shape[:2]; halves = {"L": im[:, :W // 2], "R": im[:, W // 2:]}
        h, w = halves["L"].shape[:2]
        res = {}
        for e in ("L", "R"):
            cc, ci, mc, mi = det.detectBoard(cv2.cvtColor(halves[e], cv2.COLOR_BGR2GRAY))
            if cc is None or len(cc) < a.min_corners: res = None; break
            o, i = board.matchImagePoints(cc, ci); res[e] = (o.reshape(-1, 3), i.reshape(-1, 2), ci.ravel())
        if res is None: print(f"  skip {p} (board not found in both eyes)"); continue
        # keep only corners seen in BOTH eyes (needed for stereoCalibrate)
        common = np.intersect1d(res["L"][2], res["R"][2])
        for e in ("L", "R"):
            o, i, ci = res[e]; m = np.isin(ci, common)
            order = np.argsort(ci[m])            # identical id order in both eyes (stereoCalibrate pairs by index)
            obj[e].append(o[m][order].astype(np.float32)); img[e].append(i[m][order].astype(np.float32)); ids[e].append(ci[m][order])
        used.append(p); print(f"  {p}: {len(common)} shared corners, board spans x {res['L'][1][:,0].min():.0f}-{res['L'][1][:,0].max():.0f} y {res['L'][1][:,1].min():.0f}-{res['L'][1][:,1].max():.0f}")
    n = len(used); print(f"{n} stereo views usable, {sum(len(x) for x in obj['L'])} corner observations per eye")
    size = (w, h); f_nom = a.f_init if a.f_init else 4.806 / (6.2868 / 4056)
    print(f"initial focal guess {f_nom:.1f} px for a {w}x{h} eye")
    K0 = np.array([[f_nom, 0, w / 2], [0, f_nom, h / 2], [0, 0, 1]])
    cal = {}
    for e in ("L", "R"):
        flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3
        rms, K, d, rv, tv, sdi, sde, pve = cv2.calibrateCameraExtended(obj[e], img[e], size, K0.copy(), None, flags=flags)
        # focal observability: sigma of fx from the covariance
        print(f"eye {e}: rms {rms:.3f}px  fx {K[0,0]:.1f}±{sdi[0,0]:.1f} fy {K[1,1]:.1f}±{sdi[1,0]:.1f}  cx {K[0,2]:.1f}±{sdi[2,0]:.1f} cy {K[1,2]:.1f}±{sdi[3,0]:.1f}  "
              f"k1 {d[0,0]:+.4f} k2 {d[0,1]:+.4f} p1 {d[0,2]:+.5f} p2 {d[0,3]:+.5f}   per-view rms {np.round(pve.ravel(),2)}")
        cal[e] = (K, d, rms)
    KL, dL, _ = cal["L"]; KR, dR, _ = cal["R"]
    print(f"R/L focal ratio: fx {KR[0,0]/KL[0,0]:.4f}  fy {KR[1,1]/KL[1,1]:.4f}   principal point offset R-L: dx {KR[0,2]-KL[0,2]:+.1f} dy {KR[1,2]-KL[1,2]:+.1f} px")
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)
    rms_s, KL, dL, KR, dR, R, T, E, F = cv2.stereoCalibrate(obj["L"], img["L"], img["R"], KL, dL, KR, dR, size, flags=cv2.CALIB_FIX_INTRINSIC, criteria=crit)
    rvec = cv2.Rodrigues(R)[0].ravel()
    print(f"stereo: rms {rms_s:.3f}px  baseline |T| = {np.linalg.norm(T):.2f} mm  T = {T.ravel().round(2)} mm  "
          f"rotation {np.degrees(np.linalg.norm(rvec)):.3f} deg (rx {np.degrees(rvec[0]):+.3f} ry {np.degrees(rvec[1]):+.3f} rz {np.degrees(rvec[2]):+.3f})")
    # also a stereo solve with intrinsics free (checks consistency)
    rms_f, KL2, dL2, KR2, dR2, R2, T2, _, _ = cv2.stereoCalibrate(obj["L"], img["L"], img["R"], KL.copy(), dL.copy(), KR.copy(), dR.copy(), size, flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3, criteria=crit)
    print(f"stereo (intrinsics refined jointly): rms {rms_f:.3f}px  |T| {np.linalg.norm(T2):.2f} mm  fxL {KL2[0,0]:.1f} fxR {KR2[0,0]:.1f}  rot {np.degrees(np.linalg.norm(cv2.Rodrigues(R2)[0])):.3f} deg")
    # rectification check on the board corners
    R1, R2r, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(KL, dL, KR, dR, size, R, T, alpha=0)
    vy = []; dx = []
    for oL, iL, iR in zip(obj["L"], img["L"], img["R"]):
        pl = cv2.undistortPoints(iL.reshape(-1, 1, 2), KL, dL, R=R1, P=P1).reshape(-1, 2)
        pr = cv2.undistortPoints(iR.reshape(-1, 1, 2), KR, dR, R=R2r, P=P2).reshape(-1, 2)
        vy.append(pr[:, 1] - pl[:, 1]); dx.append(pl[:, 0] - pr[:, 0])
    vy = np.concatenate(vy); dx = np.concatenate(dx)
    print(f"after rectification: vertical disparity on board corners mean {vy.mean():+.2f} px, sd {vy.std():.2f} px, |max| {np.abs(vy).max():.2f}; "
          f"horizontal disparity range {dx.min():.1f}..{dx.max():.1f} px (all positive = consistent)")
    print(f"rectified focal {P1[0,0]:.1f} px, cx L {P1[0,2]:.1f} / R {P2[0,2]:.1f}  -> depth = {P1[0,0]:.0f} * {np.linalg.norm(T):.2f} / disparity  (mm)")
    np.savez(a.out, KL=KL, dL=dL, KR=KR, dR=dR, R=R, T=T, size=np.array(size), R1=R1, R2=R2r, P1=P1, P2=P2, Q=Q,
             rms_stereo=rms_s, square_mm=a.square, files=np.array(used))
    json.dump({"KL": KL.tolist(), "dL": dL.ravel().tolist(), "KR": KR.tolist(), "dR": dR.ravel().tolist(), "R": R.tolist(), "T_mm": T.ravel().tolist(),
               "size": size, "rms_stereo": rms_s, "square_mm": a.square, "n_views": n}, open(a.out.replace(".npz", ".json"), "w"), indent=1)
    print(f"wrote {a.out} / .json")

if __name__ == "__main__": main()
