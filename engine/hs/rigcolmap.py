#!/usr/bin/env python3
"""
rigcolmap.py — rig-aware COLMAP reconstruction for RED Hydrogen One stereo capture.

Replaces the old mv/rigsolve.py architecture (left-eye-only incremental SfM, right eyes
attached afterwards through the calibration, metric scale recovered by a post-hoc linear
fit). Here both eyes are in the reconstruction from the start, COLMAP's native rig/frame
model ties them with one shared sensor_from_rig, and because that transform is given in
metres the reconstruction is metric by construction — there is no scale to fit.

A 55-capture clip becomes 55 frame poses + 1 rig transform, not 110 free cameras.

  prep    split _2x1 side-by-side frames into images/L and images/R (and optionally
          apply a per-capture convergence shift to the right eye)
  sfm     features -> matches -> rig config -> rig-aware incremental mapping
  conv    measure the per-capture right-eye convergence offset against a reconstruction
  export  undistort to pinhole and write a Brush/COLMAP training dataset

Subcommands are separate because `conv` needs a reconstruction to measure against, so the
convergence-corrected run is necessarily a second pass:

    prep -> sfm -> conv -> prep --shift -> sfm -> export

Runs anywhere pycolmap installs (pip install pycolmap; wheels exist for macOS arm64
including cp314). Feature extraction is CPU SIFT — there is no CUDA on Apple Silicon —
which is a few seconds per 1920x1080 image.
"""
import argparse, json, os, shutil, sys
import numpy as np

try:
    import pycolmap
except ImportError:
    sys.exit("pip install pycolmap")


# ---------------------------------------------------------------- helpers

def quat_wxyz(R):
    """Rotation matrix -> [w,x,y,z], which is the order COLMAP's rig json expects."""
    R = np.asarray(R, float)
    t = R.trace()
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0.0, 0.0, 0.0]
        q[i] = 0.25 * s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        w = (R[k, j] - R[j, k]) / s
        x, y, z = q
    q = np.array([w, x, y, z])
    return (q / np.linalg.norm(q)).tolist()


def load_calib(path):
    """stereocal.py npz -> dict. T is mm in that file; COLMAP gets metres."""
    z = np.load(path, allow_pickle=True)
    return {
        "KL": z["KL"].astype(float), "dL": z["dL"].astype(float).ravel(),
        "KR": z["KR"].astype(float), "dR": z["dR"].astype(float).ravel(),
        "R": z["R"].astype(float), "T_mm": z["T"].astype(float).ravel(),
        "size": tuple(int(v) for v in z["size"]),
    }


def opencv_params(K, d):
    """COLMAP OPENCV model: fx fy cx cy k1 k2 p1 p2."""
    d = np.asarray(d, float).ravel()
    return [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]),
            float(d[0]), float(d[1]), float(d[2] if len(d) > 2 else 0.0),
            float(d[3] if len(d) > 3 else 0.0)]


def rig_config(calib, scale_baseline=1.0):
    """One rig: L is the reference sensor, R sits at the calibrated sensor_from_rig.

    The calibrated intrinsics always go in, whether or not BA is allowed to refine them
    afterwards — without them COLMAP falls back to a default focal guess (2304 px here
    against a true 1704), which is enough to stop registration dead.

    Because sensor_from_rig is metres, the whole reconstruction is metres. scale_baseline
    exists to test the 10.64-vs-13.64 mm calibration disagreement without recalibrating —
    though see the note in cmd_sfm: scale is pure gauge here, so it rescales the result
    rather than telling you which baseline is right."""
    T = calib["T_mm"] * scale_baseline / 1000.0
    return [{"cameras": [
        {"image_prefix": "L/", "ref_sensor": True,
         "camera_model_name": "OPENCV",
         "camera_params": opencv_params(calib["KL"], calib["dL"])},
        {"image_prefix": "R/",
         "camera_model_name": "OPENCV",
         "camera_params": opencv_params(calib["KR"], calib["dR"]),
         "cam_from_rig_rotation": quat_wxyz(calib["R"]),
         "cam_from_rig_translation": [float(v) for v in T]},
    ]}]


# ---------------------------------------------------------------- prep

def cmd_prep(a):
    import cv2
    src = sorted([f for f in os.listdir(a.frames)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if a.stride > 1:
        src = src[::a.stride]
    if not src:
        sys.exit(f"no images in {a.frames}")

    shifts = {}
    if a.shift:
        shifts = {k: v for k, v in json.load(open(a.shift))["shift"].items()}
        print(f"applying convergence shifts from {a.shift} "
              f"({len(shifts)} captures, "
              f"dx {min(v[0] for v in shifts.values()):+.1f}..{max(v[0] for v in shifts.values()):+.1f} px)")

    for sub in ("L", "R"):
        d = os.path.join(a.out, "images", sub)
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))

    captures = []
    for i, f in enumerate(src):
        im = cv2.imread(os.path.join(a.frames, f), cv2.IMREAD_COLOR)
        if im is None:
            print(f"  skip unreadable {f}")
            continue
        h, w = im.shape[:2]
        if w % 2:
            w -= 1
        half = w // 2
        L, Rg = im[:, :half], im[:, half:w]
        cap = f"cap{i:03d}"
        if cap in shifts:
            dx, dy = shifts[cap]
            # Integer translation only: a convergence shift is a 2D image translation to
            # well under a pixel at these magnitudes, and an integer roll resamples nothing.
            M = np.float32([[1, 0, -round(dx)], [0, 1, -round(dy)]])
            Rg = cv2.warpAffine(Rg, M, (half, h), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        q = [cv2.IMWRITE_JPEG_QUALITY, 95]
        cv2.imwrite(os.path.join(a.out, "images", "L", cap + ".jpg"), L, q)
        cv2.imwrite(os.path.join(a.out, "images", "R", cap + ".jpg"), Rg, q)
        captures.append({"capture": cap, "source": f,
                         "shift": shifts.get(cap, [0.0, 0.0])})

    json.dump({"captures": captures, "eye_size": [half, h]},
              open(os.path.join(a.out, "captures.json"), "w"), indent=1)
    print(f"{len(captures)} captures -> {a.out}/images/{{L,R}}  ({half}x{h} per eye)")


# ---------------------------------------------------------------- sfm

def cmd_sfm(a):
    calib = load_calib(a.calib)
    work = a.work
    db = os.path.join(work, "database.db")
    imgs = os.path.join(work, "images")
    sparse = os.path.join(work, "sparse")
    if not os.path.isdir(imgs):
        sys.exit(f"{imgs} missing — run prep first")

    if a.fresh and os.path.exists(db):
        os.remove(db)
    if os.path.isdir(sparse):
        shutil.rmtree(sparse)
    os.makedirs(sparse, exist_ok=True)

    n_img = sum(len(os.listdir(os.path.join(imgs, s))) for s in ("L", "R"))

    if not os.path.exists(db):
        fe = pycolmap.FeatureExtractionOptions()
        fe.sift.max_num_features = a.features
        fe.sift.peak_threshold = a.peak_threshold
        fe.sift.estimate_affine_shape = a.affine
        fe.sift.domain_size_pooling = a.affine
        ro = pycolmap.ImageReaderOptions()
        if a.masks:
            ro.mask_path = a.masks
        print(f"extracting SIFT from {n_img} images "
              f"(max {a.features} features, peak {a.peak_threshold}, CPU)...")
        pycolmap.extract_features(db, imgs,
                                  camera_mode=pycolmap.CameraMode.PER_FOLDER,
                                  reader_options=ro, extraction_options=fe)
        dbh = pycolmap.Database.open(db)
        nk = [dbh.num_keypoints_for_image(im.image_id) for im in dbh.read_all_images()]
        dbh.close()
        print(f"  features per image: min {min(nk)}, median {int(np.median(nk))}, max {max(nk)}")

        fm = pycolmap.FeatureMatchingOptions()
        fm.guided_matching = a.guided
        print(f"exhaustive matching {n_img} images ({n_img*(n_img-1)//2} pairs)...")
        pycolmap.match_exhaustive(db, matching_options=fm)
    else:
        print(f"reusing features/matches in {db}")

    cfg_path = os.path.join(work, "rig_config.json")
    cfg = rig_config(calib, a.baseline_scale)
    json.dump(cfg, open(cfg_path, "w"), indent=1)
    rigs = pycolmap.read_rig_config(cfg_path)
    dbh = pycolmap.Database.open(db)
    pycolmap.apply_rig_config(rigs, dbh)
    print(f"rig applied: {dbh.num_rigs()} rig(s), {dbh.num_frames()} frames, "
          f"{dbh.num_cameras()} cameras, baseline "
          f"{np.linalg.norm(calib['T_mm'])*a.baseline_scale:.2f} mm")
    orig_rigs = None
    if a.float_rig:
        orig_rigs = pycolmap.RigMap()
        for r in dbh.read_all_rigs():
            orig_rigs[r.rig_id] = r
    dbh.close()

    o = pycolmap.IncrementalPipelineOptions()
    o.ba_refine_focal_length = a.refine_intrinsics
    o.ba_refine_principal_point = a.refine_intrinsics
    o.ba_refine_extra_params = a.refine_intrinsics
    o.ba_refine_sensor_from_rig = a.float_rig
    o.mapper.init_min_tri_angle = a.init_min_tri_angle
    o.mapper.filter_min_tri_angle = a.min_tri_angle
    o.triangulation.min_angle = a.min_tri_angle
    o.min_model_size = 3
    o.multiple_models = False
    print(f"mapping (rig {'refined then rescaled' if a.float_rig else 'fixed'}, "
          f"intrinsics {'refined' if a.refine_intrinsics else 'fixed'})...")
    recs = pycolmap.incremental_mapping(db, imgs, sparse, options=o)
    if not recs:
        sys.exit("mapping produced no reconstruction")

    best = max(recs.values(), key=lambda r: r.num_reg_images())
    if a.float_rig:
        for rig in best.rigs.values():
            for sid in rig.non_ref_sensors:
                print(f"  BA-refined sensor_from_rig before rescale: "
                      f"|t| {np.linalg.norm(rig.sensor_from_rig(sid).translation)*1000:.3f}, "
                      f"rotation vs calibrated "
                      f"{np.degrees(np.arccos(np.clip((np.trace(rig.sensor_from_rig(sid).rotation.matrix() @ calib['R'].T)-1)/2,-1,1))):.4f} deg")
        ok = pycolmap.align_reconstruction_to_orig_rig_scales(orig_rigs, best)
        print(f"rescaled to the calibrated rig scale: {ok}")

    report(best, calib, work)
    out = os.path.join(sparse, "rig")
    os.makedirs(out, exist_ok=True)
    best.write(out)
    best.export_PLY(os.path.join(out, "points3D.ply"))
    print(f"wrote {out}")


def report(rec, calib, work):
    nframes = sum(1 for f in rec.frames.values() if f.has_pose())
    errs = []
    for p in rec.points3D.values():
        errs.append(p.error)
    errs = np.array([e for e in errs if e >= 0])
    print(f"\n  reconstruction: {rec.num_reg_images()} images in {nframes} posed frames, "
          f"{rec.num_points3D()} points")
    if len(errs):
        print(f"  reprojection error: mean {errs.mean():.3f} px, median {np.median(errs):.3f} px")
    tl = np.array([p.track.length() for p in rec.points3D.values()], float)
    print(f"  track length: mean {tl.mean():.2f}, max {int(tl.max())}")

    # per-eye error: if the right eye is much worse, the right-eye model is still wrong
    per = {}
    for im in rec.images.values():
        if not im.has_pose:
            continue
        o, x = [], []
        for p2 in im.points2D:
            if p2.has_point3D():
                o.append(p2.xy); x.append(rec.point3D(p2.point3D_id).xyz)
        if not o:
            continue
        loc = im.cam_from_world() * np.asarray(x, float)
        ok = loc[:, 2] > 0
        if ok.sum() == 0:
            continue
        pr = np.asarray(rec.camera(im.camera_id).img_from_cam(loc[ok]), float)
        per.setdefault(im.name.split("/")[0], []).append(
            np.linalg.norm(np.asarray(o)[ok] - pr, axis=1))
    for eye in sorted(per):
        e = np.concatenate(per[eye])
        print(f"  eye {eye}: {len(e)} observations, rms {np.sqrt((e**2).mean()):.3f} px, "
              f"median {np.median(e):.3f} px")

    for rig in rec.rigs.values():
        for sid in rig.non_ref_sensors:
            T = rig.sensor_from_rig(sid).translation
            print(f"  rig sensor_from_rig: |t| = {np.linalg.norm(T)*1000:.3f} mm "
                  f"(calibrated {np.linalg.norm(calib['T_mm']):.3f} mm)")

    # order by capture name, not by frame id — frame ids come out in registration order,
    # which made "frame spacing" read 27-262 mm on a path whose real steps are 9-72 mm
    ref = {}
    for im in rec.images.values():
        if im.has_pose and im.name.startswith("L/"):
            ref[im.name] = np.asarray(im.projection_center())
    C = np.array([ref[k] for k in sorted(ref)])
    if len(C) > 1:
        step = np.linalg.norm(np.diff(C, axis=0), axis=1) * 1000
        print(f"  step between consecutive captures: {step.min():.0f}-{step.max():.0f} mm "
              f"(median {np.median(step):.0f}), path length {step.sum():.0f} mm")
        print(f"  path extent: {np.round(np.ptp(C, axis=0)*1000, 0)} mm")
    P = np.array([p.xyz for p in rec.points3D.values()])
    if len(P):
        d = np.linalg.norm(P - C.mean(axis=0), axis=1) * 1000
        print(f"  scene depth from path centre: p5 {np.percentile(d,5):.0f} mm, "
              f"median {np.median(d):.0f} mm, p95 {np.percentile(d,95):.0f} mm")
    json.dump({"num_images": rec.num_reg_images(), "num_frames": int(nframes),
               "num_points": rec.num_points3D(),
               "mean_reproj_px": float(errs.mean()) if len(errs) else None},
              open(os.path.join(work, "sfm_report.json"), "w"), indent=1)


# ---------------------------------------------------------------- conv

def cmd_conv(a):
    """Measure each capture's right-eye convergence offset.

    The right eye's offset is unobservable from one stereo pair — a pure horizontal shift
    preserves the epipolar constraint exactly — so it has to be measured against structure
    that the left eyes established over time. With the rig baseline fixed in metres the
    scale is already known, which makes this a per-capture 2-parameter fit rather than the
    old ill-conditioned joint fit over a shared unknown scale.
    """
    rec = pycolmap.Reconstruction(a.recon)
    caps = json.load(open(os.path.join(a.work, "captures.json")))["captures"]
    prev = {c["capture"]: c["shift"] for c in caps}

    byname = {im.name: im for im in rec.images.values()}
    out, rows = {}, []
    for c in caps:
        cap = c["capture"]
        iL, iR = byname.get("L/" + cap + ".jpg"), byname.get("R/" + cap + ".jpg")
        if iR is None or not iR.has_pose:
            continue
        cam = rec.camera(iR.camera_id)
        obs, xyz = [], []
        for p2 in iR.points2D:
            if p2.has_point3D():
                obs.append(p2.xy)
                xyz.append(rec.point3D(p2.point3D_id).xyz)
        if len(obs) < a.min_obs:
            continue
        obs = np.asarray(obs, float)
        loc = iR.cam_from_world() * np.asarray(xyz, float)
        ok = loc[:, 2] > 0
        if ok.sum() < a.min_obs:
            continue
        pred = cam.img_from_cam(loc[ok])
        res = obs[ok] - np.asarray(pred, float)
        med = np.median(res, axis=0)
        # robust: one reweighted pass against the median
        keep = np.linalg.norm(res - med, axis=1) < max(3.0, 3 * np.median(
            np.linalg.norm(res - med, axis=1)))
        med = np.median(res[keep], axis=0)
        p = prev.get(cap, [0.0, 0.0])
        out[cap] = [float(p[0] + med[0]), float(p[1] + med[1])]
        rows.append((cap, len(obs), int(keep.sum()), med[0], med[1],
                     np.std(res[keep, 0]), np.std(res[keep, 1])))

    if not rows:
        sys.exit("no capture had enough right-eye observations")
    print(f"{'capture':9s} {'obs':>5s} {'kept':>5s} {'dx':>8s} {'dy':>8s} {'sdx':>6s} {'sdy':>6s}")
    for r in rows:
        print(f"{r[0]:9s} {r[1]:5d} {r[2]:5d} {r[3]:+8.2f} {r[4]:+8.2f} {r[5]:6.2f} {r[6]:6.2f}")
    dx = np.array([r[3] for r in rows]); dy = np.array([r[4] for r in rows])
    print(f"\nresidual right-eye offset: dx {dx.min():+.2f}..{dx.max():+.2f} px "
          f"(range {np.ptp(dx):.2f}), dy {dy.min():+.2f}..{dy.max():+.2f} px "
          f"(range {np.ptp(dy):.2f})")
    print("a shared-camera rig can absorb the mean, not the spread — the spread is what "
          "the pre-warp removes")
    p = os.path.join(a.work, "conv_shift.json")
    json.dump({"note": "cumulative right-eye shift in px to subtract at prep time",
               "shift": out}, open(p, "w"), indent=1)
    print(f"wrote {p} — rerun prep with --shift {p}, then sfm")


# ---------------------------------------------------------------- export

def cmd_export(a):
    rec = pycolmap.Reconstruction(a.recon)
    os.makedirs(a.out, exist_ok=True)
    print(f"undistorting {rec.num_reg_images()} images to pinhole -> {a.out}")
    opts = pycolmap.UndistortCameraOptions()
    if a.max_size:
        opts.max_image_size = a.max_size
    pycolmap.undistort_images(a.out, a.recon, a.images,
                              output_type="COLMAP", undistort_options=opts,
                              jpeg_quality=a.jpeg_quality)
    sp = os.path.join(a.out, "sparse")
    r2 = pycolmap.Reconstruction(sp)
    r2.export_PLY(os.path.join(sp, "points3D.ply"))
    r2.write_text(sp)
    print(f"{r2.num_reg_images()} images, {r2.num_points3D()} points; "
          f"points3D.ply written for the path builder")
    write_rig_npz(r2, os.path.join(a.out, "rig.npz"))


def write_rig_npz(rec, path):
    """Emit the rig.npz layout the old solver produced, so arc_path.py / spline_path.py
    keep working against a COLMAP reconstruction without changes: views interleaved
    L,R,L,R in capture order, lengths in mm."""
    byname = {im.name: im for im in rec.images.values() if im.has_pose}
    caps = sorted({n.split("/", 1)[1] for n in byname})
    names, K, R, t, C = [], [], [], [], []
    for cap in caps:
        pair = [byname.get(e + "/" + cap) for e in ("L", "R")]
        if any(p is None for p in pair):
            continue
        for im in pair:
            cam = rec.camera(im.camera_id)
            p = cam.params
            names.append(os.path.splitext(cap)[0] + "_" + im.name[0])
            K.append([[p[0], 0.0, p[2]], [0.0, p[1], p[3]], [0.0, 0.0, 1.0]])
            w2c = im.cam_from_world()
            R.append(w2c.rotation.matrix())
            t.append(np.asarray(w2c.translation) * 1000.0)
            C.append(np.asarray(im.projection_center()) * 1000.0)
    pts = np.array([p.xyz for p in rec.points3D.values()]) * 1000.0
    cam0 = rec.camera(rec.images[list(rec.images)[0]].camera_id)
    np.savez(path, names=np.array(names), K=np.array(K), R=np.array(R),
             t=np.array(t), C=np.array(C), pts=pts,
             w=cam0.width, h=cam0.height, s_mm=1.0,
             photos=np.array([os.path.splitext(c)[0] for c in caps]))
    print(f"wrote {path}: {len(names)//2} captures interleaved L,R for the path builders")


# ---------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prep")
    p.add_argument("frames", help="directory of _2x1 side-by-side frames")
    p.add_argument("-o", "--out", required=True, help="work directory")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--shift", default=None, help="conv_shift.json from the conv step")
    p.set_defaults(fn=cmd_prep)

    p = sub.add_parser("sfm")
    p.add_argument("work")
    p.add_argument("--calib", required=True, help="stereocal npz")
    p.add_argument("--features", type=int, default=16384)
    p.add_argument("--peak-threshold", type=float, default=0.0025,
                   help="SIFT DoG peak threshold; COLMAP's 0.0067 default finds only ~1700 "
                        "features on these video frames, which starves the init cloud")
    p.add_argument("--affine", action="store_true",
                   help="affine-shape + DSP SIFT: slower, more robust across the baseline")
    p.add_argument("--guided", action="store_true",
                   help="guided matching: second matching pass constrained by the two-view geometry")
    p.add_argument("--masks", default=None,
                   help="directory of <image>.png masks mirroring images/ (black = ignore)")
    p.add_argument("--baseline-scale", type=float, default=1.0,
                   help="multiply the calibrated baseline (test 1.28 for the 13.64 mm fit)")
    p.add_argument("--refine-intrinsics", action="store_true")
    p.add_argument("--float-rig", action="store_true",
                   help="let BA refine sensor_from_rig, then restore the calibrated scale")
    p.add_argument("--init-min-tri-angle", type=float, default=8.0)
    p.add_argument("--min-tri-angle", type=float, default=0.8)
    p.add_argument("--fresh", action="store_true", help="rebuild features and matches")
    p.set_defaults(fn=cmd_sfm)

    p = sub.add_parser("conv")
    p.add_argument("work")
    p.add_argument("--recon", required=True)
    p.add_argument("--min-obs", type=int, default=40)
    p.set_defaults(fn=cmd_conv)

    p = sub.add_parser("export")
    p.add_argument("recon")
    p.add_argument("--images", required=True)
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--max-size", type=int, default=0)
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.set_defaults(fn=cmd_export)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
