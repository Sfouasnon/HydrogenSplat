#!/usr/bin/env python3
"""
monocolmap.py — COLMAP reconstruction for a camera *array*: one photograph per camera,
every camera the same body and lens (RED KOMODO-X rigs and the like), no stereo rig.

rigcolmap.py's sibling for the ``array`` source. Same three steps, same outputs, so
``hs solve`` and everything after it (train, move, render, views) read the result the
way they read a Hydrogen solve:

  prep    copy the frames into images/L/<camera>.jpg (the L folder is where every later
          stage looks for the reference view; an array has only reference views)
  sfm     SIFT -> exhaustive matching -> incremental mapping with ONE shared OPENCV camera
          (focal and distortion refined, principal point held at the centre: with 12 views
          of a mostly planar set, freeing it let the 2026-04-03 array solve collapse to two
          images), then an optional metric scale from a known camera-to-camera distance
  export  undistort to pinhole, write the COLMAP training set and a mono rig.npz
          (``stereo = False``; see hs/rig.py)

Scale. Photogrammetry from unknown cameras has no unit; the Hydrogen's calibrated
baseline gives its solves millimetres for free, an array has to be told. ``--scale-pair
GA,GB,MM`` sets the unit from the measured distance between two camera centres; ``--scale
S`` multiplies by a known factor. Without either the reconstruction is exported as it
comes and ``hs solve`` fails the ``scene_scaled`` check so nobody mistakes it for metric.
"""
import argparse, json, os, shutil, sys
import numpy as np

try:
    import pycolmap
except ImportError:
    sys.exit("pip install pycolmap")

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


# ---------------------------------------------------------------- prep

def cmd_prep(a):
    import cv2
    src = sorted(f for f in os.listdir(a.frames) if f.lower().endswith(IMAGE_EXTS))
    if not src:
        sys.exit(f"no images in {a.frames}")
    d = os.path.join(a.out, "images", "L")
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(d):
        os.remove(os.path.join(d, f))
    captures, sizes = [], set()
    for f in src:
        im = cv2.imread(os.path.join(a.frames, f), cv2.IMREAD_COLOR)
        if im is None:
            print(f"  skip unreadable {f}")
            continue
        cam = os.path.splitext(f)[0]
        cv2.imwrite(os.path.join(d, cam + ".jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        sizes.add((im.shape[1], im.shape[0]))
        captures.append({"capture": cam, "source": f})
    if len(sizes) != 1:
        sys.exit(f"frames differ in size: {sorted(sizes)} — one camera model needs one size")
    (w, h), = sizes
    json.dump({"captures": captures, "eye_size": [w, h], "stereo": False},
              open(os.path.join(a.out, "captures.json"), "w"), indent=1)
    print(f"{len(captures)} cameras -> {a.out}/images/L  ({w}x{h})")


# ---------------------------------------------------------------- sfm

def cmd_sfm(a):
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
    n_img = len(os.listdir(os.path.join(imgs, "L")))

    if not os.path.exists(db):
        fe = pycolmap.FeatureExtractionOptions()
        fe.sift.max_num_features = a.features
        fe.sift.peak_threshold = a.peak_threshold
        ro = pycolmap.ImageReaderOptions()
        ro.camera_model = "OPENCV"
        if a.focal_px:
            # pycolmap takes the prior as "fx,fy,cx,cy,k1,k2,p1,p2"; only the focal matters
            cap = json.load(open(os.path.join(work, "captures.json")))
            w, h = cap["eye_size"]
            ro.camera_params = f"{a.focal_px},{a.focal_px},{w / 2},{h / 2},0,0,0,0"
        if a.masks:
            ro.mask_path = a.masks
        print(f"extracting SIFT from {n_img} images "
              f"(max {a.features} features, peak {a.peak_threshold}, CPU)...")
        pycolmap.extract_features(db, imgs, camera_mode=pycolmap.CameraMode.SINGLE,
                                  reader_options=ro, extraction_options=fe)
        dbh = pycolmap.Database.open(db)
        nk = [dbh.num_keypoints_for_image(im.image_id) for im in dbh.read_all_images()]
        dbh.close()
        print(f"  features per image: min {min(nk)}, median {int(np.median(nk))}, max {max(nk)}")
        print(f"exhaustive matching {n_img} images ({n_img * (n_img - 1) // 2} pairs)...")
        pycolmap.match_exhaustive(db)
    else:
        print(f"reusing features/matches in {db}")

    o = pycolmap.IncrementalPipelineOptions()
    o.ba_refine_focal_length = not a.fix_intrinsics
    o.ba_refine_extra_params = not a.fix_intrinsics
    o.ba_refine_principal_point = a.refine_principal_point
    o.mapper.init_min_tri_angle = a.init_min_tri_angle
    o.mapper.filter_min_tri_angle = a.min_tri_angle
    o.triangulation.min_angle = a.min_tri_angle
    o.min_model_size = 3
    o.multiple_models = False
    print(f"mapping (one shared camera; focal/distortion "
          f"{'fixed' if a.fix_intrinsics else 'refined'}, principal point "
          f"{'refined' if a.refine_principal_point else 'fixed at centre'})...")
    recs = pycolmap.incremental_mapping(db, imgs, sparse, options=o)
    if not recs:
        sys.exit("mapping produced no reconstruction")
    best = max(recs.values(), key=lambda r: r.num_reg_images())

    scale, how = 1.0, None
    if a.scale_pair:
        n1, n2, mm = a.scale_pair.split(",")
        byname = {os.path.splitext(os.path.basename(im.name))[0]: im
                  for im in best.images.values() if im.has_pose}
        for n in (n1, n2):
            if n not in byname:
                sys.exit(f"--scale-pair: camera {n!r} is not registered ({sorted(byname)})")
        d = float(np.linalg.norm(np.asarray(byname[n1].projection_center())
                                 - np.asarray(byname[n2].projection_center())))
        scale = float(mm) / 1000.0 / d
        how = f"{n1}-{n2} = {float(mm):.1f} mm, was {d:.4f} units"
    elif a.scale:
        scale, how = a.scale, f"given factor {a.scale}"
    if how:
        best.transform(pycolmap.Sim3d(scale, pycolmap.Rotation3d(), np.zeros(3)))
        print(f"  scaled to metres: x{scale:.6f} ({how})")
    else:
        print("  NOT scaled: no --scale-pair / --scale; units are arbitrary")

    report(best, work, scale if how else None)
    out = os.path.join(sparse, "rig")     # the folder name every later stage expects
    os.makedirs(out, exist_ok=True)
    best.write(out)
    best.export_PLY(os.path.join(out, "points3D.ply"))
    print(f"wrote {out}")


def report(rec, work, scale):
    errs = np.array([p.error for p in rec.points3D.values() if p.error >= 0])
    print(f"\n  reconstruction: {rec.num_reg_images()} images in {rec.num_reg_images()} posed frames, "
          f"{rec.num_points3D()} points")
    if len(errs):
        print(f"  reprojection error: mean {errs.mean():.3f} px, median {np.median(errs):.3f} px")
    tl = np.array([p.track.length() for p in rec.points3D.values()], float)
    if len(tl):
        print(f"  track length: mean {tl.mean():.2f}, max {int(tl.max())}")
    cam = list(rec.cameras.values())[0]
    p = cam.params
    print(f"  camera: {cam.model.name} {cam.width}x{cam.height} fx {p[0]:.1f} fy {p[1]:.1f} "
          f"cx {p[2]:.1f} cy {p[3]:.1f} dist {np.round(p[4:], 4).tolist()}")
    ref = {im.name: np.asarray(im.projection_center()) for im in rec.images.values() if im.has_pose}
    C = np.array([ref[k] for k in sorted(ref)])
    unit = "mm" if scale else "units"
    k = 1000.0 if scale else 1.0
    if len(C) > 1:
        step = np.linalg.norm(np.diff(C, axis=0), axis=1) * k
        print(f"  step between consecutive cameras: {step.min():.0f}-{step.max():.0f} {unit} "
              f"(median {np.median(step):.0f}), path length {step.sum():.0f} {unit}")
    P = np.array([p.xyz for p in rec.points3D.values()])
    if len(P) and len(C):
        d = np.linalg.norm(P - C.mean(axis=0), axis=1) * k
        print(f"  scene depth from array centre: p5 {np.percentile(d, 5):.0f} {unit}, "
              f"median {np.median(d):.0f} {unit}, p95 {np.percentile(d, 95):.0f} {unit}")
    json.dump({"num_images": rec.num_reg_images(), "num_frames": rec.num_reg_images(),
               "num_points": rec.num_points3D(),
               "mean_reproj_px": float(errs.mean()) if len(errs) else None,
               "scale_to_m": scale, "camera": {"model": cam.model.name, "width": cam.width,
                                               "height": cam.height, "params": [float(x) for x in p]}},
              open(os.path.join(work, "sfm_report.json"), "w"), indent=1)


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
    r2 = pycolmap.Reconstruction(os.path.join(a.out, "sparse"))
    finish_dataset(r2, a.out)


def finish_dataset(rec, out, write_binary=False):
    """The training set's sparse model as text + points3D.ply, and its rig.npz. Export calls
    this on the undistorted model; ``hs scale`` calls it again after applying the board's
    Sim3d to that same model (``write_binary=True``: the undistorter's .bin files must follow),
    so a board-scaled dataset is written by exactly the code that writes a --scale one."""
    sp = os.path.join(out, "sparse")
    if write_binary:
        rec.write(sp)
    rec.export_PLY(os.path.join(sp, "points3D.ply"))
    rec.write_text(sp)
    print(f"{rec.num_reg_images()} images, {rec.num_points3D()} points; "
          f"points3D.ply written for the path builder")
    write_rig_npz(rec, os.path.join(out, "rig.npz"))


def write_rig_npz(rec, path):
    """The mono layout of rig.npz (hs/rig.py): one view per camera, named ``<cam>_L``,
    ``stereo = False``; lengths in mm like the stereo file."""
    byname = {im.name: im for im in rec.images.values() if im.has_pose}
    names, K, R, t, C, WH, photos = [], [], [], [], [], [], []
    for name in sorted(byname):
        im = byname[name]
        cam = rec.camera(im.camera_id)
        p = cam.params
        stem = os.path.splitext(os.path.basename(name))[0]
        names.append(stem + "_L")
        photos.append(stem)
        K.append([[p[0], 0.0, p[2]], [0.0, p[1], p[3]], [0.0, 0.0, 1.0]])
        WH.append([int(cam.width), int(cam.height)])
        w2c = im.cam_from_world()
        R.append(w2c.rotation.matrix())
        t.append(np.asarray(w2c.translation) * 1000.0)
        C.append(np.asarray(im.projection_center()) * 1000.0)
    pts = np.array([p.xyz for p in rec.points3D.values()]) * 1000.0
    w, h = (WH[0] if WH else [0, 0])
    np.savez(path, names=np.array(names), K=np.array(K), R=np.array(R),
             t=np.array(t), C=np.array(C), pts=pts,
             wh=np.array(WH, int).reshape(-1, 2), w=w, h=h, s_mm=1.0,
             photos=np.array(photos), stereo=False)
    print(f"wrote {path}: {len(names)} cameras (mono) for the path builders")


# ---------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prep")
    p.add_argument("frames", help="directory with one frame per camera, named by camera")
    p.add_argument("-o", "--out", required=True, help="work directory")
    p.set_defaults(fn=cmd_prep)

    p = sub.add_parser("sfm")
    p.add_argument("work")
    p.add_argument("--features", type=int, default=16384)
    p.add_argument("--peak-threshold", type=float, default=0.0025)
    p.add_argument("--masks", default=None, help="directory of <image>.png masks mirroring images/")
    p.add_argument("--focal-px", type=float, default=None,
                   help="focal length prior in pixels (lens mm / sensor width mm x image width)")
    p.add_argument("--fix-intrinsics", action="store_true", help="keep the prior; do not refine focal/distortion")
    p.add_argument("--refine-principal-point", action="store_true",
                   help="NOT recommended on small arrays: the 12-view 2026-04-03 solve collapsed with it")
    p.add_argument("--scale-pair", default=None, metavar="CAM1,CAM2,MM",
                   help="metric scale from the measured distance between two camera centres")
    p.add_argument("--scale", type=float, default=None, help="multiply the reconstruction by this to get metres")
    p.add_argument("--init-min-tri-angle", type=float, default=8.0)
    p.add_argument("--min-tri-angle", type=float, default=0.8)
    p.add_argument("--fresh", action="store_true", help="rebuild features and matches")
    p.set_defaults(fn=cmd_sfm)

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
