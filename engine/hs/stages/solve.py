"""hs solve — rigcolmap.py prep → sfm --float-rig → export, plus per_image.json and
coverage.json (strategy §4.4). The calibrated intrinsics from the profile go into the COLMAP
rig config through the same npz path the by-hand run used; --refine-intrinsics is not
exposed (it collapses the reconstruction without --float-rig and buys ~0.03 px with it).

Checks: registered frames == selected frames; mean reprojection ≤ 1.8 px; no image > 3 px;
L–R separation equals the profile baseline. A partial registration fails the stage with the
remedies in order (re-select with --max-gap 45; trim the tail with --end) — never silently
proceed to train on a partial solve.
"""
import json
import os
import re
import shutil
import sys

import numpy as np

from .. import calib, coverage, events, runner

STAGE = "solve"

RE_FEAT = re.compile(r"Processed file \[(\d+)/(\d+)\]")
RE_MATCH = re.compile(r"Processing block \[(\d+)/(\d+), (\d+)/(\d+)\]")
RE_REG = re.compile(r"Registering image #(\d+) \((\d+)\)")
RE_FEATSTAT = re.compile(r"features per image: min (\d+), median (\d+), max (\d+)")
RE_RECON = re.compile(r"reconstruction: (\d+) images in (\d+) posed frames, (\d+) points")
RE_REPROJ = re.compile(r"reprojection error: mean ([\d.]+) px, median ([\d.]+) px")
RE_EYE = re.compile(r"eye (\w): (\d+) observations, rms ([\d.]+) px, median ([\d.]+) px")
RE_RIG = re.compile(r"rig sensor_from_rig: \|t\| = ([\d.]+) mm \(calibrated ([\d.]+) mm\)")
RE_BA = re.compile(r"BA-refined sensor_from_rig before rescale: \|t\| ([\d.]+), rotation vs calibrated ([\d.]+) deg")
RE_STEP = re.compile(r"step between consecutive captures: (\d+)-(\d+) mm \(median (\d+)\), path length (\d+) mm")
RE_DEPTH = re.compile(r"scene depth from path centre: p5 (\d+) mm, median (\d+) mm, p95 (\d+) mm")
RE_UNDIST = re.compile(r"Undistorting image \[(\d+)/(\d+)\]")


def add_parser(sub):
    p = sub.add_parser("solve", help="split eyes, rig-aware COLMAP SfM, export the training set (rigcolmap.py)")
    p.add_argument("--profile", default=None, help="override the profile chosen at ingest")
    p.add_argument("--peak-threshold", type=float, default=0.0025)
    p.add_argument("--features", type=int, default=16384)
    p.add_argument("--masks", default=None, help="mask dir mirroring images/ (wired, unverified)")
    p.add_argument("--no-float-rig", action="store_true", help="NOT recommended: keep sensor_from_rig fixed")
    p.add_argument("--reuse-matches", action="store_true", help="keep an existing database.db (skip features/matching)")
    p.add_argument("--max-reproj", type=float, default=1.8, help="check threshold, px")
    return p


def run(a, pj):
    pj.require(STAGE)
    frames = pj.frames_dir
    n_sel = len([f for f in os.listdir(frames) if f.lower().endswith(".jpg")]) if os.path.isdir(frames) else 0
    if n_sel == 0:
        raise events.StageError("no selected frames", hint="hs select --project ...")
    prof_ref = a.profile or pj.m.get("profile_path") or pj.m.get("profile_id")
    if not prof_ref:
        raise events.StageError("project has no calibration profile", hint="re-run hs ingest")
    prof_path, prof = calib.load_profile(prof_ref)
    work = pj.stage_dir(STAGE)
    keep_db = None
    if a.reuse_matches and os.path.exists(os.path.join(work, "database.db")):
        keep_db = os.path.join(pj.root, "database.db.keep")
        shutil.move(os.path.join(work, "database.db"), keep_db)
    pj.begin(STAGE, argv=sys.argv)
    if keep_db:
        shutil.move(keep_db, os.path.join(work, "database.db"))
    # the train dataset is written by export; it belongs to train/ but is produced here
    if os.path.isdir(pj.dataset_dir):
        shutil.rmtree(pj.dataset_dir)
    os.makedirs(pj.path("train"), exist_ok=True)

    calib_npz = calib.profile_to_npz(prof, os.path.join(work, "calib.npz"))
    json.dump(prof, open(os.path.join(work, "profile.json"), "w"), indent=1)
    pj.m["profile_id"] = prof.get("profile_id")
    pj.m["profile_path"] = prof_path
    baseline = calib.baseline_mm(prof)
    pj.metric(STAGE, "profile_baseline_mm", round(baseline, 4))
    log = pj.log_path(STAGE)

    # ---- prep
    events.start(STAGE, "prep")
    runner.run(runner.python_argv("rigcolmap.py", "prep", frames, "-o", work), STAGE, log_path=log)
    caps = json.load(open(os.path.join(work, "captures.json")))["captures"]
    pj.metric(STAGE, "captures", len(caps))
    n_img = 2 * len(caps)

    # ---- sfm
    events.start(STAGE, "sfm")
    argv = runner.python_argv("rigcolmap.py", "sfm", work, "--calib", calib_npz,
                              "--peak-threshold", a.peak_threshold, "--features", a.features)
    if not a.no_float_rig:
        argv.append("--float-rig")
    if a.masks:
        argv += ["--masks", a.masks]
    state = {"step": "features", "reg": 0, "metrics": {}}

    def on_line(line):
        m = RE_FEAT.search(line)
        if m:
            state["step"] = "features"
            events.progress(STAGE, int(m.group(1)), int(m.group(2)), step="features")
            return
        m = RE_MATCH.search(line)
        if m:
            state["step"] = "matching"
            i, ni, j, nj = (int(x) for x in m.groups())
            events.progress(STAGE, (i - 1) * nj + j - 1, ni * nj, step="matching",
                            detail=f"block {i}/{ni},{j}/{nj}")
            return
        m = RE_REG.search(line)
        if m:
            state["step"] = "mapping"
            state["reg"] = int(m.group(2))
            events.progress(STAGE, state["reg"], n_img, step="mapping", detail="images registered")
            return
        for key, rx in (("featstat", RE_FEATSTAT), ("recon", RE_RECON), ("reproj", RE_REPROJ),
                        ("rig", RE_RIG), ("ba", RE_BA), ("step", RE_STEP), ("depth", RE_DEPTH)):
            m = rx.search(line)
            if m:
                state["metrics"][key] = m.groups()
                return
        m = RE_EYE.search(line)
        if m:
            state["metrics"]["eye_" + m.group(1)] = m.groups()[1:]

    runner.run(argv, STAGE, log_path=log, on_line=on_line)
    rep_path = os.path.join(work, "sfm_report.json")
    if not os.path.exists(rep_path):
        raise events.StageError("sfm wrote no sfm_report.json", hint="see logs/solve.log")
    rep = json.load(open(rep_path))
    mm = state["metrics"]
    if "featstat" in mm:
        pj.metric(STAGE, "features_per_image_median", int(mm["featstat"][1]))
    pj.metric(STAGE, "num_images", rep["num_images"])
    pj.metric(STAGE, "num_frames", rep["num_frames"])
    pj.metric(STAGE, "num_points", rep["num_points"])
    pj.metric(STAGE, "mean_reproj_px", round(float(rep["mean_reproj_px"]), 4))
    if "reproj" in mm:
        pj.metric(STAGE, "median_reproj_px", float(mm["reproj"][1]))
    for eye in ("L", "R"):
        if "eye_" + eye in mm:
            pj.metric(STAGE, f"eye_{eye}_rms_px", float(mm["eye_" + eye][1]))
    if "ba" in mm:
        pj.metric(STAGE, "ba_rig_rotation_shift_deg", float(mm["ba"][1]))
    if "rig" in mm:
        pj.metric(STAGE, "rig_baseline_mm", float(mm["rig"][0]))
    if "step" in mm:
        pj.metric(STAGE, "path_length_mm", int(mm["step"][3]))
        pj.metric(STAGE, "capture_step_median_mm", int(mm["step"][2]))
    if "depth" in mm:
        pj.metric(STAGE, "scene_depth_p5_mm", int(mm["depth"][0]))
        pj.metric(STAGE, "scene_depth_median_mm", int(mm["depth"][1]))
        pj.metric(STAGE, "scene_depth_p95_mm", int(mm["depth"][2]))
    pj.artifact(STAGE, rep_path, "json")

    all_reg = rep["num_frames"] == len(caps)
    pj.check(STAGE, "all_frames_registered", all_reg, value=f"{rep['num_frames']}/{len(caps)}")
    pj.check(STAGE, "mean_reproj_ok", float(rep["mean_reproj_px"]) <= a.max_reproj,
             value=f"{float(rep['mean_reproj_px']):.3f} px (want ≤ {a.max_reproj}; rig6 1.394)")

    # ---- per-image table (needs the reconstruction; pycolmap is already a dependency)
    events.start(STAGE, "per_image")
    per = per_image_table(os.path.join(work, "sparse", "rig"))
    per_path = os.path.join(work, "per_image.json")
    json.dump(per, open(per_path, "w"), indent=1)
    pj.artifact(STAGE, per_path, "json")
    worst = max(per["images"], key=lambda r: r["mean_reproj_px"]) if per["images"] else None
    if worst:
        pj.metric(STAGE, "worst_image_reproj_px", worst["mean_reproj_px"])
        pj.check(STAGE, "no_image_over_3px", worst["mean_reproj_px"] <= 3.0,
                 value=f"worst {worst['name']} {worst['mean_reproj_px']:.2f} px")

    if not all_reg:
        pj.finish(STAGE, ok=False, error="partial registration")
        missing = sorted(set(c["capture"] for c in caps) - set(r["name"].split("/")[1].split(".")[0] for r in per["images"]))
        raise events.StageError(
            f"only {rep['num_frames']} of {len(caps)} frames registered — the chain broke "
            f"(unregistered: {', '.join(missing[:12])}{'…' if len(missing) > 12 else ''})",
            hint="in order: `hs select --max-gap 45`; trim the clip's tail with `hs select --end N`; "
                 "then look at solve/per_image.json. Do not train on a partial solve.")

    # ---- export
    events.start(STAGE, "export")
    runner.run(runner.python_argv("rigcolmap.py", "export", os.path.join(work, "sparse", "rig"),
                                  "--images", os.path.join(work, "images"), "-o", pj.dataset_dir),
               STAGE, log_path=log,
               on_line=lambda l: (lambda m: m and events.progress(STAGE, int(m.group(1)), int(m.group(2)), step="export"))(RE_UNDIST.search(l)))
    if not os.path.exists(pj.rig_npz):
        raise events.StageError("export wrote no rig.npz")
    # rig.npz's w/h drive the aim check's in-frame bounds and every path's output canvas, and
    # the two eyes are undistorted to different sizes — so they must be the LEFT views' size.
    # This caught write_rig_npz handing over the right eye's canvas (off by 4x2 px).
    import cv2
    G = np.load(pj.rig_npz, allow_pickle=True)
    l_dir = os.path.join(pj.dataset_dir, "images", "L")
    l_first = sorted(f for f in os.listdir(l_dir) if f.lower().endswith(".jpg"))[0]
    im = cv2.imread(os.path.join(l_dir, l_first))
    rig_wh, img_wh = (int(G["w"]), int(G["h"])), (im.shape[1], im.shape[0])
    pj.metric(STAGE, "view_size_L", list(img_wh))
    if "wh" in G.files and len(G["wh"]) > 1:
        pj.metric(STAGE, "view_size_R", [int(x) for x in G["wh"][1]])
    pj.check(STAGE, "rig_size_matches_L_views", rig_wh == img_wh,
             value=f"rig.npz {rig_wh[0]}x{rig_wh[1]} vs {l_first} {img_wh[0]}x{img_wh[1]}")
    pj.artifact(STAGE, pj.dataset_dir, "dataset")
    pj.artifact(STAGE, pj.rig_npz, "rig")
    ply = os.path.join(pj.dataset_dir, "sparse", "points3D.ply")
    if os.path.exists(ply):
        pj.artifact(STAGE, ply, "pointcloud")

    # ---- coverage (in the frame the path builders use)
    events.start(STAGE, "coverage")
    cov = coverage.write(pj.rig_npz, os.path.join(work, "coverage.json"))
    pj.artifact(STAGE, os.path.join(work, "coverage.json"), "json")
    pj.metric(STAGE, "subject_mm", cov["subject_mm"])
    pj.metric(STAGE, "azimuth_range_deg", cov["azimuth_range_deg"])
    pj.metric(STAGE, "elevation_range_deg", cov["elevation_range_deg"])
    pj.metric(STAGE, "distance_range_mm", cov["distance_range_mm"])
    seps = np.array([c["lr_separation_mm"] for c in cov["captures"]])
    dev = float(np.abs(seps - baseline).max())
    pj.metric(STAGE, "lr_separation_max_dev_mm", round(dev, 4))
    pj.check(STAGE, "rig_constraint_held", dev < 0.01,
             value=f"L–R separation {seps.min():.3f}–{seps.max():.3f} mm vs profile {baseline:.3f} mm")
    # the stage completed; a failed quality check stays visible in the manifest and the
    # event stream rather than blocking (partial registration raised above — that one blocks)
    pj.finish(STAGE, ok=True)


def per_image_table(recon_dir):
    """Per-image mean reprojection error and observation count, ordered by name."""
    import pycolmap
    rec = pycolmap.Reconstruction(recon_dir)
    rows = []
    for im in rec.images.values():
        if not im.has_pose:
            continue
        obs, xyz = [], []
        for p2 in im.points2D:
            if p2.has_point3D():
                obs.append(p2.xy)
                xyz.append(rec.point3D(p2.point3D_id).xyz)
        if not obs:
            rows.append({"name": im.name, "n_obs": 0, "mean_reproj_px": None})
            continue
        loc = im.cam_from_world() * np.asarray(xyz, float)
        ok = loc[:, 2] > 0
        pr = np.asarray(rec.camera(im.camera_id).img_from_cam(loc[ok]), float)
        e = np.linalg.norm(np.asarray(obs)[ok] - pr, axis=1)
        rows.append({"name": im.name, "n_obs": int(ok.sum()),
                     "mean_reproj_px": round(float(e.mean()), 4), "median_reproj_px": round(float(np.median(e)), 4)})
    rows.sort(key=lambda r: r["name"])
    return {"images": rows}
