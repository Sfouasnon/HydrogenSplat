"""hs scale — metric scale (and a ground-plane estimate) from a ChArUco board in the scene.

Photogrammetry from uncalibrated cameras has no unit. The Hydrogen's calibrated baseline gives
its solves millimetres for free; an array gets them from `hs solve --scale-pair`; a mono take
had nothing but a hand-measured `--scale S`. A ChArUco board lying anywhere in the scene gives
one without measuring anything but the print: its corners are detected with sub-pixel accuracy
and each carries an id, so the same corner is found in every view that sees it.

Method (hs/board.py):
  1. detect the board in the undistorted training images (train/dataset/images/<eye>),
     cv2.aruco.CharucoDetector;
  2. triangulate every corner id seen in >= --min-views views by linear DLT from rig.npz's
     own K, R, t — the poses the model trains on, not a separate solve;
  3. scale = median over every corner pair of known_mm / reconstructed distance. The spread
     (MAD) and each view's reprojection RMS of the triangulated corners are the checks: a
     corner the solve places badly shows up there rather than in the scale;
  4. fit the board's plane: its normal, oriented towards the cameras, is the up-axis a board
     lying on the table implies. Reported with its angle to coverage's current up (the mean
     camera -y); not applied.

Applying it is exactly what `monocolmap.py sfm --scale S` does, done after the fact: one
pycolmap.Sim3d(s) on the sparse model (train/dataset/sparse, and solve/sparse/rig so a later
re-export agrees), then monocolmap.finish_dataset rewrites the text model, points3D.ply and
rig.npz (t, C, pts; s_mm stays 1.0 because the file is now in mm, which is what it means),
coverage.write rewrites solve/coverage.json, and solve's `scene_scaled` check is set ok with
source "board" — `hs masks` and the app read that check. Images and masks are untouched: a
similarity changes no pixel and no projection, so exposure and masks stay current, and only
train / move / prune / render / views go stale (their outputs are in the old units).

A stereo (Hydrogen) solve is already metric — the calibrated baseline — so applying a board
scale to it is refused; `--dry-run` still measures, and on a stereo project the factor it
reports is the baseline's own scale error (1.0 = the calibration is right).

`--dry-run` measures and reports and writes nothing but stages.scale.dry_run.

`--lidar SCAN` is the second measurement source, feeding the same apply path (`_apply_and_mark`).
A phone LiDAR scan (hs/lidar.py) is registered to rig.npz's sparse points — solve -> scan,
Sim(3) whatever the source — by global registration (or `--pairs`), then trimmed ICP against
the dense scan. On a mono/array project the scale is applied (`--dry-run` measures only); on a
stereo project the fit gives `scale_ratio` (the factor the solve is off by, 1.0 = the baseline
is right) and `implied_baseline_mm`, and nothing is applied unless `--trust-scan` says the scan
is the better reference — which past a metre or two it is: the H1's 10.6 mm baseline puts 5 px
of disparity on a subject at 4 m, the floating rig's rotation shift is ten times that, and the
realigned CirclesSculpture solve came out 5.8x small (2026-09-23). The scan's up axis is read
off the scan (`--scan-up auto`: Polycam keeps ARKit's +Y, Scaniverse's PLY is +Z). Outdoors the
SfM points run far past the phone's 5 m LiDAR range, so only points near the camera path are
registered, ICP's trim follows the measured overlap, and the tolerances grow with the scene
(2 % of the camera-subject distance). `--factor F` applies a factor found some other way (a
tape, the ring-silhouette fit) through the same path.
Checks: lidar_aligned, lidar_scale_agrees (stereo), lidar_covers_captures, lidar_up_agrees.
Written: scale/lidar_report.json and scale/lidar_aligned.ply (the scan in the solve frame, mm,
for the viewer). A measurement that is not applied (stereo without --trust-scan, --dry-run, or
an alignment that failed) leaves the stage's status alone and records under
stages.scale.lidar_check; a failed alignment exits 1.
"""
import contextlib
import json
import os
import re
import sys

import numpy as np

from .. import board as boardlib
from .. import coverage, events, rig
from .. import lidar as lidarlib
from ..project import now_iso

STAGE = "scale"
MAX_MAD_REL = 0.005          # 0.5 %: the pair ratios agree, i.e. the board is rigid in the solve
MAX_VIEW_RMS_PX = 2.0        # a view whose triangulated corners miss by more is posed badly
REFUSE_MAD_REL = 0.05        # 5 %: past this the factor is refused outright, not applied with a failed check

# --lidar (see run_lidar)
lidarlib_defaults = {"min_inlier": 0.5, "max_rms_mm": 30.0, "inlier_mm": 50.0, "min_scale_sensitivity": 0.1}
LIDAR_SCALE_TOL = 0.02       # stereo: the scan and the calibrated baseline agree within 2 %
LIDAR_UP_TOL_DEG = 10.0      # the scan's gravity and coverage's mean-camera up agree within 10 degrees
LIDAR_CAM_MAX_MM = 3000.0    # every camera within 3 m of the scan...
LIDAR_PTS_MAX_MM = 50.0      # ...and the sparse points it sees within 50 mm of it (median)
LIDAR_REG_POINTS = 5000      # subsample sizes: global registration (each side)...
LIDAR_ICP_SOLVE = 20000      # ...ICP's solve points...
LIDAR_ICP_SCAN = 400000      # ...and its scan (the target, and what the coverage check measures to)
LIDAR_VIEW_POINTS = 100000   # scale/lidar_aligned.ply
LIDAR_MIN_POSE_EIG = 0.01    # lidar_geometry_constrains: every direction of motion is seen (lidarlib.constraints)
LIDAR_REL_INLIER = 0.02      # tolerances grow with the scene: --inlier-mm is at least 2 % of the camera-subject
LIDAR_REL_RMS = 0.015         # distance (the SfM's own depth noise at 4 m is tens of mm), --max-rms-mm 1.5 %,
LIDAR_REL_PTS = 0.02          # and the coverage check's per-frustum median 2 %; indoors the flags' values hold
LIDAR_NEAR_PATH = 2.0        # solve points further than this x the camera path's extent from every camera are not registered
LIDAR_OVERLAP_FACTOR = 10.0  # a solve point within this x --inlier-mm of the scan is one the scan could contain
LIDAR_MIN_TRIM = 0.15        # ICP keeps at least this share of the solve's points, however small the overlap
LIDAR_MIN_OVERLAP_POINTS = 2000   # lidar_aligned: the scan must contain at least this many of ICP's solve points...
LIDAR_MIN_OVERLAP = 0.05          # ...and this share of them


def add_parser(sub):
    p = sub.add_parser("scale", help="metric scale from a ChArUco board seen in the training views, "
                                     "or from a LiDAR scan of the scene (--lidar)")
    add_board_args(p, required=False)
    p.add_argument("--eye", choices=("L", "R", "both"), default="L",
                   help="which views to detect in (a mono or array rig has only L)")
    p.add_argument("--min-views", type=int, default=3, help="triangulate a corner seen in at least this many views")
    p.add_argument("--dry-run", action="store_true", help="measure and report, write nothing")
    g = p.add_argument_group("LiDAR scan (instead of --board)")
    g.add_argument("--lidar", metavar="SCAN", default=None,
                   help="a phone LiDAR scan of the scene (Polycam / Scaniverse PLY point cloud or mesh, OBJ, "
                        "XYZ/CSV): aligned to the solve's sparse points; its metric scale is applied to a "
                        "mono/array solve and checked against a stereo one")
    g.add_argument("--units", choices=tuple(lidarlib.UNIT_MM), default=None,
                   help="the scan file's units (default: a header comment, else inferred from its extent)")
    g.add_argument("--scan-up", choices=("auto",) + tuple(lidarlib.SCAN_UP), default="auto",
                   help="the scan's gravity axis: auto (default: the axis whose lowest band is a floor), "
                        "y (ARKit / Polycam) or z (Scaniverse PLY, Blender-style exports)")
    g.add_argument("--trust-scan", action="store_true",
                   help="stereo projects: apply the scan's scale even though the solve is nominally metric "
                        "(the H1 baseline is unobservable past a metre or two: CirclesSculpture came out "
                        "5.8x small); implies --apply")
    g.add_argument("--pairs", default=None, metavar="'SX,SY,SZ=PX,PY,PZ;...'",
                   help=">= 3 matching points: scan coordinates (scan units) = solve coordinates (rig.npz mm) "
                        "— the initial alignment instead of the automatic search")
    g.add_argument("--init", choices=("auto", "pairs"), default=None,
                   help="initial alignment: auto (global registration; the default) or pairs (--pairs)")
    g.add_argument("--apply", action="store_true",
                   help="apply the scan's scale (the default on a mono/array project unless --dry-run; "
                        "refused on a stereo project)")
    g.add_argument("--min-inlier", type=float, default=lidarlib_defaults["min_inlier"],
                   help="lidar_aligned: share of solve points within --inlier-mm of the scan after ICP")
    g.add_argument("--max-rms-mm", type=float, default=lidarlib_defaults["max_rms_mm"],
                   help="lidar_aligned: trimmed (best 70%%) RMS after ICP, mm")
    g.add_argument("--inlier-mm", type=float, default=lidarlib_defaults["inlier_mm"],
                   help="a solve point within this distance of the scan counts as on it")
    g.add_argument("--min-scale-sensitivity", type=float, default=lidarlib_defaults["min_scale_sensitivity"],
                   help="lidar_geometry_constrains: how strongly the overlap's shape fixes the scale (0 for floor "
                        "and walls alone; ~0.23 for a room with furniture); a mono scale below it is not applied")
    g.add_argument("--icp-iters", type=int, default=100,
                   help="ICP iterations at most (a room converges slowly along its scale: the walls do not "
                        "care, only the furniture pulls; from --pairs' few-percent start it takes ~60)")
    g.add_argument("--seed", type=int, default=0, help="random seed of the subsamples and the global search")
    g.add_argument("--factor", type=float, default=None, metavar="F",
                   help="apply a known scale factor instead of measuring one: the solve's units x F = mm "
                        "(a tape measure, or the ring-silhouette fit of 2026-09-23 that gave Circles 5.8); "
                        "stereo projects need --trust-scan; --note says where F came from")
    g.add_argument("--note", default=None, help="with --factor: where the factor came from (recorded)")
    return p


def add_board_args(p, required=True):
    p.add_argument("--board", required=required, metavar="SX,SY,SQUARE_MM,MARKER_MM[,DICT]",
                   help="the ChArUco board: squares across, squares down, printed square and marker side "
                        f"in mm, ArUco dictionary (default {boardlib.DEFAULT_DICT}), e.g. 7,5,40,30")
    p.add_argument("--legacy-board", action="store_true",
                   help="the board was printed from a pre-4.6 OpenCV generator (old even-row layout)")


def spec_from_args(a, pj=None):
    """The board: --board, else the one the project's last `hs scale` used."""
    s = getattr(a, "board", None)
    if not s and pj is not None:
        s = (pj.m.get("scale") or {}).get("board")
    if not s:
        raise events.StageError("no board given", hint="--board SX,SY,SQUARE_MM,MARKER_MM[,DICT], e.g. 7,5,40,30; "
                                                       "or --lidar SCAN for a LiDAR scan of the scene")
    try:
        spec = boardlib.parse_spec(s, legacy=bool(getattr(a, "legacy_board", False)))
        boardlib.make_board(spec)                  # the legacy layout needs a recent OpenCV
        return spec
    except ValueError as e:
        raise events.StageError(str(e))


def _view_image(pj, name):
    eye = name[-1] if name.endswith(("_L", "_R")) else "L"
    stem = rig.capture_name(name)
    d = os.path.join(pj.dataset_dir, "images", eye)
    for ext in (".jpg", ".jpeg", ".png"):
        p = os.path.join(d, stem + ext)
        if os.path.exists(p):
            return eye, p
    return eye, None


def detect_views(pj, det, names, eyes, stage=STAGE):
    """{view index: (ids, xy)} for every view of the chosen eyes whose board was found, and the
    list of view names looked at."""
    import cv2
    found, looked = {}, []
    todo = [(v, n) for v, n in enumerate(names) if (n[-1] if n.endswith(("_L", "_R")) else "L") in eyes]
    events.start(stage, "detect")
    for i, (v, n) in enumerate(todo):
        _eye, path = _view_image(pj, n)
        events.progress(stage, i + 1, len(todo), step="detect", detail=n)
        if path is None:
            continue
        looked.append(n)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        ids, xy = det.detect(img)
        if len(ids):
            found[v] = (ids, xy)
    return found, looked


def measure(pj, spec, eyes, min_views):
    """Detection + triangulation + scale + plane on the project as it is now. -> dict."""
    G, names, L = rig.load(pj.rig_npz)
    K, R, t = (G[k].astype(np.float64) for k in ("K", "R", "t"))
    det = boardlib.Detector(spec)
    found, looked = detect_views(pj, det, names, eyes)
    if not looked:
        raise events.StageError(f"no training images for eye {'/'.join(sorted(eyes))} under train/dataset/images",
                                hint="hs solve first; a mono or array rig has only --eye L")
    obs = {}
    for v, (ids, xy) in found.items():
        for c, p in zip(ids, xy):
            obs.setdefault(int(c), []).append((v, p))
    ids, X, res = boardlib.triangulate(obs, K, R, t, min_views=min_views)
    out = {"views_looked": len(looked), "views_with_board": [names[v] for v in sorted(found)],
           "corners_detected": int(sum(len(i) for i, _ in found.values())),
           "corners_triangulated": int(len(ids)), "names": names, "stereo": rig.is_stereo(G)}
    if len(ids) < 4:
        best = max((len({vv for vv, _ in o}) for o in obs.values()), default=0)
        raise events.StageError(
            f"only {len(ids)} board corners were seen in >= {min_views} views "
            f"(board found in {len(found)} of {len(looked)} views; most views any corner was seen in: {best})",
            hint="check --board against the print (squares across, down, square and marker mm, dictionary); "
                 "lower --min-views; or the board is too small or oblique in these views")
    obj = boardlib.corner_points(spec)
    sc = boardlib.scale_from_pairs(ids, X, obj)
    rms = boardlib.per_view_rms(res)
    Cs = -np.einsum("nji,nj->ni", R, t)                   # camera centres, rig units
    centroid, normal, plane_rms = boardlib.fit_plane(X)
    normal = boardlib.orient_towards(normal, centroid, Cs[sorted(found)].mean(axis=0))
    # coverage's up (hs/coverage.py): minus the mean camera +y axis over the reference views
    down = np.mean([R[v].T @ np.array([0.0, 1.0, 0.0]) for v in L], axis=0)
    cov_up = (-down / np.linalg.norm(down)).tolist()
    s_um, _Ru, _tu = boardlib.umeyama(obj[ids], X)           # board mm -> rig units
    out.update({
        "scale": sc["scale"], "mad": sc["mad"], "mad_rel": sc["mad_rel"], "n_pairs": sc["n_pairs"],
        "umeyama_scale": 1.0 / s_um if s_um else None,
        "view_rms_px": {names[v]: round(e, 4) for v, e in sorted(rms.items())},
        "plane_normal": normal.tolist(), "plane_centroid": centroid.tolist(),
        "plane_rms_mm": plane_rms * sc["scale"],
        "angle_to_up_deg": boardlib.angle_deg(normal, cov_up), "current_up": cov_up,
        "corners": {int(c): X[i].tolist() for i, c in enumerate(ids)},
    })
    return out


def run(a, pj):
    if getattr(a, "factor", None) is not None:
        if getattr(a, "board", None) or getattr(a, "lidar", None):
            raise events.StageError("give --factor alone, without --board or --lidar",
                                    hint="--factor applies a number you already have; the others measure one")
        return run_factor(a, pj)
    if getattr(a, "lidar", None):
        if getattr(a, "board", None):
            raise events.StageError("give --board or --lidar, not both",
                                    hint="one measurement per run: `hs scale --board ...` or `hs scale --lidar SCAN`")
        return run_lidar(a, pj)
    pj.require(STAGE)
    spec = spec_from_args(a, pj)
    if not os.path.exists(pj.rig_npz):
        raise events.StageError("no train/dataset/rig.npz", hint="hs solve first")
    G = np.load(pj.rig_npz, allow_pickle=True)
    # the camera contract decides: a mono rig.npz comes from monocolmap.py (array / mono
    # sources), whose scale is whatever it was told; a stereo one carries the baseline's
    stereo = rig.is_stereo(G)
    if stereo and not a.dry_run:
        raise events.StageError(
            "a stereo solve is already metric: its scale comes from the calibrated baseline, so a board "
            "scale is not applied to it",
            hint="`hs scale --dry-run --board ...` measures the baseline's scale error on this project; "
                 "to change the baseline re-solve with `hs solve --baseline-mm MM`")
    eyes = {"L", "R"} if a.eye == "both" else {a.eye}
    if not a.dry_run:
        pj.acquire(STAGE)        # measure under the lock: a concurrent solve would rewrite rig.npz
    m = measure(pj, spec, eyes, a.min_views)
    s = float(m["scale"])
    worst = max(m["view_rms_px"].items(), key=lambda kv: kv[1])
    metrics = {
        "board": spec.text(), "scale_factor": round(s, 8), "scale_mad_rel": round(m["mad_rel"], 6),
        "scale_umeyama": round(m["umeyama_scale"], 8) if m["umeyama_scale"] else None,
        "board_pairs": m["n_pairs"], "views_with_board": len(m["views_with_board"]),
        "views_looked": m["views_looked"], "corners_triangulated": m["corners_triangulated"],
        "board_reproj_rms_max_px": round(worst[1], 4), "board_reproj_rms_worst_view": worst[0],
        "board_plane_rms_mm": round(m["plane_rms_mm"], 4),
        "board_normal_world": [round(x, 6) for x in m["plane_normal"]],
        "board_normal_to_up_deg": round(m["angle_to_up_deg"], 3),
    }
    checks = [
        ("board_pair_ratios_agree", m["mad_rel"] <= MAX_MAD_REL,
         f"MAD {100 * m['mad_rel']:.3f}% of the scale over {m['n_pairs']} corner pairs (want <= {100 * MAX_MAD_REL:.1f}%)"),
        ("board_corners_reproject", worst[1] <= MAX_VIEW_RMS_PX,
         f"worst view {worst[0]} {worst[1]:.2f} px RMS (want <= {MAX_VIEW_RMS_PX})"),
    ]
    if stereo:
        prof = pj.stage("solve").get("metrics", {}).get("profile_baseline_mm")
        metrics["implied_baseline_mm"] = round(prof * s, 4) if prof else None

    if a.dry_run:
        for k, v in metrics.items():
            events.metric(STAGE, k, v)
        for name, ok, val in checks:
            events.check(STAGE, name, ok, value=val)
        pj.stage(STAGE)["dry_run"] = {"at": now_iso(), "argv": list(sys.argv), "metrics": metrics,
                                      "checks": [{"name": n, "ok": bool(ok), "value": v} for n, ok, v in checks]}
        pj.save()
        return
    if m["mad_rel"] > REFUSE_MAD_REL:
        # a spread this wide is not a board measured by a sound solve; nothing is changed
        raise events.StageError(
            f"the board's corner pairs disagree by {100 * m['mad_rel']:.1f}% (MAD) about a scale of {s:.6g}: "
            "not applied",
            hint="look at it with --dry-run; check --board against the print, raise --min-views, "
                 "or the solve itself is distorted (see solve/per_image.json)")

    pj.begin(STAGE, argv=sys.argv)
    for k, v in metrics.items():
        pj.metric(STAGE, k, v)
    for name, ok, val in checks:
        pj.check(STAGE, name, ok, value=val)
    if not (np.isfinite(s) and s > 0):
        raise events.StageError(f"the board gave a scale of {s}: not applied")

    to_m = _apply_and_mark(pj, s, "board", f"x{s:.6f} from board {spec.text()} (hs scale)")

    rep = {"note": "hs scale: metric scale from a ChArUco board, applied as one Sim3d to the solve",
           "board": spec.text(), "legacy_pattern": spec.legacy, "scale_factor": s, "scale_to_m": to_m,
           "min_views": a.min_views, "eyes": sorted(eyes),
           **{k: m[k] for k in ("mad", "mad_rel", "n_pairs", "umeyama_scale", "views_with_board",
                                "view_rms_px", "plane_normal", "angle_to_up_deg", "current_up")},
           "plane_centroid_mm": (np.asarray(m["plane_centroid"]) * s).tolist(),
           "corners_mm": {c: (np.asarray(x) * s).tolist() for c, x in m["corners"].items()}}
    rp = os.path.join(pj.stage_dir(STAGE), "scale_report.json")
    json.dump(rep, open(rp, "w"), indent=1)
    pj.artifact(STAGE, rp, "json")
    pj.m["scale"] = {"source": "board", "board": spec.text(), "scale_factor": s, "scale_to_m": to_m,
                     "at": now_iso(), "argv": list(sys.argv)}
    pj.finish(STAGE, ok=True)


def apply_scale(pj, s):
    """One Sim3d(s) on the training set's sparse model (and on solve/sparse/rig, the model it was
    undistorted from), then rig.npz / text model / points3D.ply rewritten by the export's own
    writer. Nothing else in the dataset changes."""
    import pycolmap
    from .. import monocolmap
    sim = pycolmap.Sim3d(float(s), pycolmap.Rotation3d(), np.zeros(3))
    sp = os.path.join(pj.dataset_dir, "sparse")
    rec = pycolmap.Reconstruction(sp)
    rec.transform(sim)
    # finish_dataset is the export's writer and prints like a script; stdout is the event
    # stream here, so its lines go to logs/scale.log where a child's output would
    os.makedirs(os.path.dirname(pj.log_path(STAGE)), exist_ok=True)
    with open(pj.log_path(STAGE), "a") as log, contextlib.redirect_stdout(log):
        if rig.is_stereo(np.load(pj.rig_npz, allow_pickle=True)):
            # the stereo dataset is written by the rig route's own writer (views interleaved L,R)
            from .. import rigcolmap
            rec.write(sp)
            rec.export_PLY(os.path.join(sp, "points3D.ply"))
            rec.write_text(sp)
            rigcolmap.write_rig_npz(rec, pj.rig_npz)
        else:
            monocolmap.finish_dataset(rec, pj.dataset_dir, write_binary=True)
    src = pj.path("solve", "sparse", "rig")
    if os.path.isdir(src) and any(f.endswith((".bin", ".txt")) for f in os.listdir(src)):
        r0 = pycolmap.Reconstruction(src)
        r0.transform(sim)
        r0.write(src)
        if os.path.exists(os.path.join(src, "points3D.ply")):
            r0.export_PLY(os.path.join(src, "points3D.ply"))


def _mark_solve_scaled(pj, s, to_m, source, desc):
    """solve's own `scene_scaled` record is what `hs masks` and the app read: set it ok, with its
    source ("board" or "lidar"), instead of leaving an unscaled solve's failure standing next to
    a scaled dataset."""
    st = pj.stage("solve")
    st.setdefault("metrics", {})["scale_to_m"] = to_m
    st["metrics"]["scale_source"] = source
    checks = [c for c in st.get("checks", []) if c.get("name") != "scene_scaled"]
    rec = {"name": "scene_scaled", "ok": True, "source": source, "value": desc}
    checks.append(rec)
    st["checks"] = checks
    pj.check(STAGE, "scene_scaled", True, value=rec["value"], source=source)
    pj.stage(STAGE)["checks"][-1]["source"] = source


def _apply_and_mark(pj, s, source, desc):
    """The one apply path both measurements feed: Sim3d(s) on the model and rig.npz, the unit
    chain, coverage.json rewritten, solve's scene_scaled set. -> scale_to_m."""
    events.start(STAGE, "apply")
    prev = pj.stage("solve").get("metrics", {}).get("scale_to_m")
    apply_scale(pj, s)
    # the unit chain: raw solve units -> metres. An unscaled solve had none (raw = 1 unit).
    to_m = (prev if prev else 1.0) * s
    pj.metric(STAGE, "scale_to_m", round(to_m, 10))
    cov = coverage.write(pj.rig_npz, pj.path("solve", "coverage.json"))
    pj.artifact(STAGE, pj.path("solve", "coverage.json"), "json")
    pj.artifact(STAGE, pj.rig_npz, "rig")
    pj.metric(STAGE, "subject_mm", cov["subject_mm"])
    pj.metric(STAGE, "distance_range_mm", cov["distance_range_mm"])
    _mark_solve_scaled(pj, s, to_m, source, desc)
    return to_m


# ============================================================================ --factor

def run_factor(a, pj):
    """Apply a scale factor the user already has (solve units x F = mm) through the same path as a
    board or a scan: for a length measured with a tape, or a factor found outside the stage — the
    2026-09-23 ring-silhouette fit on CirclesSculpture, where the sparse cloud was a lawn and no
    registration could fix the scale but the sculpture's outline in the photographs could."""
    pj.require(STAGE)
    if not os.path.exists(pj.rig_npz):
        raise events.StageError("no train/dataset/rig.npz", hint="hs solve first")
    f = float(a.factor)
    if not (np.isfinite(f) and f > 0):
        raise events.StageError(f"--factor {a.factor}: want a positive number")
    if a.dry_run:
        raise events.StageError("--factor with --dry-run does nothing: the factor is already known",
                                hint="drop --dry-run to apply it")
    stereo = rig.is_stereo(rig.load(pj.rig_npz)[0])
    if stereo and not getattr(a, "trust_scan", False):
        raise events.StageError(
            "a stereo solve is nominally metric: its scale comes from the calibrated baseline, so a factor "
            "is not applied to it unless you say your measurement is the better reference",
            hint="add --trust-scan (the H1 baseline is unobservable past a metre or two: Circles came out 5.8x small)")
    pj.begin(STAGE, argv=sys.argv)
    pj.metric(STAGE, "scale_factor", round(f, 8))
    pj.metric(STAGE, "scale_source", "manual")
    if a.note:
        pj.metric(STAGE, "scale_note", a.note)
    if stereo:
        prof = pj.stage("solve").get("metrics", {}).get("profile_baseline_mm")
        pj.metric(STAGE, "implied_baseline_mm", round(prof * f, 4) if prof else None)
    desc = f"x{f:.6f} given by hand (hs scale --factor" + (f": {a.note}" if a.note else "") + ")"
    to_m = _apply_and_mark(pj, f, "manual", desc)
    rep = {"note": "hs scale: a scale factor given by hand, applied as one Sim3d to the solve",
           "scale_factor": f, "scale_to_m": to_m, "source": "manual", "from": a.note, "stereo": stereo}
    rp = os.path.join(pj.stage_dir(STAGE), "scale_report.json")
    json.dump(rep, open(rp, "w"), indent=1)
    pj.artifact(STAGE, rp, "json")
    pj.m["scale"] = {"source": "manual", "note": a.note, "scale_factor": f, "scale_to_m": to_m,
                     "at": now_iso(), "argv": list(sys.argv)}
    pj.finish(STAGE, ok=True)


# ============================================================================ --lidar

def parse_pairs(text):
    """``"sx,sy,sz=px,py,pz; ..."`` -> (scan (n, 3), solve (n, 3)); ValueError with the reason."""
    rows = [r.strip() for r in re.split(r"[;\n]", text or "") if r.strip()]
    S, P = [], []
    for r in rows:
        if r.count("=") != 1:
            raise ValueError(f"pair {r!r}: want SX,SY,SZ=PX,PY,PZ")
        a, b = r.split("=")
        try:
            s = [float(x) for x in re.split(r"[,\s]+", a.strip()) if x]
            p = [float(x) for x in re.split(r"[,\s]+", b.strip()) if x]
        except ValueError:
            raise ValueError(f"pair {r!r}: coordinates must be numbers")
        if len(s) != 3 or len(p) != 3:
            raise ValueError(f"pair {r!r}: three coordinates on each side of '='")
        S.append(s)
        P.append(p)
    if len(S) < 3:
        raise ValueError(f"{len(S)} pair(s): need at least 3 points found in both the scan and the solve")
    S, P = np.array(S), np.array(P)
    for A, what in ((S, "scan"), (P, "solve")):
        sv = np.linalg.svd(A - A.mean(0), compute_uv=False)
        if sv[1] <= 1e-6 * max(sv[0], 1e-12):
            raise ValueError(f"the {what} points of --pairs are collinear: pick three that make a triangle")
    return S, P


def _mat4(s, R, t):
    M = np.eye(4)
    M[:3, :3] = s * np.asarray(R)
    M[:3, 3] = t
    return [[round(float(x), 9) for x in row] for row in M]


def _fit_record(r):
    return {"s": r["s"], "R": np.asarray(r["R"]).tolist(), "t": np.asarray(r["t"]).tolist(),
            "rms_mm": round(r["rms"], 4), "median_mm": round(r["median"], 4),
            "inlier_fraction": round(r.get("inlier_fraction", float("nan")), 5),
            "inlier_fraction_in_scan": round(r.get("inlier_fraction_in_scan", float("nan")), 5),
            "overlap_fraction": round(r.get("overlap_fraction", float("nan")), 5),
            "iterations": r["iterations"], "converged": r["converged"],
            "rms_mm_per_iteration": [round(x, 4) for x in r["rms_history"]]}


def measure_lidar(pj, G, names, L, scan, a, init, pairs, stereo):
    """Register the scan to rig.npz's sparse points and measure everything the report holds.
    Pure measurement: nothing in the project is written. -> dict."""
    pts_all = G["pts"].astype(np.float64)
    if len(pts_all) < 50:
        raise events.StageError(f"the solve has only {len(pts_all)} sparse points: too few to align a scan to",
                                hint="hs solve first; a solve this thin will not train either")
    # SfM strays out before any subsample: a voxel subsample keeps every isolated stray and thins
    # the surfaces, so 20 % strays in the cloud become ~40 % of what the registration sees
    keep = lidarlib.denoise(pts_all)
    pts = pts_all[keep]
    C, Rc, Kc, tc = (G[k].astype(np.float64) for k in ("C", "R", "K", "t"))
    # only what a phone scan can reach: SfM triangulates the trees at 30 m, the LiDAR stops at
    # 5, so the solve's far points are kept out of the registration (a voxel subsample of the
    # whole cloud was 98 % trees on Circles). "Far" is measured against the camera path's own
    # size, the one length known in the solve's units: further than LIDAR_NEAR_PATH x its
    # extent from every camera is out. Cameras placed off in the distance by a bad solve do
    # not count (the outlier rule of hs solve).
    Cg = C[np.linalg.norm(C - np.median(C, 0), axis=1) <= 10 * np.median(np.linalg.norm(C - np.median(C, 0), axis=1)) + 1e-9]
    path_extent = float(np.linalg.norm(Cg.max(0) - Cg.min(0))) if len(Cg) else 0.0
    if path_extent > 0 and len(pts) > LIDAR_ICP_SOLVE:
        dcam, _ = lidarlib.NN(Cg).query(pts)
        near = dcam <= LIDAR_NEAR_PATH * path_extent
        if near.sum() >= LIDAR_MIN_OVERLAP_POINTS:
            pts = pts[near]
    n_near = int(len(pts))
    if "wh" in G.files and len(G["wh"]) == len(names):
        wh = G["wh"]
    else:
        wh = np.tile([int(G["w"]), int(G["h"])], (len(names), 1))
    X = scan.points
    # the scale is always solved for: a stereo solve is nominally metric, but its baseline is
    # only observable when the scene is close (rig6's bust at 0.3 m: 60 px of disparity; the
    # Circles ring at 4 m: 5 px, and the float rig's 1.5-degree rotation shift swamped it, so
    # the realigned solve came out 5.8x small). An SE(3) search there could only fail; the
    # Sim(3) fit gives the ratio and the pose whatever the baseline did.
    with_scale = True
    seed = a.seed

    events.start(STAGE, "lidar_subsample")
    dense = X[lidarlib.subsample(X, LIDAR_ICP_SCAN, seed=seed)]
    nn = lidarlib.NN(dense)
    icp_i = lidarlib.subsample(pts, LIDAR_ICP_SOLVE, method="random", seed=seed)
    sub = {"scan_points": int(len(X)), "scan_icp": int(len(dense)), "solve_points": int(len(pts_all)),
           "solve_strays_removed": int((~keep).sum()), "solve_points_near_path": n_near,
           "path_extent_solve_units": round(path_extent, 1),
           "solve_icp": int(len(icp_i)), "scan_icp_spacing_mm": round(lidarlib.spacing(dense, nn, seed=seed), 3)}

    glob = None
    if init == "auto":
        reg_scan = lidarlib.subsample(X, LIDAR_REG_POINTS, seed=seed)
        reg_solve = lidarlib.subsample(pts, LIDAR_REG_POINTS, method="random", seed=seed)
        sub.update(scan_register=int(len(reg_scan)), solve_register=int(len(reg_solve)))
        events.start(STAGE, "lidar_register")
        glob = lidarlib.global_register(
            pts[reg_solve], X[reg_scan], with_scale, seed=seed, min_inlier=a.min_inlier, dst_ref=dense,
            min_scale_sensitivity=a.min_scale_sensitivity,
            progress=lambda d, n: events.progress(STAGE, d, n, step="lidar_register", detail=f"sample {d}"))
        T0 = (glob["s"], glob["R"], glob["t"])
        init_rep = {"method": "auto", "aligned": glob["aligned"], "reason": glob["reason"],
                    **{k: glob.get(k) for k in ("inlier_fraction", "rms", "cost", "runner_up_cost", "tau_used",
                                                "samples", "hypotheses", "confirmations", "seconds", "ambiguous",
                                                "alternatives", "constraints", "dst_spacing", "src_spacing")}}
        bounds = 1.1
    else:
        S_mm = pairs[0] * scan.meta["to_mm"]
        s_p, R_p, t_p, res = lidarlib.umeyama(S_mm, pairs[1], with_scale)       # scan mm -> solve
        T0 = lidarlib.invert((s_p, R_p, t_p))
        init_rep = {"method": "pairs", "pairs": int(len(S_mm)),
                    "residuals_solve_units": [round(float(x), 4) for x in res],
                    "scale_solve_to_scan": T0[0]}
        bounds = 1.25

    events.start(STAGE, "lidar_icp")
    step = "lidar_icp"

    def cb(it, n, rms):
        events.progress(STAGE, it, n, step=step, detail=f"trimmed RMS {rms:.2f} mm", force=(it == n))

    Ps = pts[icp_i]
    # how much of the solve the scan can contain at all: indoors nearly all of it, but outdoors
    # the SfM points run out to the trees at 30 m while the phone's LiDAR stops at 5 (Circles,
    # 2026-09-23: 15 % of the sparse cloud lay on the scan). The trim follows that overlap, else
    # the points with no surface under them drag the fit off; the verdict below is read over
    # the points the scan does contain, with a floor on how many that must be.
    # the scene's size in mm, from the initial scale: the tolerances are relative to it
    subject0 = np.median(pts, axis=0)
    scene_mm = float(T0[0] * np.median(np.linalg.norm(Cg - subject0, axis=1))) if len(Cg) else 0.0
    inlier_mm = max(float(a.inlier_mm), LIDAR_REL_INLIER * scene_mm)
    max_rms_mm = max(float(a.max_rms_mm), LIDAR_REL_RMS * scene_mm)
    pts_max_mm = max(LIDAR_PTS_MAX_MM, LIDAR_REL_PTS * scene_mm)
    overlap_mm = LIDAR_OVERLAP_FACTOR * inlier_mm
    d0, _ = nn.query(lidarlib.apply(T0, Ps))
    overlap0 = float(np.mean(d0 <= overlap_mm))
    trim = float(np.clip(0.9 * overlap0, LIDAR_MIN_TRIM, 0.7))
    # a global search that found nothing has decided the verdict: a few iterations fill the report
    iters = a.icp_iters if glob is None or glob["aligned"] else min(10, a.icp_iters)
    fit = lidarlib.icp(Ps, None, T0, max_iter=iters, trim=trim, with_scale=with_scale, index=nn,
                       inlier_dist=inlier_mm, scale_bounds=bounds, keep_within=inlier_mm, callback=cb)
    sim = fit
    s = float(sim["s"])                    # solve -> scan scale: what turns solve units into mm
    R, t = fit["R"], fit["t"]              # the reported alignment, Sim(3) on every source
    T = (float(fit["s"]), R, t)
    d_fit = np.asarray(sim["distances"])
    in_scan = d_fit <= overlap_mm
    n_in = int(in_scan.sum())
    overlap = float(np.mean(in_scan))
    inl_in = float(np.mean(d_fit[in_scan] <= inlier_mm)) if n_in else 0.0
    sim["inlier_fraction_in_scan"] = inl_in
    sim["overlap_fraction"] = overlap
    sim["rms_in_scan"] = float(np.sqrt(np.mean(np.minimum(d_fit[in_scan], overlap_mm) ** 2))) if n_in else float("inf")

    # ---- the verdict, on the fit the scale is read from
    reasons = []
    if glob is not None and not glob["aligned"]:
        reasons.append(f"global registration: {glob['reason']}")
    if n_in < LIDAR_MIN_OVERLAP_POINTS or overlap < LIDAR_MIN_OVERLAP:
        reasons.append(f"only {n_in} of the solve's {len(Ps)} points ({100 * overlap:.0f}%) lie within "
                       f"{overlap_mm:g} mm of the scan after ICP (want >= {LIDAR_MIN_OVERLAP_POINTS} and "
                       f">= {100 * LIDAR_MIN_OVERLAP:.0f}%): the scan does not cover what the cameras saw")
    if inl_in < a.min_inlier:
        reasons.append(f"{100 * inl_in:.0f}% of the solve's points on the scan are within {inlier_mm:g} mm "
                       f"of it after ICP (want >= {100 * a.min_inlier:.0f}%; {100 * overlap:.0f}% of the "
                       f"solve is on the scan at all)")
    if sim["rms"] > max_rms_mm:
        reasons.append(f"trimmed RMS {sim['rms']:.1f} mm after ICP (want <= {max_rms_mm:g})")
    aligned = not reasons

    # ---- does the overlap's shape fix the pose and the scale at all? (lidarlib.constraints)
    Y = lidarlib.apply((sim["s"], sim["R"], sim["t"]), Ps)
    dY, jY = nn.query(Y)
    inl = dY <= inlier_mm
    if inl.sum() >= 10:
        nY, _ = lidarlib.estimate_normals(dense[jY[inl]], 16, ref=dense, index=nn)
        con = lidarlib.constraints(Y[inl], nY)
    else:
        con = lidarlib.constraints(Y[:0], Y[:0])
    weak = []
    if con["pose_min_eig"] < LIDAR_MIN_POSE_EIG:
        weak.append(f"the overlap does not fix the pose (smallest pose eigenvalue {con['pose_min_eig']:.4f} < "
                    f"{LIDAR_MIN_POSE_EIG:g}): the solve sees too few of the scan's surfaces")
    if con["scale_sensitivity"] < a.min_scale_sensitivity:
        weak.append(f"the overlap does not fix the scale (sensitivity {con['scale_sensitivity']:.3f} < "
                    f"{a.min_scale_sensitivity:g}): e.g. floor and walls alone fit at any scale about their corner")
    constrained = not weak

    # ---- frames: scan mm -> the solve as it is now, and as it will be once scaled
    R_st = np.asarray(R).T                                  # scan -> solve rotation
    to_current = lidarlib.invert(T)
    # once the solve is multiplied by s (T = (s, R, t)), scan -> solve is rigid: (1, R^T, -R^T t)
    to_metric = (1.0, R_st, -(R_st @ t))

    # ---- up and ground
    cov_up = np.asarray(coverage.compute(pj.rig_npz)["up_world"], np.float64)
    up_world, up_angle = lidarlib.up_axis(R_st, a.scan_up, cov_up)
    gp_scan = lidarlib.ground_plane(dense, lidarlib.SCAN_UP[a.scan_up], seed=seed)
    CL_scan = lidarlib.apply(T, C[L])
    ground = None
    if gp_scan is not None:
        n_sc, p_sc = np.asarray(gp_scan["normal"]), np.asarray(gp_scan["point_mm"])
        heights = (CL_scan - p_sc) @ n_sc
        ground = {k: gp_scan[k] for k in ("inliers", "band_points", "inlier_share", "rms_mm", "tilt_to_up_deg")}
        ground.update({"normal_scan": n_sc.tolist(), "point_scan_mm": p_sc.tolist(),
                       "normal_solve": (R_st @ n_sc).tolist(),
                       "camera_height_mm": {"median": round(float(np.median(heights)), 1),
                                            "min": round(float(heights.min()), 1),
                                            "max": round(float(heights.max()), 1)}})

    # ---- coverage: per capture, in the scan's mm
    Pscan = lidarlib.apply(T, Ps)
    RcL = Rc[L] @ R_st                      # camera rotations in the scan frame
    tcL = T[0] * tc[L] - np.einsum("nij,j->ni", RcL, t)
    rows = lidarlib.coverage_check(CL_scan, None, pts=Pscan, K=Kc[L], R=RcL, t=tcL, wh=wh[L],
                                   names=[rig.capture_name(names[v]) for v in L], index=nn,
                                   cam_max_mm=LIDAR_CAM_MAX_MM, pts_max_mm=pts_max_mm)
    bad = [r["capture"] for r in rows if not r["ok"]]

    meta = scan.meta
    units_ok = meta["units_source"] != "extent" or meta["units_confident"]
    metrics = {
        "lidar_scan": os.path.basename(a.lidar), "scan_points": meta["points"], "scan_faces": meta["faces"],
        "scan_units": meta["units"], "scan_units_source": meta["units_source"],
        "scan_up": a.scan_up, "scan_up_source": a.scan_up_source,
        "lidar_init": init, "lidar_icp_iterations": fit["iterations"],
        "lidar_rms_mm": round(sim["rms"], 3), "lidar_inlier_fraction": round(sim["inlier_fraction"], 4),
        "lidar_inlier_fraction_in_scan": round(sim["inlier_fraction_in_scan"], 4),
        "solve_overlap_fraction": round(sim["overlap_fraction"], 4), "lidar_icp_trim": round(trim, 3),
        "lidar_inlier_mm": round(inlier_mm, 1), "lidar_max_rms_mm": round(max_rms_mm, 1),
        "scene_distance_mm": round(scene_mm, 1),
        "lidar_aligned": aligned,
        "solve_strays_removed": sub["solve_strays_removed"],
        "lidar_scale_sensitivity": round(con["scale_sensitivity"], 4),
        "lidar_pose_min_eig": round(con["pose_min_eig"], 5),
        "lidar_up_world": [round(float(x), 6) for x in up_world],
        "lidar_up_to_current_up_deg": round(up_angle, 3),
        "captures_off_scan": len(bad),
    }
    if glob is not None:
        metrics["lidar_register_s"] = glob["seconds"]
    if ground is not None:
        metrics["ground_normal_world"] = [round(float(x), 6) for x in ground["normal_solve"]]
        metrics["camera_height_mm_median"] = ground["camera_height_mm"]["median"]
    implied = None
    if stereo:
        prof = pj.stage("solve").get("metrics", {}).get("profile_baseline_mm")
        implied = round(prof * s, 4) if prof else None
        metrics.update({"scale_ratio": round(s, 6), "implied_baseline_mm": implied})
    else:
        metrics["scale_factor"] = round(s, 8)

    checks = [{"name": "lidar_aligned", "ok": aligned,
               "value": (f"{100 * sim['inlier_fraction_in_scan']:.1f}% of the {n_in} solve points on the scan "
                         f"({100 * overlap:.0f}% of {len(Ps)}) within {inlier_mm:g} mm, trimmed RMS "
                         f"{sim['rms']:.2f} mm (want >= {100 * a.min_inlier:.0f}% and <= {max_rms_mm:g} mm)"
                         + ("" if aligned else "; " + "; ".join(reasons)))}]
    checks.append({"name": "lidar_geometry_constrains", "ok": constrained, "needs_human": not constrained,
                   "value": (f"scale sensitivity {con['scale_sensitivity']:.3f} (want >= {a.min_scale_sensitivity:g}), "
                             f"smallest pose eigenvalue {con['pose_min_eig']:.4f} (want >= {LIDAR_MIN_POSE_EIG:g}) "
                             f"over {con['n']} points on the scan")
                            + ("" if constrained else "; " + "; ".join(weak))})
    if stereo:
        ok = abs(s - 1.0) <= LIDAR_SCALE_TOL
        checks.append({"name": "lidar_scale_agrees", "ok": ok, "needs_human": not ok,
                       "value": f"scale ratio {s:.5f} ({100 * (s - 1):+.2f}%; want within {100 * LIDAR_SCALE_TOL:.0f}%)"
                                + (f", implied baseline {implied:.3f} mm" if implied else "")})
    checks.append({"name": "lidar_covers_captures", "ok": not bad,
                   "value": (f"all {len(rows)} captures within {LIDAR_CAM_MAX_MM / 1000:g} m of the scan, their points "
                             f"within {pts_max_mm:g} mm" if not bad else
                             f"{len(bad)} of {len(rows)} off the scan: {', '.join(bad[:12])}"
                             + (" ..." if len(bad) > 12 else ""))})
    ok = up_angle <= LIDAR_UP_TOL_DEG
    checks.append({"name": "lidar_up_agrees", "ok": ok, "needs_human": not ok,
                   "value": f"scan up {a.scan_up.upper()} is {up_angle:.1f} deg from coverage's up (want <= {LIDAR_UP_TOL_DEG:g})"})
    checks.append({"name": "lidar_units_known", "ok": units_ok, "needs_human": not units_ok,
                   "value": f"{meta['units']} from {meta['units_source']} (extent {meta['extent_file_units']:g} file units)"})

    report = {
        "note": "hs scale --lidar: a LiDAR scan registered to the solve's sparse points (solve -> scan, trimmed "
                "ICP); lengths in mm unless a key says otherwise",
        "scan": {k: v for k, v in meta.items()},
        "stereo": stereo, "mode": "sim3", "scan_up": a.scan_up, "scan_up_source": a.scan_up_source,
        "scan_up_detection": a.scan_up_detection,
        "thresholds": {"min_inlier": a.min_inlier, "max_rms_mm": max_rms_mm, "inlier_mm": inlier_mm,
                       "max_rms_mm_flag": a.max_rms_mm, "inlier_mm_flag": a.inlier_mm, "scene_distance_mm": round(scene_mm, 1),
                       "scale_tol": LIDAR_SCALE_TOL, "up_tol_deg": LIDAR_UP_TOL_DEG,
                       "camera_max_mm": LIDAR_CAM_MAX_MM, "points_max_mm": pts_max_mm},
        "subsample": sub, "init": init_rep,
        "icp": _fit_record(fit),
        "aligned": aligned, "reasons": reasons, "constrained": constrained, "weak": weak, "constraints": con,
        "rms_mm": round(sim["rms"], 4), "inlier_fraction": round(sim["inlier_fraction"], 5),
        "solve_to_scan": {"s": T[0], "R": np.asarray(R).tolist(), "t": np.asarray(t).tolist()},
        "scale_factor": None if stereo else s, "scale_ratio": s if stereo else None,
        "implied_baseline_mm": implied,
        "up": {"up_world": up_world.tolist(), "current_up": cov_up.tolist(), "angle_deg": up_angle},
        "ground_plane": ground,
        "coverage": {"captures": rows, "failing": bad},
    }
    return {"report": report, "metrics": metrics, "checks": checks, "aligned": aligned, "reasons": reasons,
            "constrained": constrained, "weak": weak,
            "scale": s, "to_current": to_current, "to_metric": to_metric, "stereo": stereo}


def _write_lidar_outputs(pj, m, scan, a, to_frame, frame_note):
    """scale/lidar_report.json (with the scan -> solve transform the viewer needs) and
    scale/lidar_aligned.ply (a subsample of the scan in that frame). -> (report path, ply path)."""
    d = pj.stage_dir(STAGE)
    os.makedirs(d, exist_ok=True)
    s_o, R_o, t_o = to_frame
    k = scan.meta["to_mm"]
    rep = m["report"]
    rep["transform"] = {"frame": frame_note,
                        "scan_mm_to_solve": _mat4(s_o, R_o, t_o),
                        "scan_file_to_solve": _mat4(s_o * k, R_o, t_o),
                        "note": "4x4, column vectors: X_solve = M @ [x, y, z, 1]; 'scan_file' takes the file's own units"}
    if rep.get("ground_plane"):
        rep["ground_plane"]["point_solve"] = lidarlib.apply(to_frame, np.asarray(rep["ground_plane"]["point_scan_mm"])[None])[0].tolist()
    rp = os.path.join(d, "lidar_report.json")
    tmp = rp + ".partial"
    with open(tmp, "w") as f:
        json.dump(rep, f, indent=1, default=events._default)
    os.replace(tmp, rp)
    vi = lidarlib.subsample(scan.points, LIDAR_VIEW_POINTS, seed=a.seed)
    ply = os.path.join(d, "lidar_aligned.ply")
    lidarlib.write_points_ply(ply, lidarlib.apply(to_frame, scan.points[vi]),
                              scan.colors[vi] if scan.colors is not None else None,
                              comments=[f"hs scale --lidar: {os.path.basename(a.lidar)} in the solve frame, {frame_note}"])
    return rp, ply


def run_lidar(a, pj):
    pj.require(STAGE)
    if not os.path.exists(pj.rig_npz):
        raise events.StageError("no train/dataset/rig.npz", hint="hs solve first")
    G, names, L = rig.load(pj.rig_npz)
    stereo = rig.is_stereo(G)
    trust = bool(getattr(a, "trust_scan", False))
    if trust:
        a.apply = True
    if a.apply and a.dry_run:
        raise events.StageError("--apply and --dry-run contradict each other")
    if stereo and a.apply and not trust:
        raise events.StageError(
            "a stereo solve is nominally metric: its scale comes from the calibrated baseline, so a LiDAR "
            "scale is not applied to it unless you say the scan is the better reference",
            hint="`hs scale --lidar SCAN` without --apply measures the baseline's scale error against the scan; "
                 "`--trust-scan` applies the scan's scale (the H1 baseline is unobservable past a metre or two); "
                 "to change the baseline itself re-solve with `hs solve --baseline-mm MM`")
    apply = (not stereo or trust) and not a.dry_run
    init = a.init or ("pairs" if a.pairs else "auto")
    pairs = None
    if init == "pairs" and not a.pairs:
        raise events.StageError("--init pairs needs --pairs", hint="--pairs 'sx,sy,sz=px,py,pz;...' with >= 3 points")
    if a.pairs:
        try:
            pairs = parse_pairs(a.pairs)
        except ValueError as e:
            raise events.StageError(str(e), hint="--pairs 'sx,sy,sz=px,py,pz;...': scan coordinates in the scan's "
                                                 "units = solve coordinates in rig.npz mm, three or more")
    pj.acquire(STAGE)               # measure under the lock: a concurrent solve would rewrite rig.npz
    events.start(STAGE, "lidar_load")
    try:
        scan = lidarlib.load_scan(a.lidar, units=a.units)
    except (ValueError, OSError) as e:
        raise events.StageError(f"cannot read the scan {os.path.basename(a.lidar)}: {e}",
                                hint="export a PLY (point cloud or mesh), OBJ or XYZ/CSV from Polycam / Scaniverse")
    # which of the file's axes is up: told, or read off the scan (the lowest band along the
    # true up is a floor). Polycam keeps ARKit's +Y; Scaniverse's PLY is +Z.
    a.scan_up_source, a.scan_up_detection = "flag", None
    if a.scan_up == "auto":
        axis, det = lidarlib.detect_up(scan.points, seed=a.seed)
        a.scan_up_detection = det
        if axis is None:
            pj.release()
            raise events.StageError(f"cannot tell which axis of the scan is up: {det['reason']}",
                                    hint="pass --scan-up y (ARKit / Polycam) or --scan-up z (Scaniverse PLY, "
                                         "Blender-style exports); scale/lidar_report.json is not written")
        a.scan_up, a.scan_up_source = axis, "detected"
    m = measure_lidar(pj, G, names, L, scan, a, init, pairs, stereo)
    metrics, checks = m["metrics"], m["checks"]

    if not (apply and m["aligned"] and m["constrained"]):
        # measured, not applied: stereo, --dry-run, or an alignment that failed or fixes no scale
        if stereo:
            to_frame, note = m["to_current"], ("the current solve units (stereo: nominally mm, off by the "
                                               "scale ratio); nothing was applied")
        else:
            to_frame, note = m["to_current"], "the current (unscaled) solve units: nothing was applied"
        m["report"]["applied"] = False
        rp, ply = _write_lidar_outputs(pj, m, scan, a, to_frame, note)
        for k, v in metrics.items():
            events.metric(STAGE, k, v)
        for c in checks:
            events.check(STAGE, c["name"], c["ok"], value=c["value"], needs_human=c.get("needs_human", False))
        events.artifact(STAGE, pj.rel(rp), "json")
        events.artifact(STAGE, pj.rel(ply), "ply")
        pj.stage(STAGE)["lidar_check"] = {
            "at": now_iso(), "argv": list(sys.argv), "scan": os.path.abspath(a.lidar), "applied": False,
            "aligned": m["aligned"], "metrics": metrics, "report": pj.rel(rp), "aligned_ply": pj.rel(ply),
            "checks": [{k: c[k] for k in ("name", "ok", "value", "needs_human") if k in c} for c in checks]}
        pj.save()
        pj.release()
        if not m["aligned"]:
            meta = scan.meta
            raise events.StageError(
                "the scan did not align with the solve: " + "; ".join(m["reasons"]),
                hint=f"the scan was read as {meta['units']} (from {meta['units_source']}): check --units; give "
                     "--pairs with 3+ points found in both (scan units = solve mm); make sure the scan covers "
                     "what the cameras saw; scale/lidar_report.json has the details")
        if apply:
            raise events.StageError(
                "the scan aligned, but what it shares with the solve does not fix the scale: " + "; ".join(m["weak"]),
                hint="the solve must see shapes that are not just floor and walls (furniture, the subject, a "
                     "doorway) inside the scanned area; or scale with --board; --min-scale-sensitivity to override")
        return

    s = float(m["scale"])
    if not (np.isfinite(s) and s > 0):
        raise events.StageError(f"the scan gave a scale of {s}: not applied")
    pj.begin(STAGE, argv=sys.argv)
    for k, v in metrics.items():
        pj.metric(STAGE, k, v)
    for c in checks:
        pj.check(STAGE, c["name"], c["ok"], value=c["value"], needs_human=c.get("needs_human", False))
    name = os.path.basename(a.lidar)
    to_m = _apply_and_mark(pj, s, "lidar", f"x{s:.6f} from LiDAR scan {name} (hs scale --lidar)")
    m["report"].update({"applied": True, "scale_to_m": to_m})
    rp, ply = _write_lidar_outputs(pj, m, scan, a, m["to_metric"], "mm, after the scale was applied")
    pj.artifact(STAGE, rp, "json")
    pj.artifact(STAGE, ply, "ply")
    rep = m["report"]
    pj.m["scale"] = {"source": "lidar", "scan": os.path.abspath(a.lidar), "scale_factor": s, "scale_to_m": to_m,
                     "up_world": rep["up"]["up_world"],
                     "ground_normal_world": (rep["ground_plane"] or {}).get("normal_solve"),
                     "scan_mm_to_solve": rep["transform"]["scan_mm_to_solve"],
                     "at": now_iso(), "argv": list(sys.argv)}
    pj.finish(STAGE, ok=True)
