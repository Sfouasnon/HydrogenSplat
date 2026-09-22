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
"""
import contextlib
import json
import os
import sys

import numpy as np

from .. import board as boardlib
from .. import coverage, events, rig
from ..project import now_iso

STAGE = "scale"
MAX_MAD_REL = 0.005          # 0.5 %: the pair ratios agree, i.e. the board is rigid in the solve
MAX_VIEW_RMS_PX = 2.0        # a view whose triangulated corners miss by more is posed badly
REFUSE_MAD_REL = 0.05        # 5 %: past this the factor is refused outright, not applied with a failed check


def add_parser(sub):
    p = sub.add_parser("scale", help="metric scale from a ChArUco board seen in the training views")
    add_board_args(p)
    p.add_argument("--eye", choices=("L", "R", "both"), default="L",
                   help="which views to detect in (a mono or array rig has only L)")
    p.add_argument("--min-views", type=int, default=3, help="triangulate a corner seen in at least this many views")
    p.add_argument("--dry-run", action="store_true", help="measure and report, write nothing")
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
        raise events.StageError("no board given", hint="--board SX,SY,SQUARE_MM,MARKER_MM[,DICT], e.g. 7,5,40,30")
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
    _mark_solve_scaled(pj, s, to_m, spec)

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
        monocolmap.finish_dataset(rec, pj.dataset_dir, write_binary=True)
    src = pj.path("solve", "sparse", "rig")
    if os.path.isdir(src) and any(f.endswith((".bin", ".txt")) for f in os.listdir(src)):
        r0 = pycolmap.Reconstruction(src)
        r0.transform(sim)
        r0.write(src)
        if os.path.exists(os.path.join(src, "points3D.ply")):
            r0.export_PLY(os.path.join(src, "points3D.ply"))


def _mark_solve_scaled(pj, s, to_m, spec):
    """solve's own `scene_scaled` record is what `hs masks` and the app read: set it ok, with its
    source, instead of leaving an unscaled solve's failure standing next to a scaled dataset."""
    st = pj.stage("solve")
    st.setdefault("metrics", {})["scale_to_m"] = to_m
    st["metrics"]["scale_source"] = "board"
    checks = [c for c in st.get("checks", []) if c.get("name") != "scene_scaled"]
    rec = {"name": "scene_scaled", "ok": True, "source": "board",
           "value": f"x{s:.6f} from board {spec.text()} (hs scale)"}
    checks.append(rec)
    st["checks"] = checks
    pj.check(STAGE, "scene_scaled", True, value=rec["value"], source="board")
    pj.stage(STAGE)["checks"][-1]["source"] = "board"
