"""hs cameras — write the solved cameras as JSON for the app's splat viewer (hs/cameras.py),
or choose which captures to hold out of training.

    hs cameras -p P                   train/dataset/rig.npz      -> viewer/cameras_current.json
    hs cameras -p P --archive NAME    archive/NAME/rig.npz       -> viewer/cameras_NAME.json
    hs cameras -p P --holdout N [--method fps|interval|azimuth] [--seed 0] [--write]
                                      choose N hold-out captures; --write -> solve/holdout.json

Read-only towards the pipeline: it takes no project lock, touches no stage and leaves
manifest.json alone, so the viewer can ask for cameras while `hs train` holds the lock. The
output goes under viewer/, never into archive/NAME — an archive is written whole or not at
all, and nothing is added to one afterwards.

Hold-outs. "Every 10th capture from 5" left the 09-16 head's +90…+135 band with no hold-out
at all, so the one place the model was weakest was never scored. ``--holdout N`` picks by
where the cameras are instead of by index (coverage.select_holdout): ``fps`` is farthest-point
sampling over the camera centres, seeded at the capture farthest from their centroid;
``azimuth`` takes one capture per equal-width azimuth band; ``interval`` is the old every-Nth,
for comparison. It prints the choice and its spread (stderr, for a person) and emits the same
numbers as metrics; ``--write`` saves ``solve/holdout.json`` ``{method, n, names, captures,
exclude, stats}``, which ``hs train --exclude @holdout`` and ``hs views --captures holdout``
read. A re-solve deletes solve/, and the hold-out file with it: capture indices do not survive
a new solve. The file is the one thing this command writes outside viewer/.
"""
import json
import os
import sys

import numpy as np

from .. import cameras, coverage, events, rig
from ..project import md5_file, now_iso

STAGE = "cameras"


def add_parser(sub):
    p = sub.add_parser("cameras", help="write viewer/cameras_*.json (poses + pinholes, metres) for the app's splat viewer, "
                                       "or choose hold-out captures (--holdout N)")
    p.add_argument("--archive", default=None, help="use archive/NAME/rig.npz (the cameras that model was trained against)")
    p.add_argument("--holdout", type=int, default=None, metavar="N",
                   help="choose N captures to hold out of training instead of writing cameras json")
    p.add_argument("--method", choices=coverage.HOLDOUT_METHODS, default="fps",
                   help="fps: farthest-point over camera centres (default); interval: every Nth; azimuth: one per band")
    p.add_argument("--seed", type=int, default=0, help="interval: phase offset (0 = half a step in, the old cap005,015,...)")
    p.add_argument("--write", action="store_true", help="save the choice to solve/holdout.json")
    return p


def run(a):
    if not getattr(a, "project", None):
        raise events.StageError("hs cameras needs --project DIR")
    root = os.path.abspath(os.path.expanduser(a.project))
    if not os.path.exists(os.path.join(root, "manifest.json")):
        raise events.StageError(f"{root} is not a project (no manifest.json)")
    events.start(STAGE)
    if getattr(a, "holdout", None) is not None:
        return holdout(a, root)
    if a.archive:
        if os.sep in a.archive or a.archive.startswith("."):
            raise events.StageError(f"archive name must be a plain folder name, not {a.archive!r}")
        rig_npz = os.path.join(root, "archive", a.archive, "rig.npz")
        out = os.path.join(root, "viewer", f"cameras_{a.archive}.json")
    else:
        rig_npz = os.path.join(root, "train", "dataset", "rig.npz")
        out = os.path.join(root, "viewer", "cameras_current.json")
    if not os.path.exists(rig_npz):
        raise events.StageError(f"no {os.path.relpath(rig_npz, root)}",
                                hint="hs solve first" if not a.archive else "that archive has no rig.npz")
    t = cameras.write(rig_npz, out)
    events.metric(STAGE, "views", len(t["views"]))
    events.metric(STAGE, "stereo", t["stereo"])
    events.metric(STAGE, "rig_npz_md5", t["rig_npz_md5"])
    events.artifact(STAGE, os.path.relpath(out, root), "json")
    return t


def holdout(a, root):
    rig_npz = os.path.join(root, "train", "dataset", "rig.npz")
    if not os.path.exists(rig_npz):
        raise events.StageError("no train/dataset/rig.npz", hint="hs solve first")
    G, names, L = rig.load(rig_npz)
    stereo = rig.is_stereo(G)
    C = G["C"].astype(np.float64)[L]
    cov_path = os.path.join(root, "solve", "coverage.json")
    cov = json.load(open(cov_path)) if os.path.exists(cov_path) else coverage.compute(rig_npz)
    caps = sorted(cov["captures"], key=lambda c: c["capture"])
    if len(caps) != len(L):
        raise events.StageError(f"solve/coverage.json has {len(caps)} captures, rig.npz {len(L)}",
                                hint="the two are from different solves; re-run hs solve")
    az = [c["azimuth_deg"] for c in caps]
    try:
        chosen = coverage.select_holdout(C, a.holdout, a.method, azimuth=az, seed=a.seed)
    except ValueError as e:
        raise events.StageError(str(e))
    stats = coverage.holdout_stats(C, chosen, azimuth=az)
    cap_names = [rig.capture_name(names[int(L[i])]) for i in chosen]
    exclude = [f"{e}/{n}" for n in cap_names for e in (("L", "R") if stereo else ("L",))]

    # the part a person reads; stdout carries only events
    w = sys.stderr.write
    w(f"hold-out {a.method}: {len(chosen)} of {len(C)} captures\n")
    for i, n in zip(chosen, cap_names):
        c = caps[i]
        w(f"  {i:4d}  {n:10s} az {c['azimuth_deg']:+7.1f}  el {c['elevation_deg']:+6.1f}  {c['distance_mm']:7.1f} mm\n")
    w(f"  nearest other hold-out   min {stats['nn_holdout_mm']['min']} mm, median {stats['nn_holdout_mm']['median']} mm\n")
    w(f"  nearest training capture min {stats['holdout_to_rest_mm']['min']} mm, median {stats['holdout_to_rest_mm']['median']} mm\n")
    for b in stats.get("bands", []):
        w(f"  az {b['azimuth_deg'][0]:+5d}…{b['azimuth_deg'][1]:+5d}  {b['holdouts']} of {b['captures']}"
          f"{'   <- no hold-out' if not b['holdouts'] else ''}\n")

    events.metric(STAGE, "holdout_method", a.method)
    events.metric(STAGE, "holdout_captures", chosen)
    events.metric(STAGE, "holdout_names", cap_names)
    events.metric(STAGE, "holdout_nn_min_mm", stats["nn_holdout_mm"]["min"])
    events.metric(STAGE, "holdout_nn_median_mm", stats["nn_holdout_mm"]["median"])
    events.metric(STAGE, "holdout_to_rest_min_mm", stats["holdout_to_rest_mm"]["min"])
    events.metric(STAGE, "holdout_to_rest_median_mm", stats["holdout_to_rest_mm"]["median"])
    events.metric(STAGE, "holdout_bands", stats.get("bands"))
    events.check(STAGE, "every_azimuth_band_has_a_holdout", not stats.get("empty_bands"),
                 value=("bands with captures but no hold-out: "
                        + ", ".join(f"{lo}…{hi}" for lo, hi in stats["empty_bands"])
                        if stats.get("empty_bands") else f"all {len(stats.get('bands', []))} covered bands hold one"))
    out = {"method": a.method, "n": len(chosen), "seed": a.seed, "names": cap_names, "captures": chosen,
           "exclude": exclude, "stereo": stereo, "stats": stats,
           "rig_npz_md5": md5_file(rig_npz), "created": now_iso()}
    if a.write:
        p = os.path.join(root, coverage.HOLDOUT_JSON)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=1)
        os.replace(tmp, p)
        events.artifact(STAGE, coverage.HOLDOUT_JSON, "json")
    return out
