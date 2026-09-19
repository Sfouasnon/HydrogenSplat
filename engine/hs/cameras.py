"""cameras.json — the solved cameras in a form the app's splat viewer can read.

Swift cannot open ``rig.npz``, and a viewer that guesses poses is worse than none: the whole
point of looking at a model from capture 65 is that it is *exactly* capture 65. This writes
every view's pose and pinhole, straight from the rig.npz a model was trained against, plus
the coverage frame (subject, up) so an orbit has something to turn about.

Conventions, the same ones ``move/*.json`` uses, so one decoder serves both:
  c2w      4x4, OpenCV camera (x right, y down, z forward), **metres** — the unit of the .ply
           (rig.npz is mm; sparse points and Brush exports are rig.npz / 1000)
  K        fx, fy, cx, cy in pixels of that view's own w x h (the undistorted canvas; the two
           eyes differ by a few pixels, so the size is per view, never assumed)

``rig_npz_md5`` is the fingerprint `hs render` already trusts (render.model_matches_solve):
an archive's manifest carries the same md5, so the viewer can refuse to pose a model with
cameras from a different solve instead of showing a convincing wrong picture.
"""
import json
import os

import numpy as np

from . import coverage, rig
from .project import md5_file

SCHEMA = 1


def compute(rig_npz):
    G, names, L = rig.load(rig_npz)
    cov = coverage.compute(rig_npz)
    by_capture = {c["name"]: c for c in cov["captures"]}
    R = G["R"].astype(np.float64)
    C = G["C"].astype(np.float64)
    K = G["K"].astype(np.float64)
    if "wh" in G.files and len(G["wh"]) == len(names):
        WH = [[int(w), int(h)] for w, h in G["wh"]]
    else:                       # rig.npz from before per-view sizes were recorded
        WH = [[int(G["w"]), int(G["h"])]] * len(names)
    views = []
    for i, name in enumerate(names):
        c2w = np.eye(4)
        c2w[:3, :3] = R[i].T                    # R is world->camera
        c2w[:3, 3] = C[i] / 1000.0              # mm -> m
        cap = rig.capture_name(name)
        cv = by_capture.get(cap, {})
        views.append({
            "name": name, "capture": cap,
            "eye": name[-1] if name.endswith(("_L", "_R")) else "L",
            "w": WH[i][0], "h": WH[i][1],
            "fx": float(K[i][0, 0]), "fy": float(K[i][1, 1]),
            "cx": float(K[i][0, 2]), "cy": float(K[i][1, 2]),
            "c2w": [[float(x) for x in row] for row in c2w],
            "azimuth_deg": cv.get("azimuth_deg"), "elevation_deg": cv.get("elevation_deg"),
            "distance_mm": cv.get("distance_mm"),
        })
    return {
        "schema": SCHEMA,
        "note": "OpenCV camera convention (x right, y down, z forward), metres; one entry per "
                "view of rig.npz, in rig.npz order",
        "rig_npz": os.path.abspath(rig_npz),
        "rig_npz_md5": md5_file(rig_npz),
        "stereo": bool(cov["stereo"]),
        "subject_m": [x / 1000.0 for x in cov["subject_mm"]],
        "up_world": cov["up_world"],
        "views": views,
    }


def write(rig_npz, out_path):
    t = compute(rig_npz)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(t, f)
    os.replace(tmp, out_path)
    return t
