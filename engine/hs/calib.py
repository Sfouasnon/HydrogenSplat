"""Calibration profiles (strategy §4.2).

A profile is a JSON file in ``profiles/``: per-eye OPENCV intrinsics, the R-from-L stereo
transform, its provenance, and the *match keys* that tie it to a capture mode. Ingest picks
the profile whose match keys equal the clip's ``leia3d_*`` comment tags; no match blocks
rather than guesses.

``rigcolmap.py sfm`` reads a stereocal ``.npz`` (KL dL KR dR R T size, T in mm) and is
vendored unchanged, so ``profile_to_npz`` writes that file from the profile's ``stereocal``
block. Numerically it is the same calibration ``h1_video_stereo.npz`` carried.
"""
import glob
import json
import os
import re

import numpy as np

from . import events

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROFILE_DIRS = [
    os.environ.get("HS_PROFILES", ""),
    os.path.join(REPO_ROOT, "profiles"),
    os.path.expanduser("~/Library/Application Support/HydrogenSplat/profiles"),
]

MATCH_KEYS = ("leia3d_layout", "leia3d_width_per_view", "leia3d_height_per_view",
              "leia3d_recording_software_version")


def profile_dirs():
    return [d for d in PROFILE_DIRS if d and os.path.isdir(d)]


def list_profiles():
    out = {}
    for d in profile_dirs():
        for p in sorted(glob.glob(os.path.join(d, "*.json"))):
            try:
                j = json.load(open(p))
            except Exception:
                continue
            pid = j.get("profile_id") or os.path.splitext(os.path.basename(p))[0]
            out.setdefault(pid, (p, j))
    return out


def load_profile(profile_id_or_path):
    if os.path.isfile(profile_id_or_path):
        j = json.load(open(profile_id_or_path))
        return profile_id_or_path, j
    profs = list_profiles()
    if profile_id_or_path in profs:
        return profs[profile_id_or_path]
    raise events.StageError(f"no calibration profile '{profile_id_or_path}'",
                            hint="available: " + ", ".join(sorted(profs)) if profs else
                            "no profiles/ directory found")


def parse_leia_comment(comment):
    """'leia3d_layout=2x1;leia3d_width_per_view=1920;...' -> dict with ints where numeric."""
    tags = {}
    for kv in (comment or "").split(";"):
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        k, v = k.strip(), v.strip()
        tags[k] = int(v) if re.fullmatch(r"-?\d+", v) else v
    return tags


def match_profile(tags):
    """Return (profile_id, path, profile) whose match keys equal the clip's tags, else None."""
    for pid, (path, j) in list_profiles().items():
        want = j.get("match", {})
        if all(str(tags.get(k)) == str(want.get(k)) for k in MATCH_KEYS):
            return pid, path, j
    return None


def profile_to_npz(profile, out_path, baseline_mm=None):
    """Write the stereocal-style npz that rigcolmap.py --calib expects.

    `baseline_mm` rescales T to a different separation without touching its direction. The
    baseline is the weakest number in a short ChArUco session — this profile's own notes give a
    best joint fit of 11.71 mm with a 1-sigma of 10.1-13.3 mm against the 10.595 mm it ships —
    and it is not "pure gauge" in a rig solve: it fixes the relative pose of two cameras that
    observe the scene at the same instant, so an error in it is a systematic, depth-dependent,
    purely horizontal disagreement between the eyes, which no amount of bundle adjustment can
    absorb while the scale is restored after BA."""
    s = profile["stereocal"]
    T = np.array(s["T_mm"], float)
    if baseline_mm:
        n = float(np.linalg.norm(T))
        if n > 1e-9:
            T = T * (float(baseline_mm) / n)
    np.savez(out_path,
             KL=np.array(s["KL"], float), dL=np.array(s["dL"], float),
             KR=np.array(s["KR"], float), dR=np.array(s["dR"], float),
             R=np.array(s["R"], float), T=T,
             size=np.array(s["size"], int))
    return out_path


def baseline_mm(profile):
    return float(np.linalg.norm(np.array(profile["stereocal"]["T_mm"], float)))
