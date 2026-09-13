"""hs calib — stereocal.py wrapper (the v1.1 "Calibrate from board clip" back end).

Takes board photos (2x1 JPEGs) or a board *video* (frames are extracted at --every seconds),
runs stereocal.py unchanged, and writes a calibration profile JSON next to the npz with the
match keys you give. Not part of the golden test; the shipped profile was made this way by
hand (`stereocal.py --f-init 1865`).
"""
import glob
import json
import os
import re
import sys

import numpy as np

from .. import events, runner

STAGE = "calib"
RE_EYE = re.compile(r"eye (\w): rms ([\d.]+)px")
RE_STEREO = re.compile(r"stereo: rms ([\d.]+)px\s+baseline \|T\| = ([\d.]+) mm")
RE_VIEWS = re.compile(r"(\d+) stereo views usable")


def add_parser(sub):
    p = sub.add_parser("calib", help="calibrate from ChArUco board captures (stereocal.py)")
    p.add_argument("--photos", nargs="*", default=[], help="IMG_*_2x1.jpg board captures")
    p.add_argument("--video", default=None, help="a 2x1 board video; frames extracted with ffmpeg")
    p.add_argument("--every", type=float, default=1.0, help="video: seconds between extracted frames")
    p.add_argument("--square", type=float, default=15.45, help="mm")
    p.add_argument("--marker", type=float, default=12.36, help="mm")
    p.add_argument("--min-corners", type=int, default=20)
    p.add_argument("--f-init", type=float, default=1865.0)
    p.add_argument("-o", "--out", required=True, help="output .npz (a .json profile is written beside it)")
    p.add_argument("--profile-id", default=None)
    p.add_argument("--match", default="leia3d_layout=2x1;leia3d_width_per_view=1920;leia3d_height_per_view=1080;leia3d_recording_software_version=1.18.2")
    p.add_argument("--ffmpeg", default=os.environ.get("HS_FFMPEG", "ffmpeg"))
    return p


def run(a, pj=None):
    photos = []
    for pat in a.photos:
        photos += sorted(glob.glob(os.path.expanduser(pat)))
    out = os.path.abspath(os.path.expanduser(a.out))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    log = os.path.join(os.path.dirname(out), "calib.log")
    if a.video:
        events.start(STAGE, "extract")
        fdir = os.path.splitext(out)[0] + "_frames"
        os.makedirs(fdir, exist_ok=True)
        runner.run([a.ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", os.path.expanduser(a.video),
                    "-vf", f"fps=1/{a.every:g}", "-q:v", "2", os.path.join(fdir, "IMG_%04d_2x1.jpg")],
                   STAGE, log_path=log)
        photos += sorted(glob.glob(os.path.join(fdir, "*.jpg")))
    if not photos:
        raise events.StageError("no board images", hint="--photos 'board/*.jpg' or --video board.h4v")
    events.start(STAGE, "stereocal")
    st = {}

    def on_line(line):
        m = RE_EYE.search(line)
        if m:
            st["rms_" + m.group(1)] = float(m.group(2))
        m = RE_STEREO.search(line)
        if m:
            st["rms_stereo"], st["baseline"] = float(m.group(1)), float(m.group(2))
        m = RE_VIEWS.search(line)
        if m:
            st["views"] = int(m.group(1))

    argv = runner.python_argv("stereocal.py", *photos, "--square", a.square, "--marker", a.marker,
                              "--min-corners", a.min_corners, "--f-init", a.f_init, "-o", out)
    runner.run(argv, STAGE, log_path=log, on_line=on_line)
    for k, v in st.items():
        events.metric(STAGE, k, v)
    # profile JSON in the shipped layout
    z = np.load(out, allow_pickle=True)
    match = {}
    for kv in a.match.split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            match[k.strip()] = int(v) if v.strip().isdigit() else v.strip()
    from ..calib import REPO_ROOT  # noqa
    sys.path.insert(0, runner.VENDORED)
    from rigcolmap import rig_config  # the same function that writes the COLMAP rig config
    calib = {"KL": z["KL"].astype(float), "dL": z["dL"].astype(float).ravel(),
             "KR": z["KR"].astype(float), "dR": z["dR"].astype(float).ravel(),
             "R": z["R"].astype(float), "T_mm": z["T"].astype(float).ravel(),
             "size": tuple(int(v) for v in z["size"])}
    prof = {
        "profile_id": a.profile_id or os.path.splitext(os.path.basename(out))[0],
        "device": "RED Hydrogen One (H1A1000)", "mode": "Holocam 3D video, 2x1 side-by-side",
        "match": match,
        "source": {"method": f"hs calib (stereocal.py --f-init {a.f_init:g})",
                   "board": f"ChArUco 8x11 DICT_4X4_100, square {a.square} mm, marker {a.marker} mm",
                   "n_views": st.get("views"), "rms_L_px": st.get("rms_L"), "rms_R_px": st.get("rms_R"),
                   "rms_stereo_px": st.get("rms_stereo")},
        "units": {"intrinsics": "pixels at 1920x1080 per eye", "T_mm": "millimetres",
                  "cam_from_rig_translation": "metres"},
        "stereocal": {"KL": calib["KL"].tolist(), "dL": calib["dL"].tolist(), "KR": calib["KR"].tolist(),
                      "dR": calib["dR"].tolist(), "R": calib["R"].tolist(), "T_mm": calib["T_mm"].tolist(),
                      "size": list(calib["size"])},
        "colmap_rig": rig_config(calib),
    }
    pjson = os.path.splitext(out)[0] + ".profile.json"
    json.dump(prof, open(pjson, "w"), indent=1)
    events.artifact(STAGE, pjson, "profile")
    events.artifact(STAGE, out, "npz")
