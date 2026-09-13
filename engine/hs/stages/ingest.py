"""hs ingest — bring a clip into a project and validate it (strategy §4.1, file-drop path).

M0 covers the "clip dropped onto the window" route: copy (or link) the .h4v into
``source/``, MD5 it, ffprobe it, and refuse anything that is not one 3840x1080 video stream
whose comment tag says ``leia3d_layout=2x1`` / ``leia3d_width_per_view=1920``. The matching
calibration profile is chosen by the clip's match keys and recorded in the manifest; no
match blocks with "no calibration for this mode". adb listing / pulling is M1.
"""
import json
import os
import shutil
import subprocess

from .. import calib, events
from ..project import md5_file, tool_versions

STAGE = "ingest"


def add_parser(sub):
    p = sub.add_parser("ingest", help="copy a clip into the project, MD5 + ffprobe + validate, pick the profile")
    p.add_argument("--clip", required=True, help="VID_*_2x1.h4v (a plain MP4)")
    p.add_argument("--link", action="store_true", help="symlink instead of copying (golden test on a big clip)")
    p.add_argument("--profile", default=None, help="force a calibration profile id/path instead of matching")
    p.add_argument("--ffprobe", default=os.environ.get("HS_FFPROBE", "ffprobe"))
    return p


def ffprobe(ffprobe_bin, path):
    exe = shutil.which(os.path.expanduser(ffprobe_bin)) or ffprobe_bin
    argv = [exe, "-v", "error", "-show_entries",
            "format=duration,size,format_name:format_tags:stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames",
            "-of", "json", path]
    try:
        out = subprocess.run(argv, capture_output=True, text=True, check=True).stdout
    except FileNotFoundError:
        raise events.StageError(f"ffprobe not found ({ffprobe_bin})", hint="brew install ffmpeg")
    except subprocess.CalledProcessError as e:
        raise events.StageError(f"ffprobe failed on {path}", hint=e.stderr.strip()[-300:])
    return json.loads(out)


def validate_probe(probe):
    """Returns (ok, problems, info) — info carries the parsed leia tags and video stream facts."""
    problems = []
    vids = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"]
    if len(vids) != 1:
        problems.append(f"expected one video stream, found {len(vids)}")
    v = vids[0] if vids else {}
    w, h = int(v.get("width", 0) or 0), int(v.get("height", 0) or 0)
    if (w, h) != (3840, 1080):
        problems.append(f"video is {w}x{h}, expected 3840x1080 (2x1 side-by-side)")
    tags = calib.parse_leia_comment(probe.get("format", {}).get("tags", {}).get("comment", ""))
    if tags.get("leia3d_layout") != "2x1":
        problems.append(f"leia3d_layout={tags.get('leia3d_layout')!r}, expected '2x1' (a stills-mode or 4V file?)")
    if tags.get("leia3d_width_per_view") != 1920:
        problems.append(f"leia3d_width_per_view={tags.get('leia3d_width_per_view')!r}, expected 1920")
    fr = v.get("avg_frame_rate", "0/1")
    try:
        num, den = fr.split("/")
        fps = float(num) / float(den) if float(den) else 0.0
    except Exception:
        fps = 0.0
    info = {"width": w, "height": h, "codec": v.get("codec_name"), "fps": round(fps, 3),
            "nb_frames": int(v.get("nb_frames", 0) or 0),
            "duration_s": float(probe.get("format", {}).get("duration", 0) or 0),
            "leia": tags}
    return not problems, problems, info


def run(a, pj):
    src = os.path.abspath(os.path.expanduser(a.clip))
    if not os.path.isfile(src):
        raise events.StageError(f"clip not found: {src}")
    pj.begin(STAGE, argv=_argv(a))
    events.start(STAGE, "copy")
    dst = pj.path("source", os.path.basename(src))
    if a.link:
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)
    events.start(STAGE, "md5")
    digest = md5_file(dst)
    with open(dst + ".md5", "w") as f:
        f.write(f"{digest}  {os.path.basename(dst)}\n")
    pj.metric(STAGE, "clip_md5", digest)
    pj.metric(STAGE, "clip_bytes", os.path.getsize(dst))

    events.start(STAGE, "probe")
    probe = ffprobe(a.ffprobe, dst)
    json.dump(probe, open(pj.path("source", "probe.json"), "w"), indent=1)
    pj.artifact(STAGE, pj.path("source", "probe.json"), "json")
    ok, problems, info = validate_probe(probe)
    for k in ("width", "height", "fps", "nb_frames", "duration_s", "codec"):
        pj.metric(STAGE, k, info[k])
    pj.check(STAGE, "clip_is_2x1_video", ok, value="3840x1080 2x1" if ok else "; ".join(problems))
    if not ok:
        raise events.StageError("clip rejected: " + "; ".join(problems),
                                hint="only Holocam 3D *video* in 2x1 layout is supported in v1")

    # profile
    if a.profile:
        path, prof = calib.load_profile(a.profile)
        pid = prof.get("profile_id") or os.path.basename(path)
        pj.check(STAGE, "profile_forced", True, value=pid)
    else:
        m = calib.match_profile(info["leia"])
        if m is None:
            want = {k: info["leia"].get(k) for k in calib.MATCH_KEYS}
            pj.check(STAGE, "profile_matched", False, value=json.dumps(want))
            raise events.StageError("no calibration for this mode: " + json.dumps(want),
                                    hint="shoot a ChArUco board clip in this mode and add a profile (v1.1), "
                                         "or force one with --profile if you know it applies")
        pid, path, prof = m
        pj.check(STAGE, "profile_matched", True, value=pid)
    pj.m["profile_id"] = pid
    pj.m["profile_path"] = path
    pj.m["source"] = {"clip": pj.rel(dst), "md5": digest, "original_path": src,
                      "probe": info}
    for k, v in tool_versions().items():
        pj.record_tool(k, v)
    pj.finish(STAGE, ok=True)


def _argv(a):
    import sys
    return sys.argv
