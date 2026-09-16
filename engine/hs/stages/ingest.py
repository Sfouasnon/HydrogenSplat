"""hs ingest — bring a clip into a project and validate it (strategy §4.1, file-drop path).

Two routes: a clip file (``--clip``, the "dropped onto the window" path; copied or linked), or
a clip on the phone (``--phone SERIAL --remote /sdcard/DCIM/Camera/VID_…``) pulled over adb
straight into ``source/`` with byte progress and an MD5 compared against the phone's own.
Then MD5 it, ffprobe it, and refuse anything that is not one 3840x1080 video stream
whose comment tag says ``leia3d_layout=2x1`` / ``leia3d_width_per_view=1920``. The matching
calibration profile is chosen by the clip's match keys and recorded in the manifest; no
match blocks with "no calibration for this mode". Listing the phone is ``hs phone``.
"""
import json
import os
import shutil
import subprocess
import time

from .. import calib, events, runner
from ..project import md5_file, tool_versions

STAGE = "ingest"


def add_parser(sub):
    p = sub.add_parser("ingest", help="copy a clip into the project, MD5 + ffprobe + validate, pick the profile")
    p.add_argument("--clip", default=None, help="VID_*_2x1.h4v (a plain MP4)")
    p.add_argument("--phone", default=None, metavar="SERIAL", help="pull from this adb device instead of --clip")
    p.add_argument("--remote", default=None, help="with --phone: the clip's path on the phone")
    p.add_argument("--adb", default=os.environ.get("HS_ADB", "adb"))
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


def pull(a, pj, dst):
    """adb pull into dst with byte progress; returns the phone's md5 (or None if it has no md5sum)."""
    from .phone import adb, adb_exe
    exe = adb_exe(a.adb)
    r = adb(exe, "shell", f"stat -c %s '{a.remote}'", serial=a.phone)
    try:
        total = int(r.stdout.strip())
    except ValueError:
        raise events.StageError(f"not on the phone: {a.remote}", hint=(r.stdout + r.stderr).strip()[-200:])
    pj.metric(STAGE, "remote_bytes", total)
    t0 = time.monotonic()

    def tick():
        n = os.path.getsize(dst) if os.path.exists(dst) else 0
        el = time.monotonic() - t0
        rate = n / el if el > 0.5 and n else None
        events.progress(STAGE, n, total, rate=rate, eta_s=(total - n) / rate if rate else None,
                        detail=f"{n / 1e6:.1f} / {total / 1e6:.1f} MB", step="pull")

    events.start(STAGE, "pull")
    runner.run([exe, "-s", a.phone, "pull", a.remote, dst], STAGE, log_path=pj.log_path(STAGE),
               tick=tick, tick_interval=0.5)
    events.progress(STAGE, total, total, detail=f"{total / 1e6:.1f} MB", step="pull", force=True)
    got = os.path.getsize(dst)
    if got != total:
        raise events.StageError(f"pulled {got} bytes, phone has {total}", hint="replug the phone and ingest again")
    events.start(STAGE, "phone_md5")
    r = adb(exe, "shell", f"md5sum '{a.remote}'", serial=a.phone, timeout=120)
    m = r.stdout.strip().split()
    return m[0] if m and len(m[0]) == 32 else None


def run(a, pj):
    if bool(a.clip) == bool(a.phone):
        raise events.StageError("give exactly one of --clip or --phone SERIAL --remote PATH")
    if a.phone:
        if not a.remote:
            raise events.StageError("--phone needs --remote /sdcard/DCIM/Camera/VID_…_2x1.h4v")
        src = f"adb:{a.phone}:{a.remote}"
        pj.begin(STAGE, argv=_argv(a))
        dst = pj.path("source", os.path.basename(a.remote))
        phone_md5 = pull(a, pj, dst)
    else:
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
        phone_md5 = None
    events.start(STAGE, "md5")
    digest = md5_file(dst)
    if a.phone:
        pj.check(STAGE, "pull_matches_phone", phone_md5 in (None, digest),
                 value=f"md5 {digest}" + (" (phone has no md5sum; size matched)" if phone_md5 is None
                                          else " on both" if phone_md5 == digest else f" vs phone {phone_md5}"))
        if phone_md5 not in (None, digest):
            raise events.StageError("the pulled clip differs from the phone's", hint="ingest again")
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
