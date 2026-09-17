"""hs ingest — bring a clip into a project and validate it (strategy §4.1, file-drop path).

Two routes: a clip file (``--clip``, the "dropped onto the window" path; copied or linked), or
a clip on the phone (``--phone SERIAL --remote /sdcard/DCIM/Camera/VID_…``) pulled over adb
straight into ``source/`` with byte progress and an MD5 compared against the phone's own.
Then MD5 it, ffprobe it, and refuse anything that is not one 3840x1080 video stream
whose comment tag says ``leia3d_layout=2x1`` / ``leia3d_width_per_view=1920``. The matching
calibration profile is chosen by the clip's match keys and recorded in the manifest; no
match blocks with "no calibration for this mode". Listing the phone is ``hs phone``.

Third route, the **array** source (one photograph per camera, every camera the same body and
lens): ``--frames DIR`` takes a folder of PNG/JPEG/TIFF frames named by camera, ``--r3d DIR
--take NNN`` finds every ``*_?NNN_*.RDC/*_001.R3D`` under an RDM tree, names each by its
camera position (``G007_A067`` -> ``GA``) and renders its first frame through REDline (16-bit
BT.709 TIFF, converted to PNG). Frames land in ``source/frames/``; there is no clip and
nothing to select, so ingest also marks ``select`` done and ``hs solve`` runs monocolmap.py.
"""
import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
import time

from .. import calib, events, runner
from ..project import dir_digest, md5_file, tool_versions

STAGE = "ingest"
FRAME_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
# RED clip names: <reel><cam>NNN_<take>_<hash>: G007_A067_04036V -> camera "GA", take "067"
RE_R3D_NAME = re.compile(r"^([A-Z])\d{3}_([A-Z])(\d{3})_")
# REDline (--format 1 = TIFF, --gammaCurve 1 / --colorSpace 1 = BT.709): a display-referred
# render with no look applied, the same thing the Hydrogen's own h264 gives the pipeline
REDLINE_ARGS = ["--format", "1", "--gammaCurve", "1", "--colorSpace", "1", "--frameCount", "1"]


def add_parser(sub):
    p = sub.add_parser("ingest", help="copy a clip into the project, MD5 + ffprobe + validate, pick the profile")
    p.add_argument("--clip", default=None, help="VID_*_2x1.h4v (a plain MP4)")
    p.add_argument("--phone", default=None, metavar="SERIAL", help="pull from this adb device instead of --clip")
    p.add_argument("--remote", default=None, help="with --phone: the clip's path on the phone")
    p.add_argument("--adb", default=os.environ.get("HS_ADB", "adb"))
    p.add_argument("--link", action="store_true", help="symlink instead of copying (golden test on a big clip)")
    p.add_argument("--profile", default=None, help="force a calibration profile id/path instead of matching")
    p.add_argument("--ffprobe", default=os.environ.get("HS_FFPROBE", "ffprobe"))
    p.add_argument("--frames", default=None, metavar="DIR",
                   help="array source: a folder with one frame per camera, named by camera (GA.png …)")
    p.add_argument("--r3d", default=None, metavar="DIR",
                   help="array source: an RDM tree of KOMODO-style clips; the first frame of every "
                        "camera's clip of --take is rendered through REDline")
    p.add_argument("--take", default=None, help="with --r3d: the 3-digit take number (067)")
    p.add_argument("--redline", default=os.environ.get("HS_REDLINE", "REDline"),
                   help="REDline executable (HS_REDLINE); ~/bin/REDline is tried when it is not on PATH")
    p.add_argument("--res", type=int, default=1, help="with --r3d: REDline --res (1 full, 2 half …)")
    return p


def redline_runs(exe):
    """True when this REDline starts and answers --help. A REDline whose Rosetta translation
    is broken exists, is executable and aborts before printing anything, so existence is not
    enough; this is what tells ~/bin/REDline (the re-signed copy) from /usr/local/bin/REDline."""
    try:
        r = subprocess.run([exe, "--help"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return "--format" in (r.stdout + r.stderr)


def redline_exe(name):
    """An explicit path (--redline / HS_REDLINE with a slash) must work; a bare name is looked
    for on PATH, then ~/bin, then /usr/local/bin, then inside REDCINE-X PRO — first one that runs."""
    name = os.path.expanduser(name)
    if os.sep in name:
        if not os.path.exists(name):
            raise events.StageError(f"REDline not found: {name}")
        if not redline_runs(name):
            raise events.StageError(f"REDline at {name} does not run", hint="try it by hand: REDline --help")
        return name
    tried = []
    for c in ([shutil.which(name)] if shutil.which(name) else []) + [
            os.path.expanduser("~/bin/REDline"), "/usr/local/bin/REDline",
            "/Applications/REDCINE-X Professional/REDCINE-X PRO.app/Contents/MacOS/REDline"]:
        if c in tried or not os.path.exists(c):
            continue
        tried.append(c)
        if redline_runs(c):
            return c
    raise events.StageError("no working REDline" + (f" (tried {', '.join(tried)})" if tried else " found"),
                            hint="install REDCINE-X PRO, or point HS_REDLINE / --redline at an executable that "
                                 "answers `REDline --help`; on Apple silicon a Rosetta cache fault aborts the "
                                 "installed copy — a re-signed copy in ~/bin is picked up automatically")


def r3d_clips(root, take):
    """{camera: path} of the ``*_001.R3D`` files for one take under an RDM tree."""
    take = f"{int(take):03d}"
    out = {}
    for p in sorted(glob.glob(os.path.join(root, "**", "*.R3D"), recursive=True)):
        m = RE_R3D_NAME.match(os.path.basename(p))
        if not m or m.group(3) != take or not p.endswith("_001.R3D"):
            continue
        cam = m.group(1) + m.group(2)
        if cam in out:
            raise events.StageError(f"two clips for camera {cam} take {take}: {out[cam]} and {p}")
        out[cam] = p
    if not out:
        raise events.StageError(f"no *_?{take}_*.RDC/*_001.R3D under {root}")
    return out


def transcode_r3d(exe, r3d, dst_png, res, log):
    """First frame of an R3D -> 8-bit PNG via a temporary 16-bit TIFF. cv2 reads the TIFF
    (ffmpeg would too, but cv2 is already a dependency)."""
    import cv2
    tmp = tempfile.mkdtemp(prefix="hs_redline_")
    try:
        stem = os.path.splitext(os.path.basename(dst_png))[0]
        argv = [exe, "--i", r3d, "--res", str(res), "--outDir", tmp, "--o", stem] + REDLINE_ARGS
        runner.run(argv, STAGE, log_path=log)
        tifs = sorted(f for f in os.listdir(tmp) if f.lower().endswith((".tif", ".tiff")))
        if not tifs:
            raise events.StageError(f"REDline wrote no TIFF for {os.path.basename(r3d)}",
                                    hint=f"see {log}")
        im = cv2.imread(os.path.join(tmp, tifs[0]), cv2.IMREAD_UNCHANGED)
        if im is None:
            raise events.StageError(f"cannot read REDline's TIFF {tifs[0]}")
        if im.dtype != "uint8":
            im = (im.astype("uint32") * 255 // 65535).astype("uint8")
        if im.ndim == 3 and im.shape[2] == 4:
            im = im[:, :, :3]
        cv2.imwrite(dst_png, im)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_array(a, pj):
    """--frames DIR or --r3d DIR --take NNN -> source/frames/<cam>.png, select marked done."""
    import cv2
    a.frames, a.r3d = getattr(a, "frames", None), getattr(a, "r3d", None)
    a.take, a.res = getattr(a, "take", None), getattr(a, "res", 1)
    a.redline = getattr(a, "redline", os.environ.get("HS_REDLINE", "REDline"))
    if a.frames and a.r3d:
        raise events.StageError("give --frames or --r3d, not both")
    if a.r3d and not a.take:
        raise events.StageError("--r3d needs --take NNN (the 3-digit take number in the clip names)")
    pj.begin(STAGE, argv=_argv(a))
    fdir = pj.path("source", "frames")
    os.makedirs(fdir, exist_ok=True)
    src_root = os.path.abspath(os.path.expanduser(a.r3d or a.frames))
    origin = {}
    if a.r3d:
        exe = redline_exe(a.redline)
        clips = r3d_clips(src_root, a.take)
        pj.record_tool("redline", {"path": exe})
        events.start(STAGE, "transcode")
        log = pj.log_path(STAGE)
        for i, (cam, r3d) in enumerate(sorted(clips.items())):
            events.progress(STAGE, i, len(clips), detail=f"{cam} {os.path.basename(r3d)}", step="transcode", force=True)
            transcode_r3d(exe, r3d, os.path.join(fdir, cam + ".png"), a.res, log)
            origin[cam] = r3d
        events.progress(STAGE, len(clips), len(clips), step="transcode", force=True)
    else:
        if not os.path.isdir(src_root):
            raise events.StageError(f"not a folder: {src_root}")
        events.start(STAGE, "copy")
        files = sorted(f for f in os.listdir(src_root) if f.lower().endswith(FRAME_EXTS))
        if not files:
            raise events.StageError(f"no frames ({', '.join(FRAME_EXTS)}) in {src_root}")
        for f in files:
            cam = os.path.splitext(f)[0]
            if not re.fullmatch(r"[A-Za-z0-9-]+", cam):
                raise events.StageError(f"frame name {f!r} must be letters, digits or '-' (it becomes the view name)")
            dst = os.path.join(fdir, f)
            if a.link:
                os.symlink(os.path.join(src_root, f), dst)
            else:
                shutil.copy2(os.path.join(src_root, f), dst)
            origin[cam] = os.path.join(src_root, f)

    events.start(STAGE, "probe")
    frames, sizes = [], {}
    for f in sorted(os.listdir(fdir)):
        if not f.lower().endswith(FRAME_EXTS):
            continue
        im = cv2.imread(os.path.join(fdir, f), cv2.IMREAD_COLOR)
        if im is None:
            raise events.StageError(f"unreadable frame {f}")
        cam = os.path.splitext(f)[0]
        frames.append({"camera": cam, "file": f, "width": im.shape[1], "height": im.shape[0],
                       "md5": md5_file(os.path.join(fdir, f)), "origin": origin.get(cam)})
        sizes.setdefault((im.shape[1], im.shape[0]), []).append(cam)
    digest, n = dir_digest(fdir, FRAME_EXTS)
    pj.metric(STAGE, "cameras", [fr["camera"] for fr in frames])
    pj.metric(STAGE, "frames_md5", digest)
    pj.metric(STAGE, "width", frames[0]["width"])
    pj.metric(STAGE, "height", frames[0]["height"])
    pj.check(STAGE, "one_frame_size", len(sizes) == 1,
             value=f"{frames[0]['width']}x{frames[0]['height']} x {n}" if len(sizes) == 1
             else "; ".join(f"{w}x{h}: {','.join(c)}" for (w, h), c in sizes.items()))
    pj.check(STAGE, "enough_cameras", n >= 3, value=f"{n} cameras (COLMAP needs 3+)")
    if len(sizes) != 1 or n < 3:
        raise events.StageError("array rejected", hint="one size for every camera, at least three of them")
    pj.m["profile_id"] = None
    pj.m["profile_path"] = None
    pj.m["source"] = {"kind": "array", "frames": pj.rel(fdir), "md5": digest, "original_path": src_root,
                      "take": a.take, "cameras": frames,
                      "probe": {"width": frames[0]["width"], "height": frames[0]["height"], "nb_frames": n}}
    for k, v in tool_versions().items():
        pj.record_tool(k, v)
    pj.finish(STAGE, ok=True)

    # nothing to select from a one-frame-per-camera capture: select/frames is the source set
    pj.begin("select", argv=_argv(a))
    os.makedirs(pj.frames_dir, exist_ok=True)
    for fr in frames:
        os.symlink(os.path.join(fdir, fr["file"]), os.path.join(pj.frames_dir, fr["file"]))
    pj.metric("select", "selected", n)
    pj.metric("select", "note", "array source: one frame per camera, nothing to select")
    pj.finish("select", ok=True)


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
    if getattr(a, "frames", None) or getattr(a, "r3d", None):
        if a.clip or a.phone:
            raise events.StageError("an array source (--frames / --r3d) cannot be combined with --clip / --phone")
        return run_array(a, pj)
    if bool(a.clip) == bool(a.phone):
        raise events.StageError("give exactly one of --clip, --phone SERIAL --remote PATH, --frames DIR, --r3d DIR --take NNN")
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
