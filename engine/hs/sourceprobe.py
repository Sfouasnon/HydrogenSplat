"""What is this file or folder, as a source for a project?

`hs ingest` takes four things, and until 2026-10-04 only the first could be given to the app:

  stereo   a RED Hydrogen One 3D clip (VID_*_2x1.h4v): one 3840x1080 stream, two eyes side by side,
           tagged ``leia3d_layout=2x1`` in the container's comment;
  video    any other video with one picture stream: one ordinary camera walking round the subject
           (an iPhone orbit). SDR, or HLG HDR; PQ is not decoded (select_frames.py --hdr);
  stills   a folder of photographs: one camera's set, or one frame from each camera of an array;
  r3d      a RED camera array: a folder of R3D clips, one per camera, from which one take is used.

This module is the part of that decision that needs no project: reading a video's stream, turning
file names into view names, listing the takes under a RED folder. `hs source PATH`
(stages/source.py) puts it together for the app's New Project page; `hs ingest` uses the same
functions, so what the page says about a source is what ingest then does with it.
"""
import datetime
import json
import os
import re
import shutil
import subprocess

from . import calib, events

VIDEO_EXTS = (".h4v", ".mp4", ".mov", ".m4v")
STILL_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
HEIC_EXTS = (".heic", ".heif")          # read through macOS's sips (convert_heic)
R3D_EXTS = (".r3d",)
# RED clip names: <reel><cam>NNN_<take>_<hash>: G007_A067_04036V -> camera "GA", take "067"
RE_R3D_NAME = re.compile(r"^([A-Z])\d{3}_([A-Z])(\d{3})_")
RE_UNSAFE = re.compile(r"[^A-Za-z0-9-]+")
MIN_VIDEO_FRAMES = 30


def hidden(name):
    """.DS_Store, and the ._IMG_0001.JPG twins macOS leaves on a camera card."""
    return name.startswith(".")


# --------------------------------------------------------------------------- video
def ffprobe_json(ffprobe_bin, path):
    """ffprobe's format and streams for a file, as a dict. StageError when ffprobe is missing or
    cannot read the file."""
    exe = shutil.which(os.path.expanduser(ffprobe_bin)) or ffprobe_bin
    argv = [exe, "-v", "error", "-show_format", "-show_streams", "-of", "json", path]
    try:
        out = subprocess.run(argv, capture_output=True, text=True, check=True).stdout
    except FileNotFoundError:
        raise events.StageError(f"ffprobe not found ({ffprobe_bin})", hint="brew install ffmpeg")
    except subprocess.CalledProcessError as e:
        raise events.StageError(f"ffprobe cannot read {os.path.basename(path)}",
                                hint=(e.stderr or "").strip()[-300:] or "not a video file?")
    try:
        return json.loads(out)
    except ValueError:
        raise events.StageError(f"ffprobe gave no JSON for {os.path.basename(path)}")


def leia_tags(probe):
    """The Hydrogen's own tags from the container comment ({} for any other camera's file)."""
    tags = calib.parse_leia_comment(((probe.get("format") or {}).get("tags") or {}).get("comment", ""))
    return {k: v for k, v in tags.items() if str(k).startswith("leia3d_")}


def video_facts(probe):
    """-> (facts, problems) for the picture stream of an ordinary (one-view) video.

    facts: width, height (as displayed: a portrait iPhone clip is stored 3840x2160 with a 90 degree
    display matrix and reads 2160x3840), codec, pix_fmt, fps, nb_frames, duration_s, rotation,
    transfer, hdr ("hlg" / "pq" / None). problems: why it cannot be used, empty when it can."""
    from . import select_frames
    vids = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"
            and not (s.get("disposition") or {}).get("attached_pic")]
    if not vids:
        return None, ["no video stream in this file"]
    v = select_frames.parse_probe(json.dumps({"streams": [vids[0]]})) or {}
    fmt = probe.get("format") or {}
    try:
        duration = float(fmt.get("duration") or vids[0].get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0.0
    nb = v.get("nb_frames")
    if not nb and duration and v.get("fps"):
        nb = int(round(duration * v["fps"]))
    hdr = select_frames.hdr_kind(v)
    facts = {"width": v.get("width") or 0, "height": v.get("height") or 0, "codec": v.get("codec"),
             "pix_fmt": v.get("pix_fmt"), "fps": round(v["fps"], 3) if v.get("fps") else None,
             "nb_frames": int(nb or 0), "duration_s": round(duration, 3), "rotation": v.get("rotation") or 0.0,
             "transfer": v.get("transfer"), "hdr": hdr}
    problems = []
    if not facts["width"] or not facts["height"]:
        problems.append("the video stream has no picture size")
    if hdr == "pq":
        problems.append("this clip is PQ HDR (HDR10, or Dolby Vision without an HLG base layer); only HLG HDR "
                        "is decoded. On an iPhone: Settings > Camera > Record Video > HDR Video records HLG")
    if facts["nb_frames"] <= 1:
        # ffprobe reads one picture as a "video" of no length, or of one frame: an EXR, a DNG, a BMP
        problems.append("no frame count or duration in this file: a single picture (EXR, DNG, BMP …) "
                        "or a raw stream, not a clip")
    elif facts["nb_frames"] < MIN_VIDEO_FRAMES:
        problems.append(f"only {facts['nb_frames']} frames: too short to pick a set of views from")
    return facts, problems


def video_line(f):
    """One line for a person: 2160×3840 · hevc · 30 fps · 2,541 frames · 84.7 s · HLG HDR."""
    parts = [f"{f['width']}×{f['height']}", f.get("codec") or "?"]
    if f.get("fps"):
        parts.append(f"{f['fps']:g} fps")
    if f.get("nb_frames"):
        parts.append(f"{f['nb_frames']:,} frames")
    if f.get("duration_s"):
        parts.append(f"{f['duration_s']:.1f} s")
    if f.get("hdr"):
        parts.append(f"{f['hdr'].upper()} HDR")
    return " · ".join(parts)


def recorded_date(path, probe=None):
    """YYYY-MM-DD for the project folder: the container's creation time when it has one, else the
    file's modification date."""
    ct = (((probe or {}).get("format") or {}).get("tags") or {}).get("creation_time")
    if ct:
        try:
            t = datetime.datetime.fromisoformat(str(ct).replace("Z", "+00:00"))
            if t.year >= 1990:                   # a camera with no clock writes 1904 or 1970
                return t.astimezone().strftime("%Y-%m-%d")
        except ValueError:
            pass
    try:
        return datetime.date.fromtimestamp(os.path.getmtime(path)).isoformat()
    except OSError:
        return datetime.date.today().isoformat()


# --------------------------------------------------------------------------- names
def safe_stem(stem):
    """A file name as a view name: letters, digits and '-' only. An underscore would collide with
    the _L / _R suffix every view carries, so IMG_0001 becomes IMG-0001."""
    s = RE_UNSAFE.sub("-", stem).strip("-")
    return s or "frame"


def safe_names(stems):
    """One view name per stem, in order: every name safe, and different from the others (two
    files that come out the same — IMG_0001.jpg beside IMG-0001.jpg — get -2, -3 …)."""
    out, used = [], set()
    for stem in stems:
        base = safe_stem(stem)
        name, k = base, 2
        while name.lower() in used:
            name = f"{base}-{k}"
            k += 1
        used.add(name.lower())
        out.append(name)
    return out


# --------------------------------------------------------------------------- HEIC
def convert_heic(src, dst_jpg, sips_bin=None):
    """An iPhone photograph (HEIC) as a JPEG, through macOS's sips. OpenCV has no HEIC reader."""
    name = sips_bin or os.environ.get("HS_SIPS", "sips")
    exe = shutil.which(os.path.expanduser(name))
    if not exe:
        raise events.StageError(f"{os.path.basename(src)} is HEIC, and sips ({name}) is not there to convert it",
                                hint="sips is part of macOS; elsewhere, export the photographs as JPEG first")
    r = subprocess.run([exe, "-s", "format", "jpeg", "-s", "formatOptions", "95", src, "--out", dst_jpg],
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(dst_jpg):
        raise events.StageError(f"sips could not convert {os.path.basename(src)}",
                                hint=(r.stderr or r.stdout or "").strip()[-300:])


# --------------------------------------------------------------------------- RED
R3D_DEPTH = 6        # root / camera / volume.RDM / clip.RDC / file is four; two to spare


def r3d_files(root, max_depth=R3D_DEPTH):
    """Every .R3D under a folder, sorted. A walk, not a glob: it does not follow links (a card
    can hold one to /), does not go deeper than a card does, skips hidden folders, and takes a
    folder name with [brackets] in it as a name."""
    out = []
    root = os.path.abspath(root)
    base = root.rstrip(os.sep).count(os.sep)
    for d, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(x for x in dirs if not hidden(x))
        if d.rstrip(os.sep).count(os.sep) - base >= max_depth:
            dirs[:] = []
        out += [os.path.join(d, f) for f in files if f.upper().endswith(".R3D") and not hidden(f)]
    return sorted(out)


def r3d_index(root):
    """({take: {camera: path of its _001.R3D}}, {take: why it cannot be used}) for every take
    under a RED media folder. A camera with two clips of one take spoils that take, not the rest."""
    takes, bad = {}, {}
    for p in r3d_files(root):
        base = os.path.basename(p)
        m = RE_R3D_NAME.match(base)
        if not m or not base.endswith("_001.R3D"):
            continue
        cam, take = m.group(1) + m.group(2), m.group(3)
        cams = takes.setdefault(take, {})
        if cam in cams:
            bad.setdefault(take, f"two clips for camera {cam} take {take}: {cams[cam]} and {p}")
            continue
        cams[cam] = p
    return takes, bad


def r3d_root(path):
    """A dropped .R3D file, .RDC clip or .RDM volume -> (the folder that holds the array, take).

    Climbs out of the clip and volume folders. Then, only if a clip was given and its take has
    under three cameras there, up to two levels more for as long as each adds cameras and until
    there are three: one card per camera is one folder per camera. It stops at three so that the
    shoot in the next folder, with the same take number, is not swept in; it never climbs past a
    mount point or the home folder. A folder that was given is taken as given."""
    take = None
    p = os.path.abspath(path)
    m = RE_R3D_NAME.match(os.path.basename(p))
    if m:
        take = m.group(3)
    inside = os.path.isfile(p)                  # was a clip, or part of one, given?
    if inside:
        p = os.path.dirname(p)
    while os.path.basename(p).upper().endswith((".RDC", ".RDM")):
        m = RE_R3D_NAME.match(os.path.basename(p))
        if m and take is None:
            take = m.group(3)
        p = os.path.dirname(p)
        inside = True

    def cameras(d):
        idx, _bad = r3d_index(d)
        return len(idx.get(take, {})) if take else max((len(c) for c in idx.values()), default=0)

    home = os.path.expanduser("~")
    best, n = p, cameras(p)
    for _ in range(2 if inside else 0):
        up = os.path.dirname(best)
        if n >= 3 or up == best or os.path.ismount(best) or up in (home, os.path.dirname(home), "/Volumes", os.sep):
            break
        k = cameras(up)
        if k <= n:
            break
        best, n = up, k
    return best, take
