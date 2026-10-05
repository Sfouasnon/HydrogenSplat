"""hs source PATH — what a file or folder would be taken as, before there is a project.

    hs source ~/Movies/IMG_2525.MOV
    hs source ~/Pictures/helmet_stills
    hs source "~/Desktop/Camera Footage/RED_Footage"

Emits one ``source`` metric (nothing is written, no lock is taken): the kind (``stereo`` a
Hydrogen 3D clip, ``video`` one ordinary camera's clip, ``stills`` a folder of photographs,
``r3d`` a RED camera array, ``unknown``), a title and one line of facts for a person, whether
``hs ingest`` would accept it and why not, and the ``hs ingest`` arguments that bring it in. The
app's New Project page shows exactly this; ``hs/sourceprobe.py`` has the rules, and ``hs ingest``
applies the same ones.

``hs source -p P --subject KIND`` (matte | glossy | bright | person | scene) is the Footage step's
one choice about the subject, kept in the manifest as ``project.subject_kind``. ``hs exposure
--analyze`` reads it for its recommendation and ``hs train --recipe`` is normally given the same
word; nothing else in the project changes, so no stage goes stale.
"""
import argparse
import os

from .. import calib, events, sourceprobe
from . import ingest

STAGE = "source"
SUBJECT_KINDS = ("matte", "glossy", "bright", "person", "scene")
TITLES = {"stereo": "Hydrogen One 3D clip", "video": "Video from one camera", "stills": "Photographs",
          "r3d": "RED camera array", "unknown": "Not a source"}


def add_parser(sub):
    p = sub.add_parser("source", help="say what a file or folder would be ingested as (no project needed)")
    p.add_argument("path", nargs="?", default=None, help="a video file, a folder of photographs, or a RED media folder")
    p.add_argument("--ffprobe", default=os.environ.get("HS_FFPROBE", "ffprobe"))
    p.add_argument("--subject", choices=SUBJECT_KINDS, default=None,
                   help="record what the subject is in the project's manifest (project.subject_kind); needs -p")
    p.add_argument("-p", "--project", default=argparse.SUPPRESS, help="project folder, for --subject")
    return p


def _report(path, kind, summary, problems=(), notes=(), **extra):
    r = {"path": path, "name": os.path.basename(path.rstrip(os.sep)), "kind": kind, "title": TITLES[kind],
         "summary": summary, "accepted": kind != "unknown" and not problems,
         "problems": list(problems), "notes": list(notes)}
    r.update(extra)
    return r


def probe_video(path, ffprobe_bin):
    try:
        probe = sourceprobe.ffprobe_json(ffprobe_bin, path)
    except events.StageError as e:
        return _report(path, "unknown", "not readable as a video", [str(e) + (f" ({e.hint})" if e.hint else "")])
    date = sourceprobe.recorded_date(path, probe)
    stem = sourceprobe.safe_stem(os.path.splitext(os.path.basename(path))[0])
    if sourceprobe.leia_tags(probe):
        ok, problems, info = ingest.validate_probe(probe)
        line = (f"{info['width']}×{info['height']} · {info.get('codec') or '?'} · {info['fps']:g} fps · "
                f"{info['nb_frames']:,} frames · {info['duration_s']:.1f} s · two eyes side by side")
        notes = []
        if ok and calib.match_profile(info["leia"]) is None:
            problems = problems + ["no calibration profile for this recording mode"]
        return _report(path, "stereo", line, problems, notes, date=date, stem=stem,
                       ingest=["--clip", path], video=info)
    facts, problems = sourceprobe.video_facts(probe)
    if facts is None:
        return _report(path, "unknown", "no video stream", problems)
    if facts["nb_frames"] <= 1:                  # one picture, which ffprobe calls a video stream
        return _report(path, "unknown", "a single picture, not a clip", problems)
    if not problems and ((facts["width"], facts["height"]) == (3840, 1080) or "_2x1" in os.path.basename(path)):
        problems = ["looks like a Hydrogen 2x1 clip without its leia3d tags (re-encoded?); use the clip as the phone wrote it"]
    notes = ["Frames are picked from the clip in the project's Select step."]
    if facts["hdr"] == "hlg":
        notes.append("HLG HDR: decoded through ffmpeg with one fixed curve; the Dolby Vision layer is ignored.")
    if os.path.splitext(path)[1].lower() not in sourceprobe.VIDEO_EXTS:
        notes.append("This container has not been tried; .mov and .mp4 have.")
    return _report(path, "video", sourceprobe.video_line(facts), problems, notes, date=date, stem=stem,
                   ingest=["--clip", path], video=facts)


def probe_r3d(path):
    root, take = sourceprobe.r3d_root(path)
    idx, bad = sourceprobe.r3d_index(root)
    takes = [{"take": t, "cameras": sorted(c), "count": len(c), "problem": bad.get(t),
              "date": sourceprobe.recorded_date(sorted(c.values())[0])} for t, c in sorted(idx.items())]

    def usable(t):
        return t["count"] >= 3 and not t["problem"]

    good = [t for t in takes if usable(t)]
    pick = take if take in idx else (good[0]["take"] if good else (takes[0]["take"] if takes else None))
    chosen = next((t for t in takes if t["take"] == pick), None)
    problems = [] if takes else ["no *_001.R3D clips named like G007_A067_… under this folder"]
    # the words "three cameras" mark a problem as one take's, not the folder's (SourceProbe.blocker)
    if chosen and chosen["count"] < 3:
        other = f"; take {good[0]['take']} has {good[0]['count']}" if good else ""
        one = ("; if this is one camera's folder, give the folder above it"
               if max(t["count"] for t in takes) == 1 else "")
        problems.append(f"take {pick} has {chosen['count']} camera{'s' if chosen['count'] != 1 else ''} here; "
                        f"an array needs three cameras or more{other}{one}")
    elif chosen and chosen["problem"]:
        problems.append(chosen["problem"] + " (three cameras or more, one clip each, make a take)")
    notes = ["One frame per camera: the first frame of each clip, rendered through REDline (BT.709, 8-bit)."]
    if os.path.abspath(os.path.expanduser(path)) != root:
        notes.append(f"The array is read from {root}.")
    try:
        exe = ingest.redline_exe(os.environ.get("HS_REDLINE", "REDline"))
    except events.StageError as e:
        exe = None
        problems.append(str(e) + (f" ({e.hint})" if e.hint else ""))
    cams = max((t["count"] for t in takes), default=0)
    ok = bool(chosen) and usable(chosen) and exe is not None
    return _report(root, "r3d", f"{len(takes)} take{'s' if len(takes) != 1 else ''}, up to {cams} cameras",
                   problems, notes, date=chosen["date"] if chosen else sourceprobe.recorded_date(root),
                   stem=sourceprobe.safe_stem(f"array{pick}" if pick else "array"),
                   ingest=(["--r3d", root, "--take", pick] if ok else None),
                   r3d={"root": root, "takes": takes, "take": pick, "redline": exe})


def probe_stills(root, dropped_file=None):
    import cv2
    try:
        kind, views, why = ingest.frames_layout(root)
    except events.StageError as e:
        return _report(root, "stills", "photographs", [str(e) + (f" ({e.hint})" if e.hint else "")])
    if not views:
        return _report(root, "unknown", "no photographs here",
                       [f"no {', '.join(ingest.SOURCE_EXTS)} files in this folder (or in one folder per camera inside it)"])
    given = [v for v, _ in views]
    names = sourceprobe.safe_names(given)
    renamed = [(o, n) for o, n in zip(given, names) if o != n]
    layout = "one folder per camera" if kind == "array" else "one folder"
    if kind is None:
        kind, why = ingest.guess_flat_kind(os.path.dirname(views[0][1]), names)
    heic = sum(1 for _v, p in views if p.lower().endswith(sourceprobe.HEIC_EXTS))
    size = None
    first = next((p for _v, p in views if not p.lower().endswith(sourceprobe.HEIC_EXTS)), None)
    if first:
        im = cv2.imread(first, cv2.IMREAD_COLOR)
        if im is not None:
            size = [int(im.shape[1]), int(im.shape[0])]
    problems = []
    if len(views) < 3:
        problems.append(f"{len(views)} photograph{'s' if len(views) != 1 else ''}: the solve needs at least three")
    notes = []
    if renamed:
        notes.append(f"{len(renamed)} file name{'s' if len(renamed) != 1 else ''} will be changed to view names "
                     f"(letters, digits, '-'): {renamed[0][0]} → {renamed[0][1]}. The originals are not touched.")
    if heic:
        notes.append(f"{heic} HEIC photograph{'s' if heic != 1 else ''} will be converted to JPEG (sips).")
    if dropped_file:
        notes.append(f"The whole folder is the set, not only {os.path.basename(dropped_file)}.")
    notes.append("Every photograph must be the same size, from one camera and one lens setting.")
    what = "from one camera" if kind == "mono" else "one per camera"
    line = f"{len(views):,} photographs" + (f" · {size[0]}×{size[1]}" if size else "") + f" · {what} ({why})"
    return _report(root, "stills", line, problems, notes,
                   date=sourceprobe.recorded_date(views[0][1]), stem=sourceprobe.safe_stem(os.path.basename(root.rstrip(os.sep))),
                   ingest=["--frames", root],
                   stills={"count": len(views), "kind": kind, "kind_why": why, "layout": layout, "size": size,
                           "renamed": len(renamed), "heic": heic})


def probe(path, ffprobe_bin="ffprobe"):
    """The report for one path (see the module docstring)."""
    p = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(p):
        return _report(p, "unknown", "not there", [f"no such file or folder: {p}"])
    low = p.lower().rstrip(os.sep)
    if os.path.isdir(p):
        if low.endswith((".rdc", ".rdm")):
            return probe_r3d(p)
        # a RED folder holds R3D clips somewhere below; look only as deep as a card goes
        if sourceprobe.r3d_files(p, max_depth=4):
            return probe_r3d(p)
        return probe_stills(p)
    if low.endswith(sourceprobe.R3D_EXTS):
        return probe_r3d(p)
    if low.endswith(ingest.SOURCE_EXTS):
        return probe_stills(os.path.dirname(p), dropped_file=p)
    return probe_video(p, ffprobe_bin)


def set_subject(pj, kind):
    """project.subject_kind in the manifest; -> the previous value (None when unset)."""
    if kind not in SUBJECT_KINDS:
        raise events.StageError(f"--subject {kind!r}: one of {', '.join(SUBJECT_KINDS)}")
    sect = pj.m.setdefault("project", {})
    prev = sect.get("subject_kind")
    pj.acquire(STAGE)
    try:
        sect["subject_kind"] = kind
        pj.save()
    finally:
        pj.release()
    events.metric(STAGE, "subject_kind", kind, previous=prev)
    return prev


def run(a):
    kind = getattr(a, "subject", None)
    path = getattr(a, "path", None)
    if kind:
        from ..project import Project
        root = getattr(a, "project", None)
        if not root:
            raise events.StageError("hs source --subject KIND needs --project DIR: the kind is kept in that manifest")
        set_subject(Project(root), kind)
    if path is None:
        if not kind:
            raise events.StageError("hs source needs a PATH to look at, or --subject KIND with -p DIR")
        return
    events.start(STAGE, "probe")
    r = probe(path, a.ffprobe)
    events.metric(STAGE, "source", r)
    events.check(STAGE, "source_accepted", r["accepted"], value=r["summary"] if r["accepted"] else "; ".join(r["problems"]) or r["summary"])
