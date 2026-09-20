"""hs movepreview — compile an .hsmove script for the app's viewer, without touching the pipeline.

    hs movepreview -p P --script move/shot03.hsmove   -> viewer/move_shot03.json
                                                          viewer/move_shot03.report.json
    hs movepreview -p P --frame                       -> viewer/move_frame.json (the captures
                                                          in script coordinates; no script needed)
    hs movepreview -p P --locate x,y,z [--hull MM]    -> a `locate` metric: where a camera
                                                          centre (solve mm) sits as az / el / dolly

The same compiler `hs move --script` runs (movescript.build), against the same rig.npz
(train/dataset/rig.npz), so what the viewer previews is the path the stage will write. Like
`hs cameras` it takes no project lock and leaves manifest.json alone: the move panel
recompiles on every edit, including while `hs train` holds the lock. It writes no aim-check
image and records no checks — the aim is confirmed on the move `hs move` builds, never on a
preview.

A script error is an ``error`` event whose message starts with the line number, so the app
can point at the line.
"""
import json
import math
import os

from .. import events, movescript, rig

STAGE = "movepreview"


def add_parser(sub):
    p = sub.add_parser("movepreview", help="compile an .hsmove script into viewer/move_<name>.json (no lock, no manifest)")
    p.add_argument("--script", default=None, help="the .hsmove file")
    p.add_argument("--name", default=None, help="default: the script's file name")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--frame", action="store_true", help="only write viewer/move_frame.json (captures in script coordinates)")
    p.add_argument("--locate", default=None, help="x,y,z mm (solve coordinates): report az / el / dolly for a start line there")
    p.add_argument("--hull", type=float, default=movescript.HULL_MM, help="hull limit for --locate, mm")
    return p


def _clean(x):
    """JSON has no NaN: the report carries NaN radii where a cue never reached the hull."""
    if isinstance(x, float):
        return None if (math.isnan(x) or math.isinf(x)) else x
    if hasattr(x, "item") and not isinstance(x, (list, tuple, dict)):   # numpy scalar
        return _clean(x.item())
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_clean(obj), f)
    os.replace(tmp, path)


def write_frame(rig_npz, out):
    """The captures in script coordinates — what the move panel's map draws before (or without)
    a script that compiles, and what a click on the map is turned into a start line against."""
    fr = movescript.frame(rig_npz)
    reach = movescript.Reach(fr["CL"], fr["subject"], fr["up"], fr["ref"], fr["right"])
    locs = [movescript.locate(rig_npz, c, _fr=fr, _reach=reach) for c in fr["CL"]]
    t = {"schema": 1, "rig_npz_md5": _md5(rig_npz), "subject_mm": fr["subject"].tolist(),
         "up": fr["up"].tolist(), "ref": fr["ref"].tolist(), "right": fr["right"].tolist(),
         "hull_default_mm": movescript.HULL_MM,
         "captures": [[round(d["az"], 2), round(d["el"], 2), round(d["r_mm"], 1)] for d in locs],
         "capture_names": [rig.capture_name(fr["names"][v]) for v in fr["L"]]}
    _write_json(out, t)
    return t


def _md5(p):
    from ..project import md5_file
    return md5_file(p)


def run(a):
    if not getattr(a, "project", None):
        raise events.StageError("hs movepreview needs --project DIR")
    root = os.path.abspath(os.path.expanduser(a.project))
    if not os.path.exists(os.path.join(root, "manifest.json")):
        raise events.StageError(f"{root} is not a project (no manifest.json)")
    rig_npz = os.path.join(root, "train", "dataset", "rig.npz")
    if not os.path.exists(rig_npz):
        raise events.StageError("no train/dataset/rig.npz", hint="hs solve first")
    vdir = os.path.join(root, "viewer")
    os.makedirs(vdir, exist_ok=True)
    events.start(STAGE)

    if a.locate:
        try:
            p = [float(x) for x in a.locate.replace(" ", "").split(",")]
            assert len(p) == 3
        except (ValueError, AssertionError):
            raise events.StageError(f"--locate wants x,y,z in mm, not {a.locate!r}")
        d = _clean(movescript.locate(rig_npz, p, hull=a.hull))
        events.metric(STAGE, "locate", d)
        return d

    if a.frame or not a.script:
        out = os.path.join(vdir, "move_frame.json")
        write_frame(rig_npz, out)
        events.artifact(STAGE, os.path.relpath(out, root), "json")
        return None

    spath = os.path.abspath(os.path.expanduser(a.script))
    if not os.path.exists(spath):
        raise events.StageError(f"no script at {spath}")
    name = a.name or os.path.splitext(os.path.basename(spath))[0]
    if os.sep in name or name.startswith("."):
        raise events.StageError(f"move name must be a plain name, not {name!r}")
    out = os.path.join(vdir, f"move_{name}.json")
    rep_path = os.path.join(vdir, f"move_{name}.report.json")
    for f in (out, rep_path):
        if os.path.exists(f):
            os.remove(f)
    try:
        rep = movescript.build(rig_npz, open(spath).read(), out, fps=a.fps)
    except movescript.ScriptError as e:
        raise events.StageError(str(e), hint="cue grammar is at the top of engine/hs/movescript.py")
    rep = _clean(rep)
    rep["schema"] = 1
    rep["script"] = spath
    rep["move_json"] = out
    rep["rig_npz_md5"] = _md5(rig_npz)
    _write_json(rep_path, rep)
    events.metric(STAGE, "frames", rep["frames"])
    events.metric(STAGE, "duration_s", round(rep["frames"] / rep["fps"], 2))
    events.metric(STAGE, "hull_max_mm", round(rep["hull_mm"][1], 1))
    events.metric(STAGE, "cues_clamped", len(rep["clamped"]))
    events.artifact(STAGE, os.path.relpath(out, root), "move")
    events.artifact(STAGE, os.path.relpath(rep_path, root), "json")
    return rep
