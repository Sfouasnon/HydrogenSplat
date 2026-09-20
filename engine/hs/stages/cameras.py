"""hs cameras — write the solved cameras as JSON for the app's splat viewer (hs/cameras.py).

    hs cameras -p P                   train/dataset/rig.npz      -> viewer/cameras_current.json
    hs cameras -p P --archive NAME    archive/NAME/rig.npz       -> viewer/cameras_NAME.json

Read-only towards the pipeline: it takes no project lock, touches no stage and leaves
manifest.json alone, so the viewer can ask for cameras while `hs train` holds the lock. The
output goes under viewer/, never into archive/NAME — an archive is written whole or not at
all, and nothing is added to one afterwards.
"""
import os

from .. import cameras, events

STAGE = "cameras"


def add_parser(sub):
    p = sub.add_parser("cameras", help="write viewer/cameras_*.json (poses + pinholes, metres) for the app's splat viewer")
    p.add_argument("--archive", default=None, help="use archive/NAME/rig.npz (the cameras that model was trained against)")
    return p


def run(a):
    if not getattr(a, "project", None):
        raise events.StageError("hs cameras needs --project DIR")
    root = os.path.abspath(os.path.expanduser(a.project))
    if not os.path.exists(os.path.join(root, "manifest.json")):
        raise events.StageError(f"{root} is not a project (no manifest.json)")
    events.start(STAGE)
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
