"""hs tools — find the tools and report versions (the Setup page's back end, strategy §4.0)."""
import os
import platform
import shutil
import subprocess
import sys

from .. import events
from ..project import tool_versions
from .render import DEFAULT_RENDER
from .train import DEFAULT_BRUSH

STAGE = "tools"


def add_parser(sub):
    p = sub.add_parser("tools", help="check python packages, brush, brush-path-render, ffmpeg, adb")
    p.add_argument("--brush", default=os.environ.get("HS_BRUSH", DEFAULT_BRUSH))
    p.add_argument("--render-bin", default=os.environ.get("HS_PATH_RENDER", DEFAULT_RENDER))
    return p


def _ver(argv):
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=20)
        return (r.stdout or r.stderr).strip().splitlines()[0] if (r.stdout or r.stderr).strip() else "?"
    except Exception as e:
        return None


def run(a, pj=None):
    events.start(STAGE)
    info = tool_versions()
    info["platform"] = f"{platform.system()} {platform.machine()}"
    ok_all = True
    for mod in ("numpy", "cv2", "pycolmap"):
        ok = info.get(mod) is not None
        ok_all &= ok
        events.check(STAGE, mod, ok, value=info.get(mod) or f"pip install {'opencv-python-headless' if mod == 'cv2' else mod}")
    if info.get("pycolmap") and not str(info["pycolmap"]).startswith("4.2"):
        events.check(STAGE, "pycolmap_version", False, value=f"{info['pycolmap']} (rig6 used 4.2.0; the rig API changed across versions)")
    for name, path, hint in (("brush", a.brush, "cargo build --release -p brush-app in the Brush fork"),
                             ("brush-path-render", a.render_bin, "cargo build --release -p brush-path-render")):
        p = os.path.expanduser(path)
        ok = os.path.isfile(p) and os.access(p, os.X_OK)
        events.check(STAGE, name, ok, value=p if ok else f"not at {p}: {hint}")
        info[name] = p if ok else None
    for name, argv, hint in (("ffmpeg", ["ffmpeg", "-version"], "brew install ffmpeg"),
                             ("adb", ["adb", "version"], "brew install android-platform-tools")):
        exe = shutil.which(name)
        v = _ver(argv) if exe else None
        events.check(STAGE, name, bool(exe), value=v if exe else hint)
        info[name] = v
    # hs stability's optical flow: dis is always there with cv2, raft is the optional hs[metrics]
    from .stability import available_backends
    for name, (ok, detail) in available_backends().items():
        events.check(STAGE, f"flow_{name}", ok, value=detail)
        info[f"flow_{name}"] = detail if ok else None
    events.metric(STAGE, "python_exe", sys.executable)
    if pj is not None:
        for k, v in info.items():
            pj.record_tool(k, v)
        pj.save()
    return info
