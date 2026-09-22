"""hs tools — find the tools and report versions (the Setup page's back end, strategy §4.0).

``hs tools --fetch-vocab-tree`` also downloads COLMAP's vocabulary tree for ``hs solve
--matcher sequential --loop vocab`` (optional: the default stride loop pass needs nothing).
The file is the FAISS tree COLMAP ≥ 3.12 / pycolmap ≥ 3.12 fetch as their own default
(``VOCAB_TREE_URL``, checked against ``VOCAB_TREE_SHA256``; source: colmap/colmap issue
#3464 and PR #3036). It goes to ``vocab_tree_path()``: ``$HS_VOCAB_TREE`` if set, else
``~/Library/Caches/HydrogenSplat/`` on macOS (``~/.cache/hydrogensplat/`` elsewhere). Not
downloaded or run in the Linux container this was written in (GitHub is not reachable from
it), so the ``--loop vocab`` path is wired and unexercised.
"""
import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import time

from .. import events, runner
from ..project import tool_versions
from .render import DEFAULT_RENDER
from .train import DEFAULT_BRUSH

STAGE = "tools"

VOCAB_TREE_FILE = "vocab_tree_faiss_flickr100K_words256K.bin"
VOCAB_TREE_URL = "https://github.com/colmap/colmap/releases/download/3.11.1/" + VOCAB_TREE_FILE
VOCAB_TREE_SHA256 = "96ca8ec8ea60b1f73465aaf2c401fd3b3ca75cdba2d3c50d6a2f6f760f275ddc"


def cache_dir():
    if sys.platform == "darwin":
        return os.path.expanduser("~/Library/Caches/HydrogenSplat")
    return os.path.expanduser("~/.cache/hydrogensplat")


def vocab_tree_path():
    """Where the vocabulary tree is (or would be): $HS_VOCAB_TREE, else the cache."""
    env = os.environ.get("HS_VOCAB_TREE")
    return os.path.expanduser(env) if env else os.path.join(cache_dir(), VOCAB_TREE_FILE)


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def fetch_vocab_tree(url=VOCAB_TREE_URL, dest=None, sha256=VOCAB_TREE_SHA256):
    """Download the tree to ``dest`` (default vocab_tree_path()) through a .part file, check its
    sha256 (skipped when sha256 is empty) and move it into place. Progress events on the way.
    Returns the path. An existing file with the right checksum is kept, not re-downloaded."""
    import urllib.request
    dest = dest or vocab_tree_path()
    if os.path.isfile(dest) and (not sha256 or sha256_file(dest) == sha256):
        events.check(STAGE, "vocab_tree_fetched", True, value=f"already at {dest}")
        return dest
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    part = dest + ".part"
    events.start(STAGE, "fetch_vocab_tree")
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0) or None
            got, t0 = 0, time.monotonic()
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
                got += len(b)
                el = time.monotonic() - t0
                events.progress(STAGE, got, total, rate=got / el if el > 0.5 else None,
                                detail=f"{got / 1e6:.0f} MB", step="fetch_vocab_tree")
    except Exception as e:  # noqa: BLE001 — urllib raises several unrelated types
        if os.path.exists(part):
            os.remove(part)
        raise events.StageError(f"could not download the vocabulary tree: {e}",
                                hint=f"{url} — or download it by hand and set HS_VOCAB_TREE")
    if sha256:
        got_sha = sha256_file(part)
        if got_sha != sha256:
            os.remove(part)
            raise events.StageError(f"vocabulary tree checksum mismatch: {got_sha}",
                                    hint=f"expected {sha256} from {url}")
    os.replace(part, dest)
    events.check(STAGE, "vocab_tree_fetched", True, value=dest)
    return dest


def vocab_tree_status(info):
    """The `vocab_tree` line: optional, so its check is always ok and the value says why."""
    path = vocab_tree_path()
    ok = os.path.isfile(path)
    events.check(STAGE, "vocab_tree", True,
                 value=(f"{path} ({os.path.getsize(path) / 1e6:.0f} MB; hs solve --loop vocab)" if ok else
                        "optional, not fetched: hs tools --fetch-vocab-tree (only for hs solve --loop vocab; "
                        "the default stride loop pass needs nothing)"))
    info["vocab_tree"] = path if ok else None


def add_parser(sub):
    p = sub.add_parser("tools", help="check python packages, brush, brush-path-render, ffmpeg, adb, node/splat-transform")
    p.add_argument("--brush", default=os.environ.get("HS_BRUSH", DEFAULT_BRUSH))
    p.add_argument("--render-bin", default=os.environ.get("HS_PATH_RENDER", DEFAULT_RENDER))
    p.add_argument("--splat-transform", default=os.environ.get("HS_SPLAT_TRANSFORM"))
    p.add_argument("--fetch-vocab-tree", action="store_true",
                   help=f"download COLMAP's vocabulary tree (~{VOCAB_TREE_FILE}) for hs solve --loop vocab")
    p.add_argument("--vocab-tree-url", default=os.environ.get("HS_VOCAB_TREE_URL", VOCAB_TREE_URL),
                   help=argparse.SUPPRESS)
    p.add_argument("--vocab-tree-sha256", default=VOCAB_TREE_SHA256, help=argparse.SUPPRESS)
    return p



def _ver(argv):
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=20)
        return (r.stdout or r.stderr).strip().splitlines()[0] if (r.stdout or r.stderr).strip() else "?"
    except Exception as e:
        return None


def run(a, pj=None):
    events.start(STAGE)
    if getattr(a, "fetch_vocab_tree", False):
        fetch_vocab_tree(a.vocab_tree_url, sha256=a.vocab_tree_sha256)
    info = tool_versions()
    info["platform"] = f"{platform.system()} {platform.machine()}"
    ok_all = True
    for mod in ("numpy", "cv2", "pycolmap"):
        ok = info.get(mod) is not None
        ok_all &= ok
        events.check(STAGE, mod, ok, value=info.get(mod) or f"pip install {'opencv-python-headless' if mod == 'cv2' else mod}")
    if info.get("pycolmap") and not str(info["pycolmap"]).startswith("4.2"):
        events.check(STAGE, "pycolmap_version", False, value=f"{info['pycolmap']} (rig6 used 4.2.0; the rig API changed across versions)")
    if info.get("cv2"):
        import cv2
        # hs scale / exposure --reference board need the ChArUco detector (main OpenCV >= 4.7);
        # --reference checker needs cv2.mcc, which not every build carries (optional)
        aruco = hasattr(getattr(cv2, "aruco", None), "CharucoDetector")
        events.check(STAGE, "cv2_aruco_charuco", aruco,
                     value="cv2.aruco.CharucoDetector (hs scale, exposure --reference board)" if aruco
                     else "needs opencv-python-headless >= 4.7 for hs scale")
        from .exposure import checker_available
        mcc = checker_available()
        events.check(STAGE, "cv2_mcc", True, value="cv2.mcc (exposure --reference checker)" if mcc
                     else "optional: no cv2.mcc — exposure --reference checker unavailable "
                          "(opencv-contrib-python-headless has it)")
        info["cv2_mcc"] = mcc
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
    splat_transform(a, info)
    vocab_tree_status(info)
    events.metric(STAGE, "python_exe", sys.executable)
    if pj is not None:
        for k, v in info.items():
            pj.record_tool(k, v)
        pj.save()
    return info


def splat_transform(a, info):
    """node / npx / splat-transform for hs export. Optional: only spz, sog and html need them.
    Never downloads anything: an npx route is probed with --no, which answers 'no' to the
    install prompt, so an uncached package reports as 'fetched on first export'."""
    from .export import NPX_PACKAGE
    for name, hint in (("node", "brew install node (hs export's spz/sog/html need it; ply does not)"),
                       ("npx", "comes with node")):
        exe = shutil.which(name)
        v = _ver([exe, "--version"]) if exe else None
        events.check(STAGE, name, bool(exe), value=v if exe else hint)
        info[name] = v
    explicit = getattr(a, "splat_transform", None)
    exe = runner.which(explicit) if explicit else shutil.which("splat-transform")
    if explicit and not exe:
        events.check(STAGE, "splat-transform", False, value=f"not at {explicit} (HS_SPLAT_TRANSFORM / --splat-transform)")
        info["splat-transform"] = None
        return
    if exe:
        v = _ver([exe, "--version"])
        events.check(STAGE, "splat-transform", bool(v), value=f"{v} ({exe})" if v else f"{exe} did not answer --version")
        info["splat-transform"] = {"argv": [exe], "version": v}
        return
    npx = shutil.which("npx")
    if not npx:
        events.check(STAGE, "splat-transform", False, value="needs node: brew install node, or npm install -g " + NPX_PACKAGE)
        info["splat-transform"] = None
        return
    v = _ver([npx, "--no", "--", NPX_PACKAGE, "--version"])
    # an uncached package makes npx --no print "npm error npx canceled ... [\"@playcanvas/splat-transform@x\"]",
    # which also contains the name: only a real version line (no "npm" prefix) counts
    v = v if v and "splat-transform" in v and not v.lower().startswith("npm") else None
    events.check(STAGE, "splat-transform", True,
                 value=f"{v} via npx" if v else f"via npx; {NPX_PACKAGE} is fetched on the first hs export")
    info["splat-transform"] = {"argv": [npx, "-y", "--", NPX_PACKAGE], "version": v}
