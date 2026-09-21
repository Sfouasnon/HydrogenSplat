"""Apple Vision foreground segmentation, for `hs masks --method vision`.

The helper is engine/hs/tools/hs_segment.swift. The engine builds it itself on first use —
`xcrun swiftc`, cached under ~/Library/Caches/HydrogenSplat by the source's hash, so an edit to
the Swift rebuilds it and nothing else ever does — so the CLI works from a fresh checkout without
building the app. HS_SEGMENT_BIN overrides the binary (the tests use a fake).

Why the engine and not the app: `hs masks` is a stage like every other, runnable from Terminal,
and it must not depend on which app build is installed.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys

from . import events

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "tools", "hs_segment.swift")
MIN_MACOS = "14.0"          # VNGenerateForegroundInstanceMaskRequest


def cache_dir():
    return os.path.expanduser(os.environ.get("HS_CACHE_DIR", "~/Library/Caches/HydrogenSplat"))


def helper(stage):
    """-> path to a runnable hs-segment, building it if needed. Raises StageError with a fix."""
    env = os.environ.get("HS_SEGMENT_BIN")
    if env:
        if not os.access(env, os.X_OK):
            raise events.StageError(f"HS_SEGMENT_BIN={env} is not an executable file")
        return env
    if sys.platform != "darwin":
        raise events.StageError("--method vision uses Apple Vision, which needs macOS",
                                hint="hs masks --method geometry")
    src = open(SOURCE, "rb").read()
    tag = hashlib.md5(src).hexdigest()[:10]
    out = os.path.join(cache_dir(), f"hs-segment-{tag}")
    if os.access(out, os.X_OK):
        return out
    os.makedirs(cache_dir(), exist_ok=True)
    arch = platform.machine() or "arm64"
    cmd = ["xcrun", "swiftc", "-O", "-target", f"{arch}-apple-macos{MIN_MACOS}", "-o", out + ".tmp", SOURCE]
    events.start(stage, "build_vision_helper")   # once per helper version; about a minute
    events.log(stage, "[hs] " + " ".join(cmd))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except FileNotFoundError:
        raise events.StageError("xcrun is not installed, so the Vision helper cannot be built",
                                hint="xcode-select --install   (or: hs masks --method geometry)")
    if r.returncode != 0:
        tail = "\n".join((r.stderr or r.stdout).strip().splitlines()[-12:])
        raise events.StageError(f"building the Vision helper failed:\n{tail}",
                                hint="hs masks --method geometry works without it; send this error to fix the helper")
    os.replace(out + ".tmp", out)
    return out


def run(stage, images, work_dir, progress=None):
    """Segment every path in `images`. -> list (same order) of dicts:
    {"instances": n, "dir": work_dir/i} or {"error": "..."}. Instance k's soft mask is dir/k.png."""
    exe = helper(stage)
    os.makedirs(work_dir, exist_ok=True)
    lst = os.path.join(work_dir, "images.txt")
    with open(lst, "w") as f:
        f.write("\n".join(os.path.abspath(p) for p in images) + "\n")
    res = [{"error": "no result from the helper"} for _ in images]
    p = subprocess.Popen([exe, "--out", work_dir, "--list", lst], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    done = 0
    for line in p.stdout:
        try:
            ev = json.loads(line)
        except ValueError:
            events.log(stage, "[hs-segment] " + line.rstrip())
            continue
        i = ev.get("index")
        if not isinstance(i, int) or not 0 <= i < len(images):
            continue
        if "error" in ev:
            res[i] = {"error": ev["error"]}
        else:
            res[i] = {"instances": int(ev.get("instances", 0)), "dir": os.path.join(work_dir, str(i))}
        done += 1
        if progress:
            progress(done, len(images))
    err = p.stderr.read()
    if p.wait() != 0:
        raise events.StageError(f"the Vision helper exited {p.returncode}: {err.strip()[-400:]}",
                                hint="hs masks --method geometry works without it")
    return res
