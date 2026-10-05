"""Lens profiles: one camera body + lens + recording size -> K, distortion and how good the fit was.

`hs calibrate` writes one from a clip of the ChArUco board; `hs solve --lens auto` looks one up by
the project's source metadata and seeds COLMAP's OPENCV camera with it. The store is a folder of
JSON files named by key::

    ~/Library/Application Support/HydrogenSplat/lenses/<key>.json      (macOS)
    ~/.hydrogensplat/lenses/<key>.json                                 (elsewhere)
    $HS_LENSES/<key>.json                                              (either, when set: tests)

key = safe(make) _ safe(model) _ safe(lens) _ WxH, e.g. ``apple_iphone-15-pro_back-camera_3840x2160``.
Same body, same lens, same recording size is what makes a calibration transferable: a different
resolution or crop is a different profile, so the size is part of the key and never scaled.

Profile shape (version 1)::

    {"version": 1, "key": ..., "camera": {"make", "model", "lens", "software"}, "image_size": [w, h],
     "model": "OPENCV", "K": 3x3, "dist": [k1, k2, p1, p2, k3], "rms_px": .., "frames_used": n,
     "frames_seen": n, "corners_total": n, "coverage": {"cells": [6, 4], "share": 0..1},
     "board": "SX,SY,SQ,MK,DICT", "source": "clip or folder name", "date": ISO-8601, "ok": bool,
     "hs_version": ..}

`colmap_params` turns it into the "fx,fy,cx,cy,k1,k2,p1,p2" string pycolmap takes for OPENCV
(k3 has no slot there; a lens whose k3 matters should be refined by COLMAP from this prior).
"""
import datetime
import glob
import json
import os
import re
import sys

import numpy as np

RE_UNSAFE = re.compile(r"[^a-z0-9]+")
UNKNOWN = {"make": "unknown", "model": "unknown", "lens": "default"}
# ffprobe format/stream tag names that carry the camera's identity, in order of preference
TAG_MAKE = ("com.apple.quicktime.make", "make", "manufacturer", "com.android.manufacturer")
TAG_MODEL = ("com.apple.quicktime.model", "model", "com.android.model", "device_model")
TAG_LENS = ("com.apple.quicktime.camera.lens_model", "lens_model", "lens", "com.apple.quicktime.camera.identifier")
TAG_SOFTWARE = ("com.apple.quicktime.software", "software")


def profile_dir():
    """The store folder (not created here)."""
    env = os.environ.get("HS_LENSES")
    if env:
        return os.path.expanduser(env)
    if sys.platform == "darwin":
        return os.path.expanduser("~/Library/Application Support/HydrogenSplat/lenses")
    return os.path.expanduser("~/.hydrogensplat/lenses")


def safe(s):
    s = RE_UNSAFE.sub("-", str(s or "").strip().lower()).strip("-")
    return s or "unknown"


def key_for(meta):
    """meta: {"make", "model", "lens", "width", "height"} (missing identity -> UNKNOWN's words)."""
    m = dict(UNKNOWN)
    m.update({k: v for k, v in (meta or {}).items() if v not in (None, "")})
    w, h = int(m.get("width") or 0), int(m.get("height") or 0)
    return f"{safe(m['make'])}_{safe(m['model'])}_{safe(m['lens'])}_{w}x{h}"


def _first(tags, names):
    low = {str(k).lower(): v for k, v in (tags or {}).items()}
    for n in names:
        v = low.get(n.lower())
        if v not in (None, ""):
            return str(v).strip()
    return None


def meta_from_probe(probe, width=None, height=None):
    """ffprobe -show_format -show_streams JSON -> {"make", "model", "lens", "software", "width",
    "height"}. Tags are read from the container first, then the picture stream (Android puts them
    there). width/height are the DISPLAYED size when given (sourceprobe.video_facts knows the
    rotation); else the stream's stored size."""
    fmt_tags = ((probe or {}).get("format") or {}).get("tags") or {}
    vids = [s for s in (probe or {}).get("streams", []) if s.get("codec_type") == "video"]
    st_tags = (vids[0].get("tags") or {}) if vids else {}
    tags = dict(st_tags)
    tags.update(fmt_tags)
    meta = {"make": _first(tags, TAG_MAKE), "model": _first(tags, TAG_MODEL), "lens": _first(tags, TAG_LENS),
            "software": _first(tags, TAG_SOFTWARE)}
    if width is None and vids:
        width, height = vids[0].get("width"), vids[0].get("height")
    meta["width"], meta["height"] = int(width or 0), int(height or 0)
    return meta


def meta_from_project(pj):
    """What the project's source says about its camera: the mono clip's own probe.json tags and
    the displayed size; an array/stills source has a size and, unless ingest recorded a camera,
    nothing else (key unknown_unknown_default_WxH). None when the size is not known."""
    src = pj.m.get("source") or {}
    facts = src.get("probe") or {}
    w, h = facts.get("width"), facts.get("height")
    if not w or not h:
        return None
    meta = {"width": int(w), "height": int(h)}
    pj_probe = pj.path("source", "probe.json")
    if os.path.isfile(pj_probe):
        try:
            full = json.load(open(pj_probe))
        except ValueError:
            full = {}
        meta.update({k: v for k, v in meta_from_probe(full, w, h).items() if k not in ("width", "height")})
    cam = src.get("camera") or {}
    for k in ("make", "model", "lens"):
        if cam.get(k):
            meta[k] = cam[k]
    return meta


def path_for(key):
    return os.path.join(profile_dir(), f"{key}.json")


def save(profile, path=None):
    """Write the profile under its key (or at `path`); the folder is created. -> path."""
    key = profile.get("key") or key_for({**(profile.get("camera") or {}),
                                         "width": profile["image_size"][0], "height": profile["image_size"][1]})
    profile = dict(profile)
    profile["key"] = key
    profile.setdefault("version", 1)
    profile.setdefault("date", datetime.datetime.now().astimezone().isoformat(timespec="seconds"))
    out = path or path_for(key)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(profile, f, indent=1)
    return out


def load(path):
    """A profile file -> dict, or ValueError saying what is wrong with it."""
    try:
        j = json.load(open(os.path.expanduser(path)))
    except (OSError, ValueError) as e:
        raise ValueError(f"cannot read lens profile {path}: {e}")
    for k in ("K", "dist", "image_size"):
        if k not in j:
            raise ValueError(f"{path} is not a lens profile (no {k!r})")
    K = np.asarray(j["K"], float)
    if K.shape != (3, 3) or len(j["image_size"]) != 2:
        raise ValueError(f"{path}: K must be 3x3 and image_size [w, h]")
    return j


def find(meta):
    """The stored profile for this camera + size -> (path, profile), or None."""
    p = path_for(key_for(meta))
    if not os.path.isfile(p):
        return None
    try:
        return p, load(p)
    except ValueError:
        return None


def list_profiles():
    """[(path, profile)] in the store, newest first; unreadable files are skipped."""
    out = []
    for p in glob.glob(os.path.join(profile_dir(), "*.json")):
        try:
            out.append((p, load(p)))
        except ValueError:
            continue
    out.sort(key=lambda pp: pp[1].get("date") or "", reverse=True)
    return out


def colmap_params(profile):
    """"fx,fy,cx,cy,k1,k2,p1,p2" for pycolmap's OPENCV model (k3 dropped: the model has no slot)."""
    K = np.asarray(profile["K"], float)
    d = list(np.asarray(profile.get("dist") or [], float).ravel()) + [0.0] * 4
    vals = [K[0, 0], K[1, 1], K[0, 2], K[1, 2], d[0], d[1], d[2], d[3]]
    return ",".join(f"{float(v):.10g}" for v in vals)


def summary(path, profile):
    """What solve records as `lens_profile`: where it came from and how good it is."""
    return {"path": path, "key": profile.get("key"), "rms_px": profile.get("rms_px"), "date": profile.get("date"),
            "frames_used": profile.get("frames_used"), "coverage_share": (profile.get("coverage") or {}).get("share"),
            "fx_px": float(np.asarray(profile["K"], float)[0, 0])}
