"""hs grade — lift / gamma / gain, sharpening and a head-following crop on a rendered move.

    hs grade -p P --move arc180 [--lift 0.02] [--gamma 1.1] [--gain 1.05]
             [--lift-rgb 0,0,0] [--gamma-rgb 1,1,1] [--gain-rgb 1.03,1,0.97]
             [--sharpen 0.35] [--aspect 2.35] [--headroom-mm 25.4]
    hs grade -p P --move arc180 --still 225 [--graded]   # one frame for the app's preview

Per channel, on 8-bit code values x in [0, 1] (the same curve the app previews):

    out = clip(gain·x + lift·(1 − x), 0, 1) ^ (1 / gamma)

with lift = master + channel offset, gamma and gain = master × channel factor. It is applied
with ffmpeg ``lutrgb``, so preview and export use one 256-entry table per channel.

Crop: if ``move/<name>_frame.json`` exists (written by ``hs move --headroom-mm``), a full-width
crop of ``--aspect`` follows the crown of the head so it stays ``--headroom-mm`` below the top
edge (ffmpeg ``sendcmd`` drives the crop row frame by frame, smoothed over 15 frames).
Without it the crop is centred. ``--aspect 0`` keeps the full frame.

Input ``render/<move>_1920.mp4``; output ``render/<move>_graded.mp4``; the settings used are
kept in ``grade/<move>.json`` and are the defaults for the next run.
"""
import json
import os
import shutil
import subprocess

import numpy as np

from .. import events, framing, runner

STAGE = "grade"
DEFAULTS = {"lift": 0.0, "gamma": 1.0, "gain": 1.0, "lift_rgb": [0.0, 0.0, 0.0],
            "gamma_rgb": [1.0, 1.0, 1.0], "gain_rgb": [1.0, 1.0, 1.0], "sharpen": 0.35,
            "aspect": 2.35, "headroom_mm": 25.4}


def add_parser(sub):
    p = sub.add_parser("grade", help="lift/gamma/gain, sharpen and head-following crop on a rendered move")
    p.add_argument("--move", required=True, help="move name; reads render/<move>_1920.mp4")
    p.add_argument("--lift", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--gain", type=float, default=None)
    p.add_argument("--lift-rgb", default=None, help="r,g,b offsets added to --lift")
    p.add_argument("--gamma-rgb", default=None, help="r,g,b factors on --gamma")
    p.add_argument("--gain-rgb", default=None, help="r,g,b factors on --gain")
    p.add_argument("--sharpen", type=float, default=None, help="contrast-adaptive sharpening 0-1 (0 = off)")
    p.add_argument("--aspect", type=float, default=None, help="crop aspect, 0 = no crop")
    p.add_argument("--headroom-mm", type=float, default=None)
    p.add_argument("--still", type=int, default=None, help="write one frame (0-based) instead of the video")
    p.add_argument("--graded", action="store_true", help="with --still: apply the grade too (default: crop only)")
    p.add_argument("--reset", action="store_true", help="ignore saved settings")
    p.add_argument("--ffmpeg", default=os.environ.get("HS_FFMPEG", "ffmpeg"))
    p.add_argument("--crf", type=int, default=17)
    return p


def _rgb(s):
    v = [float(x) for x in s.split(",")]
    if len(v) != 3:
        raise events.StageError(f"expected r,g,b: {s}")
    return v


def settings(a, saved):
    s = dict(DEFAULTS)
    if saved and not a.reset:
        s.update({k: saved[k] for k in DEFAULTS if k in saved})
    for k in ("lift", "gamma", "gain", "sharpen", "aspect", "headroom_mm"):
        v = getattr(a, k)
        if v is not None:
            s[k] = v
    for k in ("lift_rgb", "gamma_rgb", "gain_rgb"):
        v = getattr(a, k)
        if v is not None:
            s[k] = _rgb(v)
    for k in ("gamma",):
        if s[k] <= 0 or min(s["gamma_rgb"]) <= 0:
            raise events.StageError("gamma must be > 0")
    return s


def channel_params(s):
    return [(s["lift"] + s["lift_rgb"][c], s["gamma"] * s["gamma_rgb"][c], s["gain"] * s["gain_rgb"][c])
            for c in range(3)]


def lut(lift, gamma, gain):
    """The 256-entry table the app previews; matches the lutrgb expression below."""
    x = np.arange(256) / 255.0
    y = np.clip(gain * x + lift * (1.0 - x), 0.0, 1.0) ** (1.0 / gamma)
    return np.clip(np.floor(y * 255.0 + 0.5), 0, 255).astype(np.uint8)


def _expr(lift, gamma, gain):
    # val/maxval in [0,1]; floor(+0.5) rounds like lut() above
    return (f"clip(floor(pow(clip({gain:.6f}*val/maxval+{lift:.6f}*(1-val/maxval)\\,0\\,1)\\,{1.0 / gamma:.6f})"
            f"*maxval+0.5)\\,0\\,maxval)")


def grade_filter(s):
    r, g, b = (_expr(*p) for p in channel_params(s))
    parts = [f"lutrgb=r={r}:g={g}:b={b}"]
    if s["sharpen"] > 0:
        parts.append(f"cas=strength={min(max(s['sharpen'], 0.0), 1.0):.3f}")
    return parts


def is_identity(s):
    return all(abs(v - d) < 1e-9 for v, d in ((s["lift"], 0), (s["gamma"], 1), (s["gain"], 1))) and \
        s["lift_rgb"] == [0, 0, 0] and s["gamma_rgb"] == [1, 1, 1] and s["gain_rgb"] == [1, 1, 1]


def probe(ffmpeg, path):
    ffprobe = os.path.join(os.path.dirname(ffmpeg), "ffprobe")
    if not os.path.exists(ffprobe):
        ffprobe = shutil.which("ffprobe") or "ffprobe"
    r = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-count_packets",
                        "-show_entries", "stream=width,height,r_frame_rate,nb_read_packets", "-of", "json", path],
                       capture_output=True, text=True)
    try:
        st = json.loads(r.stdout)["streams"][0]
        num, den = st["r_frame_rate"].split("/")
        return int(st["width"]), int(st["height"]), float(num) / float(den), int(st["nb_read_packets"])
    except Exception:
        raise events.StageError(f"cannot probe {path}", hint=(r.stderr or "")[-200:])


def run(a, pj):
    ffmpeg = shutil.which(os.path.expanduser(a.ffmpeg))
    if not ffmpeg:
        raise events.StageError("ffmpeg not found", hint="brew install ffmpeg")
    src = pj.path("render", f"{a.move}_1920.mp4")
    if not os.path.exists(src):
        raise events.StageError(f"no render for {a.move}: {pj.rel(src)}", hint=f"hs render --move {a.move}")
    os.makedirs(pj.path("grade"), exist_ok=True)
    spath = pj.path("grade", f"{a.move}.json")
    saved = json.load(open(spath)) if os.path.exists(spath) else None
    s = settings(a, saved)
    W, H, fps, n = probe(ffmpeg, src)
    events.metric(STAGE, "source", {"path": pj.rel(src), "size": [W, H], "fps": fps, "frames": n})

    # ---- crop
    vf, cmds = [], None
    track = framing.load_track(pj, a.move) if s["aspect"] > 0 else None
    if s["aspect"] > 0:
        if track:
            if len(track["crown_row"]) != n:
                events.check(STAGE, "frame_track_matches_render", False,
                             value=f"track has {len(track['crown_row'])} frames, render {n}: centred crop instead")
                track = None
        if track:
            tops, ch = framing.crop_track_px(track, W, H, s["aspect"], s["headroom_mm"])
            events.metric(STAGE, "crop", {"height": ch, "top_min": int(tops.min()), "top_max": int(tops.max()),
                                          "mode": "follow head", "headroom_mm": s["headroom_mm"]})
        else:
            ch = framing.even(W / s["aspect"])
            tops = np.full(n, (H - ch) // 2)
            events.metric(STAGE, "crop", {"height": ch, "top_min": int(tops[0]), "top_max": int(tops[0]), "mode": "centred"})
        if ch > H:
            raise events.StageError(f"aspect {s['aspect']} is taller than the render")

    if a.still is not None:
        i = min(max(a.still, 0), n - 1)
        if s["aspect"] > 0:
            vf.append(f"crop={W}:{ch}:0:{int(tops[i])}")
        if a.graded:
            vf += grade_filter(s)
        out = pj.path("grade", f"{a.move}_still{'_graded' if a.graded else ''}.png")
        argv = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", src,
                "-vf", f"select=eq(n\\,{i})" + ("," + ",".join(vf) if vf else ""), "-frames:v", "1", out]
        runner.run(argv, STAGE, log_path=pj.log_path(STAGE))
        if not os.path.exists(out):
            raise events.StageError("ffmpeg wrote no still", hint="see logs/grade.log")
        events.metric(STAGE, "still", {"frame": i, "frames": n, "crop_top": int(tops[i]) if s["aspect"] > 0 else None})
        events.artifact(STAGE, pj.rel(out), "image")
        return

    if s["aspect"] > 0:
        if track:
            cmds = pj.path("grade", f"{a.move}_crop.cmd")
            with open(cmds, "w") as f:
                for k, y in enumerate(tops):
                    f.write(f"{max(0.0, k / fps - 0.5 / fps):.4f} crop y {int(y)};\n")
            vf += [f"sendcmd=f='{cmds}'", f"crop={W}:{ch}:0:{int(tops[0])}"]
        else:
            vf.append(f"crop={W}:{ch}:0:{int(tops[0])}")
    if not is_identity(s) or s["sharpen"] > 0:
        vf += grade_filter(s)
    out = pj.path("render", f"{a.move}_graded.mp4")
    events.start(STAGE, "encode")
    argv = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-stats", "-i", src,
            "-vf", ",".join(vf) if vf else "null", "-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    import re
    rx = re.compile(r"frame=\s*(\d+)")

    def on_line(line):
        m = rx.search(line)
        if m:
            events.progress(STAGE, int(m.group(1)), n, step="encode")

    runner.run(argv, STAGE, log_path=pj.log_path(STAGE), on_line=on_line)
    if not os.path.exists(out) or os.path.getsize(out) == 0:
        raise events.StageError("ffmpeg produced no graded video", hint="see logs/grade.log")
    with open(spath, "w") as f:
        json.dump({**s, "move": a.move, "output": pj.rel(out)}, f, indent=1)
    events.metric(STAGE, "settings", s)
    events.artifact(STAGE, pj.rel(out), "video")
    events.progress(STAGE, n, n, step="encode", force=True)
