"""hs move (alias: paths) — camera moves through the existing builders (strategy §4.6).

Presets, all from coverage.json + rig.npz:
  sweep   spline_path.py --aim-median over the longest run of captures within ±6° of the
          median elevation (rig6: 8:34), 240 frames
  boom    key_path.py with boom–slide–settle keys chosen from the coverage table
          (rig6: 51,56,58,60,62,64,16,62 ≈ yesterday's hand-picked move_boom.json), 360 frames, 0.5 s holds
  custom  key_path.py with --keys you give (the click-to-key designer's back end)

Every build's aim-check line is parsed into a check event with needs_human=true, and the real
image the aim point was projected into is written with a crosshair as move/<name>_aim_check.jpg.
"In frame" is not "on subject" — a person confirms before render enables. The hull number
("virtual cameras sit … from the nearest real camera") must be ≤ 25 mm.
"""
import json
import os
import re
import sys

from .. import coverage, events, runner

STAGE = "move"
RE_AIM = re.compile(r"aim check: projects to \(([-\d.]+), ([-\d.]+)\) in view (\d+) \((\w+), (\d+)x(\d+)\) at ([-\d.]+) mm — (.*)$")
RE_BEHIND = re.compile(r"aim point is BEHIND view (\d+) \((\w+)\)")
RE_HULL = re.compile(r"virtual cameras sit ([\d.]+)-([\d.]+) mm from the nearest real camera \(median ([\d.]+)\)")
# numpy prints mixed-magnitude rows in scientific notation ([-1.260e+01  2.000e-01  2.021e+02])
RE_OBJ = re.compile(r"object centre \(mm\), (?:median of \d+ points|given|densest cell of \d+ points): \[\s*([-\d.e+]+)\s+([-\d.e+]+)\s+([-\d.e+]+)\s*\]")
RE_LEN = re.compile(r"path length (\d+) mm")
RE_SPEED = re.compile(r"peak speed (\d+) mm/s, mean (\d+) mm/s")
RE_DIST = re.compile(r"virtual camera distance to object: (\d+)-(\d+) mm")
HULL_MAX_MM = 25.0


def add_parser(sub):
    p = sub.add_parser("move", aliases=["paths"], help="build a camera move (spline_path.py / key_path.py) + aim check")
    p.add_argument("--preset", choices=["sweep", "boom", "custom"], default="sweep")
    p.add_argument("--name", default=None, help="move name (default: the preset)")
    p.add_argument("--frames", type=int, default=None, help="default: sweep 240, boom/custom 360")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--captures", default=None, help="sweep: A:B window (default: auto from coverage)")
    p.add_argument("--keys", default=None, help="custom: ordered capture indices 52,54,56,...")
    p.add_argument("--hold", type=float, default=0.5, help="boom/custom: seconds held at each end")
    p.add_argument("--ease", type=float, default=1.0)
    p.add_argument("--aim-point", default=None, help="x,y,z mm — override the SfM median (you must then check the image)")
    p.add_argument("--span", type=float, default=0.8, help="sweep: fraction of the window to sweep")
    p.add_argument("--smooth", type=int, default=3, help="sweep: moving-average window over camera centres")
    return p


def run(a, pj):
    pj.require(STAGE)
    rig = pj.rig_npz
    if not os.path.exists(rig):
        raise events.StageError("no train/dataset/rig.npz", hint="hs solve --project ...")
    cov_path = pj.path("solve", "coverage.json")
    if os.path.exists(cov_path):
        cov = json.load(open(cov_path))
    else:
        cov = coverage.write(rig, cov_path)
    name = a.name or a.preset
    out = pj.path("move", f"{name}.json")
    # a move folder holds several moves; don't wipe siblings
    pj.begin(STAGE, argv=sys.argv, clean=False)
    for f in (out, pj.path("move", f"{name}_aim_check.jpg")):
        if os.path.exists(f):
            os.remove(f)
    events.start(STAGE, a.preset)

    frames = a.frames or (240 if a.preset == "sweep" else 360)
    info = {}
    if a.preset == "sweep":
        if a.captures:
            lo, hi = (int(x) for x in a.captures.split(":"))
        else:
            (lo, hi), med = coverage.sweep_window(cov, rig_npz=rig)
            info["auto_window_median_el"] = round(med, 2)
        info["window"] = f"{lo}:{hi}"
        pj.metric(STAGE, f"{name}.window", f"{lo}:{hi}")
        argv = runner.python_argv("spline_path.py", rig, "-o", out, "--captures", f"{lo}:{hi}",
                                  "--frames", frames, "--fps", a.fps, "--span", a.span, "--smooth", a.smooth)
    else:
        if a.preset == "boom":
            keys, kinfo = coverage.boom_keys(cov)
            info.update(kinfo)
            if kinfo.get("fallback"):
                pj.check(STAGE, f"{name}.preset_fallback", False, value=kinfo["fallback"])
        else:
            if not a.keys:
                raise events.StageError("--preset custom needs --keys")
            keys = [int(k) for k in a.keys.split(",")]
        info["keys"] = keys
        pj.metric(STAGE, f"{name}.keys", keys)
        argv = runner.python_argv("key_path.py", rig, "-o", out, "--keys", ",".join(map(str, keys)),
                                  "--frames", frames, "--fps", a.fps, "--hold", a.hold, "--ease", a.ease)
    if a.aim_point:
        argv += ["--aim-point", a.aim_point]
    else:
        argv.append("--aim-median")

    st = {}

    def on_line(line):
        for key, rx in (("aim", RE_AIM), ("behind", RE_BEHIND), ("hull", RE_HULL), ("obj", RE_OBJ),
                        ("len", RE_LEN), ("speed", RE_SPEED), ("dist", RE_DIST)):
            m = rx.search(line)
            if m:
                st[key] = m.groups()

    runner.run(argv, STAGE, log_path=pj.log_path(STAGE), on_line=on_line)
    if not os.path.exists(out):
        raise events.StageError("the path builder wrote nothing", hint="see logs/move.log")
    move = json.load(open(out))
    pj.artifact(STAGE, out, "move")
    pj.metric(STAGE, f"{name}.frames", len(move["frames"]))
    pj.metric(STAGE, f"{name}.size", [move["width"], move["height"]])
    if "obj" in st:
        pj.metric(STAGE, f"{name}.aim_mm", [float(x) for x in st["obj"]])
    if "len" in st:
        pj.metric(STAGE, f"{name}.path_length_mm", int(st["len"][0]))
    if "speed" in st:
        pj.metric(STAGE, f"{name}.peak_speed_mm_s", int(st["speed"][0]))
    if "dist" in st:
        pj.metric(STAGE, f"{name}.distance_to_subject_mm", [int(st["dist"][0]), int(st["dist"][1])])

    # ---- hull
    if "hull" in st:
        hmin, hmax, hmed = (float(x) for x in st["hull"])
        pj.metric(STAGE, f"{name}.hull_max_mm", hmax)
        pj.check(STAGE, "hull_within_25mm", hmax <= HULL_MAX_MM,
                 value=f"virtual camera never more than {hmax:.0f} mm from a real camera (median {hmed:.0f}; pass ≤ {HULL_MAX_MM:.0f})",
                 move=name)
    else:
        pj.check(STAGE, "hull_within_25mm", False, value="hull line not found in builder output", move=name)

    # ---- aim check (human)
    if "behind" in st:
        pj.check(STAGE, "aim_in_frame", False, value=f"aim point is BEHIND view {st['behind'][1]}",
                 needs_human=True, move=name)
        raise events.StageError("aim point is behind the check view — the path is aimed at nothing",
                                hint="pass --aim-point x,y,z (mm) from a point you have verified")
    if "aim" not in st:
        pj.check(STAGE, "aim_in_frame", False, value="aim check line not found", needs_human=True, move=name)
        raise events.StageError("no aim check in builder output", hint="see logs/move.log")
    u, v, vidx, vname, W, H, depth, verdict = st["aim"]
    u, v, depth = float(u), float(v), float(depth)
    inside = verdict.startswith("in frame")
    pj.metric(STAGE, f"{name}.aim_px", [round(u), round(v)])
    pj.metric(STAGE, f"{name}.aim_view", vname)
    pj.metric(STAGE, f"{name}.aim_depth_mm", round(depth))
    img_path = aim_check_image(pj, vname, u, v, depth, name)
    if img_path:
        pj.artifact(STAGE, img_path, "image")
    pj.check(STAGE, "aim_in_frame", inside,
             value=f"({u:.0f},{v:.0f}) in {vname} at {depth:.0f} mm" + ("" if inside else " — OUT OF FRAME"),
             needs_human=True, move=name,
             image=pj.rel(img_path) if img_path else None)
    pj.record_run(STAGE, name, preset=a.preset, info=info, path=pj.rel(out))
    pj.finish(STAGE, ok=True)


def aim_check_image(pj, view_name, u, v, depth_mm, move_name):
    """Crosshair on the real (undistorted) training image the aim point projects into."""
    try:
        import cv2
    except ImportError:
        return None
    m = re.match(r"(cap\d+)_([LR])$", view_name)
    if not m:
        return None
    cap, eye = m.groups()
    src = os.path.join(pj.dataset_dir, "images", eye, cap + ".jpg")
    if not os.path.exists(src):
        return None
    im = cv2.imread(src, cv2.IMREAD_COLOR)
    if im is None:
        return None
    h, w = im.shape[:2]
    x, y = int(round(u)), int(round(v))
    inside = 0 <= x < w and 0 <= y < h
    col = (0, 255, 0) if inside else (0, 0, 255)
    cx, cy = min(max(x, 0), w - 1), min(max(y, 0), h - 1)
    cv2.line(im, (cx - 60, cy), (cx + 60, cy), col, 2, cv2.LINE_AA)
    cv2.line(im, (cx, cy - 60), (cx, cy + 60), col, 2, cv2.LINE_AA)
    cv2.circle(im, (cx, cy), 28, col, 2, cv2.LINE_AA)
    label = f"{move_name}: aim -> ({x},{y}) in {view_name} at {depth_mm:.0f} mm" + ("" if inside else "  OUT OF FRAME")
    cv2.rectangle(im, (0, h - 34), (w, h), (0, 0, 0), -1)
    cv2.putText(im, label, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    out = pj.path("move", f"{move_name}_aim_check.jpg")
    cv2.imwrite(out, im, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out
