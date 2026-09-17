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

from .. import coverage, events, framing, runner

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
    p.add_argument("--script", default=None,
                   help="an .hsmove script in capture coordinates (grammar: engine/hs/movescript.py); overrides --preset")
    p.add_argument("--name", default=None, help="move name (default: the preset)")
    p.add_argument("--frames", type=int, default=None, help="default: sweep 240, boom/custom 360")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--captures", default=None, help="sweep: A:B window (default: auto from coverage)")
    p.add_argument("--keys", default=None, help="custom: ordered capture indices 52,54,56,...")
    p.add_argument("--hold", type=float, default=0.5, help="boom/custom: seconds held at each end")
    p.add_argument("--ease", type=float, default=1.0)
    p.add_argument("--aim-point", default=None, help="x,y,z mm — override the SfM median (you must then check the image)")
    p.add_argument("--span", type=float, default=0.8, help="sweep: fraction of the window to sweep")
    p.add_argument("--headroom-mm", type=float, default=None,
                   help="frame a person: solve the aim height so the crown sits this far below the top of the "
                        "--frame-aspect crop on every frame (needs move/subject.json); overrides --aim-point")
    p.add_argument("--frame-aspect", type=float, default=2.35, help="crop aspect the framing solves for")
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
    name = a.name or (os.path.splitext(os.path.basename(a.script))[0] if a.script else a.preset)
    out = pj.path("move", f"{name}.json")
    # a move folder holds several moves; don't wipe siblings
    pj.begin(STAGE, argv=sys.argv, clean=False)
    for f in (out, pj.path("move", f"{name}_aim_check.jpg")):
        if os.path.exists(f):
            os.remove(f)
    events.start(STAGE, "script" if a.script else a.preset)

    if a.script:
        _from_script(a, pj, rig, name, out)
        return

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
    if a.headroom_mm is not None:
        crown = framing.load_subject(pj)
        if crown is None:
            raise events.StageError("--headroom-mm needs move/subject.json with the subject's crown_mm",
                                    hint="write {\"crown_mm\": [x, y, z]} (solve coordinates, mm)")
        events.start(STAGE, "framing")
        t, fmove, rows, depths, fy = framing.solve_aim(framing.builder(argv, out), crown, framing.up_axis(pj),
                                                       a.frame_aspect, a.headroom_mm)
        aim = crown - t * framing.up_axis(pj)
        a.aim_point = ",".join(f"{x:.1f}" for x in aim)
        pj.metric(STAGE, f"{name}.aim_below_crown_mm", round(t, 1))
        pj.metric(STAGE, f"{name}.crown_row_px", [round(float(rows.min())), round(float(rows.max()))])
        info["framing"] = {"headroom_mm": a.headroom_mm, "aspect": a.frame_aspect}
    if a.aim_point:
        argv.append(f"--aim-point={a.aim_point}")
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
    if a.headroom_mm is not None:
        rows, depths, fy = framing.crown_track(move, crown)
        tp = framing.write_track(pj, name, move, rows, depths, fy, a.frame_aspect, a.headroom_mm, aim, t)
        pj.artifact(STAGE, tp, "json")
        tops = framing.crop_rows(rows, depths, fy, a.headroom_mm)
        pj.check(STAGE, "headroom_fits_every_frame", bool(tops.min() >= 0 and tops.max() <= framing.room(move, a.frame_aspect) + 0.5),
                 value=f"crop starts {tops.min():.0f}-{tops.max():.0f} px down (room {framing.room(move, a.frame_aspect):.0f}) "
                       f"for {a.headroom_mm:g} mm above the crown at {depths.min():.0f}-{depths.max():.0f} mm", move=name)
    else:
        stale = pj.path("move", framing.FRAME_FILE.format(name=name))
        if os.path.exists(stale):
            os.remove(stale)
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
    m = re.match(r"([A-Za-z0-9-]+)_([LR])$", view_name)
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


def _from_script(a, pj, rig, name, out):
    """hs move --script shot.hsmove — cues in capture coordinates, clamped to the captured hull."""
    from .. import movescript
    spath = os.path.abspath(os.path.expanduser(a.script))
    if not os.path.exists(spath):
        raise events.StageError(f"no script at {spath}", hint="hs move --script move/shot.hsmove")
    aim = [float(x) for x in a.aim_point.split(",")] if a.aim_point else None
    try:
        rep = movescript.build(rig, open(spath).read(), out, fps=a.fps, aim_point=aim)
    except movescript.ScriptError as e:
        raise events.StageError(str(e), hint="cue grammar is at the top of engine/hs/movescript.py")
    move = json.load(open(out))
    pj.artifact(STAGE, out, "move")
    pj.metric(STAGE, f"{name}.frames", rep["frames"])
    pj.metric(STAGE, f"{name}.size", [move["width"], move["height"]])
    pj.metric(STAGE, f"{name}.aim_mm", [round(x, 1) for x in rep["subject_mm"]])
    pj.metric(STAGE, f"{name}.path_length_mm", int(round(rep["path_length_mm"])))
    pj.metric(STAGE, f"{name}.peak_speed_mm_s", int(round(rep["peak_speed_mm_s"])))
    pj.metric(STAGE, f"{name}.p90_speed_mm_s", int(round(rep["p90_speed_mm_s"])))
    pj.metric(STAGE, f"{name}.distance_to_subject_mm", [int(round(x)) for x in rep["distance_mm"]])
    pj.metric(STAGE, f"{name}.duration_s", round(rep["frames"] / rep["fps"], 2))
    cues = [{"cue": c["cue"], "want": round(c["want"], 1), "got": round(c["got"], 1),
             "unit": c["unit"], "secs": round(c["secs"], 2), "clamped": c["clamped"],
             "r_mm": [round(c["r0"]), round(c["r1"])] if c.get("r0") == c.get("r0") else None}
            for c in rep["cues"]]
    pj.metric(STAGE, f"{name}.cues", cues)

    hmin, hmax, hmed = rep["hull_mm"]
    pj.metric(STAGE, f"{name}.hull_max_mm", round(hmax, 1))
    pj.check(STAGE, "hull_within_25mm", hmax <= HULL_MAX_MM,
             value=f"virtual camera never more than {hmax:.0f} mm from a real camera "
                   f"(median {hmed:.0f}; pass \u2264 {HULL_MAX_MM:.0f})", move=name)

    short = [c for c in rep["cues"] if c.get("clamped")]
    pj.check(STAGE, "every_cue_completed", not short,
             value="all cues ran to their target" if not short else
                   "; ".join(f"{c['cue']} got {c['got']:.0f}{c['unit']} of {c['want']:.0f}{c['unit']} "
                             f"(az {c['at']['az']:+.0f} el {c['at']['el']:+.0f})" for c in short),
             move=name)
    ran = len(rep["cues"])
    if short and any(c["cue"] != "dolly" for c in short):
        pj.metric(STAGE, f"{name}.cues_run", ran)

    # the reachable band steps where the nearest real camera changes; a track that has to
    # follow that step pops in one frame, which no amount of smoothing inside the hull removes
    sp = rep.get("spike")
    pj.check(STAGE, "no_speed_spike", sp is None,
             value="speed stays under 3x its own 90th percentile" if sp is None else
                   f"{sp['mm_s']:.0f} mm/s in one frame at {sp['t']:.2f} s (frame {sp['frame']}), "
                   f"against a 90th percentile of {rep['p90_speed_mm_s']:.0f} \u2014 the reachable band "
                   f"steps here; move the cue off this bearing or slow it through", move=name)

    aimr = rep["aim"]
    if aimr.get("behind"):
        pj.check(STAGE, "aim_in_frame", False, value=f"aim point is BEHIND view {aimr['view']}",
                 needs_human=True, move=name)
        raise events.StageError("aim point is behind the check view — the path is aimed at nothing",
                                hint="pass --aim-point x,y,z (mm), or an `aim x,y,z` line in the script")
    u, v, depth = aimr["u"], aimr["v"], aimr["depth_mm"]
    pj.metric(STAGE, f"{name}.aim_px", [round(u), round(v)])
    pj.metric(STAGE, f"{name}.aim_view", aimr["view"])
    pj.metric(STAGE, f"{name}.aim_depth_mm", round(depth))
    img_path = aim_check_image(pj, aimr["view"], u, v, depth, name)
    if img_path:
        pj.artifact(STAGE, img_path, "image")
    pj.check(STAGE, "aim_in_frame", aimr["inside"],
             value=f"({u:.0f},{v:.0f}) in {aimr['view']} at {depth:.0f} mm"
                   + ("" if aimr["inside"] else " \u2014 OUT OF FRAME"),
             needs_human=True, move=name, image=pj.rel(img_path) if img_path else None)
    pj.record_run(STAGE, name, preset="script", info={"script": pj.rel(spath) if spath.startswith(pj.root) else spath,
                                                      "start": rep["start"], "cues": cues},
                  path=pj.rel(out))
    pj.finish(STAGE, ok=True)
