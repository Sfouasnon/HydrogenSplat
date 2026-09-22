"""hs render — brush-path-render + ffmpeg (strategy §4.7).

    brush-path-render <ply> --path <move.json> -o <frames> --width 2400
    ffmpeg … <name>_1920.mp4 (H.264 crf 17, faststart) and a centred 4:5 crop <name>_1080x1350.mp4

Guards, learned the hard way on rig6 (HANDOFF §4): the .ply is MD5'd against the train
stage's recorded final export, and the move file must be newer than the rig.npz it was built
from and older than every frame written. Neither says the model and the cameras belong
together: a model can outlive a re-solve, its md5 still matching train's record, while a
freshly built move passes the mtime check in a frame the model was never trained in. So the
third guard compares the rig.npz md5 the train stage fingerprinted (or the archive manifest
carries, for a --ply from archive/) against the rig.npz the move was built from. A pruned
ply is accepted only when prune's recorded input md5 is train's export. Models trained
before the fingerprint existed fall back to timestamps — rig.npz must predate the training
run — and say so. brush-path-render prints ``path: N frames, …``,
``loaded N splats``, ``  frame i/N  (t elapsed)`` every 10 frames and ``wrote N frames to …``;
the PNG count on disk is polled as the ground truth.
"""
import os
import re
import shutil
import sys
import time

from .. import events, runner
from ..project import md5_file, parse_iso
from .prune import final_export

STAGE = "render"
DEFAULT_RENDER = "~/Desktop/Apps/brush/target/release/brush-path-render"
RE_PATH = re.compile(r"path: (\d+) frames, native (\d+)x(\d+), rendering (\d+)x(\d+)")
RE_LOADED = re.compile(r"loaded (\d+) splats")
RE_FRAME = re.compile(r"frame (\d+)/(\d+)\s+\(([\d.]+)s elapsed\)")
RE_WROTE = re.compile(r"wrote (\d+) frames to")
RE_FF_FRAME = re.compile(r"frame=\s*(\d+)")


def add_parser(sub):
    p = sub.add_parser("render", help="render a move with brush-path-render and encode with ffmpeg")
    p.add_argument("--move", required=True, help="move name (move/<name>.json) or a path to a move json")
    p.add_argument("--ply", default=None, help="default: the train stage's final export")
    p.add_argument("--name", default=None, help="output name (default: <move>[_pruned])")
    p.add_argument("--width", type=int, default=2400)
    p.add_argument("--keep-frames", action="store_true")
    p.add_argument("--stability", action="store_true",
                   help="measure temporal stability on the frames before they are deleted (hs stability): "
                        "render/<name>_stability/, stability_* metrics")
    p.add_argument("--stability-k", default="1,7", help="with --stability: frame gaps (default 1,7)")
    p.add_argument("--stability-backend", choices=("dis", "raft"), default="dis", help="with --stability: flow backend")
    p.add_argument("--no-crop", action="store_true", help="skip the 4:5 crop")
    p.add_argument("--crf", type=int, default=17)
    p.add_argument("--render-bin", default=os.environ.get("HS_PATH_RENDER", DEFAULT_RENDER))
    p.add_argument("--ffmpeg", default=os.environ.get("HS_FFMPEG", "ffmpeg"))
    p.add_argument("--allow-mismatch", action="store_true",
                   help="render even if the ply MD5 or the move mtime check fails (you have been warned)")
    return p


def run(a, pj):
    move = a.move if a.move.endswith(".json") else pj.path("move", f"{a.move}.json")
    move = os.path.abspath(move)
    # The move file is the prerequisite, not the `move` stage: the app's keyframe editor bakes
    # move/<name>.json itself and never runs `hs move`, so on those projects stages.move is
    # pending forever. An existing file stands in for the stage; a missing one still gets the
    # stage's own hint.
    move_from_stage = pj.status("move") in ("done", "stale")
    pj.require(STAGE, satisfied=("move",) if os.path.exists(move) else ())
    if not os.path.exists(move):
        raise events.StageError(f"move not found: {move}",
                                hint="hs move --project ... --preset boom, or bake one in the app's Move editor")
    move_name = os.path.splitext(os.path.basename(move))[0]
    ply = os.path.abspath(a.ply) if a.ply else final_export(pj)
    if not ply or not os.path.exists(ply):
        raise events.StageError("no .ply to render", hint="hs train first, or --ply PATH")
    name = a.name or (move_name + ("_pruned" if "pruned" in os.path.basename(ply) else ""))
    rbin = runner.which(a.render_bin)
    if not rbin:
        raise events.StageError(f"brush-path-render not found at {a.render_bin}",
                                hint="cargo build --release -p brush-path-render, or --render-bin PATH / HS_PATH_RENDER")
    ffmpeg = shutil.which(os.path.expanduser(a.ffmpeg))
    if not ffmpeg:
        raise events.StageError("ffmpeg not found", hint="brew install ffmpeg, or --ffmpeg PATH")

    out_dir = pj.path("render", name)
    pj.begin(STAGE, argv=sys.argv, clean=False)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    for f in (pj.path("render", f"{name}_1920.mp4"), pj.path("render", f"{name}_1080x1350.mp4")):
        if os.path.exists(f):
            os.remove(f)

    # ---- guards
    events.start(STAGE, "guards")
    pj.check(STAGE, "move_source", True,
             value=("hs move stage" if move_from_stage else
                    "move json authored outside `hs move` (keyframe editor or hand-written); stages.move is pending"))
    mt_move = guards(a, pj, ply, move)

    # ---- render
    events.start(STAGE, "render")
    argv = [rbin, ply, "--path", move, "-o", out_dir, "--width", str(a.width)]
    st = {"n": 0, "t0": time.monotonic(), "written": 0}

    def _progress(done, force=False):
        el = time.monotonic() - st["t0"]
        rate = done / el if el > 0.5 and done else None
        eta = (st["n"] - done) / rate if rate and st["n"] else None
        events.progress(STAGE, done, st["n"] or None, rate=rate, eta_s=eta, step="render", force=force)

    def on_line(line):
        m = RE_PATH.search(line)
        if m:
            st["n"] = int(m.group(1))
            pj.metric(STAGE, "frames", st["n"])
            pj.metric(STAGE, "render_size", [int(m.group(4)), int(m.group(5))])
            return
        m = RE_LOADED.search(line)
        if m:
            pj.metric(STAGE, "splats", int(m.group(1)))
            return
        m = RE_FRAME.search(line)
        if m:
            _progress(int(m.group(1)))

    def tick():
        n = len([f for f in os.listdir(out_dir) if f.endswith(".png")])
        if n != st["written"]:
            st["written"] = n
            _progress(n, force=True)

    res = runner.run(argv, STAGE, log_path=pj.log_path(STAGE), on_line=on_line, tick=tick, tick_interval=2.0,
                     env={"RUST_LOG": os.environ.get("RUST_LOG", "warn")})
    pngs = sorted(f for f in os.listdir(out_dir) if f.endswith(".png"))
    pj.metric(STAGE, "frames_written", len(pngs))
    pj.metric(STAGE, "render_elapsed_s", round(res.elapsed, 1))
    if not pngs:
        raise events.StageError("brush-path-render wrote no frames", hint="see logs/render.log")
    if st["n"] and len(pngs) != st["n"]:
        pj.check(STAGE, "all_frames_rendered", False, value=f"{len(pngs)}/{st['n']}")
    oldest = min(os.path.getmtime(os.path.join(out_dir, f)) for f in pngs)
    pj.check(STAGE, "frames_newer_than_move", oldest >= mt_move,
             value="every frame was written after the move file" if oldest >= mt_move else "a frame predates the move file")

    # ---- encode
    events.start(STAGE, "encode")
    fps = _move_fps(move)
    pattern = os.path.join(out_dir, "frame_%04d.png")
    out16 = pj.path("render", f"{name}_1920.mp4")
    _ffmpeg(ffmpeg, fps, pattern, "scale=1920:-2", a.crf, out16, pj, len(pngs), "encode_1920")
    pj.artifact(STAGE, out16, "video")
    pj.metric(STAGE, "mp4_1920", pj.rel(out16))
    if not a.no_crop:
        out45 = pj.path("render", f"{name}_1080x1350.mp4")
        # scale so the height is 1350, then crop the centre 1080 wide -> exact 4:5, no stretch
        _ffmpeg(ffmpeg, fps, pattern, "scale=-2:1350,crop=1080:1350", a.crf, out45, pj, len(pngs), "encode_4x5")
        pj.artifact(STAGE, out45, "video")
        pj.metric(STAGE, "mp4_1080x1350", pj.rel(out45))
    if getattr(a, "stability", False):
        # on the PNGs, before they go: the mp4 is H.264 at crf 17, and its own compression
        # flicker would be measured along with the model's
        from . import stability
        events.start(STAGE, "stability")
        s = stability.measure(frames=out_dir, ks=stability.parse_k(getattr(a, "stability_k", "1,7")),
                              backend=getattr(a, "stability_backend", "dis"),
                              out_dir=pj.path("render", f"{name}_stability"), name=name, stage=STAGE,
                              record=lambda n, v, **x: pj.metric(STAGE, "stability_" + n, v, **x))
        for kind, key in (("json", "json"), ("image", "png")):
            pj.artifact(STAGE, s["meta"]["outputs"][key], kind)
    if not a.keep_frames:
        shutil.rmtree(out_dir)
        pj.metric(STAGE, "frames_kept", False)
    else:
        pj.artifact(STAGE, out_dir, "frames")
        pj.metric(STAGE, "frames_kept", True)
    pj.record_run(STAGE, name, move=move_name, ply=pj.rel(ply) if ply.startswith(pj.root) else ply)
    pj.finish(STAGE, ok=True)


def model_lineage(pj, ply, digest, explicit):
    """Which solve was this ply trained against? Returns (rig_npz_md5 | None, how, md5_ok).

    how is a short provenance string for the check value. md5_ok is the old identity guard:
    the ply is the train stage's final export, prune's output of that export, or a --ply the
    user named explicitly."""
    tm = pj.stage("train").get("metrics", {})
    want = tm.get("final_export_md5")
    fp = tm.get("dataset_fingerprint") or {}
    pm = pj.stage("prune").get("metrics", {})
    pruned_out = pm.get("output_ply")
    if pruned_out and os.path.abspath(pj.path(pruned_out)) == ply:
        pin = pm.get("input_ply_md5")
        if pin and want and pin != want:
            return None, f"pruned from {pin}, but train's export is {want}", False
        if not pin:
            # A legacy prune has no input md5, so all it can prove is ORDER: it must have run on
            # the current export, i.e. after train finished. On 2026-09-13_coins a prune from
            # 18:26 (408,807 splats in, an older solve) was assumed current and rendered in a
            # different world frame from the other two panels of a three-up.
            p_fin = parse_iso(pj.stage("prune").get("finished"))
            t_fin = parse_iso(pj.stage("train").get("finished"))
            if p_fin and t_fin and p_fin >= t_fin:
                return fp.get("rig_npz_md5"), "pruned after train finished (prune recorded no input md5)", True
            return None, (f"pruned {pj.stage('prune').get('finished')} but train finished "
                          f"{pj.stage('train').get('finished')} — stale prune, re-run hs prune"), False
        return fp.get("rig_npz_md5"), "pruned variant of train's export", True
    if want and digest == want:
        return fp.get("rig_npz_md5"), "train's final export", True
    # a ply from an archive folder carries its own frame in manifest.json
    man = os.path.join(os.path.dirname(ply), "manifest.json")
    if os.path.exists(man):
        try:
            import json
            m = json.load(open(man))
            if isinstance(m, dict) and m.get("rig_npz_md5"):
                arch_ok = (m.get("ply") or {}).get("md5") in (None, digest)
                return m["rig_npz_md5"], "archive manifest", bool(explicit and arch_ok)
        except (OSError, ValueError):
            pass
    if want:
        return None, f"{digest} vs train's {want}", False
    return None, "train stage recorded no export MD5" + (" (--ply given)" if explicit else ""), bool(explicit)


def guards(a, pj, ply, move):
    """The three render guards; records the checks and raises unless --allow-mismatch.
    Returns the move file's mtime (the frames must be newer than it)."""
    digest = md5_file(ply)
    pj.metric(STAGE, "ply", pj.rel(ply) if ply.startswith(pj.root) else ply)
    pj.metric(STAGE, "ply_md5", digest)
    pj.metric(STAGE, "move", pj.rel(move) if move.startswith(pj.root) else move)
    rig_want, how, md5_ok = model_lineage(pj, ply, digest, a.ply is not None)
    pj.check(STAGE, "ply_md5_matches", md5_ok, value=f"{digest}: {how}")

    rig = pj.rig_npz
    mt_move = os.path.getmtime(move)
    mt_rig = os.path.getmtime(rig) if os.path.exists(rig) else 0
    mtime_ok = pj.check(STAGE, "move_newer_than_solve", mt_move >= mt_rig,
                        value=f"move {time.strftime('%H:%M:%S', time.localtime(mt_move))} vs rig.npz "
                              f"{time.strftime('%H:%M:%S', time.localtime(mt_rig))}")

    # does the model live in the frame the move was built in?
    rig_now = md5_file(rig) if os.path.exists(rig) else None
    pj.metric(STAGE, "rig_npz_md5", rig_now)
    if rig_want:
        lineage_ok = pj.check(STAGE, "model_matches_solve", rig_want == rig_now,
                              value=f"trained against rig.npz {rig_want}; current solve's is {rig_now}"
                                    if rig_want != rig_now else f"rig.npz {rig_now} ({how})")
    elif md5_ok and pj.stage("train").get("started"):
        # trained before the fingerprint existed: the rig must at least predate the training run
        t_train = parse_iso(pj.stage("train").get("started"))
        older = bool(t_train and mt_rig and mt_rig <= t_train)
        lineage_ok = pj.check(STAGE, "model_matches_solve", older,
                              value=("legacy model (no dataset fingerprint recorded): rig.npz predates the "
                                     "training run, checked by timestamp only — retrain to record lineage")
                                    if older else
                                    "rig.npz is newer than the training run: the model was trained against an "
                                    "earlier solve and cannot be rendered in this frame")
    else:
        lineage_ok = pj.check(STAGE, "model_matches_solve", False,
                              value=f"no record of which solve this ply was trained against ({how})")
    if not (md5_ok and mtime_ok and lineage_ok) and not a.allow_mismatch:
        raise events.StageError("render guard failed — the ply, the move and the solve do not belong together",
                                hint="fix the inputs (usually: retrain on the current solve), or --allow-mismatch if you really mean it")
    return mt_move


def _move_fps(move):
    import json
    try:
        return float(json.load(open(move)).get("fps", 30.0))
    except Exception:
        return 30.0


def _ffmpeg(ffmpeg, fps, pattern, vf, crf, out, pj, n, step):
    argv = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-stats", "-framerate", f"{fps:g}",
            "-i", pattern, "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
            "-movflags", "+faststart", out]

    def on_line(line):
        m = RE_FF_FRAME.search(line)
        if m:
            events.progress(STAGE, int(m.group(1)), n, step=step)

    runner.run(argv, STAGE, log_path=pj.log_path(STAGE), on_line=on_line)
    if not os.path.exists(out) or os.path.getsize(out) == 0:
        raise events.StageError(f"ffmpeg produced no {os.path.basename(out)}", hint="see logs/render.log")
