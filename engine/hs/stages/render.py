"""hs render — brush-path-render + ffmpeg (strategy §4.7).

    brush-path-render <ply> --path <move.json> -o <frames> --width 2400
    ffmpeg … <name>_1920.mp4 (H.264 crf 17, faststart) and a centred 4:5 crop <name>_1080x1350.mp4

Guards, both learned the hard way on rig6 (HANDOFF §4): the .ply is MD5'd against the train
stage's recorded final export, and the move file must be newer than the rig.npz it was built
from and older than every frame written. brush-path-render prints ``path: N frames, …``,
``loaded N splats``, ``  frame i/N  (t elapsed)`` every 10 frames and ``wrote N frames to …``;
the PNG count on disk is polled as the ground truth.
"""
import os
import re
import shutil
import sys
import time

from .. import events, runner
from ..project import md5_file
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
    p.add_argument("--no-crop", action="store_true", help="skip the 4:5 crop")
    p.add_argument("--crf", type=int, default=17)
    p.add_argument("--render-bin", default=os.environ.get("HS_PATH_RENDER", DEFAULT_RENDER))
    p.add_argument("--ffmpeg", default=os.environ.get("HS_FFMPEG", "ffmpeg"))
    p.add_argument("--allow-mismatch", action="store_true",
                   help="render even if the ply MD5 or the move mtime check fails (you have been warned)")
    return p


def run(a, pj):
    pj.require(STAGE)
    move = a.move if a.move.endswith(".json") else pj.path("move", f"{a.move}.json")
    move = os.path.abspath(move)
    if not os.path.exists(move):
        raise events.StageError(f"move not found: {move}", hint="hs move --project ... --preset boom")
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
    digest = md5_file(ply)
    pj.metric(STAGE, "ply", pj.rel(ply) if ply.startswith(pj.root) else ply)
    pj.metric(STAGE, "ply_md5", digest)
    pj.metric(STAGE, "move", pj.rel(move) if move.startswith(pj.root) else move)
    want = pj.stage("train").get("metrics", {}).get("final_export_md5")
    pruned_out = pj.stage("prune").get("metrics", {}).get("output_ply")
    if pruned_out and os.path.abspath(pj.path(pruned_out)) == ply:
        md5_ok = pj.check(STAGE, "ply_md5_matches", True, value=f"pruned variant {digest} (from prune stage)")
    elif want:
        md5_ok = pj.check(STAGE, "ply_md5_matches", digest == want,
                          value=f"{digest} vs train's {want}" if digest != want else digest)
    else:
        md5_ok = pj.check(STAGE, "ply_md5_matches", a.ply is not None,
                          value=f"{digest} (train stage recorded no export MD5; --ply given)" if a.ply
                          else "train stage recorded no export MD5")
    rig = pj.rig_npz
    mt_move = os.path.getmtime(move)
    mt_rig = os.path.getmtime(rig) if os.path.exists(rig) else 0
    mtime_ok = pj.check(STAGE, "move_newer_than_solve", mt_move >= mt_rig,
                        value=f"move {time.strftime('%H:%M:%S', time.localtime(mt_move))} vs rig.npz "
                              f"{time.strftime('%H:%M:%S', time.localtime(mt_rig))}")
    if not (md5_ok and mtime_ok) and not a.allow_mismatch:
        raise events.StageError("render guard failed — the ply or the move is not what the manifest says it is",
                                hint="fix the inputs, or --allow-mismatch if you really mean it")

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
    if not a.keep_frames:
        shutil.rmtree(out_dir)
        pj.metric(STAGE, "frames_kept", False)
    else:
        pj.artifact(STAGE, out_dir, "frames")
        pj.metric(STAGE, "frames_kept", True)
    pj.record_run(STAGE, name, move=move_name, ply=pj.rel(ply) if ply.startswith(pj.root) else ply)
    pj.finish(STAGE, ok=True)


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
