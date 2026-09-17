"""hs train — wraps the Brush trainer (strategy §4.5).

    brush <project>/train/dataset --total-train-iters 40000 --growth-stop-iter 30000
          --refine-every 130 --export-every 2500 --export-path <project>/train/exports
          --export-name export_{iter}.ply

Facts settled by reading the Brush fork's source (crates/brush-process/src/{config,train_stream}.rs,
apps/brush-cli/src/lib.rs) — the "still open" items of strategy §9:

* ``--export-path`` is joined onto the dataset's *parent* directory; an absolute path works
  as-is. ``--export-name`` is a template: ``{iter}`` is zero-padded to the digit count of
  ``--total-train-iters`` (40000 → 5 digits → ``export_02500.ply`` … ``export_40000.ply``).
* Progress: the indicatif bars go to stderr and are hidden when stderr is not a TTY, so
  under a pipe the spinner never appears. What does appear (stdout, with RUST_LOG=info) is
  the env_logger line ``Refine iter 130, 74699 splats.`` every ``--refine-every`` steps,
  plus ``Loaded dataset with N training, M eval views`` and ``Done training! Took …``.
  We parse both the log lines and the TTY forms (``NNNN/40000 Steps``, ``Current splat
  count N``) and split on \\r as well as \\n, and treat export files on disk as the ground
  truth.
* Resume: ``--start-iter N`` only moves the loop start; the initial splats come from
  whatever ``.ply`` the dataset folder holds (``init.ply`` wins, else the last one sorted).
  ``--resume-from PLY`` therefore copies that export in as ``init.ply`` for the run and
  removes it afterwards. Mechanically supported; its effect on quality is untested — M3
  decides whether the UI says "resume" or "restart". The checkpoint is validated and
  staged as init.ply *before* train/exports is cleared, and when it lives in train/exports
  (the natural place) that folder keeps every export up to the checkpoint's iteration and
  drops only the later ones — a resume must never delete the file it resumes from.
* Lineage: before brush starts, the dataset it will train on is fingerprinted (rig.npz md5,
  digest of the images and masks) into the train metrics as ``dataset_fingerprint``.
  ``hs render`` compares the rig.npz md5 against the current solve, so a model that
  survived a re-solve cannot render silently in a frame it does not belong to.

* Views: Brush trains on every image under the folder it is given, applies any ``masks/``
  folder it finds, and SKIPS an image listed in ``sparse`` whose file is missing (a warning,
  not an error). ``--exclude L/cap064,R/cap069`` and ``--no-masks`` therefore build
  ``train/view/``: ``sparse`` symlinked, ``images/`` (and ``masks/`` unless ``--no-masks``)
  as per-file symlinks minus the excluded views. The dataset itself is never touched, and
  the init cloud is still ``sparse/points3D.ply``. The view is rebuilt on every run and the
  loaded view count is checked against it.

Keep-awake: cli.py holds ``caffeinate -d -i -m -s`` for the whole ``hs`` process (keepawake.py);
runner.py notices sleeps anyway (lid closed, Apple menu > Sleep) and train records ``slept_s``.

Checks: no sleep over 60 s during the run; final export present; ``element vertex`` within 0.6–1.4× of 2,628 splats per
registered frame (rig6: 170,841 / 65); splat count monotone through growth.
"""
import os
import re
import shutil
import subprocess
import sys
import time

from .. import events, runner
from ..project import dir_digest, md5_file

STAGE = "train"
DEFAULT_BRUSH = "~/Desktop/Apps/brush/target/release/brush"
RE_REFINE = re.compile(r"Refine iter (\d+), (\d+) splats")
RE_STEPS = re.compile(r"(\d+)/(\d+)\s+Steps")
RE_SPLATS = re.compile(r"Current splat count (\d+)")
RE_DATASET = re.compile(r"Loaded dataset with (\d+) training, (\d+) eval views")
RE_EVAL = re.compile(r"Eval iter (\d+): PSNR ([\d.]+), ssim ([\d.]+)")
RE_EXPORT = re.compile(r"export_(\d+)\.ply$")
RE_ERR = re.compile(r"(❌|Error|error:|panicked)")
SPLATS_PER_FRAME_REF = 170841 / 65.0
HEARTBEAT_S = 15.0   # re-emit progress if the trainer has printed nothing for this long


def add_parser(sub):
    p = sub.add_parser("train", help="train a splat with Brush (wraps the brush binary)")
    p.add_argument("--brush", default=os.environ.get("HS_BRUSH", DEFAULT_BRUSH))
    p.add_argument("--total-train-iters", type=int, default=40000)
    p.add_argument("--growth-stop-iter", type=int, default=30000)
    p.add_argument("--refine-every", type=int, default=130)
    p.add_argument("--split-at-screen-size", type=float, default=None, help="Brush default 0.5")
    p.add_argument("--min-scale-factor", type=float, default=None,
                   help="Mip-Splatting 3D-filter strength (Brush >= #541 default 0.1; 0 = off, the pre-#541 behaviour). "
                        "Always passed explicitly when the binary supports it, so the manifest records it")
    p.add_argument("--export-every", type=int, default=2500)
    p.add_argument("--resume-from", default=None, help="an export_NNNNN.ply to continue from (experimental)")
    p.add_argument("--start-iter", type=int, default=None, help="with --resume-from; default: parsed from its name")
    p.add_argument("--no-caffeinate", action="store_true",
                   help="do not hold the Mac awake (also HS_NO_CAFFEINATE=1)")
    p.add_argument("--brush-args", default="", help="extra arguments passed to brush verbatim")
    p.add_argument("--exclude", default="",
                   help="views to leave out, comma separated: L/cap064,R/cap069 (cap064_L also accepted)")
    p.add_argument("--no-masks", action="store_true",
                   help="train without train/dataset/masks even though it exists")
    return p


MIN_SCALE_DEFAULT = 0.1   # Brush #541 (dd5ea36, 2026-09-13)


def brush_info(brush):
    """What we know about the binary: git commit/branch of its checkout, and which flags it has."""
    info = {"path": brush}
    try:
        info["mtime"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(brush)))
    except OSError:
        pass
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(brush))))  # <repo>/target/release/brush
    if os.path.isdir(os.path.join(root, ".git")):
        def git(*args):
            r = subprocess.run(["git", "-C", root] + list(args), capture_output=True, text=True, timeout=10)
            return r.stdout.strip() if r.returncode == 0 else None
        info["git_commit"] = git("rev-parse", "--short", "HEAD")
        info["git_branch"] = git("rev-parse", "--abbrev-ref", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
        info["git_dirty"] = bool(dirty) if dirty is not None else None
    try:
        r = subprocess.run([brush, "--help"], capture_output=True, text=True, timeout=30)
        info["flags"] = sorted(set(re.findall(r"(--[a-z][a-z0-9-]+)", r.stdout + r.stderr)))
    except (OSError, subprocess.TimeoutExpired):
        info["flags"] = None
    return info


def ply_vertex_count(path):
    with open(path, "rb") as f:
        head = f.read(4096)
    m = re.search(rb"element vertex (\d+)", head)
    return int(m.group(1)) if m else None


def list_exports(d):
    if not os.path.isdir(d):
        return []
    out = []
    for f in os.listdir(d):
        m = RE_EXPORT.match(f)
        if m:
            out.append((int(m.group(1)), os.path.join(d, f)))
    return sorted(out)


def parse_exclude(text):
    """'L/cap064, cap069_R, R/cap070.jpg' -> {'L/cap064', 'R/cap069', 'R/cap070'}"""
    out = set()
    for tok in (t.strip() for t in (text or "").split(",")):
        if not tok:
            continue
        tok = os.path.splitext(tok)[0]
        m = re.fullmatch(r"([A-Za-z0-9-]+)_([LR])", tok)
        if m:
            tok = f"{m.group(2)}/{m.group(1)}"
        if re.fullmatch(r"[A-Za-z0-9-]+", tok):        # a bare camera id on an array: GA -> L/GA
            tok = "L/" + tok
        if not re.fullmatch(r"[LR]/[A-Za-z0-9-]+", tok):
            raise events.StageError(f"--exclude: cannot read '{tok}'", hint="write views as L/cap064,R/cap069 (or GA,GB on an array)")
        out.add(tok)
    return out


def build_view(dataset, view, exclude, use_masks):
    """train/view: the dataset as Brush should see it. -> {'images': n, 'masks': n}"""
    if os.path.islink(view):
        os.remove(view)
    elif os.path.isdir(view):
        shutil.rmtree(view)          # removes the links, never what they point at
    os.makedirs(view)
    os.symlink(os.path.join(dataset, "sparse"), os.path.join(view, "sparse"))
    counts = {}
    for top in ("images", "masks") if use_masks else ("images",):
        src_top = os.path.join(dataset, top)
        if not os.path.isdir(src_top):
            continue
        n = 0
        for root, _dirs, files in os.walk(src_top):
            rel = os.path.relpath(root, src_top)
            for f in sorted(files):
                key = os.path.normpath(os.path.join(rel, os.path.splitext(f)[0]))
                if key in exclude:
                    continue
                dst_dir = os.path.normpath(os.path.join(view, top, rel))
                os.makedirs(dst_dir, exist_ok=True)
                os.symlink(os.path.join(root, f), os.path.join(dst_dir, f))
                n += 1
        counts[top] = n
    return counts


def run(a, pj):
    pj.require(STAGE)
    brush = runner.which(a.brush)
    if not brush:
        raise events.StageError(f"brush binary not found at {a.brush}",
                                hint="build the Brush fork (cargo build --release -p brush-app) or pass --brush PATH / HS_BRUSH")
    dataset = pj.dataset_dir
    if not os.path.isdir(dataset) or not os.path.exists(os.path.join(dataset, "sparse")):
        raise events.StageError("no train/dataset (undistorted COLMAP set)", hint="hs solve first")
    n_frames = int(pj.stage("solve").get("metrics", {}).get("num_frames") or 0)

    # exposure and masks write into train/dataset and a re-solve wipes it; they are outside
    # the STAGES chain so they cannot block, but training on a dataset whose normalisation or
    # silhouettes were deleted underneath it is a silent wrong answer, not a warning.
    exclude = parse_exclude(getattr(a, "exclude", ""))
    use_masks = not getattr(a, "no_masks", False)
    for opt in ("exposure", "masks") if use_masks else ("exposure",):
        st_opt = pj.status(opt)
        if st_opt == "stale":
            pj.check(STAGE, f"{opt}_still_applied", False,
                     value=f"'{opt}' ran earlier but a later stage rewrote train/dataset; "
                           f"re-run `hs {opt} --project {pj.root}` or accept a dataset without it")

    # validate the checkpoint before anything is deleted: the natural place for it is
    # train/exports, which this stage clears, and a resume that wipes its own checkpoint
    # fails with "not found" after the damage is done
    start_iter, src, src_md5 = 0, None, None
    if a.resume_from:
        src = os.path.abspath(a.resume_from)
        if not os.path.isfile(src):
            raise events.StageError(f"--resume-from not found: {src}")
        if ply_vertex_count(src) is None:
            raise events.StageError(f"--resume-from is not a PLY with a vertex element: {src}")
        m = RE_EXPORT.search(os.path.basename(src))
        start_iter = a.start_iter if a.start_iter is not None else (int(m.group(1)) if m else 0)
        src_md5 = md5_file(src)

    # train/ holds the dataset written by solve; wipe only exports + our own files
    pj.begin(STAGE, argv=sys.argv, clean=False)
    exports = pj.exports_dir
    view = pj.path("train", "view")
    view_counts = None
    if exclude or not use_masks:
        img_root = os.path.join(dataset, "images")
        missing = sorted(e for e in exclude if not any(
            os.path.exists(os.path.join(img_root, e + ext)) for ext in (".jpg", ".jpeg", ".png")))
        if missing:
            raise events.StageError(f"--exclude names views that are not in the dataset: {', '.join(missing)}")
        view_counts = build_view(dataset, view, exclude, use_masks)
        brush_root = view
        pj.metric(STAGE, "excluded_views", sorted(exclude))
        pj.metric(STAGE, "masks_used", use_masks and bool(view_counts.get("masks")))
        pj.metric(STAGE, "view", {"path": pj.rel(view), **view_counts})
    else:
        if os.path.islink(view) or os.path.isdir(view):
            (os.remove if os.path.islink(view) else shutil.rmtree)(view)   # a stale view must not linger
        brush_root = dataset
        pj.metric(STAGE, "excluded_views", [])
        pj.metric(STAGE, "masks_used", os.path.isdir(os.path.join(dataset, "masks")))
    init_ply = os.path.join(brush_root, "init.ply")
    if os.path.exists(init_ply):
        os.remove(init_ply)  # a stale resume file would silently seed a fresh run
    if src:
        shutil.copy2(src, init_ply)   # staged before the clear, so the source may live in exports
        if md5_file(init_ply) != src_md5:
            raise events.StageError(f"copy of --resume-from does not match its source: {src}")
    resuming_from_exports = bool(src) and os.path.dirname(src) == os.path.abspath(exports)
    if os.path.isdir(exports):
        if resuming_from_exports:
            # keep the history this run continues (everything up to the checkpoint); drop
            # only the exports the resumed run will supersede
            for it, p in list_exports(exports):
                if it > start_iter and p != src:
                    os.remove(p)
        else:
            shutil.rmtree(exports)
    os.makedirs(exports, exist_ok=True)
    if src:
        pj.metric(STAGE, "resume_from", pj.rel(src) if src.startswith(pj.root) else src)
        pj.metric(STAGE, "resume_from_md5", src_md5)
        pj.metric(STAGE, "start_iter", start_iter)
        pj.check(STAGE, "resume_experimental", False,
                 value=f"resuming from {os.path.basename(src)} at iter {start_iter}: mechanically supported, quality unverified")

    # what this model is trained against — render checks the rig against the current solve
    events.start(STAGE, "fingerprint")
    img_d, img_n = dir_digest(os.path.join(dataset, "images"), (".jpg", ".jpeg", ".png"))
    msk_d, msk_n = dir_digest(os.path.join(dataset, "masks"), (".png",))
    fp = {"rig_npz_md5": md5_file(pj.rig_npz) if os.path.exists(pj.rig_npz) else None,
          "images": {"digest": img_d, "count": img_n},
          "masks": {"digest": msk_d, "count": msk_n} if msk_n else None,
          "exposure": pj.status("exposure"), "masks_stage": pj.status("masks"),
          "excluded_views": sorted(exclude), "masks_used": use_masks and bool(msk_n)}
    pj.metric(STAGE, "dataset_fingerprint", fp)

    argv = [brush, brush_root,
            "--total-train-iters", str(a.total_train_iters),
            "--growth-stop-iter", str(a.growth_stop_iter),
            "--refine-every", str(a.refine_every),
            "--export-every", str(a.export_every),
            "--export-path", exports,
            "--export-name", "export_{iter}.ply"]
    binfo = brush_info(brush)
    flags = binfo.get("flags")
    msf = a.min_scale_factor
    if flags is not None and "--min-scale-factor" in flags:
        msf = MIN_SCALE_DEFAULT if msf is None else msf
        if "--min-scale-factor" not in a.brush_args:
            argv += ["--min-scale-factor", str(msf)]
    elif msf is not None:
        raise events.StageError("--min-scale-factor given but this brush has no such flag",
                                hint="update Brush to >= #541 (dd5ea36) or drop the option")
    else:
        msf = 0.0   # pre-#541 binaries have no 3D filter
    pj.metric(STAGE, "brush_config", {"min_scale_factor": msf, "commit": binfo.get("git_commit"),
                                      "branch": binfo.get("git_branch"), "dirty": binfo.get("git_dirty")})
    if a.split_at_screen_size is not None:
        argv += ["--split-at-screen-size", str(a.split_at_screen_size)]
    if start_iter:
        argv += ["--start-iter", str(start_iter)]
    if a.brush_args:
        argv += a.brush_args.split()
    # keep-awake is held by cli.py for the whole stage (keepawake.py), not wrapped around brush
    pj.record_tool("brush", {k: v for k, v in binfo.items() if k != "flags"})
    pj.m["stages"][STAGE]["brush_argv"] = argv
    pj.save()
    env = {"RUST_LOG": os.environ.get("RUST_LOG", "info,wgpu=warn,wgpu_core=warn,wgpu_hal=warn,naga=warn")}

    total = a.total_train_iters
    st = {"iter": start_iter, "splats": None, "t0": time.monotonic(), "iter0": start_iter,
          "seen_exports": set(p for _, p in list_exports(exports)),   # retained pre-resume history
          "growth": [], "errors": [], "last_emit": 0.0}

    def _progress(force=False):
        el = time.monotonic() - st["t0"]
        done = st["iter"]
        rate = (done - st["iter0"]) / el if el > 1 and done > st["iter0"] else None
        eta = (total - done) / rate if rate else None
        st["last_emit"] = time.monotonic()
        events.progress(STAGE, done, total, rate=rate, eta_s=eta,
                        detail=f"{st['splats']} splats" if st["splats"] else None, step="train", force=force)

    def on_line(line):
        m = RE_REFINE.search(line)
        if m:
            st["iter"], st["splats"] = int(m.group(1)), int(m.group(2))
            st["growth"].append((st["iter"], st["splats"]))
            _progress()
            return
        m = RE_STEPS.search(line)
        if m:
            st["iter"] = int(m.group(1))
            _progress()
            return
        m = RE_SPLATS.search(line)
        if m:
            st["splats"] = int(m.group(1))
            return
        m = RE_DATASET.search(line)
        if m:
            pj.metric(STAGE, "train_views", int(m.group(1)))
            return
        m = RE_EVAL.search(line)
        if m:
            events.metric(STAGE, "eval_psnr", float(m.group(2)), iter=int(m.group(1)))
            return
        if RE_ERR.search(line):
            st["errors"].append(line)

    def tick(final=False):
        for it, p in list_exports(exports):
            if p in st["seen_exports"]:
                continue
            # a file still being written is shorter than its header claims; wait a tick
            if not final and time.time() - os.path.getmtime(p) < 2.0:
                continue
            st["seen_exports"].add(p)
            n = ply_vertex_count(p)
            pj.artifact(STAGE, p, "ply")
            events.metric(STAGE, "export_splats", n, iter=it)
            st["iter"] = max(st["iter"], it)
            _progress(force=True)
        # Brush stops refining ~2,000 iterations before the end (rig6: last "Refine iter" at
        # 37961 of 40000), so the step counter goes quiet for the last ~2.5 minutes. Re-emit
        # the last known position as a heartbeat so a consumer can tell "still training" from
        # "hung". `done` deliberately does not advance — we do not know the iteration.
        if not final and time.monotonic() - st["last_emit"] > HEARTBEAT_S:
            _progress(force=True)

    events.start(STAGE, "brush")
    pid_file = pj.path("train", "brush.pid")
    try:
        res = runner.run(argv, STAGE, log_path=pj.log_path(STAGE), on_line=on_line, tick=tick,
                         tick_interval=3.0, pid_file=pid_file, env=env, check=False)
    finally:
        # Brush may write an export AT the start iteration, i.e. over the checkpoint this run
        # resumed from (the fake does: range(start, total + 1)). init.ply is a verified copy of
        # it, so put the original back before init.ply goes.
        if resuming_from_exports and os.path.exists(init_ply) and \
                (not os.path.exists(src) or md5_file(src) != src_md5):
            shutil.copy2(init_ply, src)
            events.log(STAGE, f"[hs] brush overwrote the checkpoint {os.path.basename(src)}; restored it")
        if os.path.exists(init_ply):
            os.remove(init_ply)
    tick(final=True)
    pj.metric(STAGE, "elapsed_s", round(res.elapsed, 1))
    pj.metric(STAGE, "wall_s", round(res.wall_s, 1))
    pj.metric(STAGE, "slept_s", round(res.slept_s, 1))
    pj.check(STAGE, "no_sleep_during_run", res.slept_s <= 60,
             value=(f"slept {res.slept_s / 60:.1f} min in {len(res.sleeps)} gaps over {res.wall_s / 60:.1f} min wall"
                    if res.sleeps else f"awake for all {res.wall_s / 60:.1f} min"))
    exps = list_exports(exports)
    final = [p for it, p in exps if it == total]
    if res.returncode != 0 and not final:
        hint = " | ".join(st["errors"][-3:]) or " | ".join(res.lines[-3:])
        raise events.StageError(f"brush exited {res.returncode}", hint=hint)

    if view_counts is not None:
        got = pj.stage(STAGE).get("metrics", {}).get("train_views")
        pj.check(STAGE, "view_count_matches", got == view_counts["images"],
                 value=f"brush loaded {got} views; train/view holds {view_counts['images']} "
                       f"({len(exclude)} excluded, masks {'on' if use_masks else 'off'})")
    ok_final = pj.check(STAGE, "final_export_present", bool(final),
                        value=os.path.basename(final[0]) if final else f"no export_{total:05d}.ply in train/exports")
    if final:
        n = ply_vertex_count(final[0])
        pj.metric(STAGE, "final_export", pj.rel(final[0]))
        pj.metric(STAGE, "final_export_md5", md5_file(final[0]))
        pj.metric(STAGE, "final_splats", n)
        if n_frames and n:
            lo, hi = 0.6 * SPLATS_PER_FRAME_REF * n_frames, 1.4 * SPLATS_PER_FRAME_REF * n_frames
            pj.check(STAGE, "splat_count_in_range", lo <= n <= hi,
                     value=f"{n:,} splats for {n_frames} frames (want {lo:,.0f}–{hi:,.0f}; rig6 170,841/65)")
    growth = [(it, s) for it, s in st["growth"] if it <= a.growth_stop_iter]
    if growth:
        mono = all(b >= a_ for (_, a_), (_, b) in zip(growth, growth[1:]))
        pj.metric(STAGE, "growth_curve", [[it, s] for it, s in st["growth"]])
        pj.check(STAGE, "splat_count_monotone_through_growth", mono,
                 value=f"{growth[0][1]} → {growth[-1][1]} splats over {len(growth)} refine steps")
    else:
        pj.check(STAGE, "progress_lines_seen", False,
                 value="no 'Refine iter N, M splats.' lines seen — is RUST_LOG reaching brush? exports on disk are the truth")
    pj.finish(STAGE, ok=ok_final)
    if not ok_final:
        raise events.StageError("training finished without the final export", hint="see logs/train.log")
