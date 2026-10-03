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
* ``--init lidar`` stages ``scale/lidar_init.ply`` (``hs scale --lidar SCAN --init-points``: the
  LiDAR scan as splats in the training set's metres, plus the sparse points past its reach) as
  ``init.ply`` the same way: validated before the clear (the file's md5 and the rig.npz it was
  written for, from the record in ``manifest.scale`` or ``stages.scale.lidar_check``), md5-checked
  after the copy, removed after the run. It and ``--resume-from`` both name the initial splats, so
  they are refused together. ``init`` ("sparse", "lidar" or "resume") and the file's md5 go into
  the metrics and ``dataset_fingerprint``, which ``hs archive`` copies.
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

* Layers: one model explains one part of the scene, named by ``--layer``. ``full`` trains without
  masks at all (the whole room; what ``--no-masks`` has always done). ``subject`` trains with
  ``train/dataset/masks`` as they are. ``background`` trains the complement from those same files
  through Brush's ``--invert-masks`` — the masks are written once and read both ways. The default
  is ``subject`` when the folder exists and ``full`` when it does not, so old invocations are
  unchanged. ``--alpha-mode`` picks what the mask means to the loss (see ``hs masks``): Brush's
  default ``masked`` leaves those pixels out of the loss, ``transparent`` turns on the L1 on
  rendered alpha that pushes the model to be empty outside the silhouette. Both land in the
  manifest (``layer``, ``brush_config.alpha_mode``, ``brush_config.invert_masks``) and in
  ``dataset_fingerprint``, so two models trained off one dataset can always be told apart.

* Depth: ``--depth-weight W`` adds Brush's depth loss (fork branch depth-loss:
  ``--depth-loss-weight``). The reference is the LiDAR scan `hs scale --lidar SCAN --init-points`
  registered — the scan rows of ``scale/lidar_init.ply``, validated against the current rig.npz
  exactly as ``--init lidar`` is — rendered into every training view by hs/depthmaps.py:
  ``train/depth/<eye>/<view>.png`` (16-bit, ``--depth-res`` px on the long edge), linked into
  ``train/view/depths`` where Brush looks for them, with ``train/depth/depth_report.json`` and a
  contact sheet beside them. Photometric loss alone draws a reflection as real content behind a
  translucent surface; the scan says where the surface is. A subject layer is validated on a
  real capture (Stormtrooper helmet, 2026-10-01); background and full layers use the simpler
  visibility test and are not. What the scan did not reach has no reference and trains as before:
  ``depth_reference_coverage`` says how much that is. Brush logs the mean relative depth error
  every 500 steps; it lands in the metrics as ``depth_error_curve``, and ``depth_error_held`` asks
  that the last one be within twice the tolerance.
* Spread: ``--depth-spread-weight W`` adds Brush's depth spread term (``--depth-spread-weight``):
  per ray, the standard deviation of depth over the blend weights as a share of the mean depth,
  whatever exceeds ``--depth-spread-tolerance``. It comes out of the same extra render, needs no
  scan, and applies to every view and every solid pixel — so it reaches what the scan did not
  (run C, 2026-10-02: the depth loss halved the depth error where the scan was and left the
  helmet's back, which it never saw, 40 mm thick). It makes a surface thin and does not say
  where it is; with ``--depth-weight`` the scan anchors it where there is one. Brush logs the
  mean relative spread every 500 steps (``depth_spread_curve``, ``depth_spread_held``).
  It counts in the loss from step ``--depth-spread-from`` (an eighth of the run, at most 5000)
  and is only measured before: a ray through a faint new surface and an opaque one behind it is
  thinnest with the new surface gone, so from step 0 the term can stop a surface from forming.
  The lines before that step are the model without the term (``depth_rel_spread_before``).

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
RE_DEPTH_VIEWS = re.compile(r"Depth maps for (\d+) of (\d+) training views")
RE_DEPTH_ERR = re.compile(r"Depth loss at step (\d+): mean relative depth error ([\d.]+)")
RE_DEPTH_SPREAD = re.compile(r"Depth spread at step (\d+): mean relative depth spread ([\d.]+)")
RE_EXPORT = re.compile(r"export_(\d+)\.ply$")
RE_ERR = re.compile(r"(❌|Error|error:|panicked)")
SPLATS_PER_FRAME_REF = 170841 / 65.0          # rig6, kept as the historical reference only
# A 0.6-1.4x band around rig6's 2,628 splats/frame failed every model trained since: the 09-15
# head 8,833/frame, the body 13,567, the 04-03 array 15,394 -- three captures, three subjects,
# all 3-6x rig6, none of them wrong. rig6 was a small coin on a short orbit. The check is worth
# keeping only as a sanity bound on the runaway and the collapse, so it is one: under 500 means
# growth never took, over 40,000 means a model too heavy to render at speed and probably
# floaters. Recalibrate again when there are enough subjects to know what normal is.
SPLATS_PER_FRAME_MIN, SPLATS_PER_FRAME_MAX = 500, 40000
HEARTBEAT_S = 15.0   # re-emit progress if the trainer has printed nothing for this long
DEPTH_COVERAGE_OK = 0.7   # depth_reference_coverage: the scan should see this share of the layer's pixels
DEPTH_ERROR_OK = 2.0      # depth_error_held: the final mean relative depth error, in tolerances


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
    p.add_argument("--init", choices=("sparse", "lidar"), default="sparse",
                   help="the initial splats. sparse (default): Brush's own start, the solve's sparse points. "
                        "lidar: scale/lidar_init.ply, the LiDAR scan in the training set's frame "
                        "(`hs scale --lidar SCAN --init-points`); not with --resume-from")
    p.add_argument("--start-iter", type=int, default=None, help="with --resume-from; default: parsed from its name")
    p.add_argument("--no-caffeinate", action="store_true",
                   help="do not hold the Mac awake (also HS_NO_CAFFEINATE=1)")
    p.add_argument("--brush-args", default="", help="extra arguments passed to brush verbatim")
    p.add_argument("--exclude", default="",
                   help="views to leave out, comma separated: L/cap064,R/cap069 (cap064_L also accepted); "
                        "@holdout = the captures in solve/holdout.json (hs cameras --holdout N --write)")
    p.add_argument("--no-masks", action="store_true",
                   help="train without train/dataset/masks even though it exists")
    p.add_argument("--layer", choices=("full", "subject", "background"), default=None,
                   help="which part of the scene this model explains. full: no masks, the whole room. "
                        "subject: the masks as they are. background: the same masks inverted, the room "
                        "without the subject. Default: subject if train/dataset/masks exists, else full")
    p.add_argument("--alpha-mode", choices=("masked", "transparent"), default=None,
                   help="how Brush reads the mask alpha. masked (Brush's default): masked-out pixels are "
                        "left out of the loss, so outside the silhouette is unsupervised, not empty. "
                        "transparent: the ground truth is premultiplied and an L1 on rendered alpha "
                        "(--match-alpha-weight, default 0.1) pushes the model to be empty out there")
    p.add_argument("--depth-weight", type=float, default=0.0,
                   help="weight of Brush's depth loss against the registered LiDAR scan "
                        "(hs scale --lidar SCAN --init-points); 0 (default) = off. The loss is the mean over "
                        "the measured pixels of max(0, |ln(rendered depth / scan depth)| - tolerance)")
    p.add_argument("--depth-tolerance", type=float, default=0.01,
                   help="--depth-weight: relative depth error that costs nothing (0.01 = 1%% of the "
                        "distance: 5 mm at 0.5 m), sized to the scan's own error")
    p.add_argument("--depth-res", type=int, default=512,
                   help="--depth-weight: long edge of the depth maps, px. Brush renders the depth pass at "
                        "this size every step")
    p.add_argument("--depth-every", type=int, default=1,
                   help="--depth-weight / --depth-spread-weight: apply them every Nth step (the pass costs a "
                        "second, small render)")
    p.add_argument("--depth-spread-weight", type=float, default=0.0,
                   help="weight of Brush's depth spread term: per ray, std(depth) / mean depth over the blend "
                        "weights, above the tolerance. Thins a smoky surface; needs no scan and reaches every "
                        "view. 0 (default) = off")
    p.add_argument("--depth-spread-tolerance", type=float, default=0.005,
                   help="--depth-spread-weight: relative spread that costs nothing (0.005 = a standard "
                        "deviation of 0.5%% of the distance: 2.5 mm at 0.5 m)")
    p.add_argument("--depth-spread-from", type=int, default=None,
                   help="--depth-spread-weight: first step the term counts in the loss; before it the spread "
                        "is only measured and logged. Default: an eighth of the run, at most 5000. From step 0 "
                        "it can keep a surface from forming")
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
        help_text = r.stdout + r.stderr
        info["flags"] = sorted(set(re.findall(r"(--[a-z][a-z0-9-]+)", help_text)))
        # clap prints a value placeholder only for options that take one:
        #   --invert-masks            (bare: ArgAction::SetTrue)
        #   --alpha-mode <ALPHA_MODE> (takes a value)
        # A flag can appear twice (usage line, then the options list); a placeholder anywhere wins.
        takes = {}
        for m in re.finditer(r"(--[a-z][a-z0-9-]+)([ \t]*[<=][^>\n]*>?)?", help_text):
            takes[m.group(1)] = takes.get(m.group(1), False) or bool(m.group(2))
        info["flag_args"] = takes
    except (OSError, subprocess.TimeoutExpired):
        info["flags"] = None
        info["flag_args"] = None
    return info


def brush_flag(binfo, name, value=None):
    """argv for a Brush flag, asking the binary itself whether it wants the value spelled out.

    `--invert-masks` is `#[arg(long, default_value = "false")] bool`, which clap renders as a bare
    flag; a future build could give it an explicit value. Read it off --help rather than guess.
    """
    args = (binfo.get("flag_args") or {})
    if value is None:
        return [name, "true"] if args.get(name) else [name]
    return [name, str(value)]


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


def parse_exclude(text, root=None):
    """'L/cap064, cap069_R, R/cap070.jpg' -> {'L/cap064', 'R/cap069', 'R/cap070'}

    ``@holdout`` expands to every view solve/holdout.json holds out (both eyes of each capture
    on a stereo rig; `hs cameras --holdout N --write` makes it), and mixes with names:
    ``@holdout,L/cap099``. It needs ``root``, the project folder."""
    out = set()
    for tok in (t.strip() for t in (text or "").split(",")):
        if not tok:
            continue
        if tok.startswith("@"):
            if tok != "@holdout":
                raise events.StageError(f"--exclude: unknown list '{tok}'", hint="the one list is @holdout (solve/holdout.json)")
            if root is None:
                raise events.StageError("--exclude @holdout needs the project folder")
            from ..coverage import load_holdout
            try:
                h = load_holdout(root)
            except ValueError as e:
                raise events.StageError(str(e))
            eyes = ("L", "R") if h.get("stereo", True) else ("L",)
            out |= set(h.get("exclude") or [f"{e}/{n}" for n in h["names"] for e in eyes])
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


def build_view(dataset, view, exclude, use_masks, depths=None):
    """train/view: the dataset as Brush should see it. -> {'images': n, 'masks': n[, 'depths': n]}

    `depths` is a folder of depth maps laid out like the masks (hs/depthmaps.py); its PNGs are
    linked as ``depths/`` next to ``images/``, which is where Brush looks."""
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
    if depths and os.path.isdir(depths):
        n = 0
        for root, _dirs, files in os.walk(depths):
            rel = os.path.relpath(root, depths)
            for f in sorted(files):
                if not f.endswith(".png") or os.path.normpath(os.path.join(rel, f[:-4])) in exclude:
                    continue
                dst_dir = os.path.normpath(os.path.join(view, "depths", rel))
                os.makedirs(dst_dir, exist_ok=True)
                os.symlink(os.path.join(root, f), os.path.join(dst_dir, f))
                n += 1
        counts["depths"] = n
    return counts


def lidar_init_ply(pj, what="--init lidar"):
    """The scan init `hs scale --init-points` recorded, verified: (absolute path, md5, record).
    `what` names the option that needs it, for the messages.
    Records live in manifest.scale (an applied scale) and stages.scale.lidar_check (a measurement
    that was not applied); the newest whose file is unchanged and whose rig.npz is the current one
    wins. A re-solve or a scale applied since the file was written moved the frame under it."""
    from ..project import md5_file
    recs = [r for r in ((pj.m.get("scale") or {}), (pj.stage("scale").get("lidar_check") or {})) if r.get("init_ply")]
    if not recs:
        raise events.StageError(f"{what}: no LiDAR init points recorded for this project",
                                hint=f"hs scale -p {pj.root} --lidar SCAN --init-points (with --dry-run to leave "
                                     "the scale alone)")
    rig_md5 = md5_file(pj.rig_npz) if os.path.exists(pj.rig_npz) else None
    why = []
    for r in sorted(recs, key=lambda r: r.get("at") or "", reverse=True):
        p = pj.path(r["init_ply"]) if not os.path.isabs(r["init_ply"]) else r["init_ply"]
        if not os.path.isfile(p):
            why.append(f"{r['init_ply']} is missing")
            continue
        md5 = md5_file(p)
        if md5 != r.get("init_ply_md5"):
            why.append(f"{r['init_ply']} changed since hs scale wrote it")
            continue
        if r.get("init_rig_md5") != rig_md5:
            why.append(f"{r['init_ply']} was written for another rig.npz (re-solved or scaled since)")
            continue
        return p, md5, r
    raise events.StageError(f"{what}: " + "; ".join(why),
                            hint=f"re-run hs scale -p {pj.root} --lidar SCAN --init-points on the solve as it is now")


def build_depth(pj, scan_ply, rec, layer, use_masks, exclude, res):
    """train/depth: the registered scan as every training view sees it (hs/depthmaps.py).
    -> the report (also written to train/depth/depth_report.json).

    The scan is the first ``init_points_scan`` rows of scale/lidar_init.ply, which holds it in the
    training set's units (metres = rig.npz mm / 1000); the rows after are the solve's own sparse
    points and are no measurement of anything."""
    import json

    from .. import depthmaps, lidar as lidarlib, rig as riglib
    out = pj.path("train", "depth")
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    try:
        pts = lidarlib.load_scan(scan_ply, units="m").points
    except (ValueError, OSError) as e:
        raise events.StageError(f"--depth-weight: cannot read {pj.rel(scan_ply)}: {e}")
    n_scan = rec.get("init_points_scan")
    if n_scan:
        pts = pts[:int(n_scan)]
    G, names, _L = riglib.load(pj.rig_npz)
    dataset = pj.dataset_dir
    masks = os.path.join(dataset, "masks")
    map_layer = layer if (use_masks and os.path.isdir(masks)) else "full"
    try:
        rep = depthmaps.build(pts, G, names, out, layer=map_layer, masks_dir=masks,
                              images_dir=os.path.join(dataset, "images"), exclude=exclude, long_edge=int(res),
                              progress=lambda done, total: events.progress(STAGE, done, total, step="depth"))
    except ValueError as e:
        raise events.StageError(f"--depth-weight: {e}",
                                hint="scale/lidar_aligned.ply against the solve shows where the scan sits; "
                                     "train/depth/depth_sheet.jpg is written when any view gets depth")
    rep["source"] = pj.rel(scan_ply)
    with open(os.path.join(out, "depth_report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    return rep


def run(a, pj):
    pj.require(STAGE)
    init_mode = getattr(a, "init", None) or "sparse"
    if init_mode == "lidar" and a.resume_from:
        raise events.StageError("--init lidar and --resume-from both name the initial splats: give one",
                                hint="--resume-from continues a model; --init lidar starts a new one from the scan")
    brush = runner.which(a.brush)
    if not brush:
        raise events.StageError(f"brush binary not found at {a.brush}",
                                hint="build the Brush fork (cargo build --release -p brush-app) or pass --brush PATH / HS_BRUSH")
    binfo = brush_info(brush)
    flags = binfo.get("flags")
    depth_w = float(getattr(a, "depth_weight", 0.0) or 0.0)
    depth_tol = float(getattr(a, "depth_tolerance", 0.01))
    depth_res = int(getattr(a, "depth_res", 512))
    depth_every = int(getattr(a, "depth_every", 1))
    spread_w = float(getattr(a, "depth_spread_weight", 0.0) or 0.0)
    spread_tol = float(getattr(a, "depth_spread_tolerance", 0.005))
    if depth_w < 0 or depth_tol < 0 or depth_every < 1 or depth_res < 32 or spread_w < 0 or spread_tol < 0:
        raise events.StageError("--depth-weight, --depth-tolerance, --depth-spread-weight and "
                                "--depth-spread-tolerance must be >= 0, --depth-every >= 1, --depth-res >= 32")
    spread_from = getattr(a, "depth_spread_from", None)
    spread_from = min(5000, a.total_train_iters // 8) if spread_from is None else int(spread_from)
    if spread_w and not 0 <= spread_from < a.total_train_iters:
        raise events.StageError(f"--depth-spread-from {spread_from} leaves the spread term no steps: the run "
                                f"has {a.total_train_iters}")
    if spread_w and flags is not None and "--depth-spread-from-iter" not in flags:
        raise events.StageError("--depth-spread-weight needs Brush's --depth-spread-weight and "
                                "--depth-spread-from-iter, and this binary lacks them",
                                hint="update the Brush fork to its depth-loss branch and rebuild "
                                     "(cargo build --release)")
    if depth_w and flags is not None and "--depth-loss-weight" not in flags:
        raise events.StageError("--depth-weight needs Brush's --depth-loss-weight, and this binary has no such flag",
                                hint="update the Brush fork to its depth-loss branch and rebuild "
                                     "(cargo build --release)")
    dataset = pj.dataset_dir
    if not os.path.isdir(dataset) or not os.path.exists(os.path.join(dataset, "sparse")):
        raise events.StageError("no train/dataset (undistorted COLMAP set)", hint="hs solve first")
    n_frames = int(pj.stage("solve").get("metrics", {}).get("num_frames") or 0)

    # exposure and masks write into train/dataset and a re-solve wipes it; they are outside
    # the STAGES chain so they cannot block, but training on a dataset whose normalisation or
    # silhouettes were deleted underneath it is a silent wrong answer, not a warning.
    exclude = parse_exclude(getattr(a, "exclude", ""), root=pj.root)
    # One model explains one part of the scene. `--layer` names which; masks are how Brush is told.
    layer = getattr(a, "layer", None)
    alpha_mode = getattr(a, "alpha_mode", None)
    no_masks = bool(getattr(a, "no_masks", False))
    have_masks = os.path.isdir(os.path.join(dataset, "masks"))
    if layer == "full":
        no_masks = True
    elif layer in ("subject", "background"):
        if no_masks:
            raise events.StageError(f"--layer {layer} is trained with the masks; --no-masks contradicts it",
                                    hint="drop --no-masks, or ask for --layer full")
        if not have_masks:
            raise events.StageError(f"--layer {layer} needs train/dataset/masks, which is not there",
                                    hint=f"hs masks --project {pj.root}")
    use_masks = not no_masks
    if alpha_mode and not (use_masks and have_masks):
        raise events.StageError("--alpha-mode describes how Brush reads the masks, and this run has none",
                                hint="drop --alpha-mode, or ask for --layer subject")
    if layer is None:
        layer = "subject" if (use_masks and have_masks) else "full"
    invert_masks = layer == "background"
    # what Brush will actually do: no masks at all, or masked (its default) unless asked otherwise
    effective_alpha = (alpha_mode or "masked") if (use_masks and have_masks) else None
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
    # --init lidar: the scan's splats (hs scale --init-points), checked against the current rig
    # before anything is cleared, staged as init.ply exactly as a resume checkpoint is
    init_src, init_md5 = None, None
    if init_mode == "lidar":
        init_src, init_md5, _rec = lidar_init_ply(pj)
    # --depth-weight: the same file is the depth reference, and is checked the same way
    depth_src, depth_src_md5, depth_rec = lidar_init_ply(pj, what="--depth-weight") if depth_w else (None, None, None)

    # train/ holds the dataset written by solve; wipe only exports + our own files
    pj.begin(STAGE, argv=sys.argv, clean=False)
    exports = pj.exports_dir
    view = pj.path("train", "view")
    view_counts = None
    depth_rep, depth_dir = None, pj.path("train", "depth")
    if depth_w:
        events.start(STAGE, "depth")
        depth_rep = build_depth(pj, depth_src, depth_rec, layer, use_masks, exclude, depth_res)
        pj.artifact(STAGE, os.path.join(depth_dir, "depth_report.json"), "json")
        if depth_rep.get("sheet"):
            pj.artifact(STAGE, os.path.join(depth_dir, depth_rep["sheet"]), "jpg")
    elif os.path.isdir(depth_dir):
        shutil.rmtree(depth_dir)      # maps of an earlier run must not pass for this one's
    if exclude or not use_masks or depth_w:
        img_root = os.path.join(dataset, "images")
        missing = sorted(e for e in exclude if not any(
            os.path.exists(os.path.join(img_root, e + ext)) for ext in (".jpg", ".jpeg", ".png")))
        if missing:
            raise events.StageError(f"--exclude names views that are not in the dataset: {', '.join(missing)}")
        view_counts = build_view(dataset, view, exclude, use_masks, depths=depth_dir if depth_w else None)
        brush_root = view
        pj.metric(STAGE, "excluded_views", sorted(exclude))
        pj.metric(STAGE, "masks_used", use_masks and bool(view_counts.get("masks")))
        pj.metric(STAGE, "layer", layer)
        pj.metric(STAGE, "view", {"path": pj.rel(view), **view_counts})
    else:
        if os.path.islink(view) or os.path.isdir(view):
            (os.remove if os.path.islink(view) else shutil.rmtree)(view)   # a stale view must not linger
        brush_root = dataset
        pj.metric(STAGE, "excluded_views", [])
        pj.metric(STAGE, "masks_used", os.path.isdir(os.path.join(dataset, "masks")))
        pj.metric(STAGE, "layer", layer)
    init_ply = os.path.join(brush_root, "init.ply")
    if os.path.exists(init_ply):
        os.remove(init_ply)  # a stale resume file would silently seed a fresh run
    if src:
        shutil.copy2(src, init_ply)   # staged before the clear, so the source may live in exports
        if md5_file(init_ply) != src_md5:
            raise events.StageError(f"copy of --resume-from does not match its source: {src}")
    elif init_src:
        shutil.copy2(init_src, init_ply)
        if md5_file(init_ply) != init_md5:
            os.remove(init_ply)
            raise events.StageError(f"copy of the LiDAR init does not match its source: {init_src}")
    init_used = "resume" if src else init_mode
    pj.metric(STAGE, "init", init_used)
    if init_src:
        pj.metric(STAGE, "init_ply", pj.rel(init_src))
        pj.metric(STAGE, "init_ply_md5", init_md5)
        pj.metric(STAGE, "init_splats", ply_vertex_count(init_src))
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
          "excluded_views": sorted(exclude), "masks_used": use_masks and bool(msk_n),
          "layer": layer, "alpha_mode": effective_alpha, "invert_masks": invert_masks,
          "init": init_used}
    if init_src:
        fp["init_md5"] = init_md5
    depth_cfg = None
    if depth_w:
        dep_d, dep_n = dir_digest(depth_dir, (".png",))
        depth_unit = depth_rep["unit_mm"] / 1000.0          # dataset units (rig.npz mm / 1000) per count
        depth_cfg = {"weight": depth_w, "tolerance": depth_tol, "every": depth_every, "res": depth_res,
                     "unit": depth_unit, "source_md5": depth_src_md5}
        fp["depth"] = dict(depth_cfg, digest=dep_d, count=dep_n)
        pj.metric(STAGE, "depth", dict(
            depth_cfg, source=pj.rel(depth_src), maps=pj.rel(depth_dir), layer=depth_rep["layer"],
            views=depth_rep["views"], views_with_depth=depth_rep["views_with_depth"],
            coverage=depth_rep["coverage"], coverage_percentiles=depth_rep["coverage_percentiles"],
            views_under_half=depth_rep["views_under_half"], subject=depth_rep.get("subject"),
            hull=depth_rep.get("hull")))
        pc = depth_rep["coverage_percentiles"] or {}
        of = "the subject's" if depth_rep["layer"] == "subject" else "the layer's"
        pj.check(STAGE, "depth_reference_coverage", depth_rep["coverage"] >= DEPTH_COVERAGE_OK,
                 value=f"the scan gives depth for {100 * depth_rep['coverage']:.1f}% of {of} pixels over "
                       f"{depth_rep['views']} views (median view {100 * pc.get('50', 0):.0f}%, "
                       f"{depth_rep['views_under_half']} views under half, {depth_rep['views_with_depth']} with a "
                       f"map); the rest trains without a depth reference")
    spread_cfg = None
    if spread_w:
        spread_cfg = {"weight": spread_w, "tolerance": spread_tol, "from": spread_from, "every": depth_every,
                      "res": depth_res}
        fp["depth_spread"] = spread_cfg
        pj.metric(STAGE, "depth_spread", spread_cfg)
    pj.metric(STAGE, "dataset_fingerprint", fp)

    argv = [brush, brush_root,
            "--total-train-iters", str(a.total_train_iters),
            "--growth-stop-iter", str(a.growth_stop_iter),
            "--refine-every", str(a.refine_every),
            "--export-every", str(a.export_every),
            "--export-path", exports,
            "--export-name", "export_{iter}.ply"]
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
    # the layer, spelled out to Brush. Both flags are recorded whether or not they are passed,
    # so a manifest says which part of the scene the model was asked to explain.
    if invert_masks and "--invert-masks" not in a.brush_args:
        if flags is not None and "--invert-masks" not in flags:
            raise events.StageError("--layer background needs Brush's --invert-masks, and this binary has no such flag",
                                    hint="update the Brush fork (brush-dataset/src/config.rs: invert_masks)")
        argv += brush_flag(binfo, "--invert-masks")
    if alpha_mode and "--alpha-mode" not in a.brush_args:
        if flags is not None and "--alpha-mode" not in flags:
            raise events.StageError("--alpha-mode given but this brush has no such flag",
                                    hint="update the Brush fork, or drop the option")
        argv += brush_flag(binfo, "--alpha-mode", alpha_mode)
    depth_flags = []
    if depth_w:
        depth_flags += [("--depth-loss-weight", depth_w), ("--depth-loss-tolerance", depth_tol),
                        ("--depth-unit", depth_cfg["unit"])]
    if spread_w:
        # views without a depth map are rendered for the spread at --depth-res, as the maps are
        depth_flags += [("--depth-spread-weight", spread_w), ("--depth-spread-tolerance", spread_tol),
                        ("--depth-spread-from-iter", spread_from),
                        ("--depth-pass-res", depth_res)]
    if depth_flags:
        depth_flags.append(("--depth-loss-every", depth_every))
    for flag, value in depth_flags:
        if flag not in a.brush_args:
            argv += [flag, f"{value:g}"]
    pj.metric(STAGE, "brush_config", {"min_scale_factor": msf, "commit": binfo.get("git_commit"),
                                      "branch": binfo.get("git_branch"), "dirty": binfo.get("git_dirty"),
                                      "layer": layer, "alpha_mode": effective_alpha,
                                      "invert_masks": invert_masks,
                                      "depth_loss_weight": depth_w,
                                      "depth_loss_tolerance": depth_tol if depth_w else None,
                                      "depth_loss_every": depth_every if (depth_w or spread_w) else None,
                                      "depth_spread_weight": spread_w,
                                      "depth_spread_tolerance": spread_tol if spread_w else None,
                                      "depth_spread_from": spread_from if spread_w else None})
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
          "growth": [], "errors": [], "last_emit": 0.0, "depth_err": [], "depth_views": None,
          "depth_spread": []}

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
        m = RE_DEPTH_VIEWS.search(line)
        if m:
            st["depth_views"] = int(m.group(1))
            return
        m = RE_DEPTH_ERR.search(line)
        if m:
            st["depth_err"].append((int(m.group(1)), float(m.group(2))))
            events.metric(STAGE, "depth_rel_error", float(m.group(2)), iter=int(m.group(1)))
            return
        m = RE_DEPTH_SPREAD.search(line)
        if m:
            st["depth_spread"].append((int(m.group(1)), float(m.group(2))))
            events.metric(STAGE, "depth_rel_spread", float(m.group(2)), iter=int(m.group(1)))
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
                       f"({len(exclude)} excluded, layer {layer}, masks {'on' if use_masks else 'off'})")
    if depth_w:
        want = view_counts.get("depths", 0)
        pj.check(STAGE, "depth_maps_loaded", st["depth_views"] == want,
                 value=f"brush found depth maps for {st['depth_views'] or 0} views; train/view/depths holds {want}")
        curve = st["depth_err"]
        if curve:
            pj.metric(STAGE, "depth_error_curve", [[it, e] for it, e in curve])
            pj.metric(STAGE, "depth_rel_error_final", curve[-1][1])
            # not "it fell": a run started from the scan (--init lidar) begins on the reference, and
            # the question is whether the photographs pulled it off again
            worst = max(e for _it, e in curve)
            pj.check(STAGE, "depth_error_held", curve[-1][1] <= DEPTH_ERROR_OK * depth_tol,
                     value=f"mean relative depth error {curve[-1][1]:.4f} at step {curve[-1][0]} "
                           f"(first {curve[0][1]:.4f}, worst {worst:.4f}; ok up to {DEPTH_ERROR_OK:g} x the "
                           f"tolerance {depth_tol:g}). Times the camera distance it is a length: "
                           f"0.02 at 0.5 m is 10 mm. Above the bound the reflections are still drawn "
                           f"behind the surface: raise --depth-weight")
        else:
            pj.check(STAGE, "depth_error_reported", False,
                     value="no 'Depth loss at step N' line seen: the depth loss did not run")
    if spread_w:
        curve = st["depth_spread"]
        if curve:
            pj.metric(STAGE, "depth_spread_curve", [[it, e] for it, e in curve])
            pj.metric(STAGE, "depth_rel_spread_final", curve[-1][1])
            # medians of a few lines, not single ones: each line is one view, and a view of the
            # subject's edge-on side is thick whatever the model
            def median(xs):
                return sorted(xs)[len(xs) // 2]
            # Brush counts the steps of this run (a resumed run starts again at 1). The lines
            # before --depth-spread-from are the model without the term: the "before" for free.
            head = [e for it, e in curve if 1 < it < spread_from][-5:]
            tail = [e for it, e in curve if it >= spread_from][-5:] or [curve[-1][1]]
            late = median(tail)
            before = median(head) if head else None
            if before is not None:
                pj.metric(STAGE, "depth_rel_spread_before", before)
            was = (f"{before:.4f} over the last {len(head)} lines before the term started at step {spread_from}"
                   if before is not None else f"first line {curve[0][1]:.4f}")
            pj.check(STAGE, "depth_spread_held", late <= DEPTH_ERROR_OK * spread_tol,
                     value=f"mean relative depth spread {late:.4f} over the last {len(tail)} lines "
                           f"({was}; last {curve[-1][1]:.4f}; ok up to {DEPTH_ERROR_OK:g} x the "
                           f"tolerance {spread_tol:g}). Times the camera distance it is the standard deviation "
                           f"of depth along a ray: 0.01 at 0.5 m is 5 mm, a surface about 13 mm thick. Above "
                           f"the bound the surface is still smoky: raise --depth-spread-weight")
        else:
            pj.check(STAGE, "depth_spread_reported", False,
                     value="no 'Depth spread at step N' line seen: the spread term did not run")
    ok_final = pj.check(STAGE, "final_export_present", bool(final),
                        value=os.path.basename(final[0]) if final else f"no export_{total:05d}.ply in train/exports")
    if final:
        n = ply_vertex_count(final[0])
        pj.metric(STAGE, "final_export", pj.rel(final[0]))
        pj.metric(STAGE, "final_export_md5", md5_file(final[0]))
        pj.metric(STAGE, "final_splats", n)
        if n_frames and n:
            per = n / n_frames
            pj.metric(STAGE, "splats_per_frame", round(per, 1))
            pj.check(STAGE, "splat_count_in_range", SPLATS_PER_FRAME_MIN <= per <= SPLATS_PER_FRAME_MAX,
                     value=f"{n:,} splats for {n_frames} frames = {per:,.0f}/frame "
                           f"(sanity band {SPLATS_PER_FRAME_MIN:,}–{SPLATS_PER_FRAME_MAX:,}; "
                           f"rig6 {SPLATS_PER_FRAME_REF:,.0f}, 09-15 head 8,833, body 13,567, 04-03 array 15,394)")
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
