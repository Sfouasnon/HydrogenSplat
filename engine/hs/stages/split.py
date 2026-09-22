"""hs split — label a trained model's splats subject / background from the 2D masks, after training.

    hs split -p P --ply archive/holdout-base/export_40000.ply --exclude @holdout --masks vision

Why post hoc. A subject layer trained against masks (hs train --layer subject) is a second 40k-
iteration train and still leaves the silhouette band to the alpha loss. The model we already
have knows, per pixel of every training photograph, which splats painted it and how much
(hs.splatweights). FlashSplat's observation is that with those weights the best 0/1 label per
splat under the masks has a closed form: ``p_i = sum(w * M) / sum(w)`` over every (view, cell)
the splat touches, with M the mask's mean over the cell. So a split is one pass of the renderer's
forward weights, no optimisation, no retraining.

Steps:
  1. weights over the TRAINING views only. The hold-outs never touch a label, or a hold-out
     score of the split layers would be graded on views that shaped them. Default: the views the
     model itself was trained without (its archive manifest, or the train stage's fingerprint);
     --exclude overrides (a list as for hs train, or @holdout for solve/holdout.json).
  2. uplift ``p_i`` as above. Views without a mask contribute nothing.
  3. refine the ambiguous ones (0.2 < p < 0.8, and splats no training view saw): a KNN over
     (xyz, DC colour) with cv2.flann, and ``p <- mean(p of neighbours)`` for --iters rounds. p
     stays a float; only the export thresholds it.
  4. export split/<name>/ full_labelled.ply (every property + ``subject_p``), subject.ply
     (p >= 0.5 + bias), background.ply, report.json, and a manifest.json naming the rig, so
     hs render / hs views take the layers like any archive.

Checks: masks_cover_training_views (share of training views with a mask), labels_bimodal (under
5 % of splats still ambiguous after refining; needs_human otherwise).

Not in STAGES: it reads the model and writes only split/<name>/, so it neither blocks nor stales
anything. Requires train, or an explicit --ply.
"""
import json
import os
import sys
import time

import numpy as np

from .. import events
from .. import splatweights as sw
from ..project import md5_file, now_iso

STAGE = "split"
AMBIG_LO, AMBIG_HI = 0.2, 0.8
BIMODAL_MAX = 0.05
COVER_MIN = 0.9


def add_parser(sub):
    p = sub.add_parser("split", help="label a trained model's splats subject/background from the masks (post hoc)")
    p.add_argument("--ply", default=None, help="default: the train stage's final export")
    p.add_argument("--masks", default=None,
                   help="vision | region | DIR. vision / region: train/dataset/masks, checked to have been made "
                        "by hs masks --method vision / geometry; DIR: a tree <eye>/<image>.png. "
                        "Default: train/dataset/masks as it is")
    p.add_argument("--exclude", default=None,
                   help="views to leave out: L/cap064,R/cap064 as for hs train, or @holdout (solve/holdout.json). "
                        "Default: what the model was trained without")
    p.add_argument("--name", default=None, help="split/<name>/ (default: the archive's name, else the ply's stem)")
    p.add_argument("--cell", type=int, default=8, help="weight grid cell in px")
    p.add_argument("--bias", type=float, default=0.0, help="subject when p >= 0.5 + bias")
    p.add_argument("--refine", choices=("knn", "none"), default="knn")
    p.add_argument("--k", type=int, default=16, help="neighbours for the refine")
    p.add_argument("--iters", type=int, default=4, help="diffusion rounds for the refine")
    p.add_argument("--colour-scale", type=float, default=0.2,
                   help="refine: a DC colour difference this large counts as one splat spacing")
    p.add_argument("--max-cells-per-splat", type=int, default=sw.MAX_CELLS_PER_SPLAT)
    p.add_argument("--jobs", type=int, default=None, help="views composited in parallel (default: up to 4)")
    return p


# --------------------------------------------------------------------------- shared with prune --score
def resolve_ply(pj, ply_arg):
    """--ply (absolute, project-relative or cwd-relative), else the train stage's final export."""
    from .prune import final_export
    if ply_arg:
        ply = ply_arg if os.path.isabs(ply_arg) else (
            pj.path(ply_arg) if os.path.exists(pj.path(ply_arg)) else os.path.abspath(ply_arg))
    else:
        ply = final_export(pj)
    if not ply or not os.path.exists(ply):
        raise events.StageError(f"no .ply to read{': ' + ply_arg if ply_arg else ''}",
                                hint="hs train first, or --ply PATH")
    return os.path.abspath(ply)


def default_name(pj, ply):
    d = os.path.dirname(os.path.abspath(ply))
    if os.path.dirname(d) == pj.path("archive"):
        return os.path.basename(d)
    return os.path.splitext(os.path.basename(ply))[0]


def check_name(name):
    if not name or name != os.path.basename(name) or name.startswith("."):
        raise events.StageError(f"bad --name: {name!r}")
    return name


def run_views(stage, sp, G, views, cell, max_cells, per_view, jobs=None):
    """Composite each view (in a small thread pool: numpy's sort and ufuncs release the GIL) and
    hand the result to per_view(v, vw) in the calling thread, in order. -> seconds."""
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    sp.cov_mm                                             # build once, before the threads share it
    jobs = jobs or min(4, os.cpu_count() or 1)

    def one(v):
        return v, sw.view_weights(sp, G["K"][v], G["R"][v], G["t"][v], G["wh"][v], cell=cell, max_cells=max_cells)

    done = 0
    if jobs <= 1:
        it = map(one, views)
        for v, vw in it:
            per_view(v, vw)
            done += 1
            events.progress(stage, done, len(views), step="weights")
    else:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            # a bounded window keeps at most ~2x jobs views' triples in memory at once
            pending, queue = [], list(views)
            while queue or pending:
                while queue and len(pending) < 2 * jobs:
                    pending.append(ex.submit(one, queue.pop(0)))
                v, vw = pending.pop(0).result()
                per_view(v, vw)
                done += 1
                events.progress(stage, done, len(views), step="weights")
    events.progress(stage, len(views), len(views), step="weights", force=True)
    return time.time() - t0


# --------------------------------------------------------------------------- masks
def resolve_masks(pj, spec):
    """-> (masks_root, source label)."""
    mdir = os.path.join(pj.dataset_dir, "masks")
    method = ((pj.m["stages"].get("masks") or {}).get("metrics") or {}).get("method")
    if spec in (None, "", "vision", "region"):
        if not os.path.isdir(mdir):
            raise events.StageError("no train/dataset/masks", hint=f"hs masks --project {pj.root}")
        want = {"vision": "vision", "region": "geometry"}.get(spec)
        if want and method and method != want:
            raise events.StageError(f"train/dataset/masks was made by hs masks --method {method}, not {want}",
                                    hint=f"hs masks --method {want}, or --masks {'region' if method == 'geometry' else 'vision'}")
        return mdir, f"train/dataset/masks ({method or 'method not recorded'})"
    d = spec if os.path.isabs(spec) else pj.path(spec)
    if not os.path.isdir(d):
        d = os.path.abspath(spec)
    if not os.path.isdir(d):
        raise events.StageError(f"--masks {spec}: no such folder", hint="vision, region, or a folder of <eye>/<image>.png")
    return d, pj.rel(d) if d.startswith(pj.root) else d


def load_mask_cells(root, name, wh, cell):
    """The mask's mean over each cell (>=128 is subject), or None when the view has no mask."""
    import cv2
    eye, stem = sw.view_key(name).split("/", 1)
    p = os.path.join(root, eye, stem + ".png")
    if not os.path.exists(p):
        return None
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    W, H = int(wh[0]), int(wh[1])
    if m.shape != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    return sw.cell_mean((m >= 128).astype(np.float32), cell)


# --------------------------------------------------------------------------- the maths
def refine_knn(p, xyz, rgb, k=16, iters=4, colour_scale=0.2):
    """Label diffusion over a KNN graph in (xyz, colour): the ambiguous splats (0.2 < p < 0.8, and
    unseen ones, which start at 0.5) take the mean p of their k nearest neighbours, `iters` times.
    Confident splats never move. -> (refined p, number of splats refined)."""
    import cv2
    p = p.copy()
    unseen = np.isnan(p)
    amb = unseen | ((p > AMBIG_LO) & (p < AMBIG_HI))
    if not amb.any() or len(p) < 2:
        p[unseen] = 0.0
        return p, 0
    lo, hi = np.percentile(xyz, [5, 95], axis=0)
    vol = float(np.prod(np.maximum(hi - lo, 1e-9)))
    spacing = max((vol / max(0.9 * len(p), 1)) ** (1.0 / 3.0), 1e-9)   # mean inter-splat spacing
    feat = np.hstack([xyz / spacing, rgb / max(colour_scale, 1e-6)]).astype(np.float32)
    kk = int(min(k + 1, len(p)))
    index = cv2.flann_Index(feat, dict(algorithm=1, trees=4))           # randomised kd-trees
    q = np.flatnonzero(amb)
    nbr = np.empty((len(q), kk), np.int32)
    step = 200000
    for i in range(0, len(q), step):
        ind, _d = index.knnSearch(feat[q[i:i + step]], kk, params=dict(checks=64))
        nbr[i:i + step] = ind
    # drop each point itself (usually column 0; not guaranteed with an approximate search)
    self_hit = nbr == q[:, None]
    nbr = np.where(self_hit, -1, nbr)
    valid = nbr >= 0
    p[unseen] = 0.5
    for _ in range(max(0, iters)):
        vals = np.where(valid, p[np.maximum(nbr, 0)], 0.0)
        p[q] = vals.sum(1) / np.maximum(valid.sum(1), 1)
    return p, int(len(q))


def histogram(p, bins=10):
    h, e = np.histogram(p[np.isfinite(p)], bins=bins, range=(0.0, 1.0))
    return {"edges": [round(float(x), 3) for x in e], "counts": [int(x) for x in h]}


# --------------------------------------------------------------------------- the stage
def run(a, pj):
    pj.require(STAGE, satisfied=("train",) if a.ply else ())
    ply = resolve_ply(pj, a.ply)
    name = check_name((a.name or default_name(pj, ply)).strip())
    G = sw.load_rig(pj.rig_npz)
    mroot, msrc = resolve_masks(pj, a.masks)
    exclude, ex_src = sw.resolve_exclude(pj, ply, a.exclude, G["names"])
    views = sw.training_views(G["names"], exclude)
    if not views:
        raise events.StageError("every view is excluded; nothing to label from")

    pj.acquire(STAGE)
    st = pj.stage(STAGE)
    st.update({"status": "running", "started": now_iso(), "finished": None, "argv": list(sys.argv),
               "metrics": {}, "checks": [], "artifacts": [], "error": None})
    pj.save()
    t0 = time.time()

    events.start(STAGE, "read")
    sp = sw.Splats(ply)
    pj.metric(STAGE, "ply", pj.rel(ply) if ply.startswith(pj.root) else ply)
    pj.metric(STAGE, "splats", sp.n)
    pj.metric(STAGE, "live_splats", int(sp.live.sum()))
    pj.metric(STAGE, "mask_source", msrc)
    pj.metric(STAGE, "excluded_views", sorted(exclude))
    pj.metric(STAGE, "exclude_source", ex_src)

    masks = {}
    for v in views:
        mc = load_mask_cells(mroot, G["names"][v], G["wh"][v], a.cell)
        if mc is not None:
            masks[v] = mc
    used = [v for v in views if v in masks]
    cover = len(used) / len(views)
    pj.metric(STAGE, "training_views", len(views))
    pj.metric(STAGE, "views_used", len(used))
    pj.check(STAGE, "masks_cover_training_views", cover >= COVER_MIN,
             value=f"{len(used)} of {len(views)} training views have a mask ({100 * cover:.0f}%)")
    if not used:
        raise events.StageError(f"none of the {len(views)} training views has a mask in {msrc}",
                                hint=f"hs masks --project {pj.root}, or --masks DIR")

    events.start(STAGE, "weights")
    num = np.zeros(sp.n)
    den = np.zeros(sp.n)
    clipped = [0]

    def per_view(v, vw):
        m = masks[v].ravel()[vw.cell]
        num[:] += np.bincount(vw.splat, weights=vw.w * m, minlength=sp.n)
        den[:] += np.bincount(vw.splat, weights=vw.w, minlength=sp.n)
        clipped[0] += vw.clipped

    secs_w = run_views(STAGE, sp, G, used, a.cell, a.max_cells_per_splat, per_view, a.jobs)
    p0 = np.full(sp.n, np.nan)
    seen = den > 1e-9
    p0[seen] = num[seen] / den[seen]
    amb_before = int(((p0 > AMBIG_LO) & (p0 < AMBIG_HI)).sum())
    unseen = int((~seen).sum())
    pj.metric(STAGE, "weights_s", round(secs_w, 1))
    pj.metric(STAGE, "clipped_footprints", clipped[0])
    pj.metric(STAGE, "unseen_splats", unseen)
    pj.metric(STAGE, "ambiguous_before_refine", amb_before)

    events.start(STAGE, "refine")
    if a.refine == "knn":
        p, n_ref = refine_knn(p0, sp.xyz_mm / 1000.0, sp.rgb, a.k, a.iters, a.colour_scale)
    else:
        p, n_ref = p0.copy(), 0
        p[np.isnan(p)] = 0.0
    amb_after = int(((p > AMBIG_LO) & (p < AMBIG_HI)).sum())
    pj.metric(STAGE, "refined_splats", n_ref)
    pj.metric(STAGE, "ambiguous_after_refine", amb_after)
    share = amb_after / max(sp.n, 1)
    pj.check(STAGE, "labels_bimodal", share < BIMODAL_MAX, needs_human=share >= BIMODAL_MAX,
             value=f"{100 * share:.2f}% of splats have {AMBIG_LO} < p < {AMBIG_HI} after refine "
                   f"({100 * amb_before / max(sp.n, 1):.2f}% before)")

    events.start(STAGE, "export")
    subj = p >= 0.5 + a.bias
    out = pj.path("split", name)
    os.makedirs(out, exist_ok=True)
    for f in ("full_labelled.ply", "subject.ply", "background.ply", "report.json", "manifest.json"):
        if os.path.exists(os.path.join(out, f)):
            os.remove(os.path.join(out, f))
    paths = {"full_labelled": os.path.join(out, "full_labelled.ply"),
             "subject": os.path.join(out, "subject.ply"), "background": os.path.join(out, "background.ply")}
    sw.write_ply(paths["full_labelled"], sp, extra=("subject_p", p))
    sw.write_ply(paths["subject"], sp, keep=subj)
    sw.write_ply(paths["background"], sp, keep=~subj)
    pj.metric(STAGE, "subject_splats", int(subj.sum()))
    pj.metric(STAGE, "background_splats", int((~subj).sum()))
    op = sp.opacity
    pj.metric(STAGE, "subject_opacity_share", round(float(op[subj].sum() / max(op.sum(), 1e-12)), 4))

    secs = time.time() - t0
    rep = {
        "note": "hs split: FlashSplat closed-form uplift of the 2D masks onto the model's splats",
        "ply": pj.rel(ply) if ply.startswith(pj.root) else ply, "ply_md5": md5_file(ply),
        "rig_npz_md5": md5_file(pj.rig_npz), "name": name,
        "counts": {"splats": sp.n, "live": int(sp.live.sum()), "subject": int(subj.sum()),
                   "background": int((~subj).sum()), "unseen": unseen},
        "ambiguous": {"before_refine": amb_before, "after_refine": amb_after, "refined": n_ref,
                      "band": [AMBIG_LO, AMBIG_HI]},
        "p_histogram_before_refine": histogram(p0), "p_histogram": histogram(p),
        "views_used": [G["names"][v] for v in used],
        "training_views_without_mask": [G["names"][v] for v in views if v not in masks],
        "excluded_views": sorted(exclude), "exclude_source": ex_src, "mask_source": msrc,
        "params": {"cell": a.cell, "bias": a.bias, "refine": a.refine, "k": a.k, "iters": a.iters,
                   "colour_scale": a.colour_scale, "max_cells_per_splat": a.max_cells_per_splat},
        "clipped_footprints": clipped[0], "seconds": round(secs, 1), "weights_seconds": round(secs_w, 1),
    }
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1)
    # beside the layers, the frame they live in: hs render's lineage guard reads this like an archive's
    json.dump({"note": "hs split layers of one model", "name": name, "split_of": rep["ply"],
               "split_of_md5": rep["ply_md5"], "rig_npz_md5": rep["rig_npz_md5"], "made": now_iso(),
               "layers": {"subject": "subject.ply", "background": "background.ply"}},
              open(os.path.join(out, "manifest.json"), "w"), indent=1)
    for k_, pth in paths.items():
        pj.artifact(STAGE, pth, "ply")
    pj.artifact(STAGE, os.path.join(out, "report.json"), "report")
    pj.metric(STAGE, "seconds", round(secs, 1))
    pj.metric(STAGE, "output", pj.rel(out))
    st["status"] = "done"
    st["finished"] = now_iso()
    pj.save()
    pj.release()
