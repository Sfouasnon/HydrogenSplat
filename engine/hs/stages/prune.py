"""hs prune — optional prune of a trained .ply (prune_splats.py, unchanged).

--center is always given explicitly, as the SfM median in metres (spec §6): the script's own
opacity-weighted centre lands on bright background as readily as on the subject. Note the
``--center=`` form — a leading minus is otherwise parsed as a flag.

``--score`` / ``--floaters`` is a second, photometric mode that does not touch prune_splats.py.
From the renderer's forward weights over the training views (hs.splatweights) it scores every
splat:

  importance        sum of w = alpha * T over every (view, cell) -- LightGaussian's global
                    significance without the volume term;
  top_contributor   the splat holds the largest w in at least one cell of one view --
                    Mini-Splatting's guard: the only thing drawing some pixel is not pruned;
  views_seen        training views in which it carries any weight;
  blame             sum(w * e) / sum(w), e = per cell |composited DC colour - photograph|
                    in linear RGB (mean over channels): how wrong the pixels it paints are.

``--score`` alone writes prune/<name>_scores.npz and <name>_report.json and prunes nothing. It is
a report: prune's manifest status and metrics (and render's staleness) are left as they were,
and the run is kept under stages.prune.runs.<name>_score. ``--floaters`` removes the splats that
are low-importance (under the --min-importance-quantile of the splats seen at all) AND top
contributor nowhere AND high-blame (> --max-blame), writing <name>_nofloat.ply and
<name>_floaters_only.ply; it is a prune run like the geometric one (render goes stale, and its
output is what render's lineage guard knows as prune's output_ply).
"""
import json
import os
import re
import sys
import time

import numpy as np

from .. import events, runner
from ..project import md5_file, now_iso

STAGE = "prune"
RE_WROTE = re.compile(r"wrote .*: (\d+) splats \(([\d.]+)% of the input\)")
RE_TOTAL = re.compile(r"^(\d+) splats, subject centre")


def add_parser(sub):
    p = sub.add_parser("prune", help="prune background / invisible / oversized splats (prune_splats.py)")
    p.add_argument("--ply", default=None, help="default: the train stage's final export")
    p.add_argument("--radius", type=float, default=0.3, help="metres (nominal) about the SfM median")
    p.add_argument("--min-opacity", type=float, default=0.05)
    p.add_argument("--max-aniso", type=float, default=50.0)
    p.add_argument("--max-scale", type=float, default=0.2)
    p.add_argument("--center", default=None, help="x,y,z metres; default: SfM median from coverage.json")
    p.add_argument("--report-only", action="store_true")
    sc = p.add_argument_group("photometric score (hs.splatweights; prune_splats.py is not run)")
    sc.add_argument("--score", action="store_true",
                    help="score every splat (importance, top contributor, views seen, blame) over the training "
                         "views; writes prune/<name>_scores.npz + _report.json and prunes nothing")
    sc.add_argument("--floaters", action="store_true",
                    help="prune low-importance, never-top-contributor, high-blame splats: prune/<name>_nofloat.ply "
                         "and _floaters_only.ply (scores are reused from --score when they match the ply)")
    sc.add_argument("--name", default=None, help="score / floaters file stem (default: the ply's stem)")
    sc.add_argument("--exclude", default=None,
                    help="views to leave out (as hs train, or @holdout); default: what the model was trained without")
    sc.add_argument("--cell", type=int, default=8, help="weight grid cell in px")
    sc.add_argument("--max-cells-per-splat", type=int, default=None)
    sc.add_argument("--min-importance-quantile", type=float, default=0.02,
                    help="--floaters: low importance = under this quantile of the splats seen in any view")
    sc.add_argument("--max-blame", type=float, default=0.25,
                    help="--floaters: high blame = over this mean linear-RGB error of the cells a splat paints")
    sc.add_argument("--jobs", type=int, default=None, help="views composited in parallel (default: up to 4)")
    return p


def final_export(pj):
    p = pj.stage("train").get("metrics", {}).get("final_export")
    if p and os.path.exists(pj.path(p)):
        return pj.path(p)
    d = pj.exports_dir
    if os.path.isdir(d):
        plys = sorted(f for f in os.listdir(d) if re.fullmatch(r"export_\d+\.ply", f))
        if plys:
            return os.path.join(d, plys[-1])
    return None


def run(a, pj):
    if getattr(a, "score", False) or getattr(a, "floaters", False):
        return run_score(a, pj)
    pj.require(STAGE)
    ply = os.path.abspath(a.ply) if a.ply else final_export(pj)
    if not ply or not os.path.exists(ply):
        raise events.StageError("no trained .ply to prune", hint="hs train first, or --ply PATH")
    if a.center:
        center = a.center
    else:
        sub = pj.stage("solve").get("metrics", {}).get("subject_mm")
        if not sub:
            import json
            cov = json.load(open(pj.path("solve", "coverage.json")))
            sub = cov["subject_mm"]
        center = ",".join(f"{v / 1000.0:.6f}" for v in sub)
    pj.begin(STAGE, argv=sys.argv, clean=not a.report_only)
    events.start(STAGE, "prune")
    base = os.path.splitext(os.path.basename(ply))[0]
    out = pj.path("prune", f"{base}_pruned_r{str(a.radius).replace('.', '')}.ply")  # r0.3 -> _r03
    argv = runner.python_argv("prune_splats.py", ply) + ([] if a.report_only else [out]) + [
        f"--center={center}", "--radius", str(a.radius), "--min-opacity", str(a.min_opacity),
        "--max-aniso", str(a.max_aniso), "--max-scale", str(a.max_scale)]
    if a.report_only:
        argv.append("--report-only")
    st = {}

    def on_line(line):
        m = RE_TOTAL.search(line)
        if m:
            st["total"] = int(m.group(1))
        m = RE_WROTE.search(line)
        if m:
            st["kept"], st["pct"] = int(m.group(1)), float(m.group(2))

    runner.run(argv, STAGE, log_path=pj.log_path(STAGE), on_line=on_line)
    pj.metric(STAGE, "center_m", [float(v) for v in center.split(",")])
    pj.metric(STAGE, "input_ply", pj.rel(ply) if ply.startswith(pj.root) else ply)
    pj.metric(STAGE, "input_ply_md5", md5_file(ply))   # render ties the pruned cloud back to train's export
    if "total" in st:
        pj.metric(STAGE, "splats_in", st["total"])
    if "kept" in st:
        pj.metric(STAGE, "splats_out", st["kept"])
        pj.metric(STAGE, "kept_fraction", st["pct"] / 100.0)
    if not a.report_only:
        if not os.path.exists(out):
            raise events.StageError("prune wrote no output", hint="see logs/prune.log")
        pj.metric(STAGE, "output_ply", pj.rel(out))
        pj.artifact(STAGE, out, "ply")
    pj.finish(STAGE, ok=True)


# --------------------------------------------------------------------------- --score / --floaters
BACKGROUND = 0.0     # what shows through where the splats run out (brush-path-render draws black)


class _Recorder:
    """pj.metric/check/artifact for a run that owns the prune stage (--floaters); for a report-only
    --score run the same events are emitted but the record goes to stages.prune.runs instead of
    overwriting the last prune's metrics, which render's lineage guard reads."""

    def __init__(self, pj, owns_stage):
        self.pj, self.owns = pj, owns_stage
        self.rec = {"metrics": {}, "checks": [], "artifacts": []}

    def metric(self, name, value):
        if self.owns:
            self.pj.metric(STAGE, name, value)
        else:
            self.rec["metrics"][name] = value
            events.metric(STAGE, name, value)

    def check(self, name, ok, value=None, needs_human=False):
        if self.owns:
            return self.pj.check(STAGE, name, ok, value=value, needs_human=needs_human)
        r = {"name": name, "ok": bool(ok)}
        if value is not None:
            r["value"] = value
        if needs_human:
            r["needs_human"] = True
        self.rec["checks"].append(r)
        events.check(STAGE, name, ok, value, needs_human)
        return bool(ok)

    def artifact(self, path, kind):
        if self.owns:
            self.pj.artifact(STAGE, path, kind)
        else:
            self.rec["artifacts"].append({"path": self.pj.rel(path), "kind": kind})
            events.artifact(STAGE, self.pj.rel(path), kind)


def load_photo_cells(dataset, name, wh, cell):
    """The undistorted training photograph's per-cell mean, linear RGB, or None."""
    import cv2
    from .. import splatweights as sw
    p = sw.image_path(dataset, name)
    img = cv2.imread(p, cv2.IMREAD_COLOR) if p else None
    if img is None:
        return None
    W, H = int(wh[0]), int(wh[1])
    if img.shape[:2] != (H, W):
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    return sw.srgb_to_linear(sw.cell_mean(img[:, :, ::-1].astype(np.float32) / 255.0, cell))


def score_splats(pj, sp, G, views, cell, max_cells, jobs=None):
    """-> dict of per-splat arrays + the names of the views that had a photograph."""
    from .. import splatweights as sw
    from .split import run_views
    n = sp.n
    importance = np.zeros(n)
    blame_num = np.zeros(n)
    views_seen = np.zeros(n, np.int32)
    top_cells = np.zeros(n, np.int64)
    photos, err_cells = {}, []
    for v in views:
        ph = load_photo_cells(pj.dataset_dir, G["names"][v], G["wh"][v], cell)
        if ph is not None:
            photos[v] = ph
    used = [v for v in views if v in photos]
    clipped = [0]

    def per_view(v, vw):
        comp = sw.srgb_to_linear(np.clip(vw.rgb + vw.T[..., None] * BACKGROUND, 0.0, 1.0))
        e = np.abs(comp - photos[v]).mean(axis=-1).ravel()
        err_cells.append(float(e.mean()))
        importance[:] += np.bincount(vw.splat, weights=vw.w, minlength=n)
        blame_num[:] += np.bincount(vw.splat, weights=vw.w * e[vw.cell], minlength=n)
        views_seen[:] += np.bincount(vw.splat, minlength=n) > 0
        top = sw.cell_max_mask(vw)
        top_cells[:] += np.bincount(vw.splat[top], minlength=n)
        clipped[0] += vw.clipped

    secs = run_views(STAGE, sp, G, used, cell, max_cells, per_view, jobs)
    blame = np.where(importance > 0, blame_num / np.maximum(importance, 1e-12), 0.0)
    return {"importance": importance, "blame": blame, "views_seen": views_seen,
            "top_cells": top_cells, "top_contributor": top_cells > 0, "clipped": clipped[0],
            "views": [G["names"][v] for v in used],
            "views_without_photo": [G["names"][v] for v in views if v not in photos],
            "cell_error_mean": float(np.mean(err_cells)) if err_cells else None, "seconds": secs}


def flag_floaters(sc, live, q=0.02, max_blame=0.25):
    """The prune rule: low importance AND top contributor nowhere AND high blame. Only splats that
    some training view saw can be judged; the quantile is taken over those. -> (mask, threshold)."""
    seen = live & (sc["views_seen"] > 0)
    if not seen.any():
        return np.zeros(len(live), bool), 0.0
    thr = float(np.quantile(sc["importance"][seen], q))
    flag = seen & (sc["importance"] <= thr) & ~sc["top_contributor"] & (sc["blame"] > max_blame)
    return flag, thr


def _hist(x, bins, rng=None, log=False):
    x = np.asarray(x, float)
    if log:
        x = np.log10(np.maximum(x, 1e-6))
    h, e = np.histogram(x, bins=bins, range=rng)
    return {"edges": [round(float(v), 4) for v in e], "counts": [int(v) for v in h],
            **({"log10": True} if log else {})}


def run_score(a, pj):
    from .. import splatweights as sw
    from .split import check_name, resolve_ply
    pj.require(STAGE, satisfied=("train",) if a.ply else ())
    ply = None
    if not a.ply and not a.score and a.name:
        # --floaters on its own picks up the model its scores were computed from
        old = pj.path("prune", f"{a.name}_scores.npz")
        if os.path.exists(old):
            stored = str(np.load(old, allow_pickle=True)["ply"])
            ply = stored if os.path.isabs(stored) else pj.path(stored)
    ply = resolve_ply(pj, a.ply) if not ply else ply
    name = check_name((a.name or os.path.splitext(os.path.basename(ply))[0]).strip())
    G = sw.load_rig(pj.rig_npz)
    exclude, ex_src = sw.resolve_exclude(pj, ply, a.exclude, G["names"])
    views = sw.training_views(G["names"], exclude)
    if not views:
        raise events.StageError("every view is excluded; nothing to score against")
    scores_path = pj.path("prune", f"{name}_scores.npz")
    report_path = pj.path("prune", f"{name}_report.json")
    digest = md5_file(ply)

    owns = bool(a.floaters)
    if owns:
        pj.begin(STAGE, argv=sys.argv, clean=False)     # keep earlier prune outputs and the scores
    else:
        pj.acquire(STAGE)
    os.makedirs(pj.path("prune"), exist_ok=True)
    rec = _Recorder(pj, owns)
    t0 = time.time()
    events.start(STAGE, "read")
    sp = sw.Splats(ply)
    rec.metric("input_ply", pj.rel(ply) if ply.startswith(pj.root) else ply)
    rec.metric("input_ply_md5", digest)
    rec.metric("splats_in", sp.n)

    reuse = None
    if not a.score and os.path.exists(scores_path):
        z = np.load(scores_path, allow_pickle=True)
        if str(z["ply_md5"]) == digest and int(z["cell"]) == a.cell and len(z["importance"]) == sp.n:
            reuse = {k: z[k] for k in ("importance", "blame", "views_seen", "top_cells", "top_contributor")}
            reuse["views"] = [str(x) for x in z["views"]]
        else:
            events.log(STAGE, f"[hs] {pj.rel(scores_path)} is for another model or cell size; rescoring")
    if reuse is not None:
        sc = reuse
        rec.metric("scores", f"reused {pj.rel(scores_path)}")
    else:
        events.start(STAGE, "weights")
        sc = score_splats(pj, sp, G, views, a.cell, a.max_cells_per_splat or sw.MAX_CELLS_PER_SPLAT, a.jobs)
        if not sc["views"]:
            raise events.StageError(f"none of the {len(views)} training views has a photograph in train/dataset/images")
        rec.metric("views_used", len(sc["views"]))
        rec.metric("weights_s", round(sc["seconds"], 1))
        rec.metric("clipped_footprints", sc["clipped"])
        rec.metric("excluded_views", sorted(exclude))
        rec.metric("exclude_source", ex_src)
        rec.check("photos_cover_training_views", not sc["views_without_photo"],
                  value=f"{len(sc['views'])} of {len(views)} training views have a photograph")
        np.savez_compressed(scores_path, importance=sc["importance"].astype(np.float32),
                            blame=sc["blame"].astype(np.float32), views_seen=sc["views_seen"],
                            top_cells=sc["top_cells"].astype(np.int32), top_contributor=sc["top_contributor"],
                            opacity=sp.opacity.astype(np.float32), live=sp.live,
                            ply=pj.rel(ply) if ply.startswith(pj.root) else ply, ply_md5=digest,
                            cell=a.cell, views=np.array(sc["views"]))
        seen = sp.live & (sc["views_seen"] > 0)
        would, thr = flag_floaters(sc, sp.live, a.min_importance_quantile, a.max_blame)
        rep = {
            "note": "hs prune --score: per-splat importance / top contributor / views seen / blame "
                    "from the renderer's forward weights over the training views",
            "ply": pj.rel(ply) if ply.startswith(pj.root) else ply, "ply_md5": digest, "cell": a.cell,
            "views": sc["views"], "excluded_views": sorted(exclude), "exclude_source": ex_src,
            "counts": {"splats": sp.n, "live": int(sp.live.sum()), "seen": int(seen.sum()),
                       "never_seen_live": int((sp.live & ~seen).sum()),
                       "top_contributors": int(sc["top_contributor"].sum())},
            "cell_error_mean_linear": sc["cell_error_mean"],
            "importance_quantiles": {str(q): float(np.quantile(sc["importance"][seen], q)) if seen.any() else None
                                     for q in (0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.9)},
            "histograms": {
                "importance_log10": _hist(sc["importance"][seen], 24, (-4, 4), log=True),
                "blame": _hist(sc["blame"][seen], 20, (0.0, 1.0)),
                "views_seen": _hist(sc["views_seen"][sp.live], min(20, max(1, len(sc["views"]))),
                                    (0, max(1, len(sc["views"])))),
            },
            "would_flag": {"min_importance_quantile": a.min_importance_quantile, "max_blame": a.max_blame,
                           "importance_threshold": thr, "splats": int(would.sum()),
                           "opacity_mass_share": round(float(sp.opacity[would].sum() / max(sp.opacity.sum(), 1e-12)), 5)},
            "clipped_footprints": sc["clipped"], "seconds": round(time.time() - t0, 1),
        }
        json.dump(rep, open(report_path, "w"), indent=1)
        rec.metric("seen_splats", int(seen.sum()))
        rec.metric("top_contributors", int(sc["top_contributor"].sum()))
        rec.metric("would_flag_floaters", int(would.sum()))
        rec.artifact(scores_path, "scores")
        rec.artifact(report_path, "report")

    if a.floaters:
        events.start(STAGE, "floaters")
        flag, thr = flag_floaters(sc, sp.live, a.min_importance_quantile, a.max_blame)
        out = pj.path("prune", f"{name}_nofloat.ply")
        only = pj.path("prune", f"{name}_floaters_only.ply")
        sw.write_ply(out, sp, keep=~flag)
        sw.write_ply(only, sp, keep=flag)
        op = sp.opacity
        left = float(op[~flag].sum() / max(op.sum(), 1e-12))
        rec.metric("importance_threshold", thr)
        rec.metric("floaters", int(flag.sum()))
        rec.metric("splats_out", int((~flag).sum()))
        rec.metric("kept_fraction", round(float((~flag).mean()), 5))
        rec.metric("opacity_mass_kept", round(left, 5))
        rec.metric("output_ply", pj.rel(out))           # what render's lineage guard ties to input_ply_md5
        rec.check("floater_prune_is_small", left >= 0.9,
                  value=f"{int(flag.sum()):,} splats, {100 * (1 - left):.2f}% of the opacity mass removed",
                  needs_human=left < 0.9)
        json.dump({"ply": pj.rel(ply) if ply.startswith(pj.root) else ply, "ply_md5": digest,
                   "min_importance_quantile": a.min_importance_quantile, "importance_threshold": thr,
                   "max_blame": a.max_blame, "floaters": int(flag.sum()), "splats_in": sp.n,
                   "opacity_mass_kept": left, "made": now_iso()},
                  open(pj.path("prune", f"{name}_floaters.json"), "w"), indent=1)
        rec.artifact(out, "ply")
        rec.artifact(only, "ply")

    rec.metric("seconds", round(time.time() - t0, 1))
    if owns:
        pj.record_run(STAGE, f"{name}_floaters", ply=rec.pj.rel(ply) if ply.startswith(pj.root) else ply)
        pj.finish(STAGE, ok=True)
    else:
        st = pj.stage(STAGE)
        r = dict(rec.rec)
        r.update({"finished": now_iso(), "argv": list(sys.argv),
                  "ply": pj.rel(ply) if ply.startswith(pj.root) else ply})
        st.setdefault("runs", {})[f"{name}_score"] = r
        pj.save()
        pj.release()
