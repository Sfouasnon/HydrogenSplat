"""Freeze the regression suite: one JSON + one Markdown table of what each project measures today.

Read-only over Projects/<p>/{solve,views,archive,scale}. Run it before any change to the solve or the
trainer (rig lock, COLMAP mapper, depth loss ...) and diff the next snapshot against it.

    python3 engine/tools/regression_snapshot.py --out Projects/_regression/baseline-YYYY-MM-DD
"""
import argparse, glob, hashlib, json, os, statistics as st, sys, datetime

SUITE = ["2026-09-20_GreetingCard", "2026-09-13_coins", "2026-09-22_CirclesSculpture",
         "2026-09-16_Stormtrooper", "2026-09-28_Stormtrooper_iPhone"]
VIEW_KEYS = ["psnr_db", "displaced_fraction", "displaced_fraction_detrended", "bulk_shift_px",
             "displacement_median_px", "retained_edge_energy"]


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def med(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(st.median(xs), 4) if xs else None


def views_summary(rep):
    vs = rep.get("views") if isinstance(rep, dict) else None
    if not isinstance(vs, list) or not vs:
        return None
    out = {"n": len(vs), "ply": rep.get("ply")}
    for k in VIEW_KEYS:
        out["median_" + k] = med([v.get(k) for v in vs])
    out["max_bulk_shift_px"] = max((v.get("bulk_shift_px") or 0) for v in vs)
    worst = max(vs, key=lambda v: v.get("bulk_shift_px") or 0)
    out["worst_bulk_shift_view"] = worst.get("view")
    return out


def project(root, name):
    P = os.path.join(root, name)
    snap = {"project": name, "exists": os.path.isdir(P)}
    if not snap["exists"]:
        return snap
    m = load(os.path.join(P, "manifest.json")) or {}
    snap["stages_done"] = sorted(k for k, v in (m.get("stages") or {}).items()
                                 if isinstance(v, dict) and v.get("status") == "done")
    sfm = load(os.path.join(P, "solve", "sfm_report.json"))
    if sfm:
        snap["sfm"] = {k: v for k, v in sfm.items() if not isinstance(v, (list, dict))}
    for s in ("scale_report", "lidar_report"):
        r = load(os.path.join(P, "scale", s + ".json"))
        if r:
            snap[s] = {k: v for k, v in r.items() if not isinstance(v, (list, dict))}
    snap["views"] = {}
    for f in sorted(glob.glob(os.path.join(P, "views", "*_report.json"))):
        s = views_summary(load(f))
        if s:
            snap["views"][os.path.basename(f)[:-12]] = s
    snap["archives"] = {}
    for f in sorted(glob.glob(os.path.join(P, "archive", "*", "manifest.json"))):
        a = load(f) or {}
        ply = a.get("ply") or {}
        fp = a.get("train_dataset_fingerprint") or {}
        snap["archives"][os.path.basename(os.path.dirname(f))] = {
            "ply_md5": ply.get("md5"), "archived": a.get("archived"),
            "excluded_views": fp.get("excluded_views"), "masks_used": fp.get("masks_used"),
            "pycolmap": (a.get("tools") or {}).get("pycolmap")}
    return snap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--projects", default=os.path.join(os.path.dirname(__file__), "..", "..", "Projects"))
    ap.add_argument("--out", required=True, help="path stem; writes <stem>.json and <stem>.md")
    ap.add_argument("--suite", default=",".join(SUITE))
    a = ap.parse_args()
    root = os.path.abspath(a.projects)
    snaps = [project(root, n) for n in a.suite.split(",")]
    doc = {"created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "note": "regression baseline -- numbers as measured by earlier runs, not re-measured here",
           "projects": snaps}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(doc, open(a.out + ".json", "w"), indent=1)
    L = [f"# Regression baseline ({doc['created']})", "", doc["note"], ""]
    for s in snaps:
        L.append(f"## {s['project']}")
        if not s["exists"]:
            L += ["missing", ""]; continue
        if "sfm" in s:
            L.append("solve: " + ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in s["sfm"].items()))
        if s["views"]:
            L += ["", "| report | n | PSNR dB | displaced | detrended | bulk shift px | worst view (px) |",
                  "|---|---:|---:|---:|---:|---:|---|"]
            for k, v in s["views"].items():
                f = lambda x: "—" if x is None else x
                L.append(f"| {k} | {v['n']} | {f(v['median_psnr_db'])} | {f(v['median_displaced_fraction'])} | "
                         f"{f(v['median_displaced_fraction_detrended'])} | {f(v['median_bulk_shift_px'])} | "
                         f"{v['worst_bulk_shift_view']} ({v['max_bulk_shift_px']}) |")
        if s["archives"]:
            L += ["", "archives: " + ", ".join(f"{k} ({(v['ply_md5'] or '?')[:8]})" for k, v in s["archives"].items())]
        L.append("")
    open(a.out + ".md", "w").write("\n".join(L))
    print(a.out + ".json", a.out + ".md")


if __name__ == "__main__":
    sys.exit(main())
