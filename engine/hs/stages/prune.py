"""hs prune — optional prune of a trained .ply (prune_splats.py, unchanged).

--center is always given explicitly, as the SfM median in metres (spec §6): the script's own
opacity-weighted centre lands on bright background as readily as on the subject. Note the
``--center=`` form — a leading minus is otherwise parsed as a flag.
"""
import os
import re
import sys

from .. import events, runner

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
