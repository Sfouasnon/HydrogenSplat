"""hs selftest — the golden test on the rig6 clip (strategy §5, fixtures/README.md).

Runs ingest → select → solve → move (sweep, auto window) → move (boom, auto keys) into a
project folder and asserts:

  60–70 frames selected                       (reference 65/66)
  100% registered                             (rig6 65/65)
  mean reprojection 1.2–1.6 px                (reference 1.394)
  coverage min elevation < −5°, max > +20°    (two low passes and the high pass)
  SfM median projects within 60 px of (959, 552) in cap021_L  (reference aim check)
  both moves: aim in frame, hull ≤ 25 mm

Train is not part of the automated test (55 min); a manual golden train should land at
170k ± 20k splats — `hs train` checks that band itself. Exit 0 when every assertion holds.
"""
import json
import os
import sys
import time
from argparse import Namespace

import numpy as np

from .. import calib, events
from ..project import Project, md5_file
from . import ingest, move, select, solve

STAGE = "selftest"
DEFAULT_CLIP = os.path.join(calib.REPO_ROOT, "fixtures", "VID_20260912_151351_2x1.h4v")
DEFAULT_PROJECT = os.path.join(calib.REPO_ROOT, "fixtures", "selftest")
REF_VIEW, REF_UV, REF_TOL_PX = "cap021_L", (959.0, 552.0), 60.0


def add_parser(sub):
    p = sub.add_parser("selftest", help="golden test: select → solve → paths on the rig6 clip, assert the fixture numbers")
    p.add_argument("--clip", default=DEFAULT_CLIP)
    # --project comes from the shared flag cli.py adds to every stage; default below
    p.add_argument("--resume", action="store_true", help="skip stages the project manifest already shows done")
    p.add_argument("--fresh", action="store_true", help="delete the project folder first")
    return p


def run(a, _pj=None):
    t0 = time.monotonic()
    results = []
    collected = []
    events.set_sink(collected.append)

    def assert_(name, ok, value):
        results.append((name, bool(ok), value))
        events.check(STAGE, name, ok, value=value)
        return bool(ok)

    clip = os.path.abspath(os.path.expanduser(a.clip))
    if not os.path.exists(clip):
        raise events.StageError(f"fixture clip missing: {clip}",
                                hint="cp ~/Desktop/Apps/REDHydrogenOne/capture_video/VID_20260912_151351_2x1.h4v fixtures/")
    events.start(STAGE, "fixture")
    md5_path = os.path.join(calib.REPO_ROOT, "fixtures", "rig6.md5")
    if os.path.exists(md5_path):
        want = open(md5_path).read().split()[0]
        got = md5_file(clip)
        assert_("fixture_md5", got == want, f"{got}" + ("" if got == want else f" != {want}"))

    root = os.path.abspath(os.path.expanduser(getattr(a, "project", None) or DEFAULT_PROJECT))
    if a.fresh and os.path.isdir(root):
        import shutil
        shutil.rmtree(root)
    pj = Project(root, create=True)
    events.metric(STAGE, "project", root)

    def done(stage):
        return a.resume and pj.status(stage) == "done"

    # ---- stages
    if not done("ingest"):
        ingest.run(Namespace(clip=clip, link=True, profile=None, ffprobe=os.environ.get("HS_FFPROBE", "ffprobe")), pj)
    if not done("select"):
        select.run(Namespace(residual=1.5, min_gap=6, max_gap=90, search=4, max_clip=0.02, work_width=480,
                             start=0, end=-1, dry_run=False), pj)
    if not done("solve"):
        solve.run(Namespace(profile=None, peak_threshold=0.0025, features=16384, masks=None, no_float_rig=False,
                            reuse_matches=False, max_reproj=1.8), pj)
    moves_done = a.resume and pj.status("move") == "done" and \
        all(os.path.exists(pj.path("move", f"{n}.json")) and n in pj.stage("move").get("runs", {})
            for n in ("sweep", "boom"))
    if not moves_done:
        base = dict(name=None, frames=None, fps=30.0, captures=None, keys=None, hold=0.5, ease=1.0,
                    aim_point=None, span=0.8, smooth=3)
        move.run(Namespace(preset="sweep", **base), pj)
        move.run(Namespace(preset="boom", **base), pj)
    pj = Project(root)  # reload the manifest the stages wrote

    # ---- assertions
    events.start(STAGE, "assert")
    sm, vm = pj.stage("select").get("metrics", {}), pj.stage("solve").get("metrics", {})
    n_sel = int(sm.get("frames_selected", 0))
    assert_("frames_selected_60_70", 60 <= n_sel <= 70, f"{n_sel} (reference 65/66)")
    n_reg, n_cap = int(vm.get("num_frames", 0)), int(vm.get("captures", 0))
    assert_("all_registered", n_reg == n_cap and n_cap > 0, f"{n_reg}/{n_cap}")
    mr = float(vm.get("mean_reproj_px", 0))
    assert_("mean_reproj_1.2_1.6", 1.2 <= mr <= 1.6, f"{mr:.3f} px (reference 1.394)")
    el = vm.get("elevation_range_deg", [0, 0])
    assert_("coverage_low_pass", el[0] < -5.0, f"min elevation {el[0]:.1f}° (want < −5°)")
    assert_("coverage_high_pass", el[1] > 20.0, f"max elevation {el[1]:.1f}° (want > +20°)")
    dev = vm.get("lr_separation_max_dev_mm")
    if dev is not None:
        assert_("rig_constraint_held", dev < 0.01, f"L–R separation within {dev:.4f} mm of the profile baseline")

    # reference aim check: project the SfM median into cap021_L exactly as the builders do
    rig = pj.rig_npz
    G = np.load(rig, allow_pickle=True)
    names = [str(x) for x in G["names"]]
    if REF_VIEW in names:
        v = names.index(REF_VIEW)
        obj = np.median(G["pts"].astype(float), axis=0)
        Xc = G["R"].astype(float)[v] @ obj + G["t"].astype(float)[v]
        uv = G["K"].astype(float)[v] @ Xc
        uv = uv[:2] / uv[2]
        d = float(np.hypot(uv[0] - REF_UV[0], uv[1] - REF_UV[1]))
        assert_("median_aim_near_reference", Xc[2] > 0 and d <= REF_TOL_PX,
                f"({uv[0]:.0f}, {uv[1]:.0f}) in {REF_VIEW} at {Xc[2]:.0f} mm — {d:.0f} px from {tuple(int(x) for x in REF_UV)} (want ≤ {REF_TOL_PX:.0f})")
    else:
        assert_("median_aim_near_reference", False, f"{REF_VIEW} not in the solve")

    runs = pj.stage("move").get("runs", {})
    for mv in ("sweep", "boom"):
        checks = {c["name"]: c for c in runs.get(mv, {}).get("checks", [])}
        for cname in ("aim_in_frame", "hull_within_25mm"):
            c = checks.get(cname)
            assert_(f"{mv}.{cname}", bool(c and c["ok"]), c.get("value") if c else "check not recorded")

    # ---- summary (human-readable, stderr) + verdict
    elapsed = time.monotonic() - t0
    n_ok = sum(1 for _, ok, _ in results if ok)
    lines = [f"hs selftest — {n_ok}/{len(results)} assertions passed in {elapsed / 60:.1f} min  ({root})"]
    for name, ok, value in results:
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name:34s} {value}")
    summary = "\n".join(lines)
    sys.stderr.write(summary + "\n")
    json.dump({"passed": n_ok == len(results), "results": [{"name": n, "ok": o, "value": v} for n, o, v in results],
               "elapsed_s": round(elapsed, 1), "project": root},
              open(pj.path("selftest.json"), "w"), indent=1)
    events.artifact(STAGE, pj.rel(pj.path("selftest.json")), "json")
    events.metric(STAGE, "passed", n_ok)
    events.metric(STAGE, "total", len(results))
    events.metric(STAGE, "elapsed_s", round(elapsed, 1))
    return n_ok == len(results)
