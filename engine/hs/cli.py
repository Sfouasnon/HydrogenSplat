"""hs — the HydrogenSplat engine CLI.

    hs <stage> --project DIR [stage options]
    hs tools | hs calib ... | hs selftest [--clip CLIP]

Stages: ingest, select, solve, train, move (alias paths), prune, render. Each wraps one of
the vendored scripts or a Brush binary as a subprocess and speaks the JSON-lines event
contract on stdout (events.py). Exit codes: 0 ok, 1 stage error (an ``error`` event says
why), 2 unexpected exception, 130 interrupted.
"""
import argparse
import os
import sys
import traceback

from . import __version__, events
from .project import Project
from .stages import calibrate, ingest, move, prune, render, select, selftest, solve, tools, train

PROJECT_STAGES = {
    "ingest": ingest, "select": select, "solve": solve, "train": train,
    "move": move, "paths": move, "prune": prune, "render": render,
}
FREE_STAGES = {"tools": tools, "calib": calibrate, "selftest": selftest}


def build_parser():
    ap = argparse.ArgumentParser(prog="hs", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"hs {__version__}")
    ap.add_argument("-p", "--project", default=None, help="project folder (created by ingest)")
    ap.add_argument("-v", "--verbose", action="store_true", help="also emit child output as {\"ev\":\"log\"} events")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="<stage>")
    for mod in (ingest, select, solve, train, move, prune, render, tools, calibrate, selftest):
        mod.add_parser(sub)
    return ap


def main(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)
    events.set_verbose(a.verbose)
    stage_name = a.cmd
    mod = PROJECT_STAGES.get(stage_name) or FREE_STAGES.get(stage_name)
    ev_stage = getattr(mod, "STAGE", stage_name)
    pj = None
    try:
        if stage_name in PROJECT_STAGES:
            if not a.project:
                ap.error(f"hs {stage_name} needs --project DIR")
            pj = Project(a.project, create=(stage_name == "ingest"))
            mod.run(a, pj)
            code = 0
        elif stage_name == "tools":
            pj = Project(a.project) if a.project and os.path.exists(os.path.join(a.project, "manifest.json")) else None
            mod.run(a, pj)
            code = 0
        elif stage_name == "selftest":
            code = 0 if mod.run(a) else 1
        else:
            mod.run(a)
            code = 0
        events.done(ev_stage, code)
        return code
    except events.StageError as e:
        events.error(ev_stage, str(e), hint=e.hint)
        if pj is not None and pj.status(ev_stage) == "running":
            pj.finish(ev_stage, ok=False, error=str(e))
        events.done(ev_stage, 1)
        return 1
    except KeyboardInterrupt:
        events.error(ev_stage, "interrupted")
        if pj is not None and pj.status(ev_stage) == "running":
            pj.finish(ev_stage, ok=False, error="interrupted")
        events.done(ev_stage, 130)
        return 130
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        if pj is not None:
            try:
                os.makedirs(pj.path("logs"), exist_ok=True)
                with open(pj.log_path(ev_stage), "a") as f:
                    f.write("\n### hs traceback\n" + tb)
                if pj.status(ev_stage) == "running":
                    pj.finish(ev_stage, ok=False, error=repr(e))
            except Exception:
                pass
        sys.stderr.write(tb)
        events.error(ev_stage, f"{type(e).__name__}: {e}", hint="traceback on stderr and in logs/<stage>.log")
        events.done(ev_stage, 2)
        return 2


if __name__ == "__main__":
    sys.exit(main())
