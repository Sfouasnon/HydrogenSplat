"""hs — the HydrogenSplat engine CLI.

    hs <stage> --project DIR [stage options]
    hs cameras -p DIR | hs movepreview -p DIR --script F | hs tools | hs calib ... | hs selftest [--clip CLIP] | hs phone | hs replay EVENTS.jsonl

Stages: ingest, select, solve, scale, train, move (alias paths), prune, render, views. Each wraps one of
the vendored scripts or a Brush binary as a subprocess and speaks the JSON-lines event
contract on stdout (events.py). Exit codes: 0 ok, 1 stage error (an ``error`` event says
why), 2 unexpected exception, 130 interrupted.
"""
import argparse
import os
import sys
import traceback

from . import __version__, events, keepawake
from .project import Project
from .stages import archive, calibrate, cameras, exposure, grade, ingest, masks, merge, move, movepreview, phone, prune, render, replay, scale, select, selftest, solve, tools, train, views

PROJECT_STAGES = {
    "ingest": ingest, "select": select, "solve": solve, "scale": scale, "train": train,
    "move": move, "paths": move, "prune": prune, "render": render, "views": views,
    "exposure": exposure, "masks": masks, "merge": merge, "archive": archive, "grade": grade,
}
FREE_STAGES = {"cameras": cameras, "movepreview": movepreview, "tools": tools, "calib": calibrate, "selftest": selftest, "phone": phone, "replay": replay}


# --project and --verbose are accepted on either side of the stage name: `hs -p DIR solve`
# and `hs solve -p DIR` both work. argparse hands everything after the stage name to the
# subparser, so the flags have to exist on both; SUPPRESS on the subparser's copy means an
# omitted flag leaves the top-level value in place instead of overwriting it with None.
GLOBAL_ON_SUBPARSER = set(PROJECT_STAGES) | {"tools", "selftest", "cameras", "movepreview"}


def build_parser():
    ap = argparse.ArgumentParser(prog="hs", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"hs {__version__}")
    ap.add_argument("-p", "--project", default=None, help="project folder (created by ingest)")
    ap.add_argument("-v", "--verbose", action="store_true", help="also emit child output as {\"ev\":\"log\"} events")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="<stage>")
    for mod in (ingest, select, solve, scale, exposure, masks, train, archive, merge, move, prune, render, grade, views, cameras, movepreview, tools, calibrate, selftest, phone, replay):
        mod.add_parser(sub)
    seen = set()
    for name, sp in sub.choices.items():
        if id(sp) in seen:      # aliases (paths -> move) share one parser object
            continue
        seen.add(id(sp))
        if name in GLOBAL_ON_SUBPARSER:
            sp.add_argument("-p", "--project", default=argparse.SUPPRESS,
                            help="project folder (created by ingest)")
        sp.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS,
                        help="also emit child output as {\"ev\":\"log\"} events")
    return ap


def main(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)
    events.set_verbose(a.verbose)
    stage_name = a.cmd
    mod = PROJECT_STAGES.get(stage_name) or FREE_STAGES.get(stage_name)
    ev_stage = getattr(mod, "STAGE", stage_name)
    pj = None
    awake = None
    try:
        if stage_name in PROJECT_STAGES or stage_name == "selftest":
            # hold the Mac awake for the whole stage, not just the child (keepawake.py)
            awake = keepawake.hold(ev_stage, off=keepawake.disabled(a)).__enter__()
        if stage_name in PROJECT_STAGES:
            if not a.project:
                ap.error(f"hs {stage_name} needs --project DIR")
            pj = Project(a.project, create=(stage_name == "ingest"))
            # the run's own event stream, replayable with `hs replay` (events.open_file_log)
            events.open_file_log(pj.path("logs", f"{ev_stage}.events.jsonl"))
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
    finally:
        events.close_file_log()
        if awake is not None:
            awake.__exit__(None, None, None)
        # finish() releases the project lock on the normal path; make sure an exception
        # raised before finish (or a stage that never called it) does not leave it behind
        if pj is not None:
            try:
                pj.release()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
