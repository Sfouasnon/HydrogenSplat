"""hs replay — play a recorded JSON-lines event log back on stdout, with its timing.

    hs replay tests/events/train_body.jsonl [--speed 20] [--fail-at N]

The app's stub (strategy §2: "one on the app against a stub that replays recorded event
logs"): every stage page can be built and exercised without a phone, a GPU or 85 minutes.
A line may carry ``"t"`` — seconds since the start of the recording — which paces the replay
and is stripped before the event is written. Lines without ``t`` follow the previous one
after ``--gap`` seconds. ``--fail-at N`` stops after N events with an ``error`` event and
exit 1, so failure handling can be exercised too.
"""
import json
import sys
import time

from .. import events

STAGE = "replay"


def add_parser(sub):
    p = sub.add_parser("replay", help="replay a recorded event log (the app's stub stage)")
    p.add_argument("file")
    p.add_argument("--speed", type=float, default=1.0, help="time compression (20 = 20x faster)")
    p.add_argument("--gap", type=float, default=0.05, help="seconds between lines that carry no t")
    p.add_argument("--max-wait", type=float, default=2.0, help="cap on any single pause, after --speed")
    p.add_argument("--fail-at", type=int, default=None)
    p.add_argument("--last", action="store_true",
                   help="only the last run in the file (from its final `run` event): what a project's "
                        "logs/<stage>.events.jsonl holds after several runs")
    return p


def run(a):
    try:
        lines = [l for l in open(a.file, encoding="utf-8").read().splitlines() if l.strip()]
    except OSError as e:
        raise events.StageError(f"cannot read {a.file}: {e}")
    if getattr(a, "last", False):
        starts = [i for i, l in enumerate(lines) if '"ev":"run"' in l.replace(" ", "")]
        if starts:
            lines = lines[starts[-1]:]
    t_prev, n, last_stage = 0.0, 0, STAGE
    for i, line in enumerate(lines, 1):
        try:
            ev = json.loads(line)
        except ValueError:
            raise events.StageError(f"{a.file}:{i} is not JSON")
        t = ev.pop("t", None)
        wait = (t - t_prev) / max(a.speed, 1e-6) if t is not None else a.gap
        if t is not None:
            t_prev = t
        time.sleep(max(0.0, min(wait, a.max_wait)))
        if a.fail_at is not None and n >= a.fail_at:
            raise events.StageError(f"replay stopped after {n} events (--fail-at)", hint="this failure is simulated")
        last_stage = ev.get("stage", last_stage)
        sys.stdout.write(json.dumps(ev, ensure_ascii=False, separators=(",", ":")) + "\n")
        sys.stdout.flush()
        n += 1
