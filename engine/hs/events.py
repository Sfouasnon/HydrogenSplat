"""The one JSON-lines emitter every stage uses (docs/hydrogensplat-strategy-v1.md §2).

One JSON object per line on stdout, flushed immediately. Child-process output never goes
to stdout; it is teed verbatim into ``<project>/logs/<stage>.log`` (see runner.py) and, with
``--verbose``, surfaced as ``{"ev":"log"}`` events so the terminal user can watch it.

Event shapes (all carry ``"stage"``)::

    {"ev":"start","stage":"solve","step":"sfm"}
    {"ev":"progress","stage":"train","done":23970,"total":40000,"rate":9.98,"eta_s":1605,"detail":"139455 splats"}
    {"ev":"estimate","stage":"solve","captures":422,"matchers":{...},...}   (hs solve --estimate; timing.py)
    {"ev":"metric","stage":"solve","name":"mean_reproj_px","value":1.394}
    {"ev":"artifact","stage":"select","path":"select/contact.jpg","kind":"image"}
    {"ev":"check","stage":"solve","name":"all_frames_registered","ok":true,"value":"65/65"}
    {"ev":"check","stage":"move","name":"aim_in_frame","ok":true,"value":"(959,552) in cap021_L at 230 mm","needs_human":true}
    {"ev":"done","stage":"solve","exit":0}
    {"ev":"error","stage":"train","message":"...","hint":"..."}
    {"ev":"log","stage":"solve","line":"..."}          (only with --verbose)

Consumers must ignore event kinds and keys they do not know.

Every ``progress`` with a ``total`` carries an ``eta_s`` once it can: stages that do not
compute one get the mean rate since the step's first progress call (``auto_eta``), reset
by each ``start`` of the stage.

Every project stage also records its own stream to ``<project>/logs/<stage>.events.jsonl``
(``open_file_log``), with a ``run`` event first carrying the argv, the wall clock and a
monotonically increasing id, and a ``t`` on every line (seconds since that run started) --
the shape ``hs replay`` reads. So any run can be played back into the app afterwards, which
is how a chart that looks wrong gets answered from the run's own events instead of from
reasoning about what the app might have received. The text log beside it is the child's raw
output; this is the engine's own.
"""
import json
import os
import sys
import time

_VERBOSE = False
_SINK = None            # optional callable(dict) that also receives every event (selftest uses it)
_FILE = None            # open file handle for <project>/logs/<stage>.events.jsonl
_FILE_T0 = None
_PROGRESS_MIN_DT = 0.25  # seconds between progress events for the same stage/step
_last_progress = {}
_eta_first = {}          # (stage, step) -> [t, done, last done]: the sample auto-ETA measures from
_eta_start = {}          # stage -> time of its last `start`, while no step of it has been sampled
AUTO_ETA_MIN_S = 2.0     # no auto-ETA until this long after the first sample


def set_verbose(flag):
    global _VERBOSE
    _VERBOSE = bool(flag)


def set_sink(fn):
    """Register a function that receives every emitted event dict (in addition to stdout)."""
    global _SINK
    _SINK = fn


def open_file_log(path, argv=None):
    """Append this run's events to `path`. Never fatal: a log that cannot be written must not
    take the stage down with it."""
    global _FILE, _FILE_T0
    close_file_log()
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        _FILE = open(path, "a", encoding="utf-8")
    except OSError:
        _FILE, _FILE_T0 = None, None
        return None
    _FILE_T0 = time.monotonic()
    run_id = int(time.time())
    _emit({"ev": "run", "stage": "hs", "run": run_id,
           "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "argv": list(argv or sys.argv)})
    return path


def close_file_log():
    global _FILE, _FILE_T0
    if _FILE is not None:
        try:
            _FILE.close()
        except OSError:
            pass
    _FILE, _FILE_T0 = None, None


def _emit(obj):
    line = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=_default)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()
    if _FILE is not None:
        # t is what paces `hs replay`; it is added only to the file so stdout stays byte-identical
        rec = dict(obj)
        rec["t"] = round(time.monotonic() - _FILE_T0, 3)
        try:
            _FILE.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":"), default=_default) + "\n")
            _FILE.flush()
        except (OSError, ValueError):
            pass
    if _SINK is not None:
        _SINK(obj)


def _default(o):
    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    return str(o)


def start(stage, step=None, **extra):
    reset_eta(stage)
    ev = {"ev": "start", "stage": stage}
    if step is not None:
        ev["step"] = step
    ev.update(extra)
    _emit(ev)


def reset_eta(stage=None, now=None):
    """Forget auto-ETA samples for one stage (every ``start`` does this, and remembers when it
    started) or, with no stage, for all."""
    for k in [k for k in _eta_first if stage is None or k[0] == stage]:
        del _eta_first[k]
    if stage is None:
        _eta_start.clear()
    else:
        _eta_start[stage] = time.monotonic() if now is None else now


def auto_eta(key, done, total, now):
    """Seconds left from the mean rate since the first (t, done) sample of this (stage, step).
    None until AUTO_ETA_MIN_S have passed and done has moved; 0 once done reaches total. A
    done that goes backwards (a second pass over the same step) restarts the measurement.

    Most loops report ``i + 1`` after item i, so their first sample already has work behind
    it. For the first step sampled after a ``start`` of the stage, that work is anchored at the
    start: (t_start, 0). Later steps under the same start are measured from their own first
    sample (solve's matching must not inherit the time features took)."""
    try:
        done, total = float(done), float(total)
    except (TypeError, ValueError):
        return None
    first = _eta_first.get(key)
    if first is None or done < first[2]:
        t_start = _eta_start.pop(key[0], None)
        if first is None and t_start is not None and done > 0 and t_start <= now:
            _eta_first[key] = first = [t_start, 0.0, done]
        else:
            _eta_first[key] = [now, done, done]
            return 0.0 if done >= total else None
    _eta_start.pop(key[0], None)
    first[2] = done
    if done >= total:
        return 0.0
    t0, d0, _ = first
    el = now - t0
    if el < AUTO_ETA_MIN_S or done <= d0:
        return None
    return (total - done) * el / (done - d0)


def progress(stage, done, total=None, rate=None, eta_s=None, detail=None, step=None, force=False, rest_s=None):
    """Rate-limited: at most one progress event per stage/step per 0.25 s unless force.

    ETA for free: when the caller gives a ``total`` but no ``eta_s``, ``eta_s`` is the mean
    rate since the first progress call of this (stage, step) — every call is sampled, even
    one the rate limit drops — applied to what is left (``auto_eta``). A caller that knows
    better (train's own rate, solve's mapping from the cost model) passes ``eta_s`` itself.
    ``rest_s`` is what the stage's LATER steps are expected to take (solve: the phases after
    this one, from the cost model); with it the event also carries ``stage_eta_s`` =
    ``eta_s + rest_s``, the whole stage's remaining time, which is what a person waiting for
    the stage wants — the per-step ETA alone read "49 min" while the solve had hours to go."""
    key = (stage, step)
    now = time.monotonic()
    if eta_s is None and total is not None:
        eta_s = auto_eta(key, done, total, now)
    if not force and now - _last_progress.get(key, 0.0) < _PROGRESS_MIN_DT:
        return
    _last_progress[key] = now
    ev = {"ev": "progress", "stage": stage, "done": done}
    if step is not None:
        ev["step"] = step
    if total is not None:
        ev["total"] = total
    if rate is not None:
        ev["rate"] = round(float(rate), 3)
    if eta_s is not None:
        ev["eta_s"] = int(round(eta_s))
        if rest_s is not None:
            ev["stage_eta_s"] = int(round(eta_s + max(float(rest_s), 0.0)))
    if detail is not None:
        ev["detail"] = detail
    _emit(ev)


def metric(stage, name, value, **extra):
    ev = {"ev": "metric", "stage": stage, "name": name, "value": value}
    ev.update(extra)
    _emit(ev)


def artifact(stage, path, kind):
    _emit({"ev": "artifact", "stage": stage, "path": path, "kind": kind})


def check(stage, name, ok, value=None, needs_human=False, **extra):
    ev = {"ev": "check", "stage": stage, "name": name, "ok": bool(ok)}
    if value is not None:
        ev["value"] = value
    if needs_human:
        ev["needs_human"] = True
    ev.update(extra)
    _emit(ev)


def done(stage, exit_code=0, **extra):
    ev = {"ev": "done", "stage": stage, "exit": int(exit_code)}
    ev.update(extra)
    _emit(ev)


def error(stage, message, hint=None, **extra):
    ev = {"ev": "error", "stage": stage, "message": str(message)}
    if hint:
        ev["hint"] = hint
    ev.update(extra)
    _emit(ev)


def log(stage, line):
    if _VERBOSE:
        _emit({"ev": "log", "stage": stage, "line": line})


class StageError(Exception):
    """Raised by a stage to abort with a message and an optional hint; cli.py turns it into
    an ``error`` event, a failed manifest entry and a non-zero exit."""

    def __init__(self, message, hint=None):
        super().__init__(message)
        self.hint = hint
