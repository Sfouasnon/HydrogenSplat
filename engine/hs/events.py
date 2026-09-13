"""The one JSON-lines emitter every stage uses (docs/hydrogensplat-strategy-v1.md §2).

One JSON object per line on stdout, flushed immediately. Child-process output never goes
to stdout; it is teed verbatim into ``<project>/logs/<stage>.log`` (see runner.py) and, with
``--verbose``, surfaced as ``{"ev":"log"}`` events so the terminal user can watch it.

Event shapes (all carry ``"stage"``)::

    {"ev":"start","stage":"solve","step":"sfm"}
    {"ev":"progress","stage":"train","done":23970,"total":40000,"rate":9.98,"eta_s":1605,"detail":"139455 splats"}
    {"ev":"metric","stage":"solve","name":"mean_reproj_px","value":1.394}
    {"ev":"artifact","stage":"select","path":"select/contact.jpg","kind":"image"}
    {"ev":"check","stage":"solve","name":"all_frames_registered","ok":true,"value":"65/65"}
    {"ev":"check","stage":"move","name":"aim_in_frame","ok":true,"value":"(959,552) in cap021_L at 230 mm","needs_human":true}
    {"ev":"done","stage":"solve","exit":0}
    {"ev":"error","stage":"train","message":"...","hint":"..."}
    {"ev":"log","stage":"solve","line":"..."}          (only with --verbose)

Consumers must ignore event kinds and keys they do not know.
"""
import json
import sys
import time

_VERBOSE = False
_SINK = None            # optional callable(dict) that also receives every event (selftest uses it)
_PROGRESS_MIN_DT = 0.25  # seconds between progress events for the same stage/step
_last_progress = {}


def set_verbose(flag):
    global _VERBOSE
    _VERBOSE = bool(flag)


def set_sink(fn):
    """Register a function that receives every emitted event dict (in addition to stdout)."""
    global _SINK
    _SINK = fn


def _emit(obj):
    line = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=_default)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()
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
    ev = {"ev": "start", "stage": stage}
    if step is not None:
        ev["step"] = step
    ev.update(extra)
    _emit(ev)


def progress(stage, done, total=None, rate=None, eta_s=None, detail=None, step=None, force=False):
    """Rate-limited: at most one progress event per stage/step per 0.25 s unless force."""
    key = (stage, step)
    now = time.monotonic()
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
