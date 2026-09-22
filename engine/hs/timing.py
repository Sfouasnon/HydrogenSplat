"""How long a solve will take — the timing store, the cost model and the estimate (pure: no
pycolmap, no project; ``hs solve --estimate`` and the tests import it directly).

Cost model, per phase, in seconds::

    features = a · images
    matching = b · pairs
    mapping  = c · images ** p        (incremental mapping is superlinear; p = 1.5 until
                                       ≥ 3 runs spanning ≥ 1.5× in images let it be fitted)
    export   = d · images

Every completed ``hs solve`` appends one run to the store (``append_solve_run``): images,
pairs, matcher and the four phase times the vendored scripts print as ``timing: <phase> N s``
(export is the export child's wall time). ``fit`` refits the coefficients by least squares
through the origin, one phase at a time, from the runs that measured that phase; a phase no
run has measured keeps its default. The store lives at

    $HS_TIMING_FILE                                              (tests point it at a temp file)
    ~/Library/Application Support/HydrogenSplat/timing.json      (macOS)
    ~/.hydrogensplat/timing.json                                 (anywhere else)

and is never fatal: a missing or unreadable file is an empty store.

Defaults (``calibration: "defaults"`` until the store has a run). Matching is the only one
measured: CirclesSculpture 2026-09-22, 844 images, 88–159 s per 50×50 exhaustive block on
Apple Silicon. pycolmap 4.2.0 runs all 17×17 = 289 blocks of that match (not the 153 of
i ≤ j — see pairs.exhaustive_block_progress), 355,746 / 289 ≈ 1,231 pairs each, so 0.071–
0.129 s per pair; 0.10 is the midpoint. Features 0.6 s/image, mapping 0.5·images^1.5 s and
export 0.3 s/image are conservative guesses, to be replaced by the first real runs.

The ``estimate`` event (one line on stdout from ``hs solve --estimate``; the app parses it)::

    {"ev": "estimate", "stage": "solve",
     "captures": 422, "images_per_capture": 2,
     "auto": "sequential",                      # what --matcher auto would run
     "calibration": "defaults",                 # or "measured on this Mac (3 runs)"
     "runs": 0,
     "matchers": {
       "exhaustive": {"images": 844, "pairs": 355746,
                      "seconds": {"features": 506, "matching": 35575, "mapping": 12260,
                                  "export": 253, "total": 48594}},
       "sequential": {"images": 844, "pairs": 30566,
                      "seconds": {"features": 506, "matching": 3057, "mapping": 12260,
                                  "export": 253, "total": 16076},
                      "overlap": 15, "loop_stride": 8}}}

(422 stereo captures, default coefficients.) ``total`` is the sum of the unrounded phases,
so it can differ from the sum of the rounded ones by a second or two.

Seconds are integers. Consumers must ignore keys they do not know (events.py).
"""
import json
import math
import os
import sys
import time

from . import pairs as _pairs

PHASES = ("features", "matching", "mapping", "export")
DEFAULTS = {
    "features": 0.6,       # s per image
    "matching": 0.10,      # s per pair
    "mapping_c": 0.5,      # s · images ** mapping_p
    "mapping_p": 1.5,
    "export": 0.3,         # s per image
}
EXPONENT_RANGE = (1.0, 2.5)
MAX_RUNS = 500


# ------------------------------------------------------------------ store

def store_path():
    env = os.environ.get("HS_TIMING_FILE")
    if env:
        return os.path.expanduser(env)
    if sys.platform == "darwin":
        return os.path.expanduser("~/Library/Application Support/HydrogenSplat/timing.json")
    return os.path.expanduser("~/.hydrogensplat/timing.json")


def load(path=None):
    """The store as {"version": 1, "solve": [run, ...]}; empty when absent or unreadable."""
    path = path or store_path()
    try:
        with open(path) as f:
            d = json.load(f)
        if not isinstance(d, dict):
            raise ValueError
    except (OSError, ValueError):
        d = {}
    d.setdefault("version", 1)
    if not isinstance(d.get("solve"), list):
        d["solve"] = []
    return d


def append_solve_run(run, path=None):
    """Append one solve run (a dict with images, pairs and any of features_s … export_s) and
    write the store atomically. Returns the path, or None if it could not be written."""
    path = path or store_path()
    d = load(path)
    rec = dict(run)
    rec.setdefault("at", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    d["solve"].append(rec)
    d["solve"] = d["solve"][-MAX_RUNS:]
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(d, f, indent=1)
        os.replace(tmp, path)
    except OSError:
        return None
    return path


# ------------------------------------------------------------------ model

def _num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


def _ls_origin(xs, ys):
    """y = k·x through the origin, least squares."""
    sxx = sum(x * x for x in xs)
    return sum(x * y for x, y in zip(xs, ys)) / sxx if sxx > 0 else None


def fit(runs):
    """Coefficients from the stored runs. Returns a dict with DEFAULTS' keys plus ``runs``
    (how many runs contributed anything) and ``measured`` (the phases that came from data)."""
    m = dict(DEFAULTS)
    used, measured = set(), []
    for phase, xkey in (("features", "images"), ("matching", "pairs"), ("export", "images")):
        pts = [(_num(r.get(xkey)), _num(r.get(phase + "_s")), i) for i, r in enumerate(runs)]
        pts = [(x, y, i) for x, y, i in pts if x and y]
        if pts:
            k = _ls_origin([x for x, _, _ in pts], [y for _, y, _ in pts])
            if k:
                m[phase] = k
                measured.append(phase)
                used.update(i for _, _, i in pts)
    pts = [(_num(r.get("images")), _num(r.get("mapping_s")), i) for i, r in enumerate(runs)]
    pts = [(x, y, i) for x, y, i in pts if x and y]
    if pts:
        p = DEFAULTS["mapping_p"]
        xs = [x for x, _, _ in pts]
        if len(pts) >= 3 and max(xs) >= 1.5 * min(xs):
            lx = [math.log(x) for x in xs]
            ly = [math.log(y) for _, y, _ in pts]
            mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
            sxx = sum((a - mx) ** 2 for a in lx)
            if sxx > 0:
                p = sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sxx
                p = min(max(p, EXPONENT_RANGE[0]), EXPONENT_RANGE[1])
        c = _ls_origin([x ** p for x in xs], [y for _, y, _ in pts])
        if c:
            m["mapping_c"], m["mapping_p"] = c, p
            measured.append("mapping")
            used.update(i for _, _, i in pts)
    m["runs"] = len(used)
    m["measured"] = [ph for ph in PHASES if ph in measured]
    return m


def model(path=None):
    """fit(load()) — the coefficients this machine has earned so far."""
    return fit(load(path)["solve"])


def calibration_label(m):
    n = m.get("runs", 0)
    return f"measured on this Mac ({n} run{'s' if n != 1 else ''})" if n else "defaults"


def phase_seconds(m, images, pairs):
    s = {
        "features": m["features"] * images,
        "matching": m["matching"] * pairs,
        "mapping": m["mapping_c"] * images ** m["mapping_p"] if images > 0 else 0.0,
        "export": m["export"] * images,
    }
    s["total"] = sum(s[p] for p in PHASES)
    return s


def estimate(n_captures, images_per_capture=2, m=None, overlap=_pairs.DEFAULT_OVERLAP,
             loop_stride=_pairs.DEFAULT_LOOP_STRIDE):
    """The ``estimate`` event (without "ev"/"stage") for both matchers. See the module doc."""
    m = m or fit([])
    n_img = int(n_captures) * int(images_per_capture)
    out = {"captures": int(n_captures), "images_per_capture": int(images_per_capture),
           "auto": _pairs.resolve_matcher("auto", n_captures),
           "calibration": calibration_label(m), "runs": int(m.get("runs", 0)), "matchers": {}}
    for matcher in _pairs.MATCHERS:
        n_pairs = _pairs.pair_count(matcher, n_captures, images_per_capture, overlap, loop_stride)
        sec = phase_seconds(m, n_img, n_pairs)
        rec = {"images": n_img, "pairs": n_pairs,
               "seconds": {k: int(round(v)) for k, v in sec.items()}}
        if matcher == "sequential":
            rec.update(overlap=int(overlap), loop_stride=int(loop_stride))
        out["matchers"][matcher] = rec
    return out


def mapping_eta(elapsed, registered, total_images, predicted_s=None, exponent=DEFAULTS["mapping_p"]):
    """Seconds of incremental mapping left. Registration slows as the model grows (each new
    image costs more bundle adjustment), so a linear rate says too little too early. Early on
    the cost model's prediction minus the elapsed time; once a quarter of the images (at least
    10) are in, the run's own curve: elapsed · ((total / registered) ** p − 1)."""
    if not total_images or total_images <= 0 or elapsed is None:
        return None
    if registered >= total_images:
        return 0.0
    enough = registered >= max(10, 0.25 * total_images)
    if registered > 0 and enough and elapsed > 0:
        return elapsed * ((total_images / registered) ** exponent - 1.0)
    if predicted_s:
        left = predicted_s - elapsed
        if left > 0:
            return left
    if registered > 0 and elapsed > 0:
        return elapsed * ((total_images / registered) ** exponent - 1.0)
    return None


def fmt_duration(s):
    """'≈ 35 min' style, for the CLI's stderr summary."""
    s = int(round(s))
    if s < 90:
        return f"{s} s"
    if s < 3600:
        return f"{round(s / 60)} min"
    h, mnt = divmod(round(s / 60), 60)
    return f"{h} h {mnt:02d} min"
