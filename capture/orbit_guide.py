"""orbit_guide — capture assist for the Hydrogen One, run in Termux while Holocam records.

    python orbit_guide.py [--out TRACK.jsonl] [--duration S] [--quiet] [--no-vibrate]
    python orbit_guide.py --replay TRACK.jsonl          # off-phone, no sensors needed

Why this exists
---------------
``hs views`` can tell you, after a 75-minute training run, that the azimuth band at −135°…−90°
registers badly and that the high elevation ring is empty. That is a useless thing to learn
after the shoot, and "film a high elevation band" is not an instruction anyone can follow while
holding a phone. This turns it into something you can act on at the time: it reads the phone's
orientation, works out which cell of the azimuth × elevation grid the camera is looking from,
and says "raise", "left", "hold" until every cell has been held long enough to have clean
frames in it.

It draws nothing. A HUD would fight Holocam for the viewfinder, and while you are orbiting a
subject you are looking at the subject. Speech and vibration do not need the screen — and the
speech lands in Holocam's own audio track, which makes aligning ``--out`` against the clip
afterwards a correlation of cues against cues rather than two unsynchronised clocks.

What it measures, and what it does not
--------------------------------------
Elevation comes from the Gravity sensor and is exact: with the rear camera looking along the
device −Z axis, the camera's elevation above the subject is ``asin(−ĝz)``, assuming you keep
the subject centred in frame.

Azimuth comes from Game Rotation Vector — gyro and accelerometer, magnetometer deliberately
excluded. Magnetic heading is absolute but wrong near steel, and walking a circle round a
subject on a stage floor drags the local field with you, so the error would correlate with
position: the worst possible shape of error for a coverage map. Game Rotation Vector drifts
instead, slowly and smoothly, which over a two-minute orbit is a fraction of a band.

This gives orientation, not position. It knows which way you are pointing; it infers where you
are standing from that, which holds while you orbit a centred subject and breaks the moment you
pan from a fixed spot. It cannot tell you that you drifted closer.

Sensors are read through Termux:API (``pkg install termux-api`` plus the F-Droid APK). Pure
stdlib otherwise — there is no numpy on the phone.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
# in the repo hs/ lives under engine/; deployed to the phone it sits beside this file
sys.path[:0] = [os.path.join(_HERE, "..", "engine"), _HERE]
from hs.bands import (AZIMUTH_BAND, DWELL_S, SLOW_DEG_S, Grid,  # noqa: E402
                      elevation_band, wrap180)

GRAVITY = "gravity"
ROTATION = "game rotation"
SPEAK_GAP_S = 2.5          # never repeat the same instruction faster than this
MIN_RATE_DT = 0.05         # shorter than this and the interval is clock noise, not a measurement
RATE_SMOOTH = 0.5          # weight on the newest turn-rate estimate


# ------------------------------------------------------------------ orientation

def elevation_from_gravity(g):
    """Camera elevation above the subject, degrees, from a device-frame gravity vector."""
    if len(g) < 3:
        return None
    n = math.sqrt(sum(c * c for c in g[:3]))
    if n < 1e-6:
        return None
    return math.degrees(math.asin(max(-1.0, min(1.0, -g[2] / n))))


def yaw_from_quaternion(v):
    """Heading of the rear camera, degrees, from a rotation-vector reading.

    Android rotation vectors carry (x, y, z) and sometimes w; w is recovered from the unit
    norm when absent. The matrix maps device to world, the rear camera looks along device −Z,
    so the world-frame view direction is minus the matrix's third column. Yaw is measured from
    the world reference axis (+Y) towards +X — an arbitrary zero, which is all we need.

    It increases clockwise seen from above, i.e. as you turn to your right, which is what makes
    a positive azimuth error in Grid.cue() mean "go right" rather than the reverse.
    """
    if len(v) < 3:
        return None
    x, y, z = v[0], v[1], v[2]
    w = v[3] if len(v) >= 4 else math.sqrt(max(0.0, 1.0 - x * x - y * y - z * z))
    fx = -2.0 * (x * z + y * w)
    fy = -2.0 * (y * z - x * w)
    return math.degrees(math.atan2(fx, fy))


def orbit_azimuth(yaw_deg):
    """Where you are standing, given where you are looking: the far side of the subject."""
    return wrap180(yaw_deg + 180.0)


# ------------------------------------------------------------------ sensor stream

def sensor_argv(period_ms):
    return ["termux-sensor", "-s", "Gravity,Game Rotation Vector", "-d", str(int(period_ms))]


def pick(sample, needle):
    """The values array of the first sensor whose name contains ``needle`` (case-insensitive)."""
    for name, body in sample.items():
        if needle in name.lower():
            v = body.get("values") if isinstance(body, dict) else body
            if isinstance(v, list) and v:
                return v
    return None


def decode_stream(chunks):
    """Yield JSON objects from an iterable of text chunks.

    termux-sensor writes pretty-printed objects back to back with no separator and no newline
    discipline, so the stream is decoded incrementally rather than parsed a line at a time.
    """
    buf, dec = "", json.JSONDecoder()
    for chunk in chunks:
        buf += chunk
        while True:
            buf = buf.lstrip()
            if not buf:
                break
            try:
                obj, end = dec.raw_decode(buf)
            except ValueError:
                break                      # partial object; wait for more
            buf = buf[end:]
            yield obj


def read_sensors(period_ms):
    """(sample dict, wall clock) pairs from termux-sensor until it is killed."""
    if not shutil.which("termux-sensor"):
        raise SystemExit("termux-sensor not found — pkg install termux-api, and install the "
                         "Termux:API APK from F-Droid (they are two separate halves)")
    p = subprocess.Popen(sensor_argv(period_ms), stdout=subprocess.PIPE, text=True, bufsize=1)
    try:
        # readline, not read(n): a blocking block read hands over several samples at once and
        # they all get the same arrival clock, which collapses dt and makes the measured
        # turn rate meaningless. A line arrives as soon as termux-sensor writes it.
        for obj in decode_stream(iter(p.stdout.readline, "")):
            yield obj, time.time()
    finally:
        p.terminate()


def replay_track(path):
    """(sample-shaped dict, wall clock) pairs rebuilt from a track written by --out."""
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "az" not in r:
                continue                    # header
            yield r, r["t"]


# ------------------------------------------------------------------ voice

class Voice:
    """Speaks an instruction, at most one every SPEAK_GAP_S, and never the same one twice
    running — a guide that repeats itself every sample is one you stop listening to."""

    def __init__(self, enabled=True, vibrate=True):
        self.speak_ok = enabled and bool(shutil.which("termux-tts-speak"))
        self.vibrate_ok = vibrate and bool(shutil.which("termux-vibrate"))
        self.last, self.last_t = None, 0.0

    def _run(self, argv):
        try:
            subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            pass

    def say(self, text, now, force=False):
        if not force and (text == self.last or now - self.last_t < SPEAK_GAP_S):
            return False
        self.last, self.last_t = text, now
        if self.speak_ok:
            self._run(["termux-tts-speak", "-n", "0", "-r", "1.1", text])
        return True

    def tick(self):
        if self.vibrate_ok:
            self._run(["termux-vibrate", "-d", "120"])


# ------------------------------------------------------------------ the loop

CUE_WORDS = {"raise": "raise", "lower": "lower", "left": "left", "right": "right",
             "hold": "hold", "done": "all bands covered"}


def run(samples, grid, voice, out=None, duration=None, quiet=False):
    """Consume (sample, clock) pairs, guide, log. Returns (grid report, run stats).

    The stats exist because "0 of 24 cells covered" on its own does not say whether the
    operator moved too fast, held the phone outside every elevation ring, or the stream stopped
    after two samples — three different problems with the same summary line.
    """
    stats = {"samples": 0, "elapsed_s": 0.0, "rates": [], "out_of_band_s": 0.0, "too_fast_s": 0.0}
    t0 = prev_t = None
    prev_az = prev_el = None
    rate = 0.0
    for sample, clock in samples:
        if isinstance(sample, dict) and "az" in sample:
            az, el = sample["az"], sample["el"]          # replay
        else:
            g, q = pick(sample, GRAVITY), pick(sample, ROTATION)
            if g is None or q is None:
                continue
            el, yaw = elevation_from_gravity(g), yaw_from_quaternion(q)
            if el is None or yaw is None:
                continue
            az = orbit_azimuth(yaw)

        if t0 is None:
            t0 = clock
        t = clock - t0
        dt = 0.0 if prev_t is None else max(0.0, t - prev_t)
        # Two samples arriving within a millisecond of each other measure nothing: the angle
        # moved is real but the interval is clock noise, and dividing by it reports hundreds of
        # degrees a second for a phone sitting still. Wait for a real interval, keeping the old
        # anchor so the next measurement spans it, and smooth what comes out.
        if dt >= MIN_RATE_DT and prev_az is not None:
            instant = math.hypot(wrap180(az - prev_az), el - prev_el) / dt
            rate = RATE_SMOOTH * instant + (1.0 - RATE_SMOOTH) * rate
            prev_az, prev_el = az, el
        elif prev_az is None:
            prev_az, prev_el = az, el
        prev_t = t

        stats["samples"] += 1
        stats["elapsed_s"] = t
        stats["rates"].append(rate)
        if elevation_band(el) is None:
            stats["out_of_band_s"] += dt
        elif rate > grid.slow:
            stats["too_fast_s"] += dt

        done_cell = grid.mark(az, el, dt, rate)
        if done_cell:
            voice.tick()

        if rate > grid.slow:
            spoke = voice.say("slower", clock)
            cue = "fast"
        else:
            cue, target = grid.cue(az, el)
            spoke = voice.say(CUE_WORDS[cue], clock, force=(cue == "done"))

        if out:
            out.write(json.dumps({"t": round(t, 3), "az": round(az, 2), "el": round(el, 2),
                                  "rate": round(rate, 1), "cue": cue,
                                  "cue_spoken": bool(spoke),
                                  "ring": elevation_band(el),
                                  "completed": list(done_cell) if done_cell else None}) + "\n")
        if not quiet:
            n = len(grid.covered())
            sys.stderr.write(f"\r az {az:+7.1f}  el {el:+6.1f}  {rate:5.1f}°/s  "
                             f"{n:2d}/{len(grid.time)} cells  {cue:<6s}")
            sys.stderr.flush()
        if cue == "done" or (duration and t >= duration):
            break
    if not quiet:
        sys.stderr.write("\n")
    return grid.report(), stats


def median(xs):
    s = sorted(xs)
    return 0.0 if not s else (s[len(s) // 2] if len(s) % 2 else (s[len(s) // 2 - 1] + s[len(s) // 2]) / 2)


def summarise(report, dwell, stats=None):
    miss = [r for r in report if not r["covered"]]
    lines = [f"{len(report) - len(miss)}/{len(report)} cells covered "
             f"(a cell needs {dwell:g}s held steady)"]
    if miss:
        by_ring = {}
        for r in miss:
            by_ring.setdefault(r["ring"], []).append(f"{r['azimuth_deg'][0]:+d}")
        for ring, los in by_ring.items():
            lines.append(f"  {ring:<5s} still open at az {', '.join(los)}")
    if stats:
        best = max((r["seconds"] for r in report), default=0.0)
        lines.append(f"  {stats['samples']} samples over {stats['elapsed_s']:.1f}s, "
                     f"median {median(stats['rates']):.1f}°/s, "
                     f"{stats['out_of_band_s']:.1f}s outside every ring, "
                     f"{stats['too_fast_s']:.1f}s too fast, "
                     f"best cell {best:.1f}s")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="write the orientation track here as JSON lines")
    ap.add_argument("--replay", help="replay a track instead of reading sensors")
    ap.add_argument("--duration", type=float, default=None, help="stop after this many seconds")
    ap.add_argument("--period-ms", type=int, default=100, help="sensor period (default 100)")
    ap.add_argument("--band", type=int, default=AZIMUTH_BAND)
    ap.add_argument("--dwell", type=float, default=DWELL_S)
    ap.add_argument("--slow", type=float, default=SLOW_DEG_S, help="°/s above which frames smear")
    ap.add_argument("--quiet", action="store_true", help="no live status line")
    ap.add_argument("--silent", action="store_true", help="no speech")
    ap.add_argument("--no-vibrate", action="store_true")
    a = ap.parse_args(argv)

    grid = Grid(band=a.band, dwell=a.dwell, slow_deg_s=a.slow)
    voice = Voice(enabled=not (a.silent or a.replay), vibrate=not (a.no_vibrate or a.replay))
    samples = replay_track(a.replay) if a.replay else read_sensors(a.period_ms)

    out = open(a.out, "w") if a.out else None
    if out:
        out.write(json.dumps({"started": time.time(), "band": a.band, "dwell": a.dwell,
                              "slow_deg_s": a.slow, "source": a.replay or "termux-sensor"}) + "\n")
    stats = None
    try:
        report, stats = run(samples, grid, voice, out=out, duration=a.duration, quiet=a.quiet)
    except KeyboardInterrupt:
        report = grid.report()
        if not a.quiet:
            sys.stderr.write("\n")
    finally:
        if out:
            out.close()
    print(summarise(report, a.dwell, stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
