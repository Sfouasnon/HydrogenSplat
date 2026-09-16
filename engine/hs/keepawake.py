"""Keep the Mac awake while a stage runs, and notice when it slept anyway.

Every project stage (and selftest) runs inside ``hold()`` (cli.py). On macOS that starts

    caffeinate -d -i -m -s -w <hs pid>

so the display, idle sleep, disk idle and -- on AC power -- system sleep are all held for as
long as the ``hs`` process lives; ``-w`` means a killed ``hs`` never leaves caffeinate behind.
``HS_CAFFEINATE_FLAGS="-i -m -s"`` changes the flags (e.g. to let the display sleep),
``HS_NO_CAFFEINATE=1`` or ``hs train --no-caffeinate`` turns it off.

What caffeinate cannot do: closing the lid (without an external display), Apple menu > Sleep,
or a low-battery shutdown still sleep the machine. The 2026-09-15 head train had ``caffeinate
-i`` around Brush alone and still spent ~10 h doing ~35 min of work: ~30 s of training every
~16 min, the signature of a sleeping Mac's maintenance wakes. So the engine also measures
sleep: ``SleepWatch`` compares a clock that runs through sleep with one that stops during
it, and every gap is reported (runner.py emits a ``sleep_detected`` metric live, train
persists ``slept_s`` / ``wall_s`` and the ``no_sleep_during_run`` check).
"""
import os
import platform
import re
import shlex
import shutil
import subprocess
import time

from . import events

DEFAULT_FLAGS = "-d -i -m -s"
SLEEP_GAP_S = 20.0   # through-sleep clock minus awake clock over one poll that counts as a sleep


def _clock_ids():
    """(runs through sleep, stops during sleep) clock ids, or None where unknown."""
    if platform.system() == "Darwin" and hasattr(time, "CLOCK_UPTIME_RAW"):
        # Apple: CLOCK_MONOTONIC_RAW keeps counting while asleep, CLOCK_UPTIME_RAW does not
        return time.CLOCK_MONOTONIC_RAW, time.CLOCK_UPTIME_RAW
    if hasattr(time, "CLOCK_BOOTTIME"):
        # Linux: BOOTTIME includes suspend, MONOTONIC does not
        return time.CLOCK_BOOTTIME, time.CLOCK_MONOTONIC
    return None


_CLOCKS = _clock_ids()


def clocks():
    """(through_sleep_s, awake_s). Tests replace this function."""
    if _CLOCKS is None:
        t = time.monotonic()
        return t, t
    return time.clock_gettime(_CLOCKS[0]), time.clock_gettime(_CLOCKS[1])


class SleepWatch:
    """Call poll() now and then; it returns the seconds slept since the previous poll
    (0.0 when none) and keeps the running totals."""

    def __init__(self):
        self.through0, self.awake0 = clocks()
        self._through, self._awake = self.through0, self.awake0
        self.slept_s = 0.0
        self.sleeps = []          # [(local time the gap was noticed, seconds)]

    def poll(self):
        through, awake = clocks()
        gap = (through - self._through) - (awake - self._awake)
        self._through, self._awake = through, awake
        if gap < SLEEP_GAP_S:
            return 0.0
        self.slept_s += gap
        self.sleeps.append((time.strftime("%H:%M:%S"), round(gap, 1)))
        return gap

    @property
    def wall_s(self):
        return clocks()[0] - self.through0

    def summary(self):
        return {"slept_s": round(self.slept_s, 1), "sleeps": len(self.sleeps),
                "wall_s": round(self.wall_s, 1)}


def disabled(a=None):
    if os.environ.get("HS_NO_CAFFEINATE", "").lower() in ("1", "true", "yes"):
        return True
    return bool(getattr(a, "no_caffeinate", False))


def _out(argv):
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=3).stdout
    except Exception:
        return ""


def power_state():
    """{"power": "AC"|"battery"|None, "lid_closed": bool|None} -- macOS only, best effort."""
    batt = _out(["pmset", "-g", "batt"])
    power = "AC" if "AC Power" in batt else "battery" if "Battery Power" in batt else None
    m = re.search(r'"AppleClamshellState"\s*=\s*(Yes|No)',
                  _out(["ioreg", "-r", "-k", "AppleClamshellState", "-d", "1"]))
    return {"power": power, "lid_closed": (m.group(1) == "Yes") if m else None}


class hold:
    """Context manager: caffeinate for the life of this process (macOS only)."""

    def __init__(self, stage, off=False):
        self.stage = stage
        self.off = off
        self.proc = None

    def __enter__(self):
        if self.off or platform.system() != "Darwin" or not shutil.which("caffeinate"):
            return self
        flags = shlex.split(os.environ.get("HS_CAFFEINATE_FLAGS", DEFAULT_FLAGS))
        argv = ["caffeinate"] + flags + ["-w", str(os.getpid())]
        try:
            self.proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL,
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as e:
            events.metric(self.stage, "keep_awake", {"argv": argv, "error": str(e)})
            return self
        ps = power_state()
        notes = []
        if ps["power"] == "battery":
            notes.append("on battery: -s is ignored and macOS may still sleep -- plug in")
        if ps["lid_closed"]:
            notes.append("lid is closed: without an external display the Mac will sleep")
        notes.append("closing the lid or Apple menu > Sleep still sleeps the Mac")
        events.metric(self.stage, "keep_awake", {"argv": argv, **ps, "note": "; ".join(notes)})
        return self

    def __exit__(self, *exc):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        return False
