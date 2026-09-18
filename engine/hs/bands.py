"""Capture bands — one definition of "a band", shared by the scorer and the phone.

``hs views`` reports per-azimuth-band medians and flags bands that register badly; the phone
guide (``capture/orbit_guide.py``) tells you, while you are shooting, which bands you have not
covered yet. Those two must agree about what a band *is*, or the guide sends you to fill a gap
the scorer does not believe in. Hence this module: pure stdlib, no numpy, so the phone can
import the same file the engine does.

Frames
------
The engine's azimuth zero (coverage.py) is the mean horizontal direction from the subject to
the cameras — recoverable only *after* the solve. The phone has no subject and no such frame;
it has device orientation. So the two disagree about where zero is, by a constant. That is
fine and deliberate: everything here is about *gaps and spans*, which are the same in any
frame rotated about the vertical. Nothing compares an absolute azimuth from the phone against
an absolute azimuth from the solve.

Elevation is not frame-relative — it is measured from gravity at both ends — so elevation
bands do mean the same thing on the phone and in the report.

The band edges
--------------
``AZIMUTH_BAND`` 45° is the reporting band ``hs views`` has used since it grew a per-band
table. The elevation rings are the observed spans of the captures we have, not validated
optima: rig6 ran −12°…+26° with its low passes clustered near −7°, and the 2026-09-15 head
runs sit in the same place. "high" exists because the coverage map keeps showing it empty.
Treat the ring edges as a starting point to be moved once a capture shot *to* them has been
scored.
"""
import math

AZIMUTH_BAND = 45          # degrees per reporting band

# (low, high, name); half-open [low, high). Anything outside every ring is out of band.
ELEVATION_BANDS = ((-25.0, 5.0, "low"), (5.0, 20.0, "mid"), (20.0, 45.0, "high"))

DWELL_S = 1.5              # seconds inside a cell before it counts as covered
SLOW_DEG_S = 25.0          # above this angular rate the frames smear; dwell does not accrue


def wrap180(deg):
    """Fold an angle into (−180, 180]."""
    return -((-float(deg) + 180.0) % 360.0 - 180.0)


def azimuth_band_lo(az, band=AZIMUTH_BAND):
    """The low edge of the azimuth band holding ``az`` — matches views.azimuth_table()."""
    return int(math.floor(wrap180(az) / band) * band)


def elevation_band(el):
    """The ring name holding ``el``, or None when it is outside every ring."""
    for lo, hi, name in ELEVATION_BANDS:
        if lo <= el < hi:
            return name
    return None


def ring_position(el):
    """Ring index holding ``el``; −1 below the lowest ring, len(ELEVATION_BANDS) above the top.

    Out of band is not one place: below the low ring the fix is to raise the camera, above the
    high ring it is to lower it. Treating both as ring 0 told an operator already under the low
    ring to "lower", and take04 shows him doing it — from −26° to −46°.
    """
    ring = elevation_band(el)
    if ring is not None:
        return ring_index(ring)
    return -1 if el < ELEVATION_BANDS[0][0] else len(ELEVATION_BANDS)


def cell(az, el, band=AZIMUTH_BAND):
    """(azimuth band low edge, ring name) — or None if the elevation is outside every ring."""
    ring = elevation_band(el)
    return None if ring is None else (azimuth_band_lo(az, band), ring)


def all_cells(band=AZIMUTH_BAND):
    """Every cell of the grid, in az-then-ring order."""
    los = range(-180, 180, band)
    return [(lo, name) for lo in los for _, _, name in ELEVATION_BANDS]


def band_centre(lo, band=AZIMUTH_BAND):
    return wrap180(lo + band / 2.0)


def ring_centre(name):
    for lo, hi, n in ELEVATION_BANDS:
        if n == name:
            return (lo + hi) / 2.0
    raise KeyError(name)


def ring_index(name):
    for i, (_, _, n) in enumerate(ELEVATION_BANDS):
        if n == name:
            return i
    raise KeyError(name)


class Grid:
    """Dwell-weighted coverage of the azimuth × elevation grid.

    ``mark`` accrues time in the cell the camera is currently looking from, but only while the
    camera is turning slowly enough that the frames are not smeared — a cell swept through at
    speed is not covered, which is the same judgement ``captures_sharper_than_the_model`` makes
    after the fact. A cell reaching ``dwell`` seconds is covered.
    """

    def __init__(self, band=AZIMUTH_BAND, dwell=DWELL_S, slow_deg_s=SLOW_DEG_S):
        self.band, self.dwell, self.slow = band, float(dwell), float(slow_deg_s)
        self.time = {c: 0.0 for c in all_cells(band)}

    def mark(self, az, el, dt, rate_deg_s=0.0):
        """Accrue ``dt`` seconds at (az, el). Returns the cell if this call completed it."""
        c = cell(az, el, self.band)
        if c is None or rate_deg_s > self.slow or dt <= 0:
            return None
        before = self.time[c]
        self.time[c] = before + float(dt)
        return c if before < self.dwell <= self.time[c] else None

    def covered(self):
        return {c for c, t in self.time.items() if t >= self.dwell}

    def missing(self):
        return [c for c in all_cells(self.band) if self.time[c] < self.dwell]

    def ring_complete(self, name):
        return all(self.time[c] >= self.dwell for c in self.time if c[1] == name)

    def nearest_missing(self, az, el):
        """The uncovered cell that is the least work to reach from here, or None if done.

        Cost is the azimuth turn in degrees plus a fixed penalty per ring you would have to
        change — walking 40° round the subject is easier than raising the camera a ring, and
        this keeps the guide from ping-ponging between rings on every step.
        """
        miss = self.missing()
        if not miss:
            return None
        here_i = min(max(ring_position(el), 0), len(ELEVATION_BANDS) - 1)

        def cost(c):
            lo, name = c
            return abs(wrap180(band_centre(lo, self.band) - az)) + 60.0 * abs(ring_index(name) - here_i)

        return min(miss, key=cost)

    def cue(self, az, el):
        """(instruction, target cell) for where to go next — ('done', None) when the grid is full.

        The instruction is what the operator has to *do*: raise or lower the camera to change
        ring, otherwise turn left or right round the subject, otherwise hold still because the
        cell you are in has not clocked its dwell yet.
        """
        target = self.nearest_missing(az, el)
        if target is None:
            return "done", None
        lo, name = target
        here = elevation_band(el)
        if here != name:
            return ("raise" if ring_index(name) > ring_position(el) else "lower"), target
        delta = wrap180(band_centre(lo, self.band) - az)
        if abs(delta) <= self.band / 2.0:
            return "hold", target
        return ("right" if delta > 0 else "left"), target

    def report(self):
        """Serialisable coverage, cells in grid order."""
        return [{"azimuth_deg": [lo, lo + self.band], "ring": name,
                 "seconds": round(self.time[(lo, name)], 2),
                 "covered": self.time[(lo, name)] >= self.dwell}
                for lo, name in all_cells(self.band)]
