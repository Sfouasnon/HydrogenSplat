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
BACK_COST = 3.0            # one band backwards costs as much as this many bands ahead
FLIP_DEG = 30.0            # net backward travel that means the operator has turned round


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
        self.direction = +1        # +1 walking the way azimuth increases ("right"), −1 the other
        self._last_az = None
        self._backward = 0.0       # net travel against self.direction since last going forward
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

    def observe(self, az):
        """Track which way round the operator is walking.

        Direction flips only after FLIP_DEG of net travel the other way, so hand jitter and a
        step back to re-frame do not reverse the guide; turning round and walking does.
        """
        if self._last_az is not None:
            step = wrap180(az - self._last_az) * self.direction
            self._backward = min(0.0, self._backward + step)
            if self._backward <= -FLIP_DEG:
                self.direction, self._backward = -self.direction, 0.0
        self._last_az = az

    def _band_steps(self, az, lo):
        """(bands ahead, bands behind) from the band holding ``az`` to band ``lo``."""
        n = 360 // self.band
        i = (azimuth_band_lo(az, self.band) + 180) // self.band
        j = (lo + 180) // self.band
        ahead = ((j - i) * self.direction) % n
        return ahead, (n - ahead) % n

    def _plan(self, az, el):
        """(target cell, go ahead?) or (None, None) when the grid is full."""
        miss = self.missing()
        if not miss:
            return None, None
        here_i = min(max(ring_position(el), 0), len(ELEVATION_BANDS) - 1)
        best = None
        for lo, name in miss:
            ahead, behind = self._band_steps(az, lo)
            fwd = ahead <= BACK_COST * behind
            walk = ahead if fwd else BACK_COST * behind
            key = (walk, abs(ring_index(name) - here_i))
            if best is None or key < best[0]:
                best = (key, (lo, name), fwd)
        return best[1], best[2]

    def nearest_missing(self, az, el):
        """The uncovered cell to go to next, or None if done.

        One pass, all rings: finish every ring in the band you are standing in before moving
        on — nearest ring first, so consecutive bands run low→high then high→low — then go to
        the next band *ahead* in the direction you are walking. A band behind costs BACK_COST
        bands ahead: a gap one band back is only worth turning round for once nothing is
        missing within three bands ahead. take04 is why — at +150° walking right, a gap one
        band behind made the old nearest-by-angle rule say left, right, left for 15 s.
        """
        return self._plan(az, el)[0]

    def cue(self, az, el):
        """(instruction, target cell) for where to go next — ('done', None) when the grid is full.

        The instruction is what the operator has to *do*: raise or lower the camera to change
        ring, otherwise turn left or right round the subject, otherwise hold still because the
        cell you are in has not clocked its dwell yet.
        """
        target, fwd = self._plan(az, el)
        if target is None:
            return "done", None
        lo, name = target
        if lo != azimuth_band_lo(az, self.band):
            # walk first, change ring on arrival — the walk is the bigger move
            way = self.direction if fwd else -self.direction
            return ("right" if way > 0 else "left"), target
        if elevation_band(el) != name:
            return ("raise" if ring_index(name) > ring_position(el) else "lower"), target
        return "hold", target

    def report(self):
        """Serialisable coverage, cells in grid order."""
        return [{"azimuth_deg": [lo, lo + self.band], "ring": name,
                 "seconds": round(self.time[(lo, name)], 2),
                 "covered": self.time[(lo, name)] >= self.dwell}
                for lo, name in all_cells(self.band)]
