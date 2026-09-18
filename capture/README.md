# capture — assist on the phone, not diagnosis afterwards

`hs views` grades a finished model and draws a coverage map. Both are post-mortems: by the time
they tell you the high elevation ring is empty, the subject has gone home. `orbit_guide.py` is
the other half of that loop — it runs on the Hydrogen One while Holocam records, and turns the
same grid into instructions you can follow with the phone in your hands.

Both ends import `hs/bands.py`, which is the only place the grid is defined. That is deliberate:
a guide that sends you to fill a band the scorer does not believe in is worse than no guide.

## What it needs on the phone

Termux, plus **both halves** of Termux:API — the APK and the CLI package. They are separate and
each is useless alone:

```
adb install termux-api.apk       # from https://f-droid.org/repo/com.termux.api_1002.apk
pkg install -y termux-api        # inside Termux
```

Sensors need no runtime permission, so there is no grant prompt to chase. Speech and vibration
come from the same package and are optional (`--silent`, `--no-vibrate`).

## Deploying

The guide needs three files: itself, `hs/bands.py`, and the package's `__init__.py`. Nothing
else from the engine, and no numpy — Termux Python is stdlib only here.

```
adb push capture/orbit_guide.py engine/hs/bands.py engine/hs/__init__.py /sdcard/hsguide/
```

Then in Termux, with `orbit_guide.py` beside an `hs/` directory holding the other two.

## Shooting with it

```
python orbit_guide.py --out ~/storage/shared/hsguide/take07.jsonl
```

Start it, start Holocam, shoot. It says *raise*, *lower*, *left*, *right*, *hold* and *slower*,
and buzzes each time a cell clocks its dwell. It stops itself when the grid is full, or on
Ctrl-C, and prints which cells are still open.

`--out` writes the orientation track as JSON lines. Because the spoken cues land in Holocam's
own audio track, that track can be aligned to the clip afterwards by correlating cues against
cues, rather than trusting two unsynchronised clocks.

## Reading a track back

Everything runs off the track file, so a take can be re-scored, or the tuning changed and
re-judged, without the phone:

```
python capture/orbit_guide.py --replay take07.jsonl --dwell 2.5
```

## What it does not know

It has orientation, not position. It infers where you are standing from where you are pointing,
which holds while you orbit a subject you keep centred and breaks the moment you pan from a
fixed spot. It cannot tell you that you drifted closer to the subject, and it has no idea what
is in frame.

The ring edges in `hs/bands.py` are the observed spans of the captures we have, not validated
optima. They should move once something shot *to* them has been scored.
