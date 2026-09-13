# engine

One CLI, `hs`, that wraps the rig6 pipeline with the JSON-lines event contract of
`docs/hydrogensplat-strategy-v1.md` §2. The scripts vendored from
`~/Desktop/Apps/REDHydrogenOne/mv` on 2026-09-14 still run standalone exactly as documented
in `docs/capture-to-splat-spec-v1.md`; `hs` runs them as subprocesses and never touches their
numerics. Five of the six are byte-identical to the scaffold commit. The exception is
`rigcolmap.py`, whose `write_rig_npz` took `rig.npz`'s `w`/`h` from `rec.images`' arbitrary
first entry: undistortion crops each eye to its own valid region (rig6: L 1913×1073,
R 1909×1071), so the path builders were handed the right eye's canvas for left-eye views —
the aim check's in-frame bounds and every path's output size were off by (4, 2) px. It now
takes them from the reference (left) view and also stores a per-view `wh`. No computed value
changed; `hs solve` checks the two agree (`rig_size_matches_L_views`).

| script | stage | needs |
|---|---|---|
| select_frames.py | `hs select` | numpy, cv2 |
| rigcolmap.py | `hs solve` (prep / sfm / export) | numpy, cv2, pycolmap 4.2.0 |
| spline_path.py | `hs move --preset sweep` | numpy |
| key_path.py | `hs move --preset boom\|custom` — imports spline_path | numpy |
| prune_splats.py | `hs prune` | numpy |
| stereocal.py | `hs calib` (v1.1 UI) | numpy, cv2 |

## Install

```
cd ~/Desktop/Apps/HydrogenSplat
python3 -m venv .venv && .venv/bin/pip install -e engine
.venv/bin/hs tools
```

`pip install -e engine` puts an `hs` entry point in `.venv/bin`; `python -m hs` works too.

## Layout

```
hs/
  cli.py        hs <stage> --project DIR [...]  — exit 0 ok, 1 stage error, 2 crash, 130 ^C
  events.py     the one JSON-lines emitter (start/progress/metric/artifact/check/done/error)
  runner.py     subprocess runner: streams lines split on \r and \n, tees to logs/<stage>.log
  project.py    project folder + manifest.json (status per stage, argv, metrics, checks; stale propagation)
  calib.py      profile JSON <-> the stereocal npz rigcolmap.py reads; match keys
  coverage.py   azimuth / elevation / distance per capture from rig.npz; sweep-window and boom-key presets
  stages/       ingest select solve train move prune render tools calibrate selftest
  <six vendored scripts>
```

## Stages

```
hs ingest  -p P --clip VID_..._2x1.h4v [--link]                       copy, MD5, ffprobe, validate 2x1 video, match the calibration profile
hs select  -p P [--residual 1.5 --max-gap 90 --end N --dry-run]      frames + selection.json + contact.jpg
hs solve   -p P                                                       prep → sfm --float-rig → export; per_image.json, coverage.json
hs train   -p P [--brush PATH]                                        brush → train/exports/export_NNNNN.ply   (Mac only)
hs move    -p P --preset sweep|boom|custom [--keys ...] [--name N]    move/N.json + move/N_aim_check.jpg
hs prune   -p P [--radius 0.3]                                        prune/<export>_pruned_r03.ply
hs render  -p P --move N [--ply PATH] [--width 2400] [--keep-frames]  render/N_1920.mp4, N_1080x1350.mp4  (Mac only)
hs views   -p P [--captures 5,15,55] [--ply PATH]                     grade the model against the photographs  (Mac only)
hs tools                                                              versions of python packages, brush, brush-path-render, ffmpeg, adb
hs calib   --photos 'board/*.jpg' | --video board.h4v -o cal.npz      stereocal.py + a profile JSON
hs selftest [--clip CLIP] [--project DIR] [--resume|--fresh]          golden test (below)
```

`-p/--project` and `-v/--verbose` are accepted on either side of the stage name: `hs -p DIR
solve` and `hs solve -p DIR` are the same command.

Every stage: `pj.require()` checks its prerequisites are `done` in the manifest, wipes only its
own folder, marks downstream stages `stale`, writes `logs/<stage>.log`, and records argv,
metrics, checks and artifacts in `manifest.json`. `-v` also streams the child's output as
`{"ev":"log"}` events. Opening a project reconciles it first: a stage the manifest still calls
`running` whose recorded pid is gone becomes `failed — interrupted`, so a ^C'd or crashed run
reports that instead of blocking the next stage with "solve is running".

## Golden test

```
cp ~/Desktop/Apps/REDHydrogenOne/capture_video/VID_20260912_151351_2x1.h4v fixtures/
.venv/bin/hs selftest
```

Runs ingest → select → solve → move (sweep, auto window) → move (boom, auto keys) into
`fixtures/selftest/` (gitignored) and asserts the numbers in `fixtures/README.md`. A
human-readable summary goes to stderr; the events go to stdout; `fixtures/selftest/selftest.json`
keeps the verdict. Exit 0 only if every assertion holds. ~3 min on the Mac, ~10 on 2 cores.

Results, 2026-09-13 — 12/12 on both platforms. COLMAP mapping is not deterministic and the
two builds differ, so these are the spread to expect, not fixed numbers:

| | Apple Silicon (python 3.14, numpy 2.5.3, cv2 5.0.0, pycolmap 4.2.0) | x86_64 container (python 3.11, numpy 2.4.6) |
|---|---|---|
| runtime | 3.1 min | 9.9 min |
| frames selected | 65 (= frames4) | 66 |
| registered | 65/65 | 66/66 |
| mean reprojection | 1.352 px | 1.343 px |
| elevation range | −11.9° … +26.0° | −12.3° … +25.2° |
| median aim in cap021_L | (969, 532), 23 px from (959, 552) | (974, 512), 43 px |
| rig baseline | 10.642 mm, zero spread | same |
| sweep | window 2:33, hull 16 mm | window 3:34, hull 17 mm |
| boom | keys 50,55,57,59,61,63,14,5, hull 22 mm | keys 51,56,58,60,62,64,14,61, hull 21 mm |

The Mac is the reference platform and lands closer to rig6 on every number. The 60 px aim
tolerance is the one with the least headroom (23–43 px observed).

## Grading the model against the photographs

```
.venv/bin/hs views -p fixtures/selftest
```

Renders the trained `.ply` from real capture poses — by default both azimuth extremes, the
most central view and the highest — at each capture's own K and canvas, so each render lands
pixel-aligned on its training image. Per view it reports the edge energy the model retains
from that photograph, PSNR and correlation against it, and a phase-correlation displacement
field over 64 px patches; it writes a four-panel comparison (photograph, model, difference,
patch map) per view plus `views/views_report.json`.

The point is that every other number in the pipeline is the solver grading its own homework.
The reprojection residual says how well the SfM fits the features it picked; the hull check
says how far a virtual camera strays from a real one. Neither can say the model is wrong.
This can, because the reference is the photograph.

The two metrics separate failure modes that look identical in a rendered move: blur costs
edge energy and leaves displacement alone, misregistration keeps edge energy and displaces
patches. On the rig6 golden train:

| view | az | edge energy kept | PSNR | displaced > 4 px | p90 |
|---|---|---|---|---|---|
| cap005 | +1° | 60% | 35.1 dB | 0% | 1.8 px |
| cap015 | −42° | 60% | 29.3 dB | 1% | 1.6 px |
| cap055 | +43° | 50% | 28.2 dB | **20%** | **10.2 px** |
| cap061 | −11° | 61% | 27.8 dB | 6% | 2.3 px |

Equal edge energy with 20× the displaced fraction is what identified the soft right side of
that render as thin coverage and weak registration rather than the motion blur it resembled;
10.2 px at 235 mm is 1.4 mm of world error. The displaced fraction is diluted by background
inside the crop, so compare views within a capture, not across differently framed runs.

## Brush facts the wrappers rely on (read from the fork's source, not yet exercised)

* `brush <dataset> --total-train-iters 40000 --growth-stop-iter 30000 --refine-every 130
  --export-every 2500 --export-path <abs> --export-name export_{iter}.ply`. `--export-path`
  is joined onto the dataset's *parent* directory (absolute paths pass through); `{iter}` is
  zero-padded to the digit count of the total → `export_02500.ply` … `export_40000.ply`.
* The indicatif progress bar is hidden when stderr is not a TTY, so a piped `brush` never
  shows `NNNN/40000 Steps`. `hs train` sets `RUST_LOG=info` and parses env_logger's
  `Refine iter N, M splats.` (every `--refine-every` steps) for progress; export files on disk
  are the ground truth. Both parsers exist in case the output changes.
* Resume: `--start-iter N` only moves the loop; the init splats are whatever `.ply` the
  dataset holds (`init.ply` wins). `hs train --resume-from export_NNNNN.ply` copies it in as
  `init.ply` for the run and removes it after. Mechanically supported, quality unverified (M3).
* `brush-path-render <ply> --path move.json -o DIR --width 2400` prints `path: N frames…`,
  `loaded N splats`, `  frame i/N  (t elapsed)` every 10 frames, `wrote N frames to …`.

`hs train` and `hs render` were exercised end to end against stand-in binaries that print
those exact lines and write real PLY/PNG files (`scratch/fakebin` in the M0 session). One run
each on the Mac against the real binaries is what closes M0 for them.
