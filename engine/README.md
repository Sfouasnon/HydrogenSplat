# engine

One CLI, `hs`, that wraps the rig6 pipeline with the JSON-lines event contract of
`docs/hydrogensplat-strategy-v1.md` §2. The six scripts vendored from
`~/Desktop/Apps/REDHydrogenOne/mv` on 2026-09-14 are **byte-identical** to the scaffold
commit and still run standalone exactly as documented in `docs/capture-to-splat-spec-v1.md`;
`hs` runs them as subprocesses and never touches their numerics.

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
hs ingest  -p P --clip VID_..._2x1.h4v [--link]   copy, MD5, ffprobe, validate 2x1 video, match the calibration profile
hs select  -p P [--residual 1.5 --max-gap 90 --end N --dry-run]      frames + selection.json + contact.jpg
hs solve   -p P                                                       prep → sfm --float-rig → export; per_image.json, coverage.json
hs train   -p P [--brush PATH]                                        brush → train/exports/export_NNNNN.ply   (Mac only)
hs move    -p P --preset sweep|boom|custom [--keys ...] [--name N]    move/N.json + move/N_aim_check.jpg
hs prune   -p P [--radius 0.3]                                        prune/<export>_pruned_r03.ply
hs render  -p P --move N [--ply PATH] [--width 2400] [--keep-frames]  render/N_1920.mp4, N_1080x1350.mp4  (Mac only)
hs tools                                                              versions of python packages, brush, brush-path-render, ffmpeg, adb
hs calib   --photos 'board/*.jpg' | --video board.h4v -o cal.npz      stereocal.py + a profile JSON
hs selftest [--clip CLIP] [--project DIR] [--resume|--fresh]          golden test (below)
```

Every stage: `pj.require()` checks its prerequisites are `done` in the manifest, wipes only its
own folder, marks downstream stages `stale`, writes `logs/<stage>.log`, and records argv,
metrics, checks and artifacts in `manifest.json`. `-v` also streams the child's output as
`{"ev":"log"}` events.

## Golden test

```
cp ~/Desktop/Apps/REDHydrogenOne/capture_video/VID_20260912_151351_2x1.h4v fixtures/
.venv/bin/hs selftest
```

Runs ingest → select → solve → move (sweep, auto window) → move (boom, auto keys) into
`fixtures/selftest/` (gitignored) and asserts the numbers in `fixtures/README.md`. A
human-readable summary goes to stderr; the events go to stdout; `fixtures/selftest/selftest.json`
keeps the verdict. Exit 0 only if every assertion holds. ~11 min on 2 cores, ~15 on the Mac.

Result on 2026-09-13 (x86_64 container, pycolmap 4.2.0, OpenCV 5.0.0):
66 frames selected · 66/66 registered · mean reprojection 1.348 px · elevation −12.9° … +24.6° ·
SfM median → (975, 496) in cap021_L at 226 mm, 59 px from the reference (959, 552) · rig held at 10.642 mm ·
sweep auto window 3:34, hull 17 mm · boom keys 51,56,58,60,62,64,14,61, hull 21 mm.

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
