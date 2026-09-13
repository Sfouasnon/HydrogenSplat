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
hs ingest  -p P --clip VID_..._2x1.h4v [--link]                       copy, MD5, ffprobe, validate 2x1 video, match the calibration profile
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

## The Brush wrappers, verified 2026-09-13 on Apple Silicon

`hs train` and `hs render` drove the real binaries end to end on the rig6 selftest project.
What the wrappers assume, read from the fork's source and now confirmed by that run:

* `brush <dataset> --total-train-iters 40000 --growth-stop-iter 30000 --refine-every 130
  --export-every 2500 --export-path <abs> --export-name export_{iter}.ply`. `--export-path`
  is joined onto the dataset's *parent* directory (absolute paths pass through) and
  `--export-name` is a template whose `{iter}` is zero-padded to the digit count of the
  total — confirmed: `export_02500.ply` … `export_40000.ply`.
* The indicatif progress bar is hidden when stderr is not a TTY, so a piped `brush` never
  prints `NNNN/40000 Steps`. With `RUST_LOG=info` (which `hs train` sets) env_logger's
  `Refine iter N, M splats.` arrives every `--refine-every` steps and is what drives
  progress — confirmed, ~9 s apart. Export files on disk remain the ground truth.
* **Brush stops refining before the end.** The last `Refine iter` of the golden train was
  37961 of 40000, so the step counter goes quiet for the final ~2,000 iterations (148 s at
  the observed rate). `hs train` re-emits its last position every 15 s (`HEARTBEAT_S`) so a
  consumer can tell "still training" from "hung"; `done` deliberately does not advance,
  because the iteration is genuinely unknown. `engine/tests/fake_brush.py` reproduces this
  tail via `HS_FAKE_REFINE_STOP` / `HS_FAKE_QUIET_TAIL`.
* **The early ETA is optimistic.** Throughput falls as the model grows — 23.1 it/s at the
  first refine, 13.8 it/s by the end — so the ETA at iteration 131 read 29 min against an
  actual 48.5 min. Treat the first few minutes' ETA as a lower bound.
* Resume: `--start-iter N` only moves the loop start; the init splats are whatever `.ply`
  the dataset holds (`init.ply` wins). `hs train --resume-from export_NNNNN.ply` copies it
  in as `init.ply` for the run and removes it after. Mechanically supported, quality
  unverified — M3 decides whether the UI says "resume" or "restart".
* `brush-path-render <ply> --path move.json -o DIR --width 2400` prints `path: N frames…`,
  `loaded N splats`, `  frame i/N  (t elapsed)` every 10 frames, `wrote N frames to …`.

### The golden train and render (65 frames, M-series Mac)

| | this run | rig6 reference |
|---|---|---|
| train wall clock | 48.5 min (40k iters, 13.76 it/s average) | ~55 min |
| final splats | 157,252 | 170,841 (−8.0%) |
| growth at 5k / 10k / 15k | 75,408 / 91,678 / 106,387 | 74,699 / 90,850 / 107,198 |
| growth at 25k / 30k / 40k | 137,221 / 152,939 / 157,252 | 143,399 / 162,742 / 170,841 |
| render | 360 frames at 2400×1346 in 21.9 s (16.4 fps) | ~14 fps |

The growth curves track within 1% to 15k and then diverge, the deficit reaching 8% at 40k —
consistent with a different seed and a slightly different solve, not with a wrapper problem.
Both land inside the check's 0.6–1.4× band and inside the strategy's 170k ± 20k. Per spec §5
the count is not what decides quality here; a visual comparison against
`compare/rig6_boom_12s.mp4` is the open question this run does not answer.

### Testing the wrappers without a GPU

`engine/tests/run_fake_pipeline.sh [project]` runs train → prune → render against stand-ins
(`engine/tests/fakebin/`) that print the same lines and write real PLY and PNG files, on top
of a finished selftest project. ~95 s.
