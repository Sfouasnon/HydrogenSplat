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
  movescript.py the .hsmove cue language: boom / arc / dolly / hold in capture coordinates, clamped to the hull
  splatweights.py the renderer's forward weights w = alpha*T per (splat, view, cell), in numpy — under split and prune --score
  stages/       ingest select solve exposure masks train move prune render tools calibrate selftest
  <six vendored scripts>
```

## Stages

```
hs ingest  -p P --clip VID_..._2x1.h4v [--link]                       copy, MD5, ffprobe, validate 2x1 video, match the calibration profile
hs ingest  -p P --frames DIR | --r3d RDM_DIR --take 067 [--res 1]     array source: one frame per camera (REDline renders the R3Ds); select is marked done
hs select  -p P [--residual 1.5 --max-gap 90 --end N --dry-run]      frames + selection.json + quality.json + thumbs/ + contact.jpg
hs solve   -p P                                                       prep → sfm --float-rig → export; per_image.json, coverage.json
hs solve   -p P --scale-pair GA,GB,700 [--focal-px F]                 array project: monocolmap.py, one shared camera, metric scale from a measured spacing
hs exposure -p P [--reference median|auto|capNNN] [--mode rgb|luma] [--restore] [--dry-run]   match every view to one reference (auto = select's best-exposed clean pick)
hs masks   -p P [--radius 0.10] [--margin-mm 5] [--min-opacity 0.2]   per-view subject silhouettes for Brush's mask channel
hs train   -p P [--brush PATH]                                        brush → train/exports/export_NNNNN.ply   (Mac only)
hs move    -p P --preset sweep|boom|custom [--keys ...] [--name N]    move/N.json + move/N_aim_check.jpg
hs move    -p P --script shot.hsmove [--name N]                      compile a cue sheet against the captured hull (movescript.py)
hs prune   -p P [--radius 0.3]                                        prune/<export>_pruned_r03.ply
hs prune   -p P --score [--ply PLY] [--name N] [--cell 8]            prune/N_scores.npz + N_report.json — a report, prunes nothing
hs prune   -p P --floaters [--name N] [--min-importance-quantile 0.02] [--max-blame 0.25]   prune/N_nofloat.ply + N_floaters_only.ply
hs split   -p P [--ply PLY] [--masks vision|region|DIR] [--exclude L/cap064,…|@holdout] [--name N] [--cell 8] [--refine knn|none]
                                                                      split/N/{full_labelled,subject,background}.ply + report.json
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

## Post-hoc split and floater score (2026-09-21)

Both read one thing: for every training view, which splats the renderer composites into each
`--cell`-px cell of the photograph, front to back, and with what weight `w = α·T`
(`hs/splatweights.py` — the 3DGS forward pass in numpy: EWA footprint from scale/rotation,
0.3 px dilation, α clamped to 0.99, per-cell depth sort and cumulative log-transmittance, all
vectorised; no GPU, no Brush). Footprints are prefiltered to the cell so a splat smaller than a
cell contributes its area rather than hitting or missing (the module docstring says why a quarter
cell², not a box's 1/12). Only the training views are used: by default the views the model was
trained without (its archive manifest, or the train stage's fingerprint) are excluded, and
`--exclude` takes `hs train`'s syntax or `@holdout` (`solve/holdout.json`).

**`hs split`** labels the model instead of training a second one: `p = Σ w·M / Σ w` per splat
(FlashSplat's closed form; `M` the mask's mean over the cell), the ambiguous band (0.2–0.8) and
the unseen splats refined by label diffusion over a `cv2.flann` KNN in (xyz, DC colour), then
exported to `split/<name>/` as `full_labelled.ply` (all properties + `subject_p`), `subject.ply`,
`background.ply`, `report.json` and a `manifest.json` naming the rig so `hs render` / `hs views
--ply` take the layers like an archive. Checks: `masks_cover_training_views`, `labels_bimodal`
(< 5 % still ambiguous, `needs_human` otherwise). `--masks vision|region` checks that
`train/dataset/masks` was made by `hs masks --method vision|geometry`; a folder works too.

**`hs prune --score`** writes per-splat `importance` (Σ w, LightGaussian without the volume
term), `top_contributor` (largest w in some cell — Mini-Splatting's guard), `views_seen` and
`blame` (Σ w·e / Σ w, `e` the cell's |composited DC colour − photograph| in linear RGB) to
`prune/<name>_scores.npz`, with histograms in `_report.json`. It is a report: prune's status and
metrics, and render's, are left alone (the run is kept under `stages.prune.runs`).
**`--floaters`** removes splats that are low-importance *and* top contributor nowhere *and*
high-blame, writing `<name>_nofloat.ply` and `<name>_floaters_only.ply` (look at the second in the
viewer before trusting the first) and the opacity mass kept; it reuses the scores when they match
the ply and cell, and records `output_ply` / `input_ply_md5` so render's lineage guard accepts
the result as prune's output. The geometric `hs prune` is unchanged.

```
hs split -p P --ply archive/holdout-base/export_40000.ply --masks vision
hs prune -p P --score --ply archive/holdout-base/export_40000.ply --name holdout-base
hs prune -p P --floaters --name holdout-base
```

Colour is DC only (SH off), so blame is coarse on view-dependent surfaces. The 0.25 blame default
is the plan's and untested on real data: on the synthetic test a 25 %-opaque black floater over a
white card scores 0.18 — a floater cannot darken a cell much further before it becomes that cell's
top contributor and the guard keeps it. Speed on the 2-core cloud container: 200k splats × 30
views at 1920×1080, cell 8, 45 s (`HS_TIMING=1 pytest tests/test_splatweights.py`); views run in
a small thread pool (`--jobs`, default up to 4).

## Where the failure sits, not just how much (2026-09-17)

Three hold-out runs all said the same thing in a form too coarse to act on: one median per view,
one median per run. `hs views` now also reports

- **`displaced_fraction_core` / `_surround`** per view, and their medians. The crop is 2×half
  across and centred on the subject, so patches within half that radius are the subject and the
  rest is what surrounds it. Radial, not segmented: it says which ring the displacement is in and
  claims nothing more. Check `subject_registers_better_than_its_surroundings` fires when the
  subject is the worse of the two, which would be a different (and worse) failure.
- **`by_azimuth`**: medians per 45° band, with `every_azimuth_band_registers` naming the worst
  band against the same threshold. Every hold-out run so far has been fine where the camera went
  often and bad at the edges of its coverage; a per-run median cannot show that.
- **`views_render_sharper_than_photo`** and `captures_sharper_than_the_model`: views whose
  normalised edge energy exceeds 1.0 — the photograph is blurrier than the render, so its score
  is a floor, not a model fault. Replaces `edge_energy_consistent_across_views`, whose spread
  bound measured variation in the photographs (the 09-16 head spanned 39–103% because of its two
  motion-blurred captures). The new bound, `no_view_much_softer_than_achievable`, flags any view
  under 0.45 of its grain ceiling.

`splat_count_in_range` is recalibrated too. A 0.6–1.4× band around rig6's 2,628 splats/frame
failed every model since — head 8,833/frame, body 13,567, array 15,394, none of them wrong — so
it is now a sanity band, 500–40,000 per frame: under it growth never took, over it the model is
too heavy to render and probably full of floaters. `splats_per_frame` is recorded either way.

## Every run records its own events (2026-09-17)

Each project stage appends its event stream to `<project>/logs/<stage>.events.jsonl`: a `run`
event first (argv, wall clock), then every event the stage emitted, each with `t` — seconds
since that run started. `hs replay <file> --last` plays back the final run at its own pace, so
a finished run can be pushed through the app's Event Replay view afterwards. The text log
beside it (`logs/<stage>.log`) is still the child's raw output; this one is the engine's own.
stdout is unchanged — `t` exists only in the file. A log that cannot be opened is skipped, never
fatal.

Use it when the app shows something the numbers do not explain (the 2026-09-17 growth chart drew
a 16M spike on a 1.0M model): replay the run instead of reasoning about what the app received.

```
hs replay Projects/2026-09-15_head/logs/train.events.jsonl --last --speed 50
```

## The grain ceiling in `hs views` (2026-09-17)

`retained_edge_energy` is the render's 90th-percentile |Laplacian| over contrast divided by the
photograph's. A splat render carries no sensor noise and a photograph does, so **removing the
photograph's grain alone** — a 3×3 median, which moves no edge — already costs a large share of
the metric:

| operation on the photograph | array GA (KOMODO-X, ISO 800, 4K) | head cap004 (Hydrogen) |
|---|---:|---:|
| 3×3 median (grain gone, edges kept) | 0.600 | 0.774 |
| 5×5 median | 0.403 | 0.622 |
| Gaussian 0.5 px | 1.000 | 0.928 |
| Gaussian 1.0 px | 0.800 | 0.541 |
| Gaussian 1.5 px | 0.400 | 0.391 |

That share is the ceiling: the most a noise-free render can score on that clip. So `views` now
reports `src_denoised_sharpness`, `grain_ceiling` (= denoised/raw source sharpness) and
**`retained_edge_energy_norm` = render sharpness / denoised source sharpness**, with medians
`retained_edge_energy_norm_median` and `grain_ceiling_median`, and
`edge_energy_consistent_across_views` now spans the normalised values. **Compare runs by
`_norm`**; the raw number carries the clip's grain and is not comparable between clips.

What it is worth: the 2026-04-03 array scored 0.457 raw on views that match the photographs at
42 dB PSNR with a black difference image — 0.76 of its ceiling, i.e. sub-half-pixel blur. The
head hold-out baseline scored 0.44 raw = 0.57 of its 0.77 ceiling, about 1 px equivalent. The
synthetic harness in `tests/test_lifecycle.py::ViewsEdgeMetric` makes the same point: a
geometrically exact, noise-free render of a grainy photograph scores 0.65 raw and 0.99 norm.

## Array source: R3D camera arrays (2026-09-17)

The pipeline also takes a **camera array** — one photograph per camera, every camera the same
body and lens (the 2026-04-03 KOMODO-X 4×3 rig: 12 cameras, three synchronised takes of a
gray sphere + tag board / gray card + focus chart / Macbeth). Nothing else about the project
changes: `train`, `archive`, `views`, `move` and `render` read the same `train/dataset` and
`rig.npz`.

```
hs ingest -p Projects/2026-04-03_array067 --r3d ~/Desktop/Camera\ Footage/RED_Footage --take 067
hs ingest -p P --frames DIR            # or: a folder of PNG/JPEG/TIFF frames already named by camera
hs solve  -p P --scale-pair GA,GB,700  # the measured distance in mm between two camera centres
```

- **ingest** finds every `*_?067_*.RDC/*_001.R3D` under the RDM tree, names each by its camera
  position (`G007_A067` → `GA`) and renders the first frame through REDline (`--format 1
  --gammaCurve 1 --colorSpace 1`, i.e. a 16-bit BT.709 TIFF, converted to 8-bit PNG). REDline is
  found on `PATH`, then `~/bin/REDline`, then `/usr/local/bin/REDline`, or given with
  `--redline` / `HS_REDLINE`. Frames land in `source/frames/`; `source.kind` is `array` in the
  manifest; `select` is marked done with symlinks (there is nothing to select from one frame per
  camera, and `hs select` refuses an array project).
- **solve** runs `monocolmap.py` instead of `rigcolmap.py`: SIFT → exhaustive matching → incremental
  mapping with ONE shared `OPENCV` camera, focal and distortion refined, **principal point held at
  the centre** (freeing it collapsed the 12-view solve to two images: a mostly planar set gives it
  nothing to pull against). Then the metric scale: photogrammetry from uncalibrated cameras has no
  unit, so `--scale-pair CAM1,CAM2,MM` (a measured camera spacing) or `--scale S` sets it; without
  one the `scene_scaled` check fails (`needs_human`) and the dataset is exported unscaled.
- **rig.npz** gets the mono layout (`hs/rig.py`): one view per camera named `<cam>_L`, `stereo =
  False`, images in `train/dataset/images/L/<cam>.jpg`. Every consumer of the old `[::2]` idiom
  (coverage, spline_path, key_path, movescript, views) goes through `rig.left_indices()`, so a
  stereo file without the key still reads as pairs. `hs views --eye R` and `key_path --eye R|mid`
  refuse a mono rig. `--exclude` takes camera ids (`GA,HB`) as well as `L/cap064`.
- On the 2026-04-03 take 067 (12 × 3840×2160): 12/12 registered, 5,375 points, mean reprojection
  0.51 px, fx ≈ 5,700 px (37° horizontal FOV), 4×3 grid at ≈ 700 mm spacing once scaled.
- **Known limit**: coverage's `up_world` is the mean camera −y axis. The array cameras are mounted
  portrait, so that axis is horizontal and the azimuth/elevation table (and any `hs move` preset
  built on it) is rotated 90°. Fine for train/archive/views; fix before relying on move presets.

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

| view | az | el | edge energy kept | PSNR | corr | displaced > 4 px | p90 |
|---|---|---|---|---|---|---|---|
| cap005 | +1° | −8° | 60% | 35.09 dB | 0.995 | 0% | 1.86 px (0.27 mm) |
| cap039 | −7° | +25° | 69% | 29.59 dB | 0.980 | 1% | 1.28 px (0.14 mm) |
| cap015 | −42° | −0° | 61% | 29.28 dB | 0.992 | 3% | 1.60 px (0.23 mm) |
| cap055 | +43° | −3° | 49% | 28.09 dB | 0.974 | **24%** | **10.44 px (1.46 mm)** |

Displacement separates the extremes by 8× while edge energy differs by 1.24×, which is what
identified the soft right side of rig6_boom as thin coverage and weak registration rather
than the motion blur it resembled — the source frames at ±42° are equally sharp (0.037 and
0.036 on a scale-normalised subject crop). The highest view scores best of the four, so a
fast pass is not automatically a bad one; what failed at az +43° was density, 5 captures
within 8° spread over 18° of elevation against 9 within an 8° band at az −42°. The displaced
fraction is diluted by background inside the crop, so compare views within a capture rather
than across differently framed runs.

## Keeping the Mac awake (2026-09-16)

Every project stage and `hs selftest` holds `caffeinate -d -i -m -s -w <hs pid>` for its whole
run (`hs/keepawake.py`); a `keep_awake` metric at the start reports AC/battery and whether the
lid is closed. `HS_CAFFEINATE_FLAGS` overrides the flags; `HS_NO_CAFFEINATE=1` or
`hs train --no-caffeinate` turns it off. caffeinate cannot stop a closed lid (no external
display), Apple menu > Sleep or a battery shutdown, so `runner.py` also measures sleep (a clock
that runs through sleep vs one that does not): any gap over 20 s is a live `sleep_detected`
metric and a line in `logs/<stage>.log`, and `hs train` records `wall_s`, `slept_s` and the
`no_sleep_during_run` check (fails above 60 s). Why: the 2026-09-15 head train had
`caffeinate -i` around Brush only and took ~16 h for ~1.5 h of work, training ~30 s per
~16 min overnight.

## Framing a person and grading (2026-09-16)

`move/subject.json` holds the subject's crown (`{"crown_mm": [x, y, z]}`, solve coordinates).
`hs move ... --headroom-mm 25.4 [--frame-aspect 2.35]` then solves the aim height, straight below
the crown, so a full-width crop of that aspect keeps the crown 25.4 mm (one inch) below its top
edge on every frame while cutting as much ceiling as the aspect allows; the per-frame crown row
goes to `move/<name>_frame.json` and the `headroom_fits_every_frame` check reports the range.
On the 2026-09-15 body arc the fixed 220 px crop cut the head off entirely (the arc pushes in to
728 mm); framed, the crop row follows the head from 102 to 259 px down.

`hs grade -p P --move NAME` applies lift / gamma / gain (master plus per-channel), contrast-adaptive
sharpening and that head-following crop (ffmpeg `lutrgb`, `cas`, `sendcmd` + `crop`) to
`render/NAME_1920.mp4` → `render/NAME_graded.mp4`, and keeps the settings in `grade/NAME.json`.
`--still N` writes one cropped frame for the app, which previews the grade with the identical
256-entry table (`out = clip(gain·x + lift·(1−x), 0, 1)^(1/gamma)` on code values; tested equal to
lutrgb). Note for ffmpeg ≥ 8: `selectivecolor` values must be quoted and space-separated.

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


## Exposure, masks, and the .hsmove cue language (2026-09-14, coin capture)

Three additions, each from a measured failure on `Projects/2026-09-13_coins` (70 captures of
four challenge coins in a clear plastic case, shot against a backlit sliding door).

### `hs exposure`

The phone's auto-exposure walks during a capture, and the walk tracks camera position: on this
set the 140 undistorted views spanned **2.49x in linear luma**, 2.74x in contrast and 8% in
R/B ratio, with the five brightest at cap019–024 and the five darkest at cap064–069. Brush has
no per-image exposure or appearance parameter — there is nothing for it in `ProcessConfig`,
`TrainConfig`, `ModelConfig` or `LoadDatasetConfig` — so a brightness that varies with camera
position can only be explained as a property of the object, and the spherical harmonics absorb
it as shading.

This decodes sRGB to linear, takes each channel's median per view, and scales it onto the
dataset's own reference (the median of those medians). Medians, not means, so a blown window
cannot drag the gain. It runs after `hs solve` and touches nothing the solver produced — the
sparse model, poses and rig.npz are computed from the original frames — so a train before and
after differs in one variable. Originals go to `solve/exposure_backup/`; `--restore` puts them
back. 2.49x → **1.03x** on this set, gains 0.76–2.08x, 11 s for 140 views.

Result on the golden four views: worst displaced fraction 40.5% → 28.3%,
`edge_energy_consistent_across_views` flipped to pass. PSNR is *not* comparable across the two
trains — the reference images changed too.

`hs exposure` **refuses an array project** (`source.kind == "array"`) unless `--force`:
median-matching is a correction only when every view frames the same thing. On the 2026-04-03
take 067 the A-column cameras fill the frame with the lit cyc (median linear luma 0.08–0.15) and
the D column with black drape (0.008) — a 19× spread that is framing, not exposure, and matching
it would darken the A column ~7× and brighten the D column ~2×. `--dry-run` always measures.
Matching an array's exposure properly means using a shared neutral target (the gray sphere, gray
card or Macbeth in the frame); that is not built.

### `hs masks`

`3DGS_4DGS_Challenging_Materials_Guide.docx` §1: "duplicate or ghosted objects through glass →
straight-ray model fits incompatible refracted correspondences → mask glass and retrain
background". Masking is the one thing that guide recommends which Brush actually has: its
loader finds a `masks/` tree mirroring `images/`, matches by file stem, takes white as keep and
selects `AlphaMode::Masked` automatically; `match_alpha_weight` then puts an L1 on rendered
alpha, so the model is pushed to be *empty* outside the silhouette rather than merely
unsupervised there.

No segmentation model is needed — the solve already knows where the subject is. Splats (or SfM
points) within `--radius` of the subject centre are projected into every view at that view's own
K, R, t, each drawn as a disc the size of its own projected footprint; then close, fill interior
holes, keep the largest connected component (detached blobs are haze and table caught by the
radius), and dilate by a world-space margin so the silhouette errs outward. A generous mask
costs background supervision; a tight one deletes real observations.

Verified with a 34-second control — same 1000 iterations, masks the only difference:

| | splats | inside 120 mm | beyond 1 m |
|---|---|---|---|
| initialization | 14,033 | 82.0% | 14.8% |
| masked, 1000 iters | 21,795 | 81.5% | 10.4% |
| unmasked, 1000 iters | 24,712 | 67.8% | 18.9% |

Full run: **245,351 splats against 408,807 unmasked**, and `splat_count_in_range` passes for the
first time on this project. Displacement improved on five of seven graded views (cap007
18.8 → 5.8%, cap040 2.6 → 1.3%) and worsened on two (cap009, cap050). PSNR rose on six of seven.

The limit worth knowing: **a 2D mask constrains the silhouette, not depth.** A splat two metres
behind the subject that projects inside the silhouette is fully supervised, and a transparent
subject needs something back there to show through. 35.4% of the masked model still sits beyond
a metre — though only 30% of that population projects into any sampled view at all.

### `.hsmove` and `hs move --script`

Presets build a path through real camera *indices*. A script describes the shot the way it is
described on set, in the capture's own spherical coordinates:

```
start  az -15  el +16  dolly +0
boom   to el +1        speed 4>1
hold   0.4s
dolly  back 20         speed 2
arc    left 24         speed 2
hold   0.5s
arc    right 90        speed 3
```

Speed is tenths: 10/10 is 30°/s of orbit and 150 mm/s of dolly, and `4>1` ramps across the cue.
Every cue is walked one frame at a time against the hull; a cue that runs out of capture is
CLAMPED, keeps what it achieved, and the report says where and how much — on the coin set that
last arc gets 46° of the 90° asked for, which is the capture's longest continuous arc.

Two things the compiler had to get right that a 1°-quantised map gets wrong. The reachable
radii along one bearing are **not a single interval** — two cameras at different depths leave an
unreachable gap between them, and taking the outer bounds puts the camera in it. And the shell
radius, taken as "the radius closest to a real camera", steps every time the nearest camera
changes: up to 30 mm between consecutive frames, a 900 mm/s lurch. The shell is now a
Gaussian-weighted mean over nearby captures (σ 8°), and the radius track is smoothed by
alternating projection — smooth, clamp into the band, repeat — which is what guarantees the
hull. p90 speed 48 mm/s against rig6's boom peaking at 87.

`no_speed_spike` reports what smoothing cannot fix: where the band itself has a seam, the track
jumps it in one frame, and the check names the time and the frame rather than hiding it.
