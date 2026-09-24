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
changed; `hs solve` checks the two agree (`rig_size_matches_L_views`). On 2026-09-22
`rigcolmap.py` and `monocolmap.py` gained `--matcher` (default `exhaustive`, whose branch is
unchanged) and print `timing: <phase> N s` lines; see "Matching, estimates and ETAs" below.

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
  pairs.py      which image pairs the solve matches: exhaustive, or the rig-aware sequential list (pure)
  timing.py     the solve timing store, cost model and `estimate` event (pure)
  events.py     the one JSON-lines emitter (start/progress/metric/artifact/check/done/error)
  runner.py     subprocess runner: streams lines split on \r and \n, tees to logs/<stage>.log
  project.py    project folder + manifest.json (status per stage, argv, metrics, checks; stale propagation)
  calib.py      profile JSON <-> the stereocal npz rigcolmap.py reads; match keys
  coverage.py   azimuth / elevation / distance per capture from rig.npz; sweep-window and boom-key presets
  movescript.py the .hsmove cue language: boom / arc / dolly / hold in capture coordinates, clamped to the hull
  board.py      ChArUco detection, DLT triangulation from rig.npz, pair-median scale, board plane, white-paper samples
  lidar.py      phone LiDAR scans: PLY/OBJ/XYZ loading + units, subsample, Umeyama, trimmed ICP, global registration, up, ground, coverage
  splatweights.py the renderer's forward weights w = alpha*T per (splat, view, cell), in numpy — under split and prune --score
  overlay.py    sparse points on photograph + render; global / per-quadrant phase-correlation shift
  shotsheet.py  the delivery shot sheet: manifest dicts -> Markdown + HTML (pure; hs export writes it)
  stages/       ingest select solve scale exposure masks train move prune render views stability split export tools calibrate selftest
  <six vendored scripts>
```

## Stages

```
hs ingest  -p P --clip VID_..._2x1.h4v [--link]                       copy, MD5, ffprobe, validate 2x1 video, match the calibration profile
hs ingest  -p P --frames DIR | --r3d RDM_DIR --take 067 [--res 1]     frames source, select is marked done: array (one frame per camera; REDline renders the R3Ds)
           [--kind auto|mono|array]                                   or mono (one camera's frames, e.g. select_frames.py --mono picks) — see below
hs select  -p P [--residual 1.5 --max-gap 90 --end N --dry-run]      frames + selection.json + quality.json + thumbs/ + contact.jpg
           [--keyframes [--search-keyframes 2 --min-sharp-rel 0.6]] [--highlight-knee 0.85]   I-frames only; soft highlight knee on the written frames
hs solve   -p P [--matcher auto|sequential|exhaustive] [--overlap 15 --loop-stride 8]   prep → sfm --float-rig → export; per_image.json, coverage.json
hs solve   -p P --estimate | --estimate --captures N [--eyes 1|2]     one `estimate` event: pairs and seconds per phase for both matchers (no lock, no writes)
hs solve   -p P --scale-pair GA,GB,700 [--focal-px F] [--board B]     array/mono project: monocolmap.py, one shared camera, metric scale from a measured spacing (or --board: hs scale after)
hs scale   -p P --board SX,SY,SQ_MM,MK_MM[,DICT] [--min-views 3] [--eye L] [--dry-run]   metric scale from a ChArUco board in the views, applied to the solve; board plane vs up
hs scale   -p P --lidar SCAN.ply [--units m|mm|cm] [--scan-up auto|y|z] [--pairs 'sx,sy,sz=px,py,pz;…'] [--init auto|pairs] [--apply|--dry-run] [--trust-scan]
                                                                      a phone LiDAR scan aligned to the solve: mono/array scale applied, stereo scale checked (applied with --trust-scan); up, ground
hs scale   -p P --factor F [--note '…'] [--trust-scan]                 apply a scale factor you already have (solve units x F = mm)
hs exposure -p P [--reference median|auto|capNNN|board|checker] [--reference-view V] [--mode rgb|luma] [--restore] [--dry-run]   match every view to one reference (board/checker: a shared target, fine on an array)
hs masks   -p P [--radius 0.10] [--margin-mm 5] [--min-opacity 0.2]   per-view subject silhouettes for Brush's mask channel
hs train   -p P [--brush PATH]                                        brush → train/exports/export_NNNNN.ply   (Mac only)
hs move    -p P --preset sweep|boom|custom [--keys ...] [--name N]    move/N.json + move/N_aim_check.jpg
hs move    -p P --script shot.hsmove [--name N]                      compile a cue sheet against the captured hull (movescript.py)
hs prune   -p P [--radius 0.3]                                        prune/<export>_pruned_r03.ply
hs prune   -p P --score [--ply PLY] [--name N] [--cell 8]            prune/N_scores.npz + N_report.json — a report, prunes nothing
hs prune   -p P --floaters [--name N] [--min-importance-quantile 0.02] [--max-blame 0.25]   prune/N_nofloat.ply + N_floaters_only.ply
hs split   -p P [--ply PLY] [--masks vision|region|DIR] [--exclude L/cap064,…|@holdout] [--name N] [--cell 8] [--refine knn|none]
                                                                      split/N/{full_labelled,subject,background}.ply + report.json
hs render  -p P --move N [--ply PATH] [--width 2400] [--keep-frames] [--stability]  render/N_1920.mp4, N_1080x1350.mp4  (Mac only)
hs views   -p P [--captures 5,15,55|holdout] [--ply PATH]                     grade the model against the photographs  (Mac only)
hs cameras -p P --holdout N [--method fps|interval|azimuth] [--write]  choose hold-out captures by camera position -> solve/holdout.json
hs stability --frames DIR | --video MP4 [--k 1,7] [--backend dis|raft]  flow-warped temporal stability of a rendered move (no project needed)
hs export -p P [--ply PLY|--archive NAME] [--formats ply,spz,sog,html] [--min-opacity X] [--subject PLY] [--shot-sheet]   deliver/NAME/: PLY + web formats + shot sheet
hs tools   [--fetch-vocab-tree]                                       versions of python packages, brush, brush-path-render, ffmpeg, adb, node, splat-transform, vocab tree
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

## Keyframes only, and a highlight knee on the frames (2026-09-23)

CirclesSculpture (3 minutes, 5,880 frames, every angle) showed two things about the Hydrogen's
video that the selector did not know. **The codec, not motion, limits sharpness.** The clip is
H.264 Baseline at ~12 Mbit/s with a GOP of 30: one I-frame, then 29 P-frames, each coded as a
change from the one before. The Laplacian rises through every GOP — the P-frames' accumulated
compression artefacts read as detail — so "sharpest of the next 4" drifts to the most artefacted
frame; only 7 of 422 picks were I-frames. The shutter was 1/1000 s, so motion blur was never the
issue. `hs select --keyframes` makes only I-frames candidates. The I-frame indices come from
ffprobe (`-show_entries frame=pict_type -of csv=p=0`, one line per frame; Baseline has no
B-frames, so that is display order too), found the way `hs ingest` finds it (`--ffprobe`,
`HS_FFPROBE`); without ffprobe keyframe mode stops with that said. The parallax rule still decides
*when* a frame is due (the residual crossing, `--min-gap`, `--max-gap`); the pick is then the first
keyframe at or after the crossing that passes quality, looking at no more than
`--search-keyframes` (2: the first keyframe at or after the crossing, and the next). Quality is
clipping under `--max-clip`, focus (the contrast-normalised Laplacian the report's "soft" flag
uses) at least `--min-sharp-rel` (0.6, the "soft" threshold) of the running median over the picks,
and crushed shadows under the report's 10%. If none passes, the least bad is taken and its record
says why (`quality_fallback`). The first pick is a keyframe too; `--mono` works the same. Expect
gaps in whole GOPs (30, 60, …) and fewer, cleaner picks; `median_gap_in_range` then asks for at
most 2 GOPs, and `keyframes_used` checks every pick is an I-frame. In the default mode ffprobe runs
alongside the decode, best effort, so every run's report says how many picks happen to be
I-frames (`keyframes_picked` of `frames_selected`); the picks themselves do not change.

**Highlights at the ceiling.** The clip was ISO 250, 1/1000 — the lowest the phone offers — in
sun and shade: ~14% of pixels at ≥ 250 in some channel, 1.8% at 255. `hs exposure` is not the
tool: it moves whole views to a common median and would pull the sunny side down. `--highlight-knee
K` puts a soft knee on every written frame, per channel, in linear light: sRGB → linear, above K
`x → K + (1−K)(1 − e^−(x−K)/(1−K))`, back to sRGB, from a 256-entry table. It meets the identity with
slope 1 at K, keeps order, never reaches 1, and touches nothing below K; both eyes get the same
curve. K = 0.85 starts at code 238 and takes 255 to 249 (18 codes into 12 — neighbouring highlight
values can merge in 8 bits); 0.9 starts at 244 and takes 255 to 251. It lives in select because
select writes the photographs: solve, masks, train and views all read `select/frames`, so every
consumer sees the same pictures, and `hs views` compares renders against the knee'd frames the
model was trained on. It is not exposure matching (`hs exposure`) and not a look (`hs grade`), and
it cannot bring back what the sensor clipped. The selector's measurements (trace, per-pick clip
and sharpness) are of the source before the knee — the knee changes no pick — while the report's
right-eye columns are measured on the written frames, so `clip_R` reads near zero with a knee.
`highlight_knee_applied` in the checks says K and how many frames.

selection.json gains `keyframes` (`frames`, `total`, `gop_median`, `probe_frames`, `source`; null
when ffprobe could not list the types), `params.mode` (`parallax` | `keyframes`) and the new flags
in `params`; every pick gets `keyframe` (true/false, null unknown), keyframe picks also `crossing`,
`trigger` (`parallax` | `max-gap` | `untracked`) and, when nothing passed, `quality_fallback`.
quality.json gains a `keyframes` block (`mode`, `picks`, `picks_known`, `picks_keyframes`,
`keyframes_total`, `gop_median`, `quality_fallbacks`) and `keyframe` / `quality_fallback` per
frame; in keyframe mode only keyframes count as a "sharper frame nearby", and `stood_still` follows
the trigger rather than the gap (a gap includes the wait for the next I-frame). Metrics:
`selection_mode`, `keyframes_picked`, `keyframes_total`, `gop_median`, and in the new modes
`quality_fallbacks`, `highlight_knee`.

## Matching, estimates and ETAs (2026-09-22)

CirclesSculpture (422 stereo captures, 844 images) went into an exhaustive match of 355,746
pairs with no ETA on screen. Three changes.

**`--matcher exhaustive|sequential|auto`** on `hs solve` (default `auto`) and on both scripts
(default `exhaustive`, so a standalone run is what it always was). `auto` is sequential above
60 captures, exhaustive at or below. Sequential is built explicitly by `hs/pairs.py` from
`captures.json`'s order and fed to `pycolmap.match_image_pairs` (the list is written to
`solve/match_pairs.txt`):

- a window: capture i against i+1 … i+`--overlap` (15), every eye against every eye;
- every rig mate L_i–R_i;
- a loop pass: every `--loop-stride`-th capture (8) against every other one, all eyes, so the
  two ends of an orbit and a walk-around that comes back meet without a vocabulary tree.

COLMAP's own `SequentialPairingOptions` is not used: it orders images by name, so over `L/` and
`R/` it pairs `L/cap421` with `R/cap000` and never pairs a capture's two eyes (checked on pycolmap
4.2.0: 7 captures, overlap 2, 25 pairs, no L_i–R_i). On the synthetic 7-capture set COLMAP matched
exactly the 63 pairs of the list. `--loop vocab` replaces the stride pass with
`pycolmap.match_vocabtree` over a vocabulary tree (`--vocab-tree`, `HS_VOCAB_TREE`, or the file
`hs tools --fetch-vocab-tree` downloads: COLMAP's FAISS tree `vocab_tree_faiss_flickr100K_words256K.bin`
from its 3.11.1 release, sha256-checked, into `~/Library/Caches/HydrogenSplat/`). That path is
wired and **not exercised**: GitHub is unreachable from the container, so neither the download nor
a vocab-tree match has run. The stride pass needs nothing and is the default.

| captures (stereo) | images | exhaustive pairs | sequential pairs |
|---:|---:|---:|---:|
| 60 | 120 | 7,140 | (auto: exhaustive) |
| 150 | 300 | 44,850 | 9,282 |
| 422 | 844 | 355,746 | 30,566 |

Sequential is right for orbits and walk-arounds. Use exhaustive when the capture revisits a view
from far apart in time and neither the window nor a stride capture would pair the two visits.

**The exhaustive block count.** pycolmap 4.2.0 logs `Processing block [i/ni, j/nj]` for **every**
one of the ni×nj blocks, not only i ≤ j: each block holds about half its 50×50 pairs and together
they are each pair once (reproduced on synthetic images, and replayed in `tests/test_pairs.py`).
So solve's matching progress total of ni·nj was right and stays, and CirclesSculpture's 88–159 s
per block is 1,231 pairs per block: 0.07–0.13 s per pair, 289 blocks, 7–13 hours (about 10), not
the 4–5 hours that 153 blocks would take. Counting the block lines in that run's `logs/solve.log`
(a `[2/17, 1/17]` line settles it) confirms this on the Mac.

**`hs solve --estimate`** takes no lock, constructs no Project (which would reconcile the
manifest), opens no events log and prints one line:

```
hs solve -p P --estimate                   # captures from select/frames (1 image each on array/mono)
hs solve --estimate --captures 150         # no project needed: what the Frames page plans
{"ev":"estimate","stage":"solve","captures":150,"images_per_capture":2,"auto":"sequential",
 "calibration":"defaults","runs":0,"matchers":{
   "exhaustive":{"images":300,"pairs":44850,"seconds":{"features":180,"matching":4485,"mapping":2598,"export":90,"total":7353}},
   "sequential":{"images":300,"pairs":9282,"seconds":{"features":180,"matching":928,"mapping":2598,"export":90,"total":3796},"overlap":15,"loop_stride":8}}}
```

The cost model (`hs/timing.py`): features = a·images, matching = b·pairs, mapping = c·images^p,
export = d·images, each fitted by least squares through the origin from the runs that measured
that phase, p fitted in log–log from ≥ 3 runs spanning ≥ 1.5× in images (clamped 1.0–2.5), else
1.5. The scripts print `timing: features|matching|mapping N s`; `hs solve` records `matcher`,
`num_pairs`, `features_s`, `matching_s`, `mapping_s`, `export_s` and appends a completed run to
`~/Library/Application Support/HydrogenSplat/timing.json` (`HS_TIMING_FILE` overrides;
`~/.hydrogensplat/timing.json` off macOS). A `--reuse-matches` run records mapping and export only.
Until the store has a run, `calibration` is `"defaults"`: matching 0.10 s/pair (the measured block
times above), and **guesses** for the rest — features 0.6 s/image, mapping 0.5·images^1.5 s,
export 0.3 s/image. The mapping guess dominates the sequential estimate (3.4 h of its 4.5 h at 422
captures) and is the number most likely to be wrong; one real run replaces it.

**ETA on every progress event.** `events.progress` fills `eta_s` whenever a stage gives a `total`
and no ETA: the mean rate since the step's first sample (every call is sampled, even ones the
0.25 s rate limit drops), reset by each `start` of the stage. The first step after a `start` is
anchored at the start time with done 0, because most loops report `i + 1` after item i; later
steps under the same `start` measure from their own first sample. A `done` that goes backwards
restarts the measurement. Audit of the stages that report done/total:

| stage · step | ETA from |
|---|---|
| train · train | its own iteration rate (unchanged); auto until the first rate exists |
| render · render | its own frame rate (unchanged); render · encode, auto |
| ingest · pull | its own byte rate (unchanged); ingest · transcode (R3D), auto |
| solve · mapping | the cost model, then the run's registration curve elapsed·((n/k)^p − 1) once a quarter of the images are in (`timing.mapping_eta`): incremental mapping slows as it grows, so a linear rate would promise too early |
| solve · features, matching, export; select; masks · project/segment; views · render; split / prune --score · weights; stability; export · convert; exposure · measure/apply; scale · detect; merge · write; grade · encode | auto |
| select without `nb_frames` in the probe | none: there is no total |

`hs train|views|render --estimate` (iterations, captures or frames × a measured rate) are not
built: they need those stages to record their rates in the store first.

## Scale from a board, exposure on a target, mono as its own source (2026-09-21)

```
hs scale    -p P --board 7,5,40,30                  # squares across, down, square mm, marker mm [,DICT_5X5_100]
hs scale    -p P --board 7,5,40,30 --dry-run        # measure only; on a Hydrogen project: the baseline's error
hs solve    -p P --board 7,5,40,30                  # the same, straight after an array/mono solve
hs exposure -p P --reference board                  # the board hs scale used; --reference-view GA to pick the view
hs exposure -p P --reference checker                # a ColorChecker Classic's white patch (cv2.mcc)
hs ingest   -p P --frames picks/                    # selNNN-FFFFF.jpg → source.kind "mono"; GA.png … → "array"
```

**`hs scale`** (stage between solve and train, optional — train requires solve only). Detects
the ChArUco board (`cv2.aruco.CharucoDetector`) in the undistorted training images, triangulates
every corner id seen in ≥ `--min-views` views by linear DLT from rig.npz's own K, R, t, and takes
the scale as the **median over every corner pair** of printed mm / reconstructed distance
(`hs/board.py`). Checks: the pairs' MAD ≤ 0.5 % of the scale (`board_pair_ratios_agree`) and each
view's reprojection RMS of the triangulated corners ≤ 2 px (`board_corners_reproject`); a Umeyama
fit of the board is reported beside it as a cross-check. The board's plane normal, oriented
towards the cameras, is reported with its angle to coverage's up (`board_normal_to_up_deg`) — the
up a board lying on the table implies; nothing is rotated yet. Applying the factor is the
`monocolmap.py --scale` path after the fact: one `pycolmap.Sim3d(s)` on `train/dataset/sparse`
(and `solve/sparse/rig`), then `monocolmap.finish_dataset` — the export's own writer — rewrites the
text model, points3D.ply and rig.npz, `coverage.write` rewrites `solve/coverage.json`, and solve's
`scene_scaled` check becomes ok with `source: board` (`hs masks` and the app read it). A
similarity moves no pixel, so exposure and masks stay current; train, move, prune, render and
views go stale. A second run measures ≈ 1.0. A stereo rig.npz is refused (the calibrated
baseline is its scale); `--dry-run` on one reports `implied_baseline_mm`. Synthetic test: 5 views
of a generated 7×5 board warped in by homography with 2 DN noise: scale within 0.03 %, normal
within 0.1°, corner reprojection RMS ≈ 0.4 px.

### `hs scale --lidar`: a phone LiDAR scan as the metric reference (2026-09-22)

```
hs scale -p P --lidar room.ply                        # mono/array: align, apply the scan's scale (as a board would)
hs scale -p P --lidar room.ply --dry-run              # measure only; scale/lidar_report.json + scale/lidar_aligned.ply
hs scale -p P --lidar room.ply                        # stereo: never applied — reports scale_ratio, implied_baseline_mm
hs scale -p P --lidar room.obj --units cm --scan-up z # a Z-up export in centimetres
hs scale -p P --lidar room.ply --pairs '2.61,0.50,1.79=812.4,-310.2,955.0;1.30,0.60,2.80=...;0.02,2.31,0.41=...'
```

```
hs scale -p P --lidar room.ply --trust-scan            # stereo: apply the scan's scale after all (the baseline was wrong)
hs scale -p P --factor 5.8 --trust-scan --note 'ring silhouette fit'   # a factor found outside the stage
```

```
hs scale -p P --lidar scan.ply --init silhouette --dry-run      # the subject's outline fixes the scale (needs hs masks)
hs scale -p P --lidar scan.ply --init silhouette --init-points  # ... and write scale/lidar_init.ply
hs train -p P --init lidar                                      # start Brush from the scan instead of the SfM points
```

Exactly one of `--board` / `--lidar` / `--factor`. The scan (Polycam / Scaniverse from an iPhone Pro, scanned
before the shoot) is registered to rig.npz's sparse points **solve → scan**: the solve's points
are a subset of what the scan saw, so every one has a surface under it and the scan's extra
rooms need no explaining, and the fitted scale is then exactly the factor that turns the solve's
units into mm. It feeds the board's apply path unchanged (`_apply_and_mark`: Sim3d on the model,
`finish_dataset`, coverage.json, `scene_scaled` ok with `source: lidar`).

1. **Load** (`lidar.load_scan`): PLY ascii / binary little- or big-endian with any vertex property
   order and type (x/y/z found by name, float or double; colour from red/green/blue, r/g/b,
   diffuse_* — uchar, ushort or float 0..1 — or a splat PLY's f_dc_*; normals nx/ny/nz; other
   elements before or after the vertices skipped, faces counted), OBJ `v` lines (with
   `v x y z r g b` colour), XYZ/CSV/TXT/PTS (delimiter and header guessed; a CloudCompare `//X,Y,Z`
   header or a PTS count line understood). Non-finite rows dropped. **Units**: `--units`, else a
   header comment naming one (`units: meters`), else the extent: a largest robust side ≤ 150 is
   metres, above is mm (a room is 2–20 m: 2–20 or 2,000–20,000); cm is never guessed. An inferred
   unit outside the confident bands (0.3–60 m, 1.5–60 m in mm) fails `lidar_units_known`
   (needs_human).
2. **Subsample** (voxel, seeded): the solve is first cleaned of strays (`lidar.denoise`: mean
   distance to 8 neighbours > 3× the median) — a voxel subsample keeps every isolated stray and
   thins the surfaces, so 20 % strays became 43 % of the registration's points before this. 5,000
   points a side for the global search; ICP runs the solve's 20,000 against the scan's 400,000.
3. **Initial alignment**. `--init auto` (default): `lidar.global_register`, a RANSAC over
   congruent triples. A well-separated solve triple on locally planar surface is sampled (rare
   normal directions preferred — a room is mostly floor and walls, three planes that fit a corner
   turned 120° as well as the right one; the furniture decides); its first edge is looked up in
   a hash of 1,200 scan points' pairs keyed by three scale-invariant angles (each PCA normal to
   the edge, normal to normal), which with the first normal gives a pose per candidate and sign
   (the edge's length ratio is the scale; with the scale known the lengths must agree within
   6 %); the third point must land on the scan with the triple's distance ratios and normal;
   Umeyama on the triple; a voxel distance grid prefilters thousands of poses per sample, the
   best 12 get a short trimmed ICP ranked by truncated-quadratic cost; it stops once the best
   pose has been found from two different triples (at least 15 samples), and the best four are
   refined against the dense scan. `--pairs` (≥ 3, scan units = solve mm) is Umeyama instead.
4. **ICP**: trimmed point-to-point (the best 70 % of correspondences — less when the measured
   overlap is smaller — plus every one within `--inlier-mm`), Sim(3), scale bounded about its start. The
   "plus" matters: scaling about the corner where floor and walls meet moves no wall point off its
   plane, so a few percent off it is the furniture that is the worst 30 % — a pure trim throws
   away exactly the correspondences that pull the scale back. Progress events carry a total.
5. **Stereo**: the same Sim(3) fit; its scale is `scale_ratio` (the factor the solve is off by
   — 1.0 = the baseline is right; the same sense as the board's factor) and
   `implied_baseline_mm = profile_baseline × ratio`. Nothing is applied and the stage's status
   is left alone unless `--trust-scan`, which applies it through the rig route's own writer.

**Honest failure.** A scan that does not align, or aligns ambiguously, exits 1 with an error and
a hint (`--units`, `--pairs`, coverage) and still writes the report; the project is untouched
(`stages.scale.lidar_check` records the attempt). Not aligned means any of: the global search's
best pose has < `--min-inlier` of the solve on the scan; another *different* pose (> 3° / 3 % away)
fits within 1.5× its cost (**ambiguous** — a bare corner is symmetric under 120° turns); or the
final ICP has < `--min-inlier` (0.5) within `--inlier-mm` (50) or a trimmed RMS > `--max-rms-mm`
(30). Separately, `lidar_geometry_constrains` measures what the overlap can observe
(`lidar.constraints`: the 7×7 point-to-plane information matrix at the solution — the Schur
complement of the scale, and the smallest eigenvalue of the pose block). Floor and two walls fit
at any scale about their corner: sensitivity 0 for that shape, 0.02–0.06 for the synthetic room
seen only near its corner, 0.23 for the room with its furniture. Below `--min-scale-sensitivity`
(0.1) a mono scale is not applied (exit 1); on stereo it is a needs_human check.

**Checks**: `lidar_aligned`, `lidar_geometry_constrains`, `lidar_scale_agrees` (stereo: ratio
within 2 %, needs_human otherwise), `lidar_covers_captures` (every camera centre within 3 m of
the scan and the median distance to the scan of the sparse points in its frustum ≤ 50 mm; names
the captures that fail), `lidar_up_agrees` (the scan's +Y — or `--scan-up z` — in the solve frame
within 10° of coverage's mean-camera up, needs_human otherwise), `lidar_units_known`.

**Written**: `scale/lidar_report.json` — scan metadata (format, vertex/face counts, properties,
comments, units and their source, extent), subsample sizes and strays removed, the init (global:
samples, hypotheses, confirmations, alternatives, seconds; pairs: residuals), ICP per-iteration
trimmed RMS, final RMS and inlier fraction, solve → scan (s, R, t), the scale factor or ratio and
implied baseline, the constraints, up vector and angle, the ground plane (RANSAC on the lowest
150 mm band of the scan, with camera heights above it), the per-capture coverage rows, and
`transform`: 4×4 `scan_mm_to_solve` and `scan_file_to_solve` (the file's own units) into the solve
frame — after scaling when applied, the current units on a dry run. `scale/lidar_aligned.ply`: a
100,000-point subsample of the scan in that frame, coloured when the scan was, for the viewer.
Applied runs also put `up_world`, `ground_normal_world` and the transform in `manifest.scale`
(source `lidar`); nothing is rotated yet — coverage's up is still the mean camera up (which is
wrong on the portrait array; the scan's is not).

**Synthetic results** (`tests/lidar_synth.py`: a 5 × 4 m room, 2.6 m walls, a box turned 30°, a
ball, a column, ~280k scan points in metres; the "solve" 12,000 of them from the 60 % the cameras
covered, 3 mm noise, a random Sim(3) with scale 0.7–1.4, stereo 1.0; 2-core container):

| | result |
|---|---|
| global registration alone, 24 random scenes | 24/24 within 0.3 % / 0.3°; worst 0.18 % scale, 0.02° |
| global registration time, 5,000 × 5,000 points | median 3.1 s, 1.9–6.8 s (to failure on no overlap: 5–9 s) |
| ICP after it (12,000 vs 280,000) | trimmed RMS 3.6 mm, scale 0.001–0.005 %, rotation < 0.01° |
| three hand-picked `--pairs` with 10 mm error | start 0–1.2 % / 0.4–1.6° off; ICP converges to the same pose (< 0.05 %) |
| 10 mm noise; 20 % strays; 3,000 points | 6/6, 12/12, 6/6 within 0.3 % / 0.3° |
| solve sees only the corner (30 % coverage) | 6/6 refused (ambiguous, or scale not fixed) — never a silent 120° / 25 % answer |
| stereo, the profile baseline 3 % long | scale_ratio 0.9709 (true 1/1.03 = 0.9709), implied baseline 10.600 mm (true 10.600), needs_human |
| mono stage end to end (`hs scale --lidar`, 4 s) | scale 0.826438 against 0.826446 (0.001 %); every camera spacing in mm within 0.001 % |

**The first real export (Scaniverse Classic, Mesh capture, Share → Export → PLY, iPhone 17 Pro
Max, CirclesSculpture 2026-09-23):** binary little-endian, 733,944 vertices of float `x y z` +
uchar `red green blue`, no faces, no normals, one comment (`Created with Scaniverse`), metres
(extent 10.8 × 8.6 × 2.6 m, read as m from the extent), 10 mm point spacing — and **+Z up**, not
ARKit's +Y. So `--scan-up` defaults to `auto`: `lidar.detect_up` takes the axis with the
smallest robust extent (a scanned world is wider than tall) whose lowest band is a plane lying
along it (a floor); Polycam's +Y and Scaniverse's +Z both come out right, a scan that cannot
say (a stairwell, no floor) exits with a hint to pass `--scan-up y|z`. A Scaniverse splat PLY
reads as its centres with DC colour — usable, floaters and all. LAS/LAZ/E57/USDZ/GLB are
refused with a hint to export PLY, OBJ or XYZ. A header comment is the only declared unit
understood; PLY has no standard one.

**What the first real registration taught (Circles, H1 stereo, 373 captures at 3–5 m from a
2 m glossy red ring on a lawn).** The scan was fine; the solve was not what the design assumed.
(1) The solve's sparse cloud was the lawn (a plane) and the trees at 20–30 m (past the LiDAR's
range): the sculpture itself, glossy and uniform, contributed almost no points — so the overlap
was a plane, which fixes no scale, and the registration refused, ambiguous, as designed. (2) The
solve was **5.8x too small**: the profile's 10.642 mm baseline puts ~5 px of disparity on a
subject at 4 m, the floating rig's 1.47° rotation shift is ~44 px, and after `--float-rig`'s
realignment to the profile baseline the whole scene shrank (the reconstruction behaves as if
the eyes were 62 mm apart). A stereo SE(3) search could only fail there; the stage now fits
Sim(3) on every source and reports the ratio, `--trust-scan` applies it, and `--factor` applies
a number found elsewhere — the 5.8 came from a ring-silhouette fit: the scan's red points
projected through the solve's cameras against the red mask of 14 photographs, gravity locked,
Sim(3) by Nelder–Mead (IoU 0.49; scale 5.81 with tilt fixed, 5.69 free). (3) Outdoors the
"solve ⊂ scan" premise fails the other way — the SfM sees further than the LiDAR — so the
registration keeps only solve points within 2x the camera path's extent of a camera, subsamples
them at random rather than by voxel (a voxel grid over a 70 m cloud was 98 % trees), trims ICP
to the measured overlap, reads the verdict over the points the scan contains (with a floor of
2,000 / 5 %), and scales the tolerances with the camera–subject distance (inlier 2 %, RMS 1.5 %,
per-frustum 2 %; indoors the flags' 50 / 30 / 50 mm hold). With `--pairs` from the silhouette
fit Circles then aligns to 55 % within 91 mm, RMS 92 mm — and ICP still slides the scale 4.6–5.4
with the ring absent from the sparse cloud, so the number to apply is the silhouette's, by
`--factor`. The lesson for the H1: its baseline gives a trustworthy scale only within a metre
or so (rig6's bust at 0.3 m: 60 px of disparity); anything further needs the scan or a board.
That fit is now in the stage: `--init silhouette`, below.

**`--init silhouette`: the scale from the subject's outline (2026-09-23).** For a scene whose
sparse cloud does not hold the subject (glossy, uniform, thin), `hs/silhouette.py`:

1. **The scan's subject**: the scan's ground plane (`lidar.ground_plane` along its up), the points
   more than `--subject-above-mm` (150) above it, and the largest 26-connected cluster of them on a
   100 mm voxel grid (`scipy.ndimage.label`). A plinth or a rolling lawn joins the subject at the
   first cut — Circles at 150 mm was a 6.2 × 5.7 m "subject" of lawn humps and plinth, and the fit
   on it gave 6.45 at IoU 0.37 — so the cut rises (1, 1.5, 2, 2.5, 3, 4, 5, 6 × the flag) while the
   cluster's footprint (occupied voxel columns seen from above) keeps shrinking, and stops at the
   first cut whose next one keeps ≥ 85 % of it: what is left stands up rather than lies on the
   ground (Circles: 2154, 1492, 307, 219, 217 columns at 150–450 mm → cut 375 mm, the ring, 132,917
   points, 3.0 × 2.1 × 0.6 m). Never past half the cluster's height; a box stops at the first cut.
   The subject must be the largest thing standing in the scan: indoors the walls are, and win.
2. **The masks**: `train/dataset/masks/{eye}/capNNN.png` (`--eye`, default L) of the views with an
   image and a mask, outlier cameras skipped (hs solve's rule, > 10× the median distance from the
   median centre), evenly spaced down to `--silhouette-views` (24); fewer than 6 is refused with a
   hint to run `hs masks`. Read at 1/4 resolution, nearest.
3. **The fit** (the ring fit of 2026-09-23, cleaned up): gravity locked — the scan's up onto
   coverage's `up_world` — so yaw, log-scale and translation; the scan subject's centroid starts on
   the solve's subject guess (`subject_mm`, the median sparse point, recomputed from the current
   rig). Cost 1 − mean IoU of the projected points (6,000, a seeded voxel subsample; rasterised at
   1/4, dilated 2 px) against the masks. A coarse grid over yaw (4°) × 12 log-spaced scales — 0.5–10
   on a nominally metric solve (stereo, or scaled already), 0.25–4 × an apparent-size prior on an
   unscaled one (mask box against the subject's size at the camera distance) — then Nelder–Mead,
   then Nelder–Mead again over the views with IoU ≥ 0.3. `--silhouette-tilt` frees two small tilts;
   off, because on Circles a free tilt overfitted the masks and put the cameras through the lawn.
4. **ICP, the scale locked**: the reported scale is the silhouettes'. By default
   (`--silhouette-icp vertical`, `lidar.icp_vertical`) ICP moves only height and the two tilts —
   what a floor or lawn observes — point-to-plane, over the solve points within 10 × the inlier
   distance of the scan (the far ones matched nothing and their exact kd-tree queries were 2 s of
   every iteration). `se3` is the rigid ICP. `none` (or `--icp-iters 0`) keeps the silhouettes'
   pose. **The subject decides**: when the ICP's move costs the silhouettes more than 10 % of their
   IoU or tilts the solve more than 3°, the pose reverts to the silhouettes' and the ICP's move is
   only reported — on Circles the lawn ICP lowered the cameras 284 mm and tilted the solve 8.1°
   (IoU 0.45 → 0.37, and in the overlays the cyan ring sits above the red one while the yellow one
   is on it): the sparse lawn is a bowl (SfM drift), the ring is where the cameras see it, and an
   init cloud has to put the subject there.
5. **Verdict**: `lidar_aligned` is the silhouette check plus the overlap floor (the trimmed RMS and
   the ground's inlier fraction are reported, not judged — on a lawn the RMS is grass, and the
   ground's fit is the cross-check below; the global search is not run). `lidar_silhouette_fits`:
   mean IoU ≥ `--min-iou` (0.35) over ≥ 6 views, at the final pose.
   `lidar_ground_agrees_with_silhouettes` (needs_human): ICP kept ≥ 90 % of the fit's IoU, tilted it
   ≤ 3°, and ≥ 50 % of the solve's on-scan points sit within the inlier distance — when not, the
   solve's floor and the subject's outline disagree (a bent solve, a drifted scan), the silhouettes'
   pose stands, and the overlays are the thing to look at. `lidar_geometry_constrains` is not
   emitted (the scale is not the geometry's). Metrics: `silhouette_scale`, `silhouette_yaw_deg`,
   `silhouette_iou_mean` (final) / `_fit`, `silhouette_views_used` / `_offered`,
   `scan_subject_points`, `scan_subject_extent_mm`, `scan_subject_cut_mm`,
   `silhouette_camera_height_mm` and `silhouette_walk_radius_mm` (p5, median, p95 over the
   non-outlier cameras, above the scan's ground / from the subject's axis). The report's `init` has
   the per-view IoU (first pass and final), the grid, the prior, the subject's cut levels, and
   `icp_moved` (tilt, and the cameras' mean move along the up); `silhouette` has the IoU after ICP
   and the heights at both poses. **`scale/silhouette_overlay_capNNN.jpg`** for 3 evenly spaced
   views used: the photograph, the mask's outline (magenta), the subject at the fit's pose
   (yellow) and at the final one (cyan) — the human check.

**Circles** (`hs scale --lidar scan.ply --dry-run --init silhouette --init-points`, 19 masked views —
stand-in red-threshold masks, which miss the ring's pale sunlit faces — 2-core container, 80 s):
scale **5.802** (the ring fit's 5.81), yaw 83.1°, IoU 0.446 over 13 views at the fit (cap000 and
cap180–cap275, the stretch next to the unregistered captures, dropped under 0.3), grid best 3.36
at 80° before Nelder–Mead. Vertical ICP, 19 iterations: 57.9 % of the solve's points on the scan
within 91 mm (lidar_aligned ok), but it tilted the solve 8.1° and moved the cameras −284 mm to put
the SfM lawn on the scanned one, and the silhouettes' IoU fell to 0.366 —
`lidar_ground_agrees_with_silhouettes` asks for a look. The overlays show the ring on the ring at
the fit's pose and higher in the frame after ICP (cap140: IoU 0.39 → 0.16; cap410 0.49 → 0.56, where
the fit's pose sat low on the plinth); the scan's glossy ring is a fuzzy cloud with a halo of
strays, which caps the IoU as much as the stand-in masks do. Every subject tried (the
cluster, the red points only, both) puts the cameras 1.73–1.77 m above the scan's ground at the
silhouettes' pose against 1.51 m after the lawn: the conflict is in the solve, not the subject.
Cameras at the final pose: 0.78 / 1.51 / 2.14 m above the ground (p5 / median / p95), 2.1 / 4.4 /
5.6 m from the ring's axis. The rigid `se3` ICP did no better (IoU 0.36, 48.5 % in-scan inliers —
not aligned — 100 iterations in 4 minutes); keeping the silhouettes' pose (`--icp-iters 0`) keeps
IoU 0.446 with 27 % of the lawn on the scan (not aligned). Refitting yaw, scale and position after
each lawn ICP was tried and dropped: with the height the lawn's, the silhouettes pay for it in
scale (6.44).

**Synthetic** (`tests/lidar_synth.YardScene`: an 8 × 8 m lawn, the subject a box with a ball and a
1.2 m column against it — box and ball alone are nearly the same silhouette turned half round,
IoU 0.85 there against 0.82 at the truth — a post as a distractor, a solve of the lawn only with
5 mm noise, 24 cameras at 1.5 m on a 3 m orbit, masks the subject's projected points closed): five
scenes, solve 5×, 2×, 2.7× too small and 2.3× too big, stereo and mono: scale within 0.6 %,
rotation within 0.7°, every camera within 47 mm, IoU 0.83–0.84, camera height 1.50 m; 12 s each.

**`--init-points`** (any `--init`): after an alignment the run stands behind (dry run or applied),
**`scale/lidar_init.ply`** — the scan as Brush initial splats (`hs/initsplats.py`), in exactly
Brush's export layout (`x y z scale_0..2 opacity rot_0..3 f_dc_0..2 f_rest_0..44`, float, binary
little-endian, comments `Exported from Brush` and `SH degree: 3` plus one saying who wrote it) and
in the training set's units: the report's `scan_mm_to_solve` / 1000 (COLMAP metres = rig.npz mm /
1000; after the scale when applied, the current units on a dry run). The scan voxel-subsampled to
`--init-points-max` (400,000), coloured from the scan (f_dc = (c/255 − 0.5)/0.28209479),
opacity logit(`--init-opacity`, 0.3), one isotropic scale = log(mean distance to the 3 nearest
neighbours) clamped to 0.002–0.1, rot (1, 0, 0, 0), f_rest 0; plus the solve's own sparse points
(denoised) further than 10 × the inlier distance from the scan, so the trees and the far lawn keep
their usual start — with the training set's COLMAP colours when `train/dataset/sparse` reads, else
mid-grey. Recorded with the file's md5 and the rig.npz md5 it was written for, in
`manifest.scale` (applied) or `stages.scale.lidar_check` (not); metrics `init_points_scan`,
`init_points_sparse`, `init_ply`. Circles: 400,000 + 199,825 splats, 142 MB.

**`hs train --init lidar`** stages that file as `init.ply` in the folder Brush trains on, the way
`--resume-from` stages a checkpoint: validated before train/exports is cleared (the newest record
whose file md5 is unchanged and whose rig.npz is the current one — a re-solve or a scale applied
since moved the frame under it, and is refused with the reason), md5-checked after the copy,
removed after the run; `--resume-from` with it is refused. `init` ("sparse", "lidar", "resume"),
`init_ply`, `init_ply_md5` and `init_splats` go in the train metrics, `init` and `init_md5` in
`dataset_fingerprint`, which `hs archive` copies whole. What it does to a model is untested here
(no GPU): the file is Brush's own layout, and the start is where the scan says the scene is.

**Dependency**: scipy (≥ 1.10) joins the engine's for `ndimage.label` (3-D connected components)
and `optimize.minimize` (Nelder–Mead); only `--init silhouette` imports it, and says to reinstall
the engine when it is missing.

**Limits.** ICP is point-to-point: it converges slowly along a room's weak scale direction (60
iterations from 1 % off; `--icp-iters` 100). The scale's accuracy on a real capture will be
bounded by the solve (SfM noise, a mono solve's own drift) and the scan (ARKit's few-mm to cm
LiDAR noise, drift over a large room), not by the registration. A scene that is only planes —
an empty room, a wall — cannot give a scale, and says so.

**`hs exposure --reference board|checker`**. The array refusal was right for a median and wrong
for a shared target: the same physical white is in every view. `board` samples the white paper of
the board — in a ChArUco board the white squares carry the markers, so the sample is the margin
between marker and square edge, inset 20 % from both — `checker` the interior of a ColorChecker's
white patch (mcc's own false positives on plain texture are rejected: the neutral row must fall
white to black). Per-channel gains bring each view's white onto the reference view's (default the
median one), through the same LUT, backup and `--restore`. A view without the target keeps gain
1.0 and is named in a failing `<target>_seen_in_every_view` check. `exposure_target` holds the
per-view gains and `white_rms_before/after` (relative RMS of the white's luma across views).
Synthetic test: five views with per-channel gains 0.80–1.20, JPEG q95: worst recovered gain 0.4 % off (bound 1 %), white spread 7.8 % → 0.13 % RMS.

**`source.kind = "mono"`**. `hs ingest --frames DIR` now says which of two things the frames are:
`array` (per-camera subfolders with one frame each, an R3D take, or a flat folder of camera-named
frames as before) or `mono` (one camera: `select_frames.py --mono` picks, a selection.json that
says mono, or any one numbered sequence ≥ 3 digits); `--kind` overrides the guess for a flat
folder, and `source.kind_why` records why. Both take the frames route (`Project.frames_route`:
select refused, solve through monocolmap.py, `source_kind` metric on solve). They differ where the
camera count matters: the median refusal in `hs exposure` is an array's only — a mono orbit is the
case the median was built for. A manifest without `kind` is a Hydrogen clip, and mono data
ingested earlier as `array` keeps working as an array.
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

**Partial registration and export-only (2026-09-22).** A solve that registers only some
captures fails by design — never train on a partial solve without saying so. Two flags say so:
`hs solve -p P --export-only --allow-partial` exports `solve/sparse/rig` from the earlier run as
it stands (no prep, no sfm, the earlier run's metrics kept), records `unregistered_captures`,
fails `partial_solve_accepted` with `needs_human`, and leaves the hole in the coverage for the
move editor to see. `--export-only` alone re-exports a complete solve. Two more checks come out
of the same failure: `registered_images_have_observations` (an image COLMAP posed with no
surviving 3D observation — L/cap256 on the 09-22 solve — has a pose nothing supports; exclude
it from train) and `no_outlier_camera_positions` (a camera centre more than 10× the median
distance from the rest; such a capture dragged that solve's path length to 24.6 km). Both name
the captures; `hs train --exclude` takes them.

## Evaluation trio: hold-outs, reproject overlay, temporal stability (2026-09-21)

**Hold-outs by position.** "Every 10th from 5" left the 09-16 head's +90…+135 band without a
hold-out. `hs cameras -p P --holdout 16 --write` picks by where the cameras are: `fps`
(default) is farthest-point sampling over the camera centres, seeded at the capture farthest
from their centroid; `azimuth` takes one capture per equal-width azimuth band; `interval` is the
old every-Nth (`--seed` shifts its phase). It prints the picks, nearest-neighbour spread among
hold-outs and to the training set, and hold-outs per 45° band (stderr; events on stdout), and
`--write` saves `solve/holdout.json`. Then `hs train -p P --exclude @holdout` (both eyes of each
capture on a stereo rig) and `hs views -p P --captures holdout` score exactly what was left out.
FPS reaches the ends of the orbit first, so its hold-outs include the coverage extremes: they are
extrapolations, which is what a move to the edge of the capture asks of the model too.

**Reproject overlay** — solve fault or training fault?

```
python3 engine/tools/reproject_overlay.py -p P --capture cap045 [--eye R] [--ply PLY | --render IMG]
hs views -p P --overlay            # the same for every scored view, on the renders views makes
```

The sparse points the image observes (rig.npz `pts` when the model has no tracks) are drawn on
the photograph at their reprojection with a red line to their keypoint, and on the render of the
same pose; phase correlation (×4 coarse, whole-pixel back-shift, fine residual; on the synthetic
test whole-pixel shifts come back exact and fractional ones within 0.06 px) gives `shift_px` and per-quadrant shifts in `report.json`.
Long keypoint lines = the solve is wrong. Short lines and a near-uniform render shift = the splats
or the pose used in training disagree with a solve that is fine. Without `--ply`/`--render` it
uses the frame `hs views --keep-frames` left for that pose, if any.

**Temporal stability** — does the model pop?

```
hs stability --frames render/boom/ | --video render/boom_1920.mp4 [--k 1,7] [-p P]
hs render -p P --move boom --stability          # on the PNGs before they are deleted
```

For each pair (t, t+k): DIS optical flow both ways, t+k warped onto t, occlusions masked by
forward–backward consistency (> 1 px), then warped PSNR / MSE, mean |Δ| (flicker) and the fraction
of pixels over 0.1 error ("popping") on the valid mask. Frames are scored by the mean over the
pairs they are in, so a one-frame pop is blamed on itself rather than a neighbour. Output:
`<name>_pairs.csv`, `<name>_stability.json`, `<name>_stability.png`; `hs render --stability` records
the numbers as `stability_*` render metrics. Frames are downscaled to 960 px wide first; compare
runs at the same width, k and backend. On the synthetic test (a textured square moving 2 px/frame,
320×240): steady 58.4 dB median at k = 1; a 6 px jump on one frame drops its pair to 42.6 dB
and names it worst; a one-frame 1.35× brightness flash drops it to 26.8 dB with 8% popping pixels.
A rigid jump is the weak case, because the flow tracks most of it; appearance pops are the strong
one. `--backend raft` uses torchvision's raft_small (`pip install -e 'engine[metrics]'`; torch is
imported only then, and it downloads its weights on first use); `hs tools` reports `flow_dis` and
`flow_raft`.

## Delivering a model: `hs export` (2026-09-21)

```
hs export -p P                                         # train's final export -> deliver/export_40000/
hs export -p P --archive masked-exposure --shot-sheet  # an archived model, with its provenance page
hs export -p P --ply prune/x_pruned_r03.ply --formats ply,spz --min-opacity 0.05 --name x
hs export -p P --archive head --subject split/1/subject.ply --shot-sheet
```

Writes `deliver/<name>/` (built beside and renamed into place, replaced whole on a re-run; the
name defaults to the archive name, else the ply's stem):

| file | what |
|---|---|
| `<name>.ply` | the source PLY byte for byte, SH intact — what Nuke and Houdini read |
| `<name>.spz`, `<name>.sog` | compressed web formats |
| `<name>.html` | a self-contained single-page viewer (SOG inside) |
| `<name>_subject.*` | the same set for `--subject` (a split subject layer) |
| `manifest.json` | source ply + md5 + splat count, every file's md5 / bytes / splats, the exact argv that made it |
| `shot-sheet.md`, `.html` | with `--shot-sheet` |

The web formats come from PlayCanvas **splat-transform** (`@playcanvas/splat-transform`, MIT),
one call per output in its own syntax `splat-transform [GLOBAL] input [ACTIONS] output`:

```
splat-transform -w in.ply out.spz
splat-transform -w in.ply -V opacity,gte,0.05 out.sog        # --min-opacity 0.05
```

It is found as `--splat-transform PATH` / `HS_SPLAT_TRANSFORM`, else `splat-transform` on
`PATH`, else `npx -y -- @playcanvas/splat-transform` (node ≥ 18; the first run downloads it).
Node is optional: `--formats ply` needs nothing. `hs tools` reports node, npx and splat-transform
(probing npx with `--no`, so it never downloads).

- `--min-opacity` is splat-transform's `-V/--filter-value` action, which compares **linear**
  opacity (0–1, after the sigmoid), not the PLY's stored logit. With a filter the delivered PLY
  is written by splat-transform too, so every format carries the same splats (`splats_exported`);
  without one the PLY is a copy and `model_ply_copy_matches_source` checks its md5.
- SOG and HTML compress on the GPU through WebGPU. On a machine without an adapter (the Linux
  container: "vkCreateInstance: Found no drivers") that fails; with no `--gpu` given the file is
  retried once with `-g cpu` and `model_sog_on_gpu` / `model_html_on_gpu` fail as a note. `--gpu
  N|cpu` is passed through and never retried.
- `--archive NAME` exports `archive/NAME/<ply>` and refuses it if its md5 no longer matches the
  archive manifest. A `--ply` inside an archive folder picks the manifest up the same way. Without
  either, `train` must be done (`REQUIRES["export"]`).
- The **shot sheet** (`hs/shotsheet.py`, a pure function of the manifest dicts) lists the project,
  the source (kind, clip md5 or frames digest), solve metrics, the model's provenance (archive,
  ply md5, splat count, layer, masks, train argv, Brush commit), the move (frames, fps, duration,
  keyframes from `move/<name>.keys.json` or the `hs move` record), the hold-out medians of every
  `views/*_report.json` — each marked "this model" only when it scored this very file — the grade
  look, and the export md5s. The archive's stage snapshot is preferred to the project's, since the
  project may have been re-solved since; a training record whose final-export md5 is not this ply's
  is flagged. The move is `--move`, else the latest render of this ply, else the only move in
  `move/`. Anything absent says "not recorded".

Checked against the real splat-transform v3.5.1 (c23730c) in the container: a 2,000-splat PLY
with `--min-opacity 0.5` delivered 966 splats in every format, exactly the count with
sigmoid(opacity) ≥ 0.5; spz 38 KB, sog 56 KB, html 3.2 MB (SOG and HTML via the cpu retry). The
tests run against `tests/fakebin/splat-transform` (`tests/fake_splat_transform.py`), which parses
the same argv and logs it (`HS_FAKE_ST_LOG`); `HS_FAKE_ST_NO_GPU=1` reproduces the GPU failure.

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
Matching an array's exposure properly means using a shared neutral target: `--reference board`
or `checker` (2026-09-21, above) do that and are not refused.

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
