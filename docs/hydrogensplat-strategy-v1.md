# HydrogenSplat — v1 strategy (2026-09-14)

Desktop macOS app that turns a RED Hydrogen One stereo video clip into a rendered Gaussian-splat camera move with as little human input as the pipeline allows. Decisions taken with Stephen before this was written: SwiftUI native app from day one; new repo `~/Desktop/Apps/HydrogenSplat`, under git from the first commit; video input only. The pipeline it automates is the one that produced rig6 — every stage below already runs by hand and is specified with measured numbers in `capture-to-splat-spec-v1.md`. The app's job is orchestration, checking, and the two decisions that still need a human eye.

## 1. What v1 is, and is not

v1 takes a clip off the phone and produces: a solved rig dataset, a trained `.ply`, a camera-move MP4 (16:9 and a 4:5 crop), with every intermediate kept in a project folder and every stage re-runnable. It supports static subjects only, one clip per project, one calibration profile (H1 video mode, 1920×1080 per eye, Holocam 1.18.2). It does not: handle the 4V stills format, do moving subjects, mask, run on anything but Apple Silicon macOS, or ship inside the App Store sandbox (it spawns adb, Python and GPU binaries and writes to user folders).

Automation does not raise the quality ceiling. The 1.5 px reprojection residual and the soft right side of rig6 are capture and solver questions, not app questions; the app makes the run reproducible and shows the numbers so those questions can be worked on.

## 2. Architecture — the one decision that matters

**The app is an orchestrator; the engine is subprocesses.** SwiftUI owns the project, the state machine, progress, and the two human checkpoints. Everything that computes runs as a child process with arguments in and JSON-lines events on stdout, and is testable from a terminal without the app. The app never links Python or Rust.

```
HydrogenSplat.app (SwiftUI)
 ├─ ProjectStore        manifest.json, stage state machine, resumability
 ├─ ToolRunner          Process + pipe reader → JSON-lines events → UI
 ├─ Stages              Ingest · Select · Solve · Train · Move · Render
 └─ Views               Setup · Projects · Stage pages · Aim check · Move designer · Result
engine/  (Python package `hs`, one CLI: `hs <stage> --project DIR [...]`)
 ├─ ingest.py     adb list/pull, ffprobe validate, thumbnails
 ├─ select.py     = mv/select_frames.py
 ├─ solve.py      = mv/rigcolmap.py prep+sfm+export, plus coverage table
 ├─ paths.py      = mv/spline_path.py + mv/key_path.py + auto-key presets
 ├─ prune.py      = mv/prune_splats.py
 ├─ calib.py      profile load/validate; stereocal wrapper (v1.1)
 └─ events.py     the one JSON-lines emitter every stage uses
tools/   brush, brush-path-render (from the Brush fork), ffmpeg, adb — paths configured in Setup, bundled later
```

Why this and not "SwiftUI calls into Python": the pipeline already exists as six Python scripts and two Rust binaries that work; wrapping them costs days, porting them costs weeks and re-introduces every bug already fixed. It also lets two people (or two Claude sessions) work in parallel: one on the engine CLI with the rig6 clip as a golden test, one on the app against a stub that replays recorded event logs.

**Event contract** (stdout, one JSON object per line; stderr is captured to `logs/<stage>.log` verbatim):

```
{"ev":"start","stage":"solve","step":"sfm"}
{"ev":"progress","stage":"train","done":23970,"total":40000,"rate":9.98,"eta_s":1605,"detail":"139455 splats"}
{"ev":"metric","stage":"solve","name":"mean_reproj_px","value":1.394}
{"ev":"artifact","stage":"select","path":"select/contact.jpg","kind":"image"}
{"ev":"check","stage":"solve","name":"all_frames_registered","ok":true,"value":"65/65"}
{"ev":"check","stage":"move","name":"aim_in_frame","ok":true,"value":"(959,552) in cap021_L at 230 mm","needs_human":true}
{"ev":"done","stage":"solve","exit":0}
{"ev":"error","stage":"train","message":"...","hint":"..."}
```

Brush and brush-path-render don't speak this; `hs train` and `hs render` wrap them, parse their stdout (Brush's spinner writes `NNNN/40000 Steps (r/s, remaining)` and `Current splat count N` with carriage returns — split on `\r` as well as `\n`) and watch the export folder as the ground truth for progress.

## 3. Project folder and manifest

One folder per capture. Everything the run touches lives inside it; nothing is written elsewhere.

```
~/HydrogenSplat/Projects/2026-09-12_figurine/
  manifest.json          state machine + metrics + tool versions + calibration profile id
  source/VID_..._2x1.h4v + .md5 + probe.json
  select/frames/VID_NNN_FFFF_2x1.jpg, selection.json, contact.jpg
  solve/images/{L,R}, sparse/rig, sfm_report.json, per_image.json, coverage.json
  train/dataset/ (undistorted PINHOLE + rig.npz), exports/export_40000.ply, train.log
  move/<name>.json, aim_check.jpg, coverage_plot.png
  render/<name>/frame_%04d.png (deleted after encode unless kept), <name>_1920.mp4, <name>_1080x1350.mp4
  logs/
```

`manifest.json` holds, per stage: `status` (pending / running / done / failed / stale), started/finished times, the exact argv used, metrics, and check results. A stage becomes `stale` when an upstream stage re-runs; the app offers to re-run downstream. Re-running a stage deletes only that stage's folder. This is what makes the app resumable across sleep, crashes and quit — training is ~55 min and the Mac will be closed on it.

## 4. Stages

### 4.0 Setup (first launch and on demand)
Finds or installs the tools and records versions in the manifest of every project it runs. Python: Homebrew `python3` (3.14 today) → app creates `~/Library/Application Support/HydrogenSplat/venv` and pip-installs a pinned `requirements.txt` (numpy, opencv-python-headless, pycolmap==4.2.0). Brush binaries: path to `~/Desktop/Apps/brush/target/release/{brush,brush-path-render}` (configurable; bundling is a later milestone — they are ~arm64 Metal binaries and relocate fine). ffmpeg: Homebrew or bundled static build. adb: Android platform-tools (Homebrew `android-platform-tools`). Green/red per tool with the fix command shown.

Prevent idle sleep while a stage runs (`IOPMAssertionCreateWithName` / `caffeinate -i` around the child). Show the estimated total: ~15 min solve + ~55 min train + ~2 min render on the M-series Mac used for rig6.

### 4.1 Ingest
`adb devices` (USB or the phone's wireless adb at its LAN address — both are just adb serials); list `/sdcard/DCIM/Camera/VID_*_2x1.h4v` with sizes and mtimes; show clips newer than the last ingest with a 1-fps thumbnail strip pulled via `adb exec-out` of a small ffmpeg pass, or after the pull. Pull selected clip, MD5, ffprobe; **validate**: one video stream 3840×1080, comment tag contains `leia3d_layout=2x1` and `leia3d_width_per_view=1920`, `recording_software_version` recorded. Mismatch → refuse with the reason (this is the check that stops a stills-mode or 4V-layout file from silently producing garbage).

Also accept a clip dropped onto the window (no phone needed) — that's how the golden test runs.

### 4.2 Calibration profile
A profile is a JSON resource in the app: intrinsics for both eyes (OPENCV model), `cam_from_rig` rotation quaternion and translation, source (board spec, date, rms), and the match keys: `width_per_view`, `height_per_view`, `recording_software_version`. v1 ships one profile, converted from `capture_video/h1_video_stereo.npz` (values in spec §2). Ingest picks the profile whose match keys equal the clip's; no match → block with "no calibration for this mode" rather than guess. Profiles are versioned; the manifest records which one solved the project so a later recalibration never silently changes an old project.

v1.1: a "Calibrate from board clip" screen wrapping `stereocal.py` (board spec fields, frame extraction, rms readout) and the depth-walk board clip that pins the baseline.

### 4.3 Select
`hs select` = `select_frames.py` with the calibrated default (residual 1.5 px at 480 wide, min gap 6, max 90, search 4, clip <2%). Outputs the frames, `selection.json`, and a contact sheet. **Checks**: frame count in 30–120 (rig6: 65); median gap 6–15; fraction of picks that hit `max-gap` < 15% (a high fraction means the phone stood still — coverage will be thin there). The page shows the contact sheet, the count, and a residual/sharpness plot along the clip; a slider for the residual threshold re-runs select in ~5 s. This is the cheap place to catch a bad capture before spending 70 minutes.

Exposure: the H1 writes no per-frame exposure metadata, so the app can't read ISO. It reports what it can measure — clipping fraction and a noise estimate (median absolute Laplacian in flat regions) per frame — and flags a clip whose noise is well above the rig6 reference. Yesterday's clip was auto at 1/250 s, ISO 1280; that shutter is right for blur, that ISO is costing SIFT features and splat cleanliness. Capture guidance, not app logic: more light, lower ISO, same shutter.

### 4.4 Solve
`hs solve` runs `rigcolmap.py prep → sfm --float-rig → export` with the profile written into the rig config (calibrated intrinsics are mandatory — spec §4). Progress from pycolmap's log lines (feature extraction N/M, matching N/M pairs, registered image count). **Outputs** beyond today's: `per_image.json` (per-image reprojection error, observations) and `coverage.json` — for every capture: azimuth, elevation, distance about the SfM median, in the frame the path builders use. The move designer needs this and it's free at solve time.

**Checks**: registered frames == selected frames (rig6: 65/65); mean reprojection ≤ 1.8 px (rig6: 1.394); no image > 3 px; L–R separation equals the profile baseline (rig constraint held). Partial registration → the app proposes, in order: re-select with `--max-gap 45`; trim the clip's tail (`--end`) — the chain usually breaks on the last fast move; then stop and show the per-image table. Never proceed to train on a partial solve silently.

### 4.5 Train
`hs train` wraps `brush <dataset> --total-train-iters 40000 --growth-stop-iter 30000 --refine-every 130 --export-every 2500 --export-path <project>/train/exports --export-name export_{iter}.ply` (verify the exact `export-name` template Brush expects — it's a config field, not yet exercised). Progress: parse the step counter and splat count from the spinner, confirm against export files appearing. Runs with `caffeinate`. On app relaunch with a `running` train stage and a live PID → reattach to the log; with a dead PID → offer resume from the last export via `--start-iter` (needs verification that Brush resumes from a `.ply` — if not, "restart" is the honest option).

**Checks**: final export present and its `element vertex` count within 0.6–1.4× of the rig6 reference per registered frame (rig6: 170,841 for 65 frames ≈ 2,600/frame); splat count monotone through growth. The train page shows the growth curve live.

Defaults are the rig6 settings verbatim. Advanced disclosure exposes `refine-every`, `growth-stop-iter`, `total-train-iters`, `split-at-screen-size`; nothing else in v1.

### 4.6 Move (the first human checkpoint)
Three presets, all built from `coverage.json` and `rig.npz`, all through the existing builders:

- **Sweep loop** — `spline_path.py --aim-median`, window chosen automatically as the longest contiguous run of captures within ±6° of the median elevation (rig6: 8:34), 240 frames.
- **Boom–slide–settle** — `key_path.py` with keys chosen automatically from the coverage table. Define "low" as within 6° of the low-pass median elevation (rig6: −7°, 41 of 65 captures). Start = the highest-elevation capture in the right third of the azimuth range; boom = walk forward in capture order to the first low capture; slide = every other low capture from there, in capture order, then the leftmost low capture; settle = the low capture nearest azimuth 0. Verified on rig6: start 51 (az 32°, el 19°), boom to 56 (az 50°, el −10°), slide through 58–64 to 16 (az −25°), settle at 62 (az −1°) — equivalent to yesterday's hand-picked move_boom.json (52→56→64→16→22), not identical keys. 360 frames, 0.5 s holds.
- **Custom** — a 2-D plot of every capture at (azimuth, elevation) with its thumbnail on hover; click captures in order to make keys; the same builder.

Every build prints the aim check; the app parses it and **renders the aim-check image itself**: the real image the aim point was projected into with a crosshair at the projected pixel, alongside the hull number ("virtual camera never more than N mm from a real camera", pass ≤ 25 mm). The user must click "Aim is on the subject" before Render enables. This is the rule that four shipped bad paths bought; it stays a human step in v1.

### 4.7 Render (the second human checkpoint)
`brush-path-render <ply> --path <move> --out <frames> --width 2400`, then ffmpeg to `<name>_1920.mp4` (H.264, crf 17, faststart) and a centred 4:5 crop `<name>_1080x1350.mp4` for Instagram; frames deleted after encode unless "keep frames" is on (500 MB per 360 frames). Plays the result in the app; "Reveal in Finder"; "Render pruned variant" runs `prune_splats.py --center=<SfM median> --radius 0.3` first.

Before every render the app MD5s the `.ply` against the manifest's train output and checks the move file's mtime is older than nothing it depends on — both mistakes happened in the rig6 session.

## 5. Golden test

The rig6 clip is the regression fixture. `hs selftest --clip VID_20260912_151351_2x1.h4v` runs select → solve and asserts: 60–70 frames selected; 100% registered; mean reprojection 1.2–1.6 px; coverage table contains the two low passes and the high pass (min elevation < −5°, max > +20°); aim check for the median lands within 60 px of (959, 552) in cap021_L. Train is not part of the automated test (55 min); a manual golden train should land at 170k ± 20k splats. This test runs in CI-less fashion from the app's Setup page ("Verify installation") and from the terminal. It is the acceptance test for every engine milestone.

## 6. Risks and how each is handled

- **Brush progress parsing** is fragile (a TUI spinner, not a log). Mitigation: exports on disk are the source of truth; the parse only feeds the ETA. If it proves too flaky, build `apps/brush-cli` (headless trainer in the same workspace, currently not built) and add a `--json-progress` flag there — a 30-line Rust change, contained.
- **Python relocation.** A venv created from Homebrew Python is machine-local; fine for v1 on Stephen's Mac. For distribution, swap to python-build-standalone in the bundle; the `hs` CLI contract doesn't change.
- **Long runs and sleep.** `caffeinate` + manifest-based resume. Resume from a Brush export is unverified — test early (M3), and if it doesn't work, say "restart" in the UI rather than pretending.
- **Disk.** A render is 0.5 GB of PNGs, a train dataset 60 MB, exports 12 MB each at 2,500-step cadence × 16 = 190 MB. Project folder shows its size; "Clean intermediates" removes render frames and pre-final exports.
- **Coverage edge cases in auto-keys.** A clip with no high pass has no "top right" — the preset must detect that (max elevation < 8°) and fall back to a low-only slide, saying so. The hull check (≤ 25 mm) is the backstop for any key set.
- **Calibration drift.** A firmware or app update on the phone changes the crop; the match keys catch a version change but not a same-version drift. v1.1's board-clip recalibration is the answer; until then the solve's reprojection check is the canary.
- **GPU.** Brush uses wgpu/Metal; brush-path-render likewise. Both already run on this Mac. No CUDA anywhere.

## 7. Milestones (each ends with a demo and the golden test where it applies)

**M0 — Repo and engine CLI (1–2 days).** `git init`, `.gitignore` (venv, Projects, frames, ply), vendor the six scripts into `engine/hs/`, add `events.py`, one `hs` entry point with the stage subcommands, a `requirements.txt`, and `hs selftest`. Acceptance: `hs select`, `hs solve`, `hs paths` reproduce the rig6 numbers from the terminal; `hs train` and `hs render` run the binaries and emit events.

**M1 — App skeleton, Setup, Projects, Ingest (2–3 days).** SwiftUI app with ProjectStore, ToolRunner reading JSON-lines from a Process, Setup page with tool detection, project list, Ingest page with adb listing and drop-a-clip, validation. Acceptance: pull a clip from the phone into a new project; manifest written; events from a stub script render live.

**M2 — Select + Solve pages (2–3 days).** Contact sheet, threshold slider, solve progress, per-image table, coverage table, the checks with the proposed remedies. Acceptance: rig6 clip solves inside the app with 65/65 and the coverage plot shows the three passes.

**M3 — Train (1–2 days plus a 55-min run).** Brush driver, live growth curve, caffeinate, relaunch/reattach, resume-or-restart decided by test. Acceptance: a train started from the app survives closing the lid and finishes with export_40000.ply.

**M4 — Move designer + Render (3–4 days).** Three presets, coverage plot with thumbnails, aim-check image with the required confirmation, render, encode, playback, 4:5 crop. Acceptance: the boom–slide–settle preset on rig6 produces a move whose azimuth/elevation trajectory stays within 5° of move_boom.json's at every 10% of the path, passes the hull check, and renders without the aim leaving the subject.

**M5 — Golden test in-app, packaging, docs (1–2 days).** "Verify installation" runs selftest; app builds as a signed, notarized, non-sandboxed .app with the binaries' paths configurable; README with the capture guidance from spec §1.

**M6 — v1.1 candidates.** Board-clip calibration screen; depth-walk baseline clip; left-eye-only training toggle for the residual investigation; pruned-model variant; bundling Python and Brush.

## 8. Hand-off to whoever builds it

Read, in this order: `capture-to-splat-spec-v1.md` (what the pipeline is, with numbers), this document, `HANDOFF-2026-09-13-colmap-rig.md` §4 (mistakes not to repeat), `colmap-rig-pipeline.md` (solver internals), `video-capture-pipeline.md` (why frame selection is parallax-based). The code is `~/Desktop/Apps/REDHydrogenOne/mv/` (six files) and `~/Desktop/Apps/brush/apps/{brush-app,brush-path-render}`. The golden data is `~/Desktop/Apps/REDHydrogenOne/capture_video/` — do not modify it; copy the clip into the new repo's `fixtures/`.

Rules that are not negotiable in v1: calibrated intrinsics go into the COLMAP rig config; `--float-rig` on, `--refine-intrinsics` off; frame selection by homography residual with no optical-flow escape hatch; aim point = SfM median, projected into a real view and confirmed by a human before render; MD5 the ply and check mtimes before render; never train on a partial solve without saying so.

## 9. Still open (decide during M0, not before)

Whether Brush resumes from an export (`--start-iter` + a ply source) — decides the resume UI. Whether `--export-name` is a template or a fixed name — decides how exports are found. Whether the wireless-adb address is stable enough to default to, or USB is the v1 story. Whether ffmpeg is bundled (static build, ~80 MB) or required from Homebrew. None of these change the architecture.
