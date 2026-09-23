# HydrogenSplat user guide v1 (2026-09-22)

What every page and control does, what it runs, how long it takes, what its checks mean and
what lands on disk. Written from the SwiftUI views, the HSCore models and the engine stage
docstrings on `research-fixes-2026-09-21`. Two controls marked **New on this branch** are
specified in the 2026-09-22 plan (sequential matcher, progress with ETA, this guide) and were
being built when this was written; check their labels against the build. Capture guidance is in `docs/capture-sop-v1.md`.
In commands, `P` is the project folder; the Console runs them from the repository folder.

## How the app is organised

The sidebar holds Setup, New Project, Console, Event Replay and the project list. A project
has two pages, **Pipeline** and **Viewer** (toolbar picker, ⌘1 / ⌘2). The Pipeline page has a
header (with a lock line when a run holds the project), a one-line strip for the latest run,
the **rail**, the selected item's workspace and, on the right, a **CHECKS AND METRICS**
inspector. The rail is grouped PREPARE (Source, Frames), DATASET (Exposure, Masks), MODEL
(Train), SHOT (Move, Grade, Render). `opt` marks optional items; an arrow marks items worked in
the Viewer, and selecting one switches to the Viewer page. ⌘[ and ⌘] walk the rail. Each dot is
the state of the engine stage behind it (green done, blue running, red failed, orange stale,
grey pending); Frames shows the less advanced of select and solve. A project opens on the first
required item not done.

A button that runs a stage starts `hs <stage> -p <project> …` as a child process and draws its
JSON events; only the move editor and the Look write files themselves.
Each stage records in the project's `manifest.json` its status, times, exact command,
metrics, checks and artifacts. Re-running a stage marks everything downstream **stale** and
wipes only its own folder. One run at a time may write to a project: the engine holds
`.hs.lock` (pid and stage) and refuses a second with "project is busy". Projects are folders
in the Projects folder set in Setup; the appendix lists what each stage writes. The app
re-reads manifests and locks every 3 seconds, so a Terminal run shows up without a reload.

## New Project (Source)

Sidebar ▸ **New Project** (⌘N). **Source** picks Phone or File.

| Control | What it does |
|---|---|
| **Refresh** (Phone) | `hs phone`: lists adb devices (USB or wireless, USB debugging on) and their `VID_*_2x1.h4v` clips. The newest clip not yet in a project is preselected. |
| Drop zone / **Choose…** (File) | One `.h4v` or `.mp4`. |
| **Label** | Folder becomes `YYYY-MM-DD_<label>` (default: the clip's time); `-2`, `-3` are added rather than reuse a folder. |
| **Ingest** (⌘↩) | Phone: `hs ingest --phone SERIAL --remote PATH`, pulled into `source/` and MD5-compared with the phone. File: `hs ingest --clip PATH`, copied. **Open Project** appears on success. |

The pull runs at USB/Wi-Fi speed with an ETA; the checks after it take seconds:
`clip_is_2x1_video` (one 3840×1080 stream tagged `leia3d_layout=2x1`, anything else refused),
`pull_matches_phone`, `profile_matched` (no calibration profile for the mode blocks). Writes
`source/<clip>`, `<clip>.md5`, `probe.json`.

No button for frames sources (a mono take, camera-named frames, an R3D array); ingest marks
select done for them:

```
hs ingest -p P --frames DIR [--kind mono|array]
```

```
hs ingest -p P --r3d RDM_DIR --take 067
```

The **Source** rail item shows the clip or cameras, MD5, format and profile, and the full
**Stages** table; click a row to jump to its rail item.

## Frames: select, then solve

### Select frames

Hydrogen clips only; array and mono projects say there is nothing to select.

| Control | What it does |
|---|---|
| **Parallax residual** | Take a frame once the camera has moved this far (px) after rotation is removed. Lower = more frames. Default 1.5. |
| **More settings** | **Gap between picks** (6 to 90 frames; a max-gap pick is taken even if the camera stood still), **Search window** (sharpest of 4), **Max clipped** (0.02), **Frame range** (end −1 = to the end), **Defaults**. |
| **Select Frames** (⌘⇧S) | `hs select` with the values you changed. Again asks "Select the frames again?"; **Select again** marks solve and everything after it stale. |

It reads every frame of the clip, then measures each pick, so time grows with clip length.
Checks: `frame_count_in_range` (30–120), `median_gap_in_range` (6–15), `maxgap_fraction_low`
(under 15%; high means the camera stood still and coverage is thin there). Remedy: change the
residual or gaps and select again, or reshoot.

The report: counts, median gap and Laplacian; chips **All**, **Flagged**, **Clean** and one
per flag (hover for its meaning); **Sort**; focus and exposure traces along the clip with the
picks on top; **Show stills** for the contact sheet. Flags are a report: nothing is dropped. On
a still, double-click opens the 2×1 frame; right-click offers **Open frame**, **Reveal in
Finder**, **Use as exposure reference** and **Copy sharper frame number**; nothing in the app
swaps a pick for that frame. Writes `select/frames/` (picks, `selection.json`),
`select/quality.json`, `select/thumbs/`, `select/contact.jpg`.

### Solve

**Solve** appears once select is done (**Solve again**, or **Solve (stale)** after an earlier
stage re-ran). `hs solve`: split the eyes, CPU SIFT features, matching, incremental mapping
with both eyes as one rigid rig on the calibrated intrinsics, then export the undistorted
training set. Bars: `solve · features`, `· matching` (block i/n), `· mapping`, `· export`.

Matching is almost all of the time. Exhaustive matching compares every image with every other,
so it grows with the square of the frame count: 422 captures (844 images, 355,746 pairs) on
CirclesSculpture ran at 88–159 s per 50×50 block, and COLMAP visits all 289 blocks, so that
match alone is 7–13 hours on Apple Silicon; ~150 captures (300 images, 44,850 pairs) is about
an hour and a quarter of matching. Features grow in proportion to the image count, mapping
faster than that. The matching bar counts blocks and is honest; before this branch it had no
ETA, on this branch every bar does. The chooser below shows the estimate for both matchers
before you press Solve; trust it over any figure written here.

**New on this branch, the matcher chooser.** When the solve step opens and whenever the frame
count changes, the app runs `hs solve --estimate` (no lock, writes nothing) and shows
**Exhaustive / Sequential / Auto** as a segmented control with an estimate beside each ("≈ 1 h
15 min", "≈ 12 h"), Auto preselected — sequential above 60 captures, exhaustive below.
Sequential matches each capture with the 15 before and after it (both eyes) and its rig mate,
plus every 8th capture against every other 8th so the orbit's closure is caught; it takes
minutes at any count. Use it for orbits and walk-arounds; use exhaustive when the camera
returns to a view from far apart in time and the every-8th pass would miss it. Solve passes
`--matcher`. Estimates say "measured on this Mac (N runs)" once solves have been timed, else
"defaults". The frame count gains the matching cost beside it. Solve lives on the Frames page; there is
no Solve page. Until the first solve has been timed on this Mac the estimates come from default
rates and say so; each finished solve refines them. The window and stride have no control:

```
hs solve -p P --matcher sequential --overlap 15 --loop-stride 8
```

Checks: `all_frames_registered` fails the stage and names the unregistered captures. In order:
select again with max gap 45; trim the tail with Frame range end; read `solve/per_image.json`.
Never train on a partial solve. `mean_reproj_ok` (≤ 1.8 px), `no_image_over_3px` (names the
worst), `rig_constraint_held` (L–R separation equals the profile baseline),
`rig_size_matches_L_views` are recorded without blocking. Writes `solve/` (`sparse/rig`,
`per_image.json`, `coverage.json`) and `train/dataset/` (images, `sparse/`, `rig.npz`). A
re-solve replaces `train/dataset`, so exposure and masks go stale.

Array and mono projects get the same Solve box and matcher chooser on this branch (before it
they had no Solve button). What the button cannot pass is a metric reference, so without one
`scene_scaled` fails and needs you; give it in the Console:

```
hs solve -p P --scale-pair GA,GB,700
```

or from a ChArUco board in view (`hs scale` has no button):

```
hs scale -p P --board 7,5,40,30
```

## Exposure (optional)

Rewrites the training images to one exposure and white point; poses untouched. Shown once
solve is done or stale.

| Control | What it does |
|---|---|
| **Match to** | **Best frame** (`--reference auto`: the clean pick within a third of a stop of the median that clips least). **A frame I pick**: **Capture**, **Use suggested (#N)**, or the contact sheet. **Median of all views**. |
| **Correct** | **Exposure + white balance** (per channel) or **Brightness only**. |
| **Preview** | `--dry-run`: reports gains, writes nothing, changes no status. |
| **Match Exposure** (⌘⇧E) | Writes the images, originals to `solve/exposure_backup/`. With a model it asks "Rewrite the training images?": train, prune, render and views go stale. |
| **Restore originals** | `--restore`, after a match. |

Seconds (140 views in 11 s). Checks: `views_share_one_exposure` (luma spread ≤ 1.10× after),
`no_view_needed_an_extreme_gain` (gains within 0.35–3.0). A failure usually means one clipped
or badly framed view: find its gain, leave it out in Train, or restore. Writes the images in
place and `train/dataset/exposure.json`.

Skip it when exposure was locked: if **Preview** shows `luma_spread_before` near 1.0 there is
nothing to correct. On an array **Best frame** and **Median** are refused (the views frame
different things), and array and mono projects have no quality report for **Best frame**. Match
an array on a shared target:

```
hs exposure -p P --reference board
```

## Masks (optional)

One silhouette of the subject per view: a subject layer trains on it, a background layer reads
it inverted. Shown once solve is done.

| Control | What it does |
|---|---|
| **Method** | **Object (Vision)**: Apple Vision's object in each photo that sits inside the region projected from the solve. **Region only**: the projected region, which includes whatever is near the subject. |
| **Built from** | **Trained model** (prune output, else final export) or **SfM points**; points only until a model exists. |
| **Size** | ×0.5–2 on the radius fitted to the orbit (subject ≈ 70% of frame). Needs no metric scale. |
| **Margin** | Outward dilation, 0–20% of the radius. |
| **Detail** | **Edge** (px), **Minimum opacity**, **Close gaps**, **Keep the largest piece only**, **Preview** views. |
| **Build Masks** | `hs masks` with every value explicit; asks "Rebuild the masks?" when a model and masks exist. **Reveal in Finder**. |

One Vision pass and one projection per view. Checks: `vision_found_the_subject` (fallback to
the region on ≤ 10% of views; needs you), `every_view_has_a_silhouette`, `coverage_sane`
(median 2–75% of frame; needs you). Under 2%: raise Size or lower Minimum opacity; over 75%:
lower Size. Look at the preview sheet (yellow edge, magenta Vision region) before training.
Writes `train/dataset/masks/{L,R}/capNNN.png`, `select/masks_preview.jpg`, `masks_vision/`;
marks train, prune, render and views stale.

Skip it unless you want a layer. What has worked: train a full model, build masks from it,
train the layer.

## Train

Shown once solve is done. **Current model** gives splats, training time, Brush commit, and
"kept as <archive>" or "not archived — training replaces it", with the last growth curve.

| Control | What it does |
|---|---|
| **Length** | Total iterations; default 40,000. |
| **Stop growing at** | 15,000, 20,000 or 30,000 (default); after it splats are only refined. |
| **Hold out** | **Nothing**, **Every 10th capture** (default), **Every 5th capture**, counted from capture 5, both eyes on a stereo rig. |
| **Layer** | **Full scene**, **Subject**, **Background**; needs masks. |
| **Outside the mask** | Subject only: **Pushed empty** (default; nothing outside) or **Left unsupervised** (Brush's default). Only a pushed-empty subject merges cleanly with a background. |
| **Detail and experiments** | **Refine every** (130), **Split big splats**, **Anti-aliasing**, **Background noise**, **Leave out views** (`L/cap064,R/cap069`), **Brush arguments**. |
| **Keep it as** | Archive name; empty means the next train replaces this model. |
| **Then score** | **Score views**, **both eyes**, crop mm (auto, 350, 2000). Warns when earlier reports used another crop: those scores do not compare. |
| **Start training** (⌘↩) | Runs the queue. Without a name it asks "Train without keeping the result?". |

Beside Start: on battery (macOS will sleep and stall training) or plugged in (keep the lid
open). Every stage holds `caffeinate`; a closed lid still sleeps.

A 40,000-iteration train takes 50–110 minutes; the splat count decides it, so hold-outs, masks
and anti-aliasing change the time. Checks: `final_export_present` (else failed),
`splat_count_in_range` (500–40,000 per registered frame: below, growth never took; above, too
heavy and probably floaters), `splat_count_monotone_through_growth`, `view_count_matches`,
`exposure_still_applied` / `masks_still_applied` (a re-solve undid them: re-run or accept),
`no_sleep_during_run` (≤ 60 s asleep). Writes `train/exports/export_NNNNN.ply` every 2,500
iterations; the dataset is never modified.

Hold-outs chosen by camera position have no control:

```
hs cameras -p P --holdout 16 --write
```

```
hs train -p P --exclude @holdout
```

```
hs views -p P --captures holdout
```

A train started in Terminal is followed from `logs/train.log`: "Training in another window
(pid N)", iteration, splats, it/s, ETA, and "no progress for … — asleep or finishing?" after two
quiet minutes. Stop it where it was started.

## The queue: train → archive → score

**Start training** queues **Train**, **Archive <name>** (if named), **Score views (L)** and,
with both eyes, **Score views (R)**; each runs only if the one before succeeded. Chips show
each step's state; the chart plots splats against iterations. **Stop** interrupts the current
step (SIGINT): the stage is marked failed, the lock released, later steps skipped. Settings are
taken when you press Start.

Archive (`hs archive`) copies the model, `rig.npz`, the COLMAP camera records and
`exposure.json` into `archive/<name>/` with a manifest of MD5s for everything it was trained
on, whole or not at all.

Score (`hs views`) renders the model from each chosen capture's pose and compares it with the
photograph, a few seconds per view: the hold-outs, or with none four in-sample views. Subject
layers are scored inside the masks, background layers outside. Report name `views_<archive>`,
else `holdout_latest` or `views_latest`. Checks: `model_registers_to_photographs` (no view with
over 10% of patches displaced more than 4 px; needs you), `every_azimuth_band_registers`,
`no_view_much_softer_than_achievable`, `captures_sharper_than_the_model` (the photograph is
blurred; its score is a floor), `subject_registers_better_than_its_surroundings`. A bad band at
the edge of coverage is thin capture, not training. Writes `views/<name>_report.json`, a
comparison image per view, `<name>_coverage.jpg`.

## The Models box

Under Train: every `archive/*`, the live export (`current`) and every `prune/*.ply`, each with
layer, splats, training time, date, and **this solve** or **other solve** (trained against
another `rig.npz`; render refuses it).

| Control | What it does |
|---|---|
| **View** | Opens it on the Viewer page. Right-click: **Open in New Window** (for an A/B), **Reveal reports**. |
| **Archive…** | `current` only: `hs archive` under a name. |
| **Score** | **Score whole crop**, **Score inside masks**, **Score outside masks**: `hs views --ply` with the Train page's hold-out and crop, as `score_<model>_<region>`. |
| **Reveal** | The ply in Finder. |

Scores list under the row (PSNR, interior, edge, displaced, views). Archive and Score run in
the Train page's queue slot and are disabled while any run holds the project. No buttons for these;
their plys then appear here:

```
hs prune -p P --score --ply P/archive/NAME/export_40000.ply --name NAME
```

```
hs split -p P --ply P/archive/NAME/export_40000.ply --masks vision
```

```
hs merge -p P --models subject-a,background-a --name merged
```

## Move (the Viewer)

If the page says **Viewer unavailable — no compiled Metal shaders**, the app was launched with
`swift run`; build it with `app/scripts/make_app.sh --install` (see Setup and About).

The Viewer page: **Model**, **Reload**, **Open in New Window**, **Unload** (frees 1–2 GB),
**Move** (the move panel). Loading runs `hs cameras`, which takes no lock and works during a
train. It is a preview, not brush-path-render: judge frames from a render. **Capture** ‹ › (←
→), **Eye**, **Stand here** (↩: that capture's pose and lens), **Orbit** (O); drag to orbit,
shift-drag to slide, scroll to move in. With no move open, a **Move** picker plays any
`move/*.json`.

A move is keys: camera positions at times, every frame looking at the anchor. Frame the start,
then **New Move From Here…**.

| Control | What it does |
|---|---|
| + menu | **New Move From Here…**, **New From Preset** (**180° orbit**, **360° orbit**, **Slow push in**, **Slow pull out**, **Arc and push**, **Crane up and out**; 8–36 s, from where the viewer stands), **Duplicate…**, **Rename…**, **Close Move**, **Reveal in Finder**. |
| **Add from the last key** | arc 15° / 45°, boom 3° / 10°, dolly 5 / 20 cm, hold 0.5–2 s, at slow, normal or fast. Centimetres are scene units, real only on a metric solve. |
| **Anchor** | **Pick on model** then click the subject; **Reset to subject** (SfM median). |
| Selected key | **Delete**, **at** (s), **Ease (come to rest)**. |
| Timeline | Play (space), **Key** (K, from the viewer's camera), delete (⌫), drag keys to retime. |
| **Lens** | **Use <capture> <eye>**: render through that pinhole. |

**The move** lists length, peak speed, distance to anchor and coverage, the angle from the
anchor to the nearest real camera: green under 3°, amber to 6°, red beyond (least to go on;
nothing blocked). A speed jump names the frame to retime around. Edits save after a pause to
`move/<name>.keys.json` and `move/<name>.json`, the file render reads. The editor never runs
`hs move`, so the Move dot stays pending; render accepts the file. Scripted and preset moves
(`hs move`) are Console only.

## Grade (optional)

Set the look in the move panel's **Look**: **Preview**, **Lift**, **Gamma**, **Gain**, **Crop**,
**Reset**; it saves to `grade/<render name>.json` and, with Preview on, Render bakes it.

The **Grade** rail item (⌘1 from the Viewer) checks the bake on a rendered frame: **Move**,
**Before**, **Frame** / **Middle** (`hs grade --still`), the three sliders, **Per channel**,
**Sharpen** (export only), **Crop**, **Headroom**, **Reset**, **Export graded video** (`hs
grade`, an ffmpeg pass), **Play**. Writes `render/<move>_graded.mp4`, `grade/<move>.json`. The
crop follows the head only with `move/<name>_frame.json` from `hs move --headroom-mm`; keyframed
moves get a centred crop. For a render of an archive (`<move>_<model>`) it looks for
`move/<move>_<model>.json`, which is never written, and shows 0 frames. Skip Grade when no look
is wanted.

## Render

In the move panel. **Look at** **First**, **Middle**, **Last** each shows that frame of this
exact move; Render unlocks after all three, and any edit resets them. **Width** (1920, 2400,
3840), **4:5 crop**, **Keep frames**. **Render <model>** (**Render + grade <model>** with the
Look previewed) runs `hs render --move <name> --ply <model>`, then `hs grade`.

A few seconds per frame plus the encode; frames = seconds × 30. The render fails unless its
guards hold: `ply_md5_matches` (train's export, prune's output of it, or an archive matching its
manifest), `move_newer_than_solve`, `model_matches_solve` (the model was trained on the current
`rig.npz`). A failed guard means model, move and solve do not belong together: retrain on the
current solve, or remake the move on a model from it. Writes `render/<name>_1920.mp4` and
`_1080x1350.mp4` (name = move, plus `_<model>` for an archive or prune ply); frames only with
Keep frames. The result plays in the panel. No buttons for flicker or delivery:

```
hs stability --video P/render/NAME_1920.mp4 -p P
```

```
hs export -p P --archive NAME --shot-sheet
```

## The run panel and progress

The strip under the project header shows the latest run, running or finished in the last 10
minutes; "N check(s) need you" counts failed checks. Click it for the RunPanel: **Stop**, copy
command, one bar per stage · step (done/total, ETA when the engine sends one, detail), an
**Error** box (message, hint, stderr tail), **Checks**, **Metrics**, **Events**. An orange cross
is a failed check; **needs you** means a person must judge it, pass or fail. The inspector
shows the stored record: command, error, checks, metrics, **Open log**, artifacts.

Today only the phone pull, train and render send an ETA. **New on this branch:** every bar
with a total gets an ETA, and a one-line strip (stage · step · fraction · ETA) stays visible
while any run is live, repeated in the sidebar row and window title, with the percentage on
the Dock icon.

## Console and Event Replay

**Console**: a command in `zsh -l` from the repository folder, `hs` on the PATH; **Run**,
**Stop** (Ctrl-C), **Clear**, copy output. **Event Replay** plays a recording from
`engine/tests/events` through the RunPanel: **Recording**, **Speed**, **Fail after**, **Play**.
A project's own runs replay from the Console:

```
hs replay P/logs/train.events.jsonl --last --speed 50
```

## Setup and About

**Building and launching the app.** Two ways to build, and they are not equivalent:

| command | what you get | when |
|---|---|---|
| `app/scripts/make_app.sh --install` | `HydrogenSplat.app` in `~/Applications`, release build through `xcodebuild`, **with the splat viewer** | the app you use |
| `swift build` / `swift run HydrogenSplat` (in `app/`) | a debug build, everything works **except the Viewer**, which says "Viewer unavailable … no compiled Metal shaders" | quick checks that a change compiles |

The viewer's renderer (MetalSplatter) loads its shaders from a compiled `default.metallib` in
its resource bundle; `swift build` copies the `.metal` sources without compiling them and writes
no Info.plist into resource bundles, so only the `xcodebuild` route produces a working viewer.
`make_app.sh` needs the full Xcode (`sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`)
and the Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`, a separate download
since Xcode 26); it prints those commands if either is missing. After pulling a change, rebuild
and relaunch from `~/Applications`; a running solve or train is unaffected, and the new app
re-attaches to it within a few seconds.

```
cd ~/Desktop/Apps/HydrogenSplat && app/scripts/make_app.sh --install
```

**Setup**: **Repository**, **hs executable**, **Projects folder** (**Choose…**), **Extra PATH**
(so a Finder-launched app finds ffmpeg, ffprobe, adb), problems with their fix command, **Use
this checkout's defaults**, **Check tools** (`hs tools`, also run on opening). Ingest, Select,
Solve, Exposure, Masks, Train and Render stay disabled while Setup lists a problem. **About HydrogenSplat** (app menu):
version and licences.

## Troubleshooting

**Project is busy / lock.** "<stage> is running (pid N)" means a live process holds `.hs.lock`;
buttons that write are disabled until it ends. "stale lock from pid N — the next run reclaims it" needs
nothing. If the pid in "project is busy" is not an hs process, delete `P/.hs.lock`.

**Stale stages.** Re-run them in rail order. The engine runs on stale inputs with an
`upstream_<stage>_fresh` warning, but the app's Train and Masks pages need solve done.

**A stage failed.** The RunPanel or inspector has the error and its hint; **Open log** has the
tool's own output.

**The app closed mid-train.** The app does not re-attach its own runs. If hs survived, it holds
the lock and Train follows it like a Terminal run. If it died, the manifest says running until
the next `hs` command on the project marks it "failed — interrupted"; Start works again, and
exports up to the last 2,500 remain. Resuming from one is experimental:

```
hs train -p P --resume-from P/train/exports/export_25000.ply
```

**A Terminal run shows up in the app.** Expected: manifests and locks are polled every 3
seconds. A train is followed from its log; other stages disable Start and name themselves.

## Appendix: stages, folders, outputs

| Stage | Button | Folder | Main outputs |
|---|---|---|---|
| ingest | Ingest | `source/` | clip, `.md5`, `probe.json`; `frames/` for frames sources |
| select | Select Frames | `select/` | `frames/`, `quality.json`, `thumbs/`, `contact.jpg` |
| solve | Solve | `solve/`, `train/dataset/` | `sparse/rig`, `per_image.json`, `coverage.json`; images, `rig.npz` |
| scale | none | `scale/` | `scale_report.json`; rescales the solve |
| exposure | Match Exposure | `train/dataset/`, `solve/exposure_backup/` | rewritten images, `exposure.json` |
| masks | Build Masks | `train/dataset/masks/` | `{L,R}/capNNN.png` |
| train | Start training | `train/exports/` | `export_NNNNN.ply` |
| archive | Keep it as, Archive… | `archive/<name>/` | ply, `rig.npz`, `sparse/`, `manifest.json` |
| views | Score views, Score | `views/` | `<name>_report.json`, comparisons |
| move | editor saves | `move/` | `<name>.keys.json`, `<name>.json` |
| prune | none | `prune/` | `_pruned_r03.ply`, `_scores.npz`, `_nofloat.ply` |
| split | none | `split/<name>/` | `subject.ply`, `background.ply` |
| render | Render | `render/` | `<name>_1920.mp4`, `<name>_1080x1350.mp4` |
| grade | Export graded video | `grade/`, `render/` | `<move>.json`, `<move>_graded.mp4` |
| export | none | `deliver/<name>/` | ply, spz, sog, html, shot sheet |
| cameras | Viewer load | `viewer/` | `cameras_<model>.json`; `solve/holdout.json` |
| all | | `logs/` | `<stage>.log`, `<stage>.events.jsonl` |
