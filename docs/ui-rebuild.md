# UI rebuild (approved 2026-10-05): six steps, one verdict each

The mockup Stephen approved (Design canvas "HydrogenSplat UI Rebuild") replaces the stage rail and
its panels with six steps a non-technical user can follow, plus Calibrate as its own door and
Settings for tools and paths. This page is the contract every worker builds to. Branch `ui-rebuild`.

## The rail

| # | Step | Engine stages behind it | The verdict it states |
|---|---|---|---|
| 1 | Footage | ingest (`hs source`) | what was dropped, what kind it is, and the four facts that decide later steps: kind, useful resolution, exposure (fixed / auto), lens (calibrated / not) |
| 2 | Frames | select, solve | frames picked and placed, reprojection in px, coverage (8 directions × 3 heights) with the gap named |
| 3 | Scan (optional) | scale (`hs scale --lidar`) | whether the scan lines up with the placed cameras, the fit in mm, how much of the solve lies on it; then it gives real size, up, a starting shape and a surface to hold to. Takes PLY, OBJ or text points |
| 4 | Look | exposure | drift in stops across the orbit, clipped share, and ONE recommendation picked from those numbers (match to a reference / global drop / softer shoulder) |
| 5 | Subject | masks | whether masking is worth it (subject share of frame → share of the model the room would take), outlines made and tightened, how many frames need a look and the three things to do about them |
| 6 | Train | train, archive, views | three choices (what is in the shot · how long · score it), live progress in words, then the scores in words against the last run and what would help next |
| 7 | Shot | move, render, grade | Move (viewer, keys, coverage on the timeline) and Render & grade (checks, graded preview, frame/grade/export with the FULL `hs grade` scope) |
| — | Calibrate | calibrate | lens: board shown on screen or saved as PDF, film it, drop the clip, a verdict |
| — | Settings | tools | the old Setup page, unchanged |

Rail groups are gone; the numbered steps and the two doors are the whole left column. Scan was a
card behind the Calibrate door in the approved mockup; on the first walk-through it was not found
there (a scan belongs to a project, not to a camera), so it became a step (2026-10-05). A step's
dot is the least advanced of its engine stages (as `PipelineStage.status` does today).

## Page anatomy (StepUI.swift — use these, do not invent parallel ones)

`StepPage { StepHeader(title:lead:) ; VerdictCard(.good|.attention|.blocked|.info, headline:, detail:) { chips, charts, choices } ; … ; StepFooter(primary:…) }`.
`ChoiceTile` for a small set of exclusive choices, `MetricChip` for a short fact. The verdict
headline is one sentence in the user's terms with the number in it ("243 frames picked and all 243
placed. Cameras agree to 1.1 px."); the detail says what, if anything, to do about it.

## Copy rules

- Every sentence is about THIS project. No other project, clip, run or date by name ("rig6", "the
  coins run", "the 09-15 clips", "the head clip" are all gone). A fact that came from another project
  is stated as a rule ("Brush's late passes scale with the splat count"), not as that project's story.
- Numbers come from the manifest's metrics and checks; a page never shows a number the engine did
  not produce. Where the engine has no number yet, the page says what it will say once it has one.
- Flags are not copy. "Pushed empty" is "the model holds nothing outside the outline".
- Help text is one line. Longer explanations live in the engine README, linked by a `?` button.
- No emoji. Buttons are verbs ("Match the frames and continue", not "OK").

## Work split (Phase A, parallel; disjoint files)

**W1 — app shell and steps 1–4** (`PipelineRail.swift`, `ProjectDetailView.swift`, `AppModel.swift`,
`RootView.swift`, `IngestView.swift` → Footage, `SelectView.swift` → Frames, `ExposureView.swift` → Look,
`MasksView.swift` + `MaskReviewSection.swift` → Subject, new `FootagePage.swift`, `FramesPage.swift`,
`LookPage.swift`, `SubjectPage.swift`). Mounts W2's pages by these names: `TrainStepPage(project:)`,
`ShotStepPage(project:)`, `CalibrateStepPage(project:)`. Settings mounts the existing `SetupView`.
Subject kind (matte / glossy / bright / person / scene) is chosen on Footage and stored in the manifest
by `hs source --subject KIND` (W3 adds the flag; until it lands, store it in `AppModel` keyed by project path).

**W2 — Train, Shot, Calibrate** (`TrainView.swift` → `TrainStepPage`, `MovePanel.swift` + `MoveEditor.swift`
→ the Move half of `ShotStepPage`, `GradeView.swift` → the Render & grade half, `ScaleView.swift` →
the LiDAR half of `CalibrateStepPage`, new `ResultsSection.swift`, new `LensBoardView.swift`).
Train's three choices map to `hs train --recipe KIND --effort quick|standard|final` (W3); until
W3 lands, the page builds the same flags itself through `TrainSettings` (see the recipe table below).

**W3 — engine verdicts** (`engine/hs/stages/exposure.py`, `solve.py`/`masks.py`, `train.py`, `source.py`,
`views.py`, tests): `hs exposure --analyze` (metrics `drift_stops`, `clipped_share`, `clipped_where`
(highlights|whole), `recommendation` ∈ match|global_drop|shoulder|none and a check); `hs masks
--suggest` or a solve metric `subject_share_of_frame` and `room_share_estimate`; `hs source --subject`;
`hs train --recipe --effort` recorded in the manifest; `hs views` adds `summary` (plain-language
lines the Results page shows) and `next_steps` derived from checks.

**W4 — calibrate** (`engine/hs/stages/calibrate.py` (new), `engine/hs/board.py` if it fits, `solve.py`
to consume a lens profile, tests): `hs calibrate --board-image OUT.png|OUT.pdf --squares 7x5
--square-mm 35` (ChArUco, OpenCV `cv2.aruco`), `hs calibrate -p P --clip CLIP.mov` → lens profile
JSON (K, distortion, image size, rms px) under the user's Application Support, keyed by camera model
+ lens + recording size (from the clip's metadata), with a verdict check; `hs solve` takes
`--lens PROFILE` or finds the matching profile automatically and records which it used.

## Recipe table (Train, "what is in the shot" × "how long")

| Kind | masks | alpha | init | depth / spread | SH | glints | notes |
|---|---|---|---|---|---|---|---|
| matte | yes | transparent | lidar if present | 0.2 / 0.2 if scan | 3 | kept | |
| glossy | yes | transparent | lidar if present | 0.2 / 0.2 (spread on even without scan) | 3 | kept | |
| bright | yes | transparent | lidar if present | 0.2 / 0.2 | 3 | kept | Look recommends the softer shoulder |
| person | yes | transparent | lidar if present | 0.2 / 0.2 | 3 | kept | |
| scene | no | — | lidar if present | 0 / 0 | 3 | — | full-scene layer |

| Effort | max resolution | total iters | growth stop | refine every |
|---|---|---|---|---|
| quick | 1920 | 20,000 | 15,000 | 130 |
| standard | 1920 | 40,000 | 30,000 | 130 |
| final | source (≤ 3840) | 40,000 | 30,000 | 130 |

Run H (2026-10-04) showed 1920 scores the same as 3840 on the helmet in half the time; that is why
standard is 1920. No recipe passes `hs masks --snap-edge`: run I (2026-10-05) trained on snapped
outlines and scored 5.6 dB lower along the outline than run H on the same outlines.

## Build loop

Swift is built on the Mac only: after each phase Stephen runs
`cd ~/Desktop/Apps/HydrogenSplat/app && swift build 2>&1 | grep -E "error|warning: unused" | head -40`
and pastes the output; the engine's tests run anywhere (`cd engine && python3 -m unittest`).
Workers never commit; commits happen per phase after the build is clean.

## Added after approval

- **Scan** is step 3 (optional, after Frames); Look, Subject, Train, Shot are 4–7. Calibrate keeps the lens.
- **One project at a time** (2026-10-05, `RootView.swift`, `OpenProjectView.swift`). The sidebar that
  listed every project beside the work is gone; the step rail is the only left column. The window
  works in one project, named in a menu at the left of the toolbar. That menu lists every project
  (the open one ticked, one with a run going marked "running") and holds All Projects…, New
  Project…, Console, Event Replay and Settings…. File › New Project… ⌘N, Open Project… ⌘O; the app
  menu's Settings… ⌘, ; View › Console, Event Replay. **Open a project** is a page: every project,
  newest first, with where it has got to, its clip and its date; one click opens it. A launch goes
  back to the project the last session worked in (`openProject.v1` in the app's defaults), or to
  that page when there is none. On an app page with a project open the toolbar has Back. The strip
  under every page still shows a run in any project and opens that project on a click.
- **Shot › Clean up** (2026-10-05, `CleanUpView.swift`) is the first of Shot's three tabs
  (1 Clean up · 2 Move · 3 Render & grade). The viewer is the Move tab's, with a clean-up panel
  beside it instead of the move panel. Everything writes a copy; the trained model is never changed,
  and the model picker names the copies ("path cleaned · export_20000", "erased by hand · …").
  - **On the camera's path.** Gentle or Strong, one action, "Clean a Copy": `hs prune --clear-path
    1.3x|2.4x` on the model showing. A switch, "Do this at the end of every training run" (off
    until turned on; `CleanAfterTraining`), makes the Train step finish with the gentle clean and
    keep both models, so the two can be compared after a run.
  - **By hand.** Look Around | Select. In Select a drag draws a box over the picture; the app
    counts the splats inside it and marks, with red dots, the ones less than half way to the middle
    of what the box holds. "How deep" moves that limit. Look Around turns the view with the dots
    still on, to check they sit on the floater. "Erase What Is Marked" writes the boxes to
    `prune/<model>_hand_erase.json` and runs `hs prune --erase`; the result is one hand-cleaned copy
    per model (`prune/<model>_hand_erased.ply`) that later erases add to in place.
  - **Where it went.** After either clean the copy is shown with yellow dots where the removed
    splats were (read from the `_path_only` / `_erased_only` file the engine writes), with a switch
    to hide them. On an orbit they sit out where the camera stood, behind the usual view.
  - A box is the viewer's projection × view, a rectangle in its screen and a reach in metres from
    that camera; the engine runs the same test on splat centres (`clearpath.inside_boxes`), so
    what the dots show is what goes.
