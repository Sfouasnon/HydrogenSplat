# H1 capture → splat: working spec v1 (2026-09-13)

The first end-to-end run that produced a usable splat from RED Hydrogen One footage: `capture_video/video_rig6_train_exports/export_40000.ply`, rendered along `move_boom.json` as `compare/rig6_boom_12s.mp4`. Every number below was measured on that run. Where a step is known to be weak it says so.

## 1. Capture

| | |
|---|---|
| Device / app | RED Hydrogen One, Holocam 3D video, 2x1 layout |
| Clip | `capture_video/VID_20260912_151351_2x1.h4v` — 23.5 s, 706 stereo frames, 30 fps |
| Container | plain MP4 (H.264 Baseline) despite the `.h4v` extension; ffprobe reads it directly |
| Frame | 3840×1080 = two 1920×1080 eyes side by side, hardware-synchronised |
| Subject | ~100 mm figurine on a table in a lit room, static; nothing masked |
| Distance | 209–260 mm camera-to-subject along the useful part of the path; scene depth p5 183 mm / median 213 mm / p95 1,905 mm |
| Motion | handheld, one continuous move: low pass right→left→right (captures 0–32, el −9° to −1°), rise and cross at height (34–54, el up to +26°), drop and second low pass right→left (56–64, el −12° to −7°). Azimuth −26° to +50° about the subject. Path length 1,729 mm, bounding box 339 × 136 × 121 mm |

What made it work: continuous handheld motion with real translation (not a pan), staying inside ~25 cm of the subject, and covering the same azimuths twice at two heights. What didn't matter: no lab discipline, no turntable, no masking.

## 2. Calibration (video mode — stills calibration does NOT transfer)

`capture_video/h1_video_stereo.npz`, from a ChArUco board video (SplatVizLive 8×11 DICT_4X4_100, square 15.45 mm, marker 12.36 mm; 45–60 s, 0.3–1.0 m, tilts to 30–45°, ~56 sharp frames with ≥30 corners; `stereocal.py --f-init 1865`).

| eye | fx | fy | cx | cy | k1 | k2 | p1 | p2 | rms |
|---|---|---|---|---|---|---|---|---|---|
| L | 1703.65 | 1690.98 | 910.92 | 550.79 | +0.04094 | −0.16449 | −0.00108 | −0.00288 | 0.60 px |
| R | 1688.52 | 1673.55 | 993.21 | 539.27 | +0.05008 | −0.14240 | −0.00050 | −0.00266 | 0.71 px |

Rig (R from L, calibrated): rotation quaternion [w,x,y,z] = [0.99997, −0.00160, −0.00764, 0.00028], translation −10.595 mm in x. Bundle adjustment later moves the rotation by 0.54°; that correction is real and should be kept (`--float-rig`).

**Weak number:** the baseline. Best fit 11.71 mm, 1σ band 10.1–13.3 mm; 10.64 mm is what the rig config carries. In the rig formulation this is pure gauge — it rescales the reconstruction uniformly and changes nothing about the splat or the path — but any absolute distance quoted from this pipeline carries up to ±25% until a board video walking 30 cm → 1.5 m is shot.

Video convergence is fixed across a clip (per-capture right-eye residual spread 1.63 px x / 0.74 px y). No per-capture shift is needed in video mode; stills still jump per shutter press.

## 3. Frame selection

65 stereo frames out of 706, chosen by **parallax, not optical flow**: track features from the last selected frame, fit a homography, select when the median residual after removing it crosses a threshold (a pure rotation is exactly a homography, so the residual isolates translation). Minimum gap 6 frames, maximum 90; among the 4 frames past the crossing take the sharpest with <2% clipping. Selected source frames on this clip: 0, 25, 43, 52, … 677, 685 (gaps 6–85, median 9). Output: `capture_video/frames4/VID_NNN_FFFF_2x1.jpg`.

The script that made frames4 was never committed. **`mv/select_frames.py`** (written 2026-09-13) implements the method and was calibrated against frames4 on the same clip: with its residual measured as the median over forward–backward-verified tracks after a RANSAC homography, **1.5 px at 480 wide** reproduces the selection (66 frames vs 65, median gap 9, 58 of 66 within 3 frames of the original picks). The old note of "~3.5 px" belonged to a cruder residual and gives only 35 frames with this script.

```
.venv/bin/python mv/select_frames.py capture_video/VID_20260912_151351_2x1.h4v -o capture_video/frames5
```

Runs in ~5 s per 700 frames in dry-run, ~30 s writing JPEGs; writes `selection.json` with per-frame residual, sharpness and clipping. Do not add a flow-magnitude escape hatch — it reintroduces the pan failure.

## 4. Structure from motion — `mv/rigcolmap.py` (pycolmap 4.2.0)

```
cd ~/Desktop/Apps/REDHydrogenOne
.venv/bin/python mv/rigcolmap.py prep capture_video/frames4 -o capture_video/video_rig6
.venv/bin/python mv/rigcolmap.py sfm capture_video/video_rig6 --calib capture_video/h1_video_stereo.npz --float-rig
.venv/bin/python mv/rigcolmap.py export capture_video/video_rig6/sparse/rig --images capture_video/video_rig6/images -o capture_video/video_rig6_train
```

`prep` splits each 2x1 frame into `images/L/capNNN.jpg` and `images/R/capNNN.jpg` (1920×1080 each). `sfm` runs CPU SIFT (`--peak-threshold 0.0025`, ~2,700–3,000 features per frame), exhaustive matching (8,385 pairs, ~10 min), writes the rig config from the calibration (both cameras OPENCV with the calibrated intrinsics — mandatory, or registration collapses), then rig-aware incremental mapping with `camera_mode=PER_FOLDER`.

Result (`video_rig6/sfm_report.json`): 65/65 frames registered, 130 images, 20,687 points, mean reprojection 1.394 px. Per-image error is flat across the clip, 1.3–2.2 px (L eye 1.59, R eye 1.54 after undistortion). Every L–R separation exactly 10.642 mm — the rig constraint held.

Do not use `--refine-intrinsics` without `--float-rig` (it collapses the reconstruction) and it buys ~0.03 px with it. Leave it off.

`export` undistorts to two PINHOLE cameras (L 1913×1073, R 1909×1071 — COLMAP crops each to its own valid region; Brush handles per-image intrinsics), writes a COLMAP dataset under `video_rig6_train/` and `rig.npz` (views interleaved L,R in capture order, mm) plus `sparse/points3D.ply` for the path builders.

## 5. Training — Brush

```
~/Desktop/Apps/brush/target/release/brush ~/Desktop/Apps/REDHydrogenOne/capture_video/video_rig6_train --total-train-iters 40000 --growth-stop-iter 30000 --refine-every 130
```

Exports every 5k to `capture_video/video_rig6_train_exports/`. ≈55 min on the Mac (5k iterations per ~6.5 min while growing; faster after 30k). Splat count: 5k 74,699 · 10k 90,850 · 15k 107,198 · 20k 124,561 · 25k 143,399 · 30k 162,742 · 35k 167,903 · **40k 170,841**.

`--refine-every 130` probably cost splats (the rigsolve model at the default 200 reached 199,181) but the render is better anyway; the count is not what decides quality here. `--split-at-screen-size` stays at its 0.5 default for a room scene. No masks: the one masked model tried so far had white spikes from the silhouette; masked-vs-unmasked is untested under controlled conditions.

The trained .ply is in the SfM frame — splats sit on the SfM points to ~1.5 mm; Brush does not recentre. Camera paths built from `rig.npz` apply directly.

## 6. Optional prune

```
.venv/bin/python mv/prune_splats.py capture_video/video_rig6_train_exports/export_40000.ply capture_video/video_rig6_train_exports/export_40000_pruned_r03.ply --center=0.0195,0.0083,0.2078 --radius 0.3 --max-scale 0.2
```

Filters opacity <0.05, anisotropy >50, scale ≥0.2, then keeps everything within 0.3 (metres, nominal) of the given centre → 52,659 splats. **Give `--center` explicitly** (the SfM median in metres; note the `=` — a leading minus is otherwise parsed as a flag). The script's own opacity-weighted centre lands on bright background as readily as on the subject. Pruning removes the room; for a wiggle the room's parallax is worth keeping, so the unpruned model was used.

## 7. Camera path

Aim point = **median of points3D** (`--aim-median`), on this capture the chest plate, 230 mm from the mid camera. Every builder prints an aim check — the pixel the aim point projects to in a real view and whether it is in frame. Read it, then look at that image once: "in frame" is not "on subject". Four wrong aim points shipped before that rule existed.

Loop (sinusoid over a window of captures):

```
.venv/bin/python mv/spline_path.py capture_video/video_rig6_train/rig.npz -o capture_video/video_rig6_train/sweep_path_slow.json --captures 8:34 --aim-median --frames 240
```

One-way keyframed move through chosen real camera positions, eased, with holds (`mv/key_path.py`, new today):

```
.venv/bin/python mv/key_path.py capture_video/video_rig6_train/rig.npz -o capture_video/video_rig6_train/move_boom.json --keys 52,54,56,58,60,62,64,14,16,18,20,22 --aim-median --frames 360 --hold 0.5
```

That move: start top-right (az 39°, el 15°), boom down the right side to el −11°, slide left along the low pass through centre to az −24°, arc back and settle at centre. 568 mm in 12 s, peak 78 mm/s. The builders keep the virtual camera within a few mm of a real one (spline: 1–13 mm; keyed move: 0–21 mm, worst at the 64→14 stitch). Stay on the captured hull — a path that leaves it is invention, not interpolation.

## 8. Render and encode

```
~/Desktop/Apps/brush/target/release/brush-path-render capture_video/video_rig6_train_exports/export_40000.ply --path capture_video/video_rig6_train/move_boom.json -o capture_video/rig6_boom --width 2400
```

~14 frames/s at 2400 wide on the Mac. Path JSON: `width`, `height`, `K`, `fps`, `frames[].c2w` (4×4, OpenCV convention x right / y down / z forward, metres). Encode (Linux VM or Mac with ffmpeg):

```
ffmpeg -framerate 30 -i capture_video/rig6_boom/frame_%04d.png -vf scale=1920:-2 -c:v libx264 -pix_fmt yuv420p -crf 17 -movflags +faststart capture_video/compare/rig6_boom_12s.mp4
```

## 9. Known limits of v1

- **1.5 px residual reprojection error** is the quality ceiling. It shows as ghosting/doubling on the thinly covered right side (az ≳ 40°) and on the finest detail (base lettering). Not local — flat across all captures — so it is rolling shutter, motion blur, or the rigid rig coupling. Untested which. Left-eye-only training from the same solve is the cheap discriminator.
- Baseline gauge ±25% on absolute distances (§2).
- 1920×1080 per eye is the resolution; a 4:5 portrait crop is 864×1080.
- Static subjects only. Sequential SfM cannot do people or animals; that needs the feed-forward stereo-pair route.
- Masks untested; `rigcolmap.py sfm --masks DIR` is wired but unverified.
- `select_frames.py` is calibrated to reproduce one selection on one clip; its threshold has not yet been tested on a second capture.

## 10. Runbook — reproduce on a new clip

All commands from the Mac Terminal after `cd ~/Desktop/Apps/REDHydrogenOne`. Replace `CLIP` with the `VID_*_2x1.h4v` name and `NAME` with a working name (e.g. `rig7`). Expected outputs are what this run produced; a new subject will differ but should be in range.

Shoot: stereo video mode, 20–30 s, static subject, camera 20–30 cm from a ~10 cm subject (scale up proportionally), keep moving with real translation — arcs and a height change, no pans — and cover the same azimuths twice at two heights. Avoid ending the clip with the fastest move; the tail is where registration is weakest.

```
# 1. pull the clip off the phone
adb pull /sdcard/DCIM/Camera/CLIP capture_video/

# 2. select frames by parallax  (expect ~3 frames per second of clip, gaps 6–90)
.venv/bin/python mv/select_frames.py capture_video/CLIP -o capture_video/frames_NAME

# 3. split eyes, solve as a rig, export the training set  (~15 min for 65 captures, CPU)
.venv/bin/python mv/rigcolmap.py prep capture_video/frames_NAME -o capture_video/video_NAME
.venv/bin/python mv/rigcolmap.py sfm capture_video/video_NAME --calib capture_video/h1_video_stereo.npz --float-rig
.venv/bin/python mv/rigcolmap.py export capture_video/video_NAME/sparse/rig --images capture_video/video_NAME/images -o capture_video/video_NAME_train
```

Check `capture_video/video_NAME/sfm_report.json`: num_frames should equal the frame count, mean_reproj_px ≈ 1.3–1.5. Fewer frames registered than selected means the chain broke — usually a fast move; re-select with `--max-gap 45` or trim the clip with `--end`.

```
# 4. train  (~55 min; exports every 5k into capture_video/video_NAME_train_exports/)
~/Desktop/Apps/brush/target/release/brush ~/Desktop/Apps/REDHydrogenOne/capture_video/video_NAME_train --total-train-iters 40000 --growth-stop-iter 30000 --refine-every 130

# 5. build a path, READ THE AIM CHECK LINE, then open the named view and confirm the pixel is on the subject
.venv/bin/python mv/spline_path.py capture_video/video_NAME_train/rig.npz -o capture_video/video_NAME_train/sweep_path.json --captures 8:34 --aim-median --frames 240
```

To design a keyframed move, print the per-capture azimuth/elevation table first (the snippet is in HANDOFF-2026-09-13-colmap-rig.md §0, or ask for it), pick capture indices, then:

```
.venv/bin/python mv/key_path.py capture_video/video_NAME_train/rig.npz -o capture_video/video_NAME_train/move.json --keys 52,54,56,58,60,62,64,14,16,18,20,22 --aim-median --frames 360 --hold 0.5
```

Keep "virtual cameras sit … from the nearest real camera" under ~25 mm.

```
# 6. render and encode
~/Desktop/Apps/brush/target/release/brush-path-render capture_video/video_NAME_train_exports/export_40000.ply --path capture_video/video_NAME_train/move.json -o capture_video/NAME_move --width 2400
ffmpeg -framerate 30 -i capture_video/NAME_move/frame_%04d.png -vf scale=1920:-2 -c:v libx264 -pix_fmt yuv420p -crf 17 -movflags +faststart capture_video/NAME_move.mp4
```

Before trusting any render: MD5 the .ply you rendered against the export you meant, and check the path file's mtime is older than the frames. Both bit this run.

## 11. Environments

Mac Terminal: `.venv/bin/python` (numpy, scipy, cv2, pycolmap), `brush`, `brush-path-render` — the only place the binaries run. `cd ~/Desktop/Apps/REDHydrogenOne` first. Linux VM (device_bash): system python3 with numpy/cv2/ffmpeg, no scipy; runs `select_frames.py`, `spline_path.py`, `key_path.py`, `prune_splats.py`, ffmpeg; 175 s per call; cannot delete files. Cloud container: x86_64, pycolmap, network; can run the full SfM if images are staged in.
