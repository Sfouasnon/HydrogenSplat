# HANDOFF — COLMAP rig migration + camera-path debugging (2026-09-13)

Copied from the RED Hydrogen One research project on 2026-09-14. Companion docs in this folder: capture-to-splat-spec-v1.md (**the reproduction runbook — start there to run the pipeline**), hydrogensplat-strategy-v1.md (the app plan). The research repo is `~/Desktop/Apps/REDHydrogenOne` (not git); its `capture_video/` is the golden data — read it, never modify it.

## Repo state of REDHydrogenOne after the 2026-09-13 cleanup

Cut from ~11 GB to 900 MB; everything not behind the rig6 splat was deleted. Kept, under `capture_video/`: `VID_20260912_151351_2x1.h4v`, `frames4`, `board`, `board2`, `h1_video_stereo.*`, `video_rig6`, `video_rig6_train` (rig.npz, sweep_path.json, sweep_path_slow.json, move_boom.json), `video_rig6_train_exports/export_40000.ply` + `_pruned_r03.ply`, `compare/` (rig6_boom_12s.mp4, rig6_slow_8s.mp4, threeway_2x2.mp4, contact sheets). `mv/` holds exactly: rigcolmap.py, select_frames.py, spline_path.py, key_path.py, prune_splats.py, stereocal.py — vendored into this repo's `engine/hs/`.

## 0. Three-way comparison — rig6 wins on pose, not splat count

**rig6 training**: `export_40000.ply`, 170,841 splats. Growth curve (rig5 / rig6): 5k 53,273/74,699 · 10k 91,489/90,850 · 15k 121,787/107,198 · 20k 154,316/124,561 · 25k 191,785/143,399 · 30k 194,505/162,742 · 35k 197,189/167,903 · 40k 199,181/170,841. Growth stops at 30k in both. Final deficit 14.2%; the deficit did not matter.

**Verdict from full-res crops**: at frame 0 (mid-path) and frame 90 (one sweep extreme) rig6 is clearly better — helmet, bandolier and base lettering sharp, while rig5 is doubled/ghosted throughout. At frame 30 (other extreme) rig6 ghosts on the base lettering and pauldron edges. rig5 carries a dark floater smear on the floor that survives an r=0.3 prune; rig6 has none. Background is markedly sharper in rig6.

**The frame-30 ghost is not a local registration failure.** Per-image mean reprojection error in the rig6 solve is flat: captures 0–8 1.72 px, 8–16 1.66, 16–24 1.44, 24–34 1.53, 34–50 1.43, 50–65 1.63; L eye 1.59, R eye 1.54; worst image cap063 at 2.2 px. It is the global ~1.5 px residual showing at the edge of the sweep. Candidates, untested: rolling shutter, motion blur, the rigid rig coupling. A left-eye-only training run from the same solve would isolate the coupling.

**Path-builder trust established numerically**: the trained .ply sits on the SfM points (NN median 1.5 mm rig6, 0.8 mm rig5 → same frame, no Brush recentring); the rig6 path's mid-frame c2w is within 3.6° of real cap021_L with forward·forward = 0.9996.

**Aim point on rig6, checked by eye on cap021_L**: density cell [-4.7, 8.1, 195.0] mm → (811, 560), left pauldron edge; median [19.5, 8.3, 207.8] mm → (959, 552), chest plate. Both on subject; median is better centred and is what every rig6 path uses.

**Camera moves**: `key_path.py` — keyframed one-way move through chosen real camera positions with smoothstep ease and end holds. The per-capture azimuth/elevation table showed two low passes and one high pass; the move (top-right, boom down, slide left, settle at centre) was built as `move_boom.json` from keys 52,54,56,58,60,62,64,14,16,18,20,22 — 568 mm in 12 s, never >21 mm from a real camera. Its weak stretch is 2–4 s on the thinly covered right side.

**Next steps for the pipeline itself** (not app work): reproduce on a new clip; chase the 1.5 px residual (left-eye-only training, then a `--float-rig` re-solve); controlled masked-vs-unmasked test; board sweep clip (30 cm → 1.5 m) for the baseline; moving subjects via feed-forward stereo-pair models.

## 1. The solver

mv/rigsolve.py (left-eye-only SfM, right eyes attached afterwards, post-hoc metric fit; deleted) is superseded by mv/rigcolmap.py on pycolmap 4.2.0 and COLMAP's native rig/frame model. Both eyes are in bundle adjustment, tied by one sensor_from_rig given in metres, so the reconstruction is metric by construction.

| | rig5 (rigsolve.py) | rig6 (rigcolmap.py) |
|---|---|---|
| captures registered | 55 of 65 | 65 of 65 |
| images | 55 left + attached right | 130 in 65 rig frames |
| init points | 7,752 | 20,687 |
| metric scale | post-hoc fit, 17.11 mm/unit | by construction |
| mean reprojection | 0.59 px (left only, free per-view K, self-selected inliers) | 1.394 px (all 130, shared calibrated cameras, rigid coupling) |

Those two reprojection numbers are not comparable. BA moved the rig rotation 0.54° off the ChArUco value (~16 px at fx 1690) — that correction is the real content of the migration. CPU SIFT only; ~10 min to match 130 images exhaustively.

## 2. The camera-path bug, and how it was settled

Four wrong aim points shipped in a row: a path from a different reconstruction rendered against the wrong model (an MD5 showed two differently named .ply files were the same file); a "fix" to densest-cell aiming that crashed then moved the aim 147 mm; and the trained model's opacity-weighted densest cell, which projects off frame — a trained splat model piles opacity into bright background as readily as into the subject. What settled it: projecting each candidate into a real training image. The lasting fix: every path builder projects the chosen aim point into a real view and prints the pixel, the depth, and whether it is in frame. Read that line every time, and look at the image once — "in frame" is not "on subject".

## 3. Findings that changed the plan

a) The subject was never under-densified; what is wasted is elsewhere (37% of rig5 splats below 0.1 opacity, thousands of oversized splats far from the subject).
b) The bigger init cloud does not survive densification; likely cause `--refine-every 130`; the count deficit does not decide quality.
c) Masked training may be worse than unmasked (white spikes from the silhouette in a masked model). Suggestive, not settled.
d) The baseline is unresolved — 10.1–13.3 mm at 1σ, best fit 11.71 mm. Pure gauge in the rig formulation; corrupts only absolute distance claims.

## 4. Mistakes to not repeat

- Never ship a camera path without projecting the aim point into a real view — and look at the image.
- Check file mtimes before believing a render came from the path file the terminal names. A path rewritten after the render leaves old frames on disk under a name that looks current.
- Filtering a verification run through grep hides tracebacks. Run verification unfiltered.
- Variable shadowing in spline_path.py: lo, hi, k, d are all live.
- Always put the calibrated intrinsics in the COLMAP rig config; `--refine-intrinsics` alone collapses the reconstruction, safe only with `--float-rig`.
- Folder names do not identify a capture. MD5 the file.
- `--center -0.0009,...` fails argparse; write `--center=-0.0009,...`.
- Broadcasting an N×M×3 distance array OOM-kills at ~2000×75000 on a small machine; loop instead.

## 5. Environment

- Mac Terminal — the only place the research repo's `.venv/bin/python` works and the only place `brush` / `brush-path-render` run (arm64 Metal binaries at `~/Desktop/Apps/brush/target/release/`).
- The engine scripts need only numpy + cv2 (+ pycolmap for rigcolmap.py) and run anywhere those import.
