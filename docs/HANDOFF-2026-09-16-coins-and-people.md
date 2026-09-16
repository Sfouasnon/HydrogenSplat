# HANDOFF — coins diagnosis, floaters, and the two person clips (2026-09-16)

Companion to HANDOFF-2026-09-13-colmap-rig.md. Full evidence for the cap064/cap009 work is in the
Claude project doc `claude/cap064-cap009-diagnosis-2026-09-15.md`; this file is the working state.

## Repo (main, last pushed 6404103)
Commits since 09-14: a6f7658 masks · cade89b/50cffe1/61db7ad views metrics + `--eye R` ·
e28f580 `engine/tools/rig_consistency.py` + three_up bounded encode · 7bb5a15 project lock, safe
resume/archive/exposure dry-run, render lineage guard (incl. stale legacy prune), resume restores a
checkpoint Brush overwrites · 22e5520/a5b20fb three_up labels · 9360d05 `hs train --exclude/--no-masks`
· 6404103 three_up checks every model before rendering. Tests: `cd engine && ../.venv/bin/python -W
ignore -m unittest tests.test_lifecycle` → 23 OK.
Uncommitted: this file; `docs/Specular_Stereo_and_3DGS_Mechanisms.docx` (untracked, never added).

## Gotchas learned the hard way
- Mac system `python3` has no numpy — always `.venv/bin/python`.
- Any git command run from the Claude VM leaves `.git/index.lock`; every commit line starts `rm -f .git/index.lock`.
- `sed -i` from the VM drops the exec bit (three_up.sh "permission denied"); edit via python read/write. Git mode is now +x.
- `hs prune` names output from `str(radius)`: `--radius 10` → `_pruned_r100.ply`, `0.08` → `_r008`.
- `hs views` with `--eye R` needs its own `--name` or it overwrites the L report (default `views` → `views_R` automatically).
- `hs train` holds the project lock for its whole run; `hs render`/three_up on the same project fails until it ends.
- ffmpeg on this Mac has no `drawtext`; three_up draws labels with OpenCV. A looped still in `-filter_complex` ignores `-shortest` (the 17 h / 2.36 GB hang) — fixed with `shortest=1` + `-frames:v`.
- `splat_count_in_range` (≤ 257,576 for 70 frames) is rig6-calibrated and fails every no-mask room model; recalibrate, don't chase.

## Project 2026-09-13_coins (70 captures, 140 views, solve 1.30 px)
Archives (all `archive/<name>/export_40000.ply`):
| name | splats | md5 | notes |
|---|---|---|---|
| exposure-only | 379,307 | 8f306a81 | 140 views, no masks |
| exposure-masks | 245,085 | 7d4ec8b2 | masks; best subject, shredded room |
| exposure-excl | 385,967 | 4ef9e42e | `--exclude L/cap064,R/cap069 --no-masks`; working base |
| exposure-excl-decay8 | 282,914 | 41894797 | same + `--brush-args "--opac-decay 0.008"`; cleanest shot02, current train/exports |
prune/export_40000_pruned_r100.ply (quality prune of exposure-excl) — REJECTED, exposes rainbow SH streaks.

Findings (details in the project doc):
- L064 is a bad photo: neighbours unchanged when it is excluded; held out it is a 7.8 px near-uniform shift. Sparse PnP says its pose is fine to ~1 px — that contradiction is still unexplained. Not timing, baseline, distortion, exposure, masks.
- R069 is a pose error (bulk 2.8–3.2 px, detrended 2–4%); `rig_consistency.py` flags it. It is blind to L064-type faults.
- cap009 "displacement" is metric noise on clipped specular. cap020 R stays ~18% on every model (az −47, capture edge).
- Floaters: no statistical prune separates them from the translucent subject (median opacity 0.12 within 100 mm). opac-decay 0.008 visibly removes them (f170 ghost gone) but costs close-view fit: L062 8.5→20.2%, L063 4.5→15.0%.
- ON HOLD: `--opac-decay 0.006` midpoint run (command below). Real fix = multi-view cull of splats in front of the surface (not built).

Moves: shot01 (3.4 s) was clamped — dolly had no room, arc hit a coverage hole, a clamped arc ends the move.
shot02.hsmove rides the el +6 pass: 12.7 s, 305 mm, 54° arc, hull 24.8 mm, all cues complete.
Open move-engine bugs: disjoint radius bands near az −2 el +6 give a 4 mm/frame pop (arc right 39 hits it,
36 does not — needs continuity-preferring band choice); `dolly` targets are absolute offsets, not relative;
a clamped arc should arguably end only that cue.

## New clips (filmed 2026-09-15 on the H1, pulled to fixtures/, gitignored)
| clip | project | content | select | solve |
|---|---|---|---|---|
| VID_20260915_145111_2x1.h4v (md5 56dcc989) | Projects/2026-09-15_body | full body, ~1 orbit | 84 frames, median gap 16 (check wants ≤15) | 84/84, 1.38 px, worst 1.98, 115,227 pts, az −59…+109 (168°), el +2.9…+13.6, 1.14–1.70 m |
| VID_20260915_145235_2x1.h4v (md5 e40705de) | Projects/2026-09-15_head | head, ~3.5 loops, <1 m | 160 frames (check wants ≤120) | was still running at handoff |
Clip QA (Projects/_clipqa/ sheets): body sharpness median 0.071 vs head 0.030; motion 5.9 vs 11.9 px/frame;
clipped 0.64% vs 1.28% (head max 18.5%, LED panel). Recommendation: train body first. Head has glasses,
hair, earbud and multi-loop drift risk; body has only ~130 px of face and a flat single-height orbit.
Train estimate (fit from coins: 2,235 s + 0.0035 s/splat): 500k splats ≈ 66 min, 700k ≈ 78 min.

## Next commands
Read the head solve result first:
```
cd ~/Desktop/Apps/HydrogenSplat && .venv/bin/python -c "import json;m=json.load(open('Projects/2026-09-15_head/manifest.json'))['stages']['solve'];print(m['status']);print({k:m['metrics'].get(k) for k in ('mean_reproj_px','num_points','azimuth_range_deg','elevation_range_deg','distance_range_mm')});[print(c['name'],c['ok'],c['value']) for c in m['checks']]"
```
Train + archive the body clip (after the head solve has finished):
```
cd ~/Desktop/Apps/HydrogenSplat && P=Projects/2026-09-15_body && .venv/bin/hs train -p $P | grep -v progress && .venv/bin/hs archive -p $P --name base | grep -v progress
```
Coins decay-0.006 midpoint (on hold):
```
cd ~/Desktop/Apps/HydrogenSplat && P=Projects/2026-09-13_coins && .venv/bin/hs train -p $P --exclude L/cap064,R/cap069 --no-masks --brush-args "--opac-decay 0.006" | grep -v progress && .venv/bin/hs archive -p $P --name exposure-excl-decay6 | grep -v progress
```
