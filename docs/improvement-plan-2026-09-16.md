# HydrogenSplat — improvement plan from the Brush / 3DGS research notes (2026-09-16, rev 2)

Source: external research notes pasted 2026-09-16, checked against the repo, the 09-16 handoff,
the two 2026-09-15 person models, and (rev 2) an audit of the local Brush checkout on the Mac.
Where the notes and the measurements disagree, the measurement wins and the disagreement is
called out.

**Rev 2 changes:** subject mapping confirmed; Brush audited and updated (§2.3); `--subject-mm` on
every `hs views` command (§3); head re-select tested and deferred (§5.1); run order reworked so no
step throws away an earlier one (§6); decisions 1 and 3 closed (§7).

**Subjects (confirmed):** A = `Projects/2026-09-15_body` (room visible, shelving 5–10 m behind).
B = `Projects/2026-09-15_head` (hair, glasses, earbud).

**Software state.** Brush: `~/Desktop/Apps/brush`, branch `hs` = `fbdebfb3` (brush-path-render)
on upstream `5f6e45cc` (#546). HydrogenSplat: `main` @ 44416a1 plus uncommitted work (M1 app,
`hs phone` / `replay` / `grade`, framing, keep-awake, brush config recording, this plan).

---

## 0. TL;DR

| Subject | Primary diagnosis | First moves |
|---|---|---|
| A — body | Far background is under-constrained (little parallax at 5–10 m), not a global pose error. Possibly also renderer sort-popping. | 1) render-popping A/B · 2) background-noise A/B (free) · 3) per-view appearance (PR #483, open) · 4) background-only depth loss (PR #497, open) |
| B — head | Source detail is the limit: hair, glasses rims and earbud are all soft. | 1) growth schedule back to Brush's default (15k) · 2) `--split-at-screen-size` and `--min-scale-factor` sweeps · 3) AbsGS · 4) region-scored frame selection (then re-select) |

Both models are **growth-schedule-bound, not signal-bound** (§2.2). "Densify more" is the wrong
direction.

Before any of it: **there is no held-out evaluation.** Every `hs views` number so far is measured
on training views. Build the fixed hold-out first (§3), on the **updated** Brush.

Operational rules for every run below: **plugged in, lid open** (the 09-15 head train slept ~14.5 h
on battery; `hs` now reports sleeps but can't prevent them on battery). One mechanism per run.
Archive every run under a descriptive `--name`.

---

## 1. Current state (measured)

### 1.1 Training runs (old Brush, `063945e7` / #534)

| | body (A) | head (B) |
|---|---:|---:|
| captures / views | 84 / 168 | 160 / 320 |
| masks used | no | no |
| final splats | 1,139,613 | 1,413,218 |
| awake train time | 5,068.4 s (84.5 min) | 6,525.3 s (108.8 min) |
| hs schedule | 40k iters, growth stop 30k, refine every 130 | same |
| 3D filter | none (pre-#541) | none |

Both are archived as `archive/base`. They stay valid for rendering (§2.3: the new renderer draws
them at 56.3 dB PSNR against the old one), but they are **not** a controlled baseline for new
training runs — the trainer changed (#541, #545, #546).

### 1.2 Splat density vs the rig6 reference

Reference: 170,841 / 65 = **2,628.32 splats per registered frame**.

| | splats / frame | × reference |
|---|---:|---:|
| body | 1,139,613 / 84 = 13,566.82 | **5.162×** |
| head | 1,413,218 / 160 = 8,832.61 | **3.361×** |

The notes say "~4×". The `splat_count_in_range` check (0.6–1.4×) is rig6-calibrated and fails
both — recalibrate, don't chase.

### 1.3 In-sample view metrics (`hs views`, 4 views each, all training views)

Scored with `--subject-mm 2000` (body) and `--subject-mm 350` (head).

| view | depth mm | src sharp | retained edge | PSNR dB | displaced | p90 px |
|---|---:|---:|---:|---:|---:|---:|
| body cap000_L | 1102.7 | 0.0940 | 0.553 | 20.92 | 0.058 | 2.93 |
| body cap071_L | 1462.5 | 0.0553 | 0.466 | 23.27 | 0.149 | 4.74 |
| body cap041_L | 1662.1 | 0.0360 | 0.502 | 27.54 | 0.047 | 3.12 |
| body cap064_L | 1093.7 | 0.0492 | 0.556 | 23.04 | 0.178 | 5.61 |
| head cap132_L | 464.3 | 0.0506 | 0.465 | 24.00 | 0.039 | 2.28 |
| head cap004_L | 548.3 | 0.0594 | 0.386 | 22.12 | 0.094 | 3.91 |
| head cap108_L | 822.2 | 0.0347 | 0.512 | 28.35 | 0.100 | 3.86 |
| head cap131_L | 467.2 | 0.0585 | 0.410 | 22.99 | 0.053 | 2.60 |

Means: retained edge energy body 0.519, head 0.443 — on views the models were trained on.

### 1.4 Clip QA (handoff)

Sharpness median body 0.071 vs head 0.030 (head = 42.3% of body); inter-frame motion 5.9 vs
11.9 px/frame; clipped 0.64% vs 1.28% (head max 18.5%, LED panel).

### 1.5 Head: is the softness movement between orbit passes? (2026-09-16, no)

Head points (< 130 mm from the head centre) triangulated separately from each pass past a
direction (front 4 passes, sides 3 each): median shift between passes 0.2–1.2 mm, per-point
scatter 1.5–2.7 mm (triangulation noise). Worst 1.2 mm at ~650 mm ≈ 3.1 px, typical ≈ 1.3 px.
R-only vs L-only triangulation differs by 0.7–1.4 mm, the same size. Far points (1.5–4 m) move
7–53 mm between passes and 11–14 mm between eyes — largely single-pass depth noise, consistent
with the background smear. Limit: hair and rims yield no matched features, so sub-mm movement
there can't be excluded. The doubled look on glasses/ear/hair is translucent edge splats plus
refraction through the lenses, exaggerated by a heavy grade (lift −0.133, sharpen 0.52).

---

## 2. Where the notes need correcting or qualifying

### 2.1 Numbers that don't match

- **"~35% softer" (head vs body).** Clip QA says ~58% lower (0.030 / 0.071 = 0.423); the 4
  in-sample views say ~13% lower (0.0508 / 0.0586 = 0.867). No measurement gives 35%.
- **"~12 px/frame motion blur."** 11.9 px/frame is inter-frame displacement, not blur width.
  Blur ≈ displacement × shutter duty cycle: 5.95 px at a 1/60 s shutter, 2.98 px at 1/120 s. The
  shutter **can't be read from the clip**: the H1 writes no per-frame exposure metadata (strategy
  §4.3) and the probe tags carry only leia3d / OS fields. Estimate blur from the frames instead
  (edge width vs per-frame motion).
- **"~4× splats per frame."** 5.16× / 3.36× (§1.2).

### 2.2 Growth is schedule-bound

| | splats at ~15k | share of final | growth per refine at 29.4k→29.9k | added after 30k |
|---|---:|---:|---:|---:|
| body | 799,056 | 70.1% | ~3,388 / 130 iters | 3,039 (0.27%) |
| head | 972,650 | 68.8% | ~4,076 / 130 iters | 9,132 (0.65%) |

Neither curve flattens before the 30k stop. **Brush's own default is `--growth-stop-iter 15000`**;
`hs train` overrides it to 30000 (rig6 settings).

### 2.3 Brush audit (verified 2026-09-16)

`origin` is upstream (`ArthurBrussee/brush`). The checkout was at `063945e7` (#534), 9 commits
behind; `brush-path-render` was untracked with uncommitted `Cargo.toml` / `Cargo.lock` edits.
Now: branch `hs` = the renderer commit rebased onto `5f6e45cc`; `main` = upstream. Only
`Cargo.lock` conflicted (resolved by taking upstream's and letting cargo add the renderer's 19
lines — never `cargo generate-lockfile`, which re-resolves every pin).

| PR | state | what it means here |
|---|---|---|
| #541 (dd5ea36) | merged 2026-09-13 | adds `--min-scale-factor` and `--growth-start-iter` |
| #545 | merged | `--units-per-meter`, default 1.0. HydrogenSplat datasets are already metres (sparse points = rig.npz mm / 1000), so nothing to change |
| #546 | merged | burn fuses the training step — speed and possibly results change |
| #483 appearance (bilateral grid + PPISP) | **open**, last updated 2026-07-21 | predates #545/#546: expect conflicts; try on its own branch |
| #497 depth loss | **open**, last updated 2026-07-27 | same |

Flags in the updated build (`brush --help`):

| flag | default | note |
|---|---|---|
| `--min-scale-factor` | **0.1 (on)** | Mip-Splatting 3D filter: per-splat world-space floor `sqrt(s) · pixel size at the nearest observing camera` = 0.316 px std-dev at 0.1. Baked at export, never optimized. 0 = off = pre-#541 behaviour. (The notes' "≈ 0.32 px" is confirmed.) |
| `--growth-start-iter` | 0 | clamped to growth stop |
| `--growth-stop-iter` | 15000 | hs passes 30000 |
| `--units-per-meter` | 1.0 | correct for our datasets |
| `--background-color` | 0,0,0 | |
| `--background-noise-strength` | **0.1** | uniform noise on the background each step (§4.2) |
| `--split-at-screen-size`, `--growth-select-fraction`, `--opac-decay`, `--mean-noise-weight`, `--lod-*` | present | `--lod-*` unused by hs |
| bilateral / ppisp / depth | absent | #483 / #497 |

Renderer check: the rebuilt `brush-path-render` re-rendered head `arc180h` from the unchanged model
at PSNR 56.3 dB average (min 50.4, some frames identical) against the old build. Old and new
renders are comparable.

`hs train` now records the Brush checkout (`tools.brush`: commit, branch, dirty flag, binary mtime)
and a `brush_config` metric, and passes `--min-scale-factor` explicitly (default 0.1) when the
binary has it, so every manifest shows the filter strength. On a pre-#541 binary it records 0 and
refuses an explicit value.

### 2.4 Things the notes get right that match repo history

- Masks-as-loss-exclusion → "clean subject, shredded room" is the coins `exposure-masks` result.
- `--opac-decay 0.008` too destructive: coins L062 8.5→20.2%, L063 4.5→15.0%.
- Brush already does MCMC-like relocation, so porting 3DGS-MCMC is low value.

---

## 3. Prerequisite: a fixed hold-out evaluation (do first, on the updated Brush)

Hold out every 10th capture starting at 5, **both eyes** (the other eye, 10.6 mm away, would leak
the answer). Head: 16 captures / 32 views (10%). Body: 8 captures / 16 views (9.5%). Scores always
use the same crop sizes as §1.3: `--subject-mm 350` (head), `--subject-mm 2000` (body) — without
them the crop is measured from the room-wide point cloud and the numbers aren't comparable.

The baseline uses the updated Brush with the current hs schedule (30k growth stop) and the new
default filter (`--min-scale-factor 0.1`, now recorded). It is the reference for everything after.

Head baseline + hold-out report (L and R), ~110 min:

```
cd ~/Desktop/Apps/HydrogenSplat && P=Projects/2026-09-15_head && EX=$(python3 -c "print(','.join(f'{e}/cap{i:03d}' for i in range(5,160,10) for e in 'LR'))") && CAP=$(python3 -c "print(','.join(str(i) for i in range(5,160,10)))") && .venv/bin/hs train -p $P --exclude $EX | grep -v progress && .venv/bin/hs archive -p $P --name holdout-base | grep -v progress && .venv/bin/hs views -p $P --captures $CAP --subject-mm 350 --name holdout | grep -v progress && .venv/bin/hs views -p $P --captures $CAP --subject-mm 350 --eye R --name holdout_R | grep -v progress
```

Body baseline (after the head finishes — `hs train` holds the project lock and two Brush runs
fight for the GPU), ~85 min:

```
cd ~/Desktop/Apps/HydrogenSplat && P=Projects/2026-09-15_body && EX=$(python3 -c "print(','.join(f'{e}/cap{i:03d}' for i in range(5,84,10) for e in 'LR'))") && CAP=$(python3 -c "print(','.join(str(i) for i in range(5,84,10)))") && .venv/bin/hs train -p $P --exclude $EX | grep -v progress && .venv/bin/hs archive -p $P --name holdout-base | grep -v progress && .venv/bin/hs views -p $P --captures $CAP --subject-mm 2000 --name holdout | grep -v progress && .venv/bin/hs views -p $P --captures $CAP --subject-mm 2000 --eye R --name holdout_R | grep -v progress
```

Every experiment replaces `train/exports`. To render the old models afterwards, pass
`--ply Projects/<p>/archive/base/export_40000.ply` (the lineage guard compares rig.npz, which is
unchanged).

**Gaps in `hs views` for this plan** (small engine work):

1. **Region split.** Retained edge energy / displacement for subject vs far background (body) and
   hair / glasses-ear / silhouette strip (head).
2. **Temporal stability metric** for camera-path renders (§4.1).
3. **Aggregate row** (median over held-out views) so runs compare as one line each.
4. **Per-pass reporting on the head** — the hold-out captures fall in 4 front passes and 3 per
   side; report each pass separately so a pass-specific fault shows.

---

## 4. Subject A (body) — far-background instability

| # | Intervention | Value | Effort | Where |
|---|---|---|---|---|
| 1 | Render-popping diagnostic | decides the branch | small | hand-written move + metric |
| 2 | `--background-noise-strength 0` A/B | unknown, free | none | `--brush-args` |
| 3 | Per-view appearance (PR #483) | high | medium (stale PR) | Brush branch |
| 4 | Background-only depth loss (PR #497 + detach fix) | very high | medium–high | preprocessing + Brush |
| 5 | Two-model fg/bg + compositor | high | medium | masks stage + render |
| 6 | Multi-run disagreement as confidence (CoR-GS principle) | medium | small–medium | two seeds + diff tool |
| 7 | Rig-locked pose refinement | low here | medium | Brush or pycolmap |
| — | Scene contraction, env map / skybox | low | — | skip |

### 4.1 Render-popping vs geometry (first)

Does the shimmer come from geometry that moves, or from approximate per-tile sorting (StopThePop)?

Test: one model; ~100 frames of sub-degree rotation / sub-mm translation; for background pixels
measure (a) frame-to-frame temporal variance and (b) tracked positions of strong shelf edges
against the known path.

- Edges follow the path but pixel values flicker → **renderer**. Port a StopThePop-style
  hierarchical sort to WGSL.
- Edges wander → **geometry**. Go to 4.3–4.5.

**`hs move` can't make this path** — the presets and `.hsmove` cues don't go below mm/degree
steps. Write the move JSON directly (same format as `move/*.json`: `width`, `height`, `K`, `fps`,
`frames[].c2w` in metres, OpenCV convention): 100 copies of one real capture's pose with a
0.02°/frame yaw step. Render it with `hs render --move <path.json>`. The temporal metric is still
to build.

### 4.2 Background noise A/B (free)

Every run so far trained with Brush's default `--background-noise-strength 0.1`: noise on the
background colour each step, which pushes splats to cover any pixel the photograph doesn't explain.
On a room capture a real background is always there, so the effect is probably small — but it is
a single flag and costs one run.

```
cd ~/Desktop/Apps/HydrogenSplat && P=Projects/2026-09-15_body && EX=$(python3 -c "print(','.join(f'{e}/cap{i:03d}' for i in range(5,84,10) for e in 'LR'))") && CAP=$(python3 -c "print(','.join(str(i) for i in range(5,84,10)))") && .venv/bin/hs train -p $P --exclude $EX --brush-args "--background-noise-strength 0" | grep -v progress && .venv/bin/hs archive -p $P --name holdout-bgnoise0 | grep -v progress && .venv/bin/hs views -p $P --captures $CAP --subject-mm 2000 --name holdout_bgnoise0 | grep -v progress
```

### 4.3 Per-view appearance compensation (PR #483)

A per-view scalar gain (`hs exposure`) can't correct tone curve, channel-wise WB, highlight
roll-off or lens shading, and the H1 ISP varies all of them. Mismatched photometry is cheaper for
the optimizer to "explain" with duplicated/offset background Gaussians.

The PR is open and was last updated 2026-07-21, before #545/#546 — expect conflicts. Merge it on
its own branch off `hs`, never on `hs` itself:

```
cd ~/Desktop/Apps/brush && git switch hs && git fetch origin pull/483/head:pr-483 && git switch -c hs-bilateral && git merge --no-ff pr-483; git status --short | head -20
```

If it merges and builds, the flag name comes from `--help` (the notes' `--bilateral-grid` is
unverified). Change only this in the run. Appearance state isn't in checkpoints (matters for
`--resume-from`); novel views render with canonical colour (what HydrogenSplat wants). Run once
with and once without `hs exposure` (decision 4).

### 4.4 Background-only depth regularization

`L = L_RGB + λ_d · w(x) · ρ(1/d̂(x) − 1/d_mono(x))`, `w ≈ 0` on the subject, hair, silhouette strip
and clipped pixels, high only on confident far background.

New `hs depth` stage:

1. **Depth Anything V2 Small** (Apache-2.0, Core ML path) per view; Depth Pro / MoGe-2 as
   alternates. DAV2 Base/Large are CC-BY-NC — don't use them.
2. Per-view scale/shift on inverse depth against the COLMAP sparse points (robust fit).
3. `w`: complement of a dilated subject mask × agreement with sparse points × not clipped.
4. Float32 TIFFs where PR #497 expects them.
5. Brush: PR #497 (open, last updated 2026-07-27; expect conflicts) **plus** the reviewed fix that
   detaches the blending weights. Don't merge #497 as-is.

The body project has no masks (`masks_used: false`); build them before `w`.

### 4.5 Two models: subject + background

Subject model: subject masks, full resolution, normal growth. Background model: complementary
masks, lower resolution and SH degree, depth loss, smaller budget. Composite by rendered alpha /
depth. Needs brush-path-render to emit alpha + depth (or a two-pass compositor in `hs render`) and
the lineage guard extended to a pair of models.

### 4.6 Disagreement as confidence

Train the body twice with different seeds; depths that disagree at the hold-out poses mark
unreliable background → prune candidates for the floater problem (coins: "real fix = multi-view
cull", unbuilt) and an extra down-weight in `w`. ~2 × 85 min.

### 4.7 Pose refinement — deprioritised

The subject registers well (§1.5: < 1.2 mm between passes). If pursued: one SE(3) per stereo
timestamp, both eyes, 10.64 mm baseline rigid. `rig_consistency.py` flags R069-type errors;
L064-type faults remain unexplained and pose refinement won't fix them.

---

## 5. Subject B (head) — soft hair, glasses, earbud

| # | Intervention | Value | Effort | Where |
|---|---|---|---|---|
| 1 | Growth schedule to Brush default (stop 15k, refine 200) | diagnostic | none | `hs train` flags |
| 2 | `--split-at-screen-size` sweep | medium | none | already plumbed |
| 3 | `--min-scale-factor` sweep (0 / 0.03 / 0.1) | medium | none | now plumbed |
| 4 | AbsGS absolute-gradient densification | high | medium | Brush backward/refine |
| 5 | Region-scored frame selection, then re-select | high if the scorer shows a gain | small + re-solve | `hs select` |
| 6 | Error/edge-guided densification statistic | medium–high | medium | Brush |
| 7 | Taming-3DGS budgeted growth | medium | medium | Brush |
| — | 3DGS-MCMC, Mini-Splatting | low | — | skip |

### 5.1 Frame selection — tested, deferred

`select_frames.py` into `/tmp` (never `hs select --dry-run` on the live project: it overwrites
`selection.json` and the select metrics), head clip, `--search 8`:

| selection | frames | sharpness median | vs current | median gap (max) |
|---|---:|---:|---:|---|
| current (residual 1.5, search 4) | 160 | 952.1 | — | 12 (41) |
| residual 1.5, search 8 | 133 | 1012.6 | +6.4% | 14.5 (47) |
| residual 2.0, search 8 | 119 | 964.8 | +1.3% | 16 (48) |
| residual 2.5, search 8 | 109 | 1025.1 | +7.7% | 17.5 (48) |

The gain is within noise (2.0 came out less sharp than 1.5) and nowhere near the 58% clip gap.
The score is whole-frame and dominated by the LED panel and background, so it can't say whether
hair got sharper. A wider search does cut frames (each later pick delays the next crossing), and
109 frames would cut matching to (109/160)² = 0.464 of the 52:37 solve — but a re-select costs a
re-solve, a new baseline and new moves.

**Decision:** keep the current selection and solve for the hold-out work. Build a region-scored
sharpness (head/hair crop, glasses/ear crop, silhouette strip) first; re-select only if it shows a
real difference, and then before training a new baseline.

### 5.2 Schedule diagnostic

Brush's own default growth stop is 15k. Tests whether the extra ~0.4–0.6M splats carry detail:

```
cd ~/Desktop/Apps/HydrogenSplat && P=Projects/2026-09-15_head && EX=$(python3 -c "print(','.join(f'{e}/cap{i:03d}' for i in range(5,160,10) for e in 'LR'))") && CAP=$(python3 -c "print(','.join(str(i) for i in range(5,160,10)))") && .venv/bin/hs train -p $P --exclude $EX --growth-stop-iter 15000 --refine-every 200 | grep -v progress && .venv/bin/hs archive -p $P --name holdout-g15r200 | grep -v progress && .venv/bin/hs views -p $P --captures $CAP --subject-mm 350 --name holdout_g15r200 | grep -v progress
```

If hold-out retained edge energy and PSNR are within noise of `holdout-base` at ~⅔ the splats,
the extra growth is waste and later runs use this schedule.

### 5.3 AbsGS (the densification change to build)

Signed screen-space gradients from different pixels/views cancel, so a large Gaussian straddling
hair looks like it doesn't need splitting. AbsGS accumulates `|∂L/∂x_screen|`, `|∂L/∂y_screen|`
instead. gsplat Mip-NeRF360 (CUDA, not predictive for us): 29.00 dB / 0.14 LPIPS / 3.24M →
29.11 dB / 0.12 LPIPS / 2.47M with absgrad.

Brush work: accumulate the per-Gaussian absolute 2D gradient in the rasterizer backward
(`crates/brush-render/src/bwd/…`, heavily changed by #546 — build on `hs`), switch the refine
statistic behind a flag, raise the growth threshold to match. Run with the §5.2 schedule.

### 5.4 Screen-space split sweep

`--split-at-screen-size 0.5` rarely fires on the mid-size blobs that smear hair. Log the
distribution of visible Gaussians' major-axis screen extent in the hair region first, then set the
threshold near its upper tail. Bracket until then:

```
cd ~/Desktop/Apps/HydrogenSplat && P=Projects/2026-09-15_head && EX=$(python3 -c "print(','.join(f'{e}/cap{i:03d}' for i in range(5,160,10) for e in 'LR'))") && CAP=$(python3 -c "print(','.join(str(i) for i in range(5,160,10)))") && for S in 0.25 0.1; do .venv/bin/hs train -p $P --exclude $EX --growth-stop-iter 15000 --refine-every 200 --split-at-screen-size $S | grep -v progress && .venv/bin/hs archive -p $P --name holdout-split$S | grep -v progress && .venv/bin/hs views -p $P --captures $CAP --subject-mm 350 --name holdout_split$S | grep -v progress; done
```

### 5.5 `--min-scale-factor` sweep (now runnable)

0.1 is Brush's default and the baseline's value; 0 reproduces the old models. Anti-aliasing, not
hair recovery: likely helps the body background, may hurt already-soft head edges. Judge on
hold-out edge energy **and** temporal stability.

```
cd ~/Desktop/Apps/HydrogenSplat && P=Projects/2026-09-15_head && EX=$(python3 -c "print(','.join(f'{e}/cap{i:03d}' for i in range(5,160,10) for e in 'LR'))") && CAP=$(python3 -c "print(','.join(str(i) for i in range(5,160,10)))") && for M in 0 0.03; do .venv/bin/hs train -p $P --exclude $EX --growth-stop-iter 15000 --refine-every 200 --min-scale-factor $M | grep -v progress && .venv/bin/hs archive -p $P --name holdout-msf$M | grep -v progress && .venv/bin/hs views -p $P --captures $CAP --subject-mm 350 --name holdout_msf$M | grep -v progress; done
```

(Compare against `holdout-g15r200`, which is the 0.1 point on the same schedule.)

### 5.6 Later

- **Edge/error-guided densification:** weight the growth statistic (not the RGB loss) by
  `1 + λ|∇I_gt|` or persistent residual.
- **Taming-3DGS budgets:** port the scoring/budget logic, not the CUDA speedups. After AbsGS.

---

## 6. Run order and cost

Costs from the measured old-Brush runs; #546 changes speed, so the first new run re-measures it.

| Step | Run | Change | Est. time |
|---|---|---|---:|
| ✓ | Brush audit + update + renderer check | — | done |
| ✓ | head re-select test | — | done, deferred (§5.1) |
| 1 | head `holdout-base` | hold-out only, new Brush | ~110 min |
| 2 | body `holdout-base` | same | ~85 min |
| 3 | head `holdout-g15r200` | schedule | ≤ 110 min |
| 4 | head `holdout-split0.25`, `-split0.1` | 1 flag each | 2 runs |
| 5 | head `holdout-msf0`, `-msf0.03` | 1 flag each | 2 runs |
| 6 | body `holdout-bgnoise0` | 1 flag | ~85 min |
| 7 | body popping test (§4.1) | render only | < 30 min + tooling |
| 8 | `hs views` region split + aggregate + per-pass | code | — |
| 9 | #483 on `hs-bilateral` → body run | PR merge | build + ~85 min |
| 10 | region-scored select → (maybe) head re-select, re-solve, new baseline | frames | solve + ~110 min |
| 11 | Brush: AbsGS | code | — |
| 12 | `hs depth` + #497 (+ fix) → body depth run | code | — |
| 13 | two-model fg/bg | code | — |

Disk: each archive is 343–519 MB; ~10 runs ≈ 4–5 GB (the Mac had 674 GB free).

---

## 7. Decisions

1. ~~A = body, B = head~~ — **confirmed.**
2. Hold-out: every 10th capture from 5, both eyes — proposed; add per-pass reporting on the head.
3. ~~Brush fork strategy~~ — **done:** `hs` branch (renderer only) rebased on upstream `main`;
   open PRs go on their own branches off `hs`.
4. Whether `hs exposure` stays on once #483 is in — open; test both.

## 8. Sources cited by the notes

- Brush PRs: [#483 appearance](https://github.com/ArthurBrussee/brush/pull/483) ·
  [#497 depth](https://github.com/ArthurBrussee/brush/pull/497) ·
  [#353 multi-view densification](https://github.com/ArthurBrussee/brush/pull/353) (state not checked) ·
  [#541](https://github.com/ArthurBrussee/brush/pull/541) (merged, verified)
- [DNGaussian (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Li_DNGaussian_Optimizing_Sparse-View_3D_Gaussian_Radiance_Fields_with_Global-Local_Depth_CVPR_2024_paper.html)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) ·
  [Depth Pro](https://github.com/apple/ml-depth-pro) · [MoGe](https://github.com/microsoft/moge)
- [StopThePop](https://arxiv.org/abs/2402.00525) · [CoR-GS](https://arxiv.org/abs/2405.12110) ·
  [GaussianPro](https://proceedings.mlr.press/v235/cheng24f.html) ·
  [Mip-NeRF 360](https://openaccess.thecvf.com/content/CVPR2022/html/Barron_Mip-NeRF_360_Unbounded_Anti-Aliased_Neural_Radiance_Fields_CVPR_2022_paper.html)
- [gsplat eval benchmarks](https://docs.gsplat.studio/main/tests/eval.html) ·
  [Taming 3DGS](https://humansensinglab.github.io/taming-3dgs/)
