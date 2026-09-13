# HydrogenSplat

Stereo Gaussian splats from a RED Hydrogen One. A macOS app that takes a Holocam 3D video
clip off the phone, selects frames by parallax, solves the two eyes as one rigid COLMAP rig,
trains a splat in Brush, builds a camera move inside the captured hull, and renders it.

Status: **M0 — engine CLI.** No app yet. `engine/` has the `hs` CLI: every pipeline stage as
a subcommand speaking JSON-lines events, wrapping the six scripts unchanged; `hs selftest`
reproduces the rig6 numbers from the terminal (see `engine/README.md`). `hs train` and
`hs render` wrap Brush and are exercised only against stand-ins until one run on the Mac.
The pipeline is specified, with the numbers from the first successful capture, in
`docs/capture-to-splat-spec-v1.md`. The build plan is `docs/hydrogensplat-strategy-v1.md`.
Read those two first; then `docs/HANDOFF-2026-09-13-colmap-rig.md` §4 for the mistakes
that must not be repeated.

## Layout

```
docs/       spec, strategy, handoff — the contract for what the app does
engine/hs/  Python engine: the `hs` CLI, JSON-lines events on stdout; the six vendored scripts unchanged
profiles/   calibration profiles (JSON); one ships: H1 video mode, 1920x1080/eye, Holocam 1.18.2
fixtures/   golden-test clip (not committed) + expected numbers
app/        SwiftUI macOS app (from M1)
```

## Requirements

Apple Silicon Mac. Homebrew `python3` (3.14 at time of writing) for the engine venv;
Brush and brush-path-render built from the Brush fork at `~/Desktop/Apps/brush`
(`target/release/{brush,brush-path-render}`); `ffmpeg`; `adb` (android-platform-tools).

```
python3 -m venv .venv && .venv/bin/pip install -e engine
.venv/bin/hs tools        # what is installed, what is missing and how to fix it
.venv/bin/hs selftest     # golden test on fixtures/VID_20260912_151351_2x1.h4v, ~15 min
```

## Non-negotiable rules (from the first successful run)

- Calibrated intrinsics go into the COLMAP rig config; `--float-rig` on; `--refine-intrinsics` off.
- Frame selection by homography residual (parallax). No optical-flow escape hatch.
- Aim point = SfM median, projected into a real view, confirmed by a human before render.
- MD5 the .ply and check the path file's mtime before every render.
- Never train on a partial solve without saying so.
