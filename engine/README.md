# engine

`hs/` holds the pipeline scripts, vendored verbatim from `~/Desktop/Apps/REDHydrogenOne/mv`
on 2026-09-14. Each still runs standalone exactly as documented in
`docs/capture-to-splat-spec-v1.md`:

| script | stage | needs |
|---|---|---|
| select_frames.py | Select | numpy, cv2 |
| rigcolmap.py | Solve (prep / sfm / export) | numpy, cv2, pycolmap 4.2.0 |
| spline_path.py | Move (sweep loop) | numpy |
| key_path.py | Move (keyframed) — imports spline_path | numpy |
| prune_splats.py | optional prune | numpy |
| stereocal.py | Calibration (v1.1 UI) | numpy, cv2 |

M0 turns these into one `hs` CLI with JSON-lines events (strategy §2) without changing
their numerics. Golden test: `hs selftest --clip fixtures/VID_20260912_151351_2x1.h4v`.
