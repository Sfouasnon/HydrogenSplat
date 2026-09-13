# Fixtures

The golden-test clip is not committed (this repo is public; it is Stephen's footage).
Copy it here locally:

    cp ~/Desktop/Apps/REDHydrogenOne/capture_video/VID_20260912_151351_2x1.h4v fixtures/

Run `.venv/bin/hs selftest` (writes into `fixtures/selftest/`, gitignored; `--resume` skips
finished stages, `--fresh` starts over). Expected on this clip (see docs/capture-to-splat-spec-v1.md):
60–70 frames selected (reference 65/66), 100% registered, mean reprojection 1.2–1.6 px
(reference 1.394), coverage min elevation < −5° and max > +20°, median aim within 60 px
of (959, 552) in cap021_L. Reference MD5 of the clip is recorded in `fixtures/rig6.md5`.

Training is not part of the automated test. Run by hand on this project, the golden train
(`hs train -p fixtures/selftest`, 40k iterations, ~48 min on an M-series Mac) produced
157,252 splats against rig6's 170,841 — inside the strategy's 170k ± 20k band and inside the
0.6–1.4×-per-frame check `hs train` applies. Rendering `boom` took 21.9 s for 360 frames.
