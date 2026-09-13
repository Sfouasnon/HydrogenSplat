# Fixtures

The golden-test clip is not committed (this repo is public; it is Stephen's footage).
Copy it here locally:

    cp ~/Desktop/Apps/REDHydrogenOne/capture_video/VID_20260912_151351_2x1.h4v fixtures/

Expected from `hs selftest` on this clip (see docs/capture-to-splat-spec-v1.md):
60–70 frames selected (reference 65/66), 100% registered, mean reprojection 1.2–1.6 px
(reference 1.394), coverage min elevation < −5° and max > +20°, median aim within 60 px
of (959, 552) in cap021_L. Reference MD5 of the clip is recorded in `fixtures/rig6.md5`.
