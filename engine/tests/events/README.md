Recorded `hs` event logs for `hs replay` — the app's stub stage.

- `train_body.jsonl` — 2026-09-15 body train (84 frames, 5,068 s, 1,139,613 splats). Export,
  check and fingerprint events are the ones the run printed; the `progress` events are rebuilt
  from its growth curve at the measured 7.49 it/s (the terminal hid them with `grep -v progress`).
- `ingest_phone.jsonl` — a phone pull of the 105 MB head clip at the measured 14.8 MB/s, then
  md5, probe and profile events in the shape `hs ingest --phone` emits. The md5 is illustrative.

`"t"` is seconds since the start; `hs replay FILE --speed 20` plays them 20x faster.
