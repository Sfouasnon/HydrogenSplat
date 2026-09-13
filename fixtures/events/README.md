# Recorded event streams

`selftest-2026-09-13.jsonl` is the complete stdout of `hs selftest --fresh` on the rig6 clip
(x86_64 container, 2 cores, 9.9 min): ingest → select → solve → move ×2 → the selftest
assertions. One JSON object per line, the contract in docs/hydrogensplat-strategy-v1.md §2.
M1's ToolRunner stub replays it (`ev:progress` lines carry `step`, `done`, `total`, and
`rate`/`eta_s` where known; a consumer must ignore unknown keys and event kinds).
