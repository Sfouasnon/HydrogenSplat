#!/bin/sh
# Exercise hs train / prune / render without Brush, on top of a finished selftest project
# (needs fixtures/selftest with solve + move done: run `hs selftest` first).
#   engine/tests/run_fake_pipeline.sh [project]
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
P="${1:-$ROOT/fixtures/selftest}"
HS="$ROOT/.venv/bin/hs"
"$HS" -p "$P" train --brush "$ROOT/engine/tests/fakebin/brush" --total-train-iters 4000 --growth-stop-iter 3000 --export-every 1000 --no-caffeinate
"$HS" -p "$P" prune
"$HS" -p "$P" render --move boom --render-bin "$ROOT/engine/tests/fakebin/brush-path-render"
"$HS" -p "$P" render --move sweep --ply "$P"/prune/export_4000_pruned_r03.ply --render-bin "$ROOT/engine/tests/fakebin/brush-path-render" --no-crop
ls -la "$P/render"
