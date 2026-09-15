#!/usr/bin/env bash
# Render several models along ONE move and stack them side by side into a single mp4.
#
# Comparing splat models by flipping between separate files does not work: the eye cannot
# hold a floater's position across a cut. Put them on the same move, in the same frame, at
# the same moment, and the differences are obvious.
#
#   engine/tools/three_up.sh                 # the three coin models on shot01
#   MOVE=boom engine/tools/three_up.sh       # same three, different move
#
# Labels are drawn into a strip with OpenCV rather than ffmpeg's drawtext: drawtext needs
# ffmpeg built against libfreetype, which many Homebrew and system builds are not.
set -euo pipefail

PROJ=${PROJ:-Projects/2026-09-13_coins}
MOVE=${MOVE:-shot01}
HS=${HS:-.venv/bin/hs}
PY=${PY:-.venv/bin/python}
FFMPEG=${FFMPEG:-ffmpeg}
FFPROBE=${FFPROBE:-ffprobe}
PANEL_W=${PANEL_W:-960}          # per panel; 3 panels -> 2880 wide
BAR_H=${BAR_H:-56}

# label|ply  — edit these three lines to compare something else
MODELS=(
  "no L064/R069, opac-decay 0.008   282,914 splats|$PROJ/archive/exposure-excl-decay8/export_40000.ply"
  "exposure, no L064/R069   385,967 splats|$PROJ/archive/exposure-excl/export_40000.ply"
  "no L064/R069, quality prune   298,703 splats|$PROJ/prune/export_40000_pruned_r100.ply"
)

# check every model before rendering any: a typo in the last path should not cost two renders
for m in "${MODELS[@]}"; do
  [ -f "${m##*|}" ] || { echo "missing: ${m##*|}" >&2; exit 1; }
done

inputs=(); filters=""; labels=""; texts=()
for i in "${!MODELS[@]}"; do
  label="${MODELS[$i]%%|*}"; ply="${MODELS[$i]##*|}"
  [ -f "$ply" ] || { echo "missing: $ply" >&2; exit 1; }
  name="cmp$i"
  echo "=== [$((i+1))/${#MODELS[@]}] $label"
  # --allow-mismatch is the point of this script: panels other than the current train are
  # deliberately NOT the manifest's export, which is exactly what the render guard rejects.
  "$HS" render -p "$PROJ" --move "$MOVE" --ply "$ply" --name "$name" --no-crop --allow-mismatch \
    | grep -Ev '"ev":"progress"' || true
  mp4="$PROJ/render/${name}_1920.mp4"
  [ -f "$mp4" ] || { echo "render produced no $mp4" >&2; exit 1; }
  inputs+=(-i "$mp4")
  filters+="[$i:v]scale=${PANEL_W}:-2[v$i];"
  labels+="[v$i]"
  texts+=("$label")
done

N=${#MODELS[@]}
STRIP="$PROJ/render/.${MOVE}_labels.png"
"$PY" - "$STRIP" "$PANEL_W" "$BAR_H" "$N" "${texts[@]}" <<'PY'
import sys, numpy as np, cv2
out, pw, bh, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
bar = np.zeros((bh, pw * n, 3), np.uint8)
for i, t in enumerate(sys.argv[5:5 + n]):
    cv2.putText(bar, t, (i * pw + 20, int(bh * 0.66)), cv2.FONT_HERSHEY_SIMPLEX,
                0.72, (255, 255, 255), 2, cv2.LINE_AA)
    if i:
        cv2.line(bar, (i * pw, 0), (i * pw, bh), (70, 70, 70), 1)
cv2.imwrite(out, bar)
PY

# The label strip is a looped still, i.e. an endless stream. -shortest does NOT stop a
# filter_complex graph: vstack keeps pulling the strip and repeats the last panel frame
# forever (this ran 17 h and wrote 2.3 GB on 2026-09-14). So cap it twice: vstack ends
# with its shortest input, and -frames:v pins the count to the first panel's frame count.
FRAMES=$("$FFPROBE" -v error -select_streams v:0 -count_packets \
  -show_entries stream=nb_read_packets -of csv=p=0 "$PROJ/render/cmp0_1920.mp4")
FPS=$("$FFPROBE" -v error -select_streams v:0 -show_entries stream=r_frame_rate \
  -of csv=p=0 "$PROJ/render/cmp0_1920.mp4")
OUT="$PROJ/render/${MOVE}_${N}up.mp4"
"$FFMPEG" -y -hide_banner -loglevel error "${inputs[@]}" -loop 1 -framerate "$FPS" -i "$STRIP" \
  -filter_complex "${filters}${labels}hstack=inputs=${N}:shortest=1[row];[${N}:v][row]vstack=inputs=2:shortest=1[out]" \
  -map "[out]" -frames:v "$FRAMES" -c:v libx264 -crf 17 -pix_fmt yuv420p -movflags +faststart "$OUT"
rm -f "$STRIP"
echo
echo "wrote $OUT"
"$FFMPEG" -hide_banner -i "$OUT" 2>&1 | grep -E "Duration|Stream #0:0" || true
