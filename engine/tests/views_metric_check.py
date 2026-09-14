#!/usr/bin/env python3
"""Does `hs views` report the displacement that is actually there?

Every conclusion this project has drawn about model quality rests on one measurement:
phase correlation on 64 px patches, thresholded at 4 px. That instrument has never been
given a case with a known answer. This does: it takes a real training photograph, makes a
second copy displaced, blurred or noised by a known amount, and asks `compare()` what it
sees. A metric that cannot recover a 6 px shift it was handed cannot be used to argue that
a model is 6 px wrong.

    python3 engine/tests/views_metric_check.py <project>
"""
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hs.stages.views import compare, PATCH          # noqa: E402

DISP = 4.0


def case(name, src, ren, uv, half, expect):
    r = compare(src, ren, uv, half, DISP)
    if r is None:
        print(f"{name:28s} -> crop too small")
        return
    med = r["displacement_median_px"] or 0.0
    p90 = r["displacement_p90_px"] or 0.0
    frac = r["displaced_fraction"] or 0.0
    print(f"{name:28s} n={r['patches']:4d}  median {med:6.2f}  p90 {p90:6.2f}  "
          f">4px {100 * frac:5.1f}%  edge {r['retained_edge_energy'] or 0:5.2f}  "
          f"psnr {r['psnr_db']:5.1f}   expect {expect}")


def main(project):
    img = None
    for eye in ("L",):
        d = os.path.join(project, "train", "dataset", "images", eye)
        for f in sorted(os.listdir(d)):
            if f.endswith(".jpg"):
                img = os.path.join(d, f)
                break
    if img is None:
        sys.exit("no training images")
    print(f"reference: {img}\n")
    bgr = cv2.imread(img)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = g.shape
    uv, half = (w / 2, h / 2), min(h, w) // 2 - 4

    def shift(a, dx, dy):
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(a, M, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)

    case("identical", g, g.copy(), uv, half, "0 px, 0%")
    for dx in (1, 2, 4, 6, 10, 20):
        case(f"shifted {dx} px horizontally", g, shift(g, dx, 0), uv, half,
             f"{dx} px, {'100' if dx > DISP else '0'}%")
    case("shifted 6 px diagonally", g, shift(g, 6 / np.sqrt(2), 6 / np.sqrt(2)), uv, half, "6 px, 100%")
    for s in (1.0, 2.0, 4.0):
        case(f"blurred sigma {s:g}", g, cv2.GaussianBlur(g, (0, 0), s), uv, half, "0 px, 0%, edge down")
    case("blur 2 + shift 6 px", g, cv2.GaussianBlur(shift(g, 6, 0), (0, 0), 2.0), uv, half, "6 px, 100%")
    case("noise sigma 8", g, (g + np.random.default_rng(0).normal(0, 8, g.shape)).astype(np.float32),
         uv, half, "0 px, 0%")
    case("half the frame shifted 6 px", g,
         np.hstack([shift(g, 6, 0)[:, :w // 2], g[:, w // 2:]]), uv, half, "~6 px on half, ~50%")
    case("contrast x0.5", g, (g * 0.5).astype(np.float32), uv, half, "0 px, 0%")
    case("brightness +40", g, (g + 40).astype(np.float32), uv, half, "0 px, 0%")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Projects/2026-09-13_coins")
