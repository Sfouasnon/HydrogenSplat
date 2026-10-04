#!/usr/bin/env python3
"""Stand-in for macOS's sips: `sips -s format jpeg -s formatOptions 95 SRC --out DST`.

The "HEIC" is any file; its first byte picks the grey level and bytes 1-2 the width and height
(0 = 64 x 48), so tests can tell the photographs apart and make one a different size."""
import sys

import cv2
import numpy as np

a = sys.argv[1:]
src = a[a.index("--out") - 1]
dst = a[a.index("--out") + 1]
raw = open(src, "rb").read()
if raw[:4] == b"FAIL":
    sys.stderr.write("Error: Cannot extract image from file.\n")
    sys.exit(1)
w, h = (raw[1] or 64) if len(raw) > 1 else 64, (raw[2] or 48) if len(raw) > 2 else 48
cv2.imwrite(dst, np.full((h, w, 3), raw[0] if raw else 128, np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95])
