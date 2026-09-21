#!/usr/bin/env python3
"""Stand-in for hs-segment (Apple Vision) in tests. Same CLI and output contract.

HS_FAKE_SEGMENT: "two" (default) — instance 1 a disc at the image centre (the subject), instance 2
a block in the top-left corner (clutter beside it); "none" — no foreground; "error" — every image
reports an error; "crash" — exit 3."""
import json
import os
import sys

import cv2
import numpy as np

a = sys.argv[1:]
out, lst = a[a.index("--out") + 1], a[a.index("--list") + 1]
mode = os.environ.get("HS_FAKE_SEGMENT", "two")
if mode == "crash":
    sys.stderr.write("simulated crash\n")
    sys.exit(3)
for i, path in enumerate(p for p in open(lst).read().splitlines() if p):
    h, w = cv2.imread(path).shape[:2]
    if mode == "error":
        print(json.dumps({"index": i, "error": "simulated"}), flush=True)
        continue
    d = os.path.join(out, str(i))
    os.makedirs(d, exist_ok=True)
    if mode == "none":
        print(json.dumps({"index": i, "instances": 0}), flush=True)
        continue
    disc = np.zeros((h, w), np.uint8)
    cv2.circle(disc, (w // 2, h // 2), int(os.environ.get("HS_FAKE_SEGMENT_R", "90")), 255, -1)
    disc = cv2.GaussianBlur(disc, (0, 0), 3)            # Vision's masks are soft at the edge
    block = np.zeros((h, w), np.uint8)
    block[0:60, 0:80] = 255
    cv2.imwrite(os.path.join(d, "1.png"), disc)
    cv2.imwrite(os.path.join(d, "2.png"), block)
    print(json.dumps({"index": i, "instances": 2, "width": w, "height": h}), flush=True)
