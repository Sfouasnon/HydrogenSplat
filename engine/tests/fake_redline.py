#!/usr/bin/env python3
"""Stand-in for REDline: renders the "first frame" of a fake R3D as a 16-bit TIFF.

The R3D is any file; its first byte picks the grey level so tests can tell cameras apart.
Understands the flags hs ingest sends (--i, --res, --outDir, --o, --format 1 …) and writes
<outDir>/<o>_000000.tif the way REDline names single frames."""
import os
import sys

import cv2
import numpy as np


def main(argv):
    a = {}
    i = 0
    while i < len(argv):
        if argv[i].startswith("--") and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            a[argv[i][2:]] = argv[i + 1]
            i += 2
        else:
            a[argv[i].lstrip("-")] = True
            i += 1
    if "help" in a:
        print("--format <int>        - output formats, DPX = 0, Tiff = 1 ...")
        return 0
    src, out, stem = a["i"], a["outDir"], a["o"]
    if not os.path.exists(src):
        print(f"no such clip {src}")
        return 1
    if a.get("format") != "1":
        print("fake REDline only writes TIFF (--format 1)")
        return 1
    level = open(src, "rb").read(1)[0] if os.path.getsize(src) else 0
    res = int(a.get("res", "1"))
    w, h = 64 // res, 36 // res
    im = np.full((h, w, 3), level * 257, np.uint16)
    im[:, : w // 2, 0] = 65535           # a feature so it is not a flat frame
    cv2.imwrite(os.path.join(out, f"{stem}_000000.tif"), im)
    print(f"rendered 1 frame of {os.path.basename(src)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
