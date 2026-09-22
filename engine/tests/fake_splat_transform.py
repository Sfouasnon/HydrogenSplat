#!/usr/bin/env python3
"""Stand-in for PlayCanvas splat-transform (@playcanvas/splat-transform, v3.5.1 syntax).

    splat-transform [GLOBAL] input [ACTIONS] ... output [ACTIONS]

Understands what hs export sends: -w, -q, -g <n|cpu> globals, the -V name,cmp,value action on
``opacity`` (linear, after the sigmoid, as the real tool compares it), one input, one output.
A .ply output is a real binary PLY with the filter applied (float properties only, like the
fakes' PLYs); .spz / .sog / .html are small placeholder files naming their format and the
input's md5, so a test can tell them apart. Refuses an existing output without -w, as the real
one does. Knobs:
    HS_FAKE_ST_LOG=path    append each argv as one JSON line (tests assert it)
    HS_FAKE_ST_NO_GPU=1    .sog / .html fail unless -g cpu is given (a machine with no WebGPU adapter)
"""
import hashlib
import json
import os
import sys

import numpy as np

VALUE_FLAGS = {"-g", "--gpu", "-V", "--filter-value", "-H", "--filter-harmonics", "-s", "--scale",
               "-t", "--translate", "-r", "--rotate", "-i", "--sh-iterations", "--viewer-settings"}
CMP = {"lt": np.less, "lte": np.less_equal, "gt": np.greater, "gte": np.greater_equal,
       "eq": np.equal, "neq": np.not_equal}


def read_ply(path):
    raw = open(path, "rb").read()
    k = raw.index(b"end_header\n") + len(b"end_header\n")
    head = raw[:k].decode("ascii")
    n = int(next(l.split()[2] for l in head.splitlines() if l.startswith("element vertex")))
    props = [l.split()[2] for l in head.splitlines() if l.startswith("property")]
    arr = np.frombuffer(raw[k:k + n * 4 * len(props)], "<f4").reshape(n, len(props))
    return props, arr, hashlib.md5(raw).hexdigest()


def write_ply(path, props, arr):
    hdr = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % len(arr)
           + "".join(f"property float {p}\n" for p in props) + "end_header\n")
    with open(path, "wb") as f:
        f.write(hdr.encode() + np.ascontiguousarray(arr, "<f4").tobytes())


def main(argv):
    if os.environ.get("HS_FAKE_ST_LOG"):
        with open(os.environ["HS_FAKE_ST_LOG"], "a") as f:
            f.write(json.dumps(argv) + "\n")
    if "--version" in argv or "-v" in argv:
        print("splat-transform v0.0.0-fake (fake)")
        return 0
    if "--help" in argv or "-h" in argv or not argv:
        print("USAGE\n  splat-transform [GLOBAL] input [ACTIONS]  ...  output [ACTIONS]")
        return 0
    files, filters, opts, i = [], [], {}, 0
    while i < len(argv):
        t = argv[i]
        if t in VALUE_FLAGS:
            if t in ("-V", "--filter-value"):
                filters.append(argv[i + 1])
            else:
                opts[t] = argv[i + 1]
            i += 2
        elif t.startswith("-"):
            opts[t] = True
            i += 1
        else:
            files.append(t)
            i += 1
    if len(files) < 2:
        print("error: need an input and an output")
        return 1
    src, out = files[0], files[-1]
    if os.path.exists(out) and not (opts.get("-w") or opts.get("--overwrite")):
        print(f"error: File '{out}' already exists. Use -w option to overwrite.")
        return 1
    ext = os.path.splitext(out)[1].lower()
    gpu = opts.get("-g") or opts.get("--gpu")
    if ext in (".sog", ".html") and os.environ.get("HS_FAKE_ST_NO_GPU") and gpu != "cpu":
        print("Warning: vkCreateInstance: Found no drivers!")
        print("    ✗ TypeError: Cannot read properties of null (reading 'features')")
        return 1
    props, arr, md5 = read_ply(src)
    keep = np.ones(len(arr), bool)
    for f in filters:
        name, cmp, val = f.split(",")
        if name == "opacity":
            col = 1.0 / (1.0 + np.exp(-arr[:, props.index("opacity")].astype(np.float64)))
        else:
            col = arr[:, props.index(name.removesuffix("_raw"))]
        keep &= CMP[cmp](col, float(val))
    arr = arr[keep]
    print(f"  · {len(arr)} gaussians")
    if ext == ".ply":
        write_ply(out, props, arr)
    elif ext in (".spz", ".sog", ".html"):
        with open(out, "w") as f:
            f.write(f"FAKE {ext[1:].upper()} of {md5}: {len(arr)} gaussians, filters {filters}, gpu {gpu}\n")
    else:
        print(f"error: unsupported output {ext}")
        return 1
    print(f"    · {os.path.basename(out)} ({os.path.getsize(out)} B)")
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
