"""What hs reads out of a scan file, set against what the file says about itself.

    python3 engine/tools/scan_report.py fixtures/scans/pump.e57 [more files ...]

For an E57 it prints, per scan, the scanner the file names, the record count, the fields and how
each is stored, and the scan's own declared bounds (cartesianBounds, in the scanner's frame) next
to the extent of the points this reader decoded BEFORE the pose is applied. Decoded points must sit
inside the declared box and, unless the scan is thinned hard, reach its faces. For a LAS it does the
same against the header's bounds. Then it runs the file through hs.lidar.load_scan, which is what
`hs scale --lidar` calls. Reads only; writes nothing.
"""
import os
import sys
import time
import xml.etree.ElementTree as ET
import struct

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hs import lidar, scanformats as F  # noqa: E402


def _text(el, name):
    c = F._child(el, name) if el is not None else None
    return (c.text or "").strip() if c is not None and c.text else ""


def e57(path, max_points):
    buf = np.memmap(path, dtype=np.uint8, mode="r")
    major, minor, flen, xml_at, xml_len, page = struct.unpack_from("<IIQQQQ", bytes(buf[8:48]))
    xml, _ = F._read(buf, xml_at, xml_len)
    root = ET.fromstring(xml)
    scans = [s for s in (F._child(root, "data3D") or []) if F._child(s, "points") is not None]
    total = sum(int(F._child(s, "points").get("recordCount", 0)) for s in scans)
    stride = max(1, -(-total // max_points))
    print(f"  E57 {major}.{minor}, file length in header {flen:,} (on disk {len(buf):,}), "
          f"{len(scans)} scan(s), {total:,} records, read 1 in {stride}")
    worst, declared = 0.0, 0
    for i, s in enumerate(scans):
        pel = F._child(s, "points")
        proto = F._prototype(F._child(pel, "prototype"))
        print(f"  scan {i}: '{_text(s, 'name')}'  {_text(s, 'sensorVendor')} {_text(s, 'sensorModel')}"
              f"  records {int(pel.get('recordCount', 0)):,}  pose {'yes' if F._child(s, 'pose') is not None else 'no'}")
        print("    fields: " + ", ".join(f"{f.name}[{f.kind}{'' if f.kind != 'int' else ' %d bit' % f.bits}]" for f in proto))
        t = time.time()
        cols, _n = F._decode_points(buf, pel, stride)
        dt = time.time() - t
        if "cartesianX" not in cols:
            print(f"    spherical scan, decoded in {dt:.1f} s (no cartesian bounds to compare)")
            continue
        p = np.stack([cols["cartesianX"], cols["cartesianY"], cols["cartesianZ"]], 1)
        ok = np.isfinite(p).all(1) & (np.abs(p).sum(1) > 0)
        if cols.get("cartesianInvalidState") is not None:
            ok &= cols["cartesianInvalidState"] == 0
        p = p[ok]
        b = F._child(s, "cartesianBounds")
        print(f"    decoded {len(p):,} valid of {len(ok):,} in {dt:.1f} s")
        for k, ax in enumerate("xyz"):
            lo, hi = float(p[:, k].min()), float(p[:, k].max())
            line = f"    {ax}: read {lo:10.3f} .. {hi:10.3f} m"
            if b is not None:
                declared += 1
                dlo, dhi = F._num(F._child(b, ax + "Minimum")), F._num(F._child(b, ax + "Maximum"))
                out = max(dlo - lo, hi - dhi, 0.0)
                worst = max(worst, out)
                line += f"   file says {dlo:10.3f} .. {dhi:10.3f}   outside by {out * 1000:.2f} mm"
            print(line)
    if worst:
        print(f"  WORST: a decoded point lies {worst * 1000:.2f} mm outside its scan's declared bounds")
    elif declared:
        print("  every decoded point is inside its scan's declared bounds")
    else:
        print("  the file declares no bounds to check the points against")


def las(path):
    h = open(path, "rb").read(375)
    mx = struct.unpack_from("<6d", h, 179)
    print(f"  LAS {h[24]}.{h[25]} point format {h[104] & 0x3F}, header bounds "
          f"x {mx[1]:.3f}..{mx[0]:.3f}  y {mx[3]:.3f}..{mx[2]:.3f}  z {mx[5]:.3f}..{mx[4]:.3f}")


def main(paths, max_points=8_000_000):
    for path in paths:
        print(f"{path}  ({os.path.getsize(path) / 1e6:.1f} MB)")
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".e57":
                e57(path, max_points)
            elif ext in (".las", ".laz"):
                las(path)
            t = time.time()
            scan = lidar.load_scan(path)
            m = scan.meta
            ext_mm = scan.points.max(0) - scan.points.min(0)
            print(f"  load_scan: {len(scan.points):,} points in {time.time() - t:.1f} s, colour "
                  f"{'yes' if scan.colors is not None else 'no'}, units {m.get('units')} ({m.get('units_source')}), "
                  f"extent {ext_mm[0] / 1000:.2f} x {ext_mm[1] / 1000:.2f} x {ext_mm[2] / 1000:.2f} m, "
                  f"origin shift {m.get('origin_shift_mm')}, dropped invalid {m.get('dropped_invalid')}")
        except Exception as e:                                # report and go on to the next file
            print(f"  FAILED: {type(e).__name__}: {e}")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
