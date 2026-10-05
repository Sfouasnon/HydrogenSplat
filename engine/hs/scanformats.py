"""Survey scan formats — E57 and LAS — read with numpy alone.

`hs scale --lidar` took what a phone scanning app exports (PLY, OBJ, text points). A scan from a
terrestrial scanner on a set (Leica, Faro, Trimble …) arrives as E57, or as LAS from a survey
package, and had to go through CloudCompare first. These readers take them as they are.

Why not a library. The E57 reference reader (libE57, through pye57) ships no wheel for the Python
the engine runs on and builds against Xerces; LAS is a fixed-width record that numpy maps
directly. So both are read here, and checked against those libraries where they exist
(tests/test_scanformats.py reads files this module's test writer made; the writer's files and
libE57's own were cross-read once with pye57 0.4.19 and laspy 2.7 — see that test's docstring).
LAZ (compressed LAS) is the exception: its arithmetic coder is not something to re-write, so it
goes through laspy when laspy and a LAZ backend are installed and is refused by name otherwise.

Size. A phone scan is under a million points; a set scan is hundreds of millions. Everything
after loading (registration, the kd-trees, the init points) wants a few million at most, so both
readers take every k-th record, k chosen so that no more than ``max_points`` come back
(MAX_POINTS). E57 records are decoded straight from the packets at that stride — a bit-packed
field is gathered at the wanted bit offsets, not unpacked whole — so memory follows what is kept,
not what is in the file.

What comes back is what hs/lidar.py's other loaders return: ``(points (n, 3) float64 in file
units, colours uint8 (n, 3) or None, normals None, meta dict, declared units or None)``. E57 is
metres by its standard. LAS has no unit in its header; the coordinate system it carries (a WKT
string or GeoTIFF keys, in a variable-length record) usually names one — metres, feet or US survey
feet — and that is what is declared. A LAS with no coordinate system is taken as metres and marked
as assumed (``units_assumed``), which `hs scale` reports as units it was not told: LAS extents run
past what the phone-scan rule in hs/lidar.py (over 150 units across means millimetres) can judge.

E57 in one paragraph (ASTM E2807). A 48-byte header; then 1024-byte pages whose last four bytes
are a checksum, so offsets in the file are physical and lengths are logical (checksums not
counted). An XML section describes each scan (``data3D``): its pose, and a ``points`` vector with
a ``prototype`` naming the fields per record — ``cartesianX/Y/Z`` (or ``sphericalRange/Azimuth/
Elevation``), ``colorRed/Green/Blue``, ``cartesianInvalidState`` … — each a Float (raw IEEE,
single or double), an Integer or a ScaledInteger (bit-packed, least significant bit first, in just
enough bits for maximum − minimum; a ScaledInteger is then × scale + offset). The records
themselves sit in a binary section as data packets; a packet holds one byte stream per field, and
a stream runs on from packet to packet, so a value may be split across two. The page checksums
are stepped over, not verified (CRC-32C over gigabytes in Python is minutes): a damaged file reads
as damaged points, not as an error.
"""
import math
import os
import re
import struct
import xml.etree.ElementTree as ET

import numpy as np

MAX_POINTS = 8_000_000        # points kept from a survey scan; the stride is file records / this, rounded up

PAGE, PAGE_DATA = 1024, 1020  # an E57 page, and the bytes of it that are not checksum


# ======================================================================== LAS

# byte offset of the 16-bit R, G, B in a point record, by point data record format
_LAS_RGB_AT = {2: 20, 3: 28, 5: 28, 7: 30, 8: 30, 10: 30}


# metres per unit -> the engine's unit word (hs/lidar.py UNIT_MM)
_LINEAR_UNITS = ((1.0, "m"), (0.3048, "ft"), (1200.0 / 3937.0, "usft"))
_GEOKEY_UNITS = {9001: "m", 9002: "ft", 9003: "usft"}          # ProjLinearUnitsGeoKey (3076) codes
_ANGULAR = ("degree", "radian", "grad", "gon", "arc")


def las_declared_units(path, header_size, n_vlrs):
    """The linear unit of the coordinate system a LAS / LAZ file carries ("m", "ft", "usft"), or
    None when it carries none or names one that is not one of those. Read from the
    LASF_Projection records after the header: OGC WKT (2112) or the GeoTIFF key directory (34735)."""
    found = None
    try:
        with open(path, "rb") as f:
            f.seek(header_size)
            for _ in range(min(int(n_vlrs), 256)):
                h = f.read(54)
                if len(h) < 54:
                    break
                user = h[2:18].split(b"\0")[0]
                rec_id, length = struct.unpack_from("<HH", h, 18)
                body = f.read(length)
                if user != b"LASF_Projection":
                    continue
                if rec_id == 2112:                             # WKT: the first unit that is not an angle
                    for name, factor in re.findall(r'UNIT\s*\[\s*"([^"]+)"\s*,\s*([0-9.eE+-]+)', body.decode("latin-1")):
                        if any(a in name.lower() for a in _ANGULAR):
                            continue
                        f_m = float(factor)
                        found = next((u for m, u in _LINEAR_UNITS if abs(f_m - m) < 1e-7), None)
                        break
                elif rec_id == 34735 and found is None:        # GeoTIFF keys: KeyID, location, count, value
                    keys = np.frombuffer(body[:len(body) // 8 * 8], "<u2").reshape(-1, 4)
                    for key_id, where, _count, value in keys[1:]:
                        if key_id == 3076 and where == 0:
                            found = _GEOKEY_UNITS.get(int(value))
    except (OSError, ValueError):
        return None
    return found


def load_las(path, max_points=MAX_POINTS):
    """LAS 1.0–1.4 (ASPRS). x, y, z are int32 × scale + offset from the header; colour where the
    point format carries it. A compressed file (.laz, or the format byte's high bit) goes to laspy."""
    with open(path, "rb") as f:
        head = f.read(375)
    if len(head) < 227 or head[:4] != b"LASF":
        raise ValueError("not a LAS file (too short, or no LASF signature)")
    major, minor = head[24], head[25]
    header_size, offset, n_vlrs = struct.unpack_from("<HII", head, 94)
    declared = las_declared_units(path, header_size, n_vlrs)
    fmt, reclen = struct.unpack_from("<BH", head, 104)
    n = struct.unpack_from("<I", head, 107)[0]
    scale = np.array(struct.unpack_from("<3d", head, 131))
    shift = np.array(struct.unpack_from("<3d", head, 155))
    if (major, minor) >= (1, 4) and len(head) >= 255:
        n = struct.unpack_from("<Q", head, 247)[0] or n       # 1.4 counts in 64 bits; the old field may be 0
    if (fmt & 0x80) or path.lower().endswith(".laz"):
        return _load_laz(path, max_points, declared)
    fmt &= 0x3F
    if reclen < 12:
        raise ValueError(f"LAS point records are {reclen} bytes; x, y, z alone need 12")
    n = min(int(n), max(0, (os.path.getsize(path) - offset) // reclen))
    if n == 0:
        raise ValueError("LAS file holds no point records")
    stride = max(1, math.ceil(n / max_points))
    rows = np.ascontiguousarray(np.memmap(path, dtype=np.uint8, mode="r", offset=offset, shape=(n, reclen))[::stride])
    pts = rows[:, :12].copy().view("<i4").reshape(-1, 3).astype(np.float64) * scale + shift
    rgb = None
    at = _LAS_RGB_AT.get(fmt)
    if at is not None and reclen >= at + 6:
        c = rows[:, at:at + 6].copy().view("<u2").reshape(-1, 3)
        rgb = (c >> 8).astype(np.uint8) if c.max() > 255 else c.astype(np.uint8)   # 16-bit by the standard; some writers store 8
    meta = {"format": "las", "faces": 0, "las_version": f"{major}.{minor}", "las_point_format": int(fmt),
            "records": n, "stride": stride, "colour_from": "rgb" if rgb is not None else None,
            "units_assumed": None if declared else "m"}
    return pts, rgb, None, meta, declared


def _load_laz(path, max_points, declared=None):
    try:
        import laspy
    except ImportError:
        raise ValueError("LAZ is compressed LAS and is read through the laspy package, which is not installed: "
                         "pip install 'laspy[lazrs]' into the engine's Python, or save the scan as LAS, E57 or PLY")
    try:
        with laspy.open(path) as r:
            n = int(r.header.point_count)
            if n == 0:
                raise ValueError("LAZ file holds no point records")
            stride = max(1, math.ceil(n / max_points))
            names = set(r.header.point_format.dimension_names)
            has_rgb = {"red", "green", "blue"} <= names
            xyz, cols, seen = [], [], 0
            for chunk in r.chunk_iterator(1_000_000):
                first = (-seen) % stride
                xyz.append(np.stack([np.asarray(chunk.x), np.asarray(chunk.y), np.asarray(chunk.z)], 1)[first::stride])
                if has_rgb:
                    cols.append(np.stack([np.asarray(chunk.red), np.asarray(chunk.green), np.asarray(chunk.blue)], 1)[first::stride])
                seen += len(chunk)
            version = f"{r.header.version.major}.{r.header.version.minor}"
            fmt = int(r.header.point_format.id)
    except ValueError:
        raise
    except Exception as e:      # noqa: BLE001 - laspy without a backend, or a file it cannot read: say which
        raise ValueError(f"laspy could not read this LAZ file ({e}); it needs a LAZ backend: "
                         "pip install 'laspy[lazrs]', or save the scan as LAS, E57 or PLY")
    pts = np.concatenate(xyz).astype(np.float64)
    rgb = None
    if has_rgb:
        c = np.concatenate(cols)
        rgb = (c >> 8).astype(np.uint8) if c.max() > 255 else c.astype(np.uint8)
    meta = {"format": "laz", "faces": 0, "las_version": version, "las_point_format": fmt,
            "records": n, "stride": stride, "colour_from": "rgb" if rgb is not None else None,
            "units_assumed": None if declared else "m"}
    return pts, rgb, None, meta, declared


# ======================================================================== E57

def _to_logical(phys):
    return (phys // PAGE) * PAGE_DATA + min(phys % PAGE, PAGE_DATA)


def _to_physical(logical):
    return (logical // PAGE_DATA) * PAGE + logical % PAGE_DATA


def _read(buf, phys, n):
    """n logical bytes starting at physical offset phys, and the physical offset after them:
    the page checksums in between are stepped over."""
    end = _to_physical(_to_logical(phys) + n)
    if end > len(buf):
        raise ValueError("E57 file ends inside a section (truncated?)")
    span = np.asarray(buf[phys:end])
    if end - phys != n:                                   # the span crosses at least one checksum
        span = span[(np.arange(phys, end) % PAGE) < PAGE_DATA]
    return span.tobytes(), end


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def _child(el, name):
    return next((c for c in el if _local(c.tag) == name), None)


def _num(el, default=0.0):
    """An element's value: E57 writes a zero as an empty element."""
    if el is None:
        return default
    t = (el.text or "").strip()
    return float(t) if t else 0.0


class _Field:
    """One prototype field and the byte stream that carries it."""

    def __init__(self, name, el):
        self.name = name
        kind = el.get("type")
        self.buf, self.bit, self.n, self.out = b"", 0, 0, []
        self.scale, self.offset, self.minimum = 1.0, 0.0, 0
        if kind == "Float":
            self.kind = "float"
            self.dtype = "<f4" if el.get("precision", "double") == "single" else "<f8"
            self.size = 4 if self.dtype == "<f4" else 8
            self.maximum = el.get("maximum")
        elif kind in ("Integer", "ScaledInteger"):
            self.kind = "int"
            self.minimum = int(el.get("minimum", -(2 ** 63)))
            self.maximum = int(el.get("maximum", 2 ** 63 - 1))
            self.bits = (self.maximum - self.minimum).bit_length()
            if kind == "ScaledInteger":
                self.scale, self.offset = float(el.get("scale", 1.0)), float(el.get("offset", 0.0))
            self.scaled = kind == "ScaledInteger"
        else:
            raise ValueError(f"E57 field {name} has type {kind}; Float, Integer and ScaledInteger are read")

    @property
    def constant(self):
        return self.kind == "int" and self.bits == 0     # maximum == minimum: no bytes are stored at all

    def feed(self, chunk, stride):
        """Take the next bytes of this field's stream; keep every stride-th record of what is whole."""
        data = self.buf + chunk if self.buf else chunk
        if self.kind == "float":
            k = len(data) // self.size
            if k:
                first = (-self.n) % stride
                self.out.append(np.frombuffer(data, dtype=self.dtype, count=k)[first::stride].copy())
            used = k * self.size
        else:
            k = (len(data) * 8 - self.bit) // self.bits
            if k:
                idx = np.arange((-self.n) % stride, k, stride, dtype=np.int64)
                pos = self.bit + idx * self.bits
                a = np.frombuffer(data + b"\0" * 8, dtype=np.uint8)
                # eight bytes from each record's first byte, as one little-endian word, shifted down
                word = np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(a, 8)[pos >> 3]).view("<u8").reshape(-1)
                self.out.append((word >> (pos & 7).astype(np.uint64)) & np.uint64((1 << self.bits) - 1))
            done = self.bit + k * self.bits
            used, self.bit = done // 8, done % 8
        self.n += k
        self.buf = data[used:]

    def values(self, count):
        """The kept records as float64 (a ScaledInteger scaled; an Integer as it is)."""
        if self.constant:
            return np.full(count, self.minimum * self.scale + self.offset)
        v = np.concatenate(self.out)[:count] if self.out else np.zeros(0)
        if self.kind == "int":
            v = (v.astype(np.int64) + self.minimum).astype(np.float64) * self.scale + self.offset
        return v.astype(np.float64)


def _prototype(el, prefix=""):
    """The prototype's fields in file order; a nested structure contributes its leaves."""
    out = []
    for c in el:
        name = prefix + _local(c.tag)
        if c.get("type") == "Structure":
            out += _prototype(c, name + ".")
        else:
            out.append(_Field(name, c))
    return out


# the fields a cloud is made of; every other stream in a packet (intensity, row and column index,
# time stamps, a maker's extensions) is stepped over without being decoded
_WANTED = {"cartesianX", "cartesianY", "cartesianZ", "cartesianInvalidState",
           "sphericalRange", "sphericalAzimuth", "sphericalElevation", "sphericalInvalidState",
           "colorRed", "colorGreen", "colorBlue"}


def _decode_points(buf, points_el, stride):
    """{field name: float64 array} of every stride-th record of one scan's points vector, for
    the fields in _WANTED."""
    proto = _child(points_el, "prototype")
    if proto is None or len(proto) == 0:
        raise ValueError("E57 scan's points have no prototype (no fields are described)")
    fields = _prototype(proto)
    for f in fields:
        if f.name in _WANTED and f.kind == "int" and f.bits > 56:
            raise ValueError(f"E57 field {f.name} is packed in {f.bits} bits; more than 56 is not read")
    count = int(points_el.get("recordCount", 0))
    start = int(points_el.get("fileOffset"))
    head, _ = _read(buf, start, 32)
    section_id, _r, logical_len, data_at, _index_at = struct.unpack("<B7sQQQ", head)
    if section_id != 1:
        raise ValueError(f"E57 points section starts with id {section_id}, not a compressed vector")
    live = [f for f in fields if f.name in _WANTED and not f.constant]
    end_logical = _to_logical(start) + logical_len
    phys = data_at
    while live and min(f.n for f in live) < count and _to_logical(phys) < end_logical:
        h, _ = _read(buf, phys, 4)
        length = struct.unpack_from("<H", h, 2)[0] + 1
        body, phys = _read(buf, phys, length)
        if h[0] == 1:                                      # a data packet; 0 is an index packet, 2 is padding
            n_streams = struct.unpack_from("<H", body, 4)[0]
            if n_streams != len(fields):
                raise ValueError(f"E57 data packet carries {n_streams} streams for a prototype of {len(fields)} fields")
            at = 6 + 2 * n_streams
            for f, size in zip(fields, struct.unpack_from(f"<{n_streams}H", body, 6)):
                if size and f.name in _WANTED:
                    f.feed(body[at:at + size], stride)
                at += size
        elif h[0] not in (0, 2):
            raise ValueError(f"E57 packet of unknown type {h[0]}")
    if live and min(f.n for f in live) < count:
        raise ValueError(f"E57 points section ended after {min(f.n for f in live):,} of {count:,} records")
    kept = len(range(0, count, stride))
    return {f.name: f.values(kept) for f in fields if f.name in _WANTED}, count


def _quat_matrix(w, x, y, z):
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _colour(cols, field):
    """One colour channel as 0–255, from whatever range the file declares for it."""
    top = float(field.maximum) if field.maximum is not None else float(cols.max() if len(cols) else 255)
    if top <= 0:
        return np.zeros(len(cols), np.uint8)
    return np.clip(np.round(cols * (255.0 / top)) if top != 255 else cols, 0, 255).astype(np.uint8)


def load_e57(path, max_points=MAX_POINTS):
    """Every scan in an E57 file, each moved by its own pose into the file's common frame, as one
    cloud in metres. Invalid returns (``cartesianInvalidState`` / ``sphericalInvalidState`` not 0,
    and points at the scanner's own origin) are dropped."""
    buf = np.memmap(path, dtype=np.uint8, mode="r")
    if len(buf) < 48 or bytes(buf[:8]) != b"ASTM-E57":
        raise ValueError("not an E57 file (no ASTM-E57 signature)")
    major, minor, _flen, xml_at, xml_len, page = struct.unpack_from("<IIQQQQ", bytes(buf[8:48]))
    if page != PAGE:
        raise ValueError(f"E57 page size {page}; the standard's 1024 is what is read")
    xml, _ = _read(buf, xml_at, xml_len)
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as e:
        raise ValueError(f"E57 XML section does not parse: {e}")
    data3d = _child(root, "data3D")
    scans = [s for s in (data3d if data3d is not None else []) if _child(s, "points") is not None]
    if not scans:
        raise ValueError("E57 file holds no scans (no data3D with points); a file of images only cannot be used")
    total = sum(int(_child(s, "points").get("recordCount", 0)) for s in scans)
    if total == 0:
        raise ValueError("E57 scans hold no records")
    stride = max(1, math.ceil(total / max_points))
    all_pts, all_rgb, names, dropped, how = [], [], [], 0, set()
    fields_by_name = {}
    for s in scans:
        pel = _child(s, "points")
        cols, _count = _decode_points(buf, pel, stride)
        fields_by_name = {f.name: f for f in _prototype(_child(pel, "prototype"))}     # ranges, for the colours
        if {"cartesianX", "cartesianY", "cartesianZ"} <= set(cols):
            p = np.stack([cols["cartesianX"], cols["cartesianY"], cols["cartesianZ"]], 1)
            bad = cols.get("cartesianInvalidState")
            how.add("cartesian")
        elif {"sphericalRange", "sphericalAzimuth", "sphericalElevation"} <= set(cols):
            r, az, el = cols["sphericalRange"], cols["sphericalAzimuth"], cols["sphericalElevation"]
            p = np.stack([r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el)], 1)
            bad = cols.get("sphericalInvalidState")
            how.add("spherical")
        else:
            raise ValueError("E57 scan has neither cartesianX/Y/Z nor sphericalRange/Azimuth/Elevation fields: "
                             + ", ".join(sorted(cols)))
        keep = np.isfinite(p).all(1) & (np.abs(p).sum(1) > 0)
        if bad is not None:
            keep &= bad == 0
        dropped += int((~keep).sum())
        pose = _child(s, "pose")
        if pose is not None:
            rot, tr = _child(pose, "rotation"), _child(pose, "translation")
            if rot is not None:
                p = p @ _quat_matrix(*(_num(_child(rot, k), 1.0 if k == "w" else 0.0) for k in "wxyz")).T
            if tr is not None:
                p = p + np.array([_num(_child(tr, k)) for k in "xyz"])
        all_pts.append(p[keep])
        if {"colorRed", "colorGreen", "colorBlue"} <= set(cols):
            all_rgb.append(np.stack([_colour(cols[c], fields_by_name[c])
                                     for c in ("colorRed", "colorGreen", "colorBlue")], 1)[keep])
        else:
            all_rgb.append(None)
        nm = _child(s, "name")
        names.append((nm.text or "").strip() if nm is not None else "")
    pts = np.concatenate(all_pts)
    rgb = None
    if all(c is not None for c in all_rgb):
        rgb = np.concatenate(all_rgb)
    meta = {"format": "e57", "faces": 0, "e57_version": f"{major}.{minor}", "scans": len(scans),
            "scan_names": names[:20], "records": total, "stride": stride, "dropped_invalid": dropped,
            "coordinates": "+".join(sorted(how)), "colour_from": "colorRed/Green/Blue" if rgb is not None else None}
    return pts, rgb, None, meta, "m"
