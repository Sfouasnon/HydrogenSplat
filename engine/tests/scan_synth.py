"""Small E57 and LAS files for the tests, written without any library.

`write_e57` lays a file out as the standard has it — 48-byte header, binary sections, XML, all cut
into 1024-byte pages with a CRC-32C on each — and can encode coordinates as single or double
floats, as bit-packed ScaledIntegers, or as spherical range / azimuth / elevation. Every field's
byte stream is cut into packets at byte counts that are NOT multiples of a record, so a reader has
to carry a half-read value from one packet into the next.

These writers are test fixtures, but they are not only self-consistent: on 2026-10-05 their files
were read back with libE57 (pye57 0.4.19) and laspy 2.7 and gave the points that were written, and
hs/scanformats.py read libE57Format's own reference files (bunnyDouble, bunnyInt32, the coloured
cubes, las2e57's 10-bit ColourRepresentation) to the same values libE57 did.
"""
import struct

import numpy as np

PAGE, PAGE_DATA = 1024, 1020
NS = "http://www.astm.org/COMMIT/E57/2010-e57-v1.0"

_CRC = []


def crc32c(data):
    if not _CRC:
        for i in range(256):
            c = i
            for _ in range(8):
                c = (c >> 1) ^ 0x82F63B78 if c & 1 else c >> 1
            _CRC.append(c)
    crc = 0xFFFFFFFF
    for x in data:
        crc = _CRC[(crc ^ x) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def physical(logical):
    return (logical // PAGE_DATA) * PAGE + logical % PAGE_DATA


def pack_bits(values, bits):
    """Unsigned integers, `bits` each, least significant bit first, as bytes."""
    v = np.asarray(values, np.uint64)
    b = ((v[:, None] >> np.arange(bits, dtype=np.uint64)) & np.uint64(1)).astype(np.uint8).reshape(-1)
    return np.packbits(b, bitorder="little").tobytes()


def _field(name, values, kind, scale=None, lo=None, hi=None):
    """(xml for the prototype, the field's whole byte stream)."""
    v = np.asarray(values)
    if kind in ("single", "double"):
        dt = "<f4" if kind == "single" else "<f8"
        return (f'<{name} type="Float" precision="{kind}" minimum="{float(v.min())!r}" maximum="{float(v.max())!r}"/>',
                v.astype(dt).tobytes())
    if kind == "scaled":
        raw = np.round(v / scale).astype(np.int64)
        lo = int(raw.min()) if lo is None else lo
        hi = int(raw.max()) if hi is None else hi
        bits = (hi - lo).bit_length()
        return (f'<{name} type="ScaledInteger" minimum="{lo}" maximum="{hi}" scale="{scale!r}" offset="0"/>',
                pack_bits(raw - lo, bits) if bits else b"")
    lo = int(v.min()) if lo is None else lo                      # Integer
    hi = int(v.max()) if hi is None else hi
    bits = (hi - lo).bit_length()
    return (f'<{name} type="Integer" minimum="{lo}" maximum="{hi}"/>',
            pack_bits(v.astype(np.int64) - lo, bits) if bits else b"")


def _packets(streams, cuts=(1021, 37, 4093, 613)):
    """The streams cut into data packets: packet k takes the next cuts[k % len] bytes of each."""
    out, at, k = b"", [0] * len(streams), 0
    while any(a < len(s) for a, s in zip(at, streams)):
        parts = []
        for i, s in enumerate(streams):
            take = cuts[(k + i) % len(cuts)]
            parts.append(s[at[i]:at[i] + take])
            at[i] += len(parts[-1])
        body = struct.pack(f"<{len(parts)}H", *[len(p) for p in parts]) + b"".join(parts)
        total = 6 + len(body)
        pad = (-total) % 4
        out += struct.pack("<BBHH", 1, 0, total + pad - 1, len(parts)) + body + b"\0" * pad
        k += 1
    return out


def write_e57(path, scans):
    """scans: list of dicts — ``xyz`` (n, 3) metres, in the scan's own frame; optional ``rgb``
    (n, 3), ``colour_max`` (255 or 65535), ``pose`` ((w, x, y, z), (tx, ty, tz)), ``encoding``
    "single" | "double" | "scaled" (with ``scale``, default 0.001) | "spherical", ``invalid``
    (n,) 0/1/2 states, ``intensity`` (n,) to put a stream in that readers must step over,
    ``name``."""
    logical = bytearray(48)                                   # the header is filled in last
    scan_xml = []
    for i, s in enumerate(scans):
        xyz = np.asarray(s["xyz"], np.float64)
        enc = s.get("encoding", "double")
        fields = []
        if enc == "spherical":
            r = np.linalg.norm(xyz, axis=1)
            el = np.arcsin(np.divide(xyz[:, 2], r, out=np.zeros_like(r), where=r > 0))
            az = np.arctan2(xyz[:, 1], xyz[:, 0])
            fields += [_field("sphericalRange", r, "double"), _field("sphericalAzimuth", az, "double"),
                       _field("sphericalElevation", el, "double")]
        else:
            for k, nm in enumerate(("cartesianX", "cartesianY", "cartesianZ")):
                fields.append(_field(nm, xyz[:, k], enc, scale=s.get("scale", 0.001)))
        if "intensity" in s:
            fields.append(_field("intensity", s["intensity"], "single"))
        if "rgb" in s:
            top = s.get("colour_max", 255)
            c = np.asarray(s["rgb"], np.int64) * (top // 255)
            for k, nm in enumerate(("colorRed", "colorGreen", "colorBlue")):
                fields.append(_field(nm, c[:, k], "integer", lo=0, hi=top))
        if "invalid" in s:
            fields.append(_field("sphericalInvalidState" if enc == "spherical" else "cartesianInvalidState",
                                 s["invalid"], "integer", lo=0, hi=2))
        data = _packets([f[1] for f in fields])
        start = len(logical)
        section = struct.pack("<B7sQQQ", 1, b"\0" * 7, 32 + len(data), physical(start + 32), 0)
        logical += section + data
        pose = ""
        if "pose" in s:
            (w, x, y, z), (tx, ty, tz) = s["pose"]
            pose = (f'<pose type="Structure"><rotation type="Structure"><w type="Float">{w!r}</w><x type="Float">{x!r}</x>'
                    f'<y type="Float">{y!r}</y><z type="Float">{z!r}</z></rotation><translation type="Structure">'
                    f'<x type="Float">{tx!r}</x><y type="Float">{ty!r}</y><z type="Float">{tz!r}</z></translation></pose>')
        scan_xml.append(
            f'<vectorChild type="Structure"><guid type="String"><![CDATA[{{00000000-0000-0000-0000-{i:012d}}}]]></guid>'
            f'<name type="String"><![CDATA[{s.get("name", f"scan {i}")}]]></name>{pose}'
            f'<points type="CompressedVector" fileOffset="{physical(start)}" recordCount="{len(xyz)}">'
            f'<prototype type="Structure">{"".join(f[0] for f in fields)}</prototype>'
            f'<codecs type="Vector" allowHeterogeneousChildren="1"></codecs></points></vectorChild>')
    xml = (f'<?xml version="1.0" encoding="UTF-8"?>\n<e57Root type="Structure" xmlns="{NS}">'
           '<formatName type="String"><![CDATA[ASTM E57 3D Imaging Data File]]></formatName>'
           '<guid type="String"><![CDATA[{11111111-2222-3333-4444-555555555555}]]></guid>'
           '<versionMajor type="Integer">1</versionMajor><versionMinor type="Integer">0</versionMinor>'
           '<coordinateMetadata type="String"/>'
           f'<data3D type="Vector" allowHeterogeneousChildren="1">{"".join(scan_xml)}</data3D>'
           '<images2D type="Vector" allowHeterogeneousChildren="1"></images2D></e57Root>\n').encode()
    xml_at = len(logical)
    logical += xml
    pages = -(-len(logical) // PAGE_DATA)
    logical[:48] = struct.pack("<8sIIQQQQ", b"ASTM-E57", 1, 0, pages * PAGE, physical(xml_at), len(xml), PAGE)
    logical += b"\0" * (pages * PAGE_DATA - len(logical))
    with open(path, "wb") as f:
        for p in range(pages):
            chunk = bytes(logical[p * PAGE_DATA:(p + 1) * PAGE_DATA])
            f.write(chunk + struct.pack(">I", crc32c(chunk)))


# LAS point record lengths and where the 16-bit colour sits, by point format
_LAS = {0: (20, None), 2: (26, 20), 3: (34, 28), 6: (30, None), 7: (36, 30)}


def write_las(path, xyz, rgb=None, fmt=None, version=(1, 2), scale=0.001, vlrs=()):
    """A LAS file of the given points. fmt defaults to 2 / 7 with colour and 0 / 6 without.
    vlrs: (user id, record id, payload bytes) records written between the header and the points —
    a coordinate system is ("LASF_Projection", 2112, WKT) or ("LASF_Projection", 34735, GeoTIFF keys)."""
    xyz = np.asarray(xyz, np.float64)
    n = len(xyz)
    new = tuple(version) >= (1, 4)
    if fmt is None:
        fmt = (7 if new else 2) if rgb is not None else (6 if new else 0)
    reclen, rgb_at = _LAS[fmt]
    head_size = 375 if new else 227
    offset = np.floor(xyz.min(0))
    raw = np.round((xyz - offset) / scale).astype("<i4")
    rec = np.zeros((n, reclen), np.uint8)
    rec[:, :12] = raw.view(np.uint8).reshape(n, 12)
    if rgb is not None and rgb_at is not None:
        rec[:, rgb_at:rgb_at + 6] = (np.asarray(rgb, np.uint16) << 8).astype("<u2").view(np.uint8).reshape(n, 6)
    extra = b"".join(struct.pack("<H16sHH32s", 0, user.encode(), rec_id, len(body), b"") + body
                     for user, rec_id, body in vlrs)
    h = bytearray(head_size)
    h[:4] = b"LASF"
    h[24], h[25] = version
    struct.pack_into("<HI", h, 94, head_size, head_size + len(extra))
    struct.pack_into("<I", h, 100, len(vlrs))
    struct.pack_into("<BH", h, 104, fmt, reclen)
    struct.pack_into("<I", h, 107, 0 if new else n)
    struct.pack_into("<3d", h, 131, scale, scale, scale)
    struct.pack_into("<3d", h, 155, *offset)
    mx, mn = xyz.max(0), xyz.min(0)
    struct.pack_into("<6d", h, 179, mx[0], mn[0], mx[1], mn[1], mx[2], mn[2])
    if new:
        struct.pack_into("<Q", h, 247, n)
    with open(path, "wb") as f:
        f.write(bytes(h) + extra + rec.tobytes())
