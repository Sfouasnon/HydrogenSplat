"""hs calibrate — a ChArUco board to print or show, and a lens profile from a clip of it.

Two jobs, one door (the app's Calibrate page):

  hs calibrate --board-image board.png [--board-pdf board.pdf] [--squares 7x5 --square-mm 35 --marker-mm 26]
      the board at print size: 300 dpi for the millimetres asked, a 10 mm quiet border, the
      board's name and sizes in the margin, the PNG tagged with its dpi so Preview prints it at
      100 %. The PDF holds the same picture on a Letter page and an A4 page, placed at exact size.
  hs calibrate --board-screen screen.png --screen-in 15 --screen-px 1728x1117
      the board sized for a screen: the diagonal in inches and the pixel size give the pixels per
      millimetre, so a square measures --square-mm on that screen when the app shows the picture
      full screen, pixel for pixel.
  hs calibrate [-p P] --clip BOARD.mov [--every 10 --max-frames 60] | --frames DIR
      every --every'th frame is decoded with ffmpeg (grey, autorotated, as ingest sees the clip),
      the board's corners are found in each (hs/board.py), cv2.calibrateCamera fits one OPENCV
      camera (fx fy cx cy, k1 k2 p1 p2 k3; a second pass drops the frames the first could not
      explain — motion blur, a half-seen board) and the result is a lens profile (hs/lens.py)
      keyed by the clip's camera make, model, lens and recording size. `hs solve --lens auto`
      finds it by that key. With -p the frames, detections and a copy of the profile go to
      P/calibrate/ and the verdict into the manifest; without, only the store is written.

Checks say what to do: board_seen_in_enough_frames (>= 15 frames with >= 20 corners),
calibration_covers_the_frame (corners in >= 70 % of a 6 x 4 grid of cells: the distortion is
in the corners of the frame, so film the board there too), reprojection_rms_small (<= 0.5 px;
above it the fit needs a human: hold the board still, keep it flat, use the shoot's recording
setting). The board's physical size does not enter the intrinsics, so a board shown on a screen
calibrates as well as a print — what matters is that it is flat and sharp.
"""
import datetime
import glob
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib

import numpy as np

from .. import __version__, board as B, events, lens, sourceprobe

STAGE = "calibrate"
PRINT_DPI = 300.0
BORDER_MM = 10.0            # quiet border round the board: detection wants white round the outer markers
LABEL_MM = 7.0              # text strip under the board
PAGES_MM = (("Letter", 215.9, 279.4), ("A4", 210.0, 297.0))
COVER_CELLS = (6, 4)
MIN_FRAMES, MIN_CORNERS, MAX_RMS_PX, MIN_COVER = 15, 20, 0.5, 0.70
FRAME_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def add_parser(sub):
    p = sub.add_parser("calibrate", help="ChArUco board to print/show; lens profile (K, distortion) from a clip of it")
    b = p.add_argument_group("the board")
    b.add_argument("--squares", default="7x5", metavar="SXxSY", help="squares across x down (default 7x5)")
    b.add_argument("--square-mm", type=float, default=35.0, help="printed square side, mm")
    b.add_argument("--marker-mm", type=float, default=26.0, help="marker side inside the white squares, mm")
    b.add_argument("--dict", default=B.DEFAULT_DICT, help="ArUco dictionary (default DICT_5X5_100)")
    b.add_argument("--board-image", default=None, metavar="OUT.png", help="write the board at 300 dpi print size")
    b.add_argument("--board-pdf", default=None, metavar="OUT.pdf", help="write the board on a Letter and an A4 page")
    b.add_argument("--board-screen", default=None, metavar="OUT.png", help="write the board sized for a screen")
    b.add_argument("--screen-in", type=float, default=None, help="--board-screen: the screen's diagonal, inches")
    b.add_argument("--screen-px", default=None, metavar="WxH", help="--board-screen: the screen's pixel size")
    c = p.add_argument_group("the lens")
    c.add_argument("--clip", default=None, help="a video of the board, from the camera and setting the shoot uses")
    c.add_argument("--frames", default=None, metavar="DIR", help="instead of --clip: a folder of stills of the board")
    c.add_argument("--every", type=int, default=10, help="--clip: use every Nth frame (default 10)")
    c.add_argument("--max-frames", type=int, default=60, help="at most this many frames (default 60)")
    c.add_argument("--min-corners", type=int, default=MIN_CORNERS, help="a frame counts with this many corners")
    c.add_argument("--make", default=None, help="camera make when the clip's tags lack it (or to override)")
    c.add_argument("--model", default=None, help="camera model, likewise")
    c.add_argument("--lens-name", default=None, help="lens name, likewise (default: the clip's tag, else 'default')")
    c.add_argument("--keep-frames", action="store_true", help="-p: keep the decoded frames in P/calibrate/frames")
    c.add_argument("--out", default=None, metavar="PROFILE.json", help="also write the profile here")
    c.add_argument("--ffmpeg", default=os.environ.get("HS_FFMPEG", "ffmpeg"))
    c.add_argument("--ffprobe", default=os.environ.get("HS_FFPROBE", "ffprobe"))
    return p


# ------------------------------------------------------------------ the board picture

def parse_squares(s):
    m = re.fullmatch(r"\s*(\d+)\s*[xX,]\s*(\d+)\s*", str(s))
    if not m:
        raise events.StageError(f"--squares {s!r}: want SXxSY, e.g. 7x5")
    return int(m.group(1)), int(m.group(2))


def spec_from_args(a):
    sx, sy = parse_squares(a.squares)
    try:
        return B.parse_spec(f"{sx},{sy},{a.square_mm:g},{a.marker_mm:g},{a.dict}")
    except ValueError as e:
        raise events.StageError(str(e))


def board_label(spec, square_px, px_per_mm):
    sq = square_px / px_per_mm
    return (f"ChArUco {spec.sx}x{spec.sy} {spec.dictionary}  square {sq:.2f} mm  marker "
            f"{sq * spec.marker_mm / spec.square_mm:.2f} mm  -  HydrogenSplat, print at 100%")


def render_board(spec, px_per_mm, border_mm=BORDER_MM, label_mm=LABEL_MM, label=True, exact=True):
    """The board as a grey uint8 image: squares of round(square_mm * px_per_mm) px (whole
    pixels, so every edge is sharp), a white border and a text strip under it. With `exact`
    (print) the pixel pitch is then taken as square_px / square_mm, so the dpi written into the
    file makes a square print at exactly --square-mm; on a screen the pitch is the screen's and
    the square is whatever whole number of pixels is nearest.
    -> (img, info) with info = {square_px, square_mm_actual, border_px, origin (x, y) of the
    board's top-left corner in the image, px_per_mm (the pitch the picture is to be shown at)}."""
    import cv2
    square_px = int(round(spec.square_mm * px_per_mm))
    if square_px < 8:
        raise events.StageError(f"a {spec.square_mm:g} mm square is {square_px} px here: too small to detect")
    if exact:
        px_per_mm = square_px / float(spec.square_mm)
    border = int(round(border_mm * px_per_mm))
    bw, bh = spec.sx * square_px, spec.sy * square_px
    core = B.generate_image(spec, (bw, bh), margin=0)
    strip = int(round(label_mm * px_per_mm)) if label else 0
    img = np.full((bh + 2 * border + strip, bw + 2 * border), 255, np.uint8)
    img[border:border + bh, border:border + bw] = core
    if label:
        text = board_label(spec, square_px, px_per_mm)
        scale = max(0.3, 2.6 * px_per_mm / 11.811 * 0.33)      # ~3 mm cap height
        th = max(1, int(round(px_per_mm / 6)))
        y = border + bh + int(strip * 0.65)
        cv2.putText(img, text, (border, y), cv2.FONT_HERSHEY_SIMPLEX, scale, 0, th, cv2.LINE_AA)
    return img, {"square_px": square_px, "square_mm_actual": square_px / px_per_mm, "border_px": border,
                 "origin": (border, border), "px_per_mm": px_per_mm, "label_px": strip}


def write_png(path, img, dpi=None):
    """cv2.imwrite, then the pHYs chunk cv2 does not write, so a viewer prints the picture at
    its size instead of fitting it to the page."""
    import cv2
    if not cv2.imwrite(path, img):
        raise events.StageError(f"cannot write {path}")
    if dpi:
        _png_set_dpi(path, dpi)


def _png_set_dpi(path, dpi):
    data = open(path, "rb").read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return
    ppm = int(round(dpi / 0.0254))
    body = struct.pack(">IIB", ppm, ppm, 1)
    chunk = struct.pack(">I", len(body)) + b"pHYs" + body + struct.pack(">I", zlib.crc32(b"pHYs" + body) & 0xFFFFFFFF)
    # after IHDR (8 sig + 4 len + 4 type + 13 data + 4 crc)
    end_ihdr = 8 + 4 + 4 + 13 + 4
    out = data[:end_ihdr] + chunk + data[end_ihdr:]
    with open(path, "wb") as f:
        f.write(out)


def write_pdf(path, img, px_per_mm, pages=PAGES_MM, spec=None):
    """A minimal PDF, no library: the grey image Flate-compressed once, drawn at exact size on a
    page per paper size (landscape when the picture is wider than tall). StageError, naming the
    largest square that would fit (when `spec` is given), when the picture does not fit a page."""
    h, w = img.shape[:2]
    w_mm, h_mm = w / px_per_mm, h / px_per_mm
    pt = 72.0 / 25.4
    objs = []

    def add(body):
        objs.append(body)
        return len(objs)

    cat = add(None)
    pages_obj = add(None)
    im_obj = add(b"<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceGray "
                 b"/BitsPerComponent 8 /Filter /FlateDecode /Length %d >>\nstream\n" % (w, h, 0))   # length patched below
    im_stream = zlib.compress(np.ascontiguousarray(img).tobytes(), 9)
    objs[im_obj - 1] = (b"<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceGray "
                        b"/BitsPerComponent 8 /Filter /FlateDecode /Length %d >>\nstream\n" % (w, h, len(im_stream))
                        + im_stream + b"\nendstream")
    kids = []
    for name, pw, ph in pages:
        if w_mm > h_mm and pw < ph:
            pw, ph = ph, pw
        if w_mm > pw + 1e-6 or h_mm > ph + 1e-6:
            raise events.StageError(f"the board ({w_mm:.0f} x {h_mm:.0f} mm) does not fit a {name} page ({pw:g} x {ph:g} mm)",
                                    hint=(f"--square-mm {np.floor(max_square_mm(spec, pw, ph)):.0f} or fewer squares" if spec
                                          else "a smaller square or fewer squares"))
        W, H = pw * pt, ph * pt
        iw, ih = w_mm * pt, h_mm * pt
        x, y = (W - iw) / 2, (H - ih) / 2
        content = f"q {iw:.4f} 0 0 {ih:.4f} {x:.4f} {y:.4f} cm /Im1 Do Q".encode()
        c_obj = add(b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream")
        p_obj = add(f"<< /Type /Page /Parent {pages_obj} 0 R /MediaBox [0 0 {W:.4f} {H:.4f}] "
                    f"/Resources << /XObject << /Im1 {im_obj} 0 R >> >> /Contents {c_obj} 0 R >>".encode())
        kids.append(p_obj)
    objs[cat - 1] = f"<< /Type /Catalog /Pages {pages_obj} 0 R >>".encode()
    objs[pages_obj - 1] = ("<< /Type /Pages /Kids [" + " ".join(f"{k} 0 R" for k in kids) +
                           f"] /Count {len(kids)} >>").encode()
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for o in offsets:
        out += b"%010d 00000 n \n" % o
    out += b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, cat, xref)
    with open(path, "wb") as f:
        f.write(out)
    return [n for n, _, _ in pages]


def max_square_mm(spec, pw, ph, border_mm=BORDER_MM, label_mm=LABEL_MM):
    """The largest square (mm) whose board fits a pw x ph mm page in either orientation."""
    best = 0.0
    for a, b in ((pw, ph), (ph, pw)):
        best = max(best, min((a - 2 * border_mm) / spec.sx, (b - 2 * border_mm - label_mm) / spec.sy))
    return best


def screen_px_per_mm(diag_in, w_px, h_px):
    """Pixels per millimetre of a screen from its diagonal and pixel size."""
    if diag_in <= 0 or w_px <= 0 or h_px <= 0:
        raise events.StageError("--screen-in and --screen-px must be positive")
    return float(np.hypot(w_px, h_px) / diag_in / 25.4)


def render_screen(spec, diag_in, w_px, h_px):
    """The board centred on a white w_px x h_px picture, squares of --square-mm at this screen's
    pixel pitch; smaller (and said so) when that does not fit. -> (img, info)."""
    ppmm = screen_px_per_mm(diag_in, w_px, h_px)
    border_mm = BORDER_MM
    fit = max_square_mm(spec, w_px / ppmm, h_px / ppmm, border_mm, LABEL_MM)
    sq_mm = min(spec.square_mm, fit)
    if sq_mm < 4:
        raise events.StageError(f"a {spec.sx}x{spec.sy} board does not fit a {w_px}x{h_px} screen", hint="fewer squares")
    s2 = B.BoardSpec(spec.sx, spec.sy, sq_mm, sq_mm * spec.marker_mm / spec.square_mm, spec.dictionary, spec.legacy)
    img, info = render_board(s2, ppmm, border_mm, LABEL_MM, exact=False)
    if img.shape[1] > w_px or img.shape[0] > h_px:        # whole-pixel rounding can overshoot by a px
        img = img[:h_px, :w_px]
    canvas = np.full((h_px, w_px), 255, np.uint8)
    y0, x0 = (h_px - img.shape[0]) // 2, (w_px - img.shape[1]) // 2
    canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    info.update({"origin": (x0 + info["origin"][0], y0 + info["origin"][1]), "px_per_mm": ppmm,
                 "square_mm_asked": spec.square_mm, "fits": sq_mm >= spec.square_mm - 1e-9})
    return canvas, info


# ------------------------------------------------------------------ frames in

def decode_frames(ffmpeg, clip, out_dir, every=10, max_frames=60):
    """Every `every`'th frame of the clip as grey PNGs (autorotated, as ffmpeg displays it),
    at most max_frames of them, in out_dir. -> sorted paths."""
    exe = shutil.which(os.path.expanduser(ffmpeg)) or ffmpeg
    os.makedirs(out_dir, exist_ok=True)
    every = max(1, int(every))
    argv = [exe, "-y", "-v", "error", "-nostdin", "-i", clip, "-map", "0:v:0", "-an",
            "-vf", f"select=not(mod(n\\,{every}))", "-vsync", "vfr", "-frames:v", str(int(max_frames)),
            "-pix_fmt", "gray", os.path.join(out_dir, "f%05d.png")]
    try:
        r = subprocess.run(argv, capture_output=True, text=True)
    except FileNotFoundError:
        raise events.StageError(f"ffmpeg not found ({ffmpeg})", hint="brew install ffmpeg")
    paths = sorted(glob.glob(os.path.join(out_dir, "f*.png")))
    if r.returncode != 0 or not paths:
        raise events.StageError(f"ffmpeg could not decode {os.path.basename(clip)}",
                                hint=(r.stderr or "").strip()[-300:] or "not a video file?")
    return paths


def still_paths(folder, max_frames=60):
    """The stills in a folder, evenly thinned to max_frames."""
    ps = sorted(p for p in glob.glob(os.path.join(os.path.expanduser(folder), "*"))
                if p.lower().endswith(FRAME_EXTS) and not os.path.basename(p).startswith("."))
    if len(ps) > max_frames > 0:
        idx = np.round(np.linspace(0, len(ps) - 1, max_frames)).astype(int)
        ps = [ps[i] for i in sorted(set(idx))]
    return ps


def read_gray(path):
    import cv2
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise events.StageError(f"unreadable frame {os.path.basename(path)}")
    return im


# ------------------------------------------------------------------ the fit

def coverage_share(xys, size, cells=COVER_CELLS):
    """Share of a cells[0] x cells[1] grid over the frame that saw at least one corner."""
    w, h = size
    seen = set()
    for xy in xys:
        for x, y in np.asarray(xy, float).reshape(-1, 2):
            i = min(cells[0] - 1, max(0, int(x * cells[0] / w)))
            j = min(cells[1] - 1, max(0, int(y * cells[1] / h)))
            seen.add((i, j))
    return len(seen) / float(cells[0] * cells[1]), seen


def calibrate(frames, spec, min_corners=MIN_CORNERS, detector=None, on_frame=None):
    """frames: iterable of (name, grey uint8 image), all one size. Detects the board in each,
    fits one camera over the frames with >= min_corners corners, then refits without the frames
    whose own reprojection error is over 3x the median (and over 1 px): a frame the first fit
    cannot explain is blurred or half-seen, not evidence about the lens.

    -> {"K", "dist" (5,), "rms_px", "image_size", "frames": [{"name", "corners", "used", "err_px"}],
        "frames_seen", "frames_used", "corners_total", "coverage" {"cells", "share"},
        "std_intrinsics" [fx fy cx cy k1 k2 p1 p2 k3]}. ValueError when fewer than 3 frames see it."""
    import cv2
    det = detector or B.Detector(spec)
    obj_all = B.corner_points(spec).astype(np.float32)
    rows, obj, img, size = [], [], [], None
    for i, (name, im) in enumerate(frames):
        if size is None:
            size = (im.shape[1], im.shape[0])
        elif (im.shape[1], im.shape[0]) != size:
            raise ValueError(f"{name} is {im.shape[1]}x{im.shape[0]}, the first frame {size[0]}x{size[1]}: one size per calibration")
        ids, xy = det.detect(im, min_corners)
        rows.append({"name": name, "corners": int(len(ids)), "used": len(ids) >= min_corners, "err_px": None})
        if len(ids) >= min_corners:
            obj.append(obj_all[ids])
            img.append(xy.astype(np.float32).reshape(-1, 1, 2))
        if on_frame:
            on_frame(i + 1, rows[-1])
    used = [r for r in rows if r["used"]]
    if len(used) < 3:
        raise ValueError(f"the board was found in {len(used)} of {len(rows)} frames; a fit needs at least 3")

    def fit(oo, ii):
        res = cv2.calibrateCameraExtended(oo, ii, size, None, None)
        rms, K, dist, _rv, _tv, std_i, _std_e, per = res[:8]
        return float(rms), K, np.asarray(dist, float).ravel()[:5], np.asarray(per, float).ravel(), np.asarray(std_i, float).ravel()

    rms, K, dist, per, std_i = fit(obj, img)
    for r, e in zip(used, per):
        r["err_px"] = round(float(e), 4)
    med = float(np.median(per))
    keep = per <= max(3.0 * med, 1.0)
    if keep.sum() >= 3 and not keep.all():
        rms, K, dist, per2, std_i = fit([o for o, k in zip(obj, keep) if k], [p for p, k in zip(img, keep) if k])
        it = iter(per2)
        for r, k in zip(used, keep):
            if k:
                r["err_px"] = round(float(next(it)), 4)
            else:
                r["used"] = False
                r["dropped"] = "could not be explained by the first fit (blur, or a half-seen board)"
    used_xy = [p for p, r in zip(img, used) if r["used"]]
    share, _cells = coverage_share(used_xy, size)
    return {"K": np.asarray(K, float), "dist": np.asarray(dist, float).ravel()[:5], "rms_px": rms, "image_size": list(size),
            "frames": rows, "frames_seen": len(rows), "frames_used": int(sum(r["used"] for r in rows)),
            "corners_total": int(sum(len(p) for p in used_xy)),
            "coverage": {"cells": list(COVER_CELLS), "share": round(share, 4)},
            "std_intrinsics": [round(float(x), 6) for x in std_i[:9]]}


def coverage_picture(result, xys, path):
    """Every used corner on a dark frame, the 6 x 4 grid over it: the Calibrate page's picture."""
    import cv2
    w, h = result["image_size"]
    s = 960.0 / max(w, 1)
    im = np.full((int(round(h * s)), 960, 3), 40, np.uint8)
    cw, ch = 960 / COVER_CELLS[0], im.shape[0] / COVER_CELLS[1]
    _, seen = coverage_share(xys, (w, h))
    for i in range(COVER_CELLS[0]):
        for j in range(COVER_CELLS[1]):
            col = (60, 90, 60) if (i, j) in seen else (40, 40, 90)
            cv2.rectangle(im, (int(i * cw), int(j * ch)), (int((i + 1) * cw) - 1, int((j + 1) * ch) - 1), col, -1)
    for xy in xys:
        for x, y in np.asarray(xy, float).reshape(-1, 2):
            cv2.circle(im, (int(round(x * s)), int(round(y * s))), 2, (230, 230, 230), -1, cv2.LINE_AA)
    cv2.imwrite(path, im)


# ------------------------------------------------------------------ run

def run(a, pj=None):
    R = pj if pj is not None else events          # metric/check/artifact have the same shape on both
    any_board = a.board_image or a.board_pdf or a.board_screen
    if not (any_board or a.clip or a.frames):
        raise events.StageError("nothing to do", hint="--board-image OUT.png, --board-screen OUT.png, or --clip BOARD.mov / --frames DIR")
    if B.aruco_api() is None:
        raise events.StageError("this OpenCV has no cv2.aruco", hint="pip install opencv-contrib-python (same version as opencv-python)")
    spec = spec_from_args(a)
    if any_board:
        _boards(a, spec, R)
    if a.clip or a.frames:
        return _lens(a, spec, pj, R)        # the profile, for callers in-process (tests)


def _boards(a, spec, R):
    import cv2
    R.metric(STAGE, "board", spec.text())
    if a.board_image or a.board_pdf:
        img, info = render_board(spec, PRINT_DPI / 25.4)
        ppmm = info["px_per_mm"]                 # ~300 dpi, nudged so the whole-pixel square is exactly --square-mm
        R.metric(STAGE, "print_dpi", round(ppmm * 25.4, 2))
        R.metric(STAGE, "print_square_px", info["square_px"])
        R.metric(STAGE, "print_size_mm", [round(img.shape[1] / ppmm, 1), round(img.shape[0] / ppmm, 1)])
        if a.board_image:
            out = os.path.abspath(os.path.expanduser(a.board_image))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            write_png(out, img, ppmm * 25.4)
            events.artifact(STAGE, out, "board_image")       # outside the project: the event only
        if a.board_pdf:
            out = os.path.abspath(os.path.expanduser(a.board_pdf))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            for name, pw, ph in PAGES_MM:
                fit = max_square_mm(spec, pw, ph)
                R.check(STAGE, f"board_fits_{name.lower()}", spec.square_mm <= fit + 1e-9,
                        value=f"{spec.sx}x{spec.sy} at {spec.square_mm:g} mm needs {img.shape[1] / ppmm:.0f} x {img.shape[0] / ppmm:.0f} mm; "
                              f"{name} takes squares up to {fit:.1f} mm" + ("" if spec.square_mm <= fit + 1e-9 else
                              f" — print with --square-mm {np.floor(fit):.0f}, or fewer squares"))
            write_pdf(out, img, ppmm, spec=spec)
            events.artifact(STAGE, out, "board_pdf")
    if a.board_screen:
        if not (a.screen_in and a.screen_px):
            raise events.StageError("--board-screen needs --screen-in DIAGONAL and --screen-px WxH")
        w_px, h_px = parse_squares(a.screen_px)
        img, info = render_screen(spec, a.screen_in, w_px, h_px)
        out = os.path.abspath(os.path.expanduser(a.board_screen))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        write_png(out, img)
        events.artifact(STAGE, out, "board_screen")
        R.metric(STAGE, "screen_px_per_mm", round(info["px_per_mm"], 4))
        R.metric(STAGE, "screen_square_px", info["square_px"])
        R.metric(STAGE, "screen_square_mm", round(info["square_mm_actual"], 3))
        R.check(STAGE, "board_fits_screen", info["fits"],
                value=(f"{spec.sx}x{spec.sy} at {info['square_mm_actual']:.1f} mm squares on a {a.screen_in:g}\" "
                       f"{w_px}x{h_px} screen ({info['px_per_mm'] * 25.4:.0f} ppi)" +
                       ("" if info["fits"] else f" — {a.square_mm:g} mm does not fit; the squares are {info['square_mm_actual']:.1f} mm "
                                                "(fine for a lens calibration: the board's size does not enter K)")))


def _lens(a, spec, pj, R):
    import cv2
    if a.clip and a.frames:
        raise events.StageError("--clip or --frames, not both")
    src = os.path.abspath(os.path.expanduser(a.clip or a.frames))
    if not os.path.exists(src):
        raise events.StageError(f"{src} does not exist")
    work = pj.path(STAGE) if pj is not None else tempfile.mkdtemp(prefix="hs-calibrate-")
    if pj is not None:
        pj.acquire(STAGE)
        st = pj.stage(STAGE)
        st.update({"status": "running", "started": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                   "finished": None, "argv": list(sys.argv), "metrics": {}, "checks": [], "artifacts": [], "error": None})
        pj.save()
        if os.path.isdir(work):
            shutil.rmtree(work)
    os.makedirs(work, exist_ok=True)
    R.metric(STAGE, "board", spec.text())

    # ---- the camera's identity
    meta = {"make": a.make, "model": a.model, "lens": a.lens_name}
    if a.clip:
        events.start(STAGE, "probe")
        probe = sourceprobe.ffprobe_json(a.ffprobe, src)
        facts, problems = sourceprobe.video_facts(probe)
        if facts is None:
            raise events.StageError("clip rejected: " + "; ".join(problems))
        tags = lens.meta_from_probe(probe, facts["width"], facts["height"])
        for k in ("make", "model", "lens", "software"):
            meta.setdefault(k, None)
            meta[k] = meta[k] or tags.get(k)
        events.start(STAGE, "decode")
        paths = decode_frames(a.ffmpeg, src, os.path.join(work, "frames"), a.every, a.max_frames)
    else:
        paths = still_paths(src, a.max_frames)
        if not paths:
            raise events.StageError(f"no stills in {src}", hint=f"{', '.join(FRAME_EXTS)}")
    R.metric(STAGE, "frames_decoded", len(paths))

    # ---- detect + fit
    events.start(STAGE, "detect")
    n = len(paths)

    def frames():
        for p in paths:
            yield os.path.basename(p), read_gray(p)

    def on_frame(i, row):
        events.progress(STAGE, i, n, step="detect", detail=f"{row['corners']} corners in {row['name']}")

    try:
        res = calibrate(frames(), spec, a.min_corners, on_frame=on_frame)
    except ValueError as e:
        raise events.StageError(str(e), hint="film the whole board, well lit and in focus, filling a third of the frame or more; "
                                             "move it to every part of the frame and tilt it")
    w, h = res["image_size"]
    meta["width"], meta["height"] = w, h
    camera = {k: meta.get(k) or lens.UNKNOWN.get(k) for k in ("make", "model", "lens")}
    camera["software"] = meta.get("software")
    K, dist = res["K"], res["dist"]
    hfov = float(2 * np.degrees(np.arctan(w / (2 * K[0, 0]))))
    for name, val in (("frames_with_board", sum(1 for r in res["frames"] if r["corners"] >= a.min_corners)),
                      ("frames_used", res["frames_used"]), ("corners_total", res["corners_total"]),
                      ("rms_px", round(res["rms_px"], 4)), ("coverage_share", res["coverage"]["share"]),
                      ("fx_px", round(float(K[0, 0]), 2)), ("fy_px", round(float(K[1, 1]), 2)),
                      ("cx_px", round(float(K[0, 2]), 2)), ("cy_px", round(float(K[1, 2]), 2)),
                      ("k1", round(float(dist[0]), 6)), ("k2", round(float(dist[1]), 6)), ("p1", round(float(dist[2]), 6)),
                      ("p2", round(float(dist[3]), 6)), ("k3", round(float(dist[4]), 6)), ("hfov_deg", round(hfov, 2)),
                      ("fx_std_px", res["std_intrinsics"][0]), ("image_size", [w, h]), ("camera", camera)):
        R.metric(STAGE, name, val)
    dropped = [r["name"] for r in res["frames"] if r.get("dropped")]
    if dropped:
        R.metric(STAGE, "frames_dropped", dropped)

    # ---- checks: a number, and what to do
    ok_frames = res["frames_used"] >= MIN_FRAMES
    ok_cover = res["coverage"]["share"] >= MIN_COVER
    ok_rms = res["rms_px"] <= MAX_RMS_PX
    R.check(STAGE, "board_seen_in_enough_frames", ok_frames,
            value=f"{res['frames_used']} frames with >= {a.min_corners} corners (want >= {MIN_FRAMES}) — "
                  + ("good" if ok_frames else "film longer with the whole board in view, in focus, filling a third of the frame"))
    R.check(STAGE, "calibration_covers_the_frame", ok_cover,
            value=f"corners in {res['coverage']['share'] * 100:.0f}% of a {COVER_CELLS[0]}x{COVER_CELLS[1]} grid (want >= {MIN_COVER * 100:.0f}%) — "
                  + ("good" if ok_cover else "move the board to the edges and corners of the frame, where the distortion is"))
    R.check(STAGE, "reprojection_rms_small", ok_rms, needs_human=not ok_rms,
            value=f"{res['rms_px']:.3f} px rms (want <= {MAX_RMS_PX}) — "
                  + ("good" if ok_rms else "hold the board still and flat, keep it sharp, and record with the same setting as the shoot; "
                                           f"{len(dropped)} blurred frames were already dropped"))
    ok = bool(ok_frames and ok_cover and ok_rms)

    # ---- the profile
    profile = {"version": 1, "camera": camera, "image_size": [w, h], "model": "OPENCV",
               "K": np.round(K, 6).tolist(), "dist": np.round(dist, 8).tolist(), "rms_px": round(res["rms_px"], 4),
               "frames_used": res["frames_used"], "frames_seen": res["frames_seen"], "corners_total": res["corners_total"],
               "coverage": res["coverage"], "board": spec.text(), "source": os.path.basename(src),
               "std_intrinsics": res["std_intrinsics"], "ok": ok, "hs_version": __version__,
               "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
    profile["key"] = lens.key_for({**camera, "width": w, "height": h})
    path = lens.save(profile)
    R.metric(STAGE, "profile_key", profile["key"])
    R.metric(STAGE, "profile_path", path)
    events.artifact(STAGE, path, "lens_profile")              # the store is outside any project
    if a.out:
        lens.save(profile, os.path.abspath(os.path.expanduser(a.out)))
        events.artifact(STAGE, os.path.abspath(os.path.expanduser(a.out)), "lens_profile")
    if pj is not None:
        local = os.path.join(work, "profile.json")
        lens.save(profile, local)
        pj.artifact(STAGE, local, "lens_profile")
        det_path = os.path.join(work, "detections.json")
        json.dump({"frames": res["frames"], "image_size": [w, h], "board": spec.text()}, open(det_path, "w"), indent=1)
        pj.artifact(STAGE, det_path, "json")
        xys = _used_corners(paths, res, spec, a.min_corners)
        cov_path = os.path.join(work, "coverage.jpg")
        coverage_picture(res, xys, cov_path)
        pj.artifact(STAGE, cov_path, "image")
        pj.m["lens_profile"] = {"path": path, "key": profile["key"], "rms_px": profile["rms_px"], "date": profile["date"]}
        if a.clip and not getattr(a, "keep_frames", False):
            shutil.rmtree(os.path.join(work, "frames"), ignore_errors=True)    # 60 grey 4K PNGs: the fit has what it needs
        pj.finish(STAGE, ok=True)      # the fit ran; a failed check is the verdict, not a crash
    else:
        shutil.rmtree(work, ignore_errors=True)
    return profile


def _used_corners(paths, res, spec, min_corners):
    """Re-detect on the used frames for the coverage picture (cheap next to the fit)."""
    det = B.Detector(spec)
    used = {r["name"] for r in res["frames"] if r["used"]}
    out = []
    for p in paths:
        if os.path.basename(p) in used:
            ids, xy = det.detect(read_gray(p), min_corners)
            if len(ids):
                out.append(xy)
    return out
