"""Measure the frames hs select wrote, both eyes, and draw the contact sheet.

The selector sees only the left eye at work width while it walks the clip. The written frames
are full-resolution 2x1 JPEGs, so this is where the right eye and noise get measured, the
per-frame thumbnails the app shows are cut, and contact.jpg is drawn with each pick's numbers
and flags on it.
"""
import os
import re

import numpy as np

FILE_RE = re.compile(r"^VID_(\d+)_(\d+)_2x1\.jpg$", re.I)
_LINEAR = ((np.arange(256) / 255.0) ** 2.2).astype(np.float64)


def frame_files(frames_dir):
    """{source frame index: file name} for the frames the selector wrote."""
    out = {}
    for f in sorted(os.listdir(frames_dir)):
        m = FILE_RE.match(f)
        if m:
            out[int(m.group(2))] = f
    return out


def noise_estimate(gray):
    """Median |Laplacian| over flat regions at 960x540 — the estimator hs select always used."""
    import cv2
    gs = cv2.resize(gray, (960, 540), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(gs, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gs, cv2.CV_32F, 0, 1, ksize=3)
    flat = (np.abs(gx) + np.abs(gy)) < 8.0
    if flat.sum() <= 1000:
        return None
    lap = np.abs(cv2.Laplacian(gs, cv2.CV_32F, ksize=1))
    return float(lap[flat].mean())


def measure(frames_dir, thumbs_dir, work_width=480, thumb_w=360):
    """{source frame: {file, thumb, sharp_R, std_R, luma_R, clip_R, noise}}; writes one left-eye
    thumbnail per frame into thumbs_dir."""
    import cv2
    os.makedirs(thumbs_dir, exist_ok=True)
    out = {}
    for fi, name in frame_files(frames_dir).items():
        im = cv2.imread(os.path.join(frames_dir, name), cv2.IMREAD_COLOR)
        if im is None:
            continue
        half = im.shape[1] // 2
        L, R = im[:, :half], im[:, half:]
        gL = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY)
        gR = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)
        h = int(round(gR.shape[0] * work_width / gR.shape[1]))
        sR = cv2.resize(gR, (work_width, h), interpolation=cv2.INTER_AREA)
        th = cv2.resize(L, (thumb_w, int(round(thumb_w * L.shape[0] / L.shape[1]))), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(thumbs_dir, name), th, [cv2.IMWRITE_JPEG_QUALITY, 88])
        out[fi] = {
            "file": name,
            "thumb": os.path.join(os.path.basename(thumbs_dir), name),
            "sharp_R": float(cv2.Laplacian(sR, cv2.CV_64F).var()),
            "std_R": float(sR.std()),
            "luma_R": float(_LINEAR[sR].mean()),
            "clip_R": float((gR >= 250).mean()),
            "noise": noise_estimate(gL),
        }
    return out


def contact_sheet(frames_dir, quality, out_path, thumb_w=240, per_row=8):
    """contact.jpg: every pick, left eye, labelled with its numbers; flagged picks framed in
    orange with their flags written underneath."""
    import cv2
    rows_out = []
    for fr in quality["frames"]:
        if not fr.get("file"):
            continue
        im = cv2.imread(os.path.join(frames_dir, fr["file"]), cv2.IMREAD_COLOR)
        if im is None:
            continue
        L = im[:, : im.shape[1] // 2]
        th = cv2.resize(L, (thumb_w, int(round(thumb_w * L.shape[0] / L.shape[1]))), interpolation=cv2.INTER_AREA)
        h, w = th.shape[:2]
        cv2.rectangle(th, (0, 0), (w, 30), (0, 0, 0), -1)
        ev = fr.get("ev")
        line1 = f"#{fr['sel']}  f{fr['frame']}  gap {fr.get('gap') or '-'}"
        fo = fr.get("focus_rel")
        line2 = (f"lap {fr['sharp']:.0f}  focus {'-' if fo is None else f'{fo:.2f}x'}"
                 f"  {'' if ev is None else f'{ev:+.2f} EV'}")
        cv2.putText(th, line1, (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(th, line2, (4, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
        if fr["flags"]:
            cv2.rectangle(th, (0, h - 16), (w, h), (0, 0, 0), -1)
            cv2.putText(th, " ".join(fr["flags"])[:44], (4, h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                        (0, 165, 255), 1, cv2.LINE_AA)
            cv2.rectangle(th, (0, 0), (w - 1, h - 1), (0, 165, 255), 3)
        rows_out.append(th)
    if not rows_out:
        return None
    h, w = rows_out[0].shape[:2]
    rows = (len(rows_out) + per_row - 1) // per_row
    sheet = np.zeros((rows * h, per_row * w, 3), np.uint8)
    for i, th in enumerate(rows_out):
        r, c = divmod(i, per_row)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = th[:h, :w]
    cv2.imwrite(out_path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out_path
