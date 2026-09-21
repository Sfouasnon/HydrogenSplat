#!/usr/bin/env python3
"""
select_frames.py — pick stereo video frames for SfM by PARALLAX, not by motion.

Reads a Holocam 2x1 clip (VID_*_2x1.h4v — a plain MP4 with two 1920x1080 eyes side by
side), walks it frame by frame on the left eye, and selects a frame whenever the camera
has TRANSLATED enough since the last selected frame.

Why not optical flow: flow magnitude conflates rotation with translation. A fast pan gives
large flow and near-zero baseline, the essential matrix degenerates and the solve collapses
(seen: 1.7-2.1 mm camera steps, negative scales, 16/68 registered). A pure rotation is
exactly a homography, so tracking features from the last selected frame, fitting a
homography, and taking the median residual after removing it isolates the depth-bearing
translation. Select when that residual reaches --residual px at --work-width.

The residual here is the median over forward-backward-verified tracks after a RANSAC
homography, which reads lower than the earlier ad-hoc selector's figure: the default 1.5 px
was calibrated on VID_20260912_151351 to reproduce the frames4 selection that solved
65/65 (66 frames vs 65, median gap 9, 58 of 66 within 3 frames of the original picks;
3.5 px gives only 35 frames with this measurement).

Among the frames just past the crossing (--search frames), the sharpest one with clipping
below --max-clip is taken, so the SfM sees the least motion blur available.

Do NOT add a "flow > N" escape hatch — it reintroduces the pan problem.

Output: DIR/VID_NNN_FFFF_2x1.jpg (full-resolution 2x1 frames, NNN = selection index,
FFFF = source frame index) plus DIR/selection.json. --dry-run writes only the JSON.

selection.json also carries a per-frame "trace" of every frame read (left eye, at work width):
Laplacian variance, contrast (grey std), linear-light mean luma, clipped and crushed fractions — and, for each pick,
the candidates it beat. hs select turns those into quality.json: what the picks look like
against their neighbours, and where a better frame sat.

Usage:
  python3 select_frames.py capture_video/VID_..._2x1.h4v -o capture_video/frames5
        [--residual 1.5] [--min-gap 6] [--max-gap 90] [--search 4] [--max-clip 0.02]
        [--work-width 480] [--start 0] [--end -1] [--dry-run] [--mono]

--mono: the clip is one ordinary camera, not a 2x1 pair — an iPhone orbit, say. Nothing about
the selection changes, because none of it was ever stereo: the residual is fitted between two
frames of ONE view (rotation is a homography, translation is not), and the left-eye crop was
only ever a way to get one view out of a side-by-side frame. In mono the whole frame is that
view, and the picks are written as selNNN-FFFFF.jpg, a name `hs ingest --frames` accepts as a view name
(letters, digits and '-' only). Scale is the thing a mono clip loses,
not selection: see monocolmap.py --scale / --scale-pair.
"""
import argparse, json, os, sys
import numpy as np
import cv2


def left_eye(frame):
    return frame[:, : frame.shape[1] // 2]


def whole(frame):
    """--mono: the frame already IS one view."""
    return frame


def prep(gray_full, work_width):
    h, w = gray_full.shape
    s = work_width / w
    return cv2.resize(gray_full, (work_width, int(round(h * s))), interpolation=cv2.INTER_AREA)


def parallax_residual(ref_small, ref_pts, cand_small):
    """Median residual (px at work width) after the best homography from ref to cand.
    Returns (residual, n_tracked). residual is None when tracking fails."""
    if ref_pts is None or len(ref_pts) < 12:
        return None, 0
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(ref_small, cand_small, ref_pts, None,
                                          winSize=(21, 21), maxLevel=3,
                                          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    if nxt is None:
        return None, 0
    ok = st.reshape(-1) == 1
    # back-track to reject drifting tracks
    back, st2, _ = cv2.calcOpticalFlowPyrLK(cand_small, ref_small, nxt, None, winSize=(21, 21), maxLevel=3)
    if back is not None:
        ok &= (st2.reshape(-1) == 1) & (np.linalg.norm((back - ref_pts).reshape(-1, 2), axis=1) < 1.0)
    p0 = ref_pts.reshape(-1, 2)[ok]
    p1 = nxt.reshape(-1, 2)[ok]
    if len(p0) < 12:
        return None, int(ok.sum())
    H, _ = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
    if H is None:
        return None, len(p0)
    proj = cv2.perspectiveTransform(p0.reshape(-1, 1, 2), H).reshape(-1, 2)
    res = np.linalg.norm(proj - p1, axis=1)
    return float(np.median(res)), len(p0)


def sharpness(gray_small):
    return float(cv2.Laplacian(gray_small, cv2.CV_64F).var())


def clipping(gray_full):
    return float((gray_full >= 250).mean())


# 8-bit sRGB-ish code value -> linear light (gamma 2.2), so an exposure offset reads in stops
_LINEAR = ((np.arange(256) / 255.0) ** 2.2).astype(np.float64)


def linear_luma(gray):
    """Mean linear-light luminance, 0..1. log2 of a ratio of these is an offset in stops."""
    return float(_LINEAR[gray].mean())


def crushed(gray):
    return float((gray <= 5).mean())


def features(gray_small):
    return cv2.goodFeaturesToTrack(gray_small, maxCorners=1500, qualityLevel=0.01, minDistance=6, blockSize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--residual", type=float, default=1.5, help="median homography residual, px at work width")
    ap.add_argument("--min-gap", type=int, default=6)
    ap.add_argument("--max-gap", type=int, default=90)
    ap.add_argument("--search", type=int, default=4, help="frames after the crossing to search for the sharpest")
    ap.add_argument("--max-clip", type=float, default=0.02, help="max fraction of pixels >= 250")
    ap.add_argument("--work-width", type=int, default=480)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1, help="last frame index to consider (-1 = end of clip)")
    ap.add_argument("--dry-run", action="store_true", help="write selection.json only, no frames")
    ap.add_argument("--mono", action="store_true",
                    help="one ordinary camera, not a 2x1 stereo pair: use the whole frame as the view")
    a = ap.parse_args()
    view = whole if a.mono else left_eye
    # Stereo picks keep the VID_NNN_FFFF_2x1 name hs select's measurements parse. A mono pick
    # becomes a *view name* in `hs ingest --frames`, which allows only letters, digits and '-'
    # (an underscore would collide with the _L/_R suffix rig.py strips), so: selNNN-FFFFF.
    def pick_name(k, fi):
        return f"sel{k:03d}-{fi:05d}.jpg" if a.mono else f"VID_{k:03d}_{fi:04d}_2x1.jpg"

    cap = cv2.VideoCapture(a.clip)
    if not cap.isOpened():
        sys.exit(f"cannot open {a.clip}")
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); Hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{a.clip}: {n_total} frames at {fps:.2f} fps, {W}x{Hh}"
          + (" (mono)" if a.mono else f" (2x1 -> {W//2}x{Hh} per eye)"))
    os.makedirs(a.out, exist_ok=True)

    selected = []          # dicts: sel, frame, residual, sharpness, clip
    ref = None             # (small_gray, pts) of the last selected frame
    last_sel = None
    pending = []           # candidates past the crossing: (frame_idx, frame_bgr, res, sharp, clip)
    crossing_at = None
    trace = {"frame": [], "sharp": [], "std": [], "luma": [], "clip": [], "dark": [], "residual": []}
    fi = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        if fi < a.start:
            continue
        if a.end >= 0 and fi > a.end:
            break
        gl = cv2.cvtColor(view(frame), cv2.COLOR_BGR2GRAY)
        small = prep(gl, a.work_width)
        # every frame read, not only the picks: the picks are judged against their neighbours
        trace["frame"].append(fi)
        trace["sharp"].append(round(sharpness(small), 2))
        trace["std"].append(round(float(small.std()), 3))
        trace["luma"].append(round(linear_luma(small), 6))
        trace["clip"].append(round(clipping(gl), 5))
        trace["dark"].append(round(crushed(small), 5))
        trace["residual"].append(None)

        if ref is None:
            # first frame is always selected
            selected.append({"sel": 0, "frame": fi, "residual": 0.0, "sharpness": sharpness(small), "clip": clipping(gl)})
            ref = (small, features(small)); last_sel = fi
            if not a.dry_run:
                cv2.imwrite(os.path.join(a.out, pick_name(0, fi)), frame, [cv2.IMWRITE_JPEG_QUALITY, 97])
            continue

        gap = fi - last_sel
        if gap < a.min_gap:
            continue
        res, ntr = parallax_residual(ref[0], ref[1], small)
        trace["residual"][-1] = None if res is None else round(res, 3)
        crossed = (res is not None and res >= a.residual) or gap >= a.max_gap or (res is None and gap >= a.min_gap * 2)
        if crossing_at is None and crossed:
            crossing_at = fi
        if crossing_at is not None:
            pending.append((fi, frame, res if res is not None else -1.0, sharpness(small), clipping(gl)))
            done = fi >= crossing_at + a.search - 1 or gap >= a.max_gap
            if done:
                # sharpest with acceptable clipping; fall back to sharpest overall
                good = [p for p in pending if p[4] < a.max_clip] or pending
                pick = max(good, key=lambda p: p[3])
                pfi, pframe, pres, psharp, pclip = pick
                k = len(selected)
                selected.append({"sel": k, "frame": pfi, "residual": pres, "sharpness": psharp, "clip": pclip,
                                 "tracked": ntr, "gap": pfi - last_sel,
                                 "candidates": [[p[0], round(p[3], 2), round(p[4], 5)] for p in pending]})
                if not a.dry_run:
                    cv2.imwrite(os.path.join(a.out, pick_name(k, pfi)), pframe, [cv2.IMWRITE_JPEG_QUALITY, 97])
                pg = cv2.cvtColor(view(pframe), cv2.COLOR_BGR2GRAY)
                psmall = prep(pg, a.work_width)
                ref = (psmall, features(psmall)); last_sel = pfi
                pending = []; crossing_at = None
                why = "max-gap" if pfi - (selected[-2]["frame"]) >= a.max_gap else "parallax"
                print(f"  sel {k:3d}  frame {pfi:5d}  gap {selected[-1]['gap']:3d}  residual {pres:5.2f} px  "
                      f"sharp {psharp:7.1f}  clip {pclip*100:4.1f}%  [{why}]")
    cap.release()

    gaps = [s["gap"] for s in selected if "gap" in s]
    print(f"\n{len(selected)} frames selected of {fi+1}; gaps {min(gaps) if gaps else 0}-{max(gaps) if gaps else 0} "
          f"(median {int(np.median(gaps)) if gaps else 0})")
    json.dump({"clip": os.path.abspath(a.clip), "fps": fps, "frames_total": fi + 1, "mono": bool(a.mono),
               "params": vars(a), "selected": selected, "trace": trace},
              open(os.path.join(a.out, "selection.json"), "w"), indent=1)
    print(f"wrote {os.path.join(a.out, 'selection.json')}" + ("" if a.dry_run else f" and {len(selected)} frames"))


if __name__ == "__main__":
    main()
