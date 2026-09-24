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
        [--keyframes [--search-keyframes 2] [--min-sharp-rel 0.6]] [--highlight-knee 0.85]
        [--ffprobe ffprobe]

--mono: the clip is one ordinary camera, not a 2x1 pair — an iPhone orbit, say. Nothing about
the selection changes, because none of it was ever stereo: the residual is fitted between two
frames of ONE view (rotation is a homography, translation is not), and the left-eye crop was
only ever a way to get one view out of a side-by-side frame. In mono the whole frame is that
view, and the picks are written as selNNN-FFFFF.jpg, a name `hs ingest --frames` accepts as a view name
(letters, digits and '-' only). Scale is the thing a mono clip loses,
not selection: see monocolmap.py --scale / --scale-pair.

--keyframes: only H.264 I-frames are candidates. The Hydrogen One records Baseline H.264 at
~12 Mbit/s with a GOP of 30: one I-frame, then 29 P-frames each coded as a change from the one
before. The Laplacian rises through every GOP (CirclesSculpture, 2026-09-22) — not because the
P-frames are sharper but because their accumulated compression artefacts read as detail — so
"take the sharpest" drifts to the most artefacted frame. At 1/1000 s motion blur is not what
limits sharpness; the codec is, and the I-frame is the cleanest picture in its GOP. The
I-frame indices come from ffprobe (frame=pict_type, one line per frame; Baseline has no
B-frames, so decode order is display order) before the clip is decoded. The parallax rule
still decides WHEN a frame is due (the residual crossing, --min-gap, --max-gap); the pick is
then the first keyframe at or after the crossing whose quality passes, looking at no more than
--search-keyframes keyframes (the first at or after the crossing, and the ones after it).
Quality = clipped fraction < --max-clip, focus (Laplacian / grey variance, the contrast-
normalised sharpness frame_quality's "soft" flag uses) at least --min-sharp-rel of the running
median over the picks so far, and crushed fraction <= frame_quality.CRUSH. If none passes,
the least bad is taken and the pick says why ("quality_fallback"). The first pick is a keyframe
too. Every pick records "keyframe": true/false in both modes (in the default mode ffprobe runs
alongside the decode, best effort — null when it cannot), and selection.json carries
"keyframes": the I-frame indices, their count and the median spacing (the GOP).

--highlight-knee K (0 < K < 1): each written frame goes through a soft knee, per channel, in
linear light: sRGB 8-bit -> linear, x > K becomes K + (1-K)(1 - exp(-(x-K)/(1-K))), back to
sRGB. [K, 1] compresses smoothly into [K, 1) — same slope at K, order kept, nothing below K
touched — so the top of the range comes off the ceiling with the order of what was recorded
kept (K=0.85: 255 -> 249), instead of hs exposure or training pushing it into a hard clip. In
8 bits the compressed top has fewer codes than it came from (K=0.85: 18 codes into 12), so
neighbouring highlight values can merge; K=0.9 is gentler. A 256-entry table does it; both
eyes get the same curve. It lives here, in select, because
select writes the photographs: solve, masks, train and views all read select/frames, so every
downstream consumer sees the same pictures, and `hs views` compares renders to the knee'd
frames, which is consistent (the model was trained on them). What it is NOT: not exposure
matching (that is `hs exposure`, which moves whole views to a common median) and not a look
(that is `hs grade`, applied to renders). It cannot recover what the sensor clipped: 255 stays
the brightest value, just lower. The selector's own measurements (trace, per-pick clip and
sharpness) are of the decoded source, before the knee — the knee changes no pick; hs select's
right-eye columns are measured on the written, knee'd frames (clip_R reads ~0 there).
"""
import argparse, json, os, shutil, subprocess, sys
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


# --keyframes quality thresholds: the same numbers as frame_quality.SOFT_REL / CRUSH (this script
# runs standalone, so they are copied; tests/test_select_keyframes.py keeps them equal)
SOFT_REL = 0.6
CRUSH = 0.10


def focus(sharp, std):
    """Contrast-normalised Laplacian (= frame_quality.focus): sharp / grey variance. None if flat."""
    if std is None or std < 1.0:
        return None
    return sharp / (std * std)


# ---- keyframes (ffprobe)

PICT_TYPES = {"I", "P", "B", "S", "SI", "SP", "BI", "?"}


def ffprobe_argv(exe, clip):
    """One line per frame of the first video stream, its picture type (I / P / B)."""
    return [exe, "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pict_type",
            "-of", "csv=p=0", clip]


def parse_pict_types(text):
    """ffprobe's frame=pict_type CSV -> ["I", "P", ...], one per frame in decode order.

    Lines can carry a trailing comma ("I," when the frame has side data); blank lines are
    skipped. Anything that is not a picture type raises ValueError, so a stand-in ffprobe or a
    wrong -of reads as an error, not as a clip without keyframes."""
    out = []
    for line in text.splitlines():
        fields = [f.strip() for f in line.strip().split(",") if f.strip()]
        if not fields:
            continue
        t = next((f for f in fields if f in PICT_TYPES), None)
        if t is None:
            raise ValueError(f"not an ffprobe pict_type line: {line.strip()[:60]!r}")
        out.append(t)
    return out


def keyframe_indices(types):
    return [i for i, t in enumerate(types) if t == "I"]


def gop_median(keyframes):
    """Median spacing of the keyframes, in frames (the GOP); None with fewer than two."""
    if len(keyframes) < 2:
        return None
    return float(np.median(np.diff(keyframes)))


def find_ffprobe(name):
    """The ffprobe executable, found the way hs ingest finds it (a name on PATH or a path), or None."""
    return shutil.which(os.path.expanduser(name))


def keyframe_info(types):
    kf = keyframe_indices(types)
    return {"source": "ffprobe frame=pict_type", "frames": kf, "total": len(kf),
            "gop_median": gop_median(kf), "probe_frames": len(types)}


# ---- highlight knee

def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(y):
    return np.where(y <= 0.0031308, y * 12.92, 1.055 * np.power(np.maximum(y, 0.0), 1 / 2.4) - 0.055)


def knee_lut(k):
    """256-entry uint8 table: sRGB code -> sRGB code after a soft knee at linear K.

    Below K (in linear light) the code is returned unchanged; above it
    x -> K + (1-K)(1 - exp(-(x-K)/(1-K))), which meets the identity with slope 1 at K and
    approaches 1 without reaching it, so order is kept (non-decreasing once rounded to 8 bits)."""
    if not 0.0 < k < 1.0:
        raise ValueError(f"highlight knee must be between 0 and 1, got {k}")
    codes = np.arange(256)
    lin = srgb_to_linear(codes / 255.0)
    hi = lin > k
    out = lin.copy()
    out[hi] = k + (1.0 - k) * (1.0 - np.exp(-(lin[hi] - k) / (1.0 - k)))
    lut = np.clip(np.round(linear_to_srgb(out) * 255.0), 0, 255).astype(np.uint8)
    lut[~hi] = codes[~hi]          # exactly the identity below the knee, whatever the float round trip does
    return lut


def apply_knee(img, lut):
    """Same table on every channel of an 8-bit image (both eyes of a 2x1 frame alike)."""
    return img if lut is None else cv2.LUT(img, lut)


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
    ap.add_argument("--keyframes", action="store_true",
                    help="only H.264 I-frames are candidates: the first one at or after each parallax crossing that passes quality")
    ap.add_argument("--search-keyframes", type=int, default=2,
                    help="--keyframes: keyframes looked at per pick, from the first at or after the crossing")
    ap.add_argument("--min-sharp-rel", type=float, default=SOFT_REL,
                    help="--keyframes: focus must be at least this fraction of the running median of the picks")
    ap.add_argument("--highlight-knee", type=float, default=None, metavar="K",
                    help="soft knee at linear K (0 < K < 1) on every written frame; off by default")
    ap.add_argument("--ffprobe", default=os.environ.get("HS_FFPROBE", "ffprobe"),
                    help="ffprobe, for the keyframe indices")
    a = ap.parse_args()
    if a.highlight_knee is not None and not 0.0 < a.highlight_knee < 1.0:
        ap.error(f"--highlight-knee must be between 0 and 1 (exclusive), got {a.highlight_knee}")
    if a.search_keyframes < 1:
        ap.error("--search-keyframes must be at least 1")
    lut = knee_lut(a.highlight_knee) if a.highlight_knee is not None else None
    view = whole if a.mono else left_eye
    # Stereo picks keep the VID_NNN_FFFF_2x1 name hs select's measurements parse. A mono pick
    # becomes a *view name* in `hs ingest --frames`, which allows only letters, digits and '-'
    # (an underscore would collide with the _L/_R suffix rig.py strips), so: selNNN-FFFFF.
    def pick_name(k, fi):
        return f"sel{k:03d}-{fi:05d}.jpg" if a.mono else f"VID_{k:03d}_{fi:04d}_2x1.jpg"

    def write(k, fi, img):
        cv2.imwrite(os.path.join(a.out, pick_name(k, fi)), apply_knee(img, lut), [cv2.IMWRITE_JPEG_QUALITY, 97])

    # keyframe indices: before decoding in --keyframes mode (they decide the candidates); otherwise
    # alongside the decode, best effort, only to say which picks happened to be keyframes
    ffprobe = find_ffprobe(a.ffprobe)
    kf_info, kf_set, probe = None, None, None
    if a.keyframes:
        if ffprobe is None:
            sys.exit(f"--keyframes needs ffprobe to find the I-frames, and {a.ffprobe!r} is not there "
                     f"(install ffmpeg: brew install ffmpeg; or pass --ffprobe /path/to/ffprobe)")
        r = subprocess.run(ffprobe_argv(ffprobe, a.clip), capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"ffprobe failed on {a.clip}: {r.stderr.strip()[-300:]}")
        try:
            kf_info = keyframe_info(parse_pict_types(r.stdout))
        except ValueError as e:
            sys.exit(f"--keyframes: could not read ffprobe's frame types ({e})")
        if not kf_info["frames"]:
            sys.exit(f"--keyframes: ffprobe found no I-frames in {a.clip}")
        kf_set = set(kf_info["frames"])
        print(f"keyframes: {kf_info['total']} I-frames of {kf_info['probe_frames']}, "
              f"GOP {kf_info['gop_median']:g}" if kf_info["gop_median"] else
              f"keyframes: {kf_info['total']} I-frame(s) of {kf_info['probe_frames']}")
    elif ffprobe is not None:
        try:
            probe = subprocess.Popen(ffprobe_argv(ffprobe, a.clip), stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True)
        except OSError:
            probe = None

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

    # --keyframes: pending holds dicts (keyframes only), judged as they arrive
    trigger = None         # what made the current crossing: parallax | max-gap | untracked
    picked_focus = []      # focus of every pick so far: the running median --min-sharp-rel is against

    def judge(c):
        """Why keyframe candidate c fails quality ([] = it passes), and how badly: each failed test adds
        how far past its threshold the candidate sits (>= 1), so the least bad fails fewest, least."""
        med = float(np.median(picked_focus)) if picked_focus else None
        why, bad = [], 0.0
        if c["clip"] >= a.max_clip:
            why.append(f"clip {100 * c['clip']:.1f}% >= {100 * a.max_clip:g}%")
            bad += c["clip"] / a.max_clip
        if med:
            if c["focus"] is None:
                why.append("flat: no focus measure")
                bad += 1.0
            elif c["focus"] < a.min_sharp_rel * med:
                why.append(f"soft: focus {c['focus'] / med:.2f}x the picks' median (< {a.min_sharp_rel:g})")
                bad += a.min_sharp_rel * med / c["focus"]
        if c["dark"] > CRUSH:
            why.append(f"crushed {100 * c['dark']:.1f}% at <= 5 (> {100 * CRUSH:g}%)")
            bad += c["dark"] / CRUSH
        c["why"], c["bad"] = why, bad
        return c

    def take_keyframe(first):
        """Commit the pick for the current search: the first candidate that passed, else the least bad."""
        nonlocal ref, last_sel, pending, crossing_at, trigger
        passed = [c for c in pending if not c["why"]]
        pick = passed[0] if passed else min(pending, key=lambda c: c["bad"])
        k = len(selected)
        rec = {"sel": k, "frame": pick["frame"], "residual": pick["residual"], "sharpness": pick["sharp"],
               "clip": pick["clip"]}
        if not first:
            rec.update({"tracked": pick["tracked"], "gap": pick["frame"] - last_sel})
        rec["keyframe"] = pick["frame"] in kf_set
        if not first:
            rec["crossing"] = crossing_at
            rec["trigger"] = trigger
        if not first or len(pending) > 1:
            rec["candidates"] = [[c["frame"], round(c["sharp"], 2), round(c["clip"], 5)] for c in pending]
        if not passed:
            rec["quality_fallback"] = (f"none of {len(pending)} keyframe{'s' if len(pending) != 1 else ''} passed ("
                                       + "; ".join(f"{c['frame']}: {', '.join(c['why'])}" for c in pending)
                                       + "); took the least bad")
        selected.append(rec)
        picked_focus.extend([pick["focus"]] if pick["focus"] is not None else [])
        if not a.dry_run:
            write(k, pick["frame"], pick["img"])
        pg = cv2.cvtColor(view(pick["img"]), cv2.COLOR_BGR2GRAY)
        psmall = prep(pg, a.work_width)
        ref = (psmall, features(psmall)); last_sel = pick["frame"]
        pending = []; crossing_at = None
        if not first:
            print(f"  sel {k:3d}  frame {pick['frame']:5d}  gap {rec['gap']:3d}  residual {rec['residual']:5.2f} px  "
                  f"sharp {rec['sharpness']:7.1f}  clip {rec['clip']*100:4.1f}%  [{trigger}]  keyframe"
                  + ("" if passed else "  (quality fallback)"))
        trigger = None

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

        if a.keyframes:
            is_key = fi in kf_set

            def candidate(res, ntr):
                sh = sharpness(small)
                return judge({"frame": fi, "img": frame, "residual": res, "tracked": ntr, "sharp": sh,
                              "clip": clipping(gl), "dark": crushed(small), "focus": focus(sh, float(small.std()))})

            if ref is None:
                # the first pick: the first keyframe from --start on that passes (no parallax yet)
                if is_key:
                    pending.append(candidate(0.0, None))
                    if not pending[-1]["why"] or len(pending) >= a.search_keyframes:
                        take_keyframe(first=True)
                continue
            gap = fi - last_sel
            if gap < a.min_gap:
                continue
            res, ntr = parallax_residual(ref[0], ref[1], small)
            trace["residual"][-1] = None if res is None else round(res, 3)
            if crossing_at is None:
                if res is not None and res >= a.residual:
                    trigger = "parallax"
                elif gap >= a.max_gap:
                    trigger = "max-gap"
                elif res is None and gap >= a.min_gap * 2:
                    trigger = "untracked"
                if trigger is not None:
                    crossing_at = fi
            # the crossing says a frame is due; the pick waits for a keyframe at or after it
            if crossing_at is not None and is_key:
                pending.append(candidate(res if res is not None else -1.0, ntr))
                if not pending[-1]["why"] or len(pending) >= a.search_keyframes:
                    take_keyframe(first=False)
            continue

        if ref is None:
            # first frame is always selected
            selected.append({"sel": 0, "frame": fi, "residual": 0.0, "sharpness": sharpness(small), "clip": clipping(gl)})
            ref = (small, features(small)); last_sel = fi
            if not a.dry_run:
                write(0, fi, frame)
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
                    write(k, pfi, pframe)
                pg = cv2.cvtColor(view(pframe), cv2.COLOR_BGR2GRAY)
                psmall = prep(pg, a.work_width)
                ref = (psmall, features(psmall)); last_sel = pfi
                pending = []; crossing_at = None
                why = "max-gap" if pfi - (selected[-2]["frame"]) >= a.max_gap else "parallax"
                print(f"  sel {k:3d}  frame {pfi:5d}  gap {selected[-1]['gap']:3d}  residual {pres:5.2f} px  "
                      f"sharp {psharp:7.1f}  clip {pclip*100:4.1f}%  [{why}]")
    cap.release()

    if a.keyframes and pending:
        # the clip ended inside a search whose keyframes all failed: keep the least bad rather than
        # lose the last stretch of the orbit
        take_keyframe(first=ref is None)
        selected[-1]["quality_fallback"] = selected[-1].get("quality_fallback", "") + " (end of clip)"

    if probe is not None:
        out, err = probe.communicate()
        try:
            if probe.returncode != 0:
                raise ValueError(err.strip()[-200:] or f"exit {probe.returncode}")
            kf_info = keyframe_info(parse_pict_types(out))
            kf_set = set(kf_info["frames"])
        except ValueError as e:
            print(f"(ffprobe could not list the keyframes, so the picks are not marked: {e})")
    if kf_info is not None and kf_info["probe_frames"] != fi + 1:
        print(f"note: ffprobe listed {kf_info['probe_frames']} frames, the decoder read {fi + 1}")
    if not a.keyframes:
        for s in selected:
            s["keyframe"] = (s["frame"] in kf_set) if kf_set is not None else None

    gaps = [s["gap"] for s in selected if "gap" in s]
    print(f"\n{len(selected)} frames selected of {fi+1}; gaps {min(gaps) if gaps else 0}-{max(gaps) if gaps else 0} "
          f"(median {int(np.median(gaps)) if gaps else 0})")
    if kf_set is not None:
        n_key = sum(1 for s in selected if s.get("keyframe"))
        print(f"{n_key} of {len(selected)} picks are keyframes ({kf_info['total']} in the clip"
              + (f", GOP {kf_info['gop_median']:g})" if kf_info["gop_median"] else ")"))
    params = dict(vars(a), mode="keyframes" if a.keyframes else "parallax")
    json.dump({"clip": os.path.abspath(a.clip), "fps": fps, "frames_total": fi + 1, "mono": bool(a.mono),
               "params": params, "keyframes": kf_info, "selected": selected, "trace": trace},
              open(os.path.join(a.out, "selection.json"), "w"), indent=1)
    print(f"wrote {os.path.join(a.out, 'selection.json')}" + ("" if a.dry_run else f" and {len(selected)} frames")
          + (f" (highlight knee K={a.highlight_knee:g})" if lut is not None and not a.dry_run else ""))


if __name__ == "__main__":
    main()
