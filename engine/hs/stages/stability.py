"""hs stability — temporal stability of a rendered move: does the model pop?

    hs stability --frames render/boom/ | --video render/boom_1920.mp4
                 [--k 1,7] [--backend dis|raft] [--stride 1] [--name N] [--out DIR] [-p P]
    hs render -p P --move boom --stability          the same, in-process, on the frames before
                                                    they are deleted

A splat model can look right in every still and still flicker in motion: splats that pop in and
out as the camera crosses a sorting or culling boundary, view-dependent colour that swings. No
photograph exists for a virtual camera, so this is no-reference: each frame is compared with
its neighbour after the neighbour has been warped onto it by optical flow, which removes the
motion the camera is meant to produce and leaves what the model changed.

For every pair (t, t+k):
  flow          forward t -> t+k and backward t+k -> t (DIS, or RAFT with --backend raft)
  warp          frame t+k resampled onto frame t along the forward flow
  valid mask    forward-backward consistency: |fw(x) + bw(x + fw(x))| <= --fb-px (1 px), and the
                warped position inside the frame. Occluded and disoccluded pixels have no
                counterpart and are left out rather than counted as popping.
  warped MSE / PSNR  on the valid mask, colour in [0, 1] (PoppingDetection's view-consistency
                measure). At k = 1 the camera barely moves, so a low value is change the flow
                could not explain; k = 7 measures the same over a quarter second at 30 fps,
                where slow drift accumulates.
  flicker       mean |difference| after warping (the no-reference flicker proxy)
  popping       fraction of valid pixels whose largest channel error exceeds --pop-thresh (0.1)

Per k the summary reports the median warped PSNR and MSE, the 95th percentile popping fraction,
the worst pairs by warped MSE, and the worst FRAMES: each frame scored by the mean warped PSNR of
the pairs it takes part in. A frame that pops disagrees with the frame before it and the one
after, so both its pairs go bad; filing a pair under either of its frames blames a neighbour
half the time (a one-frame brightness flash came out as frame t+1). Within k of either end of the
sequence a frame has one pair, not two, so at k = 7 a pop at frame 10 of 20 also marks frames 3
and 17, whose only pair is with frame 10; the pop is the frame the others have in common.

Outputs (default <project>/stability/<name>/ with -p, else <source's folder>/<name>_stability/):
  <name>_pairs.csv        one row per pair
  <name>_stability.json   {"1": {warped_psnr_median, warped_mse_median, popping_pixel_fraction_p95,
                          flicker_median, worst_frames: [...]}, "7": {...}, "meta": {...}}
  <name>_stability.png    warped PSNR per frame, one panel per k, worst frames marked

Frames are downscaled to --max-width (960) before the flow: DIS on a 2400-px render costs about
half a second a flow, and four flows per frame is the difference between a minute and ten.
Numbers are only comparable between runs at the same width, k and backend.

Backends: ``dis`` is OpenCV's DISOpticalFlow (PRESET_MEDIUM) and needs nothing new. ``raft`` is
torchvision's raft_small (pip install -e 'engine[metrics]'); torch is imported only when it is
asked for, and `hs tools` says which backends this machine has. This stage takes no lock and
writes nothing to manifest.json; `hs render --stability` records the summary as render metrics.
"""
import csv
import importlib.util
import os
from collections import deque

import numpy as np

from .. import events

STAGE = "stability"
BACKENDS = ("dis", "raft")
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
WORST_N = 5


def add_parser(sub):
    p = sub.add_parser("stability", help="temporal stability of a rendered move (flow-warped PSNR, popping pixels)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--frames", default=None, help="folder of frames, read in name order")
    src.add_argument("--video", default=None, help="a video file (anything cv2 can decode)")
    p.add_argument("--k", default="1,7", help="frame gaps to compare, comma separated (default 1,7)")
    p.add_argument("--backend", choices=BACKENDS, default="dis",
                   help="dis: OpenCV DIS (default, no extra dependency); raft: torchvision raft_small (hs[metrics])")
    p.add_argument("--stride", type=int, default=1, help="start a pair every Nth frame (k stays in frames)")
    p.add_argument("--max-width", type=int, default=960, help="downscale frames to this width first (0 = full size)")
    p.add_argument("--fb-px", type=float, default=1.0, help="forward-backward inconsistency above this is occlusion")
    p.add_argument("--pop-thresh", type=float, default=0.1, help="per-pixel error (0..1) that counts as popping")
    p.add_argument("--name", default=None, help="output name (default: the folder or video name)")
    p.add_argument("--out", default=None, help="output folder")
    return p


# ------------------------------------------------------------------ backends

def available_backends():
    """{name: (ok, detail)} without importing torch: `hs tools` asks this on every Setup page load."""
    import cv2
    out = {"dis": (hasattr(cv2, "DISOpticalFlow_create"), f"cv2 {cv2.__version__} DISOpticalFlow")}
    have = [m for m in ("torch", "torchvision") if importlib.util.find_spec(m) is not None]
    if len(have) == 2:
        try:
            from importlib.metadata import version
            detail = f"torch {version('torch')}, torchvision {version('torchvision')} raft_small"
        except Exception:
            detail = "torch + torchvision raft_small"
        out["raft"] = (True, detail)
    else:
        out["raft"] = (False, "pip install -e 'engine[metrics]' (torch + torchvision)")
    return out


class _DIS:
    def __init__(self):
        import cv2
        self.cv2 = cv2
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    def __call__(self, a, b):
        """grey uint8 a, b -> flow (H, W, 2) float32 with a(x) ~ b(x + flow(x))"""
        return self.dis.calc(a, b, None)


class _RAFT:
    """torchvision raft_small. Imported here and nowhere else, so torch is loaded only on request."""

    def __init__(self):
        try:
            import torch
            from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
        except ImportError as e:
            raise events.StageError(f"--backend raft needs torch and torchvision ({e})",
                                    hint="pip install -e 'engine[metrics]', or use --backend dis")
        self.torch = torch
        dev = ("cuda" if torch.cuda.is_available() else
               "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        self.dev = torch.device(dev)
        self.model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).eval().to(self.dev)

    def _t(self, img):
        x = self.torch.from_numpy(np.ascontiguousarray(img)).float()
        x = x[None].repeat(3, 1, 1) if x.ndim == 2 else x.permute(2, 0, 1)
        return (x[None] / 127.5 - 1.0).to(self.dev)

    def __call__(self, a, b):
        h, w = a.shape[:2]
        ph, pw = (-h) % 8, (-w) % 8                      # RAFT wants sides divisible by 8
        pad = lambda im: np.pad(im, ((0, ph), (0, pw)) + ((0, 0),) * (im.ndim - 2), mode="edge")
        with self.torch.no_grad():
            flows = self.model(self._t(pad(a)), self._t(pad(b)))
        return flows[-1][0].permute(1, 2, 0).cpu().numpy()[:h, :w].astype(np.float32)


def flow_backend(name):
    if name == "dis":
        return _DIS()
    if name == "raft":
        return _RAFT()
    raise events.StageError(f"unknown flow backend {name!r}", hint=f"one of {', '.join(BACKENDS)}")


# ------------------------------------------------------------------ frames

def _prep(img, max_width):
    import cv2
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.dtype != np.uint8:
        img = (np.clip(img.astype(np.float64) / (65535.0 if img.dtype == np.uint16 else 1.0), 0, 1) * 255).astype(np.uint8)
    if max_width and img.shape[1] > max_width:
        h = int(round(img.shape[0] * max_width / img.shape[1]))
        img = cv2.resize(img, (max_width, h), interpolation=cv2.INTER_AREA)
    return img


def frame_source(frames=None, video=None):
    """-> (count or None, iterator of BGR uint8 frames)"""
    import cv2
    if frames:
        if not os.path.isdir(frames):
            raise events.StageError(f"--frames: no folder {frames}")
        files = sorted(f for f in os.listdir(frames) if f.lower().endswith(IMAGE_EXT))
        if len(files) < 2:
            raise events.StageError(f"{frames} holds {len(files)} frame(s); stability needs at least two")

        def it():
            for f in files:
                im = cv2.imread(os.path.join(frames, f), cv2.IMREAD_UNCHANGED)
                if im is None:
                    raise events.StageError(f"cannot read {f}")
                if im.ndim == 3 and im.shape[2] == 4:
                    im = im[:, :, :3]
                yield im
        return len(files), it()
    if not video or not os.path.exists(video):
        raise events.StageError(f"--video: no file {video}")
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise events.StageError(f"cv2 cannot open {video}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    def it():
        try:
            while True:
                ok, im = cap.read()
                if not ok:
                    return
                yield im
        finally:
            cap.release()
    return n, it()


# ------------------------------------------------------------------ the measure

def pair_metrics(a, b, flow, fb_px=1.0, pop_thresh=0.1, grey=None):
    """Frame a (time t) against frame b (t + k) warped onto it. a, b: BGR uint8, same size.
    -> dict of warped_mse, warped_psnr (None when nothing is valid), flicker_mad,
       popping_fraction, valid_fraction, flow_mean_px"""
    import cv2
    ga, gb = grey if grey is not None else (cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY))
    h, w = ga.shape
    fw, bw = flow(ga, gb), flow(gb, ga)
    gy, gx = np.mgrid[:h, :w].astype(np.float32)
    mx, my = gx + fw[..., 0], gy + fw[..., 1]
    bw_at = cv2.remap(bw, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    fb = np.hypot(fw[..., 0] + bw_at[..., 0], fw[..., 1] + bw_at[..., 1])
    valid = (fb <= fb_px) & (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
    warped = cv2.remap(b.astype(np.float32) / 255.0, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    d = np.abs(a.astype(np.float32) / 255.0 - warped)
    nv = int(valid.sum())
    out = {"valid_fraction": round(nv / valid.size, 4),
           "flow_mean_px": round(float(np.hypot(fw[..., 0], fw[..., 1]).mean()), 3)}
    if nv == 0:
        out.update(warped_mse=None, warped_psnr=None, flicker_mad=None, popping_fraction=None)
        return out
    dv = d[valid]
    mse = float((dv ** 2).mean())
    out.update(warped_mse=mse, warped_psnr=round(float(10 * np.log10(1.0 / max(mse, 1e-12))), 3),
               flicker_mad=round(float(dv.mean()), 6),
               popping_fraction=round(float((dv.max(axis=1) > pop_thresh).mean()), 6))
    return out


def parse_k(text):
    try:
        ks = sorted({int(x) for x in str(text).split(",") if x.strip()})
    except ValueError:
        raise events.StageError(f"--k: cannot read {text!r}", hint="frame gaps like 1,7")
    if not ks or ks[0] < 1:
        raise events.StageError(f"--k: gaps must be >= 1, got {text!r}")
    return ks


def summarise(rows, ks, worst_n=WORST_N):
    out = {}
    for k in ks:
        r = [x for x in rows if x["k"] == k and x["warped_mse"] is not None]
        if not r:
            out[str(k)] = {"pairs": 0}
            continue
        mse = np.array([x["warped_mse"] for x in r])
        worst = sorted(r, key=lambda x: -x["warped_mse"])[:worst_n]
        # A frame that pops disagrees with the frame before it AND the one after, so both pairs
        # it is in go bad; filing a pair under one of its frames blames the neighbour half the
        # time. Score each frame by the mean warped PSNR of every pair it takes part in (one
        # pair at the ends of the sequence, two inside it).
        by_frame = {}
        for x in r:
            for f in (x["t"], x["frame"]):
                by_frame.setdefault(f, []).append(x)
        frames = sorted(by_frame.items(), key=lambda kv: float(np.mean([x["warped_psnr"] for x in kv[1]])))
        out[str(k)] = {
            "pairs": len(r),
            "warped_psnr_median": round(float(np.median([x["warped_psnr"] for x in r])), 3),
            "warped_psnr_min": round(float(min(x["warped_psnr"] for x in r)), 3),
            "warped_mse_median": float(np.median(mse)),
            "flicker_median": round(float(np.median([x["flicker_mad"] for x in r])), 6),
            "popping_pixel_fraction_p95": round(float(np.percentile([x["popping_fraction"] for x in r], 95)), 6),
            "valid_fraction_median": round(float(np.median([x["valid_fraction"] for x in r])), 4),
            "worst_frames": [{"frame": int(f), "pairs": [[x["t"], x["frame"]] for x in ps],
                              "warped_psnr": round(float(np.mean([x["warped_psnr"] for x in ps])), 3),
                              "popping_fraction": max(x["popping_fraction"] for x in ps)}
                             for f, ps in frames[:worst_n]],
            "worst_pairs": [{"from": x["t"], "to": x["frame"], "warped_psnr": x["warped_psnr"],
                             "warped_mse": x["warped_mse"], "popping_fraction": x["popping_fraction"]} for x in worst],
        }
    return out


def plot(rows, ks, summary, path, title=""):
    """Warped PSNR per frame, one panel per k, worst frames in red. cv2 only."""
    import cv2
    W, PH, PAD_L, PAD_R, PAD_T, PAD_B = 1000, 200, 60, 16, 30, 26
    panels = []
    for k in ks:
        r = [x for x in rows if x["k"] == k and x["warped_psnr"] is not None]
        img = np.full((PH, W, 3), 22, np.uint8)
        s = summary.get(str(k), {})
        cv2.putText(img, f"{title}  k={k}: warped PSNR median {s.get('warped_psnr_median', 0):.1f} dB, "
                         f"popping p95 {100 * s.get('popping_pixel_fraction_p95', 0):.2f}%",
                    (PAD_L, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        if r:
            per = {}
            for x in r:                                   # the per-frame score summarise() ranks by
                for fr in (x["t"], x["frame"]):
                    per.setdefault(fr, []).append(x["warped_psnr"])
            f = np.array(sorted(per), float)
            p = np.array([np.mean(per[fr]) for fr in sorted(per)], float)
            f_lo, f_hi = f.min(), max(f.max(), f.min() + 1)
            p_lo, p_hi = np.floor(p.min() - 1), np.ceil(p.max() + 1)

            def xy(fr, ps):
                return (int(PAD_L + (fr - f_lo) / (f_hi - f_lo) * (W - PAD_L - PAD_R)),
                        int(PH - PAD_B - (ps - p_lo) / max(p_hi - p_lo, 1e-6) * (PH - PAD_T - PAD_B)))
            for ps in np.linspace(p_lo, p_hi, 4):
                _, y = xy(f_lo, ps)
                cv2.line(img, (PAD_L, y), (W - PAD_R, y), (50, 50, 50), 1)
                cv2.putText(img, f"{ps:.0f}", (8, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
            cv2.polylines(img, [np.array([xy(a, b) for a, b in zip(f, p)], np.int32)], False, (90, 200, 90), 1, cv2.LINE_AA)
            for w_ in s.get("worst_frames", []):
                pt = xy(w_["frame"], w_["warped_psnr"])
                cv2.circle(img, pt, 4, (60, 60, 240), -1, cv2.LINE_AA)
                cv2.putText(img, str(w_["frame"]), (pt[0] + 5, pt[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (120, 120, 255), 1, cv2.LINE_AA)
            cv2.putText(img, f"frame {int(f_lo)} .. {int(f.max())}   (dB)", (PAD_L, PH - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
        panels.append(img)
    cv2.imwrite(path, np.vstack(panels))
    return path


def measure(frames=None, video=None, ks=(1, 7), backend="dis", stride=1, max_width=960, fb_px=1.0,
            pop_thresh=0.1, out_dir=None, name=None, record=None, stage=STAGE):
    """Run the measure over a frames folder or a video. -> summary dict (also written to out_dir).

    ``record(name, value, **extra)`` receives every summary number (default: a metric event from
    ``stage``); `hs render --stability` passes one that also writes the manifest."""
    ks = sorted(set(int(k) for k in ks))
    stride = max(int(stride), 1)
    record = record or (lambda n, v, **x: events.metric(stage, n, v, **x))
    src = frames or video
    name = name or os.path.splitext(os.path.basename(os.path.normpath(src)))[0]
    out_dir = out_dir or os.path.join(os.path.dirname(os.path.abspath(os.path.normpath(src))), f"{name}_stability")
    os.makedirs(out_dir, exist_ok=True)
    n, it = frame_source(frames, video)
    flow = flow_backend(backend)
    import cv2

    kmax = max(ks)
    buf = deque(maxlen=kmax + 1)           # (index, bgr, grey)
    rows, size = [], None
    total = sum(len(range(0, n - k, stride)) for k in ks) if n else None
    for j, raw in enumerate(it):
        img = _prep(raw, max_width)
        if size is None:
            size = [img.shape[1], img.shape[0]]
        elif [img.shape[1], img.shape[0]] != size:
            raise events.StageError(f"frame {j} is {img.shape[1]}x{img.shape[0]}, the first was {size[0]}x{size[1]}")
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        buf.append((j, img, grey))
        for k in ks:
            t = j - k
            if t < 0 or t % stride:
                continue
            _, a, ga = buf[-1 - k]
            m = pair_metrics(a, img, flow, fb_px, pop_thresh, grey=(ga, grey))
            rows.append({"k": k, "t": t, "frame": j, **m})
            events.progress(stage, len(rows), total, step="stability")
    if not rows:
        raise events.StageError(f"{src}: no frame pairs at k={','.join(map(str, ks))}",
                                hint="fewer frames than the smallest gap")
    events.progress(stage, len(rows), total, step="stability", force=True)

    summary = summarise(rows, ks)
    summary["meta"] = {"source": os.path.abspath(src), "backend": backend, "frames": j + 1, "size": size,
                       "k": ks, "stride": stride, "max_width": max_width, "fb_px": fb_px, "pop_thresh": pop_thresh}
    csv_path = os.path.join(out_dir, f"{name}_pairs.csv")
    cols = ["k", "t", "frame", "warped_psnr", "warped_mse", "flicker_mad", "popping_fraction",
            "valid_fraction", "flow_mean_px"]
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    json_path = os.path.join(out_dir, f"{name}_stability.json")
    import json
    json.dump(summary, open(json_path, "w"), indent=1)
    png_path = plot(rows, ks, summary, os.path.join(out_dir, f"{name}_stability.png"), title=name)
    summary["meta"]["outputs"] = {"csv": csv_path, "json": json_path, "png": png_path}

    for k in ks:
        s = summary[str(k)]
        for key in ("warped_psnr_median", "warped_mse_median", "flicker_median", "popping_pixel_fraction_p95"):
            if key in s:
                record(f"{key}_k{k}", s[key], k=k)
        if s.get("worst_frames"):
            record(f"worst_frames_k{k}", [w["frame"] for w in s["worst_frames"]], k=k)
    return summary


def run(a):
    events.start(STAGE)
    out = a.out
    src = a.frames or a.video
    name = a.name or os.path.splitext(os.path.basename(os.path.normpath(src)))[0]
    root = getattr(a, "project", None)
    if not out and root:
        root = os.path.abspath(os.path.expanduser(root))
        if not os.path.exists(os.path.join(root, "manifest.json")):
            raise events.StageError(f"{root} is not a project (no manifest.json)")
        out = os.path.join(root, "stability", name)
    s = measure(frames=a.frames, video=a.video, ks=parse_k(a.k), backend=a.backend, stride=a.stride,
                max_width=a.max_width, fb_px=a.fb_px, pop_thresh=a.pop_thresh, out_dir=out, name=name)
    base = root if root and not a.out else None
    for kind, key in (("csv", "csv"), ("json", "json"), ("image", "png")):
        p = s["meta"]["outputs"][key]
        events.artifact(STAGE, os.path.relpath(p, base) if base else p, kind)
    return s
