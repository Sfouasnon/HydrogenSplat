"""hs exposure — match every training view's exposure and white balance.

The Hydrogen One's auto-exposure walks during a capture. On the 2026-09-13 coin set the
70 left-eye views span 1.82x in mean luma (62.3 to 113.2), 2.74x in contrast and 8% in
R/B ratio, and the drift tracks camera position: the five brightest views are cap019-024,
the five darkest cap064-069. Brush has no per-image exposure or appearance parameter --
there is nothing for it in ProcessConfig, TrainConfig, ModelConfig or LoadDatasetConfig --
so a brightness that varies with camera position can only be explained as a property of
the object, and the spherical harmonics absorb it as shading.

This rewrites the undistorted training images so every view shares one exposure and one
white point. It runs after `hs solve` has exported the dataset and touches nothing the
solver produced: the sparse model, the poses and rig.npz are all computed from the
original frames and are left exactly as they were. Only pixels change, so a train before
and a train after differ in one variable.

Method: decode sRGB to linear light, take each channel's median over the image, and scale
it onto the dataset's own reference (the median of those medians, so the correction is
centred and no view is privileged). Medians, not means, so a blown window or a specular
highlight does not drag the gain. Clipped highlights are not recoverable -- a view that
was already clipping still clips, and the report says how much.

`--reference` changes what the views are matched TO. `median` (the default) is the centred
reference above. `auto` takes the pick hs select's quality report names (frame_quality.
pick_reference: a clean pick within a third of a stop of the median with the least clipping),
and `capNNN` / `NNN` names a capture. With a single-frame reference that frame's left eye keeps
its exposure and white point exactly (gain 1.0), every other view — both eyes — is scaled onto
it, so the set looks like that photograph and the left/right sensor offset goes too.

`board` and `checker` match views on a shared physical target instead of on their content, which
is what a camera array needs (its views frame different things, so their medians differ for
reasons that are not exposure). `board`: the ChArUco board of `hs scale` (--board, default the
one `hs scale` recorded) is detected in every view and the mean linear BGR of its white paper
sampled — in a ChArUco board the white squares carry the markers, so the sample is the margin
between each marker and its square's edge, inset 20 % from both (hs/board.py
white_sample_points). `checker`: cv2.mcc finds an X-Rite ColorChecker Classic and the interior
60 % of its white patch is sampled. Every view's per-channel gain then brings its white onto the
reference view's (--reference-view, default the view whose white is the median). A view where the
target is not found, or its white is clipped, keeps gain 1.0 and is named in a failing
`<target>_seen_in_every_view` check. The `exposure_target` metric keeps the per-view gains and the
white's RMS spread across views before and after. The array refusal below does not apply.

The originals are copied to solve/exposure_backup/ first, so `--restore` puts them back and
a second run always starts from the untouched export.

`--dry-run` measures the originals (the backup when one exists, else the current images) and
writes nothing: not a pixel, not the stage's status. It records what it measured under
stages.exposure.dry_run so the numbers are kept, but `exposure` stays whatever it was and
`train` is not marked stale, because nothing it trained on changed.
"""
import json
import os
import shutil
import sys

import numpy as np

from .. import events
from ..project import now_iso

STAGE = "exposure"
GAIN_MIN, GAIN_MAX = 0.35, 3.0
STRIDE = 3                      # subsample for the statistic; medians are stable under it


def add_parser(sub):
    p = sub.add_parser("exposure", help="match exposure and white balance across the training views")
    p.add_argument("--mode", choices=("rgb", "luma"), default="rgb",
                   help="rgb also neutralises white-balance drift (default); luma matches brightness only")
    p.add_argument("--reference", default="median",
                   help="match to: median (of all views, default) | auto (hs select's best-exposed clean pick) "
                        "| capNNN or NNN (that capture's left eye) | board (the white paper of the ChArUco "
                        "board in view: a shared target, so it works on an array) | checker (the white patch "
                        "of a Macbeth chart; needs cv2.mcc)")
    p.add_argument("--reference-view", default=None, metavar="VIEW",
                   help="with board/checker: match every view's target to this view's (GA, cap012, L/cap012); "
                        "default the view whose target is the median")
    p.add_argument("--board", default=None, metavar="SX,SY,SQUARE_MM,MARKER_MM[,DICT]",
                   help="with --reference board: the ChArUco board (default: the one `hs scale` recorded)")
    p.add_argument("--legacy-board", action="store_true", help="with --board: pre-4.6 OpenCV board layout")
    p.add_argument("--restore", action="store_true", help="put the original undistorted images back")
    p.add_argument("--dry-run", action="store_true", help="measure and report, write nothing")
    p.add_argument("--force", action="store_true",
                   help="run on an array project anyway (see the warning in run(); measure first with --dry-run)")
    return p


def _srgb_to_linear_lut():
    s = np.arange(256, dtype=np.float64) / 255.0
    return np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(lin):
    lin = np.clip(lin, 0.0, 1.0)
    s = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return np.clip(s * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _views(images_dir):
    out = []
    for eye in ("L", "R"):
        d = os.path.join(images_dir, eye)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                out.append((eye, f))
    return out


def measure(path, lut):
    import cv2
    bgr = cv2.imread(path)
    if bgr is None:
        raise events.StageError(f"cannot read {path}")
    lin = lut[bgr[::STRIDE, ::STRIDE, :]]
    med = np.median(lin.reshape(-1, 3), axis=0)                  # B, G, R in linear light
    clipped = float((bgr.max(axis=2) >= 250).mean())
    return med, clipped


def run(a, pj):
    import cv2
    if pj.status("solve") not in ("done", "stale"):
        raise events.StageError(f"stage 'solve' is {pj.status('solve')}; 'exposure' needs it done",
                                hint=f"run `hs solve --project {pj.root}` first")
    # The method matches each view's median onto the set's. That is only a correction when every
    # view frames the same thing, which is true of one camera orbiting a subject and false of an
    # array: on the 2026-04-03 take 067 the A-column cameras fill the frame with the lit cyc
    # (median linear luma 0.08-0.15) and the D column with black drape (0.008), a 19x spread that
    # is framing, not exposure. Matching it would darken the A column ~7x and brighten the D column
    # ~2x, giving one physical surface a different brightness in each camera -- the fault this
    # stage exists to remove. An array's exposure should be matched on a shared neutral target
    # (the gray sphere, gray card or Macbeth in the frame): --reference board / checker, which
    # compare the same physical white in every view and so are exactly what an array needs.
    # The refusal is about the number of cameras, so it is an array's only: a mono source is one
    # camera orbiting the subject, the case the median was built for (the Hydrogen's left eye).
    target = _target_mode(a)
    if pj.source_kind == "array" and not target and not (a.force or a.dry_run or a.restore):
        raise events.StageError(
            "refusing to match medians across an array: the views frame different content, so the "
            "gains would be framing, not exposure",
            hint="match a shared target instead: --reference board (a ChArUco board in view) or checker "
                 "(a Macbeth chart); measure it with --dry-run; if the cameras really do differ in exposure "
                 "and frame the same content, --force")
    if target == "checker" and not checker_available():
        raise events.StageError("--reference checker needs cv2.mcc (the Macbeth chart detector), which this "
                                "OpenCV build does not have",
                                hint="pip install opencv-contrib-python-headless, or use --reference board")
    images = os.path.join(pj.dataset_dir, "images")
    if not os.path.isdir(images):
        raise events.StageError("no train/dataset/images", hint="hs solve first")
    backup = pj.path("solve", "exposure_backup")
    report_path = os.path.join(pj.dataset_dir, "exposure.json")
    if a.dry_run and a.restore:
        raise events.StageError("--dry-run and --restore together: a dry run writes nothing, so there is nothing to restore")
    if a.dry_run:
        return _dry_run(a, pj, images, backup)
    pj.acquire(STAGE)
    st = pj.stage(STAGE)
    st.update({"status": "running", "started": now_iso(), "finished": None, "argv": list(sys.argv),
               "metrics": {}, "checks": [], "artifacts": []})
    pj.save()

    if a.restore:
        if not os.path.isdir(backup):
            raise events.StageError("no solve/exposure_backup to restore from")
        shutil.rmtree(images)
        shutil.copytree(backup, images)
        for f in (report_path,):
            if os.path.exists(f):
                os.remove(f)
        pj.m.pop("exposure", None)
        _mark_stale(pj)
        events.start(STAGE, "restore")
        pj.metric(STAGE, "restored", True)
        _done(pj)
        return

    views = _views(images)
    if not views:
        raise events.StageError("no images in train/dataset/images")
    # a second run must start from the untouched export, not from an already-corrected set
    if os.path.isdir(backup):
        shutil.rmtree(images)
        shutil.copytree(backup, images)
    else:
        shutil.copytree(images, backup)

    lut = _srgb_to_linear_lut()
    if target:
        return _run_target(a, pj, target, images, backup, report_path, views, lut)
    views, med, clip, M, ref, gains, luma, ref_label, ref_why = _measure_set(a, images, views, lut, pj)
    pj.metric(STAGE, "views", len(views))
    pj.metric(STAGE, "reference", ref_label)
    pj.metric(STAGE, "reference_why", ref_why)
    pj.metric(STAGE, "luma_spread_before", round(float(luma.max() / max(luma.min(), 1e-9)), 3))
    pj.metric(STAGE, "gain_min", round(float(gains.min()), 3))
    pj.metric(STAGE, "gain_max", round(float(gains.max()), 3))
    pj.metric(STAGE, "clipped_fraction_max", round(float(max(clip.values())), 4))
    events.metric(STAGE, "reference_linear_bgr", [round(float(x), 5) for x in ref])

    after = _apply(images, views, gains, lut)
    spread = float(after.max() / max(after.min(), 1e-9))
    pj.metric(STAGE, "luma_spread_after", round(spread, 3))
    pj.check(STAGE, "views_share_one_exposure", spread <= 1.10,
             value=f"{spread:.2f}x spread in linear luma across {len(views)} views "
                   f"(was {luma.max() / max(luma.min(), 1e-9):.2f}x; want <= 1.10)")
    pj.check(STAGE, "no_view_needed_an_extreme_gain",
             bool(gains.min() > GAIN_MIN and gains.max() < GAIN_MAX),
             value=f"gains {gains.min():.2f}-{gains.max():.2f}x")

    rows = [{"eye": e, "image": f, "gain_bgr": [round(float(x), 4) for x in g],
             "linear_median_bgr": [round(float(x), 5) for x in med[(e, f)]],
             "clipped_fraction": round(clip[(e, f)], 4)}
            for (e, f), g in zip(views, gains)]
    json.dump({"note": "hs exposure: per-view gains applied in linear light to the undistorted "
                       "training images; the solve is untouched",
               "mode": a.mode, "reference": ref_label, "reference_why": ref_why,
               "reference_linear_bgr": [float(x) for x in ref],
               "backup": pj.rel(backup), "views": rows},
              open(report_path, "w"), indent=1)
    pj.artifact(STAGE, report_path, "json")
    pj.m["exposure"] = {"mode": a.mode, "views": len(views), "reference": ref_label,
                        "luma_spread_before": round(float(luma.max() / max(luma.min(), 1e-9)), 3),
                        "luma_spread_after": round(spread, 3),
                        "argv": list(sys.argv)}
    _mark_stale(pj)
    _done(pj)


def _apply(images, views, gains, lut, after_fn=None):
    """Write every view with its per-channel linear gain. -> per-view median linear luma after
    (and, with after_fn(i, corrected_bgr), whatever that returns, collected in a second list)."""
    import cv2
    events.start(STAGE, "apply")
    after, extra = [], []
    for i, ((eye, f), g) in enumerate(zip(views, gains)):
        p = os.path.join(images, eye, f)
        bgr = cv2.imread(p)
        # the correction is a per-channel curve, so it collapses to one 256-entry table per
        # channel: decode sRGB, scale in linear light, re-encode. Same arithmetic as doing it
        # per pixel, ~100x faster.
        tbl = np.stack([_linear_to_srgb(lut * g[ch]) for ch in range(3)], axis=1)
        out = cv2.LUT(bgr, tbl.reshape(1, 256, 3))
        cv2.imwrite(p, out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if after_fn is not None:
            extra.append(after_fn(i, cv2.imread(p)))       # what was written, JPEG and all
        after.append(float(np.median(lut[out[::STRIDE, ::STRIDE, :]].reshape(-1, 3) @
                                     np.array([0.0722, 0.7152, 0.2126]))))
        events.progress(STAGE, i + 1, len(views), step="apply")
    after = np.array(after)
    return (after, extra) if after_fn is not None else after


# ------------------------------------------------------------------ shared targets

LUMA_BGR = np.array([0.0722, 0.7152, 0.2126])
TARGET_CLIP_MAX = 0.2           # a target whose white is clipped over this share is not a measurement


def _target_mode(a):
    r = str(getattr(a, "reference", None) or "median").strip().lower()
    return r if r in ("board", "checker") else None


def checker_available():
    try:
        import cv2
        return hasattr(cv2, "mcc") and hasattr(cv2.mcc, "CCheckerDetector_create")
    except ImportError:
        return False


class BoardTarget:
    """The white paper of a ChArUco board (hs/board.py): detected once per view on the
    originals; the same board-plane samples are re-read on the corrected image."""
    label = "board"

    def __init__(self, spec):
        from .. import board as boardlib
        self.b = boardlib
        self.spec = spec
        self.det = boardlib.Detector(spec)
        self.obj = boardlib.corner_points(spec)

    def locate(self, bgr):
        ids, xy = self.det.detect(bgr, min_corners=6)
        if not len(ids):
            return None
        H = self.b.board_homography(ids, xy, self.obj)
        return None if H is None else self.b.white_mask(self.spec, H, bgr.shape)

    def sample(self, bgr, where, lut):
        return self.b.sample_white(bgr, where, lut=lut)


class CheckerTarget:
    """The white patch (19 of 24) of an X-Rite ColorChecker Classic, found by cv2.mcc. The
    detector reports each patch's mean code value; decoded to linear per channel."""
    label = "checker"
    WHITE = 18

    def __init__(self):
        import cv2
        self.cv2 = cv2

    def _detect(self, bgr):
        mcc = self.cv2.mcc
        d = mcc.CCheckerDetector_create()
        if hasattr(d, "setColorChartType"):          # OpenCV 5: the chart type is a property
            d.setColorChartType(mcc.MCC24)
            found = d.process(bgr, 1)
        else:                                        # 4.x contrib: process(image, chartType, nc)
            found = d.process(bgr, mcc.MCC24, 1)
        if not found:
            return None
        c = d.getBestColorChecker()
        if c is None:
            return None
        rgb = np.asarray(c.getChartsRGB(), np.float64)
        # rows are (patch, channel) in R, G, B order; column 1 is the mean code value
        charts = rgb.reshape(-1, 3, rgb.shape[-1])[:, :, 1]
        if not self.plausible(charts):
            return None
        polys = np.asarray(c.getColorCharts(), np.float64).reshape(-1, 4, 2)
        return polys[self.WHITE] if len(polys) > self.WHITE else None

    @classmethod
    def plausible(cls, charts):
        """The detector also 'finds' charts in plain texture (on a ChArUco board it returned 24
        patches all of the background's grey). A real ColorChecker's bottom row runs white to
        black: brightest first, falling patch by patch, several stops of range."""
        if charts is None or len(charts) < 24:
            return False
        y = charts[cls.WHITE:cls.WHITE + 6].mean(axis=1)
        return bool(np.all(np.diff(y) < 0) and y[0] > 3.0 * max(y[-1], 1.0))

    def locate(self, bgr):
        return self._detect(bgr)

    def sample(self, bgr, where, lut):
        # measured where locate() found it; re-detected on the corrected image (where=None).
        # The patch polygon is mcc's own sampling quad; its interior 60 % is decoded to linear
        # light pixel by pixel (the detector's mean of code values is not a linear mean, and its
        # edge pixels carry the JPEG's chroma bleed from the black surround).
        quad = where if where is not None else self._detect(bgr)
        if quad is None:
            return None, 0, 0.0
        c = quad.mean(axis=0)
        inner = c + 0.6 * (quad - c)
        mask = np.zeros(bgr.shape[:2], np.uint8)
        self.cv2.fillConvexPoly(mask, np.round(inner).astype(np.int32), 1)
        raw = bgr[mask.astype(bool)].reshape(-1, 3)
        if not len(raw):
            return None, 0, 0.0
        clipped = raw.max(axis=1) >= 250
        frac = float(clipped.mean())
        raw = raw[~clipped]
        if not len(raw):
            return None, 0, frac
        return lut[raw].mean(axis=0), int(len(raw)), frac


def _target(a, pj, mode):
    if mode == "checker":
        return CheckerTarget()
    from .scale import spec_from_args
    return BoardTarget(spec_from_args(a, pj))


def _named_view(views, name):
    """--reference-view: GA, cap012, L/cap012, cap012_R -> index into views, or None."""
    n = str(name).strip()
    eye = None
    if "/" in n:
        eye, n = n.split("/", 1)
    elif n.endswith(("_L", "_R")):
        n, eye = n[:-2], n[-1]
    for i, (e, f) in enumerate(views):
        stem = os.path.splitext(f)[0]
        stem = stem[:-2] if stem.endswith(("_" + e)) else stem
        if stem == n and (eye or "L") == e:
            return i
    return None


def _measure_target(a, pj, src_dir, views, lut, tgt):
    """Per-view target white (linear BGR, or None) and the gains onto the reference view's."""
    import cv2
    events.start(STAGE, "measure")
    where, white, why_missing = [], [], {}
    for i, (eye, f) in enumerate(views):
        bgr = cv2.imread(os.path.join(src_dir, eye, f))
        if bgr is None:
            raise events.StageError(f"cannot read {eye}/{f}")
        loc = tgt.locate(bgr)
        w = None
        if loc is None:
            why_missing[(eye, f)] = f"no {tgt.label} found"
        else:
            w, n, clipped = tgt.sample(bgr, loc, lut)
            if w is None or n == 0:
                why_missing[(eye, f)] = f"{tgt.label} found but no usable white samples"
                w = None
            elif clipped > TARGET_CLIP_MAX:
                why_missing[(eye, f)] = f"{tgt.label} white clipped in {100 * clipped:.0f}% of samples"
                w = None
        where.append(loc)
        white.append(w)
        events.progress(STAGE, i + 1, len(views), step="measure")
    seen = [i for i, w in enumerate(white) if w is not None]
    if not seen:
        reasons = {}
        for why in why_missing.values():
            key = why.split(" in ")[0] if "clipped" in why else why
            reasons[key] = reasons.get(key, 0) + 1
        raise events.StageError(f"the {tgt.label} was not measured in any of {len(views)} views ("
                                + "; ".join(f"{k}: {n}" for k, n in reasons.items()) + ")",
                                hint="check --board against the print; the target must be in view, "
                                     "in focus and not clipped")
    Wl = {i: float(white[i] @ LUMA_BGR) for i in seen}
    ref_name = getattr(a, "reference_view", None)
    if ref_name:
        ri = _named_view(views, ref_name)
        if ri is None:
            raise events.StageError(f"--reference-view {ref_name}: no such view in the dataset")
        if white[ri] is None:
            raise events.StageError(f"--reference-view {ref_name}: {why_missing[views[ri]]}")
        why = "chosen by hand"
    else:
        order = sorted(seen, key=lambda i: Wl[i])
        ri = order[(len(order) - 1) // 2]
        why = f"the view whose {tgt.label} white is the median of {len(seen)}"
    ref = np.asarray(white[ri], np.float64)
    gains = np.ones((len(views), 3))
    for i in seen:
        if a.mode == "luma":
            gains[i] = float(ref @ LUMA_BGR) / Wl[i]
        else:
            gains[i] = ref / np.asarray(white[i])
    gains = np.clip(gains, GAIN_MIN, GAIN_MAX)
    e, f = views[ri]
    label = f"{tgt.label}:{e}/{os.path.splitext(f)[0]}"
    return {"where": where, "white": white, "seen": seen, "missing": why_missing, "ref_index": ri,
            "ref": ref, "gains": gains, "label": label, "why": why}


def _white_rms(whites):
    """Relative RMS of the per-view white luma about its mean: 0 = every view sees one white."""
    y = np.array([float(np.asarray(w) @ LUMA_BGR) for w in whites if w is not None])
    if len(y) < 2:
        return 0.0
    return float(np.sqrt(np.mean((y / y.mean() - 1.0) ** 2)))


def _target_metrics(views, T):
    per_view = {f"{e}/{f}": [round(float(x), 4) for x in T["gains"][i]] for i, (e, f) in enumerate(views)}
    return {"target": T["label"].split(":")[0], "reference": T["label"], "reference_why": T["why"],
            "views_with_target": len(T["seen"]), "views_without_target": len(views) - len(T["seen"]),
            "missing": {f"{e}/{f}": why for (e, f), why in T["missing"].items()},
            "white_rms_before": round(_white_rms(T["white"]), 5), "gains": per_view}


def _run_target(a, pj, mode, images, backup, report_path, views, lut):
    tgt = _target(a, pj, mode)
    T = _measure_target(a, pj, images, views, lut, tgt)
    gains = T["gains"]
    pj.metric(STAGE, "views", len(views))
    pj.metric(STAGE, "reference", T["label"])
    pj.metric(STAGE, "reference_why", T["why"])
    pj.metric(STAGE, "gain_min", round(float(gains.min()), 3))
    pj.metric(STAGE, "gain_max", round(float(gains.max()), 3))
    events.metric(STAGE, "reference_linear_bgr", [round(float(x), 5) for x in T["ref"]])

    def white_after(i, bgr):
        if T["white"][i] is None:
            return None
        return tgt.sample(bgr, T["where"][i] if mode == "board" else None, lut)[0]

    after, white2 = _apply(images, views, gains, lut, after_fn=white_after)
    tm = _target_metrics(views, T)
    tm["white_rms_after"] = round(_white_rms(white2), 5)
    pj.metric(STAGE, "exposure_target", tm)
    pj.metric(STAGE, "white_rms_before", tm["white_rms_before"])
    pj.metric(STAGE, "white_rms_after", tm["white_rms_after"])
    pj.metric(STAGE, "luma_spread_after", round(float(after.max() / max(after.min(), 1e-9)), 3))
    pj.check(STAGE, "views_share_one_white", tm["white_rms_after"] <= 0.01,
             value=f"{100 * tm['white_rms_after']:.2f}% RMS in the {mode} white across {len(T['seen'])} views "
                   f"(was {100 * tm['white_rms_before']:.2f}%; want <= 1%)")
    pj.check(STAGE, f"{mode}_seen_in_every_view", not T["missing"], needs_human=bool(T["missing"]),
             value=(f"all {len(views)} views" if not T["missing"] else
                    f"{len(T['missing'])} of {len(views)} views left uncorrected (gain 1.0): "
                    + ", ".join(f"{e}/{f}" for e, f in list(T["missing"])[:8])
                    + (" …" if len(T["missing"]) > 8 else "")))
    pj.check(STAGE, "no_view_needed_an_extreme_gain",
             bool(gains.min() > GAIN_MIN and gains.max() < GAIN_MAX),
             value=f"gains {gains.min():.2f}-{gains.max():.2f}x")
    rows = [{"eye": e, "image": f, "gain_bgr": [round(float(x), 4) for x in gains[i]],
             "target_linear_bgr": ([round(float(x), 5) for x in T["white"][i]] if T["white"][i] is not None else None),
             "target_after_linear_bgr": ([round(float(x), 5) for x in white2[i]] if white2[i] is not None else None),
             "missing": T["missing"].get((e, f))}
            for i, (e, f) in enumerate(views)]
    json.dump({"note": f"hs exposure --reference {mode}: per-view gains that bring every view's {mode} white "
                       "onto the reference view's, applied in linear light; the solve is untouched",
               "mode": a.mode, "reference": T["label"], "reference_why": T["why"],
               "reference_linear_bgr": [float(x) for x in T["ref"]],
               "white_rms_before": tm["white_rms_before"], "white_rms_after": tm["white_rms_after"],
               "backup": pj.rel(backup), "views": rows},
              open(report_path, "w"), indent=1)
    pj.artifact(STAGE, report_path, "json")
    pj.m["exposure"] = {"mode": a.mode, "views": len(views), "reference": T["label"], "target": mode,
                        "white_rms_before": tm["white_rms_before"], "white_rms_after": tm["white_rms_after"],
                        "argv": list(sys.argv)}
    _mark_stale(pj)
    _done(pj)


def _view_for(views, cap):
    """The left-eye view of capture ``cap`` ("cap042"): cap042.jpg, or cap042_L.jpg style names."""
    for eye, f in views:
        stem = os.path.splitext(f)[0]
        if eye == "L" and (stem == cap or stem.startswith(cap + "_")):
            return (eye, f)
    return None


def _reference(a, pj, views, med, M):
    """(linear BGR target, label, why) for --reference."""
    r = str(getattr(a, "reference", None) or "median").strip()
    if r.lower() == "median":
        return np.median(M, axis=0), "median", "the median of every view's channel medians"
    if r.lower() == "auto":
        from .. import frame_quality
        qp = pj.path("select", "quality.json")
        if not os.path.exists(qp):
            raise events.StageError("--reference auto needs select/quality.json",
                                    hint="re-run hs select (it writes the quality report), or name a capture")
        q = json.load(open(qp))
        n_caps = sum(1 for e, _ in views if e == "L")
        if n_caps != len(q.get("frames", [])):
            raise events.StageError(f"the quality report has {len(q.get('frames', []))} picks but the dataset "
                                    f"has {n_caps} captures, so pick N is not capture N",
                                    hint="re-run hs select and hs solve, or name a capture")
        pick = q.get("exposure_reference") or frame_quality.pick_reference(q["frames"])
        if not pick:
            raise events.StageError("no pick in the quality report has an exposure measurement")
        cap, why = pick["cap"], f"auto: {pick['why']} (source frame {pick['frame']})"
    else:
        digits = r.lower().removeprefix("cap")
        if not digits.isdigit():
            raise events.StageError(f"--reference {r!r}: use median, auto, capNNN, NNN, board or checker")
        cap, why = f"cap{int(digits):03d}", "chosen by hand"
    key = _view_for(views, cap)
    if key is None:
        raise events.StageError(f"--reference {cap}: no left-eye view {cap} in the dataset")
    return med[key], cap, why


def _measure_set(a, src_dir, views, lut, pj):
    """Per-view linear medians, clipped fraction, the reference, and the gains onto it."""
    med, clip = {}, {}
    events.start(STAGE, "measure")
    for i, (eye, f) in enumerate(views):
        med[(eye, f)], clip[(eye, f)] = measure(os.path.join(src_dir, eye, f), lut)
        events.progress(STAGE, i + 1, len(views), step="measure")
    M = np.array([med[k] for k in views])                        # (n, 3) B G R
    ref, ref_label, ref_why = _reference(a, pj, views, med, M)
    if a.mode == "luma":
        w = np.array([0.0722, 0.7152, 0.2126])                   # BGR weights
        y = M @ w
        gains = np.repeat((float(ref @ w) / y)[:, None], 3, axis=1)
    else:
        gains = ref[None, :] / M
    gains = np.clip(gains, GAIN_MIN, GAIN_MAX)
    luma = (M @ np.array([0.0722, 0.7152, 0.2126]))
    return views, med, clip, M, ref, gains, luma, ref_label, ref_why


def _dry_run(a, pj, images, backup):
    """Measure and report. Reads the originals (the backup if a run already happened) and
    writes only stages.exposure.dry_run in the manifest: no pixels, no status, no stale marks."""
    src_dir = backup if os.path.isdir(backup) else images
    views = _views(src_dir)
    if not views:
        raise events.StageError(f"no images in {pj.rel(src_dir)}")
    lut = _srgb_to_linear_lut()
    mode = _target_mode(a)
    if mode:
        T = _measure_target(a, pj, src_dir, views, lut, _target(a, pj, mode))
        tm = _target_metrics(views, T)
        g = T["gains"]
        metrics = {"views": len(views), "reference": T["label"], "reference_why": T["why"],
                   "gain_min": round(float(g.min()), 3), "gain_max": round(float(g.max()), 3),
                   "white_rms_before": tm["white_rms_before"], "exposure_target": tm,
                   "measured": pj.rel(src_dir)}
        for k, v in metrics.items():
            events.metric(STAGE, k, v)
        ok = bool(g.min() > GAIN_MIN and g.max() < GAIN_MAX)
        events.check(STAGE, "gains_within_range", ok, value=f"{g.min():.2f}-{g.max():.2f}x")
        pj.stage(STAGE)["dry_run"] = {"at": now_iso(), "mode": a.mode, "argv": list(sys.argv), "metrics": metrics,
                                      "checks": [{"name": "gains_within_range", "ok": ok}]}
        pj.save()
        return
    views, med, clip, M, ref, gains, luma, ref_label, ref_why = _measure_set(a, src_dir, views, lut, pj)
    metrics = {"views": len(views), "reference": ref_label, "reference_why": ref_why,
               "luma_spread_before": round(float(luma.max() / max(luma.min(), 1e-9)), 3),
               "gain_min": round(float(gains.min()), 3), "gain_max": round(float(gains.max()), 3),
               "clipped_fraction_max": round(float(max(clip.values())), 4),
               "measured": pj.rel(src_dir)}
    for k, v in metrics.items():
        events.metric(STAGE, k, v)
    events.metric(STAGE, "reference_linear_bgr", [round(float(x), 5) for x in ref])
    ok = bool(gains.min() > GAIN_MIN and gains.max() < GAIN_MAX)
    events.check(STAGE, "gains_within_range", ok,
                 value=f"{gains.min():.2f}-{gains.max():.2f}x (clamped outside {GAIN_MIN}-{GAIN_MAX})")
    pj.stage(STAGE)["dry_run"] = {"at": now_iso(), "mode": a.mode, "argv": list(sys.argv), "metrics": metrics,
                                  "checks": [{"name": "gains_within_range", "ok": ok}]}
    pj.save()


def _done(pj):
    st = pj.stage(STAGE)
    st["status"] = "done"
    st["finished"] = now_iso()
    pj.save()
    pj.release()


def _mark_stale(pj):
    for s in ("train", "prune", "render", "views"):
        if pj.status(s) in ("done", "failed", "running"):
            pj.m["stages"][s]["status"] = "stale"
    pj.save()
