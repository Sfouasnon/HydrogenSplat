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
    # (the gray sphere, gray card or Macbeth in the frame), which this does not do yet.
    if pj.m.get("source", {}).get("kind") == "array" and not (a.force or a.dry_run or a.restore):
        raise events.StageError(
            "refusing to match medians across an array: the views frame different content, so the "
            "gains would be framing, not exposure",
            hint="measure it with --dry-run; if the cameras really do differ in exposure and frame "
                 "the same content, --force")
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
    views, med, clip, M, ref, gains, luma = _measure_set(a, images, views, lut)
    pj.metric(STAGE, "views", len(views))
    pj.metric(STAGE, "luma_spread_before", round(float(luma.max() / max(luma.min(), 1e-9)), 3))
    pj.metric(STAGE, "gain_min", round(float(gains.min()), 3))
    pj.metric(STAGE, "gain_max", round(float(gains.max()), 3))
    pj.metric(STAGE, "clipped_fraction_max", round(float(max(clip.values())), 4))
    events.metric(STAGE, "reference_linear_bgr", [round(float(x), 5) for x in ref])

    events.start(STAGE, "apply")
    after = []
    for i, ((eye, f), g) in enumerate(zip(views, gains)):
        p = os.path.join(images, eye, f)
        bgr = cv2.imread(p)
        # the correction is a per-channel curve, so it collapses to one 256-entry table per
        # channel: decode sRGB, scale in linear light, re-encode. Same arithmetic as doing it
        # per pixel, ~100x faster.
        tbl = np.stack([_linear_to_srgb(lut * g[ch]) for ch in range(3)], axis=1)
        out = cv2.LUT(bgr, tbl.reshape(1, 256, 3))
        cv2.imwrite(p, out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        after.append(float(np.median(lut[out[::STRIDE, ::STRIDE, :]].reshape(-1, 3) @
                                     np.array([0.0722, 0.7152, 0.2126]))))
        events.progress(STAGE, i + 1, len(views), step="apply")
    after = np.array(after)
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
               "mode": a.mode, "reference_linear_bgr": [float(x) for x in ref],
               "backup": pj.rel(backup), "views": rows},
              open(report_path, "w"), indent=1)
    pj.artifact(STAGE, report_path, "json")
    pj.m["exposure"] = {"mode": a.mode, "views": len(views),
                        "luma_spread_before": round(float(luma.max() / max(luma.min(), 1e-9)), 3),
                        "luma_spread_after": round(spread, 3),
                        "argv": list(sys.argv)}
    _mark_stale(pj)
    _done(pj)


def _measure_set(a, src_dir, views, lut):
    """Per-view linear medians, clipped fraction and the gains that would centre them."""
    med, clip = {}, {}
    events.start(STAGE, "measure")
    for i, (eye, f) in enumerate(views):
        med[(eye, f)], clip[(eye, f)] = measure(os.path.join(src_dir, eye, f), lut)
        events.progress(STAGE, i + 1, len(views), step="measure")
    M = np.array([med[k] for k in views])                        # (n, 3) B G R
    ref = np.median(M, axis=0)
    if a.mode == "luma":
        y = M @ np.array([0.0722, 0.7152, 0.2126])               # BGR weights
        gains = np.repeat((np.median(y) / y)[:, None], 3, axis=1)
    else:
        gains = ref[None, :] / M
    gains = np.clip(gains, GAIN_MIN, GAIN_MAX)
    luma = (M @ np.array([0.0722, 0.7152, 0.2126]))
    return views, med, clip, M, ref, gains, luma


def _dry_run(a, pj, images, backup):
    """Measure and report. Reads the originals (the backup if a run already happened) and
    writes only stages.exposure.dry_run in the manifest: no pixels, no status, no stale marks."""
    src_dir = backup if os.path.isdir(backup) else images
    views = _views(src_dir)
    if not views:
        raise events.StageError(f"no images in {pj.rel(src_dir)}")
    lut = _srgb_to_linear_lut()
    views, med, clip, M, ref, gains, luma = _measure_set(a, src_dir, views, lut)
    metrics = {"views": len(views),
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
