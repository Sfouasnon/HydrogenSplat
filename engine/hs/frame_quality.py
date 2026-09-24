"""Frame quality for hs select — what each pick looks like, and where a better frame sat.

select_frames.py chooses frames by parallax and, inside a four-frame search window, takes the
sharpest. That makes every pick locally sensible and says nothing about the set: whether a
pick is soft next to the rest of the clip, whether its exposure wanders off the others, or
whether a clearly sharper frame sat a few frames away, outside the window. This module turns
selection.json's per-frame trace (every frame read) plus a measurement of the written frames
(both eyes) into quality.json, which the app draws as the contact sheet. With --keyframes the
selector takes H.264 I-frames only; then only keyframes count as a "sharper frame nearby", and
the "keyframes" block says how many picks are I-frames and what the clip's GOP is.

Everything here is a *report*. The thresholds are starting guesses, not calibrated against a
solve — they flag frames for a human to look at; nothing is dropped automatically. Pure numpy
+ stdlib so it is testable without a clip.

Units
-----
sharp   Laplacian variance of the left eye at the selector's work width (480 px by default) —
        the same number the selector maximised. Only comparable within one clip, and it scales
        with contrast squared: a frame half a stop darker reads soft without being any blurrier.
focus   sharp / grey variance — the Laplacian with the contrast divided out, so exposure and
        scene brightness stop masquerading as blur. The "soft" and "sharper_nearby" flags use
        this, not raw sharp. On a synthetic clip a 0.55× exposure dip dropped raw sharp to 0.27×
        the median with no blur at all; focus held.
ev      log2(linear-light mean luma / the median over the picks): stops off the set's median.
        Linear light is a gamma-2.2 decode of 8-bit values, so this is approximate for Holocam's
        actual transfer curve, but monotonic and right in sign.
eye_ev  log2(right / left linear luma). A constant offset between eyes is normal (two sensors;
        hs exposure matches them); a frame is flagged when it departs from the clip's median
        offset.
"""
import math

import numpy as np

# --- flag thresholds (report-only; move them once a flagged frame has been shown to hurt) ---
SOFT_REL = 0.6          # focus below this fraction of the picks' median → "soft"
NEARBY_GAIN = 1.25      # a frame with this much more focus inside the pick's interval → "sharper_nearby"
EV_TOL = 0.33           # |ev| above a third of a stop → "exposure"
EYE_EV_TOL = 0.25       # eye offset this far from the clip's median offset → "eye_exposure"
EYE_SHARP_REL = 0.6     # right-eye focus below this fraction of the left's → "right_soft"
CRUSH = 0.10            # fraction of pixels <= 5 → "crushed"
NOISY_REL = 1.5         # noise above this multiple of the picks' median → "noisy"
CLIP_REL = 1.5          # clipped beyond --max-clip AND this multiple of the picks' median → "clipped"

FLAG_TEXT = {
    "soft": "soft — well below the other picks' sharpness",
    "sharper_nearby": "a sharper frame sits inside this pick's interval",
    "exposure": "exposure off the set's median",
    "eye_exposure": "left and right eyes differ in exposure more than usual",
    "right_soft": "right eye much softer than the left",
    "clipped": "more highlights clipped than --max-clip and than the other picks",
    "crushed": "shadows crushed",
    "noisy": "noisier than the other picks",
    "stood_still": "picked on max-gap — the camera barely translated",
    "untracked": "parallax could not be measured (tracking lost)",
}


# flags that disqualify a frame as the exposure reference ("sharper_nearby" and "stood_still"
# say nothing about how the frame is exposed)
REFERENCE_BLOCKERS = {"soft", "exposure", "eye_exposure", "right_soft", "clipped", "crushed", "noisy",
                      "untracked"}


def pick_reference(frames, ev_tol=EV_TOL):
    """The pick to match every view's exposure to, with the reason, or None.

    Among picks with none of REFERENCE_BLOCKERS and within ev_tol of the set's median exposure,
    the one with the least clipping (worse eye). Least clipping because a gain can only move
    what was recorded: matching to a hotter frame pushes every darker view's highlights into
    the clipping that frame has, while matching to the cleanest frame in the family mostly
    darkens, which loses nothing. The ev window keeps that from drifting to the darkest frame —
    the whole set moves by at most ev_tol. Ties go to the smaller |ev|, then the higher focus.
    Falls back to all picks with an exposure if nothing qualifies.
    """
    def worst_clip(f):
        return max(f.get("clip") or 0.0, f.get("clip_R") or 0.0)

    usable = [f for f in frames if f.get("ev") is not None]
    family = [f for f in usable if abs(f["ev"]) <= ev_tol and not (set(f.get("flags", [])) & REFERENCE_BLOCKERS)]
    pool, relaxed = (family, False) if family else (usable, True)
    if not pool:
        return None
    best = min(pool, key=lambda f: (round(worst_clip(f), 4), abs(f["ev"]), -(f.get("focus_rel") or 0.0)))
    clips = sorted(worst_clip(f) for f in usable)
    med = clips[len(clips) // 2] if clips else 0.0
    why = (f"least clipping of the {len(pool)} {'picks' if relaxed else 'clean picks within ' + format(ev_tol, '.2f') + ' EV of the median'}: "
           f"{100 * worst_clip(best):.1f}% at 250+ vs {100 * med:.1f}% median, {best['ev']:+.2f} EV"
           + (f", focus {best['focus_rel']:.2f}x" if best.get("focus_rel") is not None else ""))
    return {"sel": best["sel"], "frame": best["frame"], "cap": f"cap{best['sel']:03d}",
            "ev": best["ev"], "clip": round(worst_clip(best), 5), "relaxed": relaxed, "why": why,
            "pool": len(pool)}


def _median(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.median(xs)) if xs else None


def _log2_ratio(a, b):
    if a is None or b is None or a <= 0 or b <= 0:
        return None
    return math.log2(a / b)


def _r(x, n):
    return None if x is None else round(float(x), n)


def focus(sharp, std):
    """Contrast-normalised Laplacian: sharp / variance of the grey image. None if flat."""
    if sharp is None or std is None or std < 1.0:
        return None
    return sharp / (std * std)


def neighbourhood(picks, i, first, last):
    """Frame range [lo, hi] that belongs to pick i: halfway to the pick before and after.

    A frame inside it can replace the pick without changing the order of the picks, so the
    parallax spacing moves by at most half an interval either way.
    """
    f = picks[i]
    lo = first if i == 0 else (picks[i - 1] + f) // 2 + 1
    hi = last if i == len(picks) - 1 else (f + picks[i + 1]) // 2
    return lo, hi


def keyframe_summary(selection):
    """How many picks are H.264 I-frames, and the clip's keyframe count and GOP.

    mode: "keyframes" when only I-frames were candidates (select_frames.py --keyframes), else
    "parallax". picks_known is how many picks carry a keyframe mark at all (none when ffprobe
    could not list the frame types); keyframes_total / gop_median are None then too."""
    sel = selection.get("selected", [])
    kf = selection.get("keyframes") or {}
    known = [s for s in sel if s.get("keyframe") is not None]
    return {
        "mode": "keyframes" if (selection.get("params") or {}).get("keyframes") else "parallax",
        "picks": len(sel),
        "picks_known": len(known),
        "picks_keyframes": sum(1 for s in known if s["keyframe"]),
        "keyframes_total": kf.get("total"),
        "gop_median": kf.get("gop_median"),
        "quality_fallbacks": sum(1 for s in sel if s.get("quality_fallback")),
    }


def analyse(selection, measured=None):
    """quality.json content from selection.json (with its trace) and optional per-frame
    measurements of the written frames.

    measured: {source frame index: {"file", "thumb", "sharp_R", "std_R", "luma_R", "clip_R", "noise"}} — absent on
    a dry run, in which case the eye and noise columns are None and never flag.
    """
    measured = measured or {}
    params = selection.get("params", {})
    max_clip = float(params.get("max_clip", 0.02))
    max_gap = int(params.get("max_gap", 90))
    tr = selection.get("trace") or {}
    t_frames = tr.get("frame", [])
    idx = {f: k for k, f in enumerate(t_frames)}
    sel = selection["selected"]
    picks = [s["frame"] for s in sel]

    def tv(key, f):
        k = idx.get(f)
        if k is None or key not in tr:
            return None
        return tr[key][k]

    luma_L = {s["frame"]: tv("luma", s["frame"]) for s in sel}
    focus_L = {s["frame"]: focus(s["sharpness"], tv("std", s["frame"])) for s in sel}
    med_sharp = _median([s["sharpness"] for s in sel])
    med_focus = _median(list(focus_L.values()))
    t_focus = [focus(a, b) for a, b in zip(tr.get("sharp", []), tr.get("std", []))]
    med_luma = _median(list(luma_L.values()))
    eye_evs = {f: _log2_ratio(m.get("luma_R"), luma_L.get(f)) for f, m in measured.items()}
    med_eye = _median(list(eye_evs.values()))
    med_noise = _median([m.get("noise") for m in measured.values()])
    med_clip = _median([s.get("clip") for s in sel]) or 0.0
    clip_ok = max(max_clip, CLIP_REL * med_clip)   # a usable alternative clips no worse than this
    fps = float(selection.get("fps") or 0) or None
    # --keyframes: only an I-frame could have replaced a pick, so only keyframes are "nearby"
    # (a P-frame's higher Laplacian is its compression artefacts — select_frames.py)
    kf = selection.get("keyframes") or {}
    kf_only = set(kf.get("frames") or []) if params.get("keyframes") else None
    first = t_frames[0] if t_frames else (picks[0] if picks else 0)
    last = t_frames[-1] if t_frames else (picks[-1] if picks else 0)

    frames = []
    for i, s in enumerate(sel):
        f = s["frame"]
        m = measured.get(f, {})
        sharp = float(s["sharpness"])
        ev = _log2_ratio(luma_L[f], med_luma)
        eye = eye_evs.get(f)
        foc = focus_L[f]
        flags = []
        if foc is not None and med_focus and foc < SOFT_REL * med_focus:
            flags.append("soft")

        # the sharpest usable frame in this pick's own interval
        best = None
        if t_frames:
            lo, hi = neighbourhood(picks, i, first, last)
            for k in range(idx.get(lo, 0), len(t_frames)):
                g = t_frames[k]
                if g < lo:
                    continue
                if g > hi:
                    break
                if g == f or tr["clip"][k] >= clip_ok or t_focus[k] is None:
                    continue
                if kf_only is not None and g not in kf_only:
                    continue
                if best is None or t_focus[k] > best[1]:
                    best = (g, t_focus[k], tr["sharp"][k])
        nearby = None
        if best and foc and best[1] >= NEARBY_GAIN * foc:
            nearby = {"frame": best[0], "sharp": _r(best[2], 1), "gain": _r(best[1] / foc, 2),
                      "offset": best[0] - f}
            flags.append("sharper_nearby")

        if ev is not None and abs(ev) > EV_TOL:
            flags.append("exposure")
        if eye is not None and med_eye is not None and abs(eye - med_eye) > EYE_EV_TOL:
            flags.append("eye_exposure")
        sharp_R = m.get("sharp_R")
        foc_R = focus(sharp_R, m.get("std_R"))
        if foc_R is not None and foc and foc_R < EYE_SHARP_REL * foc:
            flags.append("right_soft")
        # relative as well as absolute: on a white subject every frame clips, and a flag on every
        # frame points at nothing (2026-09-16 Stormtrooper: median 7.7% at 250+, 26 of 29 flagged)
        worst_clip = max(s.get("clip", 0) or 0, m.get("clip_R") or 0)
        if worst_clip >= max_clip and worst_clip >= CLIP_REL * med_clip:
            flags.append("clipped")
        dark = tv("dark", f)
        if dark is not None and dark > CRUSH:
            flags.append("crushed")
        noise = m.get("noise")
        if noise is not None and med_noise and noise > NOISY_REL * med_noise:
            flags.append("noisy")
        # a keyframe pick says what made the frame due; its gap also includes the wait for the
        # next I-frame, so a parallax pick can overshoot max-gap without the camera standing still
        on_max_gap = s["trigger"] == "max-gap" if "trigger" in s else s.get("gap", 0) >= max_gap
        if on_max_gap:
            flags.append("stood_still")
        if s.get("residual", 0) is not None and s.get("residual", 0) < 0:
            flags.append("untracked")

        frames.append({
            "sel": s["sel"], "frame": f, "file": m.get("file"), "thumb": m.get("thumb"),
            "t_s": _r(f / fps, 3) if fps else None,
            "gap": s.get("gap"), "residual": _r(s.get("residual"), 3), "tracked": s.get("tracked"),
            "sharp": _r(sharp, 1), "sharp_rel": _r(sharp / med_sharp, 3) if med_sharp else None,
            "focus": _r(foc, 4), "focus_rel": _r(foc / med_focus, 3) if foc and med_focus else None,
            "sharp_R": _r(sharp_R, 1), "focus_R_rel": _r(foc_R / foc, 3) if foc_R and foc else None,
            "ev": _r(ev, 3), "eye_ev": _r(eye, 3),
            "clip": _r(s.get("clip"), 5), "clip_R": _r(m.get("clip_R"), 5), "dark": _r(dark, 5),
            "noise": _r(noise, 3), "noise_rel": _r(noise / med_noise, 3) if noise is not None and med_noise else None,
            "candidates": s.get("candidates", []),
            "keyframe": s.get("keyframe"), "quality_fallback": s.get("quality_fallback"),
            "sharper_nearby": nearby,
            "flags": flags,
        })

    counts = {}
    for fr in frames:
        for fl in fr["flags"]:
            counts[fl] = counts.get(fl, 0) + 1
    ev_trace = [_r(_log2_ratio(l, med_luma), 3) for l in tr.get("luma", [])]

    # what is wrong with the whole set rather than with a frame — a capture problem, not a pick
    warnings = []
    if med_clip >= max_clip:
        warnings.append(f"every pick clips: median {100 * med_clip:.1f}% of pixels at 250+ (limit "
                        f"{100 * max_clip:.0f}%). That is exposure or a specular subject, not frame "
                        f"choice — only frames clipping well beyond the rest are flagged.")
    stuck = [fr["frame"] for fr in frames if "soft" in fr["flags"] and not fr["sharper_nearby"]]
    if stuck:
        warnings.append(f"{len(stuck)} soft pick{'s' if len(stuck) != 1 else ''} with nothing sharper in "
                        f"reach (frames {', '.join(map(str, stuck[:8]))}{'…' if len(stuck) > 8 else ''}): "
                        f"the camera was moving too fast there for any frame to be sharp.")
    return {
        "version": 1,
        "clip": selection.get("clip"), "fps": selection.get("fps"),
        "frames_total": selection.get("frames_total"),
        "params": params,
        "warnings": warnings,
        "thresholds": {"soft_rel": SOFT_REL, "clip_rel": CLIP_REL, "nearby_gain": NEARBY_GAIN, "ev_tol": EV_TOL,
                       "eye_ev_tol": EYE_EV_TOL, "eye_sharp_rel": EYE_SHARP_REL, "crush": CRUSH,
                       "noisy_rel": NOISY_REL, "max_clip": max_clip, "max_gap": max_gap},
        "flag_text": FLAG_TEXT,
        "medians": {"sharp": _r(med_sharp, 1), "focus": _r(med_focus, 4), "clip": _r(med_clip, 5), "luma": _r(med_luma, 6), "eye_ev": _r(med_eye, 3),
                    "noise": _r(med_noise, 3)},
        "measured_eyes": bool(measured),
        "flag_counts": counts,
        "flagged": sum(1 for fr in frames if fr["flags"]),
        "exposure_reference": pick_reference(frames),
        "keyframes": keyframe_summary(selection),
        "frames": frames,
        "trace": {"frame": t_frames, "sharp": tr.get("sharp", []),
                  "focus_rel": [_r(x / med_focus, 3) if x and med_focus else None for x in t_focus],
                  "ev": ev_trace,
                  "clip": tr.get("clip", []), "residual": tr.get("residual", [])},
    }
