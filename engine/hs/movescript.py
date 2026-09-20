"""The .hsmove script: a shot described in cues, compiled against what the capture can reach.

The presets build a path through real camera *indices*; this builds one in the capture's own
spherical coordinates — azimuth, elevation, and a dolly offset from the captured shell — which
is how a move is actually described on set ("boom down, decelerate, pull back, arc right").

    # hsmove 1
    start  az -15  el +16  dolly +0
    boom   to el +1        speed 4>1
    hold   0.4s
    dolly  back 20         speed 2
    arc    left 24         speed 2
    hold   0.5s
    arc    right 90        speed 3
    hull   50              # optional: the hull limit in mm for this move (default 25)

Speed is tenths: 10/10 is 30 deg/s of orbit and 150 mm/s of dolly, and `4>1` ramps across the
cue, which is how a deceleration into a landing is written. Every cue is walked one frame at a
time and each frame is checked against the hull: the virtual camera has to stay within 25 mm of
a real one or the render smears. A cue that runs out of capture is CLAMPED — it stops where the
data stops, keeps what it achieved, and the report says where and how much, because the useful
answer on set is "you got 46 of the 90 you asked for", not a failed build.

The capture is a shell of varying radius, not a sphere, so boom and arc ride it: a 90 deg arc on
the 2026-09-13 coin set also moves the camera from 315 mm to 220 mm whether or not a dolly was
asked for. `dolly` is an offset from that shell, bounded by how deep the hull runs at that
position (±25 mm where a real camera sits on the ray, and much less at the edges) — a short
dolly is reported but does not end the move.
"""
import json
import os
import re

import numpy as np

from . import rig

DEG_PER_10 = 30.0          # 10/10 orbit rate, deg/s
MM_PER_10 = 150.0          # 10/10 dolly rate, mm/s
HULL_MM = 25.0
R_LO, R_HI, R_STEP = 120.0, 600.0, 2.0


class ScriptError(Exception):
    pass


def parse(text):
    """-> (start, cues, opts). Raises ScriptError with the line number."""
    start, cues, opts = None, [], {}
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        w = line.replace(">", " > ").split()
        head = w[0].lower()
        try:
            if head == "start":
                d = _kv(w[1:])
                start = {"az": float(d["az"]), "el": float(d["el"]), "off": float(d.get("dolly", 0))}
            elif head == "aim":
                opts["aim"] = [float(x) for x in re.split(r"[,\s]+", " ".join(w[1:])) if x]
            elif head == "hull":
                opts["hull"] = float(w[1].lower().rstrip("m"))
                if not 5.0 <= opts["hull"] <= 200.0:
                    raise ScriptError("hull must be 5-200 mm")
            elif head == "fps":
                opts["fps"] = float(w[1])
            elif head == "hold":
                cues.append({"type": "hold", "secs": float(w[1].rstrip("s"))})
            elif head == "boom":
                sin_, sout = _speed(w)
                cues.append({"type": "boom", "el": float(_after(w, "el")), "sIn": sin_, "sOut": sout})
            elif head == "arc":
                sin_, sout = _speed(w)
                side = w[1].lower()
                if side not in ("left", "right"):
                    raise ScriptError("arc needs left or right")
                cues.append({"type": "arc", "dir": -1 if side == "left" else 1,
                             "amt": abs(float(w[2])), "sIn": sin_, "sOut": sout})
            elif head == "dolly":
                sin_, sout = _speed(w)
                side = w[1].lower()
                if side not in ("back", "in"):
                    raise ScriptError("dolly needs back or in")
                cues.append({"type": "dolly", "dir": 1 if side == "back" else -1,
                             "amt": abs(float(w[2])), "sIn": sin_, "sOut": sout})
            else:
                raise ScriptError(f"unknown cue '{w[0]}'")
        except ScriptError as e:
            raise ScriptError(f"line {n}: {e} — {raw.strip()}")
        except (IndexError, ValueError, KeyError) as e:
            raise ScriptError(f"line {n}: cannot read '{raw.strip()}' ({e})")
    if start is None:
        raise ScriptError("no `start az .. el .. dolly ..` line")
    if not cues:
        raise ScriptError("no cues")
    return start, cues, opts


def _kv(words):
    return {words[i].lower(): words[i + 1] for i in range(0, len(words) - 1, 2)}


def _after(words, key):
    low = [x.lower() for x in words]
    return words[low.index(key) + 1]


def _speed(words):
    low = [x.lower() for x in words]
    if "speed" not in low:
        return 3.0, 3.0
    i = low.index("speed")
    a = float(words[i + 1])
    if i + 3 < len(words) and words[i + 2] == ">":
        return a, float(words[i + 3])
    return a, a


def radius_grid(cr, hull=HULL_MM):
    """The radii a bearing's shell is searched over. It has to cover the captures' own distance
    plus the hull either side: a fixed 120-600 mm (the coins sat 220-315 mm out) made every
    start on a person captured from 0.6-1.7 m "out of reach", including a real camera's own
    position. The grid keeps its 2 mm steps on the old origin, so sets inside 120-600 mm
    compile exactly as before."""
    lo = min(R_LO, float(np.min(cr)) - 2 * hull)
    hi = max(R_HI, float(np.max(cr)) + 2 * hull)
    lo = R_LO - R_STEP * np.ceil(max(R_LO - max(lo, R_STEP), 0.0) / R_STEP)
    return np.arange(lo, hi + 1e-9, R_STEP)


class Reach:
    """Where a virtual camera may stand, measured against the real camera centres."""

    SIGMA_DEG = 8.0        # angular width of the shell smoother

    def __init__(self, CL, subject, up, ref, right, hull=HULL_MM):
        self.CL, self.subject, self.up, self.ref, self.right = CL, subject, up, ref, right
        self.hull = float(hull)
        self._cache = {}
        V = CL - subject
        self.cr = np.linalg.norm(V, axis=1)
        self.cel = np.degrees(np.arcsin(np.clip(V @ up / self.cr, -1, 1)))
        Hh = V - np.outer(V @ up, up)
        self.caz = np.degrees(np.arctan2(Hh @ right, Hh @ ref))
        self.radii = radius_grid(self.cr, self.hull)

    def smooth_radius(self, az, el):
        """The captured shell as a continuous function of bearing.

        Taking the radius that happens to be closest to a real camera makes the shell step
        every time the nearest camera changes — up to 30 mm between one frame and the next,
        which is a 900 mm/s lurch in the render. A Gaussian-weighted mean over the captures
        near this bearing gives the same surface without the cliffs."""
        w = np.exp(-(((self.caz - az) ** 2 + (self.cel - el) ** 2) / (2 * self.SIGMA_DEG ** 2)))
        tot = w.sum()
        return float((w @ self.cr) / tot) if tot > 1e-12 else float(np.median(self.cr))

    def direction(self, az, el):
        a, e = np.radians(az), np.radians(el)
        return np.cos(e) * (np.cos(a) * self.ref + np.sin(a) * self.right) + np.sin(e) * self.up

    def shell(self, az, el):
        """(rnom, rlo, rhi, dmin) at this bearing, or None if nothing is within the hull."""
        key = (round(az, 2), round(el, 2))
        if key in self._cache:
            return self._cache[key]
        d = self.direction(az, el)
        rs = self.radii
        P = self.subject[None, :] + rs[:, None] * d[None, :]
        near = np.sqrt(((P[:, None, :] - self.CL[None, :, :]) ** 2).sum(-1)).min(1)
        k = int(np.argmin(near))
        out = None
        if near[k] <= self.hull:
            # the run containing the closest radius, not the outer bounds: two cameras at
            # different depths on one bearing leave a gap between them that is NOT reachable
            ok = near <= self.hull
            lo_i = k
            while lo_i > 0 and ok[lo_i - 1]:
                lo_i -= 1
            hi_i = k
            while hi_i < len(rs) - 1 and ok[hi_i + 1]:
                hi_i += 1
            rsm = min(max(self.smooth_radius(az, el), rs[lo_i]), rs[hi_i])
            out = (rsm, float(rs[lo_i]), float(rs[hi_i]), float(near[k]))
        self._cache[key] = out
        return out

    def radius(self, sh, off):
        return min(max(sh[0] + off, sh[1]), sh[2])


def walk(start, cues, reach, fps=30.0):
    """One frame at a time, clamping at the edge of the capture. -> (samples, reports)."""
    p = dict(start)
    sh0 = reach.shell(p["az"], p["el"])
    samples, reports = [], []

    def emit(seg, rate):
        sh = reach.shell(p["az"], p["el"])
        samples.append({"az": p["az"], "el": p["el"], "r": reach.radius(sh, p["off"]) if sh else float("nan"),
                        "rate": rate, "seg": seg})

    if sh0 is None:
        return samples, [{"seg": -1, "error": "the start mark is out of reach"}]
    emit(-1, 0.0)
    for s, q in enumerate(cues):
        r0 = samples[-1]["r"]
        if q["type"] == "hold":
            n = max(1, int(round(q["secs"] * fps)))
            for _ in range(n):
                emit(s, 0.0)
            reports.append({"seg": s, "cue": "hold", "want": q["secs"], "got": n / fps,
                            "secs": n / fps, "clamped": False, "peak": 0.0, "unit": "s",
                            "r0": r0, "r1": r0})
            continue
        if q["type"] == "boom":
            axis, target, unit, per = "el", q["el"], "deg", DEG_PER_10
        elif q["type"] == "arc":
            axis, target, unit, per = "az", p["az"] + q["dir"] * q["amt"], "deg", DEG_PER_10
        else:
            axis, target, unit, per = "off", q["dir"] * q["amt"], "mm", MM_PER_10
        frm = p[axis]
        total = abs(target - frm)
        sign = np.sign(target - frm) or 1.0
        if total < 1e-6:
            reports.append({"seg": s, "cue": q["type"], "want": 0.0, "got": 0.0, "secs": 0.0,
                            "clamped": False, "peak": 0.0, "unit": unit, "r0": r0, "r1": r0})
            continue
        done, n, peak, clamped = 0.0, 0, 0.0, False
        while done < total - 1e-9 and n < 3600:
            u = done / total
            rate = ((q["sIn"] + (q["sOut"] - q["sIn"]) * u) / 10.0) * per
            step = min(rate / fps, total - done)
            nxt = dict(p)
            nxt[axis] = frm + sign * (done + step)
            sh = reach.shell(nxt["az"], nxt["el"])
            if sh is None:
                clamped = True
                break
            if q["type"] == "dolly" and abs(reach.radius(sh, nxt["off"]) - (sh[0] + nxt["off"])) > 0.5:
                clamped = True
                break
            p, done, n = nxt, done + step, n + 1
            peak = max(peak, rate)
            emit(s, rate)
        reports.append({"seg": s, "cue": q["type"], "want": total, "got": done, "secs": n / fps,
                        "clamped": clamped, "peak": peak, "unit": unit,
                        "at": {"az": p["az"], "el": p["el"]}, "r0": r0, "r1": samples[-1]["r"]})
        if clamped and q["type"] != "dolly":
            break                      # a short dolly does not end the move
    return samples, reports


def frame(rig_npz, aim_point=None):
    """The script's coordinate frame: subject, up, and the azimuth reference, from rig.npz.

    Azimuth 0 is the mean horizontal bearing of the captures seen from the median of the sparse
    points; `right` (positive azimuth) is the cameras' mean x axis. `ref` and `right` are always
    measured about the SfM median, even when an aim point moves the subject, so an `aim` line
    does not rotate the script's azimuths."""
    G = np.load(rig_npz, allow_pickle=True)
    names = [str(x) for x in G["names"]]
    L = rig.left_indices(G, names)
    C = G["C"].astype(np.float64)
    R = G["R"].astype(np.float64)
    pts = G["pts"].astype(np.float64)
    subject = np.array(aim_point, float) if aim_point is not None else np.median(pts, axis=0)

    down = np.mean([R[v].T @ np.array([0.0, 1.0, 0.0]) for v in L], axis=0)
    up = -down / np.linalg.norm(down)
    down = down / np.linalg.norm(down)
    xr = np.mean([R[v].T @ np.array([1.0, 0.0, 0.0]) for v in L], axis=0)
    CL = C[L]
    V = CL - subject
    H = V - np.outer(V @ up, up)
    ref = H.mean(axis=0)
    ref -= (ref @ up) * up
    ref /= np.linalg.norm(ref)
    right = xr - (xr @ up) * up
    # Gram-Schmidt against ref: azimuth is atan2(H.right, H.ref) one way and
    # cos(az) ref + sin(az) right the other, which are inverses only on an orthonormal basis.
    # The cameras' mean x axis is ~90 deg from ref on a narrow set (coins: 97 deg) and nothing
    # like it on a wide one (Stormtrooper: 69 deg), where a capture's own az/el pointed 260 mm
    # away from it.
    right -= (right @ ref) * ref
    right /= np.linalg.norm(right)
    return {"G": G, "names": names, "L": L, "subject": subject, "up": up, "down": down,
            "ref": ref, "right": right, "CL": CL}


def locate(rig_npz, p_mm, aim_point=None, hull=HULL_MM, _fr=None, _reach=None):
    """Where a camera centre (solve coordinates, mm) sits in script terms: az, el, and the dolly
    offset from the captured shell at that bearing. -> dict; `reachable` says whether a `start`
    line there would compile (within the hull, and inside the reachable band on its ray)."""
    fr = _fr or frame(rig_npz, aim_point)
    reach = _reach or Reach(fr["CL"], fr["subject"], fr["up"], fr["ref"], fr["right"], hull)
    V = np.asarray(p_mm, float) - fr["subject"]
    r = float(np.linalg.norm(V))
    if r < 1e-6:
        return {"reachable": False, "why": "that is the subject itself"}
    el = float(np.degrees(np.arcsin(np.clip(V @ fr["up"] / r, -1, 1))))
    Hh = V - (V @ fr["up"]) * fr["up"]
    az = float(np.degrees(np.arctan2(Hh @ fr["right"], Hh @ fr["ref"])))
    near = np.linalg.norm(fr["CL"] - np.asarray(p_mm, float), axis=1)
    k = int(np.argmin(near))
    out = {"az": az, "el": el, "r_mm": r, "nearest_capture": rig.capture_name(fr["names"][fr["L"][k]]),
           "nearest_mm": float(near[k])}
    sh = reach.shell(az, el)
    if sh is None:
        out.update(reachable=False, why=f"nothing was captured within {hull:.0f} mm of this bearing")
        return out
    rnom, rlo, rhi, _ = sh
    off = r - rnom
    out.update(shell_mm=rnom, band_mm=[rlo, rhi], dolly=off, reachable=bool(rlo - 1e-6 <= r <= rhi + 1e-6))
    if not out["reachable"]:
        out["why"] = (f"{r:.0f} mm from the subject; the capture reaches {rlo:.0f}-{rhi:.0f} mm on this bearing "
                      f"— a start here is pulled to {min(max(r, rlo), rhi):.0f} mm")
    return out


def build(rig_npz, text, out_path, fps=30.0, aim_point=None):
    """Compile a script into the move JSON the path renderer reads. -> report dict."""
    fr = frame(rig_npz, aim_point)
    G, names, L = fr["G"], fr["names"], fr["L"]
    R = G["R"].astype(np.float64)
    t = G["t"].astype(np.float64)
    K = G["K"].astype(np.float64)
    subject, up, down, ref, right, CL = fr["subject"], fr["up"], fr["down"], fr["ref"], fr["right"], fr["CL"]

    start, cues, opts = parse(text)
    fps = opts.get("fps", fps)
    if "aim" in opts and aim_point is None:
        subject = np.array(opts["aim"], float)
    hull = opts.get("hull", HULL_MM)
    reach = Reach(CL, subject, up, ref, right, hull)
    samples, reports = walk(start, cues, reach, fps)
    if reports and reports[0].get("error"):
        raise ScriptError(reports[0]["error"] + f" (az {start['az']:+.0f} el {start['el']:+.0f})")
    if len(samples) < 2:
        raise ScriptError("the move is empty — every cue clamped at the start mark")

    # The radius track wants to be smooth, but each frame has its own reachable band and the
    # band steps as the nearest real camera changes. Alternating projection: smooth a little,
    # pull back into the band, repeat. It converges on the smoothest track the capture allows,
    # and the final projection is what guarantees the hull — an unconstrained smoother lifts
    # the camera 6 mm outside it on this set.
    bands = []
    for s_ in samples:
        sh = reach.shell(s_["az"], s_["el"])
        bands.append((sh[1], sh[2]) if sh else (s_["r"], s_["r"]))
    lo = np.array([b[0] for b in bands]); hi = np.array([b[1] for b in bands])
    rtrack = np.clip(np.array([s["r"] for s in samples], float), lo, hi)
    if len(rtrack) >= 5:
        k = np.ones(5) / 5.0
        for _ in range(60):
            rtrack = np.convolve(np.pad(rtrack, 2, mode="edge"), k, mode="valid")
            rtrack = np.clip(rtrack, lo, hi)
    for i, s_ in enumerate(samples):
        s_["r"] = float(rtrack[i])
    pos = np.array([reach.subject + s["r"] * reach.direction(s["az"], s["el"]) for s in samples])
    near = np.array([np.linalg.norm(CL - q, axis=1).min() for q in pos])
    mid = int(np.argmin(np.linalg.norm(CL - pos[len(pos) // 2], axis=1)))
    vmid = int(L[mid])
    wh = G["wh"] if "wh" in G.files else None
    W0, H0 = (int(wh[vmid][0]), int(wh[vmid][1])) if wh is not None else (int(G["w"]), int(G["h"]))

    frames = []
    for q in pos:
        fwd = subject - q
        fwd /= np.linalg.norm(fwd)
        rt = np.cross(down, fwd)
        nr = np.linalg.norm(rt)
        if nr < 1e-6:
            raise ScriptError("degenerate up vector — the move looks straight down the world up axis")
        rt /= nr
        dn = np.cross(fwd, rt)
        c2w = np.eye(4)
        c2w[:3, 0] = rt
        c2w[:3, 1] = dn
        c2w[:3, 2] = fwd
        c2w[:3, 3] = q / 1000.0
        frames.append({"c2w": c2w.tolist()})

    json.dump({"note": "OpenCV camera convention (x right, y down, z forward), metres; compiled "
                       "from an .hsmove script in capture coordinates, clamped to the captured hull",
               "width": W0, "height": H0, "K": K[vmid].tolist(), "fps": fps, "frames": frames},
              open(out_path, "w"), indent=1)

    Xc = R[vmid] @ subject + t[vmid]
    aim = {"view": names[vmid], "depth_mm": float(Xc[2]), "w": W0, "h": H0}
    if Xc[2] > 0:
        uv = K[vmid] @ Xc
        aim["u"], aim["v"] = float(uv[0] / uv[2]), float(uv[1] / uv[2])
        aim["inside"] = bool(0 <= aim["u"] < W0 and 0 <= aim["v"] < H0)
    else:
        aim["behind"] = True
    d = np.linalg.norm(np.diff(pos, axis=0), axis=1) * fps
    dist = np.linalg.norm(pos - subject, axis=1)
    p90 = float(np.percentile(d, 90))
    spike = int(np.argmax(d))
    has_spike = bool(d[spike] > max(3.0 * p90, 150.0))
    return {
        "frames": len(frames), "fps": fps, "subject_mm": [float(x) for x in subject],
        "path_length_mm": float(d.sum() / fps), "peak_speed_mm_s": float(d.max()),
        "mean_speed_mm_s": float(d.mean()), "p90_speed_mm_s": p90,
        "spike": {"frame": spike, "t": spike / fps, "mm_s": float(d[spike])} if has_spike else None,
        "distance_mm": [float(dist.min()), float(dist.max())],
        "hull_mm": [float(near.min()), float(near.max()), float(np.median(near))],
        "aim": aim, "cues": reports, "start": start, "hull_limit_mm": hull,
        "clamped": [r for r in reports if r.get("clamped")],
        # for the app's move panel: the path and the captures in the script's own coordinates
        "track": [[round(s_["az"], 2), round(s_["el"], 2), round(s_["r"], 1), s_["seg"]] for s_ in samples],
        "captures": [[round(float(a_), 2), round(float(e_), 2), round(float(r_), 1)]
                     for a_, e_, r_ in zip(reach.caz, reach.cel, reach.cr)],
        "capture_names": [rig.capture_name(names[v]) for v in L],
        "view": names[vmid],
    }
