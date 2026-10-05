"""The mask review: what `hs masks` found wrong with the masks, and what was decided about it.

`maskcheck` does the measuring. This module is the project side of it: it runs the check on
train/dataset/masks, writes masks_review/ (review.json, a picture per flagged view, a repaired
mask where one can be made), applies decisions, and is what `hs train` asks before it starts.

The contract with the app is docs/mask-review.md: the app reads review.json, shows the
pictures, and runs `hs masks --decide`. Nothing else writes these files.

A decision belongs to a mask file, not to a view name: every flagged entry carries the md5 of
the mask it was made about. Rebuild the masks and a decision survives only if the new file is
byte for byte the old one.
"""
import json
import os
import shutil
import time

import numpy as np

from . import events, maskcheck
from .depthmaps import map_size, view_key
from .project import dir_digest, md5_file, now_iso

VERSION = 1
DIR = "masks_review"
REPORT = "review.json"
CHOICES = ("repair", "exclude", "keep", "undo")
DOWNSTREAM = ("train", "prune", "render", "views")
LABELS = {
    ("reselect", False): "Vision's own object, which the build had turned down",
    ("reselect", True): "Vision's own object, plus the missing piece with the hull's outline",
    ("fill", False): "the hole in the mask filled in",
    ("fill", True): "the missing piece filled in up to the frame edge",
    ("hull", True): "the missing piece added with the hull's outline",
    "whole": "the whole mask drawn from the hull (Vision found no object here)",
}
LISTS = ("@undecided", "@repairable", "@exact", "@decided")


def report_path(pj):
    return pj.path(DIR, REPORT)


def load(pj):
    p = report_path(pj)
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def retire(pj):
    """A build is about to overwrite the masks: their review goes with them. -> the review that
    was there (its exclude / keep decisions carry over to masks that come out byte for byte the
    same), or None."""
    old = load(pj)
    shutil.rmtree(pj.path(DIR), ignore_errors=True)
    return old


def masks_dir(pj):
    return os.path.join(pj.dataset_dir, "masks")


def digest(pj):
    return dir_digest(masks_dir(pj), (".png",))[0]


def summarise(review):
    fl = review.get("flagged", [])
    s = {"flagged": len(fl), "undecided": sum(1 for f in fl if not f.get("decision")),
         "repairable": sum(1 for f in fl if f.get("repair")),
         "exact": sum(1 for f in fl if f.get("repair") and not f["repair"].get("approximate", True))}
    for c in ("repair", "exclude", "keep"):
        s[c] = sum(1 for f in fl if f.get("decision") == c)
    review["summary"] = s
    return s


def _finite(o):
    """NaN and infinity are not JSON; the app's reader would turn the whole report down."""
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    return o


def save(pj, review):
    summarise(review)
    os.makedirs(pj.path(DIR), exist_ok=True)
    tmp = report_path(pj) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_finite(json.loads(json.dumps(review, default=events._default))), f, indent=1, allow_nan=False)
    os.replace(tmp, report_path(pj))


def record(pj, stage, review):
    """The review's state in the manifest: one metric, one check (replacing the last ones)."""
    s = review["summary"]
    m = dict(s)
    m["report"] = pj.rel(report_path(pj))
    pj.metric(stage, "mask_review", m)
    st = pj.stage(stage)
    st["checks"] = [c for c in st.get("checks", []) if c.get("name") != "masks_reviewed"]
    if review.get("note") and not s["flagged"]:
        value = review["note"]
    elif not s["flagged"]:
        value = f"all {review.get('views', 0)} masks agree with each other"
    else:
        value = (f"{s['flagged']} of {review.get('views', 0)} masks flagged; {s['undecided']} without a decision"
                 f" ({s['repair']} repaired, {s['exclude']} excluded, {s['keep']} kept)"
                 + ("" if not s["undecided"] else f" — decide them in the app's Masks panel or with "
                    f"`hs masks -p {pj.root} --decide VIEW=repair|exclude|keep`; hs train refuses until then"))
    pj.check(stage, "masks_reviewed", s["undecided"] == 0, value=value, needs_human=bool(s["undecided"]))


# --------------------------------------------------------------------------- the check
def _views(pj):
    """-> (G, names, [(rig index, key, mask path, image path)]) for every view with a mask file."""
    G = np.load(pj.rig_npz, allow_pickle=True)
    names = [str(x) for x in G["names"]]
    out = []
    for v, name in enumerate(names):
        key = view_key(name)
        mp = os.path.join(masks_dir(pj), key + ".png")
        if os.path.exists(mp):
            out.append((v, key, mp, os.path.join(pj.dataset_dir, "images", key + ".jpg")))
    return G, names, out


def _wh(G, v):
    if "wh" in G.files:
        return int(G["wh"][v][0]), int(G["wh"][v][1])
    return int(G["w"]), int(G["h"])


def _epoch(iso):
    """now_iso() back to epoch seconds, honouring its UTC offset (None when unreadable)."""
    import datetime
    try:
        return datetime.datetime.strptime(iso, "%Y-%m-%dT%H:%M:%S%z").timestamp()
    except (TypeError, ValueError):
        return None


def _vision_dirs(pj):
    """{view key: folder of Vision's instance masks} when they are the ones these masks were built
    from: masks_vision/images.txt lists the photographs in the order of the numbered folders."""
    root = pj.path("masks_vision")
    lst = os.path.join(root, "images.txt")
    st = pj.stage("masks")
    if not os.path.exists(lst) or st.get("metrics", {}).get("method") != "vision":
        return {}
    # written during the build that made the masks on disk, not by an older one
    t0, t1 = _epoch(st.get("started")), _epoch(st.get("finished"))
    mt = os.path.getmtime(lst)
    if (t0 and mt < t0 - 5) or (t1 and mt > t1 + 5):
        return {}
    out = {}
    for i, line in enumerate(open(lst).read().splitlines()):
        parts = line.strip().replace("\\", "/").split("/")
        if len(parts) >= 2 and os.path.isdir(os.path.join(root, str(i))):
            out[f"{parts[-2]}/{os.path.splitext(parts[-1])[0]}"] = os.path.join(root, str(i))
    return out


def _instances(folder):
    import cv2
    out = []
    if folder and os.path.isdir(folder):
        for f in sorted(os.listdir(folder), key=lambda x: (len(x), x)):
            if f.endswith(".png"):
                sm = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
                if sm is not None:
                    out.append(sm)
    return out


def known_fall_backs(pj, prev, md5s):
    """A check of masks built earlier is not told which views fell back to the rough region. What
    is on record: the last review's entries (while the mask file is still the one they were about),
    and the build's own metric, which names the first twelve."""
    out = {}
    for line in pj.stage("masks").get("metrics", {}).get("vision_fell_back_views") or []:
        name, _, why = str(line).partition(": ")
        out[view_key(name)] = why or "fell back to the rough region"
    for key, f in prev.items():
        if f.get("fell_back") and f.get("mask_md5") == md5s.get(key):
            out[key] = f["fell_back"]
    return out


def build_settings(pj, a=None):
    """How the build finished its masks, so a repaired one is finished the same way: from the
    build's own arguments, or (check-only on masks built earlier) from the recorded command."""
    if a is not None and getattr(a, "method", None):
        ns = a
    else:
        ns = None
        argv = pj.stage("masks").get("argv") or []
        if "masks" in argv:
            try:
                import argparse
                from .stages import masks as stage
                sub = argparse.ArgumentParser(add_help=False).add_subparsers()
                p = stage.add_parser(sub)
                p.add_argument("-p", "--project")
                rest = [x for x in argv[argv.index("masks") + 1:]]
                ns, _unknown = p.parse_known_args(rest)
            except (SystemExit, Exception):   # noqa: BLE001 - an old command line we cannot read: defaults
                ns = None
    method = getattr(ns, "method", None) or pj.stage("masks").get("metrics", {}).get("method") or "vision"
    return {"method": method,
            "feather_px": float(getattr(ns, "feather_px", 1.0) or 0.0) if method == "vision" else 0.0,
            "grow_px": int(getattr(ns, "grow_px", 0) or 0) if method == "vision" else 0,
            "snap_edge": bool(getattr(ns, "snap_edge", False)) if method == "vision" else False,
            "snap_fallback_px": int(getattr(ns, "snap_fallback_px", 3) or 0),
            "snap_max_px": int(getattr(ns, "snap_max_px", 8) or 0),
            "exclude_highlights": getattr(ns, "exclude_highlights", None),
            "highlight_grow_px": getattr(ns, "highlight_grow_px", None)}


def finish(hard, settings, image_path, grow=False):
    """A hard repaired mask finished the way `hs masks` finishes its own: grown by --grow-px and
    snapped to the photograph's edge by --snap-edge (both for an object fresh from Vision only:
    `grow`; a patched mask keeps its boundary, which the build already snapped), the feathered rim
    and, when the build cut the glints out, the same cut."""
    import cv2
    m = hard
    g = int(settings.get("grow_px") or 0)
    if grow and g > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1)))
    if grow and settings.get("snap_edge") and os.path.exists(image_path):
        from . import edgesnap
        photo = cv2.imread(image_path)
        r = edgesnap.measure(photo, m >= 128)
        px = (edgesnap.pixels(r["offset"], settings.get("snap_max_px", 8)) if r
              else int(settings.get("snap_fallback_px", 3)))
        m = edgesnap.snap(m >= 128, px)
    if settings.get("feather_px", 0) > 0:
        m = cv2.GaussianBlur(m, (0, 0), settings["feather_px"])
    code = settings.get("exclude_highlights")
    if code is not None and os.path.exists(image_path):
        photo = cv2.imread(image_path)
        hot = (photo.min(axis=2) >= code).astype(np.uint8)
        if hot.shape != m.shape:
            hot = cv2.resize(hot, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_NEAREST)
        grow = settings.get("highlight_grow_px")
        if grow is None:
            from .stages.masks import HIGHLIGHT_GROW_FRAC
            grow = int(round(HIGHLIGHT_GROW_FRAC * m.shape[1]))
        if grow > 0 and hot.any():
            hot = cv2.dilate(hot, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
        m = np.where(hot > 0, 0, m).astype(np.uint8)
    return m


def run_check(pj, stage, fell_back=None, settings=None, progress=None, previous=None):
    """Check the masks on disk, write masks_review/, return the review. `fell_back`:
    {view key: reason} from the build (None on a check of masks built earlier). `previous`: the
    review `retire` took away, on a build."""
    import cv2
    t0 = time.time()
    events.start(stage, "check")
    G, _names, items = _views(pj)
    if not items:
        raise events.StageError("no masks to check in train/dataset/masks", hint=f"hs masks -p {pj.root}")
    K, R, t = (G[k].astype(np.float64) for k in ("K", "R", "t"))
    pts = G["pts"].astype(np.float64) if "pts" in G.files else np.zeros((0, 3))
    settings = settings or build_settings(pj)
    previous = previous or load(pj) or {}
    prev = {f["view"]: f for f in previous.get("flagged", [])}

    views, small, md5s = [], [], {}
    for n, (v, key, mp, _ip) in enumerate(items):
        wh = _wh(G, v)
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise events.StageError(f"cannot read the mask {pj.rel(mp)}")
        W, H = map_size(wh, maskcheck.CHECK_EDGE)
        small.append(cv2.resize(m, (W, H), interpolation=cv2.INTER_AREA))
        views.append((K[v], R[v], t[v], wh))
        md5s[key] = md5_file(mp)
        if progress:
            progress(n + 1, 2 * len(items))
    keys = [it[1] for it in items]
    if fell_back is None:
        fell_back = known_fall_backs(pj, prev, md5s)
    # a repair already installed stays a decision as long as its file is the one on disk; such a
    # view keeps its entry whatever the new verdict is, and its mask is no longer the rough
    # region: it votes like any other
    kept = {k: f for k, f in prev.items() if f.get("decision") == "repair" and f.get("mask_md5") == md5s.get(k)}
    fb = {keys.index(k): why for k, why in (fell_back or {}).items() if k in keys and k not in kept}
    rep = maskcheck.check(views, small, pts, fell_back=fb,
                          progress=(lambda d, n: progress(len(items) + d, 2 * len(items))) if progress else None)
    vdirs = _vision_dirs(pj)
    out_root = pj.path(DIR)
    os.makedirs(out_root, exist_ok=True)

    flagged = []
    order = sorted((r for r in rep["views"] if r["reasons"]), key=lambda r: -r["score"])
    for r in order:
        i = r["index"]
        _v, key, mp, ip = items[i]
        if key in kept:
            continue
        entry = {"view": key, "reasons": r["reasons"], "score": r["score"],
                 "why": maskcheck.why(r), "agreement": r["agreement"], "piece_share": r["piece_share"],
                 "fell_back": fb.get(i), "preview": None, "repair": None, "mask_md5": md5s[key],
                 "decision": None, "decided": None}
        os.makedirs(os.path.join(out_root, os.path.dirname(key)), exist_ok=True)
        photo = cv2.imread(ip, cv2.IMREAD_REDUCED_COLOR_2) if os.path.exists(ip) else None
        if photo is None:
            photo = np.full((small[i].shape[0] * 2, small[i].shape[1] * 2, 3), 96, np.uint8)
        hard = small[i] >= 128
        S, core = rep["work"].get(i, (np.zeros_like(hard), np.zeros_like(hard)))
        pv = os.path.join(out_root, key + ".jpg")
        cv2.imwrite(pv, maskcheck.preview(photo, hard, S, core), [cv2.IMWRITE_JPEG_QUALITY, 85])
        entry["preview"] = pj.rel(pv)
        if i in rep["work"]:
            full = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            got = maskcheck.repair(full, S, instances=_instances(vdirs.get(key)), fell_back=i in fb,
                                   subject_uv=maskcheck.subject_uv(rep["subject"], rep["prepared"][i]))
            if got is not None:
                fixed, kind, approx, whole = got
                fixed = finish(fixed, settings, ip, grow=kind == "reselect")
                fp = os.path.join(out_root, key + "_repaired.png")
                cv2.imwrite(fp, fixed)
                rp = os.path.join(out_root, key + "_repair.jpg")
                cv2.imwrite(rp, maskcheck.preview(photo, hard, S, core, repaired=fixed >= 128),
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
                entry["repair"] = {"kind": kind, "approximate": approx,
                                   "label": LABELS["whole"] if whole else LABELS.get((kind, approx), kind),
                                   "preview": pj.rel(rp), "mask": pj.rel(fp)}
        # the same mask, flagged before and decided: the decision stands
        old = prev.get(key)
        if old and old.get("mask_md5") == md5s[key] and old.get("decision") in ("exclude", "keep"):
            entry["decision"], entry["decided"] = old["decision"], old.get("decided")
        flagged.append(entry)
    flagged = flagged + [kept[k] for k in keys if k in kept]   # decided long ago: last
    method = dict(rep["method"])
    method["seconds"] = round(time.time() - t0, 1)
    review = {"version": VERSION, "created": now_iso(), "views": len(items), "masks_digest": digest(pj),
              "method": method, "note": rep["note"], "flagged": flagged}
    save(pj, review)
    record(pj, stage, review)
    _sweep(pj, review)
    sheet = contact_sheet(pj, review)
    if sheet:
        pj.artifact(stage, sheet, "image")
    return review


def _sweep(pj, review):
    """Pictures and repaired masks of views that are no longer flagged, and originals of repairs
    that are no longer installed, are removed: the folder says what the report says."""
    root = pj.path(DIR)
    want = {os.path.normpath(report_path(pj))}
    for f in review["flagged"]:
        r = f.get("repair") or {}
        for p in (f.get("preview"), r.get("preview"), r.get("mask")):
            if p:
                want.add(os.path.normpath(pj.path(p)))
        if f.get("decision") == "repair":
            want.add(os.path.normpath(_original(pj, f["view"])))
    for d, _dirs, files in os.walk(root, topdown=False):
        for name in files:
            p = os.path.normpath(os.path.join(d, name))
            if p not in want:
                os.remove(p)
        if d != root and not os.listdir(d):
            os.rmdir(d)


def contact_sheet(pj, review, n=8):
    """The worst flagged views on one picture, for the log and the app's preview."""
    import cv2
    tiles = []
    for f in review["flagged"][:n]:
        p = pj.path(f["preview"]) if f.get("preview") else None
        im = cv2.imread(p) if p and os.path.exists(p) else None
        if im is None:
            continue
        s = 300.0 / im.shape[0]
        im = cv2.resize(im, (int(im.shape[1] * s), 300), interpolation=cv2.INTER_AREA)
        cv2.putText(im, f["view"], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        tiles.append(im)
    if not tiles:
        return None
    w = max(x.shape[1] for x in tiles)
    tiles = [cv2.copyMakeBorder(x, 0, 0, 0, w - x.shape[1], cv2.BORDER_CONSTANT) for x in tiles]
    if len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    out = pj.path(DIR, "flagged_sheet.jpg")
    cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out


# --------------------------------------------------------------------------- decisions
def parse_decisions(texts):
    """["L/a=repair,L/b=keep", "@undecided=exclude"] -> [("L/a", "repair"), ...] in order."""
    out = []
    for text in texts or []:
        for tok in str(text).split(","):
            tok = tok.strip()
            if not tok:
                continue
            if "=" not in tok:
                raise events.StageError(f"--decide: '{tok}' is not VIEW=CHOICE",
                                        hint="e.g. --decide L/cap012=repair,L/cap040=exclude   (or @undecided=exclude)")
            view, choice = (x.strip() for x in tok.rsplit("=", 1))
            if choice not in CHOICES:
                raise events.StageError(f"--decide: '{choice}' is not one of {', '.join(CHOICES)}")
            out.append((os.path.normpath(view) if not view.startswith("@") else view, choice))
    if not out:
        raise events.StageError("--decide needs at least one VIEW=CHOICE")
    return out


def _mask_file(pj, key):
    return os.path.join(masks_dir(pj), key + ".png")


def _original(pj, key):
    return pj.path(DIR, "original", key + ".png")


def _undo_repair(pj, entry):
    key = entry["view"]
    orig = _original(pj, key)
    if not os.path.exists(orig):
        raise events.StageError(f"the mask of {key} as it was built is no longer in {DIR}/original",
                                hint=f"rebuild the masks: hs masks -p {pj.root}")
    shutil.copyfile(orig, _mask_file(pj, key))
    os.remove(orig)
    for d in (os.path.dirname(orig), pj.path(DIR, "original")):      # leave no empty folders behind
        try:
            os.rmdir(d)
        except OSError:
            break
    entry["mask_md5"] = md5_file(_mask_file(pj, key))
    entry["decision"], entry["decided"] = None, None


def decide(pj, stage, texts):
    """Apply decisions. -> (review, number of mask files changed)."""
    review = load(pj)
    if review is None:
        raise events.StageError("there is no mask review to decide on", hint=f"hs masks -p {pj.root} --check-only")
    if int(review.get("version", 1)) > VERSION:
        raise events.StageError(f"{DIR}/{REPORT} was written by a newer hs (version {review.get('version')})")
    by = {f["view"]: f for f in review.get("flagged", [])}
    tokens = parse_decisions(texts)
    for view, choice in tokens:                  # what can be refused is refused before anything changes
        if view.startswith("@"):
            if view not in LISTS:
                raise events.StageError(f"--decide: unknown list '{view}'", hint="the lists are " + ", ".join(LISTS))
        elif view not in by:
            raise events.StageError(f"--decide: {view} is not a flagged view",
                                    hint=f"the flagged views are in {DIR}/{REPORT}")
        elif choice == "repair" and not by[view].get("repair"):
            raise events.StageError(f"--decide: there is no repair for {view}", hint="exclude it or keep it")
    count = {"changed": 0, "applied": 0}
    try:
        _apply(pj, by, tokens, count)
    finally:
        # also when a file turned out to be missing half-way: what was done is on record
        review["masks_digest"] = digest(pj)
        save(pj, review)
        if count["changed"]:
            for s in DOWNSTREAM:
                if pj.status(s) in ("done", "failed", "running"):
                    pj.m["stages"][s]["status"] = "stale"
        record(pj, stage, review)
    events.metric(stage, "mask_decisions", {"applied": count["applied"], "masks_changed": count["changed"]})
    return review, count["changed"]


def _apply(pj, by, tokens, count):
    # one token at a time, in the order given: "@exact=repair,@undecided=exclude" repairs what
    # needs no look and excludes what is then still undecided
    for view, choice in tokens:
        if view == "@undecided":
            keys = [k for k, f in by.items() if not f.get("decision")]
        elif view == "@repairable":
            keys = [k for k, f in by.items() if not f.get("decision") and f.get("repair")]
        elif view == "@exact":                   # the repairs that need no look (maskcheck.py)
            keys = [k for k, f in by.items() if not f.get("decision") and f.get("repair")
                    and not f["repair"].get("approximate", True)]
        elif view == "@decided":                 # for "=undo": start the review over
            keys = [k for k, f in by.items() if f.get("decision")]
        else:
            keys = [view]
        if choice == "repair":                   # a list: repair what can be, leave the rest as it is
            keys = [k for k in keys if by[k].get("repair")]
        for key in keys:
            e = by[key]
            if e.get("decision") == choice:
                continue
            count["applied"] += 1
            if e.get("decision") == "repair":    # whatever comes next starts from the mask as built
                _undo_repair(pj, e)
                count["changed"] += 1
            if choice == "undo":
                e["decision"], e["decided"] = None, None
                continue
            if choice == "repair":
                src = pj.path(e["repair"]["mask"])
                if not os.path.exists(src):
                    raise events.StageError(f"the repaired mask of {key} is missing ({e['repair']['mask']})",
                                            hint=f"hs masks -p {pj.root} --check-only writes it again")
                if md5_file(_mask_file(pj, key)) != e.get("mask_md5"):
                    raise events.StageError(f"the mask of {key} is not the one that was checked",
                                            hint=f"hs masks -p {pj.root} --check-only")
                orig = _original(pj, key)
                os.makedirs(os.path.dirname(orig), exist_ok=True)
                shutil.copyfile(_mask_file(pj, key), orig)
                shutil.copyfile(src, _mask_file(pj, key))
                e["mask_md5"] = md5_file(_mask_file(pj, key))
                count["changed"] += 1
            e["decision"], e["decided"] = choice, now_iso()


# --------------------------------------------------------------------------- train's question
def gate(pj, exclude, allow=False):
    """May a run that uses the masks start? -> (views to leave out, a record for the metrics).
    Raises StageError when the masks were never checked, changed since, or a flagged view is
    undecided — unless `allow`."""
    review = load(pj)
    info = {"reviewed": review is not None, "allowed_unreviewed": bool(allow)}
    hint_allow = "or pass --allow-unreviewed-masks to train on them as they are"
    if review is None:
        if allow:
            return set(), info
        raise events.StageError("these masks have not been checked against each other",
                                hint=f"hs masks -p {pj.root} --check-only   (about ten seconds; {hint_allow})")
    s = summarise(review)
    info.update({k: s[k] for k in ("flagged", "undecided", "repair", "exclude", "keep")})
    if review.get("masks_digest") != digest(pj):
        info["masks_changed"] = True
        if not allow:
            raise events.StageError("the masks on disk are not the ones that were reviewed",
                                    hint=f"hs masks -p {pj.root} --check-only   ({hint_allow})")
    open_ = sorted(f["view"] for f in review.get("flagged", []) if not f.get("decision") and f["view"] not in exclude)
    if open_ and not allow:
        show = ", ".join(open_[:8]) + (f" and {len(open_) - 8} more" if len(open_) > 8 else "")
        raise events.StageError(
            f"{len(open_)} flagged mask(s) have no decision: {show}",
            hint=(f"look at them in the app's Masks panel ({DIR}/), then e.g. "
                  f"hs masks -p {pj.root} --decide @exact=repair,@undecided=exclude   ({hint_allow})"))
    info["undecided_trained_on"] = len(open_)
    out = {f["view"] for f in review.get("flagged", []) if f.get("decision") == "exclude"}
    return out, info
