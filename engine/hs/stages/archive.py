"""hs archive — a model and the frame it lives in, kept together.

A .ply is meaningless without the cameras it was trained against. We learned that the
expensive way: a baseline sweep re-solved the project seven times, each solve rewriting
train/dataset, and the best model of the session survived only as splats. Its rig.npz was
gone, so it could no longer be rendered from a training pose, graded against a photograph,
or compared with anything — 57 minutes of training reduced to a point cloud nobody can
register.

This writes one self-contained folder: the splats, rig.npz, the COLMAP camera / image / rig /
frame records (text form, small), the exposure report, and a manifest carrying md5s for the
ply, the rig, every training image and every mask, plus the argv and metrics of the stages
that produced them. Points3D is deliberately excluded — it is large and reconstructible —
but its md5 is recorded so a later solve can be recognised as the same one or not.

    hs archive -p P --name masked-exposure [--ply PATH] [--link]
"""
import hashlib
import json
import os
import shutil
import sys

from .. import events
from ..project import md5_file, now_iso, tool_versions

STAGE = "archive"
COLMAP_RECORDS = ("cameras.txt", "images.txt", "rigs.txt", "frames.txt")


def add_parser(sub):
    p = sub.add_parser("archive", help="store a trained model together with the cameras it was trained against")
    p.add_argument("--name", default=None, help="folder name under archive/ (default: the ply's stem plus the date)")
    p.add_argument("--ply", default=None, help="default: the train stage's final export")
    p.add_argument("--link", action="store_true", help="hardlink the .ply instead of copying it")
    p.add_argument("--no-images", action="store_true", help="skip hashing the training images (faster, weaker provenance)")
    return p


def _dir_digest(root, exts):
    """md5 of (relative path, size, content md5) for every matching file, and the file count."""
    h, n = hashlib.md5(), 0
    for dirpath, _, files in os.walk(root):
        for f in sorted(files):
            if not f.lower().endswith(exts):
                continue
            p = os.path.join(dirpath, f)
            rel = os.path.relpath(p, root)
            h.update(rel.encode()); h.update(str(os.path.getsize(p)).encode())
            h.update(md5_file(p).encode())
            n += 1
    return h.hexdigest(), n


def run(a, pj):
    pj.require(STAGE)
    from .prune import final_export
    ply = os.path.abspath(a.ply) if a.ply else final_export(pj)
    if not ply or not os.path.exists(ply):
        raise events.StageError("no .ply to archive", hint="hs train first, or --ply PATH")
    rig = pj.rig_npz
    if not os.path.exists(rig):
        raise events.StageError("no train/dataset/rig.npz — the frame is exactly what this is for")

    name = a.name or f"{os.path.splitext(os.path.basename(ply))[0]}-{now_iso()[:10]}"
    out = pj.path("archive", name)
    os.makedirs(out, exist_ok=True)
    events.start(STAGE, "copy")

    dst_ply = os.path.join(out, os.path.basename(ply))
    if not os.path.exists(dst_ply):
        if a.link:
            try:
                os.link(ply, dst_ply)
            except OSError:
                shutil.copy2(ply, dst_ply)
        else:
            shutil.copy2(ply, dst_ply)
    shutil.copy2(rig, os.path.join(out, "rig.npz"))
    sparse = os.path.join(pj.dataset_dir, "sparse")
    recs = {}
    os.makedirs(os.path.join(out, "sparse"), exist_ok=True)
    for f in COLMAP_RECORDS:
        src = os.path.join(sparse, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out, "sparse", f))
            recs[f] = md5_file(src)
    for f in ("exposure.json",):
        src = os.path.join(pj.dataset_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out, f))

    events.start(STAGE, "hash")
    man = {
        "note": "a trained model and the coordinate frame it lives in; the ply alone cannot be registered",
        "archived": now_iso(), "project": pj.root, "name": name,
        "ply": {"source": pj.rel(ply) if ply.startswith(pj.root) else ply,
                "md5": md5_file(ply), "bytes": os.path.getsize(ply)},
        "rig_npz_md5": md5_file(rig),
        "colmap_records_md5": recs,
        "points3D_md5": md5_file(os.path.join(sparse, "points3D.bin"))
                        if os.path.exists(os.path.join(sparse, "points3D.bin")) else None,
        "tools": tool_versions(), "argv": list(sys.argv),
        "stages": {k: {"status": v.get("status"), "argv": v.get("argv"),
                       "metrics": v.get("metrics"), "checks": v.get("checks")}
                   for k, v in pj.m["stages"].items()},
        "source": pj.m.get("source", {}), "profile_id": pj.m.get("profile_id"),
        "exposure": pj.m.get("exposure"),
    }
    if not a.no_images:
        d, n = _dir_digest(os.path.join(pj.dataset_dir, "images"), (".jpg", ".jpeg", ".png"))
        man["images"] = {"digest": d, "count": n}
        pj.metric(STAGE, "images_hashed", n)
        mdir = os.path.join(pj.dataset_dir, "masks")
        if os.path.isdir(mdir):
            d, n = _dir_digest(mdir, (".png",))
            man["masks"] = {"digest": d, "count": n}
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)

    pj.metric(STAGE, "archive", pj.rel(out))
    pj.metric(STAGE, "ply_md5", man["ply"]["md5"])
    pj.metric(STAGE, "bytes", sum(os.path.getsize(os.path.join(dp, f))
                                  for dp, _, fs in os.walk(out) for f in fs))
    pj.artifact(STAGE, out, "archive")
    pj.check(STAGE, "frame_archived_with_model",
             all(f in recs for f in ("cameras.txt", "images.txt")) and os.path.exists(os.path.join(out, "rig.npz")),
             value=f"rig.npz + {len(recs)} COLMAP records beside {os.path.basename(ply)}")
    st = pj.stage(STAGE)
    st.update({"status": "done", "finished": now_iso()})
    pj.save()
