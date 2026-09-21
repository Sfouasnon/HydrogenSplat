"""hs merge — one .ply from several, so a subject model and a background model render together.

Why a merge and not a compositor: both layers were trained against the same solve, so they
already live in the same world frame and at the same scale. Concatenating the two vertex lists
therefore needs no resampling and no per-pixel compositing — and because Brush sorts every splat
by depth at render time, sorting *across* the layers comes out right for free. The alternative
(render each layer to RGBA + depth and composite) needs a renderer that emits alpha, which
brush-path-render does not: it writes RGB with the background already flattened in.

    hs merge -p P --models subject-a,background-a --name subject-plus-background

The output is written as an ordinary archive — archive/<name>/ with the merged ply, the rig and a
manifest.json — so `hs cameras --archive`, the viewer's model list and `hs render --ply` all take
it with no special case. `hs render`'s lineage guard reads that manifest's rig_npz_md5, so a
merged model can no more be rendered in the wrong frame than any other archive.

Refused rather than guessed: models whose PLY property lists differ. In the 3DGS layout f_rest
is channel-major — degree 3 is f_rest_0..14 for R, 15..29 for G, 30..44 for B — so padding a
lower-degree model up to a higher one is an interleave, not an append, and getting it subtly
wrong shifts colour rather than failing. Train both layers with the same SH degree instead.
"""
import json
import os
import shutil
import sys

from .. import events
from ..project import md5_file, now_iso

STAGE = "merge"


def add_parser(sub):
    p = sub.add_parser("merge", help="combine several models (e.g. subject + background) into one archive")
    p.add_argument("--models", required=True,
                   help="comma separated: archive names, or paths to .ply files. Two or more")
    p.add_argument("--name", required=True, help="the archive to write: archive/<name>/")
    p.add_argument("--force", action="store_true",
                   help="merge even when the sources do not agree on the solve they came from")
    return p


def read_header(path):
    """-> (header_bytes, n_vertices, [property names], body_offset). Binary little-endian only."""
    with open(path, "rb") as f:
        raw = f.read(1 << 16)
    k = raw.find(b"end_header\n")
    if k < 0:
        raise events.StageError(f"no end_header in {os.path.basename(path)} — is it a PLY?")
    head = raw[:k].decode("ascii", "replace")
    if "binary_little_endian" not in head:
        raise events.StageError(f"{os.path.basename(path)} is not binary_little_endian",
                                hint="hs only writes and reads the binary Gaussian PLY")
    n, props, seen_vertex = None, [], False
    for line in head.split("\n"):
        w = line.split()
        if len(w) == 3 and w[0] == "element":
            if w[1] == "vertex":
                n, seen_vertex = int(w[2]), True
            elif seen_vertex:
                raise events.StageError(f"{os.path.basename(path)} has more than one element",
                                        hint="a Gaussian PLY carries only 'vertex'")
        elif len(w) == 3 and w[0] == "property" and seen_vertex:
            props.append((w[2], w[1]))
    if n is None:
        raise events.StageError(f"no 'element vertex' in {os.path.basename(path)}")
    return raw[:k + len(b"end_header\n")], n, props, k + len(b"end_header\n")


SIZES = {"float": 4, "float32": 4, "double": 8, "uchar": 1, "uint8": 1, "char": 1, "int8": 1,
         "short": 2, "ushort": 2, "int": 4, "uint": 4}


def row_bytes(props):
    try:
        return sum(SIZES[t] for _n, t in props)
    except KeyError as e:
        raise events.StageError(f"unknown PLY property type {e}")


def resolve(pj, spec):
    """An archive name or a path -> (ply_path, archive_dir | None, manifest | None)."""
    arch = pj.path("archive", spec)
    if os.path.isdir(arch):
        plys = sorted(f for f in os.listdir(arch) if f.endswith(".ply"))
        if not plys:
            raise events.StageError(f"archive/{spec} holds no .ply")
        # an archive holds one model; if a second was dropped in, take the largest, as the app does
        ply = max((os.path.join(arch, f) for f in plys), key=os.path.getsize)
        man_p = os.path.join(arch, "manifest.json")
        man = json.load(open(man_p)) if os.path.exists(man_p) else None
        return ply, arch, man
    ply = spec if os.path.isabs(spec) else pj.path(spec)
    if not os.path.exists(ply):
        raise events.StageError(f"no archive and no file called {spec}",
                                hint="hs merge --models takes archive names or paths to .ply files")
    man_p = os.path.join(os.path.dirname(ply), "manifest.json")
    man = json.load(open(man_p)) if os.path.exists(man_p) else None
    return ply, None, man


def layer_of(man):
    """What layer a source archive says it was trained as, if it says."""
    if not man:
        return None
    fp = ((man.get("train_dataset_fingerprint")) or {})
    if fp.get("layer"):
        return fp["layer"]
    tm = ((man.get("stages") or {}).get("train") or {}).get("metrics") or {}
    return tm.get("layer")


def run(a, pj):
    pj.require(STAGE)
    specs = [s.strip() for s in a.models.split(",") if s.strip()]
    if len(specs) < 2:
        raise events.StageError(f"--models needs two or more, got {len(specs)}")
    name = a.name.strip()
    if not name or name != os.path.basename(name) or name.startswith("."):
        raise events.StageError(f"bad archive name: {a.name!r}")
    final = pj.path("archive", name)
    if os.path.exists(final) and not a.force:
        raise events.StageError(f"archive/{name} already exists", hint="pick another --name, or --force")

    pj.acquire(STAGE)
    st = pj.stage(STAGE)
    st.update({"status": "running", "started": now_iso(), "finished": None, "argv": list(sys.argv),
               "metrics": {}, "checks": [], "artifacts": []})
    pj.save()

    events.start(STAGE, "read")
    sources = []
    for s in specs:
        ply, arch, man = resolve(pj, s)
        head, n, props, off = read_header(ply)
        sources.append({"spec": s, "ply": ply, "archive": arch, "manifest": man, "head": head,
                        "n": n, "props": props, "offset": off, "layer": layer_of(man),
                        "rig_npz_md5": (man or {}).get("rig_npz_md5"), "md5": md5_file(ply)})

    # every layer must be the same world, or the merge silently interleaves two coordinate frames
    rigs = {s["rig_npz_md5"] for s in sources}
    same_frame = len(rigs) == 1 and None not in rigs
    if not same_frame and not a.force:
        detail = ", ".join(f"{s['spec']}: {s['rig_npz_md5'] or 'no manifest'}" for s in sources)
        raise events.StageError("the models do not agree on the solve they came from",
                                hint=f"{detail} — merging them would interleave two world frames; --force to override")
    pj.check(STAGE, "sources_share_a_frame", same_frame,
             value=(f"rig.npz {next(iter(rigs))}" if same_frame
                    else "sources disagree or carry no manifest (merged under --force)"))

    # property lists must match exactly; see the module docstring on why padding is not attempted
    base = sources[0]["props"]
    odd = [s["spec"] for s in sources[1:] if s["props"] != base]
    if odd:
        want = len([p for p, _t in base if p.startswith("f_rest_")])
        got = {s["spec"]: len([p for p, _t in s["props"] if p.startswith("f_rest_")]) for s in sources}
        raise events.StageError(f"these models have different PLY properties: {', '.join(odd)}",
                                hint=f"f_rest counts {got} (the first has {want}) — retrain the layers "
                                     f"with the same SH degree; hs will not pad, because f_rest is "
                                     f"channel-major and padding it is an interleave, not an append")
    pj.check(STAGE, "properties_match", True,
             value=f"{len(base)} properties per splat, identical across {len(sources)} models")

    stride = row_bytes(base)
    total = sum(s["n"] for s in sources)

    events.start(STAGE, "write")
    out = pj.path("archive", name + ".partial")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    dst_ply = os.path.join(out, "merged.ply")
    head = sources[0]["head"].replace(b"element vertex %d" % sources[0]["n"],
                                      b"element vertex %d" % total)
    if b"element vertex %d" % total not in head:
        shutil.rmtree(out, ignore_errors=True)
        raise events.StageError("could not rewrite the vertex count in the header")
    with open(dst_ply, "wb") as w:
        w.write(head)
        for i, s in enumerate(sources):
            with open(s["ply"], "rb") as r:
                r.seek(s["offset"])
                left = s["n"] * stride
                while left > 0:
                    chunk = r.read(min(1 << 22, left))
                    if not chunk:
                        shutil.rmtree(out, ignore_errors=True)
                        raise events.StageError(f"{s['spec']} ended early: {left} bytes short of "
                                                f"{s['n']} splats — is the file truncated?")
                    w.write(chunk)
                    left -= len(chunk)
            events.progress(STAGE, i + 1, len(sources), step="write")

    # the frame travels with the model, exactly as hs archive writes it
    src_arch = next((s["archive"] for s in sources if s["archive"]), None)
    for f in ("rig.npz", "cameras.txt", "images.txt", "points3D.bin"):
        if src_arch and os.path.exists(os.path.join(src_arch, f)):
            shutil.copy2(os.path.join(src_arch, f), os.path.join(out, f))
    if not os.path.exists(os.path.join(out, "rig.npz")) and os.path.exists(pj.rig_npz):
        shutil.copy2(pj.rig_npz, os.path.join(out, "rig.npz"))

    man = {
        "note": "several models in one frame: a merged ply, and the coordinate frame it lives in",
        "archived": now_iso(), "project": pj.root, "name": name, "merged": True,
        "ply": {"source": pj.rel(dst_ply), "md5": md5_file(dst_ply),
                "bytes": os.path.getsize(dst_ply)},
        "rig_npz_md5": md5_file(os.path.join(out, "rig.npz")) if os.path.exists(os.path.join(out, "rig.npz")) else None,
        "merged_from": [{"spec": s["spec"], "ply": pj.rel(s["ply"]) if s["ply"].startswith(pj.root) else s["ply"],
                         "md5": s["md5"], "splats": s["n"], "layer": s["layer"],
                         "rig_npz_md5": s["rig_npz_md5"]} for s in sources],
        "train_dataset_fingerprint": (sources[0]["manifest"] or {}).get("train_dataset_fingerprint"),
        "argv": list(sys.argv),
        "source": pj.m.get("source", {}), "profile_id": pj.m.get("profile_id"),
    }
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)

    if os.path.exists(final):
        shutil.rmtree(final)
    os.rename(out, final)
    dst_ply = os.path.join(final, "merged.ply")

    _h, got, _p, _o = read_header(dst_ply)
    pj.check(STAGE, "splat_count_is_the_sum", got == total,
             value=" + ".join(f"{s['n']:,}" for s in sources) + f" = {got:,}")
    layers = [s["layer"] for s in sources]
    pj.check(STAGE, "layers_are_complementary", sorted(x for x in layers if x) == ["background", "subject"],
             value=", ".join(f"{s['spec']}: {s['layer'] or 'layer not recorded'}" for s in sources),
             needs_human=True)
    pj.metric(STAGE, "archive", pj.rel(final))
    pj.metric(STAGE, "splats", total)
    pj.metric(STAGE, "models", len(sources))
    pj.metric(STAGE, "ply_md5", man["ply"]["md5"])
    pj.metric(STAGE, "bytes", os.path.getsize(dst_ply))
    pj.metric(STAGE, "merged_from", [{"spec": s["spec"], "splats": s["n"], "layer": s["layer"]} for s in sources])
    pj.artifact(STAGE, final, "archive")
    st["status"] = "done"
    st["finished"] = now_iso()
    pj.save()
    pj.release()
