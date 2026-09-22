"""hs export — hand a trained model over: plain PLY plus the web formats, and a shot sheet.

    hs export -p P                                   # train's final export -> deliver/<stem>/
    hs export -p P --archive masked-exposure         # an archived model -> deliver/masked-exposure/
    hs export -p P --ply prune/x_pruned_r03.ply --formats ply,spz --min-opacity 0.05 --name x
    hs export -p P --archive head --subject split/1/subject.ply --shot-sheet

Everything lands in ``deliver/<name>/`` (rebuilt whole on every run, like an archive):

    <name>.ply                the source PLY, byte for byte (SH intact — Nuke and Houdini read it)
    <name>.spz / .sog         compressed web formats, via PlayCanvas splat-transform
    <name>.html               splat-transform's self-contained single-page viewer
    <name>_subject.<fmt>      the same set for --subject (a split subject layer), when given
    manifest.json             source ply + md5, every file's md5 / bytes, the exact splat-transform argv
    shot-sheet.md / .html     with --shot-sheet (hs/shotsheet.py)

splat-transform is found as ``--splat-transform PATH`` / ``HS_SPLAT_TRANSFORM``, else a
``splat-transform`` on PATH, else ``npx -y -- @playcanvas/splat-transform`` (node ≥ 18; the first
run downloads the package). One invocation per output, in its own syntax
``splat-transform [GLOBAL] input [ACTIONS] output``::

    splat-transform -w in.ply -V opacity,gte,0.05 out.sog

``--min-opacity`` is its ``-V/--filter-value`` action, which compares *linear* opacity (0–1,
after the sigmoid), not the PLY's stored logit. With a filter the delivered PLY is written by
splat-transform too, so every format carries the same splats; without one it is a plain copy
and its md5 equals the source's. SOG and HTML compress on the GPU (WebGPU via Dawn); when that
fails and no ``--gpu`` was given, the output is retried once with ``-g cpu`` and a check says so.
"""
import json
import os
import shutil
import sys

from .. import events, runner, shotsheet
from ..project import md5_file, now_iso

STAGE = "export"
FORMATS = ("ply", "spz", "sog", "html")
GPU_FORMATS = ("sog", "html")
NPX_PACKAGE = "@playcanvas/splat-transform"


def add_parser(sub):
    p = sub.add_parser("export", help="deliver a model: plain PLY + spz / sog / html viewer (splat-transform), shot sheet")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--ply", default=None, help="default: the train stage's final export")
    src.add_argument("--archive", default=None, metavar="NAME", help="export the model stored in archive/NAME")
    p.add_argument("--formats", default=",".join(FORMATS), help=f"comma list from {','.join(FORMATS)} (default: all)")
    p.add_argument("--subject", default=None, metavar="PLY",
                   help="also export this subject layer (e.g. split/N/subject.ply, project-relative) in the same formats")
    p.add_argument("--min-opacity", type=float, default=0.0,
                   help="drop splats under this linear opacity (0-1) via splat-transform -V opacity,gte,X (default 0 = keep all)")
    p.add_argument("--name", default=None, help="deliver/<name> (default: the archive name, else the ply's stem)")
    p.add_argument("--shot-sheet", action="store_true", help="also write shot-sheet.md and shot-sheet.html")
    p.add_argument("--move", default=None, help="the move the shot sheet describes (default: the latest render of this ply)")
    p.add_argument("--splat-transform", default=os.environ.get("HS_SPLAT_TRANSFORM"),
                   help="splat-transform executable (HS_SPLAT_TRANSFORM); default: on PATH, else npx")
    p.add_argument("--gpu", default=None, help="splat-transform -g: GPU adapter index or 'cpu'")
    return p


# ---------------------------------------------------------------------------- the tool

def splat_transform_argv(explicit=None):
    """-> the argv prefix that runs splat-transform, or raise StageError.

    Like ingest.redline_exe: an explicit path must exist and be executable; a bare name is
    looked up on PATH. Without one, an installed ``splat-transform`` wins over npx."""
    if explicit:
        p = os.path.expanduser(explicit)
        found = runner.which(p)
        if not found:
            raise events.StageError(f"splat-transform not found: {explicit}",
                                    hint="npm install -g @playcanvas/splat-transform, or fix --splat-transform / HS_SPLAT_TRANSFORM")
        return [found]
    found = shutil.which("splat-transform")
    if found:
        return [found]
    npx = shutil.which("npx")
    if npx:
        return [npx, "-y", "--", NPX_PACKAGE]
    raise events.StageError("splat-transform needs node: no splat-transform and no npx on PATH",
                            hint="brew install node (npx then fetches @playcanvas/splat-transform), "
                                 "or --splat-transform PATH; --formats ply needs neither")


def splat_transform_version(prefix, timeout=60):
    """First line of `splat-transform --version`, or None. Never raises."""
    import subprocess
    try:
        r = subprocess.run(list(prefix) + ["--version"], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    out = (r.stdout or r.stderr or "").strip()
    return out.splitlines()[0] if r.returncode == 0 and out else None


def transform_argv(prefix, src, out, min_opacity=0.0, gpu=None):
    """One splat-transform invocation: [GLOBAL] input [ACTIONS] output."""
    argv = list(prefix) + ["-w"]
    if gpu is not None:
        argv += ["-g", str(gpu)]
    argv.append(src)
    if min_opacity and min_opacity > 0:
        argv += ["-V", f"opacity,gte,{min_opacity:g}"]
    argv.append(out)
    return argv


def parse_formats(s):
    fmts = [f.strip().lower().lstrip(".") for f in (s or "").split(",") if f.strip()]
    bad = [f for f in fmts if f not in FORMATS]
    if bad or not fmts:
        raise events.StageError(f"unknown format(s) {', '.join(bad) or '(none)'}; choose from {', '.join(FORMATS)}")
    return [f for f in FORMATS if f in fmts]       # fixed order, no duplicates


# ---------------------------------------------------------------------------- sources

def _load_json(path):
    try:
        with open(path) as f:
            v = json.load(f)
        return v if isinstance(v, dict) else None
    except (OSError, ValueError):
        return None


def archive_source(pj, name):
    """-> (ply path, archive manifest | None) for archive/<name>."""
    d = pj.path("archive", name)
    if not os.path.isdir(d):
        raise events.StageError(f"no archive/{name}", hint="hs archive -p P --name NAME first, or --ply PATH")
    man = _load_json(os.path.join(d, "manifest.json"))
    plys = sorted(f for f in os.listdir(d) if f.lower().endswith(".ply"))
    want = os.path.basename(str((man or {}).get("ply", {}).get("source") or ""))
    if want in plys:
        return os.path.join(d, want), man
    if len(plys) == 1:
        return os.path.join(d, plys[0]), man
    raise events.StageError(f"archive/{name} holds {len(plys)} .ply files and its manifest names none of them",
                            hint="pass the one you mean with --ply")


def splat_count(path):
    from .merge import read_header
    try:
        return read_header(path)[1]
    except (events.StageError, OSError):
        return None


def resolve_subject(pj, s):
    for p in (s if os.path.isabs(s) else pj.path(s), os.path.abspath(s)):
        if os.path.isfile(p):
            return p
    raise events.StageError(f"--subject not found: {s}", hint="project-relative (split/N/subject.ply) or absolute")


# ---------------------------------------------------------------------------- the shot sheet's inputs

def _plyrel(pj, p):
    ap = os.path.abspath(p)
    return pj.rel(ap) if ap.startswith(pj.root + os.sep) else ap


def holdout_reports(pj, src_rel, arch):
    """Every views/*_report.json, each marked by whether it scored this very file."""
    d = pj.path("views")
    if not os.path.isdir(d):
        return []
    arch_src = ((arch or {}).get("ply") or {}).get("source")
    out = []
    for f in sorted(os.listdir(d)):
        if not f.endswith("_report.json"):
            continue
        rep = _load_json(os.path.join(d, f))
        if rep is None:
            continue
        scored = rep.get("ply")
        if scored == src_rel:
            rel = "this model"
        elif arch_src and scored == arch_src:
            rel = "archive's source path (scored before archiving; md5 not in the report)"
        else:
            rel = "another model"
        out.append({"name": f[:-len("_report.json")], "report": rep, "relation": rel})
    out.sort(key=lambda r: r["relation"] != "this model")
    return out


def move_and_grade(pj, src_rel, move_name):
    """The move the sheet describes and its grade look: --move, else the latest render of this
    ply (stages.render.runs records move and ply), else the only move in move/."""
    runs = pj.stage("render").get("runs") or {}
    mine = sorted(((r.get("finished") or "", name, r) for name, r in runs.items()
                   if isinstance(r, dict) and r.get("ply") == src_rel), key=lambda x: x[0])
    render_name = None
    if move_name:
        for _, rn, r in reversed(mine):
            if r.get("move") == move_name:
                render_name = rn
                break
    elif mine:
        _, render_name, r = mine[-1]
        move_name = r.get("move") or render_name
    else:
        mdir = pj.path("move")
        cands = sorted(f[:-5] for f in os.listdir(mdir)
                       if f.endswith(".json") and not f.endswith((".keys.json", "_frame.json"))
                       and f != "subject.json") if os.path.isdir(mdir) else []
        if len(cands) == 1:
            move_name = cands[0]
        else:
            why = (f"no render of this ply and {len(cands)} moves in move/; pass --move" if cands
                   else "no move in the project")
            return {"why": why}, None
    path = _load_json(pj.path("move", f"{move_name}.json"))
    mv = {"name": move_name, "path": path or {}, "keys": _load_json(pj.path("move", f"{move_name}.keys.json")),
          "run": (pj.stage("move").get("runs") or {}).get(move_name), "render": render_name}
    if path is None:
        mv["why"] = f"move/{move_name}.json is missing"
    grade = None
    for g in ([render_name] if render_name else []) + [move_name]:
        grade = _load_json(pj.path("grade", f"{g}.json"))
        if grade:
            break
    return mv, grade


def _build(a, pj, ply, arch, subject, name, fmts, prefix, out):
    src_rel = _plyrel(pj, ply)
    src_md5 = md5_file(ply)
    if arch is not None:
        amd5 = (arch.get("ply") or {}).get("md5")
        pj.check(STAGE, "archive_ply_md5_matches", amd5 in (None, src_md5),
                 value=f"{src_md5}" + ("" if amd5 in (None, src_md5) else f" vs the archive manifest's {amd5}"))
        if amd5 not in (None, src_md5):
            raise events.StageError(f"{os.path.basename(ply)} does not match its archive manifest",
                                    hint="the archived ply was changed after hs archive; re-archive it")
    pj.metric(STAGE, "ply", src_rel)
    pj.metric(STAGE, "ply_md5", src_md5)
    n_src = splat_count(ply)
    pj.metric(STAGE, "splats", n_src)

    version = None
    if prefix:
        version = splat_transform_version(prefix)
        pj.record_tool("splat-transform", {"argv": prefix, "version": version})
        pj.metric(STAGE, "splat_transform", {"argv": prefix, "version": version})

    files = []
    jobs = [("model", ply, name)] + ([("subject", subject, f"{name}_subject")] if subject else [])
    total = len(jobs) * len(fmts)
    done = 0
    for role, src, stem in jobs:
        for fmt in fmts:
            events.progress(STAGE, done, total, detail=f"{stem}.{fmt}", step="convert", force=True)
            dst = os.path.join(out, f"{stem}.{fmt}")
            rec = {"path": os.path.basename(dst), "role": role, "format": fmt, "source": _plyrel(pj, src)}
            if fmt == "ply" and not a.min_opacity:
                events.start(STAGE, f"copy {role}")
                shutil.copy2(src, dst)
                rec["argv"] = "copy"
                ok = md5_file(dst) == md5_file(src)
                pj.check(STAGE, f"{role}_ply_copy_matches_source", ok, value=os.path.basename(dst))
                if not ok:
                    raise events.StageError(f"copy of {os.path.basename(src)} does not match its source")
            else:
                events.start(STAGE, f"{fmt} {role}")
                argv = transform_argv(prefix, src, dst, a.min_opacity, a.gpu)
                try:
                    runner.run(argv, STAGE, log_path=pj.log_path(STAGE))
                except events.StageError:
                    if fmt not in GPU_FORMATS or a.gpu is not None:
                        raise
                    # SOG/HTML compress on the GPU; a machine without a WebGPU adapter fails there
                    argv = transform_argv(prefix, src, dst, a.min_opacity, "cpu")
                    runner.run(argv, STAGE, log_path=pj.log_path(STAGE))
                    pj.check(STAGE, f"{role}_{fmt}_on_gpu", False,
                             value="GPU compression failed; retried with -g cpu (same output, slower)")
                if not os.path.isfile(dst) or os.path.getsize(dst) == 0:
                    raise events.StageError(f"splat-transform wrote no {os.path.basename(dst)}",
                                            hint="see logs/export.log")
                rec["argv"] = argv
            rec["bytes"] = os.path.getsize(dst)
            rec["md5"] = md5_file(dst)
            if fmt == "ply":
                rec["splats"] = splat_count(dst)
            files.append(rec)
            done += 1
    events.progress(STAGE, done, total, step="convert", force=True)

    exp = {
        "note": "hs export: a delivered model; every file's md5, and the exact command that made it",
        "name": name, "exported": now_iso(), "project": pj.root, "project_name": pj.m.get("name"),
        "source": {"ply": src_rel, "md5": src_md5, "bytes": os.path.getsize(ply), "splats": n_src,
                   "archive": a.archive or ((arch or {}).get("name"))},
        "subject": ({"ply": _plyrel(pj, subject), "md5": md5_file(subject), "splats": splat_count(subject)}
                    if subject else None),
        "formats": fmts, "min_opacity": a.min_opacity,
        "splat_transform": {"argv": prefix, "version": version} if prefix else None,
        "files": files, "argv": list(sys.argv),
    }
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(exp, f, indent=1)
    sheets = []
    if a.shot_sheet:
        events.start(STAGE, "shot sheet")
        mv, grade = move_and_grade(pj, src_rel, a.move)
        md, html = shotsheet.shot_sheet(pj.m, exp, archive=arch, reports=holdout_reports(pj, src_rel, arch),
                                        move=mv, grade=grade)
        for ext, body in (("md", md), ("html", html)):
            with open(os.path.join(out, f"shot-sheet.{ext}"), "w", encoding="utf-8") as f:
                f.write(body)
            sheets.append(f"shot-sheet.{ext}")

    return files, exp, sheets


# ---------------------------------------------------------------------------- run

def run(a, pj):
    fmts = parse_formats(a.formats)
    if not 0.0 <= a.min_opacity < 1.0:
        raise events.StageError(f"--min-opacity {a.min_opacity} is a linear opacity: 0 <= x < 1")
    arch = None
    if a.archive:
        ply, arch = archive_source(pj, a.archive)
    elif a.ply:
        ply = os.path.abspath(a.ply)
        arch = _load_json(os.path.join(os.path.dirname(ply), "manifest.json"))
        if arch is not None and "rig_npz_md5" not in arch:
            arch = None                               # a manifest.json that is not an archive's
    else:
        pj.require(STAGE)
        from .prune import final_export
        ply = final_export(pj)
    if not ply or not os.path.isfile(ply):
        raise events.StageError("no .ply to export", hint="hs train first, or --ply PATH / --archive NAME")
    subject = resolve_subject(pj, a.subject) if a.subject else None
    name = a.name or a.archive or os.path.splitext(os.path.basename(ply))[0]
    if not name or name.startswith(".") or os.sep in name or "/" in name:
        raise events.StageError(f"export name must be a plain folder name, not {name!r}")
    need_tool = any(f != "ply" for f in fmts) or (a.min_opacity > 0 and "ply" in fmts)
    prefix = splat_transform_argv(a.splat_transform) if need_tool else None

    pj.acquire(STAGE)
    st = pj.stage(STAGE)
    prev_runs = st.get("runs")
    st.clear()
    st.update({"status": "running", "started": now_iso(), "finished": None, "argv": list(sys.argv),
               "pid": os.getpid(), "metrics": {}, "checks": [], "artifacts": []})
    if prev_runs:
        st["runs"] = prev_runs
    pj.save()

    final = pj.path("deliver", name)
    out = pj.path("deliver", f".{name}.building")
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    try:
        files, exp, sheets = _build(a, pj, ply, arch, subject, name, fmts, prefix, out)
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)   # never leave a half-built deliver folder behind
        raise
    if os.path.exists(final):
        shutil.rmtree(final)
    os.rename(out, final)

    for rec in files:
        pj.artifact(STAGE, os.path.join(final, rec["path"]), rec["format"])
    pj.artifact(STAGE, os.path.join(final, "manifest.json"), "json")
    for s in sheets:
        pj.artifact(STAGE, os.path.join(final, s), "shot-sheet")
    pj.metric(STAGE, "deliver", pj.rel(final))
    pj.metric(STAGE, "files", {r["path"]: r["md5"] for r in files})
    kept = [r["splats"] for r in files if r["role"] == "model" and r.get("splats") is not None]
    if kept:
        pj.metric(STAGE, "splats_exported", kept[0])
    pj.record_run(STAGE, name, ply=exp["source"]["ply"], ply_md5=exp["source"]["md5"], deliver=pj.rel(final))
    pj.finish(STAGE, ok=True)
