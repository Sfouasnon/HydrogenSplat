"""The shot sheet: one page that says what a delivered model is and where it came from.

A pure function of the records the pipeline already keeps — the project manifest, the archive
manifest a model was stored with, the hold-out reports, the move, the grade settings and the
export's own manifest — to a Markdown string and an HTML string. Nothing here reads a file,
so it can be tested on dicts and cannot drift from what ``hs export`` wrote.

Every piece is optional except the export manifest. A project with no archive, no hold-out
scores, no move or no grade gets a sheet that says "not recorded" for that section rather than
an exception or a silently empty line: a sheet that omits the gap reads as a model with nothing
to declare.

    md, html = shot_sheet(project_manifest, export_manifest, archive=archive_manifest,
                          reports=[{"name": "views", "report": {...}, "relation": "this model"}],
                          move={"name": "arc180", "path": {...}, "keys": {...}, "run": {...}},
                          grade={"lift": 0.0, ...})
"""
import html as _html
import statistics

NR = "not recorded"

SOLVE_KEYS = ("source_kind", "num_images", "num_frames", "num_points", "mean_reproj_px", "median_reproj_px",
              "worst_image_reproj_px", "rig_baseline_mm", "scale_to_m", "subject_mm", "focal_px",
              "azimuth_range_deg", "elevation_range_deg", "distance_range_mm")
SCORE_KEYS = (("psnr_db", "PSNR dB"), ("psnr_interior_db", "PSNR interior dB"), ("psnr_edge_db", "PSNR edge dB"),
              ("displaced_fraction", "displaced"), ("retained_edge_energy_norm", "edge energy (norm)"))
GRADE_KEYS = ("lift", "gamma", "gain", "lift_rgb", "gamma_rgb", "gain_rgb", "sharpen", "aspect", "headroom_mm")


def _d(x):
    return x if isinstance(x, dict) else {}


def _fmt(v):
    if v is None:
        return NR
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, (list, tuple)):
        return ", ".join(_fmt(x) for x in v) if v else "none"
    if isinstance(v, dict):
        return ", ".join(f"{k} {_fmt(x)}" for k, x in v.items()) if v else "none"
    return str(v)


def _median(rows, key):
    vals = [r.get(key) for r in rows if isinstance(r, dict) and isinstance(r.get(key), (int, float))
            and not isinstance(r.get(key), bool)]
    return round(float(statistics.median(vals)), 3) if vals else None


def _argv(v):
    if isinstance(v, (list, tuple)) and v:
        return " ".join(str(x) for x in v)
    return NR


# ---------------------------------------------------------------------------- sections
# A section is (title, note, body) with body either [(label, value), ...] ("kv") or
# {"head": [...], "rows": [[...], ...]} ("table"), or None for "not recorded" as a whole.

def _project(pm, ex):
    return ("Project", None, [
        ("name", pm.get("name") or ex.get("project_name") or NR),
        ("folder", ex.get("project") or NR),
        ("created", pm.get("created") or NR),
        ("calibration profile", pm.get("profile_id") or NR),
    ])


def _source(pm, arch):
    src = _d(arch.get("source")) if arch.get("source") else _d(pm.get("source"))
    whence = "archive manifest" if arch.get("source") else "project manifest"
    if not src:
        return ("Source", None, None)
    kind = src.get("kind") or ("video clip" if src.get("clip") else NR)
    rows = [("kind", kind)]
    if src.get("clip"):
        rows.append(("clip", src["clip"]))
        rows.append(("clip md5", src.get("md5") or NR))
    else:
        rows.append(("frames", src.get("frames") or NR))
        rows.append(("frames digest (md5)", src.get("md5") or NR))
    rows.append(("original path", src.get("original_path") or NR))
    return ("Source", f"from the {whence}", rows)


def _stages(pm, arch):
    """The stage records that describe this model: the archive's snapshot when there is one
    (the project may have been re-solved and retrained since), else the project's own."""
    if arch.get("stages"):
        return _d(arch.get("stages")), "archive manifest"
    return _d(pm.get("stages")), "project manifest"


def _solve(pm, arch):
    stages, whence = _stages(pm, arch)
    sm = _d(_d(stages.get("solve")).get("metrics"))
    rows = [(k, sm[k]) for k in SOLVE_KEYS if k in sm]
    if not rows:
        return ("Solve", None, None)
    st = _d(stages.get("solve")).get("status")
    return ("Solve", f"from the {whence}" + (f"; solve status {st}" if st and st != "done" else ""), rows)


def _model(pm, ex, arch):
    stages, whence = _stages(pm, arch)
    tr = _d(stages.get("train"))
    tm = _d(tr.get("metrics"))
    src = _d(ex.get("source"))
    fp = _d(tm.get("dataset_fingerprint"))
    bc = _d(tm.get("brush_config"))
    tools_brush = _d(_d(pm.get("tools")).get("brush"))
    rows = [
        ("archive", arch.get("name") or src.get("archive") or "not archived"),
        ("ply", src.get("ply") or NR),
        ("ply md5", src.get("md5") or NR),
        ("splats", src.get("splats") if src.get("splats") is not None else NR),
    ]
    note = f"training record from the {whence}"
    want = tm.get("final_export_md5")
    arch_md5 = _d(arch.get("ply")).get("md5")
    # does the training record describe THIS file? Only an md5 says so.
    if src.get("md5") and (src.get("md5") == arch_md5 or src.get("md5") == want):
        describes = True
    elif not tr:
        describes = None
    else:
        describes = False
    if describes is False:
        note += (f"; its final export md5 is {want or NR}, not this ply's — the training fields below "
                 "may describe a different model")
    if not tr:
        rows.append(("training", NR))
        return ("Model", note, rows)
    rows += [
        ("train final_splats", tm.get("final_splats") if tm.get("final_splats") is not None else NR),
        ("layer", tm.get("layer") or bc.get("layer") or fp.get("layer") or NR),
        ("masks used", tm.get("masks_used") if tm.get("masks_used") is not None else NR),
        ("train views", tm.get("train_views") if tm.get("train_views") is not None else NR),
        ("excluded views", tm.get("excluded_views") if tm.get("excluded_views") is not None else NR),
        ("train argv", _argv(tr.get("argv"))),
        ("Brush commit", _brush_commit(bc, tools_brush)),
        ("dataset rig.npz md5", fp.get("rig_npz_md5") or arch.get("rig_npz_md5") or NR),
    ]
    return ("Model", note, rows)


def _brush_commit(bc, tools_brush):
    c = bc.get("commit") or tools_brush.get("git_commit")
    if not c:
        return NR
    extra = []
    br = bc.get("branch") or tools_brush.get("git_branch")
    if br:
        extra.append(br)
    dirty = bc.get("dirty") if "dirty" in bc else tools_brush.get("git_dirty")
    if dirty:
        extra.append("dirty tree")
    return c + (f" ({', '.join(extra)})" if extra else "")


def _move(move):
    mv = _d(move)
    if not mv.get("name"):
        why = mv.get("why")
        return ("Move", why, None)
    path = _d(mv.get("path"))
    frames = path.get("frames")
    n = len(frames) if isinstance(frames, list) else None
    fps = path.get("fps")
    rows = [("name", mv["name"]), ("frames", n if n is not None else NR), ("fps", fps if fps else NR)]
    if n and isinstance(fps, (int, float)) and fps > 0:
        rows.append(("duration", f"{n / fps:.2f} s"))
    if path.get("width") and path.get("height"):
        rows.append(("frame size", f"{path['width']}x{path['height']}"))
    keys = _d(mv.get("keys"))
    run = _d(mv.get("run"))
    info = _d(run.get("info"))
    if isinstance(keys.get("keys"), list):
        ts = [k.get("t") for k in keys["keys"] if isinstance(k, dict)]
        rows.append(("keyframes", f"{len(ts)} (app keyframes, move/{mv['name']}.keys.json)"))
        rows.append(("key times s", ", ".join(f"{t:.2f}" for t in ts if isinstance(t, (int, float))) or NR))
    elif info.get("keys"):
        rows.append(("keyframes", f"{len(info['keys'])} capture keys ({run.get('preset') or 'hs move'})"))
        rows.append(("key captures", info["keys"]))
    elif info.get("window"):
        rows.append(("keyframes", f"sweep over captures {info['window']}"))
    elif info.get("script"):
        rows.append(("keyframes", f"cue script {info['script']}"))
    else:
        rows.append(("keyframes", NR))
    if mv.get("render"):
        rows.append(("render", mv["render"]))
    return ("Move", None, rows)


def _scores(reports):
    reps = [r for r in (reports or []) if isinstance(r, dict) and isinstance(r.get("report"), dict)]
    if not reps:
        return ("Hold-out scores", "no views/*_report.json in the project", None)
    head = ["report", "scored ply", "model", "views"] + [lab for _, lab in SCORE_KEYS]
    rows = []
    for r in reps:
        views = r["report"].get("views") if isinstance(r["report"].get("views"), list) else []
        row = [r.get("name") or "?", r["report"].get("ply") or NR, r.get("relation") or NR, len(views)]
        for key, _ in SCORE_KEYS:
            m = _median(views, key)
            if m is None:
                row.append("—")
            elif key == "displaced_fraction":
                row.append(f"{100 * m:.1f}%")
            else:
                row.append(f"{m:.3f}" if key.startswith("retained") else f"{m:.2f}")
        rows.append(row)
    return ("Hold-out scores", "medians over each report's views (hs views); only rows whose model column says "
            "\"this model\" scored the file delivered here", {"head": head, "rows": rows})


def _grade(grade):
    g = _d(grade)
    if not g:
        return ("Grade", None, None)
    rows = [(k, g[k]) for k in GRADE_KEYS if k in g]
    if g.get("move"):
        rows.insert(0, ("look", f"grade/{g['move']}.json"))
    if g.get("output"):
        rows.append(("graded video", g["output"]))
    return ("Grade", None, rows or None)


def _export(ex):
    files = [f for f in (ex.get("files") or []) if isinstance(f, dict)]
    st = _d(ex.get("splat_transform"))
    rows = [
        ("exported", ex.get("exported") or NR),
        ("min opacity", ex.get("min_opacity") if ex.get("min_opacity") else "0 (no filter)"),
        ("splat-transform", st.get("version") or ("not used" if not st else NR)),
    ]
    table = {"head": ["file", "role", "format", "splats", "bytes", "md5"],
             "rows": [[f.get("path") or "?", f.get("role") or "model", f.get("format") or "?",
                       f.get("splats") if f.get("splats") is not None else "—",
                       f.get("bytes") if f.get("bytes") is not None else "—", f.get("md5") or NR]
                      for f in files]}
    return [("Export", None, rows), ("Files", None, table if files else None)]


def sections(project, export, archive=None, reports=(), move=None, grade=None):
    pm, ex, arch = _d(project), _d(export), _d(archive)
    return [_project(pm, ex), _source(pm, arch), _solve(pm, arch), _model(pm, ex, arch), _move(move),
            _scores(reports), _grade(grade)] + _export(ex)


# ---------------------------------------------------------------------------- rendering

def _md_cell(v):
    return _fmt(v).replace("|", "\\|").replace("\n", " ")


def to_markdown(title, secs):
    out = [f"# {title}", ""]
    for name, note, body in secs:
        out.append(f"## {name}")
        out.append("")
        if note:
            out.append(f"_{note}_")
            out.append("")
        if body is None:
            out.append(NR)
        elif isinstance(body, dict):
            out.append("| " + " | ".join(body["head"]) + " |")
            out.append("|" + "---|" * len(body["head"]))
            for row in body["rows"]:
                out.append("| " + " | ".join(_md_cell(c) for c in row) + " |")
        else:
            for label, value in body:
                out.append(f"- **{label}**: {_md_cell(value)}")
        out.append("")
    return "\n".join(out)


CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1f;--muted:#6b6b70;--rule:#dcdcdf;--head:#f0f0f2}
@media (prefers-color-scheme: dark){:root{--bg:#161618;--fg:#ececef;--muted:#9a9aa1;--rule:#34343a;--head:#222226}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Helvetica Neue",Arial,sans-serif;
max-width:980px;margin:0 auto;padding:24px 16px}
h1{font-size:22px;margin:0 0 16px}h2{font-size:15px;margin:28px 0 8px;border-bottom:1px solid var(--rule);padding-bottom:4px}
.note{color:var(--muted);font-size:12px;margin:0 0 8px}.nr{color:var(--muted);font-style:italic}
dl{display:grid;grid-template-columns:minmax(140px,220px) 1fr;gap:4px 16px;margin:0}
dt{color:var(--muted)}dd{margin:0;word-break:break-all;font-variant-numeric:tabular-nums}
.wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:4px 8px;border-bottom:1px solid var(--rule);white-space:nowrap}th{background:var(--head)}
"""


def _h(v):
    s = _fmt(v)
    esc = _html.escape(s)
    return f'<span class="nr">{esc}</span>' if s == NR else esc


def to_html(title, secs):
    out = ["<!doctype html>", '<html lang="en"><head><meta charset="utf-8">',
           '<meta name="viewport" content="width=device-width,initial-scale=1">',
           f"<title>{_html.escape(title)}</title><style>{CSS}</style></head><body>",
           f"<h1>{_html.escape(title)}</h1>"]
    for name, note, body in secs:
        out.append(f"<h2>{_html.escape(name)}</h2>")
        if note:
            out.append(f'<p class="note">{_html.escape(note)}</p>')
        if body is None:
            out.append(f'<p class="nr">{NR}</p>')
        elif isinstance(body, dict):
            out.append('<div class="wrap"><table><thead><tr>'
                       + "".join(f"<th>{_html.escape(h)}</th>" for h in body["head"]) + "</tr></thead><tbody>")
            for row in body["rows"]:
                out.append("<tr>" + "".join(f"<td>{_h(c)}</td>" for c in row) + "</tr>")
            out.append("</tbody></table></div>")
        else:
            out.append("<dl>" + "".join(f"<dt>{_html.escape(lab)}</dt><dd>{_h(val)}</dd>" for lab, val in body)
                       + "</dl>")
    out.append("</body></html>")
    return "\n".join(out)


def shot_sheet(project, export, archive=None, reports=(), move=None, grade=None):
    """-> (markdown, html). Every argument but ``export`` may be None or empty."""
    ex = _d(export)
    title = f"Shot sheet — {ex.get('name') or _d(project).get('name') or 'export'}"
    secs = sections(project, export, archive, reports, move, grade)
    return to_markdown(title, secs), to_html(title, secs)
