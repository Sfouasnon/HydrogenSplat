#!/usr/bin/env python3
"""Reproject the solve's points into one capture's photograph and the model's render of it.

Is a render that is "off" the solve's fault or training's? The sparse points this image
observed are drawn on the photograph at their reprojection, with a line to the keypoint they
came from: long lines mean the solve itself is wrong. The same points go on the render of that
exact pose, and phase correlation (global, then per quadrant) says how far the render sits from
the photograph. A near-uniform shift with well-placed sparse points is a splat / pose-in-training
fault; misplaced sparse points are a solve fault. hs/overlay.py has the details and the sign.

The render is, in order: --render IMG; a fresh brush-path-render of the pose when --ply is
given; else the frame `hs views --keep-frames` left in views/frames for this pose (matched by
pose against views/<name>.json). Without any, only the photograph overlay is written.

  python3 engine/tools/reproject_overlay.py -p Projects/2026-09-15_head --capture cap045
  python3 engine/tools/reproject_overlay.py -p P --capture cap045 --eye R --ply archive/holdout-base/export_40000.ply

Writes <out>/overlay_gt.jpg, overlay_render.jpg, report.json
(default out: <project>/views/overlay_<capture>_<eye>/).
"""
import argparse
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # engine/, so `hs` imports without an install

import numpy as np  # noqa: E402

from hs import overlay, rig  # noqa: E402

DEFAULT_RENDER = "~/Desktop/Apps/brush/target/release/brush-path-render"


def c2w(G, v):
    M = np.eye(4)
    M[:3, :3] = G["R"][v].astype(float).T
    M[:3, 3] = G["C"][v].astype(float) / 1000.0
    return M


def find_view(G, names, capture, eye):
    """--capture 'cap045' | 'cap045_L' | 'L/cap045' | '12' (a capture index) -> view index."""
    L = rig.left_indices(G, names)
    c = capture.strip()
    if "/" in c:
        eye, c = c.split("/", 1)
    c = os.path.splitext(c)[0]
    if c.endswith(("_L", "_R")):
        c, eye = c[:-2], c[-1]
    if eye == "R" and not rig.is_stereo(G):
        sys.exit("--eye R needs a stereo rig; this rig.npz is mono")
    if c.isdigit() and int(c) < len(L) and f"{c}_L" not in names:
        v = int(L[int(c)])
    else:
        want = f"{c}_L"
        if want not in names:
            sys.exit(f"no capture {capture!r} in rig.npz (e.g. {rig.capture_name(names[int(L[0])])})")
        v = names.index(want)
    return v + (1 if eye == "R" else 0), eye


def existing_render(root, M):
    """A frame hs views --keep-frames left for this pose, or None."""
    frames = os.path.join(root, "views", "frames")
    if not os.path.isdir(frames):
        return None
    n_png = len([f for f in os.listdir(frames) if f.endswith(".png")])
    for pj in sorted(glob.glob(os.path.join(root, "views", "*.json")), key=os.path.getmtime, reverse=True):
        try:
            p = json.load(open(pj))
        except (OSError, ValueError):
            continue
        fr = p.get("frames") if isinstance(p, dict) else None
        if not fr or len(fr) != n_png:          # views/frames holds the most recent pass only
            continue
        for i, f in enumerate(fr):
            if np.allclose(np.array(f.get("c2w")), M, atol=1e-6):
                img = os.path.join(frames, f"frame_{i:04d}.png")
                if os.path.exists(img):
                    return img
    return None


def fresh_render(G, v, ply, render_bin, out_dir):
    """One frame at view v's own pose, K and canvas."""
    w, h = (int(G["wh"][v][0]), int(G["wh"][v][1])) if "wh" in G.files else (int(G["w"]), int(G["h"]))
    path = os.path.join(out_dir, "pose.json")
    json.dump({"note": "reproject_overlay: one frame at a capture's own pose", "width": w, "height": h,
               "K": G["K"][v].astype(float).tolist(), "fps": 30.0, "frames": [{"c2w": c2w(G, v).tolist()}]},
              open(path, "w"), indent=1)
    fdir = os.path.join(out_dir, "render")
    os.makedirs(fdir, exist_ok=True)
    r = subprocess.run([os.path.expanduser(render_bin), ply, "--path", path, "-o", fdir],
                       capture_output=True, text=True)
    img = os.path.join(fdir, "frame_0000.png")
    if r.returncode != 0 or not os.path.exists(img):
        sys.exit(f"brush-path-render failed ({r.returncode}): {(r.stderr or r.stdout).strip()[-400:]}")
    return img


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--project", required=True)
    ap.add_argument("--capture", required=True, help="cap045, cap045_R, L/cap045, GA, or a capture index")
    ap.add_argument("--eye", choices=("L", "R"), default="L")
    ap.add_argument("--ply", default=None, help="render this model at the capture's pose (needs brush-path-render)")
    ap.add_argument("--render-bin", default=os.environ.get("HS_PATH_RENDER", DEFAULT_RENDER))
    ap.add_argument("--render", default=None, help="an image already rendered at this pose")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    import cv2

    root = os.path.abspath(os.path.expanduser(a.project))
    ds = os.path.join(root, "train", "dataset")
    rig_npz = os.path.join(ds, "rig.npz")
    if not os.path.exists(rig_npz):
        sys.exit(f"no {rig_npz}: hs solve first")
    G, names, _ = rig.load(rig_npz)
    v, eye = find_view(G, names, a.capture, a.eye)
    cap = rig.capture_name(names[v])
    out = os.path.abspath(a.out or os.path.join(root, "views", f"overlay_{cap}_{eye}"))
    os.makedirs(out, exist_ok=True)

    gt_path = next((p for p in (os.path.join(ds, "images", eye, cap + e) for e in (".jpg", ".jpeg", ".png"))
                    if os.path.exists(p)), None)
    if gt_path is None:
        sys.exit(f"no photograph train/dataset/images/{eye}/{cap}.*")
    gt = cv2.imread(gt_path)
    if a.render:
        ren_path, how = os.path.abspath(a.render), "given"
    elif a.ply:
        ren_path, how = fresh_render(G, v, os.path.abspath(a.ply), a.render_bin, out), "brush-path-render"
    else:
        ren_path, how = existing_render(root, c2w(G, v)), "views/frames"
    ren = cv2.imread(ren_path) if ren_path else None

    pts = overlay.points_for_view(G, v, ds, eye)
    rep = overlay.make(gt, ren, pts, out, label=f"{cap} {eye}")
    rep.update({"capture": cap, "view": names[v], "eye": eye, "photograph": os.path.relpath(gt_path, root),
                "render": (os.path.relpath(ren_path, root) if ren_path and ren_path.startswith(root) else ren_path),
                "render_source": how if ren_path else None})
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1)

    print(f"{cap} {eye}: {rep['n_points']} points ({rep['points_source']})"
          + (f", sparse residual median {rep['sparse_residual_px']['median']:.2f} px" if "sparse_residual_px" in rep else ""))
    if "shift_px" in rep:
        q = rep["quadrant_shifts"]
        print(f"  render ({how}): shift {rep['shift_px'][0]:+.2f}, {rep['shift_px'][1]:+.2f} px, response {rep['response']:.2f}")
        print("  quadrants: " + "  ".join(f"{k} {q[k][0]:+.2f},{q[k][1]:+.2f}" for k in ("TL", "TR", "BL", "BR")))
    else:
        print("  no render at this pose (--ply PLY, --render IMG, or hs views --keep-frames first)")
    print(f"  {rep['diagnosis']}")
    print("wrote", out)
    return rep


if __name__ == "__main__":
    main()
