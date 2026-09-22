# HydrogenSplat.app (M1)

SwiftUI orchestrator over the `hs` engine (strategy §2): it never links Python; every stage is
`hs <stage> …` as a child process and the UI is built from its JSON-lines events.

```
cd app && swift build && swift run HydrogenSplat     # or: open Package.swift (Xcode) and Run
cd app && swift test
```

`swift build` / `swift test` verify the code; they do not touch the bundle you double-click.
After a change, rebuild that with `app/scripts/make_app.sh --install` (below) — otherwise the
app you launch is still the previous build, however green the tests are.

M1 scope (strategy §7): Setup (engine paths, `hs tools`), the project list (every
`Projects/*/manifest.json`), New Project (phone clips via `hs phone`, pull + validate via
`hs ingest --phone`, or a dropped clip via `hs ingest --clip`), a read-only project page
(stages, checks, metrics, logs), and Event Replay (`hs replay` of `engine/tests/events/*.jsonl`
through the same live panel a real run uses).

Layout: `Sources/HSCore` — events, the process runner, manifests, projects (no views, unit
tested); `Sources/HydrogenSplat` — the views.

The engine config (repo, `hs` path, projects folder, extra PATH) is saved in UserDefaults;
defaults come from the checkout the app was built from. A Finder-launched app gets a bare
PATH, so `/opt/homebrew/bin:/usr/local/bin` is prepended for every child.

## Training in the app (M3 start)

Each project page has a **Train** panel: length, growth stop, hold-out, masks, and (under
"Detail and experiments") refine interval, split size, anti-aliasing (`--min-scale-factor`),
background noise, extra excluded views and raw Brush arguments — each with a one-line
explanation. It shows the exact `hs` commands, then runs train → archive → score views as one
queue with a live splat-growth chart. If the current model isn't in any archive it asks before
replacing it. A train started in Terminal is detected through the project lock and followed from
`logs/train.log` (iteration, splats, it/s, a stall warning).

**Console** (sidebar) runs any command in `zsh -l` from the repository folder, with live output
and Stop (Ctrl-C to the command).

## A double-clickable app

```
app/scripts/make_app.sh --install
```

builds `app/build/HydrogenSplat.app` (release, ad-hoc signed, icon from `Branding/AppIcon.icns`)
and copies it to `~/Applications`. The .app keeps its own settings (bundle id
`com.sfouasnon.hydrogensplat`); the engine paths default to this checkout.

It builds with **xcodebuild**, not `swift build`, because of the splat viewer (below): full
Xcode 16.3+ (Swift 6.1) and, on Xcode 26, the Metal toolchain
(`xcodebuild -downloadComponent MetalToolchain`). `make_app.sh --spm` still builds with
`swift build`; that app works except for the viewer, which says why it is off. macOS 15+.

## Looking at a model (splat viewer)

The project page lists every `archive/*/` model, the live `train/exports` model and any
`prune/*.ply`; **View** opens one in its own window (two windows = an A/B). The renderer is
[MetalSplatter](https://github.com/scier/MetalSplatter) (MIT), pinned by commit in
`Package.swift`.

- **Stand here** puts the camera exactly at a capture — its pose *and* its pinhole, fitted into
  the window — so cap065 in the viewer is cap065. ← → step captures, L/R picks the eye.
- **Move** scrubs or plays any `move/*.json` through the same path.
- Drag to orbit (it leaves a capture from where it stands), shift- or right-drag to slide,
  scroll or pinch to move in and out, **O** to return to the orbit about the subject.

Poses come from `hs cameras -p P [--archive NAME]` → `viewer/cameras_<NAME|current>.json`
(metres, OpenCV c2w — the `move/*.json` convention). An archive is posed with *its own*
`rig.npz`; the live export warns when `train/dataset/rig.npz` is no longer the one it was
trained on. The command takes no lock, so the viewer works during a train run.

**It is a preview, not `brush-path-render`**: sorting and anti-aliasing differ and there is no
grade. Use it for geometry, floaters and coverage; judge frames from a render.

`swift run` cannot show models: SwiftPM compiles no Metal, and MetalSplatter traps without its
`default.metallib`, so `SplatViewerSupport` checks for it first and disables **View** instead.

Third-party notices: HydrogenSplat ▸ About HydrogenSplat, `HSCore/Acknowledgements.swift`, and
`THIRD_PARTY_LICENSES.md` (copied into the .app).
