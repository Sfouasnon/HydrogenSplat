# HydrogenSplat.app (M1)

SwiftUI orchestrator over the `hs` engine (strategy §2): it never links Python; every stage is
`hs <stage> …` as a child process and the UI is built from its JSON-lines events.

```
cd app && swift build && swift run HydrogenSplat     # or: open Package.swift (Xcode) and Run
cd app && swift test
```

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
