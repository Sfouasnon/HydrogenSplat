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
