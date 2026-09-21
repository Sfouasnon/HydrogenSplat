import SwiftUI
import AppKit
import HSCore

/// Build the per-view subject silhouettes, look at them, and see what they cost.
///
/// The masks are written once and read two ways: the subject model trains on them as they are,
/// the background model reads the same files inverted. What they mean to the loss is a Train
/// setting, not a Masks one — see the Layer control there.
struct MasksView: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var store: ProjectStore
    let project: ProjectSummary
    let manifest: Manifest

    @State private var showAdvanced = false
    @State private var confirmStale = false
    /// Listed on appear and whenever the stage changes, never in body: body runs on every poll.
    @State private var files = MaskFiles(counts: [:], newest: nil)

    private var settings: Binding<MaskSettings> {
        Binding(get: { model.maskSettings[project.path] ?? defaults },
                set: { model.maskSettings[project.path] = $0 })
    }

    /// Without a model there is nothing to project but the sparse cloud.
    private var defaults: MaskSettings {
        var s = MaskSettings()
        if !trainDone { s.source = .points }
        return s
    }

    private var stage: StageState? { manifest.stage("masks") }
    private var queue: RunQueue? { model.maskQueues[project.path] }
    private var running: Bool { queue?.isRunning ?? false }
    private var solveReady: Bool { manifest.stage("solve")?.status == .done }
    /// A model exists to project. Stale counts: building masks marks train stale, and the model is
    /// still on disk — reading "not done" as "no model" flipped the source to SfM points after the
    /// first build, so a second press would have rebuilt from the worse source.
    private var trainDone: Bool {
        let s = manifest.stage("train")?.status
        return s == .done || s == .stale
    }
    private var lockAlive: Bool { project.lock?.alive == true }
    private var blocked: Bool { running || lockAlive || !model.config.problems.isEmpty }
    /// The solve ran without a metric reference: the engine's _m/_mm figures are in its own units.
    private var unscaled: Bool {
        manifest.stage("solve")?.checks.contains { $0.name == "scene_scaled" && !$0.ok } ?? false
    }

    private var reloadKey: String {
        "\(project.path)|\(stage?.finished?.timeIntervalSince1970 ?? 0)|\(stage?.status.rawValue ?? "")"
    }

    /// Masks built before the model they would now be rebuilt from: still valid silhouettes, but
    /// not the tightest ones available. This is the coins case — the set on disk came from an
    /// export that training has since replaced.
    private var olderThanModel: Bool {
        guard let built = files.newest, let trained = manifest.stage("train")?.finished else { return false }
        return built < trained
    }

    /// The preview sheet the stage recorded, if it is still there.
    private var previewSheet: String? {
        guard let a = stage?.artifacts.first(where: { $0.hasSuffix(".jpg") }) else { return nil }
        let p = a.hasPrefix("/") ? a : (project.path as NSString).appendingPathComponent(a)
        return FileManager.default.fileExists(atPath: p) ? p : nil
    }

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 14) {
                if !solveReady {
                    Text("Solve first — the silhouettes are projected with each view's own K, R and t.")
                        .foregroundStyle(.secondary)
                } else {
                    intro
                    handles
                    commandPreview
                    buttons
                    results
                }
            }
            .padding(4)
            .frame(maxWidth: .infinity, alignment: .leading)
        } label: {
            HStack {
                Text("Masks").font(.headline)
                if let s = stage { StatusBadge(status: s.status) }
                if !files.isEmpty {
                    Text(files.summary).font(.caption.monospaced()).foregroundStyle(.secondary)
                }
            }
        }
        .task(id: reloadKey) { files = MaskFiles.read(project: project.path) }
        .confirmationDialog("Rebuild the masks?", isPresented: $confirmStale) {
            Button("Build masks", role: .destructive) { run() }
        } message: {
            Text("The current model was trained against the masks on disk; replacing them marks train, prune, render and views stale.")
        }
    }

    private var intro: some View {
        Text("A silhouette of the subject in every view, projected from geometry the solve already has — no segmentation model. White is what the trainer keeps. The same files serve both layers: the background model reads them inverted.")
            .font(.callout).foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }

    // MARK: handles

    private var handles: some View {
        VStack(alignment: .leading, spacing: 10) {
            Handle(title: "Built from", help: sourceHelp) {
                Picker("", selection: settings.source) {
                    ForEach(MaskSettings.Source.allCases) { Text($0.title).tag($0) }
                }
                .labelsHidden().pickerStyle(.segmented).frame(width: 260)
                .disabled(!trainDone)
            }
            Handle(title: "Size", help: "The radius is fitted to your camera orbit — the subject fills about 70% of the frame at the median camera distance — so it works whether or not the scene has metric scale. Below 1 tightens it; above 1 takes in more around the subject.") {
                Slider(value: settings.radiusScale, in: 0.5...2.0, step: 0.05) { EmptyView() }
                    .frame(width: 220)
                Text(String(format: "×%.2f", settings.wrappedValue.radiusScale))
                    .monospacedDigit().frame(width: 60, alignment: .leading)
            }
            Handle(title: "Margin", help: "The silhouette is dilated outward by this share of the radius. One-sided on purpose: a generous mask costs a little background supervision, a tight one deletes real observations of the subject.") {
                Slider(value: settings.marginFrac, in: 0...0.2, step: 0.01) { EmptyView() }
                    .frame(width: 220)
                Text(String(format: "%.0f%%", settings.wrappedValue.marginFrac * 100))
                    .monospacedDigit().frame(width: 60, alignment: .leading)
            }
            DisclosureGroup("Detail", isExpanded: $showAdvanced) {
                VStack(alignment: .leading, spacing: 10) {
                    Handle(title: "Minimum opacity", help: "Splats fainter than this are not drawn. Raise it when a decayed model leaves haze inside the radius; lower it when the silhouette comes out patchy.") {
                        Slider(value: settings.minOpacity, in: 0...0.6, step: 0.05) { EmptyView() }
                            .frame(width: 220)
                        Text(String(format: "%.2f", settings.wrappedValue.minOpacity))
                            .monospacedDigit().frame(width: 60, alignment: .leading)
                    }
                    Handle(title: "Close gaps", help: "Morphological close, in pixels, to bridge the gaps between projected splats before the interior is filled.") {
                        Stepper(value: settings.closePx, in: 1...81, step: 2) {
                            Text("\(settings.wrappedValue.closePx) px").monospacedDigit()
                        }
                        .frame(width: 160)
                    }
                    Handle(title: "One object", help: "Keep only the largest connected silhouette. Turn it off when the subject really is in separate pieces.") {
                        Toggle("Keep the largest piece only", isOn: settings.keepLargest)
                    }
                    Handle(title: "Preview", help: "How many views go into the preview sheet written beside the masks.") {
                        Stepper(value: settings.previewViews, in: 2...12) {
                            Text("\(settings.wrappedValue.previewViews) views").monospacedDigit()
                        }
                        .frame(width: 160)
                    }
                }
                .padding(.top, 6)
            }
        }
    }

    private var sourceHelp: String {
        if !trainDone {
            return "No model in this project yet, so the sparse SfM points are all there is to project. They are thin and scattered, so the silhouette leans on the close and fill steps — check the preview before you train on it. Once a model exists, rebuild from that instead."
        }
        return settings.wrappedValue.source == .model
            ? "The trained model: the prune output if there is one, else the final export. Dense, so the silhouette needs the least bridging. This is the order that has worked — train once without masks, build the masks from that model, then train the layer."
            : "The sparse SfM points, which need no model at all. The cloud is thin, so the discs are inflated and the close step does the rest; on the coins set this produced silhouettes that were rejected in favour of ones projected from a trained model."
    }

    // MARK: run

    private var commandPreview: some View {
        let args = settings.wrappedValue.arguments(project: project.path)
        let line = (["hs"] + args.map { $0 == project.path ? "$P" : $0 }).map(shellQuote).joined(separator: " ")
        return VStack(alignment: .leading, spacing: 4) {
            Text("Runs (P = this project):").font(.caption).foregroundStyle(.secondary)
            CopyableCommand(text: line)
        }
    }

    private var buttons: some View {
        HStack(spacing: 12) {
            Button(running ? "Building…" : "Build Masks") {
                if trainDone && !files.isEmpty { confirmStale = true } else { run() }
            }
            .disabled(blocked)
            .help("Project the silhouettes into every view and write train/dataset/masks.")
            if !files.isEmpty {
                Button("Reveal in Finder") {
                    let p = (project.path as NSString).appendingPathComponent("train/dataset/masks")
                    NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: p)])
                }
                .disabled(running)
            }
            if lockAlive, let l = project.lock {
                Label("\(l.stage ?? "a stage") is running", systemImage: "lock.fill").foregroundStyle(.secondary)
            }
        }
    }

    private func run() {
        let steps = [RunQueue.Step(title: "Build masks", arguments: settings.wrappedValue.arguments(project: project.path))]
        let q = RunQueue(config: model.config, steps: steps)
        let path = project.path
        q.onStep = { s in model.projectRuns[path] = s }
        q.onFinish = { _ in store.reload() }
        model.maskQueues[path] = q
        q.start()
    }

    // MARK: results

    @ViewBuilder private var results: some View {
        if let s = stage, s.status != .pending, !s.metrics.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                Divider()
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Text("What was built").font(.subheadline.weight(.semibold))
                    if s.status == .stale {
                        Text("the solve was rewritten since — rebuild before training")
                            .font(.caption).foregroundStyle(.orange)
                    }
                }
                if let p = previewSheet {
                    ThumbImage(path: p)
                        .frame(maxWidth: .infinity)
                        .clipShape(RoundedRectangle(cornerRadius: 4))
                    Text("Yellow is the silhouette edge; everything outside it is dimmed.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                MetricGrid(rows: rows(s.metrics, keys: ["source", "views", "points_available", "points_used",
                                                        "radius_source", "radius_scale",
                                                        "coverage_median", "coverage_min", "coverage_max"]))
                if let r = s.metrics["radius_m"]?.double {
                    Text(unscaled
                         ? String(format: "Radius %.4g scene-metres — this solve has no metric scale, so that is its own unit, not a real metre. The masks don't depend on it.", r)
                         : String(format: "Radius %.0f mm.", r * 1000))
                        .font(.caption).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                ForEach(s.checks) { c in
                    CheckRow(name: c.name, ok: c.ok, value: c.value?.display, needsHuman: c.needsHuman)
                }
                if olderThanModel {
                    Label("These were built before the model now in train/exports. They are still valid silhouettes — rebuild only if you want them projected from the current model.",
                          systemImage: "clock.arrow.circlepath")
                        .font(.caption).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if let cov = s.metrics["coverage_median"]?.double {
                    Text(coverageNote(cov))
                        .font(.caption).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    private func coverageNote(_ median: Double) -> String {
        let pct = median * 100
        if pct < 2 {
            return String(format: "The subject covers %.1f%% of the frame — almost nothing is being kept. Raise the radius, or lower the minimum opacity.", pct)
        }
        if pct > 75 {
            return String(format: "The subject covers %.1f%% of the frame, so the mask is keeping most of the picture and buying little. Lower the radius.", pct)
        }
        return String(format: "The subject covers %.1f%% of the frame. The background layer gets the other %.1f%%.", pct, 100 - pct)
    }

    private func rows(_ m: [String: JSONValue], keys: [String]) -> [(String, String)] {
        keys.compactMap { k in m[k].map { (k, $0.display) } }
    }
}
