import SwiftUI
import AppKit
import HSCore

/// Match every training view's exposure (and white balance) to one reference: by default the
/// best-exposed clean pick from hs select's quality report, or a frame chosen in the contact
/// sheet, or the median of all views. Preview measures without writing; Match rewrites the
/// solved training images (originals kept in solve/exposure_backup); Restore puts them back.
struct ExposureView: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var store: ProjectStore
    let project: ProjectSummary
    let manifest: Manifest

    @State private var quality: FrameQuality?
    @State private var confirmStale = false

    private var settings: Binding<ExposureSettings> {
        Binding(get: { model.exposureSettings[project.path] ?? ExposureSettings() },
                set: { model.exposureSettings[project.path] = $0 })
    }
    private var queue: RunQueue? { model.exposureQueues[project.path] }
    private var running: Bool { queue?.isRunning ?? false }
    private var stage: StageState? { manifest.stage("exposure") }
    private var solveReady: Bool {
        let s = manifest.stage("solve")?.status
        return s == .done || s == .stale
    }
    private var trainDone: Bool { manifest.stage("train")?.status == .done }
    private var lockAlive: Bool { project.lock?.alive == true }
    private var blocked: Bool { running || lockAlive || !model.config.problems.isEmpty }
    private var dryRun: [String: JSONValue]? {
        manifest.raw["stages"]?["exposure"]?["dry_run"]?["metrics"]?.object
    }
    private var reloadKey: String {
        "\(project.path)|\(manifest.stage("select")?.finished?.timeIntervalSince1970 ?? 0)"
    }

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 14) {
                if !solveReady {
                    Text("Solve first — exposure rewrites the solved training images.").foregroundStyle(.secondary)
                } else {
                    referencePicker
                    Handle(title: "Correct",
                           help: "Exposure + white balance scales each colour channel onto the reference, so drift in white point goes too. Brightness only leaves colour alone.") {
                        Picker("", selection: settings.whiteBalance) {
                            Text("Exposure + white balance").tag(true)
                            Text("Brightness only").tag(false)
                        }
                        .pickerStyle(.segmented).frame(width: 340)
                    }
                    commandPreview
                    buttons
                    results
                }
            }
            .padding(4)
            .frame(maxWidth: .infinity, alignment: .leading)
        } label: {
            HStack {
                Text("Exposure").font(.headline)
                if let s = stage { StatusBadge(status: s.status) }
            }
        }
        .task(id: reloadKey) { quality = FrameQuality.load(project: project.path) }
        .confirmationDialog("Rewrite the training images?", isPresented: $confirmStale) {
            Button("Match exposure", role: .destructive) { run(dry: false) }
        } message: {
            Text("The current model was trained on the uncorrected images; this marks it stale. The originals stay in solve/exposure_backup and Restore puts them back.")
        }
    }

    // MARK: reference

    private var referencePicker: some View {
        VStack(alignment: .leading, spacing: 8) {
            Handle(title: "Match to",
                   help: helpText) {
                Picker("", selection: settings.reference) {
                    ForEach(ExposureSettings.Reference.allCases) { Text($0.title).tag($0) }
                }
                .pickerStyle(.segmented).frame(width: 420)
            }
            switch settings.wrappedValue.reference {
            case .auto:
                if let r = quality?.exposureReference {
                    referenceCard(cap: r.cap, why: r.why + (r.relaxed == true ? " — nothing was clean, so this is the least-bad pick" : ""))
                } else {
                    Text("No quality report with a suggested frame — run Select Frames, or pick one yourself.")
                        .font(.caption).foregroundStyle(.orange).padding(.leading, 140)
                }
            case .chosen:
                HStack(spacing: 8) {
                    Text("Capture").foregroundStyle(.secondary)
                    TextField("#", value: settings.chosenCapture, format: .number)
                        .textFieldStyle(.roundedBorder).frame(width: 70)
                    if let s = quality?.exposureReference {
                        Button("Use suggested (#\(s.sel))") { settings.wrappedValue.chosenCapture = s.sel }
                            .controlSize(.small)
                    }
                    Text("or right-click a frame in the contact sheet → Use as exposure reference")
                        .font(.caption).foregroundStyle(.secondary)
                }
                .padding(.leading, 140)
                if let n = settings.wrappedValue.chosenCapture {
                    referenceCard(cap: String(format: "cap%03d", n), why: "chosen by hand")
                }
            case .median:
                EmptyView()
            }
        }
    }

    private var helpText: String {
        switch settings.wrappedValue.reference {
        case .auto: return "The clean pick (no exposure, focus, eye or clipping flags) within a third of a stop of the set's median that clips least. Matching to it mostly darkens, which loses nothing; matching to a hotter frame would push the darker views' highlights into clipping."
        case .chosen: return "Every view, both eyes, is scaled onto this capture's left eye. It keeps its own exposure and white point exactly."
        case .median: return "No single frame: every view is scaled onto the median of all views, so the correction is centred."
        }
    }

    @ViewBuilder private func referenceCard(cap: String, why: String) -> some View {
        let f = quality?.frame(cap: cap)
        HStack(alignment: .top, spacing: 12) {
            if let t = f?.thumb {
                ThumbImage(path: (project.path as NSString).appendingPathComponent("select/" + t))
                    .frame(width: 200)
                    .clipShape(RoundedRectangle(cornerRadius: 4))
            }
            VStack(alignment: .leading, spacing: 4) {
                Text(cap + (f.map { " · pick #\($0.sel) · source frame \($0.frame)" } ?? ""))
                    .font(.callout.weight(.semibold))
                if let f = f {
                    Text([f.ev.map { String(format: "%+.2f EV off median", $0) },
                          f.clip.map { String(format: "%.1f%% clipped (L)", $0 * 100) },
                          f.clipR.map { String(format: "%.1f%% (R)", $0 * 100) },
                          f.focusRel.map { String(format: "focus %.2f×", $0) }]
                        .compactMap { $0 }.joined(separator: " · "))
                        .font(.caption).monospacedDigit()
                    if !f.flags.isEmpty {
                        Text("flags: " + f.flags.joined(separator: ", ")).font(.caption).foregroundStyle(.orange)
                    }
                } else if quality != nil {
                    Text("not in the quality report").font(.caption).foregroundStyle(.orange)
                }
                Text(why).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(.leading, 140)
    }

    // MARK: run

    private var commandPreview: some View {
        let args = settings.wrappedValue.arguments(project: project.path)
        let line = (["hs"] + args.map { $0.hasPrefix(project.path) ? "$P" : $0 }).map(shellQuote).joined(separator: " ")
        return VStack(alignment: .leading, spacing: 4) {
            Text("Runs (P = this project; Preview adds --dry-run):").font(.caption).foregroundStyle(.secondary)
            CopyableCommand(text: line)
        }
    }

    private var buttons: some View {
        HStack(spacing: 12) {
            let problem = settings.wrappedValue.problem
            Button(running && queue?.steps.first?.title == "Exposure preview" ? "Measuring…" : "Preview") { run(dry: true) }
                .disabled(blocked || problem != nil)
                .help("Measure every view and report the gains; writes nothing.")
            Button(running && queue?.steps.first?.title == "Match exposure" ? "Matching…" : "Match Exposure") {
                if trainDone { confirmStale = true } else { run(dry: false) }
            }
            .keyboardShortcut("e", modifiers: [.command, .shift])
            .disabled(blocked || problem != nil)
            if stage?.status == .done && stage?.metrics["restored"] == nil {
                Button("Restore originals") { restore() }
                    .disabled(blocked)
                    .help("Put the uncorrected training images back from solve/exposure_backup.")
            }
            if let p = problem {
                Label(p, systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
            } else if lockAlive, let l = project.lock {
                Label("\(l.stage ?? "a stage") is running", systemImage: "lock.fill").foregroundStyle(.secondary)
            }
        }
    }

    private func run(dry: Bool) {
        let args = settings.wrappedValue.arguments(project: project.path, dryRun: dry)
        start([RunQueue.Step(title: dry ? "Exposure preview" : "Match exposure", arguments: args)])
    }

    private func restore() {
        start([RunQueue.Step(title: "Restore exposure", arguments: ExposureSettings.restoreArguments(project: project.path))])
    }

    private func start(_ steps: [RunQueue.Step]) {
        let q = RunQueue(config: model.config, steps: steps)
        let path = project.path
        q.onStep = { s in model.projectRuns[path] = s }
        q.onFinish = { _ in store.reload() }
        model.exposureQueues[path] = q
        q.start()
    }

    // MARK: results

    @ViewBuilder private var results: some View {
        if let s = stage, s.status == .done || s.status == .stale || s.status == .failed {
            VStack(alignment: .leading, spacing: 6) {
                Text(s.metrics["restored"] != nil ? "Originals restored" : "Applied").font(.subheadline.weight(.semibold))
                if s.metrics["restored"] == nil {
                    MetricGrid(rows: rows(s.metrics, keys: ["reference", "reference_why", "views", "luma_spread_before",
                                                             "luma_spread_after", "gain_min", "gain_max",
                                                             "clipped_fraction_max"]))
                    ForEach(s.checks) { c in CheckRow(name: c.name, ok: c.ok, value: c.value?.display) }
                }
            }
        }
        if let d = dryRun {
            VStack(alignment: .leading, spacing: 6) {
                Text("Last preview (nothing written)").font(.subheadline.weight(.semibold))
                MetricGrid(rows: rows(d, keys: ["reference", "reference_why", "views", "luma_spread_before",
                                                 "gain_min", "gain_max", "clipped_fraction_max", "measured"]))
            }
        }
    }

    private func rows(_ m: [String: JSONValue], keys: [String]) -> [(String, String)] {
        keys.compactMap { k in m[k].map { (k, $0.display) } }
    }
}
