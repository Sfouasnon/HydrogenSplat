import SwiftUI
import Charts
import HSCore

/// Train a model from the app: handles with plain explanations, the exact command, a live
/// growth chart, and train → archive → score views as one queue. A train started in Terminal
/// (the project lock is held by another process) is followed from its log instead.
struct TrainView: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var store: ProjectStore
    let project: ProjectSummary
    let manifest: Manifest

    @State private var showAdvanced = false
    @State private var confirmReplace = false
    @State private var lock: ProjectSummary.LockInfo?
    @State private var onBattery = PowerStatus.onBattery
    private let tick = Timer.publish(every: 3, on: .main, in: .common).autoconnect()

    private var settings: Binding<TrainSettings> {
        Binding(get: { model.trainSettings[project.path] ?? defaults },
                set: { model.trainSettings[project.path] = $0 })
    }

    private var defaults: TrainSettings {
        var s = TrainSettings()
        // score with the crop size the last views run on this project used
        if let mm = manifest.stage("views")?.metrics["subject_extent_mm"]?.double { s.subjectMM = mm }
        if !hasMasks { s.useMasks = false }
        return s
    }

    private var captureSet: CaptureSet { CaptureSet.read(project: project.path) }
    private var captures: Int { captureSet.count }

    private var hasMasks: Bool {
        FileManager.default.fileExists(atPath: (project.path as NSString).appendingPathComponent("train/dataset/masks"))
    }

    private var solveDone: Bool { manifest.stage("solve")?.status == .done }
    private var queue: RunQueue? { model.trainQueues[project.path] }
    /// The lock as either source sees it: this view polls it every 3 s, and the store carries the
    /// value the sidebar shows. Taking whichever is alive means a Terminal run is never missed
    /// because one of the two had not refreshed yet — missing it offers a Start button that the
    /// engine would only refuse.
    private var activeLock: ProjectSummary.LockInfo? {
        [lock, project.lock].compactMap { $0 }.first { $0.alive } ?? lock ?? project.lock
    }
    private var externalTrain: Bool {
        guard let l = activeLock, l.alive else { return false }
        return !(queue?.isRunning ?? false)
    }

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 14) {
                if !solveDone {
                    Text("Solve this project first — training needs the solved dataset.").foregroundStyle(.secondary)
                } else if externalTrain, let l = activeLock {
                    ExternalTrainView(project: project, manifest: manifest, lock: l)
                } else {
                    currentModel
                    handles
                    commandPreview
                    startRow
                    if let q = queue {
                        QueueView(queue: q, totalIters: settings.wrappedValue.totalIters)
                    }
                }
            }
            .padding(4)
            .frame(maxWidth: .infinity, alignment: .leading)
        } label: {
            HStack {
                Text("Train").font(.headline)
                if let s = manifest.stage("train") { StatusBadge(status: s.status) }
            }
        }
        .onAppear(perform: refresh)
        .onReceive(tick) { _ in refresh() }
        .confirmationDialog("Train without keeping the result?", isPresented: $confirmReplace) {
            Button("Train without keeping it", role: .destructive) { start() }
        } message: {
            Text((currentMD5 != nil && currentArchive == nil
                  ? "The current export is not in any archive and training deletes train/exports. "
                  : "")
                 + "No archive name is set, so the model this run makes will be replaced by the next run. Type a name in \"Keep it as\" to keep it.")
        }
    }

    private func refresh() {
        lock = ProjectStore.readLock(project.path)
        onBattery = PowerStatus.onBattery
    }

    // MARK: current model

    private var archives: [ArchiveInfo] { ArchiveInfo.list(project: project.path) }
    private var currentMD5: String? { manifest.stage("train")?.metrics["final_export_md5"]?.string }
    private var currentArchive: ArchiveInfo? {
        guard let md5 = currentMD5 else { return nil }
        return archives.first { $0.plyMD5 == md5 }
    }

    @ViewBuilder private var currentModel: some View {
        if let t = manifest.stage("train"), t.status == .done {
            let m = t.metrics
            HStack(alignment: .firstTextBaseline, spacing: 16) {
                Label("Current model", systemImage: "cube.transparent").font(.subheadline.weight(.semibold))
                Text("\(m["final_splats"]?.display ?? "?") splats").monospacedDigit()
                if let e = m["elapsed_s"]?.double { Text("trained in \(Format.duration(e))").foregroundStyle(.secondary) }
                if let c = m["brush_config"]?["commit"]?.string { Text("brush \(c)").font(.caption.monospaced()).foregroundStyle(.secondary) }
                Spacer()
                if let a = currentArchive {
                    Label("kept as \(a.name)", systemImage: "archivebox").foregroundStyle(.green)
                } else if currentMD5 != nil {
                    Label("not archived — training replaces it", systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.orange)
                }
            }
            .font(.callout)
        }
    }

    // MARK: handles

    private var handles: some View {
        VStack(alignment: .leading, spacing: 10) {
            Handle(title: "Length", help: "Total iterations. The rig6 schedule is 40,000. Time grows with the splat count: the head clip took about 86 minutes on the current Brush.") {
                Stepper(value: settings.totalIters, in: 1_000...100_000, step: 5_000) {
                    Text("\(settings.wrappedValue.totalIters) iterations").monospacedDigit()
                }
            }
            Handle(title: "Stop growing at", help: "After this iteration no new splats are added; the rest only refines. Brush's default is 15,000. On the 09-15 clips about 70% of the final splats already existed by 15,000, and growth was still climbing when it stopped at 30,000.") {
                Picker("", selection: settings.growthStopIter) {
                    Text("15,000 (Brush default)").tag(15_000)
                    Text("20,000").tag(20_000)
                    Text("30,000 (rig6)").tag(30_000)
                }
                .labelsHidden().frame(width: 200)
            }
            Handle(title: "Hold out", help: "Leaves every Nth \(captureSet.noun)\(captureSet.stereo ? " (both eyes)" : "") out of training, so the view score measures the model on photographs it never saw. Use the same setting for every run you want to compare.") {
                Picker("", selection: settings.holdoutEvery) {
                    Text("Nothing").tag(0)
                    Text("Every 10th \(captureSet.noun)").tag(10)
                    Text("Every 5th \(captureSet.noun)").tag(5)
                }
                .labelsHidden().frame(width: 200)
                let held = settings.wrappedValue.holdoutCaptures(total: captures)
                if !held.isEmpty {
                    Text("\(held.count) of \(captures) \(captureSet.noun)s · \(captureSet.views - held.count * (captureSet.stereo ? 2 : 1)) views train")
                        .foregroundStyle(.secondary).monospacedDigit()
                }
            }
            Handle(title: "Masks", help: hasMasks
                   ? "This project has masks. With masks on, Brush ignores everything outside the subject: a cleaner subject, but no usable room."
                   : "No masks in this project (run hs masks to make them).") {
                Toggle("Train with masks", isOn: settings.useMasks).disabled(!hasMasks)
            }
            DisclosureGroup("Detail and experiments", isExpanded: $showAdvanced) {
                VStack(alignment: .leading, spacing: 10) {
                    Handle(title: "Refine every", help: "How many iterations between split / prune steps. 130 is the rig6 value; Brush's own default is 200. Larger means fewer, calmer growth steps.") {
                        Stepper(value: settings.refineEvery, in: 50...500, step: 10) {
                            Text("\(settings.wrappedValue.refineEvery)").monospacedDigit()
                        }
                    }
                    Handle(title: "Split big splats", help: "Splats that cover more than this fraction of the image get split. Lower values make more, smaller splats — the lever for soft hair. Brush default 0.5.") {
                        OptionalNumber(value: settings.splitAtScreenSize, placeholder: "0.5 default", choices: [0.25, 0.1])
                    }
                    Handle(title: "Anti-aliasing", help: "Minimum splat width, about √s pixels (0.1 → 0.32 px). Reduces shimmer on fine background detail; may soften the finest edges. 0 turns it off, which is how the 09-15 models were trained.") {
                        Picker("", selection: settings.minScaleFactor) {
                            Text("Off (0)").tag(0.0)
                            Text("Light (0.03)").tag(0.03)
                            Text("Default (0.1)").tag(0.1)
                            Text("Strong (0.3)").tag(0.3)
                        }
                        .labelsHidden().frame(width: 200)
                    }
                    Handle(title: "Background noise", help: "Random noise on the background colour each step pushes splats to cover every pixel. Brush default 0.1. A room capture always has a real background, so try 0 when the far background ghosts.") {
                        OptionalNumber(value: settings.backgroundNoise, placeholder: "0.1 default", choices: [0])
                    }
                    Handle(title: "Leave out views", help: "Extra views to exclude, e.g. a bad photograph: L/cap064,R/cap069.") {
                        TextField("L/cap064,R/cap069", text: settings.excludeExtra).textFieldStyle(.roundedBorder).frame(maxWidth: 320)
                    }
                    Handle(title: "Brush arguments", help: "Passed to Brush unchanged, for flags the app doesn't have a control for.") {
                        TextField("--opac-decay 0.006", text: settings.extraBrushArgs).textFieldStyle(.roundedBorder).frame(maxWidth: 320)
                    }
                }
                .padding(.top, 6)
            }
            Divider()
            Handle(title: "Keep it as", help: "Copies the finished model and its coordinate frame to archive/<name>, so the next run doesn't replace it. Leave empty to skip.") {
                TextField("holdout-base", text: settings.archiveName).textFieldStyle(.roundedBorder).frame(width: 200)
                if let p = settings.wrappedValue.archiveNameProblem {
                    Text(p).foregroundStyle(.red)
                } else if archives.contains(where: { $0.name == settings.wrappedValue.archiveName }) {
                    Text("already exists — pick a new name").foregroundStyle(.red)
                } else if settings.wrappedValue.archiveName.trimmingCharacters(in: .whitespaces).isEmpty {
                    Text("empty — the result won't be kept; the next run replaces it").foregroundStyle(.orange)
                }
            }
            Handle(title: "Then score", help: "Renders the model from the held-out (or chosen) capture poses and compares with the photographs. The crop size must match between runs you compare: 350 mm head, 2000 mm body.") {
                Toggle("Score views", isOn: settings.scoreViews)
                Toggle("both eyes", isOn: settings.scoreBothEyes)
                    .disabled(!settings.wrappedValue.scoreViews || !captureSet.stereo)
                    .help(captureSet.stereo ? "Render and grade the right eye too: only eL - eR is a disparity."
                                            : "This project has one view per camera — there is no second eye.")
                OptionalNumber(value: settings.subjectMM, placeholder: "crop mm (auto)", choices: [350, 2000])
                    .disabled(!settings.wrappedValue.scoreViews)
            }
        }
    }

    private var steps: [RunQueue.Step] {
        settings.wrappedValue.steps(project: project.path, in: captureSet)
    }

    private var commandPreview: some View {
        let lines = steps.map { (["hs"] + $0.arguments.map { $0.hasPrefix(project.path) ? "$P" : $0 })
            .map(shellQuote).joined(separator: " ") }
        return VStack(alignment: .leading, spacing: 4) {
            Text("Runs, in order (P = this project):").font(.caption).foregroundStyle(.secondary)
            CopyableCommand(text: lines.joined(separator: " && \\\n  "))
        }
    }

    private var nameBlocked: Bool {
        let s = settings.wrappedValue
        return s.archiveNameProblem != nil || archives.contains { $0.name == s.archiveName }
    }

    private var startRow: some View {
        HStack(spacing: 12) {
            let running = queue?.isRunning ?? false
            Button(running ? "Training…" : "Start training") {
                // Ask whenever nothing will be archived: the queue is built now, so a name typed
                // later is ignored, and the model this run makes is replaced by the next one.
                if settings.wrappedValue.archiveName.trimmingCharacters(in: .whitespaces).isEmpty {
                    confirmReplace = true
                } else {
                    start()
                }
            }
            .keyboardShortcut(.defaultAction)
            .disabled(running || nameBlocked || captures == 0 || !model.config.problems.isEmpty
                      || activeLock?.alive == true)
            if onBattery {
                Label("On battery — macOS will sleep and stall training. Plug in.", systemImage: "battery.25")
                    .foregroundStyle(.orange)
            } else {
                Label("Plugged in. Keep the lid open.", systemImage: "powerplug").foregroundStyle(.secondary)
            }
        }
    }

    private func start() {
        let q = RunQueue(config: model.config, steps: steps)
        let path = project.path
        q.onStep = { s in model.projectRuns[path] = s }
        q.onFinish = { _ in store.reload() }
        model.trainQueues[path] = q
        q.start()
    }
}

/// A label, its control(s), and a one-line explanation under them.
struct Handle<Content: View>: View {
    let title: String
    let help: String
    @ViewBuilder let content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 10) {
                Text(title).frame(width: 130, alignment: .leading)
                content()
            }
            Text(help)
                .font(.caption).foregroundStyle(.secondary)
                .padding(.leading, 140)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

/// "Default" or a number, with a few quick choices.
struct OptionalNumber: View {
    @Binding var value: Double?
    let placeholder: String
    let choices: [Double]

    var body: some View {
        HStack(spacing: 6) {
            TextField(placeholder, value: $value, format: .number)
                .textFieldStyle(.roundedBorder).frame(width: 120)
            ForEach(choices, id: \.self) { c in
                Button(c == c.rounded() ? String(Int(c)) : String(c)) { value = c }.controlSize(.small)
            }
            if value != nil {
                Button("Default") { value = nil }.controlSize(.small)
            }
        }
    }
}

// MARK: - the running queue

struct QueueView: View {
    @ObservedObject var queue: RunQueue
    let totalIters: Int

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                ForEach(Array(queue.steps.enumerated()), id: \.offset) { i, s in
                    StepChip(title: s.title, state: chipState(i))
                }
                Spacer()
                if queue.isRunning {
                    Button("Stop") { queue.cancel() }
                        .help("Interrupts the current step; the stage is marked failed and the project unlocked")
                }
            }
            if let first = queue.sessions.first {
                GrowthChart(session: first, totalIters: totalIters)
            }
            if let cur = queue.current {
                RunPanel(session: cur, showMetrics: queue.index > 0)
            }
        }
    }

    private func chipState(_ i: Int) -> StepChip.State {
        if i < queue.index { return .done }
        if i > queue.index { return queue.state == .running ? .waiting : .skipped }
        switch queue.state {
        case .running: return .running
        case .finished: return .done
        case .failed, .cancelled: return .failed
        case .idle: return .waiting
        }
    }
}

struct StepChip: View {
    enum State { case waiting, running, done, failed, skipped }
    let title: String
    let state: State

    var body: some View {
        HStack(spacing: 5) {
            switch state {
            case .waiting: Image(systemName: "circle").foregroundStyle(.secondary)
            case .running: ProgressView().controlSize(.mini)
            case .done: Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
            case .failed: Image(systemName: "xmark.circle.fill").foregroundStyle(.red)
            case .skipped: Image(systemName: "minus.circle").foregroundStyle(.secondary)
            }
            Text(title).font(.caption)
        }
        .padding(.horizontal, 8).padding(.vertical, 3)
        .background(Capsule().fill(Color.secondary.opacity(0.12)))
    }
}

/// Splats against iterations, live from the train step's progress events.
struct GrowthChart: View {
    @ObservedObject var session: RunSession
    let totalIters: Int

    var body: some View {
        let pts = session.samples.filter { $0.value != nil }
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Splats").font(.caption.weight(.semibold))
                Spacer()
                if let last = pts.last {
                    Text("\(JSONValue.number(last.value ?? 0).display) at \(JSONValue.number(last.done).display)")
                        .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                }
            }
            Chart(pts, id: \.self) { p in
                AreaMark(x: .value("Iteration", p.done), y: .value("Splats", p.value ?? 0))
                    .foregroundStyle(Brand.tally.opacity(0.12))
                LineMark(x: .value("Iteration", p.done), y: .value("Splats", p.value ?? 0))
                    .foregroundStyle(Brand.tally)
            }
            .chartXScale(domain: 0...Double(max(totalIters, 1)))
            .chartYAxis { AxisMarks(format: FloatingPointFormatStyle<Double>.number.notation(.compactName)) }
            .frame(height: 140)
        }
    }
}

// MARK: - a train started from Terminal

struct ExternalTrainView: View {
    let project: ProjectSummary
    let manifest: Manifest
    let lock: ProjectSummary.LockInfo
    @State private var status: TrainLogStatus?
    @State private var showLog = false
    private let tick = Timer.publish(every: 2, on: .main, in: .common).autoconnect()

    private var total: Int {
        // brush_argv is what Brush actually got; hs's own argv only has the flag when it was overridden
        let brush = manifest.raw["stages"]?["train"]?["brush_argv"]?.array?.compactMap { $0.string } ?? []
        for argv in [brush, manifest.stage("train")?.argv ?? []] {
            if let i = argv.firstIndex(of: "--total-train-iters"), i + 1 < argv.count, let n = Int(argv[i + 1]) { return n }
        }
        return TrainSettings().totalIters
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("Training in another window (pid \(lock.pid.map(String.init) ?? "?")). Following its log; stop it from where it was started.",
                  systemImage: "terminal")
                .foregroundStyle(.secondary)
            if let s = status {
                let frac = min(Double(s.iter) / Double(max(total, 1)), 1)
                ProgressView(value: frac)
                HStack(spacing: 16) {
                    Text("\(s.iter) / \(total)").monospacedDigit()
                    Text("\(JSONValue.number(Double(s.splats)).display) splats").monospacedDigit()
                    if let r = s.rate { Text(String(format: "%.2f it/s", r)).monospacedDigit() }
                    let quiet = s.lastLineAt.map { Date().timeIntervalSince($0) } ?? 0
                    if quiet <= TrainLogStatus.sleepGap, let eta = s.etaSeconds(total: total) {
                        Text("ETA \(Format.eta(eta))").monospacedDigit()
                            .help("Brush iterations left at the recent awake rate. Splat growth slows it, so this runs a little short; archive and view scoring follow.")
                    }
                    if s.lastLineAt != nil {
                        Text(quiet > TrainLogStatus.sleepGap ? "no progress for \(Format.duration(quiet)) — asleep or finishing?" : "updated \(Format.duration(quiet)) ago")
                            .foregroundStyle(quiet > TrainLogStatus.sleepGap ? Color.orange : Color.secondary)
                    }
                    Spacer()
                    if let st = s.runStarted { Text("started \(st)").foregroundStyle(.secondary) }
                }
                .font(.callout)
                DisclosureGroup("Log", isExpanded: $showLog) {
                    ScrollView {
                        Text(s.tail.joined(separator: "\n"))
                            .font(.system(.caption, design: .monospaced))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
                    }
                    .frame(height: 200)
                }
            } else {
                Text("No train log yet.").foregroundStyle(.secondary)
            }
        }
        .onAppear(perform: read)
        .onReceive(tick) { _ in read() }
    }

    private func read() {
        let p = (project.path as NSString).appendingPathComponent("logs/train.log")
        DispatchQueue.global(qos: .utility).async {
            let s = TrainLogStatus.read(path: p)
            DispatchQueue.main.async { status = s }
        }
    }
}
