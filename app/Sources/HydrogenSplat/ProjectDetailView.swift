import SwiftUI
import AppKit
import HSCore

/// One project: the rail of six steps and two doors on the left, the selected step's page on the
/// right. Each page states one verdict and holds the one action that moves the project on.
struct ProjectDetailView: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var store: ProjectStore
    let project: ProjectSummary
    /// The engine's checks and metrics for the selected step, beside the page, for a closer look.
    @State private var showDetails = false

    private var page: Binding<ProjectPage> {
        Binding(get: { model.projectPage[project.path] ?? .pipeline },
                set: { model.projectPage[project.path] = $0 })
    }

    var body: some View {
        Group {
            if page.wrappedValue == .viewer, project.manifest != nil {
                ModelViewerPane(scene: model.viewerScene, project: project)
            } else {
                pipeline
            }
        }
        .onAppear { model.syncSubjectKind(project) }
        .onChange(of: project.path) { _, _ in model.syncSubjectKind(project) }
        .onChange(of: model.subjectKind[project.path]) { _, _ in model.syncSubjectKind(project) }
        .toolbar {
            ToolbarItem(placement: .principal) {
                Picker("Page", selection: page) {
                    ForEach(ProjectPage.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .disabled(project.manifest == nil)
                .help("Steps: the project, one step at a time. Viewer: the model and its camera move (⌘1 / ⌘2).")
            }
            ToolbarItemGroup {
                Toggle(isOn: $showDetails) { Label("Details", systemImage: "list.bullet.rectangle") }
                    .help("The engine's checks and metrics for this step")
                Button { store.reload() } label: { Label("Reload", systemImage: "arrow.clockwise") }
                Button {
                    NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: project.path)])
                } label: { Label("Reveal in Finder", systemImage: "folder") }
            }
        }
        .background {
            // ⌘1 / ⌘2 switch pages without reaching for the toolbar
            Group {
                Button("") { page.wrappedValue = .pipeline }.keyboardShortcut("1")
                Button("") { page.wrappedValue = .viewer }.keyboardShortcut("2")
            }
            .opacity(0).frame(width: 0, height: 0).accessibilityHidden(true)
        }
    }

    private func stageBinding(_ m: Manifest) -> Binding<PipelineStage> {
        Binding(get: { model.pipelineStage[project.path] ?? PipelineStage.next(in: m, lock: project.lock) },
                set: { item in model.pipelineStage[project.path] = item })
    }

    private func step(_ d: Int, _ m: Manifest) {
        let all = PipelineStage.steps + PipelineStage.doors
        let cur = stageBinding(m).wrappedValue
        guard let i = all.firstIndex(of: cur) else { return }
        let j = min(max(i + d, 0), all.count - 1)
        if j != i { stageBinding(m).wrappedValue = all[j] }
    }

    /// Labels the rail cannot read off the manifest: a scene has no outlines by choice.
    private var railLabels: [PipelineStage: String] {
        var out: [PipelineStage: String] = [:]
        if let raw = model.subjectKind[project.path], let k = SubjectKind(rawValue: raw), !k.usesMasks,
           project.manifest?.stage("masks") == nil {
            out[.subject] = "everything"
        }
        return out
    }

    private var pipeline: some View {
        VStack(alignment: .leading, spacing: 0) {
            header.padding(.horizontal, 20).padding(.top, 14).padding(.bottom, 8)
            if let run = model.projectRuns[project.path] {
                Observing(run) { r in
                    if r.isRunning || r.finishedAt.map({ Date().timeIntervalSince($0) < 600 }) == true {
                        RunStrip(run: r).padding(.horizontal, 20).padding(.bottom, 8)
                    }
                }
            }
            Divider()
            if let m = project.manifest {
                let sel = stageBinding(m)
                HStack(alignment: .top, spacing: 0) {
                    PipelineRail(manifest: m, lock: project.lock, selection: sel, labels: railLabels)
                        .frame(width: 230)
                    Divider()
                    stepPage(sel.wrappedValue)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                    if showDetails {
                        Divider()
                        ScrollView {
                            inspector(sel.wrappedValue, m).padding(14)
                        }
                        .frame(width: 340)
                    }
                }
                .frame(maxHeight: .infinity)
                .layoutPriority(1)
                .background {
                    // ⌘[ / ⌘] walk the rail
                    Group {
                        Button("") { step(-1, m) }.keyboardShortcut("[", modifiers: .command)
                        Button("") { step(1, m) }.keyboardShortcut("]", modifiers: .command)
                    }
                    .opacity(0).frame(width: 0, height: 0).accessibilityHidden(true)
                }
            } else {
                Text(project.manifestError ?? "no manifest").foregroundStyle(.red).padding(20)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    /// One page per step. Train, Shot and Calibrate are TrainStepPage, ShotStepPage and
    /// CalibrateStepPage (docs/ui-rebuild.md, W2); Settings is the Setup page as it was.
    @ViewBuilder private func stepPage(_ item: PipelineStage) -> some View {
        switch item {
        case .footage:
            FootagePage(project: project)
        case .frames:
            FramesPage(project: project)
        case .look:
            LookPage(project: project)
        case .subject:
            SubjectPage(project: project)
        case .train:
            TrainStepPage(project: project)
        case .shot:
            ShotStepPage(project: project)
        case .calibrate:
            CalibrateStepPage(project: project)
        case .settings:
            SetupView()
        }
    }

    @ViewBuilder private func inspector(_ item: PipelineStage, _ m: Manifest) -> some View {
        let states = item.engineStages.compactMap { m.stage($0) }.filter { $0.status != .pending }
        VStack(alignment: .leading, spacing: 12) {
            Text("CHECKS AND METRICS").font(.caption2.weight(.semibold)).tracking(1.2).foregroundStyle(.secondary)
            if states.isEmpty {
                Text("Nothing has run here yet.").foregroundStyle(.secondary)
            }
            ForEach(states) { st in StageDetail(project: project, stage: st) }
            if item == .footage { stagesBox(m) }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(project.displayName).font(.largeTitle.bold()).textSelection(.enabled)
            Text(project.path).font(.caption.monospaced()).foregroundStyle(.secondary).textSelection(.enabled)
            if let lock = project.lock {
                Label(lock.alive ? "\(lock.stage ?? "a stage") is running (pid \(lock.pid.map(String.init) ?? "?"))"
                                 : "stale lock from pid \(lock.pid.map(String.init) ?? "?") — the next run reclaims it",
                      systemImage: lock.alive ? "lock.fill" : "lock.open")
                    .foregroundStyle(lock.alive ? Color.blue : Color.secondary)
            }
        }
    }

    /// Every engine stage in one table, for the Details column of the Footage step.
    private func stagesBox(_ m: Manifest) -> some View {
        GroupBox("Stages") {
            Table(m.stages, selection: Binding(get: { nil as String? }, set: { name in
                if let n = name, let item = PipelineStage.containing(n) { stageBinding(m).wrappedValue = item }
            })) {
                TableColumn("Stage") { s in Text(s.name).font(.system(.body, design: .monospaced)) }
                    .width(76)
                TableColumn("Status") { s in StatusBadge(status: s.status) }
                    .width(64)
                TableColumn("Took") { s in Text(Format.duration(s.duration)).monospacedDigit() }
                    .width(64)
            }
            .frame(height: CGFloat(m.stages.count) * 24 + 32)
        }
    }
}

struct StatusBadge: View {
    let status: StageStatus

    var body: some View {
        Text(status.rawValue)
            .font(.caption.weight(.semibold))
            .padding(.horizontal, 6).padding(.vertical, 1)
            .background(Capsule().fill(color.opacity(0.18)))
            .foregroundStyle(color)
    }

    private var color: Color {
        switch status {
        case .done: return .green
        case .running: return .blue
        case .failed: return .red
        case .stale: return .orange
        case .pending, .unknown: return .secondary
        }
    }
}

struct StageDetail: View {
    let project: ProjectSummary
    let stage: StageState

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 10) {
                if !stage.argv.isEmpty {
                    CopyableCommand(text: stage.argv.map(shellQuote).joined(separator: " "))
                }
                if let e = stage.error {
                    Label(e, systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(stage.status == .failed ? Color.red : Color.orange)
                        .textSelection(.enabled)
                }
                if !stage.checks.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        ForEach(stage.checks) { c in
                            CheckRow(name: c.name, ok: c.ok, value: c.value?.display, needsHuman: c.needsHuman)
                        }
                    }
                }
                if !stage.metrics.isEmpty {
                    MetricGrid(rows: stage.displayMetrics)
                }
                HStack {
                    if FileManager.default.fileExists(atPath: logPath) {
                        Button("Open log") { NSWorkspace.shared.open(URL(fileURLWithPath: logPath)) }
                    }
                    ForEach(stage.artifacts.prefix(6), id: \.self) { a in
                        Button((a as NSString).lastPathComponent) {
                            let p = a.hasPrefix("/") ? a : (project.path as NSString).appendingPathComponent(a)
                            NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: p)])
                        }
                        .help(a)
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(4)
        } label: {
            HStack {
                Text(stage.name).font(.headline)
                StatusBadge(status: stage.status)
            }
        }
    }

    private var logPath: String {
        (project.path as NSString).appendingPathComponent("logs/\(stage.name).log")
    }
}


/// One line for the latest run: what, how it went, how long. The full panel — progress, checks,
/// events — opens on click, bounded, so a run can never push the rail off the window.
struct RunStrip: View {
    @ObservedObject var run: RunSession
    @State private var open = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button { open.toggle() } label: {
                HStack(spacing: 8) {
                    icon
                    Text(run.title).fontWeight(.semibold)
                    Text(detail).foregroundStyle(.secondary).lineLimit(1).truncationMode(.tail)
                    Spacer(minLength: 8)
                    Text(Format.duration(elapsed)).monospacedDigit().foregroundStyle(.secondary)
                    Image(systemName: open ? "chevron.up" : "chevron.down").font(.caption).foregroundStyle(.secondary)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            if open {
                ScrollView { RunPanel(session: run, showMetrics: false) }.frame(maxHeight: 240)
            }
        }
        .padding(.horizontal, 12).padding(.vertical, 8)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color.secondary.opacity(0.08)))
    }

    @ViewBuilder private var icon: some View {
        if run.isRunning {
            ProgressView().controlSize(.small)
        } else if run.succeeded {
            Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
        } else {
            Image(systemName: "xmark.circle.fill").foregroundStyle(.red)
        }
    }

    private var detail: String {
        if run.isRunning { return run.liveProgress.short }
        if run.succeeded { return run.failedChecks.isEmpty ? "finished" : "finished · \(run.failedChecks.count) check(s) need you" }
        return run.errors.last?.message ?? "failed"
    }

    private var elapsed: Double? {
        guard let s = run.startedAt else { return nil }
        return (run.finishedAt ?? Date()).timeIntervalSince(s)
    }
}
