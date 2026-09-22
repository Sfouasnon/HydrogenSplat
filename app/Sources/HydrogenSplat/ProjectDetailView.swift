import SwiftUI
import AppKit
import HSCore

/// One project: its source, every stage's state, and a stage's metrics / checks / log.
/// The app runs ingest, select (+ solve), exposure, masks and train; the other stages are read from
/// the manifest the CLI writes.
struct ProjectDetailView: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var store: ProjectStore
    let project: ProjectSummary

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
        .toolbar {
            ToolbarItem(placement: .principal) {
                Picker("Page", selection: page) {
                    ForEach(ProjectPage.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .disabled(project.manifest == nil)
                .help("Pipeline: every stage. Viewer: look at a trained model (⌘1 / ⌘2).")
            }
            ToolbarItemGroup {
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
                set: { item in
                    model.pipelineStage[project.path] = item
                    if item.inViewer { page.wrappedValue = .viewer }   // the move panel is unchanged
                })
    }

    private func step(_ d: Int, _ m: Manifest) {
        let all = PipelineStage.allCases
        let cur = stageBinding(m).wrappedValue
        guard let i = all.firstIndex(of: cur) else { return }
        let j = min(max(i + d, 0), all.count - 1)
        if j != i { stageBinding(m).wrappedValue = all[j] }
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
            if let m = project.manifest {
                let sel = stageBinding(m)
                Divider()
                PipelineRail(manifest: m, lock: project.lock, selection: sel)
                Divider()
                HStack(alignment: .top, spacing: 0) {
                    ScrollView {
                        workspace(sel.wrappedValue, m)
                            .padding(20)
                            .frame(maxWidth: 980, alignment: .leading)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    Divider()
                    ScrollView {
                        inspector(sel.wrappedValue, m).padding(14)
                    }
                    .frame(width: 340)
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

    @ViewBuilder private func workspace(_ item: PipelineStage, _ m: Manifest) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            switch item {
            case .source:
                sourceBox(m)
                stagesBox(m)
            case .frames:
                SelectView(project: project, manifest: m)
            case .exposure:
                ExposureView(project: project, manifest: m)
            case .masks:
                MasksView(project: project, manifest: m)
            case .train:
                TrainView(project: project, manifest: m)
                ModelsBox(scene: model.viewerScene, project: project)
            case .grade:
                viewerCard(item)
                GradeView(project: project)
            case .move, .render:
                viewerCard(item)
            }
        }
    }

    private func viewerCard(_ item: PipelineStage) -> some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 10) {
                Text(item == .grade
                     ? "The look is set in the Viewer's move panel, live on the model, and Render bakes it. Below: the baked result, checked on a rendered frame."
                     : "\(item.title) is worked in the Viewer, beside the model: key the camera, set the look, look at the first, middle and last frame, then render.")
                    .fixedSize(horizontal: false, vertical: true)
                Button("Open the Viewer") { page.wrappedValue = .viewer }
                    .keyboardShortcut(.defaultAction)
                Text("⌘2 does the same from anywhere; ⌘1 comes back here.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            .padding(4)
            .frame(maxWidth: .infinity, alignment: .leading)
        } label: { Text(item.title).font(.headline) }
    }

    @ViewBuilder private func inspector(_ item: PipelineStage, _ m: Manifest) -> some View {
        let states = item.engineStages.compactMap { m.stage($0) }.filter { $0.status != .pending }
        VStack(alignment: .leading, spacing: 12) {
            Text("CHECKS AND METRICS").font(.caption2.weight(.semibold)).tracking(1.2).foregroundStyle(.secondary)
            if states.isEmpty {
                Text("Nothing has run here yet.").foregroundStyle(.secondary)
            }
            ForEach(states) { st in StageDetail(project: project, stage: st) }
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

    private func sourceBox(_ m: Manifest) -> some View {
        GroupBox("Source") {
            Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 4) {
                if m.isArray {
                    row("cameras", m.cameras.isEmpty ? "—" : "\(m.cameras.count): " + m.cameras.joined(separator: " "))
                    row("from", m.originalPath ?? "—")
                    row("md5", m.clipMD5 ?? "—")
                    if let w = m.probe["width"]?.int, let h = m.probe["height"]?.int {
                        row("frames", "\(w)×\(h) · " + (m.sourceKind == "mono" ? "\(m.cameras.count) from one camera" : "one per camera"))
                    }
                } else {
                    row("clip", m.clipName ?? "—")
                    row("from", m.originalPath ?? "—")
                    row("md5", m.clipMD5 ?? "—")
                    if let w = m.probe["width"]?.int, let h = m.probe["height"]?.int {
                        row("video", "\(w)×\(h) \(m.probe["codec"]?.string ?? "") · \(m.probe["fps"]?.display ?? "?") fps · \(m.probe["nb_frames"]?.display ?? "?") frames · \(Format.duration(m.probe["duration_s"]?.double))")
                    }
                    row("profile", m.profileID ?? "—")
                }
                row("created", m.created.map { $0.formatted(date: .abbreviated, time: .shortened) } ?? "—")
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(4)
        }
    }

    private func row(_ k: String, _ v: String) -> some View {
        GridRow {
            Text(k).foregroundStyle(.secondary)
            Text(v).font(.system(.body, design: .monospaced)).textSelection(.enabled)
        }
    }

    private func stagesBox(_ m: Manifest) -> some View {
        GroupBox("Stages") {
            Table(m.stages, selection: Binding(get: { nil as String? }, set: { name in
                if let n = name, let item = PipelineStage.containing(n) { stageBinding(m).wrappedValue = item }
            })) {
                TableColumn("Stage") { s in Text(s.name).font(.system(.body, design: .monospaced)) }
                    .width(76)
                TableColumn("Status") { s in StatusBadge(status: s.status) }
                    .width(64)
                TableColumn("Checks") { s in
                    if s.checks.isEmpty {
                        Text("—").foregroundStyle(.secondary)
                    } else if s.failedChecks == 0 {
                        Text("\(s.checks.count) ok").foregroundStyle(.green)
                    } else {
                        Text("\(s.failedChecks) of \(s.checks.count) failed").foregroundStyle(.orange)
                    }
                }
                .width(110)
                TableColumn("Finished") { s in
                    Text(s.finished.map { $0.formatted(.dateTime.month(.abbreviated).day().hour().minute()) } ?? "—")
                        .foregroundStyle(.secondary)
                }
                .width(104)
                TableColumn("Took") { s in Text(Format.duration(s.duration)).monospacedDigit() }
                    .width(64)
                TableColumn("Note") { s in
                    // hs writes `error` on failure; a done stage can also carry a hand-written
                    // note saying its output was invalidated — show those as warnings, not errors
                    Text(s.error ?? "")
                        .foregroundStyle(s.status == .failed ? Color.red : Color.orange)
                        .lineLimit(1)
                        .help(s.error ?? "")
                }
                .width(min: 160, ideal: 300)
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
