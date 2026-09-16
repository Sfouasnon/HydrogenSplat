import SwiftUI
import AppKit
import HSCore

/// One project: its source, every stage's state, and a stage's metrics / checks / log.
/// M1 runs ingest only; the other stages are read from the manifest the CLI writes.
struct ProjectDetailView: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var store: ProjectStore
    let project: ProjectSummary
    @State private var selectedStage: String?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header
                if let run = model.projectRuns[project.path] {
                    Observing(run) { r in
                        if r.isRunning || r.finishedAt.map({ Date().timeIntervalSince($0) < 600 }) == true {
                            GroupBox { RunPanel(session: r, showMetrics: false).padding(4) }
                        }
                    }
                }
                if let m = project.manifest {
                    sourceBox(m)
                    stagesBox(m)
                    GradeView(project: project)
                    if let name = selectedStage ?? m.lastDone, let st = m.stage(name) {
                        StageDetail(project: project, stage: st)
                    }
                } else {
                    Text(project.manifestError ?? "no manifest").foregroundStyle(.red)
                }
            }
            .padding(24)
            .frame(maxWidth: 1100, alignment: .leading)
        }
        .toolbar {
            ToolbarItemGroup {
                Button { store.reload() } label: { Label("Reload", systemImage: "arrow.clockwise") }
                Button {
                    NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: project.path)])
                } label: { Label("Reveal in Finder", systemImage: "folder") }
            }
        }
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
                row("clip", m.clipName ?? "—")
                row("from", m.originalPath ?? "—")
                row("md5", m.clipMD5 ?? "—")
                if let w = m.probe["width"]?.int, let h = m.probe["height"]?.int {
                    row("video", "\(w)×\(h) \(m.probe["codec"]?.string ?? "") · \(m.probe["fps"]?.display ?? "?") fps · \(m.probe["nb_frames"]?.display ?? "?") frames · \(Format.duration(m.probe["duration_s"]?.double))")
                }
                row("profile", m.profileID ?? "—")
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
            Table(m.stages, selection: Binding(get: { selectedStage ?? m.lastDone }, set: { selectedStage = $0 })) {
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
