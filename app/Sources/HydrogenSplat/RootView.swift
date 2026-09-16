import SwiftUI
import HSCore

struct RootView: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var store: ProjectStore

    var body: some View {
        NavigationSplitView {
            List(selection: $model.selection) {
                Section("App") {
                    Label("Setup", systemImage: model.config.problems.isEmpty ? "checkmark.seal" : "exclamationmark.triangle")
                        .tag(SidebarItem.setup)
                    Label("New Project", systemImage: "plus.rectangle.on.folder")
                        .tag(SidebarItem.ingest)
                    Label("Event Replay", systemImage: "play.rectangle")
                        .tag(SidebarItem.replay)
                }
                Section {
                    ForEach(store.projects) { p in
                        ProjectRow(project: p, run: model.projectRuns[p.path])
                            .tag(SidebarItem.project(p.path))
                    }
                } header: {
                    HStack {
                        Text("Projects")
                        Spacer()
                        Button { store.reload() } label: { Image(systemName: "arrow.clockwise") }
                            .buttonStyle(.borderless)
                            .help("Reload projects")
                    }
                }
            }
            .navigationSplitViewColumnWidth(min: 230, ideal: 270)
        } detail: {
            switch model.selection {
            case .setup, .none:
                SetupView()
            case .ingest:
                IngestView()
            case .replay:
                ReplayView()
            case .project(let path):
                if let p = store.project(at: path) {
                    ProjectDetailView(project: p)
                        .id(path)
                } else {
                    ContentUnavailableView("Project not found", systemImage: "questionmark.folder",
                                           description: Text(path))
                }
            }
        }
    }
}

struct ProjectRow: View {
    let project: ProjectSummary
    @ObservedObject var run: RunSession

    init(project: ProjectSummary, run: RunSession?) {
        self.project = project
        self._run = ObservedObject(wrappedValue: run ?? RunSession.placeholder)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(project.displayName).lineLimit(1)
            HStack(spacing: 4) {
                if run.isRunning {
                    ProgressView().controlSize(.mini)
                    Text(run.currentStep ?? "running").font(.caption).foregroundStyle(.secondary)
                } else if let lock = project.lock, lock.alive {
                    Image(systemName: "lock.fill").font(.caption2)
                    Text("\(lock.stage ?? "?") running (pid \(lock.pid.map(String.init) ?? "?"))")
                        .font(.caption).foregroundStyle(.secondary)
                } else if let m = project.manifest {
                    Text(m.lastDone.map { "through \($0)" } ?? "nothing done yet")
                        .font(.caption).foregroundStyle(.secondary)
                } else {
                    Text(project.manifestError ?? "?").font(.caption).foregroundStyle(.red)
                }
            }
        }
        .padding(.vertical, 2)
    }
}

extension RunSession {
    /// A never-started session so rows can always observe something.
    static let placeholder = RunSession(title: "", config: EngineConfig(repoRoot: "/"), arguments: [])
}
