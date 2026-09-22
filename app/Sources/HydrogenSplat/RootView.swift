import SwiftUI
import HSCore

struct RootView: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var store: ProjectStore
    // a stage run from Terminal changes manifest.json and .hs.lock and tells the app nothing
    private let poll = Timer.publish(every: 3, on: .main, in: .common).autoconnect()

    var body: some View {
        NavigationSplitView {
            List(selection: $model.selection) {
                BrandHeader()
                Section("App") {
                    Label("Setup", systemImage: model.config.problems.isEmpty ? "checkmark.seal" : "exclamationmark.triangle")
                        .tag(SidebarItem.setup)
                    Label("New Project", systemImage: "plus.rectangle.on.folder")
                        .tag(SidebarItem.ingest)
                    Label("Console", systemImage: "terminal")
                        .tag(SidebarItem.console)
                    Label("Event Replay", systemImage: "play.rectangle")
                        .tag(SidebarItem.replay)
                }
                Section {
                    ForEach(store.projects) { p in
                        ProjectRow(project: p, run: model.projectRuns[p.path])
                            .tag(SidebarItem.project(p.path))
                    }
                    if let e = store.lastError {
                        // an unreadable Projects folder used to look like "no projects"
                        Label(e, systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
                            .font(.caption).fixedSize(horizontal: false, vertical: true)
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
            .onReceive(poll) { _ in
                store.refreshIfChanged()
                // also when nothing changed on disk: a Terminal run an app-started one displaced
                model.attachExternalRuns()
            }
        } detail: {
            Group {
                switch model.selection {
                case .setup, .none:
                    SetupView()
                case .ingest:
                    IngestView()
                case .replay:
                    ReplayView()
                case .console:
                    ConsoleView()
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
            // under every page, so a live run is never out of sight
            .safeAreaInset(edge: .bottom, spacing: 0) {
                LiveRunsBar()
            }
            // the window title: "HydrogenSplat — solve · matching 42% · 18 min" while a run is live
            .navigationTitle(model.windowTitle)
        }
    }
}

/// The progress strip: one line per live run, pinned under every page (Setup, Console, the
/// Viewer, every project), so a run is never out of sight — stage · step · done/total · ETA.
/// Click a line to go to its project. Nothing at all when no run is live.
struct LiveRunsBar: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var store: ProjectStore

    var body: some View {
        VStack(spacing: 0) {
            // every session is observed by its own row; a row that is not running draws nothing
            ForEach(model.projectRuns.keys.sorted(), id: \.self) { path in
                if let run = model.projectRuns[path] {
                    LiveRunLine(path: path, name: store.project(at: path)?.displayName ?? (path as NSString).lastPathComponent,
                                run: run) { model.selection = .project(path) }
                }
            }
        }
    }
}

struct LiveRunLine: View {
    let path: String
    let name: String
    @ObservedObject var run: RunSession
    let open: () -> Void
    @State private var now = Date()
    private let clock = Timer.publish(every: 5, on: .main, in: .common).autoconnect()

    var body: some View {
        if run.isRunning {
            let p = run.liveProgress
            VStack(spacing: 0) {
                Divider()
                Button(action: open) {
                    HStack(spacing: 8) {
                        ProgressView().controlSize(.small)
                        Text(name).fontWeight(.semibold).lineLimit(1)
                        Text(p.line(now: now)).monospacedDigit().foregroundStyle(.secondary)
                            .lineLimit(1).truncationMode(.middle)
                        Spacer(minLength: 8)
                        if let f = p.fraction {
                            ProgressView(value: f).frame(width: 120)
                        }
                    }
                    .font(.callout)
                    .padding(.horizontal, 14).padding(.vertical, 6)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help("\(run.title) on \(name) — click to open the project")
            }
            .background(.bar)
            .onReceive(clock) { now = $0 }
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
                    Text(run.liveProgress.short).font(.caption).monospacedDigit().foregroundStyle(.secondary)
                        .lineLimit(1)
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
