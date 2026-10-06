import SwiftUI
import HSCore

/// The window works in one project at a time. The project's page fills it (its step rail is the
/// only column on the left); the project is changed from the menu at the left of the toolbar, or
/// from the list of projects (File › Open Project…, ⌘O), which is also what a window with no
/// project shows. A list of every project beside the work was a second left column the pages
/// could not spare.
struct RootView: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var store: ProjectStore
    // a stage run from Terminal changes manifest.json and .hs.lock and tells the app nothing
    private let poll = Timer.publish(every: 3, on: .main, in: .common).autoconnect()

    /// The open project, while something other than its page is in front.
    private var backTarget: ProjectSummary? {
        guard let path = model.openProject, model.selection != .project(path) else { return nil }
        return store.project(at: path)
    }

    var body: some View {
        Group {
            switch model.selection {
            case .open, .none:
                OpenProjectView()
            case .setup:
                SetupView()
            case .ingest:
                FootagePage(project: nil)
            case .replay:
                ReplayView()
            case .console:
                ConsoleView()
            case .project(let path):
                if let p = store.project(at: path) {
                    ProjectDetailView(project: p)
                        .id(path)
                } else {
                    ContentUnavailableView {
                        Label("Project not found", systemImage: "questionmark.folder")
                    } description: {
                        Text(path)
                    } actions: {
                        Button("Show All Projects") { model.selection = .open }
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        // under every page, so a live run is never out of sight
        .safeAreaInset(edge: .bottom, spacing: 0) {
            LiveRunsBar()
        }
        // the window title: "HydrogenSplat — solve · matching 42% · 18 min" while a run is live
        .navigationTitle(model.windowTitle)
        .toolbar {
            // on one of the app's own pages with a project open: the way back to it
            if let p = backTarget {
                ToolbarItem(placement: .navigation) {
                    Button { model.selection = .project(p.path) } label: {
                        Label("Back", systemImage: "chevron.backward").labelStyle(.titleAndIcon)
                    }
                    .help("Back to \(p.displayName)")
                }
            }
            ToolbarItem(placement: .navigation) { ProjectSwitcher() }
        }
        .onReceive(poll) { _ in
            store.refreshIfChanged()
            // also when nothing changed on disk: a Terminal run an app-started one displaced
            model.attachExternalRuns()
        }
    }
}

/// The project the window is in, as a menu: every project (the open one ticked, one with a run
/// going says so), the list of them, New Project, and the app's own pages.
struct ProjectSwitcher: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var store: ProjectStore

    private var current: ProjectSummary? { model.openProject.flatMap { store.project(at: $0) } }

    var body: some View {
        Menu {
            Section("Open") {
                ForEach(store.projects) { p in
                    Toggle(isOn: Binding(get: { model.selection == .project(p.path) },
                                         set: { _ in model.selection = .project(p.path) })) {
                        Text(title(p))
                    }
                }
            }
            Divider()
            Button("All Projects…") { model.selection = .open }
            Button("New Project…") { model.selection = .ingest }
            Divider()
            Button("Console") { model.selection = .console }
            Button("Event Replay") { model.selection = .replay }
            Button("Settings…") { model.selection = .setup }
        } label: {
            Label {
                Text(current?.displayName ?? "No Project").lineLimit(1)
            } icon: {
                Image(systemName: "folder")
            }
            .labelStyle(.titleAndIcon)
        }
        .help("The project this window is working in. Open another one, or start a new one, from here.")
    }

    private func title(_ p: ProjectSummary) -> String {
        let live = model.projectRuns[p.path]?.isRunning == true || p.lock?.alive == true
        return live ? "\(p.displayName) — running" : p.displayName
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

extension RunSession {
    /// A never-started session so rows can always observe something.
    static let placeholder = RunSession(title: "", config: EngineConfig(repoRoot: "/"), arguments: [])
}
