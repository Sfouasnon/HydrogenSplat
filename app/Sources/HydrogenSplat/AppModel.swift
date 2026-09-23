import SwiftUI
import AppKit
import Combine
import HSCore

enum ProjectPage: String, CaseIterable, Identifiable {
    case pipeline = "Pipeline"
    case viewer = "Viewer"
    var id: String { rawValue }
}

enum SidebarItem: Hashable {
    case setup
    case ingest
    case replay
    case console
    case project(String)
}

/// App-wide state: engine config (persisted), the project store, and the runs in flight.
@MainActor
final class AppModel: ObservableObject {
    @Published var config: EngineConfig {
        didSet {
            save()
            if store.root != config.projectsRoot { store.root = config.projectsRoot }
        }
    }
    @Published var selection: SidebarItem? = .setup {
        didSet {
            if case .project(let p)? = selection { attachExternalRun(path: p, lock: ProjectStore.readLock(p)) }
            updateLiveStatus()
        }
    }
    /// The latest run per project path: the app's own, or one attached to a run started in
    /// Terminal (`attachExternalRuns`).
    @Published var projectRuns: [String: RunSession] = [:] {
        didSet {
            // an app-started run for the same project replaces an attached one: stop its tail
            for (path, old) in oldValue where old.isAttached && projectRuns[path] !== old { old.detach() }
            watchRuns()
        }
    }
    /// "solve · matching 42% · 18 min" for the window title while a run is live, nil otherwise.
    /// Published only when the text changes (every percent, every minute of ETA), not on every
    /// progress event: every view holding the model re-renders when it does.
    @Published private(set) var liveTitle: String?
    private var runWatch: AnyCancellable?
    private var projectsWatch: AnyCancellable?
    /// "path|pid" of lock holders whose events file shows nothing to follow (its last run is
    /// already done — e.g. a lock stage that logs under another name), so they are not re-read.
    private var unfollowable: Set<String> = []
    /// The Solve matcher per project path (Auto unless changed), kept for the session.
    @Published var solveMatcher: [String: SolveMatcher] = [:]
    /// The train → archive → views queue per project path; survives leaving the page.
    @Published var trainQueues: [String: RunQueue] = [:]
    /// Train panel settings per project path, kept for the session.
    @Published var trainSettings: [String: TrainSettings] = [:]
    /// The select / solve run per project path, started from the Select frames panel.
    @Published var selectQueues: [String: RunQueue] = [:]
    /// Select panel settings per project path, kept for the session.
    @Published var selectSettings: [String: SelectSettings] = [:]
    /// The exposure run per project path, and its panel settings.
    @Published var exposureQueues: [String: RunQueue] = [:]
    @Published var exposureSettings: [String: ExposureSettings] = [:]
    /// The masks run per project path, and its panel settings.
    @Published var maskQueues: [String: RunQueue] = [:]
    @Published var maskSettings: [String: MaskSettings] = [:]
    /// The scale run (LiDAR scan measured / applied, or a factor) per project path, and its settings.
    @Published var scaleQueues: [String: RunQueue] = [:]
    @Published var scaleSettings: [String: ScaleSettings] = [:]
    @Published var consoleHistory: [ConsoleSession] = []
    /// Pipeline or Viewer, per project path.
    @Published var projectPage: [String: ProjectPage] = [:]
    /// The pipeline rail's selected stage, per project path. Unset means "the next thing to do".
    @Published var pipelineStage: [String: PipelineStage] = [:]
    /// The model chosen in the Viewer page, per project path.
    @Published var viewerFile: [String: ViewerModelFile] = [:]
    /// The in-window viewer's scene: one model held at a time, kept loaded while the page is
    /// switched away so coming back is instant. Separate model windows own their own scenes.
    let viewerScene = SplatScene()
    /// The move panel's build / render run per project path, and the script it has open.
    @Published var moveQueues: [String: RunQueue] = [:]
    @Published var moveSelection: [String: String] = [:]
    /// The keyframe editor per project path (it outlives leaving the Viewer page).
    private var moveEditors: [String: MoveEditor] = [:]

    func moveEditor(project: String) -> MoveEditor {
        if let e = moveEditors[project] { return e }
        let e = MoveEditor(project: project, scene: viewerScene)
        moveEditors[project] = e
        return e
    }

    let store: ProjectStore
    private static let key = "engineConfig.v1"

    init() {
        let cfg: EngineConfig
        if let data = UserDefaults.standard.data(forKey: AppModel.key),
           let saved = try? JSONDecoder().decode(EngineConfig.self, from: data) {
            cfg = saved
        } else {
            cfg = EngineConfig.guess()
        }
        config = cfg
        store = ProjectStore(root: cfg.projectsRoot)
        if cfg.problems.isEmpty { selection = store.projects.first.map { .project($0.path) } ?? .ingest }
        // every reload (the 3 s refresh sees .hs.lock appear) looks for runs started in Terminal;
        // @Published delivers the new value before the property holds it, so use the one passed
        projectsWatch = store.$projects.sink { [weak self] ps in
            MainActor.assumeIsolated { self?.attachExternalRuns(ps) }
        }
    }

    private func save() {
        if let data = try? JSONEncoder().encode(config) {
            UserDefaults.standard.set(data, forKey: AppModel.key)
        }
    }

    func resetConfig() {
        config = EngineConfig.guess()
    }

    func session(_ title: String, _ args: [String]) -> RunSession {
        RunSession(title: title, config: config, arguments: args)
    }

    var anyProjectRunRunning: Bool { projectRuns.values.contains { $0.isRunning } }

    // MARK: runs started outside the app

    /// Follow runs started in Terminal (`hs solve -p P`): a project whose lock names a stage and a
    /// live pid, while the app has no live run of its own there, gets an attached `RunSession`
    /// that tails `logs/<stage>.events.jsonl` every 2 s — so the page's run panel, the strip, the
    /// sidebar row, the title and the Dock badge follow it like a run the app started, and its
    /// Stop sends the pid the SIGINT a ^C would. It removes itself on `done` or when the pid dies.
    /// Called on every store reload, on the store's 3 s poll, and on selecting a project.
    func attachExternalRuns(_ projects: [ProjectSummary]? = nil) {
        for p in projects ?? store.projects { attachExternalRun(path: p.path, lock: p.lock) }
    }

    private func attachExternalRun(path: String, lock: ProjectSummary.LockInfo?) {
        guard let l = lock, l.alive, let pid = l.pid, let stage = l.stage else { return }
        if projectRuns[path]?.isRunning == true || appQueueRunning(path) { return }   // ours, or already followed
        let key = "\(path)|\(pid)"
        if unfollowable.contains(key) { return }
        let s = RunSession(attachingTo: path, stage: stage, pid: Int32(pid), config: config)
        s.attach()
        guard s.isRunning else {
            unfollowable.insert(key)
            return
        }
        s.onFinish = { [weak self] run in
            guard let self = self, self.projectRuns[path] === run else { return }
            self.projectRuns[path] = nil
            // a new Terminal run may already hold the lock: its .hs.lock change was seen while this one ran
            self.attachExternalRun(path: path, lock: ProjectStore.readLock(path))
        }
        projectRuns[path] = s
    }

    /// One of the app's queues is running on `path` — between its steps too, when the session in
    /// `projectRuns` has finished and the next has not been created yet.
    private func appQueueRunning(_ path: String) -> Bool {
        [trainQueues, selectQueues, exposureQueues, maskQueues, moveQueues, scaleQueues].contains { $0[path]?.isRunning == true }
    }

    // MARK: live status — window title and Dock badge

    /// Follows every project's run: any change to any of them (throttled to one a second)
    /// re-reads which runs are live. Re-subscribed whenever a run is added or replaced.
    private func watchRuns() {
        let changes = projectRuns.values.map { $0.objectWillChange }
        runWatch = Publishers.MergeMany(changes)
            .throttle(for: .seconds(1), scheduler: RunLoop.main, latest: true)
            .sink { [weak self] _ in
                MainActor.assumeIsolated { self?.updateLiveStatus() }
            }
        updateLiveStatus()
    }

    /// The live run the title and the Dock follow: the selected project's, else the first by path.
    private var followedRun: (run: RunSession, others: Int)? {
        let live = projectRuns.filter { $0.value.isRunning }
        guard !live.isEmpty else { return nil }
        if case .project(let p)? = selection, let r = live[p] { return (r, live.count - 1) }
        guard let first = live.keys.sorted().first, let r = live[first] else { return nil }
        return (r, live.count - 1)
    }

    private func updateLiveStatus() {
        let f = followedRun
        let title = f.map { $0.run.liveProgress.short + ($0.others > 0 ? " (+\($0.others) more)" : "") }
        if title != liveTitle { liveTitle = title }
        // cleared the moment nothing is live; a run without a fraction gets no badge
        let badge = f?.run.liveProgress.badge
        if let app = NSApp, app.dockTile.badgeLabel != badge { app.dockTile.badgeLabel = badge }
    }

    /// The main window's title: "HydrogenSplat — solve · matching 42% · 18 min" while a run is live.
    var windowTitle: String {
        liveTitle.map { "HydrogenSplat — \($0)" } ?? "HydrogenSplat"
    }
}
