import SwiftUI
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
    @Published var selection: SidebarItem? = .setup
    /// The latest run per project path (ingest today; every stage from M2).
    @Published var projectRuns: [String: RunSession] = [:]
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
}
