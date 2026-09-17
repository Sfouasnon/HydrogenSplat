import SwiftUI
import HSCore

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
    @Published var consoleHistory: [ConsoleSession] = []

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
