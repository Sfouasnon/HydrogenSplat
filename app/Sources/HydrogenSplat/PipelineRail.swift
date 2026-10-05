import SwiftUI
import HSCore

/// The six numbered steps and the two doors of docs/ui-rebuild.md, in the order a project walks
/// them. A step's circle is the least advanced of the engine stages behind it; the text on the
/// right is its two-word state ("243 picked", "matched").
enum PipelineStage: String, CaseIterable, Identifiable {
    case footage, frames, look, subject, train, shot, calibrate, settings

    var id: String { rawValue }

    var title: String {
        switch self {
        case .footage: return "Footage"
        case .frames: return "Frames"
        case .look: return "Look"
        case .subject: return "Subject"
        case .train: return "Train"
        case .shot: return "Shot"
        case .calibrate: return "Calibrate"
        case .settings: return "Settings"
        }
    }

    /// 1–6 for the steps; nil for the two doors.
    var number: Int? {
        switch self {
        case .footage: return 1
        case .frames: return 2
        case .look: return 3
        case .subject: return 4
        case .train: return 5
        case .shot: return 6
        case .calibrate, .settings: return nil
        }
    }

    var isDoor: Bool { number == nil }

    static let steps: [PipelineStage] = [.footage, .frames, .look, .subject, .train, .shot]
    static let doors: [PipelineStage] = [.calibrate, .settings]

    /// Engine stages whose details belong to this item.
    var engineStages: [String] {
        switch self {
        case .footage: return ["ingest"]
        case .frames: return ["select", "solve"]
        case .look: return ["exposure"]
        case .subject: return ["masks"]
        case .train: return ["train", "archive", "views"]
        case .shot: return ["move", "render", "grade"]
        case .calibrate: return ["scale", "calibrate"]
        case .settings: return []
        }
    }

    /// Engine stages that decide the circle. Archive and views follow a train; a train that was
    /// never archived is not an unfinished train.
    var statusStages: [String] {
        switch self {
        case .train: return ["train"]
        default: return engineStages
        }
    }

    /// A step a project can skip: the circle stays grey and `next` walks past it.
    var optional: Bool {
        switch self {
        case .look, .subject, .calibrate, .settings: return true
        case .footage, .frames, .train, .shot: return false
        }
    }

    /// The engine stage `name` belongs to.
    static func containing(_ name: String) -> PipelineStage? {
        allCases.first { $0.engineStages.contains(name) }
    }

    func status(in m: Manifest, lock: ProjectSummary.LockInfo?) -> StageStatus {
        if let l = lock, l.alive, let s = l.stage, statusStages.contains(s) { return .running }
        let st = statusStages.map { m.stage($0)?.status ?? .pending }
        // the least advanced wins: something failing or unfinished is what needs attention
        for s in [StageStatus.running, .failed, .stale, .pending, .unknown] where st.contains(s) {
            return s
        }
        return .done
    }

    /// Where to land when a project is opened: the first required step not yet done.
    static func next(in m: Manifest, lock: ProjectSummary.LockInfo?) -> PipelineStage {
        steps.first { !$0.optional && $0.status(in: m, lock: lock) != .done } ?? .train
    }

    /// The two words on the right of the rail row, from the manifest's own metrics. Empty when
    /// the step has nothing to say yet.
    func railLabel(in m: Manifest, lock: ProjectSummary.LockInfo?) -> String {
        let st = status(in: m, lock: lock)
        if st == .running { return "running" }
        if st == .failed { return "failed" }
        switch self {
        case .footage:
            guard m.stage("ingest")?.status == .done else { return "" }
            if m.sourceKind == "array" { return "\(m.cameras.count) cameras" }
            if m.framesOnly { return "stills" }
            return m.isArray ? "clip" : "3D clip"
        case .frames:
            let picked = m.stage("select")?.metrics["frames_selected"]?.int
                ?? (m.framesOnly && !m.cameras.isEmpty ? m.cameras.count : nil)
            if m.stage("solve")?.status == .done, let n = PipelineStage.registered(in: m) ?? picked {
                return "\(n) placed"
            }
            if let n = picked { return "\(n) picked" }
            return ""
        case .look:
            guard let e = m.stage("exposure") else { return m.stage("solve")?.status == .done ? "as shot" : "" }
            if e.metrics["restored"] != nil { return "as shot" }
            switch e.status {
            case .done: return "matched"
            case .stale: return "stale"
            case .pending, .unknown: return "as shot"
            case .running, .failed: return e.status.rawValue
            }
        case .subject:
            guard let k = m.stage("masks") else { return "" }
            if k.status == .stale { return "stale" }
            if let n = k.metrics["views"]?.int { return "\(n) outlines" }
            return k.status == .done ? "outlined" : ""
        case .train:
            switch m.stage("train")?.status {
            case .done?: return "trained"
            case .stale?: return "stale"
            default: return ""
            }
        case .shot:
            if m.stage("render")?.status == .done { return "rendered" }
            if m.stage("move")?.status == .done { return "keyed" }
            return ""
        case .calibrate:
            if m.stage("calibrate")?.status == .done { return "lens known" }
            if m.stage("scale")?.status == .done { return "scaled" }
            return ""
        case .settings:
            return ""
        }
    }

    /// Captures the solve registered, from its `all_frames_registered` check ("243/243").
    static func registered(in m: Manifest) -> Int? {
        guard let c = m.stage("solve")?.checks.first(where: { $0.name == "all_frames_registered" }),
              let v = c.value?.display.split(separator: "/").first else { return nil }
        return Int(v.trimmingCharacters(in: .whitespaces))
    }
}

/// The left column: six numbered rows, a gap, two doors.
struct PipelineRail: View {
    let manifest: Manifest
    let lock: ProjectSummary.LockInfo?
    @Binding var selection: PipelineStage
    /// A label the page knows better than the manifest ("everything" for a scene with no outlines).
    var labels: [PipelineStage: String] = [:]

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            ForEach(PipelineStage.steps) { item in row(item) }
            Divider().padding(.vertical, 8)
            ForEach(PipelineStage.doors) { item in row(item) }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 12)
    }

    private func row(_ item: PipelineStage) -> some View {
        let st = item.status(in: manifest, lock: lock)
        let on = selection == item
        let label = labels[item] ?? item.railLabel(in: manifest, lock: lock)
        return Button { selection = item } label: {
            HStack(spacing: 10) {
                circle(item, st)
                Text(item.title).font(.body.weight(on ? .semibold : .regular))
                Spacer(minLength: 6)
                if !label.isEmpty {
                    Text(label).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                }
            }
            .padding(.horizontal, 8).padding(.vertical, 7)
            .background(RoundedRectangle(cornerRadius: 7).fill(on ? Color.secondary.opacity(0.16) : Color.clear))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(item.title)
    }

    @ViewBuilder private func circle(_ item: PipelineStage, _ st: StageStatus) -> some View {
        let started = st != .pending && st != .unknown
        let fill: Color = item == .settings ? Color.secondary.opacity(0.25)
                        : (started ? Verdict.from(st).color : Color.secondary.opacity(0.25))
        ZStack {
            Circle().fill(fill).frame(width: 24, height: 24)
            if let n = item.number {
                Text(String(n))
                    .font(.caption.weight(.bold)).monospacedDigit()
                    .foregroundStyle(started ? Color.white : Color.primary)
            } else {
                Image(systemName: item == .settings ? "gearshape" : "scope")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(started && item != .settings ? Color.white : Color.primary)
            }
        }
    }

    static func color(_ s: StageStatus) -> Color {
        switch s {
        case .done: return .green
        case .running: return .blue
        case .failed: return .red
        case .stale: return .orange
        case .pending, .unknown: return .secondary
        }
    }
}
