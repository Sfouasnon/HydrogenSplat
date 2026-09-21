import SwiftUI
import HSCore

/// The pipeline in the order the engine runs it. Each item is one place to work; its dot is the
/// state of the engine stage(s) behind it. Move and Render live in the Viewer's move panel, which
/// this rail opens rather than replaces.
enum PipelineStage: String, CaseIterable, Identifiable {
    case source, frames, exposure, masks, train, move, grade, render

    var id: String { rawValue }

    var title: String {
        switch self {
        case .source: return "Source"
        case .frames: return "Frames"
        case .exposure: return "Exposure"
        case .masks: return "Masks"
        case .train: return "Train"
        case .move: return "Move"
        case .grade: return "Grade"
        case .render: return "Render"
        }
    }

    /// Engine stages whose details the inspector shows for this item.
    var engineStages: [String] {
        switch self {
        case .source: return ["ingest"]
        case .frames: return ["select", "solve"]
        case .exposure: return ["exposure"]
        case .masks: return ["masks"]
        case .train: return ["train", "archive", "views"]
        case .move: return ["move"]
        case .grade: return ["grade"]
        case .render: return ["render"]
        }
    }

    /// Engine stages that decide the item's dot. Archive and views follow a train; a train that
    /// was never archived is not an unfinished train.
    var statusStages: [String] {
        switch self {
        case .train: return ["train"]
        default: return engineStages
        }
    }

    var optional: Bool { self == .exposure || self == .masks || self == .grade }
    /// Worked in the Viewer page's move panel, not here.
    var inViewer: Bool { self == .move || self == .render }

    static let groups: [(String, [PipelineStage])] = [
        ("Prepare", [.source, .frames]),
        ("Dataset", [.exposure, .masks]),
        ("Model", [.train]),
        ("Shot", [.move, .grade, .render]),
    ]

    /// The engine stage `name` belongs to, for jumping from the full stage table.
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

    /// Where to land when a project is opened: the first required thing not yet done.
    static func next(in m: Manifest, lock: ProjectSummary.LockInfo?) -> PipelineStage {
        allCases.first { !$0.optional && !$0.inViewer && $0.status(in: m, lock: lock) != .done } ?? .train
    }
}

struct PipelineRail: View {
    let manifest: Manifest
    let lock: ProjectSummary.LockInfo?
    @Binding var selection: PipelineStage

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(alignment: .top, spacing: 14) {
                ForEach(Array(PipelineStage.groups.enumerated()), id: \.offset) { i, group in
                    if i > 0 { Divider().frame(height: 46) }
                    VStack(alignment: .leading, spacing: 5) {
                        Text(group.0.uppercased())
                            .font(.caption2.weight(.semibold)).tracking(1.2)
                            .foregroundStyle(.secondary)
                        HStack(spacing: 4) {
                            ForEach(group.1) { item in button(item) }
                        }
                    }
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
        }
    }

    private func button(_ item: PipelineStage) -> some View {
        let st = item.status(in: manifest, lock: lock)
        let on = selection == item
        return Button { selection = item } label: {
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 5) {
                    Text(item.title).font(.callout.weight(.semibold))
                    if item.optional {
                        Text("opt").font(.system(size: 9, design: .monospaced))
                            .padding(.horizontal, 3)
                            .overlay(RoundedRectangle(cornerRadius: 3).stroke(Color.secondary.opacity(0.4)))
                            .foregroundStyle(.secondary)
                    }
                    if item.inViewer {
                        Image(systemName: "arrow.up.forward.square").font(.caption2).foregroundStyle(.secondary)
                    }
                }
                HStack(spacing: 4) {
                    Circle().fill(Self.color(st)).frame(width: 6, height: 6)
                    Text(st.rawValue).font(.caption2.monospaced()).foregroundStyle(Self.color(st))
                }
            }
            .padding(.horizontal, 10).padding(.vertical, 6)
            .frame(minWidth: 78, alignment: .leading)
            .background(RoundedRectangle(cornerRadius: 6).fill(on ? Color.secondary.opacity(0.16) : .clear))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(item.inViewer ? "\(item.title) is worked in the Viewer's move panel" : item.title)
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
