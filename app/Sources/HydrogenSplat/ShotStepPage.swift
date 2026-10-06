import SwiftUI
import HSCore

/// Step 7, Shot: three tabs under one header. Clean up is the viewer with the floater tools beside
/// it (CleanUpView.swift). Move is the viewer with the keyframe editor beside
/// it (presets, keys, anchor, lens, play) and the coverage band on its timeline; Render & grade
/// is the check chips, the graded preview and filmstrip, and the Frame / Grade / Export cards
/// (`GradeView`), which carry the whole `hs grade` scope.
struct ShotStepPage: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var store: ProjectStore
    let project: ProjectSummary

    enum Half: String, CaseIterable, Identifiable {
        case clean, move, render
        var id: String { rawValue }
        var title: String {
            switch self {
            case .clean: return "1  Clean up"
            case .move: return "2  Move"
            case .render: return "3  Render & grade"
            }
        }
    }

    @AppStorage("shot.half") private var halfRaw = Half.move.rawValue
    private var half: Binding<Half> {
        Binding(get: { Half(rawValue: halfRaw) ?? .move }, set: { halfRaw = $0.rawValue })
    }

    private var models: [ViewerModelFile] { ViewerModelFile.list(project: project.path) }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            VStack(alignment: .leading, spacing: 10) {
                StepHeader(title: "Shot", lead: "Move the camera through the model, then render, grade and export the clip.")
                Picker("", selection: half) {
                    ForEach(Half.allCases) { h in Text(h.title).tag(h) }
                }
                .pickerStyle(.segmented).labelsHidden().frame(width: 520)
            }
            .padding(.horizontal, 40).padding(.top, 32).padding(.bottom, 14)
            Divider()
            switch half.wrappedValue {
            case .clean: cleanHalf
            case .move: moveHalf
            case .render: renderHalf
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    // MARK: 1 Clean up

    /// The viewer with the clean-up panel beside it: floaters taken off the camera's path, or
    /// boxed and erased by hand, each into a copy (CleanUpView.swift).
    @ViewBuilder private var cleanHalf: some View {
        if project.manifest == nil || models.isEmpty {
            StepPage {
                VerdictCard(.blocked, headline: "There is no model to clean yet.",
                            detail: "Train one on the Train step; it appears here.")
            }
        } else {
            ModelViewerPane(scene: model.viewerScene, project: project, cleaning: true)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    // MARK: 2 Move

    @ViewBuilder private var moveHalf: some View {
        if project.manifest == nil || models.isEmpty {
            StepPage {
                VerdictCard(.blocked, headline: "There is no model to move through yet.",
                            detail: "Train one on the Train step; it appears here with the cameras it was trained against.")
            }
        } else {
            VStack(spacing: 0) {
                MoveCoverageLine(editor: model.moveEditor(project: project.path))
                    // as tall as its text: the card's coloured bar takes any height it is offered,
                    // and beside the viewer in this stack that was half the page
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.horizontal, 40).padding(.vertical, 10)
                Divider()
                ModelViewerPane(scene: model.viewerScene, project: project)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }

    // MARK: 3 Render & grade

    private var renderHalf: some View {
        StepPage {
            renderChecks
            GradeView(project: project)
        }
    }

    /// The four facts about the render on disk: which model, whether stray specks were pruned,
    /// whether the stability pass ran, and how much of the open move leaves what was filmed.
    @ViewBuilder private var renderChecks: some View {
        if let m = project.manifest {
            let render = m.stage("render")
            HStack(spacing: 8) {
                if let ply = render?.metrics["ply"]?.string {
                    MetricChip(text: "rendered from " + ShotStepPage.modelLabel(ply), tint: render?.status == .done ? nil : Color.orange)
                } else {
                    MetricChip(text: "nothing rendered yet")
                }
                if let p = m.stage("prune"), p.status == .done {
                    if let kept = p.metrics["kept_fraction"]?.double {
                        MetricChip(text: String(format: "stray specks pruned · %.0f %% of the splats kept", kept * 100), tint: .green)
                    } else {
                        MetricChip(text: "stray specks pruned", tint: .green)
                    }
                } else {
                    MetricChip(text: "stray specks not pruned")
                }
                if let r = render, let f = ShotStepPage.flicker(r) {
                    MetricChip(text: String(format: "stability checked · flicker %.3f", f), tint: .green)
                } else if let r = render, r.metrics.keys.contains(where: { $0.hasPrefix("stability_") }) {
                    MetricChip(text: "stability checked", tint: .green)
                } else {
                    MetricChip(text: "stability not checked")
                }
                MoveCoverageChip(editor: model.moveEditor(project: project.path))
            }
            if let r = render, !r.checks.filter({ !$0.ok }).isEmpty {
                VerdictCard(.attention, headline: "The last render has checks that failed.",
                            detail: r.checks.filter { !$0.ok }.map { $0.name.replacingOccurrences(of: "_", with: " ") + ($0.value?.string.map { ": " + $0 } ?? "") }.joined(separator: " · "))
            }
        }
    }

    /// "archive H-1920" for archive/H-1920/export_40000.ply; "the current model" for train/exports.
    static func modelLabel(_ ply: String) -> String {
        let parts = ply.split(separator: "/").map(String.init)
        if let i = parts.firstIndex(of: "archive"), i + 1 < parts.count { return "the kept model \(parts[i + 1])" }
        if parts.contains("exports") { return "the current model" }
        if parts.contains("prune") { return "the pruned model" }
        return (ply as NSString).lastPathComponent
    }

    /// The stability pass's median flicker at the smallest frame gap it measured, if it ran.
    static func flicker(_ r: StageState) -> Double? {
        let keys = r.metrics.keys.filter { $0.hasPrefix("stability_flicker_median_k") }.sorted()
        guard let k = keys.first else { return nil }
        return r.metrics[k]?.double
    }
}

/// The open move's coverage as the step's verdict line: green when every frame is inside what
/// the cameras saw, amber with the stretch and the fix when not.
struct MoveCoverageLine: View {
    @ObservedObject var editor: MoveEditor

    var body: some View {
        if let m = editor.move, let a = editor.analysis {
            let gap = coverageGap(a)
            VerdictCard(gap == nil ? .good : .attention,
                        headline: coverageSentence(a, fps: m.fps),
                        detail: String(format: "%@ · %.1f s · %ld frames · look at the first, middle and last frame, then Render.",
                                       editor.name ?? "move", m.duration, m.frameCount))
        } else {
            VerdictCard(.info, headline: "No move is open.",
                        detail: "In the panel on the right: pick a preset or New Move From Here, then K adds a key where the viewer's camera stands. The band under the timeline is green where the frame is inside what the cameras saw and amber where it is not.")
        }
    }
}

/// How much of the open move leaves what was filmed, as one chip.
struct MoveCoverageChip: View {
    @ObservedObject var editor: MoveEditor

    var body: some View {
        if let a = editor.analysis, !a.offAngle.isEmpty {
            let share = a.fraction(over: MoveAnalysis.amberDeg)
            if share > 0 {
                MetricChip(text: String(format: "%.0f %% of the frames outside coverage", share * 100), tint: .orange)
            } else {
                MetricChip(text: "every frame inside coverage", tint: .green)
            }
        } else {
            MetricChip(text: "coverage: open a move to see it")
        }
    }
}
