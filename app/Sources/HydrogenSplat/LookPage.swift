import SwiftUI
import AppKit
import Charts
import HSCore

/// Step 3 — Look. `hs exposure --analyze` measures the subject's brightness in every frame and
/// says one thing: the drift in stops, the clipped share, where it clips, and ONE recommendation
/// (match to a reference / drop everything half a stop / soften the highlight shoulder). The page
/// states it, shows the brightness per frame, offers the three remedies and applies the chosen
/// one with `hs exposure --apply`.
struct LookPage: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var store: ProjectStore
    let project: ProjectSummary

    @State private var quality: FrameQuality?
    @State private var choice: String?
    @State private var showReference = false
    @State private var confirmStale = false

    private var manifest: Manifest? { project.manifest }
    private var stage: StageState? { manifest?.stage("exposure") }
    private var queue: RunQueue? { model.exposureQueues[project.path] }
    private var running: Bool { queue?.isRunning ?? false }
    private var lockAlive: Bool { project.lock?.alive == true }
    private var blocked: Bool { running || lockAlive || !model.config.problems.isEmpty }
    private var solveReady: Bool {
        let s = manifest?.stage("solve")?.status
        return s == .done || s == .stale
    }
    private var trainDone: Bool { manifest?.stage("train")?.status == .done }

    private var settings: Binding<ExposureSettings> {
        Binding(get: { model.exposureSettings[project.path] ?? ExposureSettings() },
                set: { model.exposureSettings[project.path] = $0 })
    }

    private var reloadKey: String {
        "\(project.path)|\(manifest?.stage("select")?.finished?.timeIntervalSince1970 ?? 0)"
    }

    // MARK: the engine's numbers (hs exposure --analyze)

    private var drift: Double? { stage?.metrics["drift_stops"]?.double }
    private var clippedShare: Double? { stage?.metrics["clipped_share"]?.double }
    private var clippedWhere: String? { stage?.metrics["clipped_where"]?.string }
    private var recommendation: String? { stage?.metrics["recommendation"]?.string }
    private var recommendationWhy: String? { stage?.metrics["recommendation_why"]?.string }
    private var measured: Bool { drift != nil && recommendation != nil }
    private var consistent: CheckResult? { stage?.checks.first { $0.name == "exposure_consistent" } }
    /// What the last `--apply` did: ["match", "shoulder"].
    private var applied: [String] {
        guard stage?.status == .done, stage?.metrics["restored"] == nil else { return [] }
        if let a = stage?.metrics["applied"]?.array { return a.compactMap { $0.string } }
        // a run from before --apply existed matched to a reference
        return stage?.metrics["reference"] != nil ? ["match"] : []
    }
    /// The subject kind, for the recommendation (bright and glossy subjects get the shoulder).
    private var subject: String? {
        model.subjectKind[project.path] ?? manifest?.raw["project"]?["subject_kind"]?.string
    }

    private struct Brightness: Identifiable {
        let id: Int
        let view: String
        let stops: Double
        let clipped: Double?
    }

    /// `brightness_by_frame`: each view's brightness in stops off the median view.
    private var brightness: [Brightness] {
        guard let rows = stage?.metrics["brightness_by_frame"]?.array else { return [] }
        var out: [Brightness] = []
        for (i, r) in rows.enumerated() {
            guard let s = r["stops"]?.double else { continue }
            out.append(Brightness(id: i, view: r["view"]?.string ?? "#\(i)", stops: s, clipped: r["clipped_share"]?.double))
        }
        return out
    }

    var body: some View {
        StepPage {
            StepHeader(title: "Look", lead: "Does the subject's brightness hold from frame to frame, and is anything blown out?")
            if manifest == nil {
                Text(project.manifestError ?? "This project has no manifest.").foregroundStyle(.red)
            } else if !solveReady {
                VerdictCard(.info, headline: "Place the cameras first.",
                            detail: "The Look step works on the placed frames; it has nothing to measure before Frames is done.")
            } else {
                verdict
                if measured { chart }
                choices
                if applied.contains("match") || (choice ?? recommendation) == "match" {
                    DisclosureGroup("Which frame to match to", isExpanded: $showReference) {
                        ExposureReferencePicker(project: project.path, settings: settings, quality: quality).padding(.top, 6)
                    }
                }
                if !applied.isEmpty { beforeAfter }
                if let s = stage, s.status == .done || s.status == .stale || s.status == .failed {
                    DisclosureGroup("What the last run did") { ExposureApplied(stage: s).padding(.top, 6) }
                }
                footer
            }
        }
        .task(id: reloadKey) { quality = FrameQuality.load(project: project.path) }
        .confirmationDialog("Rewrite the training frames?", isPresented: $confirmStale) {
            Button("Apply and continue", role: .destructive) { apply() }
        } message: {
            Text("The current model was trained on the frames as they are; applying this marks it for retraining. The originals are kept, and Keep as shot puts them back.")
        }
    }

    // MARK: verdict

    @ViewBuilder private var verdict: some View {
        if running {
            VerdictCard(.attention, headline: queue?.steps.first?.title == "Measure the frames" ? "Measuring the frames…" : "Rewriting the frames…",
                        detail: "The strip at the top follows the run.")
        } else if measured, let d = drift, let share = clippedShare {
            let ok = consistent?.ok ?? true
            VerdictCard(applied.isEmpty ? (ok ? .good : .attention) : .good,
                        headline: applied.isEmpty ? (consistent?.value?.display ?? defaultHeadline(d, share))
                                                  : "Applied: " + applied.map(appliedWord).joined(separator: ", ") + ". The frames are matched.",
                        detail: applied.isEmpty ? recommendationWhy.map { "Why: " + $0 + "." } : "Measure again to see the new drift, or go on to Subject.") {
                HStack(spacing: 8) {
                    MetricChip(text: String(format: "Drift %.1f stops", d), tint: d < 0.5 ? .green : .orange)
                    MetricChip(text: String(format: "Clipped %.1f%%", share * 100), tint: share >= 0.15 ? .orange : nil)
                    if let w = clippedWhere { MetricChip(text: clipText(w)) }
                    if let n = stage?.metrics["analysis"]?["views"]?.int { MetricChip(text: "\(n) frames measured") }
                }
            }
        } else if stage?.status == .failed {
            VerdictCard(.blocked, headline: "The frames could not be measured.", detail: stage?.error ?? "Open Details for the engine's reason.")
        } else {
            VerdictCard(.info, headline: "Not measured yet.",
                        detail: "Measuring gives the drift in stops across the orbit, the share of the subject that is blown out, and one recommendation.")
        }
    }

    private func defaultHeadline(_ d: Double, _ share: Double) -> String {
        String(format: "The subject's brightness moves %.1f stops across the frames; %.1f%% of it is blown out.", d, share * 100)
    }

    private func clipText(_ w: String) -> String {
        switch w {
        case "none": return "nothing blown out"
        case "highlights": return "clips in the highlights only"
        case "whole": return "clips in large areas"
        default: return "clips: " + w
        }
    }

    private func appliedWord(_ s: String) -> String {
        switch s {
        case "match": return "matched to a reference"
        case "global_drop": return "lowered ½ stop"
        case "shoulder": return "softer shoulder"
        default: return s
        }
    }

    // MARK: chart

    private var chart: some View {
        VerdictCard(.info, headline: "Brightness per frame", detail: "Stops off the median frame; the flat line is the goal.") {
            Chart {
                RuleMark(y: .value("Median", 0)).foregroundStyle(Color.secondary.opacity(0.4))
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3]))
                ForEach(brightness) { b in
                    LineMark(x: .value("Frame", b.id), y: .value("Stops", b.stops))
                        .foregroundStyle(Brand.accent)
                    if let c = b.clipped, c >= 0.05 {
                        PointMark(x: .value("Frame", b.id), y: .value("Stops", b.stops))
                            .foregroundStyle(Color.orange).symbolSize(24)
                    }
                }
            }
            .chartYAxisLabel("stops")
            .frame(height: 120)
            if brightness.contains(where: { ($0.clipped ?? 0) >= 0.05 }) {
                Text("Orange dots: frames where 5% or more of the subject is blown out.").font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    // MARK: choices

    private var chosen: String? { choice ?? (measured ? recommendation : nil) }

    private var choices: some View {
        VerdictCard(.info, headline: measured ? "What to do about it" : "What can be done",
                    detail: measured ? nil : "Measure first; the recommended choice is then marked.") {
            HStack(alignment: .top, spacing: 10) {
                tile("match", "Match to a reference", "Every frame is scaled onto one well-exposed frame; drift goes away.")
                tile("global_drop", "Lower everything ½ stop", "Every frame darkened the same amount; brings blown-out areas back from the ceiling.")
                tile("shoulder", "Soften the highlight shoulder", "Eases only the brightest values down; keeps glints on a bright or glossy subject.")
            }
            if let r = recommendation, r == "none" {
                Text("The measurement found nothing to fix; Keep as shot is the recommendation.").font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private func tile(_ key: String, _ title: String, _ subtitle: String) -> some View {
        let rec = recommendation == key
        return ChoiceTile(title: title + (rec ? " — recommended" : ""), subtitle: subtitle, selected: chosen == key) {
            choice = key
        }
    }

    // MARK: before / after

    private var imagesDir: String { (project.path as NSString).appendingPathComponent("train/dataset/images") }
    private var backupDir: String { (project.path as NSString).appendingPathComponent("solve/exposure_backup") }

    /// The view's file under `dir` ("L/cap012" → dir/L/cap012.jpg), whichever extension it has.
    private func imagePath(dir: String, view: String) -> String? {
        for ext in ["jpg", "jpeg", "png"] {
            let p = (dir as NSString).appendingPathComponent(view + "." + ext)
            if FileManager.default.fileExists(atPath: p) { return p }
        }
        return nil
    }

    /// The darkest and brightest measured frames, before (the backup) and after (the dataset).
    private var beforeAfter: some View {
        let rows = brightness
        let picks: [Brightness] = [rows.min { $0.stops < $1.stops }, rows.max { $0.stops < $1.stops }].compactMap { $0 }
        return VerdictCard(.info, headline: "Before and after", detail: picks.isEmpty ? "Measure the frames to pick the darkest and brightest for this comparison." : "The darkest and the brightest frame, as shot and as the trainer sees them now.") {
            HStack(alignment: .top, spacing: 16) {
                ForEach(picks) { b in
                    VStack(alignment: .leading, spacing: 4) {
                        HStack(spacing: 6) {
                            pair(imagePath(dir: backupDir, view: b.view), "as shot")
                            pair(imagePath(dir: imagesDir, view: b.view), "now")
                        }
                        Text(b.view + String(format: " · %+.2f stops", b.stops)).font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
        }
    }

    @ViewBuilder private func pair(_ path: String?, _ label: String) -> some View {
        VStack(spacing: 2) {
            if let p = path {
                ThumbImage(path: p).frame(width: 150).clipShape(RoundedRectangle(cornerRadius: 4))
            } else {
                Rectangle().fill(Color.secondary.opacity(0.12)).frame(width: 150, height: 84)
            }
            Text(label).font(.caption2).foregroundStyle(.secondary)
        }
    }

    // MARK: the one action

    private var footer: some View {
        Observing(model.projectRuns[project.path]) { _ in
            let problem = chosen == "match" ? settings.wrappedValue.problem : nil
            if !measured {
                StepFooter(primary: running ? "Measuring…" : "Measure the frames",
                           primaryEnabled: !blocked,
                           secondary: "Keep as shot",
                           note: model.config.problems.isEmpty ? "Reads every frame; writes nothing." : "Fix Settings first.") {
                    analyze()
                } secondaryAction: {
                    model.pipelineStage[project.path] = .subject
                }
            } else if let c = chosen, c != "none" {
                StepFooter(primary: running ? "Working…" : primaryTitle(c),
                           primaryEnabled: !blocked && problem == nil,
                           secondary: applied.isEmpty ? "Keep as shot" : "Put the frames back as shot",
                           note: problem ?? (blocked ? "Wait for the running step." : "Rewrites the training frames; the originals are kept.")) {
                    if trainDone { confirmStale = true } else { apply() }
                } secondaryAction: {
                    if applied.isEmpty { model.pipelineStage[project.path] = .subject } else if !blocked { restore() }
                }
            } else {
                StepFooter(primary: "Keep as shot and continue",
                           secondary: "Measure again",
                           note: "Nothing to fix; the frames go to training as they are.") {
                    model.pipelineStage[project.path] = .subject
                } secondaryAction: {
                    if !blocked { analyze() }
                }
            }
        }
    }

    private func primaryTitle(_ c: String) -> String {
        switch c {
        case "match": return "Match the frames and continue"
        case "global_drop": return "Lower the frames and continue"
        case "shoulder": return "Soften the highlights and continue"
        default: return "Apply and continue"
        }
    }

    /// `hs exposure -p P --analyze [--subject KIND]`: measures, writes nothing.
    private func analyze() {
        var args = ["exposure", "-p", project.path, "--analyze"]
        if let s = subject { args += ["--subject", s] }
        start([RunQueue.Step(title: "Measure the frames", arguments: args)])
    }

    /// `hs exposure -p P --apply STEP` with the reference and mode from the picker for a match.
    private func apply() {
        guard let c = chosen, c != "none" else { return }
        var args = settings.wrappedValue.arguments(project: project.path)
        args += ["--apply", c]
        let title = c == "match" ? "Match exposure" : (c == "global_drop" ? "Lower exposure" : "Soften highlights")
        start([RunQueue.Step(title: title, arguments: args),
               RunQueue.Step(title: "Measure the frames", arguments: ["exposure", "-p", project.path, "--analyze"])])
    }

    private func restore() {
        start([RunQueue.Step(title: "Restore exposure", arguments: ExposureSettings.restoreArguments(project: project.path))])
    }

    private func start(_ steps: [RunQueue.Step]) {
        let q = RunQueue(config: model.config, steps: steps)
        let path = project.path
        q.onStep = { s in model.projectRuns[path] = s }
        q.onFinish = { _ in store.reload() }
        model.exposureQueues[path] = q
        q.start()
    }
}
