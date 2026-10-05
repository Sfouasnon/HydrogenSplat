import SwiftUI
import AppKit
import HSCore

/// Step 4 — Subject. Is an outline worth it (the solve's `subject_share_of_frame`: the share of
/// the model the room would take), the two choices (model the subject only / model everything),
/// the outlines `hs masks` made and tightened, and how many frames need a look with the three
/// things to do about them.
struct SubjectPage: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var store: ProjectStore
    let project: ProjectSummary

    /// Listed when the stage changes, never in body: body runs on every poll.
    @State private var files = MaskFiles(counts: [:], newest: nil)
    @State private var review: MaskReview?
    @State private var reviewLoaded = false
    @State private var showList = false
    @State private var showSettings = false
    @State private var confirmRebuild = false

    private var manifest: Manifest? { project.manifest }
    private var stage: StageState? { manifest?.stage("masks") }
    private var solveStage: StageState? { manifest?.stage("solve") }
    private var queue: RunQueue? { model.maskQueues[project.path] }
    private var running: Bool { queue?.isRunning ?? false }
    private var lockAlive: Bool { project.lock?.alive == true }
    private var blocked: Bool { running || lockAlive || !model.config.problems.isEmpty }
    private var solveReady: Bool { solveStage?.status == .done }
    /// A model exists to project from (stale counts: it is still on disk).
    private var trainDone: Bool {
        let s = manifest?.stage("train")?.status
        return s == .done || s == .stale
    }

    /// What the subject is (Footage). Scene means no outlines.
    private var kind: SubjectKind? {
        if let raw = model.subjectKind[project.path], let k = SubjectKind(rawValue: raw) { return k }
        if let raw = manifest?.raw["project"]?["subject_kind"]?.string, let k = SubjectKind(rawValue: raw) { return k }
        return nil
    }
    private var wantsMasks: Bool { kind?.usesMasks ?? true }

    private var defaults: MaskSettings {
        var s = MaskSettings()
        if !trainDone { s.source = .points }
        if kind == .glossy { s.excludeHighlights = true }
        return s
    }
    private var settings: Binding<MaskSettings> {
        Binding(get: { model.maskSettings[project.path] ?? defaults },
                set: { model.maskSettings[project.path] = $0 })
    }

    private var reloadKey: String {
        "\(project.path)|\(stage?.finished?.timeIntervalSince1970 ?? 0)|\(stage?.status.rawValue ?? "")"
    }
    /// The review is read again when the stage changes, when the engine rewrites its
    /// `mask_review` metric, when review.json itself changes, and when a run here ends.
    private var reviewKey: String {
        let metric = stage?.metrics["mask_review"]?.compactJSON ?? ""
        let attrs = try? FileManager.default.attributesOfItem(atPath: MaskReview.reportPath(project: project.path))
        let stamp = (attrs?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        return "\(reloadKey)|\(running ? "running" : "idle")|\(metric)|\(stamp)"
    }

    // MARK: the engine's numbers

    private var subjectShare: Double? { solveStage?.metrics["subject_share_of_frame"]?.double }
    private var roomShare: Double? { solveStage?.metrics["room_share_estimate"]?.double }
    private var worthCheck: CheckResult? { solveStage?.checks.first { $0.name == "masking_worth_it" } }
    private var coverageMedian: Double? { stage?.metrics["coverage_median"]?.double }
    private var outlineCount: Int? { stage?.metrics["views"]?.int }
    private var snapOn: Bool { stage?.metrics["snap_edge"]?.bool ?? false }
    private var snapMoved: Double? { stage?.metrics["snap_moved_px"]?["median"]?.double }
    private var snapOffset: Double? { stage?.metrics["snap_offset_px"]?["median"]?.double }
    private var glintsCut: Bool {
        guard let v = stage?.metrics["exclude_highlights"] else { return false }
        return !v.isNull && v.bool != false
    }
    private var fellBack: Int { stage?.metrics["vision_fell_back"]?.int ?? 0 }
    private var undecided: Int { review?.flagged.filter { $0.decision == nil }.count ?? 0 }

    var body: some View {
        StepPage {
            StepHeader(title: "Subject", lead: "Model the subject alone, cut to its outline in every frame, or model everything in frame.")
            if manifest == nil {
                Text(project.manifestError ?? "This project has no manifest.").foregroundStyle(.red)
            } else if !solveReady {
                VerdictCard(.info, headline: "Place the cameras first.",
                            detail: "The outlines are projected with each placed camera; there is nothing to outline before Frames is done.")
            } else {
                worth
                choices
                if wantsMasks {
                    if stage != nil || !files.isEmpty { outlines }
                    if !files.isEmpty { reviewCard }
                    DisclosureGroup("Adjust the outlines", isExpanded: $showSettings) {
                        MaskSettingsPanel(settings: settings, modelAvailable: trainDone).padding(.top, 6)
                    }
                }
                footer
            }
        }
        .task(id: reloadKey) { files = MaskFiles.read(project: project.path) }
        .task(id: reviewKey) {
            review = MaskReview.read(project: project.path)
            reviewLoaded = true
        }
        .confirmationDialog("Make the outlines again?", isPresented: $confirmRebuild) {
            Button("Make the outlines", role: .destructive) { build() }
        } message: {
            Text("The current model was trained with the outlines on disk; replacing them marks it for retraining.")
        }
    }

    // MARK: is it worth it

    @ViewBuilder private var worth: some View {
        if let share = subjectShare {
            let room = roomShare ?? (1 - share)
            VerdictCard(.info,
                        headline: worthCheck?.value?.display
                            ?? String(format: "The subject fills %.0f%% of the frame; modelled whole, the room would take about %.0f%% of the model.", share * 100, room * 100),
                        detail: share < 0.5 ? "An outline keeps the model on the subject; the room is left out."
                                            : "The subject is most of the frame; an outline buys less here.") {
                HStack(spacing: 8) {
                    MetricChip(text: String(format: "Subject %.0f%% of frame", share * 100))
                    MetricChip(text: String(format: "Room would take ~%.0f%%", room * 100), tint: room >= 0.5 ? .orange : nil)
                }
            }
        } else if let cov = coverageMedian {
            VerdictCard(.info, headline: String(format: "The outlines cover %.0f%% of the frame in the typical view.", cov * 100),
                        detail: "From the outlines already made; placing the cameras again would measure the subject itself.")
        } else {
            VerdictCard(.info, headline: "Not measured.",
                        detail: "Placing the cameras again measures how much of the frame the subject fills; the choice below works without it.")
        }
    }

    // MARK: the two choices

    private var choices: some View {
        let share = subjectShare
        let recommendMasks = share.map { $0 < 0.75 } ?? (kind != .scene)
        return VerdictCard(.info, headline: "What should the model hold?") {
            HStack(alignment: .top, spacing: 10) {
                ChoiceTile(title: "Model the subject only" + (recommendMasks ? " — recommended" : ""),
                           subtitle: "An outline in every frame; the model holds nothing outside it.",
                           selected: wantsMasks) {
                    if kind == nil || kind == .scene { model.subjectKind[project.path] = SubjectKind.matte.rawValue }
                }
                ChoiceTile(title: "Model everything" + (recommendMasks ? "" : " — recommended"),
                           subtitle: "No outlines; the whole frame, room included, is the model.",
                           selected: !wantsMasks) {
                    model.subjectKind[project.path] = SubjectKind.scene.rawValue
                }
            }
            if let k = kind, k.usesMasks {
                Text("The subject is \(k.title.lowercased()); change that on Footage.").font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    // MARK: the outlines

    private var previewSheet: String? {
        guard let a = stage?.artifacts.first(where: { $0.hasSuffix(".jpg") }) else { return nil }
        let p = a.hasPrefix("/") ? a : (project.path as NSString).appendingPathComponent(a)
        return FileManager.default.fileExists(atPath: p) ? p : nil
    }

    /// Outlines made before the model they would now be built from: still valid, not the tightest.
    private var olderThanModel: Bool {
        guard let built = files.newest, let trained = manifest?.stage("train")?.finished else { return false }
        return built < trained
    }

    @ViewBuilder private var outlines: some View {
        let s = stage
        if running, queue?.steps.first?.title == "Build masks" {
            VerdictCard(.attention, headline: "Making the outlines…", detail: "The strip at the top follows the run.")
        } else if s?.status == .failed {
            VerdictCard(.blocked, headline: "The outlines could not be made.", detail: s?.error ?? "Open Details for the engine's reason.")
        } else if let n = outlineCount ?? (files.isEmpty ? nil : files.total) {
            VerdictCard(s?.status == .stale ? .attention : .good,
                        headline: "\(n) outlines made" + (snapOn ? " and tightened to the edge." : "."),
                        detail: s?.status == .stale ? "The frames changed since; make the outlines again before training."
                                : (olderThanModel ? "Made before the current model; still valid, make again for the tightest fit." : nil)) {
                HStack(spacing: 8) {
                    if snapOn, let m = snapMoved {
                        MetricChip(text: "Outline moved \(m >= 0 ? "in" : "out") by \(Int(abs(m).rounded())) px (typical) to sit on the edge")
                    } else if let o = snapOffset {
                        MetricChip(text: String(format: "Edge sits %.1f px off", o))
                    }
                    MetricChip(text: glintsCut ? "Glints cut from the outline" : "Glints kept")
                    if fellBack > 0 { MetricChip(text: "\(fellBack) frames used the region instead", tint: .orange) }
                    if let cov = coverageMedian { MetricChip(text: String(format: "%.0f%% of frame", cov * 100)) }
                }
                if let p = previewSheet {
                    ThumbImage(path: p).frame(maxWidth: .infinity).clipShape(RoundedRectangle(cornerRadius: 4))
                    Text("Yellow is the outline; everything outside it is dimmed.").font(.caption).foregroundStyle(.secondary)
                }
            }
        }
    }

    // MARK: the review

    @ViewBuilder private var reviewCard: some View {
        if let r = review {
            let exact = r.exactRepairs
            let n = undecided
            VerdictCard(n == 0 ? .good : .attention,
                        headline: r.newerEngine ? r.headline
                                : (r.summary.flagged == 0 ? "Every outline agrees with the others."
                                   : (n == 0 ? "Every flagged frame is decided." : "\(n) frames need a look.")),
                        detail: n == 0 ? nil : "An outline that leaves out part of the subject, or takes in something else. Training waits until each is decided.") {
                if !r.newerEngine {
                    HStack(spacing: 10) {
                        if n > 0 {
                            Button(exact == 1 ? "Fix the 1 that can be fixed" : "Fix all \(exact) that can be fixed") {
                                decide(MaskReview.exactKey, MaskReview.Choice.repair, title: "Repair masks")
                            }
                            .disabled(blocked || exact == 0)
                            .help("The repairs that need no look; the originals are kept.")
                            Button("Leave the rest out") {
                                decide(MaskReview.undecidedKey, MaskReview.Choice.exclude, title: "Exclude masks")
                            }
                            .disabled(blocked || !r.canExcludeRest)
                            .help("Every undecided frame is left out of training; nothing else changes.")
                        }
                        if !r.flagged.isEmpty {
                            Button(showList ? "Hide the list" : "Look one by one") { showList.toggle() }
                        }
                        Button("Check again") { check() }.disabled(blocked)
                            .help("Checks the outlines on disk again; a decision stays with its outline until that file changes.")
                    }
                }
                if showList && !r.flagged.isEmpty {
                    MaskReviewSection(review: r, blocked: blocked) { view, choice in
                        decide(view, choice, title: choice.title + " " + view)
                    }
                }
            }
        } else if reviewLoaded {
            VerdictCard(.attention, headline: "The outlines have not been checked against each other.",
                        detail: "The check finds outlines that leave out part of the subject; training will not start on unchecked outlines.") {
                Button("Check the outlines") { check() }.disabled(blocked)
            }
        }
    }

    // MARK: the one action

    private var footer: some View {
        Observing(model.projectRuns[project.path]) { _ in
            if !wantsMasks {
                StepFooter(primary: "Go to Train", note: "No outlines; the whole scene is the model.") {
                    model.pipelineStage[project.path] = .train
                }
            } else if files.isEmpty && stage?.status != .done {
                StepFooter(primary: running ? "Working…" : "Make the outlines",
                           primaryEnabled: !blocked,
                           note: model.config.problems.isEmpty ? "Draws the subject's outline in every frame, then checks them against each other." : "Fix Settings first.") {
                    build()
                }
            } else {
                let ready = review != nil && undecided == 0 && stage?.status != .stale
                StepFooter(primary: "Go to Train", primaryEnabled: ready,
                           secondary: running ? "Working…" : "Make the outlines again",
                           note: ready ? nil : (review == nil ? "Check the outlines first." : (undecided > 0 ? "\(undecided) frames still need a decision." : "Make the outlines again first."))) {
                    model.pipelineStage[project.path] = .train
                } secondaryAction: {
                    if blocked { return }
                    if trainDone && !files.isEmpty { confirmRebuild = true } else { build() }
                }
            }
        }
    }

    /// `hs masks` with the panel's settings; the edge snap is on for everything but a person
    /// (hair), and the outlines are checked against each other in the same run.
    private func build() {
        var args = settings.wrappedValue.arguments(project: project.path)
        if settings.wrappedValue.method == .vision && kind != .person { args.append("--snap-edge") }
        run("Build masks", args)
    }

    private func check() {
        run("Check masks", MaskReview.checkOnlyArguments(project: project.path))
    }

    private func decide(_ view: String, _ choice: MaskReview.Choice, title: String) {
        let decisions: [(view: String, choice: MaskReview.Choice)] = [(view: view, choice: choice)]
        run(title, MaskReview.decideArguments(project: project.path, decisions))
    }

    private func run(_ title: String, _ arguments: [String]) {
        guard !arguments.isEmpty else { return }
        let q = RunQueue(config: model.config, steps: [RunQueue.Step(title: title, arguments: arguments)])
        let path = project.path
        q.onStep = { s in model.projectRuns[path] = s }
        q.onFinish = { _ in store.reload() }
        model.maskQueues[path] = q
        q.start()
    }
}
