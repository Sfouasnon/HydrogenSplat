import SwiftUI
import HSCore

/// Step 5, Train: three choices (what is in the shot · how long · score it), the run in words,
/// then the scores (`ResultsSection`). Everything else — every control the old panel had — sits
/// under "All settings". The run mechanics are unchanged: `TrainSettings` → `RunQueue`
/// (train → archive → views), a Terminal train followed from its log.
struct TrainStepPage: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var store: ProjectStore
    let project: ProjectSummary

    var body: some View {
        if let m = project.manifest {
            TrainStepBody(project: project, manifest: m)
        } else {
            StepPage {
                StepHeader(title: "Train", lead: "Build the model from the frames.")
                VerdictCard(.blocked, headline: "This project cannot be read.",
                            detail: project.manifestError ?? "Its manifest is missing or damaged.")
            }
        }
    }
}

private struct TrainStepBody: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var store: ProjectStore
    let project: ProjectSummary
    let manifest: Manifest

    @State private var showAll = false
    @State private var confirmReplace = false
    @State private var lock: ProjectSummary.LockInfo?
    @State private var onBattery = PowerStatus.onBattery
    /// masks_review/review.json, whether mask files exist and the captures the hold-out is taken
    /// from — read when the masks stage changes, never in body.
    @State private var maskReview: MaskReview?
    @State private var maskFilesExist = false
    @State private var maskCaptures = CaptureSet(names: [], stereo: true)
    /// How long to spend. Read back from the settings on appear (`TrainEffort.matching`), so a
    /// hand edit under All settings shows as no tile selected rather than a wrong one.
    @State private var effort: TrainEffort? = .standard
    private let tick = Timer.publish(every: 3, on: .main, in: .common).autoconnect()

    // MARK: settings

    private var settings: Binding<TrainSettings> {
        Binding(get: { model.trainSettings[project.path] ?? defaults },
                set: { model.trainSettings[project.path] = $0 })
    }

    /// What the page starts from: the recipe for this project's subject at Standard effort,
    /// scored at the crop the last views run used.
    private var defaults: TrainSettings {
        var base = TrainSettings()
        if let mm = manifest.stage("views")?.metrics["subject_extent_mm"]?.double { base.subjectMM = mm }
        var s = TrainSettings.recipe(kind: kind, effort: .standard, hasScan: lidarInitAvailable,
                                     sourceLongEdge: sourceLongEdge, base: base)
        if !hasMasks { s.layer = .full }
        return s
    }

    /// What is in the shot, chosen on the Footage step (AppModel.subjectKind until the engine
    /// records it with `hs source --subject`).
    private var kind: SubjectKind {
        SubjectKind(rawValue: model.subjectKind[project.path] ?? "glossy") ?? .glossy
    }

    private func choose(kind k: SubjectKind) {
        model.subjectKind[project.path] = k.rawValue
        applyRecipe(kind: k, effort: effort ?? .standard)
    }

    private func choose(effort e: TrainEffort) {
        effort = e
        applyRecipe(kind: kind, effort: e)
    }

    private func applyRecipe(kind k: SubjectKind, effort e: TrainEffort) {
        var s = TrainSettings.recipe(kind: k, effort: e, hasScan: lidarInitAvailable,
                                     sourceLongEdge: sourceLongEdge, base: settings.wrappedValue)
        if !hasMasks { s.layer = .full }
        settings.wrappedValue = s
    }

    /// The training images' long edge: a Hydrogen clip is 1920 per eye; anything else as probed.
    private var sourceLongEdge: Int? {
        if manifest.sourceKind == nil { return 1920 }
        guard let w = manifest.probe["width"]?.int, let h = manifest.probe["height"]?.int else { return nil }
        return max(w, h)
    }

    private var captureSet: CaptureSet { CaptureSet.read(project: project.path) }
    private var captures: Int { captureSet.count }

    /// scale/lidar_init.ply recorded by `hs scale --lidar --init-points` (applied, or the last dry run)
    private var lidarInitAvailable: Bool {
        let rel = manifest.raw["scale"]?["init_ply"]?.string
            ?? manifest.raw["stages"]?["scale"]?["lidar_check"]?["metrics"]?["init_ply"]?.string
        guard let r = rel else { return false }
        return FileManager.default.fileExists(atPath: (project.path as NSString).appendingPathComponent(r))
    }

    private var hasMasks: Bool {
        FileManager.default.fileExists(atPath: (project.path as NSString).appendingPathComponent("train/dataset/masks"))
    }

    /// Crop sizes every previous views report in this project used: a score is only comparable
    /// with another at the same crop.
    private var previousCrops: [Double] {
        let dir = (project.path as NSString).appendingPathComponent("views")
        let files = (try? FileManager.default.contentsOfDirectory(atPath: dir)) ?? []
        var out = Set<Double>()
        for f in files where f.hasSuffix("_report.json") {
            let p = (dir as NSString).appendingPathComponent(f)
            if let d = FileManager.default.contents(atPath: p),
               let mm = JSONValue.parse(d)?["subject_extent_mm"]?.double {
                out.insert(mm)
            }
        }
        return out.sorted()
    }

    private var maskReviewKey: String {
        let s = manifest.stage("masks")
        let metric = s?.metrics["mask_review"]?.compactJSON ?? ""
        let running = model.maskQueues[project.path]?.isRunning ?? false
        let solved = manifest.stage("solve")?.finished?.timeIntervalSince1970 ?? 0
        let attrs = try? FileManager.default.attributesOfItem(atPath: MaskReview.reportPath(project: project.path))
        let stamp = (attrs?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        return "\(project.path)|\(s?.finished?.timeIntervalSince1970 ?? 0)|\(s?.status.rawValue ?? "")|\(metric)|\(running)|\(solved)|\(stamp)"
    }

    /// `hs train` refuses to start on outlines that were never checked, or while a flagged view it
    /// would train on has no decision. Only a layer that reads the outlines is affected.
    private var maskReviewWarning: String? {
        let s = settings.wrappedValue
        guard s.layer.needsMasks else { return nil }
        let excluded = s.excludedViews(in: maskCaptures)
        return MaskReview.trainWarning(maskReview, masksExist: maskFilesExist, excluded: excluded)
    }

    // MARK: run state

    private var solveDone: Bool { manifest.stage("solve")?.status == .done }
    private var queue: RunQueue? { model.trainQueues[project.path] }
    private var activeLock: ProjectSummary.LockInfo? {
        [lock, project.lock].compactMap { $0 }.first { $0.alive } ?? lock ?? project.lock
    }
    private var externalTrain: Bool {
        guard let l = activeLock, l.alive, l.stage == "train" else { return false }
        return !(queue?.isRunning ?? false)
    }
    private var otherStageRunning: String? {
        guard let l = activeLock, l.alive, l.stage != "train", !(queue?.isRunning ?? false) else { return nil }
        return l.stage ?? "another stage"
    }
    private var running: Bool { (queue?.isRunning ?? false) || externalTrain }

    private var archives: [ArchiveInfo] { ArchiveInfo.list(project: project.path) }
    private var currentMD5: String? { manifest.stage("train")?.metrics["final_export_md5"]?.string }
    private var currentArchive: ArchiveInfo? {
        guard let md5 = currentMD5 else { return nil }
        return archives.first { $0.plyMD5 == md5 }
    }

    private var steps: [RunQueue.Step] {
        settings.wrappedValue.steps(project: project.path, in: captureSet)
    }

    private var nameBlocked: Bool {
        let s = settings.wrappedValue
        return s.archiveNameProblem != nil || archives.contains { $0.name == s.archiveName }
    }

    private var startBlocked: Bool {
        running || nameBlocked || captures == 0 || !model.config.problems.isEmpty || activeLock?.alive == true || !solveDone
    }

    // MARK: body

    var body: some View {
        StepPage {
            StepHeader(title: "Train", lead: "Say what is in the shot and how long to spend; the model is built from the frames and scored on photographs it never saw.")
            if !solveDone {
                VerdictCard(.blocked, headline: "The frames are not placed yet.",
                            detail: "Training needs the solved frames. Finish the Frames step first.")
            } else {
                statusCard
                if !running {
                    choices
                    allSettings
                }
                if !running, let t = manifest.stage("train"), t.status == .done,
                   let v = manifest.stage("views"), v.status == .done {
                    ResultsSection(project: project, manifest: manifest, trainAgain: { startOrAsk() })
                }
                footer
            }
        }
        .onAppear {
            refresh()
            if model.trainSettings[project.path] == nil { model.trainSettings[project.path] = defaults }
            effort = TrainEffort.matching(settings.wrappedValue)
        }
        .onReceive(tick) { _ in refresh() }
        .task(id: maskReviewKey) {
            maskCaptures = CaptureSet.read(project: project.path)
            maskFilesExist = !MaskFiles.read(project: project.path).isEmpty
            maskReview = MaskReview.read(project: project.path)
        }
        .confirmationDialog("Train without keeping the result?", isPresented: $confirmReplace) {
            Button("Train without keeping it", role: .destructive) { start() }
        } message: {
            Text((currentMD5 != nil && currentArchive == nil
                  ? "The current model is not kept under a name and training replaces it. "
                  : "")
                 + "No name is set under All settings, so the model this run makes will be replaced by the next run.")
        }
    }

    private func refresh() {
        lock = ProjectStore.readLock(project.path)
        onBattery = PowerStatus.onBattery
    }

    // MARK: the verdict

    @ViewBuilder private var statusCard: some View {
        if externalTrain, let l = activeLock {
            VerdictCard(.attention, headline: "Training is running in another window.") {
                ExternalTrainView(project: project, manifest: manifest, lock: l)
            }
        } else if let q = queue, q.isRunning || q.state == .failed || q.state == .cancelled {
            let s = settings.wrappedValue
            // a run the user stopped is not a failure: its exports are on disk and open in Shot
            let stopped = q.state == .cancelled
            VerdictCard(q.isRunning || stopped ? .attention : .blocked,
                        headline: q.isRunning ? "Training is running." : (stopped ? "Training was stopped." : "Training failed."),
                        detail: q.isRunning ? "Keep the Mac awake and plugged in; the page follows the run."
                            : (stopped ? "The last model it saved is kept and opens in Shot. Training again starts from the beginning."
                                       : "The run's last words are below. Fix what it names and start again.")) {
                QueueView(queue: q, totalIters: s.totalIters, growthStopIter: s.growthStopIter,
                          depthHeld: s.depthWeight > 0, spreadOn: s.depthSpreadWeight > 0)
            }
        } else if let t = manifest.stage("train"), t.status == .done {
            let m = t.metrics
            let splats = m["final_splats"]?.display ?? "?"
            let took = m["elapsed_s"]?.double.map { " in " + TrainProgressCard.spoken($0) } ?? ""
            let kept = currentArchive.map { "Kept as \($0.name)." } ?? (currentMD5 != nil ? "Not kept under a name — the next run replaces it." : "")
            VerdictCard(.good, headline: "Trained: \(splats) splats\(took).", detail: kept.isEmpty ? nil : kept) {
                let pts = GrowthCurve.fromManifest(manifest)
                if !pts.isEmpty {
                    let total = pts.last.map { Int($0.done) } ?? 0
                    GrowthCurveChart(points: pts, totalIters: max(total, settings.wrappedValue.totalIters), height: 90)
                }
            }
        } else if let t = manifest.stage("train"), t.status == .failed {
            VerdictCard(.blocked, headline: "The last training run failed.", detail: t.error ?? "Its log says why (logs/train.log).")
        } else {
            let held = settings.wrappedValue.holdoutCaptures(total: captures).count
            let views = captureSet.views - held * (captureSet.stereo ? 2 : 1)
            VerdictCard(.info, headline: "Ready to train on \(views) views" + (held > 0 ? ", with \(held) \(captureSet.noun)s held back for the score." : "."),
                        detail: "Pick the three choices below and start. Nothing here is final: the model can be trained again.")
        }
    }

    // MARK: the three choices

    private var choices: some View {
        VStack(alignment: .leading, spacing: 18) {
            VStack(alignment: .leading, spacing: 8) {
                Text("What is in the shot").font(.headline)
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 150), spacing: 8)], spacing: 8) {
                    ForEach(SubjectKind.allCases) { k in
                        ChoiceTile(title: k.title, subtitle: k.usesMasks ? "Cut to the outline" : "No outline",
                                   selected: k == kind) { choose(kind: k) }
                    }
                }
                Text(kind.meaning).font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                if kind.usesMasks && !hasMasks {
                    Label("This project has no outlines yet, so the whole scene trains. Make them on the Subject step to cut the model to the outline.",
                          systemImage: "info.circle")
                        .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
                if let w = maskReviewWarning {
                    Label(w, systemImage: "exclamationmark.triangle")
                        .font(.caption).foregroundStyle(.orange).fixedSize(horizontal: false, vertical: true)
                }
            }
            VStack(alignment: .leading, spacing: 8) {
                Text("How long to spend").font(.headline)
                HStack(spacing: 8) {
                    ForEach(TrainEffort.allCases) { e in
                        ChoiceTile(title: e.title, subtitle: "about \(e.minutes) min", selected: e == effort) { choose(effort: e) }
                    }
                }
                Text((effort ?? .standard).meaning).font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                if effort == nil {
                    Text("The schedule under All settings matches none of the three; pick one to go back to it.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            VStack(alignment: .leading, spacing: 6) {
                Toggle(isOn: Binding(get: { settings.wrappedValue.scoreViews }, set: { on in
                    var s = settings.wrappedValue
                    s.scoreViews = on
                    if on && s.holdoutEvery == 0 { s.holdoutEvery = 10 }
                    if !on { s.holdoutEvery = 0 }
                    settings.wrappedValue = s
                })) {
                    Text("Score it").font(.headline)
                }
                let held = settings.wrappedValue.holdoutCaptures(total: captures)
                if settings.wrappedValue.scoreViews {
                    Text("Keeps \(held.count) of \(captures) \(captureSet.noun)s out of training and scores the model on them when it is done. Keep this the same between runs you compare.")
                        .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                } else {
                    Text("Every \(captureSet.noun) trains; there will be no score to compare this run by.")
                        .font(.callout).foregroundStyle(.secondary)
                }
            }
        }
    }

    // MARK: all settings

    private var allSettings: some View {
        DisclosureGroup("All settings", isExpanded: $showAll) {
            VStack(alignment: .leading, spacing: 10) {
                Handle(title: "Length", help: "Total steps. Time grows with the splat count, not only the step count.") {
                    Stepper(value: settings.totalIters, in: 1_000...100_000, step: 5_000) {
                        Text("\(settings.wrappedValue.totalIters) steps").monospacedDigit()
                    }
                }
                Handle(title: "Stop growing at", help: "No new splats after this step; the rest only refines.") {
                    Picker("", selection: settings.growthStopIter) {
                        Text("15,000").tag(15_000)
                        Text("20,000").tag(20_000)
                        Text("30,000").tag(30_000)
                    }
                    .labelsHidden().frame(width: 200)
                }
                Handle(title: "Resolution", help: "Long edge the training images are capped at. 1920 scores like full size in half the time.") {
                    Picker("", selection: Binding(get: { settings.wrappedValue.maxResolution ?? 0 },
                                                  set: { settings.wrappedValue.maxResolution = $0 == 0 ? nil : $0 })) {
                        Text("1920").tag(1920)
                        Text("2400").tag(2400)
                        Text("3840").tag(3840)
                        Text("Brush default (1920)").tag(0)
                    }
                    .labelsHidden().frame(width: 200)
                }
                Handle(title: "Hold out", help: "Which \(captureSet.noun)s are kept out of training for the score. Keep it the same between runs you compare.") {
                    Picker("", selection: settings.holdoutEvery) {
                        Text("Nothing").tag(0)
                        Text("Every 10th \(captureSet.noun)").tag(10)
                        Text("Every 5th \(captureSet.noun)").tag(5)
                    }
                    .labelsHidden().frame(width: 200)
                }
                Handle(title: "Layer", help: "Full scene trains without the outline; Subject uses it; Background is the same outline inverted.") {
                    Picker("", selection: settings.layer) {
                        ForEach(Layer.allCases) { Text($0.title).tag($0) }
                    }
                    .labelsHidden().pickerStyle(.segmented).frame(width: 320)
                    .disabled(!hasMasks)
                }
                if settings.wrappedValue.layer == .subject {
                    Handle(title: "Outside the outline", help: "Left unsupervised: the room is ignored, not removed. Pushed empty: the model holds nothing outside the outline.") {
                        Picker("", selection: settings.alphaMode) {
                            ForEach(AlphaMode.allCases) { Text($0.title).tag($0) }
                        }
                        .labelsHidden().pickerStyle(.segmented).frame(width: 320)
                    }
                }
                Handle(title: "Refine every", help: "Steps between split / prune passes. Larger means fewer, calmer growth steps.") {
                    Stepper(value: settings.refineEvery, in: 50...500, step: 10) {
                        Text("\(settings.wrappedValue.refineEvery)").monospacedDigit()
                    }
                }
                Handle(title: "Split big splats", help: "Splats covering more than this share of the image are split; lower makes more, smaller splats.") {
                    OptionalNumber(value: settings.splitAtScreenSize, placeholder: "0.5 default", choices: [0.25, 0.1])
                }
                Handle(title: "Anti-aliasing", help: "Minimum splat width. Reduces shimmer on fine detail; may soften the finest edges.") {
                    Picker("", selection: settings.minScaleFactor) {
                        Text("Off (0)").tag(0.0)
                        Text("Light (0.03)").tag(0.03)
                        Text("Default (0.1)").tag(0.1)
                        Text("Strong (0.3)").tag(0.3)
                    }
                    .labelsHidden().frame(width: 200)
                }
                Handle(title: "Background noise", help: "Noise on the background colour pushes splats to cover every pixel; try 0 when a real background ghosts.") {
                    OptionalNumber(value: settings.backgroundNoise, placeholder: "0.1 default", choices: [0])
                }
                if lidarInitAvailable {
                    Toggle("Start from the LiDAR scan", isOn: settings.initFromLidar)
                        .help("Brush's first splats are the scan's points, so the subject exists from step 0 instead of being grown from nothing.")
                    Handle(title: "Hold to the scan", help: "Weight of the depth term against the registered scan. 0 = off.") {
                        Slider(value: settings.depthWeight, in: 0...1, step: 0.05).frame(width: 200)
                        Text(String(format: "%.2f", settings.wrappedValue.depthWeight)).monospacedDigit()
                    }
                } else {
                    Text("No registered scan with init points in this project: the model starts from the solve's points and is not held to a scan. Calibrate → LiDAR adds one.")
                        .font(.caption).foregroundStyle(.secondary).padding(.leading, 140).fixedSize(horizontal: false, vertical: true)
                }
                Handle(title: "Thin the surface", help: "Weight of the per-ray spread term; thins a smoky surface. 0 = off.") {
                    Slider(value: settings.depthSpreadWeight, in: 0...1, step: 0.05).frame(width: 200)
                    Text(String(format: "%.2f", settings.wrappedValue.depthSpreadWeight)).monospacedDigit()
                }
                Handle(title: "Leave out views", help: "Extra views to exclude, e.g. a bad photograph: L/cap064,R/cap069.") {
                    TextField("L/cap064,R/cap069", text: settings.excludeExtra).textFieldStyle(.roundedBorder).frame(maxWidth: 320)
                }
                Handle(title: "Brush arguments", help: "Passed to Brush unchanged, for flags the app has no control for.") {
                    TextField("--opac-decay 0.006", text: settings.extraBrushArgs).textFieldStyle(.roundedBorder).frame(maxWidth: 320)
                }
                Divider()
                Handle(title: "Keep it as", help: "Copies the finished model and its frame to archive/<name>, so the next run does not replace it.") {
                    TextField("name", text: settings.archiveName).textFieldStyle(.roundedBorder).frame(width: 200)
                    if let p = settings.wrappedValue.archiveNameProblem {
                        Text(p).foregroundStyle(.red)
                    } else if archives.contains(where: { $0.name == settings.wrappedValue.archiveName }) {
                        Text("already exists — pick a new name").foregroundStyle(.red)
                    } else if settings.wrappedValue.archiveName.trimmingCharacters(in: .whitespaces).isEmpty {
                        Text("empty — the next run replaces this one").foregroundStyle(.orange)
                    }
                }
                Handle(title: "Score with", help: "The crop the score is measured in. It must match between runs you compare.") {
                    Toggle("both eyes", isOn: settings.scoreBothEyes)
                        .disabled(!settings.wrappedValue.scoreViews || !captureSet.stereo)
                    OptionalNumber(value: settings.subjectMM, placeholder: "crop mm (auto)", choices: [350, 2000])
                        .disabled(!settings.wrappedValue.scoreViews)
                    if settings.wrappedValue.scoreViews {
                        let prev = previousCrops
                        let cur = settings.wrappedValue.subjectMM
                        let list = prev.map { JSONValue.number($0).display }.joined(separator: ", ")
                        if !prev.isEmpty, let c = cur, !prev.contains(c) {
                            Label("earlier runs here scored at \(list) mm — this one will not be comparable",
                                  systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
                        } else if !prev.isEmpty, cur == nil {
                            Label("auto measures the crop from the point cloud; earlier runs here used \(list) mm",
                                  systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
                        }
                    }
                }
                commandPreview
            }
            .padding(.top, 8)
        }
    }

    private var commandPreview: some View {
        let lines = steps.map { (["hs"] + $0.arguments.map { $0.hasPrefix(project.path) ? "$P" : $0 })
            .map(shellQuote).joined(separator: " ") }
        return VStack(alignment: .leading, spacing: 4) {
            Text("Runs, in order (P = this project):").font(.caption).foregroundStyle(.secondary)
            CopyableCommand(text: lines.joined(separator: " && \\\n  "))
        }
    }

    // MARK: footer

    private var footerNote: String {
        if let st = otherStageRunning { return "\(st) is running on this project — start when it finishes." }
        if running { return "Training; the page follows it." }
        if onBattery { return "On battery — the Mac will sleep and stall training. Plug in." }
        return "Plugged in. Keep the lid open."
    }

    private var footer: some View {
        StepFooter(primary: running ? "Training…" : "Start training",
                   primaryEnabled: !startBlocked,
                   secondary: (queue?.isRunning ?? false) ? "Stop" : nil,
                   note: footerNote,
                   primaryAction: { startOrAsk() },
                   secondaryAction: { queue?.cancel() })
    }

    /// Ask first when nothing will be kept: the queue is built now, so a name typed later is
    /// ignored and the model this run makes is replaced by the next one.
    private func startOrAsk() {
        guard !startBlocked else { return }
        if settings.wrappedValue.archiveName.trimmingCharacters(in: .whitespaces).isEmpty {
            confirmReplace = true
        } else {
            start()
        }
    }

    private func start() {
        let q = RunQueue(config: model.config, steps: steps)
        let path = project.path
        q.onStep = { s in model.projectRuns[path] = s }
        q.onFinish = { _ in store.reload() }
        model.trainQueues[path] = q
        q.start()
    }
}
