import SwiftUI
import AppKit
import HSCore

/// Step 2 — Frames. Picks the frames (`hs select`) and places the cameras (`hs solve`), then states
/// one verdict: how many frames, how many placed, how well the cameras agree, and where the orbit
/// has a gap (8 directions × 3 heights, from solve/coverage.json).
struct FramesPage: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var store: ProjectStore
    let project: ProjectSummary

    @State private var quality: FrameQuality?
    @State private var coverage: CoverageGrid?
    @State private var estimate: SolveEstimate?
    @State private var estimating = false
    @State private var confirmRedo = false
    @State private var showSettings = false

    private var manifest: Manifest? { project.manifest }
    private var selectStage: StageState? { manifest?.stage("select") }
    private var solveStage: StageState? { manifest?.stage("solve") }
    private var framesOnly: Bool { manifest?.framesOnly ?? false }
    private var selectDone: Bool { framesOnly || selectStage?.status == .done }
    private var solveDone: Bool { solveStage?.status == .done }
    private var lockAlive: Bool { project.lock?.alive == true }
    private var queue: RunQueue? { model.selectQueues[project.path] }

    private var settings: Binding<SelectSettings> {
        Binding(get: { model.selectSettings[project.path] ?? SelectSettings() },
                set: { model.selectSettings[project.path] = $0 })
    }
    private var matcher: Binding<SolveMatcher> {
        Binding(get: { model.solveMatcher[project.path] ?? .auto },
                set: { model.solveMatcher[project.path] = $0 })
    }
    private var trainSettings: Binding<TrainSettings> {
        Binding(get: { model.trainSettings[project.path] ?? TrainSettings() },
                set: { model.trainSettings[project.path] = $0 })
    }

    /// Changes whenever a select or solve run finishes, so the reports are re-read.
    private var reloadKey: String {
        "\(project.path)|\(selectStage?.finished?.timeIntervalSince1970 ?? 0)|\(solveStage?.finished?.timeIntervalSince1970 ?? 0)|\(solveStage?.status.rawValue ?? "")"
    }
    private var estimateKey: String { "\(reloadKey)|\(picked ?? -1)|\(model.config.hsPath)" }

    // MARK: numbers

    /// Frames the solve sees: what select picked, or what ingest put in for an array or a set.
    private var picked: Int? {
        if let n = selectStage?.metrics["frames_selected"]?.int { return n }
        if let q = quality { return q.frames.count }
        if framesOnly, let m = manifest, !m.cameras.isEmpty { return m.cameras.count }
        return nil
    }
    private var registered: Int? { manifest.flatMap { PipelineStage.registered(in: $0) } }

    /// Set when the last solve stopped because some frames could not be placed: how many were
    /// picked and how many were placed. The placed ones can be kept (`acceptPartial`).
    private var partial: (picked: Int, placed: Int)? {
        guard solveStage?.status == .failed, let n = picked, let placed = registered, placed > 0, placed < n else { return nil }
        return (picked: n, placed: placed)
    }

    /// The selector's warning that the picks do not chain (check `picks_overlap_enough`), or nil.
    private var weakLinkWarning: String? {
        guard let c = selectStage?.checks.first(where: { $0.name == "picks_overlap_enough" }), !c.ok else { return nil }
        return c.value?.display
    }

    /// How many unbroken runs the unplaced captures form: cap21 … cap30 is one stretch. A name is
    /// a prefix and the pick's number ("cap021"), or that and the source frame ("sel149-01347").
    static func stretches(_ names: [String]) -> Int {
        let nums: [Int] = names.compactMap { name -> Int? in
            let head = name.split(separator: "-").first.map(String.init) ?? name
            let digits = String(head.reversed().prefix(while: { $0.isNumber }).reversed())
            return Int(digits)
        }.sorted()
        guard var prev = nums.first else { return 0 }
        var count = 1
        for k in nums.dropFirst() {
            if k > prev + 1 { count += 1 }
            prev = k
        }
        return count
    }
    private var reproj: Double? { solveStage?.metrics["mean_reproj_px"]?.double }

    var body: some View {
        StepPage {
            StepHeader(title: "Frames", lead: "Pick the frames and place the cameras. Everything after this is built on them.")
            if manifest == nil {
                Text(project.manifestError ?? "This project has no manifest.").foregroundStyle(.red)
            } else {
                verdict
                if solveDone { coverageCard }
                holdoutCard
                if let q = quality { framesCard(q) }
                if !framesOnly {
                    DisclosureGroup("Adjust how frames are picked", isExpanded: $showSettings) {
                        SelectSettingsPanel(settings: settings).padding(.top, 6)
                    }
                }
                matcherRow
                footer
            }
        }
        .task(id: reloadKey) {
            quality = FrameQuality.load(project: project.path)
            coverage = CoverageGrid.read(project: project.path)
        }
        .task(id: estimateKey) { await loadEstimate() }
        .confirmationDialog("Pick the frames again?", isPresented: $confirmRedo) {
            Button("Pick again", role: .destructive) { startSelect() }
        } message: {
            Text("This replaces the current frames" + (solveDone ? " and the cameras will need placing again." : "."))
        }
    }

    // MARK: verdict

    private var running: Bool { queue?.isRunning ?? false }

    @ViewBuilder private var verdict: some View {
        let stage = solveStage
        if lockAlive || running {
            let what = (project.lock?.stage ?? queue?.current?.stageName ?? "") == "select" ? "Picking frames…" : "Placing cameras…"
            VerdictCard(.attention, headline: what, detail: "The strip at the top follows the run.")
        } else if solveDone, let n = picked {
            let placed = registered ?? n
            let agree = reproj.map { String(format: "Cameras agree to %.1f px.", $0) } ?? ""
            let all = placed >= n
            let failed = stage?.checks.filter { !$0.ok } ?? []
            VerdictCard(all && failed.isEmpty ? .good : .attention,
                        headline: all ? "\(n) frames picked and all \(n) placed. \(agree)"
                                      : "\(n) frames picked, \(placed) placed. \(agree)",
                        detail: verdictDetail(all: all, failed: failed)) {
                HStack(spacing: 8) {
                    if let r = reproj { MetricChip(text: String(format: "%.2f px reprojection", r), tint: r <= 1.8 ? .green : .orange) }
                    if let d = stage?.duration { MetricChip(text: "Placed in " + Format.duration(d)) }
                    if let q = quality, q.flagged > 0 { MetricChip(text: "\(q.flagged) frames flagged", tint: .orange) }
                    if let g = coverage?.gapText { MetricChip(text: g, tint: .orange) }
                }
            }
        } else if let p = partial {
            partialVerdict(picked: p.picked, placed: p.placed)
        } else if stage?.status == .failed {
            VerdictCard(.blocked, headline: "The cameras could not be placed.",
                        detail: stage?.error ?? "Open Details for the engine's reason, then place again.")
        } else if selectDone, let n = picked {
            VerdictCard(.attention, headline: framesOnly ? "\(n) frames, one per camera. Not placed yet."
                                                        : "\(n) frames picked. Not placed yet.",
                        detail: "Place the cameras to see how they agree and where the orbit has gaps.") {
                HStack(spacing: 8) {
                    if let q = quality, let t = q.framesTotal { MetricChip(text: "picked from \(t)") }
                    if let q = quality, q.flagged > 0 { MetricChip(text: "\(q.flagged) frames flagged", tint: .orange) }
                    if let c = estimate?.framesCaption { MetricChip(text: c) }
                }
                if let w = weakLinkWarning {
                    Label(w, systemImage: "exclamationmark.triangle").font(.callout).foregroundStyle(.orange)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        } else {
            VerdictCard(.info, headline: "No frames picked yet.",
                        detail: "The picker takes a frame each time the camera has moved enough, choosing the sharpest one.")
        }
    }

    /// The solve stopped at a partial registration. The frames it did place are sound; say how many,
    /// where the rest went, and what each way on costs. The footer offers to go on with them.
    private func partialVerdict(picked n: Int, placed: Int) -> some View {
        let missing: [String] = solveStage?.metrics["unregistered_captures"]?.array?.compactMap { $0.string } ?? []
        let lost = missing.isEmpty ? n - placed : missing.count
        let runs = FramesPage.stretches(missing)
        let whereText = runs > 0 ? " in \(runs) stretch\(runs == 1 ? "" : "es") of the clip" : ""
        return VerdictCard(.attention,
                           headline: "\(placed) of \(n) frames were placed. \(lost) could not be.",
                           detail: "The solver lost the camera\(whereText): fast movement, blur, or a view with nothing to hold on to. Going on with the \(placed) takes seconds, and the model will have holes where the others looked. Placing again takes as long as it did.") {
            HStack(spacing: 8) {
                if let r = reproj { MetricChip(text: String(format: "the placed ones agree to %.2f px", r), tint: r <= 1.8 ? .green : .orange) }
                if let q = quality, q.flagged > 0 { MetricChip(text: "\(q.flagged) frames flagged", tint: .orange) }
            }
            if let w = weakLinkWarning {
                Label(w, systemImage: "exclamationmark.triangle").font(.callout).foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private func verdictDetail(all: Bool, failed: [CheckResult]) -> String {
        if !all, let s = solveStage {
            let missing = s.metrics["unregistered_captures"]?.array?.count ?? 0
            return missing > 0 ? "\(missing) frames could not be placed; the orbit has a hole there. Training can still go ahead."
                               : "Some frames could not be placed; training can still go ahead."
        }
        if let f = failed.first {
            switch f.name {
            case "mean_reproj_ok": return "The cameras agree less well than they should; softer frames or a wobbly orbit do this. Training will still run."
            case "no_image_over_3px": return "One view sits badly against the others. Training will still run; it may show as a blur from that side."
            default: return "One of the engine's checks did not pass: " + f.name.replacingOccurrences(of: "_", with: " ") + "."
            }
        }
        return coverage?.gapDetail ?? "Nothing to fix here. Go on to Look."
    }

    // MARK: coverage

    private var coverageCard: some View {
        VerdictCard(.info, headline: "Where the frames were taken from",
                    detail: coverage == nil ? "The solve wrote no coverage table; the grid will appear once it does." : coverage?.gapDetail) {
            if let c = coverage {
                CoverageGridView(grid: c)
                HStack(spacing: 8) {
                    MetricChip(text: "\(c.counted) frames in the grid")
                    if c.outside > 0 { MetricChip(text: "\(c.outside) above or below the rings", tint: .orange) }
                    if let r = solveStage?.metrics["azimuth_range_deg"]?.array, r.count == 2 {
                        MetricChip(text: "\(r[0].display)° to \(r[1].display)° around")
                    }
                }
            } else if let r = solveStage?.metrics["azimuth_range_deg"]?.array, r.count == 2 {
                MetricChip(text: "Around the subject: \(r[0].display)° to \(r[1].display)°")
            }
        }
    }

    // MARK: hold-out

    private var holdoutCard: some View {
        let on = trainSettings.wrappedValue.holdoutEvery > 0
        let n = picked ?? 0
        let held = trainSettings.wrappedValue.holdoutCaptures(total: n).count
        return VerdictCard(.info, headline: "Keep some frames out of training to score the model") {
            Toggle(isOn: Binding(get: { on }, set: { v in
                var s = trainSettings.wrappedValue
                s.holdoutEvery = v ? 10 : 0
                trainSettings.wrappedValue = s
            })) {
                Text(n > 0 && on ? "Hold out every 10th frame (\(held) of \(n)); the model is scored on them."
                                 : "Hold out every 10th frame; the model is scored on them.")
            }
            .toggleStyle(.checkbox)
            Text("A scored model tells you whether the next run was better. Held-out frames are not lost; they are just not trained on.")
                .font(.caption).foregroundStyle(.secondary).lineLimit(1)
        }
    }

    // MARK: frames

    private func framesCard(_ q: FrameQuality) -> some View {
        VerdictCard(.info, headline: "The \(q.frames.count) picked frames",
                    detail: q.flagged == 0 ? "None flagged." : "\(q.flagged) flagged for softness, exposure or clipping; orange in the strip.") {
            PickedFramesStrip(project: project.path, quality: q,
                              referenceRole: { referenceRole($0, q) },
                              useAsReference: { useAsReference($0) })
        }
    }

    private var exposureSettings: ExposureSettings { model.exposureSettings[project.path] ?? ExposureSettings() }

    private func referenceRole(_ f: FrameQuality.Frame, _ q: FrameQuality) -> String? {
        let e = exposureSettings
        switch e.reference {
        case .auto: return q.exposureReference?.sel == f.sel ? "exposure reference" : nil
        case .chosen:
            if e.chosenCapture == f.sel { return "exposure reference" }
            return q.exposureReference?.sel == f.sel ? "suggested reference" : nil
        case .median: return q.exposureReference?.sel == f.sel ? "suggested reference" : nil
        }
    }

    private func useAsReference(_ f: FrameQuality.Frame) {
        var e = exposureSettings
        e.reference = .chosen
        e.chosenCapture = f.sel
        model.exposureSettings[project.path] = e
    }

    // MARK: matcher and estimate

    @ViewBuilder private var matcherRow: some View {
        if selectDone && !solveDone {
            StepSetting(title: "Matching", help: estimate?.detail ?? "Sequential for orbits; exhaustive when the camera comes back to the same view much later.") {
                Picker("", selection: matcher) {
                    ForEach(SolveMatcher.allCases) { m in Text(estimate?.label(m) ?? m.title).tag(m) }
                }
                .labelsHidden().pickerStyle(.segmented).fixedSize()
                if estimating && estimate == nil { ProgressView().controlSize(.mini) }
            }
        }
    }

    /// `hs solve -p P --estimate`: no lock, no writes, one event. An engine without it leaves the
    /// matcher without times — never an error.
    private func loadEstimate() async {
        estimate = nil
        estimating = false
        guard selectDone, !solveDone, model.config.problems.isEmpty else { return }
        estimating = true
        let e = await SolveEstimate.load(config: model.config, project: project.path)
        if Task.isCancelled { return }
        estimate = e
        estimating = false
    }

    // MARK: the one action

    private var footer: some View {
        Observing(model.projectRuns[project.path]) { _ in
            let blocked = running || lockAlive || !model.config.problems.isEmpty
            let problem = settings.wrappedValue.problem
            if solveDone {
                StepFooter(primary: "Go to Look", secondary: "Place the cameras again",
                           note: blocked ? "Wait for the running step." : nil) {
                    model.pipelineStage[project.path] = .look
                } secondaryAction: {
                    if !blocked { startSolve() }
                }
            } else if let p = partial {
                StepFooter(primary: "Go on with the \(p.placed) placed",
                           primaryEnabled: !blocked,
                           secondary: "Place the cameras again",
                           note: blocked ? "Wait for the running step." : "Going on keeps what was solved; nothing is solved again.") {
                    acceptPartial()
                } secondaryAction: {
                    if !blocked { startSolve() }
                }
                if !framesOnly {
                    Button("Pick the frames again") { if !blocked { confirmRedo = true } }
                }
            } else if selectDone {
                StepFooter(primary: running ? "Placing…" : "Place the cameras",
                           primaryEnabled: !blocked,
                           secondary: framesOnly ? nil : "Pick the frames again",
                           note: problem ?? (model.config.problems.isEmpty ? "Takes a few minutes; the strip at the top follows it." : "Fix Settings first.")) {
                    startSolve()
                } secondaryAction: {
                    if !blocked { confirmRedo = true }
                }
            } else {
                StepFooter(primary: running ? "Picking…" : "Pick the frames",
                           primaryEnabled: !blocked && problem == nil,
                           note: problem ?? (model.config.problems.isEmpty ? "Then place the cameras." : "Fix Settings first.")) {
                    startSelect()
                }
            }
        }
    }

    private func startSelect() {
        run([RunQueue.Step(title: "Select frames", arguments: settings.wrappedValue.arguments(project: project.path))])
    }

    /// Keep the cameras the last solve placed and write them out for training (`hs solve
    /// --export-only --allow-partial`): seconds, nothing is solved again, and the solve's check
    /// keeps saying which frames are missing.
    private func acceptPartial() {
        run([RunQueue.Step(title: "Keep the placed cameras",
                           arguments: ["solve", "-p", project.path, "--export-only", "--allow-partial"])])
    }

    private func startSolve() {
        run([RunQueue.Step(title: "Solve", arguments: matcher.wrappedValue.solveArguments(project: project.path))])
    }

    private func run(_ steps: [RunQueue.Step]) {
        let q = RunQueue(config: model.config, steps: steps)
        let path = project.path
        q.onStep = { s in model.projectRuns[path] = s }
        q.onFinish = { _ in store.reload() }
        model.selectQueues[path] = q
        q.start()
    }
}

// MARK: - coverage grid

/// solve/coverage.json counted into the 8 × 3 grid `hs views` and the phone guide share
/// (engine/hs/bands.py: 45° azimuth bands from −180°, rings low −25…5°, mid 5…20°, high 20…45°).
struct CoverageGrid {
    static let bandLows: [Int] = Array(stride(from: -180, to: 180, by: 45))
    static let rings: [(name: String, lo: Double, hi: Double)] = [("high", 20, 45), ("mid", 5, 20), ("low", -25, 5)]
    static let directions: [String] = ["behind, left", "back-left", "left", "front-left",
                                       "front-right", "right", "back-right", "behind, right"]

    /// counts[ring][band]
    let counts: [[Int]]
    let outside: Int
    var counted: Int { counts.reduce(0) { $0 + $1.reduce(0, +) } }

    static func read(project: String) -> CoverageGrid? {
        let p = (project as NSString).appendingPathComponent("solve/coverage.json")
        guard let d = FileManager.default.contents(atPath: p), let j = JSONValue.parse(d),
              let caps = j["captures"]?.array, !caps.isEmpty else { return nil }
        var counts = Array(repeating: Array(repeating: 0, count: bandLows.count), count: rings.count)
        var outside = 0
        for c in caps {
            guard let az = c["azimuth_deg"]?.double, let el = c["elevation_deg"]?.double else { continue }
            var a = az.truncatingRemainder(dividingBy: 360)
            if a >= 180 { a -= 360 }
            if a < -180 { a += 360 }
            let b = min(max(Int(floor((a + 180) / 45)), 0), bandLows.count - 1)
            guard let r = rings.firstIndex(where: { el >= $0.lo && el < $0.hi }) else {
                outside += 1
                continue
            }
            counts[r][b] += 1
        }
        return CoverageGrid(counts: counts, outside: outside)
    }

    /// Directions with no frame in any ring.
    var emptyBands: [Int] {
        (0..<CoverageGrid.bandLows.count).filter { b in counts.allSatisfy { $0[b] == 0 } }
    }

    /// Rings with no frame in any direction.
    var emptyRings: [Int] {
        (0..<CoverageGrid.rings.count).filter { counts[$0].allSatisfy { $0 == 0 } }
    }

    /// "no frames from the left" — the gap, named, for a chip; nil when every direction has frames.
    var gapText: String? {
        let e = emptyBands
        guard !e.isEmpty else { return nil }
        if e.count == 1 { return "no frames from \(CoverageGrid.directions[e[0]])" }
        return "\(e.count) of 8 directions have no frames"
    }

    /// The verdict's second line.
    var gapDetail: String {
        var parts: [String] = []
        let e = emptyBands
        if e.isEmpty {
            parts.append("Every direction has frames.")
        } else {
            let names = e.map { CoverageGrid.directions[$0] }
            parts.append("No frames from " + names.joined(separator: ", ") + ": the model will be thin from there.")
        }
        for r in emptyRings {
            let name = CoverageGrid.rings[r].name
            parts.append(name == "high" ? "Nothing from high up." : name == "low" ? "Nothing from low down." : "Nothing from camera height.")
        }
        return parts.joined(separator: " ")
    }
}

/// 8 columns of direction × 3 rows of height, each cell shaded by how many frames it holds.
struct CoverageGridView: View {
    let grid: CoverageGrid

    private var peak: Int { max(grid.counts.flatMap { $0 }.max() ?? 1, 1) }

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            ForEach(0..<CoverageGrid.rings.count, id: \.self) { r in
                HStack(spacing: 3) {
                    Text(CoverageGrid.rings[r].name).font(.caption).foregroundStyle(.secondary)
                        .frame(width: 34, alignment: .trailing)
                    ForEach(0..<CoverageGrid.bandLows.count, id: \.self) { b in
                        let n = grid.counts[r][b]
                        ZStack {
                            RoundedRectangle(cornerRadius: 3)
                                .fill(n == 0 ? Color.orange.opacity(0.18) : Brand.accent.opacity(0.2 + 0.6 * Double(n) / Double(peak)))
                            Text(n == 0 ? "–" : String(n)).font(.caption2.monospacedDigit())
                                .foregroundStyle(n == 0 ? Color.orange : Color.primary)
                        }
                        .frame(width: 56, height: 26)
                        .help("\(CoverageGrid.directions[b]), \(CoverageGrid.rings[r].name): \(n) frames")
                    }
                }
            }
            HStack(spacing: 3) {
                Text("").frame(width: 34)
                ForEach(0..<CoverageGrid.bandLows.count, id: \.self) { b in
                    Text("\(CoverageGrid.bandLows[b])°").font(.caption2).foregroundStyle(.secondary)
                        .frame(width: 56)
                }
            }
            Text("Columns: direction around the subject, front at 0°. Rows: camera height. Orange: no frame from there.")
                .font(.caption).foregroundStyle(.secondary).lineLimit(1)
        }
    }
}
