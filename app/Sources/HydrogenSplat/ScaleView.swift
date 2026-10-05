import SwiftUI
import AppKit
import UniformTypeIdentifiers
import HSCore

/// The old Scale box's name, kept so an older mount still compiles: the LiDAR half of the
/// Calibrate door is `LidarScaleCard`.
struct ScaleView: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var store: ProjectStore
    let project: ProjectSummary
    let manifest: Manifest

    var body: some View {
        LidarScaleCard(project: project, manifest: manifest)
    }
}

/// A phone LiDAR scan measured against the solve and applied (`hs scale --lidar`), as one card:
/// the fit in mm, the share of the solve on the scan, the hole, and the buttons — Replace the
/// scan, Measure, Apply, See it over the frames. The last measurement's verdict comes from the
/// manifest (`stages.scale.lidar_check`), so a scan measured from Terminal shows here too. A
/// scale from a board (`hs scale --board`) or a factor already known stays under a disclosure.
struct LidarScaleCard: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var store: ProjectStore
    let project: ProjectSummary
    let manifest: Manifest

    @State private var showOther = false
    /// The apply about to run, once confirmed: its step title and arguments.
    private struct Pending: Equatable { let title: String; let arguments: [String] }
    @State private var confirmApply: Pending?
    @State private var copyError: String?
    @State private var boardAcross = 7
    @State private var boardDown = 5
    @State private var boardSquareMM = 35.0
    @State private var boardMarkerMM = 26.0

    private var settings: Binding<ScaleSettings> {
        Binding(get: { model.scaleSettings[project.path] ?? ScaleSettings() },
                set: { model.scaleSettings[project.path] = $0 })
    }
    private var queue: RunQueue? { model.scaleQueues[project.path] }
    private var running: Bool { queue?.isRunning ?? false }
    private var lockAlive: Bool { project.lock?.alive == true }
    /// A Hydrogen clip: the solve calls itself metric, so applying needs --trust-scan.
    private var stereo: Bool { !manifest.isArray }
    private var check: LidarCheck? { LidarCheck(manifest: manifest) }
    private var applied: AppliedScale? { AppliedScale(manifest: manifest) }
    private var busy: Bool { running || lockAlive || !model.config.problems.isEmpty }
    private var solveDone: Bool { manifest.stage("solve")?.status == .done }

    private var verdict: Verdict {
        guard let c = check else { return .info }
        if !c.aligned { return .blocked }
        if c.applied { return c.problems.isEmpty ? .good : .attention }
        return .attention
    }

    private var headline: String {
        if !solveDone { return "Place the frames first — the scan is measured against the solved cameras." }
        guard let c = check else { return "No scan has been measured against this solve yet." }
        return c.verdict
    }

    private var detail: String {
        if let c = check {
            if !c.aligned { return "Measure again with the scan placed by Silhouettes, or scan the ground and the backdrop the cameras see, not only the subject." }
            if c.stereo, let s = c.scale, abs(s - 1) > 0.02, !c.applied {
                return "To make this project metric: turn on \"Trust the scan\" and Apply."
            }
            return c.applied ? "The training set is in millimetres; Train can start from the scan and be held to it."
                             : "Measured, not applied. Apply writes the scan's scale into the training set."
        }
        return stereo
            ? "A Hydrogen solve is metric to about a metre from its 10.6 mm baseline; past that the baseline is too small to see, so the scan is the scale. Measure first; apply with \"Trust the scan\"."
            : "An array or one-camera solve has no scale of its own. A phone LiDAR scan of the place, taken right before the shoot, gives it one — and the ground plane and which way is up."
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            VerdictCard(verdict, headline: headline, detail: detail) {
                if let c = check { chips(c) }
                if let a = applied {
                    Label(a.sentence, systemImage: "checkmark.seal").foregroundStyle(.green)
                        .fixedSize(horizontal: false, vertical: true)
                }
                scanRow
                methodRow
                actionRow
                if let c = check {
                    ForEach(c.problems.filter { !$0.hasPrefix("lidar aligned") }, id: \.self) { p in
                        Label(p, systemImage: "exclamationmark.triangle")
                            .font(.caption).foregroundStyle(.orange)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                if let q = queue, q.state == .failed, let e = q.current?.errors.last {
                    Label((e.message ?? "failed") + (e.hint.map { " — " + $0 } ?? ""), systemImage: "xmark.octagon")
                        .foregroundStyle(.red)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if let e = copyError {
                    Label(e, systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
                }
                if let q = queue, let cur = q.current, q.isRunning {
                    RunPanel(session: cur, showMetrics: false)
                }
            }
            otherSection
        }
        .confirmationDialog("Apply this scale to the solve?", isPresented: Binding(get: { confirmApply != nil }, set: { if !$0 { confirmApply = nil } })) {
            Button("Apply", role: .destructive) { if let a = confirmApply { run(a.title, a.arguments) }; confirmApply = nil }
        } message: {
            Text("The training set is rewritten in the new units. Train, Move, Render and Views go stale and need running again; kept models keep their old units.")
        }
    }

    // MARK: the numbers

    /// Fit, coverage and the hole, each as the engine measured it.
    private func chips(_ c: LidarCheck) -> some View {
        HStack(spacing: 8) {
            if let r = c.rmsMM {
                MetricChip(text: String(format: "fit %.0f mm", r), tint: r > 30 ? Color.orange : nil)
            } else {
                MetricChip(text: "fit: measured once the scan aligns")
            }
            if let f = c.inlierFraction {
                MetricChip(text: String(format: "%.0f %% of the solve on the scan", f * 100), tint: f < 0.5 ? Color.orange : nil)
            }
            if let n = c.capturesOffScan {
                MetricChip(text: n == 0 ? "every capture inside the scan" : "\(n) capture\(n == 1 ? "" : "s") outside the scan",
                           tint: n == 0 ? Color.green : Color.orange)
            }
            if let cov = manifest.stage("train")?.metrics["depth"]?["coverage"]?.double {
                MetricChip(text: String(format: "the scan sees %.0f %% of the subject", cov * 100), tint: cov < 0.7 ? Color.orange : nil)
            } else if c.aligned {
                MetricChip(text: "hole: measured by the next train with the scan")
            }
        }
    }

    // MARK: scan

    private var scanRow: some View {
        Handle(title: "Scan",
               help: "Scaniverse on an iPhone Pro: Mesh capture, process in Detail, export PLY. Scan the ground and the backdrop the cameras see, not only the subject.") {
            Button(settings.wrappedValue.scan == nil ? (check?.scanName == nil ? "Choose the scan…" : "Replace the scan…") : "Replace the scan…") { chooseScan() }
                .disabled(busy)
            if let s = settings.wrappedValue.scan {
                Text((s as NSString).lastPathComponent).monospaced()
            } else if let c = check, let n = c.scanName {
                Button("Use \(n) again") { adoptLastScan(c) }.controlSize(.small)
                    .help("The scan the last measurement used")
            }
            Picker("", selection: settings.scanUp) {
                ForEach(ScanUp.allCases) { Text($0.title).tag($0) }
            }
            .labelsHidden().fixedSize()
            .help("Which axis of the file is up. Auto reads it off the scan; say so only if it gets it wrong.")
        }
    }

    private var methodRow: some View {
        Handle(title: "Place it by",
               help: "Geometry matches the shapes the scan and the solve share. Silhouettes projects the scan's subject through the cameras against the outlines (run Subject first) — for a subject the point cloud barely holds.") {
            Picker("", selection: settings.scanInit) {
                ForEach(ScanInit.allCases) { Text($0.title).tag($0) }
            }
            .labelsHidden().pickerStyle(.segmented).fixedSize()
            Toggle("Write init points for Train", isOn: settings.initPoints)
                .help("Also write scale/lidar_init.ply: the scan in the solve's frame as Brush's starting splats, and the depth Train can be held to.")
            if stereo {
                Toggle("Trust the scan", isOn: settings.trustScan)
                    .help("Apply the scan's scale over the Hydrogen baseline (hs scale --trust-scan). Measuring never needs this; applying does.")
            }
        }
    }

    private var actionRow: some View {
        VStack(alignment: .leading, spacing: 6) {
            let s = settings.wrappedValue
            HStack(spacing: 12) {
                Button(running && queue?.steps.first?.title == "Measure scan" ? "Measuring…" : "Measure the scan") {
                    run("Measure scan", s.measureArguments(project: project.path))
                }
                .disabled(busy || s.scanProblem != nil || !solveDone)
                .help("hs scale --lidar SCAN --dry-run: align the scan to the solve and report; nothing in the project changes.")
                Button(running && queue?.steps.first?.title == "Apply scale" ? "Applying…" : "Apply the scan's scale") {
                    confirmApply = Pending(title: "Apply scale", arguments: s.applyScanArguments(project: project.path))
                }
                .disabled(busy || s.scanProblem != nil || !solveDone || (stereo && !s.trustScan))
                .help(stereo && !s.trustScan ? "Turn on \"Trust the scan\" to apply on a Hydrogen project" : "hs scale --lidar SCAN: align and apply the scan's scale")
                Button("See it over the frames") { seeOverFrames() }
                    .disabled(check == nil)
                    .help("The scan's subject projected through the cameras over the photographs (scale/silhouette_overlay_*.jpg), else the aligned scan itself")
                Button("Open report") { openInProject("scale/lidar_report.json") }.controlSize(.small)
                    .disabled(check == nil)
                if let p = s.scanProblem {
                    Text(p).foregroundStyle(.secondary)
                } else if lockAlive, let l = project.lock {
                    Label("\(l.stage ?? "a stage") is running", systemImage: "lock.fill").foregroundStyle(.secondary)
                }
            }
            if s.scan != nil {
                commandPreview(s.measureArguments(project: project.path), title: "Measure runs")
            }
        }
    }

    // MARK: a board, or a factor already known

    private var boardSpec: String {
        "\(boardAcross),\(boardDown),\(String(format: "%g", boardSquareMM)),\(String(format: "%g", boardMarkerMM))"
    }

    private var otherSection: some View {
        DisclosureGroup("Metric scale from a board, or a factor you already have", isExpanded: $showOther) {
            VStack(alignment: .leading, spacing: 10) {
                Handle(title: "Board",
                       help: "A ChArUco board seen in the training frames: squares across and down, printed square and marker side in mm (hs scale --board).") {
                    Stepper(value: $boardAcross, in: 3...20) { Text("\(boardAcross) across").monospacedDigit() }
                    Stepper(value: $boardDown, in: 3...20) { Text("\(boardDown) down").monospacedDigit() }
                    TextField("square mm", value: $boardSquareMM, format: .number).frame(width: 70)
                    TextField("marker mm", value: $boardMarkerMM, format: .number).frame(width: 70)
                }
                HStack(spacing: 12) {
                    Button("Measure the board") { run("Measure board", ["scale", "-p", project.path, "--board", boardSpec, "--dry-run"]) }
                        .disabled(busy || !solveDone)
                    Button("Apply the board's scale") {
                        confirmApply = Pending(title: "Apply board", arguments: ["scale", "-p", project.path, "--board", boardSpec] + (stereo && settings.wrappedValue.trustScan ? ["--trust-scan"] : []))
                    }
                    .disabled(busy || !solveDone || (stereo && !settings.wrappedValue.trustScan))
                }
                commandPreview(["scale", "-p", project.path, "--board", boardSpec, "--dry-run"], title: "Measure runs")
                Divider()
                Handle(title: "Factor",
                       help: "Solve units × this = millimetres, from a tape measure or a fit made outside the app.") {
                    TextField("factor", value: settings.factor, format: .number)
                        .textFieldStyle(.roundedBorder).frame(width: 100)
                    TextField("where it came from", text: settings.note)
                        .textFieldStyle(.roundedBorder).frame(width: 260)
                }
                let s = settings.wrappedValue
                HStack(spacing: 12) {
                    Button(running && queue?.steps.first?.title == "Apply factor" ? "Applying…" : "Apply the factor") {
                        confirmApply = Pending(title: "Apply factor", arguments: s.applyFactorArguments(project: project.path))
                    }
                    .disabled(busy || s.factorProblem != nil || !solveDone || (stereo && !s.trustScan))
                    .help(stereo && !s.trustScan ? "Turn on \"Trust the scan\" above to apply on a Hydrogen project" : "hs scale --factor F")
                    if let p = s.factorProblem { Text(p).foregroundStyle(.secondary) }
                }
                if s.factorProblem == nil {
                    commandPreview(s.applyFactorArguments(project: project.path), title: "Runs")
                }
            }
            .padding(.top, 6)
        }
    }

    // MARK: helpers

    private func commandPreview(_ args: [String], title: String) -> some View {
        let line = (["hs"] + args.map { $0.hasPrefix(project.path) ? $0.replacingOccurrences(of: project.path, with: "$P") : $0 })
            .map(shellQuote).joined(separator: " ")
        return VStack(alignment: .leading, spacing: 4) {
            Text("\(title) (P = this project):").font(.caption).foregroundStyle(.secondary)
            CopyableCommand(text: line)
        }
    }

    private func chooseScan() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [UTType(filenameExtension: "ply"), UTType(filenameExtension: "obj"),
                                     UTType(filenameExtension: "xyz"), UTType(filenameExtension: "txt"),
                                     UTType(filenameExtension: "csv"), UTType(filenameExtension: "pts"),
                                     UTType(filenameExtension: "e57"), UTType(filenameExtension: "las"),
                                     UTType(filenameExtension: "laz")].compactMap { $0 }
        panel.message = "A LiDAR scan of the scene: PLY, OBJ or XYZ from a phone, E57 or LAS from a scanner"
        guard panel.runModal() == .OK, let u = panel.url else { return }
        copyError = nil
        let dir = (project.path as NSString).appendingPathComponent("lidar")
        let dest = (dir as NSString).appendingPathComponent(u.lastPathComponent)
        do {
            try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
            if u.path != dest {
                if FileManager.default.fileExists(atPath: dest) { try FileManager.default.removeItem(atPath: dest) }
                try FileManager.default.copyItem(at: u, to: URL(fileURLWithPath: dest))
            }
            settings.wrappedValue.scan = "lidar/" + u.lastPathComponent
        } catch {
            copyError = "Could not copy the scan into the project: \(error.localizedDescription)"
        }
    }

    /// The scan the last measurement used, when its file is still there.
    private func adoptLastScan(_ c: LidarCheck) {
        guard let s = c.scan else { return }
        let rel = s.hasPrefix(project.path + "/") ? String(s.dropFirst(project.path.count + 1)) : s
        if FileManager.default.fileExists(atPath: s.hasPrefix("/") ? s : (project.path as NSString).appendingPathComponent(s)) {
            settings.wrappedValue.scan = rel
        } else {
            copyError = "The last scan (\((s as NSString).lastPathComponent)) is no longer where it was; choose it again."
        }
    }

    /// The silhouette overlays the engine wrote, else the aligned scan for the viewer / CloudCompare.
    private func seeOverFrames() {
        let dir = (project.path as NSString).appendingPathComponent("scale")
        let overlays = ((try? FileManager.default.contentsOfDirectory(atPath: dir)) ?? [])
            .filter { $0.hasPrefix("silhouette_overlay_") && $0.hasSuffix(".jpg") }.sorted()
        if let first = overlays.first {
            NSWorkspace.shared.open(URL(fileURLWithPath: (dir as NSString).appendingPathComponent(first)))
        } else {
            revealInProject("scale/lidar_aligned.ply")
        }
    }

    private func openInProject(_ rel: String) {
        NSWorkspace.shared.open(URL(fileURLWithPath: (project.path as NSString).appendingPathComponent(rel)))
    }

    private func revealInProject(_ rel: String) {
        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: (project.path as NSString).appendingPathComponent(rel))])
    }

    private func run(_ title: String, _ args: [String]) {
        guard !args.isEmpty else { return }
        let q = RunQueue(config: model.config, steps: [RunQueue.Step(title: title, arguments: args)])
        let path = project.path
        q.onStep = { s in model.projectRuns[path] = s }
        q.onFinish = { _ in store.reload() }
        model.scaleQueues[path] = q
        q.start()
    }
}
