import SwiftUI
import AppKit
import UniformTypeIdentifiers
import HSCore

/// The Scale box on the Frames page, under Solve: a phone LiDAR scan measured against the solve
/// and applied (`hs scale --lidar`), or a factor found some other way (`hs scale --factor`).
/// Shows once a solve exists. The last measurement's verdict comes from the manifest
/// (`stages.scale.lidar_check`), so a scan measured from Terminal shows here too.
struct ScaleView: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var store: ProjectStore
    let project: ProjectSummary
    let manifest: Manifest

    @State private var showFactor = false
    @State private var confirmApply: [String]?     // the arguments of the apply about to run
    @State private var copyError: String?

    private var settings: Binding<ScaleSettings> {
        Binding(get: { model.scaleSettings[project.path] ?? ScaleSettings() },
                set: { model.scaleSettings[project.path] = $0 })
    }
    private var queue: RunQueue? { model.scaleQueues[project.path] }
    private var running: Bool { queue?.isRunning ?? false }
    private var lockAlive: Bool { project.lock?.alive == true }
    private var scaleStage: StageState? { manifest.stage("scale") }
    private var solveDone: Bool { manifest.stage("solve")?.status == .done }
    /// A Hydrogen clip: the solve calls itself metric, so applying needs --trust-scan.
    private var stereo: Bool { !manifest.isArray }
    private var check: LidarCheck? { LidarCheck(manifest: manifest) }
    private var applied: AppliedScale? { AppliedScale(manifest: manifest) }
    private var busy: Bool { running || lockAlive || !model.config.problems.isEmpty }

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 14) {
                Text(stereo
                     ? "A Hydrogen solve calls itself metric from the 10.6 mm baseline, and that holds within about a metre. Further away the baseline is too small to see — the Circles solve came out 5.8× small at 4 m — so past a metre the scan is the scale. Measure first; apply with \"trust the scan\"."
                     : "An array or mono solve has no scale of its own. A phone LiDAR scan of the place, taken right before the shoot, gives it one — and the ground plane and which way is up.")
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                scanRow
                trustRow
                measureRow
                if let c = check { report(c) }
                if let a = applied {
                    Label(a.sentence, systemImage: "checkmark.seal")
                        .foregroundStyle(.green)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Divider()
                factorSection
                if let e = copyError {
                    Label(e, systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
                }
            }
            .padding(4)
            .frame(maxWidth: .infinity, alignment: .leading)
        } label: {
            HStack {
                Text("Scale").font(.headline)
                if let s = scaleStage { StatusBadge(status: s.status) }
            }
        }
        .confirmationDialog("Apply this scale to the solve?", isPresented: Binding(get: { confirmApply != nil }, set: { if !$0 { confirmApply = nil } })) {
            Button("Apply", role: .destructive) { if let a = confirmApply { run("Apply scale", a) }; confirmApply = nil }
        } message: {
            Text("The training set is rewritten in the new units. Train, Move, Render and Views go stale and need running again; archives keep their old units.")
        }
    }

    // MARK: scan

    private var scanRow: some View {
        Handle(title: "Scan",
               help: "Scaniverse Classic on an iPhone Pro: Mesh capture, Large Object / Area, process in Detail, Share → Export Model → PLY. PLY, OBJ or XYZ; copied into the project's lidar/ folder. Scan the ground and the backdrop the cameras see, not only the subject.") {
            Button(settings.wrappedValue.scan == nil ? "Choose scan…" : "Change…") { chooseScan() }
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

    @ViewBuilder private var trustRow: some View {
        if stereo {
            Toggle("Trust the scan over the H1 baseline", isOn: settings.trustScan)
                .help("Apply the scan's scale to a Hydrogen solve (hs scale --trust-scan). Measuring never needs this; applying does.")
        }
    }

    private var measureRow: some View {
        VStack(alignment: .leading, spacing: 6) {
            let s = settings.wrappedValue
            HStack(spacing: 12) {
                Button(running && queue?.steps.first?.title == "Measure scan" ? "Measuring…" : "Measure") {
                    run("Measure scan", s.measureArguments(project: project.path))
                }
                .disabled(busy || s.scanProblem != nil)
                .help("hs scale --lidar SCAN --dry-run: align the scan to the solve and report; nothing in the project changes.")
                Button(running && queue?.steps.first?.title == "Apply scale" ? "Applying…" : "Apply scan scale") {
                    confirmApply = s.applyScanArguments(project: project.path)
                }
                .disabled(busy || s.scanProblem != nil || (stereo && !s.trustScan))
                .help(stereo && !s.trustScan ? "Turn on \"Trust the scan\" to apply on a Hydrogen project" : "hs scale --lidar SCAN: align and apply the scan's scale")
                if let p = s.scanProblem {
                    Text(p).foregroundStyle(.secondary)
                } else if lockAlive, let l = project.lock {
                    Label("\(l.stage ?? "a stage") is running", systemImage: "lock.fill").foregroundStyle(.secondary)
                }
            }
            if s.scan != nil {
                commandPreview(s.measureArguments(project: project.path), title: "Measure runs")
            }
            if let q = queue, q.state == .failed, let e = q.current?.errors.last {
                // a refused alignment exits 1 with the reason and a hint: the report below has the numbers
                Label((e.message ?? "failed") + (e.hint.map { " — " + $0 } ?? ""), systemImage: "xmark.octagon")
                    .foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    // MARK: the last measurement

    private func report(_ c: LidarCheck) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text("Last measurement").font(.subheadline.weight(.semibold))
                if let n = c.scanName { Text(n).monospaced().foregroundStyle(.secondary) }
                if let d = c.at { Text(d.formatted(date: .abbreviated, time: .shortened)).foregroundStyle(.secondary) }
            }
            Label(c.verdict, systemImage: c.aligned ? (c.applied ? "checkmark.seal" : "checkmark.circle") : "xmark.circle")
                .foregroundStyle(c.aligned ? Color.primary : Color.orange)
                .fixedSize(horizontal: false, vertical: true)
            if c.aligned, let s = c.scale, c.stereo, abs(s - 1) > 0.02, !c.applied {
                Text("To make this project metric: turn on \"Trust the scan\" and press Apply scan scale, or enter \(String(format: "%.3f", s)) as a factor below.")
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            ForEach(c.facts, id: \.self) { f in
                Text("· " + f).font(.caption).foregroundStyle(.secondary)
            }
            ForEach(c.problems.filter { !$0.hasPrefix("lidar aligned") }, id: \.self) { p in
                Label(p, systemImage: "exclamationmark.triangle")
                    .font(.caption).foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack(spacing: 10) {
                Button("Open report") { openInProject("scale/lidar_report.json") }.controlSize(.small)
                Button("Reveal aligned scan") { revealInProject("scale/lidar_aligned.ply") }.controlSize(.small)
                    .help("The scan in the solve's frame, 100,000 points, for the viewer or CloudCompare")
            }
        }
        .padding(8)
        .background(RoundedRectangle(cornerRadius: 6).fill(Color.secondary.opacity(0.06)))
    }

    // MARK: a known factor

    private var factorSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            DisclosureGroup("A factor you already have", isExpanded: $showFactor) {
                VStack(alignment: .leading, spacing: 8) {
                    Handle(title: "Factor",
                           help: "Solve units × this = millimetres. From a tape measure (a length in the solve, in mm, against the real one), or a fit made outside the app — the ring-silhouette fit gave Circles 5.81.") {
                        TextField("5.81", value: settings.factor, format: .number)
                            .textFieldStyle(.roundedBorder).frame(width: 100)
                        TextField("where it came from", text: settings.note)
                            .textFieldStyle(.roundedBorder).frame(width: 260)
                    }
                    let s = settings.wrappedValue
                    HStack(spacing: 12) {
                        Button(running && queue?.steps.first?.title == "Apply factor" ? "Applying…" : "Apply factor") {
                            confirmApply = s.applyFactorArguments(project: project.path)
                        }
                        .disabled(busy || s.factorProblem != nil || (stereo && !s.trustScan))
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
                                     UTType(filenameExtension: "csv"), UTType(filenameExtension: "pts")].compactMap { $0 }
        panel.message = "A phone LiDAR scan of the scene (PLY, OBJ or XYZ)"
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
