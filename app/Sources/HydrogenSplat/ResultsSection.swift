import SwiftUI
import AppKit
import UniformTypeIdentifiers
import HSCore

/// After a train with a score (`hs views`): three score cards in words, the same numbers against
/// the previous kept model, what would help most next, and the four things to do with the model.
/// Every number is the engine's (the views stage's metrics and the reports in views/).
struct ResultsSection: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var store: ProjectStore
    let project: ProjectSummary
    let manifest: Manifest
    /// Start the same recipe again (the Train page supplies it).
    var trainAgain: (() -> Void)? = nil

    @State private var exportError: String?

    private var views: StageState? { manifest.stage("views") }
    private var train: StageState? { manifest.stage("train") }
    private var currentMD5: String? { train?.metrics["final_export_md5"]?.string }
    private var archives: [ArchiveInfo] { ArchiveInfo.list(project: project.path) }
    private var currentArchive: ArchiveInfo? {
        guard let md5 = currentMD5 else { return nil }
        return archives.first { $0.plyMD5 == md5 }
    }
    /// The model the scores are about: the archive the latest train was kept as, else the live export.
    private var currentModel: ViewerModelFile? {
        let files = ViewerModelFile.list(project: project.path)
        if let a = currentArchive, let f = files.first(where: { $0.archive == a.name }) { return f }
        return files.first { $0.archive == nil && $0.name == "current" }
    }

    /// The report the latest score wrote (`hs views --name`), else the newest one for the current model.
    private var currentReport: ViewsReportMedians? {
        if let argv = views?.argv, let i = argv.firstIndex(of: "--name"), i + 1 < argv.count,
           let r = ViewsReportMedians.load(project: project.path, name: argv[i + 1]) { return r }
        guard let f = currentModel else { return nil }
        return ViewsScore.list(project: project.path, for: f).first.flatMap { ViewsReportMedians.load(project: project.path, name: $0.name) }
    }

    /// The newest kept model other than this one that has a score of its own.
    private var previousReport: (archive: ArchiveInfo, report: ViewsReportMedians)? {
        let files = ViewerModelFile.list(project: project.path)
        let others = archives.filter { $0.plyMD5 != currentMD5 }.sorted { ($0.archived ?? "") > ($1.archived ?? "") }
        for a in others {
            guard let f = files.first(where: { $0.archive == a.name }),
                  let s = ViewsScore.list(project: project.path, for: f).first,
                  let r = ViewsReportMedians.load(project: project.path, name: s.name) else { continue }
            return (a, r)
        }
        return nil
    }

    var body: some View {
        let m = views?.metrics ?? [:]
        let interior = m["psnr_interior_median"]?.double
        let edge = m["psnr_edge_median"]?.double
        let whole = m["psnr_median"]?.double
        let sharp = m["retained_edge_energy_norm_median"]?.double
        VStack(alignment: .leading, spacing: 14) {
            Text("The scores").font(.headline)
            HStack(alignment: .top, spacing: 10) {
                scoreCard(title: interior != nil ? "Inside the subject" : "Whole frame",
                          value: (interior ?? whole).map { String(format: "%.1f dB", $0) },
                          meaning: interior != nil
                            ? "How closely the model matches the photographs inside the outline. Higher is closer; the change from the last run is what to watch."
                            : "How closely the model matches the photographs over the whole crop. Train with an outline to score the subject alone.")
                scoreCard(title: "Along the outline",
                          value: edge.map { String(format: "%.1f dB", $0) },
                          meaning: edge != nil
                            ? "The same match in a thin band along the outline — where the edge sits and how soft it is."
                            : "Needs an outline: this run was scored without one, so there is no edge band to measure.")
                scoreCard(title: "Sharpness",
                          value: sharp.map { String(format: "%.0f %%", $0 * 100) },
                          meaning: "Fine detail the renders keep, as a share of what the photographs hold at their own grain. 100 % is as sharp as the photographs.")
            }
            compareTable
            nextSteps
            actions
            if let e = exportError { Label(e, systemImage: "exclamationmark.triangle").foregroundStyle(.orange) }
        }
    }

    private func scoreCard(title: String, value: String?, meaning: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title).font(.subheadline.weight(.semibold))
            Text(value ?? "—").font(.system(size: 26, weight: .bold)).monospacedDigit()
            Text(meaning).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .topLeading)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color(nsColor: .separatorColor)))
    }

    // MARK: against the previous kept model

    @ViewBuilder private var compareTable: some View {
        if let cur = currentReport, let prev = previousReport {
            VStack(alignment: .leading, spacing: 6) {
                Text("Against \(prev.archive.name), the kept model before this one").font(.subheadline.weight(.semibold))
                if let a = cur.crop, let b = prev.report.crop, abs(a - b) > 0.5 {
                    Label(String(format: "Scored at different crops (%.0f mm against %.0f mm): the numbers are not comparable.", a, b),
                          systemImage: "exclamationmark.triangle").foregroundStyle(.orange).font(.callout)
                }
                Grid(alignment: .leading, horizontalSpacing: 18, verticalSpacing: 4) {
                    GridRow {
                        Text("").frame(width: 150, alignment: .leading)
                        Text("this run").foregroundStyle(.secondary).gridColumnAlignment(.trailing)
                        Text(prev.archive.name).foregroundStyle(.secondary).gridColumnAlignment(.trailing)
                        Text("change").foregroundStyle(.secondary).gridColumnAlignment(.trailing)
                    }
                    compareRow("Inside the subject", cur.interior ?? cur.whole, prev.report.interior ?? prev.report.whole, "%.2f dB", higherIsBetter: true)
                    compareRow("Along the outline", cur.edge, prev.report.edge, "%.2f dB", higherIsBetter: true)
                    compareRow("Sharpness", cur.sharp.map { $0 * 100 }, prev.report.sharp.map { $0 * 100 }, "%.0f %%", higherIsBetter: true)
                    compareRow("Off the photograph", cur.displaced.map { $0 * 100 }, prev.report.displaced.map { $0 * 100 }, "%.1f %%", higherIsBetter: false)
                    GridRow {
                        Text("views scored").foregroundStyle(.secondary)
                        Text("\(cur.views)").monospacedDigit().gridColumnAlignment(.trailing)
                        Text("\(prev.report.views)").monospacedDigit().gridColumnAlignment(.trailing)
                        Text("")
                    }
                }
                .font(.callout)
            }
        } else if currentReport != nil {
            Text("No earlier kept model has a score in this project, so there is nothing to compare against yet. Keep this one under a name and the next run will be compared with it.")
                .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
    }

    private func compareRow(_ label: String, _ a: Double?, _ b: Double?, _ fmt: String, higherIsBetter: Bool) -> some View {
        GridRow {
            Text(label).foregroundStyle(.secondary)
            Text(a.map { String(format: fmt, $0) } ?? "—").monospacedDigit().gridColumnAlignment(.trailing)
            Text(b.map { String(format: fmt, $0) } ?? "—").monospacedDigit().gridColumnAlignment(.trailing)
            if let x = a, let y = b {
                let d = x - y
                let better = higherIsBetter ? d > 0 : d < 0
                Text((d >= 0 ? "+" : "") + String(format: fmt, d))
                    .monospacedDigit().gridColumnAlignment(.trailing)
                    .foregroundStyle(abs(d) < 1e-9 ? Color.secondary : (better ? Color.green : Color.orange))
            } else {
                Text("—").foregroundStyle(.secondary).gridColumnAlignment(.trailing)
            }
        }
    }

    // MARK: what would help most next

    private var nextLines: [String] {
        var out: [String] = []
        // the engine's own plain-language lines, when the views stage writes them
        for key in ["next_steps", "summary"] {
            if let arr = views?.metrics[key]?.array {
                out += arr.compactMap { $0.string }
            } else if let s = views?.metrics[key]?.string {
                out.append(s)
            }
        }
        let failed = (views?.checks ?? []).filter { !$0.ok } + (train?.checks ?? []).filter { !$0.ok }
        for c in failed where !c.needsHuman || out.isEmpty {
            out.append(ResultsSection.advice(for: c))
        }
        return out
    }

    /// One sentence per failed check, in the user's terms; unknown checks fall back to the engine's words.
    static func advice(for c: CheckResult) -> String {
        switch c.name {
        case "model_registers_to_photographs":
            return "Some views sit off the photographs. Look at the worst view's outline and the frame placement before training longer."
        case "no_view_much_softer_than_achievable":
            return "Some views render softer than the photographs allow. More steps, or a smaller split size under All settings, sharpens them."
        case "every_azimuth_band_registers":
            return "One side of the orbit registers worse than the rest. Add frames from that side."
        case "subject_registers_better_than_its_surroundings":
            return "The room scores better than the subject. Tighten the outline, or hold the model to a scan."
        case "depth_reference_coverage":
            return "The scan covers too little of the subject. Scan closer, or from more sides, and measure it again."
        case "depth_error_held":
            return "The model drifted from the scan. Raise Hold to the scan, or start from the scan."
        case "no_sleep_during_run":
            return "The Mac slept during the run. Keep it awake and plugged in next time."
        case "captures_sharper_than_the_model":
            return "The photographs are sharper than the model. The gain is in training, not the frames: a longer run or a finer split."
        default:
            let v = c.value?.string ?? c.value?.display ?? ""
            return c.name.replacingOccurrences(of: "_", with: " ") + (v.isEmpty ? "" : ": " + v)
        }
    }

    private var nextSteps: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("What would help most next").font(.subheadline.weight(.semibold))
            let lines = nextLines
            if lines.isEmpty {
                Text("Every check passed. The next gain is in the frames: more of the side that scored lowest, then a longer run.")
                    .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            } else {
                ForEach(Array(lines.enumerated()), id: \.offset) { _, l in
                    Label(l, systemImage: "arrow.turn.down.right").font(.callout).fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    // MARK: actions

    private var actions: some View {
        HStack(spacing: 10) {
            Button("Open in the viewer") {
                if let f = currentModel { model.viewerFile[project.path] = f }
                model.projectPage[project.path] = .viewer
            }
            .disabled(currentModel == nil)
            Button("Make a shot") { model.pipelineStage[project.path] = .shot }
            Button("Train again") { trainAgain?() }.disabled(trainAgain == nil)
            Button("Export .ply") { exportPLY() }.disabled(currentModel == nil)
            Spacer()
        }
    }

    private func exportPLY() {
        guard let f = currentModel else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [UTType(filenameExtension: "ply")].compactMap { $0 }
        panel.nameFieldStringValue = "\(project.displayName)-\(f.name).ply"
        panel.message = "A copy of the model's splats. Keep it with the project: a .ply alone cannot be posed against the cameras."
        guard panel.runModal() == .OK, let u = panel.url else { return }
        do {
            if FileManager.default.fileExists(atPath: u.path) { try FileManager.default.removeItem(at: u) }
            try FileManager.default.copyItem(at: URL(fileURLWithPath: f.ply), to: u)
            exportError = nil
        } catch {
            exportError = "Could not write the .ply: \(error.localizedDescription)"
        }
    }
}

/// The medians of one `hs views` report (views/<name>_report.json), read from its per-view rows so
/// an older report without summary metrics compares like a new one.
struct ViewsReportMedians: Equatable {
    let name: String
    let views: Int
    let crop: Double?
    let whole: Double?
    let interior: Double?
    let edge: Double?
    let sharp: Double?
    let displaced: Double?

    static func load(project: String, name: String) -> ViewsReportMedians? {
        let path = ((project as NSString).appendingPathComponent("views") as NSString).appendingPathComponent("\(name)_report.json")
        guard let d = FileManager.default.contents(atPath: path), let v = JSONValue.parse(d),
              let rows = v["views"]?.array else { return nil }
        func med(_ key: String) -> Double? {
            let xs = rows.compactMap { $0[key]?.double }.sorted()
            guard !xs.isEmpty else { return nil }
            return xs.count % 2 == 1 ? xs[xs.count / 2] : (xs[xs.count / 2 - 1] + xs[xs.count / 2]) / 2
        }
        return ViewsReportMedians(name: name, views: rows.count, crop: v["subject_extent_mm"]?.double,
                                  whole: med("psnr_db"), interior: med("psnr_interior_db"), edge: med("psnr_edge_db"),
                                  sharp: med("retained_edge_energy_norm"), displaced: med("displaced_fraction"))
    }
}
