import SwiftUI
import AppKit
import UniformTypeIdentifiers
import HSCore

/// The Calibrate door: two cards. Lens — which camera the footage came from, whether a lens
/// profile exists for it, and the three steps to make one (show or print the board, film it,
/// drop the clip: `hs calibrate -p P --clip CLIP`), with the verdict from the calibrate stage.
/// LiDAR — the scan measured against the solve (`LidarScaleCard`), with the board / factor
/// scale under its disclosure.
struct CalibrateStepPage: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var store: ProjectStore
    let project: ProjectSummary

    @State private var showBoard = false
    @State private var boardImage: NSImage?
    @State private var boardRun: RunSession?
    @State private var dropTargeted = false
    @State private var note: String?

    private var manifest: Manifest? { project.manifest }
    private var queue: RunQueue? { model.scaleQueues[project.path] }
    private var busy: Bool { (queue?.isRunning ?? false) || project.lock?.alive == true || !model.config.problems.isEmpty }

    var body: some View {
        StepPage {
            StepHeader(title: "Calibrate", lead: "Two things the solve cannot find on its own: the lens, from a board you film; and the scale, from a LiDAR scan of the place.")
            lensCard
            if let m = manifest {
                LidarScaleCard(project: project, manifest: m)
            } else {
                VerdictCard(.info, headline: "LiDAR: add footage first.", detail: "The scan is measured against the solved cameras, so a project needs frames before it can take a scan.")
            }
        }
        .sheet(isPresented: $showBoard) {
            LensBoardView(engineImage: boardImage) { showBoard = false }
        }
        .onAppear(perform: loadBoardImage)
    }

    // MARK: lens

    /// The camera, as the calibrate stage or the project's source says it.
    private var cameraIdentity: String {
        guard let m = manifest else { return "No footage yet" }
        if let cam = calibrateStage?.metrics["camera"], let make = cam["make"]?.string, let model = cam["model"]?.string {
            let lens = cam["lens"]?.string.map { $0 == "default" ? "" : " · \($0)" } ?? ""
            let size = calibrateStage?.metrics["image_size"]?.array.flatMap { a -> String? in
                guard a.count == 2, let w = a[0].int, let h = a[1].int else { return nil }
                return " · \(w)×\(h)"
            } ?? ""
            return "\(make) \(model)\(lens)\(size)"
        }
        let w = m.probe["width"]?.int, h = m.probe["height"]?.int
        let size = (w != nil && h != nil) ? " · \(w ?? 0)×\(h ?? 0)" : ""
        switch m.sourceKind {
        case nil:
            return "RED Hydrogen One · 2×1 stereo, 1920×1080 per eye"
        case "array":
            return "Camera array · \(m.cameras.count) cameras\(size)"
        default:
            let codec = m.probe["codec"]?.string.map { " \($0)" } ?? ""
            return "One camera\(size)\(codec)" + (m.clipName.map { " · \($0)" } ?? "")
        }
    }

    private var calibrateStage: StageState? { manifest?.stage("calibrate") }

    /// The lens profile this project knows: the one `hs calibrate -p P` wrote into the manifest
    /// (`lens_profile`: path, key, rms_px), else the one the last solve found (`lens_profile`
    /// metric, an object; the string "none" when it used no profile).
    private var lensProfile: (path: String, rms: Double?)? {
        guard let m = manifest else { return nil }
        if let p = m.raw["lens_profile"]?["path"]?.string {
            return (p, m.raw["lens_profile"]?["rms_px"]?.double)
        }
        if let v = m.stage("solve")?.metrics["lens_profile"], let p = v["path"]?.string {
            return (p, v["rms_px"]?.double)
        }
        return nil
    }

    /// The fit of the last board clip, from the calibrate stage.
    private var lensRMS: Double? {
        calibrateStage?.metrics["rms_px"]?.double ?? lensProfile?.rms
    }

    private var lensVerdict: (Verdict, String, String) {
        guard let m = manifest else {
            return (.info, "No footage yet.", "Drop footage on the Footage step; the camera it came from shows here.")
        }
        if m.sourceKind == nil {
            return (.good, "Calibrated: the Hydrogen's stereo profile \(m.profileID ?? "") is applied.",
                    "The phone's two eyes are calibrated as a pair; nothing to film for this project.")
        }
        if let c = calibrateStage, c.status == .failed {
            return (.blocked, "The board clip did not calibrate.", c.error ?? "The calibrate stage's checks below say why.")
        }
        if calibrateStage?.status == .done || lensProfile != nil {
            let rms = lensRMS.map { String(format: " Corners land within %.2f px.", $0) } ?? ""
            let which = lensProfile.map { " (\(($0.path as NSString).lastPathComponent))" } ?? ""
            return (.good, "Calibrated: a lens profile matches this camera\(which).\(rms)",
                    "The solve uses it instead of guessing the lens. Film the board again only if the lens or the recording size changes.")
        }
        return (.attention, "Not calibrated: the solve guesses this camera's lens.",
                "A board clip takes ten minutes and makes every solve of this camera tighter. Three steps:")
    }

    private var lensCard: some View {
        let v = lensVerdict
        return VerdictCard(v.0, headline: v.1, detail: v.2) {
            HStack(spacing: 8) {
                MetricChip(text: cameraIdentity)
                if let c = calibrateStage, c.status == .done {
                    if let n = c.metrics["frames_used"]?.int { MetricChip(text: "\(n) board frames used") }
                    if let cov = c.metrics["coverage_share"]?.double {
                        MetricChip(text: String(format: "corners in %.0f %% of the frame", cov * 100), tint: cov < 0.7 ? Color.orange : nil)
                    }
                    if let f = c.metrics["hfov_deg"]?.double { MetricChip(text: String(format: "%.0f° across", f)) }
                }
            }
            VStack(alignment: .leading, spacing: 4) {
                Label("Show the board on this screen, or print the PDF at 100 %.", systemImage: "1.circle")
                Label("Film it with the camera and settings of the shot: 10–20 s, the board a third of the frame, tilted through every corner.", systemImage: "2.circle")
                Label("Drop the clip below. The profile is kept for this camera, lens and recording size.", systemImage: "3.circle")
            }
            .font(.callout)
            HStack(spacing: 10) {
                Button("Show the board full screen") { showBoard = true }
                Button("Save as PDF") { savePDF() }
                Observing(boardRun) { r in
                    if r.isRunning { ProgressView().controlSize(.small) }
                }
            }
            dropZone
            if let c = calibrateStage, !c.checks.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(c.checks) { ch in
                        CheckRow(name: ch.name.replacingOccurrences(of: "_", with: " "), ok: ch.ok, value: ch.value?.display, needsHuman: ch.needsHuman)
                    }
                }
            }
            if let q = queue, q.isRunning, let cur = q.current, cur.stageName == "calibrate" {
                RunPanel(session: cur, showMetrics: false)
            }
            if let n = note {
                Label(n, systemImage: "info.circle").font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var dropZone: some View {
        VStack(spacing: 6) {
            Image(systemName: "film").font(.title2).foregroundStyle(.secondary)
            Text(busy ? "Busy — drop the clip when the run finishes" : "Drop the board clip here, or click to choose it")
                .foregroundStyle(.secondary)
            Text("hs calibrate -p P --clip CLIP").font(.caption.monospaced()).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, minHeight: 90)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color.secondary.opacity(dropTargeted ? 0.18 : 0.06)))
        .overlay(RoundedRectangle(cornerRadius: 10)
            .stroke(dropTargeted ? Brand.accent : Color(nsColor: .separatorColor), style: StrokeStyle(lineWidth: 1, dash: [6, 4])))
        .contentShape(Rectangle())
        .onTapGesture { chooseClip() }
        .dropDestination(for: URL.self) { urls, _ in
            guard let u = urls.first else { return false }
            calibrate(clip: u.path)
            return true
        } isTargeted: { dropTargeted = $0 }
        .disabled(busy || manifest == nil)
    }

    // MARK: actions

    /// The engine's board, when `hs calibrate --board-image` exists: kept under Application
    /// Support, made once. A failure leaves the drawn board and says the markers come from the engine.
    private func loadBoardImage() {
        let path = LensBoard.enginePNGPath
        if let img = NSImage(contentsOfFile: path) {
            boardImage = img
            return
        }
        guard boardRun == nil, model.config.problems.isEmpty else { return }
        let s = model.session("Board image", LensBoard.imageArguments(output: path))
        s.onFinish = { run in
            if run.succeeded, let img = NSImage(contentsOfFile: path) {
                boardImage = img
            } else {
                note = "The engine could not draw the board (hs calibrate --board-image), so the drawn chessboard stands in; its markers come from the engine."
            }
        }
        boardRun = s
        s.start()
    }

    private func savePDF() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [UTType.pdf]
        panel.nameFieldStringValue = "lens-board-\(LensBoard.squaresX)x\(LensBoard.squaresY)-\(Int(LensBoard.squareMM))mm.pdf"
        panel.message = "Print at 100 % — the squares are \(Int(LensBoard.squareMM)) mm."
        guard panel.runModal() == .OK, let u = panel.url else { return }
        if boardImage != nil, model.config.problems.isEmpty {
            // the engine draws the PDF with real markers, at exact size on a Letter and an A4 page
            let s = model.session("Board PDF", LensBoard.pdfArguments(output: u.path))
            s.onFinish = { run in
                if run.succeeded {
                    NSWorkspace.shared.activateFileViewerSelecting([u])
                } else {
                    writeDrawnPDF(u)
                }
            }
            boardRun = s
            s.start()
        } else {
            writeDrawnPDF(u)
        }
    }

    private func writeDrawnPDF(_ u: URL) {
        do {
            try LensBoard.writePDF(to: u)
            note = "Saved a drawn board: the grey squares stand where the engine's markers go. The engine's board (hs calibrate --board-image) is the one to print for a calibration."
            NSWorkspace.shared.activateFileViewerSelecting([u])
        } catch {
            note = "Could not write the PDF: \(error.localizedDescription)"
        }
    }

    private func chooseClip() {
        guard !busy, manifest != nil else { return }
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [UTType.movie, UTType.mpeg4Movie, UTType.quickTimeMovie, UTType.video]
        panel.message = "The clip of the board, filmed with the camera and settings of the shot"
        guard panel.runModal() == .OK, let u = panel.url else { return }
        calibrate(clip: u.path)
    }

    /// `hs calibrate -p P --clip CLIP`: the lens profile for this camera, lens and recording size.
    private func calibrate(clip: String) {
        guard !busy else { return }
        let args = ["calibrate", "-p", project.path, "--clip", clip]
        let q = RunQueue(config: model.config, steps: [RunQueue.Step(title: "Calibrate", arguments: args)])
        let path = project.path
        q.onStep = { s in model.projectRuns[path] = s }
        q.onFinish = { _ in store.reload() }
        model.scaleQueues[path] = q
        q.start()
    }
}
