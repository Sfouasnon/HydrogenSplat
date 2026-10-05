import SwiftUI
import AppKit
import ImageIO
import HSCore

/// The Render & grade half of the Shot step, below its check chips: the graded preview of one
/// rendered frame (`hs grade --still`, through the same 256-entry tables the export bakes with),
/// a filmstrip to pick the frame, then three cards — Frame (`--aspect`, `--headroom-mm`), Grade
/// (`--lift / --gamma / --gain`, their `-rgb` trims, `--sharpen`, Reset = `--reset`) and Export
/// (`hs render --width`, `hs grade [--crf]`, `hs grade --still N --graded`).
struct GradeView: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var store: ProjectStore
    let project: ProjectSummary

    @State private var moves: [RenderedMove] = []
    @State private var moveName: String = ""
    @State private var frame: Double = 0
    @State private var settings = GradeSettings()
    @State private var still: CGImage?
    @State private var graded: CGImage?
    @State private var showBefore = false
    @State private var perChannel = false
    @State private var stillRun: RunSession?
    @State private var exportRun: RunSession?
    @State private var stillFrame: Int?
    @State private var pendingFrame: Int?
    /// Reset was pressed: the next export passes `--reset` so the saved look is ignored too.
    @State private var resetPending = false
    /// The 4:5 tile: the portrait cut Render writes beside the 16:9; it is not graded here.
    @State private var portrait = false
    @State private var crf = 17
    @AppStorage("render.width") private var renderWidth = 2400

    private var move: RenderedMove? { moves.first { $0.name == moveName } }
    private var renderQueue: RunQueue? { model.moveQueues[project.path] }
    private var lockAlive: Bool { project.lock?.alive == true }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            if moves.isEmpty {
                VerdictCard(.info, headline: "Nothing is rendered yet.",
                            detail: "Render a move on the Move half; its frames come here to be framed, graded and exported.")
            } else {
                header
                preview
                filmstrip
                frameCard
                gradeCard
                exportCard
                if let r = exportRun {
                    Observing(r) { run in
                        if run.isRunning || run.finishedAt.map({ Date().timeIntervalSince($0) < 600 }) == true {
                            RunPanel(session: run, showMetrics: false)
                        }
                    }
                }
            }
        }
        .onAppear(perform: reload)
        .onChange(of: settings) { _, _ in regrade() }
    }

    // MARK: header, preview, filmstrip

    private var header: some View {
        HStack {
            Picker("Rendered move", selection: $moveName) {
                ForEach(moves) { m in Text(m.name).tag(m.name) }
            }
            .frame(maxWidth: 320)
            .onChange(of: moveName) { _, _ in selectMove() }
            if let m = move {
                Text("\(m.frames) frames").foregroundStyle(.secondary)
                Label(m.followsHead ? "the crop follows the head" : "the crop is centred",
                      systemImage: m.followsHead ? "person.crop.rectangle" : "rectangle.dashed")
                    .foregroundStyle(.secondary)
                    .help(m.followsHead ? "This move carries a head track, so the crop keeps the head under the top edge."
                                        : "Without a head track the crop stays centred.")
            }
            Spacer()
            Toggle("Before", isOn: $showBefore).toggleStyle(.button)
                .help("Hold the ungraded frame")
                .disabled(portrait)
        }
    }

    @ViewBuilder private var preview: some View {
        if portrait, let m = move {
            let p45 = ((project.path as NSString).appendingPathComponent("render") as NSString)
                .appendingPathComponent("\(m.name)_1080x1350.mp4")
            if FileManager.default.fileExists(atPath: p45) {
                PlayerView(url: URL(fileURLWithPath: p45))
                    .aspectRatio(4.0 / 5.0, contentMode: .fit)
                    .frame(maxHeight: 480)
                    .clipShape(RoundedRectangle(cornerRadius: 6))
            } else {
                VerdictCard(.attention, headline: "This move has no 4:5 cut.",
                            detail: "Render it again with the 4:5 cut on (the Export card) to get one.")
            }
        } else {
            ZStack {
                Rectangle().fill(Color.black)
                if let img = showBefore ? still : (graded ?? still) {
                    Image(decorative: img, scale: 1).resizable().aspectRatio(contentMode: .fit)
                }
                Observing(stillRun) { r in
                    if r.isRunning { ProgressView() }
                    if case .finished(let code) = r.state, code != 0 {
                        Text(r.errors.first?.message ?? "could not read the frame").foregroundStyle(.red)
                    }
                }
            }
            .aspectRatio(settings.aspect > 0 ? settings.aspect : 16.0 / 9.0, contentMode: .fit)
            .frame(maxWidth: .infinity)
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .overlay(alignment: .bottomLeading) {
                Text(showBefore ? "before" : "after · sharpening is applied on export")
                    .font(.caption).padding(4).background(.black.opacity(0.5)).foregroundStyle(.white).padding(6)
            }
        }
    }

    /// Five stops through the move plus a slider: each loads that frame through `hs grade --still`.
    private var filmstrip: some View {
        let n = max((move?.frames ?? 1), 1)
        let stops: [(String, Int)] = [("First", 0), ("Quarter", (n - 1) / 4), ("Middle", (n - 1) / 2),
                                      ("Three quarters", (n - 1) * 3 / 4), ("Last", n - 1)]
        return VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                ForEach(Array(stops.enumerated()), id: \.offset) { _, s in
                    Button(s.0) { frame = Double(s.1); loadStill(s.1) }
                        .controlSize(.small)
                        .disabled(portrait)
                }
                Spacer()
                Text("frame \(Int(frame)) of \(n)").monospacedDigit().foregroundStyle(.secondary)
            }
            Slider(value: $frame, in: 0...Double(max(n - 1, 1)), step: 1) { editing in
                if !editing { loadStill(Int(frame)) }
            }
            .disabled(portrait)
        }
    }

    // MARK: Frame

    private enum FrameChoice: String, CaseIterable, Identifiable {
        case scope, wide, portrait, full
        var id: String { rawValue }
        var title: String {
            switch self {
            case .scope: return "2.35"
            case .wide: return "16:9"
            case .portrait: return "4:5"
            case .full: return "Full frame"
            }
        }
        var subtitle: String {
            switch self {
            case .scope: return "A wide band, the height following the head"
            case .wide: return "The render's own shape, trimmed to 1920×1076"
            case .portrait: return "The portrait cut Render writes beside the 16:9"
            case .full: return "Nothing cropped"
            }
        }
        var aspect: Double {
            switch self {
            case .scope: return 2.35
            case .wide: return 1920.0 / 1076.0
            case .portrait, .full: return 0
            }
        }
    }

    private var frameChoice: FrameChoice {
        if portrait { return .portrait }
        if settings.aspect <= 0 { return .full }
        if abs(settings.aspect - 2.35) < 1e-6 { return .scope }
        if abs(settings.aspect - 1920.0 / 1076.0) < 1e-6 { return .wide }
        return .full
    }

    private func choose(_ c: FrameChoice) {
        portrait = c == .portrait
        if c != .portrait {
            settings.aspect = c.aspect
            loadStill(Int(frame))
        }
    }

    private var frameCard: some View {
        VerdictCard(.info, headline: "Frame", detail: "The crop `hs grade` cuts from the 16:9 render, full width, the height following the head when the move has a track.") {
            HStack(spacing: 8) {
                ForEach(FrameChoice.allCases) { c in
                    ChoiceTile(title: c.title, subtitle: c.subtitle, selected: c == frameChoice) { choose(c) }
                }
            }
            if portrait {
                Text("Grade works on the 16:9 render. The 4:5 cut is not graded here: export the look on the 16:9 and cut 4:5 from it, or render again with the 4:5 cut on.")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            HStack {
                Text("Headroom")
                TextField("", value: $settings.headroomMM, format: .number.precision(.fractionLength(1)))
                    .frame(width: 60)
                    .onSubmit { loadStill(Int(frame)) }
                Text("mm above the crown").foregroundStyle(.secondary)
                Spacer()
            }
            .disabled(portrait || settings.aspect <= 0)
        }
    }

    // MARK: Grade

    private var gradeCard: some View {
        VerdictCard(.info, headline: "Grade", detail: "Lift, gamma and gain on code values, through the same tables the export bakes with. The trims add to each channel.") {
            VStack(alignment: .leading, spacing: 6) {
                GradeSlider(label: "Lift", value: $settings.lift, range: -0.2...0.2, neutral: 0)
                GradeSlider(label: "Gamma", value: $settings.gamma, range: 0.5...2.0, neutral: 1)
                GradeSlider(label: "Gain", value: $settings.gain, range: 0.5...1.5, neutral: 1)
                DisclosureGroup("Red, green and blue trims", isExpanded: $perChannel) {
                    VStack(alignment: .leading, spacing: 4) {
                        ChannelRow(label: "Lift", values: $settings.liftRGB, range: -0.1...0.1, neutral: 0)
                        ChannelRow(label: "Gamma", values: $settings.gammaRGB, range: 0.8...1.25, neutral: 1)
                        ChannelRow(label: "Gain", values: $settings.gainRGB, range: 0.8...1.25, neutral: 1)
                    }
                    .padding(.top, 4)
                }
                Divider()
                GradeSlider(label: "Sharpen", value: $settings.sharpen, range: 0...1, neutral: 0.35, format: "%.2f")
                HStack {
                    Spacer()
                    Button("Reset the look") {
                        let keep = (settings.aspect, settings.headroomMM)
                        settings = GradeSettings()
                        settings.aspect = keep.0
                        settings.headroomMM = keep.1
                        resetPending = true
                    }
                    .help("Back to no grade; the next export ignores the saved look as well (hs grade --reset)")
                }
            }
            .disabled(portrait)
        }
    }

    // MARK: Export

    private var exportCard: some View {
        VerdictCard(.info, headline: "Export", detail: "Render again at another width, bake the look into an MP4, or keep this one frame.") {
            VStack(alignment: .leading, spacing: 10) {
                QueueWatch(queue: renderQueue) { q in
                    let rendering = q?.isRunning ?? false
                    HStack(spacing: 10) {
                        Text("Render width")
                        Picker("", selection: $renderWidth) {
                            Text("1920").tag(1920); Text("2400").tag(2400); Text("3840").tag(3840)
                        }
                        .labelsHidden().frame(width: 110)
                        Button(rendering ? "Rendering…" : "Render again at \(renderWidth)") { rerender() }
                            .disabled(rendering || lockAlive || !moveJSONExists || !model.config.problems.isEmpty)
                            .help(moveJSONExists ? "hs render -p P --move \(moveName) --width \(renderWidth): the current model, the 4:5 cut beside the 16:9"
                                                 : "This render's move is not in move/, so it cannot be rendered again from here")
                        if rendering { Button("Stop") { q?.cancel() } }
                    }
                    if rendering, let run = q?.current {
                        RunPanel(session: run, showMetrics: false)
                    }
                }
                Observing(exportRun) { r in
                    HStack(spacing: 10) {
                        Button(r.isRunning ? "Exporting…" : "Export graded MP4") { export() }
                            .buttonStyle(.borderedProminent)
                            .disabled(r.isRunning || move == nil || portrait)
                        Button("This frame as a still") { exportStill() }
                            .disabled(r.isRunning || move == nil || portrait)
                            .help("hs grade --still \(Int(frame)) --graded: grade/\(moveName)_still.png, cropped and graded")
                        Text("Quality")
                        Stepper(value: $crf, in: 12...28) { Text("CRF \(crf)").monospacedDigit() }
                            .help("Lower is larger and cleaner; 17 is visually lossless")
                        Spacer()
                        if let g = move?.graded, !r.isRunning {
                            Button("Play") { NSWorkspace.shared.open(URL(fileURLWithPath: g)) }
                            Button { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: g)]) } label: {
                                Image(systemName: "folder")
                            }
                            .buttonStyle(.borderless)
                        }
                    }
                    Text("render/\(moveName)_graded.mp4").font(.caption.monospaced()).foregroundStyle(.secondary)
                }
            }
        }
    }

    private var moveJSONExists: Bool {
        FileManager.default.fileExists(atPath: (project.path as NSString).appendingPathComponent("move/\(moveName).json"))
    }

    // MARK: actions

    private func reload() {
        moves = RenderedMove.list(project: project.path)
        if move == nil { moveName = moves.last(where: { $0.followsHead })?.name ?? moves.last?.name ?? "" }
        selectMove()
    }

    private func selectMove() {
        guard let m = move else { return }
        let saved = GradeSettings.lookPath(project: project.path, renderName: m.name)
        settings = GradeSettings.load(saved) ?? GradeSettings()
        resetPending = false
        frame = Double(m.frames / 2)
        still = nil
        graded = nil
        loadStill(Int(frame))
    }

    private func gradeArgs() -> [String] {
        var a = settings.arguments(project: project.path, move: moveName)
        if resetPending { a.append("--reset") }
        return a
    }

    private func stillArgs(_ n: Int, graded: Bool) -> [String] {
        var a = gradeArgs()
        a += ["--still", String(n)]
        if graded { a.append("--graded") }
        return a
    }

    private func loadStill(_ n: Int) {
        guard move != nil, !portrait else { return }
        if stillRun?.isRunning == true {
            pendingFrame = n
            return
        }
        let s = model.session("Frame \(n)", stillArgs(n, graded: false))
        s.onFinish = { run in
            let path = (project.path as NSString).appendingPathComponent("grade/\(moveName)_still.png")
            if run.succeeded, let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
               let img = CGImageSourceCreateImageAtIndex(src, 0, [kCGImageSourceShouldCache: true] as CFDictionary) {
                still = img
                stillFrame = n
                regrade()
            }
            if let p = pendingFrame {
                pendingFrame = nil
                loadStill(p)
            }
        }
        stillRun = s
        s.start()
    }

    private func regrade() {
        guard let img = still else { return }
        let s = settings
        DispatchQueue.global(qos: .userInitiated).async {
            let out = s.apply(to: img)
            DispatchQueue.main.async {
                if s == settings { graded = out }
            }
        }
    }

    private func export() {
        guard move != nil else { return }
        let s = model.session("Grade \(moveName)", gradeArgs() + ["--crf", String(crf)])
        s.onFinish = { _ in
            moves = RenderedMove.list(project: project.path)
            resetPending = false
        }
        exportRun = s
        s.start()
    }

    /// One graded, cropped frame: grade/<move>_still.png, then shown in the Finder.
    private func exportStill() {
        guard move != nil else { return }
        let n = Int(frame)
        let s = model.session("Still \(n)", stillArgs(n, graded: true))
        s.onFinish = { run in
            let path = (project.path as NSString).appendingPathComponent("grade/\(moveName)_still.png")
            if run.succeeded { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)]) }
        }
        exportRun = s
        s.start()
    }

    /// `hs render` of the move this render came from, with the current model, at the chosen width.
    private func rerender() {
        guard moveJSONExists else { return }
        let args = ["render", "-p", project.path, "--move", moveName, "--width", String(renderWidth)]
        let q = RunQueue(config: model.config, steps: [RunQueue.Step(title: "Render", arguments: args)])
        let path = project.path
        q.onStep = { s in model.projectRuns[path] = s }
        q.onFinish = { _ in
            store.reload()
            moves = RenderedMove.list(project: path)
            selectMove()
        }
        model.moveQueues[path] = q
        q.start()
    }
}

struct GradeSlider: View {
    let label: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let neutral: Double
    var format = "%.3f"

    var body: some View {
        HStack {
            Text(label).frame(width: 70, alignment: .leading)
            Slider(value: $value, in: range)
            Text(String(format: format, value)).monospacedDigit().frame(width: 56, alignment: .trailing)
            Button { value = neutral } label: { Image(systemName: "arrow.uturn.backward") }
                .buttonStyle(.borderless)
                .disabled(abs(value - neutral) < 1e-9)
                .help("Reset \(label.lowercased())")
        }
    }
}

struct ChannelRow: View {
    let label: String
    @Binding var values: [Double]
    let range: ClosedRange<Double>
    let neutral: Double

    var body: some View {
        HStack(spacing: 10) {
            Text(label).frame(width: 60, alignment: .leading).foregroundStyle(.secondary)
            ForEach(0..<3, id: \.self) { c in
                HStack(spacing: 4) {
                    Circle().fill([Color.red, Color.green, Color.blue][c]).frame(width: 8, height: 8)
                    Slider(value: Binding(get: { values.count > c ? values[c] : neutral },
                                          set: { v in if values.count > c { values[c] = v } }), in: range)
                    Text(String(format: "%.3f", values.count > c ? values[c] : neutral)).monospacedDigit().font(.caption).frame(width: 44)
                }
            }
            Button { values = [neutral, neutral, neutral] } label: { Image(systemName: "arrow.uturn.backward") }
                .buttonStyle(.borderless)
        }
    }
}
