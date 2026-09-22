import SwiftUI
import AppKit
import ImageIO
import HSCore

/// Lift / gamma / gain on one representative frame of a rendered move, previewed live with the
/// same lookup table `hs grade` exports with. The frame comes from `hs grade --still` (already
/// cropped to follow the head when the move was framed with --headroom-mm).
struct GradeView: View {
    @EnvironmentObject var model: AppModel
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

    private var move: RenderedMove? { moves.first { $0.name == moveName } }

    var body: some View {
        GroupBox {
            if moves.isEmpty {
                Text("No rendered moves yet (render/<move>_1920.mp4).").foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                VStack(alignment: .leading, spacing: 12) {
                    header
                    preview
                    frameSlider
                    sliders
                    exportRow
                    if let r = exportRun {
                        RunPanel(session: r, showMetrics: false)
                    }
                }
                .padding(4)
            }
        } label: {
            Text("Grade")
        }
        .onAppear(perform: reload)
        .onChange(of: settings) { _, _ in regrade() }
    }

    // MARK: pieces

    private var header: some View {
        HStack {
            Picker("Move", selection: $moveName) {
                ForEach(moves) { m in Text(m.name).tag(m.name) }
            }
            .frame(maxWidth: 260)
            .onChange(of: moveName) { _, _ in selectMove() }
            if let m = move {
                Text("\(m.frames) frames").foregroundStyle(.secondary)
                Label(m.followsHead ? "crop follows the head" : "centred crop",
                      systemImage: m.followsHead ? "person.crop.rectangle" : "rectangle.dashed")
                    .foregroundStyle(.secondary)
                    .help(m.followsHead ? "framed with hs move --headroom-mm" : "build the move with --headroom-mm to keep the head framed")
            }
            Spacer()
            Toggle("Before", isOn: $showBefore).toggleStyle(.button)
                .help("Hold the ungraded frame")
        }
    }

    @ViewBuilder private var preview: some View {
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

    private var frameSlider: some View {
        HStack {
            Text("Frame").frame(width: 70, alignment: .leading)
            Slider(value: $frame, in: 0...Double(max((move?.frames ?? 1) - 1, 1)), step: 1) { editing in
                if !editing { loadStill(Int(frame)) }
            }
            Text("\(Int(frame))").monospacedDigit().frame(width: 44, alignment: .trailing)
            Button("Middle") { frame = Double((move?.frames ?? 0) / 2); loadStill(Int(frame)) }
                .help("The representative frame: the middle of the move, where the subject faces the camera")
        }
    }

    private var sliders: some View {
        VStack(alignment: .leading, spacing: 6) {
            GradeSlider(label: "Lift", value: $settings.lift, range: -0.2...0.2, neutral: 0)
            GradeSlider(label: "Gamma", value: $settings.gamma, range: 0.5...2.0, neutral: 1)
            GradeSlider(label: "Gain", value: $settings.gain, range: 0.5...1.5, neutral: 1)
            DisclosureGroup("Per channel", isExpanded: $perChannel) {
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
                Text("Crop").frame(width: 70, alignment: .leading)
                Picker("", selection: $settings.aspect) {
                    ForEach(GradeSettings.aspectOptions(including: settings.aspect), id: \.value) { o in
                        Text(o.label).tag(o.value)
                    }
                }
                .labelsHidden()
                .frame(width: 130)
                .onChange(of: settings.aspect) { _, _ in loadStill(Int(frame)) }
                Text("Headroom").padding(.leading, 12)
                TextField("", value: $settings.headroomMM, format: .number.precision(.fractionLength(1)))
                    .frame(width: 60)
                    .onSubmit { loadStill(Int(frame)) }
                Text("mm above the crown").foregroundStyle(.secondary)
                Spacer()
                Button("Reset") { settings = GradeSettings() }
            }
        }
    }

    private var exportRow: some View {
        Observing(exportRun) { r in
            HStack {
                Button(r.isRunning ? "Exporting…" : "Export graded video") { export() }
                    .disabled(r.isRunning || move == nil)
                Text("render/\(moveName)_graded.mp4").font(.caption.monospaced()).foregroundStyle(.secondary)
                Spacer()
                if let g = move?.graded, !r.isRunning {
                    Button("Play") { NSWorkspace.shared.open(URL(fileURLWithPath: g)) }
                    Button { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: g)]) } label: {
                        Image(systemName: "folder")
                    }
                    .buttonStyle(.borderless)
                }
            }
        }
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
        frame = Double(m.frames / 2)
        still = nil
        graded = nil
        loadStill(Int(frame))
    }

    private func stillArgs(_ n: Int) -> [String] {
        var a = settings.arguments(project: project.path, move: moveName)
        a += ["--still", String(n)]
        return a
    }

    private func loadStill(_ n: Int) {
        guard move != nil else { return }
        if stillRun?.isRunning == true {
            pendingFrame = n
            return
        }
        let s = model.session("Frame \(n)", stillArgs(n))
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
        let s = model.session("Grade \(moveName)", settings.arguments(project: project.path, move: moveName))
        s.onFinish = { _ in moves = RenderedMove.list(project: project.path) }
        exportRun = s
        s.start()
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
                    Slider(value: Binding(get: { values[c] }, set: { values[c] = $0 }), in: range)
                    Text(String(format: "%.3f", values[c])).monospacedDigit().font(.caption).frame(width: 44)
                }
            }
            Button { values = [neutral, neutral, neutral] } label: { Image(systemName: "arrow.uturn.backward") }
                .buttonStyle(.borderless)
        }
    }
}
