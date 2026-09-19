import SwiftUI
import AppKit
import HSCore

/// Project page: every model that can be looked at, each opening in its own window — two
/// windows side by side is the A/B.
struct ModelsBox: View {
    @Environment(\.openWindow) private var openWindow
    let project: ProjectSummary

    var body: some View {
        let files = ViewerModelFile.list(project: project.path)
        GroupBox("Models") {
            VStack(alignment: .leading, spacing: 6) {
                if let why = SplatViewerSupport.problem {
                    Label(why, systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if files.isEmpty {
                    Text("No trained model yet — train, or archive one.").foregroundStyle(.secondary)
                }
                ForEach(files) { f in
                    HStack(spacing: 10) {
                        Image(systemName: f.archive == nil ? "cube.transparent" : "archivebox")
                            .foregroundStyle(.secondary).frame(width: 18)
                        Text(f.name).font(.system(.body, design: .monospaced))
                        Text(f.sizeLabel).foregroundStyle(.secondary).monospacedDigit()
                        Spacer()
                        Button("Reveal") { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: f.ply)]) }
                            .buttonStyle(.borderless)
                        Button("View") { openWindow(id: "viewer", value: f) }
                            .disabled(SplatViewerSupport.problem != nil)
                    }
                }
                if !files.isEmpty {
                    Text("A preview for looking at geometry and floaters from real camera positions. It is not brush-path-render: sorting and anti-aliasing differ, so judge final frames — and anything `hs views` scores — from a render.")
                        .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(4)
        }
    }
}

/// One model in one window.
struct SplatViewerWindow: View {
    @EnvironmentObject var model: AppModel
    let file: ViewerModelFile

    @StateObject private var scene = SplatScene()
    @State private var capture = ""
    @State private var eye = "L"
    @State private var moves: [String] = []
    @State private var movePath = ""
    @State private var move: MovePath?
    @State private var frame: Double = 0
    @State private var playing = false

    var body: some View {
        VStack(spacing: 0) {
            controls.padding(.horizontal, 12).padding(.vertical, 8)
            Divider()
            ZStack {
                Color.black
                SplatMetalView(scene: scene)
                overlay
            }
            Divider()
            status.padding(.horizontal, 12).padding(.vertical, 6)
        }
        .navigationTitle("\((file.project as NSString).lastPathComponent) — \(file.name)")
        .onAppear {
            moves = MovePath.list(project: file.project)
            scene.load(model: file, config: model.config)
        }
        .onDisappear { scene.unload() }
        .onChange(of: scene.cameras?.rigNpzMD5) { _, _ in
            if capture.isEmpty, let first = scene.cameras?.captures.first { capture = first }
        }
        .task(id: playing) {
            guard playing, let m = move, !m.frames.isEmpty else { return }
            let tick = UInt64(1_000_000_000 / max(m.fps, 1))
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: tick)
                frame = (frame + 1).truncatingRemainder(dividingBy: Double(m.frames.count))
                showFrame()
            }
        }
    }

    // MARK: controls

    private var controls: some View {
        HStack(spacing: 14) {
            if let cams = scene.cameras {
                HStack(spacing: 4) {
                    Button { step(-1) } label: { Image(systemName: "chevron.left") }
                        .keyboardShortcut(.leftArrow, modifiers: [])
                        .help("Previous capture (←)")
                    Picker("Capture", selection: $capture) {
                        ForEach(cams.captures, id: \.self) { Text($0).tag($0) }
                    }
                    .labelsHidden().frame(width: 110)
                    Button { step(+1) } label: { Image(systemName: "chevron.right") }
                        .keyboardShortcut(.rightArrow, modifiers: [])
                        .help("Next capture (→)")
                }
                if cams.stereo {
                    Picker("Eye", selection: $eye) { Text("L").tag("L"); Text("R").tag("R") }
                        .pickerStyle(.segmented).labelsHidden().frame(width: 70)
                }
                Button("Stand here") { standAtCapture() }
                    .keyboardShortcut(.return, modifiers: [])
                    .help("Put the camera exactly where this capture was, with its lens (↩)")
                Button("Orbit") { playing = false; scene.resetOrbit() }
                    .keyboardShortcut("o", modifiers: [])
                    .help("Free camera about the subject (O). Drag to turn, shift-drag or right-drag to slide, scroll to move in and out.")
            }
            Spacer(minLength: 8)
            if !moves.isEmpty {
                Picker("Move", selection: $movePath) {
                    Text("no move").tag("")
                    ForEach(moves, id: \.self) { Text(($0 as NSString).lastPathComponent).tag($0) }
                }
                .frame(maxWidth: 220)
                if let m = move, m.frames.count > 1 {
                    Button { playing.toggle() } label: { Image(systemName: playing ? "pause.fill" : "play.fill") }
                        .keyboardShortcut(.space, modifiers: [])
                    Slider(value: $frame, in: 0...Double(m.frames.count - 1), step: 1).frame(width: 200)
                    Text("\(Int(frame) + 1)/\(m.frames.count)").monospacedDigit().foregroundStyle(.secondary)
                        .frame(width: 70, alignment: .trailing)
                }
            }
        }
        .disabled(scene.phase != .ready)
        .onChange(of: capture) { _, _ in standAtCapture() }
        .onChange(of: eye) { _, _ in standAtCapture() }
        .onChange(of: movePath) { _, p in loadMove(p) }
        .onChange(of: frame) { _, _ in if !playing { showFrame() } }
    }

    @ViewBuilder private var overlay: some View {
        switch scene.phase {
        case .loading(let what):
            VStack(spacing: 10) {
                ProgressView().controlSize(.large)
                Text(what).foregroundStyle(.white.opacity(0.85))
            }
        case .failed(let why):
            VStack(spacing: 8) {
                Image(systemName: "exclamationmark.triangle.fill").font(.largeTitle).foregroundStyle(.orange)
                Text(why).foregroundStyle(.white).multilineTextAlignment(.center).textSelection(.enabled)
                    .frame(maxWidth: 560)
            }
        case .idle, .ready:
            EmptyView()
        }
    }

    private var status: some View {
        HStack(spacing: 14) {
            if scene.phase == .ready {
                Text("\(scene.splatCount.formatted()) splats").monospacedDigit()
                Text(String(format: "loaded in %.1f s", scene.loadSeconds)).foregroundStyle(.secondary)
            }
            switch scene.mode {
            case .pose(_, let label): Label(label, systemImage: "camera.viewfinder")
            case .orbit: Label("orbit", systemImage: "rotate.3d").foregroundStyle(.secondary)
            }
            Spacer()
            if let w = lineageWarning {
                Label(w, systemImage: "exclamationmark.triangle.fill").foregroundStyle(.orange).help(w)
            }
            Text("preview — not brush-path-render").foregroundStyle(.secondary)
        }
        .font(.callout)
        .lineLimit(1)
    }

    /// The live export is posed with train/dataset/rig.npz as it is *now*. If solve re-ran
    /// since training, that is a different frame — the render guard's model_matches_solve.
    private var lineageWarning: String? {
        guard file.archive == nil, let cams = scene.cameras,
              let trained = CameraExport.trainedRigMD5(project: file.project), trained != cams.rigNpzMD5 else { return nil }
        return "cameras are from a later solve than this model — capture poses will not line up"
    }

    // MARK: actions

    private func step(_ d: Int) {
        guard let caps = scene.cameras?.captures, !caps.isEmpty else { return }
        let i = caps.firstIndex(of: capture) ?? 0
        capture = caps[(i + d + caps.count) % caps.count]
    }

    private func standAtCapture() {
        guard scene.phase == .ready, let v = scene.cameras?.view(capture: capture, eye: eye) else { return }
        playing = false
        scene.stand(at: v)
    }

    private func loadMove(_ path: String) {
        playing = false
        frame = 0
        if path.isEmpty { move = nil } else { move = try? MovePath.load(path: path) }
        if move != nil { showFrame() }
    }

    private func showFrame() {
        guard let m = move, let p = m.pose(at: Int(frame)) else { return }
        scene.stand(at: p, label: "\((movePath as NSString).lastPathComponent) · frame \(Int(frame) + 1) of \(m.frames.count)")
    }
}
