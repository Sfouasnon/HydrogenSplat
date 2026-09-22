import SwiftUI
import AppKit
import HSCore

/// Project page: every model that can be looked at, with where it came from and how it scored.
/// View opens it in the project's Viewer page; right-click → Open in New Window for two side by
/// side (the A/B). Archive keeps the live export under a name; Score runs `hs views --ply` on any
/// model after the fact, with the Train page's hold-out and crop, so a model can be scored at a
/// different crop or against a new mask set without retraining.
struct ModelsBox: View {
    @EnvironmentObject var model: AppModel
    @ObservedObject var scene: SplatScene
    @Environment(\.openWindow) private var openWindow
    let project: ProjectSummary
    @State private var revision = 0
    @State private var askArchive = false
    @State private var archiveName = ""

    private var queue: RunQueue? { model.trainQueues[project.path] }
    private var busy: Bool { (queue?.isRunning ?? false) || project.lock?.alive == true }
    private var hasMasks: Bool {
        FileManager.default.fileExists(atPath: (project.path as NSString).appendingPathComponent("train/dataset/masks"))
    }
    private var archiveNameProblem: String? {
        let n = archiveName.trimmingCharacters(in: .whitespaces)
        if n.isEmpty { return "a name is needed" }
        if n.range(of: #"^[A-Za-z0-9][A-Za-z0-9._-]*$"#, options: .regularExpression) == nil { return "letters, digits, . _ - only" }
        if FileManager.default.fileExists(atPath: (project.path as NSString).appendingPathComponent("archive/\(n)")) { return "an archive named \(n) exists" }
        return nil
    }

    var body: some View {
        let _ = revision
        let files = ViewerModelFile.list(project: project.path)
        let scores = ViewsScore.list(project: project.path)
        let rigNow = ModelProvenance.currentRigMD5(project: project.path)
        GroupBox("Models") {
            VStack(alignment: .leading, spacing: 8) {
                if let why = SplatViewerSupport.problem {
                    Label(why, systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if files.isEmpty {
                    Text("No trained model yet — train, or archive one.").foregroundStyle(.secondary)
                }
                // one Grid for every model, so name / layer / splats / time / date line up across rows;
                // a model's scores are a second row spanning the grid, as a small table of their own
                Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 6) {
                    ForEach(files) { f in
                        let prov = ModelProvenance.load(for: f)
                        let mine = scores.filter { $0.scores(f) }
                        modelRow(f, prov: prov, rigNow: rigNow, scoreCount: mine.count)
                        if !mine.isEmpty {
                            GridRow {
                                scoreTable(mine)
                                    .gridCellColumns(8)
                            }
                        }
                        if f != files.last {
                            GridRow { Divider().gridCellColumns(8) }
                        }
                    }
                }
                if !files.isEmpty {
                    Text("A preview for looking at geometry and floaters from real camera positions. It is not brush-path-render: sorting and anti-aliasing differ, so judge final frames — and anything `hs views` scores — from a render. Scores are medians over the report's views; a report is only comparable with another at the same crop and mask set.")
                        .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(4)
        }
        .alert("Archive the current model", isPresented: $askArchive) {
            TextField("name", text: $archiveName)
            Button("Archive") { archive() }.disabled(archiveNameProblem != nil)
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Copies train/exports into archive/<name> with its rig.npz and manifest. "
                 + (archiveNameProblem.map { "(\($0))" } ?? ""))
        }
    }

    @ViewBuilder private func modelRow(_ f: ViewerModelFile, prov: ModelProvenance?, rigNow: String?, scoreCount: Int) -> some View {
        GridRow {
            Image(systemName: f.archive == nil ? "cube.transparent" : "archivebox")
                .foregroundStyle(.secondary).frame(width: 18)
            Text(f.name).font(.system(.body, design: .monospaced))
                .contextMenu {
                    Button("View") { view(f) }
                    Button("Open in New Window") { openWindow(id: "viewer", value: f) }
                    if scoreCount > 0 {
                        Button("Reveal reports") {
                            NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: (project.path as NSString).appendingPathComponent("views"))])
                        }
                    }
                }
            Group {
                if let l = prov?.layer { chip(l, tint: l == "full" ? .secondary : .blue) } else { Text("") }
            }
            Group {
                if let n = prov?.splats { Text("\(n.formatted()) splats") } else { Text(f.sizeLabel) }
            }
            .foregroundStyle(.secondary).monospacedDigit().gridColumnAlignment(.trailing)
            Text(prov?.trainSeconds.map { Format.duration($0) } ?? "")
                .foregroundStyle(.secondary).monospacedDigit().gridColumnAlignment(.trailing)
            Text(prov?.archivedLabel ?? "").foregroundStyle(.secondary).monospacedDigit()
            Group {
                if let want = prov?.rigNpzMD5, let now = rigNow {
                    if want == now {
                        Label("this solve", systemImage: "checkmark.circle").foregroundStyle(.green)
                            .help("Trained against the current train/dataset/rig.npz — renders and scores line up with the moves")
                    } else {
                        Label("other solve", systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
                            .help("Trained against a different rig.npz (\(want.prefix(8))… vs \(now.prefix(8))…); hs render's model_matches_solve guard will refuse it")
                    }
                } else {
                    Text("")
                }
            }
            .font(.caption)
            HStack(spacing: 8) {
                if scene.loaded == f { Text("loaded").font(.caption).foregroundStyle(.green) }
                Spacer(minLength: 0)
                if f.name == "current" {
                    Button("Archive…") { archiveName = ""; askArchive = true }
                        .buttonStyle(.borderless).disabled(busy)
                        .help("Keep this export under a name before the next train replaces it")
                }
                Menu("Score") {
                    ForEach(ScoreRegion.allCases, id: \.self) { r in
                        Button(r.title) { score(f, r) }
                            .disabled(r != .whole && !hasMasks)
                    }
                }
                .menuStyle(.button).buttonStyle(.borderless).fixedSize().disabled(busy)
                .help(busy ? "wait for the running stage" : "hs views --ply on this model with the Train page's hold-out and crop; the report appears below")
                Button("Reveal") { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: f.ply)]) }
                    .buttonStyle(.borderless)
                Button("View") { view(f) }
                    .disabled(SplatViewerSupport.problem != nil)
                    .help("Open in this project's Viewer page (right-click for a separate window)")
            }
            .frame(maxWidth: .infinity)
        }
    }

    /// The model's `hs views` reports as a table: one row per report, numbers right-aligned in
    /// their own columns, so five reports read as a table and not as five pills of different widths.
    private func scoreTable(_ scores: [ViewsScore]) -> some View {
        let shown = Array(scores.prefix(6))
        func num(_ v: Double?, _ fmt: String) -> some View {
            Text(v.map { String(format: fmt, $0) } ?? "—").monospacedDigit().gridColumnAlignment(.trailing)
        }
        return Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 2) {
            GridRow {
                Text("report").foregroundStyle(.secondary)
                Text("PSNR").foregroundStyle(.secondary).gridColumnAlignment(.trailing)
                Text("interior").foregroundStyle(.secondary).gridColumnAlignment(.trailing)
                Text("edge").foregroundStyle(.secondary).gridColumnAlignment(.trailing)
                Text("displaced").foregroundStyle(.secondary).gridColumnAlignment(.trailing)
                Text("views").foregroundStyle(.secondary).gridColumnAlignment(.trailing)
            }
            ForEach(shown) { s in
                GridRow {
                    Text(s.name).font(.caption.monospaced())
                    num(s.psnr, "%.2f")
                    num(s.psnrInterior, "%.2f")
                    num(s.psnrEdge, "%.2f")
                    num(s.displaced.map { $0 * 100 }, "%.1f%%")
                    Text("\(s.views)").monospacedDigit().gridColumnAlignment(.trailing)
                }
                .help("views/\(s.name)_report.json")
            }
            if scores.count > shown.count {
                GridRow {
                    Text("+\(scores.count - shown.count) more in views/").foregroundStyle(.secondary).gridCellColumns(6)
                }
            }
        }
        .font(.caption)
        .padding(.leading, 30)
        .padding(.bottom, 2)
    }

    private func chip(_ text: String, tint: Color) -> some View {
        Text(text).font(.caption.weight(.medium)).foregroundStyle(tint)
            .padding(.horizontal, 6).padding(.vertical, 1)
            .background(tint.opacity(0.12), in: Capsule())
    }

    private func view(_ f: ViewerModelFile) {
        model.viewerFile[project.path] = f
        model.projectPage[project.path] = .viewer
    }

    private func start(_ steps: [RunQueue.Step]) {
        let q = RunQueue(config: model.config, steps: steps)
        let path = project.path
        q.onStep = { s in model.projectRuns[path] = s }
        q.onFinish = { _ in revision += 1 }
        model.trainQueues[path] = q
        q.start()
    }

    private func archive() {
        let n = archiveName.trimmingCharacters(in: .whitespaces)
        guard archiveNameProblem == nil else { return }
        start([.init(title: "Archive \(n)", arguments: ["archive", "-p", project.path, "--name", n])])
    }

    private func score(_ f: ViewerModelFile, _ region: ScoreRegion) {
        let settings = model.trainSettings[project.path] ?? TrainSettings()
        let set = CaptureSet.read(project: project.path)
        let (name, argv) = region.arguments(project: project.path, model: f, settings: settings, set: set)
        start([.init(title: "Score \(f.name) (\(name))", arguments: argv)])
    }
}

/// The project's Viewer page: a model picker over the shared scene, then the viewer itself,
/// filling the detail pane.
struct ModelViewerPane: View {
    @EnvironmentObject var model: AppModel
    @ObservedObject var scene: SplatScene
    @Environment(\.openWindow) private var openWindow
    let project: ProjectSummary
    @AppStorage("viewer.showMovePanel") private var showMovePanel = true
    /// Listed on appear and on Reload, not in body: the scene publishes on every drag and every
    /// played frame, and this view observes it.
    @State private var files: [ViewerModelFile] = []

    /// What the page shows: the pick made here or from the Models box; else what the scene
    /// already holds for this project; else the newest export.
    private var chosen: ViewerModelFile? {
        let fs = files
        if let f = model.viewerFile[project.path], fs.contains(f) { return f }
        if let l = scene.loaded, l.project == project.path, fs.contains(l) { return l }
        return fs.first
    }

    private var selection: Binding<ViewerModelFile?> {
        Binding(get: { chosen }, set: { model.viewerFile[project.path] = $0 })
    }

    var body: some View {
        let fs = files
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                Picker("Model", selection: selection) {
                    ForEach(fs) { f in
                        Text("\(f.name)  ·  \(f.sizeLabel)").tag(Optional(f))
                    }
                }
                .frame(maxWidth: 360)
                .disabled(fs.isEmpty)
                Spacer()
                if let f = chosen {
                    Button("Reload") {
                        files = ViewerModelFile.list(project: project.path)
                        scene.load(model: f, config: model.config)
                    }
                        .help("Read the .ply and the cameras again")
                    Button("Open in New Window") { openWindow(id: "viewer", value: f) }
                        .help("A second window, with its own copy in memory — for an A/B beside this one")
                }
                Button("Unload") { scene.unload() }
                    .disabled(scene.loaded == nil)
                    .help("Free the model's memory (1–2 GB for a large head)")
                Toggle(isOn: $showMovePanel) { Label("Move", systemImage: "sidebar.right") }
                    .toggleStyle(.button)
                    .help("The move panel: write, preview, build and render a camera move")
            }
            .padding(.horizontal, 12).padding(.vertical, 8)
            Divider()
            HStack(spacing: 0) {
                viewerArea
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                if showMovePanel {
                    Divider()
                    MoveInspector(project: project, editor: model.moveEditor(project: project.path), scene: scene)
                        .frame(width: 380)
                }
            }
        }
        .onAppear {
            files = ViewerModelFile.list(project: project.path)
            loadIfEmpty()
        }
        .onChange(of: model.viewerFile[project.path]) { _, _ in
            // an explicit pick (here or View in the Models box) loads it
            if let f = chosen, scene.loaded != f, SplatViewerSupport.problem == nil {
                scene.load(model: f, config: model.config)
            }
        }
    }

    @ViewBuilder private var viewerArea: some View {
        VStack(spacing: 0) {
            if let why = SplatViewerSupport.problem {
                ContentUnavailableView("Viewer unavailable", systemImage: "exclamationmark.triangle",
                                       description: Text(why))
            } else if let f = chosen {
                if scene.loaded == f {
                    SplatViewerPanel(file: f, scene: scene,
                                     editor: showMovePanel ? model.moveEditor(project: project.path) : nil)
                } else {
                    ContentUnavailableView {
                        Label(f.name, systemImage: "cube.transparent")
                    } description: {
                        Text(f.sizeLabel)
                    } actions: {
                        Button("Load") { scene.load(model: f, config: model.config) }
                            .keyboardShortcut(scene.typing ? nil : .defaultAction)
                    }
                }
            } else {
                ContentUnavailableView("No trained model yet", systemImage: "cube.transparent",
                                       description: Text("Train, or archive one."))
            }
        }
    }

    /// Opening the page loads the chosen model unless the scene already holds a model — a model
    /// from another project stays until one is picked here, so a glance at a project costs nothing.
    private func loadIfEmpty() {
        guard SplatViewerSupport.problem == nil, let f = chosen, scene.loaded != f else { return }
        if scene.loaded == nil || model.viewerFile[project.path] == f {
            scene.load(model: f, config: model.config)
        }
    }
}

/// One model in its own window.
struct SplatViewerWindow: View {
    @EnvironmentObject var model: AppModel
    let file: ViewerModelFile
    @StateObject private var scene = SplatScene()

    var body: some View {
        SplatViewerPanel(file: file, scene: scene)
            .navigationTitle("\((file.project as NSString).lastPathComponent) — \(file.name)")
            .onAppear { scene.load(model: file, config: model.config) }
            .onDisappear { scene.unload() }
    }
}

/// Controls, the Metal view and the status bar for a scene that holds `file`; with an editor,
/// the move is drawn over the view and its timeline sits under it.
struct SplatViewerPanel: View {
    let file: ViewerModelFile
    @ObservedObject var scene: SplatScene
    var editor: MoveEditor? = nil


    var body: some View {
        VStack(spacing: 0) {
            controls.padding(.horizontal, 12).padding(.vertical, 8)
            Divider()
            ZStack {
                Color.black
                SplatMetalView(scene: scene)
                    .overlay { LookFrame(look: scene.look) }
                if let e = editor { EditorOverlayHost(editor: e) }
                overlay
            }
            if let e = editor { EditorTimelineHost(editor: e, keys: keys) }
            Divider()
            status.padding(.horizontal, 12).padding(.vertical, 6)
        }
        .onAppear { pickFirstCaptureIfNeeded() }
        .onDisappear { scene.playback.pause() }
        .onChange(of: scene.cameras?.rigNpzMD5) { _, _ in pickFirstCaptureIfNeeded() }
    }

    private func pickFirstCaptureIfNeeded() {
        if scene.capture.isEmpty, let first = scene.cameras?.captures.first { scene.capture = first }
    }

    // MARK: controls

    /// Single-key shortcuts, off while the move script is being typed.
    private var keys: Bool { !scene.typing }

    private var controls: some View {
        HStack(spacing: 14) {
            if let cams = scene.cameras {
                HStack(spacing: 4) {
                    Button { step(-1) } label: { Image(systemName: "chevron.left") }
                        .keyboardShortcut(keys ? KeyboardShortcut(.leftArrow, modifiers: []) : nil)
                        .help("Previous capture (←)")
                    Picker("Capture", selection: $scene.capture) {
                        ForEach(cams.captures, id: \.self) { Text($0).tag($0) }
                    }
                    .labelsHidden().frame(width: 110)
                    Button { step(+1) } label: { Image(systemName: "chevron.right") }
                        .keyboardShortcut(keys ? KeyboardShortcut(.rightArrow, modifiers: []) : nil)
                        .help("Next capture (→)")
                }
                if cams.stereo {
                    Picker("Eye", selection: $scene.eye) { Text("L").tag("L"); Text("R").tag("R") }
                        .pickerStyle(.segmented).labelsHidden().frame(width: 70)
                }
                Button("Stand here") { standAtCapture() }
                    .keyboardShortcut(keys ? KeyboardShortcut(.return, modifiers: []) : nil)
                    .help("Put the camera exactly where this capture was, with its lens (↩)")
                Button("Orbit") { scene.resetOrbit() }
                    .keyboardShortcut(keys ? KeyboardShortcut("o", modifiers: []) : nil)
                    .help("Free camera about the subject (O). Drag to turn, shift-drag or right-drag to slide, scroll to move in and out.")
            }
            Spacer(minLength: 8)
            MoveBar(project: file.project, playback: scene.playback, keys: keys)
        }
        .disabled(scene.phase != .ready)
        .onChange(of: scene.capture) { old, _ in if !old.isEmpty { standAtCapture() } }
        .onChange(of: scene.eye) { _, _ in standAtCapture() }
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
        let i = caps.firstIndex(of: scene.capture) ?? 0
        scene.capture = caps[(i + d + caps.count) % caps.count]
    }

    private func standAtCapture() {
        guard scene.phase == .ready, let v = scene.cameras?.view(capture: scene.capture, eye: scene.eye) else { return }
        scene.stand(at: v)
    }
}

/// Pick a move, play / pause (space), scrub. Built moves (move/*.json) and live previews
/// (viewer/move_*.json) both appear.
struct MoveBar: View {
    let project: String
    @ObservedObject var playback: MovePlayback
    /// false while a text field has the keyboard: space, arrows, ↩ and O are then text
    var keys = true
    @State private var choices: [String] = []

    private var selection: Binding<String> {
        Binding(get: { playback.path }, set: { playback.load($0) })
    }

    var body: some View {
        if playback.isKeyed {
            // the keyframe editor owns the player; its timeline is under the view
            Text(playback.label).foregroundStyle(.secondary)
        } else {
            bar
        }
    }

    private var bar: some View {
        HStack(spacing: 10) {
            Picker("Move", selection: selection) {
                Text("no move").tag("")
                ForEach(choices, id: \.self) { Text(MovePath.label($0)).tag($0) }
                if !playback.path.isEmpty, !choices.contains(playback.path) {
                    Text(MovePath.label(playback.path)).tag(playback.path)
                }
            }
            .frame(maxWidth: 240)
            if let m = playback.move, m.frames.count > 1 {
                Button { playback.toggle() } label: { Image(systemName: playback.playing ? "pause.fill" : "play.fill") }
                    .keyboardShortcut(keys ? KeyboardShortcut(.space, modifiers: []) : nil)
                    .help("Play / pause (space)")
                Slider(value: $playback.frame, in: 0...Double(m.frames.count - 1), step: 1,
                       onEditingChanged: { editing in if editing { playback.pause() } })
                .frame(width: 200)
                Text("\(Int(playback.frame) + 1)/\(m.frames.count)").monospacedDigit().foregroundStyle(.secondary)
                    .frame(width: 70, alignment: .trailing)
            }
            if let e = playback.loadError {
                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange).help(e)
            }
        }
        .onAppear(perform: refresh)
        .onChange(of: playback.path) { _, _ in refresh() }
    }

    private func refresh() {
        choices = MovePath.list(project: project) + MovePath.previews(project: project)
    }
}

/// Observes the editor so the overlay appears when a move is opened.
private struct EditorOverlayHost: View {
    @ObservedObject var editor: MoveEditor
    var body: some View {
        if let m = editor.move { MoveOverlay(editor: editor, scene: editor.scene, move: m) }
    }
}

private struct EditorTimelineHost: View {
    @ObservedObject var editor: MoveEditor
    let keys: Bool
    var body: some View {
        if let m = editor.move {
            Divider()
            MoveTimeline(editor: editor, playback: editor.scene.playback, move: m, keys: keys)
        }
    }
}


/// The look's crop, drawn over the live model: full-width bands of the chosen aspect. Centred here;
/// the bake follows the head when the move has a head track.
struct LookFrame: View {
    let look: GradeSettings?

    var body: some View {
        GeometryReader { geo in
            if let a = look?.aspect, a > 0, geo.size.width / max(geo.size.height, 1) < a {
                let band = geo.size.width / a
                let bar = (geo.size.height - band) / 2
                VStack(spacing: 0) {
                    Color.black.opacity(0.6).frame(height: bar)
                    Color.clear.frame(height: band).overlay(alignment: .topLeading) {
                        Text(String(format: "%.2f crop · centred here; the bake follows the head when the move has a track", a))
                            .font(.caption2).foregroundStyle(.white.opacity(0.75)).padding(6)
                    }
                    Color.black.opacity(0.6).frame(height: bar)
                }
            }
        }
        .allowsHitTesting(false)
    }
}
