import SwiftUI
import AppKit
import AVKit
import simd
import HSCore

// MARK: - Inspector (right of the viewer)

/// Make a move by keyframes: pick or create it, place the anchor, add keys (K, or the generator
/// buttons), read what it asks of the capture, look at the first / middle / last frame, render.
struct MoveInspector: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var store: ProjectStore
    let project: ProjectSummary
    @ObservedObject var editor: MoveEditor
    @ObservedObject var scene: SplatScene

    @State private var naming: NameAction?
    @State private var newName = ""
    @State private var fileRevision = 0
    /// The move's look, edited live on the model and baked by Render. Saved to grade/<move>.json.
    @State private var look = GradeSettings()
    @State private var lookOnDisk: GradeSettings?
    @State private var showLook = false
    @FocusState private var timeFocused: Bool
    @AppStorage("render.width") private var renderWidth = 2400
    @AppStorage("render.crop") private var renderCrop = true
    @AppStorage("render.keepFrames") private var keepFrames = false

    private enum NameAction: Identifiable {
        case new, duplicate, rename
        var id: Int { hashValue }
        var title: String {
            switch self {
            case .new: return "New move"
            case .duplicate: return "Duplicate move"
            case .rename: return "Rename move"
            }
        }
    }

    private var queue: RunQueue? { model.moveQueues[project.path] }
    private var lockAlive: Bool { project.lock?.alive == true }

    var body: some View {
        VStack(spacing: 0) {
            header.padding(.horizontal, 12).padding(.vertical, 8)
            Divider()
            if let m = editor.move {
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        if editor.otherSolve {
                            Label("These keys were made on a model from another solve — they sit in a different frame here, and the render guard will refuse this pairing.",
                                  systemImage: "exclamationmark.triangle.fill")
                                .foregroundStyle(.orange).font(.callout).fixedSize(horizontal: false, vertical: true)
                        }
                        PlanViews(editor: editor, scene: scene, move: m)
                        generators
                        anchorSection(m)
                        if let id = editor.selected, let k = m.keys.first(where: { $0.id == id }) {
                            keySection(k, m)
                        }
                        numbers(m)
                        lensSection(m)
                        Divider()
                        lookSection()
                        Divider()
                        renderSection(m)
                    }
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            } else {
                emptyState
            }
        }
        .onAppear { editor.attach() }
        .onDisappear { editor.detach(); scene.typing = false }
        .onChange(of: timeFocused) { _, f in scene.typing = f }
        .onChange(of: scene.cameras?.rigNpzMD5) { _, _ in editor.attach() }   // a model finished loading
        .alert(naming?.title ?? "", isPresented: Binding(get: { naming != nil }, set: { if !$0 { naming = nil } })) {
            TextField("name", text: $newName)
            Button("Cancel", role: .cancel) { naming = nil }
            Button("OK") { applyName() }
        } message: {
            Text("Letters, digits, - and _. Saved as move/<name>.keys.json; the render reads move/<name>.json.")
        }
    }

    // MARK: header

    private var header: some View {
        HStack(spacing: 8) {
            Text("Move").font(.headline)
            Picker("Move", selection: Binding(get: { editor.name ?? "" }, set: { n in
                if n.isEmpty { editor.close() } else { editor.open(n) }
            })) {
                Text("none").tag("")
                ForEach(editor.names, id: \.self) { Text($0).tag($0) }
            }
            .labelsHidden()
            Menu {
                Button("New Move From Here…") { newName = nextName(); naming = .new }
                    .disabled(scene.cameras == nil)
                Menu("New From Preset") {
                    ForEach(KeyedMove.Preset.allCases) { p in
                        Button(p.title) { editor.create(preset: p) }
                    }
                }
                .disabled(scene.cameras == nil)
                Divider()
                Button("Duplicate…") { newName = (editor.name ?? "move") + "_b"; naming = .duplicate }
                    .disabled(editor.move == nil)
                Button("Rename…") { newName = editor.name ?? ""; naming = .rename }
                    .disabled(editor.move == nil)
                Divider()
                Button("Close Move") { editor.close() }.disabled(editor.move == nil)
                Button("Reveal in Finder") {
                    if let p = editor.keysPath { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: p)]) }
                }
                .disabled(editor.move == nil)
            } label: { Image(systemName: "plus") }
            .menuStyle(.borderlessButton)
            .fixedSize()
        }
    }

    private var emptyState: some View {
        ScrollView { emptyStateBody }
    }

    private var emptyStateBody: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("A move is keys: camera positions at times, every frame looking at the anchor.")
                .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            VStack(alignment: .leading, spacing: 4) {
                Label("Frame the start in the viewer (drag, scroll, or Stand here at a capture)", systemImage: "1.circle")
                Label("New Move From Here", systemImage: "2.circle")
                Label("K adds a key at the playhead; arc / boom / dolly / hold add keys for you", systemImage: "3.circle")
                Label("Look at the first, middle and last frame, then Render", systemImage: "4.circle")
            }
            .font(.callout)
            Button("New Move From Here…") { newName = nextName(); naming = .new }
                .disabled(scene.cameras == nil)
            Divider()
            Text("Or start from a preset").font(.subheadline.weight(.semibold))
            Text("Each starts where the viewer's camera stands and looks at the anchor. Frame the start first; the speed setting scales the length.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            ForEach(KeyedMove.Preset.allCases) { p in
                Button { editor.create(preset: p) } label: {
                    VStack(alignment: .leading, spacing: 1) {
                        Text(p.title)
                        Text(p.detail).font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.leading)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .buttonStyle(.plain)
                .padding(6)
                .background(RoundedRectangle(cornerRadius: 6).fill(Color.secondary.opacity(0.1)))
                .disabled(scene.cameras == nil)
            }
            Picker("Speed", selection: $editor.speed) {
                ForEach(KeyedMove.Speed.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented).controlSize(.small)
            if scene.cameras == nil {
                Text("Load a model first.").font(.caption).foregroundStyle(.secondary)
            }
            if let e = editor.saveError { Text(e).foregroundStyle(.orange).font(.caption) }
            Spacer()
        }
        .padding(12)
    }

    // MARK: generators

    private var generators: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Add from the last key").font(.subheadline.weight(.semibold))
                Spacer()
                Picker("", selection: $editor.speed) {
                    ForEach(KeyedMove.Speed.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented).labelsHidden().frame(width: 170).controlSize(.small)
            }
            Grid(alignment: .leading, horizontalSpacing: 6, verticalSpacing: 6) {
                GridRow {
                    Text("arc").foregroundStyle(.secondary)
                    gen("◀ 45°") { $0.arc(degrees: -45, speed: $1) }
                    gen("◀ 15°") { $0.arc(degrees: -15, speed: $1) }
                    gen("15° ▶") { $0.arc(degrees: 15, speed: $1) }
                    gen("45° ▶") { $0.arc(degrees: 45, speed: $1) }
                }
                GridRow {
                    Text("boom").foregroundStyle(.secondary)
                    gen("▲ 10°") { $0.boom(degrees: 10, speed: $1) }
                    gen("▲ 3°") { $0.boom(degrees: 3, speed: $1) }
                    gen("▼ 3°") { $0.boom(degrees: -3, speed: $1) }
                    gen("▼ 10°") { $0.boom(degrees: -10, speed: $1) }
                }
                GridRow {
                    Text("dolly").foregroundStyle(.secondary)
                    gen("in 20 cm") { $0.dolly(metres: -0.2, speed: $1) }
                    gen("in 5 cm") { $0.dolly(metres: -0.05, speed: $1) }
                    gen("out 5 cm") { $0.dolly(metres: 0.05, speed: $1) }
                    gen("out 20 cm") { $0.dolly(metres: 0.2, speed: $1) }
                }
                GridRow {
                    Text("hold").foregroundStyle(.secondary)
                    gen("0.5 s") { m, _ in m.hold(seconds: 0.5) }
                    gen("1 s") { m, _ in m.hold(seconds: 1) }
                    gen("2 s") { m, _ in m.hold(seconds: 2) }
                    Color.clear.gridCellUnsizedAxes([.horizontal, .vertical])
                }
            }
            .controlSize(.small)
            Text("Arcs and booms turn about the anchor; arc ▶ moves the camera to its own right. Keys land every 15° so the path follows the circle.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
    }

    private func gen(_ title: String, _ f: @escaping (inout KeyedMove, KeyedMove.Speed) -> Void) -> some View {
        Button(title) { editor.generate(f) }
            .frame(maxWidth: .infinity)
    }

    // MARK: anchor

    private func anchorSection(_ m: KeyedMove) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Anchor").font(.subheadline.weight(.semibold))
                Text(m.anchor == nil ? "the subject (SfM median)" : "placed by hand").foregroundStyle(.secondary)
                Spacer()
            }
            HStack {
                Toggle(isOn: $editor.pickingAnchor) { Label("Pick on model", systemImage: "scope") }
                    .toggleStyle(.button)
                    .disabled(scene.cameras?.points.isEmpty ?? true)
                    .help("Then click the person in the viewer: the anchor snaps to the nearest point of the solve under the click.")
                Button("Reset to subject") { editor.setAnchor(nil) }.disabled(m.anchor == nil)
            }
            .controlSize(.small)
            if editor.pickingAnchor {
                Text("Click the person in the viewer. Esc cancels.").font(.caption).foregroundStyle(Color.accentColor)
            } else if scene.cameras?.points.isEmpty ?? true {
                Text("This model's cameras were exported before sparse points were — press Reload in the viewer bar.")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    // MARK: selected key

    private func keySection(_ k: KeyedMove.Key, _ m: KeyedMove) -> some View {
        let i = (m.sortedKeys.firstIndex { $0.id == k.id } ?? 0) + 1
        return VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Key \(i) of \(m.keys.count)").font(.subheadline.weight(.semibold))
                Spacer()
                Button(role: .destructive) { editor.deleteSelected() } label: { Label("Delete", systemImage: "trash") }
                    .controlSize(.small).disabled(m.keys.count < 2)
            }
            HStack(spacing: 10) {
                Text("at").foregroundStyle(.secondary)
                TextField("s", value: Binding(get: { k.t }, set: { t in editor.edit { $0.retimeKey(k.id, to: t) } }),
                          format: .number.precision(.fractionLength(2)))
                    .textFieldStyle(.roundedBorder).frame(width: 70)
                    .focused($timeFocused)
                Text("s").foregroundStyle(.secondary)
                Toggle("Ease (come to rest)", isOn: Binding(get: { k.ease }, set: { e in
                    editor.edit { m in if let j = m.keys.firstIndex(where: { $0.id == k.id }) { m.keys[j].ease = e } }
                }))
                .disabled(i == 1 || i == m.keys.count)
            }
            .controlSize(.small)
            Text(String(format: "%.0f mm from the anchor", simd_length(k.position - m.anchorPoint) * 1000))
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    // MARK: numbers

    private func numbers(_ m: KeyedMove) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("The move").font(.subheadline.weight(.semibold))
            if let a = editor.analysis {
                Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 3) {
                    row("length", String(format: "%.2f s · %ld frames · %.0f mm", m.duration, m.frameCount, a.pathLength * 1000))
                    row("speed", String(format: "peak %.0f mm/s", a.peakSpeed * 1000))
                    if let lo = a.distance.min(), let hi = a.distance.max() {
                        row("to anchor", String(format: "%.0f–%.0f mm", lo * 1000, hi * 1000))
                    }
                    GridRow {
                        Text("coverage").foregroundStyle(.secondary)
                        Text(String(format: "up to %.1f° off a real camera · %.0f%% of frames past %.0f°",
                                    a.worstAngle, a.fraction(over: MoveAnalysis.redDeg) * 100, MoveAnalysis.redDeg))
                            .foregroundStyle(a.worstAngle > MoveAnalysis.redDeg ? Color.orange : Color.primary)
                    }
                }
                .font(.callout).monospacedDigit()
                if let sp = a.spikeFrame {
                    Label(String(format: "speed jump at %.2f s (frame %ld) — retime the keys around it", Double(sp) / m.fps, sp + 1),
                          systemImage: "bolt.fill").foregroundStyle(.orange).font(.callout)
                }
                Text("Coverage is the angle, seen from the anchor, to the nearest real camera: green under \(Int(MoveAnalysis.amberDeg))°, amber to \(Int(MoveAnalysis.redDeg))°, red beyond. Nothing is blocked — red is where the model has least to go on.")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private func row(_ k: String, _ v: String) -> some View {
        GridRow { Text(k).foregroundStyle(.secondary); Text(v) }
    }

    private func lensSection(_ m: KeyedMove) -> some View {
        HStack(spacing: 8) {
            Text("Lens").foregroundStyle(.secondary)
            Text(String(format: "%@ · %ld×%ld · fx %.0f", m.lens.from ?? "?", m.lens.w, m.lens.h, m.lens.fx))
                .monospacedDigit()
            Spacer()
            if let v = scene.cameras?.view(capture: scene.capture, eye: scene.eye), v.name != m.lens.from {
                Button("Use \(v.capture) \(v.eye)") { editor.useLens(of: v) }.controlSize(.small)
                    .help("Render through this capture's pinhole instead (one lens per move)")
            }
        }
        .font(.callout)
    }

    // MARK: render

    // MARK: look

    /// The look belongs to the render, so it is keyed the way hs grade and the Grade page key it:
    /// grade/<renderName>.json — <move>.json for the current model, <move>_<model>.json for an archive.
    private var lookPath: String? {
        guard let s = editor.script, let mf = scene.loaded else { return nil }
        return GradeSettings.lookPath(project: project.path, renderName: s.renderName(model: mf))
    }
    private var lookKey: String { "\(editor.name ?? "")|\(scene.loaded?.ply ?? "")" }

    private func applyLook() { scene.look = showLook ? look : nil }

    private func loadLook() {
        let saved = lookPath.flatMap { GradeSettings.load($0) }
        lookOnDisk = saved
        look = saved ?? GradeSettings()
        applyLook()
    }

    /// Written only when it differs from what is on disk, so opening a move never creates a file.
    private func saveLook() {
        guard let p = lookPath, look != (lookOnDisk ?? GradeSettings()) else { return }
        try? FileManager.default.createDirectory(atPath: (p as NSString).deletingLastPathComponent,
                                                 withIntermediateDirectories: true)
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let d = try? enc.encode(look), (try? d.write(to: URL(fileURLWithPath: p))) != nil { lookOnDisk = look }
    }

    private func lookSlider(_ name: String, _ v: Binding<Double>, _ r: ClosedRange<Double>, _ fmt: String) -> some View {
        HStack(spacing: 8) {
            Text(name).frame(width: 52, alignment: .leading)
            Slider(value: v, in: r) { EmptyView() }
            Text(String(format: fmt, v.wrappedValue)).monospacedDigit().frame(width: 52, alignment: .trailing)
        }
        .controlSize(.small)
    }

    @ViewBuilder private func lookSection() -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Look").font(.subheadline.weight(.semibold))
                Spacer()
                Toggle("Preview", isOn: $showLook).toggleStyle(.switch).controlSize(.mini)
            }
            Text("Lift, gamma and gain on the live model, through the same 256-entry tables hs grade bakes. With Preview on, Render grades the frames with this look. Sharpening is applied in the bake only.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            lookSlider("Lift", $look.lift, -0.2...0.2, "%+.3f")
            lookSlider("Gamma", $look.gamma, 0.5...2.0, "%.2f")
            lookSlider("Gain", $look.gain, 0.5...1.5, "%.2f")
            HStack {
                Picker("Crop", selection: $look.aspect) {
                    ForEach(GradeSettings.aspectOptions(including: look.aspect), id: \.value) { o in
                        Text(o.label).tag(o.value)
                    }
                }
                .frame(width: 170)
                Spacer()
                Button("Reset") { look = GradeSettings() }.controlSize(.small)
            }
            .controlSize(.small)
            if let p = scene.lookProblem {
                Text(p).font(.caption).foregroundStyle(.orange).fixedSize(horizontal: false, vertical: true)
            }
        }
        .task(id: lookKey) { loadLook() }
        .onChange(of: look) { _, _ in applyLook(); saveLook() }
        .onChange(of: showLook) { _, _ in applyLook() }
        .onDisappear { scene.look = nil }
    }

    @ViewBuilder private func renderSection(_ m: KeyedMove) -> some View {
        let _ = fileRevision
        VStack(alignment: .leading, spacing: 8) {
            Text("Render").font(.subheadline.weight(.semibold))
            if let s = editor.script, let mf = scene.loaded {
                HStack(spacing: 6) {
                    Text("Look at").foregroundStyle(.secondary)
                    ForEach(FrameCheck.Which.allCases) { w in
                        let seen = editor.check.bakedMD5 == editor.bakedMD5 && editor.check.seen.contains(w)
                        Button { editor.look(w) } label: {
                            Label(w.rawValue.capitalized, systemImage: seen ? "checkmark.circle.fill" : "eye")
                        }
                        .tint(seen ? .green : nil)
                    }
                }
                .controlSize(.small)
                Text("The render is enabled once the first, middle and last frame of this exact move have been on screen.")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                HStack(spacing: 10) {
                    Picker("Width", selection: $renderWidth) {
                        Text("1920").tag(1920); Text("2400").tag(2400); Text("3840").tag(3840)
                    }
                    .frame(width: 140)
                    Toggle("4:5 crop", isOn: $renderCrop)
                    Toggle("Keep frames", isOn: $keepFrames)
                }
                .controlSize(.small)
                QueueWatch(queue: queue) { q in
                    let running = q?.isRunning ?? false
                    let why = renderBlocker(m)
                    HStack {
                        Button(running ? "Rendering…" : (showLook ? "Render + grade \(mf.name)" : "Render \(mf.name)")) {
                            editor.flush()
                            run(s.renderArguments(model: mf, width: renderWidth, keepFrames: keepFrames, crop: renderCrop),
                                grade: showLook ? look.arguments(project: project.path, move: s.renderName(model: mf)) : nil)
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(running || why != nil)
                        if running { Button("Stop") { q?.cancel() } }
                        if let w = why {
                            Text(w).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }
                if let run = queue?.current {
                    Observing(run) { r in
                        if r.isRunning || r.finishedAt.map({ Date().timeIntervalSince($0) < 600 }) == true {
                            RunPanel(session: r, showMetrics: false)
                        }
                    }
                }
                RenderResult(paths: s.renders(model: mf))
                    .id("\(s.renderName(model: mf))|\(fileRevision)")
            } else {
                Text("Load a model in the viewer to render with it.").foregroundStyle(.secondary)
            }
        }
    }

    private func renderBlocker(_ m: KeyedMove) -> String? {
        if !model.config.problems.isEmpty { return "fix Setup first" }
        if m.keys.count < 2 || m.duration <= 0 { return "a move needs at least two keys" }
        if editor.bakedMD5 == nil { return "saving…" }
        if !editor.check.complete(for: editor.bakedMD5) {
            let seen = editor.check.bakedMD5 == editor.bakedMD5 ? editor.check.seen : []
            let left = FrameCheck.Which.allCases.filter { !seen.contains($0) }.map(\.rawValue)
            return "click " + ListFormatter.localizedString(byJoining: left) + " above — each shows that frame; Render unlocks after all three"
        }
        if lockAlive { return "\(project.lock?.stage ?? "a stage") is running" }
        return nil
    }

    private func run(_ args: [String], grade: [String]? = nil) {
        var steps = [RunQueue.Step(title: "Render", arguments: args)]
        if let g = grade { steps.append(RunQueue.Step(title: "Grade", arguments: g)) }
        let q = RunQueue(config: model.config, steps: steps)
        let path = project.path
        q.onStep = { s in model.projectRuns[path] = s }
        q.onFinish = { _ in
            store.reload()
            fileRevision += 1
        }
        model.moveQueues[path] = q
        q.start()
    }

    // MARK: naming

    private func nextName() -> String {
        let names = Set(editor.names)
        var i = 1
        while names.contains(String(format: "shot%02d", i)) { i += 1 }
        return String(format: "shot%02d", i)
    }

    private func applyName() {
        let n = newName.trimmingCharacters(in: .whitespaces)
        let action = naming
        naming = nil
        guard MoveScript.validName(n), !editor.names.contains(n) else { NSSound.beep(); return }
        switch action {
        case .new: editor.create(n)
        case .duplicate: editor.duplicate(as: n)
        case .rename: editor.rename(to: n)
        case nil: break
        }
    }
}

// MARK: - Plan and side views

/// The move from above and from the side, in the capture's real geometry: grey dots the
/// captures, the path coloured by coverage, keys, the anchor, and where the viewer stands.
/// Drawn in the app, so it is always there — including before any key exists.
struct PlanViews: View {
    @ObservedObject var editor: MoveEditor
    @ObservedObject var scene: SplatScene
    let move: KeyedMove

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                PlanCanvas(editor: editor, scene: scene, move: move, side: false)
                PlanCanvas(editor: editor, scene: scene, move: move, side: true)
            }
            .frame(height: 170)
            HStack(spacing: 10) {
                Text("above").frame(maxWidth: .infinity)
                Text("side").frame(maxWidth: .infinity)
            }
            .font(.caption2).foregroundStyle(.secondary)
        }
    }
}

private struct PlanCanvas: View {
    @ObservedObject var editor: MoveEditor
    @ObservedObject var scene: SplatScene
    let move: KeyedMove
    let side: Bool

    var body: some View {
        let frame = Frame2D(move: move, captures: captures, side: side)
        TimelineView(.animation(minimumInterval: 1 / 20)) { _ in
            Canvas { ctx, size in
                let map = frame.fit(size)
                // captures
                for c in captures {
                    let p = map(c)
                    ctx.fill(Path(ellipseIn: CGRect(x: p.x - 1.5, y: p.y - 1.5, width: 3, height: 3)),
                             with: .color(.secondary.opacity(0.55)))
                }
                // path, coloured per frame
                let ps = move.framePositions()
                if ps.count > 1, let a = editor.analysis {
                    for i in 1..<ps.count {
                        var seg = Path()
                        seg.move(to: map(ps[i - 1])); seg.addLine(to: map(ps[i]))
                        ctx.stroke(seg, with: .color(coverageColor(a.offAngle[min(i, a.offAngle.count - 1)])), lineWidth: 2)
                    }
                }
                // keys
                for k in move.keys {
                    let p = map(k.position)
                    let r: CGFloat = k.id == editor.selected ? 5 : 3.5
                    var d = Path()
                    d.move(to: CGPoint(x: p.x, y: p.y - r)); d.addLine(to: CGPoint(x: p.x + r, y: p.y))
                    d.addLine(to: CGPoint(x: p.x, y: p.y + r)); d.addLine(to: CGPoint(x: p.x - r, y: p.y)); d.closeSubpath()
                    ctx.fill(d, with: .color(k.id == editor.selected ? .white : .accentColor))
                }
                // anchor
                let a = map(move.anchorPoint)
                var cross = Path()
                cross.move(to: CGPoint(x: a.x - 6, y: a.y)); cross.addLine(to: CGPoint(x: a.x + 6, y: a.y))
                cross.move(to: CGPoint(x: a.x, y: a.y - 6)); cross.addLine(to: CGPoint(x: a.x, y: a.y + 6))
                ctx.stroke(cross, with: .color(.yellow), lineWidth: 1.5)
                // where the viewer stands, and where the playhead is
                let e = scene.cameraPosition
                let eye = map(SIMD3(Double(e.x), Double(e.y), Double(e.z)))
                ctx.stroke(Path(ellipseIn: CGRect(x: eye.x - 4, y: eye.y - 4, width: 8, height: 8)), with: .color(.white), lineWidth: 1.5)
                var look = Path(); look.move(to: eye); look.addLine(to: a)
                ctx.stroke(look, with: .color(.white.opacity(0.25)), style: StrokeStyle(lineWidth: 1, dash: [3, 3]))
            }
        }
        .background(RoundedRectangle(cornerRadius: 6).fill(Color.black.opacity(0.35)))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    private var captures: [SIMD3<Double>] {
        (scene.cameras?.views ?? []).filter { $0.eye == "L" }.compactMap { v in
            v.pose.map { SIMD3(Double($0.position.x), Double($0.position.y), Double($0.position.z)) }
        }
    }
}

/// Two axes of the move's world for a 2-D view: above = the plane square to up (x along the
/// captures' mean bearing), side = that bearing against up.
private struct Frame2D {
    let origin: SIMD3<Double>
    let ax: SIMD3<Double>
    let ay: SIMD3<Double>
    let bounds: CGRect

    init(move: KeyedMove, captures: [SIMD3<Double>], side: Bool) {
        let up = move.upVector
        origin = move.anchorPoint
        var mean = captures.reduce(SIMD3<Double>.zero) { $0 + ($1 - move.anchorPoint) }
        mean -= simd_dot(mean, up) * up
        if simd_length(mean) < 1e-9 {
            mean = simd_cross(up, SIMD3(1, 0, 0))
            if simd_length(mean) < 1e-9 { mean = simd_cross(up, SIMD3(0, 0, 1)) }
        }
        let fwd = simd_normalize(mean)
        let right = simd_normalize(simd_cross(up, fwd))
        if side {
            ax = fwd; ay = up             // side: the captures' bearing across, up is up
        } else {
            ax = right; ay = -fwd         // above, looking down: captures below the anchor, not mirrored
        }
        var r = CGRect.null
        for p in captures + move.keys.map(\.position) + [move.anchorPoint] {
            let v = p - move.anchorPoint
            r = r.union(CGRect(x: simd_dot(v, ax), y: simd_dot(v, ay), width: 0, height: 0))
        }
        bounds = r.isNull ? CGRect(x: -1, y: -1, width: 2, height: 2) : r.insetBy(dx: -0.1 * max(r.width, 0.2), dy: -0.1 * max(r.height, 0.2))
    }

    /// World point -> canvas point, uniform scale, y up.
    func fit(_ size: CGSize) -> (SIMD3<Double>) -> CGPoint {
        let s = min(size.width / max(bounds.width, 1e-6), size.height / max(bounds.height, 1e-6))
        let ox = (size.width - bounds.width * s) / 2, oy = (size.height - bounds.height * s) / 2
        let b = bounds, o = origin, ax = ax, ay = ay
        return { p in
            let v = p - o
            let x = simd_dot(v, ax), y = simd_dot(v, ay)
            return CGPoint(x: ox + (x - b.minX) * s, y: size.height - (oy + (y - b.minY) * s))
        }
    }
}

func coverageColor(_ deg: Double) -> Color {
    deg <= MoveAnalysis.amberDeg ? .green : (deg <= MoveAnalysis.redDeg ? .yellow : .red)
}

// MARK: - Timeline (under the viewer)

/// Keys on a time ruler: click to select and jump, drag to retime; drag the playhead to scrub;
/// the band under the ruler is coverage per frame, the line the speed.
struct MoveTimeline: View {
    @ObservedObject var editor: MoveEditor
    @ObservedObject var playback: MovePlayback
    let move: KeyedMove
    var keys = true

    @State private var dragging: UUID?

    var body: some View {
        VStack(spacing: 6) {
            HStack(spacing: 10) {
                Button { playback.toggle() } label: { Image(systemName: playback.playing ? "pause.fill" : "play.fill") }
                    .keyboardShortcut(keys ? KeyboardShortcut(.space, modifiers: []) : nil)
                    .help("Play / pause (space)")
                Text(String(format: "%.2f / %.2f s", editor.playhead, move.duration)).monospacedDigit()
                    .frame(width: 110, alignment: .leading)
                Button { editor.keyHere() } label: { Label("Key", systemImage: "diamond.fill") }
                    .keyboardShortcut(keys ? KeyboardShortcut("k", modifiers: []) : nil)
                    .help("K: a key at the playhead from the viewer's camera — updates the key under the playhead, or appends 2 s after the last one when the playhead is at the end")
                Button { editor.deleteSelected() } label: { Image(systemName: "trash") }
                    .keyboardShortcut(keys ? KeyboardShortcut(.delete, modifiers: []) : nil)
                    .disabled(editor.selected == nil || move.keys.count < 2)
                    .help("Delete the selected key (⌫)")
                Spacer()
                Text("drag the viewer to frame · K to key · space to play").font(.caption).foregroundStyle(.secondary)
            }
            .controlSize(.small)
            GeometryReader { geo in
                let w = geo.size.width, span = max(move.duration, 1)
                let x: (Double) -> CGFloat = { CGFloat($0 / span) * (w - 16) + 8 }
                let t: (CGFloat) -> Double = { max(0, min(span, Double(($0 - 8) / (w - 16)) * span)) }
                ZStack(alignment: .topLeading) {
                    // ruler
                    Canvas { ctx, size in
                        let step: Double = span > 30 ? 5 : (span > 10 ? 2 : 1)
                        var s = 0.0
                        while s <= span + 1e-9 {
                            var p = Path(); p.move(to: CGPoint(x: x(s), y: 0)); p.addLine(to: CGPoint(x: x(s), y: 6))
                            ctx.stroke(p, with: .color(.secondary), lineWidth: 1)
                            ctx.draw(Text(String(format: "%.0f", s)).font(.system(size: 9)).foregroundColor(.secondary),
                                     at: CGPoint(x: x(s) + 2, y: 12), anchor: .leading)
                            s += step
                        }
                        // coverage band + speed line
                        if let a = editor.analysis, move.frameCount > 1 {
                            let n = move.frameCount
                            for i in 0..<n {
                                let x0 = x(Double(i) / move.fps), x1 = x(Double(i + 1) / move.fps)
                                ctx.fill(Path(CGRect(x: x0, y: 22, width: max(x1 - x0, 1), height: 6)),
                                         with: .color(coverageColor(a.offAngle[i]).opacity(0.8)))
                            }
                            let peak = max(a.peakSpeed, 1e-6)
                            var line = Path()
                            for i in 0..<n {
                                let pt = CGPoint(x: x(Double(i) / move.fps), y: size.height - 4 - CGFloat(a.speed[i] / peak) * 22)
                                if i == 0 { line.move(to: pt) } else { line.addLine(to: pt) }
                            }
                            ctx.stroke(line, with: .color(.secondary.opacity(0.7)), lineWidth: 1)
                        }
                    }
                    // keys
                    ForEach(move.keys) { k in
                        Image(systemName: k.ease ? "diamond.fill" : "diamond")
                            .foregroundStyle(k.id == editor.selected ? Color.white : Color.accentColor)
                            .font(.system(size: 13))
                            .position(x: x(k.t), y: 40)
                            .gesture(DragGesture(minimumDistance: 2)
                                .onChanged { g in
                                    dragging = k.id
                                    editor.selected = k.id
                                    editor.edit { $0.retimeKey(k.id, to: t(g.location.x)) }
                                }
                                .onEnded { _ in dragging = nil })
                            .onTapGesture {
                                editor.selected = k.id
                                editor.seek(k.t)
                            }
                            .help(String(format: "key at %.2f s%@ — drag to retime", k.t, k.ease ? ", comes to rest" : ""))
                    }
                    // playhead
                    Rectangle().fill(Color.white).frame(width: 1.5, height: geo.size.height)
                        .position(x: x(editor.playhead), y: geo.size.height / 2)
                        .allowsHitTesting(false)
                }
                .contentShape(Rectangle())
                .gesture(DragGesture(minimumDistance: 0).onChanged { g in
                    if dragging == nil { editor.seek(t(g.location.x)) }
                })
            }
            .frame(height: 64)
        }
        .padding(.horizontal, 12).padding(.vertical, 8)
    }
}

// MARK: - Overlay on the viewer

/// The move drawn over the 3-D view: path (coverage colours), keys, the anchor, and — when the
/// view is a pinhole — the frame the render will see and its 4:5 crop. Clicks pass through
/// except while picking the anchor.
struct MoveOverlay: View {
    @ObservedObject var editor: MoveEditor
    let scene: SplatScene
    let move: KeyedMove

    var body: some View {
        GeometryReader { geo in
            TimelineView(.animation) { _ in
                Canvas { ctx, size in
                    // frame and crop guides
                    if let p = scene.screenPose {
                        let s = min(size.width / CGFloat(p.w), size.height / CGFloat(p.h))
                        let fw = CGFloat(p.w) * s, fh = CGFloat(p.h) * s
                        let frame = CGRect(x: (size.width - fw) / 2, y: (size.height - fh) / 2, width: fw, height: fh)
                        var outside = Path(CGRect(origin: .zero, size: size))
                        outside.addRect(frame)
                        ctx.fill(outside, with: .color(.black.opacity(0.45)), style: FillStyle(eoFill: true))
                        let cw = fh * 4 / 5
                        let crop = CGRect(x: frame.midX - cw / 2, y: frame.minY, width: cw, height: fh)
                        ctx.stroke(Path(crop), with: .color(.white.opacity(0.35)), style: StrokeStyle(lineWidth: 1, dash: [5, 4]))
                        ctx.draw(Text("4:5").font(.caption2).foregroundColor(.white.opacity(0.6)),
                                 at: CGPoint(x: crop.minX + 4, y: crop.minY + 4), anchor: .topLeading)
                    }
                    // path
                    let ps = move.framePositions()
                    if ps.count > 1, let a = editor.analysis {
                        var prev: CGPoint?
                        for (i, p) in ps.enumerated() {
                            let q = scene.screenPoint(SIMD3(Float(p.x), Float(p.y), Float(p.z)), size: size)
                            if let q = q, let pr = prev {
                                var seg = Path(); seg.move(to: pr); seg.addLine(to: q)
                                ctx.stroke(seg, with: .color(coverageColor(a.offAngle[min(i, a.offAngle.count - 1)]).opacity(0.9)),
                                           lineWidth: 2)
                            }
                            prev = q
                        }
                    }
                    for k in move.keys {
                        guard let q = scene.screenPoint(SIMD3(Float(k.position.x), Float(k.position.y), Float(k.position.z)), size: size)
                        else { continue }
                        let r: CGFloat = k.id == editor.selected ? 6 : 4.5
                        var d = Path()
                        d.move(to: CGPoint(x: q.x, y: q.y - r)); d.addLine(to: CGPoint(x: q.x + r, y: q.y))
                        d.addLine(to: CGPoint(x: q.x, y: q.y + r)); d.addLine(to: CGPoint(x: q.x - r, y: q.y)); d.closeSubpath()
                        ctx.fill(d, with: .color(k.id == editor.selected ? .white : .accentColor))
                    }
                    // anchor
                    let a = move.anchorPoint
                    if let q = scene.screenPoint(SIMD3(Float(a.x), Float(a.y), Float(a.z)), size: size) {
                        var c = Path()
                        c.addEllipse(in: CGRect(x: q.x - 9, y: q.y - 9, width: 18, height: 18))
                        c.move(to: CGPoint(x: q.x - 14, y: q.y)); c.addLine(to: CGPoint(x: q.x - 4, y: q.y))
                        c.move(to: CGPoint(x: q.x + 4, y: q.y)); c.addLine(to: CGPoint(x: q.x + 14, y: q.y))
                        c.move(to: CGPoint(x: q.x, y: q.y - 14)); c.addLine(to: CGPoint(x: q.x, y: q.y - 4))
                        c.move(to: CGPoint(x: q.x, y: q.y + 4)); c.addLine(to: CGPoint(x: q.x, y: q.y + 14))
                        ctx.stroke(c, with: .color(.yellow), lineWidth: 1.5)
                    }
                }
            }
            .contentShape(Rectangle())
            .onTapGesture { q in
                guard editor.pickingAnchor else { return }
                if let p = scene.pickPoint(at: q, size: geo.size) {
                    editor.setAnchor(p)
                } else {
                    NSSound.beep()
                }
            }
            .allowsHitTesting(editor.pickingAnchor)
            .onExitCommand { editor.pickingAnchor = false }
        }
        .overlay(alignment: .top) {
            if editor.pickingAnchor {
                Text("Click the person to anchor on · Esc cancels")
                    .font(.callout).padding(.horizontal, 10).padding(.vertical, 4)
                    .background(Capsule().fill(.black.opacity(0.6))).foregroundStyle(.white).padding(8)
            }
        }
    }
}

// MARK: - Pieces

/// Re-renders its content when the queue publishes (a queue held in a dictionary does not).
struct QueueWatch<Content: View>: View {
    @ObservedObject private var queue: RunQueue
    private let real: Bool
    let content: (RunQueue?) -> Content

    @MainActor
    init(queue: RunQueue?, @ViewBuilder content: @escaping (RunQueue?) -> Content) {
        self._queue = ObservedObject(wrappedValue: queue ?? IdleQueue.shared)
        self.real = queue != nil
        self.content = content
    }

    var body: some View { content(real ? queue : nil) }
}

/// A queue that never runs, for QueueWatch to observe when there is none (a generic type
/// cannot hold a static stored property).
@MainActor
private enum IdleQueue {
    static let shared = RunQueue(config: EngineConfig.guess(), steps: [])
}

/// The encoded result(s): 16:9 and the 4:5 crop, played in place.
private struct RenderResult: View {
    let paths: [String]
    @State private var chosen = ""

    var body: some View {
        if !paths.isEmpty {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Picker("", selection: $chosen) {
                        ForEach(paths, id: \.self) { Text($0.hasSuffix("_1080x1350.mp4") ? "4:5" : "16:9").tag($0) }
                    }
                    .pickerStyle(.segmented).labelsHidden()
                    Spacer()
                    Button("Open") { NSWorkspace.shared.open(URL(fileURLWithPath: chosen)) }.controlSize(.small)
                    Button("Reveal") { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: chosen)]) }
                        .controlSize(.small)
                }
                if !chosen.isEmpty {
                    PlayerView(url: URL(fileURLWithPath: chosen))
                        .aspectRatio(chosen.hasSuffix("_1080x1350.mp4") ? 4.0 / 5.0 : 16.0 / 9.0, contentMode: .fit)
                        .frame(maxHeight: 420)
                }
            }
            .onAppear { if chosen.isEmpty || !paths.contains(chosen) { chosen = paths[0] } }
        }
    }
}

/// AppKit's player. SwiftUI's VideoPlayer (the _AVKit_SwiftUI overlay) aborts at first display
/// in this app — its class metadata cannot find its superclass (crash of 2026-09-19).
struct PlayerView: NSViewRepresentable {
    let url: URL

    func makeNSView(context: Context) -> AVPlayerView {
        let v = AVPlayerView()
        v.controlsStyle = .inline
        v.player = AVPlayer(url: url)
        return v
    }

    func updateNSView(_ v: AVPlayerView, context: Context) {
        if (v.player?.currentItem?.asset as? AVURLAsset)?.url != url {
            v.player?.pause()
            v.player = AVPlayer(url: url)
        }
    }

    static func dismantleNSView(_ v: AVPlayerView, coordinator: ()) {
        v.player?.pause()
    }
}
