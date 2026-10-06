import SwiftUI
import simd
import HSCore

// Shot › Clean up. Two ways to take floaters out of a trained model, both into a copy beside it
// (the trained model is never changed):
//
//   on the camera's path   `hs prune --clear-path`: whatever sits where the camera itself went.
//                          Can also be set to run at the end of every training run.
//   by hand                drag a box round a floater in the viewer, set how deep the box
//                          reaches, erase: `hs prune --erase`.
//
// The viewer is the Move tab's (`ModelViewerPane(cleaning: true)`); this file is the panel beside
// it, the box tool, and the dots drawn over the picture.

// MARK: - On the camera's path

/// How far from the camera's path the clean reaches, in the capture's own steps between frames
/// (`--clear-path 1.3x`), so it means the same on a table-top orbit and on a walk through a garden.
enum PathStrength: String, CaseIterable, Identifiable {
    case gentle, strong
    var id: String { rawValue }
    var title: String { self == .gentle ? "Gentle" : "Strong" }
    var meaning: String {
        self == .gentle ? "Only what hugs the camera's path. Start here."
                        : "Nearly twice as far out. Clears more, and can take a leaf that brushed the lens."
    }
    var steps: String { self == .gentle ? "1.3x" : "2.4x" }
}

/// The switch that makes the Train step finish with a gentle clean of the camera's path. Off
/// unless it is turned on in Shot › Clean up; both models are kept, so the two can be compared.
enum CleanAfterTraining {
    static let key = "clean.afterTraining"
    static var isOn: Bool { UserDefaults.standard.bool(forKey: key) }
    static func step(project: String) -> RunQueue.Step {
        RunQueue.Step(title: "Clean the camera's path",
                      arguments: ["prune", "-p", project, "--clear-path", PathStrength.gentle.steps])
    }
}

// MARK: - By hand: the boxes

/// The boxes dragged over the viewer. A box is the camera it was drawn from, a rectangle on that
/// camera's screen, and how far from that camera it reaches; a splat whose centre is inside goes.
/// `hs prune --erase` runs the same test (engine/hs/clearpath.py, inside_boxes).
@MainActor
final class EraserTool: ObservableObject {
    struct Box: Identifiable {
        let id = UUID()
        /// projection × view when the box was drawn: world metres to clip, w = distance in front.
        let vp: simd_float4x4
        /// The rectangle in normalised device coordinates: x right, y up, −1…1.
        let x0: Float
        let x1: Float
        let y0: Float
        let y1: Float
        /// Distance from that camera (m) of every splat inside the rectangle, nearest first.
        let depths: [Float]
        /// Some of those splats, nearest first, for the dots.
        let sample: [SIMD3<Float>]
        /// How much of `depths` goes, nearest first: 0…1.
        var share: Double

        var count: Int { min(depths.count, Int((share * Double(depths.count)).rounded())) }
        /// The box reaches this far from the camera it was drawn from (m).
        var reach: Float { count > 0 ? depths[count - 1] : 0 }
        var dots: ArraySlice<SIMD3<Float>> {
            guard !depths.isEmpty, count > 0 else { return [] }
            let n = Int((Double(sample.count) * Double(count) / Double(depths.count)).rounded(.up))
            return sample.prefix(max(n, 1))
        }
    }

    /// Dragging in the viewer draws a box instead of turning the camera.
    @Published var selecting = false
    @Published private(set) var boxes: [Box] = []
    @Published private(set) var busy = false
    @Published var note: String?

    /// Where the last clean took splats from (some of them), the copy they are missing from, and
    /// whether to draw them: the answer to "it says 14,000 went — where?".
    @Published private(set) var tookOut: [SIMD3<Float>] = []
    @Published private(set) var tookOutFrom: String?
    @Published var showTookOut = true

    func setTookOut(_ centres: [SIMD3<Float>], from ply: String?) {
        tookOut = centres
        tookOutFrom = ply
        showTookOut = true
    }

    static let maxBoxes = 12
    static let maxDots = 4000

    /// Splats marked over all boxes (a splat in two boxes counts twice; the erase says the real number).
    var marked: Int { boxes.reduce(0) { $0 + $1.count } }

    func clear() {
        boxes = []
        note = nil
    }

    func remove(_ id: UUID) { boxes.removeAll { $0.id == id } }

    func setShare(_ id: UUID, _ share: Double) {
        guard let i = boxes.firstIndex(where: { $0.id == id }) else { return }
        boxes[i].share = min(max(share, 0), 1)
    }

    /// A box dragged over a viewer of `size` points (`r` in the same points, y down).
    func addBox(_ r: CGRect, in size: CGSize, scene: SplatScene) {
        guard !busy, size.width > 0, size.height > 0 else { return }
        guard boxes.count < EraserTool.maxBoxes else {
            note = "That is \(EraserTool.maxBoxes) boxes. Erase these first, then draw more."
            return
        }
        let centres = scene.centres
        guard !centres.isEmpty else {
            note = "The model is still loading."
            return
        }
        let m = scene.matrices(width: Float(size.width), height: Float(size.height))
        let vp = m.projection * m.view
        let x0 = Float(r.minX / size.width) * 2 - 1
        let x1 = Float(r.maxX / size.width) * 2 - 1
        let y1 = 1 - Float(r.minY / size.height) * 2
        let y0 = 1 - Float(r.maxY / size.height) * 2
        let dots = EraserTool.maxDots
        busy = true
        note = nil
        Task {
            let found = await Task.detached(priority: .userInitiated) {
                EraserTool.gather(centres: centres, vp: vp, x0: x0, x1: x1, y0: y0, y1: y1, dots: dots)
            }.value
            self.busy = false
            guard !found.depths.isEmpty else {
                self.note = "No splats in that box."
                return
            }
            // to begin with: what is less than half way to the middle of everything in the box,
            // which is where a floater in front of the scene sits
            let half = 0.5 * found.depths[found.depths.count / 2]
            var lo = 0
            var hi = found.depths.count
            while lo < hi {
                let mid = (lo + hi) / 2
                if found.depths[mid] <= half { lo = mid + 1 } else { hi = mid }
            }
            if lo == 0 {
                self.note = "Nothing in that box is clearly in front of the rest. Drag How deep to choose what goes."
            }
            self.boxes.append(Box(vp: vp, x0: x0, x1: x1, y0: y0, y1: y1, depths: found.depths, sample: found.sample,
                                  share: Double(lo) / Double(found.depths.count)))
        }
    }

    /// Every splat centre inside the rectangle and in front of the camera: distances nearest
    /// first, and an even sample of the centres in the same order.
    nonisolated static func gather(centres: [SIMD3<Float>], vp: simd_float4x4,
                                   x0: Float, x1: Float, y0: Float, y1: Float,
                                   dots: Int) -> (depths: [Float], sample: [SIMD3<Float>]) {
        var hits: [(Float, Int32)] = []
        for i in centres.indices {
            let p = centres[i]
            let c = vp * SIMD4<Float>(p.x, p.y, p.z, 1)
            let w = c.w
            if w > 1e-6 {
                let x = c.x / w
                let y = c.y / w
                if x >= x0 && x <= x1 && y >= y0 && y <= y1 { hits.append((w, Int32(i))) }
            }
        }
        hits.sort { $0.0 < $1.0 }
        let depths = hits.map { $0.0 }
        let step = max(1, hits.count / max(dots, 1))
        var sample: [SIMD3<Float>] = []
        var k = 0
        while k < hits.count {
            sample.append(centres[Int(hits[k].1)])
            k += step
        }
        return (depths, sample)
    }

    private struct FileBox: Encodable {
        let vp: [[Double]]
        let x0: Double
        let x1: Double
        let y0: Double
        let y1: Double
        let near: Double
        let far: Double
    }

    private struct File: Encodable {
        let splats: Int
        let boxes: [FileBox]
    }

    /// What `hs prune --erase` reads: each box's view as four rows, its rectangle and its reach.
    func fileData(splats: Int) throws -> Data {
        let out = boxes.filter { $0.count > 0 }.map { b -> FileBox in
            let rows: [[Double]] = (0..<4).map { r in (0..<4).map { c in Double(b.vp[c][r]) } }
            // a hair past the last splat counted, so rounding in the engine does not leave it behind
            return FileBox(vp: rows, x0: Double(b.x0), x1: Double(b.x1), y0: Double(b.y0), y1: Double(b.y1),
                           near: 0, far: Double(b.reach) * 1.000001)
        }
        return try JSONEncoder().encode(File(splats: splats, boxes: out))
    }
}

/// Over the viewer in the Clean up tab: red dots on what the boxes will take, the box being
/// dragged, and — while Select is on — the layer that takes the drag instead of the camera.
struct EraserOverlay: View {
    @ObservedObject var tool: EraserTool
    let scene: SplatScene
    @State private var from: CGPoint?
    @State private var to: CGPoint?

    private static func rect(_ a: CGPoint, _ b: CGPoint) -> CGRect {
        CGRect(x: min(a.x, b.x), y: min(a.y, b.y), width: abs(a.x - b.x), height: abs(a.y - b.y))
    }

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .topLeading) {
                TimelineView(.animation) { _ in
                    Canvas { ctx, size in
                        if tool.showTookOut, !tool.tookOut.isEmpty, tool.tookOutFrom == scene.loaded?.ply {
                            var gone = Path()
                            for p in tool.tookOut {
                                if let q = scene.screenPoint(p, size: size) {
                                    gone.addRect(CGRect(x: q.x - 1.5, y: q.y - 1.5, width: 3, height: 3))
                                }
                            }
                            ctx.fill(gone, with: .color(Color.yellow))
                        }
                        var path = Path()
                        for b in tool.boxes {
                            for p in b.dots {
                                if let q = scene.screenPoint(p, size: size) {
                                    path.addRect(CGRect(x: q.x - 1.25, y: q.y - 1.25, width: 2.5, height: 2.5))
                                }
                            }
                        }
                        ctx.fill(path, with: .color(Color.red))
                    }
                }
                .allowsHitTesting(false)
                if tool.selecting {
                    Color.clear
                        .contentShape(Rectangle())
                        .gesture(DragGesture(minimumDistance: 3)
                            .onChanged { v in
                                from = v.startLocation
                                to = v.location
                            }
                            .onEnded { v in
                                let r = EraserOverlay.rect(v.startLocation, v.location)
                                from = nil
                                to = nil
                                if r.width > 4, r.height > 4 { tool.addBox(r, in: geo.size, scene: scene) }
                            })
                    Text(tool.busy ? "Counting the splats in the box…" : "Drag a box round a floater")
                        .font(.callout)
                        .padding(.horizontal, 10).padding(.vertical, 5)
                        .background(Capsule().fill(Color.black.opacity(0.6)))
                        .foregroundStyle(Color.white)
                        .padding(10)
                        .allowsHitTesting(false)
                }
                if let a = from, let b = to {
                    let r = EraserOverlay.rect(a, b)
                    Rectangle()
                        .stroke(Color.white, style: StrokeStyle(lineWidth: 1, dash: [5, 4]))
                        .frame(width: r.width, height: r.height)
                        .offset(x: r.minX, y: r.minY)
                        .allowsHitTesting(false)
                }
            }
        }
    }
}

// MARK: - The panel beside the viewer

struct CleanUpPanel: View {
    @EnvironmentObject var model: AppModel
    let project: ProjectSummary
    @ObservedObject var scene: SplatScene
    @ObservedObject var tool: EraserTool
    /// The model the viewer is showing.
    let chosen: ViewerModelFile?
    /// A cleaned copy was written at this path: list the files again and show it.
    let onCleaned: (String) -> Void

    @AppStorage(CleanAfterTraining.key) private var afterTraining = false
    @State private var strength = PathStrength.gentle
    @State private var pathOutcome: String?
    @State private var pathFailed = false
    @State private var handOutcome: String?
    @State private var handFailed = false

    private var running: Bool { model.cleanQueues[project.path]?.isRunning == true }
    private var busyElsewhere: Bool {
        !running && (model.projectRuns[project.path]?.isRunning == true || project.lock?.alive == true)
    }
    /// The picture is the model named in the picker, fully loaded.
    private var ready: Bool { chosen != nil && scene.loaded == chosen && scene.phase == .ready }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("Clean up").font(.title2.weight(.bold))
                Text("Take floaters out of the model before you move the camera through it. Everything here writes a copy; the trained model stays as it was, and the picker above the picture switches between them.")
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                if !tool.tookOut.isEmpty, tool.tookOutFrom == chosen?.ply {
                    Toggle("Show where the last clean took splats from (yellow dots)", isOn: $tool.showTookOut)
                    Text("They can be behind you or out where the camera stood, so pull the view back to find them.")
                        .font(.caption).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if busyElsewhere {
                    Text("Another step is running on this project. Cleaning can start when it ends.")
                        .foregroundStyle(Color.orange)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Divider()
                pathCard
                Divider()
                handCard
            }
            .padding(16)
        }
        .onChange(of: scene.loaded) { _, _ in tool.clear() }
        .onDisappear { tool.selecting = false }
    }

    // MARK: on the camera's path

    private var pathBlocked: String? {
        guard let f = chosen else { return "Load a model first." }
        if f.isCleanUpOutput { return "The model showing is already a cleaned copy. Pick the trained model in the picker to clean its path." }
        return nil
    }

    private var pathCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Automatic: the camera's path").font(.headline)
            Text("Takes out the specks and haze sitting where the camera itself went. Nothing real can be there, and a move along the camera's path flies straight through them.")
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            HStack(alignment: .top, spacing: 8) {
                ForEach(PathStrength.allCases) { s in
                    ChoiceTile(title: s.title, subtitle: s.meaning, selected: strength == s) { strength = s }
                }
            }
            if let o = pathOutcome {
                Text(o).foregroundStyle(pathFailed ? Color.red : Color.primary)
                    .fixedSize(horizontal: false, vertical: true)
            } else if let b = pathBlocked {
                Text(b).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            HStack(spacing: 10) {
                Button(running ? "Working…" : "Clean a Copy") { cleanPath() }
                    .buttonStyle(.borderedProminent)
                    .disabled(running || busyElsewhere || pathBlocked != nil)
                if running { ProgressView().controlSize(.small) }
            }
            Toggle("Do this at the end of every training run", isOn: $afterTraining)
            Text("Off until you turn it on. The Train step then finishes with a gentle clean and keeps both models, so you can compare them here after a run.")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func cleanPath() {
        guard let f = chosen, pathBlocked == nil else { return }
        let path = project.path
        // gentle is what the Train step's own clean writes, so the two land in one file
        let name = strength == .gentle ? f.cleanUpStem : f.cleanUpStem + "_" + strength.rawValue
        let out = ((path as NSString).appendingPathComponent("prune") as NSString)
            .appendingPathComponent(name + "_clearpath.ply")
        run(title: "Clean the camera's path",
            arguments: ["prune", "-p", path, "--clear-path", strength.steps, "--ply", f.ply, "--name", name],
            out: out, tookOut: String(out.dropLast("_clearpath.ply".count)) + "_path_only.ply",
            countKey: "prune.on_the_path", hand: false)
    }

    // MARK: by hand

    private var handCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("By hand").font(.headline)
            Text("Choose Select and drag a box round a floater in the picture. Red dots mark what will go. Switch back to Look Around and turn the view to check the dots sit on the floater, not on the scene behind it.")
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Picker("", selection: $tool.selecting) {
                Text("Look Around").tag(false)
                Text("Select").tag(true)
            }
            .pickerStyle(.segmented).labelsHidden()
            .disabled(!ready || running)
            ForEach(Array(tool.boxes.enumerated()), id: \.element.id) { i, b in
                boxRow(i, b)
            }
            if let n = tool.note {
                Text(n).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            if let o = handOutcome {
                Text(o).foregroundStyle(handFailed ? Color.red : Color.primary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack(spacing: 10) {
                Button(running ? "Working…" : "Erase What Is Marked") { erase() }
                    .buttonStyle(.borderedProminent)
                    .disabled(running || busyElsewhere || !ready || tool.marked == 0)
                Button("Clear Boxes") { tool.clear() }
                    .disabled(tool.boxes.isEmpty || running)
            }
            Text("Erasing writes one hand-cleaned copy of the model and keeps adding to it each time. To start over, pick the trained model in the picker and erase from there.")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func boxRow(_ i: Int, _ b: EraserTool.Box) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Box \(i + 1)").fontWeight(.semibold)
                Spacer()
                Button { tool.remove(b.id) } label: { Image(systemName: "xmark.circle.fill") }
                    .buttonStyle(.plain)
                    .help("Forget this box")
            }
            Text("\(CleanUpPanel.grouped(Double(b.count))) of the \(CleanUpPanel.grouped(Double(b.depths.count))) splats in it, up to \(String(format: "%.2f", b.reach)) m from where you drew it")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            HStack(spacing: 8) {
                Text("How deep").font(.caption)
                Slider(value: Binding(get: { b.share }, set: { tool.setShare(b.id, $0) }), in: 0...1)
            }
        }
        .padding(8)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color(nsColor: .controlBackgroundColor)))
    }

    private func erase() {
        guard let f = chosen, ready, tool.marked > 0 else { return }
        let path = project.path
        let stem = ((f.ply as NSString).lastPathComponent as NSString).deletingPathExtension
        // one hand-cleaned copy per model: erasing while that copy is showing adds to it in place
        let name: String
        if f.isCleanUpOutput, stem.hasSuffix("_hand_erased") {
            name = String(stem.dropLast("_erased".count))
        } else {
            name = (f.isCleanUpOutput ? stem : f.cleanUpStem) + "_hand"
        }
        let dir = (path as NSString).appendingPathComponent("prune")
        let spec = (dir as NSString).appendingPathComponent(name + "_erase.json")
        let out = (dir as NSString).appendingPathComponent(name + "_erased.ply")
        do {
            try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
            try tool.fileData(splats: scene.centres.count).write(to: URL(fileURLWithPath: spec), options: .atomic)
        } catch {
            handFailed = true
            handOutcome = "Could not write the boxes into the project: \(error.localizedDescription)"
            return
        }
        run(title: "Erase by hand",
            arguments: ["prune", "-p", path, "--erase", spec, "--ply", f.ply, "--name", name],
            out: out, tookOut: String(out.dropLast(".ply".count)) + "_only.ply",
            countKey: "prune.erased", hand: true)
    }

    // MARK: running either

    private func run(title: String, arguments: [String], out: String, tookOut: String, countKey: String, hand: Bool) {
        let path = project.path
        let q = RunQueue(config: model.config, steps: [RunQueue.Step(title: title, arguments: arguments)])
        if hand {
            handOutcome = nil
            handFailed = false
        } else {
            pathOutcome = nil
            pathFailed = false
        }
        q.onStep = { s in model.projectRuns[path] = s }
        q.onFinish = { done in
            model.cleanQueues[path] = nil
            model.store.reload()
            let session = done.current
            var bad = false
            let text: String
            switch done.state {
            case .finished:
                let removed = session?.metrics.last(where: { $0.key == countKey })?.event.value?.double
                let total = session?.metrics.last(where: { $0.key == "prune.splats_in" })?.event.value?.double
                if let a = removed, let t = total, t > 0 {
                    text = a == 0
                        ? "Nothing was there to take out; the copy is the same as the model."
                        : "Took out \(CleanUpPanel.grouped(a)) of \(CleanUpPanel.grouped(t)) splats (\(String(format: "%.2f", 100 * a / t)) %). The copy is showing now, with yellow dots where they were; the picker above the picture has the model it came from."
                } else {
                    text = "Done. The copy is showing now."
                }
                if hand { tool.selecting = false }
                tool.setTookOut(PlyCentres.read(tookOut, limit: EraserTool.maxDots), from: out)
                onCleaned(out)
            case .cancelled:
                text = "Stopped."
            default:
                bad = true
                let e = session?.errors.last
                let said = [e?.message, e?.hint].compactMap { $0 }.joined(separator: " — ")
                text = said.isEmpty ? "It did not finish. The Console has the run's last words." : said
            }
            if hand {
                handOutcome = text
                handFailed = bad
            } else {
                pathOutcome = text
                pathFailed = bad
            }
        }
        model.cleanQueues[path] = q
        q.start()
    }

    static func grouped(_ x: Double) -> String {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        f.maximumFractionDigits = 0
        return f.string(from: NSNumber(value: x)) ?? String(Int(x))
    }
}

// MARK: - Where a clean took splats from

/// Some of the centres in a Gaussian .ply (metres), evenly thinned to about `limit`: enough to
/// draw where a clean took splats from. Empty if the file is not a little-endian binary .ply
/// with float x, y, z.
enum PlyCentres {
    private static let sizes: [String: Int] = [
        "float": 4, "float32": 4, "double": 8, "float64": 8, "uchar": 1, "uint8": 1, "char": 1, "int8": 1,
        "short": 2, "int16": 2, "ushort": 2, "uint16": 2, "int": 4, "int32": 4, "uint": 4, "uint32": 4,
    ]

    static func read(_ path: String, limit: Int) -> [SIMD3<Float>] {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path), options: .mappedIfSafe),
              let end = data.range(of: Data("end_header\n".utf8)),
              let head = String(data: data.subdata(in: 0..<end.lowerBound), encoding: .ascii),
              head.contains("binary_little_endian") else { return [] }
        var count = 0
        var stride = 0
        var at: [String: Int] = [:]
        for line in head.split(separator: "\n") {
            let w = line.split(separator: " ").map(String.init)
            guard w.count == 3 else { continue }
            if w[0] == "element", w[1] == "vertex" {
                count = Int(w[2]) ?? 0
            } else if w[0] == "property" {
                guard let n = sizes[w[1]] else { return [] }
                if n == 4, w[1].hasPrefix("float") { at[w[2]] = stride }
                stride += n
            }
        }
        guard count > 0, stride > 0, let ox = at["x"], let oy = at["y"], let oz = at["z"],
              data.count >= end.upperBound + count * stride else { return [] }
        let body = end.upperBound
        let step = max(1, count / max(limit, 1))
        var out: [SIMD3<Float>] = []
        out.reserveCapacity(count / step + 1)
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            var i = 0
            while i < count {
                let o = body + i * stride
                out.append(SIMD3<Float>(raw.loadUnaligned(fromByteOffset: o + ox, as: Float.self),
                                        raw.loadUnaligned(fromByteOffset: o + oy, as: Float.self),
                                        raw.loadUnaligned(fromByteOffset: o + oz, as: Float.self)))
                i += step
            }
        }
        return out
    }
}

// MARK: - What the copies are called

extension ViewerModelFile {
    /// A copy `hs prune` wrote (prune/…), as against a trained or a kept model.
    var isCleanUpOutput: Bool { name.hasPrefix("prune/") }

    /// What a cleaned copy of this model is filed under: the kept model's name, or the export's stem.
    var cleanUpStem: String {
        archive ?? ((ply as NSString).lastPathComponent as NSString).deletingPathExtension
    }

    /// What the model picker calls it. prune/export_20000_strong_clearpath.ply reads
    /// "path cleaned, strong · export_20000"; prune/export_20000_hand_erased.ply reads
    /// "erased by hand · export_20000".
    var pickerName: String {
        guard isCleanUpOutput else { return name }
        var stem = ((ply as NSString).lastPathComponent as NSString).deletingPathExtension
        var words: [String] = []
        var only = false
        // read the name from its end: each clean adds its own ending to the model it started from
        var found = true
        while found {
            found = false
            for (ending, word) in [("_erased_only", "what erasing took out"), ("_path_only", "what the path clean took out")]
            where stem.hasSuffix(ending) {
                stem = String(stem.dropLast(ending.count))
                words.append(word)
                only = true
                found = true
            }
            if stem.hasSuffix("_erased") {
                stem = String(stem.dropLast("_erased".count))
                if !only { words.append("erased by hand") }
                found = true
            }
            if stem.hasSuffix("_hand") {
                stem = String(stem.dropLast("_hand".count))
                found = true
            }
            if stem.hasSuffix("_clearpath") {
                stem = String(stem.dropLast("_clearpath".count))
                var how = "path cleaned"
                if stem.hasSuffix("_strong") {
                    stem = String(stem.dropLast("_strong".count))
                    how += ", strong"
                }
                if !only { words.append(how) }
                found = true
            }
            if stem.hasSuffix("_strong"), only, !words.isEmpty {
                stem = String(stem.dropLast("_strong".count))
                words[words.count - 1] += ", strong"
                found = true
            }
        }
        if words.isEmpty { return name }
        return words.reversed().joined(separator: " + ") + " · " + stem + (isFromEarlierRun ? " (earlier training run)" : "")
    }

    /// A cleaned copy of a model in train/exports that has since been trained again: exports keep
    /// their names from run to run (export_20000.ply), so a copy made before the last training run
    /// would otherwise read as a copy of the new model.
    var isFromEarlierRun: Bool {
        guard isCleanUpOutput else { return false }
        let file = (ply as NSString).lastPathComponent
        guard let r = file.range(of: #"^export_\d+"#, options: .regularExpression) else { return false }
        let source = ((project as NSString).appendingPathComponent("train/exports") as NSString)
            .appendingPathComponent(String(file[r]) + ".ply")
        let fm = FileManager.default
        func changed(_ path: String) -> Date? { (try? fm.attributesOfItem(atPath: path))?[.modificationDate] as? Date }
        guard let mine = changed(ply) else { return false }
        guard let theirs = changed(source) else { return true }     // that export is gone
        return mine < theirs
    }
}
