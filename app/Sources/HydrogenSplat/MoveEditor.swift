import SwiftUI
import CryptoKit
import simd
import HSCore

/// The open keyframed move of one project: its keys, what they ask of the capture, the render
/// check, and the link to the viewer (the free camera locked to the anchor and lens; the
/// timeline driving the scene's player).
///
/// Every edit re-bakes in memory at once (the viewer shows it) and saves after a short pause:
/// move/<name>.keys.json (the source) and move/<name>.json (what `hs render` reads).
@MainActor
final class MoveEditor: ObservableObject {
    let project: String
    let scene: SplatScene

    @Published private(set) var names: [String] = []
    @Published private(set) var name: String?
    @Published private(set) var move: KeyedMove?
    @Published var selected: UUID?
    @Published private(set) var analysis: MoveAnalysis?
    @Published private(set) var bakedMD5: String?
    @Published private(set) var saveError: String?
    @Published var check = FrameCheck()
    @Published var speed: KeyedMove.Speed = .normal
    @Published var pickingAnchor = false

    private var saveTask: Task<Void, Never>?

    init(project: String, scene: SplatScene) {
        self.project = project
        self.scene = scene
        refreshNames()
    }

    var keysPath: String? { name.map { KeyedMove.path(project: project, name: $0) } }
    var script: MoveScript? { name.map { MoveScript(project: project, name: $0) } }

    func refreshNames() { names = KeyedMove.list(project: project) }

    // MARK: open / new

    func open(_ n: String) {
        flush()
        guard let m = try? KeyedMove.load(path: KeyedMove.path(project: project, name: n)) else {
            saveError = "could not read move/\(n).keys.json"
            return
        }
        name = n
        selected = nil
        check = FrameCheck()
        pickingAnchor = false
        apply(m, save: false)
        bakedMD5 = script.flatMap { MoveScript.md5($0.builtPath) }
        scene.playback.frame = 0
        scene.playback.show()
    }

    func close() {
        flush()
        name = nil
        move = nil
        analysis = nil
        selected = nil
        pickingAnchor = false
        scene.lock(anchor: nil, lens: nil)
        if scene.playback.isKeyed { scene.playback.clear() }
    }

    /// A new move whose first key is where the viewer's camera stands now.
    func create(_ n: String) {
        guard let cams = scene.cameras else { saveError = "load a model first — a move is made in its frame"; return }
        let view = cams.view(capture: scene.capture, eye: scene.eye) ?? cams.views.first
        guard let v = view else { return }
        let s = cams.subject, u = cams.up, e = scene.cameraPosition
        var m = KeyedMove(subject: SIMD3(Double(s.x), Double(s.y), Double(s.z)),
                          up: SIMD3(Double(u.x), Double(u.y), Double(u.z)),
                          lens: KeyedMove.Lens(v), rigNpzMD5: cams.rigNpzMD5)
        m.keys = [.init(t: 0, eye: SIMD3(Double(e.x), Double(e.y), Double(e.z)), ease: true)]
        flush()
        name = n
        check = FrameCheck()
        apply(m, save: false)
        saveNow()               // now, not after the pause: the picker lists files on disk
        refreshNames()
    }

    /// A new move from a preset, starting where the viewer's camera stands.
    func create(preset p: KeyedMove.Preset) {
        var n = p.name, i = 2
        while names.contains(n) { n = "\(p.name)_\(i)"; i += 1 }
        create(n)
        guard move != nil else { return }
        let e = scene.cameraPosition
        let eye = SIMD3(Double(e.x), Double(e.y), Double(e.z))
        let s = speed
        let front = frontBearing
        edit { $0.applyPreset(p, from: eye, front: front, speed: s) }
        saveNow()
        seek(0)
    }

    /// From the anchor toward the captures, on average: the side that was filmed.
    private var frontBearing: SIMD3<Double>? {
        guard let m = move, let views = scene.cameras?.views else { return nil }
        let a = m.anchorPoint
        let sum = views.compactMap { $0.pose?.position }.reduce(SIMD3<Double>.zero) {
            $0 + SIMD3(Double($1.x), Double($1.y), Double($1.z)) - a
        }
        return simd_length(sum) > 1e-9 ? simd_normalize(sum) : nil
    }

    func duplicate(as n: String) {
        guard let m = move else { return }
        flush()
        name = n
        check = FrameCheck()
        apply(m, save: false)
        saveNow()               // now, not after the pause: the picker lists files on disk
        refreshNames()
    }

    func rename(to n: String) {
        guard let old = name, let m = move else { return }
        flush()
        let fm = FileManager.default
        try? fm.removeItem(atPath: KeyedMove.path(project: project, name: old))
        name = n
        apply(m, save: false)
        saveNow()               // now, not after the pause: the picker lists files on disk
        refreshNames()
    }

    /// The inspector is on screen: lock the viewer to this move again.
    func attach() {
        refreshNames()
        if let m = move { apply(m, save: false) }
    }

    /// The inspector went away (panel hidden, page or project changed): free the viewer.
    func detach() {
        flush()
        pickingAnchor = false
        scene.lock(anchor: nil, lens: nil)
        if scene.playback.isKeyed { scene.playback.clear() }
    }

    // MARK: editing

    func edit(_ f: (inout KeyedMove) -> Void) {
        guard var m = move else { return }
        f(&m)
        apply(m, save: true)
    }

    /// K: a key at the playhead from the viewer's camera. At the end of the move it appends two
    /// seconds later; on an existing key it updates that key.
    func keyHere() {
        guard let m = move else { return }
        let e = scene.cameraPosition
        let eye = SIMD3(Double(e.x), Double(e.y), Double(e.z))
        var t = playhead
        if !m.keys.isEmpty, m.keyIndex(near: t) == nil, t >= m.duration - 0.5 / m.fps { t = m.duration + 2 }
        var id: UUID?
        edit { id = $0.setKey(at: t, eye: eye) }
        selected = id
        seek(t)
    }

    func deleteSelected() {
        guard let id = selected, (move?.keys.count ?? 0) > 1 else { return }
        edit { $0.removeKey(id) }
        selected = nil
    }

    func generate(_ f: (inout KeyedMove, KeyedMove.Speed) -> Void) {
        let s = speed
        edit { f(&$0, s) }
        if let m = move { seek(m.duration) }
    }

    func setAnchor(_ p: SIMD3<Float>?) {
        edit { $0.anchor = p.map { [Double($0.x), Double($0.y), Double($0.z)] } }
        pickingAnchor = false
    }

    func useLens(of v: ViewerCamera) { edit { $0.lens = KeyedMove.Lens(v) } }

    // MARK: playhead

    var playhead: Double { scene.playback.frame / max(move?.fps ?? 30, 1) }

    func seek(_ t: Double) {
        guard let m = move else { return }
        scene.playback.pause()
        let f = (min(max(t, 0), m.duration) * m.fps).rounded()
        if scene.playback.frame != f { scene.playback.frame = f } else { scene.playback.show() }
    }

    func look(_ w: FrameCheck.Which) {
        guard let m = move else { return }
        scene.playback.pause()
        let f = Double(w.frame(of: m.frameCount))
        if scene.playback.frame != f { scene.playback.frame = f } else { scene.playback.show() }
        check.saw(w, md5: bakedMD5)
    }

    // MARK: apply / save

    private func apply(_ m: KeyedMove, save: Bool) {
        move = m
        let caps = (scene.cameras?.views ?? []).compactMap { v -> SIMD3<Double>? in
            guard let p = v.pose?.position else { return nil }
            return SIMD3(Double(p.x), Double(p.y), Double(p.z))
        }
        analysis = MoveAnalysis(move: m, captures: caps)
        let a = m.anchorPoint
        scene.lock(anchor: SIMD3(Float(a.x), Float(a.y), Float(a.z)), lens: m.lens)
        scene.playback.load(inMemory: m.bakedPath(), label: name ?? "move", show: false)
        if save { scheduleSave() }
    }

    private func scheduleSave() {
        saveTask?.cancel()
        saveTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 400_000_000)
            guard !Task.isCancelled else { return }
            self?.saveNow()
        }
    }

    /// Write anything pending (before switching moves or leaving).
    func flush() {
        if saveTask != nil {
            saveTask?.cancel()
            saveTask = nil
            saveNow()
        }
    }

    private func saveNow() {
        saveTask = nil
        guard let m = move, let n = name, let s = script else { return }
        do {
            try m.save(path: KeyedMove.path(project: project, name: n))
            let data = try m.bakedJSON()
            try data.write(to: URL(fileURLWithPath: s.builtPath), options: .atomic)
            bakedMD5 = Insecure.MD5.hash(data: data).map { String(format: "%02x", $0) }.joined()
            saveError = nil
        } catch {
            saveError = error.localizedDescription
        }
    }

    // MARK: facts for the inspector

    /// The keys are in the frame of a different solve than the model in the viewer.
    var otherSolve: Bool {
        guard let m = move, let c = scene.cameras else { return false }
        return m.rigNpzMD5 != c.rigNpzMD5
    }
}
