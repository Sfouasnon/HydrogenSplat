import Foundation
import simd

// MARK: - A move as keyframes (move/<name>.keys.json)

/// A camera move authored in the viewer: camera positions at times, all looking at one anchor.
/// The app bakes it into move/<name>.json — the file `hs render` reads — so the path that is
/// scrubbed in the viewer is the path that renders.
///
/// Positions are in the solve's world frame, metres (the .ply's unit). Every frame looks at the
/// anchor with the horizon level to `up`, so no key can be aimed at nothing.
public struct KeyedMove: Codable, Equatable, Sendable {
    public struct Key: Codable, Equatable, Identifiable, Sendable {
        public var id: UUID
        /// Seconds from the start.
        public var t: Double
        /// Camera centre, metres.
        public var eye: [Double]
        /// The camera comes to rest here (ease in and out). Off: it passes through.
        public var ease: Bool

        public init(id: UUID = UUID(), t: Double, eye: SIMD3<Double>, ease: Bool) {
            self.id = id
            self.t = t
            self.eye = [eye.x, eye.y, eye.z]
            self.ease = ease
        }

        public var position: SIMD3<Double> {
            eye.count == 3 ? SIMD3(eye[0], eye[1], eye[2]) : .zero
        }
    }

    /// The pinhole every frame is rendered through (a real capture's, so the render matches it).
    public struct Lens: Codable, Equatable, Sendable {
        public var w: Int
        public var h: Int
        public var fx: Double
        public var fy: Double
        public var cx: Double
        public var cy: Double
        /// Which capture it came from, for the inspector.
        public var from: String?

        public init(w: Int, h: Int, fx: Double, fy: Double, cx: Double, cy: Double, from: String?) {
            self.w = w; self.h = h; self.fx = fx; self.fy = fy; self.cx = cx; self.cy = cy; self.from = from
        }

        public init(_ c: ViewerCamera) {
            self.init(w: c.w, h: c.h, fx: c.fx, fy: c.fy, cx: c.cx, cy: c.cy, from: c.name)
        }

        public var verticalFOV: Double { 2 * atan(Double(h) / (2 * fy)) }
        public var K: [[Double]] { [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] }
    }

    public var schema = 1
    public var fps: Double
    /// The point every frame looks at, metres. nil = the subject (the SfM median).
    public var anchor: [Double]?
    /// The subject when the move was made (the anchor's default).
    public var subject: [Double]
    /// World up, unit (from the solve's cameras).
    public var up: [Double]
    public var lens: Lens
    /// The rig.npz the positions are in — a model from another solve sits in another frame.
    public var rigNpzMD5: String
    public var keys: [Key]

    enum CodingKeys: String, CodingKey {
        case schema, fps, anchor, subject, up, lens, keys
        case rigNpzMD5 = "rig_npz_md5"
    }

    public init(fps: Double = 30, anchor: SIMD3<Double>? = nil, subject: SIMD3<Double>, up: SIMD3<Double>,
                lens: Lens, rigNpzMD5: String, keys: [Key] = []) {
        self.fps = fps
        self.anchor = anchor.map { [$0.x, $0.y, $0.z] }
        self.subject = [subject.x, subject.y, subject.z]
        self.up = [up.x, up.y, up.z]
        self.lens = lens
        self.rigNpzMD5 = rigNpzMD5
        self.keys = keys
    }

    public var anchorPoint: SIMD3<Double> {
        let a = anchor ?? subject
        return a.count == 3 ? SIMD3(a[0], a[1], a[2]) : .zero
    }
    public var subjectPoint: SIMD3<Double> { subject.count == 3 ? SIMD3(subject[0], subject[1], subject[2]) : .zero }
    public var upVector: SIMD3<Double> {
        let u = up.count == 3 ? SIMD3(up[0], up[1], up[2]) : SIMD3(0, -1, 0)
        return simd_normalize(u)
    }

    public var sortedKeys: [Key] { keys.sorted { $0.t < $1.t } }
    public var duration: Double { sortedKeys.last?.t ?? 0 }
    public var frameCount: Int { keys.isEmpty ? 0 : Int((duration * fps).rounded()) + 1 }

    // MARK: files

    public static func path(project: String, name: String) -> String {
        ((project as NSString).appendingPathComponent("move") as NSString).appendingPathComponent("\(name).keys.json")
    }

    public static func load(path: String) throws -> KeyedMove {
        try JSONDecoder().decode(KeyedMove.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
    }

    public func save(path: String) throws {
        let e = JSONEncoder()
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        try FileManager.default.createDirectory(atPath: (path as NSString).deletingLastPathComponent,
                                                withIntermediateDirectories: true)
        try e.encode(self).write(to: URL(fileURLWithPath: path), options: .atomic)
    }

    /// move/*.keys.json names, newest first.
    public static func list(project: String) -> [String] {
        let dir = (project as NSString).appendingPathComponent("move")
        let fm = FileManager.default
        func mtime(_ p: String) -> Date { ((try? fm.attributesOfItem(atPath: p))?[.modificationDate] as? Date) ?? .distantPast }
        return ((try? fm.contentsOfDirectory(atPath: dir)) ?? [])
            .filter { $0.hasSuffix(".keys.json") && !$0.hasPrefix(".") }
            .sorted { mtime((dir as NSString).appendingPathComponent($0)) > mtime((dir as NSString).appendingPathComponent($1)) }
            .map { String($0.dropLast(".keys.json".count)) }
    }

    // MARK: baking

    /// Camera centre at time t (seconds): a centripetal Catmull-Rom spline through the keys.
    /// Centripetal Catmull-Rom is smooth in its own knot parameter τ (cumulative √distance
    /// between keys), so time maps to τ by one monotone cubic across all keys (Fritsch-Carlson):
    /// slope 0 at an eased key — the camera comes to rest — and at a pass-through key the
    /// harmonic mean of the neighbouring rates, so speed carries through the key instead of
    /// stepping there, however unevenly the keys are spaced. Never backs up.
    public func position(at t: Double) -> SIMD3<Double> {
        let k = sortedKeys
        guard let first = k.first else { return anchorPoint }
        if k.count == 1 || t <= first.t { return first.position }
        if t >= k[k.count - 1].t { return k[k.count - 1].position }
        var i = 0
        while i + 1 < k.count - 1 && t >= k[i + 1].t { i += 1 }
        let a = k[i], b = k[i + 1]
        let dt = max(b.t - a.t, 1e-9)
        let dtau = KeyedMove.knot(a.position, b.position)
        guard dtau > 1e-9 else { return a.position }          // a hold
        let m0 = KeyedMove.rate(k, i) * dt / dtau, m1 = KeyedMove.rate(k, i + 1) * dt / dtau
        let u = KeyedMove.retime((t - a.t) / dt, m0: m0, m1: m1)
        let p0 = i > 0 ? k[i - 1].position : a.position
        let p3 = i + 2 < k.count ? k[i + 2].position : b.position
        return KeyedMove.catmullRom(p0, a.position, b.position, p3, u)
    }

    /// √distance: the centripetal knot interval (0 for a hold).
    static func knot(_ a: SIMD3<Double>, _ b: SIMD3<Double>) -> Double { pow(simd_length(b - a), 0.5) }

    /// dτ/dt at key i: 0 when it eases (or at either end, or next to a hold); else the harmonic
    /// mean of the secants Δτ/Δt either side, which is at most twice the smaller one — the
    /// Fritsch-Carlson bound that keeps each segment's cubic monotone.
    static func rate(_ k: [Key], _ i: Int) -> Double {
        if k[i].ease || i == 0 || i == k.count - 1 { return 0 }
        let sa = knot(k[i - 1].position, k[i].position) / max(k[i].t - k[i - 1].t, 1e-9)
        let sb = knot(k[i].position, k[i + 1].position) / max(k[i + 1].t - k[i].t, 1e-9)
        guard sa > 1e-12, sb > 1e-12 else { return 0 }
        return 2 * sa * sb / (sa + sb)
    }

    /// Hermite from 0 to 1 over a segment with end slopes m0, m1 (in segment units). Monotone
    /// for slopes in [0, 3]; the rates above give at most 2.
    static func retime(_ s: Double, m0: Double, m1: Double) -> Double {
        let s = min(max(s, 0), 1)
        let s2 = s * s, s3 = s2 * s
        return (s3 - 2 * s2 + s) * m0 + (-2 * s3 + 3 * s2) + (s3 - s2) * m1
    }

    /// Centripetal Catmull-Rom (alpha 0.5): no cusps or overshoot loops when keys are uneven.
    /// Coincident points (a hold) give a still segment instead of a division by zero.
    static func catmullRom(_ p0: SIMD3<Double>, _ p1: SIMD3<Double>, _ p2: SIMD3<Double>, _ p3: SIMD3<Double>,
                           _ u: Double) -> SIMD3<Double> {
        func knot(_ a: SIMD3<Double>, _ b: SIMD3<Double>) -> Double { max(KeyedMove.knot(a, b), 1e-6) }
        if simd_length(p2 - p1) < 1e-9 { return p1 }
        let t0 = 0.0
        let t1 = t0 + knot(p0, p1)
        let t2 = t1 + knot(p1, p2)
        let t3 = t2 + knot(p2, p3)
        let t = t1 + (t2 - t1) * u
        let a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        let a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        let a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        let b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
        let b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        return (t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2
    }

    /// OpenCV camera-to-world at a position: z toward the anchor, x right, y down, level to up.
    public func cameraToWorld(eye: SIMD3<Double>) -> simd_double4x4 {
        KeyedMove.levelCameraToWorld(eye: eye, target: anchorPoint, up: upVector)
    }

    /// The one definition of a move frame's orientation, shared by the baker and the viewer.
    public static func levelCameraToWorld(eye: SIMD3<Double>, target: SIMD3<Double>, up: SIMD3<Double>) -> simd_double4x4 {
        var fwd = target - eye
        let upVector = simd_normalize(up)
        if simd_length(fwd) < 1e-9 { fwd = -upVector }
        fwd = simd_normalize(fwd)
        let down = -upVector
        var right = simd_cross(down, fwd)
        if simd_length(right) < 1e-9 {          // looking straight along up: any level right will do
            right = simd_cross(down, simd_normalize(fwd + SIMD3(1e-3, 0, 0)))
        }
        right = simd_normalize(right)
        let dn = simd_cross(fwd, right)
        return simd_double4x4(columns: (SIMD4(right, 0), SIMD4(dn, 0), SIMD4(fwd, 0), SIMD4(eye, 1)))
    }

    /// Every frame's camera centre, frame 0 at t = 0.
    public func framePositions() -> [SIMD3<Double>] {
        (0..<frameCount).map { position(at: Double($0) / fps) }
    }

    /// The move JSON `hs render` reads (same shape `hs move` writes).
    public func bakedJSON() throws -> Data {
        let frames = framePositions().map { p -> [String: Any] in
            let m = cameraToWorld(eye: p)
            let rows = (0..<4).map { r in (0..<4).map { c in m[c][r] } }
            return ["c2w": rows]
        }
        let obj: [String: Any] = [
            "note": "OpenCV camera convention (x right, y down, z forward), metres; baked by the HydrogenSplat app from "
                + "keyframes (move/<name>.keys.json), every frame looking at the anchor",
            "width": lens.w, "height": lens.h, "K": lens.K, "fps": fps, "frames": frames,
            "anchor_m": [anchorPoint.x, anchorPoint.y, anchorPoint.z],
        ]
        return try JSONSerialization.data(withJSONObject: obj, options: [])
    }

    /// The baked move for the viewer's player, without a round trip through disk.
    public func bakedPath() -> MovePath {
        MovePath(width: lens.w, height: lens.h, K: lens.K, fps: fps,
                 frames: framePositions().map { p in
                     let m = cameraToWorld(eye: p)
                     return MovePath.Frame(c2w: (0..<4).map { r in (0..<4).map { c in m[c][r] } })
                 })
    }

    // MARK: editing

    /// A key at t, or one frame away from it.
    public func keyIndex(near t: Double) -> Int? {
        keys.firstIndex { abs($0.t - t) < 0.5 / fps }
    }

    public mutating func setKey(at t: Double, eye: SIMD3<Double>) -> UUID {
        if let i = keyIndex(near: t) {
            keys[i].eye = [eye.x, eye.y, eye.z]
            return keys[i].id
        }
        let k = Key(t: max(0, t), eye: eye, ease: keys.isEmpty)
        keys.append(k)
        normaliseEasing()
        return k.id
    }

    /// The ends of a move come to rest; a key in the middle keeps whatever it was given.
    public mutating func normaliseEasing() {
        keys.sort { $0.t < $1.t }
        guard !keys.isEmpty else { return }
        keys[0].ease = true
        keys[keys.count - 1].ease = true
    }

    public mutating func removeKey(_ id: UUID) {
        keys.removeAll { $0.id == id }
        normaliseEasing()
    }

    /// Move a key in time, never onto or past its neighbours (one frame apart at least).
    public mutating func retimeKey(_ id: UUID, to t: Double) {
        let ks = sortedKeys
        guard let i = ks.firstIndex(where: { $0.id == id }) else { return }
        let lo = i > 0 ? ks[i - 1].t + 1 / fps : 0
        let hi = i + 1 < ks.count ? ks[i + 1].t - 1 / fps : .greatestFiniteMagnitude
        guard lo <= hi, let j = keys.firstIndex(where: { $0.id == id }) else { return }
        keys[j].t = (min(max(t, lo), hi) * fps).rounded() / fps
    }

    // MARK: generators (the cue vocabulary, as keys)

    public enum Speed: String, CaseIterable, Identifiable, Sendable {
        case slow, normal, fast
        public var id: String { rawValue }
        var factor: Double { self == .slow ? 0.5 : (self == .fast ? 2 : 1) }
    }

    /// Base rates at `normal`: gentle enough for a person at a metre.
    static let arcDegPerSec = 10.0
    static let boomDegPerSec = 6.0
    static let dollyMPerSec = 0.06

    /// The last key, or nil: every generator continues from the end of the move.
    private var tail: Key? { sortedKeys.last }

    /// Orbit about the anchor, around `up`. Positive degrees move the camera to its own right
    /// (facing the anchor); negative, to its left. Keys every 15° so the spline follows the
    /// circle instead of cutting its chord.
    public mutating func arc(degrees: Double, speed: Speed) {
        guard let last = tail else { return }
        let steps = max(1, Int((abs(degrees) / 15).rounded(.up)))
        let secs = abs(degrees) / (KeyedMove.arcDegPerSec * speed.factor)
        unEaseTail()
        for s in 1...steps {
            let f = Double(s) / Double(steps)
            let q = simd_quatd(angle: degrees * f * .pi / 180, axis: upVector)
            let p = anchorPoint + q.act(last.position - anchorPoint)
            keys.append(Key(t: last.t + secs * f, eye: p, ease: s == steps))
        }
        normaliseEasing()
    }

    /// Change elevation about the anchor: positive raises the camera. Stops 3° short of
    /// straight up or down.
    public mutating func boom(degrees: Double, speed: Speed) {
        guard let last = tail else { return }
        let v = last.position - anchorPoint
        let r = simd_length(v)
        guard r > 1e-6 else { return }
        let el = asin(max(-1, min(1, simd_dot(v / r, upVector))))
        let limit = (90.0 - 3.0) * .pi / 180
        let target = max(-limit, min(limit, el + degrees * .pi / 180))
        let d = target - el
        guard abs(d) > 1e-6 else { return }
        var right = simd_cross(upVector, v)
        if simd_length(right) < 1e-9 { return }
        right = simd_normalize(right)
        let steps = max(1, Int((abs(d) * 180 / .pi / 15).rounded(.up)))
        let secs = abs(d) * 180 / .pi / (KeyedMove.boomDegPerSec * speed.factor)
        unEaseTail()
        for s in 1...steps {
            let f = Double(s) / Double(steps)
            // rotating v about (up x v) by -angle raises it toward up
            let q = simd_quatd(angle: -d * f, axis: right)
            keys.append(Key(t: last.t + secs * f, eye: anchorPoint + q.act(v), ease: s == steps))
        }
        normaliseEasing()
    }

    /// Straight toward (negative) or away from (positive) the anchor, metres. Never closer than 5 cm.
    public mutating func dolly(metres: Double, speed: Speed) {
        guard let last = tail else { return }
        let v = last.position - anchorPoint
        let r = simd_length(v)
        guard r > 1e-6 else { return }
        let nr = max(0.05, r + metres)
        let secs = abs(nr - r) / (KeyedMove.dollyMPerSec * speed.factor)
        guard secs > 0 else { return }
        unEaseTail()
        keys.append(Key(t: last.t + secs, eye: anchorPoint + v / r * nr, ease: true))
        normaliseEasing()
    }

    /// Stand still: the last position again, later.
    public mutating func hold(seconds: Double) {
        guard let last = tail, seconds > 0 else { return }
        if let i = keys.firstIndex(where: { $0.id == last.id }) { keys[i].ease = true }
        keys.append(Key(t: last.t + seconds, eye: last.position, ease: true))
        normaliseEasing()
    }

    // MARK: presets

    /// Shots with a pace of their own. Each replaces the keys, starting from the viewer's camera,
    /// and eases in and out over the whole move (keys spaced on an S-curve).
    public enum Preset: String, CaseIterable, Identifiable, Sendable {
        case orbit180, orbit360, pushIn, pullOut, arcPush, craneReveal
        public var id: String { rawValue }

        public var title: String {
            switch self {
            case .orbit180: return "180° orbit"
            case .orbit360: return "360° orbit"
            case .pushIn: return "Slow push in"
            case .pullOut: return "Slow pull out"
            case .arcPush: return "Arc and push"
            case .craneReveal: return "Crane up and out"
            }
        }

        public var detail: String {
            switch self {
            case .orbit180: return "Half a turn across the captured side, centred on it, at this height and distance — 18 s"
            case .orbit360: return "A full turn from here. It will show the side nobody filmed — 36 s"
            case .pushIn: return "A third of the way in, straight at the anchor — 8 s"
            case .pullOut: return "Half as far again back — 8 s"
            case .arcPush: return "30° round while closing a fifth of the distance — 10 s"
            case .craneReveal: return "Up 15° while pulling back a quarter — 9 s"
            }
        }

        public var name: String {
            switch self {
            case .orbit180: return "orbit180"
            case .orbit360: return "orbit360"
            case .pushIn: return "push_in"
            case .pullOut: return "pull_out"
            case .arcPush: return "arc_push"
            case .craneReveal: return "crane_reveal"
            }
        }

        /// (yaw deg, pitch deg, radius factor, seconds at normal speed)
        var shape: (Double, Double, Double, Double) {
            switch self {
            case .orbit180: return (180, 0, 1, 18)
            case .orbit360: return (360, 0, 1, 36)
            case .pushIn: return (0, 0, 2.0 / 3.0, 8)
            case .pullOut: return (0, 0, 1.5, 8)
            case .arcPush: return (30, 0, 0.8, 10)
            case .craneReveal: return (0, 15, 1.25, 9)
            }
        }
    }

    /// Replace the keys with a preset. `front`: the direction from the anchor toward the
    /// captures (horizontal) — the 180 is centred on it so it stays on the filmed side.
    public mutating func applyPreset(_ p: Preset, from eye: SIMD3<Double>, front: SIMD3<Double>?, speed: Speed) {
        let (yaw, pitch, factor, base) = p.shape
        let secs = base / speed.factor
        var start = eye
        if p == .orbit180, let f = front {
            // same height and distance as the eye, at the front bearing - 90 deg
            let up = upVector
            let v = eye - anchorPoint
            let h = simd_dot(v, up)
            let horiz = simd_length(v - h * up)
            var fr = f - simd_dot(f, up) * up
            if simd_length(fr) > 1e-9 && horiz > 1e-9 {
                fr = simd_normalize(fr)
                let q = simd_quatd(angle: -.pi / 2, axis: up)
                start = anchorPoint + q.act(fr) * horiz + h * up
            }
        }
        keys = KeyedMove.sweep(anchor: anchorPoint, up: upVector, from: start, yaw: yaw, pitch: pitch,
                               radiusFactor: factor, seconds: secs)
    }

    /// One continuous move: turn `yaw` about up, `pitch` in elevation, scale the distance — all
    /// together — with keys evenly spaced in the motion and timed on a smoothstep, so the
    /// camera eases in and out over the whole shot. Keys at most 15° apart (the spline follows
    /// the circle) and at least 5 of them.
    static func sweep(anchor: SIMD3<Double>, up: SIMD3<Double>, from eye: SIMD3<Double>, yaw: Double, pitch: Double,
                      radiusFactor: Double, seconds: Double) -> [Key] {
        let v0 = eye - anchor
        let r0 = simd_length(v0)
        guard r0 > 1e-6, seconds > 0 else { return [Key(t: 0, eye: eye, ease: true)] }
        let el0 = asin(max(-1, min(1, simd_dot(v0 / r0, up))))
        let limit = (90.0 - 3.0) * .pi / 180
        let dEl = max(-limit, min(limit, el0 + pitch * .pi / 180)) - el0
        let n = max(4, Int((max(abs(yaw), abs(dEl) * 180 / .pi) / 15).rounded(.up)))
        var out: [Key] = []
        for i in 0...n {
            let f = Double(i) / Double(n)
            var v = simd_quatd(angle: yaw * f * .pi / 180, axis: up).act(v0)
            var right = simd_cross(up, v)
            if simd_length(right) > 1e-9 {
                right = simd_normalize(right)
                v = simd_quatd(angle: -dEl * f, axis: right).act(v)
            }
            v = simd_normalize(v) * r0 * (1 + (radiusFactor - 1) * f)
            out.append(Key(t: seconds * inverseSmoothstep(f), eye: anchor + v, ease: i == 0 || i == n))
        }
        return out
    }

    /// s with 3s² - 2s³ = f: where along an S-curve in time a fraction f of the motion is done.
    static func inverseSmoothstep(_ f: Double) -> Double {
        let f = min(max(f, 0), 1)
        return 0.5 - sin(asin(1 - 2 * f) / 3)
    }

    /// Before a generator adds on: the old end becomes a pass-through, so arc-then-arc flows —
    /// unless it ends a hold, which stays a stop.
    private mutating func unEaseTail() {
        let ks = sortedKeys
        guard ks.count >= 2, let i = keys.firstIndex(where: { $0.id == ks[ks.count - 1].id }) else { return }
        let prev = ks[ks.count - 2]
        if simd_length(prev.position - ks[ks.count - 1].position) > 1e-6 { keys[i].ease = false }
    }
}

// MARK: - What a move asks of the capture

/// Per frame: how far the virtual camera strays from the real ones, and how fast it moves.
/// Nothing here blocks a render — it is shown so a person can decide.
public struct MoveAnalysis: Sendable, Equatable {
    /// Degrees, seen from the anchor, between the frame's camera and the nearest real camera.
    public let offAngle: [Double]
    /// Frame camera distance over that capture's distance, as |log2| (0 = same distance, 1 = twice or half).
    public let offRange: [Double]
    public let speed: [Double]              // m/s, per frame (0 at frame 0)
    public let distance: [Double]           // m to the anchor
    public let pathLength: Double           // m

    public static let amberDeg = 3.0
    public static let redDeg = 6.0

    public init(move: KeyedMove, captures: [SIMD3<Double>]) {
        let ps = move.framePositions()
        let a = move.anchorPoint
        let dirs = captures.map { c -> (SIMD3<Double>, Double) in
            let v = c - a
            let r = simd_length(v)
            return (r > 0 ? v / r : .zero, r)
        }
        var ang: [Double] = [], rng: [Double] = [], spd: [Double] = [], dist: [Double] = []
        var length = 0.0
        for (i, p) in ps.enumerated() {
            let v = p - a
            let r = simd_length(v)
            dist.append(r)
            if r > 0, !dirs.isEmpty {
                let u = v / r
                var best = -2.0, bestR = r
                for (d, rc) in dirs {
                    let c = simd_dot(u, d)
                    if c > best { best = c; bestR = rc }
                }
                ang.append(acos(max(-1, min(1, best))) * 180 / .pi)
                rng.append(bestR > 0 ? abs(log2(r / bestR)) : 0)
            } else {
                ang.append(0); rng.append(0)
            }
            if i > 0 {
                let d = simd_length(p - ps[i - 1])
                length += d
                spd.append(d * move.fps)
            } else {
                spd.append(0)
            }
        }
        offAngle = ang; offRange = rng; speed = spd; distance = dist; pathLength = length
    }

    public var worstAngle: Double { offAngle.max() ?? 0 }
    public var peakSpeed: Double { speed.max() ?? 0 }
    public func fraction(over deg: Double) -> Double {
        offAngle.isEmpty ? 0 : Double(offAngle.filter { $0 > deg }.count) / Double(offAngle.count)
    }
    /// A frame whose speed jumps past 3x the move's 90th percentile (and 0.15 m/s): a jerk.
    public var spikeFrame: Int? {
        let s = speed.sorted()
        guard s.count > 4 else { return nil }
        let p90 = s[Int(Double(s.count - 1) * 0.9)]
        guard let i = speed.indices.max(by: { speed[$0] < speed[$1] }) else { return nil }
        return speed[i] > max(3 * p90, 0.15) ? i : nil
    }
}

// MARK: - The render check

/// Before a render: someone has looked at the first, middle and last frame of exactly this
/// baked move. Replaces the aim-check crosshair — in an editor that shows the frame, the frame is the check.
public struct FrameCheck: Equatable, Sendable {
    public enum Which: String, CaseIterable, Identifiable, Sendable {
        case first, middle, last
        public var id: String { rawValue }
        public func frame(of count: Int) -> Int {
            switch self {
            case .first: return 0
            case .middle: return max(0, (count - 1) / 2)
            case .last: return max(0, count - 1)
            }
        }
    }
    /// The baked move's md5 these were looked at on.
    public var bakedMD5: String?
    public var seen: Set<Which> = []

    public init() {}

    public mutating func saw(_ w: Which, md5: String?) {
        if md5 != bakedMD5 { bakedMD5 = md5; seen = [] }
        seen.insert(w)
    }

    public func complete(for md5: String?) -> Bool { md5 != nil && md5 == bakedMD5 && seen.count == 3 }
}
