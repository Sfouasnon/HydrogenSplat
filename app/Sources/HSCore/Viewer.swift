import Foundation
import simd

// The splat viewer's non-Metal half: what can be looked at, where the solved cameras are, and
// the matrices that put the renderer's camera exactly where a capture's was. Kept in HSCore so
// `swift test` covers it without a GPU.
//
// One convention throughout, the one `move/*.json` and `viewer/cameras_*.json` are written in
// (engine/hs/cameras.py): c2w is an OpenCV camera — x right, y down, z forward — in metres,
// the unit of the .ply. The renderer (MetalSplatter) wants a right-handed view matrix looking
// down −z with y up, so every pose goes through `ViewerMath.viewMatrix`, once.

// MARK: - What can be viewed

/// A .ply the viewer can open, and which cameras belong to it.
public struct ViewerModelFile: Identifiable, Hashable, Codable, Sendable {
    public var id: String { ply }
    public let project: String
    /// "current" for train/exports, the archive's folder name, or "prune/<file>".
    public let name: String
    public let ply: String
    /// The archive whose rig.npz this model was trained against; nil = train/dataset/rig.npz.
    public let archive: String?
    public let bytes: Int64

    public init(project: String, name: String, ply: String, archive: String?, bytes: Int64) {
        self.project = project
        self.name = name
        self.ply = ply
        self.archive = archive
        self.bytes = bytes
    }

    /// viewer/cameras_<archive|current>.json — what `hs cameras` writes for this model.
    public var camerasPath: String {
        (project as NSString).appendingPathComponent("viewer/cameras_\(archive ?? "current").json")
    }

    /// `hs cameras` arguments that (re)write `camerasPath`.
    public var camerasArguments: [String] {
        ["cameras", "-p", project] + (archive.map { ["--archive", $0] } ?? [])
    }

    public var sizeLabel: String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }

    /// Every archive's model, then the live train export, then any prune outputs.
    public static func list(project: String) -> [ViewerModelFile] {
        let fm = FileManager.default
        var out: [ViewerModelFile] = []
        func size(_ p: String) -> Int64 {
            ((try? fm.attributesOfItem(atPath: p))?[.size] as? NSNumber)?.int64Value ?? 0
        }
        func plys(in dir: String) -> [String] {
            ((try? fm.contentsOfDirectory(atPath: dir)) ?? [])
                .filter { $0.hasSuffix(".ply") && !$0.hasPrefix(".") }
                .map { (dir as NSString).appendingPathComponent($0) }
        }
        let archiveDir = (project as NSString).appendingPathComponent("archive")
        for n in ((try? fm.contentsOfDirectory(atPath: archiveDir)) ?? []).sorted() where !n.hasPrefix(".") {
            let dir = (archiveDir as NSString).appendingPathComponent(n)
            // an archive holds one model; if someone dropped a second .ply in, take the largest
            guard let ply = plys(in: dir).max(by: { size($0) < size($1) }),
                  fm.fileExists(atPath: (dir as NSString).appendingPathComponent("rig.npz")) else { continue }
            out.append(ViewerModelFile(project: project, name: n, ply: ply, archive: n, bytes: size(ply)))
        }
        let exports = plys(in: (project as NSString).appendingPathComponent("train/exports"))
        if let latest = exports.max(by: { exportIteration($0) < exportIteration($1) }) {
            out.append(ViewerModelFile(project: project, name: "current", ply: latest, archive: nil, bytes: size(latest)))
        }
        for p in plys(in: (project as NSString).appendingPathComponent("prune")).sorted() {
            out.append(ViewerModelFile(project: project, name: "prune/" + (p as NSString).lastPathComponent,
                                       ply: p, archive: nil, bytes: size(p)))
        }
        return out
    }

    /// export_40000.ply -> 40000. Brush zero-pads to the digit count of the run, so compare
    /// numbers, not strings.
    static func exportIteration(_ path: String) -> Int {
        let stem = ((path as NSString).lastPathComponent as NSString).deletingPathExtension
        return Int(stem.split(separator: "_").last ?? "") ?? -1
    }
}

// MARK: - Solved cameras (viewer/cameras_*.json)

public struct ViewerCamera: Decodable, Identifiable, Hashable, Sendable {
    public var id: String { name }
    public let name: String
    public let capture: String
    public let eye: String
    public let w: Int
    public let h: Int
    public let fx: Double
    public let fy: Double
    public let cx: Double
    public let cy: Double
    public let c2w: [[Double]]
    public let azimuthDeg: Double?
    public let elevationDeg: Double?
    public let distanceMM: Double?

    enum CodingKeys: String, CodingKey {
        case name, capture, eye, w, h, fx, fy, cx, cy, c2w
        case azimuthDeg = "azimuth_deg", elevationDeg = "elevation_deg", distanceMM = "distance_mm"
    }

    public var pose: ViewerPose? {
        guard let m = ViewerMath.matrix(c2w) else { return nil }
        return ViewerPose(c2w: m, w: Float(w), h: Float(h), fx: Float(fx), fy: Float(fy), cx: Float(cx), cy: Float(cy))
    }

    /// "cap065 L · az +22 el +6 · 612 mm"
    public var label: String {
        var s = "\(capture) \(eye)"
        if let a = azimuthDeg, let e = elevationDeg {
            s += String(format: " · az %+.0f el %+.0f", a, e)
        }
        if let d = distanceMM { s += String(format: " · %.0f mm", d) }
        return s
    }
}

public struct CameraSet: Decodable, Sendable {
    public let schema: Int
    public let rigNpzMD5: String
    public let stereo: Bool
    public let subjectM: [Double]
    public let upWorld: [Double]
    public let views: [ViewerCamera]

    enum CodingKeys: String, CodingKey {
        case schema, stereo, views
        case rigNpzMD5 = "rig_npz_md5", subjectM = "subject_m", upWorld = "up_world"
    }

    public static func load(path: String) throws -> CameraSet {
        try JSONDecoder().decode(CameraSet.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
    }

    public var subject: SIMD3<Float> { ViewerMath.vector(subjectM) ?? .zero }
    public var up: SIMD3<Float> { ViewerMath.vector(upWorld).map { simd_normalize($0) } ?? SIMD3(0, -1, 0) }

    /// Capture names in rig order, once each.
    public var captures: [String] {
        var seen = Set<String>()
        return views.compactMap { seen.insert($0.capture).inserted ? $0.capture : nil }
    }

    public func view(capture: String, eye: String) -> ViewerCamera? {
        views.first { $0.capture == capture && $0.eye == eye } ?? views.first { $0.capture == capture }
    }
}

/// Runs `hs cameras` and reads what it wrote. The command takes no project lock and leaves the
/// manifest alone (engine/hs/stages/cameras.py), so this is safe while a train run is going.
public enum CameraExport {
    public struct Failure: LocalizedError {
        public let message: String
        public var errorDescription: String? { message }
    }

    public static func run(config: EngineConfig, model: ViewerModelFile) async throws -> CameraSet {
        try await Task.detached(priority: .userInitiated) { () throws -> CameraSet in
            let p = Process()
            p.executableURL = URL(fileURLWithPath: config.hsPath)
            p.arguments = model.camerasArguments
            p.environment = config.environment()
            p.currentDirectoryURL = URL(fileURLWithPath: config.repoRoot)
            let out = Pipe()
            p.standardOutput = out
            p.standardError = out
            p.standardInput = FileHandle.nullDevice
            do { try p.run() } catch {
                throw Failure(message: "could not start \(config.hsPath): \(error.localizedDescription)")
            }
            let data = out.fileHandleForReading.readDataToEndOfFile()
            p.waitUntilExit()
            guard p.terminationStatus == 0 else {
                // the engine says why in an {"ev":"error"} line; show that rather than the raw stream
                let lines = String(decoding: data, as: UTF8.self).split(separator: "\n").map(String.init)
                let why = lines.compactMap { JSONValue.parse($0) }
                    .first { $0["ev"]?.string == "error" }?["message"]?.string
                throw Failure(message: "hs cameras failed: " + (why ?? lines.last ?? "exit \(p.terminationStatus)"))
            }
            return try CameraSet.load(path: model.camerasPath)
        }.value
    }

    /// For the live export only: the rig.npz md5 `hs train` fingerprinted, to compare with the
    /// cameras' own. They differ when solve re-ran after train — the model then sits in a frame
    /// these cameras do not describe. (An archive carries its own rig.npz, so it cannot drift.)
    public static func trainedRigMD5(project: String) -> String? {
        let path = (project as NSString).appendingPathComponent("manifest.json")
        guard let d = FileManager.default.contents(atPath: path), let v = JSONValue.parse(d) else { return nil }
        return v["stages"]?["train"]?["metrics"]?["dataset_fingerprint"]?["rig_npz_md5"]?.string
    }
}

// MARK: - Camera moves (move/*.json)

public struct MovePath: Decodable, Sendable {
    public struct Frame: Decodable, Sendable { public let c2w: [[Double]] }
    public let width: Int
    public let height: Int
    public let K: [[Double]]
    public let fps: Double
    public let frames: [Frame]

    public static func load(path: String) throws -> MovePath {
        try JSONDecoder().decode(MovePath.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
    }

    public func pose(at index: Int) -> ViewerPose? {
        guard frames.indices.contains(index), K.count == 3, K.allSatisfy({ $0.count == 3 }),
              let m = ViewerMath.matrix(frames[index].c2w) else { return nil }
        return ViewerPose(c2w: m, w: Float(width), h: Float(height),
                          fx: Float(K[0][0]), fy: Float(K[1][1]), cx: Float(K[0][2]), cy: Float(K[1][2]))
    }

    /// move/*.json in the project, newest first.
    public static func list(project: String) -> [String] {
        let dir = (project as NSString).appendingPathComponent("move")
        let fm = FileManager.default
        func mtime(_ p: String) -> Date { ((try? fm.attributesOfItem(atPath: p))?[.modificationDate] as? Date) ?? .distantPast }
        return ((try? fm.contentsOfDirectory(atPath: dir)) ?? [])
            .filter { $0.hasSuffix(".json") && !$0.hasPrefix(".") }
            .map { (dir as NSString).appendingPathComponent($0) }
            .sorted { mtime($0) > mtime($1) }
    }
}

// MARK: - Poses and matrices

/// A pinhole camera at a pose: enough to reproduce a capture's or a move frame's exact view.
public struct ViewerPose: Equatable, Sendable {
    public var c2w: simd_float4x4      // OpenCV convention, metres
    public var w: Float
    public var h: Float
    public var fx: Float
    public var fy: Float
    public var cx: Float
    public var cy: Float

    public init(c2w: simd_float4x4, w: Float, h: Float, fx: Float, fy: Float, cx: Float, cy: Float) {
        self.c2w = c2w; self.w = w; self.h = h; self.fx = fx; self.fy = fy; self.cx = cx; self.cy = cy
    }

    public var position: SIMD3<Float> { SIMD3(c2w.columns.3.x, c2w.columns.3.y, c2w.columns.3.z) }
    public var forward: SIMD3<Float> { SIMD3(c2w.columns.2.x, c2w.columns.2.y, c2w.columns.2.z) }
    public var down: SIMD3<Float> { SIMD3(c2w.columns.1.x, c2w.columns.1.y, c2w.columns.1.z) }
    public var verticalFOV: Float { 2 * atan(h / (2 * fy)) }
}

public enum ViewerMath {
    public static let near: Float = 0.02      // 20 mm: the coins clip sat 216 mm from the lens
    public static let far: Float = 200

    /// Row-major nested arrays (JSON) -> simd (column-major).
    public static func matrix(_ rows: [[Double]]) -> simd_float4x4? {
        guard rows.count == 4, rows.allSatisfy({ $0.count == 4 }) else { return nil }
        func col(_ c: Int) -> SIMD4<Float> {
            SIMD4(Float(rows[0][c]), Float(rows[1][c]), Float(rows[2][c]), Float(rows[3][c]))
        }
        return simd_float4x4(columns: (col(0), col(1), col(2), col(3)))
    }

    public static func vector(_ v: [Double]) -> SIMD3<Float>? {
        v.count == 3 ? SIMD3(Float(v[0]), Float(v[1]), Float(v[2])) : nil
    }

    /// OpenCV camera-to-world -> the view matrix of a camera looking down −z with y up.
    /// F = diag(1, −1, −1, 1) turns OpenCV camera axes into those; view = F · c2w⁻¹.
    public static func viewMatrix(c2w: simd_float4x4) -> simd_float4x4 {
        let flip = simd_float4x4(diagonal: SIMD4<Float>(1, -1, -1, 1))
        return flip * c2w.inverse
    }

    /// The pinhole's projection, fitted inside a drawable of another shape: the capture's
    /// image is scaled by s = min(W/w, H/h) and centred, and the scene simply continues past
    /// its edges. Depth maps to [0, 1] (Metal). With a centred principal point this is exactly
    /// the usual right-handed perspective matrix.
    ///
    ///   u' = s·u + (W − s·w)/2        x_ndc = 2u'/W − 1
    ///   v' = s·v + (H − s·h)/2        y_ndc = 1 − 2v'/H        (pixel v runs down, NDC y up)
    public static func projection(pose p: ViewerPose, drawableWidth W: Float, drawableHeight H: Float,
                                  near n: Float = near, far f: Float = far) -> simd_float4x4 {
        guard W > 0, H > 0, p.w > 0, p.h > 0 else { return matrix_identity_float4x4 }
        let s = min(W / p.w, H / p.h)
        let fx = p.fx * s, fy = p.fy * s
        let cx = p.cx * s + (W - p.w * s) / 2
        let cy = p.cy * s + (H - p.h * s) / 2
        let zs = f / (n - f)
        return simd_float4x4(columns: (SIMD4(2 * fx / W, 0, 0, 0),
                                       SIMD4(0, 2 * fy / H, 0, 0),
                                       SIMD4(1 - 2 * cx / W, 2 * cy / H - 1, zs, -1),
                                       SIMD4(0, 0, zs * n, 0)))
    }

    public static func perspective(fovy: Float, aspect: Float, near n: Float = near, far f: Float = far) -> simd_float4x4 {
        let ys = 1 / tan(fovy * 0.5)
        let xs = ys / max(aspect, 1e-6)
        let zs = f / (n - f)
        return simd_float4x4(columns: (SIMD4(xs, 0, 0, 0), SIMD4(0, ys, 0, 0),
                                       SIMD4(0, 0, zs, -1), SIMD4(0, 0, zs * n, 0)))
    }

    public static func lookAt(eye: SIMD3<Float>, target: SIMD3<Float>, up: SIMD3<Float>) -> simd_float4x4 {
        let fwd = simd_normalize(target - eye)
        var right = simd_cross(fwd, up)
        if simd_length(right) < 1e-6 { right = simd_cross(fwd, SIMD3(1, 0, 0)) }   // looking straight along up
        right = simd_normalize(right)
        let u = simd_cross(right, fwd)
        return simd_float4x4(columns: (SIMD4(right.x, u.x, -fwd.x, 0),
                                       SIMD4(right.y, u.y, -fwd.y, 0),
                                       SIMD4(right.z, u.z, -fwd.z, 0),
                                       SIMD4(-simd_dot(right, eye), -simd_dot(u, eye), simd_dot(fwd, eye), 1)))
    }
}

/// A free camera that turns about a point. `up` is the coverage table's up (the mean camera
/// −y), so "level" means what it means in `hs views` and `hs move`.
public struct OrbitCamera: Equatable, Sendable {
    public var eye: SIMD3<Float>
    public var target: SIMD3<Float>
    public var up: SIMD3<Float>
    public var fovy: Float

    public init(eye: SIMD3<Float>, target: SIMD3<Float>, up: SIMD3<Float>, fovy: Float) {
        self.eye = eye; self.target = target; self.up = simd_normalize(up); self.fovy = fovy
    }

    /// Leave a capture's pose without a jump: same position, turning about the point on its
    /// optical axis nearest the subject. (Roll is dropped — the orbit is level by definition.)
    public init(leaving pose: ViewerPose, subject: SIMD3<Float>, up: SIMD3<Float>) {
        let along = max(simd_dot(subject - pose.position, pose.forward), 0.05)
        self.init(eye: pose.position, target: pose.position + pose.forward * along, up: up, fovy: pose.verticalFOV)
    }

    public var distance: Float { simd_length(eye - target) }
    public var viewMatrix: simd_float4x4 { ViewerMath.lookAt(eye: eye, target: target, up: up) }

    /// Drag: yaw about `up`, pitch about the camera's right axis; radians. Pitch stops 2° short
    /// of the poles so the view never flips.
    public mutating func rotate(yaw: Float, pitch: Float) {
        var v = eye - target
        v = simd_act(simd_quatf(angle: yaw, axis: up), v)
        let right = simd_normalize(simd_cross(simd_normalize(-v), up))
        let elevation = asin(max(-1, min(1, simd_dot(simd_normalize(v), up))))
        let limit = Float.pi / 2 - 2 * .pi / 180
        let clamped = max(-limit, min(limit, elevation + pitch)) - elevation
        // a positive turn about `right` tips the eye downward, so raise by turning the other way
        v = simd_act(simd_quatf(angle: -clamped, axis: right), v)
        eye = target + v
    }

    /// Scroll / pinch: factor < 1 moves in. Never closer than the near plane allows.
    public mutating func dolly(factor: Float) {
        let v = eye - target
        let d = max(simd_length(v) * factor, ViewerMath.near * 2)
        eye = target + simd_normalize(v) * d
    }

    /// Shift-drag: slide eye and target together; dx, dy as fractions of the view height.
    public mutating func pan(dx: Float, dy: Float) {
        let fwd = simd_normalize(target - eye)
        let right = simd_normalize(simd_cross(fwd, up))
        let u = simd_cross(right, fwd)
        let scale = 2 * distance * tan(fovy / 2)
        let shift = (-right * dx + u * dy) * scale
        eye += shift
        target += shift
    }
}
