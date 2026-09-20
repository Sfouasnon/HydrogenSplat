import Foundation
import CryptoKit

// MARK: - .hsmove scripts (move/<name>.hsmove)

/// A camera move written as cues (engine/hs/movescript.py). The script is the source; `hs move
/// --script` compiles it into move/<name>.json, the file `hs render` reads.
public struct MoveScript: Identifiable, Hashable, Sendable {
    public var id: String { name }
    public let project: String
    public let name: String

    public init(project: String, name: String) {
        self.project = project
        self.name = name
    }

    public var moveDir: String { (project as NSString).appendingPathComponent("move") }
    public var scriptPath: String { (moveDir as NSString).appendingPathComponent("\(name).hsmove") }
    /// What `hs move` writes: the path the render uses.
    public var builtPath: String { (moveDir as NSString).appendingPathComponent("\(name).json") }
    public var aimCheckPath: String { (moveDir as NSString).appendingPathComponent("\(name)_aim_check.jpg") }
    /// The human's "aim is on the subject", tied to the md5 of the move it was given for.
    public var aimConfirmationPath: String { (moveDir as NSString).appendingPathComponent("\(name).aim_ok.json") }
    /// What `hs movepreview` writes: the live compile the viewer plays.
    public var previewPath: String {
        ((project as NSString).appendingPathComponent("viewer") as NSString).appendingPathComponent("move_\(name).json")
    }
    public var reportPath: String {
        ((project as NSString).appendingPathComponent("viewer") as NSString).appendingPathComponent("move_\(name).report.json")
    }

    /// move/*.hsmove, newest first.
    public static func list(project: String) -> [MoveScript] {
        let dir = (project as NSString).appendingPathComponent("move")
        let fm = FileManager.default
        return ((try? fm.contentsOfDirectory(atPath: dir)) ?? [])
            .filter { $0.hasSuffix(".hsmove") && !$0.hasPrefix(".") }
            .map { MoveScript(project: project, name: String($0.dropLast(".hsmove".count))) }
            .sorted { mtime($0.scriptPath) > mtime($1.scriptPath) }
    }

    /// A plain file name: letters, digits, - and _.
    public static func validName(_ s: String) -> Bool {
        !s.isEmpty && s.count <= 64 && !s.hasPrefix("-")
            && s.unicodeScalars.allSatisfy { CharacterSet.alphanumerics.contains($0) || $0 == "-" || $0 == "_" }
    }

    public func read() -> String {
        (try? String(contentsOfFile: scriptPath, encoding: .utf8)) ?? ""
    }

    public func write(_ text: String) throws {
        try FileManager.default.createDirectory(atPath: moveDir, withIntermediateDirectories: true)
        try text.write(toFile: scriptPath, atomically: true, encoding: .utf8)
    }

    /// A move that has been built and not edited since.
    public var buildState: BuildState {
        let fm = FileManager.default
        guard fm.fileExists(atPath: builtPath) else { return .notBuilt }
        return MoveScript.mtime(scriptPath) > MoveScript.mtime(builtPath) ? .edited : .built
    }

    public enum BuildState: Equatable, Sendable { case notBuilt, edited, built }

    // MARK: aim confirmation

    public var aimConfirmed: Bool {
        guard let d = FileManager.default.contents(atPath: aimConfirmationPath),
              let v = JSONValue.parse(d), let want = v["move_md5"]?.string,
              let have = MoveScript.md5(builtPath) else { return false }
        return want == have
    }

    public func confirmAim() throws {
        guard let md5 = MoveScript.md5(builtPath) else { throw CocoaError(.fileNoSuchFile) }
        let body: [String: String] = ["move_md5": md5, "confirmed": ISO8601DateFormatter().string(from: Date()),
                                      "note": "a person looked at \(name)_aim_check.jpg and said the aim is on the subject"]
        let data = try JSONSerialization.data(withJSONObject: body, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: aimConfirmationPath), options: .atomic)
    }

    public func revokeAim() {
        try? FileManager.default.removeItem(atPath: aimConfirmationPath)
    }

    // MARK: arguments

    public var previewArguments: [String] { ["movepreview", "-p", project, "--script", scriptPath, "--name", name] }
    public var buildArguments: [String] { ["move", "-p", project, "--script", scriptPath, "--name", name] }

    /// `hs render` of this move with one model. The output is named after the move, plus the
    /// model when it is not the live export, so renders of two models never overwrite each other.
    public func renderArguments(model: ViewerModelFile, width: Int, keepFrames: Bool, crop: Bool) -> [String] {
        var a = ["render", "-p", project, "--move", name, "--ply", model.ply, "--name", renderName(model: model),
                 "--width", String(width)]
        if keepFrames { a.append("--keep-frames") }
        if !crop { a.append("--no-crop") }
        return a
    }

    public func renderName(model: ViewerModelFile) -> String {
        guard model.archive != nil || model.name != "current" else { return name }
        let safe = model.name.map { ($0.isLetter || $0.isNumber || $0 == "-" || $0 == "_") ? $0 : "-" }
        return name + "_" + String(safe)
    }

    /// render/<renderName>_1920.mp4 and _1080x1350.mp4 that exist.
    public func renders(model: ViewerModelFile) -> [String] {
        let dir = (project as NSString).appendingPathComponent("render")
        let base = renderName(model: model)
        return ["\(base)_1920.mp4", "\(base)_1080x1350.mp4"]
            .map { (dir as NSString).appendingPathComponent($0) }
            .filter { FileManager.default.fileExists(atPath: $0) }
    }

    /// The checks `hs move` recorded for this move (stages.move.runs.<name>).
    public func builtChecks(manifest: Manifest?) -> [CheckResult] {
        manifest?.raw["stages"]?["move"]?["runs"]?[name]?["checks"]?.array?.compactMap(CheckResult.init) ?? []
    }

    public static let template = """
    # hsmove 1
    # speed is tenths: 10 = 30 deg/s of orbit, 150 mm/s of dolly; 4>1 ramps across the cue
    start  az 0  el +6  dolly +0
    hold   0.5s
    arc    left 20        speed 2
    hold   0.5s

    """

    static func mtime(_ p: String) -> Date {
        ((try? FileManager.default.attributesOfItem(atPath: p))?[.modificationDate] as? Date) ?? .distantPast
    }

    public static func md5(_ path: String) -> String? {
        guard let d = FileManager.default.contents(atPath: path) else { return nil }
        return Insecure.MD5.hash(data: d).map { String(format: "%02x", $0) }.joined()
    }
}

// MARK: - The live compile (viewer/move_<name>.report.json)

public struct MoveReport: Decodable, Sendable, Equatable {
    public struct Cue: Decodable, Sendable, Equatable, Identifiable {
        public var id: Int { seg }
        public let seg: Int
        public let cue: String
        public let want: Double?
        public let got: Double?
        public let secs: Double?
        public let clamped: Bool
        public let unit: String?
        public let at: Bearing?
    }
    public struct Bearing: Decodable, Sendable, Equatable {
        public let az: Double
        public let el: Double
    }
    public struct Spike: Decodable, Sendable, Equatable {
        public let frame: Int
        public let t: Double
        public let mmS: Double
        enum CodingKeys: String, CodingKey { case frame, t, mmS = "mm_s" }
    }
    public struct Aim: Decodable, Sendable, Equatable {
        public let view: String
        public let depthMm: Double
        public let u: Double?
        public let v: Double?
        public let inside: Bool?
        public let behind: Bool?
        enum CodingKeys: String, CodingKey { case view, u, v, inside, behind, depthMm = "depth_mm" }
    }

    public let frames: Int
    public let fps: Double
    public let pathLengthMm: Double
    public let peakSpeedMmS: Double
    public let p90SpeedMmS: Double
    public let spike: Spike?
    public let distanceMm: [Double]
    public let hullMm: [Double]
    public let hullLimitMm: Double
    public let aim: Aim
    public let cues: [Cue]
    /// Per frame: azimuth, elevation (deg), radius (mm), cue index (−1 = the start mark).
    public let track: [[Double?]]
    /// Per capture (left eye): azimuth, elevation, radius.
    public let captures: [[Double?]]
    public let captureNames: [String]
    public let rigNpzMd5: String
    public let moveJson: String

    enum CodingKeys: String, CodingKey {
        case frames, fps, spike, aim, cues, track, captures
        case pathLengthMm = "path_length_mm", peakSpeedMmS = "peak_speed_mm_s", p90SpeedMmS = "p90_speed_mm_s"
        case distanceMm = "distance_mm", hullMm = "hull_mm", hullLimitMm = "hull_limit_mm"
        case captureNames = "capture_names", rigNpzMd5 = "rig_npz_md5", moveJson = "move_json"
    }

    public var duration: Double { Double(frames) / max(fps, 1) }
    public var hullMax: Double { hullMm.count > 1 ? hullMm[1] : .nan }
    public var hullOK: Bool { hullMax <= hullLimitMm }
    public var clamped: [Cue] { cues.filter(\.clamped) }

    public static func load(path: String) throws -> MoveReport {
        try JSONDecoder().decode(MoveReport.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
    }
}

public enum MovePreview {
    public struct Failure: LocalizedError, Equatable {
        public let message: String
        /// The script line the engine named ("line 3: …"), 1-based.
        public let line: Int?
        public var errorDescription: String? { message }
        public init(message: String, line: Int?) {
            self.message = message
            self.line = line
        }
    }

    /// `hs movepreview`: compiles the saved script, returns its report. Takes no lock.
    public static func run(config: EngineConfig, script: MoveScript) async throws -> MoveReport {
        _ = try await engine(config: config, arguments: script.previewArguments)
        return try MoveReport.load(path: script.reportPath)
    }

    /// `hs movepreview --frame`: the captures in script coordinates, before any script compiles.
    public static func frame(config: EngineConfig, project: String) async throws -> MoveFrame {
        _ = try await engine(config: config, arguments: ["movepreview", "-p", project, "--frame"])
        return try MoveFrame.load(project: project)
    }

    /// `hs movepreview --locate`: where a camera centre (solve coordinates, mm) sits as a start line.
    public static func locate(config: EngineConfig, project: String, pointMM p: SIMD3<Double>,
                              hull: Double) async throws -> MoveLocation {
        let arg = String(format: "--locate=%.3f,%.3f,%.3f", p.x, p.y, p.z)
        let events = try await engine(config: config, arguments: ["movepreview", "-p", project, arg,
                                                                  "--hull", String(format: "%g", hull)])
        guard let v = events.first(where: { $0["name"]?.string == "locate" })?["value"],
              let loc = MoveLocation(v) else {
            throw Failure(message: "hs movepreview --locate printed no location", line: nil)
        }
        return loc
    }

    /// Runs hs, returns its JSON events; a non-zero exit throws the engine's own error message.
    static func engine(config: EngineConfig, arguments: [String]) async throws -> [JSONValue] {
        try await Task.detached(priority: .userInitiated) { () throws -> [JSONValue] in
            let p = Process()
            p.executableURL = URL(fileURLWithPath: config.hsPath)
            p.arguments = arguments
            p.environment = config.environment()
            p.currentDirectoryURL = URL(fileURLWithPath: config.repoRoot)
            let out = Pipe()
            p.standardOutput = out
            p.standardError = out
            p.standardInput = FileHandle.nullDevice
            do { try p.run() } catch {
                throw Failure(message: "could not start \(config.hsPath): \(error.localizedDescription)", line: nil)
            }
            let data = out.fileHandleForReading.readDataToEndOfFile()
            p.waitUntilExit()
            let lines = String(decoding: data, as: UTF8.self).split(separator: "\n").map(String.init)
            let events = lines.compactMap { JSONValue.parse($0) }
            guard p.terminationStatus == 0 else {
                let why = events.first { $0["ev"]?.string == "error" }?["message"]?.string
                    ?? lines.last ?? "exit \(p.terminationStatus)"
                throw Failure(message: why, line: lineNumber(why))
            }
            return events
        }.value
    }

    /// "line 3: unknown cue 'foo' — foo 3" -> 3
    public static func lineNumber(_ message: String) -> Int? {
        guard message.hasPrefix("line ") else { return nil }
        return Int(message.dropFirst(5).prefix { $0.isNumber })
    }
}

// MARK: - Script coordinates

/// viewer/move_frame.json: every capture as az / el / radius in the frame the script is written in.
public struct MoveFrame: Decodable, Sendable, Equatable {
    public let captures: [[Double?]]
    public let captureNames: [String]
    public let rigNpzMd5: String
    public let hullDefaultMm: Double

    enum CodingKeys: String, CodingKey {
        case captures, captureNames = "capture_names", rigNpzMd5 = "rig_npz_md5", hullDefaultMm = "hull_default_mm"
    }

    public init(captures: [[Double?]], captureNames: [String], rigNpzMd5: String, hullDefaultMm: Double) {
        self.captures = captures
        self.captureNames = captureNames
        self.rigNpzMd5 = rigNpzMd5
        self.hullDefaultMm = hullDefaultMm
    }

    public static func path(project: String) -> String {
        ((project as NSString).appendingPathComponent("viewer") as NSString).appendingPathComponent("move_frame.json")
    }

    public static func load(project: String) throws -> MoveFrame {
        try JSONDecoder().decode(MoveFrame.self, from: Data(contentsOf: URL(fileURLWithPath: path(project: project))))
    }

    /// The capture nearest a point on the az / el map: a start there always compiles.
    public func nearestCapture(az: Double, el: Double) -> (name: String, az: Double, el: Double)? {
        var best: (String, Double, Double, Double)?
        for (i, c) in captures.enumerated() {
            guard c.count >= 2, let a = c[0], let e = c[1] else { continue }
            var da = abs(a - az).truncatingRemainder(dividingBy: 360)
            if da > 180 { da = 360 - da }
            let d = da * da + (e - el) * (e - el)
            if best == nil || d < best!.3 {
                best = (i < captureNames.count ? captureNames[i] : "capture \(i)", a, e, d)
            }
        }
        return best.map { ($0.0, $0.1, $0.2) }
    }
}

/// hs movepreview --locate: a camera centre as a start line.
public struct MoveLocation: Sendable, Equatable {
    public let az: Double
    public let el: Double
    public let rMm: Double
    public let dolly: Double?
    public let reachable: Bool
    public let why: String?
    public let nearestCapture: String?
    public let nearestMm: Double?

    init?(_ v: JSONValue) {
        guard let r = v["reachable"]?.bool else { return nil }
        reachable = r
        why = v["why"]?.string
        guard let a = v["az"]?.double, let e = v["el"]?.double, let rr = v["r_mm"]?.double else {
            if r { return nil }
            az = .nan; el = .nan; rMm = .nan; dolly = nil; nearestCapture = nil; nearestMm = nil
            return
        }
        az = a; el = e; rMm = rr
        dolly = v["dolly"]?.double
        nearestCapture = v["nearest_capture"]?.string
        nearestMm = v["nearest_mm"]?.double
    }
}

/// Editing the script text the way a person would.
public enum MoveScriptText {
    /// A start line in the file's column layout.
    public static func startLine(az: Double, el: Double, dolly: Double) -> String {
        String(format: "start  az %+.1f  el %+.1f  dolly %+.0f", az, el, dolly)
    }

    /// Replace the `start` line, or put one after the leading comments when there is none.
    public static func settingStart(_ text: String, to line: String) -> String {
        var lines = text.components(separatedBy: "\n")
        if let i = lines.firstIndex(where: { firstWord($0) == "start" }) {
            // keep a trailing comment the person wrote on that line
            let old = lines[i]
            if let hash = old.firstIndex(of: "#") {
                lines[i] = line + "   " + old[hash...]
            } else {
                lines[i] = line
            }
        } else {
            let at = lines.firstIndex(where: { !$0.trimmingCharacters(in: .whitespaces).isEmpty
                && !$0.trimmingCharacters(in: .whitespaces).hasPrefix("#") }) ?? lines.count
            lines.insert(line, at: at)
        }
        return lines.joined(separator: "\n")
    }

    /// `hull 50` in the script, else nil (the engine's 25 mm).
    public static func hull(_ text: String) -> Double? {
        for l in text.components(separatedBy: "\n") where firstWord(l) == "hull" {
            let w = l.split(separator: "#", maxSplits: 1).first.map(String.init) ?? l
            let parts = w.split(whereSeparator: { $0 == " " || $0 == "\t" })
            if parts.count >= 2, let v = Double(parts[1].lowercased().replacingOccurrences(of: "mm", with: "")) { return v }
        }
        return nil
    }

    static func firstWord(_ line: String) -> String {
        let code = line.split(separator: "#", maxSplits: 1, omittingEmptySubsequences: false).first ?? ""
        return code.split(whereSeparator: { $0 == " " || $0 == "\t" }).first.map { $0.lowercased() } ?? ""
    }
}

// MARK: - Previews on disk

extension MovePath {
    /// viewer/move_*.json — the live compiles, newest first.
    public static func previews(project: String) -> [String] {
        let dir = (project as NSString).appendingPathComponent("viewer")
        let fm = FileManager.default
        return ((try? fm.contentsOfDirectory(atPath: dir)) ?? [])
            .filter { $0.hasPrefix("move_") && $0.hasSuffix(".json") && !$0.hasSuffix(".report.json") }
            .map { (dir as NSString).appendingPathComponent($0) }
            .sorted { MoveScript.mtime($0) > MoveScript.mtime($1) }
    }

    /// "shot02" for move/shot02.json, "shot02 · preview" for viewer/move_shot02.json.
    public static func label(_ path: String) -> String {
        let file = ((path as NSString).lastPathComponent as NSString).deletingPathExtension
        if (path as NSString).deletingLastPathComponent.hasSuffix("/viewer"), file.hasPrefix("move_") {
            return String(file.dropFirst(5)) + " · preview"
        }
        return file
    }
}
