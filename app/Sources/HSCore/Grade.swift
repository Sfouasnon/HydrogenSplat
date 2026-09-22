import Foundation
import CoreGraphics
import Accelerate

/// Lift / gamma / gain, the same curve `hs grade` applies with ffmpeg lutrgb (engine/hs/stages/grade.py):
///     out = clip(gain·x + lift·(1 − x), 0, 1) ^ (1 / gamma),   x = code value / 255
/// per channel, lift = master + channel offset, gamma and gain = master × channel factor.
public struct GradeSettings: Codable, Equatable, Sendable {
    public var lift: Double = 0
    public var gamma: Double = 1
    public var gain: Double = 1
    public var liftRGB: [Double] = [0, 0, 0]
    public var gammaRGB: [Double] = [1, 1, 1]
    public var gainRGB: [Double] = [1, 1, 1]
    public var sharpen: Double = 0.35
    public var aspect: Double = 2.35
    public var headroomMM: Double = 25.4

    enum CodingKeys: String, CodingKey {
        case lift, gamma, gain, sharpen, aspect
        case liftRGB = "lift_rgb", gammaRGB = "gamma_rgb", gainRGB = "gain_rgb", headroomMM = "headroom_mm"
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = GradeSettings()
        lift = try c.decodeIfPresent(Double.self, forKey: .lift) ?? d.lift
        gamma = try c.decodeIfPresent(Double.self, forKey: .gamma) ?? d.gamma
        gain = try c.decodeIfPresent(Double.self, forKey: .gain) ?? d.gain
        liftRGB = try c.decodeIfPresent([Double].self, forKey: .liftRGB) ?? d.liftRGB
        gammaRGB = try c.decodeIfPresent([Double].self, forKey: .gammaRGB) ?? d.gammaRGB
        gainRGB = try c.decodeIfPresent([Double].self, forKey: .gainRGB) ?? d.gainRGB
        sharpen = try c.decodeIfPresent(Double.self, forKey: .sharpen) ?? d.sharpen
        aspect = try c.decodeIfPresent(Double.self, forKey: .aspect) ?? d.aspect
        headroomMM = try c.decodeIfPresent(Double.self, forKey: .headroomMM) ?? d.headroomMM
    }

    /// The one crop list both grade UIs offer (the Move panel's Look and the Grade page). 0 = full frame.
    /// 16:9 is 1920×1076 (mod-4 height) — the value hs grade has always used for that choice.
    public static let aspectOptions: [(label: String, value: Double)] = [
        ("Full frame", 0.0), ("16:9", 1920.0 / 1076.0), ("1.85", 1.85), ("2.00", 2.0), ("2.35", 2.35), ("2.39", 2.39),
    ]

    /// `aspectOptions` plus the current value when a saved file holds one the list does not (an
    /// older build, or a hand-edited json) — a Picker whose selection matches no tag shows blank.
    public static func aspectOptions(including current: Double) -> [(label: String, value: Double)] {
        if aspectOptions.contains(where: { abs($0.value - current) < 1e-6 }) { return aspectOptions }
        return aspectOptions + [(String(format: "%.3f (saved)", current), current)]
    }

    /// grade/<render name>.json — the file `hs grade --move <render name>` reads and writes. The render
    /// name is the move name for the current model and <move>_<model> for an archive (Moves.renderName),
    /// so a look belongs to a render, not to a move.
    public static func lookPath(project: String, renderName: String) -> String {
        ((project as NSString).appendingPathComponent("grade") as NSString).appendingPathComponent("\(renderName).json")
    }

    public static func load(_ path: String) -> GradeSettings? {
        guard let data = FileManager.default.contents(atPath: path) else { return nil }
        return try? JSONDecoder().decode(GradeSettings.self, from: data)
    }

    public func channel(_ c: Int) -> (lift: Double, gamma: Double, gain: Double) {
        (lift + liftRGB[c], gamma * gammaRGB[c], gain * gainRGB[c])
    }

    public static func lut(lift: Double, gamma: Double, gain: Double) -> [UInt8] {
        (0..<256).map { i in
            let x = Double(i) / 255
            let y = pow(min(max(gain * x + lift * (1 - x), 0), 1), 1 / gamma)
            return UInt8(min(max((y * 255 + 0.5).rounded(.down), 0), 255))
        }
    }

    public func lut(_ c: Int) -> [UInt8] {
        let p = channel(c)
        return GradeSettings.lut(lift: p.lift, gamma: p.gamma, gain: p.gain)
    }

    private static func csv(_ v: [Double]) -> String { v.map { String(format: "%.4f", $0) }.joined(separator: ",") }

    /// Arguments for `hs grade` (every value explicit, so the saved file is not needed to reproduce it).
    public func arguments(project: String, move: String) -> [String] {
        ["grade", "-p", project, "--move", move,
         "--lift=\(String(format: "%.4f", lift))", "--gamma=\(String(format: "%.4f", gamma))",
         "--gain=\(String(format: "%.4f", gain))",
         "--lift-rgb=\(GradeSettings.csv(liftRGB))", "--gamma-rgb=\(GradeSettings.csv(gammaRGB))",
         "--gain-rgb=\(GradeSettings.csv(gainRGB))",
         "--sharpen=\(String(format: "%.3f", sharpen))", "--aspect=\(String(format: "%.3f", aspect))",
         "--headroom-mm=\(String(format: "%.2f", headroomMM))"]
    }

    /// The grade on an image's 8-bit RGB code values (no colour management: ffmpeg works on code values too).
    public func apply(to image: CGImage) -> CGImage? {
        let w = image.width, h = image.height
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: image.colorSpace ?? CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Big.rawValue)
        else { return nil }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        guard let data = ctx.data else { return nil }
        var buf = vImage_Buffer(data: data, height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: w * 4)
        let alpha = [UInt8](0...255)
        let r = lut(0), g = lut(1), b = lut(2)
        let err = alpha.withUnsafeBufferPointer { a in
            r.withUnsafeBufferPointer { rp in
                g.withUnsafeBufferPointer { gp in
                    b.withUnsafeBufferPointer { bp in
                        vImageTableLookUp_ARGB8888(&buf, &buf, a.baseAddress, rp.baseAddress, gp.baseAddress, bp.baseAddress,
                                                   vImage_Flags(kvImageNoFlags))
                    }
                }
            }
        }
        return err == kvImageNoError ? ctx.makeImage() : nil
    }
}

/// A rendered move a project can grade: render/<name>_1920.mp4 plus its move json.
public struct RenderedMove: Identifiable, Hashable, Sendable {
    public var id: String { name }
    public let name: String
    public let video: String
    public let graded: String?
    public let frames: Int
    public let followsHead: Bool

    public static func list(project: String) -> [RenderedMove] {
        let fm = FileManager.default
        let rdir = (project as NSString).appendingPathComponent("render")
        let names = (try? fm.contentsOfDirectory(atPath: rdir)) ?? []
        return names.filter { $0.hasSuffix("_1920.mp4") }.sorted().map { f in
            let name = String(f.dropLast("_1920.mp4".count))
            let move = (project as NSString).appendingPathComponent("move/\(name).json")
            var frames = 0
            if let d = fm.contents(atPath: move), let v = JSONValue.parse(d) {
                frames = v["frames"]?.array?.count ?? 0
            }
            let g = (rdir as NSString).appendingPathComponent("\(name)_graded.mp4")
            let track = (project as NSString).appendingPathComponent("move/\(name)_frame.json")
            return RenderedMove(name: name, video: (rdir as NSString).appendingPathComponent(f),
                                graded: fm.fileExists(atPath: g) ? g : nil, frames: frames,
                                followsHead: fm.fileExists(atPath: track))
        }
    }
}
