// hs-segment — Apple Vision foreground instance masks, for `hs masks --method vision`.
//
// The engine compiles this file itself on first use (xcrun swiftc, cached by the source's hash
// under ~/Library/Caches/HydrogenSplat), so neither the app nor the CLI needs a separate build.
//
//   hs-segment --out DIR --list FILE      FILE: one image path per line
//
// For line i (0-based) it writes DIR/<i>/<k>.png for every foreground instance k = 1…n Vision
// finds: an 8-bit soft mask at the image's own resolution. It prints one JSON line per image to
// stdout — {"index": i, "instances": n, "width": w, "height": h} or {"index": i, "error": "…"} —
// so the engine can report progress and tell "Vision saw nothing" from "Vision failed".
// Selection (which instance is the subject) is the engine's job, not this tool's.
import CoreGraphics
import CoreVideo
import Foundation
import ImageIO
import UniformTypeIdentifiers
import Vision

func die(_ s: String) -> Never {
    FileHandle.standardError.write((s + "\n").data(using: .utf8)!)
    exit(2)
}

func emit(_ obj: [String: Any]) {
    let d = try! JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys])
    FileHandle.standardOutput.write(d)
    FileHandle.standardOutput.write("\n".data(using: .utf8)!)
}

struct SegError: Error, CustomStringConvertible { let description: String }

func writePNG(_ buf: CVPixelBuffer, to url: URL) throws -> (Int, Int) {
    CVPixelBufferLockBaseAddress(buf, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buf, .readOnly) }
    let w = CVPixelBufferGetWidth(buf), h = CVPixelBufferGetHeight(buf)
    let rowBytes = CVPixelBufferGetBytesPerRow(buf)
    guard let base = CVPixelBufferGetBaseAddress(buf) else { throw SegError(description: "empty mask buffer") }
    let fmt = CVPixelBufferGetPixelFormatType(buf)
    var bytes = [UInt8](repeating: 0, count: w * h)
    for y in 0..<h {
        let row = base.advanced(by: y * rowBytes)
        switch fmt {
        case kCVPixelFormatType_OneComponent32Float:
            let p = row.assumingMemoryBound(to: Float32.self)
            for x in 0..<w { bytes[y * w + x] = UInt8(max(0, min(255, (p[x] * 255).rounded()))) }
        case kCVPixelFormatType_OneComponent8:
            let p = row.assumingMemoryBound(to: UInt8.self)
            for x in 0..<w { bytes[y * w + x] = p[x] }
        default:
            throw SegError(description: "unexpected mask pixel format \(fmt)")
        }
    }
    guard let provider = CGDataProvider(data: Data(bytes) as CFData),
          let img = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: w,
                            space: CGColorSpaceCreateDeviceGray(),
                            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
          let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { throw SegError(description: "could not encode \(url.lastPathComponent)") }
    CGImageDestinationAddImage(dest, img, nil)
    guard CGImageDestinationFinalize(dest) else { throw SegError(description: "could not write \(url.path)") }
    return (w, h)
}

var outDir: String?
var listFile: String?
var args = CommandLine.arguments.dropFirst().makeIterator()
while let a = args.next() {
    switch a {
    case "--out": outDir = args.next()
    case "--list": listFile = args.next()
    case "--version": print("hs-segment 1"); exit(0)
    default: die("hs-segment: unknown argument \(a)")
    }
}
guard let outDir, let listFile else { die("usage: hs-segment --out DIR --list FILE") }
guard let listText = try? String(contentsOfFile: listFile, encoding: .utf8) else { die("hs-segment: cannot read \(listFile)") }
let paths = listText.split(separator: "\n").map(String.init).filter { !$0.isEmpty }
let fm = FileManager.default

for (i, path) in paths.enumerated() {
    autoreleasepool {
        do {
            let dir = URL(fileURLWithPath: outDir).appendingPathComponent(String(i))
            try? fm.removeItem(at: dir)
            try fm.createDirectory(at: dir, withIntermediateDirectories: true)
            let handler = VNImageRequestHandler(url: URL(fileURLWithPath: path), options: [:])
            let req = VNGenerateForegroundInstanceMaskRequest()
            try handler.perform([req])
            guard let obs = req.results?.first, !obs.allInstances.isEmpty else {
                emit(["index": i, "instances": 0])
                return
            }
            var w = 0, h = 0, n = 0
            for k in obs.allInstances {
                let buf = try obs.generateScaledMaskForImage(forInstances: IndexSet(integer: k), from: handler)
                (w, h) = try writePNG(buf, to: dir.appendingPathComponent("\(k).png"))
                n += 1
            }
            emit(["index": i, "instances": n, "width": w, "height": h])
        } catch {
            emit(["index": i, "error": String(describing: error)])
        }
    }
}
