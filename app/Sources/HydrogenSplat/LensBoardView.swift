import SwiftUI
import AppKit
import HSCore

/// The 7×5 calibration board the Calibrate step shows on screen and saves as PDF. The real
/// ChArUco board (chessboard with an ArUco marker in every white square) comes from the engine,
/// `hs calibrate --board-image OUT.png|OUT.pdf --squares 7x5 --square-mm 35`; until that is
/// there, the app draws the chessboard itself with a placeholder where each marker goes, and
/// says so.
enum LensBoard {
    static let squaresX = 7
    static let squaresY = 5
    static let squareMM = 35.0
    /// ArUco markers sit inside the white squares (the engine's default, 26 mm in a 35 mm square).
    static let markerMM = 26.0
    static let marginMM = 15.0

    static var boardWidthMM: Double { Double(squaresX) * squareMM }
    static var boardHeightMM: Double { Double(squaresY) * squareMM }
    static var pageWidthMM: Double { boardWidthMM + 2 * marginMM }
    static var pageHeightMM: Double { boardHeightMM + 2 * marginMM }

    /// `hs calibrate --board-image OUT.png --squares 7x5 --square-mm 35 --marker-mm 26`: the engine's
    /// board at print size (300 dpi, tagged so Preview prints it at 100 %).
    static func imageArguments(output: String) -> [String] {
        ["calibrate", "--board-image", output] + spec
    }

    /// `hs calibrate --board-pdf OUT.pdf …`: the same board placed at exact size on a Letter and an A4 page.
    static func pdfArguments(output: String) -> [String] {
        ["calibrate", "--board-pdf", output] + spec
    }

    static var spec: [String] {
        ["--squares", "\(squaresX)x\(squaresY)", "--square-mm", String(Int(squareMM)), "--marker-mm", String(Int(markerMM))]
    }

    /// Where the engine's PNG of the board is kept between launches.
    static var enginePNGPath: String {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/Application Support")
        let dir = base.appendingPathComponent("HydrogenSplat", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("board_\(squaresX)x\(squaresY)_\(Int(squareMM))mm.png").path
    }

    /// The board's squares in a rect, the chessboard scaled to fit: black squares, and the
    /// marker placeholder inside every white square. Row 0 is the top.
    static func layout(in rect: CGRect) -> (black: [CGRect], markers: [CGRect], board: CGRect) {
        let s = min(rect.width / CGFloat(squaresX), rect.height / CGFloat(squaresY))
        let w = s * CGFloat(squaresX), h = s * CGFloat(squaresY)
        let origin = CGPoint(x: rect.midX - w / 2, y: rect.midY - h / 2)
        var black: [CGRect] = [], markers: [CGRect] = []
        let m = s * CGFloat(markerMM / squareMM)
        for j in 0..<squaresY {
            for i in 0..<squaresX {
                let r = CGRect(x: origin.x + CGFloat(i) * s, y: origin.y + CGFloat(j) * s, width: s, height: s)
                if (i + j) % 2 == 0 {
                    black.append(r)
                } else {
                    markers.append(CGRect(x: r.midX - m / 2, y: r.midY - m / 2, width: m, height: m))
                }
            }
        }
        return (black, markers, CGRect(origin: origin, size: CGSize(width: w, height: h)))
    }

    /// A PDF of the drawn board at its printed size (35 mm squares at 100 %), with the marker
    /// placeholders. The engine's own PDF, when `hs calibrate --board-image` exists, is the one
    /// to print — this one has no markers to read.
    static func writePDF(to url: URL) throws {
        let pt = 72.0 / 25.4
        var page = CGRect(x: 0, y: 0, width: pageWidthMM * pt, height: pageHeightMM * pt)
        guard let ctx = CGContext(url as CFURL, mediaBox: &page, nil) else {
            throw NSError(domain: "LensBoard", code: 1, userInfo: [NSLocalizedDescriptionKey: "could not start the PDF"])
        }
        ctx.beginPDFPage(nil)
        ctx.setFillColor(CGColor(gray: 1, alpha: 1))
        ctx.fill(page)
        let inner = page.insetBy(dx: marginMM * pt, dy: marginMM * pt)
        let l = layout(in: inner)
        ctx.setFillColor(CGColor(gray: 0, alpha: 1))
        for r in l.black { ctx.fill(r) }
        ctx.setFillColor(CGColor(gray: 0.6, alpha: 1))
        for r in l.markers { ctx.fill(r) }
        ctx.setFillColor(CGColor(gray: 0, alpha: 1))
        let note = "HydrogenSplat lens board · \(squaresX)×\(squaresY) · \(Int(squareMM)) mm squares · print at 100 % · the grey squares stand where the engine's markers go"
            as NSString
        let attrs: [NSAttributedString.Key: Any] = [.font: NSFont.systemFont(ofSize: 7)]
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(cgContext: ctx, flipped: false)
        note.draw(at: NSPoint(x: inner.minX, y: 4 * pt), withAttributes: attrs)
        NSGraphicsContext.restoreGraphicsState()
        ctx.endPDFPage()
        ctx.closePDF()
    }
}

/// The board on screen: the engine's image when there is one, else the drawn chessboard. Shown
/// as a sheet; "Fill the screen" takes the window full screen so the board is as large as the
/// display allows.
struct LensBoardView: View {
    /// The engine's rendering of the board (PNG), when `hs calibrate --board-image` produced one.
    let engineImage: NSImage?
    let dismiss: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Lens board · \(LensBoard.squaresX)×\(LensBoard.squaresY) squares")
                    .font(.headline)
                Spacer()
                Button("Fill the screen") { NSApp.keyWindow?.toggleFullScreen(nil) }
                    .help("The window goes full screen; the board fills it. Esc or the same button brings it back.")
                Button("Done", action: dismiss).keyboardShortcut(.cancelAction)
            }
            .padding(12)
            Divider()
            GeometryReader { geo in
                ZStack {
                    Color.white
                    if let img = engineImage {
                        Image(nsImage: img).resizable().interpolation(.none).aspectRatio(contentMode: .fit).padding(24)
                    } else {
                        Canvas { ctx, size in
                            let l = LensBoard.layout(in: CGRect(origin: .zero, size: size).insetBy(dx: 24, dy: 24))
                            for r in l.black { ctx.fill(Path(r), with: .color(.black)) }
                            for r in l.markers { ctx.fill(Path(r), with: .color(Color(white: 0.6))) }
                        }
                    }
                }
                .frame(width: geo.size.width, height: geo.size.height)
            }
            Divider()
            Text(engineImage != nil
                 ? "Film it with the camera and settings of the shot: 10–20 s, the board filling a third of the frame, tilted through every corner."
                 : "A drawn board: the grey squares stand where the markers go. The engine's own board (with markers the solve can read) comes from hs calibrate --board-image; this one is for framing practice and the chessboard corners only.")
                .font(.callout).foregroundStyle(.secondary).padding(12)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(minWidth: 900, minHeight: 640)
    }
}
