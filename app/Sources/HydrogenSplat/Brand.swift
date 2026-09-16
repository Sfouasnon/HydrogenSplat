import SwiftUI
import AppKit

/// Brand tokens (app/Branding/tokens.json). Red is the record tally: one red thing per screen.
enum Brand {
    static let tally = Color(light: 0xC8102E, dark: 0xE0162B)
    static let views: [Color] = [
        Color(light: 0x0E9AA6, dark: 0x5CE1E6), Color(light: 0x3F78D6, dark: 0x86B9F4),
        Color(light: 0x7359F0, dark: 0xA48CFF), Color(light: 0xA043DE, dark: 0xC88BFF),
    ]

    static var appIcon: NSImage? {
        guard let url = Bundle.module.url(forResource: "AppIcon", withExtension: "png", subdirectory: "Resources")
        else { return nil }
        return NSImage(contentsOf: url)
    }
}

extension Color {
    /// A colour that follows the system appearance.
    init(light: UInt32, dark: UInt32) {
        self.init(nsColor: NSColor(name: nil) { appearance in
            let hex = appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? dark : light
            return NSColor(srgbRed: CGFloat((hex >> 16) & 0xFF) / 255, green: CGFloat((hex >> 8) & 0xFF) / 255,
                           blue: CGFloat(hex & 0xFF) / 255, alpha: 1)
        })
    }
}

/// Sidebar header: the icon and the wordmark (HYDROGEN in text colour, SPLAT in tally red).
struct BrandHeader: View {
    var body: some View {
        HStack(spacing: 10) {
            if let icon = Brand.appIcon {
                Image(nsImage: icon).resizable().interpolation(.high).frame(width: 30, height: 30)
            }
            VStack(alignment: .leading, spacing: 1) {
                (Text("HYDROGEN ").foregroundStyle(.primary) + Text("SPLAT").foregroundStyle(Brand.tally))
                    .font(.system(size: 12, weight: .semibold, design: .default).width(.expanded))
                    .tracking(1.4)
                HStack(spacing: 3) {
                    ForEach(0..<4, id: \.self) { i in
                        Capsule().fill(Brand.views[i]).frame(width: 10, height: 2)
                    }
                }
            }
        }
        .padding(.vertical, 6)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("HydrogenSplat")
    }
}
