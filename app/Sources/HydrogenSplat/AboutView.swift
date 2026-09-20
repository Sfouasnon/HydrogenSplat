import SwiftUI
import AppKit
import HSCore

/// HydrogenSplat ▸ About HydrogenSplat. Replaces the stock panel so the third-party notices
/// (HSCore/Acknowledgements.swift) are somewhere a user can actually read them.
struct AboutView: View {
    private var version: String {
        let info = Bundle.main.infoDictionary
        guard let v = info?["CFBundleShortVersionString"] as? String else { return "development build (swift run)" }
        return "Version \(v)" + ((info?["CFBundleVersion"] as? String).map { " (\($0))" } ?? "")
    }

    var body: some View {
        VStack(spacing: 0) {
            VStack(spacing: 6) {
                if let icon = Brand.appIcon {
                    Image(nsImage: icon).resizable().interpolation(.high).frame(width: 84, height: 84)
                }
                (Text("HYDROGEN ").foregroundStyle(.primary) + Text("SPLAT").foregroundStyle(Brand.tally))
                    .font(.system(size: 17, weight: .semibold).width(.expanded)).tracking(1.6)
                Text(version).font(.callout).foregroundStyle(.secondary).textSelection(.enabled)
                Text("Stereo video and camera arrays to Gaussian-splat camera moves.")
                    .font(.callout).foregroundStyle(.secondary)
            }
            .padding(.top, 22).padding(.bottom, 16)
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    Text("Acknowledgements").font(.headline)
                    Text("HydrogenSplat includes the following open-source software.")
                        .font(.callout).foregroundStyle(.secondary)
                    ForEach(Acknowledgements.bundled) { a in AcknowledgementRow(item: a) }
                    Divider().padding(.vertical, 2)
                    Text("HydrogenSplat also drives these tools, which you install separately and which are not part of this app:")
                        .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                    VStack(alignment: .leading, spacing: 3) {
                        ForEach(Acknowledgements.drives, id: \.name) { d in
                            if let url = URL(string: d.url) { Link(d.name, destination: url).font(.callout) }
                        }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(18)
            }
        }
        .frame(width: 560, height: 620)
    }
}

private struct AcknowledgementRow: View {
    let item: Acknowledgement
    @State private var open = false

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(item.name).font(.body.weight(.semibold))
                Text("\(item.license) licence").font(.caption.weight(.semibold))
                    .padding(.horizontal, 6).padding(.vertical, 1)
                    .background(Capsule().fill(Color.secondary.opacity(0.18)))
                Spacer()
                if let url = URL(string: item.url) { Link("Source", destination: url).font(.callout) }
            }
            Text("© \(item.author)").font(.callout).foregroundStyle(.secondary)
            Text(item.use).font(.callout).fixedSize(horizontal: false, vertical: true)
            DisclosureGroup(open ? "Hide licence" : "Show licence", isExpanded: $open) {
                Text(item.licenseText)
                    .font(.system(size: 10.5, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(8)
                    .background(RoundedRectangle(cornerRadius: 6).fill(Color.secondary.opacity(0.08)))
            }
            .font(.callout)
        }
    }
}

/// The menu item. A command's content is a view, so it can reach `openWindow`.
struct AboutCommand: View {
    @Environment(\.openWindow) private var openWindow
    var body: some View {
        Button("About HydrogenSplat") { openWindow(id: "about") }
    }
}
