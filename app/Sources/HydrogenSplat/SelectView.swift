import SwiftUI
import AppKit
import HSCore

/// The pieces of the Frames step (FramesPage): the picker's settings, the strip of picked frames
/// with its contact sheet, and the frame card, flag chips and thumbnail that other steps reuse.

/// A label, its control(s), and at most one line under them.
struct StepSetting<Content: View>: View {
    let title: String
    var help: String? = nil
    @ViewBuilder let content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 10) {
                Text(title).frame(width: 130, alignment: .leading)
                content()
            }
            if let h = help {
                Text(h).font(.caption).foregroundStyle(.secondary)
                    .padding(.leading, 140).lineLimit(1)
            }
        }
    }
}

/// How `hs select` picks frames. The defaults are the engine's; only what differs is passed.
struct SelectSettingsPanel: View {
    @Binding var settings: SelectSettings
    @State private var more = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            StepSetting(title: "Camera movement", help: "Pick a frame each time the camera has moved this far; lower picks more frames.") {
                TextField("1.5", value: $settings.residual, format: .number)
                    .textFieldStyle(.roundedBorder).frame(width: 80)
                Text("px").foregroundStyle(.secondary)
            }
            StepSetting(title: "Keyframes", help: "Only the codec's clean frames; for a fast shutter where compression, not motion, limits sharpness.") {
                Toggle("Keyframes only", isOn: $settings.keyframes)
            }
            DisclosureGroup("More", isExpanded: $more) {
                VStack(alignment: .leading, spacing: 8) {
                    StepSetting(title: "Gap between picks", help: "Never closer than the first; always a pick by the second, even if the camera stood still.") {
                        TextField("6", value: $settings.minGap, format: .number)
                            .textFieldStyle(.roundedBorder).frame(width: 60)
                        Text("to").foregroundStyle(.secondary)
                        TextField("90", value: $settings.maxGap, format: .number)
                            .textFieldStyle(.roundedBorder).frame(width: 60)
                        Text("frames").foregroundStyle(.secondary)
                    }
                    StepSetting(title: "Search window", help: "Once the movement is reached, take the sharpest of this many frames.") {
                        TextField("4", value: $settings.search, format: .number)
                            .textFieldStyle(.roundedBorder).frame(width: 60)
                        Text("frames").foregroundStyle(.secondary)
                    }
                    StepSetting(title: "Max clipped", help: "Skip a frame with more than this share of blown-out pixels when a cleaner one is near.") {
                        TextField("0.02", value: $settings.maxClip, format: .number)
                            .textFieldStyle(.roundedBorder).frame(width: 80)
                    }
                    StepSetting(title: "Highlight knee", help: "Eases the brightest values down from the ceiling in every written frame; off by default.") {
                        TextField("off", value: $settings.highlightKnee, format: .number)
                            .textFieldStyle(.roundedBorder).frame(width: 80)
                        if settings.highlightKnee != nil {
                            Button("Off") { settings.highlightKnee = nil }.controlSize(.small)
                        }
                    }
                    StepSetting(title: "Frame range", help: "Only look at these source frames; −1 means to the end.") {
                        TextField("0", value: $settings.start, format: .number)
                            .textFieldStyle(.roundedBorder).frame(width: 70)
                        Text("to").foregroundStyle(.secondary)
                        TextField("-1", value: $settings.end, format: .number)
                            .textFieldStyle(.roundedBorder).frame(width: 70)
                        if settings != SelectSettings.defaults {
                            Button("Defaults") { settings = SelectSettings() }.controlSize(.small)
                        }
                    }
                }
                .padding(.top, 6)
            }
        }
    }
}

/// The picked frames: a strip of thumbnails, a filter, and the contact sheet behind "Details".
struct PickedFramesStrip: View {
    let project: String
    let quality: FrameQuality
    /// The pick `hs exposure` will match to, for the badge on its card.
    var referenceRole: (FrameQuality.Frame) -> String? = { _ in nil }
    var useAsReference: ((FrameQuality.Frame) -> Void)? = nil

    @State private var only: String?          // a flag code, or "flagged" / "clean"
    @State private var sort: QualitySort = .frame
    @AppStorage("select.stillsOpen") private var detailsOpen = false

    private var selectDir: String { (project as NSString).appendingPathComponent("select") }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 6) {
                chip("All", code: nil, count: quality.frames.count)
                chip("Flagged", code: "flagged", count: quality.flagged)
                chip("Clean", code: "clean", count: quality.frames.count - quality.flagged)
                ForEach(quality.flagsByCount, id: \.0) { item in
                    chip(item.0.replacingOccurrences(of: "_", with: " "), code: item.0, count: item.1)
                        .help(quality.flagText[item.0] ?? item.0)
                }
                Spacer()
                Toggle("Details", isOn: $detailsOpen).toggleStyle(.checkbox)
            }
            ForEach(quality.warnings ?? [], id: \.self) { w in
                Label(w, systemImage: "exclamationmark.triangle")
                    .font(.callout).foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if detailsOpen {
                HStack {
                    Picker("Sort", selection: $sort) {
                        ForEach(QualitySort.allCases) { Text($0.rawValue).tag($0) }
                    }
                    .frame(width: 260)
                    Spacer()
                }
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 230), spacing: 10, alignment: .top)],
                          alignment: .leading, spacing: 10) {
                    ForEach(visible) { f in
                        FrameCard(project: project, frame: f, quality: quality,
                                  referenceRole: referenceRole(f),
                                  useAsReference: reference(f))
                    }
                }
            } else {
                ScrollView(.horizontal, showsIndicators: true) {
                    LazyHStack(spacing: 6) {
                        ForEach(visible) { f in
                            thumb(f)
                        }
                    }
                    .padding(.vertical, 2)
                }
                .frame(height: 96)
            }
        }
    }

    private func reference(_ f: FrameQuality.Frame) -> (() -> Void)? {
        guard let use = useAsReference else { return nil }
        return { use(f) }
    }

    private var visible: [FrameQuality.Frame] {
        let f: [FrameQuality.Frame]
        switch only {
        case nil: f = quality.frames
        case "flagged": f = quality.frames.filter { $0.flagged }
        case "clean": f = quality.frames.filter { !$0.flagged }
        case let code?: f = quality.frames.filter { $0.flags.contains(code) }
        }
        return sort.sorted(f)
    }

    private func thumb(_ f: FrameQuality.Frame) -> some View {
        Group {
            if let t = f.thumb {
                ThumbImage(path: (selectDir as NSString).appendingPathComponent(t))
            } else {
                Rectangle().fill(Color.secondary.opacity(0.12)).aspectRatio(16 / 9, contentMode: .fit)
            }
        }
        .frame(height: 84)
        .clipShape(RoundedRectangle(cornerRadius: 4))
        .overlay(RoundedRectangle(cornerRadius: 4).stroke(f.flagged ? Color.orange : Color.clear, lineWidth: 2))
        .overlay(alignment: .bottomLeading) {
            Text("#\(f.sel)").font(.caption2.monospacedDigit().weight(.semibold))
                .padding(.horizontal, 4).padding(.vertical, 1)
                .background(Capsule().fill(.black.opacity(0.6))).foregroundStyle(.white)
                .padding(3)
        }
        .help(f.flags.isEmpty ? "frame \(f.frame)" : "frame \(f.frame): " + f.flags.joined(separator: ", "))
        .onTapGesture(count: 2) {
            if let file = f.file {
                NSWorkspace.shared.open(URL(fileURLWithPath: (selectDir as NSString).appendingPathComponent("frames/\(file)")))
            }
        }
    }

    private func chip(_ title: String, code: String?, count: Int) -> some View {
        let on = only == code
        return Button { only = code } label: {
            Text("\(title) \(count)")
                .font(.caption.weight(on ? .semibold : .regular))
                .padding(.horizontal, 8).padding(.vertical, 3)
                .background(Capsule().fill(on ? Brand.accent.opacity(0.25) : Color.secondary.opacity(0.12)))
        }
        .buttonStyle(.plain)
    }
}

/// One pick: its left-eye thumbnail and every number the report has for it.
struct FrameCard: View {
    let project: String
    let frame: FrameQuality.Frame
    let quality: FrameQuality
    var referenceRole: String? = nil
    var useAsReference: (() -> Void)? = nil

    private var selectDir: String { (project as NSString).appendingPathComponent("select") }
    private var framePath: String? {
        frame.file.map { (selectDir as NSString).appendingPathComponent("frames/\($0)") }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            ZStack(alignment: .topLeading) {
                if let t = frame.thumb {
                    ThumbImage(path: (selectDir as NSString).appendingPathComponent(t))
                } else {
                    Rectangle().fill(Color.secondary.opacity(0.12)).aspectRatio(16 / 9, contentMode: .fit)
                }
                HStack(spacing: 4) {
                    Text("#\(frame.sel)")
                        .font(.caption.monospacedDigit().weight(.semibold))
                        .padding(.horizontal, 5).padding(.vertical, 1)
                        .background(Capsule().fill(.black.opacity(0.6)))
                        .foregroundStyle(.white)
                    if let role = referenceRole {
                        Label(role, systemImage: "sun.max.fill")
                            .font(.caption2.weight(.semibold))
                            .padding(.horizontal, 5).padding(.vertical, 1)
                            .background(Capsule().fill(Color.yellow.opacity(0.85)))
                            .foregroundStyle(.black)
                    }
                }
                .padding(4)
            }
            .overlay(RoundedRectangle(cornerRadius: 3)
                .stroke(frame.flagged ? Color.orange : Color.clear, lineWidth: 2))

            Text("frame \(frame.frame)" + (frame.seconds.map { String(format: " · %.1f s", $0) } ?? "")
                 + (frame.gap.map { " · gap \($0)" } ?? ""))
                .font(.caption).foregroundStyle(.secondary)

            row("Laplacian", String(Int(frame.sharp.rounded())),
                detail: frame.focusRel.map { String(format: "focus %.2f×", $0) },
                warn: frame.flags.contains("soft"))
            row("Exposure", frame.ev.map { String(format: "%+.2f EV", $0) } ?? "—",
                detail: frame.dark.flatMap { $0 > 0.01 ? String(format: "%.0f%% crushed", $0 * 100) : nil },
                warn: frame.flags.contains("exposure") || frame.flags.contains("crushed"))
            if quality.hasRightEye {
                row("Right eye", frame.focusRRel.map { String(format: "focus %.2f×", $0) } ?? "—",
                    detail: frame.eyeEV.map { String(format: "%+.2f EV vs L", $0) },
                    warn: frame.flags.contains("right_soft") || frame.flags.contains("eye_exposure"))
            }
            if quality.measuredEyes {
                row("Noise", frame.noise.map { String(format: "%.2f", $0) } ?? "—",
                    detail: frame.noiseRel.map { String(format: "%.2f× median", $0) },
                    warn: frame.flags.contains("noisy"))
            }
            row("Clipped", frame.clip.map { String(format: "%.1f%%", $0 * 100) } ?? "—",
                detail: frame.residual.map { $0 < 0 ? "parallax lost" : String(format: "parallax %.2f px", $0) },
                warn: frame.flags.contains("clipped") || frame.flags.contains("untracked"))

            if let n = frame.sharperNearby {
                Label(nearbyText(n), systemImage: "arrow.left.arrow.right")
                    .font(.caption).foregroundStyle(.blue)
                    .help("The sharpest usable frame between this pick and halfway to its neighbours. Swapping to it keeps the parallax spacing within half an interval.")
            }
            if !frame.flags.isEmpty {
                FlowTags(tags: frame.flags, text: quality.flagText)
            }
        }
        .padding(6)
        .background(RoundedRectangle(cornerRadius: 6).fill(Color.secondary.opacity(0.06)))
        .contentShape(Rectangle())
        .onTapGesture(count: 2) { openFrame() }
        .contextMenu {
            Button("Open frame") { openFrame() }.disabled(framePath == nil)
            Button("Reveal in Finder") {
                if let p = framePath { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: p)]) }
            }.disabled(framePath == nil)
            if let use = useAsReference {
                Button("Use as exposure reference") { use() }
            }
            if let n = frame.sharperNearby {
                Button("Copy sharper frame number (\(n.frame))") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(String(n.frame), forType: .string)
                }
            }
        }
        .help("Double-click to open the full frame")
    }

    private func nearbyText(_ n: FrameQuality.Nearby) -> String {
        let off = n.offset.map { $0 > 0 ? "+" + String($0) : String($0) } ?? "?"
        let gain = n.gain.map { String(format: "%.1f×", $0) } ?? "much"
        return "Frame " + String(n.frame) + " (" + off + ") is " + gain + " sharper"
    }

    private func openFrame() {
        if let p = framePath { NSWorkspace.shared.open(URL(fileURLWithPath: p)) }
    }

    private func row(_ k: String, _ v: String, detail: String?, warn: Bool) -> some View {
        HStack(spacing: 6) {
            Text(k).foregroundStyle(.secondary).frame(width: 64, alignment: .leading)
            Text(v).monospacedDigit().foregroundStyle(warn ? Color.orange : Color.primary)
            if let d = detail { Text(d).foregroundStyle(.secondary).lineLimit(1) }
        }
        .font(.caption)
    }
}

/// Flag chips that wrap onto as many lines as they need.
struct FlowTags: View {
    let tags: [String]
    let text: [String: String]

    var body: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 4) { chips }
            VStack(alignment: .leading, spacing: 3) { chips }
        }
    }

    @ViewBuilder private var chips: some View {
        ForEach(tags, id: \.self) { t in
            Text(t.replacingOccurrences(of: "_", with: " "))
                .font(.caption2.weight(.semibold))
                .padding(.horizontal, 6).padding(.vertical, 1)
                .background(Capsule().fill(Color.orange.opacity(0.18)))
                .foregroundStyle(.orange)
                .help(text[t] ?? t)
        }
    }
}

/// A thumbnail from disk, cached by path and modification date (a re-selection writes new
/// frames under the same names).
struct ThumbImage: View {
    let path: String
    private static let cache = NSCache<NSString, NSImage>()

    var body: some View {
        if let img = load() {
            Image(nsImage: img).resizable().aspectRatio(contentMode: .fit)
        } else {
            Rectangle().fill(Color.secondary.opacity(0.12)).aspectRatio(16 / 9, contentMode: .fit)
        }
    }

    private func load() -> NSImage? {
        let mtime = (try? FileManager.default.attributesOfItem(atPath: path)[.modificationDate] as? Date)?
            .timeIntervalSince1970 ?? 0
        let key = "\(path)|\(mtime)" as NSString
        if let hit = ThumbImage.cache.object(forKey: key) { return hit }
        guard let img = NSImage(contentsOfFile: path) else { return nil }
        ThumbImage.cache.setObject(img, forKey: key)
        return img
    }
}
