import SwiftUI
import AppKit
import Charts
import HSCore

/// Select frames from the app: the selector's settings, the exact command, a Select Frames
/// button, and — once it has run — a contact sheet built from select/quality.json that shows
/// every pick's Laplacian, focus, exposure off the set's median, both eyes and noise, flags the
/// ones worth a look, and says where a sharper frame sat. Then a Solve button, so a new clip
/// gets from ingest to trainable without Terminal.
struct SelectView: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var store: ProjectStore
    let project: ProjectSummary
    let manifest: Manifest

    @State private var quality: FrameQuality?
    @State private var sort: QualitySort = .frame
    @State private var only: String?          // a flag code, or "flagged" / "clean"
    @State private var showAdvanced = false
    @State private var confirmRedo = false

    private var settings: Binding<SelectSettings> {
        Binding(get: { model.selectSettings[project.path] ?? SelectSettings() },
                set: { model.selectSettings[project.path] = $0 })
    }
    private var queue: RunQueue? { model.selectQueues[project.path] }
    private var selectStage: StageState? { manifest.stage("select") }
    private var selectDone: Bool { selectStage?.status == .done }
    private var solveStage: StageState? { manifest.stage("solve") }
    private var lockAlive: Bool { project.lock?.alive == true }
    /// Changes whenever a select run finishes, so the report is re-read.
    private var reloadKey: String {
        "\(project.path)|\(selectStage?.status.rawValue ?? "")|\(selectStage?.finished?.timeIntervalSince1970 ?? 0)"
    }

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 14) {
                if manifest.isArray {
                    Text("An array project has one frame per camera — ingest already did the selecting.")
                        .foregroundStyle(.secondary)
                } else {
                    handles
                    commandPreview
                    startRow
                    if let q = quality {
                        Divider()
                        summary(q)
                        charts(q)
                        contactSheet(q)
                    } else if selectDone {
                        Text("This selection was made before the app measured frames. Press Select Frames to redo it with the quality report.")
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .padding(4)
            .frame(maxWidth: .infinity, alignment: .leading)
        } label: {
            HStack {
                Text("Select frames").font(.headline)
                if let s = selectStage { StatusBadge(status: s.status) }
            }
        }
        .task(id: reloadKey) { quality = FrameQuality.load(project: project.path) }
        .confirmationDialog("Select the frames again?", isPresented: $confirmRedo) {
            Button("Select again", role: .destructive) { startSelect() }
        } message: {
            Text("This replaces the current frames and quality report"
                 + (solveStage?.status == .done ? ", and marks the solve (and everything after it) stale — it will need solving again." : "."))
        }
    }

    // MARK: settings

    private var handles: some View {
        VStack(alignment: .leading, spacing: 8) {
            Handle(title: "Parallax residual",
                   help: "Pick a frame once the camera has moved this far (px of parallax at 480 wide, after taking out rotation). Lower = more frames, closer together. 1.5 solved the reference clip 65/65.") {
                TextField("1.5", value: settings.residual, format: .number)
                    .textFieldStyle(.roundedBorder).frame(width: 80)
                Text("px").foregroundStyle(.secondary)
            }
            DisclosureGroup("More settings", isExpanded: $showAdvanced) {
                VStack(alignment: .leading, spacing: 8) {
                    Handle(title: "Gap between picks",
                           help: "Never closer than the minimum; always pick by the maximum even if the camera stood still (those picks are flagged).") {
                        TextField("6", value: settings.minGap, format: .number)
                            .textFieldStyle(.roundedBorder).frame(width: 60)
                        Text("to").foregroundStyle(.secondary)
                        TextField("90", value: settings.maxGap, format: .number)
                            .textFieldStyle(.roundedBorder).frame(width: 60)
                        Text("frames").foregroundStyle(.secondary)
                    }
                    Handle(title: "Search window",
                           help: "Once the parallax is reached, take the sharpest of this many frames.") {
                        TextField("4", value: settings.search, format: .number)
                            .textFieldStyle(.roundedBorder).frame(width: 60)
                        Text("frames").foregroundStyle(.secondary)
                    }
                    Handle(title: "Max clipped",
                           help: "Skip a candidate with more than this fraction of pixels at 250+ when a cleaner one is in the window.") {
                        TextField("0.02", value: settings.maxClip, format: .number)
                            .textFieldStyle(.roundedBorder).frame(width: 80)
                    }
                    Handle(title: "Frame range",
                           help: "Only look at these source frames. End −1 = to the end of the clip.") {
                        TextField("0", value: settings.start, format: .number)
                            .textFieldStyle(.roundedBorder).frame(width: 70)
                        Text("to").foregroundStyle(.secondary)
                        TextField("-1", value: settings.end, format: .number)
                            .textFieldStyle(.roundedBorder).frame(width: 70)
                        if settings.wrappedValue != SelectSettings.defaults {
                            Button("Defaults") { settings.wrappedValue = SelectSettings() }.controlSize(.small)
                        }
                    }
                }
                .padding(.top, 6)
            }
        }
    }

    private var commandPreview: some View {
        let args = settings.wrappedValue.arguments(project: project.path)
        let line = (["hs"] + args.map { $0.hasPrefix(project.path) ? "$P" : $0 }).map(shellQuote).joined(separator: " ")
        return VStack(alignment: .leading, spacing: 4) {
            Text("Runs (P = this project):").font(.caption).foregroundStyle(.secondary)
            CopyableCommand(text: line)
        }
    }

    private var startRow: some View {
        HStack(spacing: 12) {
            let running = queue?.isRunning ?? false
            let problem = settings.wrappedValue.problem
            Button(running && queue?.steps.first?.title == "Select frames" ? "Selecting…" : "Select Frames") {
                if selectDone { confirmRedo = true } else { startSelect() }
            }
            .keyboardShortcut("s", modifiers: [.command, .shift])
            .disabled(running || lockAlive || problem != nil || !model.config.problems.isEmpty)

            if selectDone {
                Button(running && queue?.steps.first?.title == "Solve" ? "Solving…" : solveLabel) { startSolve() }
                    .disabled(running || lockAlive || !model.config.problems.isEmpty)
                    .help("hs solve: COLMAP on the selected frames. Training unlocks when it is done.")
            }
            if let p = problem {
                Label(p, systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
            } else if lockAlive, let l = project.lock {
                Label("\(l.stage ?? "a stage") is running", systemImage: "lock.fill").foregroundStyle(.secondary)
            }
        }
    }

    private var solveLabel: String {
        switch solveStage?.status {
        case .done: return "Solve again"
        case .stale: return "Solve (stale)"
        default: return "Solve"
        }
    }

    private func startSelect() {
        run([RunQueue.Step(title: "Select frames", arguments: settings.wrappedValue.arguments(project: project.path))])
    }

    private func startSolve() {
        run([RunQueue.Step(title: "Solve", arguments: ["solve", "-p", project.path])])
    }

    private func run(_ steps: [RunQueue.Step]) {
        let q = RunQueue(config: model.config, steps: steps)
        let path = project.path
        q.onStep = { s in model.projectRuns[path] = s }
        q.onFinish = { _ in store.reload() }
        model.selectQueues[path] = q
        q.start()
    }

    // MARK: summary

    private func summary(_ q: FrameQuality) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline, spacing: 14) {
                Label("\(q.frames.count) frames", systemImage: "photo.stack").font(.subheadline.weight(.semibold))
                if let t = q.framesTotal { Text("of \(t)").foregroundStyle(.secondary) }
                if let g = q.medianGap { Text("median gap \(JSONValue.number(g).display)").foregroundStyle(.secondary) }
                if let s = q.medians.sharp { Text("median Laplacian \(Int(s.rounded()))").foregroundStyle(.secondary) }
                Text(q.flagged == 0 ? "nothing flagged" : "\(q.flagged) flagged")
                    .foregroundStyle(q.flagged == 0 ? Color.green : Color.orange)
                if let c = selectStage?.metrics["frames_selected"], c.int != q.frames.count {
                    Text("(report is from an older run)").foregroundStyle(.orange)
                }
            }
            HStack(spacing: 6) {
                chip("All", code: nil, count: q.frames.count)
                chip("Flagged", code: "flagged", count: q.flagged)
                chip("Clean", code: "clean", count: q.frames.count - q.flagged)
                ForEach(q.flagsByCount, id: \.0) { item in
                    chip(item.0.replacingOccurrences(of: "_", with: " "), code: item.0, count: item.1)
                        .help(q.flagText[item.0] ?? item.0)
                }
                Spacer()
                Picker("Sort", selection: $sort) {
                    ForEach(QualitySort.allCases) { Text($0.rawValue).tag($0) }
                }
                .frame(width: 260)
            }
            ForEach(q.warnings ?? [], id: \.self) { w in
                Label(w, systemImage: "exclamationmark.triangle")
                    .font(.callout).foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if !q.measuredEyes {
                Text("Dry run: left eye only, no thumbnails.").font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private func chip(_ title: String, code: String?, count: Int) -> some View {
        let on = only == code
        return Button { only = code } label: {
            Text("\(title) \(count)")
                .font(.caption.weight(on ? .semibold : .regular))
                .padding(.horizontal, 8).padding(.vertical, 3)
                .background(Capsule().fill(on ? Color.accentColor.opacity(0.25) : Color.secondary.opacity(0.12)))
        }
        .buttonStyle(.plain)
    }

    private func visible(_ q: FrameQuality) -> [FrameQuality.Frame] {
        let f: [FrameQuality.Frame]
        switch only {
        case nil: f = q.frames
        case "flagged": f = q.frames.filter { $0.flagged }
        case "clean": f = q.frames.filter { !$0.flagged }
        case let code?: f = q.frames.filter { $0.flags.contains(code) }
        }
        return sort.sorted(f)
    }

    // MARK: charts

    private struct TracePoint: Identifiable {
        let id: Int
        let frame: Int
        let value: Double
    }

    private func tracePoints(_ q: FrameQuality, _ series: [Double?], clamp: ClosedRange<Double>) -> [TracePoint] {
        var out: [TracePoint] = []
        out.reserveCapacity(series.count)
        for (k, v) in series.enumerated() where k < q.trace.frame.count {
            if let v = v { out.append(TracePoint(id: k, frame: q.trace.frame[k], value: min(max(v, clamp.lowerBound), clamp.upperBound))) }
        }
        return out
    }

    /// Focus and exposure along the whole clip, picks on top: whether each pick sits on a peak
    /// of its neighbourhood, and where the soft or dark stretches are.
    private func charts(_ q: FrameQuality) -> some View {
        let focusLine = tracePoints(q, q.trace.focusRel, clamp: 0...2)
        let evLine = tracePoints(q, q.trace.ev, clamp: -2...2)
        let soft = q.threshold("soft_rel", 0.6)
        let evTol = q.threshold("ev_tol", 0.33)
        return VStack(alignment: .leading, spacing: 10) {
            Text("Focus along the clip (1 = the picks' median; Laplacian with contrast divided out)")
                .font(.caption).foregroundStyle(.secondary)
            Chart {
                ForEach(focusLine) { p in
                    LineMark(x: .value("Frame", p.frame), y: .value("Focus", p.value))
                        .foregroundStyle(Color.secondary.opacity(0.6))
                        .lineStyle(StrokeStyle(lineWidth: 1))
                }
                RuleMark(y: .value("Soft", soft))
                    .foregroundStyle(Color.orange.opacity(0.5))
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3]))
                ForEach(q.frames) { f in
                    if let v = f.focusRel {
                        PointMark(x: .value("Frame", f.frame), y: .value("Focus", min(max(v, 0), 2)))
                            .foregroundStyle(f.flagged ? Color.orange : Color.green)
                            .symbolSize(28)
                    }
                    if let n = f.sharperNearby, let k = q.trace.frame.firstIndex(of: n.frame),
                       k < q.trace.focusRel.count, let v = q.trace.focusRel[k] {
                        PointMark(x: .value("Frame", n.frame), y: .value("Focus", min(max(v, 0), 2)))
                            .foregroundStyle(Color.blue)
                            .symbol(.diamond)
                            .symbolSize(36)
                    }
                }
            }
            .chartYScale(domain: 0...2)
            .frame(height: 140)

            Text("Exposure along the clip (stops off the picks' median)").font(.caption).foregroundStyle(.secondary)
            Chart {
                ForEach(evLine) { p in
                    LineMark(x: .value("Frame", p.frame), y: .value("EV", p.value))
                        .foregroundStyle(Color.secondary.opacity(0.6))
                        .lineStyle(StrokeStyle(lineWidth: 1))
                }
                RuleMark(y: .value("+tol", evTol)).foregroundStyle(Color.orange.opacity(0.5))
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3]))
                RuleMark(y: .value("-tol", -evTol)).foregroundStyle(Color.orange.opacity(0.5))
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3]))
                ForEach(q.frames) { f in
                    if let v = f.ev {
                        PointMark(x: .value("Frame", f.frame), y: .value("EV", min(max(v, -2), 2)))
                            .foregroundStyle(f.flagged ? Color.orange : Color.green)
                            .symbolSize(28)
                    }
                }
            }
            .chartYScale(domain: -2...2)
            .frame(height: 110)
            HStack(spacing: 14) {
                Label("clean pick", systemImage: "circle.fill").foregroundStyle(.green)
                Label("flagged pick", systemImage: "circle.fill").foregroundStyle(.orange)
                Label("sharper frame nearby", systemImage: "diamond.fill").foregroundStyle(.blue)
            }
            .font(.caption)
        }
    }

    // MARK: contact sheet

    private func contactSheet(_ q: FrameQuality) -> some View {
        let frames = visible(q)
        return LazyVGrid(columns: [GridItem(.adaptive(minimum: 230), spacing: 10, alignment: .top)],
                         alignment: .leading, spacing: 10) {
            ForEach(frames) { f in
                FrameCard(project: project.path, frame: f, quality: q)
            }
        }
    }
}

/// One pick: its left-eye thumbnail and every number the report has for it.
struct FrameCard: View {
    let project: String
    let frame: FrameQuality.Frame
    let quality: FrameQuality

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
                Text("#\(frame.sel)")
                    .font(.caption.monospacedDigit().weight(.semibold))
                    .padding(.horizontal, 5).padding(.vertical, 1)
                    .background(Capsule().fill(.black.opacity(0.6)))
                    .foregroundStyle(.white)
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
            if quality.measuredEyes {
                row("Right eye", frame.focusRRel.map { String(format: "focus %.2f×", $0) } ?? "—",
                    detail: frame.eyeEV.map { String(format: "%+.2f EV vs L", $0) },
                    warn: frame.flags.contains("right_soft") || frame.flags.contains("eye_exposure"))
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
            if let n = frame.sharperNearby {
                Button("Copy sharper frame number (\(n.frame))") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(String(n.frame), forType: .string)
                }
            }
        }
        .help("Double-click to open the full 2×1 frame")
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
