import SwiftUI
import Charts
import HSCore

/// The old Train panel's name, kept so an older mount still compiles: the Train step is
/// `TrainStepPage` (TrainStepPage.swift). This file holds the pieces both share — the queue
/// view, the progress card in words, the growth chart, and a train followed from Terminal.
struct TrainView: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var store: ProjectStore
    let project: ProjectSummary
    let manifest: Manifest

    var body: some View {
        TrainStepPage(project: project)
    }
}

/// A label, its control(s), and one line of help under them.
struct Handle<Content: View>: View {
    let title: String
    let help: String
    @ViewBuilder let content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 10) {
                Text(title).frame(width: 130, alignment: .leading)
                content()
            }
            Text(help)
                .font(.caption).foregroundStyle(.secondary)
                .padding(.leading, 140)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

/// "Default" or a number, with a few quick choices.
struct OptionalNumber: View {
    @Binding var value: Double?
    let placeholder: String
    let choices: [Double]

    var body: some View {
        HStack(spacing: 6) {
            TextField(placeholder, value: $value, format: .number)
                .textFieldStyle(.roundedBorder).frame(width: 120)
            ForEach(choices, id: \.self) { c in
                Button(c == c.rounded() ? String(Int(c)) : String(c)) { value = c }.controlSize(.small)
            }
            if value != nil {
                Button("Default") { value = nil }.controlSize(.small)
            }
        }
    }
}

// MARK: - the running queue

/// Train → archive → score, as one strip of step chips, the train step's progress in words with
/// its growth chart, and the later steps' panels.
struct QueueView: View {
    @ObservedObject var queue: RunQueue
    let totalIters: Int
    var growthStopIter: Int = 30_000
    var depthHeld: Bool = false
    var spreadOn: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                ForEach(Array(queue.steps.enumerated()), id: \.offset) { i, s in
                    StepChip(title: s.title, state: chipState(i))
                }
                Spacer()
                if queue.isRunning {
                    Button("Stop") { queue.cancel() }
                        .help("Interrupts the current step; the stage is marked failed and the project unlocked")
                }
            }
            if let first = queue.sessions.first {
                TrainProgressCard(session: first, totalIters: totalIters, growthStopIter: growthStopIter,
                                  depthHeld: depthHeld, spreadOn: spreadOn)
                GrowthChart(session: first, totalIters: totalIters)
            }
            if let cur = queue.current, queue.index > 0 {
                RunPanel(session: cur, showMetrics: false)
            }
        }
    }

    private func chipState(_ i: Int) -> StepChip.State {
        if i < queue.index { return .done }
        if i > queue.index { return queue.state == .running ? .waiting : .skipped }
        switch queue.state {
        case .running: return .running
        case .finished: return .done
        case .failed, .cancelled: return .failed
        case .idle: return .waiting
        }
    }
}

struct StepChip: View {
    enum State { case waiting, running, done, failed, skipped }
    let title: String
    let state: State

    var body: some View {
        HStack(spacing: 5) {
            switch state {
            case .waiting: Image(systemName: "circle").foregroundStyle(.secondary)
            case .running: ProgressView().controlSize(.mini)
            case .done: Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
            case .failed: Image(systemName: "xmark.circle.fill").foregroundStyle(.red)
            case .skipped: Image(systemName: "minus.circle").foregroundStyle(.secondary)
            }
            Text(title).font(.caption)
        }
        .padding(.horizontal, 8).padding(.vertical, 3)
        .background(Capsule().fill(Color.secondary.opacity(0.12)))
    }
}

// MARK: - progress in words

/// "Training · 38 % · about 51 minutes left", the step / splats / rate line, three chips from the
/// train log's depth metrics, and the one line that explains the quiet last 5 %. Every number
/// is the engine's own; where it has none yet the chip says what it will say.
struct TrainProgressCard: View {
    @ObservedObject var session: RunSession
    let totalIters: Int
    let growthStopIter: Int
    /// The run holds the model to a registered scan (`--depth-weight`).
    let depthHeld: Bool
    /// The run thins the surface (`--depth-spread-weight`).
    let spreadOn: Bool

    private var live: LiveProgress { session.liveProgress }

    private var headline: String {
        if case .finished(let code) = session.state {
            return code == 0 && session.errors.isEmpty ? "Training finished" : "Training stopped"
        }
        var s = "Training"
        if let p = live.percent { s += " · \(p) %" }
        if let e = live.eta { s += " · about \(TrainProgressCard.spoken(e)) left" }
        else if live.percent == nil { s += " · starting" }
        return s
    }

    /// "51 minutes", "2 hours 10 minutes", "under a minute".
    static func spoken(_ seconds: Double) -> String {
        if seconds < 60 { return "a minute" }
        let m = Int((seconds / 60).rounded())
        if m < 60 { return "\(m) minute\(m == 1 ? "" : "s")" }
        let h = m / 60, r = m % 60
        return r == 0 ? "\(h) hour\(h == 1 ? "" : "s")" : "\(h) hour\(h == 1 ? "" : "s") \(r) minutes"
    }

    private var detailLine: String {
        var parts: [String] = []
        if let d = live.done {
            parts.append("step \(JSONValue.number(d).display) of \(JSONValue.number(Double(totalIters)).display)")
        }
        if let k = session.lastProgressKey, let e = session.progress[k], let n = RunSession.leadingNumber(e.detail) {
            parts.append("\(JSONValue.number(n).display) splats")
        }
        if let k = session.lastProgressKey, let r = session.progress[k]?.rate, r > 0 {
            parts.append(String(format: "%.1f steps a second", r))
        }
        return parts.joined(separator: " · ")
    }

    private var growthChip: (String, Color?) {
        let stop = JSONValue.number(Double(growthStopIter)).display
        if let d = live.done, d >= Double(growthStopIter) { return ("growth stopped at step \(stop) · refining", nil) }
        return ("growing until step \(stop)", nil)
    }

    private var depthChip: (String, Color?) {
        guard depthHeld else { return ("held to the scan · not in this run", nil) }
        if let e = session.metric("train", "depth_rel_error")?.double {
            return (String(format: "held to the scan · %.2f %% off", e * 100), e > 0.02 ? Color.orange : Color.green)
        }
        return ("held to the scan · measured once the depth pass reports", nil)
    }

    private var surfaceChip: (String, Color?) {
        guard spreadOn else { return ("surface · thinning off", nil) }
        if let s = session.metric("train", "depth_rel_spread")?.double {
            return (String(format: "surface · spread %.2f %%", s * 100), s > 0.01 ? Color.orange : Color.green)
        }
        return ("surface · measured once the spread term starts", nil)
    }

    private var nearTheEnd: Bool {
        guard session.isRunning, let f = live.fraction else { return false }
        return f >= 0.93
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(headline).font(.headline).monospacedDigit()
            if let f = live.fraction { ProgressView(value: f) }
            if !detailLine.isEmpty { Text(detailLine).foregroundStyle(.secondary).monospacedDigit() }
            HStack(spacing: 8) {
                MetricChip(text: growthChip.0, tint: growthChip.1)
                MetricChip(text: depthChip.0, tint: depthChip.1)
                MetricChip(text: surfaceChip.0, tint: surfaceChip.1)
            }
            if nearTheEnd {
                Text("The counter goes quiet for the last 5 %: Brush stops adding and refining splats about 2,000 steps before the end and writes the export. It is still working.")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            if let e = session.errors.last, let m = e.message {
                Label(m + (e.hint.map { " — " + $0 } ?? ""), systemImage: "xmark.octagon")
                    .foregroundStyle(.red).fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color.secondary.opacity(0.08)))
    }
}

// MARK: - growth charts

/// The live chart of an in-app run: the session's progress samples.
struct GrowthChart: View {
    @ObservedObject var session: RunSession
    let totalIters: Int

    var body: some View {
        GrowthCurveChart(points: session.samples.compactMap { s in s.value.map { GrowthPoint(done: s.done, value: $0) } },
                         totalIters: totalIters)
    }
}

/// Splats over iterations from any source — a live session, a Terminal run's events log
/// (`GrowthCurve.fromEventsLog`), or a finished run's manifest (`GrowthCurve.fromManifest`).
struct GrowthCurveChart: View {
    let points: [GrowthPoint]
    let totalIters: Int
    var height: CGFloat = 140

    /// One point per iteration, in order. The engine can emit the same iteration twice (a forced
    /// tick beside Brush's own line); identical points under `id: \.self` made Charts draw the
    /// area fill between the twins as a spike to the top of the axis.
    private var series: [GrowthPoint] {
        var byIter: [Double: Double] = [:]
        for p in points { byIter[p.done] = p.value }
        return byIter.keys.sorted().compactMap { k in byIter[k].map { GrowthPoint(done: k, value: $0) } }
    }

    var body: some View {
        let pts = series
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Splats").font(.caption.weight(.semibold))
                Spacer()
                if let last = pts.last {
                    Text("\(JSONValue.number(last.value).display) at step \(JSONValue.number(last.done).display)")
                        .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                }
            }
            Chart(pts, id: \.done) { p in
                AreaMark(x: .value("Step", p.done), y: .value("Splats", p.value))
                    .foregroundStyle(Brand.tally.opacity(0.12))
                LineMark(x: .value("Step", p.done), y: .value("Splats", p.value))
                    .foregroundStyle(Brand.tally)
            }
            .chartXScale(domain: 0...Double(max(totalIters, 1)))
            .chartYAxis { AxisMarks(format: FloatingPointFormatStyle<Double>.number.notation(.compactName)) }
            .frame(height: height)
        }
    }
}

// MARK: - a train started from Terminal

struct ExternalTrainView: View {
    let project: ProjectSummary
    let manifest: Manifest
    let lock: ProjectSummary.LockInfo
    @State private var status: TrainLogStatus?
    @State private var curve: [GrowthPoint] = []
    @State private var showLog = false
    private let tick = Timer.publish(every: 2, on: .main, in: .common).autoconnect()

    private var total: Int {
        // brush_argv is what Brush actually got; hs's own argv only has the flag when it was overridden
        let brush = manifest.raw["stages"]?["train"]?["brush_argv"]?.array?.compactMap { $0.string } ?? []
        for argv in [brush, manifest.stage("train")?.argv ?? []] {
            if let i = argv.firstIndex(of: "--total-train-iters"), i + 1 < argv.count, let n = Int(argv[i + 1]) { return n }
        }
        return TrainSettings().totalIters
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("Training in another window (pid \(lock.pid.map(String.init) ?? "?")). Following its log; the run strip above can stop it.",
                  systemImage: "terminal")
                .foregroundStyle(.secondary)
            if let s = status {
                let frac = min(Double(s.iter) / Double(max(total, 1)), 1)
                let quiet = s.lastLineAt.map { Date().timeIntervalSince($0) } ?? 0
                let eta = quiet <= TrainLogStatus.sleepGap ? s.etaSeconds(total: total) : nil
                Text("Training · \(Int((frac * 100).rounded(.down))) %" + (eta.map { " · about \(TrainProgressCard.spoken($0)) left" } ?? ""))
                    .font(.headline).monospacedDigit()
                ProgressView(value: frac)
                HStack(spacing: 16) {
                    Text("step \(JSONValue.number(Double(s.iter)).display) of \(JSONValue.number(Double(total)).display)").monospacedDigit()
                    Text("\(JSONValue.number(Double(s.splats)).display) splats").monospacedDigit()
                    if let r = s.rate { Text(String(format: "%.1f steps a second", r)).monospacedDigit() }
                    if s.lastLineAt != nil {
                        Text(quiet > TrainLogStatus.sleepGap ? "no progress for \(Format.duration(quiet)) — asleep or finishing?" : "updated \(Format.duration(quiet)) ago")
                            .foregroundStyle(quiet > TrainLogStatus.sleepGap ? Color.orange : Color.secondary)
                    }
                    Spacer()
                    if let st = s.runStarted { Text("started \(st)").foregroundStyle(.secondary) }
                }
                .font(.callout)
                if !curve.isEmpty {
                    GrowthCurveChart(points: curve, totalIters: total)
                }
                DisclosureGroup("Log", isExpanded: $showLog) {
                    ScrollView {
                        Text(s.tail.joined(separator: "\n"))
                            .font(.system(.caption, design: .monospaced))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
                    }
                    .frame(height: 200)
                }
            } else {
                Text("No train log yet.").foregroundStyle(.secondary)
            }
        }
        .onAppear(perform: read)
        .onReceive(tick) { _ in read() }
    }

    private func read() {
        let p = (project.path as NSString).appendingPathComponent("logs/train.log")
        let proj = project.path
        DispatchQueue.global(qos: .utility).async {
            let s = TrainLogStatus.read(path: p)
            let c = GrowthCurve.fromEventsLog(project: proj)
            DispatchQueue.main.async { status = s; curve = c }
        }
    }
}
