import SwiftUI
import AppKit
import HSCore

/// Live view of one hs run: progress bars, checks, metrics, errors and the raw event stream.
struct RunPanel: View {
    @ObservedObject var session: RunSession
    var showMetrics = true
    @State private var showRaw = false
    @State private var now = Date()
    private let clock = Timer.publish(every: 1, on: .main, in: .common).autoconnect()

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            header
            ForEach(session.progressOrder, id: \.self) { key in
                if let e = session.progress[key] {
                    ProgressRow(event: e)
                }
            }
            if !session.errors.isEmpty || isFailure {
                failureBox
            }
            if !session.checks.isEmpty {
                GroupBox("Checks") {
                    VStack(alignment: .leading, spacing: 4) {
                        ForEach(session.checks) { c in
                            CheckRow(name: c.name ?? "?", ok: c.ok ?? false, value: c.value?.display,
                                     stage: c.stage, needsHuman: c.needsHuman)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            if showMetrics && !session.metrics.isEmpty {
                GroupBox("Metrics") {
                    MetricGrid(rows: session.metrics.map { m in
                        (m.key, summarize(m.event.value))
                    })
                }
            }
            DisclosureGroup("Events (\(session.events.count))", isExpanded: $showRaw) {
                RawEventList(events: session.events)
                    .frame(height: 220)
            }
        }
        .onReceive(clock) { now = $0 }
    }

    private var isFailure: Bool {
        switch session.state {
        case .finished(let code): return code != 0
        case .failedToStart: return true
        default: return false
        }
    }

    @ViewBuilder private var header: some View {
        HStack(spacing: 8) {
            switch session.state {
            case .idle:
                Image(systemName: "circle.dotted").foregroundStyle(.secondary)
                Text("Not started")
            case .running:
                ProgressView().controlSize(.small)
                Text(session.currentStep.map { "\(session.title): \($0)" } ?? session.title)
            case .finished(let code):
                Image(systemName: code == 0 ? "checkmark.circle.fill" : "xmark.octagon.fill")
                    .foregroundStyle(code == 0 ? .green : .red)
                Text(code == 0 ? "\(session.title) finished" : "\(session.title) failed (exit \(code))")
            case .failedToStart:
                Image(systemName: "xmark.octagon.fill").foregroundStyle(.red)
                Text("\(session.title) could not start")
            }
            Spacer()
            if session.elapsed != nil {
                Text(Format.duration(session.isRunning ? now.timeIntervalSince(session.startedAt ?? now) : session.elapsed))
                    .monospacedDigit().foregroundStyle(.secondary)
            }
            if session.isRunning {
                Button("Stop") { session.cancel() }
                    .help("Interrupt (SIGINT): the stage is marked failed and the project lock released")
            }
            Button {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(session.commandLine, forType: .string)
            } label: { Image(systemName: "doc.on.doc") }
                .buttonStyle(.borderless)
                .help("Copy the command: \(session.commandLine)")
        }
        .font(.headline)
    }

    @ViewBuilder private var failureBox: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 6) {
                if case .failedToStart(let why) = session.state {
                    Text(why).textSelection(.enabled)
                }
                ForEach(session.errors) { e in
                    Text(e.message ?? "error").bold().textSelection(.enabled)
                    if let h = e.hint {
                        Text(h).font(.callout).foregroundStyle(.secondary).textSelection(.enabled)
                    }
                }
                if isFailure && !session.stderrTail.isEmpty {
                    ScrollView {
                        Text(session.stderrTail.suffix(4000))
                            .font(.system(.caption, design: .monospaced))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
                    }
                    .frame(maxHeight: 160)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        } label: {
            Label("Error", systemImage: "exclamationmark.triangle.fill").foregroundStyle(.red)
        }
    }

    private func summarize(_ v: JSONValue?) -> String {
        guard let v = v else { return "—" }
        if let a = v.array, a.count > 12 { return "\(a.count) entries" }
        let s = v.display
        return s.count > 160 ? String(s.prefix(160)) + "…" : s
    }
}

struct ProgressRow: View {
    let event: HSEvent

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack {
                Text(event.step.map { "\(event.stage) · \($0)" } ?? event.stage)
                    .font(.subheadline.weight(.medium))
                Spacer()
                if let d = event.done, let t = event.total {
                    Text("\(JSONValue.number(d).display) / \(JSONValue.number(t).display)")
                        .monospacedDigit().foregroundStyle(.secondary)
                }
                if let eta = event.etaSeconds, (event.fraction ?? 0) < 1 {
                    Text("ETA \(Format.eta(eta))").monospacedDigit().foregroundStyle(.secondary)
                }
            }
            .font(.caption)
            if let f = event.fraction {
                ProgressView(value: f)
            } else {
                ProgressView().progressViewStyle(.linear)
            }
            if let d = event.detail {
                Text(d).font(.caption2).foregroundStyle(.secondary)
            }
        }
    }
}

struct CheckRow: View {
    let name: String
    let ok: Bool
    let value: String?
    var stage: String? = nil
    var needsHuman = false

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Image(systemName: ok ? "checkmark.circle.fill" : "xmark.circle.fill")
                .foregroundStyle(ok ? .green : .orange)
            Text(name).font(.system(.body, design: .monospaced))
            if needsHuman {
                Text("needs you").font(.caption2).padding(.horizontal, 4)
                    .background(Capsule().fill(.yellow.opacity(0.3)))
            }
            if let v = value {
                Text(v).foregroundStyle(.secondary).textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .font(.callout)
    }
}

struct MetricGrid: View {
    let rows: [(String, String)]

    var body: some View {
        Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 3) {
            ForEach(Array(rows.enumerated()), id: \.offset) { _, r in
                GridRow {
                    Text(r.0).font(.system(.callout, design: .monospaced)).foregroundStyle(.secondary)
                    Text(r.1).font(.callout).textSelection(.enabled)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct RawEventList: View {
    let events: [HSEvent]

    var body: some View {
        ScrollViewReader { proxy in
            List(events.suffix(500)) { e in
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text(e.kind).font(.caption.monospaced()).foregroundStyle(color(e)).frame(width: 60, alignment: .leading)
                    Text(e.stage).font(.caption.monospaced()).foregroundStyle(.secondary).frame(width: 60, alignment: .leading)
                    Text(JSONValue.object(e.extras).compactJSON).font(.caption.monospaced()).lineLimit(2)
                        .textSelection(.enabled)
                }
                .id(e.id)
            }
            .listStyle(.plain)
            .onChange(of: events.last?.id) { _, id in
                if let id = id { proxy.scrollTo(id, anchor: .bottom) }
            }
        }
    }

    private func color(_ e: HSEvent) -> Color {
        switch e.kind {
        case "error": return .red
        case "check": return e.ok == false ? .orange : .green
        case "done": return .blue
        default: return .primary
        }
    }
}
