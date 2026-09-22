import Foundation

/// Where a live run is, in one line: stage · step · done/total · ETA. The same reading drives the
/// progress strip, the sidebar row, the window title and the Dock badge, so they never disagree.
/// Built from the run's latest progress event; every field the engine left out is left out here.
public struct LiveProgress: Equatable, Sendable {
    public let stage: String
    public let step: String?
    public let done: Double?
    public let total: Double?
    public let etaSeconds: Double?
    /// The whole stage's remaining time, when the engine sends `stage_eta_s` (solve does).
    public let stageEtaSeconds: Double?

    public init(stage: String, step: String? = nil, done: Double? = nil, total: Double? = nil,
                etaSeconds: Double? = nil, stageEtaSeconds: Double? = nil) {
        self.stage = stage
        self.step = step
        self.done = done
        self.total = total
        self.etaSeconds = etaSeconds
        self.stageEtaSeconds = stageEtaSeconds
    }

    public init(event: HSEvent) {
        self.init(stage: event.stage, step: event.step, done: event.done, total: event.total,
                  etaSeconds: event.etaSeconds, stageEtaSeconds: event.stageEtaSeconds)
    }

    public var fraction: Double? {
        guard let d = done, let t = total, t > 0 else { return nil }
        return min(max(d / t, 0), 1)
    }

    /// Whole percent, rounded down: 100% only once it is actually done.
    public var percent: Int? { fraction.map { Int(($0 * 100).rounded(.down)) } }

    /// The ETA worth showing: present, sane, and the bar not already full.
    public var eta: Double? {
        guard let e = etaSeconds, e.isFinite, e >= 0, (fraction ?? 0) < 1 else { return nil }
        return e
    }

    /// The whole stage's remaining time worth showing: sent by the engine, sane, and more than
    /// the step's own ETA (otherwise the step ETA already says it).
    public var stageEta: Double? {
        guard let e = stageEtaSeconds, e.isFinite, e >= 0, (fraction ?? 0) < 1 else { return nil }
        if let step = eta, e <= step { return nil }
        return e
    }

    /// "solve · matching", or just "train" when the step is missing or repeats the stage.
    public var stageStep: String {
        if let s = step, !s.isEmpty, s != stage { return "\(stage) · \(s)" }
        return stage
    }

    /// "solve · matching 42% · 18 min · solve ≈ 3 h 50 min" — the sidebar row and the window
    /// title: the step's ETA, then the whole stage's when the engine can say and it is longer.
    public var short: String {
        var s = stageStep
        if let p = percent { s += " \(p)%" }
        if let e = eta { s += " · \(Format.span(e))" }
        if let w = stageEta { s += " · \(stage) ≈ \(Format.span(w))" }
        return s
    }

    /// "solve · matching · 150,123 / 355,746 · 42% · ETA 18 min · ~7:53 PM" — the progress strip.
    public func line(now: Date = Date()) -> String {
        var parts = [stage]
        if let s = step, !s.isEmpty, s != stage { parts.append(s) }
        if let d = done {
            let n = JSONValue.number(d).display
            parts.append(total.map { "\(n) / \(JSONValue.number($0).display)" } ?? n)
        }
        if let p = percent { parts.append("\(p)%") }
        if let e = eta { parts.append("ETA \(Format.eta(e, now: now))") }
        if let w = stageEta { parts.append("whole \(stage) ≈ \(Format.eta(w, now: now))") }
        return parts.joined(separator: " · ")
    }

    /// The Dock badge: "42%", or nil when there is no fraction to show.
    public var badge: String? { percent.map { "\($0)%" } }
}

extension RunSession {
    /// Where this run is now: its latest progress event, or — before the first one, and after a
    /// `start` has moved on to a step that has not reported yet — the stage and step alone.
    /// Callers show it only while `isRunning`.
    public var liveProgress: LiveProgress {
        if let k = lastProgressKey, let e = progress[k] { return LiveProgress(event: e) }
        return LiveProgress(stage: activeStage ?? stageName, step: currentStep)
    }
}
