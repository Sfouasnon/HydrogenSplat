import Foundation

/// masks_review/review.json — what `hs masks` found when it checked the masks against each other,
/// and what has been decided about each flagged view. The engine is the source of truth: the app
/// reads this file, shows the pictures the engine wrote beside it, and runs `hs masks --decide`.
///
/// Read tolerantly, like the manifest: unknown keys are ignored, missing optional keys read as
/// nil, and a report from a newer engine (`version` above `supportedVersion`) is recognised but
/// its list is not read.
public struct MaskReview: Sendable, Equatable {
    /// The report version this reader understands.
    public static let supportedVersion = 1
    /// `--decide @undecided=…`: every flagged view without a decision.
    public static let undecidedKey = "@undecided"
    /// `--decide @repairable=…`: every undecided flagged view that has a repair.
    public static let repairableKey = "@repairable"
    /// `--decide @exact=…`: every undecided flagged view whose repair is not approximate.
    public static let exactKey = "@exact"

    /// What `hs masks --decide VIEW=…` takes. A decision stored in review.json is only ever
    /// repair, exclude or keep; undo is the command that clears one.
    public enum Choice: String, Sendable, CaseIterable {
        case repair, exclude, keep, undo

        /// The button's title.
        public var title: String {
            switch self {
            case .repair: return "Repair"
            case .exclude: return "Exclude"
            case .keep: return "Keep"
            case .undo: return "Undo"
            }
        }

        /// How a decided row reads.
        public var decidedTitle: String {
            switch self {
            case .repair: return "repaired"
            case .exclude: return "excluded from training"
            case .keep: return "kept as built"
            case .undo: return "undone"
            }
        }
    }

    /// The repaired mask the engine has ready for a flagged view.
    public struct Repair: Sendable, Equatable {
        /// "reselect", "fill" or "hull".
        public let kind: String
        /// One English phrase for the UI.
        public let label: String
        /// False for the two repairs that need no look: Vision's own object, and a hole closed
        /// inside the mask. True for every repair that rests on the hull the other views voted
        /// (it can be too large near the silhouette, so a person has to look at the picture).
        public let approximate: Bool
        /// Absolute path of the repair's preview picture.
        public let preview: String?
        /// Absolute path of the repaired mask itself.
        public let mask: String?

        public init(kind: String, label: String, approximate: Bool, preview: String?, mask: String?) {
            self.kind = kind
            self.label = label
            self.approximate = approximate
            self.preview = preview
            self.mask = mask
        }

        init?(_ v: JSONValue, project: String) {
            guard v.object != nil else { return nil }
            let kind = v["kind"]?.string ?? ""
            self.init(kind: kind,
                      label: v["label"]?.string ?? kind,
                      approximate: v["approximate"]?.bool ?? true,      // not said: look first
                      preview: MaskReview.absolute(v["preview"]?.string, project: project),
                      mask: MaskReview.absolute(v["mask"]?.string, project: project))
        }

        /// The label as the row shows it.
        public var display: String {
            approximate ? label + " — approximate, look at the picture first" : label
        }
    }

    /// One flagged view.
    public struct Flagged: Sendable, Equatable, Identifiable {
        public var id: String { view }
        /// The view key, `<eye>/<cap>`: "L/sel192-01823".
        public let view: String
        /// Any of "fell_back", "holds_subject", "piece_outside".
        public let reasons: [String]
        /// 0–1, larger is worse. Only for ordering and a percent readout.
        public let score: Double
        /// One English clause, shown as is.
        public let why: String
        public let agreement: Double?
        public let pieceShare: Double?
        public let fellBack: String?
        /// Absolute path of the preview picture.
        public let preview: String?
        public let repair: Repair?
        /// nil while undecided; never `.undo`.
        public let decision: Choice?
        public let decided: Date?

        public init(view: String, reasons: [String], score: Double, why: String, agreement: Double?,
                    pieceShare: Double?, fellBack: String?, preview: String?, repair: Repair?,
                    decision: Choice?, decided: Date?) {
            self.view = view
            self.reasons = reasons
            self.score = score
            self.why = why
            self.agreement = agreement
            self.pieceShare = pieceShare
            self.fellBack = fellBack
            self.preview = preview
            self.repair = repair
            self.decision = decision
            self.decided = decided
        }

        init?(_ v: JSONValue, project: String) {
            guard let view = v["view"]?.string, !view.isEmpty else { return nil }
            var decision: Choice? = nil
            if let raw = v["decision"]?.string, let c = Choice(rawValue: raw), c != Choice.undo {
                decision = c
            }
            var repair: Repair? = nil
            if let r = v["repair"] {
                repair = Repair(r, project: project)
            }
            self.init(view: view,
                      reasons: v["reasons"]?.array?.compactMap { $0.string } ?? [],
                      score: v["score"]?.double ?? 0,
                      why: v["why"]?.string ?? "",
                      agreement: v["agreement"]?.double,
                      pieceShare: v["piece_share"]?.double,
                      fellBack: v["fell_back"]?.string,
                      preview: MaskReview.absolute(v["preview"]?.string, project: project),
                      repair: repair,
                      decision: decision,
                      decided: Manifest.date(v["decided"]?.string))
        }
    }

    /// The counts the engine keeps beside the list.
    public struct Summary: Sendable, Equatable {
        public let flagged: Int
        public let undecided: Int
        public let repair: Int
        public let exclude: Int
        public let keep: Int
        /// Flagged views with a repair, decided or not.
        public let repairable: Int
        /// Of those, the ones whose repair is not approximate.
        public let exact: Int

        public init(flagged: Int, undecided: Int, repair: Int, exclude: Int, keep: Int, repairable: Int, exact: Int) {
            self.flagged = flagged
            self.undecided = undecided
            self.repair = repair
            self.exclude = exclude
            self.keep = keep
            self.repairable = repairable
            self.exact = exact
        }

        /// Counted from the list, for a report without a "summary".
        public init(counting list: [Flagged]) {
            self.init(flagged: list.count,
                      undecided: list.filter { $0.decision == nil }.count,
                      repair: list.filter { $0.decision == Choice.repair }.count,
                      exclude: list.filter { $0.decision == Choice.exclude }.count,
                      keep: list.filter { $0.decision == Choice.keep }.count,
                      repairable: list.filter { $0.repair != nil }.count,
                      exact: list.filter { $0.repair?.approximate == false }.count)
        }
    }

    public let version: Int
    /// How many views were checked.
    public let views: Int
    /// A sentence when the check could not run fully; then `flagged` may be empty.
    public let note: String?
    /// Worst first, in the engine's order. Empty when `newerEngine`.
    public let flagged: [Flagged]
    public let summary: Summary

    /// Written by an engine newer than this app reads: say so and show no list.
    public var newerEngine: Bool { version > MaskReview.supportedVersion }

    /// `project` is the project folder: paths in the report are relative to it.
    public init?(data: Data, project: String) {
        guard let v = JSONValue.parse(data), v.object != nil else { return nil }
        let version = v["version"]?.int ?? MaskReview.supportedVersion
        let trimmed = v["note"]?.string?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        var list: [Flagged] = []
        if version <= MaskReview.supportedVersion {
            var seen = Set<String>()
            for entry in v["flagged"]?.array ?? [] {
                guard let f = Flagged(entry, project: project), !seen.contains(f.view) else { continue }
                seen.insert(f.view)
                list.append(f)
            }
        }
        let counted = Summary(counting: list)
        var summary = counted
        if let s = v["summary"], s.object != nil {
            summary = Summary(flagged: s["flagged"]?.int ?? counted.flagged,
                              undecided: s["undecided"]?.int ?? counted.undecided,
                              repair: s["repair"]?.int ?? counted.repair,
                              exclude: s["exclude"]?.int ?? counted.exclude,
                              keep: s["keep"]?.int ?? counted.keep,
                              repairable: s["repairable"]?.int ?? counted.repairable,
                              exact: s["exact"]?.int ?? counted.exact)
        }
        self.version = version
        self.views = v["views"]?.int ?? 0
        self.note = trimmed.isEmpty ? nil : trimmed
        self.flagged = list
        self.summary = summary
    }

    public static func reportPath(project: String) -> String {
        (project as NSString).appendingPathComponent("masks_review/review.json")
    }

    /// nil when masks_review/review.json is absent or unreadable: the masks were never checked.
    public static func read(project: String) -> MaskReview? {
        guard let data = FileManager.default.contents(atPath: reportPath(project: project)) else { return nil }
        return MaskReview(data: data, project: project)
    }

    /// A path from the report, made absolute against the project folder.
    static func absolute(_ path: String?, project: String) -> String? {
        guard let p = path, !p.isEmpty else { return nil }
        return p.hasPrefix("/") ? p : (project as NSString).appendingPathComponent(p)
    }

    /// An `--exclude` token as the engine reads it (engine/hs/stages/train.py parse_exclude):
    /// "cap064_L" and "L/cap064.jpg" are L/cap064, a bare camera id is L/<id>.
    static func viewKey(_ token: String) -> String {
        var t = (token as NSString).deletingPathExtension
        if !t.contains("/"), t.hasSuffix("_L") || t.hasSuffix("_R") {
            t = String(t.suffix(1)) + "/" + String(t.dropLast(2))
        }
        return t.contains("/") ? t : "L/" + t
    }

    // MARK: what the panel shows

    /// The header line.
    public var headline: String {
        if newerEngine {
            return "This review was written by a newer engine (version \(version)) — update the app to see the list."
        }
        if summary.flagged == 0 {
            if let n = note { return n }
            return views > 0 ? "All \(views) masks agree" : "All masks agree"
        }
        return "Review — \(summary.flagged) flagged of \(views) views, \(summary.undecided) undecided"
    }

    /// How many undecided views have a repair that needs no look: what `@exact=repair` would take.
    public var exactRepairs: Int {
        flagged.filter { $0.decision == nil && $0.repair?.approximate == false }.count
    }

    /// "Exclude the rest" would do something: a view is undecided.
    public var canExcludeRest: Bool {
        flagged.contains { $0.decision == nil }
    }

    /// The Train panel's warning, or nil when `hs train` would start as far as the review goes.
    /// `review` is nil when review.json is missing; `excluded` are the views the run leaves out
    /// anyway (`--exclude`), which the engine does not ask a decision for.
    public static func trainWarning(_ review: MaskReview?, masksExist: Bool, excluded: [String] = []) -> String? {
        guard masksExist else { return nil }
        guard let r = review else {
            return "These masks were never checked — run Check masks under Masks first (hs train will refuse)."
        }
        let left = Set(excluded.map(MaskReview.viewKey))
        let n = r.newerEngine
            ? r.summary.undecided
            : r.flagged.filter { $0.decision == nil && !left.contains($0.view) }.count
        if n <= 0 { return nil }
        if n == 1 {
            return "1 flagged mask has no decision — hs train will refuse to start. Decide it under Masks."
        }
        return "\(n) flagged masks have no decision — hs train will refuse to start. Decide them under Masks."
    }

    // MARK: the commands

    /// `hs masks -p P --decide L/a=repair,L/b=exclude` — one `--decide`, comma-joined. A view is a
    /// view key, or `undecidedKey` / `repairableKey` / `exactKey`. Nothing to decide: nothing to run.
    public static func decideArguments(project: String, _ decisions: [(view: String, choice: Choice)]) -> [String] {
        guard !decisions.isEmpty else { return [] }
        let joined = decisions.map { $0.view + "=" + $0.choice.rawValue }.joined(separator: ",")
        return ["masks", "-p", project, "--decide", joined]
    }

    /// `hs masks -p P --check-only`: check the masks already on disk; nothing is rebuilt.
    public static func checkOnlyArguments(project: String) -> [String] {
        ["masks", "-p", project, "--check-only"]
    }
}
