import XCTest
@testable import HSCore

/// The Masks panel's review list: masks_review/review.json as the engine writes it (contract
/// version 1), and the `hs masks --decide` / `--check-only` commands the panel runs.
final class MaskReviewTests: XCTestCase {
    private func fixtureData() throws -> Data {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "mask_review", withExtension: "json", subdirectory: "Fixtures"))
        return try Data(contentsOf: url)
    }

    /// The fixture with one top-level key replaced, or removed when `value` is nil.
    private func fixtureData(setting key: String, to value: Any?) throws -> Data {
        let parsed = try JSONSerialization.jsonObject(with: fixtureData())
        var obj = try XCTUnwrap(parsed as? [String: Any])
        if let v = value {
            obj[key] = v
        } else {
            obj.removeValue(forKey: key)
        }
        return try JSONSerialization.data(withJSONObject: obj)
    }

    private func review(_ json: String, project: String = "/p") throws -> MaskReview {
        try XCTUnwrap(MaskReview(data: Data(json.utf8), project: project))
    }

    func testDecodesEveryField() throws {
        let r = try XCTUnwrap(MaskReview(data: fixtureData(), project: "/p"))
        XCTAssertEqual(r.version, 1)
        XCTAssertFalse(r.newerEngine)
        XCTAssertEqual(r.views, 243)
        XCTAssertNil(r.note)
        XCTAssertEqual(r.flagged.map(\.view), ["L/sel192-01823", "L/sel230-02123", "L/sel079-00632", "L/sel101-00911"],
                       "the engine's order is kept: worst first")
        XCTAssertEqual(r.flagged.map(\.id), r.flagged.map(\.view))
        XCTAssertEqual(r.summary, MaskReview.Summary(flagged: 4, undecided: 3, repair: 0, exclude: 1, keep: 0, repairable: 3, exact: 2))

        // a repair of kind "reselect"
        let a = r.flagged[0]
        XCTAssertEqual(a.reasons, ["piece_outside"])
        XCTAssertEqual(a.score, 0.2667, accuracy: 1e-9)
        XCTAssertEqual(a.why, "leaves out 26.7% of the subject that the other views agree on")
        XCTAssertEqual(a.agreement ?? 0, 0.931, accuracy: 1e-9)
        XCTAssertEqual(a.pieceShare ?? 0, 0.2667, accuracy: 1e-9)
        XCTAssertNil(a.fellBack)
        XCTAssertEqual(a.preview, "/p/masks_review/L/sel192-01823.jpg")
        let reselect = MaskReview.Repair(kind: "reselect", label: "Vision's pieces re-selected against the hull",
                                         approximate: false,
                                         preview: "/p/masks_review/L/sel192-01823_repair.jpg",
                                         mask: "/p/masks_review/L/sel192-01823_repaired.png")
        XCTAssertEqual(a.repair, reselect)
        XCTAssertEqual(a.repair?.display, "Vision's pieces re-selected against the hull")
        XCTAssertNil(a.decision)
        XCTAssertNil(a.decided)

        // a repair that rests on the hull is approximate; the unknown key inside the repair is ignored
        let b = r.flagged[1]
        XCTAssertEqual(b.reasons, ["holds_subject", "piece_outside"])
        XCTAssertEqual(b.agreement ?? 0, 0.841, accuracy: 1e-9)
        XCTAssertEqual(b.repair?.kind, "hull")
        XCTAssertEqual(b.repair?.approximate, true)
        XCTAssertEqual(b.repair?.display, "Outline taken from the hull the other views voted — approximate, look at the picture first")
        XCTAssertEqual(b.repair?.mask, "/p/masks_review/L/sel230-02123_repaired.png")

        // no repair, fell back; "piece_share" and "decided" are absent from this entry
        let c = r.flagged[2]
        XCTAssertEqual(c.reasons, ["fell_back"])
        XCTAssertEqual(c.score, 0.15, accuracy: 1e-9)
        XCTAssertNil(c.agreement)
        XCTAssertNil(c.pieceShare)
        XCTAssertEqual(c.fellBack, "no instance inside the region")
        XCTAssertEqual(c.preview, "/p/masks_review/L/sel079-00632.jpg")
        XCTAssertNil(c.repair)
        XCTAssertNil(c.decision)
        XCTAssertNil(c.decided)

        // already decided
        let d = r.flagged[3]
        XCTAssertEqual(d.repair?.kind, "fill")
        XCTAssertEqual(d.repair?.approximate, false)
        XCTAssertEqual(d.decision, MaskReview.Choice.exclude)
        XCTAssertEqual(d.decided, Date(timeIntervalSince1970: 1_791_148_805))     // 14:20:05 -0700
        XCTAssertEqual(d.decided, Manifest.date("2026-10-04T14:20:05-0700"))

        XCTAssertEqual(r.exactRepairs, 1, "undecided, and its repair needs no look; the decided one is not counted")
        XCTAssertTrue(r.canExcludeRest)
    }

    func testPathsAreMadeAbsoluteAgainstTheProject() throws {
        let r = try review("""
        {"version":1,"views":2,"flagged":[
          {"view":"L/a","preview":"masks_review/L/a.jpg",
           "repair":{"kind":"fill","label":"x","preview":"/elsewhere/a_repair.jpg","mask":"masks_review/L/a_repaired.png"}},
          {"view":"R/b","repair":{"kind":"hull"}}]}
        """, project: "/Projects/trooper")
        XCTAssertEqual(r.flagged[0].preview, "/Projects/trooper/masks_review/L/a.jpg")
        XCTAssertEqual(r.flagged[0].repair?.preview, "/elsewhere/a_repair.jpg", "an absolute path is kept")
        XCTAssertEqual(r.flagged[0].repair?.mask, "/Projects/trooper/masks_review/L/a_repaired.png")
        // every optional key missing: nothing crashes, nothing is invented
        let b = r.flagged[1]
        XCTAssertNil(b.preview)
        XCTAssertEqual(b.why, "")
        XCTAssertEqual(b.reasons, [])
        XCTAssertEqual(b.score, 0)
        XCTAssertNil(b.repair?.preview)
        XCTAssertNil(b.repair?.mask)
        XCTAssertEqual(b.repair?.label, "hull")
        XCTAssertEqual(b.repair?.approximate, true, "not said: look first")
        XCTAssertEqual(r.flagged[0].repair?.approximate, true, "the same for any kind")
        XCTAssertEqual(r.exactRepairs, 0)
        XCTAssertEqual(MaskReview.reportPath(project: "/Projects/trooper"), "/Projects/trooper/masks_review/review.json")
    }

    func testReadsTheReportFromTheProjectFolder() throws {
        let p = NSTemporaryDirectory() + "hsreview-\(UUID().uuidString)"
        defer { try? FileManager.default.removeItem(atPath: p) }
        try FileManager.default.createDirectory(atPath: p + "/masks_review", withIntermediateDirectories: true)
        XCTAssertNil(MaskReview.read(project: p), "no report: the masks were never checked")

        let report = URL(fileURLWithPath: MaskReview.reportPath(project: p))
        try fixtureData().write(to: report)
        let r = try XCTUnwrap(MaskReview.read(project: p))
        XCTAssertEqual(r.flagged.count, 4)
        XCTAssertEqual(r.flagged[0].preview, p + "/masks_review/L/sel192-01823.jpg")

        try Data("not json".utf8).write(to: report)
        XCTAssertNil(MaskReview.read(project: p))
        try Data("[1, 2]".utf8).write(to: report)
        XCTAssertNil(MaskReview.read(project: p), "valid JSON that is not an object")
    }

    func testSummaryIsCountedWhenTheReportHasNone() throws {
        let r = try XCTUnwrap(MaskReview(data: fixtureData(setting: "summary", to: nil), project: "/p"))
        XCTAssertEqual(r.summary, MaskReview.Summary(flagged: 4, undecided: 3, repair: 0, exclude: 1, keep: 0, repairable: 3, exact: 2))
        XCTAssertEqual(r.headline, "Review — 4 flagged of 243 views, 3 undecided")

        // the engine's summary wins over the count when it is there
        let s = try review(#"{"version":1,"views":243,"flagged":[{"view":"L/a"}],"summary":{"flagged":48,"undecided":12,"repair":20,"exclude":10,"keep":6,"repairable":31,"exact":9}}"#)
        XCTAssertEqual(s.summary, MaskReview.Summary(flagged: 48, undecided: 12, repair: 20, exclude: 10, keep: 6, repairable: 31, exact: 9))

        // a summary missing some keys: those are counted
        let part = try review(#"{"version":1,"views":9,"flagged":[{"view":"L/a","decision":"keep"},{"view":"L/b","repair":{"kind":"fill","label":"f"}}],"summary":{"flagged":2}}"#)
        XCTAssertEqual(part.summary, MaskReview.Summary(flagged: 2, undecided: 1, repair: 0, exclude: 0, keep: 1, repairable: 1, exact: 0))
    }

    func testANewerEngineIsRecognisedAndItsListIsNotRead() throws {
        let r = try XCTUnwrap(MaskReview(data: fixtureData(setting: "version", to: 2), project: "/p"))
        XCTAssertEqual(r.version, 2)
        XCTAssertTrue(r.newerEngine)
        XCTAssertTrue(r.flagged.isEmpty)
        XCTAssertEqual(r.views, 243)
        XCTAssertEqual(r.summary.undecided, 3, "the counts are still read, for the Train warning")
        XCTAssertEqual(r.headline, "This review was written by a newer engine (version 2) — update the app to see the list.")
        XCTAssertEqual(r.exactRepairs, 0)
        XCTAssertFalse(r.canExcludeRest)

        // no version at all reads as version 1
        let old = try review(#"{"views":3,"flagged":[]}"#)
        XCTAssertEqual(old.version, 1)
        XCTAssertFalse(old.newerEngine)
    }

    func testDecisionsThatAreNotDecisionsReadAsUndecided() throws {
        let r = try review(#"{"version":1,"views":5,"flagged":[{"view":"L/a","decision":"undo"},{"view":"L/b","decision":"later"},{"view":"L/c","decision":"repair"},{"view":"L/d","decision":"keep"},{"view":"L/a","decision":"exclude"},{"reasons":["fell_back"]}]}"#)
        XCTAssertEqual(r.flagged.map(\.view), ["L/a", "L/b", "L/c", "L/d"], "a repeated view and an entry without one are dropped")
        XCTAssertNil(r.flagged[0].decision)
        XCTAssertNil(r.flagged[1].decision)
        XCTAssertEqual(r.flagged[2].decision, MaskReview.Choice.repair)
        XCTAssertEqual(r.flagged[3].decision, MaskReview.Choice.keep)
        XCTAssertEqual(r.summary, MaskReview.Summary(flagged: 4, undecided: 2, repair: 1, exclude: 0, keep: 1, repairable: 0, exact: 0))
        XCTAssertEqual(r.exactRepairs, 0, "nothing undecided has a repair")
        XCTAssertTrue(r.canExcludeRest)
        XCTAssertEqual(MaskReview.Choice.allCases.map(\.rawValue), ["repair", "exclude", "keep", "undo"])
    }

    func testDecideAndCheckOnlyArguments() {
        let one: [(view: String, choice: MaskReview.Choice)] = [(view: "L/sel192-01823", choice: MaskReview.Choice.repair)]
        XCTAssertEqual(MaskReview.decideArguments(project: "/p", one),
                       ["masks", "-p", "/p", "--decide", "L/sel192-01823=repair"])
        let several: [(view: String, choice: MaskReview.Choice)] = [
            (view: "L/sel230-02123", choice: MaskReview.Choice.exclude),
            (view: "L/sel079-00632", choice: MaskReview.Choice.keep),
            (view: "L/sel192-01823", choice: MaskReview.Choice.undo),
        ]
        XCTAssertEqual(MaskReview.decideArguments(project: "/My Projects/trooper", several),
                       ["masks", "-p", "/My Projects/trooper", "--decide",
                        "L/sel230-02123=exclude,L/sel079-00632=keep,L/sel192-01823=undo"],
                       "one --decide, comma-joined")
        let repairAll: [(view: String, choice: MaskReview.Choice)] = [(view: MaskReview.repairableKey, choice: MaskReview.Choice.repair)]
        XCTAssertEqual(MaskReview.decideArguments(project: "/p", repairAll),
                       ["masks", "-p", "/p", "--decide", "@repairable=repair"])
        let repairExact: [(view: String, choice: MaskReview.Choice)] = [(view: MaskReview.exactKey, choice: MaskReview.Choice.repair)]
        XCTAssertEqual(MaskReview.decideArguments(project: "/p", repairExact),
                       ["masks", "-p", "/p", "--decide", "@exact=repair"])
        let excludeRest: [(view: String, choice: MaskReview.Choice)] = [(view: MaskReview.undecidedKey, choice: MaskReview.Choice.exclude)]
        XCTAssertEqual(MaskReview.decideArguments(project: "/p", excludeRest),
                       ["masks", "-p", "/p", "--decide", "@undecided=exclude"])
        let none: [(view: String, choice: MaskReview.Choice)] = []
        XCTAssertEqual(MaskReview.decideArguments(project: "/p", none), [], "nothing to decide: nothing to run")
        XCTAssertEqual(MaskReview.checkOnlyArguments(project: "/p"), ["masks", "-p", "/p", "--check-only"])
    }

    func testHeadline() throws {
        let r = try XCTUnwrap(MaskReview(data: fixtureData(), project: "/p"))
        XCTAssertEqual(r.headline, "Review — 4 flagged of 243 views, 3 undecided")
        let many = try review(#"{"version":1,"views":243,"flagged":[{"view":"L/a"}],"summary":{"flagged":48,"undecided":12,"repair":20,"exclude":10,"keep":6,"repairable":31}}"#)
        XCTAssertEqual(many.headline, "Review — 48 flagged of 243 views, 12 undecided")
        let clean = try review(#"{"version":1,"views":243,"note":null,"flagged":[],"summary":{"flagged":0,"undecided":0,"repair":0,"exclude":0,"keep":0,"repairable":0}}"#)
        XCTAssertEqual(clean.headline, "All 243 masks agree")
        let unknownCount = try review(#"{"version":1,"flagged":[]}"#)
        XCTAssertEqual(unknownCount.headline, "All masks agree")
        let noted = try review(#"{"version":1,"views":243,"note":" Too few sparse points on the subject to check the masks. ","flagged":[]}"#)
        XCTAssertEqual(noted.note, "Too few sparse points on the subject to check the masks.")
        XCTAssertEqual(noted.headline, "Too few sparse points on the subject to check the masks.")
        let blank = try review(#"{"version":1,"views":7,"note":"  ","flagged":[]}"#)
        XCTAssertNil(blank.note)
        XCTAssertEqual(blank.headline, "All 7 masks agree")
    }

    func testTrainWarning() throws {
        let never = "These masks were never checked — run Check masks under Masks first (hs train will refuse)."
        XCTAssertEqual(MaskReview.trainWarning(nil, masksExist: true), never)
        XCTAssertNil(MaskReview.trainWarning(nil, masksExist: false), "no masks: nothing to check")

        let r = try XCTUnwrap(MaskReview(data: fixtureData(), project: "/p"))
        XCTAssertNil(MaskReview.trainWarning(r, masksExist: false))
        XCTAssertEqual(MaskReview.trainWarning(r, masksExist: true),
                       "3 flagged masks have no decision — hs train will refuse to start. Decide them under Masks.")
        // views the run leaves out anyway need no decision
        XCTAssertEqual(MaskReview.trainWarning(r, masksExist: true, excluded: ["L/sel192-01823", "L/sel230-02123", "R/cap005"]),
                       "1 flagged mask has no decision — hs train will refuse to start. Decide it under Masks.")
        XCTAssertNil(MaskReview.trainWarning(r, masksExist: true, excluded: ["L/sel192-01823", "L/sel230-02123", "L/sel079-00632"]))
        // "Leave out views" takes the spellings hs train --exclude takes
        XCTAssertNil(MaskReview.trainWarning(r, masksExist: true, excluded: ["sel192-01823_L", "L/sel230-02123.jpg", "L/sel079-00632"]))
        XCTAssertEqual(MaskReview.viewKey("cap064_R"), "R/cap064")
        XCTAssertEqual(MaskReview.viewKey("GA"), "L/GA", "a bare camera id on an array")
        XCTAssertEqual(MaskReview.viewKey("L/cap064"), "L/cap064")

        let decided = try review(#"{"version":1,"views":3,"flagged":[{"view":"L/a","decision":"keep"},{"view":"L/b","decision":"exclude"}]}"#)
        XCTAssertNil(MaskReview.trainWarning(decided, masksExist: true))

        // a newer engine's list is not read; its count still is
        let newer = try XCTUnwrap(MaskReview(data: fixtureData(setting: "version", to: 2), project: "/p"))
        XCTAssertEqual(MaskReview.trainWarning(newer, masksExist: true),
                       "3 flagged masks have no decision — hs train will refuse to start. Decide them under Masks.")
    }
}
