import XCTest
@testable import HSCore

/// `hs solve --estimate` → the matcher control, and the one-line progress summary every live
/// run shows (strip, sidebar, window title, Dock).
final class EstimateTests: XCTestCase {
    private func event(_ line: String) throws -> HSEvent {
        try XCTUnwrap(HSEvent(line: line, id: 0))
    }

    /// The plan's shape: matchers keyed by name, seconds per phase, an unknown key alongside.
    func testParsesTheEstimateEvent() throws {
        let e = try event(#"""
        {"ev":"estimate","stage":"solve","captures":422,"calibration":"defaults","future_key":[1,2],
         "matchers":{"exhaustive":{"images":844,"pairs":355746,"seconds":{"features":420,"matching":16008,"mapping":300,"export":72}},
                     "sequential":{"images":844,"pairs":26000,"seconds":{"features":420,"matching":1170,"mapping":300,"export":72,"total":2100}}}}
        """#)
        let est = try XCTUnwrap(SolveEstimate(event: e))
        let ex = try XCTUnwrap(est.cost(.exhaustive))
        XCTAssertEqual(ex.images, 844)
        XCTAssertEqual(ex.pairs, 355746)
        XCTAssertEqual(ex.seconds["matching"], 16008)
        XCTAssertEqual(ex.total, 16800)                    // no total given: the sum of the phases
        XCTAssertEqual(est.cost(.sequential)?.total, 2100)  // a given total wins over the sum
        XCTAssertEqual(est.calibration, "defaults")
        // no "auto" key: the engine's rule, sequential above 60 captures
        XCTAssertEqual(est.autoMatcher, .sequential)
        XCTAssertEqual(est.cost(.auto), est.cost(.sequential))

        XCTAssertEqual(est.label(.exhaustive), "Exhaustive ≈ 4 h 40 min")
        XCTAssertEqual(est.label(.sequential), "Sequential ≈ 35 min")
        XCTAssertEqual(est.label(.auto), "Auto (sequential) ≈ 35 min")
        XCTAssertEqual(est.framesCaption, "844 images, sequential ≈ 35 min")
        XCTAssertEqual(est.detail, "844 images · 355,746 pairs exhaustive · 26,000 pairs sequential. "
                       + "Times are the engine's defaults until a solve has been timed on this Mac.")
    }

    /// Other layouts the engine might settle on: a list of entries, a bare total, `<phase>_s` keys,
    /// an explicit auto choice, a measured calibration.
    func testReadsAListOfMatchersAndAnExplicitAuto() throws {
        let e = try event(#"""
        {"ev":"estimate","stage":"solve","auto":"exhaustive","calibration":"measured on this Mac (3 runs)",
         "matchers":[{"matcher":"exhaustive","images":120,"pairs":7140,"seconds":400},
                     {"matcher":"sequential","images":120,"features_s":30,"matching_s":60,"mapping_s":20,"export_s":10}]}
        """#)
        let est = try XCTUnwrap(SolveEstimate(event: e))
        XCTAssertEqual(est.cost(.exhaustive)?.total, 400)
        XCTAssertEqual(est.cost(.sequential)?.total, 120)
        XCTAssertEqual(est.cost(.sequential)?.seconds.count, 4)
        XCTAssertEqual(est.autoMatcher, .exhaustive)
        XCTAssertEqual(est.label(.auto), "Auto (exhaustive) ≈ 7 min")
        XCTAssertEqual(est.calibrationNote, "Times measured on this Mac (3 runs).")
    }

    func testMatchersAtTheTopLevelAndNoWayToResolveAuto() throws {
        let est = try XCTUnwrap(SolveEstimate(event: try event(#"{"ev":"estimate","stage":"solve","sequential":{"pairs":10,"total_s":90}}"#)))
        XCTAssertEqual(est.cost(.sequential)?.total, 90)
        XCTAssertNil(est.cost(.exhaustive))
        XCTAssertNil(est.autoMatcher)
        XCTAssertNil(est.cost(.auto))
        XCTAssertEqual(est.label(.auto), "Auto")               // no estimate shown, not a wrong one
        XCTAssertEqual(est.label(.exhaustive), "Exhaustive")
        XCTAssertEqual(est.framesCaption, "sequential ≈ 2 min")
        XCTAssertNil(est.calibrationNote)
    }

    /// Anything else is "no estimate", never a crash or a zero.
    func testNothingUsableIsNoEstimate() throws {
        XCTAssertNil(SolveEstimate(event: try event(#"{"ev":"progress","stage":"solve","done":1,"total":2}"#)))
        XCTAssertNil(SolveEstimate(event: try event(#"{"ev":"estimate","stage":"solve"}"#)))
        XCTAssertNil(SolveEstimate(event: try event(#"{"ev":"estimate","stage":"solve","matchers":{"exhaustive":{},"sequential":"soon"}}"#)))
        XCTAssertNil(SolveEstimate(event: try event(#"{"ev":"estimate","stage":"solve","matchers":{"exhaustive":{"seconds":{"total":-5}}}}"#)))
        XCTAssertNil(SolveEstimate(json: .string("x")))
    }

    func testApproxFormat() {
        XCTAssertEqual(Format.approx(nil), "—")
        XCTAssertEqual(Format.approx(-1), "—")
        XCTAssertEqual(Format.approx(10), "< 1 min")
        XCTAssertEqual(Format.approx(89), "≈ 1 min")
        XCTAssertEqual(Format.approx(2100), "≈ 35 min")
        XCTAssertEqual(Format.approx(3599), "≈ 1 h")
        XCTAssertEqual(Format.approx(16800), "≈ 4 h 40 min")
        XCTAssertEqual(Format.approx(16740), "≈ 4 h 40 min")   // 4 h 39 min: past an hour, to 5 min
        XCTAssertEqual(Format.approx(7500), "≈ 2 h 5 min")
    }

    func testArguments() {
        XCTAssertEqual(SolveMatcher.auto.solveArguments(project: "/p"), ["solve", "-p", "/p", "--matcher", "auto"])
        XCTAssertEqual(SolveMatcher.sequential.solveArguments(project: "/p"), ["solve", "-p", "/p", "--matcher", "sequential"])
        XCTAssertEqual(SolveEstimate.arguments(project: "/p"), ["solve", "-p", "/p", "--estimate"])
        XCTAssertEqual(SolveMatcher.allCases.map(\.title), ["Exhaustive", "Sequential", "Auto"])
    }

    // MARK: the loader, against a stand-in for hs

    private final class Box: @unchecked Sendable { var value: SolveEstimate?; var done = false }

    private func fakeHS(_ body: String) throws -> EngineConfig {
        let dir = NSTemporaryDirectory() + "hs-est-" + UUID().uuidString
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        let hs = dir + "/hs"
        try ("#!/bin/sh\n" + body).write(toFile: hs, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: hs)
        return EngineConfig(repoRoot: dir, hsPath: hs)
    }

    private func load(_ config: EngineConfig) -> Box {
        let box = Box()
        let done = expectation(description: "estimate")
        Task {
            box.value = await SolveEstimate.load(config: config, project: "/proj")
            box.done = true
            done.fulfill()
        }
        wait(for: [done], timeout: 10)
        return box
    }

    func testLoaderReadsTheEventAmongOtherOutput() throws {
        let c = try fakeHS(#"""
        [ "$1 $2 $3 $4" = "solve -p /proj --estimate" ] || exit 2
        echo "not json"
        echo '{"ev":"start","stage":"solve"}'
        echo '{"ev":"estimate","stage":"solve","captures":10,"matchers":{"exhaustive":{"images":20,"pairs":190,"seconds":{"total":95}}}}'
        echo '{"ev":"done","stage":"solve","exit":0}'
        """#)
        let got = load(c)
        XCTAssertTrue(got.done)
        XCTAssertEqual(got.value?.cost(.exhaustive)?.pairs, 190)
        XCTAssertEqual(got.value?.autoMatcher, .exhaustive)       // 10 captures: auto stays exhaustive
        XCTAssertEqual(got.value?.label(.auto), "Auto (exhaustive) ≈ 2 min")
    }

    /// An engine from before --estimate: argparse refuses the flag, the page shows no times.
    func testAnEngineWithoutEstimateGivesNil() throws {
        let c = try fakeHS("echo 'hs: error: unrecognized arguments: --estimate' >&2\nexit 2\n")
        let got = load(c)
        XCTAssertTrue(got.done)
        XCTAssertNil(got.value)
    }
}

@MainActor
final class LiveProgressTests: XCTestCase {
    func testShortLineAndBadge() {
        let p = LiveProgress(stage: "solve", step: "matching", done: 150_123, total: 355_746, etaSeconds: 1080)
        XCTAssertEqual(p.percent, 42)
        XCTAssertEqual(p.short, "solve · matching 42% · 18 min")
        XCTAssertEqual(p.badge, "42%")
        XCTAssertTrue(p.line(now: Date()).hasPrefix("solve · matching · 150,123 / 355,746 · 42% · ETA 18 min · ~"))
    }

    func testFullBarDropsTheETAAndPercentRoundsDown() {
        XCTAssertEqual(LiveProgress(stage: "train", step: "train", done: 40_000, total: 40_000, etaSeconds: 5).short, "train 100%")
        XCTAssertEqual(LiveProgress(stage: "train", done: 39_999, total: 40_000).percent, 99)
        XCTAssertEqual(LiveProgress(stage: "solve", step: "matching", done: 1, total: 4, etaSeconds: 16_800).short,
                       "solve · matching 25% · 4 h 40 min")
    }

    func testNoTotalMeansNoFractionNoBadge() {
        let p = LiveProgress(stage: "solve", step: "mapping", done: 12)
        XCTAssertNil(p.fraction)
        XCTAssertNil(p.badge)
        XCTAssertEqual(p.short, "solve · mapping")
        XCTAssertEqual(p.line(), "solve · mapping · 12")
        XCTAssertEqual(LiveProgress(stage: "views").line(), "views")
        XCTAssertEqual(LiveProgress(stage: "solve", etaSeconds: -3).short, "solve")
    }

    func testSpan() {
        XCTAssertEqual(Format.span(30), "under a minute")
        XCTAssertEqual(Format.span(1080), "18 min")
        XCTAssertEqual(Format.span(16_800), "4 h 40 min")
        XCTAssertEqual(Format.span(nil), "—")
        XCTAssertTrue(Format.eta(1080).hasPrefix("18 min · ~"))
    }

    /// The session's summary follows the bar that moved last, and a `start` for a new step
    /// replaces a finished bar with the step's name until it reports.
    func testSessionSummaryFollowsTheLatestStep() {
        let s = RunSession(title: "Solve", config: EngineConfig(repoRoot: "/r"), arguments: ["solve", "-p", "/p"])
        XCTAssertEqual(s.liveProgress, LiveProgress(stage: "solve"))
        let lines = [
            #"{"ev":"start","stage":"solve","step":"sfm"}"#,
            #"{"ev":"progress","stage":"solve","step":"features","done":844,"total":844}"#,
            #"{"ev":"progress","stage":"solve","step":"matching","done":10,"total":100,"eta_s":60}"#,
        ]
        s.ingest(lines.enumerated().compactMap { HSEvent(line: $1, id: $0) })
        XCTAssertEqual(s.liveProgress.short, "solve · matching 10% · 1 min")
        s.ingest([HSEvent(line: #"{"ev":"start","stage":"solve","step":"export"}"#, id: 9)!])
        XCTAssertEqual(s.liveProgress, LiveProgress(stage: "solve", step: "export"))
        XCTAssertEqual(s.liveProgress.short, "solve · export")
    }
}
