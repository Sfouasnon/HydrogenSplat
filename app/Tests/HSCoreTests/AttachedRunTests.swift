import XCTest
@testable import HSCore

// Following a run from its events file: what the app does for an `hs` started in Terminal.

/// Two runs of `hs solve` in one events file: the first done, the second mid-matching, its last
/// line not written out yet.
private let twoSolveRuns = """
{"ev":"run","stage":"hs","run":1000,"at":"x","argv":["hs","solve","-p","old"],"t":0.0}
{"ev":"start","stage":"solve","step":"matching","t":0.1}
{"ev":"progress","stage":"solve","step":"matching","done":9,"total":10,"eta_s":1,"t":1.0}
{"ev":"done","stage":"solve","exit":0,"t":2.0}
{"ev":"run","stage":"hs","run":2000,"at":"x","argv":["hs","solve","-p","P"],"t":0.0}
{"ev":"start","stage":"solve","step":"matching","t":0.5}
{"ev":"progress","stage":"solve","step":"matching","done":1,"total":4,"eta_s":30,"t":1.5}
{"ev":"progress","stage":"solve","step":"match
"""

/// The rest of the second run's half-written line, one more progress, and its done.
private let secondRunEnd = """
ing","done":2,"total":4,"eta_s":20,"t":2.5}
{"ev":"progress","stage":"solve","step":"matching","done":3,"total":4,"eta_s":10,"t":3.5}
{"ev":"done","stage":"solve","exit":0,"t":4.0}

"""

private func eventsProject(_ log: String, stage: String = "solve") throws -> String {
    let p = NSTemporaryDirectory() + "hsattach-\(UUID().uuidString)"
    try FileManager.default.createDirectory(atPath: p + "/logs", withIntermediateDirectories: true)
    try log.write(toFile: EventsLog.path(project: p, stage: stage), atomically: true, encoding: .utf8)
    return p
}

private func appendLog(_ s: String, project: String, stage: String = "solve") throws {
    let h = try XCTUnwrap(FileHandle(forWritingAtPath: EventsLog.path(project: project, stage: stage)))
    _ = h.seekToEndOfFile()
    h.write(Data(s.utf8))
    try h.close()
}

final class EventsLogTailTests: XCTestCase {
    func testTailReadsTheLastRunThenOnlyWhatIsAppended() throws {
        let p = try eventsProject(twoSolveRuns)
        var t = EventsLogTail(project: p, stage: "solve")
        let first = t.read()
        XCTAssertEqual(first.map(\.kind), ["start", "progress"], "only the last run; the half line waits")
        XCTAssertEqual(t.runID, 2000)
        XCTAssertEqual(t.runStart, Date(timeIntervalSince1970: 2000))
        XCTAssertEqual(t.runArgv, ["hs", "solve", "-p", "P"])
        XCTAssertEqual(first[1].receivedAt.timeIntervalSince1970, 2001.5, accuracy: 1e-6, "run start + t")
        XCTAssertEqual(t.read().count, 0)

        try appendLog(secondRunEnd, project: p)
        let more = t.read()
        XCTAssertEqual(more.map(\.kind), ["progress", "progress", "done"])
        XCTAssertEqual(more.map(\.done), [2, 3, nil])
        XCTAssertEqual(Set(first.map(\.id) + more.map(\.id)).count, 5, "ids stay unique across reads")
        XCTAssertEqual(t.read().count, 0)
    }

    func testMissingFileReadsNothing() {
        var t = EventsLogTail(project: "/nonexistent-\(UUID().uuidString)", stage: "solve")
        XCTAssertEqual(t.read().count, 0)
        XCTAssertNil(t.runID)
    }
}

@MainActor
final class AttachedRunTests: XCTestCase {
    private let cfg = EngineConfig(repoRoot: "/r")
    private let matching = RunSession.ProgressKey(stage: "solve", step: "matching")

    /// This test process stands in for the Terminal `hs`: its pid is alive for the whole test.
    func testFollowsTheLastRunWithEtaAndEndsOnDone() throws {
        let p = try eventsProject(twoSolveRuns)
        let s = RunSession(attachingTo: p, stage: "solve", pid: getpid(), config: cfg)
        XCTAssertTrue(s.isAttached)
        XCTAssertEqual(s.stageName, "solve")
        s.attach(interval: 3600)   // no tick during the test: poll() below stands in for it
        XCTAssertTrue(s.isRunning)
        XCTAssertEqual(s.progressOrder, [matching], "nothing from the first run")
        let live = s.liveProgress
        XCTAssertEqual(live.stage, "solve")
        XCTAssertEqual(live.step, "matching")
        XCTAssertEqual(live.done, 1)
        XCTAssertEqual(live.total, 4)
        XCTAssertEqual(live.etaSeconds, 30)
        XCTAssertEqual(live.badge, "25%")
        XCTAssertEqual(s.startedAt, Date(timeIntervalSince1970: 2000))
        XCTAssertEqual(s.samples.first?.t ?? -1, 1.5, accuracy: 1e-6, "charted on the run's own clock")
        XCTAssertEqual(s.command, ["hs", "solve", "-p", "P"], "Copy gives the command that was typed")

        var finished = 0
        s.onFinish = { _ in finished += 1 }
        try appendLog(secondRunEnd, project: p)
        s.poll()
        XCTAssertEqual(s.progress[matching]?.done, 3)
        XCTAssertFalse(s.isRunning)
        XCTAssertEqual(s.state, .finished(exit: 0))
        XCTAssertTrue(s.succeeded)
        XCTAssertEqual(finished, 1)
        s.poll()
        XCTAssertEqual(finished, 1, "a finished session reads nothing more")
    }

    func testAttachEndsAtOnceWhenTheLastRunIsAlreadyDone() throws {
        let p = try eventsProject(twoSolveRuns + secondRunEnd)
        let s = RunSession(attachingTo: p, stage: "solve", pid: getpid(), config: cfg)
        s.attach(interval: 3600)
        XCTAssertFalse(s.isRunning)
        XCTAssertEqual(s.state, .finished(exit: 0))
    }

    func testEndsWhenThePidIsGoneWithoutDone() throws {
        let p = try eventsProject(twoSolveRuns)
        let child = Process()
        child.executableURL = URL(fileURLWithPath: "/usr/bin/true")
        try child.run()
        child.waitUntilExit()
        let dead = child.processIdentifier
        XCTAssertFalse(RunSession.pidAlive(dead))
        let s = RunSession(attachingTo: p, stage: "solve", pid: dead, config: cfg)
        s.attach(interval: 3600)
        XCTAssertFalse(s.isRunning)
        XCTAssertEqual(s.state, .finished(exit: -1))
        XCTAssertEqual(s.progress[matching]?.done, 1, "what it wrote is still shown")
        XCTAssertTrue(s.stderrTail.contains("without a done"))
    }

    func testStartNeverSpawnsForAnAttachedSession() throws {
        let p = try eventsProject(twoSolveRuns)
        let s = RunSession(attachingTo: p, stage: "solve", pid: getpid(), config: cfg)
        s.start()
        XCTAssertEqual(s.state, .idle)
    }
}
