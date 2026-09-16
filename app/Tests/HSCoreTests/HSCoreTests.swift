import XCTest
@testable import HSCore

final class EventTests: XCTestCase {
    func testParsesKnownKinds() throws {
        let e = try XCTUnwrap(HSEvent(line: #"{"ev":"progress","stage":"train","done":23970,"total":40000,"rate":9.98,"eta_s":1605,"detail":"139455 splats","step":"train"}"#, id: 0))
        XCTAssertEqual(e.kind, "progress")
        XCTAssertEqual(e.stage, "train")
        XCTAssertEqual(e.done, 23970)
        XCTAssertEqual(e.fraction!, 0.59925, accuracy: 1e-9)
        XCTAssertEqual(e.etaSeconds, 1605)
        XCTAssertEqual(e.detail, "139455 splats")

        let c = try XCTUnwrap(HSEvent(line: #"{"ev":"check","stage":"solve","name":"all_frames_registered","ok":true,"value":"84/84"}"#, id: 1))
        XCTAssertEqual(c.ok, true)
        XCTAssertEqual(c.value?.string, "84/84")
    }

    func testBoolsAndNumbersStayApart() throws {
        let e = try XCTUnwrap(HSEvent(line: #"{"ev":"metric","stage":"x","name":"n","value":[1,true,0,false,1.5]}"#, id: 0))
        XCTAssertEqual(e.value, .array([.number(1), .bool(true), .number(0), .bool(false), .number(1.5)]))
        XCTAssertEqual(JSONValue.number(1139613).display, "1,139,613")
        XCTAssertEqual(JSONValue.number(1.3827).display, "1.3827")
        XCTAssertEqual(JSONValue.number(84).display, "84")
    }

    func testRejectsNonEvents() {
        XCTAssertNil(HSEvent(line: "I20260915 19:09:11 pairing.cc:212] Processing block", id: 0))
        XCTAssertNil(HSEvent(line: #"{"stage":"x"}"#, id: 0))
        XCTAssertNil(HSEvent(line: "", id: 0))
    }

    func testLineSplitterHandlesPartialLines() {
        var s = LineSplitter()
        XCTAssertEqual(s.push(Data("{\"a\":1}\n{\"b\"".utf8)), ["{\"a\":1}"])
        XCTAssertEqual(s.push(Data(":2}\r\n\n".utf8)), ["{\"b\":2}", ""])
        XCTAssertEqual(s.push(Data("tail".utf8)), [])
        XCTAssertEqual(s.finish(), ["tail"])
        XCTAssertEqual(s.finish(), [])
    }

    func testFormat() {
        XCTAssertEqual(Format.duration(5068.4), "1:24:28")
        XCTAssertEqual(Format.duration(95), "1:35")
        XCTAssertEqual(Format.duration(nil), "—")
    }
}

final class ManifestTests: XCTestCase {
    func fixture(_ name: String) throws -> URL {
        try XCTUnwrap(Bundle.module.url(forResource: name, withExtension: nil, subdirectory: "Fixtures"))
    }

    func testReadsTheBodyManifest() throws {
        let m = try XCTUnwrap(Manifest(data: Data(contentsOf: fixture("manifest_body.json"))))
        XCTAssertEqual(m.name, "2026-09-15_body")
        XCTAssertEqual(m.clipName, "VID_20260915_145111_2x1.h4v")
        XCTAssertEqual(m.profileID, "h1-video-1920x1080-holocam1.18.2-v1")
        XCTAssertEqual(m.probe["nb_frames"]?.int, 1911)
        XCTAssertEqual(Array(m.stages.prefix(3)).map(\.name), ["ingest", "select", "solve"])
        let solve = try XCTUnwrap(m.stage("solve"))
        XCTAssertEqual(solve.status, .done)
        XCTAssertEqual(solve.metrics["num_points"]?.int, 115227)
        XCTAssertTrue(solve.checks.contains { $0.name == "all_frames_registered" && $0.ok })
        let train = try XCTUnwrap(m.stage("train"))
        XCTAssertEqual(train.metrics["final_splats"]?.int, 1139613)
        XCTAssertTrue(train.displayMetrics.contains { $0.0 == "growth_curve" && $0.1.hasSuffix("entries") })
        XCTAssertNotNil(solve.duration)
    }

    func testSuggestedNames() {
        XCTAssertEqual(ProjectStore.suggestedName(clip: "VID_20260915_145235_2x1.h4v"), "2026-09-15_145235")
        XCTAssertEqual(ProjectStore.suggestedName(clip: "/x/VID_20260915_145235_2x1.h4v", label: "Head take 2"), "2026-09-15_Head-take-2")
        XCTAssertEqual(ProjectStore.suggestedName(clip: "clip.mp4"), "clip")
    }

    func testPhoneListing() throws {
        let lines = [
            #"{"ev":"metric","stage":"phone","name":"devices","value":[{"serial":"9ea4304b","state":"device","model":"H1A1000"},{"serial":"x","state":"unauthorized"}]}"#,
            #"{"ev":"metric","stage":"phone","name":"clips","value":[{"name":"VID_20260915_145235_2x1.h4v","path":"/sdcard/DCIM/Camera/VID_20260915_145235_2x1.h4v","bytes":105108436,"mtime":"2026-09-15 14:53"}],"serial":"9ea4304b","model":"H1A1000"}"#,
        ]
        let evs = lines.enumerated().compactMap { HSEvent(line: $1, id: $0) }
        let d = PhoneListing.devices(from: evs)
        XCTAssertEqual(d.map(\.serial), ["9ea4304b", "x"])
        XCTAssertEqual(d[0].clips.first?.bytes, 105108436)
        XCTAssertFalse(d[1].ready)
    }

    func testEnvironmentPutsHomebrewFirstOnce() {
        let c = EngineConfig(repoRoot: "/r")
        let env = c.environment(base: ["PATH": "/usr/bin:/opt/homebrew/bin"])
        XCTAssertEqual(env["PATH"], "/r/.venv/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin")
        XCTAssertEqual(env["PYTHONUNBUFFERED"], "1")
        XCTAssertEqual(c.projectsRoot, "/r/Projects")
    }
}

@MainActor
final class RunnerTests: XCTestCase {
    /// A real child: a shell that prints events split across writes, junk, and exits 3.
    func testRunnerDeliversEventsInOrderThenExit() throws {
        let script = """
        printf '{"ev":"start","stage":"t"}\\n{"ev":"progress","stage":"t","do'
        sleep 0.2
        printf 'ne":1,"total":2}\\nnot json\\n'
        echo oops >&2
        i=0; while [ $i -lt 300 ]; do printf '{"ev":"metric","stage":"t","name":"m","value":%d}\\n' $i; i=$((i+1)); done
        exit 3
        """
        let r = ToolRunner(executable: "/bin/sh", arguments: ["-c", script], environment: [:])
        var got: [HSEvent] = []
        let done = expectation(description: "exit")
        var result: ToolRunner.Result?
        try r.start(onEvents: { got += $0 }, onExit: { result = $0; done.fulfill() })
        wait(for: [done], timeout: 10)
        XCTAssertEqual(result?.exitCode, 3)
        XCTAssertEqual(result?.stderrTail, "oops\n")
        XCTAssertEqual(result?.nonJSONLines, ["not json"])
        XCTAssertEqual(got.count, 302)
        XCTAssertEqual(got.map(\.id), Array(0..<302))
        XCTAssertEqual(got[1].fraction, 0.5)
        XCTAssertEqual(got.last?.value?.int, 299)
    }

    func testSessionTracksLatestProgressAndMetrics() throws {
        let s = RunSession(title: "t", config: EngineConfig(repoRoot: "/r"), arguments: [])
        let lines = try String(contentsOf: Bundle.module.url(forResource: "ingest_phone", withExtension: "jsonl", subdirectory: "Fixtures")!, encoding: .utf8)
            .split(separator: "\n").map(String.init)
        s.ingest(lines.enumerated().compactMap { HSEvent(line: $1, id: $0) })
        XCTAssertEqual(s.progressOrder.count, 1)
        let p = try XCTUnwrap(s.progress[.init(stage: "ingest", step: "pull")])
        XCTAssertEqual(p.fraction, 1)
        XCTAssertEqual(s.checks.map { $0.name ?? "" }, ["pull_matches_phone", "clip_is_2x1_video", "profile_matched"])
        XCTAssertEqual(s.metric("ingest", "nb_frames")?.int, 2071)
        XCTAssertEqual(s.metrics.filter { $0.key == "ingest.width" }.count, 1)
    }
}

final class GradeTests: XCTestCase {
    /// Reference values from engine/hs/stages/grade.py lut() — preview and export must agree.
    func testLutMatchesEngine() {
        let idx = [0, 1, 64, 128, 200, 255]
        XCTAssertEqual(idx.map { Int(GradeSettings.lut(lift: 0.03, gamma: 1.2, gain: 0.9)[$0]) }, [14, 15, 80, 135, 192, 234])
        XCTAssertEqual(idx.map { Int(GradeSettings.lut(lift: 0, gamma: 1, gain: 1)[$0]) }, [0, 1, 64, 128, 200, 255])
        XCTAssertEqual(idx.map { Int(GradeSettings.lut(lift: -0.05, gamma: 0.8, gain: 1.3)[$0]) }, [0, 0, 54, 142, 255, 255])
    }

    func testChannelsAndArguments() throws {
        var s = GradeSettings()
        s.lift = 0.02; s.liftRGB = [0.01, 0, -0.01]
        s.gain = 1.1; s.gainRGB = [1.05, 1, 0.9]
        XCTAssertEqual(s.channel(0).lift, 0.03, accuracy: 1e-12)
        XCTAssertEqual(s.channel(2).gain, 0.99, accuracy: 1e-12)
        let a = s.arguments(project: "/p", move: "arc")
        XCTAssertTrue(a.contains("--gain-rgb=1.0500,1.0000,0.9000"))
        XCTAssertTrue(a.contains("--lift=0.0200"))
        // the engine's saved file decodes, missing keys fall back
        let json = #"{"lift": 0.01, "gain_rgb": [1.02, 1, 0.98], "headroom_mm": 30, "move": "arc"}"#
        let d = try JSONDecoder().decode(GradeSettings.self, from: Data(json.utf8))
        XCTAssertEqual(d.lift, 0.01)
        XCTAssertEqual(d.gainRGB, [1.02, 1, 0.98])
        XCTAssertEqual(d.headroomMM, 30)
        XCTAssertEqual(d.gamma, 1)
    }

    func testApplyUsesTheTable() throws {
        let w = 256, h = 1
        var px = [UInt8](repeating: 255, count: w * 4)
        for i in 0..<w { px[i * 4 + 1] = UInt8(i); px[i * 4 + 2] = UInt8(i); px[i * 4 + 3] = UInt8(i) }
        let ctx = CGContext(data: &px, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                            space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Big.rawValue)!
        let img = try XCTUnwrap(ctx.makeImage())
        var s = GradeSettings()
        s.gain = 0.5
        let out = try XCTUnwrap(s.apply(to: img))
        let data = try XCTUnwrap(out.dataProvider?.data) as Data
        let r = GradeSettings.lut(lift: 0, gamma: 1, gain: 0.5)
        XCTAssertEqual(data[200 * 4 + 1], r[200])
        XCTAssertEqual(data[255 * 4 + 3], r[255])
    }
}
