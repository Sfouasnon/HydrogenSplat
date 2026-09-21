import XCTest
import simd
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

final class TrainingTests: XCTestCase {
    func testDefaultsMatchTheHoldoutBaseline() {
        var s = TrainSettings()
        s.archiveName = "holdout-base"
        s.subjectMM = 350
        s.layer = .full        // the 09-16 head baseline had no masks; TrainView forces .full then
        let steps = s.steps(project: "/p", captures: 160)
        XCTAssertEqual(steps.map(\.title), ["Train", "Archive holdout-base", "Score views (L)"])
        let t = steps[0].arguments
        XCTAssertEqual(Array(t.prefix(3)), ["train", "-p", "/p"])
        XCTAssertTrue(t.contains("--min-scale-factor=0.1"))
        let ex = try! XCTUnwrap(t.first { $0.hasPrefix("--exclude=") })
        let views = ex.dropFirst("--exclude=".count).split(separator: ",")
        XCTAssertEqual(views.count, 32)                       // 16 captures x 2 eyes, as run in Terminal
        XCTAssertEqual(views.first, "L/cap005")
        XCTAssertTrue(views.contains("R/cap155"))
        XCTAssertFalse(t.contains { $0.hasPrefix("--brush-args") })
        XCTAssertTrue(t.contains("--layer=full"))
        XCTAssertFalse(t.contains { $0.hasPrefix("--alpha-mode") })   // no masks, no alpha mode
        XCTAssertEqual(steps[2].arguments, ["views", "-p", "/p", "--captures",
                                            "5,15,25,35,45,55,65,75,85,95,105,115,125,135,145,155",
                                            "--subject-mm", "350", "--name", "views_holdout-base"])
    }

    func testExperimentsBecomeFlags() {
        var s = TrainSettings()
        s.holdoutEvery = 0
        s.growthStopIter = 15_000
        s.refineEvery = 200
        s.splitAtScreenSize = 0.25
        s.minScaleFactor = 0
        s.backgroundNoise = 0
        s.extraBrushArgs = "--opac-decay 0.006"
        s.excludeExtra = "L/cap064, R/cap069"
        s.layer = .full
        s.scoreBothEyes = true
        let steps = s.steps(project: "/p", captures: 84)
        let t = steps[0].arguments
        XCTAssertTrue(t.contains("--split-at-screen-size=0.25"))
        XCTAssertTrue(t.contains("--min-scale-factor=0"))
        XCTAssertTrue(t.contains("--brush-args=--background-noise-strength 0 --opac-decay 0.006"))
        XCTAssertTrue(t.contains("--exclude=L/cap064,R/cap069"))
        XCTAssertTrue(t.contains("--layer=full"))
        XCTAssertEqual(steps.map(\.title), ["Train", "Score views (L)", "Score views (R)"])
        XCTAssertEqual(steps[2].arguments.suffix(4), ["--eye", "R", "--name", "views_latest_R"])
        s.archiveName = "bad name"
        XCTAssertNotNil(s.archiveNameProblem)
    }

    func testLayerAndAlphaModeReachTheEngine() {
        var s = TrainSettings()
        s.holdoutEvery = 0
        // the default subject layer is pushed empty, and says so explicitly
        XCTAssertTrue(s.trainArguments(project: "/p", captures: 70).contains("--layer=subject"))
        XCTAssertTrue(s.trainArguments(project: "/p", captures: 70).contains("--alpha-mode=transparent"))

        s.alphaMode = .masked
        XCTAssertTrue(s.trainArguments(project: "/p", captures: 70).contains("--alpha-mode=masked"))
        s.alphaMode = .transparent

        // the alpha mode describes the subject layer only; it must not follow the others out
        s.layer = .background
        let bg = s.trainArguments(project: "/p", captures: 70)
        XCTAssertTrue(bg.contains("--layer=background"))
        XCTAssertFalse(bg.contains { $0.hasPrefix("--alpha-mode") })

        s.layer = .full
        let full = s.trainArguments(project: "/p", captures: 70)
        XCTAssertTrue(full.contains("--layer=full"))
        XCTAssertFalse(full.contains { $0.hasPrefix("--alpha-mode") })
    }

    func testALayerIsScoredOnlyWhereItExists() {
        var s = TrainSettings()
        s.scoreViews = true
        func views(_ l: Layer) -> [String] {
            s.layer = l
            return s.steps(project: "/p", captures: 70).first { $0.title.hasPrefix("Score views") }!.arguments
        }
        XCTAssertTrue(views(.subject).contains("--inside-masks"))
        XCTAssertTrue(views(.background).contains("--outside-masks"))
        let full = views(.full)
        XCTAssertFalse(full.contains("--inside-masks") || full.contains("--outside-masks"))
    }

    func testMaskArgumentsMatchTheStage() {
        var m = MaskSettings()
        XCTAssertEqual(m.arguments(project: "/p"),
                       ["masks", "-p", "/p", "--radius-scale", "1", "--margin-frac", "0.05",
                        "--min-opacity", "0.1", "--close-px", "25", "--preview", "6"])
        XCTAssertFalse(m.arguments(project: "/p").contains("--radius"))      // never an absolute radius
        m.source = .points
        m.radiusScale = 0.8
        m.keepLargest = false
        let a = m.arguments(project: "/p")
        XCTAssertEqual(a.suffix(2), ["--from-points", "--no-keep-largest"])
        XCTAssertTrue(a.contains("0.8"))
    }

    func testLogFollowsTheLastRun() {
        let log = """
        ### 2026-09-15 21:11:41  $ /x/brush old
        [2026-09-16T04:12:17Z INFO  brush_cli] Refine iter 131, 193271 splats.
        ### 2026-09-16 16:28:47  $ /x/brush new
        [2026-09-16T23:29:00Z INFO  brush_cli] Refine iter 131, 190000 splats.
        [2026-09-16T23:29:10Z INFO  brush_train::train] screen_size iter=260
        [2026-09-16T23:29:10Z INFO  brush_cli] Refine iter 261, 196881 splats.
        """
        let st = TrainLogStatus.parse(log)
        XCTAssertEqual(st.runStarted, "2026-09-16 16:28:47")
        XCTAssertEqual(st.iter, 261)
        XCTAssertEqual(st.splats, 196881)
        XCTAssertEqual(st.rate!, 13.0, accuracy: 1e-9)
        XCTAssertEqual(st.tail.count, 3)
        XCTAssertEqual(st.etaSeconds(total: 1561)!, 100.0, accuracy: 1e-9)
        XCTAssertNil(st.etaSeconds(total: 261))
    }

    func testRateSkipsSleep() {
        let log = """
        ### 2026-09-16 16:28:47  $ /x/brush new
        [2026-09-16T23:29:00Z INFO  brush_cli] Refine iter 130, 1 splats.
        [2026-09-16T23:29:20Z INFO  brush_cli] Refine iter 260, 1 splats.
        [2026-09-17T01:29:20Z INFO  brush_cli] Refine iter 390, 1 splats.
        [2026-09-17T01:29:40Z INFO  brush_cli] Refine iter 520, 1 splats.
        """
        let st = TrainLogStatus.parse(log)
        XCTAssertEqual(st.rate!, 6.5, accuracy: 1e-9)   // 260 iters over 40 awake s; the 2 h gap is dropped
    }

    func testLeadingNumber() {
        XCTAssertEqual(RunSession.leadingNumber("1019640 splats"), 1019640)
        XCTAssertNil(RunSession.leadingNumber("splats"))
        XCTAssertNil(RunSession.leadingNumber(nil))
    }
}

final class ArrayProjectTests: XCTestCase {
    private func coverage(_ body: String) -> String {
        let d = NSTemporaryDirectory() + "hs-arr-" + UUID().uuidString
        try! FileManager.default.createDirectory(atPath: d + "/solve", withIntermediateDirectories: true)
        try! body.write(toFile: d + "/solve/coverage.json", atomically: true, encoding: .utf8)
        return d
    }

    func testCaptureSetReadsCamerasAndStereoFlag() {
        let arr = coverage(#"{"n_captures": 3, "stereo": false, "captures": [{"name":"GA"},{"name":"GB"},{"name":"HA"}]}"#)
        let s = CaptureSet.read(project: arr)
        XCTAssertFalse(s.stereo)
        XCTAssertEqual(s.names, ["GA", "GB", "HA"])
        XCTAssertEqual(s.views, 3)
        XCTAssertEqual(s.noun, "camera")
        // a coverage.json written before arrays existed has no "stereo" key: still pairs
        let old = coverage(#"{"n_captures": 2, "captures": [{"name":"cap000"},{"name":"cap001"}]}"#)
        let o = CaptureSet.read(project: old)
        XCTAssertTrue(o.stereo)
        XCTAssertEqual(o.views, 4)
        XCTAssertEqual(o.noun, "capture")
        XCTAssertEqual(CaptureSet.read(project: "/nope").count, 0)
    }

    func testArrayHoldoutExcludesCamerasAndSkipsTheRightEye() {
        var s = TrainSettings()
        s.scoreBothEyes = true
        s.holdoutEvery = 2
        s.holdoutStart = 1
        let set = CaptureSet(names: ["GA", "GB", "GC", "GD"], stereo: false)
        XCTAssertEqual(s.excludedViews(in: set), ["L/GB", "L/GD"])
        let steps = s.steps(project: "/p", in: set)
        XCTAssertTrue(steps[0].arguments.contains("--exclude=L/GB,L/GD"))
        XCTAssertEqual(steps.map(\.title), ["Train", "Score views (L)"])   // no R pass on an array
        // the same settings on a stereo set still hold out both eyes
        let stereo = CaptureSet(count: 4)
        XCTAssertEqual(s.excludedViews(in: stereo), ["L/cap001", "R/cap001", "L/cap003", "R/cap003"])
        XCTAssertEqual(s.steps(project: "/p", in: stereo).map(\.title), ["Train", "Score views (L)", "Score views (R)"])
    }

    func testManifestReadsAnArraySource() {
        let m = Manifest(data: #"""
        {"name":"a","stages":{},"source":{"kind":"array","md5":"abc","original_path":"/x/RED",
         "cameras":[{"camera":"GA"},{"camera":"GB"}],"probe":{"width":3840,"height":2160,"nb_frames":2}}}
        """#.data(using: .utf8)!)
        XCTAssertTrue(m!.isArray)
        XCTAssertEqual(m!.cameras, ["GA", "GB"])
        XCTAssertNil(m!.clipName)
    }
}

@MainActor
final class ProjectStoreWatchTests: XCTestCase {
    /// A Terminal run writes manifest.json and deletes .hs.lock without telling the app;
    /// refreshIfChanged() is what turns a finished run from "running" into done.
    func testRefreshIfChangedPicksUpAManifestWrittenOutsideTheApp() throws {
        let root = NSTemporaryDirectory() + "hs-store-" + UUID().uuidString
        let dir = root + "/proj"
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        func write(_ status: String) throws {
            try #"{"name":"proj","stages":{"train":{"status":"\#(status)"}}}"#
                .write(toFile: dir + "/manifest.json", atomically: true, encoding: .utf8)
        }
        try write("running")
        try "{\"pid\": 999999}".write(toFile: dir + "/.hs.lock", atomically: true, encoding: .utf8)

        let store = ProjectStore(root: root)
        XCTAssertEqual(store.project(at: dir)?.manifest?.stage("train")?.status, .running)
        XCTAssertNotNil(store.project(at: dir)?.lock)

        store.refreshIfChanged()        // nothing moved: still the same snapshot
        XCTAssertEqual(store.project(at: dir)?.manifest?.stage("train")?.status, .running)

        try write("done")               // what `hs train` does at the end, from another process
        try FileManager.default.removeItem(atPath: dir + "/.hs.lock")
        store.refreshIfChanged()
        XCTAssertEqual(store.project(at: dir)?.manifest?.stage("train")?.status, .done)
        XCTAssertNil(store.project(at: dir)?.lock)
        try? FileManager.default.removeItem(atPath: root)
    }
}

// MARK: - Splat viewer

final class ViewerTests: XCTestCase {
    func cameras() throws -> CameraSet {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "cameras_fixture", withExtension: "json", subdirectory: "Fixtures"))
        return try CameraSet.load(path: url.path)
    }

    /// The fixture is real `hs cameras` output (engine/hs/cameras.py), not hand-written JSON.
    func testDecodesEngineOutput() throws {
        let c = try cameras()
        XCTAssertEqual(c.schema, 1)
        XCTAssertTrue(c.stereo)
        XCTAssertEqual(c.views.count, 6)
        XCTAssertEqual(c.captures, ["cap000", "cap001", "cap002"])
        XCTAssertEqual(c.rigNpzMD5, "06fcd8d17eb03f11bfe7ea2eeffb0a05")
        let v = try XCTUnwrap(c.view(capture: "cap001", eye: "R"))
        XCTAssertEqual(v.name, "cap001_R")
        XCTAssertEqual([v.w, v.h], [1909, 1071])                 // the right eye's own canvas
        XCTAssertEqual(simd_length(c.up), 1, accuracy: 1e-5)
        XCTAssertTrue(try XCTUnwrap(c.view(capture: "cap001", eye: "L")).label.hasPrefix("cap001 L · az "))
    }

    /// A world point must land where numpy says the capture's own pinhole puts it, after the
    /// OpenCV->Metal flip and the fit into a drawable of a different shape. Reference values
    /// computed in numpy from the same fixture: pixel (1030.2045, 488.0155) in cap001_L,
    /// drawable 2400x1200, near 0.02, far 200.
    func testCapturePoseProjectsLikeThePinhole() throws {
        let pose = try XCTUnwrap(try cameras().views[2].pose)
        XCTAssertEqual(pose.w, 1913)
        let P = ViewerMath.projection(pose: pose, drawableWidth: 2400, drawableHeight: 1200, near: 0.02, far: 200)
        let V = ViewerMath.viewMatrix(c2w: pose.c2w)
        let clip = P * V * SIMD4<Float>(0.03, -0.02, 0.01, 1)
        XCTAssertEqual(clip.x / clip.w, 0.06869016, accuracy: 2e-4)
        XCTAssertEqual(clip.y / clip.w, 0.09037191, accuracy: 2e-4)
        XCTAssertEqual(clip.z / clip.w, 0.96736098, accuracy: 1e-3)
        XCTAssertGreaterThan(clip.w, 0)                           // in front of the camera
    }

    func testCentredPinholeIsTheOrdinaryPerspective() {
        let fovy: Float = 65 * .pi / 180
        let fy = 600 / tan(fovy / 2)
        let pose = ViewerPose(c2w: matrix_identity_float4x4, w: 2400, h: 1200, fx: fy, fy: fy, cx: 1200, cy: 600)
        let a = ViewerMath.projection(pose: pose, drawableWidth: 2400, drawableHeight: 1200)
        let b = ViewerMath.perspective(fovy: fovy, aspect: 2)
        for c in 0..<4 { for r in 0..<4 { XCTAssertEqual(a[c][r], b[c][r], accuracy: 1e-5, "col \(c) row \(r)") } }
        XCTAssertEqual(pose.verticalFOV, fovy, accuracy: 1e-6)
    }

    /// Looking from a capture's position along its axis, with its own "up", is the capture.
    func testLookAtReproducesACapturePose() throws {
        let pose = try XCTUnwrap(try cameras().views[0].pose)
        let L = ViewerMath.lookAt(eye: pose.position, target: pose.position + pose.forward, up: -pose.down)
        let V = ViewerMath.viewMatrix(c2w: pose.c2w)
        for c in 0..<4 { for r in 0..<4 { XCTAssertEqual(L[c][r], V[c][r], accuracy: 1e-4, "col \(c) row \(r)") } }
    }

    func testOrbitKeepsDistanceAndLeavesAPoseWithoutJumping() throws {
        let set = try cameras()
        let pose = try XCTUnwrap(set.views[0].pose)
        var o = OrbitCamera(leaving: pose, subject: set.subject, up: set.up)
        XCTAssertEqual(simd_length(o.eye - pose.position), 0, accuracy: 1e-6)           // same place
        XCTAssertEqual(simd_dot(simd_normalize(o.target - o.eye), pose.forward), 1, accuracy: 1e-5)   // same direction
        XCTAssertEqual(o.distance, 0.603, accuracy: 0.01)                                 // the ring is 603 mm out
        let d = o.distance
        let e0 = asin(simd_dot(simd_normalize(o.eye - o.target), o.up))
        o.rotate(yaw: 0.7, pitch: 0.2)
        XCTAssertEqual(o.distance, d, accuracy: 1e-4)
        let e1 = asin(simd_dot(simd_normalize(o.eye - o.target), o.up))
        XCTAssertEqual(e1 - e0, 0.2, accuracy: 1e-3)                                      // pitch up raises the eye
        o.rotate(yaw: 0, pitch: 10)                                                       // and it cannot go over the top
        XCTAssertLessThan(asin(simd_dot(simd_normalize(o.eye - o.target), o.up)), .pi / 2)
        o.dolly(factor: 0.5)
        XCTAssertEqual(o.distance, d / 2, accuracy: 1e-4)
        let t = o.target
        o.pan(dx: 0.1, dy: 0)
        XCTAssertEqual(o.distance, d / 2, accuracy: 1e-4)
        XCTAssertGreaterThan(simd_length(o.target - t), 0)
    }

    func testMovePathDecodesTheEngineFormat() throws {
        let json = #"{"note":"x","width":1913,"height":1073,"K":[[1500,0,956.5],[0,1498,536.5],[0,0,1]],"fps":30,"frames":[{"c2w":[[1,0,0,0.1],[0,1,0,0.2],[0,0,1,-0.6],[0,0,0,1]]}]}"#
        let m = try JSONDecoder().decode(MovePath.self, from: Data(json.utf8))
        let p = try XCTUnwrap(m.pose(at: 0))
        XCTAssertEqual(p.position, SIMD3<Float>(0.1, 0.2, -0.6))
        XCTAssertEqual([p.fx, p.fy, p.cx, p.cy], [1500, 1498, 956.5, 536.5])
        XCTAssertNil(m.pose(at: 1))
    }

    func testModelListingPairsEachPlyWithItsCameras() throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("hsviewer-\(UUID().uuidString)").path
        defer { try? fm.removeItem(atPath: root) }
        for d in ["archive/base", "archive/no-rig", "train/exports", "prune"] {
            try fm.createDirectory(atPath: root + "/" + d, withIntermediateDirectories: true)
        }
        for f in ["archive/base/export_40000.ply", "archive/base/rig.npz", "archive/no-rig/export_40000.ply",
                  "train/exports/export_5000.ply", "train/exports/export_40000.ply", "prune/export_40000_pruned_r100.ply"] {
            fm.createFile(atPath: root + "/" + f, contents: Data("x".utf8))
        }
        let list = ViewerModelFile.list(project: root)
        XCTAssertEqual(list.map(\.name), ["base", "current", "prune/export_40000_pruned_r100.ply"])   // no rig, no entry
        XCTAssertTrue(list[1].ply.hasSuffix("export_40000.ply"))          // 40000 > 5000 as numbers ("5000" > "40000" as strings)
        XCTAssertEqual(list[0].camerasArguments, ["cameras", "-p", root, "--archive", "base"])
        XCTAssertTrue(list[0].camerasPath.hasSuffix("viewer/cameras_base.json"))
        XCTAssertEqual(list[1].camerasArguments, ["cameras", "-p", root])
        XCTAssertTrue(list[2].camerasPath.hasSuffix("viewer/cameras_current.json"))
    }

    /// MIT's one condition: the copyright and permission notices ship with the app.
    func testAcknowledgementsCarryTheRequiredNotices() {
        let names = Acknowledgements.bundled.map(\.name)
        XCTAssertEqual(names, ["MetalSplatter", "spz-swift"])
        for a in Acknowledgements.bundled {
            XCTAssertTrue(a.licenseText.contains("Permission is hereby granted, free of charge"), a.name)
            XCTAssertTrue(a.licenseText.contains("The above copyright notice and this permission notice shall be included"), a.name)
            XCTAssertTrue(a.licenseText.contains("THE SOFTWARE IS PROVIDED \"AS IS\""), a.name)
        }
        XCTAssertTrue(Acknowledgements.bundled[0].licenseText.contains("Copyright (c) 2026 Sean Cier"))
        XCTAssertTrue(Acknowledgements.bundled[1].licenseText.contains("Copyright (c) 2024 Niantic Labs"))
    }
}
