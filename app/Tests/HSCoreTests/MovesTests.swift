import XCTest
@testable import HSCore

final class MovesTests: XCTestCase {
    /// The shape `hs movepreview` writes (trimmed from a real shot02 report), including the
    /// nulls a clamped cue leaves where JSON has no NaN.
    static let report = """
    {"frames": 382, "fps": 30.0, "subject_mm": [1, 2, 3], "path_length_mm": 304.7,
     "peak_speed_mm_s": 61.2, "mean_speed_mm_s": 24.0, "p90_speed_mm_s": 40.1, "spike": null,
     "distance_mm": [220.1, 315.4], "hull_mm": [2.68, 24.84, 17.2],
     "aim": {"view": "cap025_L", "depth_mm": 262.0, "w": 1913, "h": 1073, "u": 1280.2, "v": 567.1, "inside": true},
     "cues": [{"seg": 0, "cue": "boom", "want": 10.0, "got": 10.0, "secs": 3.2, "clamped": false, "peak": 12.0,
               "unit": "deg", "at": {"az": -15.0, "el": 6.0}, "r0": 300.0, "r1": 290.0},
              {"seg": 1, "cue": "hold", "want": 0.4, "got": 0.4, "secs": 0.4, "clamped": false, "peak": 0.0,
               "unit": "s", "r0": null, "r1": null},
              {"seg": 2, "cue": "arc", "want": 90.0, "got": 46.0, "secs": 4.1, "clamped": true, "peak": 9.0,
               "unit": "deg", "at": {"az": 31.0, "el": 6.0}, "r0": 290.0, "r1": 240.0}],
     "start": {"az": -15, "el": 16, "off": 0}, "clamped": [],
     "track": [[-15.0, 16.0, 300.0, -1], [-15.0, 15.9, null, 0]],
     "captures": [[-30.1, 5.2, 310.0], [-28.0, 5.9, 305.5]],
     "capture_names": ["cap000", "cap001"], "view": "cap025_L",
     "schema": 1, "script": "/p/move/shot02.hsmove", "move_json": "/p/viewer/move_shot02.json",
     "rig_npz_md5": "16af69c545b9c0fc2871d48f230a8c01", "hull_limit_mm": 25.0}
    """

    func testDecodesThePreviewReport() throws {
        let r = try JSONDecoder().decode(MoveReport.self, from: Data(MovesTests.report.utf8))
        XCTAssertEqual(r.frames, 382)
        XCTAssertEqual(r.duration, 382.0 / 30.0, accuracy: 1e-9)
        XCTAssertEqual(r.hullMax, 24.84, accuracy: 1e-9)
        XCTAssertTrue(r.hullOK)
        XCTAssertNil(r.spike)
        XCTAssertEqual(r.cues.count, 3)
        XCTAssertEqual(r.clamped.map(\.cue), ["arc"])
        XCTAssertEqual(r.clamped.first?.at?.az, 31.0)
        XCTAssertNil(r.cues[1].at)
        XCTAssertNil(r.track[1][2])
        XCTAssertEqual(r.captureNames, ["cap000", "cap001"])
        XCTAssertEqual(r.aim.inside, true)
    }

    func testEngineErrorLineNumber() {
        XCTAssertEqual(MovePreview.lineNumber("line 3: unknown cue 'foo' — foo 3"), 3)
        XCTAssertEqual(MovePreview.lineNumber("line 12: arc needs left or right — arc 3"), 12)
        XCTAssertNil(MovePreview.lineNumber("no `start az .. el .. dolly ..` line"))
    }

    func testNamesAndPaths() {
        let s = MoveScript(project: "/p", name: "shot03")
        XCTAssertEqual(s.scriptPath, "/p/move/shot03.hsmove")
        XCTAssertEqual(s.builtPath, "/p/move/shot03.json")
        XCTAssertEqual(s.previewPath, "/p/viewer/move_shot03.json")
        XCTAssertEqual(s.buildArguments, ["move", "-p", "/p", "--script", "/p/move/shot03.hsmove", "--name", "shot03"])
        XCTAssertEqual(MovePath.label("/p/viewer/move_shot03.json"), "shot03 · preview")
        XCTAssertEqual(MovePath.label("/p/move/shot03.json"), "shot03")
        XCTAssertTrue(MoveScript.validName("shot_03-b"))
        XCTAssertFalse(MoveScript.validName("../x"))
        XCTAssertFalse(MoveScript.validName("a b"))
        XCTAssertFalse(MoveScript.validName(""))
    }

    func testRenderNameKeepsModelsApart() {
        let s = MoveScript(project: "/p", name: "shot03")
        let live = ViewerModelFile(project: "/p", name: "current", ply: "/p/train/exports/export_40000.ply", archive: nil, bytes: 1)
        let arch = ViewerModelFile(project: "/p", name: "holdout-g15", ply: "/p/archive/holdout-g15/export.ply", archive: "holdout-g15", bytes: 1)
        let pruned = ViewerModelFile(project: "/p", name: "prune/pruned.ply", ply: "/p/prune/pruned.ply", archive: nil, bytes: 1)
        XCTAssertEqual(s.renderName(model: live), "shot03")
        XCTAssertEqual(s.renderName(model: arch), "shot03_holdout-g15")
        XCTAssertEqual(s.renderName(model: pruned), "shot03_prune-pruned-ply")
        let a = s.renderArguments(model: arch, width: 2400, keepFrames: false, crop: true)
        XCTAssertEqual(Array(a.prefix(4)), ["render", "-p", "/p", "--move"])
        XCTAssertTrue(a.contains("--ply") && a.contains("/p/archive/holdout-g15/export.ply"))
        XCTAssertFalse(a.contains("--no-crop"))
        XCTAssertTrue(s.renderArguments(model: live, width: 1920, keepFrames: true, crop: false).contains("--no-crop"))
    }

    func testAimConfirmationIsTiedToTheBuiltMove() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("hsmove-\(UUID().uuidString)").path
        defer { try? FileManager.default.removeItem(atPath: dir) }
        let s = MoveScript(project: dir, name: "m")
        try s.write(MoveScript.template)
        try Data("{\"frames\": [1]}".utf8).write(to: URL(fileURLWithPath: s.builtPath))
        XCTAssertFalse(s.aimConfirmed)
        try s.confirmAim()
        XCTAssertTrue(s.aimConfirmed)
        // a rebuild that changes the path invalidates the confirmation
        try Data("{\"frames\": [2]}".utf8).write(to: URL(fileURLWithPath: s.builtPath))
        XCTAssertFalse(s.aimConfirmed)
        try s.confirmAim()
        s.revokeAim()
        XCTAssertFalse(s.aimConfirmed)
    }

    func testSettingTheStartLine() {
        let t = "# hsmove 1\n# note\nstart  az 0  el +6  dolly +0   # the old mark\nhold 0.5s\narc left 20\n"
        let line = MoveScriptText.startLine(az: -18.086, el: 14.74, dolly: -23.4)
        XCTAssertEqual(line, "start  az -18.1  el +14.7  dolly -23")
        let out = MoveScriptText.settingStart(t, to: line)
        XCTAssertEqual(out.components(separatedBy: "\n")[2], line + "   # the old mark")
        XCTAssertEqual(out.components(separatedBy: "\n").count, t.components(separatedBy: "\n").count)
        // no start line: goes in after the leading comments
        let noStart = MoveScriptText.settingStart("# a\n\narc left 5\n", to: line)
        XCTAssertEqual(noStart.components(separatedBy: "\n")[2], line)
        XCTAssertEqual(MoveScriptText.hull("start az 0 el 0\nhull 50   # wider\n"), 50)
        XCTAssertEqual(MoveScriptText.hull("hull 40mm"), 40)
        XCTAssertNil(MoveScriptText.hull("# hull 50\narc left 3"))
    }

    func testMapClickSnapsToTheNearestCapture() {
        let f = MoveFrame(captures: [[-170, 5, 600], [10, 20, 600], [175, 8, 600]],
                          captureNames: ["cap000", "cap001", "cap002"], rigNpzMd5: "x", hullDefaultMm: 25)
        XCTAssertEqual(f.nearestCapture(az: 12, el: 18)?.name, "cap001")
        // azimuth wraps: -178 is 3 deg from 175 and 8 from -170
        XCTAssertEqual(f.nearestCapture(az: -178, el: 8)?.name, "cap002")
    }
}
