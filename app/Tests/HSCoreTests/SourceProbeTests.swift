import XCTest
@testable import HSCore

/// New Project takes any source the engine does. `hs source PATH` says what a dropped file or
/// folder is (Fixtures/source_probe.jsonl is that command's real output for a one-camera clip, a
/// folder of photographs off a card, one folder per camera, a RED folder, a Hydrogen clip, a
/// path that is not there, and one clip of a RED take that has too few cameras); SourceProbe reads
/// it and builds the `hs ingest` command.
final class SourceProbeTests: XCTestCase {
    private func reports() throws -> [SourceProbe] {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "source_probe", withExtension: "jsonl", subdirectory: "Fixtures"))
        let lines = try String(contentsOf: url, encoding: .utf8).split(separator: "\n").map(String.init)
        return try lines.enumerated().map { i, line in
            let e = try XCTUnwrap(HSEvent(line: line, id: i))
            return try XCTUnwrap(SourceProbe.from(events: [e]), "line \(i)")
        }
    }

    func testReadsEveryKindTheEngineReports() throws {
        let r = try reports()
        XCTAssertEqual(r.map(\.kind), [SourceProbe.Kind.video, SourceProbe.Kind.stills, SourceProbe.Kind.stills,
                                       SourceProbe.Kind.r3d, SourceProbe.Kind.stereo, SourceProbe.Kind.unknown,
                                       SourceProbe.Kind.r3d])
        XCTAssertEqual(r.map(\.accepted), [true, true, true, true, true, false, false])
        XCTAssertEqual(r.map(\.title), ["Video from one camera", "Photographs", "Photographs", "RED camera array",
                                        "Hydrogen One 3D clip", "Not a source", "RED camera array"])

        let video = r[0]
        XCTAssertEqual(video.path, "/Users/x/Media/IMG_2525.mp4")
        XCTAssertEqual(video.name, "IMG_2525.mp4")
        XCTAssertEqual(video.summary, "320×180 · h264 · 30 fps · 60 frames · 2.0 s")
        XCTAssertEqual(video.stem, "IMG-2525")
        XCTAssertEqual(video.date, "2026-09-28")
        XCTAssertEqual(video.notes, ["Frames are picked from the clip in the project's Select step."])
        XCTAssertFalse(video.offersStillsKind)
        XCTAssertTrue(video.takes.isEmpty)

        let stills = r[1]
        XCTAssertEqual(stills.stillsKind, "mono")
        XCTAssertEqual(stills.stillsWhy, "one numbered sequence (IMG-0001 … IMG-0007)")
        XCTAssertFalse(stills.stillsPerCamera)
        XCTAssertTrue(stills.offersStillsKind)
        XCTAssertEqual(stills.detectedStillsTitle, "one camera")
        XCTAssertTrue(stills.notes.contains { $0.contains("IMG_0001 → IMG-0001") }, "the renaming is said before it happens")
        XCTAssertTrue(stills.notes.contains { $0.contains("HEIC") })

        let perCamera = r[2]
        XCTAssertEqual(perCamera.stillsKind, "array")
        XCTAssertTrue(perCamera.stillsPerCamera)
        XCTAssertFalse(perCamera.offersStillsKind, "one folder per camera is an array, not a choice")
        XCTAssertEqual(perCamera.detectedStillsTitle, "one per camera")

        let red = r[3]
        XCTAssertEqual(red.root, "/Users/x/Media/RED_Footage")
        XCTAssertEqual(red.take, "067")
        XCTAssertEqual(red.takes, [SourceProbe.Take(take: "067", cameras: ["GA", "GB", "HA"], date: "2026-04-03"),
                                   SourceProbe.Take(take: "068", cameras: ["GA", "GB"], date: "2026-04-03")])
        XCTAssertEqual(red.takes.map(\.usable), [true, false])
        XCTAssertEqual(red.takes.map(\.label), ["take 067 · 3 cameras", "take 068 · 2 cameras"])
        XCTAssertEqual(red.date, "2026-04-03", "the day the proposed take was shot")

        // one clip of take 068 was dropped: the array's folder is found, and that take is too small
        let clip = r[6]
        XCTAssertEqual(clip.path, "/Users/x/Media/RED_Footage")
        XCTAssertEqual(clip.root, "/Users/x/Media/RED_Footage")
        XCTAssertEqual(clip.take, "068")
        XCTAssertEqual(clip.problems, ["take 068 has 2 cameras here; an array needs three cameras or more; take 067 has 3"])
        XCTAssertTrue(clip.notes.contains("The array is read from /Users/x/Media/RED_Footage."))

        let none = r[5]
        XCTAssertEqual(none.problems, ["no such file or folder: /Users/x/Media/nothing-here"])
        XCTAssertNil(none.date)
        XCTAssertNil(none.stem)
    }

    func testIngestArguments() throws {
        let r = try reports()
        let p = "/Projects/2026-10-04_x"
        XCTAssertEqual(r[0].ingestArguments(project: p), ["ingest", "-p", p, "--clip", "/Users/x/Media/IMG_2525.mp4"])
        XCTAssertEqual(r[4].ingestArguments(project: p), ["ingest", "-p", p, "--clip", "/Users/x/Media/VID_20260915_145235_2x1.h4v"])
        XCTAssertEqual(r[5].ingestArguments(project: p), [], "nothing to ingest: nothing to run")

        // photographs: the kind is the engine's unless the user says otherwise
        XCTAssertEqual(r[1].ingestArguments(project: p), ["ingest", "-p", p, "--frames", "/Users/x/Media/helmet stills"])
        var o = SourceProbe.Options()
        o.stillsKind = SourceProbe.StillsKind.array
        XCTAssertEqual(r[1].ingestArguments(project: p, options: o),
                       ["ingest", "-p", p, "--frames", "/Users/x/Media/helmet stills", "--kind", "array"])
        o.stillsKind = SourceProbe.StillsKind.mono
        XCTAssertEqual(r[2].ingestArguments(project: p, options: o), ["ingest", "-p", p, "--frames", "/Users/x/Media/array_frames"],
                       "one folder per camera cannot be called one camera: the engine would refuse --kind mono")

        // RED: the proposed take, a chosen take, a smaller frame
        XCTAssertEqual(r[3].ingestArguments(project: p), ["ingest", "-p", p, "--r3d", "/Users/x/Media/RED_Footage", "--take", "067"])
        var red = SourceProbe.Options()
        red.res = 2
        XCTAssertEqual(r[3].ingestArguments(project: p, options: red),
                       ["ingest", "-p", p, "--r3d", "/Users/x/Media/RED_Footage", "--take", "067", "--res", "2"])
        red.take = "068"
        XCTAssertEqual(r[3].ingestArguments(project: p, options: red), [], "two cameras are not an array")
        XCTAssertEqual(r[3].blocker(red), "Take 068 has 2 cameras; an array needs at least 3.")
        red.take = "099"
        XCTAssertEqual(r[3].blocker(red), "Choose a take.")
        XCTAssertNil(r[3].blocker())

        // the dropped clip's own take cannot be used; the other take of the same folder can
        XCTAssertEqual(r[6].blocker(), "Take 068 has 2 cameras; an array needs at least 3.")
        XCTAssertEqual(r[6].ingestArguments(project: p), [])
        var other = SourceProbe.Options()
        other.take = "067"
        XCTAssertNil(r[6].blocker(other), "the engine's refusal spoke for take 068 only")
        XCTAssertEqual(r[6].ingestArguments(project: p, options: other),
                       ["ingest", "-p", p, "--r3d", "/Users/x/Media/RED_Footage", "--take", "067"])
        XCTAssertEqual(SourceProbe.arguments(path: "/Users/x/My Media/a b.mov"), ["source", "/Users/x/My Media/a b.mov"])
    }

    func testBlockerSaysWhyInTheEnginesWords() throws {
        let r = try reports()
        XCTAssertNil(r[0].blocker())
        XCTAssertEqual(r[5].blocker(), "no such file or folder: /Users/x/Media/nothing-here")
        let pq = SourceProbe(path: "/a/IMG_1.MOV", name: "IMG_1.MOV", kind: SourceProbe.Kind.video, title: "Video from one camera",
                             summary: "3840×2160 · hevc", accepted: false, problems: ["this clip is PQ HDR"])
        XCTAssertEqual(pq.blocker(), "this clip is PQ HDR")
        XCTAssertEqual(pq.ingestArguments(project: "/p"), [])
        let silent = SourceProbe(path: "/a/x.mov", name: "x.mov", kind: SourceProbe.Kind.video, title: "t", summary: "s", accepted: false)
        XCTAssertEqual(silent.blocker(), "The engine would not take this.")
        // a RED folder whose problem is not about the take (REDline missing) blocks every take
        let noRedline = SourceProbe(path: "/r", name: "r", kind: SourceProbe.Kind.r3d, title: "RED camera array", summary: "1 take",
                                    accepted: false, problems: ["no working REDline found"], root: "/r",
                                    takes: [SourceProbe.Take(take: "067", cameras: ["GA", "GB", "HA"])], take: "067")
        XCTAssertEqual(noRedline.blocker(), "no working REDline found")
        // …but "no take has three cameras" speaks for the proposed take only
        let small = SourceProbe(path: "/r", name: "r", kind: SourceProbe.Kind.r3d, title: "RED camera array", summary: "2 takes",
                                accepted: false, problems: ["no take has three cameras or more (an array needs at least three)"],
                                root: "/r", takes: [SourceProbe.Take(take: "067", cameras: ["GA", "GB"])], take: nil)
        XCTAssertEqual(small.blocker(), "Choose a take.")
        // a camera with two clips of a take spoils that take whatever its size
        let twice = SourceProbe.Take(take: "067", cameras: ["GA", "GB", "HA"], problem: "two clips for camera GA take 067")
        XCTAssertFalse(twice.usable)
        XCTAssertEqual(twice.label, "take 067 · 3 cameras · cannot be used")
        let spoiled = SourceProbe(path: "/r", name: "r", kind: SourceProbe.Kind.r3d, title: "RED camera array", summary: "1 take",
                                  accepted: false, root: "/r", takes: [twice], take: "067")
        XCTAssertEqual(spoiled.blocker(), "two clips for camera GA take 067")
        // photographs the engine could not read as a set: nothing to choose between
        let unread = SourceProbe(path: "/d", name: "d", kind: SourceProbe.Kind.stills, title: "Photographs", summary: "photographs",
                                 accepted: false, problems: ["camera folder '100APPLE' holds 3 frames"])
        XCTAssertFalse(unread.offersStillsKind)
    }

    func testAReportFromALaterEngineOrABrokenOneDoesNotCrash() throws {
        let e = try XCTUnwrap(HSEvent(line: #"{"ev":"metric","stage":"source","name":"source","value":{"path":"/a/b.xyz","kind":"hologram","a_new_key":[1,2]}}"#, id: 0))
        let p = try XCTUnwrap(SourceProbe.from(events: [e]))
        XCTAssertEqual(p.kind, SourceProbe.Kind.unknown)
        XCTAssertEqual(p.name, "b.xyz")
        XCTAssertEqual(p.title, "unknown")
        XCTAssertFalse(p.accepted)
        XCTAssertEqual(p.ingestArguments(project: "/p"), [])
        let noPath = try XCTUnwrap(HSEvent(line: #"{"ev":"metric","stage":"source","name":"source","value":{"kind":"video"}}"#, id: 1))
        XCTAssertNil(SourceProbe.from(events: [noPath]))
        let other = try XCTUnwrap(HSEvent(line: #"{"ev":"metric","stage":"phone","name":"source","value":{"path":"/a"}}"#, id: 2))
        XCTAssertNil(SourceProbe.from(events: [other]), "another stage's metric is not a report")
        XCTAssertNil(SourceProbe.from(events: []))
    }

    func testProjectNames() throws {
        let r = try reports()
        // everything but a Hydrogen clip: the day it was recorded, then its own name or the label
        XCTAssertEqual(r[0].projectName(label: ""), "2026-09-28_IMG-2525")
        XCTAssertEqual(r[0].projectName(label: "Stormtrooper iPhone"), "2026-09-28_Stormtrooper-iPhone")
        XCTAssertEqual(r[1].projectName(label: ""), "2026-09-28_helmet-stills")
        XCTAssertEqual(r[3].projectName(label: ""), "2026-04-03_array067")
        var o = SourceProbe.Options()
        o.take = "068"
        XCTAssertEqual(r[3].projectName(label: "", options: o), "2026-04-03_array068", "the folder follows the take")
        XCTAssertEqual(r[3].projectName(label: "sphere", options: o), "2026-04-03_sphere")
        // a Hydrogen clip keeps the rule it had: date and time are in its name
        XCTAssertEqual(r[4].projectName(label: ""), "2026-09-15_145235")
        XCTAssertEqual(r[4].projectName(label: "head"), "2026-09-15_head")

        XCTAssertEqual(ProjectStore.suggestedName(date: "2026-09-28", stem: "IMG-2525"), "2026-09-28_IMG-2525")
        XCTAssertEqual(ProjectStore.suggestedName(date: nil, stem: "IMG-2525", label: " a/b "), "a-b")
        XCTAssertEqual(ProjectStore.suggestedName(date: "yesterday", stem: "x"), "x", "not a date: left out")
        XCTAssertEqual(ProjectStore.suggestedName(date: "2026-09-28", stem: "///"), "2026-09-28_project")
    }

    func testThePageSaysWhatItTakes() {
        XCTAssertEqual(SourceProbe.acceptedMedia.map(\.name),
                       ["RED Hydrogen One 3D clip", "Video from one camera", "Photographs", "RED camera array"])
        XCTAssertEqual(Set(SourceProbe.acceptedMedia.map(\.id)).count, 4)
        XCTAssertTrue(SourceProbe.notAccepted.contains("PQ"))
    }

    // MARK: a one-camera clip inside a project

    func testOnlyFramesGivenAsFramesHaveNothingToSelect() throws {
        func manifest(_ source: String) throws -> Manifest {
            try XCTUnwrap(Manifest(data: Data(#"{"name":"a","stages":{},"source":\#(source)}"#.utf8)))
        }
        let hydrogen = try manifest(#"{"clip":"source/VID_20260915_145235_2x1.h4v","md5":"a"}"#)
        XCTAssertFalse(hydrogen.isArray)
        XCTAssertFalse(hydrogen.framesOnly)
        let orbit = try manifest(#"{"kind":"mono","clip":"source/IMG_2525.MOV","md5":"a","probe":{"width":2160,"height":3840}}"#)
        XCTAssertTrue(orbit.isArray, "it solves like every one-camera set")
        XCTAssertFalse(orbit.framesOnly, "…but its frames are picked in Select")
        XCTAssertEqual(orbit.clipName, "IMG_2525.MOV")
        let picks = try manifest(#"{"kind":"mono","frames":"source/frames","cameras":[{"camera":"sel000-00000"}]}"#)
        XCTAssertTrue(picks.framesOnly)
        let array = try manifest(#"{"kind":"array","cameras":[{"camera":"GA"},{"camera":"GB"}]}"#)
        XCTAssertTrue(array.framesOnly)
    }

    func testAOneCameraQualityReportDecodesAndHasNoRightEye() throws {
        // select/quality.json as hs select wrote it for a one-camera clip (engine output, not hand-made)
        let url = try XCTUnwrap(Bundle.module.url(forResource: "quality_mono", withExtension: "json", subdirectory: "Fixtures"))
        let q = try JSONDecoder().decode(FrameQuality.self, from: Data(contentsOf: url))
        XCTAssertEqual(q.eyes, 1)
        XCTAssertTrue(q.measuredEyes, "the frames were measured")
        XCTAssertFalse(q.hasRightEye, "…and there is no right eye to show")
        XCTAssertEqual(q.frames.count, 9)
        XCTAssertEqual(q.frames[0].file, "sel000-00000.jpg")
        XCTAssertEqual(q.frames[0].thumb, "thumbs/sel000-00000.jpg")
        XCTAssertNil(q.frames[0].sharpR)
        XCTAssertNil(q.frames[0].eyeEV)
        XCTAssertNotNil(q.frames[0].noise)
        // the reference is named after its file, and the pick is found from that name
        XCTAssertEqual(q.exposureReference?.cap, "sel005-00110")
        XCTAssertEqual(q.frame(cap: "sel005-00110")?.sel, 5)
        XCTAssertEqual(q.frame(cap: "cap003")?.sel, 3, "a Hydrogen capture name still works")
        XCTAssertNil(q.frame(cap: "selfie"))
    }
}
