import XCTest
import simd
@testable import HSCore

final class KeyframesTests: XCTestCase {
    /// A lens like cap001_L; subject at the origin, up = -y (OpenCV world as the solves come out).
    func move(_ keys: [KeyedMove.Key] = []) -> KeyedMove {
        KeyedMove(subject: .zero, up: SIMD3(0, -1, 0),
                  lens: .init(w: 1913, h: 1073, fx: 1501, fy: 1498, cx: 956.5, cy: 539.25, from: "cap001_L"),
                  rigNpzMD5: "x", keys: keys)
    }

    func testFramesLookAtTheAnchorLevel() {
        var m = move([.init(t: 0, eye: SIMD3(0, -0.2, -1), ease: true), .init(t: 2, eye: SIMD3(1, -0.2, 0), ease: true)])
        m.anchor = [0.1, 0.05, 0]
        XCTAssertEqual(m.frameCount, 61)
        for p in [m.position(at: 0), m.position(at: 0.7), m.position(at: 2)] {
            let c = m.cameraToWorld(eye: p)
            let fwd = SIMD3(c.columns.2.x, c.columns.2.y, c.columns.2.z)
            let right = SIMD3(c.columns.0.x, c.columns.0.y, c.columns.0.z)
            let toAnchor = simd_normalize(m.anchorPoint - p)
            XCTAssertEqual(simd_dot(fwd, toAnchor), 1, accuracy: 1e-9)          // aimed at the anchor
            XCTAssertEqual(simd_dot(right, m.upVector), 0, accuracy: 1e-9)       // level horizon
            XCTAssertEqual(simd_determinant(simd_double3x3(right, SIMD3(c.columns.1.x, c.columns.1.y, c.columns.1.z), fwd)),
                           1, accuracy: 1e-9)                                     // right-handed, y down
        }
    }

    func testSplinePassesThroughKeysAndEasesToRest() {
        let m = move([.init(t: 0, eye: SIMD3(0, 0, -1), ease: true),
                      .init(t: 1, eye: SIMD3(0.5, 0, -0.8), ease: false),
                      .init(t: 3, eye: SIMD3(1, 0, 0), ease: true)])
        XCTAssertLessThan(simd_length(m.position(at: 1) - SIMD3(0.5, 0, -0.8)), 1e-9)
        XCTAssertLessThan(simd_length(m.position(at: 3) - SIMD3(1, 0, 0)), 1e-9)
        // eased ends: the first and last frame steps are far shorter than the middle ones
        let ps = m.framePositions()
        let first = simd_length(ps[1] - ps[0]), mid = simd_length(ps[46] - ps[45]), last = simd_length(ps[90] - ps[89])
        XCTAssertLessThan(first, mid * 0.1)
        XCTAssertLessThan(last, mid * 0.1)
        // passing through key 1: no stop there
        let at1 = simd_length(ps[31] - ps[30])
        XCTAssertGreaterThan(at1, mid * 0.3)
    }

    func testHoldIsStill() {
        var m = move([.init(t: 0, eye: SIMD3(0, 0, -1), ease: true)])
        m.hold(seconds: 1)
        XCTAssertEqual(m.keys.count, 2)
        XCTAssertEqual(m.duration, 1, accuracy: 1e-12)
        for p in m.framePositions() { XCTAssertLessThan(simd_length(p - SIMD3(0, 0, -1)), 1e-12) }
    }

    func testArcStaysOnTheCircleAndGoesRight() {
        var m = move([.init(t: 0, eye: SIMD3(0, -0.3, -1), ease: true)])
        m.arc(degrees: 60, speed: .normal)
        XCTAssertEqual(m.keys.count, 5)                   // 4 keys of 15 deg
        XCTAssertEqual(m.duration, 6, accuracy: 1e-9)     // 10 deg/s
        let r0 = simd_length(SIMD3(0, 0, -1.0))
        for p in m.framePositions() {
            XCTAssertEqual(p.y, -0.3, accuracy: 1e-9)                         // level: height kept
            XCTAssertEqual(simd_length(SIMD3(p.x, 0, p.z)), r0, accuracy: 0.005)   // on the circle, within 5 mm
        }
        // positive = the camera's own right at the start
        let c = m.cameraToWorld(eye: m.position(at: 0))
        let right = SIMD3(c.columns.0.x, c.columns.0.y, c.columns.0.z)
        XCTAssertGreaterThan(simd_dot(m.position(at: 0.5) - m.position(at: 0), right), 0)
    }

    func testBoomRaisesAndStopsShortOfThePole() {
        var m = move([.init(t: 0, eye: SIMD3(0, 0, -1), ease: true)])
        m.boom(degrees: 30, speed: .normal)
        let end = m.sortedKeys.last!.position
        let el = asin(simd_dot(simd_normalize(end), m.upVector)) * 180 / .pi
        XCTAssertEqual(el, 30, accuracy: 1e-6)
        XCTAssertEqual(simd_length(end), 1, accuracy: 1e-9)
        m.boom(degrees: 200, speed: .fast)
        let top = asin(simd_dot(simd_normalize(m.sortedKeys.last!.position), m.upVector)) * 180 / .pi
        XCTAssertEqual(top, 87, accuracy: 1e-6)
    }

    func testDollyAndContinuingMovesFlow() {
        var m = move([.init(t: 0, eye: SIMD3(0, 0, -1), ease: true)])
        m.arc(degrees: -30, speed: .normal)
        m.arc(degrees: -30, speed: .normal)
        // the join between the two arcs is a pass-through, the ends rest
        let ks = m.sortedKeys
        XCTAssertTrue(ks.first!.ease)
        XCTAssertTrue(ks.last!.ease)
        XCTAssertFalse(ks[2].ease)
        m.dolly(metres: -0.25, speed: .normal)
        XCTAssertEqual(simd_length(m.sortedKeys.last!.position), 0.75, accuracy: 1e-9)
    }

    func testRetimeKeepsOrderAndSetKeyUpdatesInPlace() {
        var m = move([.init(t: 0, eye: SIMD3(0, 0, -1), ease: true)])
        m.hold(seconds: 1); m.hold(seconds: 1)
        let mid = m.sortedKeys[1].id
        m.retimeKey(mid, to: 5)                        // past the last key: stops one frame short
        XCTAssertEqual(m.sortedKeys[1].t, 2 - 1.0 / 30, accuracy: 1e-9)
        let n = m.keys.count
        _ = m.setKey(at: 0.01, eye: SIMD3(0, 0, -2))   // within a frame of key 0: updates it
        XCTAssertEqual(m.keys.count, n)
        XCTAssertEqual(m.sortedKeys[0].position.z, -2)
    }

    func testBakedJSONIsWhatRenderReads() throws {
        var m = move([.init(t: 0, eye: SIMD3(0, 0, -1), ease: true)])
        m.arc(degrees: 30, speed: .fast)
        let data = try m.bakedJSON()
        let p = try JSONDecoder().decode(MovePath.self, from: data)
        XCTAssertEqual(p.frames.count, m.frameCount)
        XCTAssertEqual(p.width, 1913)
        XCTAssertEqual(p.K[0][2], 956.5)
        let pose = p.pose(at: 10)!
        let want = m.position(at: 10.0 / 30)
        XCTAssertEqual(Double(pose.position.x), want.x, accuracy: 1e-5)
        XCTAssertEqual(Double(pose.position.z), want.z, accuracy: 1e-5)
    }

    func testAnalysisMeasuresTheAngleOffCoverage() {
        var m = move([.init(t: 0, eye: SIMD3(0, 0, -1), ease: true)])
        m.arc(degrees: 20, speed: .normal)
        // one real camera where the move starts: the angle grows to 20 deg by the end
        let a = MoveAnalysis(move: m, captures: [SIMD3(0, 0, -1)])
        XCTAssertEqual(a.offAngle.first!, 0, accuracy: 1e-6)
        XCTAssertEqual(a.worstAngle, 20, accuracy: 0.05)
        XCTAssertEqual(a.pathLength, 20 * .pi / 180, accuracy: 0.002)
        XCTAssertNil(a.spikeFrame)
    }

    func testFrameCheckResetsWhenTheMoveChanges() {
        var c = FrameCheck()
        c.saw(.first, md5: "a"); c.saw(.middle, md5: "a"); c.saw(.last, md5: "a")
        XCTAssertTrue(c.complete(for: "a"))
        XCTAssertFalse(c.complete(for: "b"))
        c.saw(.first, md5: "b")
        XCTAssertEqual(c.seen, [.first])
        XCTAssertEqual(FrameCheck.Which.middle.frame(of: 91), 45)
    }

    func testSpeedCarriesThroughPassThroughKeys() {
        // uneven segments: before the fix the speed stepped at the middle key
        let m = move([.init(t: 0, eye: SIMD3(0, 0, -1), ease: true),
                      .init(t: 1, eye: SIMD3(0.2, 0, -0.98), ease: false),
                      .init(t: 4, eye: SIMD3(1, 0, 0), ease: true)])
        let ps = m.framePositions()
        let before = simd_length(ps[30] - ps[29]), after = simd_length(ps[31] - ps[30])
        XCTAssertEqual(after / before, 1, accuracy: 0.25)
    }

    func testPresetsShapeAndPace() {
        let base = move([.init(t: 0, eye: SIMD3(0, -0.2, -1), ease: true)])
        var push = base
        push.applyPreset(.pushIn, from: SIMD3(0, -0.2, -1), front: nil, speed: .normal)
        XCTAssertEqual(push.duration, 8, accuracy: 1e-9)
        let r0 = simd_length(SIMD3(0.0, -0.2, -1)), r1 = simd_length(push.sortedKeys.last!.position)
        XCTAssertEqual(r1 / r0, 2.0 / 3.0, accuracy: 1e-9)
        // a straight line at the anchor, easing in and out over the whole shot
        let ps = push.framePositions()
        let first = simd_length(ps[1] - ps[0]), mid = simd_length(ps[121] - ps[120])
        XCTAssertLessThan(first, mid * 0.05)
        for p in ps { XCTAssertLessThan(simd_length(simd_cross(simd_normalize(p), simd_normalize(SIMD3(0, -0.2, -1)))), 1e-9) }

        // 180: centred on the front, ends on the opposite side of it, same height and distance
        var half = base
        half.applyPreset(.orbit180, from: SIMD3(0, -0.2, -1), front: SIMD3(0, 0, -1), speed: .slow)
        XCTAssertEqual(half.duration, 36, accuracy: 1e-9)
        let a = half.sortedKeys.first!.position, b = half.sortedKeys.last!.position
        XCTAssertEqual(a.y, -0.2, accuracy: 1e-9)
        XCTAssertEqual(a.x, -b.x, accuracy: 1e-9)
        XCTAssertEqual(abs(a.x), 1, accuracy: 1e-9)
        let mid180 = half.position(at: 18)
        XCTAssertEqual(mid180.z, -1, accuracy: 0.01)                    // passes the front at half time
        XCTAssertEqual(KeyedMove.inverseSmoothstep(0.5), 0.5, accuracy: 1e-12)

        var full = base
        full.applyPreset(.orbit360, from: SIMD3(0, -0.2, -1), front: nil, speed: .normal)
        XCTAssertLessThan(simd_length(full.sortedKeys.last!.position - SIMD3(0, -0.2, -1)), 1e-9)
        XCTAssertGreaterThanOrEqual(full.keys.count, 25)

        var crane = base
        crane.applyPreset(.craneReveal, from: SIMD3(0, 0, -1), front: nil, speed: .fast)
        let end = crane.sortedKeys.last!.position
        XCTAssertEqual(asin(simd_dot(simd_normalize(end), crane.upVector)) * 180 / .pi, 15, accuracy: 1e-6)
        XCTAssertEqual(simd_length(end), 1.25, accuracy: 1e-9)
        XCTAssertEqual(crane.duration, 4.5, accuracy: 1e-9)
    }
}
