import XCTest
@testable import HSCore

/// The Scale box: the commands it runs and how it reads the last `hs scale --lidar` measurement
/// out of the manifest.
final class ScaleTests: XCTestCase {
    func testMeasureAndApplyArguments() {
        var s = ScaleSettings()
        XCTAssertEqual(s.measureArguments(project: "/p"), [])            // no scan: nothing to run
        XCTAssertEqual(s.scanProblem, "Choose a scan first")
        s.scan = "lidar/scan.ply"
        XCTAssertEqual(s.measureArguments(project: "/p"), ["scale", "-p", "/p", "--lidar", "/p/lidar/scan.ply", "--dry-run"])
        XCTAssertEqual(s.applyScanArguments(project: "/p"), ["scale", "-p", "/p", "--lidar", "/p/lidar/scan.ply"])
        s.scanUp = .z
        s.trustScan = true
        XCTAssertEqual(s.applyScanArguments(project: "/p"),
                       ["scale", "-p", "/p", "--lidar", "/p/lidar/scan.ply", "--scan-up", "z", "--trust-scan"])
        s.scanInit = .silhouette
        s.initPoints = true
        XCTAssertEqual(s.measureArguments(project: "/p"),
                       ["scale", "-p", "/p", "--lidar", "/p/lidar/scan.ply", "--scan-up", "z", "--init", "silhouette",
                        "--init-points", "--dry-run"])
        s.scanInit = .auto; s.initPoints = false
        s.scan = "/elsewhere/room.obj"                                     // an absolute path is passed as is
        XCTAssertEqual(s.measureArguments(project: "/p")[4], "/elsewhere/room.obj")
    }

    func testFactorArguments() {
        var s = ScaleSettings()
        XCTAssertEqual(s.applyFactorArguments(project: "/p"), [])
        XCTAssertEqual(s.factorProblem, "Enter a factor")
        s.factor = -2
        XCTAssertEqual(s.factorProblem, "The factor must be a positive number")
        s.factor = 5.81
        s.note = " ring silhouette fit "
        XCTAssertNil(s.factorProblem)
        XCTAssertEqual(s.applyFactorArguments(project: "/p"),
                       ["scale", "-p", "/p", "--factor", "5.81", "--note", "ring silhouette fit"])
        s.trustScan = true
        XCTAssertEqual(s.applyFactorArguments(project: "/p").last, "--trust-scan")
    }

    /// The real Circles measurement (2026-09-23): refused, ratio 4.65 from a lawn-only overlap,
    /// up axis detected as Z, 372 captures off the scan.
    func testReadsTheLastMeasurementFromTheManifest() throws {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "manifest_lidar_check", withExtension: "json", subdirectory: "Fixtures"))
        let m = try XCTUnwrap(Manifest(data: Data(contentsOf: url)))
        let c = try XCTUnwrap(LidarCheck(manifest: m))
        XCTAssertFalse(c.aligned)
        XCTAssertFalse(c.applied)
        XCTAssertTrue(c.stereo)
        XCTAssertEqual(c.scanName, "scan.ply")
        XCTAssertEqual(c.scale ?? 0, 4.646, accuracy: 0.001)
        XCTAssertEqual(c.impliedBaselineMM ?? 0, 49.45, accuracy: 0.01)
        XCTAssertEqual(c.scanUp, "z")
        XCTAssertEqual(c.scanUpSource, "detected")
        XCTAssertEqual(c.capturesOffScan, 372)
        XCTAssertTrue(c.verdict.hasPrefix("Not aligned — "), c.verdict)
        XCTAssertTrue(c.facts.contains { $0.hasPrefix("scan up axis Z (read off the scan)") }, "\(c.facts)")
        XCTAssertTrue(c.facts.contains { $0.contains("372 captures outside the scan") }, "\(c.facts)")
        XCTAssertTrue(c.problems.contains { $0.hasPrefix("lidar scale agrees") }, "\(c.problems)")
        XCTAssertNil(AppliedScale(manifest: m))
    }

    func testAnAlignedStereoMeasurementSaysHowFarOffTheSolveIs() throws {
        let j = """
        {"name":"x","stages":{"scale":{"lidar_check":{"at":"2026-09-23T15:00:00-0700","scan":"/p/lidar/scan.ply",
          "aligned":true,"applied":false,
          "metrics":{"scale_ratio":5.81,"implied_baseline_mm":61.8,"lidar_inlier_fraction_in_scan":0.55,"lidar_rms_mm":92.0,
                     "scan_up":"z","scan_up_source":"flag","camera_height_mm_median":1300,"captures_off_scan":0},
          "checks":[{"name":"lidar_aligned","ok":true,"value":"55%"}]}}},
         "scale":{"source":"manual","scale_factor":5.81,"note":"ring silhouette fit","at":"2026-09-23T16:00:00-0700"}}
        """
        let m = try XCTUnwrap(Manifest(data: Data(j.utf8)))
        let c = try XCTUnwrap(LidarCheck(manifest: m))
        XCTAssertEqual(c.verdict, "Aligned — the solve is 5.8× too small (scale ratio 5.810, as if the baseline were 61.8 mm); "
                       + "55% of the solve's points on the scan within tolerance, RMS 92 mm")
        XCTAssertEqual(c.facts, ["scan up axis Z", "camera 1.30 m above the scan's ground", "every capture within the scan"])
        XCTAssertEqual(c.problems, [])
        let a = try XCTUnwrap(AppliedScale(manifest: m))
        XCTAssertEqual(a.sentence, "Scale applied from a factor given by hand — ×5.8100 (ring silhouette fit)")
        XCTAssertEqual(Format.factor(1.03), "1.03×")
        XCTAssertEqual(Format.factor(5.81), "5.8×")
    }
}
