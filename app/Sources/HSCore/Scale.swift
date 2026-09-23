import Foundation

// MARK: - hs scale: settings

/// The scan's gravity axis (`hs scale --scan-up`). Auto reads it off the scan (the shortest
/// axis whose lowest band is a floor); Polycam keeps ARKit's +Y, Scaniverse's PLY is +Z.
public enum ScanUp: String, CaseIterable, Identifiable, Sendable {
    case auto, y, z
    public var id: String { rawValue }
    public var title: String {
        switch self {
        case .auto: return "Auto"
        case .y: return "Y up (Polycam)"
        case .z: return "Z up (Scaniverse)"
        }
    }
}

/// The Scale box on the Frames page: a phone LiDAR scan measured against the solve and applied,
/// or a scale factor found some other way. Per project, kept for the session.
///
/// `hs scale --lidar SCAN --dry-run` measures (writes only `scale/lidar_report.json` and the
/// aligned PLY, never the dataset); without `--dry-run` it applies the scan's scale on a
/// mono/array project, and on a Hydrogen (stereo) project only with `--trust-scan` — the H1's
/// 10.6 mm baseline is a trustworthy scale only within a metre or so (the Circles solve came
/// out 5.8x small at 4 m). `--factor F` applies a number already known through the same path.
public struct ScaleSettings: Equatable, Sendable {
    /// The scan file, project-relative ("lidar/scan.ply") once the picker has copied it in.
    public var scan: String?
    public var scanUp: ScanUp = .auto
    /// Stereo projects: apply the scan's (or the factor's) scale over the calibrated baseline.
    public var trustScan = false
    /// A known factor (solve units x F = mm), instead of the scan.
    public var factor: Double?
    public var note: String = ""

    public init() {}

    public static let defaults = ScaleSettings()

    private func scanArguments(project: String) -> [String] {
        guard let s = scan else { return [] }
        let path = s.hasPrefix("/") ? s : (project as NSString).appendingPathComponent(s)
        var a = ["scale", "-p", project, "--lidar", path]
        if scanUp != .auto { a += ["--scan-up", scanUp.rawValue] }
        return a
    }

    /// `hs scale -p P --lidar SCAN --dry-run [--scan-up …]`: measure only.
    public func measureArguments(project: String) -> [String] {
        let a = scanArguments(project: project)
        return a.isEmpty ? [] : a + ["--dry-run"]
    }

    /// `hs scale -p P --lidar SCAN [--scan-up …] [--trust-scan]`: apply the scan's scale.
    public func applyScanArguments(project: String) -> [String] {
        let a = scanArguments(project: project)
        return a.isEmpty ? [] : a + (trustScan ? ["--trust-scan"] : [])
    }

    /// `hs scale -p P --factor F [--note …] [--trust-scan]`: apply a known factor.
    public func applyFactorArguments(project: String) -> [String] {
        guard let f = factor, f.isFinite, f > 0 else { return [] }
        var a = ["scale", "-p", project, "--factor", TrainSettings.num(f)]
        let n = note.trimmingCharacters(in: .whitespacesAndNewlines)
        if !n.isEmpty { a += ["--note", n] }
        if trustScan { a.append("--trust-scan") }
        return a
    }

    /// Why the scan buttons are off, in words; nil when a scan is set.
    public var scanProblem: String? {
        scan == nil ? "Choose a scan first" : nil
    }

    /// Why the factor button is off; nil when the factor is usable.
    public var factorProblem: String? {
        guard let f = factor else { return "Enter a factor" }
        if !f.isFinite || f <= 0 { return "The factor must be a positive number" }
        return nil
    }
}

// MARK: - the last measurement, from the manifest

/// What the last `hs scale --lidar` left in `stages.scale.lidar_check` (measured, applied or
/// refused), read tolerantly from the manifest — the engine writes the full report to
/// `scale/lidar_report.json`, and this is the part a page needs: the verdict and the numbers
/// that decide what to do next.
public struct LidarCheck: Equatable, Sendable {
    public let at: Date?
    public let scan: String?
    public let aligned: Bool
    public let applied: Bool
    public let metrics: [String: JSONValue]
    public let checks: [CheckResult]

    public init?(_ v: JSONValue?) {
        guard let v = v, v.object != nil else { return nil }
        at = Manifest.date(v["at"]?.string)
        scan = v["scan"]?.string
        aligned = v["aligned"]?.bool ?? false
        applied = v["applied"]?.bool ?? false
        metrics = v["metrics"]?.object ?? [:]
        checks = v["checks"]?.array?.compactMap(CheckResult.init) ?? []
    }

    /// From a manifest: `stages.scale.lidar_check`.
    public init?(manifest: Manifest) {
        self.init(manifest.raw["stages"]?["scale"]?["lidar_check"])
    }

    public var scanName: String? { scan.map { ($0 as NSString).lastPathComponent } }
    public var stereo: Bool { metrics["scale_ratio"] != nil }
    /// Solve -> mm on a stereo project (1.0 = the baseline was right); the factor to apply on mono/array.
    public var scale: Double? { metrics["scale_ratio"]?.double ?? metrics["scale_factor"]?.double }
    public var impliedBaselineMM: Double? { metrics["implied_baseline_mm"]?.double }
    public var scanUp: String? { metrics["scan_up"]?.string }
    public var scanUpSource: String? { metrics["scan_up_source"]?.string }
    public var upAngleDeg: Double? { metrics["lidar_up_to_current_up_deg"]?.double }
    public var cameraHeightMM: Double? { metrics["camera_height_mm_median"]?.double }
    public var capturesOffScan: Int? { metrics["captures_off_scan"]?.int }
    public var inlierFraction: Double? { metrics["lidar_inlier_fraction_in_scan"]?.double ?? metrics["lidar_inlier_fraction"]?.double }
    public var rmsMM: Double? { metrics["lidar_rms_mm"]?.double }
    public var scanPoints: Int? { metrics["scan_points"]?.int }
    public var scanUnits: String? { metrics["scan_units"]?.string }

    public func check(_ name: String) -> CheckResult? { checks.first { $0.name == name } }

    /// The one-line verdict: aligned and what the scale is, or why not.
    public var verdict: String {
        if !aligned {
            let why = check("lidar_aligned")?.value?.string ?? "the scan did not align with the solve"
            return "Not aligned — " + why
        }
        var parts: [String] = []
        if let s = scale {
            if stereo {
                let how = abs(s - 1) < 0.02 ? "metric within 2%" : (s > 1 ? Format.factor(s) + " too small" : Format.factor(1 / s) + " too big")
                var line = "the solve is " + how + String(format: " (scale ratio %.3f", s)
                line += impliedBaselineMM.map { String(format: ", as if the baseline were %.1f mm)", $0) } ?? ")"
                parts.append(line)
            } else {
                parts.append(String(format: "scale ×%.4f (solve units → mm)", s))
            }
        }
        if let f = inlierFraction, let r = rmsMM {
            parts.append(String(format: "%.0f%% of the solve's points on the scan within tolerance, RMS %.0f mm", f * 100, r))
        }
        return (applied ? "Applied — " : "Aligned — ") + parts.joined(separator: "; ")
    }

    /// Up, ground and coverage as short facts for the page.
    public var facts: [String] {
        var out: [String] = []
        if let u = scanUp {
            out.append("scan up axis \(u.uppercased())" + (scanUpSource == "detected" ? " (read off the scan)" : ""))
        }
        if let a = upAngleDeg {
            out.append(String(format: "scan gravity %.1f° from the solve's up", a) + (a > 10 ? " — needs a look" : ""))
        }
        if let h = cameraHeightMM {
            out.append(String(format: "camera %.2f m above the scan's ground", h / 1000))
        }
        if let n = capturesOffScan {
            out.append(n == 0 ? "every capture within the scan" : "\(n) capture\(n == 1 ? "" : "s") outside the scan")
        }
        if let p = scanPoints {
            out.append("\(p.formatted()) scan points" + (scanUnits.map { " (\($0))" } ?? ""))
        }
        return out
    }

    /// Checks that failed, in the engine's words, for the page's warning list.
    public var problems: [String] {
        checks.filter { !$0.ok }.map { c in
            let v = c.value?.string ?? c.value?.display ?? ""
            return c.name.replacingOccurrences(of: "_", with: " ") + (v.isEmpty ? "" : ": " + v)
        }
    }
}

/// What the project's `scale` record says once a scale has been applied (board, lidar or manual).
public struct AppliedScale: Equatable, Sendable {
    public let source: String
    public let factor: Double?
    public let note: String?
    public let at: Date?

    public init?(manifest: Manifest) {
        guard let v = manifest.raw["scale"], let src = v["source"]?.string else { return nil }
        source = src
        factor = v["scale_factor"]?.double
        note = v["note"]?.string ?? v["board"]?.string
        at = Manifest.date(v["at"]?.string)
    }

    public var sentence: String {
        var s = "Scale applied from " + (source == "manual" ? "a factor given by hand" : source == "lidar" ? "the LiDAR scan" : source)
        if let f = factor { s += String(format: " — ×%.4f", f) }
        if let n = note, !n.isEmpty { s += " (\(n))" }
        return s
    }
}

extension Format {
    /// "5.8×" for a ratio; two decimals under 2, one above.
    public static func factor(_ x: Double) -> String {
        x < 2 ? String(format: "%.2f×", x) : String(format: "%.1f×", x)
    }
}
