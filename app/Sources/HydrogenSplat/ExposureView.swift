import SwiftUI
import AppKit
import HSCore

/// The pieces of the Look step (LookPage) behind the "match to a reference" choice: which frame
/// every view is matched to, and whether white balance follows. `hs exposure` rewrites the solved
/// training images; the originals stay in solve/exposure_backup and `--restore` puts them back.
struct ExposureReferencePicker: View {
    let project: String
    @Binding var settings: ExposureSettings
    let quality: FrameQuality?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            StepSetting(title: "Match to", help: helpText) {
                Picker("", selection: $settings.reference) {
                    ForEach(ExposureSettings.Reference.allCases) { Text($0.title).tag($0) }
                }
                .labelsHidden().pickerStyle(.segmented).frame(width: 400)
            }
            switch settings.reference {
            case .auto:
                if let r = quality?.exposureReference {
                    referenceCard(cap: r.cap, why: r.why + (r.relaxed == true ? " — nothing was clean, so this is the least-bad pick" : ""))
                } else {
                    Text("No frame report with a suggested frame; pick one yourself, or match to the median.")
                        .font(.caption).foregroundStyle(.orange).padding(.leading, 140)
                }
            case .chosen:
                HStack(spacing: 8) {
                    Text("Frame").foregroundStyle(.secondary)
                    TextField("#", value: $settings.chosenCapture, format: .number)
                        .textFieldStyle(.roundedBorder).frame(width: 70)
                    if let s = quality?.exposureReference {
                        Button("Use suggested (#\(s.sel))") { settings.chosenCapture = s.sel }.controlSize(.small)
                    }
                    Text("or right-click a frame in Frames → Use as exposure reference")
                        .font(.caption).foregroundStyle(.secondary)
                }
                .padding(.leading, 140)
                if let n = settings.chosenCapture {
                    referenceCard(cap: String(format: "cap%03d", n), why: "chosen by hand")
                }
            case .median:
                EmptyView()
            }
            StepSetting(title: "Correct", help: "With white balance, drift in colour goes too; brightness only leaves colour alone.") {
                Picker("", selection: $settings.whiteBalance) {
                    Text("Brightness and white balance").tag(true)
                    Text("Brightness only").tag(false)
                }
                .labelsHidden().pickerStyle(.segmented).frame(width: 340)
            }
        }
    }

    private var helpText: String {
        switch settings.reference {
        case .auto: return "The cleanest pick near the set's median brightness that clips least; matching to it mostly darkens, which loses nothing."
        case .chosen: return "Every view is scaled onto this frame; it keeps its own brightness and white point exactly."
        case .median: return "No single frame: every view is scaled onto the median of all views."
        }
    }

    @ViewBuilder private func referenceCard(cap: String, why: String) -> some View {
        let f = quality?.frame(cap: cap)
        HStack(alignment: .top, spacing: 12) {
            if let t = f?.thumb {
                ThumbImage(path: (project as NSString).appendingPathComponent("select/" + t))
                    .frame(width: 160)
                    .clipShape(RoundedRectangle(cornerRadius: 4))
            }
            VStack(alignment: .leading, spacing: 4) {
                Text(cap + (f.map { " · pick #\($0.sel) · source frame \($0.frame)" } ?? ""))
                    .font(.callout.weight(.semibold))
                if let f = f {
                    Text([f.ev.map { String(format: "%+.2f EV off median", $0) },
                          f.clip.map { String(format: "%.1f%% clipped", $0 * 100) },
                          f.focusRel.map { String(format: "focus %.2f×", $0) }]
                        .compactMap { $0 }.joined(separator: " · "))
                        .font(.caption).monospacedDigit()
                } else if quality != nil {
                    Text("not in the frame report").font(.caption).foregroundStyle(.orange)
                }
                Text(why).font(.caption).foregroundStyle(.secondary).lineLimit(1)
            }
        }
        .padding(.leading, 140)
    }
}

/// What the last `hs exposure` run did to the training images, in the engine's own numbers.
struct ExposureApplied: View {
    let stage: StageState

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if stage.metrics["restored"] != nil {
                Text("The frames are back as shot.").font(.subheadline.weight(.semibold))
            } else {
                MetricGrid(rows: rows(stage.metrics, keys: ["applied", "reference", "reference_why", "views",
                                                            "global_drop_stops", "shoulder_knee",
                                                            "luma_spread_before", "luma_spread_after",
                                                            "gain_min", "gain_max", "clipped_fraction_max"]))
                ForEach(stage.checks.filter { $0.name != "exposure_consistent" }) { c in
                    CheckRow(name: c.name, ok: c.ok, value: c.value?.display, needsHuman: c.needsHuman)
                }
            }
        }
    }

    private func rows(_ m: [String: JSONValue], keys: [String]) -> [(String, String)] {
        keys.compactMap { k in m[k].map { (k, $0.display) } }
    }
}
