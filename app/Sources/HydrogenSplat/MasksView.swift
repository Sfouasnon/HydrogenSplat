import SwiftUI
import AppKit
import HSCore

/// The pieces of the Subject step (SubjectPage) behind "Adjust the outlines": how `hs masks`
/// draws the subject's outline in every frame. The defaults are the engine's.
struct MaskSettingsPanel: View {
    @Binding var settings: MaskSettings
    /// A trained model exists to project from; without one only the sparse points can be.
    let modelAvailable: Bool
    @State private var more = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            StepSetting(title: "Method", help: "Object: the thing itself, found in each frame. Region: the projected area around it, backdrop included.") {
                Picker("", selection: $settings.method) {
                    ForEach(MaskSettings.Method.allCases) { Text($0.title).tag($0) }
                }
                .labelsHidden().pickerStyle(.segmented).frame(width: 260)
            }
            StepSetting(title: "Built from", help: modelAvailable ? "The trained model gives the tightest outline; the sparse points need no model."
                                                                 : "No model in this project yet, so the sparse points are what there is.") {
                Picker("", selection: $settings.source) {
                    ForEach(MaskSettings.Source.allCases) { Text($0.title).tag($0) }
                }
                .labelsHidden().pickerStyle(.segmented).frame(width: 260)
                .disabled(!modelAvailable)
            }
            StepSetting(title: "Size", help: "Below 1 tightens the region; above 1 takes in more around the subject.") {
                Slider(value: $settings.radiusScale, in: 0.5...2.0, step: 0.05) { EmptyView() }
                    .frame(width: 220)
                Text(String(format: "×%.2f", settings.radiusScale))
                    .monospacedDigit().frame(width: 60, alignment: .leading)
            }
            StepSetting(title: "Margin", help: "The outline grows outward by this share; generous costs little, tight deletes real subject.") {
                Slider(value: $settings.marginFrac, in: 0...0.2, step: 0.01) { EmptyView() }
                    .frame(width: 220)
                Text(String(format: "%.0f%%", settings.marginFrac * 100))
                    .monospacedDigit().frame(width: 60, alignment: .leading)
            }
            DisclosureGroup("More", isExpanded: $more) {
                VStack(alignment: .leading, spacing: 10) {
                    if settings.method == .vision {
                        StepSetting(title: "Edge", help: "Softness of the outline's rim in pixels; 1 suits an opaque subject, 0 is a hard edge.") {
                            Slider(value: $settings.featherPx, in: 0...4, step: 0.5) { EmptyView() }
                                .frame(width: 220)
                            Text(String(format: "%.1f px", settings.featherPx))
                                .monospacedDigit().frame(width: 60, alignment: .leading)
                        }
                    }
                    StepSetting(title: "Minimum opacity", help: "Splats fainter than this are not drawn; raise it against haze, lower it against a patchy outline.") {
                        Slider(value: $settings.minOpacity, in: 0...0.6, step: 0.05) { EmptyView() }
                            .frame(width: 220)
                        Text(String(format: "%.2f", settings.minOpacity))
                            .monospacedDigit().frame(width: 60, alignment: .leading)
                    }
                    StepSetting(title: "Close gaps", help: "Bridges gaps between projected splats before the inside is filled.") {
                        Stepper(value: $settings.closePx, in: 1...81, step: 2) {
                            Text("\(settings.closePx) px").monospacedDigit()
                        }
                        .frame(width: 160)
                    }
                    StepSetting(title: "Glints", help: "Cut the specular highlights out of the outline so training does not fake them with haze.") {
                        Toggle("Keep glints out of the outline", isOn: $settings.excludeHighlights)
                        if settings.excludeHighlights {
                            Stepper(value: $settings.highlightCode, in: 200...254, step: 2) {
                                Text("≥ \(settings.highlightCode)").monospacedDigit()
                            }
                            .frame(width: 130)
                        }
                    }
                    StepSetting(title: "One object", help: "Keep only the largest piece; turn off when the subject really is in separate pieces.") {
                        Toggle("Keep the largest piece only", isOn: $settings.keepLargest)
                    }
                    StepSetting(title: "Preview", help: "How many frames go into the preview sheet written beside the outlines.") {
                        Stepper(value: $settings.previewViews, in: 2...12) {
                            Text("\(settings.previewViews) frames").monospacedDigit()
                        }
                        .frame(width: 160)
                    }
                }
                .padding(.top, 6)
            }
        }
    }
}
