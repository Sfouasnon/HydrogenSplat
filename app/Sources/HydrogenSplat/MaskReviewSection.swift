import SwiftUI
import AppKit
import HSCore

private let maskReviewPage = 50
private let maskReviewPictureHeight: CGFloat = 180

/// "Look one by one": the outlines the engine flagged when it checked them against each other,
/// worst first, each with its picture, its repair (when the engine has one) and three buttons.
/// SubjectPage reads masks_review/review.json and runs `hs masks --decide`; this only draws the list.
struct MaskReviewSection: View {
    let review: MaskReview
    let blocked: Bool
    let decide: (String, MaskReview.Choice) -> Void

    /// How many rows are listed. The list is worst first, so the first page is the one that matters.
    @State private var shown = maskReviewPage

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if review.newerEngine {
                Text(review.headline).foregroundStyle(.orange).fixedSize(horizontal: false, vertical: true)
            } else if review.flagged.isEmpty {
                Text("Nothing to look at: every outline agrees with the others.").foregroundStyle(.secondary)
            } else {
                Text("Red is this frame's outline, blue is the subject as the other frames see it, yellow the piece in question, green a repair.")
                    .font(.caption).foregroundStyle(.secondary).lineLimit(1)
                LazyVStack(alignment: .leading, spacing: 10) {
                    ForEach(Array(review.flagged.prefix(shown))) { f in
                        row(f)
                        Divider()
                    }
                }
                if review.flagged.count > shown {
                    HStack(spacing: 12) {
                        Text("Showing the worst \(shown) of \(review.flagged.count).")
                            .font(.caption).foregroundStyle(.secondary)
                        Button("Show \(min(maskReviewPage, review.flagged.count - shown)) more") {
                            shown += maskReviewPage
                        }
                        .controlSize(.small)
                    }
                }
            }
        }
    }

    private func row(_ f: MaskReview.Flagged) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .top, spacing: 8) {
                if let p = f.preview { picture(p) }
                if let p = f.repair?.preview { picture(p) }
            }
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(f.view).font(.callout.monospaced()).textSelection(.enabled)
                Text(f.why.isEmpty ? f.reasons.joined(separator: ", ") : f.why)
                    .font(.callout)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let rep = f.repair {
                Text("Repair: " + rep.display).font(.caption).foregroundStyle(.secondary).lineLimit(1)
            } else {
                Text("No repair to offer for this frame.").font(.caption).foregroundStyle(.secondary)
            }
            HStack(spacing: 8) {
                if let d = f.decision {
                    Text(d.decidedTitle)
                        .font(.caption.weight(.semibold))
                        .padding(.horizontal, 6).padding(.vertical, 1)
                        .background(Capsule().fill(badgeColor(d).opacity(0.18)))
                        .foregroundStyle(badgeColor(d))
                    Button("Undo") { decide(f.view, MaskReview.Choice.undo) }
                        .disabled(blocked)
                        .help("Clear the decision; undoing a repair puts the original outline back.")
                } else {
                    Button("Repair") { decide(f.view, MaskReview.Choice.repair) }
                        .disabled(blocked || f.repair == nil)
                        .help(f.repair == nil ? "The engine has no repair to offer for this frame."
                                              : "Use the repaired outline; the original is kept.")
                    Button("Leave out") { decide(f.view, MaskReview.Choice.exclude) }
                        .disabled(blocked)
                        .help("Leave this frame out of training; nothing else changes.")
                    Button("Keep") { decide(f.view, MaskReview.Choice.keep) }
                        .disabled(blocked)
                        .help("Train on this outline as it was built.")
                }
            }
            .controlSize(.small)
        }
    }

    /// One of the engine's pictures at a fixed height, so a long list stays a list. Click opens it.
    private func picture(_ path: String) -> some View {
        ThumbImage(path: path)
            .frame(height: maskReviewPictureHeight)
            .clipShape(RoundedRectangle(cornerRadius: 4))
            .onTapGesture { NSWorkspace.shared.open(URL(fileURLWithPath: path)) }
            .help("Click to open the picture at full size.")
    }

    private func badgeColor(_ c: MaskReview.Choice) -> Color {
        switch c {
        case .repair: return .green
        case .exclude: return .orange
        case .keep, .undo: return .secondary
        }
    }
}
