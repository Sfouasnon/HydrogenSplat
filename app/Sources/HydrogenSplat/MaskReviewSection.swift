import SwiftUI
import AppKit
import HSCore

private let maskReviewPage = 50
private let maskReviewPictureHeight: CGFloat = 180

/// The masks the engine flagged when it checked them against each other, and what to do with each:
/// repair it, leave the view out of training, or keep it as built. `hs train` refuses to start
/// while a flagged view has no decision.
///
/// The engine is the source of truth. This reads masks_review/review.json, shows the pictures the
/// engine wrote beside it, and runs `hs masks --decide` through the same queue as Build Masks.
/// MasksView shows it only while mask files exist, so a missing report means "never checked".
struct MaskReviewSection: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var store: ProjectStore
    let project: ProjectSummary
    /// MasksView's reload key: the project path and the masks stage's finished time and status.
    let reloadKey: String

    /// Read when the key changes and when a run ends, never in body: body runs on every poll.
    /// (The key itself costs one stat of review.json per pass.)
    @State private var review: MaskReview?
    /// False until the first read, so "never checked" is not shown for the moment before it.
    @State private var loaded = false
    /// How many rows are listed. The list is worst first, so the first page is the one that matters.
    @State private var shown = maskReviewPage
    /// Bumped on every read, so a picture the engine rewrote under the same name is loaded again:
    /// ThumbImage looks at the file's date only when its body runs, and a view whose path did not
    /// change need not be asked for its body.
    @State private var revision = 0

    private var lockAlive: Bool { project.lock?.alive == true }

    var body: some View {
        // a queue held in a dictionary does not re-render this view when it finishes; QueueWatch does
        QueueWatch(queue: model.maskQueues[project.path]) { q in
            let running = q?.isRunning ?? false
            section(running: running)
                .task(id: readKey(running: running)) {
                    review = MaskReview.read(project: project.path)
                    loaded = true
                    revision += 1
                }
        }
    }

    /// The report is read again when the stage changes, when the engine rewrites the stage's
    /// `mask_review` metric, when review.json itself is rewritten (a decide from Terminal can swap
    /// two decisions and leave every count in the metric as it was), and when a run here ends.
    private func readKey(running: Bool) -> String {
        let metric = project.manifest?.stage("masks")?.metrics["mask_review"]?.compactJSON ?? ""
        let attrs = try? FileManager.default.attributesOfItem(atPath: MaskReview.reportPath(project: project.path))
        let stamp = (attrs?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        let state = running ? "running" : "idle"
        return "\(reloadKey)|\(state)|\(metric)|\(stamp)"
    }

    private func section(running: Bool) -> some View {
        let blocked = running || lockAlive || !model.config.problems.isEmpty
        return VStack(alignment: .leading, spacing: 8) {
            Divider()
            if let r = review {
                Text(r.headline).font(.subheadline.weight(.semibold))
                    .fixedSize(horizontal: false, vertical: true)
                if !r.flagged.isEmpty {
                    if let n = r.note {
                        Text(n).font(.caption).foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    legend
                }
                if !r.newerEngine {
                    bulk(r, running: running, blocked: blocked)
                }
                if !r.flagged.isEmpty {
                    list(r, blocked: blocked)
                }
            } else if loaded {
                Text("Review").font(.subheadline.weight(.semibold))
                Text("These masks were never checked against each other. The check finds the masks that leave out part of the subject, and hs train will not start on masks that were not checked.")
                    .font(.callout).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                HStack(spacing: 12) {
                    Button("Check masks") {
                        run("Check masks", MaskReview.checkOnlyArguments(project: project.path))
                    }
                    .disabled(blocked)
                    .help("hs masks --check-only: check the masks on disk against each other. Nothing is rebuilt and nothing goes stale.")
                    status(running: running)
                }
            }
        }
    }

    private var legend: some View {
        Text("In the pictures: red is this view's mask, blue is the subject as the other views see it, yellow is the piece in question. In a repair picture, green is the repaired mask.")
            .font(.caption).foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }

    @ViewBuilder private func status(running: Bool) -> some View {
        if running {
            ProgressView().controlSize(.small)
        } else if lockAlive, let l = project.lock {
            Label("\(l.stage ?? "a stage") is running", systemImage: "lock.fill").foregroundStyle(.secondary)
        }
    }

    // MARK: all at once

    private func bulk(_ r: MaskReview, running: Bool, blocked: Bool) -> some View {
        let exact = r.exactRepairs
        let exactTitle: String = exact == 1 ? "Apply the 1 repair that needs no look"
                                            : "Apply the \(exact) repairs that need no look"
        return HStack(spacing: 12) {
            if !r.flagged.isEmpty {
                Button(exactTitle) {
                    decide(MaskReview.exactKey, MaskReview.Choice.repair, title: "Repair masks")
                }
                .disabled(blocked || exact == 0)
                .help("Vision's own object where the build had turned it down, or a hole closed inside the mask: the repaired mask replaces the one in the dataset, the original is kept in masks_review/original, and train, prune, render and views go stale. Every other repair rests on the hull the other views voted, which can be too large near the silhouette — look at each of those before you take it.")
                Button("Exclude the rest") {
                    decide(MaskReview.undecidedKey, MaskReview.Choice.exclude, title: "Exclude masks")
                }
                .disabled(blocked || !r.canExcludeRest)
                .help("Every view still undecided is left out of training. Nothing in the dataset changes.")
            }
            Button("Check again") {
                run("Check masks", MaskReview.checkOnlyArguments(project: project.path))
            }
            .disabled(blocked)
            .help("hs masks --check-only: check the masks on disk again. A decision stays with its mask for as long as that file has not changed.")
            status(running: running)
        }
    }

    // MARK: one view at a time

    private func list(_ r: MaskReview, blocked: Bool) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            LazyVStack(alignment: .leading, spacing: 10) {
                ForEach(Array(r.flagged.prefix(shown))) { f in
                    row(f, blocked: blocked)
                    Divider()
                }
            }
            if r.flagged.count > shown {
                HStack(spacing: 12) {
                    Text("Showing the worst \(shown) of \(r.flagged.count).")
                        .font(.caption).foregroundStyle(.secondary)
                    Button("Show \(min(maskReviewPage, r.flagged.count - shown)) more") {
                        shown += maskReviewPage
                    }
                    .controlSize(.small)
                }
            }
        }
    }

    private func row(_ f: MaskReview.Flagged, blocked: Bool) -> some View {
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
                Text("Repair: " + rep.display)
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                Text("No repair to offer for this view.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            HStack(spacing: 8) {
                if let d = f.decision {
                    Text(d.decidedTitle)
                        .font(.caption.weight(.semibold))
                        .padding(.horizontal, 6).padding(.vertical, 1)
                        .background(Capsule().fill(badgeColor(d).opacity(0.18)))
                        .foregroundStyle(badgeColor(d))
                    Button("Undo") {
                        decide(f.view, MaskReview.Choice.undo, title: "Undo " + f.view)
                    }
                    .disabled(blocked)
                    .help("Clear the decision. Undoing a repair puts the original mask back.")
                } else {
                    Button("Repair") {
                        decide(f.view, MaskReview.Choice.repair, title: "Repair " + f.view)
                    }
                    .disabled(blocked || f.repair == nil)
                    .help(f.repair == nil
                          ? "The engine has no repair to offer for this view."
                          : "Replace this view's mask in the dataset with the repaired one. The original is kept in masks_review/original, and train, prune, render and views go stale.")
                    Button("Exclude") {
                        decide(f.view, MaskReview.Choice.exclude, title: "Exclude " + f.view)
                    }
                    .disabled(blocked)
                    .help("Leave this view out of training. Nothing in the dataset changes.")
                    Button("Keep") {
                        decide(f.view, MaskReview.Choice.keep, title: "Keep " + f.view)
                    }
                    .disabled(blocked)
                    .help("Train on this mask as it was built.")
                }
            }
            .controlSize(.small)
        }
    }

    /// One of the engine's pictures at a fixed height, so a long list stays a list. Click opens it.
    private func picture(_ path: String) -> some View {
        ThumbImage(path: path)
            .id("\(path)|\(revision)")
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

    // MARK: run

    private func decide(_ view: String, _ choice: MaskReview.Choice, title: String) {
        let decisions: [(view: String, choice: MaskReview.Choice)] = [(view: view, choice: choice)]
        run(title, MaskReview.decideArguments(project: project.path, decisions))
    }

    /// The same queue as Build Masks (MasksView.run), so its running and lock handling apply here.
    private func run(_ title: String, _ arguments: [String]) {
        guard !arguments.isEmpty else { return }
        let steps = [RunQueue.Step(title: title, arguments: arguments)]
        let q = RunQueue(config: model.config, steps: steps)
        let path = project.path
        q.onStep = { s in model.projectRuns[path] = s }
        q.onFinish = { _ in store.reload() }
        model.maskQueues[path] = q
        q.start()
    }
}
