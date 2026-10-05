import SwiftUI
import HSCore

/// The viewer's Clean Up control (Shot › Move). It takes out the splats that sit where the camera
/// itself went (`hs prune --clear-path`): the camera was there, so that space was empty, and a
/// move along the camera's path flies straight through whatever the training left in it. The
/// result is a copy beside the model, shown as soon as it is written; the trained model is never
/// changed, so the two can be compared from the model picker.
struct CleanUpButton: View {
    @EnvironmentObject var model: AppModel
    let project: ProjectSummary
    /// The model the viewer is showing.
    let chosen: ViewerModelFile?
    /// A cleaned copy was written at this path: list the files again and show it.
    let onCleaned: (String) -> Void

    enum Strength: String, CaseIterable, Identifiable {
        case gentle, strong
        var id: String { rawValue }
        var title: String { self == .gentle ? "Gentle" : "Strong" }
        var meaning: String {
            self == .gentle ? "Only what hugs the camera's path. Start here."
                            : "Nearly twice as far out. Clears more, and can take a leaf that brushed the lens."
        }
        /// The reach in the capture's own steps between frames (`--clear-path 1.3x`), so it means
        /// the same on a table-top orbit and on a walk through a garden.
        var steps: String { self == .gentle ? "1.3x" : "2.4x" }
    }

    @State private var open = false
    @State private var strength = Strength.gentle
    @State private var outcome: String?
    @State private var failed = false

    private var running: Bool { model.cleanQueues[project.path]?.isRunning == true }

    /// Why cleaning cannot start, or nil.
    private var blocked: String? {
        guard let f = chosen else { return "Load a model first." }
        if f.isCleanUpOutput { return "This one is already a cleaned copy. Pick the trained model at the top left to clean it again." }
        if running { return nil }
        if model.projectRuns[project.path]?.isRunning == true || project.lock?.alive == true {
            return "Wait for the step that is running."
        }
        return nil
    }

    var body: some View {
        Button { open = true } label: { Label("Clean Up", systemImage: "wand.and.stars") }
            .help("Take out the specks and haze that sit where the camera itself went")
            .popover(isPresented: $open, arrowEdge: .bottom) { panel }
    }

    private var panel: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Clean up floaters").font(.headline)
            Text("Takes out the specks and haze sitting where the camera itself went. Nothing real can be there, and a move along the camera's path flies straight through them. Your model is kept; a cleaned copy is added beside it.")
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            HStack(alignment: .top, spacing: 10) {
                ForEach(Strength.allCases) { s in
                    ChoiceTile(title: s.title, subtitle: s.meaning, selected: strength == s) { strength = s }
                }
            }
            if let o = outcome {
                Text(o).foregroundStyle(failed ? Color.red : Color.primary)
                    .fixedSize(horizontal: false, vertical: true)
            } else if let b = blocked {
                Text(b).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack(spacing: 10) {
                Button(running ? "Cleaning…" : "Clean a Copy") { start() }
                    .buttonStyle(.borderedProminent)
                    .disabled(running || blocked != nil)
                if running { ProgressView().controlSize(.small) }
                Spacer()
                Button("Done") { open = false }
            }
        }
        .padding(16)
        .frame(width: 470)
    }

    private func start() {
        guard let f = chosen, blocked == nil else { return }
        let path = project.path
        let name = f.cleanUpStem + "_" + strength.rawValue
        let out = ((path as NSString).appendingPathComponent("prune") as NSString)
            .appendingPathComponent(name + "_clearpath.ply")
        let q = RunQueue(config: model.config, steps: [
            RunQueue.Step(title: "Clean up",
                          arguments: ["prune", "-p", path, "--clear-path", strength.steps, "--ply", f.ply, "--name", name])
        ])
        outcome = nil
        failed = false
        q.onStep = { s in model.projectRuns[path] = s }
        q.onFinish = { done in
            model.cleanQueues[path] = nil
            model.store.reload()
            let run = done.current
            switch done.state {
            case .finished:
                let removed = run?.metrics.last(where: { $0.key == "prune.on_the_path" })?.event.value?.double
                let total = run?.metrics.last(where: { $0.key == "prune.splats_in" })?.event.value?.double
                if let r = removed, let t = total, t > 0 {
                    outcome = r == 0
                        ? "Nothing sat on the camera's path, so the copy is the same as the model."
                        : "Removed \(CleanUpButton.grouped(r)) of \(CleanUpButton.grouped(t)) splats (\(String(format: "%.2f", 100 * r / t)) %). The cleaned copy is showing now; pick the other model at the top left to compare."
                } else {
                    outcome = "Done. The cleaned copy is showing now."
                }
                onCleaned(out)
            case .cancelled:
                outcome = "Stopped."
            default:
                failed = true
                let e = run?.errors.last
                let said = [e?.message, e?.hint].compactMap { $0 }.joined(separator: " — ")
                outcome = said.isEmpty ? "Cleaning did not finish. The Console has the run's last words." : said
            }
        }
        model.cleanQueues[path] = q
        q.start()
    }

    static func grouped(_ x: Double) -> String {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        f.maximumFractionDigits = 0
        return f.string(from: NSNumber(value: x)) ?? String(Int(x))
    }
}

extension ViewerModelFile {
    /// A copy `hs prune` wrote (prune/…), as against a trained or a kept model.
    var isCleanUpOutput: Bool { name.hasPrefix("prune/") }

    /// What a cleaned copy of this model is filed under: the kept model's name, or the export's stem.
    var cleanUpStem: String {
        archive ?? ((ply as NSString).lastPathComponent as NSString).deletingPathExtension
    }

    /// What the model picker calls it: prune/export_20000_gentle_clearpath.ply reads
    /// "cleaned, gentle · export_20000".
    var pickerName: String {
        guard isCleanUpOutput else { return name }
        let stem = ((ply as NSString).lastPathComponent as NSString).deletingPathExtension
        for (suffix, what) in [("_clearpath", "cleaned"), ("_path_only", "what cleaning took out")] where stem.hasSuffix(suffix) {
            var base = String(stem.dropLast(suffix.count))
            var how = ""
            for s in CleanUpButton.Strength.allCases where base.hasSuffix("_" + s.rawValue) {
                base = String(base.dropLast(s.rawValue.count + 1))
                how = ", " + s.rawValue
            }
            return what + how + " · " + base
        }
        return name
    }
}
