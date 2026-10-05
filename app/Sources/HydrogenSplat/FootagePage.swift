import SwiftUI
import AppKit
import HSCore

/// Step 1 — Footage. One page in two states: with no project (New Project in the sidebar) it takes
/// footage and makes the project; on an existing project it states what the footage is. Both show
/// the four facts later steps turn on — kind, useful size, exposure, lens — and the subject tiles.
struct FootagePage: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var store: ProjectStore
    /// nil: New Project. A project whose ingest has not run yet is the same state with a folder.
    let project: ProjectSummary?

    @State private var phoneRun: RunSession?
    @State private var devices: [PhoneDevice] = []
    @State private var selectedClip: PhoneClip.ID?
    @State private var droppedFile: URL?
    @State private var dropTargeted = false
    /// `hs source` on the dropped path, and what it reported (nil while it runs, or if it failed).
    @State private var probeRun: RunSession?
    @State private var probe: SourceProbe?
    @State private var options = SourceProbe.Options()
    @State private var label = ""
    @State private var ingestRun: RunSession?
    @State private var ingestTarget: String?

    init(project: ProjectSummary?) {
        self.project = project
    }

    /// The project folder the subject choice is stored under: the project's, or the folder the
    /// new project will get.
    private var subjectKey: String? { project?.path ?? targetFolder }

    private var manifest: Manifest? { project?.manifest }
    private var ingested: Bool { manifest?.stage("ingest")?.status == .done }

    var body: some View {
        StepPage {
            StepHeader(title: "Footage",
                       lead: ingested ? "What this project was made from, and what that decides later."
                                      : "Drop the footage. The engine says what it is; you say what the subject is.")
            if ingested, let m = manifest {
                existing(m)
            } else {
                fresh
            }
        }
        .onAppear {
            if !ingested && phoneRun == nil && model.config.problems.isEmpty { refreshPhone() }
        }
    }

    // MARK: an existing project

    @ViewBuilder private func existing(_ m: Manifest) -> some View {
        VerdictCard(.good, headline: existingHeadline(m), detail: m.originalPath ?? m.clipPath) {
            factChips(kind: existingKind(m), size: existingSize(m),
                      exposure: existingExposure(m),
                      lens: existingLens(m))
        }
        subjectCard
        StepFooter(primary: "Go to Frames", secondary: "Start another project from new footage",
                   note: "The footage stays as it is; the next step picks frames from it.") {
            if let p = project { model.pipelineStage[p.path] = .frames }
        } secondaryAction: {
            model.selection = .ingest
        }
    }

    private func existingHeadline(_ m: Manifest) -> String {
        if m.sourceKind == "array" {
            return "\(m.cameras.count) cameras, one frame each."
        }
        if m.framesOnly {
            return "\(m.cameras.count) photographs from one camera."
        }
        let frames = m.probe["nb_frames"]?.display
        let dur = Format.duration(m.probe["duration_s"]?.double)
        let what = m.isArray ? "A clip from one camera" : "A 3D clip from the phone"
        if let f = frames { return "\(what): \(f) frames, \(dur)." }
        return "\(what)" + (m.clipName.map { ": \($0)." } ?? ".")
    }

    /// Whether the frames agree on brightness, once Look has measured it (`hs exposure --analyze`).
    /// A clip's file does not say whether exposure was locked, so before that there is no fact.
    private func existingExposure(_ m: Manifest) -> String? {
        guard let drift = m.stage("exposure")?.metrics["drift_stops"]?.double else { return nil }
        if drift < 0.5 { return "steady" }
        return "varies by " + String(format: "%.1f", drift) + " stops"
    }

    private func existingKind(_ m: Manifest) -> String {
        if m.sourceKind == "array" { return "camera array" }
        if m.framesOnly { return "photographs" }
        return m.isArray ? "video, one camera" : "3D clip, two eyes"
    }

    private func existingSize(_ m: Manifest) -> String? {
        guard let w = m.probe["width"]?.int, let h = m.probe["height"]?.int else { return nil }
        var s = "\(w)×\(h)"
        if let fps = m.probe["fps"]?.display, !m.framesOnly { s += " · \(fps) fps" }
        if let c = m.probe["codec"]?.string { s += " · \(c)" }
        return s
    }

    /// The lens: a stereo profile is a calibration; a solve that used a lens profile records it.
    private func existingLens(_ m: Manifest) -> String? {
        if let p = m.stage("solve")?.metrics["lens_profile"]?.string { return "calibrated · \(p)" }
        if let p = m.stage("calibrate")?.metrics["profile"]?.string { return "calibrated · \(p)" }
        if !m.isArray, let p = m.profileID { return "calibrated · \(p)" }
        return nil
    }

    // MARK: a new project

    @ViewBuilder private var fresh: some View {
        FootageDropZone(dropped: droppedFile, targeted: $dropTargeted, choose: { chooseFile() }, onDrop: { look(at: $0) })
        Observing(probeRun) { run in probeStatus(run) }
        if let p = probe {
            found(p)
        }
        PhoneClipsSection(devices: devices, selectedClip: $selectedClip, run: phoneRun,
                          ingested: store.ingestedClips, refresh: { refreshPhone() })
        subjectCard
        projectCard
        if let run = ingestRun {
            RunPanel(session: run)
        }
    }

    /// While `hs source` runs, and when it could not run or stopped with an error.
    @ViewBuilder private func probeStatus(_ run: RunSession) -> some View {
        if run.isRunning {
            HStack(spacing: 6) {
                ProgressView().controlSize(.small)
                Text("Reading it…").foregroundStyle(.secondary)
            }
        } else if case .failedToStart(let why) = run.state {
            CheckRow(name: "hs source", ok: false, value: why)
        } else {
            ForEach(run.errors) { e in
                CheckRow(name: "error", ok: false, value: [e.message, e.hint].compactMap { $0 }.joined(separator: " — "))
            }
        }
    }

    /// What the engine said the dropped thing is, as the four facts, with the choices that kind has.
    private func found(_ p: SourceProbe) -> some View {
        let blocker = p.blocker(options)
        let verdict: Verdict = blocker == nil ? .good : .blocked
        return VerdictCard(verdict, headline: p.title + (p.summary.isEmpty ? "" : ": " + p.summary),
                           detail: blocker ?? p.problems.first) {
            factChips(kind: p.title, size: nil, exposure: nil, lens: nil)
            ForEach(p.problems.filter { $0 != blocker }, id: \.self) { w in
                Label(w, systemImage: "exclamationmark.triangle").font(.callout).foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
            ForEach(p.notes, id: \.self) { n in
                Text(n).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            if p.offersStillsKind {
                HStack(spacing: 10) {
                    Text("These are").frame(width: 90, alignment: .leading)
                    Picker("", selection: $options.stillsKind) {
                        Text("As detected (\(p.detectedStillsTitle))").tag(SourceProbe.StillsKind.auto)
                        Text("One camera").tag(SourceProbe.StillsKind.mono)
                        Text("One per camera").tag(SourceProbe.StillsKind.array)
                    }
                    .labelsHidden().pickerStyle(.segmented).fixedSize()
                }
                Text("One camera moved round the subject, or one frame from each camera of a fixed array.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if p.kind == SourceProbe.Kind.r3d, !p.takes.isEmpty {
                takeChoice(p)
            }
        }
    }

    private func takeChoice(_ p: SourceProbe) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                Text("Take").frame(width: 90, alignment: .leading)
                Picker("", selection: $options.take) {
                    ForEach(p.takes) { t in Text(t.label).tag(t.take) }
                }
                .labelsHidden().frame(width: 260)
            }
            if let t = p.chosenTake(options), !t.cameras.isEmpty {
                Text("Cameras: " + t.cameras.joined(separator: " "))
                    .font(.caption.monospaced()).foregroundStyle(.secondary).textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack(spacing: 10) {
                Text("Frame size").frame(width: 90, alignment: .leading)
                Picker("", selection: $options.res) {
                    Text("Full").tag(1)
                    Text("Half").tag(2)
                    Text("Quarter").tag(4)
                }
                .labelsHidden().pickerStyle(.segmented).fixedSize()
            }
            Text("One take becomes one project; full size is slower to render, place and train.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    /// The four facts. A fact the engine has not produced reads "not known yet".
    private func factChips(kind: String?, size: String?, exposure: String?, lens: String?) -> some View {
        HStack(spacing: 8) {
            MetricChip(text: "Kind: " + (kind ?? "not known yet"))
            MetricChip(text: "Size: " + (size ?? "not known yet"))
            MetricChip(text: "Exposure: " + (exposure ?? "measured in Look"), tint: exposure == nil ? nil : Brand.accent)
            MetricChip(text: "Lens: " + (lens ?? "not known yet"), tint: lens == nil ? nil : Brand.accent)
        }
    }

    // MARK: the subject

    private var subject: SubjectKind? {
        guard let k = subjectKey else { return nil }
        if let raw = model.subjectKind[k], let s = SubjectKind(rawValue: raw) { return s }
        if let raw = manifest?.raw["project"]?["subject_kind"]?.string ?? manifest?.raw["source"]?["subject"]?.string,
           let s = SubjectKind(rawValue: raw) { return s }
        return nil
    }

    private var subjectCard: some View {
        VerdictCard(.info, headline: "What is the subject?",
                    detail: "This picks the outlines, the training recipe and what the Look step recommends.") {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 160), spacing: 10)], spacing: 10) {
                ForEach(SubjectKind.allCases) { k in
                    ChoiceTile(title: k.title, subtitle: k.meaning, selected: subject == k) {
                        if let key = subjectKey {
                            model.subjectKind[key] = k.rawValue
                        }
                    }
                    .disabled(subjectKey == nil)
                }
            }
            if subjectKey == nil {
                Text("Drop footage first; the choice is kept with the project.").font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    // MARK: the project folder and the one action

    private var projectCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Name").frame(width: 90, alignment: .leading)
                TextField("a word for this project (default: the footage's own name or time)", text: $label)
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 420)
            }
            HStack {
                Text("Folder").frame(width: 90, alignment: .leading)
                Text(targetFolder ?? "drop footage or pick a clip first")
                    .font(.system(.body, design: .monospaced))
                    .foregroundStyle(targetFolder == nil ? .secondary : .primary)
                    .textSelection(.enabled)
            }
            if let name = chosenClipName, let existing = store.ingestedClips[name] {
                Label("\(name) is already in project \(existing)", systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.orange)
            }
            Observing(ingestRun) { run in
                StepFooter(primary: run.isRunning ? "Making the project…" : "Make the project",
                           primaryEnabled: targetFolder != nil && !run.isRunning && model.config.problems.isEmpty,
                           secondary: (ingestTarget != nil && run.succeeded) ? "Open the project" : nil,
                           note: model.config.problems.isEmpty ? "Copies the footage in and reads it; nothing else runs yet." : "Fix Settings first.") {
                    startIngest()
                } secondaryAction: {
                    if let t = ingestTarget { model.selection = .project(t) }
                }
            }
        }
    }

    // MARK: phone

    private var selectedPhoneClip: PhoneClip? {
        guard let id = selectedClip else { return nil }
        return devices.flatMap { $0.clips }.first { $0.id == id }
    }

    private func refreshPhone() {
        let s = model.session("hs phone", ["phone"])
        s.onFinish = { run in
            devices = PhoneListing.devices(from: run.events)
            if selectedPhoneClip == nil {
                // newest clip that is not in a project yet
                let ingested = store.ingestedClips
                selectedClip = devices.flatMap { $0.clips }.first { ingested[$0.name] == nil }?.id
            }
        }
        phoneRun = s
        s.start()
    }

    // MARK: file

    private func chooseFile() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.message = "A clip, a folder of photographs, or a RED media folder"
        if panel.runModal() == .OK, let u = panel.url { look(at: u) }
    }

    /// Ask the engine what this is (`hs source PATH`: nothing is written, no project is needed).
    private func look(at url: URL) {
        probeRun?.cancel()                 // an earlier look still running is of no use now
        probeRun = nil
        droppedFile = url
        probe = nil
        selectedClip = nil                 // the dropped thing is the source now
        options = SourceProbe.Options()
        guard model.config.problems.isEmpty else { return }
        let s = model.session("hs source", SourceProbe.arguments(path: url.path))
        s.onFinish = { run in
            // something else was dropped meanwhile, or this look was replaced by a newer one
            guard droppedFile == url, probeRun === run else { return }
            let p = SourceProbe.from(events: run.events)
            probe = p
            options.take = p?.take ?? p?.takes.first?.take ?? ""
        }
        probeRun = s
        s.start()
    }

    // MARK: ingest

    /// A dropped file wins over a phone clip; the phone clip is used when nothing was dropped.
    private var usingPhone: Bool { droppedFile == nil && selectedPhoneClip != nil }

    /// The clip's file name when the source is a clip: what says "already in a project".
    private var chosenClipName: String? {
        if usingPhone { return selectedPhoneClip?.name }
        guard let p = probe, p.kind == SourceProbe.Kind.stereo || p.kind == SourceProbe.Kind.video else { return nil }
        return p.name
    }

    /// Where the project will go, or nil while there is nothing the engine would take.
    private var targetFolder: String? {
        if let p = project { return p.path }
        if usingPhone {
            guard let name = selectedPhoneClip?.name else { return nil }
            return store.uniqueFolder(ProjectStore.suggestedName(clip: name, label: label))
        }
        guard let p = probe, p.blocker(options) == nil else { return nil }
        return store.uniqueFolder(p.projectName(label: label, options: options))
    }

    private func startIngest() {
        guard let folder = targetFolder else { return }
        var args = ["ingest", "-p", folder]
        if usingPhone {
            guard let c = selectedPhoneClip else { return }
            args += ["--phone", c.serial, "--remote", c.path]
        } else {
            guard let p = probe else { return }
            args = p.ingestArguments(project: folder, options: options)
            guard !args.isEmpty else { return }
        }
        // the subject chosen above travels with the folder; AppModel.syncSubjectKind stores it in
        // the manifest when the project's page opens
        let chosen = subject
        let s = model.session("Ingest", args)
        let ingested: URL? = usingPhone ? nil : droppedFile
        s.onFinish = { run in
            store.reload()
            if run.succeeded {
                if let k = chosen { model.subjectKind[folder] = k.rawValue }
                label = ""
                if let f = ingested, droppedFile == f {
                    // it is in a project now: Return must not bring the same thing in again as "-2"
                    droppedFile = nil
                    probe = nil
                    probeRun = nil
                }
            }
        }
        ingestTarget = folder
        ingestRun = s
        model.projectRuns[folder] = s
        s.start()
    }
}
