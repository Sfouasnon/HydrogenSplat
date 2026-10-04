import SwiftUI
import AppKit
import UniformTypeIdentifiers
import HSCore

/// Strategy §4.1: a clip from the phone (adb), or a dropped file or folder → a new project.
///
/// File takes every source the engine does: a Hydrogen clip, any other camera's video, a folder of
/// photographs, a RED array. The page does not decide which: `hs source PATH` does, the page shows
/// what it said (SourceProbe) and the few choices that kind has, and Ingest runs the command the
/// report gives. So what is refused here is refused for the engine's reason, in its words.
struct IngestView: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var store: ProjectStore

    enum Source: String, CaseIterable, Identifiable {
        case phone = "Phone"
        case file = "File"
        var id: String { rawValue }
    }

    @State private var source: Source = .phone
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

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("New Project").font(.largeTitle.bold())
                Picker("Source", selection: $source) {
                    ForEach(Source.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                .frame(width: 260)

                switch source {
                case .phone: phoneSection
                case .file: fileSection
                }

                GroupBox("Project") {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Label").frame(width: 80, alignment: .leading)
                            TextField("e.g. body, head, coins (default: the source's own name or time)", text: $label)
                                .textFieldStyle(.roundedBorder)
                                .frame(maxWidth: 360)
                        }
                        HStack {
                            Text("Folder").frame(width: 80, alignment: .leading)
                            Text(targetFolder ?? "choose a source first")
                                .font(.system(.body, design: .monospaced))
                                .foregroundStyle(targetFolder == nil ? .secondary : .primary)
                                .textSelection(.enabled)
                        }
                        if let name = chosenClipName, let existing = store.ingestedClips[name] {
                            Label("\(name) is already in project \(existing)", systemImage: "exclamationmark.triangle")
                                .foregroundStyle(.orange)
                        }
                        Observing(ingestRun) { run in
                            HStack {
                                Button(run.isRunning ? "Ingesting…" : "Ingest") { startIngest() }
                                    .keyboardShortcut(.defaultAction)
                                    .disabled(targetFolder == nil || run.isRunning || !model.config.problems.isEmpty)
                                if !model.config.problems.isEmpty {
                                    Text("Fix Setup first").foregroundStyle(.red)
                                }
                                Spacer()
                                if let t = ingestTarget, run.succeeded {
                                    Button("Open Project") { model.selection = .project(t) }
                                }
                            }
                        }
                    }
                    .padding(4)
                }

                if let run = ingestRun {
                    RunPanel(session: run)
                }
            }
            .padding(24)
            .frame(maxWidth: 980, alignment: .leading)
        }
        .onAppear {
            if phoneRun == nil && source == .phone && model.config.problems.isEmpty { refreshPhone() }
        }
    }

    // MARK: phone

    @ViewBuilder private var phoneSection: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Observing(phoneRun) { run in
                        Button(run.isRunning ? "Looking…" : "Refresh") { refreshPhone() }
                            .disabled(run.isRunning || !model.config.problems.isEmpty)
                    }
                    Text("USB or wireless adb; the phone must allow USB debugging.")
                        .font(.caption).foregroundStyle(.secondary)
                    Spacer()
                }
                if let run = phoneRun {
                    PhoneRunStatus(session: run)
                }
                ForEach(devices) { d in
                    Text("\(d.model ?? d.product ?? "device") · \(d.serial) · \(d.state)")
                        .font(.subheadline.weight(.medium))
                        .foregroundStyle(d.ready ? Color.primary : Color.orange)
                    if d.ready && d.clips.isEmpty {
                        Text("No VID_*_2x1.h4v clips in /sdcard/DCIM/Camera").foregroundStyle(.secondary)
                    }
                }
                let clips = devices.flatMap { $0.clips }
                if !clips.isEmpty {
                    Table(clips, selection: $selectedClip) {
                        TableColumn("Clip") { c in
                            HStack(spacing: 4) {
                                Text(c.name).font(.system(.body, design: .monospaced))
                                if let p = store.ingestedClips[c.name] {
                                    Text("in \(p)").font(.caption).foregroundStyle(.secondary)
                                }
                            }
                        }
                        TableColumn("Recorded") { c in Text(c.mtime).monospacedDigit() }
                            .width(140)
                        TableColumn("Size") { c in Text(Format.bytes(Double(c.bytes))).monospacedDigit() }
                            .width(80)
                        TableColumn("Phone") { c in Text(c.serial).foregroundStyle(.secondary) }
                            .width(90)
                    }
                    .frame(height: min(40 + CGFloat(clips.count) * 24, 260))
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(4)
        } label: {
            Text("Clips on the phone")
        }
    }

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

    @ViewBuilder private var fileSection: some View {
        GroupBox("File or folder") {
            VStack(alignment: .leading, spacing: 10) {
                ZStack {
                    RoundedRectangle(cornerRadius: 10)
                        .strokeBorder(style: StrokeStyle(lineWidth: 2, dash: [6]))
                        .foregroundStyle(dropTargeted ? Color.accentColor : Color.secondary.opacity(0.5))
                    VStack(spacing: 6) {
                        Image(systemName: "film.stack").font(.largeTitle).foregroundStyle(.secondary)
                        if let f = droppedFile {
                            Text(f.lastPathComponent).font(.system(.body, design: .monospaced))
                            Text(f.deletingLastPathComponent().path).font(.caption).foregroundStyle(.secondary)
                        } else {
                            Text("Drop a clip, a folder of photographs, or a RED media folder here")
                        }
                        Button("Choose…") { chooseFile() }
                    }
                    .padding()
                }
                .frame(height: 170)
                .dropDestination(for: URL.self) { urls, _ in
                    guard let u = urls.first else { return false }
                    look(at: u)
                    return true
                } isTargeted: { dropTargeted = $0 }
                Observing(probeRun) { run in
                    probeStatus(run)
                }
                if let p = probe {
                    found(p)
                }
                acceptedMedia
            }
            .padding(4)
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

    /// What the engine said the dropped thing is, and the choices that kind has.
    private func found(_ p: SourceProbe) -> some View {
        let blocker = p.blocker(options)
        return VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Image(systemName: blocker == nil ? "checkmark.circle.fill" : "xmark.circle.fill")
                    .foregroundStyle(blocker == nil ? Color.green : Color.orange)
                Text(p.title).font(.headline)
                Text(p.summary).foregroundStyle(.secondary).textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
            }
            ForEach(p.problems, id: \.self) { w in
                Label(w, systemImage: "exclamationmark.triangle")
                    .font(.callout).foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let b = blocker, !p.problems.contains(b) {
                Label(b, systemImage: "exclamationmark.triangle")
                    .font(.callout).foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
            ForEach(p.notes, id: \.self) { n in
                Text(n).font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if p.offersStillsKind {
                stillsChoice(p)
            }
            if p.kind == SourceProbe.Kind.r3d, !p.takes.isEmpty {
                takeChoice(p)
            }
        }
    }

    /// Which cameras the chosen take has, so a take swept in from the wrong folder is seen.
    private func takeCameras(_ p: SourceProbe) -> String? {
        guard let t = p.chosenTake(options), !t.cameras.isEmpty else { return nil }
        return "Cameras: " + t.cameras.joined(separator: " ")
    }

    private func stillsChoice(_ p: SourceProbe) -> some View {
        Handle(title: "These are",
               help: "One camera: photographs taken while moving round the subject. One per camera: the same instant from each camera of a fixed array. The engine reads it off the file names (\(p.stillsWhy ?? "no reason given")); change it here if it read them wrong.") {
            Picker("", selection: $options.stillsKind) {
                Text("As detected (\(p.detectedStillsTitle))").tag(SourceProbe.StillsKind.auto)
                Text("One camera").tag(SourceProbe.StillsKind.mono)
                Text("One per camera").tag(SourceProbe.StillsKind.array)
            }
            .labelsHidden().pickerStyle(.segmented).fixedSize()
        }
    }

    private func takeChoice(_ p: SourceProbe) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Handle(title: "Take",
                   help: "One take becomes one project: the first frame of that take from every camera. A take needs three cameras or more.") {
                Picker("", selection: $options.take) {
                    ForEach(p.takes) { t in
                        Text(t.label).tag(t.take)
                    }
                }
                .labelsHidden().frame(width: 260)
            }
            if let c = takeCameras(p) {
                Text(c).font(.caption.monospaced()).foregroundStyle(.secondary)
                    .padding(.leading, 140).textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Handle(title: "Frame size",
                   help: "REDline renders each frame at this fraction of the recorded size (half of 5760×3240 is 2880×1620). Full is slower to render, solve and train.") {
                Picker("", selection: $options.res) {
                    Text("Full").tag(1)
                    Text("Half").tag(2)
                    Text("Quarter").tag(4)
                }
                .labelsHidden().pickerStyle(.segmented).fixedSize()
            }
        }
    }

    private var acceptedMedia: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("What can be dropped here").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            ForEach(SourceProbe.acceptedMedia) { m in
                (Text(m.name).fontWeight(.semibold) + Text(" — " + m.what))
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Text(SourceProbe.notAccepted).font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

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

    /// The clip's file name when the source is a clip: what says "already in a project".
    private var chosenClipName: String? {
        switch source {
        case .phone:
            return selectedPhoneClip?.name
        case .file:
            guard let p = probe, p.kind == SourceProbe.Kind.stereo || p.kind == SourceProbe.Kind.video else { return nil }
            return p.name
        }
    }

    /// Where the project will go, or nil while there is nothing the engine would take.
    private var targetFolder: String? {
        switch source {
        case .phone:
            guard let name = selectedPhoneClip?.name else { return nil }
            return store.uniqueFolder(ProjectStore.suggestedName(clip: name, label: label))
        case .file:
            guard let p = probe, p.blocker(options) == nil else { return nil }
            return store.uniqueFolder(p.projectName(label: label, options: options))
        }
    }

    private func startIngest() {
        guard let folder = targetFolder else { return }
        var args = ["ingest", "-p", folder]
        switch source {
        case .phone:
            guard let c = selectedPhoneClip else { return }
            args += ["--phone", c.serial, "--remote", c.path]
        case .file:
            guard let p = probe else { return }
            args = p.ingestArguments(project: folder, options: options)
            guard !args.isEmpty else { return }
        }
        let s = model.session("Ingest", args)
        let ingested: URL? = source == .file ? droppedFile : nil
        s.onFinish = { run in
            store.reload()
            if run.succeeded {
                label = ""
                if let f = ingested, droppedFile == f {
                    // it is in a project now: Return must not bring the same thing in again as "-2"
                    // (something else dropped meanwhile is left alone)
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

struct PhoneRunStatus: View {
    @ObservedObject var session: RunSession

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            if case .failedToStart(let why) = session.state {
                CheckRow(name: "hs phone", ok: false, value: why)
            }
            ForEach(session.checks) { c in
                CheckRow(name: c.name ?? "?", ok: c.ok ?? false, value: c.value?.display)
            }
            ForEach(session.errors) { e in
                CheckRow(name: "error", ok: false, value: [e.message, e.hint].compactMap { $0 }.joined(separator: " — "))
            }
        }
    }
}
