import SwiftUI
import AppKit
import UniformTypeIdentifiers
import HSCore

/// Strategy §4.1: a clip from the phone (adb) or a dropped file → a new project.
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
                            TextField("e.g. body, head, coins (default: the clip's time)", text: $label)
                                .textFieldStyle(.roundedBorder)
                                .frame(maxWidth: 360)
                        }
                        HStack {
                            Text("Folder").frame(width: 80, alignment: .leading)
                            Text(targetFolder ?? "choose a clip first")
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
        GroupBox("Clip file") {
            VStack(spacing: 10) {
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
                            Text("Drop a VID_*_2x1.h4v here")
                        }
                        Button("Choose…") { chooseFile() }
                    }
                    .padding()
                }
                .frame(height: 170)
                .dropDestination(for: URL.self) { urls, _ in
                    guard let u = urls.first(where: { ["h4v", "mp4"].contains($0.pathExtension.lowercased()) }) else {
                        return false
                    }
                    droppedFile = u
                    return true
                } isTargeted: { dropTargeted = $0 }
                Text("The clip is copied into the project's source/ folder, MD5'd, and checked with ffprobe: one 3840×1080 stream tagged leia3d_layout=2x1. Anything else is refused.")
                    .font(.caption).foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .padding(4)
        }
    }

    private func chooseFile() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.directoryURL = URL(fileURLWithPath: model.config.repoRoot).appendingPathComponent("fixtures")
        if panel.runModal() == .OK, let u = panel.url { droppedFile = u }
    }

    // MARK: ingest

    private var chosenClipName: String? {
        switch source {
        case .phone: return selectedPhoneClip?.name
        case .file: return droppedFile?.lastPathComponent
        }
    }

    private var targetFolder: String? {
        guard let name = chosenClipName else { return nil }
        return store.uniqueFolder(ProjectStore.suggestedName(clip: name, label: label))
    }

    private func startIngest() {
        guard let folder = targetFolder else { return }
        var args = ["ingest", "-p", folder]
        switch source {
        case .phone:
            guard let c = selectedPhoneClip else { return }
            args += ["--phone", c.serial, "--remote", c.path]
        case .file:
            guard let f = droppedFile else { return }
            args += ["--clip", f.path]
        }
        let s = model.session("Ingest", args)
        s.onFinish = { run in
            store.reload()
            if run.succeeded { label = "" }
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
