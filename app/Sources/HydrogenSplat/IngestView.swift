import SwiftUI
import AppKit
import UniformTypeIdentifiers
import HSCore

/// The pieces of the Footage step that find footage: the drop zone for a clip, a folder of
/// photographs or a RED media folder, and the list of clips on a connected phone (adb).
/// FootagePage owns the state and runs the commands; these only draw it.

/// One drop zone. `hs source PATH` decides what the dropped thing is; the page shows what it said.
struct FootageDropZone: View {
    let dropped: URL?
    @Binding var targeted: Bool
    let choose: () -> Void
    let onDrop: (URL) -> Void

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 12)
                .strokeBorder(style: StrokeStyle(lineWidth: 2, dash: [6]))
                .foregroundStyle(targeted ? Brand.accent : Color.secondary.opacity(0.5))
            VStack(spacing: 8) {
                Image(systemName: "film.stack").font(.largeTitle).foregroundStyle(.secondary)
                if let f = dropped {
                    Text(f.lastPathComponent).font(.system(.body, design: .monospaced))
                    Text(f.deletingLastPathComponent().path).font(.caption).foregroundStyle(.secondary)
                } else {
                    Text("Drop a clip, a folder of photographs, or a RED media folder here").font(.headline)
                    Text("Or pick a clip from a connected phone below.").font(.caption).foregroundStyle(.secondary)
                }
                Button("Choose…", action: choose)
            }
            .padding()
        }
        .frame(height: 160)
        .dropDestination(for: URL.self) { urls, _ in
            guard let u = urls.first else { return false }
            onDrop(u)
            return true
        } isTargeted: { targeted = $0 }
    }
}

/// The clips on every phone adb can see, one selectable row each.
struct PhoneClipsSection: View {
    @EnvironmentObject var model: AppModel
    let devices: [PhoneDevice]
    @Binding var selectedClip: PhoneClip.ID?
    let run: RunSession?
    /// Clip name → project folder, for clips already brought in.
    let ingested: [String: String]
    let refresh: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("From a connected phone").font(.headline)
                Observing(run) { r in
                    Button(r.isRunning ? "Looking…" : "Look again") { refresh() }
                        .disabled(r.isRunning || !model.config.problems.isEmpty)
                        .controlSize(.small)
                }
                Text("USB or wireless adb; the phone must allow USB debugging.")
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
            }
            if let r = run {
                PhoneRunStatus(session: r)
            }
            ForEach(devices) { d in
                Text("\(d.model ?? d.product ?? "device") · \(d.serial) · \(d.state)")
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(d.ready ? Color.primary : Color.orange)
                if d.ready && d.clips.isEmpty {
                    Text("No 3D clips in the phone's camera folder.").foregroundStyle(.secondary)
                }
            }
            let clips = devices.flatMap { $0.clips }
            if !clips.isEmpty {
                Table(clips, selection: $selectedClip) {
                    TableColumn("Clip") { c in
                        HStack(spacing: 4) {
                            Text(c.name).font(.system(.body, design: .monospaced))
                            if let p = ingested[c.name] {
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
