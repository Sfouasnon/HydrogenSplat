import SwiftUI
import AppKit
import HSCore

/// Strategy §4.0: where the engine is, and which tools it can see (`hs tools`).
struct SetupView: View {
    @EnvironmentObject var model: AppModel
    @State private var toolsRun: RunSession?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("Setup").font(.largeTitle.bold())
                GroupBox("Engine") {
                    VStack(alignment: .leading, spacing: 8) {
                        PathField(label: "Repository", path: $model.config.repoRoot, directory: true)
                        PathField(label: "hs executable", path: $model.config.hsPath, directory: false)
                        PathField(label: "Projects folder", path: $model.config.projectsRoot, directory: true)
                        HStack {
                            Text("Extra PATH").frame(width: 120, alignment: .leading)
                            TextField("", text: $model.config.extraPath)
                                .textFieldStyle(.roundedBorder)
                                .font(.system(.body, design: .monospaced))
                        }
                        Text("A Finder-launched app gets PATH=/usr/bin:/bin. These folders are put in front so the engine finds Homebrew's ffmpeg, ffprobe and adb.")
                            .font(.caption).foregroundStyle(.secondary)
                        ForEach(Array(model.config.problems.enumerated()), id: \.offset) { _, p in
                            problemRow(p)
                        }
                        HStack {
                            Button("Use this checkout's defaults") { model.resetConfig() }
                            Spacer()
                        }
                    }
                    .padding(4)
                }
                GroupBox {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Observing(toolsRun) { run in
                                Button(run.isRunning ? "Checking…" : "Check tools") { runTools() }
                                    .disabled(run.isRunning || !model.config.problems.filter(isHsProblem).isEmpty)
                            }
                            Text("runs `hs tools`: python packages, brush, brush-path-render, ffmpeg, adb")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        if let run = toolsRun {
                            ToolsResult(session: run)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(4)
                } label: {
                    Text("Tools")
                }
                GroupBox("Keeping the Mac awake") {
                    Text("Every stage holds `caffeinate -d -i -m -s` while it runs. On battery macOS still sleeps, and closing the lid always does — plug in and leave the lid open for training. Sleeps are detected and reported by the engine.")
                        .font(.callout).foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(4)
                }
            }
            .padding(24)
            .frame(maxWidth: 900, alignment: .leading)
        }
        .onAppear { if toolsRun == nil && model.config.problems.isEmpty { runTools() } }
    }

    private func isHsProblem(_ p: EngineConfig.Problem) -> Bool {
        if case .hsMissing = p { return true }
        return false
    }

    @ViewBuilder private func problemRow(_ p: EngineConfig.Problem) -> some View {
        switch p {
        case .repoMissing(let path):
            CheckRow(name: "repository", ok: false, value: "no engine/hs/cli.py under \(path)")
        case .hsMissing(let path):
            VStack(alignment: .leading, spacing: 4) {
                CheckRow(name: "hs", ok: false, value: "not executable: \(path)")
                CopyableCommand(text: "cd \(shellQuote(model.config.repoRoot)) && python3 -m venv .venv && .venv/bin/pip install -e engine")
            }
        }
    }

    private func runTools() {
        let s = model.session("hs tools", ["tools"])
        toolsRun = s
        s.start()
    }
}

struct ToolsResult: View {
    @ObservedObject var session: RunSession

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            if case .failedToStart(let why) = session.state {
                CheckRow(name: "hs", ok: false, value: why)
            }
            ForEach(session.checks) { c in
                CheckRow(name: c.name ?? "?", ok: c.ok ?? false, value: c.value?.display)
            }
            if let py = session.metric("tools", "python_exe")?.string {
                Text("python: \(py)").font(.caption).foregroundStyle(.secondary)
            }
            if case .finished(let code) = session.state, code != 0, session.checks.isEmpty {
                Text(session.stderrTail.suffix(2000)).font(.caption.monospaced()).textSelection(.enabled)
            }
        }
    }
}

struct PathField: View {
    let label: String
    @Binding var path: String
    let directory: Bool

    var body: some View {
        HStack {
            Text(label).frame(width: 120, alignment: .leading)
            TextField("", text: $path)
                .textFieldStyle(.roundedBorder)
                .font(.system(.body, design: .monospaced))
            Button("Choose…") {
                let panel = NSOpenPanel()
                panel.canChooseDirectories = directory
                panel.canChooseFiles = !directory
                panel.allowsMultipleSelection = false
                panel.directoryURL = URL(fileURLWithPath: (path as NSString).deletingLastPathComponent)
                if panel.runModal() == .OK, let url = panel.url {
                    path = url.path
                }
            }
            Button {
                NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
            } label: { Image(systemName: "arrow.right.circle") }
                .buttonStyle(.borderless)
                .help("Reveal in Finder")
        }
    }
}

struct CopyableCommand: View {
    let text: String

    var body: some View {
        HStack(alignment: .top) {
            Text(text)
                .font(.system(.callout, design: .monospaced))
                .textSelection(.enabled)
                .padding(6)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(RoundedRectangle(cornerRadius: 6).fill(Color.secondary.opacity(0.12)))
            Button {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(text, forType: .string)
            } label: { Image(systemName: "doc.on.doc") }
                .buttonStyle(.borderless)
                .help("Copy")
        }
    }
}

func shellQuote(_ s: String) -> String {
    s.contains(where: { " '\"$\\".contains($0) }) ? "'" + s.replacingOccurrences(of: "'", with: "'\\''") + "'" : s
}
