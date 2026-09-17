import SwiftUI
import AppKit
import HSCore

/// A shell in the app: each command runs in `zsh -l` from the repository folder, with
/// `.venv/bin` and Homebrew on PATH, so the same commands work here as in Terminal.
struct ConsoleView: View {
    @EnvironmentObject var model: AppModel
    @State private var command = ""
    @FocusState private var focused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Console").font(.largeTitle.bold())
            Text("Runs in zsh from \(model.config.repoRoot). `hs` is on the PATH. Stop sends Ctrl-C to the command.")
                .font(.callout).foregroundStyle(.secondary)
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        ForEach(model.consoleHistory) { s in
                            ConsoleEntry(session: s).id(s.id)
                        }
                    }
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .background(RoundedRectangle(cornerRadius: 8).fill(Color(nsColor: .textBackgroundColor)))
                .onChange(of: model.consoleHistory.count) { _, _ in
                    if let last = model.consoleHistory.last { proxy.scrollTo(last.id, anchor: .bottom) }
                }
            }
            HStack {
                Text("$").font(.system(.body, design: .monospaced)).foregroundStyle(.secondary)
                TextField("hs tools", text: $command)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(.body, design: .monospaced))
                    .focused($focused)
                    .onSubmit(run)
                Button("Run", action: run).disabled(command.trimmingCharacters(in: .whitespaces).isEmpty)
                Button("Clear") { model.consoleHistory.removeAll { !$0.running } }
            }
        }
        .padding(24)
        .onAppear { focused = true }
    }

    private func run() {
        let c = command.trimmingCharacters(in: .whitespaces)
        guard !c.isEmpty else { return }
        let s = ConsoleSession(command: c)
        model.consoleHistory.append(s)
        s.start(config: model.config)
        command = ""
    }
}

struct ConsoleEntry: View {
    @ObservedObject var session: ConsoleSession
    @State private var follow = true

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("$ " + session.command).font(.system(.body, design: .monospaced).weight(.semibold))
                    .textSelection(.enabled)
                Spacer()
                if session.running {
                    ProgressView().controlSize(.mini)
                    Button("Stop") { session.interrupt() }.controlSize(.small)
                } else if let code = session.exitCode {
                    Text("exit \(code)").font(.caption.monospaced()).foregroundStyle(code == 0 ? Color.secondary : Color.red)
                }
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(session.lines.joined(separator: "\n"), forType: .string)
                } label: { Image(systemName: "doc.on.doc") }
                    .buttonStyle(.borderless).help("Copy output")
            }
            Text(session.lines.suffix(400).joined(separator: "\n"))
                .font(.system(.caption, design: .monospaced))
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
            if session.lines.count > 400 {
                Text("showing the last 400 of \(session.lines.count) lines — Copy output for all")
                    .font(.caption2).foregroundStyle(.secondary)
            }
        }
    }
}
