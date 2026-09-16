import SwiftUI
import HSCore

/// The stub stage (strategy §2): play a recorded event log through the same RunPanel a real
/// run uses, without a phone, a GPU or 85 minutes.
struct ReplayView: View {
    @EnvironmentObject var model: AppModel
    @State private var files: [String] = []
    @State private var file: String = ""
    @State private var speed: Double = 50
    @State private var failAt = false
    @State private var failCount: Double = 40
    @State private var run: RunSession?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("Event Replay").font(.largeTitle.bold())
                Text("Plays a recorded `hs` event log (engine/tests/events) with its original timing, compressed.")
                    .foregroundStyle(.secondary)
                GroupBox {
                    VStack(alignment: .leading, spacing: 10) {
                        Picker("Recording", selection: $file) {
                            ForEach(files, id: \.self) { f in
                                Text((f as NSString).lastPathComponent).tag(f)
                            }
                        }
                        .frame(maxWidth: 420)
                        HStack {
                            Text("Speed ×\(Int(speed))").monospacedDigit().frame(width: 90, alignment: .leading)
                            Slider(value: $speed, in: 1...400).frame(width: 300)
                        }
                        HStack {
                            Toggle("Fail after", isOn: $failAt)
                            Stepper("\(Int(failCount)) events", value: $failCount, in: 1...5000, step: 10)
                                .disabled(!failAt)
                        }
                        Observing(run) { r in
                            Button(r.isRunning ? "Playing…" : "Play") { play() }
                                .disabled(file.isEmpty || r.isRunning || !model.config.problems.isEmpty)
                        }
                    }
                    .padding(4)
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                if let r = run {
                    RunPanel(session: r)
                }
            }
            .padding(24)
            .frame(maxWidth: 980, alignment: .leading)
        }
        .onAppear(perform: loadFiles)
    }

    private func loadFiles() {
        let dir = model.config.eventsDir
        let names = (try? FileManager.default.contentsOfDirectory(atPath: dir)) ?? []
        files = names.filter { $0.hasSuffix(".jsonl") }.sorted().map { (dir as NSString).appendingPathComponent($0) }
        if file.isEmpty || !files.contains(file) { file = files.first ?? "" }
    }

    private func play() {
        var args = ["replay", file, "--speed", String(Int(speed))]
        if failAt { args += ["--fail-at", String(Int(failCount))] }
        let s = model.session("Replay", args)
        run = s
        s.start()
    }
}
