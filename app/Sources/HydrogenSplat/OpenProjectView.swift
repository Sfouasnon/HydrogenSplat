import SwiftUI
import AppKit
import HSCore

/// The list of projects: what a window with no project shows, and File › Open Project… (⌘O).
/// Every project in the Projects folder, newest first; one click opens it and the window works in
/// that project until another is opened. A project with a run going says so, here and in the
/// strip under every page, so a run in a project that is not open is never out of sight.
struct OpenProjectView: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var store: ProjectStore
    @State private var filter = ""

    private var shown: [ProjectSummary] {
        let f = filter.trimmingCharacters(in: .whitespaces).lowercased()
        if f.isEmpty { return store.projects }
        return store.projects.filter {
            $0.displayName.lowercased().contains(f) || ($0.manifest?.clipName?.lowercased().contains(f) ?? false)
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            BrandHeader()
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                Text("Open a project").font(.system(size: 28, weight: .bold))
                Spacer(minLength: 12)
                Button { store.reload() } label: { Label("Reload", systemImage: "arrow.clockwise") }
                    .help("Read the Projects folder again")
                Button { model.selection = .ingest } label: { Label("New Project…", systemImage: "plus") }
                    .buttonStyle(.borderedProminent)
            }
            HStack(spacing: 8) {
                Text(store.root).font(.caption.monospaced()).foregroundStyle(.secondary)
                    .lineLimit(1).truncationMode(.middle).textSelection(.enabled)
                Button("Change…") { model.selection = .setup }
                    .buttonStyle(.link).font(.caption)
                    .help("The Projects folder is set in Settings")
            }
            if let e = store.lastError {
                // an unreadable Projects folder used to look like "no projects"
                Label(e, systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if !model.config.problems.isEmpty {
                Label("Settings needs a look before anything can run.", systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.orange)
            }
            if store.projects.count > 8 {
                TextField("Find a project", text: $filter)
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 320)
            }
            if store.projects.isEmpty {
                ContentUnavailableView("No projects yet", systemImage: "folder",
                                       description: Text("New Project makes one from your footage."))
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    LazyVStack(spacing: 6) {
                        ForEach(shown) { p in
                            OpenProjectRow(project: p, run: model.projectRuns[p.path],
                                           isOpen: model.openProject == p.path) {
                                model.selection = .project(p.path)
                            }
                        }
                        if shown.isEmpty {
                            Text("No project matches “\(filter)”.").foregroundStyle(.secondary).padding(.top, 12)
                        }
                    }
                    .padding(.bottom, 12)
                }
            }
        }
        .padding(.horizontal, 28).padding(.top, 22)
        .frame(maxWidth: 820, maxHeight: .infinity, alignment: .topLeading)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    }
}

/// One project in the list: its name, where it has got to, the footage it was made from and the
/// day it was made.
struct OpenProjectRow: View {
    let project: ProjectSummary
    @ObservedObject var run: RunSession
    let isOpen: Bool
    let open: () -> Void
    @State private var hover = false

    init(project: ProjectSummary, run: RunSession?, isOpen: Bool, open: @escaping () -> Void) {
        self.project = project
        self._run = ObservedObject(wrappedValue: run ?? RunSession.placeholder)
        self.isOpen = isOpen
        self.open = open
    }

    var body: some View {
        Button(action: open) {
            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 8) {
                        Text(project.displayName).font(.headline).lineLimit(1)
                        if isOpen {
                            Text("open").font(.caption2.weight(.semibold))
                                .padding(.horizontal, 6).padding(.vertical, 1)
                                .background(Capsule().fill(Brand.accent.opacity(0.18)))
                                .foregroundStyle(Brand.accent)
                        }
                    }
                    HStack(spacing: 6) {
                        state
                        ForEach(facts, id: \.self) { f in
                            Text("·").foregroundStyle(.tertiary)
                            Text(f).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
                        }
                    }
                    .font(.callout)
                }
                Spacer(minLength: 8)
                Image(systemName: "chevron.right").font(.caption.weight(.semibold)).foregroundStyle(.tertiary)
            }
            .padding(.horizontal, 14).padding(.vertical, 10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(RoundedRectangle(cornerRadius: 10)
                .fill(hover ? Color.secondary.opacity(0.14) : Color(nsColor: .controlBackgroundColor)))
            .overlay(RoundedRectangle(cornerRadius: 10)
                .stroke(isOpen ? Brand.accent.opacity(0.7) : Color(nsColor: .separatorColor)))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hover = $0 }
        .help(project.path)
    }

    /// Where the project has got to: a live run first, then the furthest step that is done.
    @ViewBuilder private var state: some View {
        if run.isRunning {
            ProgressView().controlSize(.mini)
            Text(run.liveProgress.short).monospacedDigit().foregroundStyle(.secondary).lineLimit(1)
        } else if let lock = project.lock, lock.alive {
            Image(systemName: "lock.fill").font(.caption2).foregroundStyle(.secondary)
            Text("\(lock.stage ?? "a stage") is running").foregroundStyle(.secondary)
        } else if let m = project.manifest {
            Text(m.lastDone.map { "done through \(PipelineStage.containing($0)?.title ?? $0)" } ?? "nothing done yet")
                .foregroundStyle(.secondary)
        } else {
            Text(project.manifestError ?? "manifest.json could not be read").foregroundStyle(.red)
        }
    }

    private var facts: [String] {
        var out: [String] = []
        if let c = project.manifest?.clipName { out.append(c) }
        if let d = project.created { out.append(d.formatted(date: .abbreviated, time: .omitted)) }
        return out
    }
}
