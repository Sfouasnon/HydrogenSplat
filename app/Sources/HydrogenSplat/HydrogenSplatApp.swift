import SwiftUI
import AppKit
import HSCore

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Launched with `swift run` there is no bundle; make it a normal foreground app.
        NSApp.setActivationPolicy(.regular)
        if let icon = Brand.appIcon { NSApp.applicationIconImage = icon }
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

@main
struct HydrogenSplatApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
    @StateObject private var model = AppModel()

    var body: some Scene {
        WindowGroup("HydrogenSplat") {
            RootView()
                .environmentObject(model)
                .environmentObject(model.store)
                .frame(minWidth: 980, minHeight: 640)
                .tint(Brand.tally)
        }
        .commands {
            CommandGroup(after: .newItem) {
                Button("New Project from Clip…") { model.selection = .ingest }
                    .keyboardShortcut("n")
                Button("Reload Projects") { model.store.reload() }
                    .keyboardShortcut("r")
            }
        }
    }
}
