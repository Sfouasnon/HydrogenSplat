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
                .tint(Brand.accent)
        }
        .commands {
            CommandGroup(replacing: .appInfo) { AboutCommand() }
            // replacing, not after: the WindowGroup's default "New Window" also claims ⌘N
            CommandGroup(replacing: .newItem) {
                Button("New Project from Clip…") { model.selection = .ingest }
                    .keyboardShortcut("n")
                Button("Reload Projects") { model.store.reload() }
                    .keyboardShortcut("r")
            }
        }

        // One model per window; two windows side by side is the A/B. No state restoration:
        // reopening a 500 MB model at every launch is not a favour.
        WindowGroup("Model", id: "viewer", for: ViewerModelFile.self) { $file in
            if let f = file {
                SplatViewerWindow(file: f)
                    .environmentObject(model)
                    .frame(minWidth: 900, minHeight: 600)
                    .tint(Brand.accent)
            }
        }
        .restorationBehavior(.disabled)
        .defaultSize(width: 1280, height: 800)

        Window("About HydrogenSplat", id: "about") { AboutView() }
            .windowResizability(.contentSize)
            .restorationBehavior(.disabled)
    }
}
