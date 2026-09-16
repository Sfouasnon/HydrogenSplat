import SwiftUI
import HSCore

/// Re-renders `content` whenever `session` publishes. A view that only holds a RunSession in
/// @State (or reads one from a dictionary) is NOT invalidated by the session's own changes, so
/// any button title or disabled state derived from `isRunning` must be built inside this.
struct Observing<Content: View>: View {
    @ObservedObject var session: RunSession
    let content: (RunSession) -> Content

    init(_ session: RunSession?, @ViewBuilder content: @escaping (RunSession) -> Content) {
        self._session = ObservedObject(wrappedValue: session ?? RunSession.placeholder)
        self.content = content
    }

    var body: some View { content(session) }
}
