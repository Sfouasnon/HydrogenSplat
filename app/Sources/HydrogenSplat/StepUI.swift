import SwiftUI
import HSCore

/// The shared vocabulary of the rebuilt pipeline pages (docs/ui-rebuild.md): every step opens with a
/// `StepHeader`, states one `VerdictCard` a non-technical user can act on, offers choices as
/// `ChoiceTile`s, summarises numbers as `MetricChip`s and ends with a `StepFooter` holding the one
/// primary action. Copy on these pages is about THIS project only — never another project, clip or
/// earlier run by name.
enum Verdict: Equatable {
    case good, attention, blocked, info
    var color: Color {
        switch self {
        case .good: return .green
        case .attention: return .orange
        case .blocked: return .red
        case .info: return Brand.accent
        }
    }
    /// The step's rail dot from its engine stages: failed or stale needs attention, done is good.
    static func from(_ status: StageStatus) -> Verdict {
        switch status {
        case .done: return .good
        case .failed: return .blocked
        case .stale, .running: return .attention
        case .pending, .unknown: return .info
        }
    }
}

/// Title and one plain sentence of what the step is for.
struct StepHeader: View {
    let title: String
    let lead: String
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title).font(.system(size: 28, weight: .bold))
            Text(lead).font(.title3).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.bottom, 4)
    }
}

/// The step's one sentence, in the user's terms, plus an optional second line and any content below
/// (chips, a chart, buttons). The coloured bar on the left is the verdict.
struct VerdictCard<Content: View>: View {
    let verdict: Verdict
    let headline: String
    var detail: String? = nil
    @ViewBuilder var content: Content
    init(_ verdict: Verdict, headline: String, detail: String? = nil, @ViewBuilder content: () -> Content) {
        self.verdict = verdict
        self.headline = headline
        self.detail = detail
        self.content = content()
    }
    var body: some View {
        HStack(alignment: .top, spacing: 0) {
            RoundedRectangle(cornerRadius: 2).fill(verdict.color).frame(width: 4)
            VStack(alignment: .leading, spacing: 10) {
                Text(headline).font(.headline)
                if let d = detail { Text(d).foregroundStyle(.secondary) }
                content
            }
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .background(RoundedRectangle(cornerRadius: 12).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color(nsColor: .separatorColor)))
    }
}

extension VerdictCard where Content == EmptyView {
    init(_ verdict: Verdict, headline: String, detail: String? = nil) {
        self.init(verdict, headline: headline, detail: detail) { EmptyView() }
    }
}

/// A short fact: "Scan coverage 46 %". Tint it when it is a warning.
struct MetricChip: View {
    let text: String
    var tint: Color? = nil
    var body: some View {
        Text(text)
            .font(.callout)
            .padding(.horizontal, 10).padding(.vertical, 5)
            .background(Capsule().fill((tint ?? Color.secondary).opacity(0.15)))
            .foregroundStyle(tint ?? Color.primary)
    }
}

/// One of a small set of mutually exclusive choices: a title, one line of what it means, selected or
/// not. Lay several out in an HStack or a LazyVGrid.
struct ChoiceTile: View {
    let title: String
    let subtitle: String
    let selected: Bool
    let action: () -> Void
    var body: some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 4) {
                Text(title).font(.headline)
                Text(subtitle).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: .infinity, minHeight: 56, alignment: .topLeading)
            .padding(12)
            .background(RoundedRectangle(cornerRadius: 10).fill(Color(nsColor: .controlBackgroundColor)))
            .overlay(RoundedRectangle(cornerRadius: 10)
                .stroke(selected ? Brand.accent : Color(nsColor: .separatorColor), lineWidth: selected ? 2 : 1))
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(selected ? .isSelected : [])
    }
}

/// The step's primary action, an optional secondary one, and one line on what happens next.
struct StepFooter: View {
    let primary: String
    var primaryEnabled: Bool = true
    var secondary: String? = nil
    var note: String? = nil
    let primaryAction: () -> Void
    var secondaryAction: (() -> Void)? = nil
    var body: some View {
        HStack(spacing: 12) {
            Button(primary, action: primaryAction)
                .buttonStyle(.borderedProminent).controlSize(.large)
                .disabled(!primaryEnabled)
            if let s = secondary, let act = secondaryAction {
                Button(s, action: act).controlSize(.large)
            }
            if let n = note { Text(n).foregroundStyle(.secondary) }
            Spacer()
        }
        .padding(.top, 4)
    }
}

/// A page column with the step's content, the width the mockups use.
struct StepPage<Content: View>: View {
    @ViewBuilder var content: Content
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) { content }
                .frame(maxWidth: 920, alignment: .leading)
                .padding(.horizontal, 40).padding(.vertical, 32)
        }
    }
}
