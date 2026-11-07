import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            logView
            Divider()
            footer
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }

    // MARK: - Header

    private var header: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .center, spacing: 14) {
                Image(nsImage: NSApp.applicationIconImage)
                    .resizable()
                    .frame(width: 52, height: 52)
                    .cornerRadius(10)
                VStack(alignment: .leading, spacing: 3) {
                    Text("Novel Formatter Studio")
                        .font(.title2.weight(.semibold))
                    Text(model.statusLine)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
                Spacer()
                stageBadge
            }

            if model.stage == .preparing {
                ProgressView()
                    .progressViewStyle(.linear)
                    .transition(.opacity)
            }

            if let error = model.lastError {
                Label(error, systemImage: "exclamationmark.triangle.fill")
                    .font(.footnote)
                    .foregroundStyle(.red)
                    .textSelection(.enabled)
            }
        }
        .padding(16)
        .animation(.default, value: model.stage)
    }

    @ViewBuilder
    private var stageBadge: some View {
        switch model.stage {
        case .checking:
            badge("Checking", systemImage: "clock", tint: .secondary)
        case .preparing:
            badge("Preparing", systemImage: "arrow.down.circle", tint: .orange)
        case .ready:
            badge("Ready", systemImage: "checkmark.circle.fill", tint: .green)
        case .running:
            badge("Running", systemImage: "play.circle.fill", tint: .blue)
        case .error:
            badge("Error", systemImage: "xmark.octagon.fill", tint: .red)
        }
    }

    private func badge(_ title: String, systemImage: String, tint: Color) -> some View {
        Label(title, systemImage: systemImage)
            .font(.callout.weight(.medium))
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(tint.opacity(0.12), in: Capsule())
            .foregroundStyle(tint)
    }

    // MARK: - Log

    private var logView: some View {
        ScrollViewReader { proxy in
            ScrollView {
                Text(model.log.isEmpty ? "Launch log will appear here." : model.log)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(model.log.isEmpty ? .tertiary : .secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(12)
                    .textSelection(.enabled)
                    .id("log-bottom")
            }
            .background(Color(nsColor: .textBackgroundColor).opacity(0.5))
            .onChange(of: model.log) {
                withAnimation(.easeOut(duration: 0.15)) {
                    proxy.scrollTo("log-bottom", anchor: .bottom)
                }
            }
        }
    }

    // MARK: - Footer

    private var footer: some View {
        HStack(spacing: 12) {
            Button(action: model.openWorkspace) {
                Label(model.stage == .running ? "Workspace running" : "Open workspace",
                      systemImage: "macwindow")
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .disabled(model.stage != .ready)

            Button("Retry") { retry() }
                .disabled(model.stage != .error)

            Spacer()

            Button {
                model.openLogsInFinder()
            } label: {
                Label("Logs", systemImage: "doc.text.magnifyingglass")
            }

            Button {
                model.revealSupportFolder()
            } label: {
                Label("Support folder", systemImage: "folder")
            }
        }
        .padding(12)
    }

    private func retry() {
        model.lastError = nil
        model.startupCheck()
    }
}
