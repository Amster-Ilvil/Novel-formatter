import SwiftUI

@main
struct NovelFormatterApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @StateObject private var model = AppModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(model)
                .frame(minWidth: 660, minHeight: 480)
                .onAppear { model.startupCheck() }
        }
        .defaultSize(width: 760, height: 560)
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationWillTerminate(_ notification: Notification) {
        // A detached GUI child would outlive the shell; stop it together.
        AppModel.current?.stopChild()
    }
}
