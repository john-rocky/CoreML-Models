import SwiftUI

@main
struct CoreMLModelsApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @StateObject private var catalog = ModelCatalog()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(catalog)
                .task { await catalog.loadInitial() }
        }
    }
}

/// Handles background URLSession completion events.
/// When the system wakes the app for a finished background download,
/// it calls `application(_:handleEventsForBackgroundURLSession:completionHandler:)`.
/// We store the handler and let the URLSession delegate's
/// `urlSessionDidFinishEvents` call it.
class AppDelegate: NSObject, UIApplicationDelegate {
    /// Stored by the system — must be called once all background events are delivered.
    static var backgroundCompletionHandler: (() -> Void)?

    func application(
        _ application: UIApplication,
        handleEventsForBackgroundURLSession identifier: String,
        completionHandler: @escaping () -> Void
    ) {
        print("[App] Background session event for: \(identifier)")
        AppDelegate.backgroundCompletionHandler = completionHandler
    }
}
