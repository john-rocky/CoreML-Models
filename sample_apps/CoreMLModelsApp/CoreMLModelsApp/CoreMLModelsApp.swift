import SwiftUI
import CoreMLLLM

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
///
/// Routes events to the correct downloader based on session identifier:
/// - `com.coreml-models.zoo.dl.*` → app's own ModelDownloader
/// - `com.coreml-llm.model-download` → CoreMLLLM package's ModelDownloader
class AppDelegate: NSObject, UIApplicationDelegate {
    /// Stored by the system — must be called once all background events are delivered.
    static var backgroundCompletionHandler: (() -> Void)?

    func application(
        _ application: UIApplication,
        handleEventsForBackgroundURLSession identifier: String,
        completionHandler: @escaping () -> Void
    ) {
        print("[App] Background session event for: \(identifier)")
        if identifier == "com.coreml-llm.model-download" {
            ModelDownloader.shared.backgroundCompletionHandler = completionHandler
        } else {
            AppDelegate.backgroundCompletionHandler = completionHandler
        }
    }

    func applicationDidBecomeActive(_ application: UIApplication) {
        KeyboardDismissInstaller.shared.install()
    }
}

/// Installs a tap-outside-to-dismiss-keyboard gesture on every app window.
/// `cancelsTouchesInView = false` keeps buttons/text fields working; tapping
/// a TextField still makes it first responder after the brief dismiss.
final class KeyboardDismissInstaller: NSObject {
    static let shared = KeyboardDismissInstaller()
    private static let recognizerName = "coreml.keyboardDismiss"

    func install() {
        for scene in UIApplication.shared.connectedScenes {
            guard let windowScene = scene as? UIWindowScene else { continue }
            for window in windowScene.windows {
                let existing = window.gestureRecognizers?.contains {
                    $0.name == Self.recognizerName
                } ?? false
                if existing { continue }
                let tap = UITapGestureRecognizer(target: self, action: #selector(dismissKeyboard))
                tap.cancelsTouchesInView = false
                tap.name = Self.recognizerName
                window.addGestureRecognizer(tap)
            }
        }
    }

    @objc private func dismissKeyboard() {
        UIApplication.shared.sendAction(
            #selector(UIResponder.resignFirstResponder),
            to: nil, from: nil, for: nil
        )
    }
}
