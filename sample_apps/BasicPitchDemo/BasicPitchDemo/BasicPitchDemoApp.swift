import SwiftUI
import AVFoundation

@main
struct BasicPitchDemoApp: App {
    init() {
        // Configure audio session so playback is captured by Screen Recording
        do {
            try AVAudioSession.sharedInstance().setCategory(
                .playback,
                mode: .default,
                options: [.mixWithOthers]
            )
            try AVAudioSession.sharedInstance().setActive(true)
            print("Audio session configured: \(AVAudioSession.sharedInstance().category.rawValue)")
        } catch {
            print("Audio session error: \(error)")
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
