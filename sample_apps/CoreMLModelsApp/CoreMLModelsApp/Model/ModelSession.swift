import Foundation
import CoreML
import SwiftUI

/// Eager model loader with load-time measurement.
///
/// Templates own a `@StateObject ModelSession<T>` and call `ensure(...)` from
/// a `.task { }` block on their root view, so loading kicks off as soon as
/// the demo screen appears (while the user is still picking material).
/// Inference handlers then `try await session.get()` to await readiness
/// without blocking the UI thread.
///
/// `T` is whatever the template needs — an `MLModel` for single-model demos,
/// a tuple / dictionary for multi-model pipelines.
@MainActor
final class ModelSession<T>: ObservableObject {
    @Published private(set) var isLoading = false
    @Published private(set) var loadTimeSec: Double?
    @Published private(set) var loadError: String?

    private var task: Task<T, Error>?

    /// Kick off loading. No-op if already started.
    func ensure(_ load: @escaping @Sendable () async throws -> T) {
        guard task == nil else { return }
        isLoading = true
        loadError = nil
        let start = ContinuousClock.now
        task = Task {
            do {
                let result = try await load()
                let elapsed = ContinuousClock.now - start
                await MainActor.run {
                    self.loadTimeSec = Self.seconds(elapsed)
                    self.isLoading = false
                }
                return result
            } catch {
                await MainActor.run {
                    self.loadError = error.localizedDescription
                    self.isLoading = false
                }
                throw error
            }
        }
    }

    /// Await the loaded value. Throws if loading failed or wasn't started.
    func get() async throws -> T {
        guard let task else {
            throw NSError(domain: "ModelSession", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Model session not started"])
        }
        return try await task.value
    }

    /// Drop the task so the next navigation re-loads. Call from `.onDisappear`
    /// for memory-heavy pipelines that should not stay resident.
    func reset() {
        task?.cancel()
        task = nil
        isLoading = false
        loadTimeSec = nil
        loadError = nil
    }

    private static func seconds(_ d: Duration) -> Double {
        let c = d.components
        return Double(c.seconds) + Double(c.attoseconds) / 1e18
    }
}

// MARK: - Timing display

/// Compact "Load 0.42s · Infer 0.18s" readout used across demo templates.
struct TimingsLabel: View {
    let loadSec: Double?
    let inferSec: Double?

    var body: some View {
        if loadSec != nil || inferSec != nil {
            HStack(spacing: 6) {
                if let l = loadSec {
                    Text("Load \(fmt(l))").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                }
                if loadSec != nil && inferSec != nil {
                    Text("·").font(.caption).foregroundStyle(.secondary)
                }
                if let i = inferSec {
                    Text("Infer \(fmt(i))").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                }
            }
        }
    }

    private func fmt(_ s: Double) -> String {
        s < 1.0 ? String(format: "%.0f ms", s * 1000) : String(format: "%.2fs", s)
    }
}
