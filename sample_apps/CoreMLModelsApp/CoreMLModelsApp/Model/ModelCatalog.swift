import Foundation
import CoreMLLLM
import StoreKit
import UIKit

/// Top-level state holder. Owns the manifest, knows which models are
/// installed locally, and vends downloaders on demand.
///
/// Loads order:
/// 1. On first launch (or no cache): try to fetch the latest manifest from HF.
/// 2. If network fails: fall back to the cached manifest on disk.
/// 3. If neither is available: show an empty list with a retry button.
@MainActor
final class ModelCatalog: ObservableObject {
    @Published private(set) var manifest: Manifest?
    @Published private(set) var loadingError: String?
    // Defaults to true so the first render shows the loading spinner
    // instead of briefly flashing the empty-state error before `.task`
    // fires `loadInitial()`.
    @Published private(set) var isLoading = true
    @Published private(set) var installedIds: Set<String> = []

    private var downloaders: [String: ModelFileDownloader] = [:]

    // MARK: - LLM Model Routing

    /// Maps a manifest model entry to a ModelDownloader.ModelInfo
    /// if it's an LLM model (chat template). Returns nil for non-LLM models.
    func llmModelInfo(for model: ModelEntry) -> ModelDownloader.ModelInfo? {
        guard model.demo.template == "chat" else { return nil }
        let normalized = model.id.replacingOccurrences(of: "_", with: "-")
        return ModelDownloader.ModelInfo.defaults.first {
            $0.folderName == normalized || $0.id == normalized
        }
    }

    // MARK: - Loading

    func loadInitial() async {
        isLoading = true
        defer { isLoading = false }

        // Always try the cached copy first for instant UI.
        if let cached = try? readCachedManifest() {
            manifest = cached
            installedIds = scanInstalled()
        }

        // Then refresh from network in the background.
        do {
            let fresh = try await fetchManifest()
            manifest = fresh
            try writeCachedManifest(fresh)
            installedIds = scanInstalled()
            loadingError = nil
        } catch {
            // If decode failed, clear stale cache so next launch retries cleanly.
            if manifest == nil {
                try? FileManager.default.removeItem(at: Paths.manifestCache)
                loadingError = (error as? LocalizedError)?.errorDescription ?? "\(error)"
            }
        }
    }

    func refresh() async { await loadInitial() }

    // MARK: - Downloaders

    func downloader(for model: ModelEntry) -> ModelFileDownloader {
        if let existing = downloaders[model.id] { return existing }
        let new = ModelFileDownloader(modelId: model.id)
        downloaders[model.id] = new
        return new
    }

    func startDownload(for model: ModelEntry) {
        Self.noteDownloadStartedForReviewPrompt()
        if let llmInfo = llmModelInfo(for: model) {
            // Client-side guard against repeated taps / SwiftUI re-entry spawning
            // parallel download tasks. The package has its own guard now, but
            // short-circuiting here also avoids spinning a stray async Task.
            let dl = ModelDownloader.shared
            if dl.isDownloading && dl.downloadingModelId == llmInfo.id {
                print("[Catalog] Already downloading \(llmInfo.id), ignoring duplicate tap")
                return
            }
            Task {
                do {
                    _ = try await dl.download(llmInfo)
                    installedIds.insert(model.id)
                } catch {
                    print("[Catalog] LLM download failed: \(error)")
                }
            }
            return
        }
        let dl = downloader(for: model)
        Task {
            await dl.run(files: model.files)
            installedIds = scanInstalled()
        }
    }

    // MARK: - App Store Review Prompt

    private static let reviewCountKey = "com.coreml-models.zoo.download_session_count"
    private static let reviewPromptedKey = "com.coreml-models.zoo.review_prompted"

    /// Increment the download-session counter and, on the second ever
    /// download, ask SKStoreReviewController for a rating prompt. The
    /// system still gates this to at most 3 requests per 365 days, so
    /// this only nudges the first qualifying window.
    private static func noteDownloadStartedForReviewPrompt() {
        let defaults = UserDefaults.standard
        guard !defaults.bool(forKey: reviewPromptedKey) else { return }

        let count = defaults.integer(forKey: reviewCountKey) + 1
        defaults.set(count, forKey: reviewCountKey)
        guard count == 2 else { return }

        defaults.set(true, forKey: reviewPromptedKey)
        // Delay slightly so the download UI is visible before the prompt.
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 1_500_000_000)
            let scene = UIApplication.shared.connectedScenes
                .first { $0.activationState == .foregroundActive } as? UIWindowScene
            if let scene {
                SKStoreReviewController.requestReview(in: scene)
            }
        }
    }

    func deleteInstall(of model: ModelEntry) {
        if let llmInfo = llmModelInfo(for: model) {
            try? ModelDownloader.shared.delete(llmInfo)
            installedIds.remove(model.id)
            return
        }
        let dir = Paths.modelDir(id: model.id)
        try? FileManager.default.removeItem(at: dir)
        installedIds.remove(model.id)
        // Keep the downloader in the dict: its background URLSession is
        // identified by modelId, and creating a second session with the
        // same identifier (on the next download tap) leaves the new
        // session's delegate callbacks undelivered — progress stalls at 0%.
        // The existing downloader's run() resets state at the top, so
        // reusing it for a fresh download is safe.
    }

    func isInstalled(_ model: ModelEntry) -> Bool {
        if let llmInfo = llmModelInfo(for: model) {
            return ModelDownloader.shared.isDownloaded(llmInfo)
        }
        return installedIds.contains(model.id)
    }

    // MARK: - Helpers

    private func scanInstalled() -> Set<String> {
        var ids: Set<String> = []
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(at: Paths.modelsDir, includingPropertiesForKeys: nil) else {
            return ids
        }
        for url in entries where url.hasDirectoryPath {
            let meta = url.appendingPathComponent(".meta.json")
            if fm.fileExists(atPath: meta.path) {
                ids.insert(url.lastPathComponent)
            }
        }
        return ids
    }

    private func fetchManifest() async throws -> Manifest {
        guard let url = URL(string: Paths.manifestURLString) else {
            throw NSError(domain: "ModelCatalog", code: 1)
        }
        let (data, _) = try await URLSession.shared.data(from: url)
        let decoded = try JSONDecoder().decode(Manifest.self, from: data)
        return decoded
    }

    private func readCachedManifest() throws -> Manifest? {
        let url = Paths.manifestCache
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(Manifest.self, from: data)
    }

    private func writeCachedManifest(_ manifest: Manifest) throws {
        let data = try JSONEncoder().encode(manifest)
        try data.write(to: Paths.manifestCache, options: [.atomic])
    }

    // MARK: - Convenience

    var categories: [Category] {
        manifest?.categories.sorted { $0.order < $1.order } ?? []
    }

    func models(in categoryId: String) -> [ModelEntry] {
        manifest?.models.filter { $0.categoryId == categoryId } ?? []
    }
}
