import Foundation
import CoreMLLLM

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
    @Published private(set) var isLoading = false
    @Published private(set) var installedIds: Set<String> = []

    private var downloaders: [String: ModelDownloader] = [:]

    // MARK: - LLM Model Routing

    /// Maps a manifest model entry to a CoreMLLLM.ModelDownloader.ModelInfo
    /// if it's an LLM model (chat template). Returns nil for non-LLM models.
    func llmModelInfo(for model: ModelEntry) -> CoreMLLLM.ModelDownloader.ModelInfo? {
        guard model.demo.template == "chat" else { return nil }
        let normalized = model.id.replacingOccurrences(of: "_", with: "-")
        return CoreMLLLM.ModelDownloader.ModelInfo.defaults.first {
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
            if manifest == nil {
                loadingError = (error as? LocalizedError)?.errorDescription ?? "\(error)"
            }
        }
    }

    func refresh() async { await loadInitial() }

    // MARK: - Downloaders

    func downloader(for model: ModelEntry) -> ModelDownloader {
        if let existing = downloaders[model.id] { return existing }
        let new = ModelDownloader(modelId: model.id)
        downloaders[model.id] = new
        return new
    }

    func startDownload(for model: ModelEntry) {
        if let llmInfo = llmModelInfo(for: model) {
            Task {
                do {
                    _ = try await CoreMLLLM.ModelDownloader.shared.download(llmInfo)
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

    func deleteInstall(of model: ModelEntry) {
        if let llmInfo = llmModelInfo(for: model) {
            try? CoreMLLLM.ModelDownloader.shared.delete(llmInfo)
            installedIds.remove(model.id)
            return
        }
        let dir = Paths.modelDir(id: model.id)
        try? FileManager.default.removeItem(at: dir)
        installedIds.remove(model.id)
        downloaders.removeValue(forKey: model.id)
    }

    func isInstalled(_ model: ModelEntry) -> Bool {
        if let llmInfo = llmModelInfo(for: model) {
            return CoreMLLLM.ModelDownloader.shared.isDownloaded(llmInfo)
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
