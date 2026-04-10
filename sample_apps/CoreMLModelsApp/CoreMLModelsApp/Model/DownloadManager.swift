import Foundation
import CryptoKit

/// Per-file download state observed by the UI.
enum DownloadState: Equatable {
    case notStarted
    case downloading(progress: Double)
    case unpacking
    case verified
    case failed(String)
}

/// Aggregated state across all files of a single model.
struct ModelDownloadState: Equatable {
    var byFile: [String: DownloadState] = [:]

    var overallProgress: Double {
        guard !byFile.isEmpty else { return 0 }
        let sum = byFile.values.reduce(0.0) { acc, state in
            switch state {
            case .notStarted: return acc
            case .downloading(let p): return acc + p
            case .unpacking, .verified: return acc + 1.0
            case .failed: return acc
            }
        }
        return sum / Double(byFile.count)
    }

    var isComplete: Bool {
        !byFile.isEmpty && byFile.values.allSatisfy { $0 == .verified }
    }

    var failure: String? {
        for (_, s) in byFile {
            if case .failed(let msg) = s { return msg }
        }
        return nil
    }
}

/// Downloads and verifies the files for a single model.
///
/// Supports:
/// - Background URLSession (downloads continue when app is suspended)
/// - Automatic resume from partial downloads on failure/relaunch
/// - SHA-256 verification
/// - ZIP unpacking
/// - `.meta.json` marker on completion
@MainActor
final class ModelDownloader: NSObject, ObservableObject {
    @Published private(set) var state = ModelDownloadState()
    let modelId: String

    private var bgSession: URLSession!
    private var activeTask: URLSessionDownloadTask?
    private var progressObserver: NSKeyValueObservation?

    // Resume data persisted to disk so we can resume after app termination
    private var resumeDataURL: URL {
        Paths.modelDir(id: modelId).appendingPathComponent(".resume_data")
    }

    // Queue of files remaining to download
    private var pendingFiles: [FileSpec] = []
    private var currentSpec: FileSpec?

    override init() { fatalError("Use init(modelId:)") }

    init(modelId: String) {
        self.modelId = modelId
        super.init()

        // Background session — downloads survive app suspension
        let config = URLSessionConfiguration.background(
            withIdentifier: "com.coreml-models.zoo.dl.\(modelId)"
        )
        config.isDiscretionary = false
        config.sessionSendsLaunchEvents = true
        config.timeoutIntervalForRequest = 60
        config.timeoutIntervalForResource = 60 * 60  // 1 hour for large models
        bgSession = URLSession(configuration: config, delegate: self, delegateQueue: nil)
    }

    // MARK: - Public API

    func cancel() {
        activeTask?.cancel()
        activeTask = nil
        pendingFiles = []
        currentSpec = nil
        clearResumeData()
    }

    func run(files: [FileSpec]) async {
        state = ModelDownloadState(byFile: Dictionary(uniqueKeysWithValues: files.map { ($0.name, .notStarted) }))

        for spec in files {
            do {
                try await downloadAndVerify(spec)
                state.byFile[spec.name] = .verified
            } catch is CancellationError {
                return
            } catch {
                let message = (error as? LocalizedError)?.errorDescription ?? "\(error)"
                state.byFile[spec.name] = .failed(message)
                return
            }
        }
        Self.writeMeta(modelId: modelId, files: files)
    }

    // MARK: - Download + Verify

    private func downloadAndVerify(_ spec: FileSpec) async throws {
        let finalURL = Paths.modelDir(id: modelId).appendingPathComponent(spec.name)

        if try isAlreadyDownloaded(spec: spec, at: finalURL) {
            state.byFile[spec.name] = .verified
            return
        }

        currentSpec = spec
        state.byFile[spec.name] = .downloading(progress: 0)

        guard let remote = URL(string: spec.url) else {
            throw DownloaderError.badURL(spec.url)
        }

        let tmpFile = try await startDownload(remote: remote, spec: spec)

        // Verify SHA-256
        let digest = try Self.sha256(of: tmpFile)
        guard digest.lowercased() == spec.sha256.lowercased() else {
            try? FileManager.default.removeItem(at: tmpFile)
            throw DownloaderError.sha256Mismatch(expected: spec.sha256, got: digest)
        }

        // Unpack if archive
        if spec.archive == "zip" {
            state.byFile[spec.name] = .unpacking
            try Self.unzip(archive: tmpFile, to: Paths.modelDir(id: modelId))
            try? FileManager.default.removeItem(at: tmpFile)
        } else {
            try? FileManager.default.removeItem(at: finalURL)
            try FileManager.default.moveItem(at: tmpFile, to: finalURL)
        }

        clearResumeData()
        currentSpec = nil
    }

    private func isAlreadyDownloaded(spec: FileSpec, at finalURL: URL) throws -> Bool {
        let fm = FileManager.default
        if spec.archive == "zip" {
            let base = (spec.name as NSString).deletingPathExtension
            let candidate = Paths.modelDir(id: modelId).appendingPathComponent(base)
            return fm.fileExists(atPath: candidate.path)
        }
        guard fm.fileExists(atPath: finalURL.path) else { return false }
        let digest = try Self.sha256(of: finalURL)
        return digest.lowercased() == spec.sha256.lowercased()
    }

    // MARK: - Background Download with Resume

    private func startDownload(remote: URL, spec: FileSpec) async throws -> URL {
        print("[DL] starting \(spec.name) ← \(remote.absoluteString)")

        return try await withCheckedThrowingContinuation { [weak self] cont in
            guard let self else { cont.resume(throwing: DownloaderError.cancelled); return }
            self.downloadContinuation = cont

            // Try to resume from saved data
            if let resumeData = self.loadResumeData() {
                print("[DL] Resuming \(spec.name) from saved data (\(resumeData.count) bytes)")
                let task = self.bgSession.downloadTask(withResumeData: resumeData)
                self.activeTask = task
                self.observeProgress(task, fileName: spec.name)
                task.resume()
            } else {
                let task = self.bgSession.downloadTask(with: remote)
                self.activeTask = task
                self.observeProgress(task, fileName: spec.name)
                task.resume()
            }
        }
    }

    private var downloadContinuation: CheckedContinuation<URL, Error>?

    private func observeProgress(_ task: URLSessionDownloadTask, fileName: String) {
        progressObserver?.invalidate()
        progressObserver = task.progress.observe(\.fractionCompleted) { [weak self] p, _ in
            let value = p.fractionCompleted
            Task { @MainActor in
                self?.state.byFile[fileName] = .downloading(progress: value)
            }
        }
    }

    // MARK: - Resume Data Persistence

    private func saveResumeData(_ data: Data) {
        try? data.write(to: resumeDataURL, options: [.atomic])
    }

    private func loadResumeData() -> Data? {
        try? Data(contentsOf: resumeDataURL)
    }

    private func clearResumeData() {
        try? FileManager.default.removeItem(at: resumeDataURL)
    }

    // MARK: - Helpers

    static func sha256(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while autoreleasepool(invoking: {
            let chunk = (try? handle.read(upToCount: 1024 * 1024)) ?? Data()
            if chunk.isEmpty { return false }
            hasher.update(data: chunk)
            return true
        }) {}
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    static func unzip(archive: URL, to dest: URL) throws {
        try ZipUnpacker.unpack(archive: archive, to: dest)
    }

    static func writeMeta(modelId: String, files: [FileSpec]) {
        let meta: [String: Any] = [
            "model_id": modelId,
            "installed_at": ISO8601DateFormatter().string(from: Date()),
            "files": files.map { ["name": $0.name, "sha256": $0.sha256, "size_bytes": $0.sizeBytes] }
        ]
        if let data = try? JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted]) {
            try? data.write(to: Paths.metaFile(modelId: modelId))
        }
    }
}

// MARK: - URLSessionDownloadDelegate (background download callbacks)

extension ModelDownloader: URLSessionDownloadDelegate {
    nonisolated func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        if let http = downloadTask.response as? HTTPURLResponse {
            print("[DL] HTTP \(http.statusCode)")
        }

        // Move temp file before URLSession deletes it
        let persisted = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString + ".download")
        do {
            try FileManager.default.moveItem(at: location, to: persisted)
            print("[DL] Downloaded to \(persisted.lastPathComponent)")
            Task { @MainActor in
                self.downloadContinuation?.resume(returning: persisted)
                self.downloadContinuation = nil
            }
        } catch {
            Task { @MainActor in
                self.downloadContinuation?.resume(throwing: error)
                self.downloadContinuation = nil
            }
        }
    }

    nonisolated func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        guard let error else { return }

        // Save resume data for later retry
        let nsError = error as NSError
        if let resumeData = nsError.userInfo[NSURLSessionDownloadTaskResumeData] as? Data {
            print("[DL] Saving resume data (\(resumeData.count) bytes)")
            Task { @MainActor in
                self.saveResumeData(resumeData)
                self.downloadContinuation?.resume(throwing: DownloaderError.interrupted)
                self.downloadContinuation = nil
            }
        } else if nsError.code == NSURLErrorCancelled {
            Task { @MainActor in
                self.downloadContinuation?.resume(throwing: CancellationError())
                self.downloadContinuation = nil
            }
        } else {
            print("[DL] Error: \(error)")
            Task { @MainActor in
                self.downloadContinuation?.resume(throwing: error)
                self.downloadContinuation = nil
            }
        }
    }

    nonisolated func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        guard totalBytesExpectedToWrite > 0 else { return }
        let progress = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
        Task { @MainActor in
            if let name = self.currentSpec?.name {
                self.state.byFile[name] = .downloading(progress: progress)
            }
        }
    }

    nonisolated func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        print("[DL] Background session events finished")
        DispatchQueue.main.async {
            AppDelegate.backgroundCompletionHandler?()
            AppDelegate.backgroundCompletionHandler = nil
        }
    }
}

enum DownloaderError: LocalizedError {
    case badURL(String)
    case noTempFile
    case sha256Mismatch(expected: String, got: String)
    case cancelled
    case interrupted

    var errorDescription: String? {
        switch self {
        case .badURL(let s): return "Invalid download URL: \(s)"
        case .noTempFile: return "Download did not produce a file"
        case .sha256Mismatch(let e, let g):
            return "Checksum mismatch. Expected \(e.prefix(12))…, got \(g.prefix(12))…"
        case .cancelled: return "Download cancelled"
        case .interrupted: return "Download interrupted — will resume automatically"
        }
    }
}
