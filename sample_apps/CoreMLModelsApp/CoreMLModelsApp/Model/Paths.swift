import Foundation

/// Central place for on-disk locations used by the hub app.
///
/// Layout:
///   Application Support/coreml-models/
///   ├── manifest.json                     (cached copy of models.json)
///   ├── manifest.etag                     (HTTP ETag for cheap revalidation)
///   └── models/
///       └── {model_id}/
///           ├── <file 1>
///           ├── <file 2>
///           └── .meta.json                (completed download marker)
enum Paths {
    static let manifestURLString =
        "https://huggingface.co/mlboydaisuke/coreml-zoo/resolve/main/models.json"

    static var appSupportRoot: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let root = base.appendingPathComponent("coreml-models", isDirectory: true)
        ensureDirectory(root)
        return root
    }

    static var manifestCache: URL {
        appSupportRoot.appendingPathComponent("manifest.json")
    }

    static var manifestEtag: URL {
        appSupportRoot.appendingPathComponent("manifest.etag")
    }

    static var modelsDir: URL {
        let url = appSupportRoot.appendingPathComponent("models", isDirectory: true)
        ensureDirectory(url)
        return url
    }

    static func modelDir(id: String) -> URL {
        let url = modelsDir.appendingPathComponent(id, isDirectory: true)
        ensureDirectory(url)
        return url
    }

    static func metaFile(modelId: String) -> URL {
        modelDir(id: modelId).appendingPathComponent(".meta.json")
    }

    private static func ensureDirectory(_ url: URL) {
        let fm = FileManager.default
        if !fm.fileExists(atPath: url.path) {
            try? fm.createDirectory(at: url, withIntermediateDirectories: true)
            var mutable = url
            var rv = URLResourceValues()
            rv.isExcludedFromBackup = true
            try? mutable.setResourceValues(rv)
        }
    }
}
