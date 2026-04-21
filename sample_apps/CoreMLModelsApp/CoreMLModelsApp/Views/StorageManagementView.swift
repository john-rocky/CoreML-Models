import SwiftUI

/// Per-model storage manager. Lists every installed model with the
/// on-disk footprint (sum of `.mlpackage` + weight files + download
/// metadata), sorted largest first, and lets the user delete each one
/// individually or wipe the whole cache.
struct StorageManagementView: View {
    @EnvironmentObject var catalog: ModelCatalog

    @State private var rows: [StorageRow] = []
    @State private var totalBytes: Int64 = 0
    @State private var freeBytes: Int64 = 0
    @State private var orphanBytes: Int64 = 0
    @State private var showingDeleteAll = false

    var body: some View {
        List {
            Section {
                summaryRow(title: "Models on disk", value: formatBytes(totalBytes))
                summaryRow(title: "Free space", value: formatBytes(freeBytes))
                if orphanBytes > 0 {
                    summaryRow(title: "Other cached files", value: formatBytes(orphanBytes))
                }
            } header: {
                Text("Summary")
            } footer: {
                Text("Delete models you no longer use to free up device storage. Models can be re-downloaded anytime from the catalog.")
            }

            if rows.isEmpty {
                Section {
                    Text("No models installed.")
                        .foregroundStyle(.secondary)
                }
            } else {
                Section("Installed models (\(rows.count))") {
                    ForEach(rows) { row in
                        rowView(row)
                            .swipeActions(edge: .trailing, allowsFullSwipe: true) {
                                Button(role: .destructive) {
                                    delete(row)
                                } label: {
                                    Label("Delete", systemImage: "trash")
                                }
                            }
                    }
                }

                Section {
                    Button(role: .destructive) {
                        showingDeleteAll = true
                    } label: {
                        Label("Delete all downloaded models", systemImage: "trash")
                    }
                }
            }
        }
        .navigationTitle("Storage")
        .navigationBarTitleDisplayMode(.inline)
        .task { reload() }
        .refreshable { reload() }
        .confirmationDialog(
            "Delete all downloaded models?",
            isPresented: $showingDeleteAll,
            titleVisibility: .visible
        ) {
            Button("Delete All (\(formatBytes(totalBytes)))", role: .destructive) {
                deleteAll()
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This removes every installed model from this device. You can re-download them from the catalog.")
        }
    }

    // MARK: - Rows

    private func summaryRow(title: String, value: String) -> some View {
        HStack {
            Text(title)
            Spacer()
            Text(value)
                .foregroundStyle(.secondary)
                .monospacedDigit()
        }
    }

    @ViewBuilder
    private func rowView(_ row: StorageRow) -> some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(row.name)
                    .font(.body)
                if let subtitle = row.subtitle {
                    Text(subtitle)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            Spacer()
            Text(formatBytes(row.bytes))
                .font(.callout)
                .foregroundStyle(.secondary)
                .monospacedDigit()
        }
        .contentShape(Rectangle())
    }

    // MARK: - Actions

    private func delete(_ row: StorageRow) {
        if let model = row.model {
            catalog.deleteInstall(of: model)
        } else {
            // Orphan directory — no manifest entry (model was removed from the
            // catalog since install, or a partial download left files behind).
            try? FileManager.default.removeItem(at: row.directory)
        }
        reload()
    }

    private func deleteAll() {
        for row in rows {
            delete(row)
        }
        // Also purge any remaining orphan files under the models root.
        let fm = FileManager.default
        if let entries = try? fm.contentsOfDirectory(at: Paths.modelsDir, includingPropertiesForKeys: nil) {
            for url in entries {
                try? fm.removeItem(at: url)
            }
        }
        reload()
    }

    // MARK: - Scan

    private func reload() {
        var collected: [StorageRow] = []
        var seenDirs: Set<URL> = []

        // 1. Models known to the manifest and installed.
        if let manifest = catalog.manifest {
            for model in manifest.models where catalog.isInstalled(model) {
                let dir = directory(for: model)
                let size = directorySize(at: dir)
                if size > 0 {
                    collected.append(StorageRow(
                        id: model.id,
                        name: model.name,
                        subtitle: model.subtitle,
                        bytes: size,
                        directory: dir,
                        model: model
                    ))
                    seenDirs.insert(dir.standardizedFileURL)
                }
            }
        }

        // 2. Orphan directories under the app's models dir that are not in
        // the current manifest (e.g. removed entries, aborted downloads).
        let fm = FileManager.default
        if let entries = try? fm.contentsOfDirectory(at: Paths.modelsDir, includingPropertiesForKeys: nil) {
            var orphan: Int64 = 0
            for url in entries where url.hasDirectoryPath {
                if seenDirs.contains(url.standardizedFileURL) { continue }
                let size = directorySize(at: url)
                if size > 0 { orphan += size }
            }
            orphanBytes = orphan
        } else {
            orphanBytes = 0
        }

        rows = collected.sorted { $0.bytes > $1.bytes }
        totalBytes = rows.reduce(0) { $0 + $1.bytes } + orphanBytes
        freeBytes = availableDiskBytes()
    }

    private func directory(for model: ModelEntry) -> URL {
        if let llm = catalog.llmModelInfo(for: model) {
            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            return docs
                .appendingPathComponent("Models", isDirectory: true)
                .appendingPathComponent(llm.folderName, isDirectory: true)
        }
        return Paths.modelDir(id: model.id)
    }

    private func directorySize(at url: URL) -> Int64 {
        let fm = FileManager.default
        guard fm.fileExists(atPath: url.path) else { return 0 }
        guard let enumerator = fm.enumerator(at: url, includingPropertiesForKeys: [.totalFileAllocatedSizeKey, .fileAllocatedSizeKey]) else {
            return 0
        }
        var total: Int64 = 0
        for case let fileURL as URL in enumerator {
            let values = try? fileURL.resourceValues(forKeys: [.totalFileAllocatedSizeKey, .fileAllocatedSizeKey])
            if let size = values?.totalFileAllocatedSize ?? values?.fileAllocatedSize {
                total += Int64(size)
            }
        }
        return total
    }

    private func availableDiskBytes() -> Int64 {
        let url = Paths.appSupportRoot
        let values = try? url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        if let bytes = values?.volumeAvailableCapacityForImportantUsage {
            return bytes
        }
        return 0
    }
}

private struct StorageRow: Identifiable {
    let id: String
    let name: String
    let subtitle: String?
    let bytes: Int64
    let directory: URL
    let model: ModelEntry?
}
