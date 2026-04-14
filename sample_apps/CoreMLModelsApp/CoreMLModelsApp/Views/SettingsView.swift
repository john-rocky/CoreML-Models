import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var catalog: ModelCatalog

    var body: some View {
        List {
            Section("Storage") {
                let totalBytes = computeStorageUsed()
                LabeledContent("Models on disk", value: formatBytes(totalBytes))
                LabeledContent("Installed count", value: "\(catalog.installedIds.count)")
                Button(role: .destructive) {
                    deleteAllInstalled()
                } label: {
                    Text("Delete all downloaded models")
                }
            }

            Section("About") {
                LabeledContent("App version", value: appVersion())
            }

            Section("Open-source models") {
                Text("Each model retains its own license. Tap a model in the catalog to see the license and a link to the upstream project.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .navigationTitle("Settings")
    }

    private func computeStorageUsed() -> Int64 {
        let fm = FileManager.default
        let root = Paths.modelsDir
        guard let enumerator = fm.enumerator(at: root, includingPropertiesForKeys: [.fileSizeKey]) else {
            return 0
        }
        var total: Int64 = 0
        for case let url as URL in enumerator {
            if let v = try? url.resourceValues(forKeys: [.fileSizeKey]),
               let size = v.fileSize {
                total += Int64(size)
            }
        }
        return total
    }

    private func deleteAllInstalled() {
        let fm = FileManager.default
        try? fm.removeItem(at: Paths.modelsDir)
    }

    private func appVersion() -> String {
        let v = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "?"
        let b = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "?"
        return "\(v) (\(b))"
    }
}
