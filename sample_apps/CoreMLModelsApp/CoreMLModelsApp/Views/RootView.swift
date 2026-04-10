import SwiftUI

struct RootView: View {
    @EnvironmentObject var catalog: ModelCatalog

    var body: some View {
        NavigationStack {
            Group {
                if catalog.isLoading && catalog.manifest == nil {
                    ProgressView("Loading model catalog…")
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if catalog.manifest == nil {
                    emptyState
                } else {
                    catalogList
                }
            }
            .navigationTitle("CoreML-Models")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    NavigationLink {
                        SettingsView()
                    } label: {
                        Image(systemName: "gear")
                    }
                }
            }
            .refreshable { await catalog.refresh() }
        }
    }

    @ViewBuilder
    private var emptyState: some View {
        VStack(spacing: 16) {
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 48))
                .foregroundStyle(.secondary)
            Text("Couldn't load model catalog")
                .font(.headline)
            if let err = catalog.loadingError {
                Text(err)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 32)
            }
            Button("Retry") {
                Task { await catalog.refresh() }
            }
            .buttonStyle(.borderedProminent)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    @ViewBuilder
    private var catalogList: some View {
        List {
            ForEach(catalog.categories) { category in
                Section {
                    ForEach(catalog.models(in: category.id)) { model in
                        NavigationLink(value: model) {
                            ModelRow(model: model)
                        }
                    }
                } header: {
                    Label(category.name, systemImage: category.icon)
                        .font(.subheadline.weight(.semibold))
                }
            }
        }
        .listStyle(.insetGrouped)
        .navigationDestination(for: ModelEntry.self) { model in
            ModelDetailView(model: model)
        }
    }
}

struct ModelRow: View {
    @EnvironmentObject var catalog: ModelCatalog
    let model: ModelEntry

    var body: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(model.name).font(.body.weight(.medium))
                if let s = model.subtitle {
                    Text(s).font(.caption).foregroundStyle(.secondary)
                }
                Text(formatBytes(model.downloadSize))
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
            Spacer()
            if catalog.isInstalled(model) {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.green)
            }
        }
    }
}

func formatBytes(_ bytes: Int64) -> String {
    let formatter = ByteCountFormatter()
    formatter.allowedUnits = [.useMB, .useGB]
    formatter.countStyle = .file
    return formatter.string(fromByteCount: bytes)
}
