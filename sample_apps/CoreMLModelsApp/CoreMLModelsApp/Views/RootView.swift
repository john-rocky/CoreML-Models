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
            if let featured = featuredModel {
                Section {
                    NavigationLink(value: featured) {
                        FeaturedHeroCard(model: featured)
                    }
                    .listRowInsets(EdgeInsets())
                    .listRowBackground(Color.clear)
                    .listRowSeparator(.hidden)
                }
            }

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

    /// Spotlight model shown at the top of the catalog. Gemma 4 E2B is the hub's
    /// flagship on-device LLM (text + vision) powered by the CoreML-LLM package.
    private var featuredModel: ModelEntry? {
        catalog.manifest?.models.first(where: { $0.id == "gemma4_e2b" })
            ?? catalog.manifest?.models.first(where: { $0.demo.template == "chat" })
    }
}

// MARK: - Featured hero card

struct FeaturedHeroCard: View {
    let model: ModelEntry

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 6) {
                Image(systemName: "sparkles")
                Text("FEATURED").tracking(1.2)
            }
            .font(.caption2.weight(.bold))
            .foregroundStyle(.white.opacity(0.9))

            VStack(alignment: .leading, spacing: 4) {
                Text(model.name)
                    .font(.title2.weight(.bold))
                    .foregroundStyle(.white)
                Text(model.subtitle ?? "On-device LLM with vision")
                    .font(.subheadline)
                    .foregroundStyle(.white.opacity(0.85))
                    .lineLimit(2)
            }

            HStack(spacing: 8) {
                FeaturedChip(icon: "text.bubble", label: "Text")
                FeaturedChip(icon: "photo", label: "Vision")
                FeaturedChip(icon: "cpu", label: "ANE")
                Spacer(minLength: 0)
            }

            HStack {
                Text("Tap to try")
                    .font(.subheadline.weight(.semibold))
                Image(systemName: "arrow.right")
                    .font(.subheadline.weight(.semibold))
                Spacer()
                Text(formatBytes(model.downloadSize))
                    .font(.caption.monospacedDigit())
                    .opacity(0.85)
            }
            .foregroundStyle(.white)
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            LinearGradient(
                colors: [Color(red: 0.28, green: 0.24, blue: 0.85),
                         Color(red: 0.72, green: 0.22, blue: 0.55)],
                startPoint: .topLeading, endPoint: .bottomTrailing
            )
        )
        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
        .padding(.horizontal)
        .padding(.vertical, 8)
    }
}

private struct FeaturedChip: View {
    let icon: String
    let label: String
    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: icon)
            Text(label)
        }
        .font(.caption2.weight(.semibold))
        .padding(.horizontal, 8).padding(.vertical, 4)
        .background(.white.opacity(0.18), in: Capsule())
        .foregroundStyle(.white)
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
