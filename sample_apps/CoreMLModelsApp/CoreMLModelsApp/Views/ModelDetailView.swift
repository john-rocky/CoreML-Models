import SwiftUI
import CoreMLLLM

struct ModelDetailView: View {
    @EnvironmentObject var catalog: ModelCatalog
    let model: ModelEntry

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                header
                DownloadSection(model: model, downloader: catalog.downloader(for: model))
                description
                metadata
            }
            .padding()
        }
        .navigationTitle(model.name)
        .navigationBarTitleDisplayMode(.large)
    }

    @ViewBuilder
    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            if let s = model.subtitle {
                Text(s).font(.subheadline).foregroundStyle(.secondary)
            }
            HStack(spacing: 8) {
                Label(formatBytes(model.downloadSize), systemImage: "tray.and.arrow.down")
                Spacer()
                Text(model.license.name)
                    .font(.caption.monospaced())
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(.thinMaterial, in: Capsule())
            }
            .font(.caption)
            .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private var description: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("About").font(.headline)
            Text(model.descriptionMd).font(.body)
        }
    }

    @ViewBuilder
    private var metadata: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Details").font(.headline)
            metaRow("Upstream", model.upstream.name, link: model.upstream.url)
            metaRow("License", model.license.name, link: model.license.url)
            if let year = model.upstream.year {
                metaRow("Year", String(year))
            }
            metaRow("Min iOS", model.requirements.minIos)
            metaRow("Min RAM", "\(model.requirements.minRamMb) MB")
            if let hf = model.derivedHfRepoUrl {
                metaRow("Hugging Face", "View files", link: hf)
            }
            if let conv = model.conversionScriptUrl {
                metaRow("Conversion script", "View source", link: conv)
            }
        }
    }

    @ViewBuilder
    private func metaRow(_ label: String, _ value: String, link: String? = nil) -> some View {
        HStack {
            Text(label).foregroundStyle(.secondary)
            Spacer()
            if let link, let url = URL(string: link) {
                Link(value, destination: url).font(.body)
            } else {
                Text(value)
            }
        }
        .font(.subheadline)
    }
}

/// Section that observes the per-model downloader directly via @ObservedObject.
/// Split out from ModelDetailView so SwiftUI re-renders when downloader.state changes.
/// For LLM models (chat template), delegates to LLMDownloadSection which observes
/// the ModelDownloader instead.
struct DownloadSection: View {
    let model: ModelEntry
    @ObservedObject var downloader: ModelFileDownloader
    @EnvironmentObject var catalog: ModelCatalog

    var body: some View {
        if catalog.llmModelInfo(for: model) != nil {
            LLMDownloadSection(model: model)
        } else {
            regularBody
        }
    }

    @ViewBuilder
    private var regularBody: some View {
        Group {
            if catalog.isInstalled(model) {
                installedControls
            } else if !downloader.state.byFile.isEmpty && !downloader.state.isComplete {
                progressView
            } else {
                downloadButton
            }
        }
    }

    @ViewBuilder
    private var installedControls: some View {
        VStack(spacing: 12) {
            HStack(spacing: 12) {
                NavigationLink {
                    DemoLauncherView(model: model)
                } label: {
                    Label("Try It", systemImage: "play.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                Button(role: .destructive) {
                    catalog.deleteInstall(of: model)
                } label: {
                    Image(systemName: "trash")
                }
                .buttonStyle(.bordered)
            }
            HStack(spacing: 12) {
                NavigationLink {
                    ModelInspectorView(model: model)
                } label: {
                    Label("Inspect", systemImage: "info.circle")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                NavigationLink {
                    BenchmarkView(model: model)
                } label: {
                    Label("Benchmark", systemImage: "timer")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
            }
        }
    }

    @ViewBuilder
    private var progressView: some View {
        VStack(alignment: .leading, spacing: 8) {
            ProgressView(value: downloader.state.overallProgress) {
                Text("Downloading… \(Int(downloader.state.overallProgress * 100))%")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if let f = downloader.state.failure {
                Text(f).font(.caption).foregroundStyle(.red)
            }
        }
    }

    @ViewBuilder
    private var downloadButton: some View {
        Button {
            print("[DL] tap → startDownload \(model.id)")
            catalog.startDownload(for: model)
        } label: {
            Label("Download \(formatBytes(model.downloadSize))", systemImage: "icloud.and.arrow.down")
                .frame(maxWidth: .infinity)
        }
        .buttonStyle(.borderedProminent)
    }
}

/// Download section for LLM models that uses ModelDownloader.
/// The package downloader is @Observable so SwiftUI tracks property reads automatically.
struct LLMDownloadSection: View {
    let model: ModelEntry
    @EnvironmentObject var catalog: ModelCatalog
    private var llmDL: ModelDownloader { .shared }

    var body: some View {
        Group {
            if catalog.isInstalled(model) {
                installedControls
            } else if llmDL.isDownloading,
                      llmDL.downloadingModelId == catalog.llmModelInfo(for: model)?.id {
                progressView
            } else {
                downloadButton
            }
        }
    }

    @ViewBuilder
    private var installedControls: some View {
        HStack(spacing: 12) {
            NavigationLink {
                DemoLauncherView(model: model)
            } label: {
                Label("Try It", systemImage: "play.fill")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            Button(role: .destructive) {
                catalog.deleteInstall(of: model)
            } label: {
                Image(systemName: "trash")
            }
            .buttonStyle(.bordered)
        }
    }

    @ViewBuilder
    private var progressView: some View {
        VStack(alignment: .leading, spacing: 8) {
            ProgressView(value: llmDL.progress) {
                Text(llmDL.status.isEmpty ? "Downloading…" : llmDL.status)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if llmDL.isPaused {
                HStack {
                    Button("Resume") { llmDL.resumeDownload() }
                        .buttonStyle(.borderedProminent).controlSize(.small)
                    Button("Cancel", role: .destructive) { llmDL.cancelDownload() }
                        .buttonStyle(.bordered).controlSize(.small)
                }
            }
        }
    }

    @ViewBuilder
    private var downloadButton: some View {
        Button {
            catalog.startDownload(for: model)
        } label: {
            Label("Download \(formatBytes(model.downloadSize))", systemImage: "icloud.and.arrow.down")
                .frame(maxWidth: .infinity)
        }
        .buttonStyle(.borderedProminent)
    }
}
