import SwiftUI

struct ModelInspectorView: View {
    let model: ModelEntry

    @State private var inspections: [ModelFileInspection] = []
    @State private var isLoading = true
    @State private var error: String?
    @State private var totalDiskSize: Int64 = 0
    @State private var copiedFile: String?

    var body: some View {
        Group {
            if isLoading {
                ProgressView("Loading model info…")
            } else if let error {
                VStack(spacing: 12) {
                    Image(systemName: "exclamationmark.triangle")
                        .font(.largeTitle).foregroundStyle(.secondary)
                    Text(error).font(.subheadline).foregroundStyle(.secondary)
                }
                .padding()
            } else {
                inspectionContent
            }
        }
        .navigationTitle("Inspector")
        .navigationBarTitleDisplayMode(.inline)
        .task { await loadInspections() }
    }

    // MARK: - Content

    @ViewBuilder
    private var inspectionContent: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                overviewSection
                ForEach(inspections) { inspection in
                    fileSection(inspection)
                    codeSection(inspection)
                }
            }
            .padding()
        }
    }

    @ViewBuilder
    private var overviewSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Overview").font(.headline)
            HStack {
                Text("Model files").foregroundStyle(.secondary)
                Spacer()
                Text("\(inspections.count)")
            }
            .font(.subheadline)
            HStack {
                Text("Size on disk").foregroundStyle(.secondary)
                Spacer()
                Text(formatBytes(totalDiskSize))
            }
            .font(.subheadline)
        }
        .padding()
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    @ViewBuilder
    private func fileSection(_ inspection: ModelFileInspection) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            // File header
            VStack(alignment: .leading, spacing: 4) {
                Text(inspection.fileName)
                    .font(.headline.monospaced())
                HStack(spacing: 12) {
                    Label(formatBytes(inspection.sizeOnDisk), systemImage: "doc")
                    Label(inspection.computeUnits, systemImage: "cpu")
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }

            // Inputs
            if !inspection.inputs.isEmpty {
                Text("Inputs").font(.subheadline.bold())
                ForEach(inspection.inputs) { tensor in
                    tensorRow(tensor)
                }
            }

            // Outputs
            if !inspection.outputs.isEmpty {
                Text("Outputs").font(.subheadline.bold())
                ForEach(inspection.outputs) { tensor in
                    tensorRow(tensor)
                }
            }

            // Metadata
            if !inspection.metadata.isEmpty {
                Text("Metadata").font(.subheadline.bold())
                ForEach(inspection.metadata, id: \.key) { item in
                    HStack(alignment: .top) {
                        Text(item.key)
                            .foregroundStyle(.secondary)
                            .frame(width: 80, alignment: .leading)
                        Text(item.value)
                            .font(.caption)
                    }
                    .font(.caption)
                }
            }
        }
        .padding()
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    @ViewBuilder
    private func tensorRow(_ tensor: TensorInfo) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(tensor.name)
                    .font(.caption.monospaced())
                Text(tensor.featureType)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 2) {
                if let shape = tensor.shape {
                    Text(shape.map(String.init).joined(separator: " \u{00D7} "))
                        .font(.caption.monospacedDigit())
                }
                Text(tensor.dataType)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 2)
    }

    // MARK: - Code snippet

    @ViewBuilder
    private func codeSection(_ inspection: ModelFileInspection) -> some View {
        let snippet = ModelInspector.generateSnippet(for: inspection)
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Swift Code").font(.headline)
                Spacer()
                Button {
                    UIPasteboard.general.string = snippet
                    copiedFile = inspection.id
                    Task {
                        try? await Task.sleep(for: .seconds(2))
                        if copiedFile == inspection.id { copiedFile = nil }
                    }
                } label: {
                    Label(
                        copiedFile == inspection.id ? "Copied" : "Copy",
                        systemImage: copiedFile == inspection.id ? "checkmark" : "doc.on.doc"
                    )
                    .font(.caption)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
            ScrollView(.horizontal, showsIndicators: false) {
                Text(snippet)
                    .font(.caption.monospaced())
                    .textSelection(.enabled)
                    .padding(12)
            }
            .background(Color(.systemFill), in: RoundedRectangle(cornerRadius: 8))
        }
        .padding()
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    // MARK: - Loading

    private func loadInspections() async {
        do {
            inspections = try await ModelInspector.inspect(model: model)
            totalDiskSize = ModelInspector.diskSize(modelId: model.id)
            isLoading = false
        } catch {
            self.error = error.localizedDescription
            isLoading = false
        }
    }
}
