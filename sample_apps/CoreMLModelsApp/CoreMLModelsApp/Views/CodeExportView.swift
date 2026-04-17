import SwiftUI
import UIKit

struct CodeExportView: View {
    let model: ModelEntry

    @State private var inspections: [ModelFileInspection] = []
    @State private var selectedId: String?
    @State private var isLoading = true
    @State private var error: String?
    @State private var justCopied = false

    private var current: ModelFileInspection? {
        inspections.first(where: { $0.id == selectedId }) ?? inspections.first
    }

    var body: some View {
        Group {
            if isLoading {
                ProgressView("Preparing snippet…")
            } else if let error {
                VStack(spacing: 12) {
                    Image(systemName: "exclamationmark.triangle")
                        .font(.largeTitle).foregroundStyle(.secondary)
                    Text(error).font(.subheadline).foregroundStyle(.secondary)
                }
                .padding()
            } else if let current {
                content(for: current)
            } else {
                Text("No model files").foregroundStyle(.secondary)
            }
        }
        .navigationTitle("Swift Code")
        .navigationBarTitleDisplayMode(.inline)
        .task { await load() }
    }

    @ViewBuilder
    private func content(for inspection: ModelFileInspection) -> some View {
        let snippet = ModelInspector.generateSnippet(for: inspection)
        VStack(alignment: .leading, spacing: 12) {
            if inspections.count > 1 {
                Picker("Target", selection: Binding(
                    get: { selectedId ?? inspections.first!.id },
                    set: { selectedId = $0 }
                )) {
                    ForEach(inspections) { ins in
                        Text(ins.fileName).tag(ins.id)
                    }
                }
                .pickerStyle(.menu)
                .padding(.horizontal)
            }

            HStack(spacing: 12) {
                Button {
                    UIPasteboard.general.string = snippet
                    justCopied = true
                    Task {
                        try? await Task.sleep(for: .seconds(2))
                        justCopied = false
                    }
                } label: {
                    Label(justCopied ? "Copied" : "Copy",
                          systemImage: justCopied ? "checkmark" : "doc.on.doc")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)

                ShareLink(item: snippet) {
                    Label("Share", systemImage: "square.and.arrow.up")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
            }
            .padding(.horizontal)

            ScrollView([.vertical, .horizontal]) {
                Text(snippet)
                    .font(.footnote.monospaced())
                    .textSelection(.enabled)
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .background(Color(.systemFill), in: RoundedRectangle(cornerRadius: 12))
            .padding(.horizontal)
            .padding(.bottom)
        }
        .padding(.top, 12)
    }

    private func load() async {
        do {
            let result = try await ModelInspector.inspect(model: model)
            inspections = result
            selectedId = result.first?.id
            isLoading = false
        } catch {
            self.error = error.localizedDescription
            isLoading = false
        }
    }
}
