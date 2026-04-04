import SwiftUI
import PhotosUI

struct ContentView: View {
    @State private var selectedItem: PhotosPickerItem?
    @State private var originalImage: UIImage?
    @State private var result: AnomalyResult?
    @State private var isProcessing = false
    @State private var status = ""
    @State private var showOriginal = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Status bar
                HStack {
                    if let r = result {
                        ScoreBadge(score: r.score)
                    }
                    Spacer()
                    if let r = result {
                        Text(String(format: "%.1fs", r.processingTime))
                            .font(.caption).foregroundColor(.secondary)
                    }
                }
                .padding(.horizontal)
                .padding(.vertical, 4)

                // Image display
                GeometryReader { geo in
                    if let r = result {
                        ZStack(alignment: .bottom) {
                            Image(uiImage: showOriginal ? originalImage! : r.heatmapOverlay)
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .frame(maxWidth: geo.size.width, maxHeight: geo.size.height)
                            Text(showOriginal ? "Original" : "Anomaly Map")
                                .font(.caption).bold()
                                .padding(.horizontal, 12).padding(.vertical, 4)
                                .background(.ultraThinMaterial, in: Capsule())
                                .padding(.bottom, 8)
                        }
                        .frame(maxWidth: geo.size.width, maxHeight: geo.size.height)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                        .onLongPressGesture(minimumDuration: .infinity,
                                            pressing: { pressing in showOriginal = pressing },
                                            perform: {})
                    } else if let original = originalImage {
                        Image(uiImage: original)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(maxWidth: geo.size.width, maxHeight: geo.size.height)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                    } else {
                        RoundedRectangle(cornerRadius: 12)
                            .fill(Color(.systemGray6))
                            .overlay {
                                VStack(spacing: 8) {
                                    Image(systemName: "eye.trianglebadge.exclamationmark")
                                        .font(.largeTitle)
                                        .foregroundStyle(.tertiary)
                                    Text("Select a photo")
                                        .font(.subheadline)
                                        .foregroundStyle(.tertiary)
                                }
                            }
                    }
                }
                .padding(.horizontal)

                // Controls
                VStack(spacing: 12) {
                    if isProcessing {
                        ProgressView(status)
                    }

                    // Sample image buttons
                    HStack(spacing: 12) {
                        Button {
                            loadSample("sample_good")
                        } label: {
                            Label("Good", systemImage: "checkmark.circle")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                        .tint(.green)

                        Button {
                            loadSample("sample_broken")
                        } label: {
                            Label("Broken", systemImage: "xmark.circle")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                        .tint(.red)

                        PhotosPicker(selection: $selectedItem, matching: .images) {
                            Label("Photo", systemImage: "photo.on.rectangle")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                    }

                    Button {
                        runDetection()
                    } label: {
                        if isProcessing {
                            ProgressView()
                                .frame(maxWidth: .infinity)
                        } else {
                            Label("Detect", systemImage: "magnifyingglass")
                                .frame(maxWidth: .infinity)
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(originalImage == nil || isProcessing)

                    if let r = result {
                        HStack(spacing: 16) {
                            ShareLink(item: Image(uiImage: r.heatmapOverlay),
                                      preview: SharePreview("Anomaly Detection", image: Image(uiImage: r.heatmapOverlay))) {
                                Label("Share", systemImage: "square.and.arrow.up")
                                    .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.bordered)

                            Text("Hold to compare")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .frame(maxWidth: .infinity)
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("EfficientAD")
            .onChange(of: selectedItem) {
                Task {
                    if let data = try? await selectedItem?.loadTransferable(type: Data.self),
                       let image = UIImage(data: data) {
                        originalImage = image
                        result = nil
                        status = ""
                    }
                }
            }
        }
    }

    private func loadSample(_ name: String) {
        guard let path = Bundle.main.path(forResource: name, ofType: "png"),
              let image = UIImage(contentsOfFile: path) else { return }
        originalImage = image
        result = nil
        status = ""
    }

    private func runDetection() {
        guard let image = originalImage else { return }
        isProcessing = true
        status = "Detecting anomalies..."
        result = nil

        Task {
            do {
                let r = try await AnomalyDetector.detect(in: image)
                await MainActor.run {
                    result = r
                    status = ""
                    isProcessing = false
                }
            } catch {
                await MainActor.run {
                    status = "Error: \(error.localizedDescription)"
                    isProcessing = false
                }
            }
        }
    }
}

struct ScoreBadge: View {
    let score: Float

    var body: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(badgeColor)
                .frame(width: 8, height: 8)
            Text(String(format: "Score: %.1f%%", score * 100))
                .font(.caption).bold()
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(badgeColor.opacity(0.15), in: Capsule())
    }

    private var badgeColor: Color {
        if score < 0.3 { return .green }
        if score < 0.6 { return .orange }
        return .red
    }
}
