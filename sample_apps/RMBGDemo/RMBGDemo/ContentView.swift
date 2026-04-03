import SwiftUI
import PhotosUI

struct ContentView: View {
    @State private var selectedItem: PhotosPickerItem?
    @State private var originalImage: UIImage?
    @State private var resultImage: UIImage?
    @State private var isProcessing = false
    @State private var status = ""
    @State private var processingTime: Double?

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Status bar
                HStack {
                    Spacer()
                    if let t = processingTime {
                        Text(String(format: "%.1fs", t))
                            .font(.caption).foregroundColor(.secondary)
                    }
                }
                .padding(.horizontal)
                .padding(.vertical, 4)

                // Image display
                GeometryReader { geo in
                    if let result = resultImage {
                        ZStack {
                            checkerboardPattern()
                            Image(uiImage: result)
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .frame(maxWidth: geo.size.width, maxHeight: geo.size.height)
                        }
                        .frame(maxWidth: geo.size.width, maxHeight: geo.size.height)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
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
                                    Image(systemName: "person.crop.rectangle")
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

                    HStack(spacing: 16) {
                        PhotosPicker(selection: $selectedItem, matching: .images) {
                            Label("Photo", systemImage: "photo.on.rectangle")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)

                        Button {
                            removeBackground()
                        } label: {
                            if isProcessing {
                                ProgressView()
                                    .frame(maxWidth: .infinity)
                            } else {
                                Label("Remove BG", systemImage: "scissors")
                                    .frame(maxWidth: .infinity)
                            }
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(originalImage == nil || isProcessing)
                    }

                    if let result = resultImage {
                        HStack(spacing: 16) {
                            ShareLink(item: Image(uiImage: result),
                                      preview: SharePreview("Background Removed", image: Image(uiImage: result))) {
                                Label("Share", systemImage: "square.and.arrow.up")
                                    .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.bordered)

                            Button {
                                savePNG(result)
                            } label: {
                                Label("Save PNG", systemImage: "square.and.arrow.down")
                                    .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.bordered)
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("RMBG")
            .onChange(of: selectedItem) {
                Task {
                    if let data = try? await selectedItem?.loadTransferable(type: Data.self),
                       let image = UIImage(data: data) {
                        originalImage = image
                        resultImage = nil
                        processingTime = nil
                        status = ""
                    }
                }
            }
        }
    }

    private func removeBackground() {
        guard let image = originalImage else { return }
        isProcessing = true
        status = "Removing background..."
        resultImage = nil
        processingTime = nil

        Task {
            let start = CFAbsoluteTimeGetCurrent()
            do {
                let result = try await BackgroundRemover.removeBackground(from: image)
                let elapsed = CFAbsoluteTimeGetCurrent() - start
                await MainActor.run {
                    resultImage = result
                    processingTime = elapsed
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

    private func savePNG(_ image: UIImage) {
        guard let data = image.pngData(),
              let pngImage = UIImage(data: data) else { return }
        UIImageWriteToSavedPhotosAlbum(pngImage, nil, nil, nil)
        status = "Saved to Photos"
    }

    @ViewBuilder
    private func checkerboardPattern() -> some View {
        Canvas { context, size in
            let tileSize: CGFloat = 16
            let rows = Int(ceil(size.height / tileSize))
            let cols = Int(ceil(size.width / tileSize))
            for row in 0..<rows {
                for col in 0..<cols {
                    let rect = CGRect(x: CGFloat(col) * tileSize, y: CGFloat(row) * tileSize,
                                      width: tileSize, height: tileSize)
                    let isLight = (row + col) % 2 == 0
                    context.fill(Path(rect), with: .color(isLight ? .white : Color(.systemGray5)))
                }
            }
        }
    }
}
