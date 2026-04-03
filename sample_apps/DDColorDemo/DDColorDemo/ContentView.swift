import SwiftUI
import PhotosUI

struct ContentView: View {
    @StateObject private var colorizer = ImageColorizer()
    @State private var selectedItem: PhotosPickerItem?
    @State private var originalImage: UIImage?
    @State private var colorizedImage: UIImage?
    @State private var isProcessing = false
    @State private var showOriginal = false
    @State private var status = ""
    @State private var processingTime: Double?

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Status bar
                HStack {
                    Circle().fill(colorizer.isReady ? .green : .red).frame(width: 8, height: 8)
                    Text(colorizer.isReady ? "Ready" : "Loading model...")
                        .font(.caption).foregroundColor(.secondary)
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
                    if let colorized = colorizedImage, let original = originalImage {
                        ZStack {
                            Image(uiImage: showOriginal ? original : colorized)
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .frame(maxWidth: .infinity, maxHeight: .infinity)
                        }
                        .contentShape(Rectangle())
                        .gesture(
                            DragGesture(minimumDistance: 0)
                                .onChanged { _ in showOriginal = true }
                                .onEnded { _ in showOriginal = false }
                        )
                        .overlay(alignment: .bottom) {
                            Text(showOriginal ? "Original" : "Colorized")
                                .font(.caption).bold()
                                .padding(.horizontal, 12).padding(.vertical, 4)
                                .background(.ultraThinMaterial)
                                .cornerRadius(8)
                                .padding(.bottom, 8)
                        }
                    } else if let original = originalImage {
                        Image(uiImage: original)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                    } else {
                        VStack(spacing: 12) {
                            Image(systemName: "photo.on.rectangle.angled")
                                .font(.system(size: 60))
                                .foregroundColor(.secondary)
                            Text("Select a black & white photo")
                                .foregroundColor(.secondary)
                        }
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                    }
                }

                // Controls
                VStack(spacing: 12) {
                    if isProcessing {
                        ProgressView(status)
                    }

                    HStack(spacing: 16) {
                        PhotosPicker(selection: $selectedItem, matching: .images) {
                            Label("Select Photo", systemImage: "photo.badge.plus")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)

                        if let original = originalImage {
                            Button {
                                colorize(original)
                            } label: {
                                Label("Colorize", systemImage: "paintpalette.fill")
                                    .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(.orange)
                            .disabled(isProcessing || !colorizer.isReady)
                        }
                    }

                    if let colorized = colorizedImage {
                        HStack(spacing: 16) {
                            ShareLink(item: Image(uiImage: colorized),
                                      preview: SharePreview("Colorized Image", image: Image(uiImage: colorized))) {
                                Label("Share", systemImage: "square.and.arrow.up")
                                    .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.bordered)

                            Button {
                                UIImageWriteToSavedPhotosAlbum(colorized, nil, nil, nil)
                                status = "Saved to Photos"
                            } label: {
                                Label("Save", systemImage: "square.and.arrow.down")
                                    .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.bordered)
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("DDColor")
            .onChange(of: selectedItem) { _ in loadImage() }
        }
    }

    private func loadImage() {
        guard let item = selectedItem else { return }
        colorizedImage = nil
        processingTime = nil
        status = ""
        Task {
            if let data = try? await item.loadTransferable(type: Data.self),
               let img = UIImage(data: data) {
                originalImage = img
            }
        }
    }

    private func colorize(_ image: UIImage) {
        isProcessing = true
        status = "Colorizing..."
        colorizedImage = nil
        processingTime = nil
        Task {
            let start = CFAbsoluteTimeGetCurrent()
            do {
                let result = try await colorizer.colorize(image: image)
                let elapsed = CFAbsoluteTimeGetCurrent() - start
                await MainActor.run {
                    colorizedImage = result
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
}
