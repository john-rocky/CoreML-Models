import SwiftUI
import PhotosUI

struct ContentView: View {
    @StateObject private var resolver = SuperResolver()
    @State private var selectedItem: PhotosPickerItem?
    @State private var originalImage: UIImage?
    @State private var resultImage: UIImage?
    @State private var isProcessing = false
    @State private var processingTime: Double?
    @State private var status = ""
    @State private var showOriginal = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Status bar
                HStack {
                    Circle()
                        .fill(resolver.isReady ? .green : .red)
                        .frame(width: 8, height: 8)
                    Text(resolver.isReady ? "Model Ready" : "Loading Models...")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    if let t = processingTime {
                        Text(String(format: "%.1fs", t))
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                .padding(.horizontal)
                .padding(.vertical, 4)

                // Image display with before/after toggle
                GeometryReader { geo in
                    if let result = resultImage, let original = originalImage {
                        ZStack {
                            Image(uiImage: showOriginal ? original : result)
                                .resizable()
                                .interpolation(.high)
                                .aspectRatio(contentMode: .fit)
                                .frame(maxWidth: geo.size.width, maxHeight: geo.size.height)
                        }
                        .frame(maxWidth: geo.size.width, maxHeight: geo.size.height)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                        .overlay(alignment: .topLeading) {
                            Text(showOriginal ? "Original" : "4x Super Resolution")
                                .font(.caption2.bold())
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(.ultraThinMaterial)
                                .clipShape(Capsule())
                                .padding(8)
                        }
                        .onTapGesture {
                            showOriginal.toggle()
                        }
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
                                    Image(systemName: "arrow.up.left.and.arrow.down.right")
                                        .font(.largeTitle)
                                        .foregroundStyle(.tertiary)
                                    Text("Select an image to upscale")
                                        .font(.subheadline)
                                        .foregroundStyle(.tertiary)
                                }
                            }
                    }
                }
                .padding(.horizontal)

                // Size info
                if let original = originalImage {
                    HStack {
                        Text("\(Int(original.size.width))x\(Int(original.size.height))")
                            .font(.caption).foregroundStyle(.secondary)
                        if resultImage != nil {
                            Image(systemName: "arrow.right")
                                .font(.caption2).foregroundStyle(.secondary)
                            Text("1024x1024")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        Spacer()
                        if resultImage != nil {
                            Text("Tap image to compare")
                                .font(.caption2).foregroundStyle(.tertiary)
                        }
                    }
                    .padding(.horizontal)
                    .padding(.top, 4)
                }

                // Status
                if !status.isEmpty {
                    Text(status)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .padding(.top, 4)
                }

                Spacer().frame(height: 16)

                // Controls
                VStack(spacing: 12) {

                    HStack(spacing: 16) {
                        PhotosPicker(selection: $selectedItem, matching: .images) {
                            Label("Photo", systemImage: "photo.on.rectangle")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)

                        Button {
                            runSuperResolution()
                        } label: {
                            if isProcessing {
                                ProgressView()
                                    .frame(maxWidth: .infinity)
                            } else {
                                Label("Upscale 4x", systemImage: "sparkles")
                                    .frame(maxWidth: .infinity)
                            }
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(originalImage == nil || !resolver.isReady || isProcessing)

                    }

                    if let result = resultImage {
                        HStack(spacing: 16) {
                            ShareLink(item: Image(uiImage: result),
                                      preview: SharePreview("Super Resolution", image: Image(uiImage: result))) {
                                Label("Share", systemImage: "square.and.arrow.up")
                                    .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.bordered)

                            Button {
                                UIImageWriteToSavedPhotosAlbum(result, nil, nil, nil)
                                status = "Saved to Photos"
                            } label: {
                                Label("Save", systemImage: "square.and.arrow.down")
                                    .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.bordered)
                        }
                    }
                }
                .padding(.horizontal)
                .padding(.bottom)
            }
            .navigationTitle("SinSR")
            .onChange(of: selectedItem) {
                Task {
                    if let data = try? await selectedItem?.loadTransferable(type: Data.self),
                       let image = UIImage(data: data) {
                        originalImage = image
                        resultImage = nil
                        processingTime = nil
                        status = ""
                        showOriginal = false
                    }
                }
            }
        }
    }

    private func runSuperResolution() {
        guard let image = originalImage else { return }
        isProcessing = true
        resultImage = nil
        processingTime = nil
        status = "Running diffusion super-resolution..."
        showOriginal = false

        Task {
            let start = CFAbsoluteTimeGetCurrent()
            do {
                let result = try await resolver.superResolve(image: image)
                let elapsed = CFAbsoluteTimeGetCurrent() - start
                await MainActor.run {
                    resultImage = result
                    processingTime = elapsed
                    status = ""
                    isProcessing = false
                }
            } catch {
                let elapsed = CFAbsoluteTimeGetCurrent() - start
                await MainActor.run {
                    status = "Error: \(error.localizedDescription)"
                    processingTime = elapsed
                    isProcessing = false
                }
            }
        }
    }

}
