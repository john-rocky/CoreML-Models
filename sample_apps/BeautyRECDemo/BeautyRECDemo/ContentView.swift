import SwiftUI
import PhotosUI

struct ContentView: View {
    @State private var sourceItem: PhotosPickerItem?
    @State private var referenceItem: PhotosPickerItem?
    @State private var sourceImage: UIImage?
    @State private var referenceImage: UIImage?
    @State private var resultImage: UIImage?
    @State private var isProcessing = false
    @State private var status = ""
    @State private var processingTime: Double?
    @State private var showOriginal = false

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

                // Result display
                GeometryReader { geo in
                    if let result = resultImage {
                        ZStack(alignment: .bottom) {
                            Image(uiImage: showOriginal ? sourceImage! : result)
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .frame(maxWidth: geo.size.width, maxHeight: geo.size.height)
                            Text(showOriginal ? "Original" : "Makeup Applied")
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
                    } else {
                        // Source + Reference side by side
                        HStack(spacing: 12) {
                            imageSlot(image: sourceImage, label: "Your Face", icon: "person.crop.circle")
                            imageSlot(image: referenceImage, label: "Makeup Ref", icon: "paintbrush")
                        }
                        .frame(maxWidth: geo.size.width, maxHeight: geo.size.height)
                    }
                }
                .padding(.horizontal)

                // Controls
                VStack(spacing: 12) {
                    if isProcessing {
                        ProgressView(status)
                    }

                    // Sample buttons
                    HStack(spacing: 12) {
                        Button {
                            loadSample(src: "sample_no_makeup", ref: "sample_makeup")
                        } label: {
                            Label("Sample", systemImage: "sparkles")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                        .tint(.pink)

                        PhotosPicker(selection: $sourceItem, matching: .images) {
                            Label("Face", systemImage: "person.crop.circle")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)

                        PhotosPicker(selection: $referenceItem, matching: .images) {
                            Label("Makeup", systemImage: "paintbrush")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                    }

                    Button {
                        runTransfer()
                    } label: {
                        if isProcessing {
                            ProgressView().frame(maxWidth: .infinity)
                        } else {
                            Label("Apply Makeup", systemImage: "wand.and.stars")
                                .frame(maxWidth: .infinity)
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.pink)
                    .disabled(sourceImage == nil || referenceImage == nil || isProcessing)

                    if let result = resultImage {
                        HStack(spacing: 16) {
                            ShareLink(item: Image(uiImage: result),
                                      preview: SharePreview("Makeup Transfer", image: Image(uiImage: result))) {
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
            .navigationTitle("BeautyREC")
            .onChange(of: sourceItem) {
                Task {
                    if let data = try? await sourceItem?.loadTransferable(type: Data.self),
                       let image = UIImage(data: data) {
                        sourceImage = image
                        resultImage = nil
                        processingTime = nil
                    }
                }
            }
            .onChange(of: referenceItem) {
                Task {
                    if let data = try? await referenceItem?.loadTransferable(type: Data.self),
                       let image = UIImage(data: data) {
                        referenceImage = image
                        resultImage = nil
                        processingTime = nil
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func imageSlot(image: UIImage?, label: String, icon: String) -> some View {
        if let img = image {
            Image(uiImage: img)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .clipShape(RoundedRectangle(cornerRadius: 12))
        } else {
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(.systemGray6))
                .overlay {
                    VStack(spacing: 6) {
                        Image(systemName: icon)
                            .font(.title2)
                            .foregroundStyle(.tertiary)
                        Text(label)
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                    }
                }
        }
    }

    private func loadSample(src: String, ref: String) {
        if let srcPath = Bundle.main.path(forResource: src, ofType: "png"),
           let refPath = Bundle.main.path(forResource: ref, ofType: "png"),
           let srcImg = UIImage(contentsOfFile: srcPath),
           let refImg = UIImage(contentsOfFile: refPath) {
            sourceImage = srcImg
            referenceImage = refImg
            resultImage = nil
            processingTime = nil
        }
    }

    private func runTransfer() {
        guard let src = sourceImage, let ref = referenceImage else { return }
        isProcessing = true
        status = "Applying makeup..."
        resultImage = nil
        processingTime = nil

        Task {
            let start = CFAbsoluteTimeGetCurrent()
            do {
                let result = try await MakeupTransfer.transfer(source: src, reference: ref)
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
}
