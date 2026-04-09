import SwiftUI
import PhotosUI

struct ContentView: View {
    @StateObject private var estimator = DepthEstimator()
    @State private var selectedItem: PhotosPickerItem?
    @State private var originalImage: UIImage?
    @State private var depthImage: UIImage?
    @State private var normalImage: UIImage?
    @State private var depthMin: Float = 0
    @State private var depthMax: Float = 0
    @State private var processingTime: Double?
    @State private var isProcessing = false
    @State private var status = ""
    @State private var view: ViewMode = .depth

    enum ViewMode: String, CaseIterable, Identifiable {
        case original, depth, normal
        var id: String { rawValue }
        var label: String { rawValue.capitalized }
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Status bar
                HStack {
                    Circle().fill(estimator.isReady ? .green : .red).frame(width: 8, height: 8)
                    Text(estimator.isReady ? "Ready" : "Loading model...")
                        .font(.caption).foregroundColor(.secondary)
                    Spacer()
                    if let t = processingTime {
                        Text(String(format: "%.2fs", t))
                            .font(.caption).foregroundColor(.secondary)
                    }
                }
                .padding(.horizontal)
                .padding(.vertical, 4)

                // Image display
                GeometryReader { _ in
                    Group {
                        if depthImage != nil {
                            displayedImage
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .frame(maxWidth: .infinity, maxHeight: .infinity)
                        } else if let original = originalImage {
                            Image(uiImage: original)
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .frame(maxWidth: .infinity, maxHeight: .infinity)
                        } else {
                            VStack(spacing: 12) {
                                Image(systemName: "cube.transparent")
                                    .font(.system(size: 60))
                                    .foregroundColor(.secondary)
                                Text("Select a photo to estimate depth + surface normals")
                                    .multilineTextAlignment(.center)
                                    .foregroundColor(.secondary)
                                    .padding(.horizontal, 24)
                            }
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                        }
                    }
                }

                // Mode picker + metric depth readout
                if depthImage != nil {
                    Picker("View", selection: $view) {
                        ForEach(ViewMode.allCases) { mode in
                            Text(mode.label).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                    .padding(.horizontal)

                    if depthMax > 0 {
                        Text(String(format: "Depth range: %.2f m – %.2f m", depthMin, depthMax))
                            .font(.caption.monospacedDigit())
                            .foregroundColor(.secondary)
                            .padding(.top, 4)
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
                                run(original)
                            } label: {
                                Label("Estimate", systemImage: "cube.fill")
                                    .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(.blue)
                            .disabled(isProcessing || !estimator.isReady)
                        }
                    }

                    if let depth = depthImage, view == .depth {
                        ShareLink(item: Image(uiImage: depth),
                                  preview: SharePreview("Depth", image: Image(uiImage: depth))) {
                            Label("Share Depth Map", systemImage: "square.and.arrow.up")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                    }
                }
                .padding()
            }
            .navigationTitle("MoGe-2")
            .onChange(of: selectedItem) { _ in loadImage() }
        }
    }

    private var displayedImage: Image {
        switch view {
        case .original:
            return Image(uiImage: originalImage ?? UIImage())
        case .depth:
            return Image(uiImage: depthImage ?? UIImage())
        case .normal:
            return Image(uiImage: normalImage ?? UIImage())
        }
    }

    private func loadImage() {
        guard let item = selectedItem else { return }
        depthImage = nil
        normalImage = nil
        processingTime = nil
        depthMin = 0
        depthMax = 0
        status = ""
        Task {
            if let data = try? await item.loadTransferable(type: Data.self),
               let img = UIImage(data: data) {
                originalImage = img
            }
        }
    }

    private func run(_ image: UIImage) {
        isProcessing = true
        status = "Estimating depth..."
        depthImage = nil
        normalImage = nil
        processingTime = nil
        Task {
            let start = CFAbsoluteTimeGetCurrent()
            do {
                let result = try await estimator.estimate(image: image)
                let elapsed = CFAbsoluteTimeGetCurrent() - start
                let depthImg = Visualization.depthImage(
                    result.depth, size: result.size, dMin: result.depthMin, dMax: result.depthMax,
                    validX: result.validX, validY: result.validY, validW: result.validW, validH: result.validH
                )
                let normalImg = Visualization.normalImage(
                    result.normal, mask: result.mask, size: result.size,
                    validX: result.validX, validY: result.validY, validW: result.validW, validH: result.validH
                )
                await MainActor.run {
                    depthImage = depthImg
                    normalImage = normalImg
                    depthMin = result.depthMin
                    depthMax = result.depthMax
                    processingTime = elapsed
                    isProcessing = false
                    status = ""
                    view = .depth
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
