import SwiftUI
import UIKit
import CoreML
import Vision
import PhotosUI

// MARK: - Lightweight Scene Segmentation
// Uses LRASPP_MobileNetV3 model (512x512 input, 1x21x512x512 segmentation map output)
// Output feature name: "var_972"
// 21 Pascal VOC classes

struct ContentView: View {
    @StateObject private var segmenter = SegmentationViewModel()

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                if let error = segmenter.errorMessage {
                    HStack {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.yellow)
                        Text(error)
                            .font(.caption)
                    }
                    .padding()
                    .background(Color(.systemOrange).opacity(0.1))
                }

                if let originalImage = segmenter.originalImage {
                    // Image display area
                    ZStack {
                        Image(uiImage: originalImage)
                            .resizable()
                            .scaledToFit()

                        if let overlayImage = segmenter.overlayImage, segmenter.showOverlay {
                            Image(uiImage: overlayImage)
                                .resizable()
                                .scaledToFit()
                                .opacity(segmenter.overlayOpacity)
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .background(Color.black)

                    // Controls
                    VStack(spacing: 12) {
                        // Overlay toggle
                        Toggle(isOn: $segmenter.showOverlay) {
                            Label("Segmentation Overlay", systemImage: "square.stack.3d.up")
                        }

                        if segmenter.showOverlay {
                            // Opacity slider
                            HStack {
                                Text("Opacity")
                                    .font(.caption)
                                Slider(value: $segmenter.overlayOpacity, in: 0.1...1.0)
                                Text(String(format: "%.0f%%", segmenter.overlayOpacity * 100))
                                    .font(.caption)
                                    .frame(width: 40)
                            }
                        }

                        // Detected classes
                        if !segmenter.detectedClasses.isEmpty {
                            VStack(alignment: .leading, spacing: 6) {
                                Text("Detected Classes")
                                    .font(.headline)

                                LazyVGrid(columns: [
                                    GridItem(.flexible()),
                                    GridItem(.flexible()),
                                    GridItem(.flexible())
                                ], spacing: 6) {
                                    ForEach(segmenter.detectedClasses, id: \.index) { cls in
                                        HStack(spacing: 4) {
                                            Circle()
                                                .fill(VOCLabels.color(for: cls.index))
                                                .frame(width: 10, height: 10)
                                            Text(cls.name)
                                                .font(.caption2)
                                                .lineLimit(1)
                                            Spacer()
                                            Text(String(format: "%.0f%%", cls.percentage))
                                                .font(.caption2)
                                                .foregroundColor(.secondary)
                                        }
                                    }
                                }
                            }
                        }

                        if segmenter.isProcessing {
                            ProgressView("Segmenting image...")
                        }
                    }
                    .padding()

                    Spacer()
                } else {
                    // Empty state
                    Spacer()
                    VStack(spacing: 16) {
                        Image(systemName: "square.stack.3d.down.right")
                            .font(.system(size: 60))
                            .foregroundColor(.secondary)
                        Text("Select a photo to perform\nscene segmentation")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)

                        PhotosPicker(
                            selection: $segmenter.selectedItem,
                            matching: .images
                        ) {
                            Label("Select Photo", systemImage: "photo")
                                .font(.headline)
                                .padding()
                                .frame(maxWidth: 280)
                                .background(Color.accentColor)
                                .foregroundColor(.white)
                                .cornerRadius(12)
                        }
                    }
                    Spacer()
                }
            }
            .navigationTitle("LRASPP Segmentation")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                if segmenter.originalImage != nil {
                    ToolbarItem(placement: .navigationBarTrailing) {
                        PhotosPicker(
                            selection: $segmenter.selectedItem,
                            matching: .images
                        ) {
                            Image(systemName: "photo.badge.plus")
                        }
                    }
                }
            }
        }
    }
}

// MARK: - Detected Class Info
struct DetectedClass {
    let index: Int
    let name: String
    let percentage: Double // percentage of pixels
}

// MARK: - Segmentation ViewModel
@MainActor
class SegmentationViewModel: ObservableObject {
    @Published var originalImage: UIImage?
    @Published var overlayImage: UIImage?
    @Published var showOverlay = true
    @Published var overlayOpacity: Double = 0.5
    @Published var isProcessing = false
    @Published var errorMessage: String?
    @Published var detectedClasses: [DetectedClass] = []

    @Published var selectedItem: PhotosPickerItem? {
        didSet { Task { await loadAndSegment() } }
    }

    private var vnModel: VNCoreMLModel?

    init() {
        loadModel()
    }

    private func loadModel() {
        // PLACEHOLDER: Add LRASPP_MobileNetV3.mlpackage to the Xcode project.
        // The compiled model class will be generated automatically by Xcode.
        // Download from the converted_models directory and drag into the project navigator.
        do {
            guard let modelURL = Bundle.main.url(forResource: "LRASPP_MobileNetV3", withExtension: "mlmodelc") else {
                errorMessage = "Model not found. Add LRASPP_MobileNetV3.mlpackage to the project."
                return
            }
            let mlModel = try MLModel(contentsOf: modelURL)
            vnModel = try VNCoreMLModel(for: mlModel)
        } catch {
            errorMessage = "Failed to load model: \(error.localizedDescription)"
        }
    }

    private func loadAndSegment() async {
        guard let item = selectedItem,
              let data = try? await item.loadTransferable(type: Data.self),
              let uiImage = UIImage(data: data) else { return }

        originalImage = uiImage
        overlayImage = nil
        detectedClasses = []
        isProcessing = true

        await performSegmentation(on: uiImage)
    }

    private func performSegmentation(on image: UIImage) async {
        guard let vnModel = vnModel else {
            isProcessing = false
            return
        }

        guard let cgImage = image.cgImage else {
            isProcessing = false
            return
        }

        let request = VNCoreMLRequest(model: vnModel)
        request.imageCropAndScaleOption = .scaleFill

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])

        do {
            try handler.perform([request])

            if let results = request.results as? [VNCoreMLFeatureValueObservation],
               let multiArray = results.first?.featureValue.multiArrayValue {
                // Output shape: 1 x 21 x 512 x 512
                processSegmentationOutput(multiArray: multiArray, originalSize: image.size)
            }
        } catch {
            errorMessage = "Segmentation failed: \(error.localizedDescription)"
        }

        isProcessing = false
    }

    private func processSegmentationOutput(multiArray: MLMultiArray, originalSize: CGSize) {
        let numClasses = 21
        let height = 512
        let width = 512
        let totalPixels = height * width

        // Find argmax class for each pixel
        var classMap = [Int](repeating: 0, count: totalPixels)
        var classCounts = [Int](repeating: 0, count: numClasses)

        for y in 0..<height {
            for x in 0..<width {
                var maxVal: Float = -Float.greatestFiniteMagnitude
                var maxClass = 0

                for c in 0..<numClasses {
                    // Index: [0, c, y, x] for shape [1, 21, 512, 512]
                    let index = c * (height * width) + y * width + x
                    let val = multiArray[index].floatValue
                    if val > maxVal {
                        maxVal = val
                        maxClass = c
                    }
                }

                let pixelIndex = y * width + x
                classMap[pixelIndex] = maxClass
                classCounts[maxClass] += 1
            }
        }

        // Build overlay image
        var pixelData = [UInt8](repeating: 0, count: totalPixels * 4) // RGBA

        for i in 0..<totalPixels {
            let classIndex = classMap[i]
            let rgb = VOCLabels.rgbColor(for: classIndex)
            pixelData[i * 4 + 0] = rgb.0
            pixelData[i * 4 + 1] = rgb.1
            pixelData[i * 4 + 2] = rgb.2
            pixelData[i * 4 + 3] = classIndex == 0 ? 0 : 200 // Transparent for background
        }

        // Create CGImage from pixel data
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)

        guard let provider = CGDataProvider(data: Data(pixelData) as CFData),
              let cgImage = CGImage(
                width: width,
                height: height,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: width * 4,
                space: colorSpace,
                bitmapInfo: bitmapInfo,
                provider: provider,
                decode: nil,
                shouldInterpolate: true,
                intent: .defaultIntent
              ) else { return }

        overlayImage = UIImage(cgImage: cgImage)

        // Build detected classes list (excluding background if small)
        var detected: [DetectedClass] = []
        for c in 0..<numClasses {
            let pct = Double(classCounts[c]) / Double(totalPixels) * 100.0
            if pct > 0.5 { // Only show classes with > 0.5% coverage
                detected.append(DetectedClass(
                    index: c,
                    name: VOCLabels.name(for: c),
                    percentage: pct
                ))
            }
        }
        detectedClasses = detected.sorted { $0.percentage > $1.percentage }
    }
}

#Preview {
    ContentView()
}
