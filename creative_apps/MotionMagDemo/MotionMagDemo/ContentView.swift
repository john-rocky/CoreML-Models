import SwiftUI
import UIKit
import CoreML
import AVFoundation
import PhotosUI

// MARK: - Video Frame Extractor

/// Extracts frames from a video asset at regular intervals
class VideoFrameExtractor {
    let asset: AVAsset

    init(asset: AVAsset) {
        self.asset = asset
    }

    /// Extract frames at the given times (in seconds)
    func extractFrames(count: Int) async throws -> [UIImage] {
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero

        let duration = try await asset.load(.duration)
        let totalSeconds = CMTimeGetSeconds(duration)
        guard totalSeconds > 0 else { return [] }

        let interval = totalSeconds / Double(count + 1)
        var frames: [UIImage] = []

        for i in 1...count {
            let time = CMTime(seconds: interval * Double(i), preferredTimescale: 600)
            do {
                let (cgImage, _) = try await generator.image(at: time)
                frames.append(UIImage(cgImage: cgImage))
            } catch {
                continue
            }
        }
        return frames
    }
}

// MARK: - Motion Magnification Processor

/// Processes pairs of frames through the STB_VMM MotionMag CoreML model
class MotionMagProcessor: ObservableObject {
    @Published var isProcessing = false
    @Published var originalFrames: [UIImage] = []
    @Published var magnifiedFrames: [UIImage] = []
    @Published var errorMessage: String?

    private var model: MLModel?
    private let inputSize = 384

    init() {
        loadModel()
    }

    private func loadModel() {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all

            // Attempt to load compiled model from bundle
            guard let modelURL = Bundle.main.url(forResource: "STB_VMM_MotionMag", withExtension: "mlmodelc") else {
                errorMessage = "Model not found. Please add STB_VMM_MotionMag.mlmodelc to the project bundle."
                return
            }
            model = try MLModel(contentsOf: modelURL, configuration: config)
        } catch {
            errorMessage = "Failed to load model: \(error.localizedDescription)"
        }
    }

    /// Preprocess a UIImage to a normalized pixel buffer (3 channels, 384x384)
    private func preprocessImage(_ image: UIImage) -> [Float]? {
        guard let cgImage = image.cgImage else { return nil }

        let width = inputSize
        let height = inputSize
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var pixelData = [UInt8](repeating: 0, count: width * height * 4)

        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else { return nil }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        // Convert to float [0, 1] in CHW format
        var floatData = [Float](repeating: 0, count: 3 * width * height)
        for y in 0..<height {
            for x in 0..<width {
                let pixelIndex = (y * width + x) * 4
                let spatialIndex = y * width + x
                floatData[0 * width * height + spatialIndex] = Float(pixelData[pixelIndex]) / 255.0     // R
                floatData[1 * width * height + spatialIndex] = Float(pixelData[pixelIndex + 1]) / 255.0 // G
                floatData[2 * width * height + spatialIndex] = Float(pixelData[pixelIndex + 2]) / 255.0 // B
            }
        }
        return floatData
    }

    /// Convert model output back to UIImage
    private func postprocessOutput(_ data: [Float], width: Int, height: Int) -> UIImage? {
        var pixelData = [UInt8](repeating: 255, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let spatialIndex = y * width + x
                let pixelIndex = spatialIndex * 4
                pixelData[pixelIndex]     = UInt8(max(0, min(255, data[0 * width * height + spatialIndex] * 255.0)))
                pixelData[pixelIndex + 1] = UInt8(max(0, min(255, data[1 * width * height + spatialIndex] * 255.0)))
                pixelData[pixelIndex + 2] = UInt8(max(0, min(255, data[2 * width * height + spatialIndex] * 255.0)))
                pixelData[pixelIndex + 3] = 255
            }
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ), let cgImage = context.makeImage() else { return nil }

        return UIImage(cgImage: cgImage)
    }

    /// Run motion magnification on a pair of frames
    func processFramePair(frameA: UIImage, frameB: UIImage, magnification: Float) async {
        guard let model = model else {
            await MainActor.run { errorMessage = "Model is not loaded." }
            return
        }

        await MainActor.run { isProcessing = true }

        do {
            guard let dataA = preprocessImage(frameA),
                  let dataB = preprocessImage(frameB) else {
                await MainActor.run {
                    errorMessage = "Failed to preprocess frames."
                    isProcessing = false
                }
                return
            }

            let size = inputSize
            // Build combined input: 7 channels = frameA(3) + frameB(3) + mag(1)
            let totalElements = 7 * size * size
            var combinedData = [Float](repeating: 0, count: totalElements)

            // Channels 0-2: frameA
            for i in 0..<(3 * size * size) {
                combinedData[i] = dataA[i]
            }
            // Channels 3-5: frameB
            for i in 0..<(3 * size * size) {
                combinedData[3 * size * size + i] = dataB[i]
            }
            // Channel 6: magnification factor (uniform)
            let normalizedMag = magnification / 50.0
            for i in 0..<(size * size) {
                combinedData[6 * size * size + i] = normalizedMag
            }

            // Create MLMultiArray
            let inputArray = try MLMultiArray(shape: [1, 7, 384, 384] as [NSNumber], dataType: .float32)
            let ptr = inputArray.dataPointer.bindMemory(to: Float.self, capacity: totalElements)
            for i in 0..<totalElements {
                ptr[i] = combinedData[i]
            }

            let inputFeatures = try MLDictionaryFeatureProvider(dictionary: ["combined_input": MLFeatureValue(multiArray: inputArray)])
            let output = try model.prediction(from: inputFeatures)

            guard let outputArray = output.featureValue(for: "magnified_frame")?.multiArrayValue else {
                await MainActor.run {
                    errorMessage = "Unexpected model output format."
                    isProcessing = false
                }
                return
            }

            // Extract output data
            let outputSize = 3 * size * size
            var outputData = [Float](repeating: 0, count: outputSize)
            let outPtr = outputArray.dataPointer.bindMemory(to: Float.self, capacity: outputSize)
            for i in 0..<outputSize {
                outputData[i] = outPtr[i]
            }

            let magnifiedImage = postprocessOutput(outputData, width: size, height: size)

            await MainActor.run {
                if let img = magnifiedImage {
                    magnifiedFrames.append(img)
                }
                isProcessing = false
            }
        } catch {
            await MainActor.run {
                errorMessage = "Inference error: \(error.localizedDescription)"
                isProcessing = false
            }
        }
    }

    /// Process all consecutive frame pairs from a video
    func processVideo(frames: [UIImage], magnification: Float) async {
        await MainActor.run {
            originalFrames = frames
            magnifiedFrames = []
            errorMessage = nil
        }

        guard frames.count >= 2 else {
            await MainActor.run { errorMessage = "Need at least 2 frames." }
            return
        }

        for i in 0..<(frames.count - 1) {
            await processFramePair(frameA: frames[i], frameB: frames[i + 1], magnification: magnification)
        }
    }
}

// MARK: - Video Picker

struct VideoPicker: UIViewControllerRepresentable {
    @Binding var videoURL: URL?

    func makeUIViewController(context: Context) -> PHPickerViewController {
        var config = PHPickerConfiguration()
        config.filter = .videos
        config.selectionLimit = 1
        let picker = PHPickerViewController(configuration: config)
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: PHPickerViewController, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    class Coordinator: NSObject, PHPickerViewControllerDelegate {
        let parent: VideoPicker

        init(_ parent: VideoPicker) {
            self.parent = parent
        }

        func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
            picker.dismiss(animated: true)
            guard let provider = results.first?.itemProvider,
                  provider.hasItemConformingToTypeIdentifier("public.movie") else { return }

            provider.loadFileRepresentation(forTypeIdentifier: "public.movie") { url, error in
                guard let url = url else { return }
                // Copy to temporary location
                let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent(url.lastPathComponent)
                try? FileManager.default.removeItem(at: tempURL)
                try? FileManager.default.copyItem(at: url, to: tempURL)
                DispatchQueue.main.async {
                    self.parent.videoURL = tempURL
                }
            }
        }
    }
}

// MARK: - Content View

struct ContentView: View {
    @StateObject private var processor = MotionMagProcessor()
    @State private var magnification: Double = 10.0
    @State private var showVideoPicker = false
    @State private var videoURL: URL?
    @State private var selectedPairIndex = 0

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    // Header
                    headerSection

                    // Error display
                    if let error = processor.errorMessage {
                        errorBanner(error)
                    }

                    // Video picker button
                    Button {
                        showVideoPicker = true
                    } label: {
                        Label("Select Video", systemImage: "video.badge.plus")
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(12)
                    }
                    .padding(.horizontal)

                    // Magnification slider
                    magnificationControl

                    // Process button
                    if videoURL != nil && !processor.isProcessing {
                        Button {
                            processSelectedVideo()
                        } label: {
                            Label("Magnify Motion", systemImage: "waveform.path.ecg")
                                .font(.headline)
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.green)
                                .foregroundColor(.white)
                                .cornerRadius(12)
                        }
                        .padding(.horizontal)
                    }

                    // Processing indicator
                    if processor.isProcessing {
                        ProgressView("Processing frames...")
                            .padding()
                    }

                    // Results comparison
                    if !processor.originalFrames.isEmpty && !processor.magnifiedFrames.isEmpty {
                        resultsSection
                    }
                }
                .padding(.vertical)
            }
            .navigationTitle("Motion Magnification")
            .sheet(isPresented: $showVideoPicker) {
                VideoPicker(videoURL: $videoURL)
            }
        }
    }

    // MARK: - Subviews

    private var headerSection: some View {
        VStack(spacing: 8) {
            Image(systemName: "waveform.path.ecg.rectangle")
                .font(.system(size: 50))
                .foregroundColor(.blue)
            Text("Video Motion Magnification")
                .font(.title2.bold())
            Text("Amplify subtle motions in video using STB-VMM")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding()
    }

    private func errorBanner(_ message: String) -> some View {
        HStack {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundColor(.yellow)
            Text(message)
                .font(.caption)
        }
        .padding()
        .background(Color.red.opacity(0.1))
        .cornerRadius(8)
        .padding(.horizontal)
    }

    private var magnificationControl: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Magnification Factor")
                    .font(.headline)
                Spacer()
                Text("\(Int(magnification))x")
                    .font(.title3.bold())
                    .foregroundColor(.blue)
            }
            Slider(value: $magnification, in: 1...50, step: 1)
                .tint(.blue)
            HStack {
                Text("1x")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Spacer()
                Text("50x")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
        .padding(.horizontal)
    }

    private var resultsSection: some View {
        VStack(spacing: 16) {
            Text("Results")
                .font(.title3.bold())

            // Frame pair selector
            if processor.magnifiedFrames.count > 1 {
                Picker("Frame Pair", selection: $selectedPairIndex) {
                    ForEach(0..<processor.magnifiedFrames.count, id: \.self) { i in
                        Text("Pair \(i + 1)").tag(i)
                    }
                }
                .pickerStyle(.segmented)
                .padding(.horizontal)
            }

            // Side by side comparison
            if selectedPairIndex < processor.originalFrames.count &&
               selectedPairIndex < processor.magnifiedFrames.count {
                HStack(spacing: 12) {
                    VStack {
                        Text("Original")
                            .font(.caption.bold())
                        Image(uiImage: processor.originalFrames[selectedPairIndex])
                            .resizable()
                            .scaledToFit()
                            .cornerRadius(8)
                    }
                    VStack {
                        Text("Magnified")
                            .font(.caption.bold())
                        Image(uiImage: processor.magnifiedFrames[selectedPairIndex])
                            .resizable()
                            .scaledToFit()
                            .cornerRadius(8)
                    }
                }
                .padding(.horizontal)
            }
        }
        .padding(.vertical)
    }

    // MARK: - Actions

    private func processSelectedVideo() {
        guard let url = videoURL else { return }
        let asset = AVAsset(url: url)
        Task {
            let extractor = VideoFrameExtractor(asset: asset)
            let frames = try await extractor.extractFrames(count: 6)
            await processor.processVideo(frames: frames, magnification: Float(magnification))
        }
    }
}

#Preview {
    ContentView()
}
