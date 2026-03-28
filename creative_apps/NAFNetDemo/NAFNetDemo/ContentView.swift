import SwiftUI
import UIKit
import CoreML
import PhotosUI

// MARK: - NAFNet Deblurring Processor

/// Handles image deblurring using the NAFNet CoreML model
class DeblurProcessor: ObservableObject {
    @Published var inputImage: UIImage?
    @Published var outputImage: UIImage?
    @Published var isProcessing = false
    @Published var errorMessage: String?
    @Published var inferenceTime: Double = 0

    private var model: MLModel?
    private let inputSize = 256

    init() {
        loadModel()
    }

    private func loadModel() {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all

            guard let modelURL = Bundle.main.url(forResource: "NAFNet_Deblur", withExtension: "mlmodelc") else {
                errorMessage = "Model not found. Please add NAFNet_Deblur.mlmodelc to the project bundle."
                return
            }
            model = try MLModel(contentsOf: modelURL, configuration: config)
        } catch {
            errorMessage = "Failed to load model: \(error.localizedDescription)"
        }
    }

    /// Convert UIImage to CHW float array normalized to [0, 1]
    private func imageToFloatArray(_ image: UIImage) -> [Float]? {
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

        var floatData = [Float](repeating: 0, count: 3 * width * height)
        for y in 0..<height {
            for x in 0..<width {
                let pixelIndex = (y * width + x) * 4
                let spatialIndex = y * width + x
                floatData[0 * width * height + spatialIndex] = Float(pixelData[pixelIndex]) / 255.0
                floatData[1 * width * height + spatialIndex] = Float(pixelData[pixelIndex + 1]) / 255.0
                floatData[2 * width * height + spatialIndex] = Float(pixelData[pixelIndex + 2]) / 255.0
            }
        }
        return floatData
    }

    /// Convert CHW float array back to UIImage
    private func floatArrayToImage(_ data: [Float], width: Int, height: Int) -> UIImage? {
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

    /// Run deblurring inference
    func deblur(image: UIImage) async {
        guard let model = model else {
            await MainActor.run { errorMessage = "Model is not loaded." }
            return
        }

        await MainActor.run {
            inputImage = image
            outputImage = nil
            isProcessing = true
            errorMessage = nil
            inferenceTime = 0
        }

        do {
            guard let inputData = imageToFloatArray(image) else {
                await MainActor.run {
                    errorMessage = "Failed to preprocess image."
                    isProcessing = false
                }
                return
            }

            let inputArray = try MLMultiArray(shape: [1, 3, 256, 256] as [NSNumber], dataType: .float32)
            let ptr = inputArray.dataPointer.bindMemory(to: Float.self, capacity: inputData.count)
            for i in 0..<inputData.count {
                ptr[i] = inputData[i]
            }

            let inputFeatures = try MLDictionaryFeatureProvider(dictionary: ["blurry_image": MLFeatureValue(multiArray: inputArray)])

            let startTime = CFAbsoluteTimeGetCurrent()
            let output = try model.prediction(from: inputFeatures)
            let endTime = CFAbsoluteTimeGetCurrent()
            let elapsed = (endTime - startTime) * 1000.0

            guard let outputArray = output.featureValue(for: "deblurred_image")?.multiArrayValue else {
                await MainActor.run {
                    errorMessage = "Unexpected model output format."
                    isProcessing = false
                }
                return
            }

            let outputSize = 3 * inputSize * inputSize
            var outputData = [Float](repeating: 0, count: outputSize)
            let outPtr = outputArray.dataPointer.bindMemory(to: Float.self, capacity: outputSize)
            for i in 0..<outputSize {
                outputData[i] = outPtr[i]
            }

            let resultImage = floatArrayToImage(outputData, width: inputSize, height: inputSize)

            await MainActor.run {
                outputImage = resultImage
                inferenceTime = elapsed
                isProcessing = false
            }
        } catch {
            await MainActor.run {
                errorMessage = "Inference error: \(error.localizedDescription)"
                isProcessing = false
            }
        }
    }
}

// MARK: - Slider Comparison View

/// A draggable slider overlay to compare before/after images
struct SliderComparisonView: View {
    let beforeImage: UIImage
    let afterImage: UIImage
    @State private var sliderPosition: CGFloat = 0.5

    var body: some View {
        GeometryReader { geometry in
            let width = geometry.size.width
            let height = geometry.size.height

            ZStack {
                // After image (full background)
                Image(uiImage: afterImage)
                    .resizable()
                    .scaledToFill()
                    .frame(width: width, height: height)
                    .clipped()

                // Before image (clipped to left of slider)
                Image(uiImage: beforeImage)
                    .resizable()
                    .scaledToFill()
                    .frame(width: width, height: height)
                    .clipped()
                    .clipShape(
                        HorizontalClipShape(position: sliderPosition)
                    )

                // Slider line and handle
                Rectangle()
                    .fill(Color.white)
                    .frame(width: 2, height: height)
                    .position(x: width * sliderPosition, y: height / 2)
                    .shadow(radius: 2)

                Circle()
                    .fill(Color.white)
                    .frame(width: 36, height: 36)
                    .shadow(radius: 3)
                    .overlay(
                        HStack(spacing: 2) {
                            Image(systemName: "chevron.left")
                                .font(.system(size: 10, weight: .bold))
                            Image(systemName: "chevron.right")
                                .font(.system(size: 10, weight: .bold))
                        }
                        .foregroundColor(.gray)
                    )
                    .position(x: width * sliderPosition, y: height / 2)

                // Labels
                VStack {
                    HStack {
                        Text("Blurry")
                            .font(.caption.bold())
                            .padding(4)
                            .background(Color.black.opacity(0.6))
                            .foregroundColor(.white)
                            .cornerRadius(4)
                        Spacer()
                        Text("Deblurred")
                            .font(.caption.bold())
                            .padding(4)
                            .background(Color.black.opacity(0.6))
                            .foregroundColor(.white)
                            .cornerRadius(4)
                    }
                    .padding(8)
                    Spacer()
                }
            }
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        sliderPosition = max(0.01, min(0.99, value.location.x / width))
                    }
            )
        }
    }
}

/// Clip shape that reveals content to the left of a horizontal position
struct HorizontalClipShape: Shape {
    var position: CGFloat

    var animatableData: CGFloat {
        get { position }
        set { position = newValue }
    }

    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.addRect(CGRect(x: 0, y: 0, width: rect.width * position, height: rect.height))
        return path
    }
}

// MARK: - Image Picker

struct ImagePicker: UIViewControllerRepresentable {
    @Binding var image: UIImage?

    func makeUIViewController(context: Context) -> PHPickerViewController {
        var config = PHPickerConfiguration()
        config.filter = .images
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
        let parent: ImagePicker

        init(_ parent: ImagePicker) {
            self.parent = parent
        }

        func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
            picker.dismiss(animated: true)
            guard let provider = results.first?.itemProvider,
                  provider.canLoadObject(ofClass: UIImage.self) else { return }
            provider.loadObject(ofClass: UIImage.self) { image, _ in
                DispatchQueue.main.async {
                    self.parent.image = image as? UIImage
                }
            }
        }
    }
}

// MARK: - Content View

struct ContentView: View {
    @StateObject private var processor = DeblurProcessor()
    @State private var showImagePicker = false
    @State private var selectedImage: UIImage?

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

                    // Pick image button
                    Button {
                        showImagePicker = true
                    } label: {
                        Label("Pick Blurry Photo", systemImage: "photo.badge.plus")
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(12)
                    }
                    .padding(.horizontal)

                    // Processing indicator
                    if processor.isProcessing {
                        ProgressView("Deblurring image...")
                            .padding()
                    }

                    // Inference time
                    if processor.inferenceTime > 0 {
                        HStack {
                            Image(systemName: "clock")
                                .foregroundColor(.orange)
                            Text(String(format: "Inference time: %.1f ms", processor.inferenceTime))
                                .font(.subheadline.bold())
                                .foregroundColor(.orange)
                        }
                        .padding(.horizontal)
                    }

                    // Comparison view
                    if let input = processor.inputImage, let output = processor.outputImage {
                        VStack(spacing: 8) {
                            Text("Drag to Compare")
                                .font(.headline)
                            SliderComparisonView(beforeImage: input, afterImage: output)
                                .frame(height: 300)
                                .cornerRadius(12)
                                .padding(.horizontal)
                        }
                    } else if let input = processor.inputImage {
                        // Show just the input if no output yet
                        VStack(spacing: 8) {
                            Text("Input Image")
                                .font(.headline)
                            Image(uiImage: input)
                                .resizable()
                                .scaledToFit()
                                .frame(maxHeight: 300)
                                .cornerRadius(12)
                                .padding(.horizontal)
                        }
                    }

                    // Side by side view
                    if let input = processor.inputImage, let output = processor.outputImage {
                        VStack(spacing: 8) {
                            Text("Side by Side")
                                .font(.headline)
                            HStack(spacing: 8) {
                                VStack {
                                    Text("Before")
                                        .font(.caption.bold())
                                        .foregroundColor(.secondary)
                                    Image(uiImage: input)
                                        .resizable()
                                        .scaledToFit()
                                        .cornerRadius(8)
                                }
                                VStack {
                                    Text("After")
                                        .font(.caption.bold())
                                        .foregroundColor(.secondary)
                                    Image(uiImage: output)
                                        .resizable()
                                        .scaledToFit()
                                        .cornerRadius(8)
                                }
                            }
                            .padding(.horizontal)
                        }
                    }

                    Spacer(minLength: 40)
                }
                .padding(.vertical)
            }
            .navigationTitle("NAFNet Deblur")
            .sheet(isPresented: $showImagePicker) {
                ImagePicker(image: $selectedImage)
            }
            .onChange(of: selectedImage) { newValue in
                guard let image = newValue else { return }
                Task {
                    await processor.deblur(image: image)
                }
            }
        }
    }

    // MARK: - Subviews

    private var headerSection: some View {
        VStack(spacing: 8) {
            Image(systemName: "camera.filters")
                .font(.system(size: 50))
                .foregroundColor(.blue)
            Text("Image Deblurring")
                .font(.title2.bold())
            Text("Remove blur from photos using NAFNet neural network")
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
}

#Preview {
    ContentView()
}
