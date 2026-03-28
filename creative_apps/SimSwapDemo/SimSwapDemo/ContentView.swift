import SwiftUI
import UIKit
import CoreML
import Vision
import PhotosUI

// MARK: - Pipeline Step

/// Represents the current step in the face swap pipeline
enum PipelineStep: String, CaseIterable {
    case idle = "Ready"
    case detectingSourceFace = "Detecting source face..."
    case extractingIdentity = "Extracting identity embedding..."
    case detectingTargetFace = "Detecting target face..."
    case generatingSwap = "Generating face swap..."
    case complete = "Complete"
    case error = "Error"
}

// MARK: - Face Swap Processor

/// Two-stage face swap pipeline: ArcFace embedding + SimSwap generator
class FaceSwapProcessor: ObservableObject {
    @Published var sourceImage: UIImage?
    @Published var targetImage: UIImage?
    @Published var resultImage: UIImage?
    @Published var isProcessing = false
    @Published var currentStep: PipelineStep = .idle
    @Published var errorMessage: String?

    // Cropped face images for display
    @Published var sourceFaceCrop: UIImage?
    @Published var targetFaceCrop: UIImage?

    private var arcFaceModel: MLModel?
    private var generatorModel: MLModel?

    init() {
        loadModels()
    }

    private func loadModels() {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all

            // Load ArcFace model
            if let arcFaceURL = Bundle.main.url(forResource: "SimSwap_ArcFace", withExtension: "mlmodelc") {
                arcFaceModel = try MLModel(contentsOf: arcFaceURL, configuration: config)
            } else {
                errorMessage = "ArcFace model not found. Please add SimSwap_ArcFace.mlmodelc to the bundle."
            }

            // Load Generator model
            if let genURL = Bundle.main.url(forResource: "SimSwap_Generator", withExtension: "mlmodelc") {
                generatorModel = try MLModel(contentsOf: genURL, configuration: config)
            } else {
                let msg = "Generator model not found. Please add SimSwap_Generator.mlmodelc to the bundle."
                errorMessage = errorMessage != nil ? "\(errorMessage!) \(msg)" : msg
            }
        } catch {
            errorMessage = "Failed to load models: \(error.localizedDescription)"
        }
    }

    /// Detect the largest face in an image using Vision and return its bounding box
    private func detectFace(in image: UIImage) async throws -> CGRect? {
        guard let cgImage = image.cgImage else { return nil }

        return try await withCheckedThrowingContinuation { continuation in
            let request = VNDetectFaceRectanglesRequest { request, error in
                if let error = error {
                    continuation.resume(throwing: error)
                    return
                }
                // Get the largest face
                let faces = request.results as? [VNFaceObservation] ?? []
                let largestFace = faces.max(by: { $0.boundingBox.width * $0.boundingBox.height < $1.boundingBox.width * $1.boundingBox.height })
                continuation.resume(returning: largestFace?.boundingBox)
            }

            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            do {
                try handler.perform([request])
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }

    /// Crop face region from image with some padding
    private func cropFace(from image: UIImage, boundingBox: CGRect, targetSize: CGSize) -> UIImage? {
        guard let cgImage = image.cgImage else { return nil }

        let imageWidth = CGFloat(cgImage.width)
        let imageHeight = CGFloat(cgImage.height)

        // Convert Vision coordinates (origin at bottom-left) to CGImage coordinates (origin at top-left)
        let x = boundingBox.origin.x * imageWidth
        let y = (1.0 - boundingBox.origin.y - boundingBox.height) * imageHeight
        let w = boundingBox.width * imageWidth
        let h = boundingBox.height * imageHeight

        // Add 20% padding
        let padding: CGFloat = 0.2
        let padX = w * padding
        let padY = h * padding
        let cropRect = CGRect(
            x: max(0, x - padX),
            y: max(0, y - padY),
            width: min(imageWidth - max(0, x - padX), w + 2 * padX),
            height: min(imageHeight - max(0, y - padY), h + 2 * padY)
        )

        guard let croppedCGImage = cgImage.cropping(to: cropRect) else { return nil }

        // Resize to target size
        let renderer = UIGraphicsImageRenderer(size: targetSize)
        let resized = renderer.image { _ in
            UIImage(cgImage: croppedCGImage).draw(in: CGRect(origin: .zero, size: targetSize))
        }
        return resized
    }

    /// Convert UIImage to CHW float array
    private func imageToFloatArray(_ image: UIImage, size: Int) -> [Float]? {
        guard let cgImage = image.cgImage else { return nil }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var pixelData = [UInt8](repeating: 0, count: size * size * 4)

        guard let context = CGContext(
            data: &pixelData,
            width: size,
            height: size,
            bitsPerComponent: 8,
            bytesPerRow: size * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else { return nil }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: size, height: size))

        var floatData = [Float](repeating: 0, count: 3 * size * size)
        for y in 0..<size {
            for x in 0..<size {
                let pixelIndex = (y * size + x) * 4
                let spatialIndex = y * size + x
                floatData[0 * size * size + spatialIndex] = Float(pixelData[pixelIndex]) / 255.0
                floatData[1 * size * size + spatialIndex] = Float(pixelData[pixelIndex + 1]) / 255.0
                floatData[2 * size * size + spatialIndex] = Float(pixelData[pixelIndex + 2]) / 255.0
            }
        }
        return floatData
    }

    /// Convert CHW float array to UIImage
    private func floatArrayToImage(_ data: [Float], size: Int) -> UIImage? {
        var pixelData = [UInt8](repeating: 255, count: size * size * 4)
        for y in 0..<size {
            for x in 0..<size {
                let spatialIndex = y * size + x
                let pixelIndex = spatialIndex * 4
                pixelData[pixelIndex]     = UInt8(max(0, min(255, data[0 * size * size + spatialIndex] * 255.0)))
                pixelData[pixelIndex + 1] = UInt8(max(0, min(255, data[1 * size * size + spatialIndex] * 255.0)))
                pixelData[pixelIndex + 2] = UInt8(max(0, min(255, data[2 * size * size + spatialIndex] * 255.0)))
                pixelData[pixelIndex + 3] = 255
            }
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &pixelData,
            width: size,
            height: size,
            bitsPerComponent: 8,
            bytesPerRow: size * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ), let cgImage = context.makeImage() else { return nil }

        return UIImage(cgImage: cgImage)
    }

    /// Run the full face swap pipeline
    func performFaceSwap() async {
        guard let sourceImage = sourceImage, let targetImage = targetImage else {
            await MainActor.run { errorMessage = "Please select both source and target images." }
            return
        }

        guard let arcFaceModel = arcFaceModel, let generatorModel = generatorModel else {
            await MainActor.run { errorMessage = "Models are not loaded." }
            return
        }

        await MainActor.run {
            isProcessing = true
            resultImage = nil
            errorMessage = nil
            currentStep = .detectingSourceFace
        }

        do {
            // Step 1: Detect and crop source face (112x112 for ArcFace)
            guard let sourceBox = try await detectFace(in: sourceImage) else {
                await MainActor.run {
                    errorMessage = "No face detected in source image."
                    currentStep = .error
                    isProcessing = false
                }
                return
            }

            let sourceFace = cropFace(from: sourceImage, boundingBox: sourceBox, targetSize: CGSize(width: 112, height: 112))
            await MainActor.run { sourceFaceCrop = sourceFace }

            // Step 2: Extract identity embedding via ArcFace
            await MainActor.run { currentStep = .extractingIdentity }

            guard let sourceFace = sourceFace,
                  let sourceData = imageToFloatArray(sourceFace, size: 112) else {
                await MainActor.run {
                    errorMessage = "Failed to preprocess source face."
                    currentStep = .error
                    isProcessing = false
                }
                return
            }

            let arcFaceInput = try MLMultiArray(shape: [1, 3, 112, 112] as [NSNumber], dataType: .float32)
            let arcPtr = arcFaceInput.dataPointer.bindMemory(to: Float.self, capacity: sourceData.count)
            for i in 0..<sourceData.count {
                arcPtr[i] = sourceData[i]
            }

            let arcFaceFeatures = try MLDictionaryFeatureProvider(dictionary: ["input": MLFeatureValue(multiArray: arcFaceInput)])
            let arcFaceOutput = try arcFaceModel.prediction(from: arcFaceFeatures)

            // Get the 512-d identity embedding
            guard let embeddingArray = arcFaceOutput.featureValue(for: "output")?.multiArrayValue else {
                await MainActor.run {
                    errorMessage = "Failed to extract identity embedding."
                    currentStep = .error
                    isProcessing = false
                }
                return
            }

            // Step 3: Detect and crop target face (224x224 for Generator)
            await MainActor.run { currentStep = .detectingTargetFace }

            guard let targetBox = try await detectFace(in: targetImage) else {
                await MainActor.run {
                    errorMessage = "No face detected in target image."
                    currentStep = .error
                    isProcessing = false
                }
                return
            }

            let targetFace = cropFace(from: targetImage, boundingBox: targetBox, targetSize: CGSize(width: 224, height: 224))
            await MainActor.run { targetFaceCrop = targetFace }

            // Step 4: Generate face swap
            await MainActor.run { currentStep = .generatingSwap }

            guard let targetFace = targetFace,
                  let targetData = imageToFloatArray(targetFace, size: 224) else {
                await MainActor.run {
                    errorMessage = "Failed to preprocess target face."
                    currentStep = .error
                    isProcessing = false
                }
                return
            }

            let targetInput = try MLMultiArray(shape: [1, 3, 224, 224] as [NSNumber], dataType: .float32)
            let targetPtr = targetInput.dataPointer.bindMemory(to: Float.self, capacity: targetData.count)
            for i in 0..<targetData.count {
                targetPtr[i] = targetData[i]
            }

            let genFeatures = try MLDictionaryFeatureProvider(dictionary: [
                "target": MLFeatureValue(multiArray: targetInput),
                "identity": MLFeatureValue(multiArray: embeddingArray)
            ])
            let genOutput = try generatorModel.prediction(from: genFeatures)

            guard let outputArray = genOutput.featureValue(for: "output")?.multiArrayValue else {
                await MainActor.run {
                    errorMessage = "Failed to generate swapped face."
                    currentStep = .error
                    isProcessing = false
                }
                return
            }

            let outputSize = 3 * 224 * 224
            var outputData = [Float](repeating: 0, count: outputSize)
            let outPtr = outputArray.dataPointer.bindMemory(to: Float.self, capacity: outputSize)
            for i in 0..<outputSize {
                outputData[i] = outPtr[i]
            }

            let result = floatArrayToImage(outputData, size: 224)

            await MainActor.run {
                resultImage = result
                currentStep = .complete
                isProcessing = false
            }
        } catch {
            await MainActor.run {
                errorMessage = "Pipeline error: \(error.localizedDescription)"
                currentStep = .error
                isProcessing = false
            }
        }
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

// MARK: - Pipeline Step View

struct PipelineStepRow: View {
    let step: PipelineStep
    let currentStep: PipelineStep
    let allSteps: [PipelineStep]

    private var stepIndex: Int { allSteps.firstIndex(of: step) ?? 0 }
    private var currentIndex: Int { allSteps.firstIndex(of: currentStep) ?? 0 }

    private var status: StepStatus {
        if currentStep == .error && step == allSteps[currentIndex] { return .error }
        if stepIndex < currentIndex { return .completed }
        if stepIndex == currentIndex { return .active }
        return .pending
    }

    enum StepStatus {
        case pending, active, completed, error
    }

    var body: some View {
        HStack(spacing: 12) {
            ZStack {
                Circle()
                    .fill(statusColor)
                    .frame(width: 28, height: 28)
                statusIcon
            }
            Text(step.rawValue)
                .font(.subheadline)
                .foregroundColor(status == .pending ? .secondary : .primary)
            Spacer()
            if status == .active {
                ProgressView()
                    .scaleEffect(0.8)
            }
        }
    }

    private var statusColor: Color {
        switch status {
        case .pending: return Color(.systemGray4)
        case .active: return .blue
        case .completed: return .green
        case .error: return .red
        }
    }

    @ViewBuilder
    private var statusIcon: some View {
        switch status {
        case .pending:
            Text("\(stepIndex + 1)")
                .font(.caption2.bold())
                .foregroundColor(.white)
        case .active:
            Text("\(stepIndex + 1)")
                .font(.caption2.bold())
                .foregroundColor(.white)
        case .completed:
            Image(systemName: "checkmark")
                .font(.caption2.bold())
                .foregroundColor(.white)
        case .error:
            Image(systemName: "xmark")
                .font(.caption2.bold())
                .foregroundColor(.white)
        }
    }
}

// MARK: - Content View

struct ContentView: View {
    @StateObject private var processor = FaceSwapProcessor()
    @State private var showSourcePicker = false
    @State private var showTargetPicker = false

    private let pipelineSteps: [PipelineStep] = [
        .detectingSourceFace,
        .extractingIdentity,
        .detectingTargetFace,
        .generatingSwap,
        .complete
    ]

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

                    // Image selection
                    imageSelectionSection

                    // Run button
                    if processor.sourceImage != nil && processor.targetImage != nil && !processor.isProcessing {
                        Button {
                            Task { await processor.performFaceSwap() }
                        } label: {
                            Label("Swap Faces", systemImage: "arrow.triangle.swap")
                                .font(.headline)
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.purple)
                                .foregroundColor(.white)
                                .cornerRadius(12)
                        }
                        .padding(.horizontal)
                    }

                    // Pipeline visualization
                    if processor.currentStep != .idle {
                        pipelineSection
                    }

                    // Face crops display
                    faceCropsSection

                    // Result
                    if let result = processor.resultImage {
                        resultSection(result)
                    }

                    Spacer(minLength: 40)
                }
                .padding(.vertical)
            }
            .navigationTitle("SimSwap Face Swap")
            .sheet(isPresented: $showSourcePicker) {
                ImagePicker(image: $processor.sourceImage)
            }
            .sheet(isPresented: $showTargetPicker) {
                ImagePicker(image: $processor.targetImage)
            }
        }
    }

    // MARK: - Subviews

    private var headerSection: some View {
        VStack(spacing: 8) {
            Image(systemName: "person.2.crop.square.stack")
                .font(.system(size: 50))
                .foregroundColor(.purple)
            Text("Face Swap")
                .font(.title2.bold())
            Text("Transfer identity from one face to another using SimSwap")
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

    private var imageSelectionSection: some View {
        HStack(spacing: 12) {
            // Source face button/preview
            VStack(spacing: 8) {
                Text("Source (Identity)")
                    .font(.caption.bold())
                    .foregroundColor(.secondary)

                Button {
                    showSourcePicker = true
                } label: {
                    if let image = processor.sourceImage {
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFill()
                            .frame(width: 140, height: 140)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                    } else {
                        VStack(spacing: 8) {
                            Image(systemName: "person.crop.rectangle")
                                .font(.title)
                            Text("Select")
                                .font(.caption)
                        }
                        .frame(width: 140, height: 140)
                        .background(Color(.systemGray5))
                        .cornerRadius(12)
                    }
                }
                .foregroundColor(.primary)
            }

            Image(systemName: "arrow.right")
                .font(.title2)
                .foregroundColor(.secondary)

            // Target face button/preview
            VStack(spacing: 8) {
                Text("Target (Pose)")
                    .font(.caption.bold())
                    .foregroundColor(.secondary)

                Button {
                    showTargetPicker = true
                } label: {
                    if let image = processor.targetImage {
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFill()
                            .frame(width: 140, height: 140)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                    } else {
                        VStack(spacing: 8) {
                            Image(systemName: "person.crop.rectangle")
                                .font(.title)
                            Text("Select")
                                .font(.caption)
                        }
                        .frame(width: 140, height: 140)
                        .background(Color(.systemGray5))
                        .cornerRadius(12)
                    }
                }
                .foregroundColor(.primary)
            }
        }
        .padding(.horizontal)
    }

    private var pipelineSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Pipeline Progress")
                .font(.headline)

            ForEach(pipelineSteps, id: \.self) { step in
                PipelineStepRow(step: step, currentStep: processor.currentStep, allSteps: pipelineSteps)
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
        .padding(.horizontal)
    }

    private var faceCropsSection: some View {
        Group {
            if processor.sourceFaceCrop != nil || processor.targetFaceCrop != nil {
                VStack(spacing: 8) {
                    Text("Detected Faces")
                        .font(.headline)
                    HStack(spacing: 16) {
                        if let crop = processor.sourceFaceCrop {
                            VStack {
                                Text("Source 112x112")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                                Image(uiImage: crop)
                                    .resizable()
                                    .scaledToFit()
                                    .frame(width: 100, height: 100)
                                    .cornerRadius(8)
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 8)
                                            .stroke(Color.purple, lineWidth: 2)
                                    )
                            }
                        }
                        if let crop = processor.targetFaceCrop {
                            VStack {
                                Text("Target 224x224")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                                Image(uiImage: crop)
                                    .resizable()
                                    .scaledToFit()
                                    .frame(width: 100, height: 100)
                                    .cornerRadius(8)
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 8)
                                            .stroke(Color.orange, lineWidth: 2)
                                    )
                            }
                        }
                    }
                }
                .padding(.horizontal)
            }
        }
    }

    private func resultSection(_ image: UIImage) -> some View {
        VStack(spacing: 8) {
            Text("Swapped Result")
                .font(.title3.bold())
            Image(uiImage: image)
                .resizable()
                .scaledToFit()
                .frame(maxHeight: 300)
                .cornerRadius(12)
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(Color.purple, lineWidth: 3)
                )
                .shadow(color: .purple.opacity(0.3), radius: 10)
                .padding(.horizontal)
        }
    }
}

#Preview {
    ContentView()
}
