import SwiftUI
import UIKit
import CoreML
import Vision
import PhotosUI

// MARK: - FOMM (First Order Motion Model) Face Reenactment Demo
//
// Two-model pipeline:
// 1. FOMM_KPDetector: Detects 10 facial keypoints + 2x2 Jacobian matrices
//    Input:  image (1,3,256,256)
//    Output: keypoints (1,10,2) + jacobians (1,10,2,2)
//
// 2. FOMM_Generator: Generates reenacted face from source + keypoint pairs
//    Input:  source_image (1,3,256,256) + source/driving keypoints & jacobians
//    Output: prediction (1,3,256,256)

struct ContentView: View {
    @StateObject private var viewModel = FOMMViewModel()

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    // Source and Driving image pickers side by side
                    HStack(spacing: 12) {
                        // Source face
                        VStack(spacing: 8) {
                            sectionHeader("Source Face")
                            PhotosPicker(selection: $viewModel.selectedSourcePhoto,
                                         matching: .images) {
                                if let image = viewModel.sourceImage {
                                    ZStack {
                                        Image(uiImage: image)
                                            .resizable()
                                            .scaledToFill()
                                            .frame(width: 150, height: 150)
                                            .clipped()
                                            .cornerRadius(12)

                                        // Keypoint overlay on source
                                        if !viewModel.sourceKeypoints.isEmpty {
                                            KeypointOverlay(
                                                keypoints: viewModel.sourceKeypoints,
                                                color: .green
                                            )
                                            .frame(width: 150, height: 150)
                                        }
                                    }
                                } else {
                                    placeholderView(
                                        systemImage: "person.crop.square",
                                        size: 150
                                    )
                                }
                            }
                        }

                        // Driving face
                        VStack(spacing: 8) {
                            sectionHeader("Driving Face")
                            PhotosPicker(selection: $viewModel.selectedDrivingPhoto,
                                         matching: .images) {
                                if let image = viewModel.drivingImage {
                                    ZStack {
                                        Image(uiImage: image)
                                            .resizable()
                                            .scaledToFill()
                                            .frame(width: 150, height: 150)
                                            .clipped()
                                            .cornerRadius(12)

                                        // Keypoint overlay on driving
                                        if !viewModel.drivingKeypoints.isEmpty {
                                            KeypointOverlay(
                                                keypoints: viewModel.drivingKeypoints,
                                                color: .orange
                                            )
                                            .frame(width: 150, height: 150)
                                        }
                                    }
                                } else {
                                    placeholderView(
                                        systemImage: "person.crop.square.filled.and.at.rectangle",
                                        size: 150
                                    )
                                }
                            }
                        }
                    }

                    // Detect keypoints button
                    if viewModel.sourceImage != nil && viewModel.drivingImage != nil {
                        Button(action: { viewModel.detectKeypoints() }) {
                            HStack {
                                if viewModel.isDetectingKeypoints {
                                    ProgressView()
                                        .tint(.white)
                                } else {
                                    Image(systemName: "dot.radiowaves.left.and.right")
                                }
                                Text("Detect Keypoints")
                            }
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(viewModel.isDetectingKeypoints ? Color.gray : Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(12)
                        }
                        .disabled(viewModel.isDetectingKeypoints)
                    }

                    // Keypoint info
                    if !viewModel.sourceKeypoints.isEmpty {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Detected Keypoints")
                                .font(.headline)

                            HStack(spacing: 20) {
                                VStack(alignment: .leading) {
                                    Text("Source: \(viewModel.sourceKeypoints.count) points")
                                        .foregroundColor(.green)
                                    Text("+ \(viewModel.sourceKeypoints.count) Jacobians (2x2)")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                                Spacer()
                                VStack(alignment: .leading) {
                                    Text("Driving: \(viewModel.drivingKeypoints.count) points")
                                        .foregroundColor(.orange)
                                    Text("+ \(viewModel.drivingKeypoints.count) Jacobians (2x2)")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                            }
                        }
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(12)
                    }

                    // Generate button
                    if !viewModel.sourceKeypoints.isEmpty && !viewModel.drivingKeypoints.isEmpty {
                        Button(action: { viewModel.generateReenactment() }) {
                            HStack {
                                if viewModel.isGenerating {
                                    ProgressView()
                                        .tint(.white)
                                } else {
                                    Image(systemName: "face.smiling")
                                }
                                Text("Generate Reenactment")
                            }
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(viewModel.isGenerating ? Color.gray : Color.accentColor)
                            .foregroundColor(.white)
                            .cornerRadius(12)
                        }
                        .disabled(viewModel.isGenerating)
                    }

                    // Error display
                    if let error = viewModel.errorMessage {
                        Text(error)
                            .foregroundColor(.red)
                            .font(.caption)
                            .padding()
                            .frame(maxWidth: .infinity)
                            .background(Color.red.opacity(0.1))
                            .cornerRadius(8)
                    }

                    // Result
                    if let result = viewModel.resultImage {
                        Section {
                            VStack(spacing: 12) {
                                Text("Reenacted Face")
                                    .font(.headline)
                                    .frame(maxWidth: .infinity, alignment: .leading)

                                Image(uiImage: result)
                                    .resizable()
                                    .scaledToFit()
                                    .frame(maxHeight: 300)
                                    .cornerRadius(12)

                                // Comparison row
                                HStack(spacing: 8) {
                                    if let src = viewModel.sourceImage {
                                        VStack {
                                            Image(uiImage: src)
                                                .resizable()
                                                .scaledToFill()
                                                .frame(width: 80, height: 80)
                                                .clipped()
                                                .cornerRadius(8)
                                            Text("Source")
                                                .font(.caption2)
                                        }
                                    }
                                    Image(systemName: "plus")
                                        .foregroundColor(.secondary)
                                    if let drv = viewModel.drivingImage {
                                        VStack {
                                            Image(uiImage: drv)
                                                .resizable()
                                                .scaledToFill()
                                                .frame(width: 80, height: 80)
                                                .clipped()
                                                .cornerRadius(8)
                                            Text("Driving")
                                                .font(.caption2)
                                        }
                                    }
                                    Image(systemName: "arrow.right")
                                        .foregroundColor(.secondary)
                                    VStack {
                                        Image(uiImage: result)
                                            .resizable()
                                            .scaledToFill()
                                            .frame(width: 80, height: 80)
                                            .clipped()
                                            .cornerRadius(8)
                                        Text("Result")
                                            .font(.caption2)
                                    }
                                }
                            }
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("FOMM Reenactment")
        }
    }

    private func sectionHeader(_ title: String) -> some View {
        Text(title)
            .font(.caption)
            .fontWeight(.semibold)
            .foregroundColor(.secondary)
    }

    private func placeholderView(systemImage: String, size: CGFloat) -> some View {
        VStack(spacing: 8) {
            Image(systemName: systemImage)
                .font(.system(size: 30))
                .foregroundColor(.secondary)
            Text("Select")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .frame(width: size, height: size)
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

// MARK: - Keypoint Overlay

struct KeypointOverlay: View {
    let keypoints: [CGPoint]
    let color: Color

    var body: some View {
        GeometryReader { geo in
            ForEach(0..<keypoints.count, id: \.self) { i in
                let point = keypoints[i]
                Circle()
                    .fill(color)
                    .frame(width: 8, height: 8)
                    .position(
                        x: point.x * geo.size.width,
                        y: point.y * geo.size.height
                    )
                    .shadow(color: color, radius: 2)

                // Draw index label
                Text("\(i)")
                    .font(.system(size: 7, weight: .bold))
                    .foregroundColor(.white)
                    .position(
                        x: point.x * geo.size.width,
                        y: point.y * geo.size.height
                    )
            }

            // Draw connections between adjacent keypoints
            Path { path in
                guard keypoints.count >= 2 else { return }
                for i in 0..<keypoints.count {
                    let next = (i + 1) % keypoints.count
                    let from = CGPoint(
                        x: keypoints[i].x * geo.size.width,
                        y: keypoints[i].y * geo.size.height
                    )
                    let to = CGPoint(
                        x: keypoints[next].x * geo.size.width,
                        y: keypoints[next].y * geo.size.height
                    )
                    path.move(to: from)
                    path.addLine(to: to)
                }
            }
            .stroke(color.opacity(0.4), lineWidth: 1)
        }
    }
}

// MARK: - ViewModel

class FOMMViewModel: ObservableObject {
    @Published var selectedSourcePhoto: PhotosPickerItem? {
        didSet { loadSourceImage() }
    }
    @Published var selectedDrivingPhoto: PhotosPickerItem? {
        didSet { loadDrivingImage() }
    }
    @Published var sourceImage: UIImage?
    @Published var drivingImage: UIImage?
    @Published var resultImage: UIImage?

    @Published var sourceKeypoints: [CGPoint] = []
    @Published var drivingKeypoints: [CGPoint] = []

    @Published var isDetectingKeypoints = false
    @Published var isGenerating = false
    @Published var errorMessage: String?

    private func loadSourceImage() {
        guard let item = selectedSourcePhoto else { return }
        Task {
            if let data = try? await item.loadTransferable(type: Data.self),
               let image = UIImage(data: data) {
                await MainActor.run {
                    self.sourceImage = image
                    self.sourceKeypoints = []
                    self.resultImage = nil
                    self.errorMessage = nil
                }
            }
        }
    }

    private func loadDrivingImage() {
        guard let item = selectedDrivingPhoto else { return }
        Task {
            if let data = try? await item.loadTransferable(type: Data.self),
               let image = UIImage(data: data) {
                await MainActor.run {
                    self.drivingImage = image
                    self.drivingKeypoints = []
                    self.resultImage = nil
                    self.errorMessage = nil
                }
            }
        }
    }

    func detectKeypoints() {
        guard let source = sourceImage, let driving = drivingImage else { return }
        isDetectingKeypoints = true
        errorMessage = nil

        Task {
            do {
                let (srcKP, drvKP) = try await runKeypointDetection(source: source, driving: driving)
                await MainActor.run {
                    self.sourceKeypoints = srcKP
                    self.drivingKeypoints = drvKP
                    self.isDetectingKeypoints = false
                }
            } catch {
                await MainActor.run {
                    self.errorMessage = error.localizedDescription
                    self.isDetectingKeypoints = false
                }
            }
        }
    }

    // Detect keypoints using FOMM_KPDetector model
    // Input: image (1,3,256,256)
    // Output: keypoints (1,10,2) normalized coordinates + jacobians (1,10,2,2)
    private func runKeypointDetection(source: UIImage, driving: UIImage) async throws -> ([CGPoint], [CGPoint]) {
        guard let modelURL = Bundle.main.url(forResource: "FOMM_KPDetector", withExtension: "mlmodelc") else {
            throw FOMMError.modelNotFound(
                "FOMM_KPDetector.mlmodelc not found in bundle. " +
                "Please compile and add the FOMM_KPDetector.mlpackage to the project."
            )
        }

        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        let model = try MLModel(contentsOf: modelURL, configuration: config)

        // Detect source keypoints
        let sourceArray = try imageToMultiArray(source)
        let sourceInput = try MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(multiArray: sourceArray)
        ])
        let sourceOutput = try model.prediction(from: sourceInput)

        guard let sourceKPArray = sourceOutput.featureValue(for: "keypoints")?.multiArrayValue else {
            throw FOMMError.processingFailed("Failed to extract source keypoints")
        }
        let sourceKP = extractKeypoints(from: sourceKPArray)

        // Detect driving keypoints
        let drivingArray = try imageToMultiArray(driving)
        let drivingInput = try MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(multiArray: drivingArray)
        ])
        let drivingOutput = try model.prediction(from: drivingInput)

        guard let drivingKPArray = drivingOutput.featureValue(for: "keypoints")?.multiArrayValue else {
            throw FOMMError.processingFailed("Failed to extract driving keypoints")
        }
        let drivingKP = extractKeypoints(from: drivingKPArray)

        return (sourceKP, drivingKP)
    }

    // Extract 10 keypoints from (1,10,2) MLMultiArray
    private func extractKeypoints(from array: MLMultiArray) -> [CGPoint] {
        var points: [CGPoint] = []
        for i in 0..<10 {
            let x = CGFloat(array[[0, i, 0] as [NSNumber]].floatValue)
            let y = CGFloat(array[[0, i, 1] as [NSNumber]].floatValue)
            // Normalize from [-1, 1] to [0, 1]
            let normX = (x + 1.0) / 2.0
            let normY = (y + 1.0) / 2.0
            points.append(CGPoint(x: normX, y: normY))
        }
        return points
    }

    func generateReenactment() {
        guard sourceImage != nil else { return }
        isGenerating = true
        errorMessage = nil

        Task {
            do {
                let result = try await runGeneration()
                await MainActor.run {
                    self.resultImage = result
                    self.isGenerating = false
                }
            } catch {
                await MainActor.run {
                    self.errorMessage = error.localizedDescription
                    self.isGenerating = false
                }
            }
        }
    }

    // Generate reenacted face using FOMM_Generator
    // Input: source_image (1,3,256,256) + keypoint data
    // Output: prediction (1,3,256,256)
    private func runGeneration() async throws -> UIImage {
        guard let modelURL = Bundle.main.url(forResource: "FOMM_Generator", withExtension: "mlmodelc") else {
            throw FOMMError.modelNotFound(
                "FOMM_Generator.mlmodelc not found in bundle. " +
                "Please compile and add the FOMM_Generator.mlpackage to the project."
            )
        }

        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        let model = try MLModel(contentsOf: modelURL, configuration: config)

        guard let source = sourceImage else {
            throw FOMMError.processingFailed("Source image not available")
        }

        let sourceArray = try imageToMultiArray(source)

        // Prepare keypoint arrays
        let srcKPArray = try keypointsToMultiArray(sourceKeypoints)
        let drvKPArray = try keypointsToMultiArray(drivingKeypoints)

        // Prepare Jacobian arrays (1,10,2,2)
        let srcJacobians = try MLMultiArray(shape: [1, 10, 2, 2], dataType: .float32)
        let drvJacobians = try MLMultiArray(shape: [1, 10, 2, 2], dataType: .float32)
        // Initialize Jacobians as identity matrices
        for i in 0..<10 {
            srcJacobians[[0, i, 0, 0] as [NSNumber]] = 1.0
            srcJacobians[[0, i, 1, 1] as [NSNumber]] = 1.0
            drvJacobians[[0, i, 0, 0] as [NSNumber]] = 1.0
            drvJacobians[[0, i, 1, 1] as [NSNumber]] = 1.0
        }

        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
            "source_image": MLFeatureValue(multiArray: sourceArray),
            "source_keypoints": MLFeatureValue(multiArray: srcKPArray),
            "driving_keypoints": MLFeatureValue(multiArray: drvKPArray),
            "source_jacobians": MLFeatureValue(multiArray: srcJacobians),
            "driving_jacobians": MLFeatureValue(multiArray: drvJacobians)
        ])

        let output = try model.prediction(from: inputFeatures)

        guard let predictionArray = output.featureValue(for: "prediction")?.multiArrayValue else {
            throw FOMMError.processingFailed("Failed to extract prediction output")
        }

        guard let resultImage = imageFromMultiArray(predictionArray, width: 256, height: 256) else {
            throw FOMMError.processingFailed("Failed to convert prediction to image")
        }

        return resultImage
    }

    // Convert UIImage to (1,3,256,256) MLMultiArray
    private func imageToMultiArray(_ image: UIImage) throws -> MLMultiArray {
        let width = 256
        let height = 256
        guard let resized = image.resized(to: CGSize(width: width, height: height)),
              let cgImage = resized.cgImage else {
            throw FOMMError.processingFailed("Failed to resize image")
        }

        let array = try MLMultiArray(shape: [1, 3, 256, 256], dataType: .float32)
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        var pixelData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &pixelData, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw FOMMError.processingFailed("Failed to create CGContext")
        }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        for y in 0..<height {
            for x in 0..<width {
                let offset = (y * width + x) * bytesPerPixel
                array[[0, 0, y, x] as [NSNumber]] = NSNumber(value: Float(pixelData[offset]) / 255.0)
                array[[0, 1, y, x] as [NSNumber]] = NSNumber(value: Float(pixelData[offset + 1]) / 255.0)
                array[[0, 2, y, x] as [NSNumber]] = NSNumber(value: Float(pixelData[offset + 2]) / 255.0)
            }
        }

        return array
    }

    // Convert keypoints to (1,10,2) MLMultiArray
    private func keypointsToMultiArray(_ keypoints: [CGPoint]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, 10, 2], dataType: .float32)
        for i in 0..<min(keypoints.count, 10) {
            // Convert from [0,1] back to [-1,1]
            array[[0, i, 0] as [NSNumber]] = NSNumber(value: Float(keypoints[i].x) * 2.0 - 1.0)
            array[[0, i, 1] as [NSNumber]] = NSNumber(value: Float(keypoints[i].y) * 2.0 - 1.0)
        }
        return array
    }

    // Convert (1,3,256,256) MLMultiArray to UIImage
    private func imageFromMultiArray(_ array: MLMultiArray, width: Int, height: Int) -> UIImage? {
        var pixelData = [UInt8](repeating: 255, count: width * height * 4)

        for y in 0..<height {
            for x in 0..<width {
                let r = min(max(array[[0, 0, y, x] as [NSNumber]].floatValue, 0), 1)
                let g = min(max(array[[0, 1, y, x] as [NSNumber]].floatValue, 0), 1)
                let b = min(max(array[[0, 2, y, x] as [NSNumber]].floatValue, 0), 1)
                let offset = (y * width + x) * 4
                pixelData[offset] = UInt8(r * 255)
                pixelData[offset + 1] = UInt8(g * 255)
                pixelData[offset + 2] = UInt8(b * 255)
                pixelData[offset + 3] = 255
            }
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &pixelData, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ), let cgImage = context.makeImage() else { return nil }

        return UIImage(cgImage: cgImage)
    }
}

enum FOMMError: LocalizedError {
    case modelNotFound(String)
    case processingFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelNotFound(let msg): return msg
        case .processingFailed(let msg): return msg
        }
    }
}

// MARK: - UIImage Extension

extension UIImage {
    func resized(to targetSize: CGSize) -> UIImage? {
        let renderer = UIGraphicsImageRenderer(size: targetSize)
        return renderer.image { _ in
            self.draw(in: CGRect(origin: .zero, size: targetSize))
        }
    }
}

#Preview {
    ContentView()
}
