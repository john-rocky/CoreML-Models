import SwiftUI
import UIKit
import CoreML
import Vision
import PhotosUI

// MARK: - 3DMM Parameter Categories

/// Decomposed 3D Morphable Model parameters from the model output
struct Face3DMMParams {
    // 12 pose parameters (rotation, translation, scale)
    var pose: [Float] = []        // indices 0-11
    // 40 shape parameters (identity basis coefficients)
    var shape: [Float] = []       // indices 12-51
    // 10 expression parameters
    var expression: [Float] = []  // indices 52-61

    /// Euler angles extracted from the pose parameters (approximated)
    var pitch: Float { pose.count >= 3 ? pose[0] * 180.0 / .pi : 0 }
    var yaw: Float { pose.count >= 3 ? pose[1] * 180.0 / .pi : 0 }
    var roll: Float { pose.count >= 3 ? pose[2] * 180.0 / .pi : 0 }

    /// Expression labels for display
    static let expressionLabels = [
        "Mouth Open", "Smile", "Brow Raise", "Brow Furrow",
        "Eye Close", "Lip Stretch", "Lip Press", "Jaw Drop",
        "Cheek Puff", "Nose Wrinkle"
    ]
}

// MARK: - Face 3D Processor

/// Processes face images through the 3DDFA_V2 CoreML model
class Face3DProcessor: ObservableObject {
    @Published var inputImage: UIImage?
    @Published var faceCrop: UIImage?
    @Published var params: Face3DMMParams?
    @Published var isProcessing = false
    @Published var errorMessage: String?
    @Published var faceLandmarks: [CGPoint] = []

    private var model: MLModel?
    private let inputSize = 120

    init() {
        loadModel()
    }

    private func loadModel() {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all

            guard let modelURL = Bundle.main.url(forResource: "3DDFA_V2", withExtension: "mlmodelc") else {
                errorMessage = "Model not found. Please add 3DDFA_V2.mlmodelc to the project bundle."
                return
            }
            model = try MLModel(contentsOf: modelURL, configuration: config)
        } catch {
            errorMessage = "Failed to load model: \(error.localizedDescription)"
        }
    }

    /// Detect face using Vision and return bounding box
    private func detectFace(in image: UIImage) async throws -> (CGRect, [CGPoint])? {
        guard let cgImage = image.cgImage else { return nil }

        return try await withCheckedThrowingContinuation { continuation in
            let request = VNDetectFaceLandmarksRequest { request, error in
                if let error = error {
                    continuation.resume(throwing: error)
                    return
                }
                guard let face = (request.results as? [VNFaceObservation])?.first else {
                    continuation.resume(returning: nil)
                    return
                }

                // Extract landmark points for overlay
                var landmarks: [CGPoint] = []
                if let allPoints = face.landmarks?.allPoints {
                    let imageWidth = CGFloat(cgImage.width)
                    let imageHeight = CGFloat(cgImage.height)
                    for point in allPoints.normalizedPoints {
                        let x = face.boundingBox.origin.x * imageWidth + point.x * face.boundingBox.width * imageWidth
                        let y = (1.0 - face.boundingBox.origin.y - face.boundingBox.height) * imageHeight + (1.0 - point.y) * face.boundingBox.height * imageHeight
                        landmarks.append(CGPoint(x: x / imageWidth, y: y / imageHeight))
                    }
                }
                continuation.resume(returning: (face.boundingBox, landmarks))
            }

            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            do {
                try handler.perform([request])
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }

    /// Crop face to 120x120 for model input
    private func cropFace(from image: UIImage, boundingBox: CGRect) -> UIImage? {
        guard let cgImage = image.cgImage else { return nil }

        let imageWidth = CGFloat(cgImage.width)
        let imageHeight = CGFloat(cgImage.height)

        let x = boundingBox.origin.x * imageWidth
        let y = (1.0 - boundingBox.origin.y - boundingBox.height) * imageHeight
        let w = boundingBox.width * imageWidth
        let h = boundingBox.height * imageHeight

        // Square crop with padding
        let side = max(w, h) * 1.3
        let centerX = x + w / 2
        let centerY = y + h / 2
        let cropRect = CGRect(
            x: max(0, centerX - side / 2),
            y: max(0, centerY - side / 2),
            width: min(imageWidth, side),
            height: min(imageHeight, side)
        )

        guard let croppedCGImage = cgImage.cropping(to: cropRect) else { return nil }

        let targetSize = CGSize(width: inputSize, height: inputSize)
        let renderer = UIGraphicsImageRenderer(size: targetSize)
        return renderer.image { _ in
            UIImage(cgImage: croppedCGImage).draw(in: CGRect(origin: .zero, size: targetSize))
        }
    }

    /// Convert UIImage to CHW float array normalized to [0, 1]
    private func imageToFloatArray(_ image: UIImage) -> [Float]? {
        guard let cgImage = image.cgImage else { return nil }

        let size = inputSize
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

    /// Run 3D face reconstruction
    func processImage(_ image: UIImage) async {
        guard let model = model else {
            await MainActor.run { errorMessage = "Model is not loaded." }
            return
        }

        await MainActor.run {
            inputImage = image
            faceCrop = nil
            params = nil
            faceLandmarks = []
            isProcessing = true
            errorMessage = nil
        }

        do {
            // Detect face
            guard let (faceBox, landmarks) = try await detectFace(in: image) else {
                await MainActor.run {
                    errorMessage = "No face detected in the image."
                    isProcessing = false
                }
                return
            }

            await MainActor.run { faceLandmarks = landmarks }

            // Crop face
            guard let cropped = cropFace(from: image, boundingBox: faceBox) else {
                await MainActor.run {
                    errorMessage = "Failed to crop face region."
                    isProcessing = false
                }
                return
            }

            await MainActor.run { faceCrop = cropped }

            // Preprocess
            guard let inputData = imageToFloatArray(cropped) else {
                await MainActor.run {
                    errorMessage = "Failed to preprocess face image."
                    isProcessing = false
                }
                return
            }

            let inputArray = try MLMultiArray(shape: [1, 3, 120, 120] as [NSNumber], dataType: .float32)
            let ptr = inputArray.dataPointer.bindMemory(to: Float.self, capacity: inputData.count)
            for i in 0..<inputData.count {
                ptr[i] = inputData[i]
            }

            let features = try MLDictionaryFeatureProvider(dictionary: ["face_image": MLFeatureValue(multiArray: inputArray)])
            let output = try model.prediction(from: features)

            guard let outputArray = output.featureValue(for: "params_3dmm")?.multiArrayValue else {
                await MainActor.run {
                    errorMessage = "Unexpected model output format."
                    isProcessing = false
                }
                return
            }

            // Parse the 62 parameters
            let outPtr = outputArray.dataPointer.bindMemory(to: Float.self, capacity: 62)
            var allParams = [Float](repeating: 0, count: 62)
            for i in 0..<62 {
                allParams[i] = outPtr[i]
            }

            var result = Face3DMMParams()
            result.pose = Array(allParams[0..<12])
            result.shape = Array(allParams[12..<52])
            result.expression = Array(allParams[52..<62])

            await MainActor.run {
                params = result
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

// MARK: - Gauge View

/// A circular gauge view for displaying a parameter value
struct GaugeView: View {
    let label: String
    let value: Float
    let range: ClosedRange<Float>
    let color: Color

    private var normalizedValue: Double {
        let clamped = max(range.lowerBound, min(range.upperBound, value))
        return Double((clamped - range.lowerBound) / (range.upperBound - range.lowerBound))
    }

    var body: some View {
        VStack(spacing: 4) {
            ZStack {
                Circle()
                    .trim(from: 0, to: 0.75)
                    .stroke(Color(.systemGray4), lineWidth: 4)
                    .rotationEffect(.degrees(135))

                Circle()
                    .trim(from: 0, to: min(0.75, normalizedValue * 0.75))
                    .stroke(color, lineWidth: 4)
                    .rotationEffect(.degrees(135))

                Text(String(format: "%.1f", value))
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
            }
            .frame(width: 50, height: 50)

            Text(label)
                .font(.system(size: 8))
                .foregroundColor(.secondary)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
        }
    }
}

// MARK: - Face Overlay View

/// Draws landmark points on top of the face image
struct FaceLandmarkOverlay: View {
    let image: UIImage
    let landmarks: [CGPoint]

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(width: geometry.size.width, height: geometry.size.height)

                // Draw landmarks
                ForEach(0..<landmarks.count, id: \.self) { i in
                    Circle()
                        .fill(Color.green)
                        .frame(width: 4, height: 4)
                        .position(
                            x: landmarks[i].x * geometry.size.width,
                            y: landmarks[i].y * geometry.size.height
                        )
                }
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

// MARK: - Camera Picker

struct CameraPicker: UIViewControllerRepresentable {
    @Binding var image: UIImage?

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.cameraDevice = .front
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: CameraPicker

        init(_ parent: CameraPicker) {
            self.parent = parent
        }

        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
            picker.dismiss(animated: true)
            if let image = info[.originalImage] as? UIImage {
                DispatchQueue.main.async {
                    self.parent.image = image
                }
            }
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            picker.dismiss(animated: true)
        }
    }
}

// MARK: - Content View

struct ContentView: View {
    @StateObject private var processor = Face3DProcessor()
    @State private var showImagePicker = false
    @State private var showCamera = false
    @State private var pickedImage: UIImage?

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

                    // Input buttons
                    HStack(spacing: 12) {
                        Button {
                            showCamera = true
                        } label: {
                            Label("Camera", systemImage: "camera.fill")
                                .font(.headline)
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.blue)
                                .foregroundColor(.white)
                                .cornerRadius(12)
                        }

                        Button {
                            showImagePicker = true
                        } label: {
                            Label("Photos", systemImage: "photo.on.rectangle")
                                .font(.headline)
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.green)
                                .foregroundColor(.white)
                                .cornerRadius(12)
                        }
                    }
                    .padding(.horizontal)

                    // Processing indicator
                    if processor.isProcessing {
                        ProgressView("Analyzing face...")
                            .padding()
                    }

                    // Face image with landmarks
                    if let image = processor.inputImage {
                        VStack(spacing: 8) {
                            Text("Detected Face with Landmarks")
                                .font(.headline)
                            FaceLandmarkOverlay(image: image, landmarks: processor.faceLandmarks)
                                .frame(height: 250)
                                .cornerRadius(12)
                                .padding(.horizontal)
                        }
                    }

                    // Cropped face
                    if let crop = processor.faceCrop {
                        VStack(spacing: 8) {
                            Text("Cropped Face (120x120)")
                                .font(.caption.bold())
                                .foregroundColor(.secondary)
                            Image(uiImage: crop)
                                .resizable()
                                .interpolation(.none)
                                .scaledToFit()
                                .frame(width: 120, height: 120)
                                .cornerRadius(8)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 8)
                                        .stroke(Color.orange, lineWidth: 2)
                                )
                        }
                    }

                    // 3DMM Parameters
                    if let params = processor.params {
                        parametersSection(params)
                    }

                    Spacer(minLength: 40)
                }
                .padding(.vertical)
            }
            .navigationTitle("3D Face Reconstruction")
            .sheet(isPresented: $showImagePicker) {
                ImagePicker(image: $pickedImage)
            }
            .sheet(isPresented: $showCamera) {
                CameraPicker(image: $pickedImage)
            }
            .onChange(of: pickedImage) { newValue in
                guard let image = newValue else { return }
                Task {
                    await processor.processImage(image)
                }
            }
        }
    }

    // MARK: - Subviews

    private var headerSection: some View {
        VStack(spacing: 8) {
            Image(systemName: "cube.transparent")
                .font(.system(size: 50))
                .foregroundColor(.orange)
            Text("3D Face Reconstruction")
                .font(.title2.bold())
            Text("Extract 3DMM parameters: pose, shape, and expression")
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

    private func parametersSection(_ params: Face3DMMParams) -> some View {
        VStack(spacing: 16) {
            // Pose (rotation angles)
            VStack(alignment: .leading, spacing: 8) {
                Text("Head Pose (Rotation)")
                    .font(.headline)

                HStack(spacing: 16) {
                    GaugeView(label: "Pitch", value: params.pitch, range: -90...90, color: .red)
                    GaugeView(label: "Yaw", value: params.yaw, range: -90...90, color: .green)
                    GaugeView(label: "Roll", value: params.roll, range: -90...90, color: .blue)
                }
                .frame(maxWidth: .infinity)

                // Pose parameter sliders
                ForEach(0..<min(params.pose.count, 12), id: \.self) { i in
                    HStack {
                        Text("P\(i)")
                            .font(.caption2)
                            .frame(width: 24, alignment: .leading)
                        ProgressView(value: normalize(params.pose[i], range: -3...3))
                            .tint(.red)
                        Text(String(format: "%.3f", params.pose[i]))
                            .font(.system(size: 9, design: .monospaced))
                            .frame(width: 50, alignment: .trailing)
                    }
                }
            }
            .padding()
            .background(Color(.systemGray6))
            .cornerRadius(12)

            // Shape parameters
            VStack(alignment: .leading, spacing: 8) {
                Text("Shape Parameters (40)")
                    .font(.headline)

                LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 4), count: 5), spacing: 8) {
                    ForEach(0..<min(params.shape.count, 40), id: \.self) { i in
                        VStack(spacing: 2) {
                            Text("S\(i)")
                                .font(.system(size: 7))
                                .foregroundColor(.secondary)
                            ProgressView(value: normalize(params.shape[i], range: -3...3))
                                .tint(.purple)
                                .scaleEffect(y: 2)
                        }
                    }
                }
            }
            .padding()
            .background(Color(.systemGray6))
            .cornerRadius(12)

            // Expression parameters
            VStack(alignment: .leading, spacing: 8) {
                Text("Expression Parameters (10)")
                    .font(.headline)

                ForEach(0..<min(params.expression.count, 10), id: \.self) { i in
                    HStack {
                        Text(Face3DMMParams.expressionLabels[i])
                            .font(.caption)
                            .frame(width: 90, alignment: .leading)
                        ProgressView(value: normalize(params.expression[i], range: -2...2))
                            .tint(expressionColor(for: i))
                        Text(String(format: "%.2f", params.expression[i]))
                            .font(.system(size: 10, design: .monospaced))
                            .frame(width: 44, alignment: .trailing)
                    }
                }
            }
            .padding()
            .background(Color(.systemGray6))
            .cornerRadius(12)
        }
        .padding(.horizontal)
    }

    // MARK: - Helpers

    private func normalize(_ value: Float, range: ClosedRange<Float>) -> Double {
        let clamped = max(range.lowerBound, min(range.upperBound, value))
        return Double((clamped - range.lowerBound) / (range.upperBound - range.lowerBound))
    }

    private func expressionColor(for index: Int) -> Color {
        let colors: [Color] = [.red, .orange, .yellow, .green, .blue, .purple, .pink, .cyan, .mint, .teal]
        return colors[index % colors.count]
    }
}

#Preview {
    ContentView()
}
