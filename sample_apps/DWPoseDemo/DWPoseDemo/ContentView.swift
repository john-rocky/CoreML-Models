import SwiftUI
import UIKit
import AVFoundation
import CoreML
import Accelerate

// MARK: - COCO Keypoint Definitions

let keypointNames: [String] = [
    "nose",           // 0
    "left_eye",       // 1
    "right_eye",      // 2
    "left_ear",       // 3
    "right_ear",      // 4
    "left_shoulder",  // 5
    "right_shoulder", // 6
    "left_elbow",     // 7
    "right_elbow",    // 8
    "left_wrist",     // 9
    "right_wrist",    // 10
    "left_hip",       // 11
    "right_hip",      // 12
    "left_knee",      // 13
    "right_knee",     // 14
    "left_ankle",     // 15
    "right_ankle",    // 16
]

let skeletonConnections: [(Int, Int)] = [
    (0, 1), (0, 2), (1, 3), (2, 4),  // Head
    (5, 6),                            // Shoulders
    (5, 7), (7, 9),                    // Left arm
    (6, 8), (8, 10),                   // Right arm
    (5, 11), (6, 12),                  // Torso
    (11, 12),                          // Hips
    (11, 13), (13, 15),                // Left leg
    (12, 14), (14, 16),               // Right leg
]

// Left-side keypoint indices (blue)
let leftIndices: Set<Int> = [1, 3, 5, 7, 9, 11, 13, 15]
// Right-side keypoint indices (red)
let rightIndices: Set<Int> = [2, 4, 6, 8, 10, 12, 14, 16]
// Center keypoint indices (green)
let centerIndices: Set<Int> = [0]

// MARK: - Keypoint

struct Keypoint {
    let x: CGFloat
    let y: CGFloat
    let confidence: Float
}

// MARK: - Connection Color Helper

func connectionColor(for connection: (Int, Int)) -> Color {
    let (a, b) = connection
    // Shoulder-to-shoulder and hip-to-hip are center connections
    if (a == 5 && b == 6) || (a == 11 && b == 12) {
        return .green
    }
    // Torso connections use the side of the limb endpoint
    if leftIndices.contains(a) || leftIndices.contains(b) {
        return .blue
    }
    if rightIndices.contains(a) || rightIndices.contains(b) {
        return .red
    }
    return .green
}

func keypointColor(for index: Int) -> Color {
    if leftIndices.contains(index) { return .blue }
    if rightIndices.contains(index) { return .red }
    return .green
}

// MARK: - Camera Manager

class CameraManager: NSObject, ObservableObject {
    let session = AVCaptureSession()
    var onFrame: ((CMSampleBuffer) -> Void)?

    private let sessionQueue = DispatchQueue(label: "camera.session")

    func configure() {
        sessionQueue.async { [weak self] in
            self?.setupSession()
        }
    }

    private func setupSession() {
        session.beginConfiguration()
        session.sessionPreset = .high

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device) else {
            session.commitConfiguration()
            return
        }

        if session.canAddInput(input) {
            session.addInput(input)
        }

        let output = AVCaptureVideoDataOutput()
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "camera.frame"))
        output.alwaysDiscardsLateVideoFrames = true

        if session.canAddOutput(output) {
            session.addOutput(output)
        }

        session.commitConfiguration()
        session.startRunning()
    }

    func stop() {
        sessionQueue.async { [weak self] in
            self?.session.stopRunning()
        }
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        onFrame?(sampleBuffer)
    }
}

// MARK: - Camera Preview

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: .zero)
        let previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)
        context.coordinator.previewLayer = previewLayer
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {
        context.coordinator.previewLayer?.frame = uiView.bounds
    }

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    class Coordinator {
        var previewLayer: AVCaptureVideoPreviewLayer?
    }
}

// MARK: - Pose Estimator

class PoseEstimator: ObservableObject {
    @Published var keypoints: [Keypoint] = []
    @Published var fps: Double = 0
    @Published var detectedKeypointCount: Int = 0
    @Published var errorMessage: String?

    private var mlModel: MLModel?
    private var isProcessing = false
    private var lastTimestamp: CFTimeInterval = 0
    private var frameCount: Int = 0
    private let fpsUpdateInterval: CFTimeInterval = 0.5

    private let confidenceThreshold: Float = 0.3
    private let smoothingFactor: CGFloat = 0.6
    private var previousKeypoints: [Keypoint] = []

    // Model input dimensions
    private let inputWidth = 192
    private let inputHeight = 256

    // SimCC output dimensions (typically 2x input + some margin)
    // For RTMPose with SimCC: x_simcc has shape (1, 17, 384), y_simcc has shape (1, 17, 512)
    private let simccXSize = 384  // inputWidth * 2
    private let simccYSize = 512  // inputHeight * 2

    init() {
        loadModel()
    }

    private func loadModel() {
        // PLACEHOLDER: Add DWPose.mlpackage to the Xcode project.
        // The compiled .mlmodelc will be bundled automatically.
        // Convert using: python conversion_scripts/convert_dwpose.py
        // Then drag DWPose.mlpackage into Xcode.

        guard let modelURL = Bundle.main.url(forResource: "DWPose", withExtension: "mlmodelc") else {
            DispatchQueue.main.async {
                self.errorMessage = "Model not found. Please add DWPose.mlpackage to the Xcode project."
            }
            return
        }

        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            mlModel = try MLModel(contentsOf: modelURL, configuration: config)
        } catch {
            DispatchQueue.main.async {
                self.errorMessage = "Failed to load model: \(error.localizedDescription)"
            }
        }
    }

    func estimatePose(sampleBuffer: CMSampleBuffer) {
        guard !isProcessing, let model = mlModel else { return }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        isProcessing = true

        // Update FPS counter
        let now = CACurrentMediaTime()
        frameCount += 1
        if now - lastTimestamp >= fpsUpdateInterval {
            let currentFPS = Double(frameCount) / (now - lastTimestamp)
            frameCount = 0
            lastTimestamp = now
            DispatchQueue.main.async {
                self.fps = currentFPS
            }
        }

        // Preprocess: resize pixel buffer and create MLMultiArray input
        guard let resizedBuffer = resizePixelBuffer(pixelBuffer, width: inputWidth, height: inputHeight) else {
            isProcessing = false
            return
        }

        do {
            let input = try createModelInput(from: resizedBuffer)
            let output = try model.prediction(from: input)
            let keypoints = postProcessSimCC(output: output)

            // Apply temporal smoothing
            let smoothed = applySmoothingFilter(keypoints)

            let detected = smoothed.filter { $0.confidence >= confidenceThreshold }.count
            DispatchQueue.main.async {
                self.keypoints = smoothed
                self.detectedKeypointCount = detected
                self.previousKeypoints = smoothed
            }
        } catch {
            // Silently skip frames with errors during inference
        }

        isProcessing = false
    }

    private func resizePixelBuffer(_ pixelBuffer: CVPixelBuffer, width: Int, height: Int) -> CVPixelBuffer? {
        var resizedBuffer: CVPixelBuffer?
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ] as CFDictionary

        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width, height,
            kCVPixelFormatType_32BGRA,
            attrs,
            &resizedBuffer
        )
        guard status == kCVReturnSuccess, let outputBuffer = resizedBuffer else { return nil }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        CVPixelBufferLockBaseAddress(outputBuffer, [])

        guard let srcData = CVPixelBufferGetBaseAddress(pixelBuffer),
              let dstData = CVPixelBufferGetBaseAddress(outputBuffer) else {
            CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
            CVPixelBufferUnlockBaseAddress(outputBuffer, [])
            return nil
        }

        var srcBuffer = vImage_Buffer(
            data: srcData,
            height: vImagePixelCount(CVPixelBufferGetHeight(pixelBuffer)),
            width: vImagePixelCount(CVPixelBufferGetWidth(pixelBuffer)),
            rowBytes: CVPixelBufferGetBytesPerRow(pixelBuffer)
        )
        var dstBuffer = vImage_Buffer(
            data: dstData,
            height: vImagePixelCount(height),
            width: vImagePixelCount(width),
            rowBytes: CVPixelBufferGetBytesPerRow(outputBuffer)
        )

        vImageScale_ARGB8888(&srcBuffer, &dstBuffer, nil, vImage_Flags(kvImageHighQualityResampling))

        CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
        CVPixelBufferUnlockBaseAddress(outputBuffer, [])

        return outputBuffer
    }

    private func createModelInput(from pixelBuffer: CVPixelBuffer) throws -> MLDictionaryFeatureProvider {
        // Create MLMultiArray with shape (1, 3, 256, 192)
        let shape: [NSNumber] = [1, 3, NSNumber(value: inputHeight), NSNumber(value: inputWidth)]
        let inputArray = try MLMultiArray(shape: shape, dataType: .float32)

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw NSError(domain: "DWPose", code: -1, userInfo: [NSLocalizedDescriptionKey: "Cannot access pixel buffer"])
        }

        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let ptr = baseAddress.assumingMemoryBound(to: UInt8.self)

        // ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        let mean: [Float] = [0.485, 0.456, 0.406]
        let std: [Float] = [0.229, 0.224, 0.225]

        let channelStride = inputHeight * inputWidth
        for y in 0..<inputHeight {
            for x in 0..<inputWidth {
                let pixelOffset = y * bytesPerRow + x * 4
                // BGRA format
                let b = Float(ptr[pixelOffset]) / 255.0
                let g = Float(ptr[pixelOffset + 1]) / 255.0
                let r = Float(ptr[pixelOffset + 2]) / 255.0

                let idx = y * inputWidth + x
                inputArray[idx] = NSNumber(value: (r - mean[0]) / std[0])                      // R channel
                inputArray[channelStride + idx] = NSNumber(value: (g - mean[1]) / std[1])       // G channel
                inputArray[2 * channelStride + idx] = NSNumber(value: (b - mean[2]) / std[2])   // B channel
            }
        }

        let featureValue = MLFeatureValue(multiArray: inputArray)
        return try MLDictionaryFeatureProvider(dictionary: ["image": featureValue])
    }

    private func postProcessSimCC(output: MLFeatureProvider) -> [Keypoint] {
        // RTMPose SimCC outputs: simcc_x (1, 17, simccXSize) and simcc_y (1, 17, simccYSize)
        // Each row contains logits for discretized coordinate bins along X or Y axis
        // The argmax of each row gives the predicted coordinate

        guard let simccX = output.featureValue(for: "simcc_x")?.multiArrayValue,
              let simccY = output.featureValue(for: "simcc_y")?.multiArrayValue else {
            // Fallback: try alternative output names
            return postProcessHeatmap(output: output)
        }

        let numKeypoints = 17
        var keypoints: [Keypoint] = []

        let xDim = simccX.shape.last?.intValue ?? simccXSize
        let yDim = simccY.shape.last?.intValue ?? simccYSize

        for k in 0..<numKeypoints {
            // Find argmax and max value for x coordinate
            var maxXVal: Float = -Float.greatestFiniteMagnitude
            var maxXIdx: Int = 0
            for i in 0..<xDim {
                let val = simccX[[0, k, i] as [NSNumber]].floatValue
                if val > maxXVal {
                    maxXVal = val
                    maxXIdx = i
                }
            }

            // Find argmax and max value for y coordinate
            var maxYVal: Float = -Float.greatestFiniteMagnitude
            var maxYIdx: Int = 0
            for i in 0..<yDim {
                let val = simccY[[0, k, i] as [NSNumber]].floatValue
                if val > maxYVal {
                    maxYVal = val
                    maxYIdx = i
                }
            }

            // Convert discretized coordinates back to normalized [0, 1]
            let normX = CGFloat(maxXIdx) / CGFloat(xDim)
            let normY = CGFloat(maxYIdx) / CGFloat(yDim)

            // Confidence is the average of softmax peaks
            let confidence = (maxXVal + maxYVal) / 2.0

            keypoints.append(Keypoint(x: normX, y: normY, confidence: confidence))
        }

        return keypoints
    }

    private func postProcessHeatmap(output: MLFeatureProvider) -> [Keypoint] {
        // Fallback heatmap-based post-processing
        // Some models output standard heatmaps instead of SimCC
        guard let featureNames = output.featureNames.first,
              let heatmaps = output.featureValue(for: featureNames)?.multiArrayValue else {
            return Array(repeating: Keypoint(x: 0, y: 0, confidence: 0), count: 17)
        }

        let numKeypoints = 17
        let heatmapH = heatmaps.shape[2].intValue
        let heatmapW = heatmaps.shape[3].intValue
        var keypoints: [Keypoint] = []

        for k in 0..<numKeypoints {
            var maxVal: Float = -Float.greatestFiniteMagnitude
            var maxRow = 0
            var maxCol = 0

            for row in 0..<heatmapH {
                for col in 0..<heatmapW {
                    let val = heatmaps[[0, k, row, col] as [NSNumber]].floatValue
                    if val > maxVal {
                        maxVal = val
                        maxRow = row
                        maxCol = col
                    }
                }
            }

            let normX = CGFloat(maxCol) / CGFloat(heatmapW)
            let normY = CGFloat(maxRow) / CGFloat(heatmapH)

            keypoints.append(Keypoint(x: normX, y: normY, confidence: maxVal))
        }

        return keypoints
    }

    private func applySmoothingFilter(_ current: [Keypoint]) -> [Keypoint] {
        guard previousKeypoints.count == current.count else { return current }

        return zip(current, previousKeypoints).map { (cur, prev) in
            // Only smooth if both frames have sufficient confidence
            if cur.confidence >= confidenceThreshold && prev.confidence >= confidenceThreshold {
                let smoothX = cur.x * (1.0 - smoothingFactor) + prev.x * smoothingFactor
                let smoothY = cur.y * (1.0 - smoothingFactor) + prev.y * smoothingFactor
                return Keypoint(x: smoothX, y: smoothY, confidence: cur.confidence)
            }
            return cur
        }
    }
}

// MARK: - Skeleton Overlay

struct SkeletonOverlay: View {
    let keypoints: [Keypoint]
    let geometrySize: CGSize
    let confidenceThreshold: Float

    var body: some View {
        Canvas { context, size in
            // Draw skeleton connections
            for connection in skeletonConnections {
                let (startIdx, endIdx) = connection
                guard startIdx < keypoints.count, endIdx < keypoints.count else { continue }

                let startKp = keypoints[startIdx]
                let endKp = keypoints[endIdx]

                guard startKp.confidence >= confidenceThreshold,
                      endKp.confidence >= confidenceThreshold else { continue }

                let startPoint = CGPoint(
                    x: startKp.x * size.width,
                    y: startKp.y * size.height
                )
                let endPoint = CGPoint(
                    x: endKp.x * size.width,
                    y: endKp.y * size.height
                )

                var path = Path()
                path.move(to: startPoint)
                path.addLine(to: endPoint)

                let color = connectionColor(for: connection)
                context.stroke(path, with: .color(color), lineWidth: 3.0)
            }

            // Draw keypoint dots
            for (index, kp) in keypoints.enumerated() {
                guard kp.confidence >= confidenceThreshold else { continue }

                let point = CGPoint(
                    x: kp.x * size.width,
                    y: kp.y * size.height
                )

                let dotSize: CGFloat = 8.0
                let rect = CGRect(
                    x: point.x - dotSize / 2,
                    y: point.y - dotSize / 2,
                    width: dotSize,
                    height: dotSize
                )

                let color = keypointColor(for: index)

                // White border
                let borderRect = CGRect(
                    x: point.x - (dotSize + 2) / 2,
                    y: point.y - (dotSize + 2) / 2,
                    width: dotSize + 2,
                    height: dotSize + 2
                )
                context.fill(Path(ellipseIn: borderRect), with: .color(.white))
                context.fill(Path(ellipseIn: rect), with: .color(color))
            }
        }
    }
}

// MARK: - FPS Counter View

struct FPSCounterView: View {
    let fps: Double

    var body: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(fps > 20 ? Color.green : (fps > 10 ? Color.yellow : Color.red))
                .frame(width: 8, height: 8)
            Text(String(format: "%.1f FPS", fps))
                .font(.system(size: 13, weight: .bold, design: .monospaced))
                .foregroundColor(.white)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(Color.black.opacity(0.6))
        .cornerRadius(8)
    }
}

// MARK: - Keypoint Count Badge

struct KeypointCountBadge: View {
    let count: Int

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: "figure.stand")
                .font(.system(size: 11))
                .foregroundColor(.white)
            Text("\(count)/17 keypoints")
                .font(.system(size: 13, weight: .bold, design: .monospaced))
                .foregroundColor(.white)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(Color.black.opacity(0.6))
        .cornerRadius(8)
    }
}

// MARK: - Content View

struct ContentView: View {
    @StateObject private var camera = CameraManager()
    @StateObject private var estimator = PoseEstimator()

    var body: some View {
        ZStack {
            // Camera feed
            CameraPreview(session: camera.session)
                .ignoresSafeArea()

            // Skeleton overlay
            GeometryReader { geometry in
                SkeletonOverlay(
                    keypoints: estimator.keypoints,
                    geometrySize: geometry.size,
                    confidenceThreshold: 0.3
                )
            }
            .ignoresSafeArea()

            VStack {
                // Top bar: FPS and keypoint count
                HStack {
                    FPSCounterView(fps: estimator.fps)
                    Spacer()
                    KeypointCountBadge(count: estimator.detectedKeypointCount)
                }
                .padding(.horizontal, 16)
                .padding(.top, 8)

                Spacer()

                // Error message if model not loaded
                if let error = estimator.errorMessage {
                    VStack(spacing: 8) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.largeTitle)
                            .foregroundColor(.yellow)
                        Text(error)
                            .font(.caption)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)
                    }
                    .padding()
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
                    .padding()
                }

                // Legend at the bottom
                HStack(spacing: 16) {
                    LegendItem(color: .blue, label: "Left")
                    LegendItem(color: .red, label: "Right")
                    LegendItem(color: .green, label: "Center")
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(Color.black.opacity(0.6))
                .cornerRadius(12)
                .padding(.bottom, 8)
            }
        }
        .onAppear {
            camera.onFrame = { [weak estimator] buffer in
                estimator?.estimatePose(sampleBuffer: buffer)
            }
            camera.configure()
        }
        .onDisappear {
            camera.stop()
        }
    }
}

// MARK: - Legend Item

struct LegendItem: View {
    let color: Color
    let label: String

    var body: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(color)
                .frame(width: 10, height: 10)
            Text(label)
                .font(.system(size: 12, weight: .medium))
                .foregroundColor(.white)
        }
    }
}

// MARK: - Preview

#Preview {
    ContentView()
}
