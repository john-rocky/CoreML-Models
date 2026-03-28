import SwiftUI
import UIKit
import AVFoundation
import CoreML
import Vision

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
        session.sessionPreset = .medium

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
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]

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

    func makeCoordinator() -> Coordinator { Coordinator() }
    class Coordinator { var previewLayer: AVCaptureVideoPreviewLayer? }
}

// MARK: - FPS Counter

class FPSCounter: ObservableObject {
    @Published var fps: Double = 0
    @Published var inferenceMs: Double = 0
    @Published var peakFps: Double = 0
    @Published var minInferenceMs: Double = Double.infinity

    private var frameTimestamps: [CFAbsoluteTime] = []
    private var inferenceTimes: [Double] = []
    private let windowSize = 30

    func recordFrame(inferenceTime: Double) {
        let now = CFAbsoluteTimeGetCurrent()
        frameTimestamps.append(now)
        inferenceTimes.append(inferenceTime)

        // Keep only recent frames
        while frameTimestamps.count > windowSize {
            frameTimestamps.removeFirst()
        }
        while inferenceTimes.count > windowSize {
            inferenceTimes.removeFirst()
        }

        // Calculate FPS from frame timestamps
        if frameTimestamps.count >= 2 {
            let duration = frameTimestamps.last! - frameTimestamps.first!
            let currentFps = duration > 0 ? Double(frameTimestamps.count - 1) / duration : 0
            DispatchQueue.main.async { [weak self] in
                self?.fps = currentFps
                if currentFps > (self?.peakFps ?? 0) {
                    self?.peakFps = currentFps
                }
            }
        }

        // Average inference time
        let avgInference = inferenceTimes.reduce(0, +) / Double(inferenceTimes.count)
        DispatchQueue.main.async { [weak self] in
            self?.inferenceMs = avgInference
            if inferenceTime < (self?.minInferenceMs ?? Double.infinity) {
                self?.minInferenceMs = inferenceTime
            }
        }
    }
}

// MARK: - MobileOne Classifier

class MobileOneClassifier: ObservableObject {
    @Published var predictions: [(label: String, confidence: Float)] = []
    @Published var errorMessage: String?

    private var vnModel: VNCoreMLModel?
    private var isProcessing = false

    let fpsCounter = FPSCounter()

    init() {
        loadModel()
    }

    private func loadModel() {
        // PLACEHOLDER: Add MobileOne_S0.mlpackage to the Xcode project.
        // The compiled .mlmodelc will be bundled automatically.
        // Download from the CoreML-Models repository and drag into Xcode.

        guard let modelURL = Bundle.main.url(forResource: "MobileOne_S0", withExtension: "mlmodelc") else {
            DispatchQueue.main.async {
                self.errorMessage = "Model not found. Please add MobileOne_S0.mlpackage to the Xcode project."
            }
            return
        }

        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            let mlModel = try MLModel(contentsOf: modelURL, configuration: config)
            vnModel = try VNCoreMLModel(for: mlModel)
        } catch {
            DispatchQueue.main.async {
                self.errorMessage = "Failed to load model: \(error.localizedDescription)"
            }
        }
    }

    func classify(sampleBuffer: CMSampleBuffer) {
        guard !isProcessing, let vnModel = vnModel else { return }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        isProcessing = true

        let request = VNCoreMLRequest(model: vnModel) { [weak self] request, error in
            defer { self?.isProcessing = false }

            if let results = request.results as? [VNCoreMLFeatureValueObservation],
               let multiArray = results.first?.featureValue.multiArrayValue {
                self?.processResults(multiArray: multiArray)
            } else if let results = request.results as? [VNClassificationObservation] {
                let top3 = results.prefix(3).map { (label: $0.identifier, confidence: $0.confidence) }
                DispatchQueue.main.async {
                    self?.predictions = top3
                }
            }
        }
        request.imageCropAndScaleOption = .centerCrop

        let startTime = CFAbsoluteTimeGetCurrent()
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right, options: [:])
        try? handler.perform([request])
        let elapsed = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0

        fpsCounter.recordFrame(inferenceTime: elapsed)
    }

    private func processResults(multiArray: MLMultiArray) {
        let count = multiArray.count
        var scores = [Float](repeating: 0, count: count)
        for i in 0..<count {
            scores[i] = multiArray[i].floatValue
        }

        // Apply softmax
        let maxScore = scores.max() ?? 0
        let expScores = scores.map { exp($0 - maxScore) }
        let sumExp = expScores.reduce(0, +)
        let probs = expScores.map { $0 / sumExp }

        let top3 = ImageNetLabels.topK(scores: probs, k: 3)
        DispatchQueue.main.async {
            self.predictions = top3.map { (label: $0.label, confidence: $0.score) }
        }
    }
}

// MARK: - Content View

struct ContentView: View {
    @StateObject private var camera = CameraManager()
    @StateObject private var classifier = MobileOneClassifier()

    var body: some View {
        ZStack {
            // Camera feed
            CameraPreview(session: camera.session)
                .ignoresSafeArea()

            VStack(spacing: 0) {
                // FPS and timing overlay at top
                HStack(spacing: 16) {
                    // FPS badge
                    VStack(spacing: 2) {
                        Text(String(format: "%.0f", classifier.fpsCounter.fps))
                            .font(.system(size: 36, weight: .bold, design: .monospaced))
                            .foregroundColor(fpsColor)
                        Text("FPS")
                            .font(.caption2)
                            .foregroundColor(.white.opacity(0.8))
                    }
                    .frame(width: 80)

                    Divider()
                        .frame(height: 50)
                        .background(Color.white.opacity(0.3))

                    // Inference time
                    VStack(spacing: 2) {
                        Text(String(format: "%.1f", classifier.fpsCounter.inferenceMs))
                            .font(.system(size: 28, weight: .semibold, design: .monospaced))
                            .foregroundColor(.cyan)
                        Text("ms avg")
                            .font(.caption2)
                            .foregroundColor(.white.opacity(0.8))
                    }

                    Divider()
                        .frame(height: 50)
                        .background(Color.white.opacity(0.3))

                    // Best inference time
                    VStack(spacing: 2) {
                        Text(classifier.fpsCounter.minInferenceMs < Double.infinity
                             ? String(format: "%.1f", classifier.fpsCounter.minInferenceMs)
                             : "--")
                            .font(.system(size: 28, weight: .semibold, design: .monospaced))
                            .foregroundColor(.green)
                        Text("ms best")
                            .font(.caption2)
                            .foregroundColor(.white.opacity(0.8))
                    }
                }
                .padding(.vertical, 12)
                .padding(.horizontal, 20)
                .background(.black.opacity(0.7), in: RoundedRectangle(cornerRadius: 16))
                .padding(.top, 60)

                // Model name badge
                Text("MobileOne S0")
                    .font(.system(.caption, design: .monospaced))
                    .fontWeight(.medium)
                    .foregroundColor(.white)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 4)
                    .background(.blue.opacity(0.7), in: Capsule())
                    .padding(.top, 8)

                Spacer()

                // Error message
                if let error = classifier.errorMessage {
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

                // Predictions overlay
                if !classifier.predictions.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        ForEach(Array(classifier.predictions.enumerated()), id: \.offset) { index, prediction in
                            HStack {
                                // Rank indicator
                                Circle()
                                    .fill(rankColor(index))
                                    .frame(width: 8, height: 8)

                                Text(prediction.label)
                                    .font(.system(.body, design: .monospaced))
                                    .foregroundColor(.white)
                                    .lineLimit(1)

                                Spacer()

                                // Confidence bar
                                ZStack(alignment: .leading) {
                                    RoundedRectangle(cornerRadius: 3)
                                        .fill(.white.opacity(0.2))
                                        .frame(width: 60, height: 6)

                                    RoundedRectangle(cornerRadius: 3)
                                        .fill(rankColor(index))
                                        .frame(width: 60 * CGFloat(prediction.confidence), height: 6)
                                }

                                Text(String(format: "%.1f%%", prediction.confidence * 100))
                                    .font(.system(.caption, design: .monospaced))
                                    .foregroundColor(.white.opacity(0.8))
                                    .frame(width: 55, alignment: .trailing)
                            }
                        }
                    }
                    .padding()
                    .background(.black.opacity(0.7), in: RoundedRectangle(cornerRadius: 16))
                    .padding()
                    .padding(.bottom, 20)
                }
            }
        }
        .onAppear {
            camera.onFrame = { [weak classifier] buffer in
                classifier?.classify(sampleBuffer: buffer)
            }
            camera.configure()
        }
        .onDisappear {
            camera.stop()
        }
    }

    private var fpsColor: Color {
        let fps = classifier.fpsCounter.fps
        if fps >= 30 { return .green }
        if fps >= 15 { return .yellow }
        return .red
    }

    private func rankColor(_ index: Int) -> Color {
        switch index {
        case 0: return .green
        case 1: return .cyan
        case 2: return .orange
        default: return .gray
        }
    }
}

#Preview {
    ContentView()
}
