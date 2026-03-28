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

// MARK: - Classifier

class MobileNetClassifier: ObservableObject {
    @Published var predictions: [(label: String, confidence: Float)] = []
    @Published var errorMessage: String?

    private var vnModel: VNCoreMLModel?
    private var isProcessing = false

    init() {
        loadModel()
    }

    private func loadModel() {
        // PLACEHOLDER: Add MobileNetV3Small.mlpackage to the Xcode project.
        // The compiled .mlmodelc will be bundled automatically.
        // Download from the CoreML-Models repository and drag into Xcode.

        guard let modelURL = Bundle.main.url(forResource: "MobileNetV3Small", withExtension: "mlmodelc") else {
            DispatchQueue.main.async {
                self.errorMessage = "Model not found. Please add MobileNetV3Small.mlpackage to the Xcode project."
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

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right, options: [:])
        try? handler.perform([request])
    }

    private func processResults(multiArray: MLMultiArray) {
        let count = multiArray.count
        var scores = [Float](repeating: 0, count: count)
        for i in 0..<count {
            scores[i] = multiArray[i].floatValue
        }

        // Apply softmax for probabilities
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
    @StateObject private var classifier = MobileNetClassifier()

    var body: some View {
        ZStack {
            // Camera feed
            CameraPreview(session: camera.session)
                .ignoresSafeArea()

            VStack {
                Spacer()

                // Error message if model not loaded
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

                // Prediction overlay
                if !classifier.predictions.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("MobileNetV3 Small")
                            .font(.headline)
                            .foregroundColor(.white)

                        ForEach(Array(classifier.predictions.enumerated()), id: \.offset) { index, prediction in
                            HStack {
                                Text("\(index + 1).")
                                    .font(.caption)
                                    .foregroundColor(.white.opacity(0.7))
                                Text(prediction.label)
                                    .font(.system(.body, design: .monospaced))
                                    .foregroundColor(.white)
                                Spacer()
                                Text(String(format: "%.1f%%", prediction.confidence * 100))
                                    .font(.system(.body, design: .monospaced))
                                    .foregroundColor(.green)
                            }
                        }
                    }
                    .padding()
                    .background(.black.opacity(0.7), in: RoundedRectangle(cornerRadius: 16))
                    .padding()
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
}

#Preview {
    ContentView()
}
