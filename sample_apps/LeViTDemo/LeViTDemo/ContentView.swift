import SwiftUI
import UIKit
import CoreML
import Vision
import AVFoundation

// MARK: - Continuous Camera Classifier with History
// Uses LeViT_128S model (224x224 input, 1000-class ImageNet output)
// Output feature name: "var_1140"

struct ClassificationEntry: Identifiable {
    let id = UUID()
    let label: String
    let confidence: Float
    let timestamp: Date
}

struct ContentView: View {
    @StateObject private var classifier = CameraClassifier()

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                if let error = classifier.errorMessage {
                    HStack {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.yellow)
                        Text(error)
                            .font(.caption)
                    }
                    .padding(8)
                    .background(Color(.systemOrange).opacity(0.1))
                }

                // Camera preview area
                ZStack(alignment: .bottom) {
                    CameraPreviewView(session: classifier.captureSession)
                        .frame(height: 320)
                        .clipped()
                        .background(Color.black)

                    // Current prediction overlay
                    if let current = classifier.currentPrediction {
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(current.label)
                                    .font(.title3)
                                    .fontWeight(.bold)
                                    .foregroundColor(.white)
                                Text(String(format: "%.1f%% confidence", current.confidence * 100))
                                    .font(.caption)
                                    .foregroundColor(.white.opacity(0.8))
                            }
                            Spacer()
                            // FPS indicator
                            Text(String(format: "%.1f fps", classifier.fps))
                                .font(.caption2)
                                .foregroundColor(.white.opacity(0.6))
                                .padding(4)
                                .background(Color.black.opacity(0.3))
                                .cornerRadius(4)
                        }
                        .padding()
                        .background(
                            LinearGradient(
                                colors: [.clear, .black.opacity(0.7)],
                                startPoint: .top,
                                endPoint: .bottom
                            )
                        )
                    }
                }

                // Controls bar
                HStack {
                    Button {
                        classifier.toggleCamera()
                    } label: {
                        Label(
                            classifier.isRunning ? "Pause" : "Resume",
                            systemImage: classifier.isRunning ? "pause.circle.fill" : "play.circle.fill"
                        )
                        .font(.subheadline)
                    }
                    .buttonStyle(.bordered)

                    Spacer()

                    Text("\(classifier.history.count) classifications")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    Spacer()

                    Button {
                        classifier.clearHistory()
                    } label: {
                        Label("Clear", systemImage: "trash")
                            .font(.subheadline)
                    }
                    .buttonStyle(.bordered)
                    .tint(.red)
                }
                .padding(.horizontal)
                .padding(.vertical, 8)

                // Classification history log
                List {
                    ForEach(classifier.history) { entry in
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(entry.label)
                                    .font(.subheadline)
                                    .fontWeight(.medium)
                                Text(entry.timestamp, style: .time)
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                            }

                            Spacer()

                            // Confidence bar
                            VStack(alignment: .trailing, spacing: 2) {
                                Text(String(format: "%.1f%%", entry.confidence * 100))
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                ProgressView(value: entry.confidence)
                                    .frame(width: 60)
                                    .tint(confidenceColor(entry.confidence))
                            }
                        }
                    }
                }
                .listStyle(.plain)
            }
            .navigationTitle("LeViT Live")
            .navigationBarTitleDisplayMode(.inline)
            .onAppear {
                classifier.startSession()
            }
            .onDisappear {
                classifier.stopSession()
            }
        }
    }

    private func confidenceColor(_ value: Float) -> Color {
        if value > 0.7 { return .green }
        if value > 0.4 { return .orange }
        return .red
    }
}

// MARK: - Camera Preview UIViewRepresentable
struct CameraPreviewView: UIViewRepresentable {
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

// MARK: - Camera Classifier ViewModel
class CameraClassifier: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var currentPrediction: ClassificationEntry?
    @Published var history: [ClassificationEntry] = []
    @Published var isRunning = false
    @Published var errorMessage: String?
    @Published var fps: Double = 0

    let captureSession = AVCaptureSession()
    private var vnModel: VNCoreMLModel?
    private var lastClassificationTime: Date = .distantPast
    private var frameCount = 0
    private var fpsTimer: Date = Date()
    private let classificationInterval: TimeInterval = 0.5 // Classify every 0.5 seconds
    private let maxHistoryCount = 100

    override init() {
        super.init()
        loadModel()
    }

    private func loadModel() {
        // PLACEHOLDER: Add LeViT_128S.mlpackage to the Xcode project.
        // The compiled model class will be generated automatically by Xcode.
        // Download from the converted_models directory and drag into the project navigator.
        do {
            guard let modelURL = Bundle.main.url(forResource: "LeViT_128S", withExtension: "mlmodelc") else {
                DispatchQueue.main.async {
                    self.errorMessage = "Model not found. Add LeViT_128S.mlpackage to the project."
                }
                return
            }
            let mlModel = try MLModel(contentsOf: modelURL)
            vnModel = try VNCoreMLModel(for: mlModel)
        } catch {
            DispatchQueue.main.async {
                self.errorMessage = "Failed to load model: \(error.localizedDescription)"
            }
        }
    }

    func startSession() {
        guard !captureSession.isRunning else { return }

        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            setupCamera()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                if granted {
                    DispatchQueue.main.async {
                        self?.setupCamera()
                    }
                } else {
                    DispatchQueue.main.async {
                        self?.errorMessage = "Camera access denied. Enable in Settings."
                    }
                }
            }
        default:
            DispatchQueue.main.async {
                self.errorMessage = "Camera access denied. Enable in Settings."
            }
        }
    }

    private func setupCamera() {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .medium

        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: camera) else {
            DispatchQueue.main.async {
                self.errorMessage = "Cannot access camera."
            }
            captureSession.commitConfiguration()
            return
        }

        if captureSession.canAddInput(input) {
            captureSession.addInput(input)
        }

        let output = AVCaptureVideoDataOutput()
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "com.coreml-models.levitdemo.camera"))
        output.alwaysDiscardsLateVideoFrames = true

        if captureSession.canAddOutput(output) {
            captureSession.addOutput(output)
        }

        captureSession.commitConfiguration()

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.captureSession.startRunning()
            DispatchQueue.main.async {
                self?.isRunning = true
            }
        }
    }

    func stopSession() {
        if captureSession.isRunning {
            captureSession.stopRunning()
        }
        DispatchQueue.main.async {
            self.isRunning = false
        }
    }

    func toggleCamera() {
        if isRunning {
            stopSession()
        } else {
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                self?.captureSession.startRunning()
                DispatchQueue.main.async {
                    self?.isRunning = true
                }
            }
        }
    }

    func clearHistory() {
        history.removeAll()
    }

    // MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // FPS calculation
        frameCount += 1
        let now = Date()
        let elapsed = now.timeIntervalSince(fpsTimer)
        if elapsed >= 1.0 {
            let currentFPS = Double(frameCount) / elapsed
            DispatchQueue.main.async {
                self.fps = currentFPS
            }
            frameCount = 0
            fpsTimer = now
        }

        // Throttle classification
        guard now.timeIntervalSince(lastClassificationTime) >= classificationInterval else { return }
        lastClassificationTime = now

        guard let vnModel = vnModel else { return }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        let request = VNCoreMLRequest(model: vnModel) { [weak self] request, error in
            self?.processResults(request: request, error: error)
        }
        request.imageCropAndScaleOption = .centerCrop

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right, options: [:])
        try? handler.perform([request])
    }

    private func processResults(request: VNRequest, error: Error?) {
        if let results = request.results as? [VNCoreMLFeatureValueObservation],
           let multiArray = results.first?.featureValue.multiArrayValue {
            let count = multiArray.count
            var scores = [Float](repeating: 0, count: count)
            for i in 0..<count {
                scores[i] = multiArray[i].floatValue
            }

            let softmaxScores = softmax(scores)
            let topResults = ImageNetLabels.topK(scores: softmaxScores, k: 1)

            if let top = topResults.first {
                let entry = ClassificationEntry(
                    label: top.label,
                    confidence: top.score,
                    timestamp: Date()
                )
                DispatchQueue.main.async {
                    self.currentPrediction = entry

                    // Only add to history if different from last entry or confidence changed significantly
                    if let lastEntry = self.history.first {
                        if lastEntry.label != entry.label || abs(lastEntry.confidence - entry.confidence) > 0.1 {
                            self.history.insert(entry, at: 0)
                        }
                    } else {
                        self.history.insert(entry, at: 0)
                    }

                    // Trim history
                    if self.history.count > self.maxHistoryCount {
                        self.history = Array(self.history.prefix(self.maxHistoryCount))
                    }
                }
            }
        } else if let results = request.results as? [VNClassificationObservation],
                  let top = results.first {
            let entry = ClassificationEntry(
                label: top.identifier,
                confidence: top.confidence,
                timestamp: Date()
            )
            DispatchQueue.main.async {
                self.currentPrediction = entry
                if let lastEntry = self.history.first {
                    if lastEntry.label != entry.label || abs(lastEntry.confidence - entry.confidence) > 0.1 {
                        self.history.insert(entry, at: 0)
                    }
                } else {
                    self.history.insert(entry, at: 0)
                }
                if self.history.count > self.maxHistoryCount {
                    self.history = Array(self.history.prefix(self.maxHistoryCount))
                }
            }
        }
    }

    private func softmax(_ input: [Float]) -> [Float] {
        let maxVal = input.max() ?? 0
        let expValues = input.map { exp($0 - maxVal) }
        let sumExp = expValues.reduce(0, +)
        return expValues.map { $0 / sumExp }
    }
}

#Preview {
    ContentView()
}
