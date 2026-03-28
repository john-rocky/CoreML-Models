import SwiftUI
import UIKit
import AVFoundation
import PhotosUI
import CoreML
import Vision

// MARK: - Benchmark Classifier

class FastViTClassifier: ObservableObject {
    @Published var predictions: [(label: String, confidence: Float)] = []
    @Published var inferenceTimeMs: Double = 0
    @Published var averageTimeMs: Double = 0
    @Published var errorMessage: String?
    @Published var isProcessing = false

    private var vnModel: VNCoreMLModel?
    private var recentTimes: [Double] = []
    private let maxRecentTimes = 20

    init() {
        loadModel()
    }

    private func loadModel() {
        // PLACEHOLDER: Add FastViT_T8.mlpackage to the Xcode project.
        // The compiled .mlmodelc will be bundled automatically.
        // Download from the CoreML-Models repository and drag into Xcode.

        guard let modelURL = Bundle.main.url(forResource: "FastViT_T8", withExtension: "mlmodelc") else {
            DispatchQueue.main.async {
                self.errorMessage = "Model not found. Please add FastViT_T8.mlpackage to the Xcode project."
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

    func classify(image: UIImage) {
        guard let vnModel = vnModel else { return }
        guard let cgImage = image.cgImage else { return }

        DispatchQueue.main.async { self.isProcessing = true }

        let request = VNCoreMLRequest(model: vnModel) { [weak self] request, error in
            if let results = request.results as? [VNCoreMLFeatureValueObservation],
               let multiArray = results.first?.featureValue.multiArrayValue {
                self?.processResults(multiArray: multiArray)
            } else if let results = request.results as? [VNClassificationObservation] {
                let top5 = results.prefix(5).map { (label: $0.identifier, confidence: $0.confidence) }
                DispatchQueue.main.async {
                    self?.predictions = top5
                    self?.isProcessing = false
                }
            }
        }
        request.imageCropAndScaleOption = .centerCrop

        DispatchQueue.global(qos: .userInitiated).async {
            let startTime = CFAbsoluteTimeGetCurrent()
            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            try? handler.perform([request])
            let elapsed = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0

            DispatchQueue.main.async { [weak self] in
                self?.inferenceTimeMs = elapsed
                self?.recentTimes.append(elapsed)
                if let count = self?.recentTimes.count, count > (self?.maxRecentTimes ?? 20) {
                    self?.recentTimes.removeFirst()
                }
                self?.averageTimeMs = (self?.recentTimes.reduce(0, +) ?? 0) / Double(self?.recentTimes.count ?? 1)
                self?.isProcessing = false
            }
        }
    }

    func classifyBuffer(sampleBuffer: CMSampleBuffer) {
        guard let vnModel = vnModel else { return }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        let request = VNCoreMLRequest(model: vnModel) { [weak self] request, error in
            if let results = request.results as? [VNCoreMLFeatureValueObservation],
               let multiArray = results.first?.featureValue.multiArrayValue {
                self?.processResults(multiArray: multiArray)
            } else if let results = request.results as? [VNClassificationObservation] {
                let top5 = results.prefix(5).map { (label: $0.identifier, confidence: $0.confidence) }
                DispatchQueue.main.async {
                    self?.predictions = top5
                }
            }
        }
        request.imageCropAndScaleOption = .centerCrop

        let startTime = CFAbsoluteTimeGetCurrent()
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right, options: [:])
        try? handler.perform([request])
        let elapsed = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0

        DispatchQueue.main.async { [weak self] in
            self?.inferenceTimeMs = elapsed
            self?.recentTimes.append(elapsed)
            if let count = self?.recentTimes.count, count > (self?.maxRecentTimes ?? 20) {
                self?.recentTimes.removeFirst()
            }
            self?.averageTimeMs = (self?.recentTimes.reduce(0, +) ?? 0) / Double(self?.recentTimes.count ?? 1)
        }
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

        let top5 = ImageNetLabels.topK(scores: probs, k: 5)
        DispatchQueue.main.async {
            self.predictions = top5.map { (label: $0.label, confidence: $0.score) }
        }
    }

    func runBenchmark(image: UIImage, iterations: Int = 10) {
        guard let vnModel = vnModel, let cgImage = image.cgImage else { return }

        DispatchQueue.main.async { self.isProcessing = true }

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            var times: [Double] = []

            for _ in 0..<iterations {
                let request = VNCoreMLRequest(model: vnModel) { _, _ in }
                request.imageCropAndScaleOption = .centerCrop

                let startTime = CFAbsoluteTimeGetCurrent()
                let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
                try? handler.perform([request])
                let elapsed = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0
                times.append(elapsed)
            }

            let avg = times.reduce(0, +) / Double(times.count)
            let minTime = times.min() ?? 0
            let maxTime = times.max() ?? 0

            DispatchQueue.main.async {
                self?.recentTimes = times
                self?.averageTimeMs = avg
                self?.inferenceTimeMs = times.last ?? 0
                self?.isProcessing = false
            }

            print("Benchmark: avg=\(String(format: "%.2f", avg))ms, min=\(String(format: "%.2f", minTime))ms, max=\(String(format: "%.2f", maxTime))ms")
        }
    }
}

// MARK: - Camera Manager

class CameraManager: NSObject, ObservableObject {
    let session = AVCaptureSession()
    var onFrame: ((CMSampleBuffer) -> Void)?
    private var isProcessing = false

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
        guard !isProcessing else { return }
        isProcessing = true
        onFrame?(sampleBuffer)
        isProcessing = false
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

// MARK: - Content View

struct ContentView: View {
    @StateObject private var classifier = FastViTClassifier()
    @StateObject private var camera = CameraManager()
    @State private var selectedItem: PhotosPickerItem?
    @State private var selectedImage: UIImage?
    @State private var mode: InputMode = .camera

    enum InputMode: String, CaseIterable {
        case camera = "Camera"
        case photo = "Photo"
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Mode picker
                Picker("Input", selection: $mode) {
                    ForEach(InputMode.allCases, id: \.self) { m in
                        Text(m.rawValue).tag(m)
                    }
                }
                .pickerStyle(.segmented)
                .padding()

                // Timing display - prominently shown
                VStack(spacing: 4) {
                    HStack(spacing: 20) {
                        VStack {
                            Text("Last")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                            Text(String(format: "%.1f ms", classifier.inferenceTimeMs))
                                .font(.system(.title, design: .monospaced))
                                .fontWeight(.bold)
                                .foregroundColor(.blue)
                        }

                        Divider().frame(height: 40)

                        VStack {
                            Text("Average")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                            Text(String(format: "%.1f ms", classifier.averageTimeMs))
                                .font(.system(.title, design: .monospaced))
                                .fontWeight(.bold)
                                .foregroundColor(.green)
                        }

                        Divider().frame(height: 40)

                        VStack {
                            Text("FPS")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                            Text(classifier.averageTimeMs > 0 ? String(format: "%.0f", 1000.0 / classifier.averageTimeMs) : "--")
                                .font(.system(.title, design: .monospaced))
                                .fontWeight(.bold)
                                .foregroundColor(.orange)
                        }
                    }
                }
                .padding(.vertical, 8)
                .frame(maxWidth: .infinity)
                .background(Color(.systemGroupedBackground))

                // Content area
                ZStack {
                    if mode == .camera {
                        CameraPreview(session: camera.session)
                    } else {
                        Color(.systemGroupedBackground)
                        if let image = selectedImage {
                            Image(uiImage: image)
                                .resizable()
                                .scaledToFit()
                                .clipShape(RoundedRectangle(cornerRadius: 12))
                                .padding()
                        } else {
                            VStack(spacing: 16) {
                                Image(systemName: "photo.on.rectangle.angled")
                                    .font(.system(size: 50))
                                    .foregroundColor(.secondary)
                                Text("Select a photo to benchmark")
                                    .foregroundColor(.secondary)
                            }
                        }
                    }

                    if classifier.isProcessing {
                        ProgressView("Running benchmark...")
                            .padding()
                            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
                    }
                }
                .frame(maxHeight: .infinity)

                // Error
                if let error = classifier.errorMessage {
                    HStack {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.yellow)
                        Text(error)
                            .font(.caption)
                    }
                    .padding(8)
                    .background(Color.red.opacity(0.1))
                    .cornerRadius(8)
                    .padding(.horizontal)
                }

                // Predictions
                if !classifier.predictions.isEmpty {
                    VStack(alignment: .leading, spacing: 6) {
                        ForEach(Array(classifier.predictions.enumerated()), id: \.offset) { index, pred in
                            HStack {
                                Text("\(index + 1). \(pred.label)")
                                    .font(.system(.caption, design: .monospaced))
                                    .fontWeight(index == 0 ? .bold : .regular)
                                Spacer()
                                Text(String(format: "%.1f%%", pred.confidence * 100))
                                    .font(.system(.caption, design: .monospaced))
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                    .padding()
                    .background(Color(.systemBackground))
                }

                // Bottom controls
                if mode == .photo {
                    HStack {
                        PhotosPicker(selection: $selectedItem, matching: .images) {
                            Label("Choose Photo", systemImage: "photo.fill")
                                .font(.headline)
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.accentColor)
                                .foregroundColor(.white)
                                .cornerRadius(12)
                        }

                        if selectedImage != nil {
                            Button {
                                if let img = selectedImage {
                                    classifier.runBenchmark(image: img, iterations: 10)
                                }
                            } label: {
                                Label("Bench x10", systemImage: "speedometer")
                                    .font(.headline)
                                    .frame(maxWidth: .infinity)
                                    .padding()
                                    .background(Color.orange)
                                    .foregroundColor(.white)
                                    .cornerRadius(12)
                            }
                        }
                    }
                    .padding()
                }
            }
            .navigationTitle("FastViT-T8 Benchmark")
            .navigationBarTitleDisplayMode(.inline)
        }
        .onChange(of: selectedItem) { newItem in
            Task {
                if let data = try? await newItem?.loadTransferable(type: Data.self),
                   let uiImage = UIImage(data: data) {
                    selectedImage = uiImage
                    classifier.classify(image: uiImage)
                }
            }
        }
        .onChange(of: mode) { newMode in
            if newMode == .camera {
                camera.onFrame = { [weak classifier] buffer in
                    classifier?.classifyBuffer(sampleBuffer: buffer)
                }
                camera.configure()
            } else {
                camera.stop()
            }
        }
        .onAppear {
            if mode == .camera {
                camera.onFrame = { [weak classifier] buffer in
                    classifier?.classifyBuffer(sampleBuffer: buffer)
                }
                camera.configure()
            }
        }
        .onDisappear {
            camera.stop()
        }
    }
}

#Preview {
    ContentView()
}
