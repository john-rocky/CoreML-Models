import SwiftUI
import UIKit
import AVFoundation
import CoreML
import Vision

// MARK: - COCO Class Labels

let cocoClassLabels: [Int: String] = [
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard", 37: "surfboard",
    38: "tennis racket", 39: "bottle", 40: "wine glass", 41: "cup", 42: "fork",
    43: "knife", 44: "spoon", 45: "bowl", 46: "banana", 47: "apple",
    48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog",
    53: "pizza", 54: "donut", 55: "cake", 56: "chair", 57: "couch",
    58: "potted plant", 59: "bed", 60: "dining table", 61: "toilet", 62: "tv",
    63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone",
    68: "microwave", 69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator",
    73: "book", 74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush"
]

// MARK: - Color Palette for Classes

struct ClassColors {
    static let palette: [Color] = [
        .red, .green, .blue, .orange, .purple,
        .pink, .yellow, .cyan, .mint, .indigo,
        .teal, .brown, Color(red: 1, green: 0.4, blue: 0.4),
        Color(red: 0.4, green: 1, blue: 0.4),
        Color(red: 0.4, green: 0.4, blue: 1),
        Color(red: 1, green: 0.8, blue: 0), Color(red: 0, green: 0.8, blue: 0.8),
        Color(red: 0.8, green: 0, blue: 0.8), Color(red: 0.6, green: 0.4, blue: 0.2),
        Color(red: 0.2, green: 0.6, blue: 0.4)
    ]

    static func color(for classIndex: Int) -> Color {
        palette[classIndex % palette.count]
    }
}

// MARK: - Detection Result

struct Detection: Identifiable {
    let id = UUID()
    let classIndex: Int
    let label: String
    let confidence: Float
    let boundingBox: CGRect // Normalized coordinates (0..1), Vision convention (origin bottom-left)
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

// MARK: - Object Detector

class ObjectDetector: ObservableObject {
    @Published var detections: [Detection] = []
    @Published var fps: Double = 0
    @Published var errorMessage: String?

    private var vnModel: VNCoreMLModel?
    private var isProcessing = false
    private var lastTimestamp: CFTimeInterval = 0
    private var frameCount: Int = 0
    private let fpsUpdateInterval: CFTimeInterval = 0.5

    private let confidenceThreshold: Float = 0.25

    init() {
        loadModel()
    }

    private func loadModel() {
        // PLACEHOLDER: Add YOLOv10N.mlpackage to the Xcode project.
        // The compiled .mlmodelc will be bundled automatically.
        // Convert using: python conversion_scripts/convert_yolov10.py
        // Then drag yolov10n.mlpackage into Xcode and rename to YOLOv10N.

        guard let modelURL = Bundle.main.url(forResource: "YOLOv10N", withExtension: "mlmodelc") else {
            DispatchQueue.main.async {
                self.errorMessage = "Model not found. Please add YOLOv10N.mlpackage to the Xcode project."
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

    func detect(sampleBuffer: CMSampleBuffer) {
        guard !isProcessing, let vnModel = vnModel else { return }
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

        let request = VNCoreMLRequest(model: vnModel) { [weak self] request, error in
            defer { self?.isProcessing = false }
            guard let self = self else { return }

            if let results = request.results as? [VNCoreMLFeatureValueObservation],
               let multiArray = results.first?.featureValue.multiArrayValue {
                self.processRawOutput(multiArray)
            } else {
                DispatchQueue.main.async {
                    self.detections = []
                }
            }
        }
        request.imageCropAndScaleOption = .scaleFill

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right, options: [:])
        try? handler.perform([request])
    }

    /// Parse raw YOLOv10 output [1, 300, 6] where each row is [x1, y1, x2, y2, confidence, class_id].
    private func processRawOutput(_ multiArray: MLMultiArray) {
        let numDetections = multiArray.shape[1].intValue  // 300
        let stride = multiArray.shape[2].intValue         // 6
        let ptr = multiArray.dataPointer.bindMemory(to: Float.self, capacity: numDetections * stride)

        var results: [Detection] = []

        for i in 0..<numDetections {
            let base = i * stride
            let confidence = ptr[base + 4]
            guard confidence >= confidenceThreshold else { continue }

            let x1 = CGFloat(ptr[base])     / 640.0
            let y1 = CGFloat(ptr[base + 1]) / 640.0
            let x2 = CGFloat(ptr[base + 2]) / 640.0
            let y2 = CGFloat(ptr[base + 3]) / 640.0
            let classId = Int(ptr[base + 5])

            let label = cocoClassLabels[classId] ?? "class_\(classId)"

            // Convert from top-left origin to Vision convention (bottom-left origin)
            let box = CGRect(x: x1, y: 1 - y2, width: x2 - x1, height: y2 - y1)

            results.append(Detection(
                classIndex: classId,
                label: label,
                confidence: confidence,
                boundingBox: box
            ))
        }

        DispatchQueue.main.async {
            self.detections = results
        }
    }
}

// MARK: - Bounding Box Overlay

struct BoundingBoxOverlay: View {
    let detections: [Detection]
    let geometrySize: CGSize

    var body: some View {
        ForEach(detections) { detection in
            let rect = convertBoundingBox(detection.boundingBox, in: geometrySize)
            let boxColor = ClassColors.color(for: detection.classIndex)

            ZStack(alignment: .topLeading) {
                // Bounding box rectangle
                Rectangle()
                    .stroke(boxColor, lineWidth: 2.5)
                    .frame(width: rect.width, height: rect.height)
                    .position(x: rect.midX, y: rect.midY)

                // Label background and text
                Text("\(detection.label) \(String(format: "%.0f%%", detection.confidence * 100))")
                    .font(.system(size: 11, weight: .semibold, design: .monospaced))
                    .foregroundColor(.white)
                    .padding(.horizontal, 4)
                    .padding(.vertical, 2)
                    .background(boxColor.opacity(0.85))
                    .cornerRadius(4)
                    .position(x: rect.minX + 40, y: rect.minY - 10)
            }
        }
    }

    /// Convert Vision normalized coordinates (origin bottom-left) to UIKit coordinates (origin top-left).
    private func convertBoundingBox(_ box: CGRect, in size: CGSize) -> CGRect {
        let x = box.origin.x * size.width
        let y = (1 - box.origin.y - box.height) * size.height
        let width = box.width * size.width
        let height = box.height * size.height
        return CGRect(x: x, y: y, width: width, height: height)
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

// MARK: - Detection Count Badge

struct DetectionCountBadge: View {
    let count: Int

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: "eye.fill")
                .font(.system(size: 11))
                .foregroundColor(.white)
            Text("\(count) object\(count == 1 ? "" : "s")")
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
    @StateObject private var detector = ObjectDetector()

    var body: some View {
        ZStack {
            // Camera feed
            CameraPreview(session: camera.session)
                .ignoresSafeArea()

            // Bounding box overlay
            GeometryReader { geometry in
                BoundingBoxOverlay(
                    detections: detector.detections,
                    geometrySize: geometry.size
                )
            }
            .ignoresSafeArea()

            VStack {
                // Top bar: FPS and detection count
                HStack {
                    FPSCounterView(fps: detector.fps)
                    Spacer()
                    DetectionCountBadge(count: detector.detections.count)
                }
                .padding(.horizontal, 16)
                .padding(.top, 8)

                Spacer()

                // Error message if model not loaded
                if let error = detector.errorMessage {
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

                // Detection list overlay at the bottom
                if !detector.detections.isEmpty {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("YOLOv10-N Detections")
                            .font(.headline)
                            .foregroundColor(.white)

                        let grouped = groupedDetections(detector.detections)
                        ForEach(Array(grouped.prefix(5).enumerated()), id: \.offset) { _, item in
                            HStack {
                                Circle()
                                    .fill(ClassColors.color(for: item.classIndex))
                                    .frame(width: 10, height: 10)
                                Text(item.label)
                                    .font(.system(.body, design: .monospaced))
                                    .foregroundColor(.white)
                                if item.count > 1 {
                                    Text("x\(item.count)")
                                        .font(.system(.caption, design: .monospaced))
                                        .foregroundColor(.white.opacity(0.7))
                                }
                                Spacer()
                                Text(String(format: "%.0f%%", item.maxConfidence * 100))
                                    .font(.system(.body, design: .monospaced))
                                    .foregroundColor(.green)
                            }
                        }
                    }
                    .padding()
                    .background(.black.opacity(0.7), in: RoundedRectangle(cornerRadius: 16))
                    .padding(.horizontal)
                    .padding(.bottom, 8)
                }
            }
        }
        .onAppear {
            camera.onFrame = { [weak detector] buffer in
                detector?.detect(sampleBuffer: buffer)
            }
            camera.configure()
        }
        .onDisappear {
            camera.stop()
        }
    }

    // Group detections by class for the summary panel
    private func groupedDetections(_ detections: [Detection]) -> [GroupedDetection] {
        var dict: [String: GroupedDetection] = [:]
        for d in detections {
            if var existing = dict[d.label] {
                existing.count += 1
                existing.maxConfidence = max(existing.maxConfidence, d.confidence)
                dict[d.label] = existing
            } else {
                dict[d.label] = GroupedDetection(
                    classIndex: d.classIndex,
                    label: d.label,
                    count: 1,
                    maxConfidence: d.confidence
                )
            }
        }
        return dict.values.sorted { $0.maxConfidence > $1.maxConfidence }
    }
}

// MARK: - Grouped Detection

struct GroupedDetection {
    let classIndex: Int
    let label: String
    var count: Int
    var maxConfidence: Float
}

// MARK: - Preview

#Preview {
    ContentView()
}
