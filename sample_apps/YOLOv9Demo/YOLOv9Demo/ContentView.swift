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

// MARK: - Detection Result

struct Detection: Identifiable {
    let id = UUID()
    let label: String
    let confidence: Float
    let boundingBox: CGRect
}

// MARK: - Object Detector (NMS-based: YOLOv9 / YOLO11)

class ObjectDetector: ObservableObject {
    @Published var detections: [Detection] = []
    @Published var errorMessage: String?

    private var vnModel: VNCoreMLModel?
    private var isProcessing = false

    // COCO class labels
    private let labels = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

    init() {
        loadModel()
    }

    private func loadModel() {
        // NMS-based models: yolov9s, yolo11s (exported with nms=True)
        // Try multiple model names
        let modelNames = ["yolov9s", "yolo11s", "YOLOv9s", "YOLO11s"]
        var modelURL: URL?

        for name in modelNames {
            if let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") {
                modelURL = url
                break
            }
        }

        guard let url = modelURL else {
            DispatchQueue.main.async {
                self.errorMessage = "Model not found. Add yolov9s.mlpackage or yolo11s.mlpackage to the Xcode project."
            }
            return
        }

        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            let mlModel = try MLModel(contentsOf: url, configuration: config)
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

        let request = VNCoreMLRequest(model: vnModel) { [weak self] request, _ in
            defer { self?.isProcessing = false }
            guard let self = self else { return }

            // NMS model outputs VNRecognizedObjectObservation
            if let results = request.results as? [VNRecognizedObjectObservation] {
                let dets = results.compactMap { obs -> Detection? in
                    guard let topLabel = obs.labels.first else { return nil }
                    return Detection(
                        label: topLabel.identifier,
                        confidence: topLabel.confidence,
                        boundingBox: obs.boundingBox
                    )
                }
                DispatchQueue.main.async {
                    self.detections = dets
                }
                return
            }

            // Fallback: raw feature values with confidence/coordinates
            if let results = request.results as? [VNCoreMLFeatureValueObservation] {
                var dets: [Detection] = []
                var confidenceArray: MLMultiArray?
                var coordinatesArray: MLMultiArray?

                for obs in results {
                    let name = obs.featureName.lowercased()
                    if name.contains("confidence") {
                        confidenceArray = obs.featureValue.multiArrayValue
                    } else if name.contains("coordinate") {
                        coordinatesArray = obs.featureValue.multiArrayValue
                    }
                }

                if let conf = confidenceArray, let coords = coordinatesArray {
                    let numDetections = conf.shape[0].intValue
                    let numClasses = conf.shape[1].intValue

                    for i in 0..<numDetections {
                        var maxConf: Float = 0
                        var maxClass = 0
                        for c in 0..<numClasses {
                            let val = conf[[i, c] as [NSNumber]].floatValue
                            if val > maxConf {
                                maxConf = val
                                maxClass = c
                            }
                        }
                        guard maxConf > 0.25 else { continue }

                        let x = coords[[i, 0] as [NSNumber]].floatValue
                        let y = coords[[i, 1] as [NSNumber]].floatValue
                        let w = coords[[i, 2] as [NSNumber]].floatValue
                        let h = coords[[i, 3] as [NSNumber]].floatValue

                        let label = maxClass < self.labels.count ? self.labels[maxClass] : "class_\(maxClass)"
                        dets.append(Detection(
                            label: label,
                            confidence: maxConf,
                            boundingBox: CGRect(x: CGFloat(x - w/2), y: CGFloat(y - h/2), width: CGFloat(w), height: CGFloat(h))
                        ))
                    }
                }

                DispatchQueue.main.async {
                    self.detections = dets
                }
            }
        }
        request.imageCropAndScaleOption = .scaleFill

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right, options: [:])
        try? handler.perform([request])
    }
}

// MARK: - Bounding Box Overlay

struct BoundingBoxOverlay: View {
    let detections: [Detection]
    let viewSize: CGSize

    private let colors: [Color] = [.red, .green, .blue, .orange, .purple, .yellow, .pink, .cyan]

    var body: some View {
        ForEach(detections) { det in
            let rect = convertBoundingBox(det.boundingBox)
            let color = colors[abs(det.label.hashValue) % colors.count]

            Rectangle()
                .stroke(color, lineWidth: 2)
                .frame(width: rect.width, height: rect.height)
                .position(x: rect.midX, y: rect.midY)

            Text("\(det.label) \(String(format: "%.0f%%", det.confidence * 100))")
                .font(.caption2)
                .fontWeight(.bold)
                .foregroundColor(.white)
                .padding(.horizontal, 4)
                .padding(.vertical, 2)
                .background(color.opacity(0.8))
                .cornerRadius(4)
                .position(x: rect.midX, y: rect.minY - 10)
        }
    }

    private func convertBoundingBox(_ box: CGRect) -> CGRect {
        // Vision bounding box: origin at bottom-left, normalized [0,1]
        let x = box.origin.x * viewSize.width
        let y = (1 - box.origin.y - box.height) * viewSize.height
        let w = box.width * viewSize.width
        let h = box.height * viewSize.height
        return CGRect(x: x, y: y, width: w, height: h)
    }
}

// MARK: - Content View

struct ContentView: View {
    @StateObject private var camera = CameraManager()
    @StateObject private var detector = ObjectDetector()

    var body: some View {
        GeometryReader { geo in
            ZStack {
                CameraPreview(session: camera.session)
                    .ignoresSafeArea()

                BoundingBoxOverlay(detections: detector.detections, viewSize: geo.size)

                VStack {
                    Spacer()

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

                    // Detection count
                    HStack {
                        Text("YOLO (NMS)")
                            .font(.headline)
                        Spacer()
                        Text("\(detector.detections.count) objects")
                            .font(.subheadline)
                    }
                    .foregroundColor(.white)
                    .padding()
                    .background(.black.opacity(0.7), in: RoundedRectangle(cornerRadius: 12))
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
}

#Preview {
    ContentView()
}
