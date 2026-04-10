import SwiftUI
import PhotosUI
import CoreML
import Vision

/// Object detection with real-time camera + photo picker.
/// Matches YOLO26Demo / YOLOv9Demo UX: live camera feed with bounding box overlays,
/// photo mode with annotated results, FPS/latency display.
struct ImageDetectionDemoView: View {
    let model: ModelEntry

    enum Mode: String, CaseIterable { case camera = "Camera", photo = "Photo" }

    @State private var mode: Mode = .camera
    @State private var inputImage: UIImage?
    @State private var annotatedImage: UIImage?
    @State private var liveDetections: [BoundingBoxOverlay.DetectionBox] = []
    @State private var isProcessing = false
    @State private var status = ""
    @State private var processingTime: Double?
    @State private var fps: Double = 0
    @State private var item: PhotosPickerItem?
    @State private var confidenceThreshold: Float = 0.25
    @State private var vnModel: VNCoreMLModel?
    @State private var mlModel: MLModel?
    @State private var isModelLoaded = false

    private var labels: [String] { model.configStringArray("labels") ?? Self.cocoLabels }
    private var inputSize: Int { model.configInt("input_size") ?? 640 }

    var body: some View {
        VStack(spacing: 0) {
            // Mode picker
            Picker("Mode", selection: $mode) {
                ForEach(Mode.allCases, id: \.self) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented).padding(.horizontal).padding(.top, 4)

            // Main content
            ZStack {
                switch mode {
                case .camera:
                    cameraView
                case .photo:
                    photoView
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            // Status bar
            HStack {
                if mode == .camera {
                    Text(String(format: "%.1f FPS", fps)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                } else if let t = processingTime {
                    Text(String(format: "%.2fs", t)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                }
                Spacer()
                if isProcessing { ProgressView().controlSize(.small) }
                Text(status).font(.caption).foregroundStyle(.secondary)
                if !isModelLoaded { Text("Loading model…").font(.caption).foregroundStyle(.orange) }
            }
            .padding(.horizontal).padding(.vertical, 4)

            // Controls
            if mode == .photo {
                VStack(spacing: 8) {
                    HStack {
                        Text("Confidence").font(.caption2).foregroundStyle(.secondary)
                        Slider(value: $confidenceThreshold, in: 0.1...0.9)
                        Text(String(format: "%.0f%%", confidenceThreshold * 100))
                            .font(.caption2.monospacedDigit()).foregroundStyle(.secondary).frame(width: 36)
                    }
                    PhotosPicker(selection: $item, matching: .images) {
                        Label("Select Photo", systemImage: "photo.badge.plus").frame(maxWidth: .infinity)
                    }.buttonStyle(.bordered).disabled(isProcessing)
                }.padding(.horizontal).padding(.bottom, 8)
            }
        }
        .task { await loadModel() }
        .onChange(of: item) { _, _ in loadAndDetectPhoto() }
    }

    // MARK: - Camera View

    @ViewBuilder
    private var cameraView: some View {
        ZStack {
            CameraView(position: .back) { pixelBuffer in
                guard isModelLoaded else { return }
                detectOnFrame(pixelBuffer)
            }
            BoundingBoxOverlay(detections: liveDetections)
        }
    }

    // MARK: - Photo View

    @ViewBuilder
    private var photoView: some View {
        if let img = annotatedImage ?? inputImage {
            Image(uiImage: img).resizable().aspectRatio(contentMode: .fit)
                .clipShape(RoundedRectangle(cornerRadius: 8)).padding(.horizontal)
        } else {
            VStack(spacing: 12) {
                Image(systemName: "viewfinder").font(.system(size: 60)).foregroundStyle(.secondary)
                Text("Select a photo for detection").foregroundStyle(.secondary)
            }
        }
    }

    // MARK: - Model Loading

    private func loadModel() async {
        status = "Compiling model…"
        do {
            let loaded = try await ModelLoader.loadPrimary(for: model)
            let vn = try VNCoreMLModel(for: loaded)
            await MainActor.run {
                mlModel = loaded; vnModel = vn; isModelLoaded = true; status = ""
            }
        } catch {
            await MainActor.run { status = "Model load failed: \(error.localizedDescription)" }
        }
    }

    // MARK: - Real-time Camera Detection

    private var lastDetectionTime: CFAbsoluteTime { CFAbsoluteTimeGetCurrent() }
    @State private var frameSkip = 0

    private func detectOnFrame(_ pixelBuffer: CVPixelBuffer) {
        // Throttle to ~15 FPS for detection
        frameSkip += 1
        guard frameSkip % 2 == 0 else { return }

        guard let vnModel else { return }
        let start = CFAbsoluteTimeGetCurrent()

        let request = VNCoreMLRequest(model: vnModel) { req, _ in
            let dets = parseResults(req.results, imageWidth: CVPixelBufferGetWidth(pixelBuffer),
                                    imageHeight: CVPixelBufferGetHeight(pixelBuffer))
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            DispatchQueue.main.async {
                liveDetections = dets
                fps = fps * 0.9 + (1.0 / max(elapsed, 0.001)) * 0.1  // EMA smoothing
            }
        }
        request.imageCropAndScaleOption = .scaleFill

        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up).perform([request])
    }

    // MARK: - Photo Detection

    private func loadAndDetectPhoto() {
        guard let item else { return }
        isProcessing = true; status = "Loading…"
        Task {
            guard let data = try? await item.loadTransferable(type: Data.self),
                  let img = UIImage(data: data) else {
                await MainActor.run { isProcessing = false; status = "Failed" }; return
            }
            await MainActor.run { inputImage = img; annotatedImage = nil }
            await detectOnPhoto(img)
        }
    }

    private func detectOnPhoto(_ image: UIImage) async {
        guard let vnModel, let cgImage = ImageUtils.normalizeOrientation(image) else {
            isProcessing = false; status = "Image error"; return
        }
        status = "Detecting…"
        let start = CFAbsoluteTimeGetCurrent()

        let request = VNCoreMLRequest(model: vnModel) { req, _ in
            let dets = parseResults(req.results, imageWidth: cgImage.width, imageHeight: cgImage.height)
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            let annotated = drawDetections(dets, on: cgImage)

            DispatchQueue.main.async {
                annotatedImage = annotated
                processingTime = elapsed
                isProcessing = false; status = ""
            }
        }
        request.imageCropAndScaleOption = .scaleFill
        try? VNImageRequestHandler(cgImage: cgImage, orientation: .up).perform([request])
    }

    // MARK: - Result Parsing

    private func parseResults(_ results: [Any]?, imageWidth: Int, imageHeight: Int) -> [BoundingBoxOverlay.DetectionBox] {
        guard let results else { return [] }
        var dets: [BoundingBoxOverlay.DetectionBox] = []
        let threshold = confidenceThreshold

        // Path 1: VNRecognizedObjectObservation (NMS pipeline)
        if let observations = results as? [VNRecognizedObjectObservation] {
            for obs in observations where obs.confidence >= threshold {
                let label = obs.labels.first?.identifier ?? "?"
                let box = obs.boundingBox
                // Vision coordinates: origin bottom-left, flip Y for top-left
                dets.append(.init(
                    label: label, confidence: obs.confidence,
                    rect: CGRect(x: box.minX, y: 1 - box.maxY, width: box.width, height: box.height),
                    classIndex: labels.firstIndex(of: label) ?? 0
                ))
            }
            return dets
        }

        // Path 2: Raw MLMultiArray [1, N, 6] (NMS-free YOLO)
        if let observations = results as? [VNCoreMLFeatureValueObservation],
           let arr = observations.first?.featureValue.multiArrayValue {
            let shape = arr.shape.map { $0.intValue }
            guard shape.count == 3 && shape[2] >= 5 else { return dets }
            let n = shape[1]
            let strides = arr.strides.map { $0.intValue }
            let sz = Float(inputSize)

            for i in 0..<n {
                let conf = ImageUtils.readFloat(arr, at: i * strides[1] + 4 * strides[2])
                guard conf >= threshold else { continue }
                let x1 = ImageUtils.readFloat(arr, at: i * strides[1] + 0) / sz
                let y1 = ImageUtils.readFloat(arr, at: i * strides[1] + 1 * strides[2]) / sz
                let x2 = ImageUtils.readFloat(arr, at: i * strides[1] + 2 * strides[2]) / sz
                let y2 = ImageUtils.readFloat(arr, at: i * strides[1] + 3 * strides[2]) / sz
                let clsId = shape[2] > 5 ? Int(ImageUtils.readFloat(arr, at: i * strides[1] + 5 * strides[2])) : 0
                let label = clsId < labels.count ? labels[clsId] : "\(clsId)"

                // NMS-free output: coordinates already in image space (0..inputSize), normalize to 0..1
                dets.append(.init(
                    label: label, confidence: conf,
                    rect: CGRect(x: CGFloat(x1), y: CGFloat(y1), width: CGFloat(x2 - x1), height: CGFloat(y2 - y1)),
                    classIndex: clsId
                ))
            }
        }
        return dets
    }

    // MARK: - Drawing

    private func drawDetections(_ dets: [BoundingBoxOverlay.DetectionBox], on cgImage: CGImage) -> UIImage? {
        let w = CGFloat(cgImage.width), h = CGFloat(cgImage.height)
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: w, height: h))
        return renderer.image { ctx in
            // UIGraphicsImageRenderer uses UIKit coordinates (origin top-left) — correct for drawing
            UIImage(cgImage: cgImage).draw(in: CGRect(x: 0, y: 0, width: w, height: h))

            let colors: [UIColor] = [.systemRed, .systemBlue, .systemGreen, .systemOrange,
                                     .systemPurple, .systemTeal, .systemYellow, .systemPink]
            for det in dets {
                let color = colors[det.classIndex % colors.count]
                let rect = CGRect(x: det.rect.origin.x * w, y: det.rect.origin.y * h,
                                  width: det.rect.width * w, height: det.rect.height * h)

                ctx.cgContext.setStrokeColor(color.cgColor)
                ctx.cgContext.setLineWidth(max(2, w / 300))
                ctx.cgContext.stroke(rect)

                let attrs: [NSAttributedString.Key: Any] = [
                    .font: UIFont.boldSystemFont(ofSize: max(12, w / 50)),
                    .foregroundColor: UIColor.white,
                    .backgroundColor: color.withAlphaComponent(0.7)
                ]
                ("\(det.label) \(Int(det.confidence * 100))%" as NSString)
                    .draw(at: CGPoint(x: rect.minX + 2, y: rect.minY + 2), withAttributes: attrs)
            }
        }
    }

    static let cocoLabels = [
        "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
        "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
        "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
        "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
        "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
        "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
        "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
        "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
        "remote","keyboard","cell phone","microwave","oven","toaster","sink",
        "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
    ]
}
