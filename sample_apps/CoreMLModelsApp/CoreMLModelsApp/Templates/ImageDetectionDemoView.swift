import SwiftUI
import PhotosUI
import CoreML
import Vision
import AVFoundation

/// Object detection with real-time camera, photo picker, and video processing.
/// Matches YOLO26Demo / YOLOv9Demo UX: live camera feed with bounding box overlays,
/// photo mode with annotated results, video mode with frame-by-frame detection.
struct ImageDetectionDemoView: View {
    let model: ModelEntry

    enum Mode: String, CaseIterable { case camera = "Camera", video = "Video", photo = "Photo" }

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
    @State private var didLoadConfidenceDefault = false
    @State private var vnModel: VNCoreMLModel?
    @State private var mlModel: MLModel?
    @State private var isModelLoaded = false
    @StateObject private var session = ModelSession<MLModel>()

    // Video state
    @State private var videoItem: PhotosPickerItem?
    @State private var videoFrame: UIImage?
    @State private var videoDetections: [BoundingBoxOverlay.DetectionBox] = []
    @State private var videoFrameSize: CGSize = .zero
    @State private var videoProgress: Double = 0
    @State private var videoFps: Double = 0
    @State private var videoTask: Task<Void, Never>?

    // Camera frame dimensions (portrait after videoRotationAngle=90) — needed
    // so the overlay can match the preview's .resizeAspectFill crop/scale.
    @State private var cameraFrameSize: CGSize = .zero

    // Multi-object tracking. Camera keeps a single tracker across frames;
    // Video creates a fresh one per loaded clip. Photo mode never uses
    // the tracker (single-frame → no motion to track).
    @State private var trackingEnabled: Bool = true
    @State private var cameraTracker = ByteTracker()

    private var labels: [String] { model.configStringArray("labels") ?? Self.cocoLabels }
    private var inputSize: Int { model.configInt("input_size") ?? 640 }

    /// For DETR-style models where the network outputs 91 logits mapped to sparse COCO
    /// category IDs (1..90, with gaps at 12/26/29/30/45/66/68/69/71/83) and index 0 = background.
    private var useSparseCoco: Bool {
        (model.configString("output_format") ?? "yolo") == "detr"
            && model.configInt("num_classes") == 91
            && (model.configStringArray("labels") == nil)
    }

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
                case .video:
                    videoView
                case .photo:
                    photoView
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            // Status bar
            HStack {
                if mode == .camera {
                    Text(String(format: "%.1f FPS", fps)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                } else if mode == .video && videoFrame != nil {
                    Text(String(format: "%.1f FPS", videoFps)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                } else {
                    TimingsLabel(loadSec: session.loadTimeSec, inferSec: processingTime)
                }
                if mode != .photo {
                    Button { trackingEnabled.toggle() } label: {
                        Label(trackingEnabled ? "Track" : "Raw",
                              systemImage: trackingEnabled ? "scope" : "circle.dashed")
                            .font(.caption2.weight(.semibold))
                            .labelStyle(.titleAndIcon)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.mini)
                    .tint(trackingEnabled ? .blue : .gray)
                }
                Spacer()
                if isProcessing { ProgressView().controlSize(.small) }
                Text(status).font(.caption).foregroundStyle(.secondary)
                if !isModelLoaded { Text("Loading model…").font(.caption).foregroundStyle(.orange) }
            }
            .padding(.horizontal).padding(.vertical, 4)

            // Confidence slider — shared across Camera / Video / Photo so the user
            // can tune the threshold live (RF-DETR in particular needs ~0.5).
            confidenceSlider

            // Mode-specific controls
            if mode == .photo {
                photoControls
            } else if mode == .video {
                videoControls
            }
        }
        .task { await loadModel() }
        .onChange(of: item) { _, _ in loadAndDetectPhoto() }
        .onChange(of: videoItem) { _, _ in loadAndProcessVideo() }
        .onChange(of: mode) { _, newMode in
            if newMode != .video { videoTask?.cancel() }
            cameraTracker.reset()
        }
        .onChange(of: trackingEnabled) { _, _ in
            // Reset camera tracker so IDs restart, and rebuild the video
            // playback so the new mode takes effect on the in-flight task.
            cameraTracker.reset()
            liveDetections = []
            if mode == .video { loadAndProcessVideo() }
        }
        .onDisappear {
            videoTask?.cancel()
            vnModel = nil
            mlModel = nil
        }
    }

    @ViewBuilder
    private var confidenceSlider: some View {
        HStack {
            Text("Confidence").font(.caption2).foregroundStyle(.secondary)
            Slider(value: $confidenceThreshold, in: 0.05...0.95, step: 0.01)
            Text(String(format: "%.0f%%", confidenceThreshold * 100))
                .font(.caption2.monospacedDigit()).foregroundStyle(.secondary).frame(width: 36)
        }
        .padding(.horizontal).padding(.bottom, 4)
    }

    @ViewBuilder
    private var photoControls: some View {
        HStack(spacing: 12) {
            PhotosPicker(selection: $item, matching: .images) {
                Label("Select Photo", systemImage: "photo.badge.plus").frame(maxWidth: .infinity)
            }.buttonStyle(.bordered).disabled(isProcessing)

            if let annotated = annotatedImage {
                Button {
                    UIImageWriteToSavedPhotosAlbum(annotated, nil, nil, nil)
                } label: {
                    Image(systemName: "arrow.down.to.line")
                }.buttonStyle(.bordered)
            }
        }
        .padding(.horizontal).padding(.bottom, 8)
    }

    @ViewBuilder
    private var videoControls: some View {
        VStack(spacing: 8) {
            if videoFrame != nil {
                ProgressView(value: videoProgress).tint(.blue).padding(.horizontal)
            }
            PhotosPicker(selection: $videoItem, matching: .videos) {
                Label("Select Video", systemImage: "video.badge.plus").frame(maxWidth: .infinity)
            }.buttonStyle(.bordered).disabled(isProcessing)
        }
        .padding(.horizontal).padding(.bottom, 8)
    }

    // MARK: - Camera View

    @ViewBuilder
    private var cameraView: some View {
        ZStack {
            CameraView(position: .back) { pixelBuffer in
                guard isModelLoaded else { return }
                if cameraFrameSize == .zero {
                    let w = CVPixelBufferGetWidth(pixelBuffer)
                    let h = CVPixelBufferGetHeight(pixelBuffer)
                    DispatchQueue.main.async {
                        cameraFrameSize = CGSize(width: w, height: h)
                    }
                }
                detectOnFrame(pixelBuffer)
            }
            BoundingBoxOverlay(
                detections: liveDetections,
                frameSize: cameraFrameSize,
                contentMode: .fill
            )
        }
    }

    // MARK: - Video View

    @ViewBuilder
    private var videoView: some View {
        if let frame = videoFrame {
            Image(uiImage: frame)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .overlay {
                    BoundingBoxOverlay(
                        detections: videoDetections,
                        frameSize: videoFrameSize,
                        contentMode: .fit
                    )
                }
        } else {
            VStack(spacing: 12) {
                Image(systemName: "video.badge.plus").font(.system(size: 60)).foregroundStyle(.secondary)
                Text("Select a video for detection").foregroundStyle(.secondary)
            }
        }
    }

    // MARK: - Video Processing

    private func loadAndProcessVideo() {
        videoTask?.cancel()
        guard let videoItem else { return }
        isProcessing = true; status = "Loading video…"
        let tracking = trackingEnabled
        Task {
            guard let transferable = try? await videoItem.loadTransferable(type: DetectionVideoTransferable.self) else {
                await MainActor.run { isProcessing = false; status = "Failed to load video" }
                return
            }
            let url = transferable.url
            await MainActor.run { isProcessing = false; status = "" }
            videoTask = Task.detached(priority: .userInitiated) {
                await processVideo(url: url, tracking: tracking)
            }
        }
    }

    private func processVideo(url: URL, tracking: Bool) async {
        let asset = AVURLAsset(url: url)
        guard let track = try? await asset.loadTracks(withMediaType: .video).first else { return }
        let duration = try? await asset.load(.duration)
        let totalSeconds = duration.map { CMTimeGetSeconds($0) } ?? 1
        let nominalFPS = (try? await track.load(.nominalFrameRate)) ?? 30
        let frameInterval = 1.0 / Double(nominalFPS)

        guard let reader = try? AVAssetReader(asset: asset) else { return }
        let outputSettings: [String: Any] = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        let trackOutput = AVAssetReaderTrackOutput(track: track, outputSettings: outputSettings)
        reader.add(trackOutput)
        reader.startReading()

        let ciContext = CIContext()
        let videoTracker = ByteTracker()

        while !Task.isCancelled, let sampleBuffer = trackOutput.copyNextSampleBuffer() {
            let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
            let currentSec = CMTimeGetSeconds(pts)

            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { continue }
            guard let vnModel else { continue }

            let start = CFAbsoluteTimeGetCurrent()

            // Run detection synchronously
            let request = VNCoreMLRequest(model: vnModel)
            request.imageCropAndScaleOption = .scaleFill
            try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up).perform([request])
            let raw = parseResults(request.results, imageWidth: CVPixelBufferGetWidth(pixelBuffer),
                                   imageHeight: CVPixelBufferGetHeight(pixelBuffer))
            let dets = tracking ? videoTracker.update(detections: raw) : raw

            let elapsed = CFAbsoluteTimeGetCurrent() - start

            // Convert pixel buffer to UIImage
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else { continue }
            let frame = UIImage(cgImage: cgImage)
            let currentFPS = 1.0 / max(elapsed, 0.001)

            let size = CGSize(width: cgImage.width, height: cgImage.height)
            await MainActor.run {
                videoFrame = frame
                videoDetections = dets
                videoFrameSize = size
                videoProgress = min(currentSec / totalSeconds, 1.0)
                videoFps = videoFps == 0 ? currentFPS : videoFps * 0.9 + currentFPS * 0.1
            }

            // Pace playback to match original frame rate
            let sleepTime = max(frameInterval - elapsed, 0)
            if sleepTime > 0 {
                try? await Task.sleep(for: .seconds(sleepTime))
            }
        }

        await MainActor.run {
            videoProgress = 1.0
            status = ""
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
        session.ensure { try await ModelLoader.loadPrimary(for: model) }
        do {
            let loaded = try await session.get()
            let vn = try VNCoreMLModel(for: loaded)
            await MainActor.run {
                mlModel = loaded; vnModel = vn; isModelLoaded = true; status = ""
                if !didLoadConfidenceDefault,
                   let defaultConf = model.configDouble("confidence_threshold") {
                    confidenceThreshold = Float(defaultConf)
                    didLoadConfidenceDefault = true
                }
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

        let tracking = trackingEnabled

        let request = VNCoreMLRequest(model: vnModel) { req, _ in
            let raw = parseResults(req.results, imageWidth: CVPixelBufferGetWidth(pixelBuffer),
                                   imageHeight: CVPixelBufferGetHeight(pixelBuffer))
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            DispatchQueue.main.async {
                liveDetections = tracking ? cameraTracker.update(detections: raw) : raw
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

        // Path 2: Raw MLMultiArray — NMS-free YOLO [1,N,6] or DETR [N,91]+[N,4]
        if let observations = results as? [VNCoreMLFeatureValueObservation] {
            let arrays = observations.compactMap { $0.featureValue.multiArrayValue }
            let outputFormat = model.configString("output_format") ?? "yolo"

            if outputFormat == "detr" {
                // DETR: two arrays — confidence [N, num_classes] and coordinates [N, 4] (cxcywh normalized)
                guard let confArr = arrays.first(where: { $0.shape.count == 2 && $0.shape[1].intValue > 4 }),
                      let boxArr = arrays.first(where: { $0.shape.count == 2 && $0.shape[1].intValue == 4 }) else { return dets }
                let n = confArr.shape[0].intValue
                let numClasses = confArr.shape[1].intValue
                let bgClass = model.configInt("background_class") ?? 0
                let confStrides = confArr.strides.map { $0.intValue }
                let boxStrides = boxArr.strides.map { $0.intValue }

                for i in 0..<n {
                    // Find best non-background class
                    var bestConf: Float = 0; var bestCls = 0
                    for c in 0..<numClasses where c != bgClass {
                        let v = ImageUtils.readFloat(confArr, at: i * confStrides[0] + c * confStrides[1])
                        if v > bestConf { bestConf = v; bestCls = c }
                    }
                    guard bestConf >= threshold else { continue }

                    let cx = ImageUtils.readFloat(boxArr, at: i * boxStrides[0] + 0)
                    let cy = ImageUtils.readFloat(boxArr, at: i * boxStrides[0] + 1 * boxStrides[1])
                    let bw = ImageUtils.readFloat(boxArr, at: i * boxStrides[0] + 2 * boxStrides[1])
                    let bh = ImageUtils.readFloat(boxArr, at: i * boxStrides[0] + 3 * boxStrides[1])

                    // Resolve label. RF-DETR uses sparse COCO IDs 1..90; YOLO-DETR hybrids
                    // or user-supplied labels use contiguous indexing.
                    let label: String
                    let classIdx: Int
                    if useSparseCoco {
                        label = Self.cocoSparse91[safe: bestCls] ?? "\(bestCls)"
                        classIdx = bestCls
                    } else {
                        let labelIdx = bestCls - 1
                        label = labelIdx >= 0 && labelIdx < labels.count ? labels[labelIdx] : "\(bestCls)"
                        classIdx = max(0, labelIdx)
                    }
                    // Skip N/A slots in the sparse COCO table
                    if label == "N/A" { continue }

                    dets.append(.init(
                        label: label, confidence: bestConf,
                        rect: CGRect(x: CGFloat(cx - bw / 2), y: CGFloat(cy - bh / 2),
                                     width: CGFloat(bw), height: CGFloat(bh)),
                        classIndex: classIdx
                    ))
                }
            } else if let arr = arrays.first {
                // NMS-free YOLO [1, N, 6]
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

                    dets.append(.init(
                        label: label, confidence: conf,
                        rect: CGRect(x: CGFloat(x1), y: CGFloat(y1), width: CGFloat(x2 - x1), height: CGFloat(y2 - y1)),
                        classIndex: clsId
                    ))
                }
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

    /// 91-entry sparse COCO lookup used by RF-DETR: index = COCO category ID,
    /// index 0 is the background / no-object class, and unused IDs (12, 26, 29,
    /// 30, 45, 66, 68, 69, 71, 83) are filled with "N/A" so `bestCls` maps
    /// directly without the -1 offset the hub template applies to other DETRs.
    static let cocoSparse91: [String] = [
        "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
        "train", "truck", "boat", "traffic light", "fire hydrant", "N/A",
        "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
        "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A",
        "backpack", "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase",
        "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
        "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
        "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A",
        "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
        "sink", "refrigerator", "N/A", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"
    ]
}

fileprivate extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}

// MARK: - Video Transferable

struct DetectionVideoTransferable: Transferable {
    let url: URL
    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { movie in
            SentTransferredFile(movie.url)
        } importing: { received in
            let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(
                UUID().uuidString + "." + received.file.pathExtension)
            try FileManager.default.copyItem(at: received.file, to: tmp)
            return Self(url: tmp)
        }
    }
}
