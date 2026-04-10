import SwiftUI
import PhotosUI
import CoreML
import Vision

/// Object detection demo template.
/// Used by: YOLOv9, YOLOv10, YOLO26.
///
/// Expected manifest config:
/// ```
/// { "input_size": 640, "confidence_threshold": 0.25, "labels": ["person", "bicycle", ...] }
/// ```
struct ImageDetectionDemoView: View {
    let model: ModelEntry

    @State private var inputImage: UIImage?
    @State private var annotatedImage: UIImage?
    @State private var detections: [Detection] = []
    @State private var isProcessing = false
    @State private var status = ""
    @State private var processingTime: Double?
    @State private var item: PhotosPickerItem?
    @State private var confidenceThreshold: Float = 0.25

    struct Detection: Identifiable {
        let id = UUID()
        let label: String
        let confidence: Float
        let box: CGRect  // normalized 0..1
    }

    var body: some View {
        VStack(spacing: 0) {
            ZStack {
                if let img = annotatedImage ?? inputImage {
                    Image(uiImage: img).resizable().aspectRatio(contentMode: .fit)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                } else {
                    VStack(spacing: 12) {
                        Image(systemName: "viewfinder").font(.system(size: 60)).foregroundStyle(.secondary)
                        Text("Select a photo for object detection").foregroundStyle(.secondary)
                    }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .padding(.horizontal)

            if !detections.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(detections) { d in
                            VStack(spacing: 2) {
                                Text(d.label).font(.caption2.bold())
                                Text(String(format: "%.0f%%", d.confidence * 100))
                                    .font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
                            }
                            .padding(.horizontal, 8).padding(.vertical, 4)
                            .background(.ultraThinMaterial)
                            .clipShape(RoundedRectangle(cornerRadius: 6))
                        }
                    }
                    .padding(.horizontal)
                }
                .frame(height: 44)
            }

            VStack(spacing: 12) {
                HStack {
                    Text("Confidence").font(.caption2).foregroundStyle(.secondary)
                    Slider(value: $confidenceThreshold, in: 0.1...0.9)
                    Text(String(format: "%.0f%%", confidenceThreshold * 100))
                        .font(.caption2.monospacedDigit()).foregroundStyle(.secondary).frame(width: 36)
                }
                HStack {
                    if let t = processingTime {
                        Text(String(format: "%.2fs • %d objects", t, detections.count))
                            .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    }
                    Spacer()
                    if isProcessing { ProgressView().controlSize(.small) }
                }
                PhotosPicker(selection: $item, matching: .images) {
                    Label("Select Photo", systemImage: "photo.badge.plus").frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered).disabled(isProcessing)
            }
            .padding()
        }
        .onChange(of: item) { _, _ in loadAndRun() }
    }

    private func loadAndRun() {
        guard let item else { return }
        isProcessing = true; status = "Loading…"
        Task {
            guard let data = try? await item.loadTransferable(type: Data.self),
                  let img = UIImage(data: data) else {
                await MainActor.run { isProcessing = false; status = "Failed" }
                return
            }
            await MainActor.run { inputImage = img; annotatedImage = nil; detections = [] }
            await runDetection(on: img)
        }
    }

    private func runDetection(on image: UIImage) async {
        await MainActor.run { status = "Compiling model…" }
        do {
            let mlModel = try await ModelLoader.loadPrimary(for: model)
            await MainActor.run { status = "Detecting…" }

            let inputSize = model.configInt("input_size") ?? 640
            guard let cgImage = ImageUtils.normalizeOrientation(image),
                  let pb = ImageUtils.pixelBuffer(from: cgImage, width: inputSize, height: inputSize) else {
                await MainActor.run { isProcessing = false; status = "Image prep failed" }
                return
            }

            let start = CFAbsoluteTimeGetCurrent()

            // Try Vision framework first (handles NMS-free YOLO output)
            let vnModel = try VNCoreMLModel(for: mlModel)
            let results = try await withCheckedThrowingContinuation { (cont: CheckedContinuation<[Any], Error>) in
                let req = VNCoreMLRequest(model: vnModel) { req, err in
                    if let err { cont.resume(throwing: err) }
                    else { cont.resume(returning: req.results ?? []) }
                }
                req.imageCropAndScaleOption = .scaleFill
                try? VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up).perform([req])
            }

            let elapsed = CFAbsoluteTimeGetCurrent() - start
            let labels = model.configStringArray("labels") ?? Self.cocoLabels
            let threshold = confidenceThreshold

            var dets: [Detection] = []

            // Parse VNRecognizedObjectObservation (standard path)
            if let observations = results as? [VNRecognizedObjectObservation] {
                for obs in observations where obs.confidence >= threshold {
                    let label = obs.labels.first?.identifier ?? "?"
                    let box = obs.boundingBox
                    dets.append(Detection(label: label, confidence: obs.confidence, box: box))
                }
            }
            // Fallback: parse raw MLMultiArray output [1, N, 6] (NMS-free YOLO)
            else if let observations = results as? [VNCoreMLFeatureValueObservation],
                    let arr = observations.first?.featureValue.multiArrayValue {
                let shape = arr.shape.map { $0.intValue }
                if shape.count == 3 && shape[2] >= 5 {
                    let n = shape[1], cols = shape[2]
                    let strides = arr.strides.map { $0.intValue }
                    let sz = Float(inputSize)
                    for i in 0..<n {
                        let conf = ImageUtils.readFloat(arr, at: i * strides[1] + 4 * strides[2])
                        guard conf >= threshold else { continue }
                        let x1 = ImageUtils.readFloat(arr, at: i * strides[1] + 0) / sz
                        let y1 = ImageUtils.readFloat(arr, at: i * strides[1] + 1) / sz
                        let x2 = ImageUtils.readFloat(arr, at: i * strides[1] + 2) / sz
                        let y2 = ImageUtils.readFloat(arr, at: i * strides[1] + 3) / sz
                        let clsId = cols > 5 ? Int(ImageUtils.readFloat(arr, at: i * strides[1] + 5 * strides[2])) : 0
                        let label = clsId < labels.count ? labels[clsId] : "\(clsId)"
                        dets.append(Detection(
                            label: label, confidence: conf,
                            box: CGRect(x: CGFloat(x1), y: CGFloat(1 - y2),
                                        width: CGFloat(x2 - x1), height: CGFloat(y2 - y1))
                        ))
                    }
                }
            }

            // Draw boxes on image
            let annotated = drawDetections(dets, on: image)

            await MainActor.run {
                detections = dets.sorted { $0.confidence > $1.confidence }
                annotatedImage = annotated
                processingTime = elapsed
                isProcessing = false; status = ""
            }
        } catch {
            await MainActor.run { isProcessing = false; status = "Error: \(error.localizedDescription)" }
        }
    }

    private func drawDetections(_ dets: [Detection], on image: UIImage) -> UIImage? {
        guard let cgImage = ImageUtils.normalizeOrientation(image) else { return image }
        let w = CGFloat(cgImage.width), h = CGFloat(cgImage.height)

        let renderer = UIGraphicsImageRenderer(size: CGSize(width: w, height: h))
        return renderer.image { ctx in
            ctx.cgContext.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))

            let colors: [UIColor] = [.systemRed, .systemBlue, .systemGreen, .systemOrange,
                                     .systemPurple, .systemTeal, .systemYellow, .systemPink]
            for (i, det) in dets.enumerated() {
                let color = colors[i % colors.count]
                let rect = CGRect(
                    x: det.box.origin.x * w,
                    y: (1 - det.box.origin.y - det.box.height) * h,
                    width: det.box.width * w,
                    height: det.box.height * h
                )

                ctx.cgContext.setStrokeColor(color.cgColor)
                ctx.cgContext.setLineWidth(max(2, w / 300))
                ctx.cgContext.stroke(rect)

                let text = "\(det.label) \(Int(det.confidence * 100))%"
                let attrs: [NSAttributedString.Key: Any] = [
                    .font: UIFont.boldSystemFont(ofSize: max(12, w / 50)),
                    .foregroundColor: UIColor.white,
                    .backgroundColor: color.withAlphaComponent(0.7)
                ]
                (text as NSString).draw(at: CGPoint(x: rect.minX + 2, y: rect.minY + 2), withAttributes: attrs)
            }
        }
    }

    // COCO 80-class labels
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
