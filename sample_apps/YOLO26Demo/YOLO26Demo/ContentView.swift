import SwiftUI
import UIKit
import AVFoundation
import CoreML
import Vision

// MARK: - Bounding Box View (CALayer pool)

class BoundingBoxView {
    let shapeLayer = CAShapeLayer()
    let fillLayer = CAShapeLayer()
    let textLayer = CATextLayer()

    init() {
        fillLayer.isHidden = true
        shapeLayer.fillColor = nil
        shapeLayer.lineWidth = 2
        shapeLayer.lineCap = .round
        shapeLayer.lineJoin = .round
        shapeLayer.isHidden = true
        textLayer.fontSize = 11
        textLayer.font = UIFont.systemFont(ofSize: 11, weight: .semibold)
        textLayer.foregroundColor = UIColor.white.cgColor
        textLayer.contentsScale = UIScreen.main.scale
        textLayer.isHidden = true
        textLayer.cornerRadius = 8
        textLayer.masksToBounds = true
        textLayer.alignmentMode = .center
    }

    func addToLayer(_ parent: CALayer) {
        parent.addSublayer(fillLayer)
        parent.addSublayer(shapeLayer)
        parent.addSublayer(textLayer)
    }

    func show(frame: CGRect, label: String, color: UIColor, alpha: CGFloat) {
        CATransaction.begin()
        CATransaction.setDisableActions(true)

        let path = UIBezierPath(roundedRect: frame, cornerRadius: 10).cgPath
        shapeLayer.path = path
        shapeLayer.strokeColor = color.withAlphaComponent(alpha).cgColor
        shapeLayer.isHidden = false

        fillLayer.path = path
        fillLayer.fillColor = color.withAlphaComponent(0.08).cgColor
        fillLayer.isHidden = false

        textLayer.string = "  \(label)  "
        textLayer.backgroundColor = color.withAlphaComponent(min(alpha + 0.1, 0.9)).cgColor
        let tw = CGFloat(label.count) * 7 + 20
        let ty = frame.minY > 28 ? frame.minY - 24 : frame.maxY + 4
        textLayer.frame = CGRect(x: frame.minX, y: ty,
                                 width: min(tw, max(frame.width + 24, 64)), height: 20)
        textLayer.isHidden = false

        CATransaction.commit()
    }

    func hide() {
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        shapeLayer.isHidden = true
        fillLayer.isHidden = true
        textLayer.isHidden = true
        CATransaction.commit()
    }
}

// MARK: - ViewController

class DetectionCameraVC: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "session")
    private let inferenceQueue = DispatchQueue(label: "inference")
    private var previewLayer: AVCaptureVideoPreviewLayer!
    private var boxViews: [BoundingBoxView] = []
    private var vnModel: VNCoreMLModel?
    private var isProcessing = false

    private var longSide: CGFloat = 1920
    private var shortSide: CGFloat = 1080
    private var frameSizeCaptured = false

    private let confThreshold: Float = 0.25
    private let colors: [UIColor] = [
        UIColor(red: 1.0, green: 0.44, blue: 0.56, alpha: 1),  // coral pink
        UIColor(red: 0.55, green: 0.82, blue: 0.96, alpha: 1), // sky blue
        UIColor(red: 0.68, green: 0.92, blue: 0.68, alpha: 1), // mint green
        UIColor(red: 0.95, green: 0.72, blue: 0.42, alpha: 1), // peach
        UIColor(red: 0.76, green: 0.62, blue: 0.95, alpha: 1), // lavender
        UIColor(red: 1.0, green: 0.85, blue: 0.45, alpha: 1),  // butter yellow
        UIColor(red: 0.95, green: 0.55, blue: 0.75, alpha: 1), // rose
        UIColor(red: 0.45, green: 0.88, blue: 0.82, alpha: 1), // teal
        UIColor(red: 0.98, green: 0.65, blue: 0.65, alpha: 1), // salmon
        UIColor(red: 0.65, green: 0.72, blue: 0.98, alpha: 1), // periwinkle
    ]
    private let labels = [
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

    override func viewDidLoad() {
        super.viewDidLoad()
        previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)

        for _ in 0..<100 {
            let bv = BoundingBoxView()
            bv.addToLayer(previewLayer)
            boxViews.append(bv)
        }

        loadModel()
        AVCaptureDevice.requestAccess(for: .video) { [weak self] ok in
            guard ok else { return }
            self?.sessionQueue.async { self?.setupCamera() }
        }
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer.frame = view.bounds
    }

    private func loadModel() {
        for name in ["yolo26s", "yolov10s"] {
            guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else { continue }
            do {
                let cfg = MLModelConfiguration()
                cfg.computeUnits = .all
                vnModel = try VNCoreMLModel(for: MLModel(contentsOf: url, configuration: cfg))
                return
            } catch { print("model load error: \(error)") }
        }
    }

    private func setupCamera() {
        session.beginConfiguration()
        session.sessionPreset = .high
        guard let dev = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: dev) else { session.commitConfiguration(); return }
        if session.canAddInput(input) { session.addInput(input) }

        let out = AVCaptureVideoDataOutput()
        out.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        out.alwaysDiscardsLateVideoFrames = true
        out.setSampleBufferDelegate(self, queue: inferenceQueue)
        if session.canAddOutput(out) { session.addOutput(out) }

        session.commitConfiguration()

        // Set video orientation to portrait AFTER addOutput (same as yolo-ios-app)
        // This makes the pixel buffer arrive already rotated to portrait
        let connection = out.connection(with: .video)
        connection?.videoOrientation = .portrait

        // Also set preview orientation
        previewLayer.connection?.videoOrientation = .portrait

        session.startRunning()
    }

    // MARK: - Capture

    func captureOutput(_ output: AVCaptureOutput, didOutput sb: CMSampleBuffer, from conn: AVCaptureConnection) {
        guard !isProcessing, let vnModel else { return }
        guard let pb = CMSampleBufferGetImageBuffer(sb) else { return }

        if !frameSizeCaptured {
            let w = CGFloat(CVPixelBufferGetWidth(pb))
            let h = CGFloat(CVPixelBufferGetHeight(pb))
            longSide = max(w, h)
            shortSide = min(w, h)
            frameSizeCaptured = true
            print("[YOLO26] frame: \(w)x\(h) long=\(longSide) short=\(shortSide)")
        }

        isProcessing = true
        let req = VNCoreMLRequest(model: vnModel) { [weak self] req, _ in
            guard let self else { return }
            defer { isProcessing = false }
            let dets = self.parseDetections(req)
            DispatchQueue.main.async { self.showBoxes(dets) }
        }
        // scaleFill: stretches full frame to 640x640
        req.imageCropAndScaleOption = .scaleFill
        // .up because frame is already rotated to portrait by videoOrientation
        try? VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up).perform([req])
    }

    // MARK: - Parse

    private struct Det {
        let label: String
        let confidence: Float
        let classIndex: Int
        let normRect: CGRect // [0,1], top-left origin
    }

    private func parseDetections(_ req: VNRequest) -> [Det] {
        if let results = req.results as? [VNRecognizedObjectObservation], !results.isEmpty {
            return results.compactMap { obs in
                guard let top = obs.labels.first, top.confidence >= confThreshold else { return nil }
                let vr = obs.boundingBox
                let nr = CGRect(x: vr.minX, y: 1 - vr.maxY, width: vr.width, height: vr.height)
                let idx = labels.firstIndex(of: top.identifier) ?? 0
                return Det(label: top.identifier, confidence: top.confidence, classIndex: idx, normRect: nr)
            }
        }

        guard let results = req.results as? [VNCoreMLFeatureValueObservation] else { return [] }
        var out: [Det] = []
        for obs in results {
            guard let arr = obs.featureValue.multiArrayValue else { continue }
            let shape = arr.shape.map { $0.intValue }
            guard shape.count == 3 && shape[2] == 6 else { continue }
            for i in 0..<shape[1] {
                let conf = arr[[0, i, 4] as [NSNumber]].floatValue
                guard conf >= confThreshold else { continue }
                // Model pixel coords [0, 640] → normalized [0, 1]
                let x1 = CGFloat(arr[[0, i, 0] as [NSNumber]].floatValue) / 640
                let y1 = CGFloat(arr[[0, i, 1] as [NSNumber]].floatValue) / 640
                let x2 = CGFloat(arr[[0, i, 2] as [NSNumber]].floatValue) / 640
                let y2 = CGFloat(arr[[0, i, 3] as [NSNumber]].floatValue) / 640
                let cid = Int(arr[[0, i, 5] as [NSNumber]].floatValue)
                let label = cid < labels.count ? labels[cid] : "\(cid)"
                // Store in Vision coordinate system (bottom-left origin) to match showBoxes transform
                out.append(Det(label: label, confidence: conf, classIndex: cid,
                               normRect: CGRect(x: x1, y: 1 - y2, width: x2 - x1, height: y2 - y1)))
            }
        }
        return out
    }

    // MARK: - Show boxes (yolo-ios-app portrait logic)

    private func showBoxes(_ dets: [Det]) {
        let width = view.bounds.width
        let height = view.bounds.height

        // Camera aspect ratio vs display aspect ratio
        // shortSide/longSide = portrait width/height of camera (e.g., 1080/1920)
        let ratio = (height / width) / (longSide / shortSide)

        for i in 0..<boxViews.count {
            guard i < dets.count && i < 50 else { boxViews[i].hide(); continue }

            var displayRect = dets[i].normRect

            if ratio >= 1 {
                // Display is taller than camera → camera fills width, height is cropped
                let offset = (1 - ratio) * (0.5 - displayRect.minX)
                let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: offset, y: -1)
                displayRect = displayRect.applying(transform)
                displayRect.size.width *= ratio
            } else {
                // Display is wider than camera → camera fills height, width is cropped
                let offset = (ratio - 1) * (0.5 - displayRect.maxY)
                let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: offset - 1)
                displayRect = displayRect.applying(transform)
                let r2 = (height / width) / (shortSide / longSide)
                displayRect.size.height /= r2
            }

            let screenRect = VNImageRectForNormalizedRect(displayRect, Int(width), Int(height))
            let det = dets[i]
            let color = colors[det.classIndex % colors.count]
            let label = String(format: "%@ %.0f%%", det.label, det.confidence * 100)
            let alpha = CGFloat(max(det.confidence - 0.2, 0.1) / 0.8 * 0.9)
            boxViews[i].show(frame: screenRect, label: label, color: color, alpha: alpha)
        }
    }
}

// MARK: - SwiftUI wrapper

struct DetectionCameraView: UIViewControllerRepresentable {
    func makeUIViewController(context: Context) -> DetectionCameraVC { DetectionCameraVC() }
    func updateUIViewController(_ vc: DetectionCameraVC, context: Context) {}
}

struct ContentView: View {
    var body: some View {
        DetectionCameraView().ignoresSafeArea()
    }
}

#Preview { ContentView() }
