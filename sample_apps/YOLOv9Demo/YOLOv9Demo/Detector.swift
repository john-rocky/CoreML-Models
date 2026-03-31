import UIKit
import CoreML
import Vision

// MARK: - Detection Result

struct Detection: Identifiable {
    let id = UUID()
    let label: String
    let confidence: Float
    let classIndex: Int
    let normRect: CGRect // [0,1], top-left origin (x, y, w, h)
}

// MARK: - Shared Detector

class Detector: ObservableObject {
    private var mlModel: MLModel?
    private var vnModel: VNCoreMLModel?
    @Published var isReady = false

    let confThreshold: Float = 0.25

    let colors: [UIColor] = [
        UIColor(red: 1.0, green: 0.44, blue: 0.56, alpha: 1),
        UIColor(red: 0.55, green: 0.82, blue: 0.96, alpha: 1),
        UIColor(red: 0.68, green: 0.92, blue: 0.68, alpha: 1),
        UIColor(red: 0.95, green: 0.72, blue: 0.42, alpha: 1),
        UIColor(red: 0.76, green: 0.62, blue: 0.95, alpha: 1),
        UIColor(red: 1.0, green: 0.85, blue: 0.45, alpha: 1),
        UIColor(red: 0.95, green: 0.55, blue: 0.75, alpha: 1),
        UIColor(red: 0.45, green: 0.88, blue: 0.82, alpha: 1),
        UIColor(red: 0.98, green: 0.65, blue: 0.65, alpha: 1),
        UIColor(red: 0.65, green: 0.72, blue: 0.98, alpha: 1),
    ]

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

    init() { loadModel() }

    private func loadModel() {
        for name in ["yolov9s", "yolo11s"] {
            guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else { continue }
            do {
                let cfg = MLModelConfiguration()
                cfg.computeUnits = .all
                mlModel = try MLModel(contentsOf: url, configuration: cfg)
                vnModel = try VNCoreMLModel(for: mlModel!)
                DispatchQueue.main.async { self.isReady = true }
                return
            } catch { print("model load error: \(error)") }
        }
    }

    var visionModel: VNCoreMLModel? { vnModel }

    // MARK: - Detect on UIImage (async, for photo mode)

    func detect(image: UIImage) async -> [Detection] {
        guard let cgImage = image.cgImage else { return [] }
        return await withCheckedContinuation { cont in
            guard let vnModel else { cont.resume(returning: []); return }
            let req = VNCoreMLRequest(model: vnModel) { [weak self] req, _ in
                cont.resume(returning: self?.parseResults(req) ?? [])
            }
            req.imageCropAndScaleOption = .scaleFill
            try? VNImageRequestHandler(cgImage: cgImage, orientation: .up).perform([req])
        }
    }

    // MARK: - Detect on CVPixelBuffer (sync, for video/camera)

    func detect(pixelBuffer: CVPixelBuffer) -> [Detection] {
        guard let vnModel else { return [] }
        var result: [Detection] = []
        let req = VNCoreMLRequest(model: vnModel) { [weak self] req, _ in
            result = self?.parseResults(req) ?? []
        }
        req.imageCropAndScaleOption = .scaleFill
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up).perform([req])
        return result
    }

    // MARK: - Parse

    func parseResults(_ req: VNRequest) -> [Detection] {
        if let results = req.results as? [VNRecognizedObjectObservation], !results.isEmpty {
            return results.compactMap { obs in
                guard let top = obs.labels.first, top.confidence >= confThreshold else { return nil }
                let vr = obs.boundingBox
                let idx = Self.cocoLabels.firstIndex(of: top.identifier) ?? 0
                return Detection(label: top.identifier, confidence: top.confidence, classIndex: idx,
                                 normRect: CGRect(x: vr.minX, y: 1 - vr.maxY, width: vr.width, height: vr.height))
            }
        }

        guard let results = req.results as? [VNCoreMLFeatureValueObservation] else { return [] }
        var out: [Detection] = []
        for obs in results {
            guard let arr = obs.featureValue.multiArrayValue else { continue }
            let shape = arr.shape.map { $0.intValue }
            guard shape.count == 3 && shape[2] == 6 else { continue }
            for i in 0..<shape[1] {
                let conf = arr[[0, i, 4] as [NSNumber]].floatValue
                guard conf >= confThreshold else { continue }
                let x1 = CGFloat(arr[[0, i, 0] as [NSNumber]].floatValue) / 640
                let y1 = CGFloat(arr[[0, i, 1] as [NSNumber]].floatValue) / 640
                let x2 = CGFloat(arr[[0, i, 2] as [NSNumber]].floatValue) / 640
                let y2 = CGFloat(arr[[0, i, 3] as [NSNumber]].floatValue) / 640
                let cid = Int(arr[[0, i, 5] as [NSNumber]].floatValue)
                let label = cid < Self.cocoLabels.count ? Self.cocoLabels[cid] : "\(cid)"
                out.append(Detection(label: label, confidence: conf, classIndex: cid,
                                     normRect: CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1)))
            }
        }
        return out
    }

    // MARK: - UIImage → CVPixelBuffer

    static func imageToPixelBuffer(_ image: UIImage, size: CGSize = CGSize(width: 640, height: 640)) -> CVPixelBuffer? {
        let attrs: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
        ]
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, Int(size.width), Int(size.height),
                            kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
        guard let pixelBuffer = pb else { return nil }
        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        let ctx = CGContext(data: CVPixelBufferGetBaseAddress(pixelBuffer),
                            width: Int(size.width), height: Int(size.height),
                            bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
                            space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
        guard let context = ctx, let cgImage = image.cgImage else {
            CVPixelBufferUnlockBaseAddress(pixelBuffer, [])
            return nil
        }
        // Normalize EXIF orientation
        let renderer = UIGraphicsImageRenderer(size: size)
        let normalized = renderer.image { _ in image.draw(in: CGRect(origin: .zero, size: size)) }
        if let normalizedCG = normalized.cgImage {
            context.draw(normalizedCG, in: CGRect(origin: .zero, size: size))
        } else {
            context.draw(cgImage, in: CGRect(origin: .zero, size: size))
        }
        CVPixelBufferUnlockBaseAddress(pixelBuffer, [])
        return pixelBuffer
    }
}
