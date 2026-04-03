import CoreML
import Vision
import UIKit

struct FaceEmbedding: Identifiable, Codable {
    let id: UUID
    let name: String
    let embedding: [Float]
    let date: Date
    let thumbnailData: Data?

    init(name: String, embedding: [Float], thumbnail: UIImage?) {
        self.id = UUID()
        self.name = name
        self.embedding = embedding
        self.date = Date()
        self.thumbnailData = thumbnail?.jpegData(compressionQuality: 0.6)
    }

    var thumbnail: UIImage? {
        guard let data = thumbnailData else { return nil }
        return UIImage(data: data)
    }
}

struct RecognitionResult {
    let faceRect: CGRect
    let embedding: [Float]
    let match: FaceEmbedding?
    let similarity: Float
}

class FaceRecognizer: ObservableObject {
    private var mlModel: MLModel?
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    @Published var isReady = false
    @Published var registered: [FaceEmbedding] = []

    // Standard 5-point alignment template for 112x112 (from insightface/ArcFace)
    private static let alignDst: [(CGFloat, CGFloat)] = [
        (38.2946, 51.6963),  // left eye
        (73.5318, 51.5014),  // right eye
        (56.0252, 71.7366),  // nose tip
        (41.5493, 92.3655),  // left mouth
        (70.7299, 92.2041),  // right mouth
    ]

    private static let storeKey = "registered_faces"

    init() {
        loadModel()
        loadRegistered()
    }

    private func loadModel() {
        guard let resourcePath = Bundle.main.resourcePath else { return }
        let fm = FileManager.default
        guard let items = try? fm.contentsOfDirectory(atPath: resourcePath) else { return }
        for item in items where item.hasSuffix(".mlmodelc") {
            let url = URL(fileURLWithPath: (resourcePath as NSString).appendingPathComponent(item))
            let config = MLModelConfiguration()
            config.computeUnits = .all
            if let model = try? MLModel(contentsOf: url, configuration: config) {
                self.mlModel = model
                DispatchQueue.main.async { self.isReady = true }
                return
            }
        }
    }

    // MARK: - Registration

    func register(name: String, image: UIImage) async -> Bool {
        let fixed = image.normalizedOrientation()
        guard let cgImage = fixed.cgImage else { return false }
        let results = await detectAndAlign(cgImage: cgImage)
        guard let first = results.first else { return false }
        guard let emb = predict(pixelBuffer: first.alignedBuffer) else { return false }

        let thumb = createThumbnail(from: fixed, faceRect: first.rect)
        let face = FaceEmbedding(name: name, embedding: emb, thumbnail: thumb)
        await MainActor.run {
            registered.append(face)
            saveRegistered()
        }
        return true
    }

    private func createThumbnail(from image: UIImage, faceRect: CGRect) -> UIImage? {
        guard let cgImage = image.cgImage else { return nil }
        let w = CGFloat(cgImage.width)
        let h = CGFloat(cgImage.height)
        let margin: CGFloat = 0.3
        let fx = faceRect.origin.x * w
        let fy = (1.0 - faceRect.origin.y - faceRect.height) * h
        let fw = faceRect.width * w
        let fh = faceRect.height * h
        let size = max(fw, fh) * (1 + margin)
        let cx = fx + fw / 2
        let cy = fy + fh / 2
        let cropRect = CGRect(x: cx - size / 2, y: cy - size / 2, width: size, height: size)
            .intersection(CGRect(x: 0, y: 0, width: w, height: h))
        guard let cropped = cgImage.cropping(to: cropRect) else { return nil }
        let thumbSize: CGFloat = 100
        UIGraphicsBeginImageContextWithOptions(CGSize(width: thumbSize, height: thumbSize), false, 1)
        UIImage(cgImage: cropped).draw(in: CGRect(x: 0, y: 0, width: thumbSize, height: thumbSize))
        let thumb = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return thumb
    }

    func deleteRegistered(at offsets: IndexSet) {
        registered.remove(atOffsets: offsets)
        saveRegistered()
    }

    private func saveRegistered() {
        if let data = try? JSONEncoder().encode(registered) {
            UserDefaults.standard.set(data, forKey: Self.storeKey)
        }
    }

    private func loadRegistered() {
        if let data = UserDefaults.standard.data(forKey: Self.storeKey),
           let faces = try? JSONDecoder().decode([FaceEmbedding].self, from: data) {
            registered = faces
        }
    }

    // MARK: - Embedding extraction (for Compare tab)

    func extractEmbedding(image: UIImage) async -> (embedding: [Float], thumbnail: UIImage?)? {
        let fixed = image.normalizedOrientation()
        guard let cgImage = fixed.cgImage else { return nil }
        let results = await detectAndAlign(cgImage: cgImage)
        guard let first = results.first else { return nil }
        guard let emb = predict(pixelBuffer: first.alignedBuffer) else { return nil }
        let thumb = createThumbnail(from: fixed, faceRect: first.rect)
        return (emb, thumb)
    }

    func similarity(_ a: [Float], _ b: [Float]) -> Float {
        cosineSimilarity(a, b)
    }

    // MARK: - Recognition (Camera)

    func recognize(pixelBuffer: CVPixelBuffer) -> [RecognitionResult] {
        let faces = detectAndAlignSync(pixelBuffer: pixelBuffer)
        return faces.compactMap { face in
            guard let emb = predict(pixelBuffer: face.alignedBuffer) else { return nil }
            let (match, sim) = findBestMatch(for: emb)
            return RecognitionResult(faceRect: face.rect, embedding: emb, match: match, similarity: sim)
        }
    }

    // MARK: - Face detection + alignment

    private struct AlignedFace {
        let rect: CGRect
        let alignedBuffer: CVPixelBuffer
    }

    private func detectAndAlign(cgImage: CGImage) async -> [AlignedFace] {
        let observations = await detectFaceLandmarks(cgImage: cgImage)
        let w = CGFloat(cgImage.width)
        let h = CGFloat(cgImage.height)
        let ciImage = CIImage(cgImage: cgImage)
        return observations.compactMap { obs in
            guard let lm = obs.landmarks else { return nil }
            guard let aligned = alignFace(ciImage: ciImage, observation: obs, landmarks: lm,
                                           imageWidth: w, imageHeight: h) else { return nil }
            return AlignedFace(rect: obs.boundingBox, alignedBuffer: aligned)
        }
    }

    private func detectAndAlignSync(pixelBuffer: CVPixelBuffer) -> [AlignedFace] {
        let observations = detectFaceLandmarksSync(pixelBuffer: pixelBuffer)
        let w = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let h = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        return observations.compactMap { obs in
            guard let lm = obs.landmarks else { return nil }
            guard let aligned = alignFace(ciImage: ciImage, observation: obs, landmarks: lm,
                                           imageWidth: w, imageHeight: h) else { return nil }
            return AlignedFace(rect: obs.boundingBox, alignedBuffer: aligned)
        }
    }

    private func detectFaceLandmarks(cgImage: CGImage) async -> [VNFaceObservation] {
        await withCheckedContinuation { cont in
            let req = VNDetectFaceLandmarksRequest { request, _ in
                cont.resume(returning: (request.results as? [VNFaceObservation]) ?? [])
            }
            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            try? handler.perform([req])
        }
    }

    private func detectFaceLandmarksSync(pixelBuffer: CVPixelBuffer) -> [VNFaceObservation] {
        var results: [VNFaceObservation] = []
        let req = VNDetectFaceLandmarksRequest { request, _ in
            results = (request.results as? [VNFaceObservation]) ?? []
        }
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try? handler.perform([req])
        return results
    }

    // MARK: - Face alignment via affine transform

    private func alignFace(ciImage: CIImage, observation: VNFaceObservation,
                            landmarks: VNFaceLandmarks2D,
                            imageWidth: CGFloat, imageHeight: CGFloat) -> CVPixelBuffer? {
        // Extract 5 key points in pixel coordinates
        guard let srcPoints = extract5Points(observation: observation, landmarks: landmarks,
                                              imageWidth: imageWidth, imageHeight: imageHeight) else { return nil }

        let dstPoints = Self.alignDst

        // Compute affine transform from src → dst (112x112)
        guard let transform = estimateAffineTransform(src: srcPoints, dst: dstPoints) else { return nil }

        // Apply transform using CIImage
        let ciTransform = CGAffineTransform(a: transform[0], b: transform[3],
                                             c: transform[1], d: transform[4],
                                             tx: transform[2], ty: transform[5])
        let transformed = ciImage.transformed(by: ciTransform)

        // Render to 112x112 pixel buffer
        var out: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, 112, 112, kCVPixelFormatType_32BGRA, nil, &out)
        guard let outBuffer = out else { return nil }
        ciContext.render(transformed, to: outBuffer,
                         bounds: CGRect(x: 0, y: 0, width: 112, height: 112),
                         colorSpace: CGColorSpaceCreateDeviceRGB())
        return outBuffer
    }

    private func extract5Points(observation: VNFaceObservation, landmarks: VNFaceLandmarks2D,
                                 imageWidth: CGFloat, imageHeight: CGFloat) -> [(CGFloat, CGFloat)]? {
        // Get landmark regions
        guard let leftEye = landmarks.leftEye,
              let rightEye = landmarks.rightEye,
              let nose = landmarks.noseCrest,
              let outerLips = landmarks.outerLips else { return nil }

        let bbox = observation.boundingBox

        // Helper: convert normalized landmark point to pixel coordinates
        // VNFaceLandmarks points are normalized relative to the face bounding box
        // Bounding box is in Vision coords (origin bottom-left)
        func toPixel(_ pts: [CGPoint]) -> CGPoint {
            let avg = pts.reduce(CGPoint.zero) { CGPoint(x: $0.x + $1.x, y: $0.y + $1.y) }
            let n = CGFloat(pts.count)
            let normX = avg.x / n
            let normY = avg.y / n
            // Convert from face-relative normalized to image pixel coords
            // CIImage uses bottom-left origin (same as Vision)
            let px = (bbox.origin.x + normX * bbox.width) * imageWidth
            let py = (bbox.origin.y + normY * bbox.height) * imageHeight
            return CGPoint(x: px, y: py)
        }

        let leftEyeCenter = toPixel(leftEye.normalizedPoints.map { $0 })
        let rightEyeCenter = toPixel(rightEye.normalizedPoints.map { $0 })
        let noseTip = toPixel([nose.normalizedPoints.last ?? nose.normalizedPoints[nose.pointCount / 2]])

        // Mouth corners: first and last points of outer lips
        let outerPts = outerLips.normalizedPoints.map { $0 }
        let leftMouth: CGPoint
        let rightMouth: CGPoint
        if outerPts.count >= 2 {
            leftMouth = toPixel([outerPts[0]])
            let midIdx = outerPts.count / 2
            rightMouth = toPixel([outerPts[midIdx]])
        } else {
            return nil
        }

        return [
            (leftEyeCenter.x, leftEyeCenter.y),
            (rightEyeCenter.x, rightEyeCenter.y),
            (noseTip.x, noseTip.y),
            (leftMouth.x, leftMouth.y),
            (rightMouth.x, rightMouth.y),
        ]
    }

    // Estimate affine transform: dst = M * src
    // Uses least-squares on first 3 point pairs (sufficient for affine)
    private func estimateAffineTransform(src: [(CGFloat, CGFloat)],
                                          dst: [(CGFloat, CGFloat)]) -> [CGFloat]? {
        guard src.count >= 3, dst.count >= 3 else { return nil }

        // Solve for 2x3 affine matrix using first 3 points
        // [dst_x] = [a b c] [src_x]
        // [dst_y]   [d e f] [src_y]
        //                   [  1  ]
        let s = src
        let d = dst

        // Build 6x6 system using all points for better accuracy
        var A = [[Double]](repeating: [Double](repeating: 0, count: 6), count: min(src.count, 5) * 2)
        var b = [Double](repeating: 0, count: min(src.count, 5) * 2)
        let n = min(src.count, 5)

        for i in 0..<n {
            let row0 = i * 2
            let row1 = i * 2 + 1
            A[row0] = [Double(s[i].0), Double(s[i].1), 1, 0, 0, 0]
            A[row1] = [0, 0, 0, Double(s[i].0), Double(s[i].1), 1]
            b[row0] = Double(d[i].0)
            b[row1] = Double(d[i].1)
        }

        // Solve using normal equations: x = (A^T A)^-1 A^T b
        guard let result = solveLinearSystem(A: A, b: b) else { return nil }
        return result.map { CGFloat($0) }
    }

    private func solveLinearSystem(A: [[Double]], b: [Double]) -> [Double]? {
        let m = A.count
        let n = 6

        // Compute A^T * A (6x6)
        var ATA = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
        var ATb = [Double](repeating: 0, count: n)

        for i in 0..<n {
            for j in 0..<n {
                for k in 0..<m {
                    ATA[i][j] += A[k][i] * A[k][j]
                }
            }
            for k in 0..<m {
                ATb[i] += A[k][i] * b[k]
            }
        }

        // Gaussian elimination with partial pivoting
        var aug = ATA
        var rhs = ATb
        for col in 0..<n {
            // Pivot
            var maxVal = abs(aug[col][col])
            var maxRow = col
            for row in (col + 1)..<n {
                if abs(aug[row][col]) > maxVal {
                    maxVal = abs(aug[row][col])
                    maxRow = row
                }
            }
            if maxVal < 1e-12 { return nil }
            if maxRow != col {
                aug.swapAt(col, maxRow)
                rhs.swapAt(col, maxRow)
            }
            // Eliminate
            for row in (col + 1)..<n {
                let factor = aug[row][col] / aug[col][col]
                for j in col..<n { aug[row][j] -= factor * aug[col][j] }
                rhs[row] -= factor * rhs[col]
            }
        }
        // Back substitution
        var x = [Double](repeating: 0, count: n)
        for i in stride(from: n - 1, through: 0, by: -1) {
            var sum = rhs[i]
            for j in (i + 1)..<n { sum -= aug[i][j] * x[j] }
            x[i] = sum / aug[i][i]
        }
        return x
    }

    // MARK: - Model prediction

    private func predict(pixelBuffer: CVPixelBuffer) -> [Float]? {
        guard let model = mlModel else { return nil }
        guard let input = try? MLDictionaryFeatureProvider(
            dictionary: ["face_image": MLFeatureValue(pixelBuffer: pixelBuffer)]
        ) else { return nil }
        guard let output = try? model.prediction(from: input) else { return nil }
        guard let multiArray = output.featureValue(for: "embedding")?.multiArrayValue else { return nil }

        let count = multiArray.count
        guard count == 512 else { return nil }
        // Use safe accessor instead of raw pointer cast
        return (0..<count).map { multiArray[$0].floatValue }
    }

    // MARK: - Matching

    private func findBestMatch(for embedding: [Float]) -> (FaceEmbedding?, Float) {
        var bestMatch: FaceEmbedding?
        var bestSim: Float = -1
        for face in registered {
            let sim = cosineSimilarity(embedding, face.embedding)
            if sim > bestSim {
                bestSim = sim
                bestMatch = face
            }
        }
        return (bestSim > 0.3 ? bestMatch : nil, bestSim)
    }

    private func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return 0 }
        var dot: Float = 0
        var normA: Float = 0
        var normB: Float = 0
        for i in 0..<a.count {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        let denom = sqrt(normA) * sqrt(normB)
        return denom > 0 ? dot / denom : 0
    }
}

extension UIImage {
    func normalizedOrientation() -> UIImage {
        guard imageOrientation != .up else { return self }
        UIGraphicsBeginImageContextWithOptions(size, false, scale)
        draw(in: CGRect(origin: .zero, size: size))
        let normalized = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return normalized ?? self
    }
}
