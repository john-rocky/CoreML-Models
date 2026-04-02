import CoreML
import Vision
import UIKit

struct FacePoseResult {
    let faceRect: CGRect
    let yaw: Float
    let pitch: Float
    let roll: Float
    let rotationMatrix: [[Float]]
    let expressionParams: [Float]
}

class FaceReconstructor: ObservableObject {
    private var mlModel: MLModel?
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    @Published var isReady = false

    init() {
        loadModel()
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

    // MARK: - Photo (async)

    func detect(image: UIImage) async -> [FacePoseResult] {
        guard let cgImage = image.cgImage, mlModel != nil else { return [] }
        let faceRects = await detectFaceRects(in: cgImage)
        return faceRects.compactMap { runModel(cgImage: cgImage, faceRect: $0) }
    }

    private func detectFaceRects(in cgImage: CGImage) async -> [CGRect] {
        await withCheckedContinuation { cont in
            let req = VNDetectFaceRectanglesRequest { request, _ in
                let faces = (request.results as? [VNFaceObservation]) ?? []
                cont.resume(returning: faces.map { $0.boundingBox })
            }
            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            try? handler.perform([req])
        }
    }

    // MARK: - Camera (sync)

    func detect(pixelBuffer: CVPixelBuffer) -> [FacePoseResult] {
        guard mlModel != nil else { return [] }
        let faceRects = detectFaceRectsSync(in: pixelBuffer)
        return faceRects.compactMap { runModel(pixelBuffer: pixelBuffer, faceRect: $0) }
    }

    private func detectFaceRectsSync(in pixelBuffer: CVPixelBuffer) -> [CGRect] {
        var rects: [CGRect] = []
        let req = VNDetectFaceRectanglesRequest { request, _ in
            let faces = (request.results as? [VNFaceObservation]) ?? []
            rects = faces.map { $0.boundingBox }
        }
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try? handler.perform([req])
        return rects
    }

    // MARK: - Model inference (Photo: CGImage path)

    // 3DDFA_V2 crop: square box, 1.58x expansion, center shifted up to include forehead
    private func roiBox(faceRect: CGRect, imageWidth: CGFloat, imageHeight: CGFloat, flipY: Bool) -> CGRect {
        let left = faceRect.origin.x * imageWidth
        let right = (faceRect.origin.x + faceRect.width) * imageWidth
        let top: CGFloat
        let bottom: CGFloat
        if flipY {
            // CGImage: Y=0 at top
            top = (1.0 - faceRect.origin.y - faceRect.height) * imageHeight
            bottom = (1.0 - faceRect.origin.y) * imageHeight
        } else {
            // CIImage: Y=0 at bottom → top has larger Y
            top = faceRect.origin.y * imageHeight
            bottom = (faceRect.origin.y + faceRect.height) * imageHeight
        }

        let oldSize = ((right - left) + (bottom - top)) / 2.0
        let centerX = (left + right) / 2.0
        // Shift center toward forehead (upward in image)
        let centerY: CGFloat
        if flipY {
            centerY = (top + bottom) / 2.0 - oldSize * 0.14
        } else {
            centerY = (top + bottom) / 2.0 + oldSize * 0.14
        }
        let size = oldSize * 1.58

        return CGRect(x: centerX - size / 2, y: centerY - size / 2, width: size, height: size)
            .intersection(CGRect(x: 0, y: 0, width: imageWidth, height: imageHeight))
    }

    private func runModel(cgImage: CGImage, faceRect: CGRect) -> FacePoseResult? {
        let w = CGFloat(cgImage.width)
        let h = CGFloat(cgImage.height)
        let cropRect = roiBox(faceRect: faceRect, imageWidth: w, imageHeight: h, flipY: true)

        guard cropRect.width > 0, cropRect.height > 0 else { return nil }
        guard let cropped = cgImage.cropping(to: cropRect) else { return nil }
        guard let pb = pixelBuffer(from: cropped, width: 120, height: 120) else { return nil }
        return predict(pixelBuffer: pb, faceRect: faceRect)
    }

    // MARK: - Model inference (Camera: CIImage path)

    private func runModel(pixelBuffer: CVPixelBuffer, faceRect: CGRect) -> FacePoseResult? {
        let w = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let h = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        let cropRect = roiBox(faceRect: faceRect, imageWidth: w, imageHeight: h, flipY: false)

        guard cropRect.width > 0, cropRect.height > 0 else { return nil }

        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            .cropped(to: cropRect)
            .transformed(by: CGAffineTransform(
                translationX: -cropRect.origin.x,
                y: -cropRect.origin.y
            ))

        let sx = 120.0 / cropRect.width
        let sy = 120.0 / cropRect.height
        let scaled = ciImage.transformed(by: CGAffineTransform(scaleX: sx, y: sy))

        var out: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, 120, 120, kCVPixelFormatType_32BGRA, nil, &out)
        guard let outBuffer = out else { return nil }
        ciContext.render(scaled, to: outBuffer, bounds: CGRect(x: 0, y: 0, width: 120, height: 120), colorSpace: CGColorSpaceCreateDeviceRGB())

        return predict(pixelBuffer: outBuffer, faceRect: faceRect)
    }

    // MARK: - Prediction

    private func predict(pixelBuffer: CVPixelBuffer, faceRect: CGRect) -> FacePoseResult? {
        guard let model = mlModel else { return nil }
        guard let input = try? MLDictionaryFeatureProvider(
            dictionary: ["face_image": MLFeatureValue(pixelBuffer: pixelBuffer)]
        ) else { return nil }
        guard let output = try? model.prediction(from: input) else { return nil }
        guard let multiArray = output.featureValue(for: "var_543")?.multiArrayValue else { return nil }

        let count = multiArray.count
        guard count == 62 else { return nil }
        let ptr = multiArray.dataPointer.bindMemory(to: Float16.self, capacity: count)
        var params = [Float](repeating: 0, count: count)
        for i in 0..<count {
            params[i] = Float(ptr[i])
        }

        let pose = Array(params[0..<12])
        let expression = Array(params[52..<62])

        // Pose is a flattened 3x4 matrix: [R | t]
        // Extract 3x3 rotation, normalize to remove scale
        var R: [[Float]] = [
            [pose[0], pose[1], pose[2]],
            [pose[4], pose[5], pose[6]],
            [pose[8], pose[9], pose[10]]
        ]

        // Remove scale from rotation matrix
        for i in 0..<3 {
            let norm = sqrt(R[i][0] * R[i][0] + R[i][1] * R[i][1] + R[i][2] * R[i][2])
            if norm > 1e-6 {
                R[i][0] /= norm
                R[i][1] /= norm
                R[i][2] /= norm
            }
        }

        let (yaw, pitch, roll) = eulerAngles(from: R)

        return FacePoseResult(
            faceRect: faceRect,
            yaw: yaw,
            pitch: pitch,
            roll: roll,
            rotationMatrix: R,
            expressionParams: expression
        )
    }

    // MARK: - Euler angles

    private func eulerAngles(from R: [[Float]]) -> (yaw: Float, pitch: Float, roll: Float) {
        let sy = sqrt(R[0][0] * R[0][0] + R[1][0] * R[1][0])
        let singular = sy < 1e-6

        let pitch: Float
        let yaw: Float
        let roll: Float

        if !singular {
            pitch = atan2(R[2][1], R[2][2])
            yaw = atan2(-R[2][0], sy)
            roll = atan2(R[1][0], R[0][0])
        } else {
            pitch = atan2(-R[1][2], R[1][1])
            yaw = atan2(-R[2][0], sy)
            roll = 0
        }

        let toDeg: Float = 180.0 / .pi
        return (yaw * toDeg, -pitch * toDeg, -roll * toDeg)
    }

    // MARK: - Photo pixel buffer helper

    private func pixelBuffer(from cgImage: CGImage, width: Int, height: Int) -> CVPixelBuffer? {
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ]
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
        guard let pixelBuffer = pb else { return nil }
        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        let ctx = CGContext(
            data: CVPixelBufferGetBaseAddress(pixelBuffer),
            width: width, height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        )
        ctx?.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        CVPixelBufferUnlockBaseAddress(pixelBuffer, [])
        return pixelBuffer
    }
}
