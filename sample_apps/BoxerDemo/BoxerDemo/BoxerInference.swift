import CoreML
import UIKit
import ARKit
import simd

// MARK: - Detection Result

struct Detection3D: Identifiable {
    let id = UUID()
    let box2D: CGRect
    let center: SIMD3<Float>    // voxel frame (meters)
    let size: SIMD3<Float>      // H×W×D (meters)
    let yaw: Float              // radians
    let confidence: Float
    let distance: Float
    let worldTransform: simd_float4x4  // for SceneKit placement
}

// MARK: - Inference Engine

@MainActor
final class BoxerInference: ObservableObject {
    private var model: MLModel?
    @Published var isReady = false

    private let imageSize = 960
    private let patchSize = 16
    private let fH = 60
    private let fW = 60
    private let maxBoxes = 20

    init() { Task { await loadModel() } }

    private func loadModel() async {
        guard let resourcePath = Bundle.main.resourcePath else { return }
        let fm = FileManager.default
        guard let items = try? fm.contentsOfDirectory(atPath: resourcePath) else { return }
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU
        for item in items where item.hasSuffix(".mlmodelc") && item.contains("Boxer") {
            let url = URL(fileURLWithPath: (resourcePath as NSString).appendingPathComponent(item))
            model = try? MLModel(contentsOf: url, configuration: config)
            if model != nil { isReady = true; return }
        }
    }

    // MARK: - AR Inference

    func predict(frame: ARFrame, boxes2D: [[Float]]) async throws -> [Detection3D] {
        guard let model else { throw BoxerError.modelsNotLoaded }

        let camera = frame.camera
        let imgW = Int(camera.imageResolution.width)
        let imgH = Int(camera.imageResolution.height)

        // ARKit → OpenCV: flip Y and Z (ARKit: Y-up, -Z forward → OpenCV: Y-down, +Z forward)
        let flipYZ = simd_float4x4(columns: (
            simd_float4( 1,  0,  0, 0),
            simd_float4( 0, -1,  0, 0),
            simd_float4( 0,  0, -1, 0),
            simd_float4( 0,  0,  0, 1)
        ))
        let T_wc = camera.transform * flipYZ

        // Gravity alignment → T_world_voxel
        let T_wv = gravityAlign(T_wc: T_wc)
        let T_vc = T_wv.inverse * T_wc

        // Scale intrinsics to 960x960 (center-crop)
        let side = min(imgW, imgH)
        let scale = Float(imageSize) / Float(side)
        let intr = camera.intrinsics
        let fx = intr[0][0] * scale
        let fy = intr[1][1] * scale
        let cx = (intr[2][0] - Float(imgW - side) / 2) * scale
        let cy = (intr[2][1] - Float(imgH - side) / 2) * scale

        // Prepare inputs
        let imageArr = try prepareImage(frame.capturedImage)
        let sdpArr = try prepareDepth(frame.sceneDepth?.depthMap, imgW: imgW, imgH: imgH)
        let bbArr = try prepareBoxes(boxes2D, imgW: imgW, imgH: imgH)
        let rayArr = try prepareRays(fx: fx, fy: fy, cx: cx, cy: cy, T_vc: T_vc)

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(multiArray: imageArr),
            "sdp_patches": MLFeatureValue(multiArray: sdpArr),
            "bb2d": MLFeatureValue(multiArray: bbArr),
            "ray_encoding": MLFeatureValue(multiArray: rayArr),
        ])
        let output = try await model.prediction(from: input)

        return decodeOutput(output, boxes2D: boxes2D, T_wv: T_wv)
    }

    // MARK: - Photo Inference (approximate)

    func predict(image: UIImage, boxes2D: [[Float]]) async throws -> [Detection3D] {
        guard let model else { throw BoxerError.modelsNotLoaded }
        guard let cgImage = normalizeOrientation(image) else { throw BoxerError.invalidImage }
        let imgW = cgImage.width, imgH = cgImage.height

        guard let pb = resizeToPB(cgImage, w: imageSize, h: imageSize) else { throw BoxerError.preprocessFailed }
        let imageArr = try pbToMLMultiArray(pb)
        let sdpArr = try MLMultiArray(shape: [1, 1, 60, 60], dataType: .float32)
        let ptr16 = sdpArr.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<3600 { ptr16[i] = Float(-1) }

        let bbArr = try prepareBoxes(boxes2D, imgW: imgW, imgH: imgH)

        // Approximate rays with default pinhole, identity pose
        let fx: Float = 720, fy: Float = 720
        let cx = Float(imageSize) / 2, cy = Float(imageSize) / 2
        let rayArr = try prepareRays(fx: fx, fy: fy, cx: cx, cy: cy, T_vc: matrix_identity_float4x4)

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(multiArray: imageArr),
            "sdp_patches": MLFeatureValue(multiArray: sdpArr),
            "bb2d": MLFeatureValue(multiArray: bbArr),
            "ray_encoding": MLFeatureValue(multiArray: rayArr),
        ])
        let output = try await model.prediction(from: input)
        return decodeOutput(output, boxes2D: boxes2D, T_wv: matrix_identity_float4x4)
    }

    // MARK: - Input Preparation

    private func prepareImage(_ pb: CVPixelBuffer) throws -> MLMultiArray {
        // Center-crop to square, resize to 960x960
        let w = CVPixelBufferGetWidth(pb)
        let h = CVPixelBufferGetHeight(pb)
        let side = min(w, h)
        let ci = CIImage(cvPixelBuffer: pb)
        let cropRect = CGRect(x: (w - side) / 2, y: (h - side) / 2, width: side, height: side)
        let cropped = ci.cropped(to: cropRect).transformed(
            by: CGAffineTransform(translationX: -cropRect.origin.x, y: -cropRect.origin.y))
        let scaled = cropped.transformed(by: CGAffineTransform(
            scaleX: CGFloat(imageSize) / CGFloat(side),
            y: CGFloat(imageSize) / CGFloat(side)))
        var outPB: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, imageSize, imageSize, kCVPixelFormatType_32BGRA, nil, &outPB)
        guard let out = outPB else { throw BoxerError.preprocessFailed }
        CIContext().render(scaled, to: out)
        return try pbToMLMultiArray(out)
    }

    private func pbToMLMultiArray(_ pb: CVPixelBuffer) throws -> MLMultiArray {
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        let arr = try MLMultiArray(shape: [1, 3, NSNumber(value: h), NSNumber(value: w)], dataType: .float32)
        let dst = arr.dataPointer.assumingMemoryBound(to: Float.self)
        CVPixelBufferLockBaseAddress(pb, .readOnly); defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return arr }
        let src = base.assumingMemoryBound(to: UInt8.self)
        let bpr = CVPixelBufferGetBytesPerRow(pb); let hw = h * w
        for y in 0..<h { for x in 0..<w {
            let off = y * bpr + x * 4; let idx = y * w + x
            dst[0 * hw + idx] = Float(Float(src[off + 2]) / 255.0)  // R
            dst[1 * hw + idx] = Float(Float(src[off + 1]) / 255.0)  // G
            dst[2 * hw + idx] = Float(Float(src[off + 0]) / 255.0)  // B
        }}
        return arr
    }

    private func prepareDepth(_ depthMap: CVPixelBuffer?, imgW: Int, imgH: Int) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: [1, 1, 60, 60], dataType: .float32)
        let dst = arr.dataPointer.assumingMemoryBound(to: Float.self)
        guard let dm = depthMap else {
            for i in 0..<3600 { dst[i] = Float(-1) }; return arr
        }
        CVPixelBufferLockBaseAddress(dm, .readOnly); defer { CVPixelBufferUnlockBaseAddress(dm, .readOnly) }
        let dW = CVPixelBufferGetWidth(dm), dH = CVPixelBufferGetHeight(dm)
        let bpr = CVPixelBufferGetBytesPerRow(dm)
        guard let base = CVPixelBufferGetBaseAddress(dm) else {
            for i in 0..<3600 { dst[i] = Float(-1) }; return arr
        }
        let src = base.assumingMemoryBound(to: Float32.self)
        let side = min(imgW, imgH)
        let cropOffX = (imgW - side) / 2, cropOffY = (imgH - side) / 2
        let scale960 = Float(imageSize) / Float(side)
        let scaleDX = Float(dW) / Float(imgW), scaleDY = Float(dH) / Float(imgH)

        for py in 0..<fH { for px in 0..<fW {
            let u960 = Float(px * patchSize + patchSize / 2)
            let v960 = Float(py * patchSize + patchSize / 2)
            let uOrig = u960 / scale960 + Float(cropOffX)
            let vOrig = v960 / scale960 + Float(cropOffY)
            let dx = Int(uOrig * scaleDX), dy = Int(vOrig * scaleDY)
            if dx >= 0 && dx < dW && dy >= 0 && dy < dH {
                let d = src[dy * (bpr / 4) + dx]
                dst[py * fW + px] = d > 0 ? Float(d) : Float(-1)
            } else { dst[py * fW + px] = Float(-1) }
        }}
        return arr
    }

    private func prepareBoxes(_ boxes: [[Float]], imgW: Int, imgH: Int) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: [1, NSNumber(value: maxBoxes), 4], dataType: .float32)
        let dst = arr.dataPointer.assumingMemoryBound(to: Float.self)
        memset(dst, 0, maxBoxes * 4 * 2)
        let side = Float(min(imgW, imgH))
        let cropOffX = Float(imgW - Int(side)) / 2
        let cropOffY = Float(imgH - Int(side)) / 2
        let scale = Float(imageSize) / side
        for (i, box) in boxes.prefix(maxBoxes).enumerated() {
            let xmin = (box[0] - cropOffX) * scale
            let ymin = (box[1] - cropOffY) * scale
            let xmax = (box[2] - cropOffX) * scale
            let ymax = (box[3] - cropOffY) * scale
            // Boxer ordering: [xmin, xmax, ymin, ymax]
            dst[i * 4 + 0] = Float((xmin + 0.5) / Float(imageSize))
            dst[i * 4 + 1] = Float((xmax + 0.5) / Float(imageSize))
            dst[i * 4 + 2] = Float((ymin + 0.5) / Float(imageSize))
            dst[i * 4 + 3] = Float((ymax + 0.5) / Float(imageSize))
        }
        return arr
    }

    private func prepareRays(fx: Float, fy: Float, cx: Float, cy: Float, T_vc: simd_float4x4) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: [1, 3600, 6], dataType: .float32)
        let dst = arr.dataPointer.assumingMemoryBound(to: Float.self)
        let R_vc = simd_float3x3(
            simd_float3(T_vc.columns.0.x, T_vc.columns.0.y, T_vc.columns.0.z),
            simd_float3(T_vc.columns.1.x, T_vc.columns.1.y, T_vc.columns.1.z),
            simd_float3(T_vc.columns.2.x, T_vc.columns.2.y, T_vc.columns.2.z)
        )
        let originVoxel = simd_float3(T_vc.columns.3.x, T_vc.columns.3.y, T_vc.columns.3.z)
        for p in 0..<3600 {
            let row = p / fW, col = p % fW
            let u = Float(col * patchSize + patchSize / 2)
            let v = Float(row * patchSize + patchSize / 2)
            let dirCam = normalize(SIMD3<Float>((u - cx) / fx, (v - cy) / fy, 1.0))
            let dirVox = normalize(R_vc * dirCam)
            let moment = cross(originVoxel, dirVox)
            let off = p * 6
            dst[off+0] = Float(dirVox.x); dst[off+1] = Float(dirVox.y); dst[off+2] = Float(dirVox.z)
            dst[off+3] = Float(moment.x); dst[off+4] = Float(moment.y); dst[off+5] = Float(moment.z)
        }
        return arr
    }

    // MARK: - Gravity Alignment
    // Matches reference: https://github.com/Barath19/Boxer3D BoxerNet.swift gravityAlign()
    // and Python gravity_align_T_world_cam(T_wc, z_grav=True)

    private func gravityAlign(T_wc: simd_float4x4) -> simd_float4x4 {
        // Extract rotation columns and translation
        let col0 = SIMD3<Float>(T_wc.columns.0.x, T_wc.columns.0.y, T_wc.columns.0.z)
        let col1 = SIMD3<Float>(T_wc.columns.1.x, T_wc.columns.1.y, T_wc.columns.1.z)
        let col2 = SIMD3<Float>(T_wc.columns.2.x, T_wc.columns.2.y, T_wc.columns.2.z)
        let t_wc = SIMD3<Float>(T_wc.columns.3.x, T_wc.columns.3.y, T_wc.columns.3.z)

        // Gravity in Aria VIO convention = (0, 0, -1)
        let g = SIMD3<Float>(0, 0, -1)

        // Camera forward (column 2 of R_wc) projected orthogonal to gravity
        // reject(a, b) = a - b * dot(a, b) / dot(b, b)
        let camZ = col2  // R_wc column 2
        var d3 = camZ - g * dot(camZ, g)  // reject camZ from gravity
        if length(d3) < 1e-6 {
            d3 = SIMD3<Float>(0, 0.001, 0)
        }

        // d2 = cross(d3, gravity)
        let d2 = cross(d3, g)

        // Build R_wcg with columns [g, d2, d3], then normalize each column
        var R_wcg = simd_float3x3(columns: (
            normalize(g),
            normalize(d2),
            normalize(d3)
        ))

        // Apply R_cg_cgz to rotate X-gravity convention to Z-gravity convention
        // Matching reference exactly: R_cg_cgz defined by rows, then transposed for inverse
        let R_cg_cgz = simd_float3x3(rows: [
            SIMD3<Float>(0, -1, 0),
            SIMD3<Float>(0,  0, 1),
            SIMD3<Float>(-1, 0, 0)
        ])
        let R_world_cgz = R_wcg * R_cg_cgz.transpose

        var T_wv = matrix_identity_float4x4
        T_wv.columns.0 = simd_float4(R_world_cgz.columns.0, 0)
        T_wv.columns.1 = simd_float4(R_world_cgz.columns.1, 0)
        T_wv.columns.2 = simd_float4(R_world_cgz.columns.2, 0)
        T_wv.columns.3 = simd_float4(t_wc, 1)
        return T_wv
    }

    // MARK: - Output Decoding

    private func decodeOutput(_ output: MLFeatureProvider, boxes2D: [[Float]], T_wv: simd_float4x4) -> [Detection3D] {
        guard let centerArr = output.featureValue(for: "center")?.multiArrayValue,
              let sizeArr = output.featureValue(for: "size")?.multiArrayValue,
              let yawArr = output.featureValue(for: "yaw")?.multiArrayValue,
              let confArr = output.featureValue(for: "confidence")?.multiArrayValue else { return [] }

        print("[Boxer] center shape=\(centerArr.shape) size shape=\(sizeArr.shape) yaw shape=\(yawArr.shape) conf shape=\(confArr.shape)")

        let R_wv = simd_float3x3(
            simd_float3(T_wv.columns.0.x, T_wv.columns.0.y, T_wv.columns.0.z),
            simd_float3(T_wv.columns.1.x, T_wv.columns.1.y, T_wv.columns.1.z),
            simd_float3(T_wv.columns.2.x, T_wv.columns.2.y, T_wv.columns.2.z)
        )
        let t_wv = SIMD3<Float>(T_wv.columns.3.x, T_wv.columns.3.y, T_wv.columns.3.z)

        var results: [Detection3D] = []
        for i in 0..<min(boxes2D.count, maxBoxes) {
            let cStrides = centerArr.strides.map { $0.intValue }
            let sStrides = sizeArr.strides.map { $0.intValue }
            let cx = readF(centerArr, at: i * cStrides[1] + 0 * cStrides[2])
            let cy = readF(centerArr, at: i * cStrides[1] + 1 * cStrides[2])
            let cz = readF(centerArr, at: i * cStrides[1] + 2 * cStrides[2])
            let sx = readF(sizeArr, at: i * sStrides[1] + 0 * sStrides[2])
            let sy = readF(sizeArr, at: i * sStrides[1] + 1 * sStrides[2])
            let sz = readF(sizeArr, at: i * sStrides[1] + 2 * sStrides[2])
            let yaw = readF(yawArr, at: i * yawArr.strides[1].intValue)
            let conf = readF(confArr, at: i * confArr.strides[1].intValue)

            print("[Boxer] box\(i): center=(\(cx),\(cy),\(cz)) size=(\(sx),\(sy),\(sz)) yaw=\(yaw) conf=\(conf)")

            let centerVoxel = SIMD3<Float>(cx, cy, cz)
            let centerWorld = R_wv * centerVoxel + t_wv

            // World rotation = R_wv * rotationZ(yaw)
            let cosY = cos(yaw), sinY = sin(yaw)
            let R_yaw = simd_float3x3(columns: (
                SIMD3<Float>(cosY, sinY, 0),
                SIMD3<Float>(-sinY, cosY, 0),
                SIMD3<Float>(0, 0, 1)
            ))
            let R_world = R_wv * R_yaw

            // Build world transform in OpenCV convention
            var worldTransformCV = matrix_identity_float4x4
            worldTransformCV.columns.0 = simd_float4(R_world.columns.0, 0)
            worldTransformCV.columns.1 = simd_float4(R_world.columns.1, 0)
            worldTransformCV.columns.2 = simd_float4(R_world.columns.2, 0)
            worldTransformCV.columns.3 = simd_float4(centerWorld, 1)

            // Convert back to ARKit/SceneKit (OpenGL) convention: flip Y and Z
            let flipYZ = simd_float4x4(columns: (
                simd_float4( 1,  0,  0, 0),
                simd_float4( 0, -1,  0, 0),
                simd_float4( 0,  0, -1, 0),
                simd_float4( 0,  0,  0, 1)
            ))
            let worldTransform = worldTransformCV * flipYZ

            let box2D = CGRect(x: CGFloat(boxes2D[i][0]), y: CGFloat(boxes2D[i][1]),
                               width: CGFloat(boxes2D[i][2] - boxes2D[i][0]),
                               height: CGFloat(boxes2D[i][3] - boxes2D[i][1]))

            results.append(Detection3D(
                box2D: box2D, center: centerVoxel, size: SIMD3(sx, sy, sz),
                yaw: yaw, confidence: conf, distance: length(centerVoxel), worldTransform: worldTransform
            ))
        }
        return results
    }

    private func readF(_ arr: MLMultiArray, at idx: Int) -> Float {
        switch arr.dataType {
        case .float16: return Float(arr.dataPointer.assumingMemoryBound(to: Float16.self)[idx])
        case .float32: return arr.dataPointer.assumingMemoryBound(to: Float.self)[idx]
        default: return arr[idx].floatValue
        }
    }

    // MARK: - Helpers

    private func normalizeOrientation(_ image: UIImage) -> CGImage? {
        guard let cg = image.cgImage else { return nil }
        if image.imageOrientation == .up { return cg }
        UIGraphicsBeginImageContextWithOptions(image.size, false, 1.0)
        image.draw(at: .zero)
        let n = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return n?.cgImage
    }

    private func resizeToPB(_ cgImage: CGImage, w: Int, h: Int) -> CVPixelBuffer? {
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, w, h, kCVPixelFormatType_32BGRA, nil, &pb)
        guard let out = pb else { return nil }
        CVPixelBufferLockBaseAddress(out, [])
        let ctx = CGContext(data: CVPixelBufferGetBaseAddress(out), width: w, height: h, bitsPerComponent: 8,
                            bytesPerRow: CVPixelBufferGetBytesPerRow(out), space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
        ctx?.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))
        CVPixelBufferUnlockBaseAddress(out, [])
        return out
    }
}

enum BoxerError: LocalizedError {
    case modelsNotLoaded, invalidImage, preprocessFailed
    var errorDescription: String? {
        switch self {
        case .modelsNotLoaded: return "Model not loaded"
        case .invalidImage: return "Invalid image"
        case .preprocessFailed: return "Preprocessing failed"
        }
    }
}
