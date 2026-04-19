import CoreML
import UIKit
import Accelerate

/// MoGe-2 ViT-B (504x504, FP16) wrapped in a small Swift driver.
///
/// The CoreML model takes a single ImageType input (`image`) and returns five
/// outputs: `points`, `depth`, `normal`, `mask`, `metric_scale`. We stretch
/// the input UIImage to a 504×504 square, run inference, and return:
///
///   - depth in metric meters (= raw_depth × metric_scale)
///   - per-pixel surface normals in [-1, 1]
///   - confidence mask in [0, 1]
///
/// The wrapper bakes `aspect_ratio = 1.0` into the UV grids, so stretching the
/// input is equivalent to what the converted graph assumes anyway; the caller
/// just applies the original image's aspect at display time to restore the
/// scene proportions so input and output views overlap pixel-for-pixel.
///
/// Visualization (turbo colormap, normal RGB) lives on the View side.
final class DepthEstimator: ObservableObject {
    static let inputSize = 504

    private var mlModel: MLModel?
    @Published var isReady = false

    struct Result {
        let depth: [Float]          // size*size, in meters
        let normal: [Float]         // size*size*3, last dim packed [nx, ny, nz]
        let mask: [Float]           // size*size, in [0, 1]
        let metricScale: Float
        let depthMin: Float
        let depthMax: Float
        let size: Int
    }

    init() {
        loadModel()
    }

    private func loadModel() {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self, let resourcePath = Bundle.main.resourcePath else { return }
            let fm = FileManager.default
            guard let items = try? fm.contentsOfDirectory(atPath: resourcePath) else { return }
            let config = MLModelConfiguration()
            // Pure ViT, runs comfortably on ANE at 504x504; .all picks the best path.
            config.computeUnits = .all
            for item in items where item.hasSuffix(".mlmodelc") && item.contains("MoGe2") {
                let url = URL(fileURLWithPath: (resourcePath as NSString).appendingPathComponent(item))
                if let model = try? MLModel(contentsOf: url, configuration: config) {
                    self.mlModel = model
                    DispatchQueue.main.async { self.isReady = true }
                    return
                }
            }
        }
    }

    // MARK: - Inference

    func estimate(image: UIImage) async throws -> Result {
        guard let model = mlModel else { throw EstimatorError.modelNotLoaded }
        let fixed = image.normalizedOrientation()
        guard let cgImage = fixed.cgImage else { throw EstimatorError.invalidImage }

        let resized = resizeCGImage(cgImage, targetSize: Self.inputSize)

        let pixelBuffer = try makeBGRAPixelBuffer(from: resized)
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(pixelBuffer: pixelBuffer)
        ])
        let output = try await model.prediction(from: input)

        guard
            let depthArr = output.featureValue(for: "depth")?.multiArrayValue,
            let normalArr = output.featureValue(for: "normal")?.multiArrayValue,
            let maskArr = output.featureValue(for: "mask")?.multiArrayValue,
            let scaleArr = output.featureValue(for: "metric_scale")?.multiArrayValue
        else {
            throw EstimatorError.predictionFailed
        }

        let metricScale = scaleArr[0].floatValue

        // depth: (1, H, W) — strides may not be C-contiguous on ANE.
        let depth = readMultiArray2D(depthArr, height: Self.inputSize, width: Self.inputSize)
        let mask = readMultiArray2D(maskArr, height: Self.inputSize, width: Self.inputSize)
        // normal: (1, H, W, 3)
        let normal = readMultiArray3D(normalArr, height: Self.inputSize, width: Self.inputSize, channels: 3)

        // Multiply by metric_scale for every pixel. Confident pixels (mask > 0.5)
        // set the dMin/dMax range so the colormap stays tight; low-confidence
        // regions (usually sky) keep their raw prediction and get clamped into
        // the same range by the visualizer, so they take the "far" end of turbo
        // instead of appearing as black bands that look like letterbox padding.
        var metricDepth = [Float](repeating: 0, count: depth.count)
        var dMin: Float = .greatestFiniteMagnitude
        var dMax: Float = 0
        for i in 0..<depth.count {
            let d = depth[i] * metricScale
            metricDepth[i] = d
            if mask[i] > 0.5 {
                if d < dMin { dMin = d }
                if d > dMax { dMax = d }
            }
        }
        if dMin == .greatestFiniteMagnitude {
            dMin = 0
            dMax = 1
        }

        return Result(
            depth: metricDepth,
            normal: normal,
            mask: mask,
            metricScale: metricScale,
            depthMin: dMin,
            depthMax: dMax,
            size: Self.inputSize
        )
    }

    // MARK: - Output reading (stride-aware)

    /// Read a (1, H, W) MLMultiArray into a row-major Float buffer.
    /// Uses vImageConvert row-by-row to handle ANE stride padding + FP16→FP32.
    private func readMultiArray2D(_ array: MLMultiArray, height: Int, width: Int) -> [Float] {
        let strides = array.strides.map { $0.intValue }
        let rowStride = strides[1]
        var result = [Float](repeating: 0, count: height * width)

        if array.dataType == .float16 {
            let src = array.dataPointer.assumingMemoryBound(to: UInt16.self)
            result.withUnsafeMutableBufferPointer { dst in
                for r in 0..<height {
                    var srcBuf = vImage_Buffer(
                        data: UnsafeMutableRawPointer(mutating: src + r * rowStride),
                        height: 1, width: vImagePixelCount(width), rowBytes: width * 2
                    )
                    var dstBuf = vImage_Buffer(
                        data: dst.baseAddress! + r * width,
                        height: 1, width: vImagePixelCount(width), rowBytes: width * 4
                    )
                    vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, 0)
                }
            }
        } else {
            let src = array.dataPointer.assumingMemoryBound(to: Float.self)
            result.withUnsafeMutableBufferPointer { dst in
                for r in 0..<height {
                    memcpy(dst.baseAddress! + r * width, src + r * rowStride, width * 4)
                }
            }
        }
        return result
    }

    /// Read a (1, H, W, C) MLMultiArray into a row-major Float buffer of size H*W*C.
    /// For FP16 with colStride == C and chStride == 1, converts each row in bulk.
    private func readMultiArray3D(_ array: MLMultiArray, height: Int, width: Int, channels: Int) -> [Float] {
        let strides = array.strides.map { $0.intValue }
        let rowStride = strides[1]
        let colStride = strides[2]
        let chStride = strides[3]
        let rowElements = width * channels
        var result = [Float](repeating: 0, count: height * width * channels)

        let interleaved = (colStride == channels && chStride == 1)

        if array.dataType == .float16 {
            let src = array.dataPointer.assumingMemoryBound(to: UInt16.self)
            if interleaved {
                // Fast path: each row is contiguous [w*C] FP16 values.
                result.withUnsafeMutableBufferPointer { dst in
                    for r in 0..<height {
                        var srcBuf = vImage_Buffer(
                            data: UnsafeMutableRawPointer(mutating: src + r * rowStride),
                            height: 1, width: vImagePixelCount(rowElements), rowBytes: rowElements * 2
                        )
                        var dstBuf = vImage_Buffer(
                            data: dst.baseAddress! + r * rowElements,
                            height: 1, width: vImagePixelCount(rowElements), rowBytes: rowElements * 4
                        )
                        vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, 0)
                    }
                }
            } else {
                // Fallback: per-element
                for r in 0..<height {
                    let baseR = r * rowStride
                    for c in 0..<width {
                        let baseC = baseR + c * colStride
                        let dst = (r * width + c) * channels
                        for ch in 0..<channels {
                            result[dst + ch] = Float(Float16(bitPattern: src[baseC + ch * chStride]))
                        }
                    }
                }
            }
        } else {
            let src = array.dataPointer.assumingMemoryBound(to: Float.self)
            if interleaved {
                result.withUnsafeMutableBufferPointer { dst in
                    for r in 0..<height {
                        memcpy(dst.baseAddress! + r * rowElements, src + r * rowStride, rowElements * 4)
                    }
                }
            } else {
                for r in 0..<height {
                    let baseR = r * rowStride
                    for c in 0..<width {
                        let baseC = baseR + c * colStride
                        let dst = (r * width + c) * channels
                        for ch in 0..<channels {
                            result[dst + ch] = src[baseC + ch * chStride]
                        }
                    }
                }
            }
        }
        return result
    }

    // MARK: - Image helpers

    /// Stretch-resize the image to `targetSize × targetSize`. The converted
    /// MoGe-2 graph hard-codes `aspect_ratio = 1.0` in its UV grids, so a plain
    /// resize matches what the model expects; the caller reapplies the original
    /// image's aspect ratio at display time so input and output views overlap.
    private func resizeCGImage(_ image: CGImage, targetSize: Int) -> CGImage {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(
            data: nil, width: targetSize, height: targetSize,
            bitsPerComponent: 8, bytesPerRow: targetSize * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: targetSize, height: targetSize))
        return ctx.makeImage()!
    }

    /// Build a kCVPixelFormatType_32BGRA CVPixelBuffer that the CoreML
    /// ImageType input accepts directly (the converter applies scale=1/255 +
    /// RGB layout for us, so we just have to deliver pixels).
    private func makeBGRAPixelBuffer(from image: CGImage) throws -> CVPixelBuffer {
        let w = image.width
        let h = image.height
        let attrs: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true,
        ]
        var pb: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, w, h,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary, &pb
        )
        guard status == kCVReturnSuccess, let buffer = pb else {
            throw EstimatorError.pixelBufferAllocFailed
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        let ctx = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: w, height: h,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                | CGBitmapInfo.byteOrder32Little.rawValue
        )
        ctx?.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        return buffer
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

enum EstimatorError: LocalizedError {
    case invalidImage, modelNotLoaded, predictionFailed, pixelBufferAllocFailed
    var errorDescription: String? {
        switch self {
        case .invalidImage: return "Invalid image"
        case .modelNotLoaded: return "Model not loaded"
        case .predictionFailed: return "Inference failed"
        case .pixelBufferAllocFailed: return "Pixel buffer allocation failed"
        }
    }
}
