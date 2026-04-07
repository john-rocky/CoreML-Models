import CoreML
import UIKit

enum AnomalyDetectorError: LocalizedError {
    case modelNotFound
    case invalidImage
    case predictionFailed

    var errorDescription: String? {
        switch self {
        case .modelNotFound: return "EfficientAD model not found"
        case .invalidImage: return "Failed to process image"
        case .predictionFailed: return "Prediction failed"
        }
    }
}

struct AnomalyResult {
    let heatmapOverlay: UIImage
    let score: Float
    let processingTime: Double
}

enum AnomalyDetector {

    static func detect(in image: UIImage) async throws -> AnomalyResult {
        let start = CFAbsoluteTimeGetCurrent()
        let fixed = image.normalizedOrientation()
        guard let cgImage = fixed.cgImage else { throw AnomalyDetectorError.invalidImage }
        let origW = cgImage.width
        let origH = cgImage.height

        guard let pixelBuffer = createPixelBuffer(from: cgImage, width: 256, height: 256) else {
            throw AnomalyDetectorError.invalidImage
        }

        let (anomalyMap, score): ([Float], Float) = try await {
            let model = try loadModel()
            let input = try MLDictionaryFeatureProvider(dictionary: ["image": pixelBuffer])
            let output = try await model.prediction(from: input)
            guard let mapArray = output.featureValue(for: "anomaly_map")?.multiArrayValue,
                  let scoreArray = output.featureValue(for: "anomaly_score")?.multiArrayValue else {
                throw AnomalyDetectorError.predictionFailed
            }
            return (extractFloats(mapArray), extractScalar(scoreArray))
        }()

        let overlay = createHeatmapOverlay(anomalyMap, modelSize: 256,
                                           originalImage: cgImage,
                                           width: origW, height: origH)

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        return AnomalyResult(heatmapOverlay: overlay, score: score, processingTime: elapsed)
    }

    // MARK: - Model

    private static func loadModel() throws -> MLModel {
        guard let resourcePath = Bundle.main.resourcePath,
              let items = try? FileManager.default.contentsOfDirectory(atPath: resourcePath) else {
            throw AnomalyDetectorError.modelNotFound
        }
        for item in items where item.hasSuffix(".mlmodelc") && item.contains("EfficientAD") {
            let url = URL(fileURLWithPath: (resourcePath as NSString).appendingPathComponent(item))
            let config = MLModelConfiguration()
            config.computeUnits = .all
            return try MLModel(contentsOf: url, configuration: config)
        }
        throw AnomalyDetectorError.modelNotFound
    }

    // MARK: - Heatmap Visualization

    private static func createHeatmapOverlay(_ map: [Float], modelSize: Int,
                                              originalImage: CGImage,
                                              width: Int, height: Int) -> UIImage {
        // Draw original image
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(data: &pixels, width: width, height: height,
                            bitsPerComponent: 8, bytesPerRow: width * 4,
                            space: colorSpace,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.draw(originalImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        // Overlay heatmap with bilinear interpolation
        for y in 0..<height {
            let srcY = Float(y) * Float(modelSize) / Float(height)
            let y0 = min(Int(srcY), modelSize - 1)
            let y1 = min(y0 + 1, modelSize - 1)
            let fy = srcY - Float(y0)
            for x in 0..<width {
                let srcX = Float(x) * Float(modelSize) / Float(width)
                let x0 = min(Int(srcX), modelSize - 1)
                let x1 = min(x0 + 1, modelSize - 1)
                let fx = srcX - Float(x0)

                let v00 = map[y0 * modelSize + x0]
                let v10 = map[y0 * modelSize + x1]
                let v01 = map[y1 * modelSize + x0]
                let v11 = map[y1 * modelSize + x1]

                let value = v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) +
                            v01 * (1 - fx) * fy + v11 * fx * fy

                let idx = (y * width + x) * 4
                let (hr, hg, hb) = heatmapColor(value)

                // Blend: alpha based on anomaly intensity
                let alpha = min(value * 1.5, 1.0)
                let origR = Float(pixels[idx])
                let origG = Float(pixels[idx + 1])
                let origB = Float(pixels[idx + 2])

                pixels[idx]     = UInt8(clamping: Int(origR * (1 - alpha) + hr * 255 * alpha))
                pixels[idx + 1] = UInt8(clamping: Int(origG * (1 - alpha) + hg * 255 * alpha))
                pixels[idx + 2] = UInt8(clamping: Int(origB * (1 - alpha) + hb * 255 * alpha))
                pixels[idx + 3] = 255
            }
        }

        let outCtx = CGContext(data: &pixels, width: width, height: height,
                               bitsPerComponent: 8, bytesPerRow: width * 4,
                               space: colorSpace,
                               bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        let outImage = outCtx.makeImage()!
        return UIImage(cgImage: outImage)
    }

    /// Red heatmap: transparent (low) -> red (high)
    private static func heatmapColor(_ value: Float) -> (Float, Float, Float) {
        return (1.0, 0.0, 0.0)
    }

    // MARK: - Utilities

    private static func extractFloats(_ array: MLMultiArray) -> [Float] {
        let count = array.count
        if array.dataType == .float16 {
            let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)
            return (0..<count).map { Float(ptr[$0]) }
        } else {
            let ptr = array.dataPointer.assumingMemoryBound(to: Float32.self)
            return (0..<count).map { ptr[$0] }
        }
    }

    private static func extractScalar(_ array: MLMultiArray) -> Float {
        if array.dataType == .float16 {
            return Float(array.dataPointer.assumingMemoryBound(to: Float16.self)[0])
        } else {
            return array.dataPointer.assumingMemoryBound(to: Float32.self)[0]
        }
    }

    private static func createPixelBuffer(from cgImage: CGImage, width: Int, height: Int) -> CVPixelBuffer? {
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
                           [kCVPixelBufferCGImageCompatibilityKey: true,
                            kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary, &pb)
        guard let buffer = pb else { return nil }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let ctx = CGContext(data: CVPixelBufferGetBaseAddress(buffer),
                                 width: width, height: height, bitsPerComponent: 8,
                                 bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                 space: CGColorSpaceCreateDeviceRGB(),
                                 bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
        else { return nil }
        ctx.interpolationQuality = .high
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
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
