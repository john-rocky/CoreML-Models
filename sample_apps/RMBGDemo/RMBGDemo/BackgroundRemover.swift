import CoreML
import UIKit

enum BackgroundRemoverError: LocalizedError {
    case modelNotFound
    case invalidImage
    case predictionFailed

    var errorDescription: String? {
        switch self {
        case .modelNotFound: return "RMBG model not found"
        case .invalidImage: return "Failed to process image"
        case .predictionFailed: return "Prediction failed"
        }
    }
}

enum BackgroundRemover {

    // MARK: - Public

    static func removeBackground(from image: UIImage) async throws -> UIImage {
        let fixed = image.normalizedOrientation()
        guard let cgImage = fixed.cgImage else { throw BackgroundRemoverError.invalidImage }
        let origW = cgImage.width
        let origH = cgImage.height

        // 1. Create 1024x1024 pixel buffer for model input
        guard let pixelBuffer = createPixelBuffer(from: cgImage, width: 1024, height: 1024) else {
            throw BackgroundRemoverError.invalidImage
        }

        // 2. Load model, predict, release
        let alphaMask: [Float] = try await {
            let model = try loadModel()
            let input = try MLDictionaryFeatureProvider(dictionary: ["image": pixelBuffer])
            let output = try await model.prediction(from: input)
            guard let maskArray = output.featureValue(for: "alpha_mask")?.multiArrayValue else {
                throw BackgroundRemoverError.predictionFailed
            }
            return extractFloats(maskArray)
        }()

        // 3. Apply mask at original resolution
        let resultImage = applyMask(alphaMask, modelSize: 1024, to: cgImage, width: origW, height: origH)
        return resultImage
    }

    // MARK: - Model

    private static func loadModel() throws -> MLModel {
        guard let resourcePath = Bundle.main.resourcePath,
              let items = try? FileManager.default.contentsOfDirectory(atPath: resourcePath) else {
            throw BackgroundRemoverError.modelNotFound
        }
        for item in items where item.hasSuffix(".mlmodelc") && item.contains("RMBG") {
            let url = URL(fileURLWithPath: (resourcePath as NSString).appendingPathComponent(item))
            let config = MLModelConfiguration()
            config.computeUnits = .cpuOnly
            return try MLModel(contentsOf: url, configuration: config)
        }
        throw BackgroundRemoverError.modelNotFound
    }

    // MARK: - Mask Application

    private static func applyMask(_ mask: [Float], modelSize: Int, to cgImage: CGImage, width: Int, height: Int) -> UIImage {
        // Extract original RGBA pixels
        var pixelData = [UInt8](repeating: 0, count: width * height * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(data: &pixelData, width: width, height: height,
                            bitsPerComponent: 8, bytesPerRow: width * 4,
                            space: colorSpace,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        // Resize mask from modelSize x modelSize to original resolution using bilinear interpolation
        // Mask shape: [1,1,modelSize,modelSize]
        var result = [UInt8](repeating: 0, count: width * height * 4)
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

                let v00 = mask[y0 * modelSize + x0]
                let v10 = mask[y0 * modelSize + x1]
                let v01 = mask[y1 * modelSize + x0]
                let v11 = mask[y1 * modelSize + x1]

                let alpha = v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) +
                            v01 * (1 - fx) * fy + v11 * fx * fy
                let a = UInt8(clamping: Int(alpha * 255))

                let idx = (y * width + x) * 4
                let r = pixelData[idx]
                let g = pixelData[idx + 1]
                let b = pixelData[idx + 2]

                // Premultiply alpha
                let af = Float(a) / 255.0
                result[idx]     = UInt8(clamping: Int(Float(r) * af))
                result[idx + 1] = UInt8(clamping: Int(Float(g) * af))
                result[idx + 2] = UInt8(clamping: Int(Float(b) * af))
                result[idx + 3] = a
            }
        }

        let outCtx = CGContext(data: &result, width: width, height: height,
                               bitsPerComponent: 8, bytesPerRow: width * 4,
                               space: colorSpace,
                               bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        let outImage = outCtx.makeImage()!
        return UIImage(cgImage: outImage)
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
