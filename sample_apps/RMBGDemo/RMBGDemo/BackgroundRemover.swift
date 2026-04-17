import CoreML
import CoreImage
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
        // Bilinear-upscale the mask and alpha-blend on the GPU. The prior
        // per-pixel Swift loop was seconds on full-res iPhone photos (12MP).
        var mask8 = [UInt8](repeating: 0, count: modelSize * modelSize)
        for i in 0..<mask8.count { mask8[i] = UInt8(clamping: Int(mask[i] * 255)) }
        let maskCG = CGDataProvider(data: Data(mask8) as CFData).flatMap {
            CGImage(width: modelSize, height: modelSize,
                    bitsPerComponent: 8, bitsPerPixel: 8,
                    bytesPerRow: modelSize,
                    space: CGColorSpaceCreateDeviceGray(),
                    bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                    provider: $0, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
        }
        guard let maskCG else { return UIImage(cgImage: cgImage) }

        let origCI = CIImage(cgImage: cgImage)
        let extent = origCI.extent
        let maskCI = CIImage(cgImage: maskCG).transformed(by:
            CGAffineTransform(scaleX: extent.width / CGFloat(modelSize),
                              y: extent.height / CGFloat(modelSize)))
        let transparent = CIImage.empty().cropped(to: extent)
        let blended = origCI.applyingFilter("CIBlendWithMask", parameters: [
            kCIInputBackgroundImageKey: transparent,
            kCIInputMaskImageKey: maskCI,
        ])
        let ciCtx = CIContext(options: [.useSoftwareRenderer: false])
        guard let outCG = ciCtx.createCGImage(blended, from: extent) else {
            return UIImage(cgImage: cgImage)
        }
        return UIImage(cgImage: outCG)
    }

    // MARK: - Utilities

    private static func extractFloats(_ array: MLMultiArray) -> [Float] {
        let count = array.count
        var raw: [Float]
        if array.dataType == .float16 {
            let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)
            raw = (0..<count).map { Float(ptr[$0]) }
        } else {
            let ptr = array.dataPointer.assumingMemoryBound(to: Float32.self)
            raw = (0..<count).map { ptr[$0] }
        }
        // Min-max normalize to [0, 1] (required by RMBG-1.4)
        let mi = raw.min() ?? 0
        let ma = raw.max() ?? 1
        let range = ma - mi
        if range > 1e-6 {
            for i in raw.indices { raw[i] = (raw[i] - mi) / range }
        }
        return raw
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
