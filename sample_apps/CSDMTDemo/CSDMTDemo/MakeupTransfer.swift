import CoreML
import UIKit

enum MakeupTransferError: LocalizedError {
    case modelNotFound(String)
    case invalidImage
    case predictionFailed

    var errorDescription: String? {
        switch self {
        case .modelNotFound(let name): return "\(name) model not found"
        case .invalidImage: return "Failed to process image"
        case .predictionFailed: return "Prediction failed"
        }
    }
}

enum MakeupTransfer {

    static func transfer(source: UIImage, reference: UIImage) async throws -> UIImage {
        let srcFixed = source.normalizedOrientation()
        let refFixed = reference.normalizedOrientation()
        guard let srcCG = srcFixed.cgImage, let refCG = refFixed.cgImage else {
            throw MakeupTransferError.invalidImage
        }
        guard let srcBuf = createPixelBuffer(from: srcCG, width: 256, height: 256),
              let refBuf = createPixelBuffer(from: refCG, width: 256, height: 256) else {
            throw MakeupTransferError.invalidImage
        }

        // 1. Parse both faces
        let parserModel = try loadModel(containing: "FaceParser")
        let (srcParse, srcMask) = try await parseFace(model: parserModel, buffer: srcBuf)
        let (refParse, refMask) = try await parseFace(model: parserModel, buffer: refBuf)

        // 2. Run makeup transfer
        let transferModel = try loadModel(containing: "MakeupTransfer")
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "source": srcBuf,
            "reference": refBuf,
            "source_parse": srcParse,
            "source_mask": srcMask,
            "ref_parse": refParse,
            "ref_mask": refMask,
        ])
        let output = try await transferModel.prediction(from: input)
        guard let resultArray = output.featureValue(for: "result")?.multiArrayValue else {
            throw MakeupTransferError.predictionFailed
        }
        return multiArrayToImage(resultArray, width: 256, height: 256)
    }

    private static func parseFace(model: MLModel, buffer: CVPixelBuffer) async throws -> (MLMultiArray, MLMultiArray) {
        let input = try MLDictionaryFeatureProvider(dictionary: ["image": buffer])
        let output = try await model.prediction(from: input)
        guard let parse = output.featureValue(for: "parse")?.multiArrayValue,
              let mask = output.featureValue(for: "face_mask")?.multiArrayValue else {
            throw MakeupTransferError.predictionFailed
        }
        return (parse, mask)
    }

    private static func loadModel(containing name: String) throws -> MLModel {
        guard let resourcePath = Bundle.main.resourcePath,
              let items = try? FileManager.default.contentsOfDirectory(atPath: resourcePath) else {
            throw MakeupTransferError.modelNotFound(name)
        }
        for item in items where item.hasSuffix(".mlmodelc") && item.contains(name) {
            let url = URL(fileURLWithPath: (resourcePath as NSString).appendingPathComponent(item))
            let config = MLModelConfiguration()
            config.computeUnits = .all
            return try MLModel(contentsOf: url, configuration: config)
        }
        throw MakeupTransferError.modelNotFound(name)
    }

    private static func multiArrayToImage(_ array: MLMultiArray, width: Int, height: Int) -> UIImage {
        var pixels = [UInt8](repeating: 255, count: width * height * 4)
        let count = width * height

        if array.dataType == .float16 {
            let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)
            for y in 0..<height {
                for x in 0..<width {
                    let idx = (y * width + x) * 4
                    pixels[idx]     = UInt8(clamping: Int(min(max(Float(ptr[0 * count + y * width + x]), 0), 1) * 255))
                    pixels[idx + 1] = UInt8(clamping: Int(min(max(Float(ptr[1 * count + y * width + x]), 0), 1) * 255))
                    pixels[idx + 2] = UInt8(clamping: Int(min(max(Float(ptr[2 * count + y * width + x]), 0), 1) * 255))
                }
            }
        } else {
            let ptr = array.dataPointer.assumingMemoryBound(to: Float32.self)
            for y in 0..<height {
                for x in 0..<width {
                    let idx = (y * width + x) * 4
                    pixels[idx]     = UInt8(clamping: Int(min(max(ptr[0 * count + y * width + x], 0), 1) * 255))
                    pixels[idx + 1] = UInt8(clamping: Int(min(max(ptr[1 * count + y * width + x], 0), 1) * 255))
                    pixels[idx + 2] = UInt8(clamping: Int(min(max(ptr[2 * count + y * width + x], 0), 1) * 255))
                }
            }
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(data: &pixels, width: width, height: height,
                            bitsPerComponent: 8, bytesPerRow: width * 4,
                            space: colorSpace,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        return UIImage(cgImage: ctx.makeImage()!)
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
