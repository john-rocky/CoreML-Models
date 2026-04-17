import Foundation
import CoreML
import Accelerate
import CoreGraphics
#if canImport(UIKit)
import UIKit
#endif

/// Mask postprocessing for SAM models. Vendored verbatim from SAMKit.
final class Postprocessor {

    private let modelSize: Int
    private let isHuggingFaceModel: Bool

    init(modelSize: Int = 1024, isHuggingFaceModel: Bool = false) {
        self.modelSize = modelSize
        self.isHuggingFaceModel = isHuggingFaceModel
    }

    func process(
        maskLogits: MLMultiArray,
        iouPredictions: MLMultiArray,
        transform: TransformParams,
        options: SamOptions
    ) throws -> SamResult {

        guard maskLogits.shape.count >= 4 else {
            throw SamError.postprocessingFailed("Expected 4D mask tensor, got shape: \(maskLogits.shape)")
        }

        let maskLogits32 = try ensureFloat32(maskLogits)
        let iouPredictions32 = try ensureFloat32(iouPredictions)

        let maskHeight = maskLogits32.shape[2].intValue
        let isLowRes = maskHeight <= 256

        // Keep masks at model resolution — the UI scales them via .scaledToFit().
        let processedMasks: MLMultiArray
        if isLowRes {
            let upsampled = try upsampleMasks(maskLogits32)
            processedMasks = try removePadding(upsampled, transform: transform)
        } else {
            processedMasks = try removePadding(maskLogits32, transform: transform)
        }

        let masks = try extractMasks(processedMasks, scores: iouPredictions32, options: options)
        return SamResult(masks: masks, scores: masks.map { $0.score })
    }

    // MARK: - Private

    private func ensureFloat32(_ array: MLMultiArray) throws -> MLMultiArray {
        if #available(iOS 16.0, macOS 13.0, *) {
            guard array.dataType == .float16 else { return array }
        } else {
            return array
        }

        let result = try MLMultiArray(shape: array.shape, dataType: .float32)
        let count = array.count
        let srcPtr = array.dataPointer.bindMemory(to: UInt16.self, capacity: count)
        let dstPtr = result.dataPointer.bindMemory(to: Float32.self, capacity: count)

        var srcBuf = vImage_Buffer(
            data: UnsafeMutableRawPointer(mutating: srcPtr),
            height: 1, width: vImagePixelCount(count),
            rowBytes: count * MemoryLayout<UInt16>.size
        )
        var dstBuf = vImage_Buffer(
            data: UnsafeMutableRawPointer(dstPtr),
            height: 1, width: vImagePixelCount(count),
            rowBytes: count * MemoryLayout<Float32>.size
        )
        vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, vImage_Flags(kvImageNoFlags))
        return result
    }

    private func upsampleMasks(_ masks: MLMultiArray) throws -> MLMultiArray {
        guard masks.shape.count == 4 else {
            throw SamError.postprocessingFailed("Invalid mask shape: \(masks.shape)")
        }

        let batchSize = masks.shape[0].intValue
        let secondDim = masks.shape[1].intValue
        let lowResH = masks.shape[2].intValue
        let lowResW = masks.shape[3].intValue

        let isHuggingFaceGrid = isHuggingFaceModel
            && batchSize == 1 && secondDim == 3 && lowResH == 256 && lowResW == 256
        if isHuggingFaceGrid {
            return try upsampleHuggingFaceGridMasks(masks)
        }

        let (actualNumMasks, channelDim): (Int, Int)
        if secondDim > 1 && batchSize == 1 {
            actualNumMasks = secondDim; channelDim = 1
        } else {
            actualNumMasks = batchSize; channelDim = secondDim
        }

        let upsampled = try MLMultiArray(
            shape: [actualNumMasks as NSNumber, 1,
                    modelSize as NSNumber, modelSize as NSNumber],
            dataType: .float32
        )

        let srcPtr = masks.dataPointer.bindMemory(to: Float32.self, capacity: masks.count)
        let dstPtr = upsampled.dataPointer.bindMemory(to: Float32.self, capacity: upsampled.count)

        if secondDim > 1 && batchSize == 1 {
            for n in 0..<actualNumMasks {
                let srcOffset = n * lowResH * lowResW
                let dstOffset = n * modelSize * modelSize
                bilinearUpsample(
                    src: srcPtr.advanced(by: srcOffset),
                    dst: dstPtr.advanced(by: dstOffset),
                    srcWidth: lowResW, srcHeight: lowResH,
                    dstWidth: modelSize, dstHeight: modelSize
                )
            }
        } else {
            for n in 0..<actualNumMasks {
                let srcOffset = n * channelDim * lowResH * lowResW
                let dstOffset = n * modelSize * modelSize
                bilinearUpsample(
                    src: srcPtr.advanced(by: srcOffset),
                    dst: dstPtr.advanced(by: dstOffset),
                    srcWidth: lowResW, srcHeight: lowResH,
                    dstWidth: modelSize, dstHeight: modelSize
                )
            }
        }

        return upsampled
    }

    private func upsampleHuggingFaceGridMasks(_ masks: MLMultiArray) throws -> MLMultiArray {
        let numMasks = masks.shape[1].intValue
        let gridSize = masks.shape[2].intValue
        let halfSize = gridSize / 2

        let upsampled = try MLMultiArray(
            shape: [numMasks as NSNumber, 1,
                    modelSize as NSNumber, modelSize as NSNumber],
            dataType: .float32
        )

        let srcPtr = masks.dataPointer.bindMemory(to: Float32.self, capacity: masks.count)
        let dstPtr = upsampled.dataPointer.bindMemory(to: Float32.self, capacity: upsampled.count)

        for maskIdx in 0..<numMasks {
            let maskSrcBase = srcPtr.advanced(by: maskIdx * gridSize * gridSize)
            let (qy, qx) = quadrantPosition(maskIndex: maskIdx)
            var quadrant = [Float32](repeating: 0, count: halfSize * halfSize)
            for y in 0..<halfSize {
                for x in 0..<halfSize {
                    quadrant[y * halfSize + x] = maskSrcBase[(qy + y) * gridSize + (qx + x)]
                }
            }

            let maskDst = dstPtr.advanced(by: maskIdx * modelSize * modelSize)
            quadrant.withUnsafeBufferPointer { buf in
                bilinearUpsample(
                    src: UnsafeMutablePointer(mutating: buf.baseAddress!),
                    dst: maskDst,
                    srcWidth: halfSize, srcHeight: halfSize,
                    dstWidth: modelSize, dstHeight: modelSize
                )
            }
        }

        return upsampled
    }

    private func quadrantPosition(maskIndex: Int) -> (y: Int, x: Int) {
        switch maskIndex {
        case 0: return (0, 0)
        case 1: return (0, 128)
        case 2: return (128, 0)
        default: return (128, 128)
        }
    }

    private func removePadding(_ masks: MLMultiArray, transform: TransformParams) throws -> MLMultiArray {
        let numMasks = masks.shape[0].intValue
        let channels = masks.shape.count > 1 ? masks.shape[1].intValue : 1
        let maskH = masks.shape[2].intValue
        let maskW = masks.shape[3].intValue

        guard maskH == modelSize && maskW == modelSize else {
            return masks
        }

        let scaledW = Int(Float(transform.originalWidth) * transform.scale)
        let scaledH = Int(Float(transform.originalHeight) * transform.scale)
        let padLeft = Int(transform.padX)
        let padTop = Int(transform.padY)

        let depadded = try MLMultiArray(
            shape: [numMasks as NSNumber, channels as NSNumber,
                    scaledH as NSNumber, scaledW as NSNumber],
            dataType: .float32
        )

        let srcPtr = masks.dataPointer.bindMemory(to: Float32.self, capacity: masks.count)
        let dstPtr = depadded.dataPointer.bindMemory(to: Float32.self, capacity: depadded.count)
        let rowBytes = scaledW * MemoryLayout<Float32>.size

        for n in 0..<numMasks {
            for c in 0..<channels {
                let srcOffset = (n * channels + c) * maskH * maskW
                let dstOffset = (n * channels + c) * scaledH * scaledW
                let maskSrc = srcPtr.advanced(by: srcOffset)
                let maskDst = dstPtr.advanced(by: dstOffset)
                for y in 0..<scaledH {
                    memcpy(
                        maskDst.advanced(by: y * scaledW),
                        maskSrc.advanced(by: (y + padTop) * maskW + padLeft),
                        rowBytes
                    )
                }
            }
        }

        return depadded
    }

    private func extractMasks(
        _ masks: MLMultiArray,
        scores: MLMultiArray,
        options: SamOptions
    ) throws -> [SamMask] {

        let numMasks = masks.shape[0].intValue
        let channels = masks.shape[1].intValue
        let height = masks.shape[2].intValue
        let width = masks.shape[3].intValue

        let isScores2D = scores.shape.count == 2 && scores.shape[0].intValue == 1
        let scoresCount = isScores2D ? scores.shape[1].intValue : scores.count

        let maskPtr = masks.dataPointer.bindMemory(to: Float32.self, capacity: masks.count)
        let scorePtr = scores.dataPointer.bindMemory(to: Float32.self, capacity: scores.count)

        let sortedIndices = sortByScore(scores: scorePtr, count: min(numMasks, scoresCount))
        let effective = min(numMasks, 3)
        let toReturn = options.multimaskOutput ? min(options.maxMasks, effective) : 1

        var result: [SamMask] = []
        for i in 0..<toReturn {
            guard i < sortedIndices.count else { break }
            let idx = sortedIndices[i]
            guard idx < scoresCount else { continue }

            let offset = idx * channels * height * width
            let maskData = maskPtr.advanced(by: offset)

            var logits: [Float]? = nil
            if options.returnLogits {
                logits = Array(UnsafeBufferPointer(start: maskData, count: height * width))
            }

            let alpha = createAlphaMask(from: maskData, width: width, height: height,
                                        threshold: options.maskThreshold)
            let cg = createCGImage(from: alpha, width: width, height: height)

            var score: Float
            if isScores2D {
                score = scores[[0, idx as NSNumber] as [NSNumber]].floatValue
            } else {
                score = scorePtr[idx]
            }
            if score.isNaN || score.isInfinite || score < 0 { score = 0 }

            result.append(SamMask(width: width, height: height, logits: logits,
                                  alpha: alpha, score: score, cgImage: cg))
        }
        return result
    }

    private func bilinearUpsample(
        src: UnsafeMutablePointer<Float32>,
        dst: UnsafeMutablePointer<Float32>,
        srcWidth: Int, srcHeight: Int,
        dstWidth: Int, dstHeight: Int
    ) {
        var srcBuf = vImage_Buffer(
            data: src, height: vImagePixelCount(srcHeight),
            width: vImagePixelCount(srcWidth),
            rowBytes: srcWidth * MemoryLayout<Float32>.size
        )
        var dstBuf = vImage_Buffer(
            data: dst, height: vImagePixelCount(dstHeight),
            width: vImagePixelCount(dstWidth),
            rowBytes: dstWidth * MemoryLayout<Float32>.size
        )
        vImageScale_PlanarF(&srcBuf, &dstBuf, nil, vImage_Flags(kvImageHighQualityResampling))
    }

    private func sortByScore(scores: UnsafeMutablePointer<Float32>, count: Int) -> [Int] {
        var indices = Array(0..<count)
        indices.sort { scores[$0] > scores[$1] }
        return indices
    }

    private func createAlphaMask(
        from logits: UnsafeMutablePointer<Float32>,
        width: Int, height: Int,
        threshold: Float
    ) -> Data {
        let count = width * height
        var alpha = Data(count: count)
        alpha.withUnsafeMutableBytes { buf in
            guard let ptr = buf.bindMemory(to: UInt8.self).baseAddress else { return }
            if threshold > 0 {
                for i in 0..<count { ptr[i] = logits[i] > threshold ? 255 : 0 }
            } else {
                // Vectorized sigmoid via Accelerate.
                var negated = [Float](repeating: 0, count: count)
                var expResult = [Float](repeating: 0, count: count)
                var sigmoid = [Float](repeating: 0, count: count)
                var scaled = [Float](repeating: 0, count: count)

                var cmin: Float = -50, cmax: Float = 50
                vDSP_vclip(logits, 1, &cmin, &cmax, &negated, 1, vDSP_Length(count))
                vDSP_vneg(negated, 1, &negated, 1, vDSP_Length(count))
                var n = Int32(count)
                vvexpf(&expResult, negated, &n)
                var one: Float = 1.0
                vDSP_vsadd(expResult, 1, &one, &expResult, 1, vDSP_Length(count))
                vvrecf(&sigmoid, expResult, &n)
                var scale: Float = 255.0
                vDSP_vsmul(sigmoid, 1, &scale, &scaled, 1, vDSP_Length(count))

                var scaledU = scaled
                vDSP_vfixu8(&scaledU, 1, ptr, 1, vDSP_Length(count))
            }
        }
        return alpha
    }

    private func createCGImage(from alpha: Data, width: Int, height: Int) -> CGImage {
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let pixelCount = width * height
        var pixelData = [UInt8](repeating: 0, count: pixelCount * bytesPerPixel)

        alpha.withUnsafeBytes { buf in
            guard let alphaPtr = buf.bindMemory(to: UInt8.self).baseAddress else { return }
            pixelData.withUnsafeMutableBufferPointer { dst in
                let base = dst.baseAddress!
                var i = 0
                while i < pixelCount {
                    let p = i * 4
                    let a = alphaPtr[i]
                    let af = Float(a) / 255.0
                    base[p]     = UInt8(30.0  * af)   // R
                    base[p + 1] = UInt8(144.0 * af)   // G
                    base[p + 2] = UInt8(255.0 * af)   // B
                    base[p + 3] = a
                    i += 1
                }
            }
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        guard let provider = CGDataProvider(data: NSData(bytes: &pixelData, length: pixelData.count)),
              let cg = CGImage(
                width: width, height: height,
                bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: bytesPerRow,
                space: colorSpace, bitmapInfo: bitmapInfo,
                provider: provider, decode: nil,
                shouldInterpolate: true, intent: .defaultIntent
              )
        else {
            let onePixel: [UInt8] = [0, 0, 0, 0]
            let p = CGDataProvider(data: NSData(bytes: onePixel, length: 4))!
            return CGImage(width: 1, height: 1,
                           bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 4,
                           space: colorSpace, bitmapInfo: bitmapInfo,
                           provider: p, decode: nil,
                           shouldInterpolate: false, intent: .defaultIntent)!
        }
        return cg
    }
}

// MARK: - SamMask Object Extraction

extension SamMask {
    /// Lift the masked subject out of the source image with a transparent background.
    func extractObject(from sourceImage: CGImage) -> CGImage? {
        let W = sourceImage.width, H = sourceImage.height
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)

        guard let ctx = CGContext(
            data: nil, width: W, height: H,
            bitsPerComponent: 8, bytesPerRow: W * 4,
            space: colorSpace, bitmapInfo: bitmapInfo.rawValue
        ) else { return nil }

        var binary = Data(count: alpha.count)
        alpha.withUnsafeBytes { src in
            binary.withUnsafeMutableBytes { dst in
                let sp = src.bindMemory(to: UInt8.self).baseAddress!
                let dp = dst.bindMemory(to: UInt8.self).baseAddress!
                for i in 0..<alpha.count { dp[i] = sp[i] >= 128 ? 255 : 0 }
            }
        }

        let gray = CGColorSpaceCreateDeviceGray()
        guard let provider = CGDataProvider(data: binary as CFData),
              let maskCG = CGImage(
                width: width, height: height,
                bitsPerComponent: 8, bitsPerPixel: 8,
                bytesPerRow: width,
                space: gray,
                bitmapInfo: CGBitmapInfo(rawValue: 0),
                provider: provider, decode: nil,
                shouldInterpolate: true, intent: .defaultIntent
              ) else { return nil }

        let rect = CGRect(x: 0, y: 0, width: W, height: H)
        ctx.clip(to: rect, mask: maskCG)
        ctx.draw(sourceImage, in: rect)
        return ctx.makeImage()
    }

    static func extractObject(from sourceImage: CGImage, masks: [SamMask]) -> CGImage? {
        guard let first = masks.first else { return nil }
        if masks.count == 1 { return first.extractObject(from: sourceImage) }

        let w = first.width, h = first.height
        let pixelCount = w * h
        var combined = Data(count: pixelCount)
        combined.withUnsafeMutableBytes { dst in
            let dp = dst.bindMemory(to: UInt8.self).baseAddress!
            for mask in masks {
                mask.alpha.withUnsafeBytes { src in
                    let sp = src.bindMemory(to: UInt8.self).baseAddress!
                    let count = min(pixelCount, mask.alpha.count)
                    for i in 0..<count { dp[i] = max(dp[i], sp[i]) }
                }
            }
            for i in 0..<pixelCount { dp[i] = dp[i] >= 128 ? 255 : 0 }
        }

        let W = sourceImage.width, H = sourceImage.height
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        guard let ctx = CGContext(
            data: nil, width: W, height: H,
            bitsPerComponent: 8, bytesPerRow: W * 4,
            space: colorSpace, bitmapInfo: bitmapInfo.rawValue
        ) else { return nil }

        let gray = CGColorSpaceCreateDeviceGray()
        guard let provider = CGDataProvider(data: combined as CFData),
              let maskCG = CGImage(
                width: w, height: h,
                bitsPerComponent: 8, bitsPerPixel: 8,
                bytesPerRow: w,
                space: gray,
                bitmapInfo: CGBitmapInfo(rawValue: 0),
                provider: provider, decode: nil,
                shouldInterpolate: true, intent: .defaultIntent
              ) else { return nil }

        let rect = CGRect(x: 0, y: 0, width: W, height: H)
        ctx.clip(to: rect, mask: maskCG)
        ctx.draw(sourceImage, in: rect)
        return ctx.makeImage()
    }

    /// Binary silhouette image with white interior and transparent exterior.
    /// Used for the glowing outline and mask overlay in the demo view.
    func toBinaryWhiteSilhouette() -> CGImage? {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        guard let ctx = CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: colorSpace, bitmapInfo: bitmapInfo.rawValue
        ) else { return nil }

        var binary = Data(count: alpha.count)
        alpha.withUnsafeBytes { src in
            binary.withUnsafeMutableBytes { dst in
                let sp = src.bindMemory(to: UInt8.self).baseAddress!
                let dp = dst.bindMemory(to: UInt8.self).baseAddress!
                for i in 0..<alpha.count {
                    let v: UInt8 = sp[i] >= 128 ? 255 : 0
                    let p = i * 4
                    dp[p] = v; dp[p+1] = v; dp[p+2] = v; dp[p+3] = v
                }
            }
        }

        guard let provider = CGDataProvider(data: binary as CFData),
              let cg = CGImage(
                width: width, height: height,
                bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: width * 4,
                space: colorSpace, bitmapInfo: bitmapInfo,
                provider: provider, decode: nil,
                shouldInterpolate: true, intent: .defaultIntent
              ) else { return nil }
        return cg
    }
}
