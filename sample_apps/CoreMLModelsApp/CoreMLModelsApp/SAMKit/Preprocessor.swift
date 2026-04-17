import Foundation
import CoreGraphics
import CoreML
import Accelerate

/// Image preprocessing for SAM models.
/// Vendored verbatim from SAMKit (~/Downloads/samkit/runtime/apple/Sources/SAMKit/).
final class Preprocessor {

    private let modelSize: Int
    private let mean: [Float]
    private let std: [Float]

    init(modelSize: Int = 1024) {
        self.modelSize = modelSize
        self.mean = [123.675, 116.28, 103.53]
        self.std = [58.395, 57.12, 57.375]
    }

    func process(_ image: CGImage) throws -> (MLMultiArray, TransformParams) {
        let transform = computeTransform(originalWidth: image.width, originalHeight: image.height)
        guard let resized = resizeAndPad(image, transform: transform) else {
            throw SamError.preprocessingFailed("Failed to resize and pad image")
        }
        let array = try imageToMLMultiArray(resized)
        normalize(array)
        return (array, transform)
    }

    func encodePoints(_ points: [SamPoint], transform: TransformParams) throws -> (MLMultiArray, MLMultiArray) {
        let count = max(1, points.count)
        let coords = try MLMultiArray(shape: [1, count as NSNumber, 2], dataType: .float32)
        let labels = try MLMultiArray(shape: [1, count as NSNumber], dataType: .float32)
        let coordsPtr = coords.dataPointer.bindMemory(to: Float32.self, capacity: coords.count)
        let labelsPtr = labels.dataPointer.bindMemory(to: Float32.self, capacity: labels.count)

        if points.isEmpty {
            coordsPtr[0] = Float(modelSize / 2)
            coordsPtr[1] = Float(modelSize / 2)
            labelsPtr[0] = -1
        } else {
            for (i, point) in points.enumerated() {
                let mp = transform.toModel(point)
                coordsPtr[i * 2] = Float(mp.x)
                coordsPtr[i * 2 + 1] = Float(mp.y)
                labelsPtr[i] = Float(point.label.rawValue)
            }
        }
        return (coords, labels)
    }

    func encodeMask(_ mask: SamMaskRef, transform: TransformParams) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: [1, 1, 256, 256], dataType: .float32)
        let ptr = arr.dataPointer.bindMemory(to: Float32.self, capacity: arr.count)
        let sx = Float(mask.width) / 256.0
        let sy = Float(mask.height) / 256.0
        for y in 0..<256 {
            for x in 0..<256 {
                let srcX = Int(Float(x) * sx)
                let srcY = Int(Float(y) * sy)
                let srcIdx = srcY * mask.width + srcX
                let v: Float
                if let logits = mask.logits, srcIdx < logits.count { v = logits[srcIdx] }
                else if srcIdx < mask.alpha.count { v = Float(mask.alpha[srcIdx]) / 255.0 }
                else { v = 0 }
                ptr[y * 256 + x] = v
            }
        }
        return arr
    }

    func encodeBox(_ box: SamBox, transform: TransformParams) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: [1, 1, 4], dataType: .float32)
        let x0 = Float(box.x0) * transform.scale + transform.padX
        let y0 = Float(box.y0) * transform.scale + transform.padY
        let x1 = Float(box.x1) * transform.scale + transform.padX
        let y1 = Float(box.y1) * transform.scale + transform.padY
        arr[[0, 0, 0] as [NSNumber]] = NSNumber(value: x0)
        arr[[0, 0, 1] as [NSNumber]] = NSNumber(value: y0)
        arr[[0, 0, 2] as [NSNumber]] = NSNumber(value: x1)
        arr[[0, 0, 3] as [NSNumber]] = NSNumber(value: y1)
        return arr
    }

    internal func computeTransform(originalWidth: Int, originalHeight: Int) -> TransformParams {
        let longSide = max(originalWidth, originalHeight)
        let scale = Float(modelSize) / Float(longSide)
        let sw = Int(Float(originalWidth) * scale)
        let sh = Int(Float(originalHeight) * scale)
        let padX = Float(modelSize - sw) / 2.0
        let padY = Float(modelSize - sh) / 2.0
        return TransformParams(scale: scale, padX: padX, padY: padY,
                               originalWidth: originalWidth, originalHeight: originalHeight,
                               modelSize: modelSize)
    }

    internal func resizeAndPad(_ image: CGImage, transform: TransformParams) -> CGImage? {
        let sw = Int(Float(image.width) * transform.scale)
        let sh = Int(Float(image.height) * transform.scale)
        guard let ctx = CGContext(
            data: nil, width: modelSize, height: modelSize,
            bitsPerComponent: 8, bytesPerRow: modelSize * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else { return nil }
        ctx.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: modelSize, height: modelSize))
        ctx.draw(image, in: CGRect(x: Int(transform.padX), y: Int(transform.padY),
                                    width: sw, height: sh))
        return ctx.makeImage()
    }

    private func imageToMLMultiArray(_ image: CGImage) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: [3, modelSize as NSNumber, modelSize as NSNumber], dataType: .float32)
        let w = image.width, h = image.height
        let bpp = 4, bpr = bpp * w
        let pixelCount = w * h

        var pixelData = [UInt8](repeating: 0, count: h * bpr)
        guard let ctx = CGContext(
            data: &pixelData, width: w, height: h,
            bitsPerComponent: 8, bytesPerRow: bpr,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else {
            throw SamError.preprocessingFailed("Failed to create bitmap context")
        }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))

        let ptr = arr.dataPointer.bindMemory(to: Float32.self, capacity: arr.count)
        let channelSize = modelSize * modelSize

        var rU8 = [UInt8](repeating: 0, count: pixelCount)
        var gU8 = [UInt8](repeating: 0, count: pixelCount)
        var bU8 = [UInt8](repeating: 0, count: pixelCount)
        pixelData.withUnsafeBufferPointer { src in
            let base = src.baseAddress!
            for i in 0..<pixelCount {
                rU8[i] = base[i * 4]
                gU8[i] = base[i * 4 + 1]
                bU8[i] = base[i * 4 + 2]
            }
        }

        let n = vDSP_Length(pixelCount)
        vDSP_vfltu8(rU8, 1, ptr, 1, n)
        vDSP_vfltu8(gU8, 1, ptr + channelSize, 1, n)
        vDSP_vfltu8(bU8, 1, ptr + channelSize * 2, 1, n)
        return arr
    }

    private func normalize(_ array: MLMultiArray) {
        let ptr = array.dataPointer.bindMemory(to: Float32.self, capacity: array.count)
        let channelSize = modelSize * modelSize
        let n = vDSP_Length(channelSize)
        for c in 0..<3 {
            let chPtr = ptr + c * channelSize
            var negMean = -mean[c]
            vDSP_vsadd(chPtr, 1, &negMean, chPtr, 1, n)
            var invStd = 1.0 / std[c]
            vDSP_vsmul(chPtr, 1, &invStd, chPtr, 1, n)
        }
    }
}

/// Letterbox transform parameters.
struct TransformParams {
    let scale: Float
    let padX: Float
    let padY: Float
    let originalWidth: Int
    let originalHeight: Int
    let modelSize: Int

    func toModel(_ point: SamPoint) -> SamPoint {
        SamPoint(
            x: point.x * CGFloat(scale) + CGFloat(padX),
            y: point.y * CGFloat(scale) + CGFloat(padY),
            label: point.label
        )
    }

    func toImage(_ point: SamPoint) -> SamPoint {
        SamPoint(
            x: (point.x - CGFloat(padX)) / CGFloat(scale),
            y: (point.y - CGFloat(padY)) / CGFloat(scale),
            label: point.label
        )
    }
}
