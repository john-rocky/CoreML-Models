import CoreML
import UIKit

enum SRError: LocalizedError {
    case modelNotLoaded
    case invalidImage
    case predictionFailed

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded: return "Model not loaded"
        case .invalidImage: return "Failed to process image"
        case .predictionFailed: return "Prediction failed"
        }
    }
}

@MainActor
class SuperResolver: ObservableObject {
    @Published var isReady = false

    private let kappa: Float = 2.0
    private let sqrtEtaT: Float = 0.99
    private let normalizeStd: Float = 2.218
    private let scaleFactor: Float = 1.0

    private let latentH = 256
    private let latentW = 256

    init() {
        checkModelsExist()
    }

    private func checkModelsExist() {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self,
                  let _ = self.modelURL(containing: "Encoder"),
                  let _ = self.modelURL(containing: "Denoiser"),
                  let _ = self.modelURL(containing: "Decoder") else { return }
            DispatchQueue.main.async { self.isReady = true }
        }
    }

    private func modelURL(containing name: String) -> URL? {
        guard let resourcePath = Bundle.main.resourcePath,
              let items = try? FileManager.default.contentsOfDirectory(atPath: resourcePath) else {
            return nil
        }
        for item in items where item.hasSuffix(".mlmodelc") && item.contains(name) {
            return URL(fileURLWithPath: (resourcePath as NSString).appendingPathComponent(item))
        }
        return nil
    }

    private func loadModel(containing name: String, cpuOnly: Bool = false) throws -> MLModel {
        guard let url = modelURL(containing: name) else { throw SRError.modelNotLoaded }
        let config = MLModelConfiguration()
        config.computeUnits = cpuOnly ? .cpuOnly : .cpuAndGPU
        return try MLModel(contentsOf: url, configuration: config)
    }

    func superResolve(image: UIImage) async throws -> UIImage {
        guard isReady else { throw SRError.modelNotLoaded }

        let h = latentH, w = latentW
        let upH = h * 4, upW = w * 4

        // Resize to 256x256 (LQ), then 4x upsample for encoder (matches original pipeline)
        guard let lqImage = resize(image, to: CGSize(width: w, height: h)),
              let lqUpImage = resize(lqImage, to: CGSize(width: upW, height: upH)) else {
            throw SRError.invalidImage
        }

        let lqArray = imageToArray(lqImage, width: w, height: h)
        let lqUpArray = imageToArray(lqUpImage, width: upW, height: upH)

        // Step 1: Encode
        let encoder = try loadModel(containing: "Encoder")
        let encOut = try await encoder.prediction(from: MLDictionaryFeatureProvider(dictionary: ["image": lqUpArray]))
        guard let zYRaw = encOut.featureValue(for: "latent")?.multiArrayValue else { throw SRError.predictionFailed }
        let zY = readArray(zYRaw)

        // Step 2: Add noise
        let spatial = h * w
        var zT = [Float](repeating: 0, count: 3 * spatial)
        let noiseScale = kappa * sqrtEtaT
        for i in 0..<zT.count {
            zT[i] = zY[i] + noiseScale * gaussianRandom()
        }

        // Step 3: Build denoiser input [scaled_zT | lq] as [1, 6, H, W]
        let lqFlat = readMLMultiArray(lqArray)
        let invStd: Float = 1.0 / normalizeStd
        var denIn = [Float](repeating: 0, count: 6 * spatial)
        for i in 0..<(3 * spatial) { denIn[i] = zT[i] * invStd }
        for i in 0..<(3 * spatial) { denIn[3 * spatial + i] = lqFlat[i] }
        let denInArray = writeArray(denIn, shape: [1, 6, NSNumber(value: h), NSNumber(value: w)])

        // Step 4: Denoise (cpuOnly to avoid FP16 overflow in Swin attention)
        let denoiser = try loadModel(containing: "Denoiser", cpuOnly: true)
        let denOut = try await denoiser.prediction(from: MLDictionaryFeatureProvider(dictionary: ["input": denInArray]))
        guard let predRaw = denOut.featureValue(for: "predicted_latent")?.multiArrayValue else { throw SRError.predictionFailed }
        let pred = readArray(predRaw)

        // Step 5: Clamp + scale for decoder
        let invScale: Float = 1.0 / scaleFactor
        var decIn = [Float](repeating: 0, count: pred.count)
        for i in 0..<pred.count { decIn[i] = min(max(pred[i], -1), 1) * invScale }
        let decInArray = writeArray(decIn, shape: [1, 3, NSNumber(value: h), NSNumber(value: w)])

        // Step 6: Decode
        let decoder = try loadModel(containing: "Decoder")
        let decOut = try await decoder.prediction(from: MLDictionaryFeatureProvider(dictionary: ["latent": decInArray]))
        guard let srRaw = decOut.featureValue(for: "image")?.multiArrayValue else { throw SRError.predictionFailed }
        let sr = readArray(srRaw)

        // Step 7: Convert to UIImage
        return arrayToImage(sr, width: upW, height: upH)
    }

    // MARK: - Safe MLMultiArray I/O (subscript-based, layout-independent)

    /// Read MLMultiArray into flat Float array using subscript (handles any stride/dtype)
    private func readArray(_ src: MLMultiArray) -> [Float] {
        let shape = src.shape.map { $0.intValue }
        let count = shape.reduce(1, *)
        var result = [Float](repeating: 0, count: count)

        guard shape.count == 4 else {
            for i in 0..<count { result[i] = src[i].floatValue }
            return result
        }

        let N = shape[0], C = shape[1], H = shape[2], W = shape[3]
        var idx = 0
        for n in 0..<N {
            for c in 0..<C {
                for h in 0..<H {
                    for w in 0..<W {
                        result[idx] = src[[n, c, h, w] as [NSNumber]].floatValue
                        idx += 1
                    }
                }
            }
        }
        return result
    }

    /// Read self-created contiguous MLMultiArray into flat Float array
    private func readMLMultiArray(_ src: MLMultiArray) -> [Float] {
        let count = src.count
        var result = [Float](repeating: 0, count: count)
        let ptr = src.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<count { result[i] = ptr[i] }
        return result
    }

    /// Write flat Float array into a new contiguous MLMultiArray
    private func writeArray(_ data: [Float], shape: [NSNumber]) -> MLMultiArray {
        let array = try! MLMultiArray(shape: shape, dataType: .float32)
        let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<data.count { ptr[i] = data[i] }
        return array
    }

    // MARK: - Image ↔ Array

    private func imageToArray(_ image: UIImage, width: Int, height: Int) -> MLMultiArray {
        let cgImage = image.cgImage!
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        let ctx = CGContext(
            data: &pixels, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        )!
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        // BGRA byte order → NCHW float32 [-1, 1]
        let spatial = height * width
        let array = try! MLMultiArray(shape: [1, 3, NSNumber(value: height), NSNumber(value: width)], dataType: .float32)
        let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)

        for y in 0..<height {
            for x in 0..<width {
                let pIdx = (y * width + x) * 4
                let b = Float(pixels[pIdx]) / 127.5 - 1.0
                let g = Float(pixels[pIdx + 1]) / 127.5 - 1.0
                let r = Float(pixels[pIdx + 2]) / 127.5 - 1.0
                let idx = y * width + x
                ptr[idx] = r
                ptr[spatial + idx] = g
                ptr[2 * spatial + idx] = b
            }
        }
        return array
    }

    private func arrayToImage(_ data: [Float], width: Int, height: Int) -> UIImage {
        let spatial = width * height
        var pixels = [UInt8](repeating: 255, count: width * height * 4)

        for y in 0..<height {
            for x in 0..<width {
                let idx = y * width + x
                let r = UInt8(min(max(data[idx] * 0.5 + 0.5, 0), 1) * 255)
                let g = UInt8(min(max(data[spatial + idx] * 0.5 + 0.5, 0), 1) * 255)
                let b = UInt8(min(max(data[2 * spatial + idx] * 0.5 + 0.5, 0), 1) * 255)
                let pIdx = (y * width + x) * 4
                // BGRA
                pixels[pIdx] = b
                pixels[pIdx + 1] = g
                pixels[pIdx + 2] = r
                pixels[pIdx + 3] = 255
            }
        }

        let provider = CGDataProvider(data: Data(pixels) as CFData)!
        let cgImage = CGImage(
            width: width, height: height,
            bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue),
            provider: provider, decode: nil,
            shouldInterpolate: false, intent: .defaultIntent
        )!
        return UIImage(cgImage: cgImage)
    }

    private func resize(_ image: UIImage, to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, true, 1.0)
        image.draw(in: CGRect(origin: .zero, size: size))
        let resized = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return resized
    }

    private func gaussianRandom() -> Float {
        let u1 = Float.random(in: Float.ulpOfOne...1)
        let u2 = Float.random(in: 0...1)
        return sqrt(-2 * log(u1)) * cos(2 * .pi * u2)
    }
}
