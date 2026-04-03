import CoreML
import UIKit
import Accelerate

class ImageColorizer: ObservableObject {
    private var mlModel: MLModel?
    @Published var isReady = false

    init() {
        loadModel()
    }

    private func loadModel() {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self, let resourcePath = Bundle.main.resourcePath else { return }
            let fm = FileManager.default
            guard let items = try? fm.contentsOfDirectory(atPath: resourcePath) else { return }
            let config = MLModelConfiguration()
            config.computeUnits = .all
            for item in items where item.hasSuffix(".mlmodelc") && item.contains("DDColor") {
                let url = URL(fileURLWithPath: (resourcePath as NSString).appendingPathComponent(item))
                if let model = try? MLModel(contentsOf: url, configuration: config) {
                    self.mlModel = model
                    DispatchQueue.main.async { self.isReady = true }
                    return
                }
            }
        }
    }

    // MARK: - Colorize

    func colorize(image: UIImage) async throws -> UIImage {
        let fixed = image.normalizedOrientation()
        guard let cgImage = fixed.cgImage else { throw ColorizerError.invalidImage }
        let origW = cgImage.width
        let origH = cgImage.height

        // 1. Extract L channel at original resolution
        let origLAB = rgbToLAB(cgImage: cgImage)
        let origL = extractL(lab: origLAB, width: origW, height: origH)

        // 2. Resize to 512x512 and create gray-RGB input
        let resized = resizeCGImage(cgImage, to: CGSize(width: 512, height: 512))
        let grayRGB = createGrayRGB(cgImage: resized)

        // 3. Run model
        guard let model = mlModel else { throw ColorizerError.modelNotLoaded }
        let inputArray = try MLMultiArray(shape: [1, 3, 512, 512], dataType: .float32)
        for i in 0..<(3 * 512 * 512) {
            inputArray[i] = NSNumber(value: grayRGB[i])
        }
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(multiArray: inputArray)
        ])
        let output = try model.prediction(from: input)
        guard let abArray = output.featureValue(for: "ab_channels")?.multiArrayValue else {
            throw ColorizerError.predictionFailed
        }

        // 4. Extract AB, resize to original size
        let abCount = abArray.count  // 1 * 2 * 512 * 512
        var ab512 = [Float](repeating: 0, count: abCount)
        for i in 0..<abCount { ab512[i] = abArray[i].floatValue }

        // Debug: check AB range
        let abMin = ab512.min() ?? 0
        let abMax = ab512.max() ?? 0
        print("[DDColor] AB range: [\(abMin), \(abMax)], count: \(abCount)")

        let abOrig = resizeAB(ab512, fromW: 512, fromH: 512, toW: origW, toH: origH)

        // 5. Combine L + AB → LAB → RGB
        let colorImage = labToRGB(l: origL, ab: abOrig, width: origW, height: origH)
        return colorImage
    }

    // MARK: - LAB Color Space

    private func rgbToLAB(cgImage: CGImage) -> [Float] {
        let w = cgImage.width, h = cgImage.height
        let pixels = extractRGBPixels(cgImage: cgImage)

        var lab = [Float](repeating: 0, count: w * h * 3)
        for i in 0..<(w * h) {
            let r = pixels[i * 3], g = pixels[i * 3 + 1], b = pixels[i * 3 + 2]
            let (l, a, bv) = srgbToLab(r: r, g: g, b: b)
            lab[i * 3] = l
            lab[i * 3 + 1] = a
            lab[i * 3 + 2] = bv
        }
        return lab
    }

    private func extractL(lab: [Float], width: Int, height: Int) -> [Float] {
        var l = [Float](repeating: 0, count: width * height)
        for i in 0..<(width * height) {
            l[i] = lab[i * 3]
        }
        return l
    }

    private func createGrayRGB(cgImage: CGImage) -> [Float] {
        // Convert to grayscale LAB [L, 0, 0] then back to RGB → model input
        let w = cgImage.width, h = cgImage.height
        let pixels = extractRGBPixels(cgImage: cgImage)

        // [CHW] format: [1, 3, 512, 512], normalized to [0, 1]
        var result = [Float](repeating: 0, count: 3 * w * h)
        for i in 0..<(w * h) {
            let r = pixels[i * 3], g = pixels[i * 3 + 1], b = pixels[i * 3 + 2]
            let (l, _, _) = srgbToLab(r: r, g: g, b: b)
            // LAB with A=0, B=0 → RGB
            let (gr, gg, gb) = labToSrgb(l: l, a: 0, b: 0)
            result[0 * w * h + i] = gr  // R channel
            result[1 * w * h + i] = gg  // G channel
            result[2 * w * h + i] = gb  // B channel
        }
        return result
    }

    private func labToRGB(l: [Float], ab: [Float], width: Int, height: Int) -> UIImage {
        let count = width * height
        var pixelData = [UInt8](repeating: 255, count: count * 4)

        for i in 0..<count {
            let lv = l[i]
            let a = ab[i]
            let b = ab[count + i]
            let (r, g, bv) = labToSrgb(l: lv, a: a, b: b)
            pixelData[i * 4] = UInt8(clamping: Int(r * 255))
            pixelData[i * 4 + 1] = UInt8(clamping: Int(g * 255))
            pixelData[i * 4 + 2] = UInt8(clamping: Int(bv * 255))
            pixelData[i * 4 + 3] = 255
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(data: &pixelData, width: width, height: height,
                            bitsPerComponent: 8, bytesPerRow: width * 4,
                            space: colorSpace,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        let cgImg = ctx.makeImage()!
        return UIImage(cgImage: cgImg)
    }

    // MARK: - sRGB ↔ LAB conversions (D65 illuminant)

    private func srgbToLab(r: Float, g: Float, b: Float) -> (Float, Float, Float) {
        // sRGB → linear
        func toLinear(_ c: Float) -> Float {
            c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4)
        }
        let rl = toLinear(r), gl = toLinear(g), bl = toLinear(b)

        // Linear RGB → XYZ (D65)
        var x = rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375
        var y = rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750
        var z = rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041

        // Normalize by D65 white point
        x /= 0.95047; y /= 1.0; z /= 1.08883

        func f(_ t: Float) -> Float {
            t > 0.008856 ? pow(t, 1.0 / 3.0) : 7.787 * t + 16.0 / 116.0
        }
        let fx = f(x), fy = f(y), fz = f(z)

        let l = 116.0 * fy - 16.0  // [0, 100]
        let a = 500.0 * (fx - fy)  // [-128, 127]
        let bv = 200.0 * (fy - fz) // [-128, 127]
        return (l, a, bv)
    }

    private func labToSrgb(l: Float, a: Float, b: Float) -> (Float, Float, Float) {
        let fy = (l + 16.0) / 116.0
        let fx = a / 500.0 + fy
        let fz = fy - b / 200.0

        func invF(_ t: Float) -> Float {
            let t3 = t * t * t
            return t3 > 0.008856 ? t3 : (t - 16.0 / 116.0) / 7.787
        }
        let x = invF(fx) * 0.95047
        let y = invF(fy) * 1.0
        let z = invF(fz) * 1.08883

        // XYZ → linear RGB
        var r = x *  3.2404542 + y * -1.5371385 + z * -0.4985314
        var g = x * -0.9692660 + y *  1.8760108 + z *  0.0415560
        var bv = x *  0.0556434 + y * -0.2040259 + z *  1.0572252

        // Linear → sRGB
        func toSRGB(_ c: Float) -> Float {
            let clamped = max(0, min(1, c))
            return clamped <= 0.0031308 ? clamped * 12.92 : 1.055 * pow(clamped, 1.0 / 2.4) - 0.055
        }
        return (toSRGB(r), toSRGB(g), toSRGB(bv))
    }

    // MARK: - Image helpers

    private func extractRGBPixels(cgImage: CGImage) -> [Float] {
        let w = cgImage.width, h = cgImage.height
        var pixelData = [UInt8](repeating: 0, count: w * h * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(data: &pixelData, width: w, height: h,
                            bitsPerComponent: 8, bytesPerRow: w * 4,
                            space: colorSpace,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))

        var result = [Float](repeating: 0, count: w * h * 3)
        for i in 0..<(w * h) {
            result[i * 3] = Float(pixelData[i * 4]) / 255.0
            result[i * 3 + 1] = Float(pixelData[i * 4 + 1]) / 255.0
            result[i * 3 + 2] = Float(pixelData[i * 4 + 2]) / 255.0
        }
        return result
    }

    private func resizeCGImage(_ image: CGImage, to size: CGSize) -> CGImage {
        let w = Int(size.width), h = Int(size.height)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(data: nil, width: w, height: h,
                            bitsPerComponent: 8, bytesPerRow: w * 4,
                            space: colorSpace,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        return ctx.makeImage()!
    }

    private func resizeAB(_ ab: [Float], fromW: Int, fromH: Int, toW: Int, toH: Int) -> [Float] {
        // ab is [2, fromH, fromW], resize each channel with bilinear interpolation
        let fromCount = fromW * fromH
        let toCount = toW * toH
        var result = [Float](repeating: 0, count: 2 * toCount)

        for ch in 0..<2 {
            let srcOffset = ch * fromCount
            let dstOffset = ch * toCount
            for y in 0..<toH {
                let srcY = Float(y) * Float(fromH) / Float(toH)
                let y0 = min(Int(srcY), fromH - 1)
                let y1 = min(y0 + 1, fromH - 1)
                let fy = srcY - Float(y0)
                for x in 0..<toW {
                    let srcX = Float(x) * Float(fromW) / Float(toW)
                    let x0 = min(Int(srcX), fromW - 1)
                    let x1 = min(x0 + 1, fromW - 1)
                    let fx = srcX - Float(x0)

                    let v00 = ab[srcOffset + y0 * fromW + x0]
                    let v10 = ab[srcOffset + y0 * fromW + x1]
                    let v01 = ab[srcOffset + y1 * fromW + x0]
                    let v11 = ab[srcOffset + y1 * fromW + x1]

                    let v = v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) +
                            v01 * (1 - fx) * fy + v11 * fx * fy
                    result[dstOffset + y * toW + x] = v
                }
            }
        }
        return result
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

enum ColorizerError: LocalizedError {
    case invalidImage, modelNotLoaded, predictionFailed
    var errorDescription: String? {
        switch self {
        case .invalidImage: return "Invalid image"
        case .modelNotLoaded: return "Model not loaded"
        case .predictionFailed: return "Colorization failed"
        }
    }
}
