import CoreML
import Foundation
import UIKit

// MARK: - Model Loading

enum ModelLoader {
    enum LoadError: LocalizedError {
        case fileNotFound(String)
        case noModelFiles
        var errorDescription: String? {
            switch self {
            case .fileNotFound(let name): return "Model file '\(name)' not found"
            case .noModelFiles: return "No model files found"
            }
        }
    }

    /// Resolve on-disk name: strip archive extension (.zip, .tar.gz) if the file
    /// was downloaded as an archive and unpacked by DownloadManager.
    private static func resolveFileName(_ name: String) -> String {
        var resolved = name
        for ext in [".zip", ".tar.gz"] {
            if resolved.hasSuffix(ext) {
                resolved = String(resolved.dropLast(ext.count))
            }
        }
        return resolved
    }

    /// Compile and load a CoreML model from the hub download directory.
    static func load(modelId: String, fileName: String, computeUnits: MLComputeUnits = .all) async throws -> MLModel {
        let dir = Paths.modelDir(id: modelId)
        let resolved = resolveFileName(fileName)

        let config = MLModelConfiguration()
        config.computeUnits = computeUnits

        // 1. Try pre-compiled .mlmodelc
        let compiledName = (resolved as NSString).deletingPathExtension + ".mlmodelc"
        let compiledURL = dir.appendingPathComponent(compiledName)
        if FileManager.default.fileExists(atPath: compiledURL.path) {
            print("[ModelLoader] Loading compiled: \(compiledURL.lastPathComponent)")
            return try MLModel(contentsOf: compiledURL, configuration: config)
        }

        // 2. Try exact resolved name
        let packageURL = dir.appendingPathComponent(resolved)
        if FileManager.default.fileExists(atPath: packageURL.path) {
            print("[ModelLoader] Compiling: \(packageURL.lastPathComponent)")
            return try await compileAndCache(packageURL, compiledURL: compiledURL, config: config)
        }

        // 3. Fallback: recursive search for .mlpackage / .mlmodelc in model dir
        //    (handles zips that extract into nested folders)
        if let found = findModelFile(in: dir, matching: resolved) {
            print("[ModelLoader] Found via search: \(found.lastPathComponent)")
            if found.pathExtension == "mlmodelc" {
                return try MLModel(contentsOf: found, configuration: config)
            }
            return try await compileAndCache(found, compiledURL: compiledURL, config: config)
        }

        // Debug: list what's actually on disk
        let items = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
        print("[ModelLoader] File not found: '\(resolved)' in \(dir.lastPathComponent)/")
        print("[ModelLoader] Directory contents: \(items)")
        throw LoadError.fileNotFound(resolved)
    }

    private static func compileAndCache(_ source: URL, compiledURL: URL, config: MLModelConfiguration) async throws -> MLModel {
        let tempCompiled = try await MLModel.compileModel(at: source)
        // Move to persistent location next to the source
        if !FileManager.default.fileExists(atPath: compiledURL.path) {
            try? FileManager.default.moveItem(at: tempCompiled, to: compiledURL)
            return try MLModel(contentsOf: compiledURL, configuration: config)
        }
        return try MLModel(contentsOf: tempCompiled, configuration: config)
    }

    /// Recursively search a directory for a .mlpackage or .mlmodelc.
    /// Prefers an exact base-name match, then falls back to any model file found.
    private static func findModelFile(in dir: URL, matching hint: String) -> URL? {
        let fm = FileManager.default
        let baseName = (hint as NSString).deletingPathExtension  // e.g. "MoGe2_ViTB_Normal_504"
        let modelExts = Set(["mlpackage", "mlmodelc", "mlmodel"])

        guard let enumerator = fm.enumerator(at: dir, includingPropertiesForKeys: [.isDirectoryKey],
                                              options: [.skipsHiddenFiles]) else { return nil }
        var bestMatch: URL?
        for case let url as URL in enumerator {
            let ext = url.pathExtension
            guard modelExts.contains(ext) else { continue }
            let name = (url.lastPathComponent as NSString).deletingPathExtension
            if name == baseName { return url }  // exact match
            if bestMatch == nil { bestMatch = url }  // first found
        }
        return bestMatch
    }

    /// Load the primary model (first file with kind == "model").
    static func loadPrimary(for model: ModelEntry) async throws -> MLModel {
        let file = model.files.first { ($0.kind ?? "model") == "model" } ?? model.files[0]
        return try await load(
            modelId: model.id, fileName: file.name,
            computeUnits: parseComputeUnits(file.computeUnits)
        )
    }

    /// Load a model file by name.
    static func load(for model: ModelEntry, named name: String) async throws -> MLModel {
        guard let file = model.files.first(where: { $0.name == name }) else {
            throw LoadError.fileNotFound(name)
        }
        return try await load(
            modelId: model.id, fileName: name,
            computeUnits: parseComputeUnits(file.computeUnits)
        )
    }

    /// Search for a model file whose base name contains `substring` (case-insensitive).
    /// Useful for finding e.g. "*_encoder*" or "*_decoder*" inside an extracted archive.
    static func loadBySubstring(modelId: String, substring: String, computeUnits: MLComputeUnits = .all) async throws -> MLModel {
        let dir = Paths.modelDir(id: modelId)
        guard let url = findModelFileBySubstring(in: dir, substring: substring) else {
            throw LoadError.fileNotFound(substring)
        }
        let compiledName = (url.lastPathComponent as NSString).deletingPathExtension + ".mlmodelc"
        let compiledURL = dir.appendingPathComponent(compiledName)
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        if url.pathExtension == "mlmodelc" {
            return try MLModel(contentsOf: url, configuration: config)
        }
        if FileManager.default.fileExists(atPath: compiledURL.path) {
            return try MLModel(contentsOf: compiledURL, configuration: config)
        }
        return try await compileAndCache(url, compiledURL: compiledURL, config: config)
    }

    private static func findModelFileBySubstring(in dir: URL, substring: String) -> URL? {
        let fm = FileManager.default
        let modelExts = Set(["mlpackage", "mlmodelc", "mlmodel"])
        let lower = substring.lowercased()
        guard let enumerator = fm.enumerator(at: dir, includingPropertiesForKeys: [.isDirectoryKey],
                                              options: [.skipsHiddenFiles]) else { return nil }
        for case let url as URL in enumerator {
            guard modelExts.contains(url.pathExtension) else { continue }
            let name = (url.lastPathComponent as NSString).deletingPathExtension.lowercased()
            if name.contains(lower) { return url }
        }
        return nil
    }

    /// URL for a non-model file (vocab, merges, voices, etc.) in the model directory.
    static func auxFileURL(modelId: String, fileName: String) -> URL {
        Paths.modelDir(id: modelId).appendingPathComponent(fileName)
    }

    /// Return the URL of a compiled `.mlmodelc` for the first model file whose
    /// base name contains `substring`. Compiles the `.mlpackage` on disk if no
    /// cached `.mlmodelc` is present. Used by SAMKit's `SamModelRef`, which
    /// accepts only pre-compiled model URLs.
    static func compiledURL(modelId: String, substring: String) async throws -> URL {
        let dir = Paths.modelDir(id: modelId)
        guard let src = findModelFileBySubstring(in: dir, substring: substring) else {
            throw LoadError.fileNotFound(substring)
        }
        if src.pathExtension == "mlmodelc" { return src }

        let compiledURL = dir.appendingPathComponent(
            (src.lastPathComponent as NSString).deletingPathExtension + ".mlmodelc"
        )
        if FileManager.default.fileExists(atPath: compiledURL.path) { return compiledURL }

        let tempCompiled = try await MLModel.compileModel(at: src)
        if !FileManager.default.fileExists(atPath: compiledURL.path) {
            try? FileManager.default.moveItem(at: tempCompiled, to: compiledURL)
            return compiledURL
        }
        return tempCompiled
    }

    static func parseComputeUnits(_ str: String?) -> MLComputeUnits {
        switch str {
        case "cpuOnly": return .cpuOnly
        case "cpuAndGPU": return .cpuAndGPU
        case "cpuAndNeuralEngine": return .cpuAndNeuralEngine
        default: return .all
        }
    }
}

// MARK: - Config Convenience

extension ModelEntry {
    func configInt(_ key: String) -> Int? { demo.config?[key]?.value as? Int }
    func configDouble(_ key: String) -> Double? { demo.config?[key]?.value as? Double }
    func configString(_ key: String) -> String? { demo.config?[key]?.value as? String }
    func configBool(_ key: String) -> Bool? { demo.config?[key]?.value as? Bool }
    func configStringArray(_ key: String) -> [String]? { demo.config?[key]?.value as? [String] }
}

// MARK: - Image Utilities

enum ImageUtils {
    /// Create a 32BGRA CVPixelBuffer from a CGImage, resized to the given dimensions.
    static func pixelBuffer(from cgImage: CGImage, width: Int, height: Int) -> CVPixelBuffer? {
        let attrs: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
        ]
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                            kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
        guard let buf = pb else { return nil }

        CVPixelBufferLockBaseAddress(buf, [])
        defer { CVPixelBufferUnlockBaseAddress(buf, []) }

        guard let ctx = CGContext(
            data: CVPixelBufferGetBaseAddress(buf),
            width: width, height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buf),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else { return nil }

        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        return buf
    }

    static func pixelBuffer(from image: UIImage, width: Int, height: Int) -> CVPixelBuffer? {
        guard let cg = normalizeOrientation(image) else { return nil }
        return pixelBuffer(from: cg, width: width, height: height)
    }

    /// Letterbox a CGImage into a square of `size`, returning the buffer and the content rect.
    static func letterbox(_ cgImage: CGImage, size: Int) -> (CVPixelBuffer, CGRect)? {
        let w = cgImage.width, h = cgImage.height
        let scale = Float(size) / Float(max(w, h))
        let dstW = Int(Float(w) * scale), dstH = Int(Float(h) * scale)
        let padX = (size - dstW) / 2, padY = (size - dstH) / 2

        let attrs: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
        ]
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, size, size,
                            kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
        guard let buf = pb else { return nil }

        CVPixelBufferLockBaseAddress(buf, [])
        defer { CVPixelBufferUnlockBaseAddress(buf, []) }

        guard let ctx = CGContext(
            data: CVPixelBufferGetBaseAddress(buf),
            width: size, height: size,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buf),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else { return nil }

        ctx.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: size, height: size))
        ctx.draw(cgImage, in: CGRect(x: padX, y: padY, width: dstW, height: dstH))

        return (buf, CGRect(x: padX, y: padY, width: dstW, height: dstH))
    }

    /// Return a CGImage oriented for display. iPhone camera photos carry
    /// EXIF orientation metadata so the pixel buffer is stored in sensor
    /// orientation; for anything other than `.up` we redraw the image with
    /// the orientation baked in. Already-upright images (web, screenshots,
    /// `UIImagePNGRepresentation` output) are returned as-is to avoid a
    /// pointless re-render that would also drop the alpha channel.
    static func normalizeOrientation(_ image: UIImage) -> CGImage? {
        if image.imageOrientation == .up, let cg = image.cgImage {
            return cg
        }
        let size = image.size
        UIGraphicsBeginImageContextWithOptions(size, true, image.scale)
        image.draw(in: CGRect(origin: .zero, size: size))
        let result = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return result?.cgImage
    }

    /// Resize UIImage.
    static func resize(_ image: UIImage, to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, true, 1.0)
        image.draw(in: CGRect(origin: .zero, size: size))
        let resized = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return resized
    }

    // MARK: - MLMultiArray Extraction

    /// Extract a flat Float array from MLMultiArray, handling FP16.
    static func extractFloats(_ array: MLMultiArray) -> [Float] {
        let count = array.count
        var result = [Float](repeating: 0, count: count)
        let ptr = array.dataPointer
        if array.dataType == .float16 {
            let fp16 = ptr.assumingMemoryBound(to: Float16.self)
            for i in 0..<count { result[i] = Float(fp16[i]) }
        } else {
            let fp32 = ptr.assumingMemoryBound(to: Float.self)
            memcpy(&result, fp32, count * MemoryLayout<Float>.size)
        }
        return result
    }

    /// Read a single float from an MLMultiArray at a flat index.
    static func readFloat(_ array: MLMultiArray, at index: Int) -> Float {
        if array.dataType == .float16 {
            return Float(array.dataPointer.assumingMemoryBound(to: Float16.self)[index])
        }
        return array.dataPointer.assumingMemoryBound(to: Float.self)[index]
    }

    /// Read a float using multi-dimensional indices and strides.
    static func readFloat(_ array: MLMultiArray, indices: [Int]) -> Float {
        let strides = array.strides.map { $0.intValue }
        var offset = 0
        for i in 0..<indices.count { offset += indices[i] * strides[i] }
        return readFloat(array, at: offset)
    }

    // MARK: - MLMultiArray → Image

    /// Convert [1,3,H,W] or [1,H,W,3] float image (0..1) to UIImage.
    static func imageFromMultiArray(_ array: MLMultiArray) -> UIImage? {
        let shape = array.shape.map { $0.intValue }
        let strides = array.strides.map { $0.intValue }

        let height: Int, width: Int
        let cStride: Int, hStride: Int, wStride: Int

        if shape.count == 4 && shape[1] == 3 {
            height = shape[2]; width = shape[3]
            cStride = strides[1]; hStride = strides[2]; wStride = strides[3]
        } else if shape.count == 4 && shape[3] == 3 {
            height = shape[1]; width = shape[2]
            hStride = strides[1]; wStride = strides[2]; cStride = strides[3]
        } else if shape.count == 3 && shape[0] == 3 {
            height = shape[1]; width = shape[2]
            cStride = strides[0]; hStride = strides[1]; wStride = strides[2]
        } else {
            return nil
        }

        // ESRGAN/GFPGAN output is 2048×2048 and bigger; parallelizing rows
        // keeps per-pixel clamp+convert from stalling on one core. Hoist the
        // FP16/FP32 branch out of the inner loop so the hot path is a single
        // typed load per channel.
        let isFP16 = array.dataType == .float16
        let basePtr = array.dataPointer
        var pixels = [UInt8](repeating: 255, count: width * height * 4)
        pixels.withUnsafeMutableBufferPointer { dstBuf in
            let dst = dstBuf.baseAddress!
            DispatchQueue.concurrentPerform(iterations: height) { y in
                let yOff = y * hStride
                let rowBase = y * width
                if isFP16 {
                    let src = basePtr.assumingMemoryBound(to: Float16.self)
                    for x in 0..<width {
                        let xOff = x * wStride
                        let di = (rowBase + x) * 4
                        for c in 0..<3 {
                            let v = Float(src[c * cStride + yOff + xOff])
                            dst[di + c] = UInt8(clamping: Int(max(0, min(1, v)) * 255))
                        }
                    }
                } else {
                    let src = basePtr.assumingMemoryBound(to: Float.self)
                    for x in 0..<width {
                        let xOff = x * wStride
                        let di = (rowBase + x) * 4
                        for c in 0..<3 {
                            let v = src[c * cStride + yOff + xOff]
                            dst[di + c] = UInt8(clamping: Int(max(0, min(1, v)) * 255))
                        }
                    }
                }
            }
        }
        return makeRGBA(pixels: pixels, width: width, height: height)
    }

    /// Convert a 1-channel depth MLMultiArray to a turbo-colormap UIImage.
    static func heatmapFromDepth(_ values: [Float], width: Int, height: Int) -> UIImage? {
        var dMin: Float = .greatestFiniteMagnitude, dMax: Float = -.greatestFiniteMagnitude
        for v in values where v > 0 && v.isFinite { dMin = min(dMin, v); dMax = max(dMax, v) }
        if dMax <= dMin { dMax = dMin + 1 }
        let range = dMax - dMin

        var pixels = [UInt8](repeating: 255, count: width * height * 4)
        pixels.withUnsafeMutableBufferPointer { dstBuf in
            values.withUnsafeBufferPointer { srcBuf in
                let dst = dstBuf.baseAddress!
                let src = srcBuf.baseAddress!
                DispatchQueue.concurrentPerform(iterations: height) { y in
                    let rowBase = y * width
                    for x in 0..<width {
                        let i = rowBase + x
                        let t = max(0, min(1, (src[i] - dMin) / range))
                        let (r, g, b) = turboColormap(t)
                        let di = i * 4
                        dst[di] = r; dst[di+1] = g; dst[di+2] = b
                    }
                }
            }
        }
        return makeRGBA(pixels: pixels, width: width, height: height)
    }

    /// Normal map: [1,H,W,3] normals in [-1,1] → RGB UIImage.
    static func normalMapImage(_ array: MLMultiArray) -> UIImage? {
        let shape = array.shape.map { $0.intValue }
        let strides = array.strides.map { $0.intValue }
        guard shape.count == 4 && shape[3] == 3 else { return nil }
        let height = shape[1], width = shape[2]
        let hStride = strides[1], wStride = strides[2], cStride = strides[3]

        let isFP16 = array.dataType == .float16
        let basePtr = array.dataPointer
        var pixels = [UInt8](repeating: 255, count: width * height * 4)
        pixels.withUnsafeMutableBufferPointer { dstBuf in
            let dst = dstBuf.baseAddress!
            DispatchQueue.concurrentPerform(iterations: height) { y in
                let yOff = y * hStride
                let rowBase = y * width
                if isFP16 {
                    let src = basePtr.assumingMemoryBound(to: Float16.self)
                    for x in 0..<width {
                        let base = yOff + x * wStride
                        let di = (rowBase + x) * 4
                        for c in 0..<3 {
                            let v = Float(src[base + c * cStride])
                            dst[di + c] = UInt8(clamping: Int((v + 1) * 0.5 * 255))
                        }
                    }
                } else {
                    let src = basePtr.assumingMemoryBound(to: Float.self)
                    for x in 0..<width {
                        let base = yOff + x * wStride
                        let di = (rowBase + x) * 4
                        for c in 0..<3 {
                            let v = src[base + c * cStride]
                            dst[di + c] = UInt8(clamping: Int((v + 1) * 0.5 * 255))
                        }
                    }
                }
            }
        }
        return makeRGBA(pixels: pixels, width: width, height: height)
    }

    /// Apply a single-channel mask [1,1,H,W] as alpha over a source image.
    static func applyMask(_ mask: MLMultiArray, over source: UIImage, inputSize: Int) -> UIImage? {
        let shape = mask.shape.map { $0.intValue }
        let strides = mask.strides.map { $0.intValue }

        let mH: Int, mW: Int, hStride: Int, wStride: Int
        if shape.count == 3 {
            mH = shape[1]; mW = shape[2]; hStride = strides[1]; wStride = strides[2]
        } else if shape.count == 4 {
            let d = shape[1] == 1 ? 2 : 1
            mH = shape[d]; mW = shape[d+1]; hStride = strides[d]; wStride = strides[d+1]
        } else { return nil }

        // Read mask values with min-max normalization
        var raw = [Float](repeating: 0, count: mH * mW)
        var mn: Float = .greatestFiniteMagnitude, mx: Float = -.greatestFiniteMagnitude
        for y in 0..<mH {
            for x in 0..<mW {
                let v = readFloat(mask, at: y * hStride + x * wStride)
                raw[y * mW + x] = v; mn = min(mn, v); mx = max(mx, v)
            }
        }
        let range = mx - mn
        if range > 0 { for i in 0..<raw.count { raw[i] = (raw[i] - mn) / range } }

        guard let srcCG = normalizeOrientation(source) else { return nil }
        let w = srcCG.width, h = srcCG.height

        let renderer = UIGraphicsImageRenderer(size: CGSize(width: w, height: h))
        return renderer.image { ctx in
            ctx.cgContext.draw(srcCG, in: CGRect(x: 0, y: 0, width: w, height: h))
            // Apply alpha from mask via bilinear sampling
            let imgData = ctx.cgContext.makeImage()
            guard let data = imgData?.dataProvider?.data,
                  let srcPtr = CFDataGetBytePtr(data) else { return }

            var pixels = [UInt8](repeating: 0, count: w * h * 4)
            let bpr = 4 * w
            memcpy(&pixels, srcPtr, min(CFDataGetLength(data), pixels.count))

            for y in 0..<h {
                let my = Float(y) / Float(h) * Float(mH)
                let yi = min(Int(my), mH - 1)
                for x in 0..<w {
                    let mx2 = Float(x) / Float(w) * Float(mW)
                    let xi = min(Int(mx2), mW - 1)
                    let alpha = raw[yi * mW + xi]
                    let idx = y * bpr + x * 4
                    pixels[idx + 3] = UInt8(clamping: Int(alpha * 255))
                }
            }

            // Re-create image with alpha
            if let provider = CGDataProvider(data: Data(pixels) as CFData),
               let result = CGImage(
                   width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32,
                   bytesPerRow: bpr, space: CGColorSpaceCreateDeviceRGB(),
                   bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                   provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent
               ) {
                ctx.cgContext.clear(CGRect(x: 0, y: 0, width: w, height: h))
                ctx.cgContext.draw(result, in: CGRect(x: 0, y: 0, width: w, height: h))
            }
        }
    }

    // MARK: - Helpers

    static func makeRGBA(pixels: [UInt8], width: Int, height: Int) -> UIImage? {
        let data = Data(pixels)
        guard let provider = CGDataProvider(data: data as CFData),
              let cgImage = CGImage(
                  width: width, height: height,
                  bitsPerComponent: 8, bitsPerPixel: 32,
                  bytesPerRow: width * 4,
                  space: CGColorSpaceCreateDeviceRGB(),
                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                  provider: provider,
                  decode: nil, shouldInterpolate: true, intent: .defaultIntent
              ) else { return nil }
        return UIImage(cgImage: cgImage)
    }

    /// Turbo colormap polynomial approximation.
    static func turboColormap(_ t: Float) -> (UInt8, UInt8, UInt8) {
        let t = max(0, min(1, t))
        let r = 0.13572138 + t * (4.6153926 + t * (-42.66032 + t * (132.13108 + t * (-152.54834 + t * 59.28788))))
        let g = 0.09140261 + t * (2.1943493 + t * (4.838359 + t * (-30.56325 + t * (42.38775 + t * (-16.89543)))))
        let b = 0.1066733 + t * (12.750753 + t * (-60.51487 + t * (109.69287 + t * (-85.36345 + t * 23.18949))))
        return (UInt8(max(0, min(255, r * 255))), UInt8(max(0, min(255, g * 255))), UInt8(max(0, min(255, b * 255))))
    }

    /// Cosine similarity between two float vectors.
    static func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var dot: Float = 0, normA: Float = 0, normB: Float = 0
        for i in 0..<a.count {
            dot += a[i] * b[i]; normA += a[i] * a[i]; normB += b[i] * b[i]
        }
        let denom = sqrt(normA) * sqrt(normB)
        return denom > 0 ? dot / denom : 0
    }
}
