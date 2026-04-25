import CoreML
import CoreImage
import UIKit

enum PixelizerError: LocalizedError {
    case modelNotFound
    case invalidImage
    case predictionFailed

    var errorDescription: String? {
        switch self {
        case .modelNotFound: return "Pixelization model not found"
        case .invalidImage: return "Failed to process image"
        case .predictionFailed: return "Prediction failed"
        }
    }
}

// MARK: - Presets

/// A named pixel-art style. `cellSize` = grid chunkiness. `palette` = optional
/// list of 0xRRGGBB colors to snap every cell to.
struct PixelArtPreset {
    let id: String
    let name: String
    let systemImage: String
    let cellSize: Int
    let palette: [UInt32]?

    // All presets default to cellSize 4 — the palette is what distinguishes
    // each mode, chunkiness is user-tuned via the slider.
    static let all: [PixelArtPreset] = [
        PixelArtPreset(id: "off",     name: "Off",      systemImage: "circle",                  cellSize: 4, palette: nil),
        PixelArtPreset(id: "gameboy", name: "Game Boy", systemImage: "gamecontroller",          cellSize: 4, palette: PixelArtPalettes.gameBoy),
        PixelArtPreset(id: "nes",     name: "NES",      systemImage: "gamecontroller.fill",     cellSize: 4, palette: PixelArtPalettes.nes),
        PixelArtPreset(id: "pico8",   name: "Pico-8",   systemImage: "square.stack.3d.up.fill", cellSize: 4, palette: PixelArtPalettes.pico8),
        PixelArtPreset(id: "c64",     name: "C64",      systemImage: "desktopcomputer",         cellSize: 4, palette: PixelArtPalettes.c64),
    ]
}

enum PixelArtPalettes {
    static let gameBoy: [UInt32] = [
        0x9BBC0F, 0x8BAC0F, 0x306230, 0x0F380F,
    ]
    static let pico8: [UInt32] = [
        0x000000, 0x1D2B53, 0x7E2553, 0x008751,
        0xAB5236, 0x5F574F, 0xC2C3C7, 0xFFF1E8,
        0xFF004D, 0xFFA300, 0xFFEC27, 0x00E436,
        0x29ADFF, 0x83769C, 0xFF77A8, 0xFFCCAA,
    ]
    static let c64: [UInt32] = [
        0x000000, 0xFFFFFF, 0x68372B, 0x70A4B2,
        0x6F3D86, 0x588D43, 0x352879, 0xB8C76F,
        0x6F4F25, 0x433900, 0x9A6759, 0x444444,
        0x6C6C6C, 0x9AD284, 0x6C5EB5, 0x959595,
    ]
    static let nes: [UInt32] = [
        0x7C7C7C, 0x0000FC, 0x0000BC, 0x4428BC,
        0x940084, 0xA80020, 0xA81000, 0x881400,
        0x503000, 0x007800, 0x006800, 0x005800, 0x004058,
        0xBCBCBC, 0x0078F8, 0x0058F8, 0x6844FC,
        0xD800CC, 0xE40058, 0xF83800, 0xE45C10,
        0xAC7C00, 0x00B800, 0x00A800, 0x00A844, 0x008888,
        0xF8F8F8, 0x3CBCFC, 0x6888FC, 0x9878F8,
        0xF878F8, 0xF85898, 0xF87858, 0xFCA044,
        0xF8B800, 0xB8F818, 0x58D854, 0x58F898,
        0x00E8D8, 0x787878,
        0xFCFCFC, 0xA4E4FC, 0xB8B8F8, 0xD8B8F8,
        0xF8B8F8, 0xF8A4C0, 0xF0D0B0, 0xFCE0A8,
        0xF8D878, 0xD8F878, 0xB8F8B8, 0xB8F8D8,
        0x00FCFC, 0xF8D8F8,
    ]
}

// MARK: - Pixelizer

enum Pixelizer {
    static let inputSize = 512

    /// Matches the upstream `test_pro.py` factor (`inputSize * 4 / cellSize`).
    /// cellSize <= 4 keeps native resolution.
    static func preBlurTargetSize(for cellSize: Int) -> Int {
        if cellSize <= 4 { return inputSize }
        return max(96, min(inputSize, inputSize * 4 / cellSize))
    }

    /// Run the network and return the raw 512×512 pixelized CGImage.
    /// `preBlurTarget` should come from `preBlurTargetSize(for:)` — pass
    /// `inputSize` (= 512) for no blur.
    static func runModel(on image: UIImage, preBlurTarget: Int = inputSize) async throws -> CGImage {
        let fixed = image.normalizedOrientation()
        guard let cgImage = fixed.cgImage else { throw PixelizerError.invalidImage }

        let blurred: CGImage = preBlurTarget < inputSize
            ? (resizeCGImageBicubic(cgImage, to: preBlurTarget) ?? cgImage)
            : cgImage

        guard let inputBuffer = createPixelBuffer(
            from: blurred, width: inputSize, height: inputSize
        ) else { throw PixelizerError.invalidImage }

        let model = try loadModel()
        let input = try MLDictionaryFeatureProvider(dictionary: ["image": inputBuffer])
        let output = try await model.prediction(from: input)
        guard let buffer = output.featureValue(for: "pixelized")?.imageBufferValue else {
            throw PixelizerError.predictionFailed
        }
        let ci = CIImage(cvPixelBuffer: buffer)
        guard let cg = CIContext(options: [.useSoftwareRenderer: false])
            .createCGImage(ci, from: ci.extent)
        else { throw PixelizerError.predictionFailed }
        return cg
    }

    static func pixelize(_ image: UIImage, preset: PixelArtPreset, cellSize: Int? = nil) async throws -> UIImage {
        let cs = cellSize ?? preset.cellSize
        let cg = try await runModel(on: image, preBlurTarget: preBlurTargetSize(for: cs))
        return postProcess(cg, cellSize: cs, palette: preset.palette) ?? UIImage(cgImage: cg)
    }

    private static func resizeCGImageBicubic(_ cg: CGImage, to size: Int) -> CGImage? {
        guard let ctx = CGContext(
            data: nil, width: size, height: size,
            bitsPerComponent: 8, bytesPerRow: size * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }
        ctx.interpolationQuality = .high
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: size, height: size))
        return ctx.makeImage()
    }

    /// Mean-sample → optional palette snap → NEAREST upscale. No edge overlay
    /// — source-resolution gradient detection adds stray lines in flat areas
    /// (texture noise), so we rely on the cells + palette for definition.
    static func postProcess(_ cg: CGImage, cellSize: Int, palette: [UInt32]?) -> UIImage? {
        let cs = max(1, cellSize)
        let gridW = cg.width / cs
        let gridH = cg.height / cs
        guard gridW > 0 && gridH > 0 else { return nil }
        let outW = gridW * cs
        let outH = gridH * cs
        let srcW = cg.width
        let srcH = cg.height

        guard let srcData = cg.dataProvider?.data,
              let srcPtr = CFDataGetBytePtr(srcData) else { return nil }
        let srcBPR = cg.bytesPerRow
        let srcBpp = cg.bitsPerPixel / 8

        var grid = [UInt8](repeating: 0, count: gridW * gridH * 3)
        grid.withUnsafeMutableBufferPointer { gbuf in
            pixelArtMeanSample(
                srcPtr: srcPtr, srcW: srcW, srcH: srcH,
                srcBPR: srcBPR, srcBpp: srcBpp,
                cs: cs, gridW: gridW, gridH: gridH,
                gbuf: gbuf.baseAddress!
            )
        }
        if let palette = palette, !palette.isEmpty {
            applyPalette(&grid, palette: palette)
        }

        let bytesPerRow = outW * 4
        var pixels = [UInt8](repeating: 0, count: bytesPerRow * outH)
        pixels.withUnsafeMutableBufferPointer { dstBuf in
            grid.withUnsafeBufferPointer { gbuf in
                pixelArtReplicate(
                    dst: dstBuf.baseAddress!,
                    gptr: gbuf.baseAddress!,
                    gridW: gridW, gridH: gridH,
                    cs: cs, bytesPerRow: bytesPerRow
                )
            }
        }

        let provider = CGDataProvider(data: Data(pixels) as CFData)!
        let space = CGColorSpaceCreateDeviceRGB()
        let bitmap = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        if let out = CGImage(
            width: outW, height: outH,
            bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: bytesPerRow,
            space: space, bitmapInfo: bitmap,
            provider: provider, decode: nil,
            shouldInterpolate: false, intent: .defaultIntent
        ) {
            return UIImage(cgImage: out)
        }
        return nil
    }

    // MARK: - Model loading

    private static func loadModel() throws -> MLModel {
        guard let resourcePath = Bundle.main.resourcePath,
              let items = try? FileManager.default.contentsOfDirectory(atPath: resourcePath)
        else { throw PixelizerError.modelNotFound }
        for item in items where item.hasSuffix(".mlmodelc") && item.contains("Pixelization") {
            let url = URL(fileURLWithPath: (resourcePath as NSString).appendingPathComponent(item))
            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndNeuralEngine
            return try MLModel(contentsOf: url, configuration: config)
        }
        throw PixelizerError.modelNotFound
    }

    // MARK: - Pixel buffer

    private static func createPixelBuffer(from cgImage: CGImage, width: Int, height: Int) -> CVPixelBuffer? {
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(
            kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
            [kCVPixelBufferCGImageCompatibilityKey: true,
             kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary,
            &pb
        )
        guard let buffer = pb else { return nil }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let ctx = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
                     | CGBitmapInfo.byteOrder32Little.rawValue
        ) else { return nil }
        ctx.interpolationQuality = .high
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        return buffer
    }
}

// MARK: - Sampling / replicate helpers

func pixelArtMeanSample(
    srcPtr: UnsafePointer<UInt8>,
    srcW: Int, srcH: Int,
    srcBPR: Int, srcBpp: Int,
    cs: Int, gridW: Int, gridH: Int,
    gbuf: UnsafeMutablePointer<UInt8>
) {
    let div: Int32 = Int32(cs * cs)
    DispatchQueue.concurrentPerform(iterations: gridH) { gy in
        for gx in 0..<gridW {
            var sumR: Int32 = 0, sumG: Int32 = 0, sumB: Int32 = 0
            for dy in 0..<cs {
                let sy: Int = min(srcH - 1, gy * cs + dy)
                let rowBase: Int = sy * srcBPR
                for dx in 0..<cs {
                    let sx: Int = min(srcW - 1, gx * cs + dx)
                    let sOff: Int = rowBase + sx * srcBpp
                    sumR += Int32(srcPtr[sOff])
                    sumG += Int32(srcPtr[sOff + 1])
                    sumB += Int32(srcPtr[sOff + 2])
                }
            }
            let gOff: Int = (gy * gridW + gx) * 3
            gbuf[gOff]     = UInt8(sumR / div)
            gbuf[gOff + 1] = UInt8(sumG / div)
            gbuf[gOff + 2] = UInt8(sumB / div)
        }
    }
}

func pixelArtReplicate(
    dst: UnsafeMutablePointer<UInt8>,
    gptr: UnsafePointer<UInt8>,
    gridW: Int, gridH: Int,
    cs: Int, bytesPerRow: Int
) {
    DispatchQueue.concurrentPerform(iterations: gridH) { gy in
        for gx in 0..<gridW {
            let gOff: Int = (gy * gridW + gx) * 3
            let r: UInt8 = gptr[gOff]
            let g: UInt8 = gptr[gOff + 1]
            let b: UInt8 = gptr[gOff + 2]
            for by in 0..<cs {
                let oy: Int = gy * cs + by
                var off: Int = oy * bytesPerRow + gx * cs * 4
                for _ in 0..<cs {
                    dst[off]     = r
                    dst[off + 1] = g
                    dst[off + 2] = b
                    dst[off + 3] = 255
                    off += 4
                }
            }
        }
    }
}

// MARK: - Palette snap

func applyPalette(_ buf: inout [UInt8], palette: [UInt32]) {
    let n = palette.count
    let pr: [Int16] = palette.map { Int16(($0 >> 16) & 0xFF) }
    let pg: [Int16] = palette.map { Int16(($0 >> 8) & 0xFF) }
    let pb: [Int16] = palette.map { Int16($0 & 0xFF) }
    let count = buf.count / 3
    buf.withUnsafeMutableBufferPointer { buf in
        let ptr = buf.baseAddress!
        pr.withUnsafeBufferPointer { prBuf in
            pg.withUnsafeBufferPointer { pgBuf in
                pb.withUnsafeBufferPointer { pbBuf in
                    let prp = prBuf.baseAddress!, pgp = pgBuf.baseAddress!, pbp = pbBuf.baseAddress!
                    DispatchQueue.concurrentPerform(iterations: count) { i in
                        let off = i * 3
                        let r = Int16(ptr[off]), g = Int16(ptr[off + 1]), b = Int16(ptr[off + 2])
                        var bestIdx = 0
                        var bestDist: Int32 = .max
                        for j in 0..<n {
                            let dr = Int32(r - prp[j]), dg = Int32(g - pgp[j]), db = Int32(b - pbp[j])
                            let d = dr*dr + dg*dg + db*db
                            if d < bestDist { bestDist = d; bestIdx = j }
                        }
                        ptr[off]     = UInt8(prp[bestIdx])
                        ptr[off + 1] = UInt8(pgp[bestIdx])
                        ptr[off + 2] = UInt8(pbp[bestIdx])
                    }
                }
            }
        }
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
