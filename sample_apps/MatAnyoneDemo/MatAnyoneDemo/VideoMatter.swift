import AVFoundation
import CoreImage
import CoreML
import UIKit
import Vision

/// Reads frames from an AVAsset, runs MatAnyone on each, and writes a
/// composited MP4 (foreground over a configurable background colour).
///
/// MatAnyone is bootstrapped from Vision's first-frame person mask, then
/// the official `InferenceCore` warmup loop is replicated (10 first-frame
/// passes plus a soft dilation of the seed mask).
///
/// The CoreML graph is locked to landscape 768×432. Portrait sources are
/// rotated to landscape before the pipeline and rotated back afterwards.
@MainActor
final class VideoMatter {
    private let engine: MatAnyoneEngine
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    struct Output {
        let videoURL: URL
        let firstFrameMask: UIImage?
    }

    init() throws {
        engine = try MatAnyoneEngine()
    }

    /// Returns the composited MP4 plus a preview of the seed mask used for
    /// MatAnyone (i.e. the Vision person mask, post dilation, in the
    /// original video orientation).
    /// `progress` is called with values in [0, 1] from the main actor.
    func process(asset: AVAsset,
                 backgroundColor: CIColor,
                 progress: @MainActor @escaping (Double) -> Void) async throws -> Output {

        // Reader
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw NSError(domain: "MatAnyone", code: 1, userInfo: [NSLocalizedDescriptionKey: "No video track"])
        }
        let nominalSize = try await track.load(.naturalSize)
        let transform = try await track.load(.preferredTransform)
        let displaySize = nominalSize.applying(transform)
        let displayW = abs(displaySize.width)
        let displayH = abs(displaySize.height)
        let isPortrait = displayH > displayW

        // The engine always processes landscape 768x432. For portrait input
        // we rotate into landscape pre-engine and back out post-engine.
        let engineW = MatAnyoneEngine.inputWidth   // 768
        let engineH = MatAnyoneEngine.inputHeight  // 432
        let outW = isPortrait ? engineH : engineW   // 432 portrait / 768 landscape
        let outH = isPortrait ? engineW : engineH   // 768 portrait / 432 landscape

        let nominalFps = (try? await track.load(.nominalFrameRate)) ?? 30.0
        let frameDuration = CMTime(value: 1, timescale: max(Int32(round(nominalFps)), 1))
        let durationCM = try await asset.load(.duration)
        let totalFrames = max(1, Int(round(CMTimeGetSeconds(durationCM) * Double(nominalFps))))

        let reader = try AVAssetReader(asset: asset)
        let readerSettings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        ]
        let readerOutput = AVAssetReaderTrackOutput(track: track, outputSettings: readerSettings)
        readerOutput.alwaysCopiesSampleData = false
        reader.add(readerOutput)
        guard reader.startReading() else {
            throw reader.error ?? NSError(domain: "MatAnyone", code: 2, userInfo: [NSLocalizedDescriptionKey: "AVAssetReader failed to start"])
        }

        // Writer (output kept in original orientation)
        let outURL = FileManager.default.temporaryDirectory.appendingPathComponent("matanyone_\(UUID().uuidString).mp4")
        try? FileManager.default.removeItem(at: outURL)
        let writer = try AVAssetWriter(outputURL: outURL, fileType: .mp4)
        let writerSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: outW,
            AVVideoHeightKey: outH,
        ]
        let writerInput = AVAssetWriterInput(mediaType: .video, outputSettings: writerSettings)
        writerInput.expectsMediaDataInRealTime = false
        let pbAdaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: writerInput,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: outW,
                kCVPixelBufferHeightKey as String: outH,
            ]
        )
        writer.add(writerInput)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)

        // Pre-allocated MLMultiArray input for the engine (landscape).
        let imageArray = try MLMultiArray(
            shape: [1, 3, NSNumber(value: engineH), NSNumber(value: engineW)],
            dataType: .float32
        )

        // Background composited at engine resolution then rotated alongside.
        let backgroundCG = ciContext.createCGImage(
            CIImage(color: backgroundColor)
                .cropped(to: CGRect(x: 0, y: 0, width: engineW, height: engineH)),
            from: CGRect(x: 0, y: 0, width: engineW, height: engineH)
        )

        var frameIndex = 0
        var seedMaskPreview: UIImage? = nil

        while reader.status == .reading {
            guard let sample = readerOutput.copyNextSampleBuffer() else { break }
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sample) else { continue }

            // 1. Apply preferredTransform so the image is display-oriented.
            var ci = CIImage(cvPixelBuffer: pixelBuffer)
                .oriented(forExifOrientation: exifOrientation(for: transform))
            // 2. For portrait input, rotate 90° clockwise into landscape so
            //    the engine sees a landscape frame.
            if isPortrait {
                ci = ci.oriented(.right)  // 90° CW: portrait -> landscape
            }
            // 3. Scale into the engine input size.
            let scaled = ci.transformed(by: CGAffineTransform(
                scaleX: CGFloat(engineW) / ci.extent.width,
                y: CGFloat(engineH) / ci.extent.height
            ))
            let renderRect = CGRect(x: 0, y: 0, width: engineW, height: engineH)
            guard let cg = ciContext.createCGImage(scaled, from: renderRect) else { continue }
            fillImageArray(cg: cg, dst: imageArray)

            // 4. Run MatAnyone (Vision-bootstrapped on the first frame).
            let alpha: [Float]
            if frameIndex == 0 {
                let visionAlpha = try generatePersonMaskFloat(cgImage: cg, width: engineW, height: engineH)
                let dilated = softDilate(visionAlpha, width: engineW, height: engineH, radius: 6)
                let seed = try makeSeedMaskMLArray(dilated, width: engineW, height: engineH)
                try await engine.initialize(firstImage: imageArray, mask: seed)
                alpha = try await engine.step(image: imageArray)
                seedMaskPreview = makeMaskPreviewImage(
                    dilated, width: engineW, height: engineH, rotateLeft: isPortrait
                )
            } else {
                alpha = try await engine.step(image: imageArray)
            }

            // 5. Composite at engine resolution and rotate back.
            let landscapePB = try makeOutputPixelBuffer(width: engineW, height: engineH)
            composite(input: cg, alpha: alpha, background: backgroundCG, into: landscapePB)
            let outPB: CVPixelBuffer
            if isPortrait {
                outPB = try rotateBuffer(landscapePB, by: .left, width: outW, height: outH)
            } else {
                outPB = landscapePB
            }

            let timestamp = CMTimeMultiply(frameDuration, multiplier: Int32(frameIndex))
            while !writerInput.isReadyForMoreMediaData {
                try await Task.sleep(nanoseconds: 5_000_000)
            }
            pbAdaptor.append(outPB, withPresentationTime: timestamp)

            frameIndex += 1
            let p = min(1.0, Double(frameIndex) / Double(totalFrames))
            await MainActor.run { progress(p) }
        }

        writerInput.markAsFinished()
        await writer.finishWriting()
        if writer.status != .completed {
            throw writer.error ?? NSError(domain: "MatAnyone", code: 3, userInfo: [NSLocalizedDescriptionKey: "AVAssetWriter failed"])
        }
        return Output(videoURL: outURL, firstFrameMask: seedMaskPreview)
    }

    // MARK: Vision person segmentation
    //
    // The "before" pane uses Vision's per-frame person segmentation as the
    // alpha — that's the same Vision API that bootstraps MatAnyone, so the
    // comparison shows exactly what video matting buys you over a per-frame
    // segmentation.

    private func generatePersonMaskFloat(cgImage: CGImage, width: Int, height: Int) throws -> [Float] {
        let request = VNGeneratePersonSegmentationRequest()
        request.qualityLevel = .accurate
        request.outputPixelFormat = kCVPixelFormatType_OneComponent8
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try handler.perform([request])

        var out = [Float](repeating: 0, count: width * height)
        guard let result = request.results?.first,
              let pixelBuffer = result.pixelBuffer as CVPixelBuffer? else {
            for i in 0..<out.count { out[i] = 1.0 }
            return out
        }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        let mw = CVPixelBufferGetWidth(pixelBuffer)
        let mh = CVPixelBufferGetHeight(pixelBuffer)
        let bpr = CVPixelBufferGetBytesPerRow(pixelBuffer)
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            for i in 0..<out.count { out[i] = 1.0 }
            return out
        }
        let bytes = base.assumingMemoryBound(to: UInt8.self)
        for y in 0..<height {
            let sy = min(mh - 1, y * mh / height)
            for x in 0..<width {
                let sx = min(mw - 1, x * mw / width)
                out[y * width + x] = Float(bytes[sy * bpr + sx]) / 255.0
            }
        }
        return out
    }

    /// Build the MatAnyone seed mask from a Vision alpha. We dilate slightly
    /// (matches the official `gen_dilate` behaviour at r=10/2 effective) so
    /// the seed covers hair/silhouette edges that Vision tends to under-cover.
    private func makeSeedMaskMLArray(_ alpha: [Float], width: Int, height: Int) throws -> MLMultiArray {
        let mask = try MLMultiArray(shape: [1, 1, NSNumber(value: height), NSNumber(value: width)], dataType: .float32)
        let dst = mask.dataPointer.assumingMemoryBound(to: Float.self)
        let dilated = softDilate(alpha, width: width, height: height, radius: 6)
        for i in 0..<(width * height) { dst[i] = dilated[i] }
        return mask
    }

    /// Separable max-filter dilation that preserves the [0,1] alpha values.
    private func softDilate(_ src: [Float], width: Int, height: Int, radius: Int) -> [Float] {
        guard radius > 0 else { return src }
        var horiz = [Float](repeating: 0, count: width * height)
        for y in 0..<height {
            let row = y * width
            for x in 0..<width {
                let lo = max(0, x - radius)
                let hi = min(width - 1, x + radius)
                var m: Float = 0
                for k in lo...hi { let v = src[row + k]; if v > m { m = v } }
                horiz[row + x] = m
            }
        }
        var out = [Float](repeating: 0, count: width * height)
        for x in 0..<width {
            for y in 0..<height {
                let lo = max(0, y - radius)
                let hi = min(height - 1, y + radius)
                var m: Float = 0
                for k in lo...hi { let v = horiz[k * width + x]; if v > m { m = v } }
                out[y * width + x] = m
            }
        }
        return out
    }

    // MARK: Mask preview

    /// Convert a [0,1] alpha into a grayscale UIImage. If `rotateLeft` is
    /// true (portrait sources) the image is rotated 90° CCW so the mask
    /// matches the displayed video orientation.
    private func makeMaskPreviewImage(_ alpha: [Float], width: Int, height: Int,
                                      rotateLeft: Bool) -> UIImage? {
        var pixels = [UInt8](repeating: 255, count: width * height * 4)
        for i in 0..<(width * height) {
            let v = UInt8(min(255, max(0, alpha[i] * 255)))
            pixels[i * 4]     = v
            pixels[i * 4 + 1] = v
            pixels[i * 4 + 2] = v
        }
        guard let provider = CGDataProvider(data: Data(pixels) as CFData),
              let cg = CGImage(
                width: width, height: height,
                bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
              ) else { return nil }
        if rotateLeft {
            let ci = CIImage(cgImage: cg).oriented(.left)
            if let rotated = ciContext.createCGImage(ci, from: ci.extent) {
                return UIImage(cgImage: rotated)
            }
        }
        return UIImage(cgImage: cg)
    }

    // MARK: Image <-> array

    private func fillImageArray(cg: CGImage, dst: MLMultiArray) {
        let w = MatAnyoneEngine.inputWidth
        let h = MatAnyoneEngine.inputHeight
        var pixels = [UInt8](repeating: 0, count: w * h * 4)
        let ctx = CGContext(
            data: &pixels, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        )!
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))

        let spatial = w * h
        let p = dst.dataPointer.assumingMemoryBound(to: Float.self)
        for y in 0..<h {
            for x in 0..<w {
                let pi = (y * w + x) * 4
                let b = Float(pixels[pi]) / 255.0
                let g = Float(pixels[pi + 1]) / 255.0
                let r = Float(pixels[pi + 2]) / 255.0
                let idx = y * w + x
                p[idx] = r
                p[spatial + idx] = g
                p[2 * spatial + idx] = b
            }
        }
    }

    private func makeOutputPixelBuffer(width: Int, height: Int) throws -> CVPixelBuffer {
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true,
        ]
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
        guard status == kCVReturnSuccess, let buf = pb else {
            throw NSError(domain: "MatAnyone", code: 4, userInfo: [NSLocalizedDescriptionKey: "CVPixelBufferCreate failed"])
        }
        return buf
    }

    private func composite(input: CGImage, alpha: [Float], background: CGImage?, into pb: CVPixelBuffer) {
        let w = CVPixelBufferGetWidth(pb)
        let h = CVPixelBufferGetHeight(pb)
        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return }
        let bpr = CVPixelBufferGetBytesPerRow(pb)
        let ctx = CGContext(
            data: base, width: w, height: h, bitsPerComponent: 8, bytesPerRow: bpr,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        )!
        if let bg = background {
            ctx.draw(bg, in: CGRect(x: 0, y: 0, width: w, height: h))
        }

        var fg = [UInt8](repeating: 0, count: w * h * 4)
        let inCtx = CGContext(
            data: &fg, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        )!
        inCtx.draw(input, in: CGRect(x: 0, y: 0, width: w, height: h))

        let bgBytes = base.assumingMemoryBound(to: UInt8.self)
        for y in 0..<h {
            for x in 0..<w {
                let aa = max(0, min(1, alpha[y * w + x]))
                let pi = (y * w + x) * 4
                let bgIdx = y * bpr + x * 4
                let fb = Float(bgBytes[bgIdx]) * (1 - aa) + Float(fg[pi]) * aa
                let fg_ = Float(bgBytes[bgIdx + 1]) * (1 - aa) + Float(fg[pi + 1]) * aa
                let fr = Float(bgBytes[bgIdx + 2]) * (1 - aa) + Float(fg[pi + 2]) * aa
                bgBytes[bgIdx]     = UInt8(min(255, max(0, fb)))
                bgBytes[bgIdx + 1] = UInt8(min(255, max(0, fg_)))
                bgBytes[bgIdx + 2] = UInt8(min(255, max(0, fr)))
                bgBytes[bgIdx + 3] = 255
            }
        }
    }

    /// Rotate a CVPixelBuffer using CIImage. `direction` matches CIImage's
    /// orientation enum: .left = 90° CCW, .right = 90° CW.
    private func rotateBuffer(_ src: CVPixelBuffer, by direction: CGImagePropertyOrientation,
                              width: Int, height: Int) throws -> CVPixelBuffer {
        let dst = try makeOutputPixelBuffer(width: width, height: height)
        let ci = CIImage(cvPixelBuffer: src).oriented(direction)
        ciContext.render(ci, to: dst)
        return dst
    }

    // MARK: Orientation

    private func exifOrientation(for transform: CGAffineTransform) -> Int32 {
        let a = transform.a, b = transform.b, c = transform.c, d = transform.d
        if a == 0  && b == 1  && c == -1 && d == 0  { return 6 } // 90 CW
        if a == 0  && b == -1 && c == 1  && d == 0  { return 8 } // 90 CCW
        if a == -1 && b == 0  && c == 0  && d == -1 { return 3 } // 180
        return 1
    }
}
