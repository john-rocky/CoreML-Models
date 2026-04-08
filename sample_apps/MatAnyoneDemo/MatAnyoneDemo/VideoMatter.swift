import Accelerate
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

        // Pre-allocate per-frame scratch buffers so the inner loop never
        // hits the allocator. All vImage ops below run against these.
        let bgraRowBytes = engineW * 4
        let bgraPlaneSize = engineH * bgraRowBytes
        let monoPlaneSize = engineW * engineH
        let inputBGRA  = UnsafeMutablePointer<UInt8>.allocate(capacity: bgraPlaneSize)
        defer { inputBGRA.deallocate() }
        let planeR = UnsafeMutablePointer<UInt8>.allocate(capacity: monoPlaneSize)
        let planeG = UnsafeMutablePointer<UInt8>.allocate(capacity: monoPlaneSize)
        let planeB = UnsafeMutablePointer<UInt8>.allocate(capacity: monoPlaneSize)
        defer { planeR.deallocate(); planeG.deallocate(); planeB.deallocate() }
        let alphaPlane = UnsafeMutablePointer<UInt8>.allocate(capacity: monoPlaneSize)
        defer { alphaPlane.deallocate() }
        // Pre-render the solid background colour into a BGRA buffer once.
        // The alpha byte is set to 255 so vImageAlphaBlend treats it as opaque.
        let bgBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: bgraPlaneSize)
        defer { bgBuffer.deallocate() }
        if let bgFillCtx = CGContext(
            data: bgBuffer, width: engineW, height: engineH, bitsPerComponent: 8, bytesPerRow: bgraRowBytes,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) {
            let bgCG = ciContext.createCGImage(
                CIImage(color: backgroundColor)
                    .cropped(to: CGRect(x: 0, y: 0, width: engineW, height: engineH)),
                from: CGRect(x: 0, y: 0, width: engineW, height: engineH)
            )
            if let bgCG { bgFillCtx.draw(bgCG, in: CGRect(x: 0, y: 0, width: engineW, height: engineH)) }
        }
        var bgViBuf = vImage_Buffer(
            data: bgBuffer,
            height: vImagePixelCount(engineH),
            width: vImagePixelCount(engineW),
            rowBytes: bgraRowBytes
        )
        vImageOverwriteChannelsWithScalar_ARGB8888(255, &bgViBuf, &bgViBuf, 0x08, vImage_Flags(kvImageNoFlags))

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
            fillImageArray(
                cg: cg,
                dst: imageArray,
                scratchBGRA: inputBGRA,
                rowBytes: bgraRowBytes,
                planeR: planeR,
                planeG: planeG,
                planeB: planeB
            )

            // 4. Run MatAnyone (Vision-bootstrapped on the first frame).
            //    Frame 0 is processed via the seed step + a single
            //    `firstFramePred=true` pass — that mirrors the official
            //    `process_video` capture point at `ti = n_warmup` and goes
            //    through the FP16-stable `read_first` path. From frame 1
            //    onwards we use the regular memory-attention `read` path.
            let alpha: [Float]
            if frameIndex == 0 {
                let visionAlpha = try generatePersonMaskFloat(cgImage: cg, width: engineW, height: engineH)
                // Match the official `gen_dilate` preprocessing: binarise the
                // soft Vision alpha at 0.5 and dilate once with radius 8 so
                // the seed comfortably covers the subject. MatAnyone can
                // shrink the matte but cannot recover from a seed that misses
                // parts of the foreground.
                let seedFloat = binaryDilate(visionAlpha, width: engineW, height: engineH,
                                             threshold: 0.5, radius: 8)
                let seed = try makeSeedMaskMLArray(seedFloat, width: engineW, height: engineH)
                try await engine.initializeSeed(firstImage: imageArray, mask: seed)
                alpha = try await engine.step(image: imageArray, providedMask: nil, firstFramePred: true)
                seedMaskPreview = makeMaskPreviewImage(
                    seedFloat, width: engineW, height: engineH, rotateLeft: isPortrait
                )
            } else {
                alpha = try await engine.step(image: imageArray)
            }

            // 5. Composite at engine resolution and rotate back.
            let landscapePB = try makeOutputPixelBuffer(width: engineW, height: engineH)
            composite(
                input: cg,
                alpha: alpha,
                bgBuffer: bgBuffer,
                bgRowBytes: bgraRowBytes,
                alphaScratch: alphaPlane,
                into: landscapePB
            )
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

    /// Wrap an already-prepared `[0, 1]` alpha array in a (1, 1, H, W) MLMultiArray
    /// for the engine. Caller is responsible for any thresholding / dilation.
    private func makeSeedMaskMLArray(_ alpha: [Float], width: Int, height: Int) throws -> MLMultiArray {
        let mask = try MLMultiArray(shape: [1, 1, NSNumber(value: height), NSNumber(value: width)], dataType: .float32)
        let dst = mask.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<(width * height) { dst[i] = alpha[i] }
        return mask
    }

    /// Threshold a soft alpha to binary `{0, 1}` then dilate with a square
    /// max filter of the given radius. Approximates the official `gen_dilate`
    /// (`np.not_equal(alpha, 0)` then `cv2.dilate`).
    private func binaryDilate(_ src: [Float], width: Int, height: Int,
                              threshold: Float, radius: Int) -> [Float] {
        var bin = [Float](repeating: 0, count: width * height)
        for i in 0..<bin.count { bin[i] = src[i] > threshold ? 1 : 0 }
        guard radius > 0 else { return bin }
        return softDilate(bin, width: width, height: height, radius: radius)
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

    /// Render `cg` into the engine's planar (1, 3, H, W) float input tensor
    /// using vImage. The interleaved BGRA8 → 3 PlanarF deinterleave + scale
    /// runs in vectorised code instead of a per-pixel Swift loop.
    ///
    /// `scratchBGRA`, `planeR`, `planeG`, `planeB` are caller-owned buffers
    /// of size `w*h*4`, `w*h`, `w*h`, `w*h` respectively, reused across frames.
    private func fillImageArray(
        cg: CGImage,
        dst: MLMultiArray,
        scratchBGRA: UnsafeMutablePointer<UInt8>,
        rowBytes: Int,
        planeR: UnsafeMutablePointer<UInt8>,
        planeG: UnsafeMutablePointer<UInt8>,
        planeB: UnsafeMutablePointer<UInt8>
    ) {
        let w = MatAnyoneEngine.inputWidth
        let h = MatAnyoneEngine.inputHeight
        let spatial = w * h

        // 1. Render CGImage into the BGRA8 scratch buffer.
        guard let ctx = CGContext(
            data: scratchBGRA, width: w, height: h, bitsPerComponent: 8, bytesPerRow: rowBytes,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else { return }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))

        // 2. Wrap the engine input tensor as 3 PlanarF buffers (R, G, B).
        let dstPtr = dst.dataPointer.assumingMemoryBound(to: Float.self)
        let rowBytesF = w * MemoryLayout<Float>.size
        var rDst = vImage_Buffer(data: dstPtr,                            height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: rowBytesF)
        var gDst = vImage_Buffer(data: dstPtr.advanced(by: spatial),      height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: rowBytesF)
        var bDst = vImage_Buffer(data: dstPtr.advanced(by: 2 * spatial),  height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: rowBytesF)

        // 3. Extract each channel from the interleaved BGRA buffer into a
        //    Planar8, then convert each Planar8 → PlanarF mapping [0,255] → [0,1].
        //    BGRA byte order: channel 0 = B, 1 = G, 2 = R, 3 = A.
        var src = vImage_Buffer(data: scratchBGRA, height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: rowBytes)
        var rPlane = vImage_Buffer(data: planeR, height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: w)
        var gPlane = vImage_Buffer(data: planeG, height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: w)
        var bPlane = vImage_Buffer(data: planeB, height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: w)
        vImageExtractChannel_ARGB8888(&src, &bPlane, 0, vImage_Flags(kvImageNoFlags))
        vImageExtractChannel_ARGB8888(&src, &gPlane, 1, vImage_Flags(kvImageNoFlags))
        vImageExtractChannel_ARGB8888(&src, &rPlane, 2, vImage_Flags(kvImageNoFlags))
        vImageConvert_Planar8toPlanarF(&rPlane, &rDst, 1.0, 0.0, vImage_Flags(kvImageNoFlags))
        vImageConvert_Planar8toPlanarF(&gPlane, &gDst, 1.0, 0.0, vImage_Flags(kvImageNoFlags))
        vImageConvert_Planar8toPlanarF(&bPlane, &bDst, 1.0, 0.0, vImage_Flags(kvImageNoFlags))
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

    /// Alpha-composite the input frame over a pre-rendered background buffer
    /// using vImage. The model alpha is converted to a Planar8 mask, written
    /// into the foreground BGRA's alpha channel, then the BGR channels are
    /// premultiplied and `vImagePremultipliedAlphaBlend_BGRA8888` performs
    /// `out = fg.rgb * α + bg.rgb * (1 - α)` over the whole frame in one
    /// vectorised call (vImage has no non-premultiplied BGRA blender, so we
    /// premultiply first).
    ///
    /// `bgBuffer` must be a BGRA8 buffer at the same WxH with alpha = 255.
    /// `alphaScratch` is a w*h Planar8 scratch slot reused across frames.
    private func composite(
        input: CGImage,
        alpha: [Float],
        bgBuffer: UnsafeMutablePointer<UInt8>,
        bgRowBytes: Int,
        alphaScratch: UnsafeMutablePointer<UInt8>,
        into pb: CVPixelBuffer
    ) {
        let w = CVPixelBufferGetWidth(pb)
        let h = CVPixelBufferGetHeight(pb)
        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return }
        let bpr = CVPixelBufferGetBytesPerRow(pb)

        // 1. Render the foreground frame into the output PB (BGRA byte order;
        //    the byte at offset 3 is currently a "skip" placeholder).
        guard let fgCtx = CGContext(
            data: base, width: w, height: h, bitsPerComponent: 8, bytesPerRow: bpr,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else { return }
        fgCtx.draw(input, in: CGRect(x: 0, y: 0, width: w, height: h))

        // 2. Convert the model alpha [Float] in [0, 1] to a Planar8 in [0, 255].
        alpha.withUnsafeBufferPointer { fp in
            var srcF = vImage_Buffer(
                data: UnsafeMutableRawPointer(mutating: fp.baseAddress),
                height: vImagePixelCount(h),
                width: vImagePixelCount(w),
                rowBytes: w * MemoryLayout<Float>.size
            )
            var dst8 = vImage_Buffer(
                data: alphaScratch,
                height: vImagePixelCount(h),
                width: vImagePixelCount(w),
                rowBytes: w
            )
            vImageConvert_PlanarFtoPlanar8(&srcF, &dst8, 1.0, 0.0, vImage_Flags(kvImageNoFlags))
        }

        // 3. Stamp the alpha plane into the foreground BGRA buffer's A channel
        //    (offset 3 = bit 3 of the channel mask).
        var alphaPlaneBuf = vImage_Buffer(
            data: alphaScratch,
            height: vImagePixelCount(h),
            width: vImagePixelCount(w),
            rowBytes: w
        )
        var fgBuf = vImage_Buffer(
            data: base,
            height: vImagePixelCount(h),
            width: vImagePixelCount(w),
            rowBytes: bpr
        )
        vImageOverwriteChannels_ARGB8888(&alphaPlaneBuf, &fgBuf, &fgBuf, 0x08, vImage_Flags(kvImageNoFlags))

        // 4. Premultiply the BGR channels by the alpha. vImage has no
        //    non-premultiplied BGRA8888 blender, only the premultiplied one.
        //    `vImagePremultiplyData_RGBA8888` is the underlying function
        //    (the BGRA-named variant is a C macro that Swift doesn't import)
        //    and works on any 4-channel 8-bit layout where alpha is the last
        //    byte — that's true for both RGBA and our BGRA buffer.
        vImagePremultiplyData_RGBA8888(&fgBuf, &fgBuf, vImage_Flags(kvImageNoFlags))

        // 5. Premultiplied src-over-dst blend. The bg buffer already has
        //    alpha = 255 so it acts as an opaque "premultiplied" bottom; the
        //    output's alpha channel ends up at 255 automatically.
        var bgBuf = vImage_Buffer(
            data: bgBuffer,
            height: vImagePixelCount(h),
            width: vImagePixelCount(w),
            rowBytes: bgRowBytes
        )
        vImagePremultipliedAlphaBlend_BGRA8888(&fgBuf, &bgBuf, &fgBuf, vImage_Flags(kvImageNoFlags))
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
