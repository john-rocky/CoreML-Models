import SwiftUI
import PhotosUI
import CoreML
import CoreImage

/// Image → image processing with model-specific UX.
/// - RMBG (mask): checkerboard transparency, Save PNG, Share
/// - DDColor (lab_ab): hold-to-compare gesture, overlay label
/// - SinSR (sinsr): tap-to-compare, dimension display, 3-model pipeline
/// - Default (image): simple before/after
struct ImageInOutDemoView: View {
    let model: ModelEntry

    @State private var inputImage: UIImage?
    @State private var outputImage: UIImage?
    @State private var isProcessing = false
    @State private var status = ""
    @State private var processingTime: Double?
    @State private var item: PhotosPickerItem?
    @State private var showOriginal = false
    @StateObject private var session = ModelSession<MLModel>()

    private var outputType: String { model.configString("output_type") ?? "image" }

    var body: some View {
        VStack(spacing: 0) {
            // Display area
            ZStack {
                if outputType == "mask", let output = outputImage, !showOriginal {
                    // RMBG: checkerboard behind transparent image
                    Canvas { ctx, size in
                        let tile: CGFloat = 20
                        for row in 0..<Int(size.height / tile) + 1 {
                            for col in 0..<Int(size.width / tile) + 1 {
                                let isLight = (row + col) % 2 == 0
                                ctx.fill(
                                    Path(CGRect(x: CGFloat(col) * tile, y: CGFloat(row) * tile, width: tile, height: tile)),
                                    with: .color(isLight ? Color(.systemGray6) : Color(.systemGray4))
                                )
                            }
                        }
                    }
                    Image(uiImage: output).resizable().aspectRatio(contentMode: .fit)
                } else if let img = showOriginal ? inputImage : outputImage ?? inputImage {
                    Image(uiImage: img).resizable().aspectRatio(contentMode: .fit)
                } else {
                    VStack(spacing: 12) {
                        Image(systemName: "photo.on.rectangle.angled").font(.system(size: 60)).foregroundStyle(.secondary)
                        Text("Select a photo to process").foregroundStyle(.secondary)
                    }
                }

                // DDColor: overlay label
                if outputType == "lab_ab" && outputImage != nil {
                    VStack {
                        Spacer()
                        Text(showOriginal ? "Original" : "Colorized")
                            .font(.caption.bold()).foregroundStyle(.white)
                            .padding(.horizontal, 12).padding(.vertical, 6)
                            .background(.ultraThinMaterial).clipShape(Capsule())
                            .padding(.bottom, 8)
                    }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .padding(.horizontal)
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { _ in if outputImage != nil { showOriginal = true } }
                    .onEnded { _ in showOriginal = false }
            )

            // Controls
            VStack(spacing: 8) {
                if outputImage != nil {
                    Text(outputType == "lab_ab" ? "Hold to see original" : "Hold to compare")
                        .font(.caption2).foregroundStyle(.tertiary)
                }

                HStack {
                    TimingsLabel(loadSec: session.loadTimeSec, inferSec: processingTime)
                    Spacer()
                    if isProcessing { ProgressView().controlSize(.small); Text(status).font(.caption).foregroundStyle(.secondary) }
                }

                HStack(spacing: 12) {
                    PhotosPicker(selection: $item, matching: .images) {
                        Label("Select Photo", systemImage: "photo.badge.plus")
                    }.buttonStyle(.bordered).disabled(isProcessing)

                    if outputType == "mask", let output = outputImage {
                        ShareLink(item: Image(uiImage: output), preview: SharePreview("Result", image: Image(uiImage: output))) {
                            Image(systemName: "square.and.arrow.up")
                        }.buttonStyle(.bordered)

                        Button {
                            UIImageWriteToSavedPhotosAlbum(output, nil, nil, nil)
                        } label: {
                            Image(systemName: "arrow.down.to.line")
                        }.buttonStyle(.bordered)
                    }
                }
            }
            .padding()
        }
        .task {
            // Eagerly start primary-model load. SinSR is a 3-model pipeline
            // where the primary file is the encoder (first `kind: model`
            // entry), so this warms the first step while the user picks
            // a photo.
            session.ensure { try await ModelLoader.loadPrimary(for: model) }
        }
        .onChange(of: item) { _, _ in loadAndRun() }
    }

    // MARK: - Load & Run

    private func loadAndRun() {
        guard let item else { return }
        isProcessing = true; status = "Loading…"
        Task {
            guard let data = try? await item.loadTransferable(type: Data.self),
                  let img = UIImage(data: data) else {
                await MainActor.run { isProcessing = false; status = "Failed to load" }; return
            }
            await MainActor.run { inputImage = img; outputImage = nil }
            await runInference(on: img)
        }
    }

    private func runInference(on image: UIImage) async {
        do {
            // SinSR: 3-model pipeline
            if outputType == "sinsr" {
                await MainActor.run { status = "Loading models…" }
                let start = CFAbsoluteTimeGetCurrent()
                let inputSize = model.configInt("input_size") ?? 256
                var result = try await processSinSRPipeline(image: image, inputSize: inputSize)
                if let r = result, let cg = ImageUtils.normalizeOrientation(image) {
                    result = restoreAspect(r, origW: cg.width, origH: cg.height, inputSize: inputSize)
                }
                let elapsed = CFAbsoluteTimeGetCurrent() - start
                await MainActor.run { outputImage = result; processingTime = elapsed; isProcessing = false; status = "" }
                return
            }

            await MainActor.run { status = session.loadTimeSec == nil ? "Loading model…" : "Processing…" }
            let mlModel = try await session.get()
            await MainActor.run { status = "Processing…" }

            let inputSize = model.configInt("input_size") ?? 512
            guard let cgImage = ImageUtils.normalizeOrientation(image) else {
                await MainActor.run { isProcessing = false; status = "Image error" }; return
            }
            let origW = cgImage.width, origH = cgImage.height

            // Build input
            let inputDesc = mlModel.modelDescription.inputDescriptionsByName
            let imageInput = inputDesc.first { $0.value.type == .image }
            let arrayInput = inputDesc.first { $0.value.type == .multiArray }

            let inputDict: [String: Any]
            if let imageInput {
                guard let pb = ImageUtils.pixelBuffer(from: cgImage, width: inputSize, height: inputSize) else {
                    await MainActor.run { isProcessing = false; status = "Prep failed" }; return
                }
                inputDict = [imageInput.key: pb]
            } else if let arrayInput {
                let constraint = arrayInput.value.multiArrayConstraint
                let shape = constraint?.shape.map { $0.intValue } ?? [1, 3, inputSize, inputSize]
                let h = shape.count >= 4 ? shape[2] : inputSize
                let w = shape.count >= 4 ? shape[3] : inputSize
                let arr = try buildMultiArrayInput(cgImage: cgImage, shape: shape, w: w, h: h)
                inputDict = [arrayInput.key: MLFeatureValue(multiArray: arr)]
            } else {
                await MainActor.run { isProcessing = false; status = "Unknown input" }; return
            }

            let start = CFAbsoluteTimeGetCurrent()
            let input = try MLDictionaryFeatureProvider(dictionary: inputDict)
            let output: MLFeatureProvider
            do {
                output = try await mlModel.prediction(from: input)
            } catch {
                // ANE inference can fail on certain device/model combinations;
                // reload the model restricted to CPU+GPU and retry.
                print("[ImageInOut] Prediction failed, retrying with cpuAndGPU: \(error.localizedDescription)")
                await MainActor.run { status = "Retrying without ANE…" }
                let file = model.files.first { ($0.kind ?? "model") == "model" } ?? model.files[0]
                let fallbackModel = try await ModelLoader.load(
                    modelId: model.id, fileName: file.name, computeUnits: .cpuAndGPU
                )
                output = try await fallbackModel.prediction(from: input)
            }

            let result: UIImage?
            switch outputType {
            case "mask":
                result = processMaskOutput(output: output, originalImage: cgImage, origW: origW, origH: origH, modelSize: inputSize)
            case "lab_ab":
                result = processLABABOutput(output: output, originalImage: cgImage, origW: origW, origH: origH, modelSize: inputSize)
            case "segmap":
                result = processSegmapOutput(output: output, originalImage: cgImage, origW: origW, origH: origH, modelSize: inputSize)
            default:
                if let r = processImageOutput(output: output) {
                    result = restoreAspect(r, origW: origW, origH: origH, inputSize: inputSize)
                } else {
                    result = nil
                }
            }
            // Measure inference + post-processing together — post-processing
            // is a meaningful fraction of wall time on large photos (RMBG mask
            // upsample, DDColor LAB→sRGB), and the user waits for both.
            let elapsed = CFAbsoluteTimeGetCurrent() - start

            await MainActor.run { outputImage = result ?? image; processingTime = elapsed; isProcessing = false; status = "" }
        } catch {
            await MainActor.run { isProcessing = false; status = "Error: \(error.localizedDescription)" }
        }
    }

    // MARK: - Input Building

    private func buildMultiArrayInput(cgImage: CGImage, shape: [Int], w: Int, h: Int) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        guard let pb = ImageUtils.pixelBuffer(from: cgImage, width: w, height: h) else { return arr }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        let base = CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self)
        let bpr = CVPixelBufferGetBytesPerRow(pb)
        let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)

        if outputType == "lab_ab" {
            for y in 0..<h { for x in 0..<w {
                let px = base + y * bpr + x * 4
                let r = Float(px[2]) / 255.0, g = Float(px[1]) / 255.0, b = Float(px[0]) / 255.0
                let l = srgbToL(r: r, g: g, b: b)
                let (gr, gg, gb) = labToSrgb(l: l, a: 0, b: 0)
                ptr[0 * h * w + y * w + x] = gr; ptr[1 * h * w + y * w + x] = gg; ptr[2 * h * w + y * w + x] = gb
            }}
        } else {
            for y in 0..<h { for x in 0..<w {
                let px = base + y * bpr + x * 4
                ptr[0 * h * w + y * w + x] = Float(px[2]) / 255.0
                ptr[1 * h * w + y * w + x] = Float(px[1]) / 255.0
                ptr[2 * h * w + y * w + x] = Float(px[0]) / 255.0
            }}
        }
        return arr
    }

    // MARK: - Aspect restoration

    // SR inputs are stretched to inputSize × inputSize, so model outputs are
    // also stretched. Resize to (origW, origH) times the upscale factor so the
    // displayed image keeps the photo's original aspect.
    private func restoreAspect(_ image: UIImage, origW: Int, origH: Int, inputSize: Int) -> UIImage? {
        guard origW > 0, origH > 0, inputSize > 0 else { return image }
        let outW = Int(image.size.width), outH = Int(image.size.height)
        let scaleX = Double(outW) / Double(inputSize)
        let scaleY = Double(outH) / Double(inputSize)
        let newW = max(1, Int((Double(origW) * scaleX).rounded()))
        let newH = max(1, Int((Double(origH) * scaleY).rounded()))
        if newW == outW && newH == outH { return image }
        return ImageUtils.resize(image, to: CGSize(width: newW, height: newH)) ?? image
    }

    // MARK: - Output: image

    private func processImageOutput(output: MLFeatureProvider) -> UIImage? {
        for name in output.featureNames {
            if let pb = output.featureValue(for: name)?.imageBufferValue {
                let ci = CIImage(cvPixelBuffer: pb)
                if let cg = CIContext().createCGImage(ci, from: ci.extent) { return UIImage(cgImage: cg) }
            }
            if let arr = output.featureValue(for: name)?.multiArrayValue {
                return ImageUtils.imageFromMultiArray(arr)
            }
        }
        return nil
    }

    // MARK: - Output: mask (RMBG)

    private func processMaskOutput(output: MLFeatureProvider, originalImage: CGImage, origW: Int, origH: Int, modelSize: Int) -> UIImage? {
        guard let maskArr = output.featureNames.compactMap({ output.featureValue(for: $0)?.multiArrayValue }).first else { return nil }
        var raw = ImageUtils.extractFloats(maskArr)
        let mi = raw.min() ?? 0, ma = raw.max() ?? 1, range = ma - mi
        if range > 1e-6 { for i in raw.indices { raw[i] = (raw[i] - mi) / range } }

        // Pack the model-resolution mask into an 8-bit grayscale CGImage, then
        // let CoreImage upscale it and blend against a transparent background.
        // GPU-accelerated bilinear upsample + composite replaces what used to
        // be a per-pixel Swift loop over the full photo (seconds on 12MP input).
        var mask8 = [UInt8](repeating: 0, count: modelSize * modelSize)
        for i in 0..<mask8.count { mask8[i] = UInt8(clamping: Int(raw[i] * 255)) }
        guard let provider = CGDataProvider(data: Data(mask8) as CFData),
              let maskCG = CGImage(
                  width: modelSize, height: modelSize,
                  bitsPerComponent: 8, bitsPerPixel: 8,
                  bytesPerRow: modelSize,
                  space: CGColorSpaceCreateDeviceGray(),
                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                  provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent
              ) else { return nil }

        let origCI = CIImage(cgImage: originalImage)
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
        guard let outCG = ciCtx.createCGImage(blended, from: extent) else { return nil }
        return UIImage(cgImage: outCG)
    }

    // MARK: - Output: LAB AB (DDColor)

    private func processLABABOutput(output: MLFeatureProvider, originalImage: CGImage, origW: Int, origH: Int, modelSize: Int) -> UIImage? {
        guard let abArr = output.featureNames.compactMap({ output.featureValue(for: $0)?.multiArrayValue }).first else { return nil }
        let ab = ImageUtils.extractFloats(abArr)
        let chSize = modelSize * modelSize

        var origPixels = [UInt8](repeating: 0, count: origW * origH * 4)
        guard let ctx = CGContext(data: &origPixels, width: origW, height: origH, bitsPerComponent: 8,
                                  bytesPerRow: origW * 4, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
        ctx.draw(originalImage, in: CGRect(x: 0, y: 0, width: origW, height: origH))

        var resultPixels = [UInt8](repeating: 255, count: origW * origH * 4)
        // Fan each row out to the concurrent pool — the LAB↔sRGB math is ~6
        // pow() calls per pixel, and serial Swift on a 12MP photo takes seconds.
        origPixels.withUnsafeBufferPointer { srcBuf in
            resultPixels.withUnsafeMutableBufferPointer { dstBuf in
                ab.withUnsafeBufferPointer { abBuf in
                    let src = srcBuf.baseAddress!
                    let dst = dstBuf.baseAddress!
                    let abPtr = abBuf.baseAddress!
                    DispatchQueue.concurrentPerform(iterations: origH) { y in
                        let sy = Float(y) * Float(modelSize) / Float(origH)
                        let y0 = min(Int(sy), modelSize-1), y1 = min(y0+1, modelSize-1)
                        let fy = sy - Float(y0)
                        let rowBase = y * origW
                        for x in 0..<origW {
                            let sx = Float(x) * Float(modelSize) / Float(origW)
                            let x0 = min(Int(sx), modelSize-1), x1 = min(x0+1, modelSize-1)
                            let fx = sx - Float(x0)
                            let w00 = (1-fx)*(1-fy), w10 = fx*(1-fy), w01 = (1-fx)*fy, w11 = fx*fy
                            let aOff = y0*modelSize+x0
                            let bOff = y0*modelSize+x1
                            let cOff = y1*modelSize+x0
                            let dOff = y1*modelSize+x1
                            let aVal = abPtr[aOff]*w00 + abPtr[bOff]*w10 + abPtr[cOff]*w01 + abPtr[dOff]*w11
                            let bVal = abPtr[chSize+aOff]*w00 + abPtr[chSize+bOff]*w10 + abPtr[chSize+cOff]*w01 + abPtr[chSize+dOff]*w11

                            let pi = (rowBase + x) * 4
                            let r = Float(src[pi]) / 255
                            let g = Float(src[pi+1]) / 255
                            let b = Float(src[pi+2]) / 255
                            let l = srgbToL(r: r, g: g, b: b)
                            let (rOut, gOut, bOut) = labToSrgb(l: l, a: aVal, b: bVal)
                            dst[pi]   = UInt8(clamping: Int(rOut * 255))
                            dst[pi+1] = UInt8(clamping: Int(gOut * 255))
                            dst[pi+2] = UInt8(clamping: Int(bOut * 255))
                        }
                    }
                }
            }
        }
        guard let outCtx = CGContext(data: &resultPixels, width: origW, height: origH, bitsPerComponent: 8,
                                     bytesPerRow: origW * 4, space: CGColorSpaceCreateDeviceRGB(),
                                     bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue),
              let cgImg = outCtx.makeImage() else { return nil }
        return UIImage(cgImage: cgImg)
    }

    // MARK: - Output: segmap (face-parsing)

    private func processSegmapOutput(output: MLFeatureProvider, originalImage: CGImage, origW: Int, origH: Int, modelSize: Int) -> UIImage? {
        guard let arr = output.featureNames.compactMap({ output.featureValue(for: $0)?.multiArrayValue }).first else { return nil }
        let shape = arr.shape.map { $0.intValue }
        // Expected [1, H, W] or [1, C, H, W] (argmax already done or need to do)
        let h: Int, w: Int
        let isArgmaxed: Bool
        if shape.count == 3 {
            h = shape[1]; w = shape[2]; isArgmaxed = true
        } else if shape.count == 4 && shape[1] > 1 {
            h = shape[2]; w = shape[3]; isArgmaxed = false
        } else {
            return nil
        }

        // Get class index per pixel
        var classMap = [Int](repeating: 0, count: h * w)
        if isArgmaxed {
            let strides = arr.strides.map { $0.intValue }
            for y in 0..<h {
                for x in 0..<w {
                    classMap[y * w + x] = Int(ImageUtils.readFloat(arr, at: y * strides[1] + x * strides[2]))
                }
            }
        } else {
            let c = shape[1]
            let strides = arr.strides.map { $0.intValue }
            for y in 0..<h {
                for x in 0..<w {
                    var maxVal: Float = -.greatestFiniteMagnitude
                    var maxIdx = 0
                    for ci in 0..<c {
                        let v = ImageUtils.readFloat(arr, at: ci * strides[1] + y * strides[2] + x * strides[3])
                        if v > maxVal { maxVal = v; maxIdx = ci }
                    }
                    classMap[y * w + x] = maxIdx
                }
            }
        }

        // Blend segmap over original image
        var origPixels = [UInt8](repeating: 0, count: origW * origH * 4)
        guard let ctx = CGContext(data: &origPixels, width: origW, height: origH, bitsPerComponent: 8,
                                  bytesPerRow: origW * 4, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
        ctx.draw(originalImage, in: CGRect(x: 0, y: 0, width: origW, height: origH))

        var result = [UInt8](repeating: 255, count: origW * origH * 4)
        let alpha: Float = 0.5
        let paletteCount = Self.segmapPalette.count
        origPixels.withUnsafeBufferPointer { srcBuf in
            result.withUnsafeMutableBufferPointer { dstBuf in
                classMap.withUnsafeBufferPointer { clsBuf in
                    let src = srcBuf.baseAddress!
                    let dst = dstBuf.baseAddress!
                    let cls = clsBuf.baseAddress!
                    DispatchQueue.concurrentPerform(iterations: origH) { y in
                        let sy = min(Int(Float(y) * Float(h) / Float(origH)), h - 1)
                        for x in 0..<origW {
                            let sx = min(Int(Float(x) * Float(w) / Float(origW)), w - 1)
                            let c = cls[sy * w + sx]
                            let (cr, cg, cb) = Self.segmapPalette[c % paletteCount]
                            let idx = (y * origW + x) * 4
                            dst[idx]   = UInt8(Float(src[idx])   * (1 - alpha) + Float(cr) * alpha)
                            dst[idx+1] = UInt8(Float(src[idx+1]) * (1 - alpha) + Float(cg) * alpha)
                            dst[idx+2] = UInt8(Float(src[idx+2]) * (1 - alpha) + Float(cb) * alpha)
                        }
                    }
                }
            }
        }
        return ImageUtils.makeRGBA(pixels: result, width: origW, height: origH)
    }

    // Face-parsing palette: background, skin, l_brow, r_brow, l_eye, r_eye, eyeglass,
    // l_ear, r_ear, earring, nose, mouth, u_lip, l_lip, neck, necklace, cloth, hair, hat
    private static let segmapPalette: [(UInt8, UInt8, UInt8)] = [
        (0, 0, 0),       // 0  background
        (255, 224, 189), // 1  skin
        (139, 90, 43),   // 2  left brow
        (139, 90, 43),   // 3  right brow
        (72, 118, 255),  // 4  left eye
        (72, 118, 255),  // 5  right eye
        (180, 180, 255), // 6  eyeglass
        (255, 182, 108), // 7  left ear
        (255, 182, 108), // 8  right ear
        (255, 215, 0),   // 9  earring
        (255, 127, 80),  // 10 nose
        (220, 20, 60),   // 11 mouth
        (199, 21, 133),  // 12 upper lip
        (199, 21, 133),  // 13 lower lip
        (210, 180, 140), // 14 neck
        (255, 215, 0),   // 15 necklace
        (100, 149, 237), // 16 cloth
        (139, 69, 19),   // 17 hair
        (160, 82, 45),   // 18 hat
    ]

    // MARK: - SinSR Pipeline

    private func processSinSRPipeline(image: UIImage, inputSize: Int) async throws -> UIImage? {
        let h = inputSize, w = inputSize, upH = h * 4, upW = w * 4
        guard let lqImage = ImageUtils.resize(image, to: CGSize(width: w, height: h)),
              let lqUpImage = ImageUtils.resize(lqImage, to: CGSize(width: upW, height: upH)) else { return nil }

        func imageToArray(_ img: UIImage, width: Int, height: Int) -> MLMultiArray {
            let cg = img.cgImage!
            var pixels = [UInt8](repeating: 0, count: width * height * 4)
            let ctx = CGContext(data: &pixels, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                                space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)!
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
            let spatial = height * width
            let array = try! MLMultiArray(shape: [1, 3, NSNumber(value: height), NSNumber(value: width)], dataType: .float32)
            let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
            for y in 0..<height { for x in 0..<width {
                let pIdx = (y * width + x) * 4
                let idx = y * width + x
                ptr[idx] = Float(pixels[pIdx + 2]) / 127.5 - 1.0
                ptr[spatial + idx] = Float(pixels[pIdx + 1]) / 127.5 - 1.0
                ptr[2 * spatial + idx] = Float(pixels[pIdx]) / 127.5 - 1.0
            }}
            return array
        }

        let lqArray = imageToArray(lqImage, width: w, height: h)
        let lqUpArray = imageToArray(lqUpImage, width: upW, height: upH)

        status = "Encoding…"
        let encFile = model.files.first { $0.name.contains("Encoder") }?.name ?? model.files[0].name
        let encoder = try await ModelLoader.load(for: model, named: encFile)
        let encOut = try await encoder.prediction(from: MLDictionaryFeatureProvider(dictionary: ["image": lqUpArray]))
        guard let zYRaw = encOut.featureValue(for: "latent")?.multiArrayValue else { return nil }
        let zY: [Float] = {
            let s = zYRaw.shape.map { $0.intValue }; var r = [Float](repeating: 0, count: s.reduce(1, *))
            if s.count == 4 { var i = 0; for n in 0..<s[0] { for c in 0..<s[1] { for hh in 0..<s[2] { for ww in 0..<s[3] {
                r[i] = zYRaw[[n,c,hh,ww] as [NSNumber]].floatValue; i += 1 } } } } } else {
                for i in 0..<r.count { r[i] = zYRaw[i].floatValue } }; return r }()

        let spatial = h * w
        var zT = [Float](repeating: 0, count: 3 * spatial)
        for i in 0..<zT.count {
            let u1 = Float.random(in: Float.ulpOfOne...1), u2 = Float.random(in: 0...1)
            zT[i] = zY[i] + 2.0 * 0.99 * sqrt(-2 * log(u1)) * cos(2 * .pi * u2)
        }

        let lqFlat = { let p = lqArray.dataPointer.assumingMemoryBound(to: Float.self); return Array(UnsafeBufferPointer(start: p, count: lqArray.count)) }()
        var denIn = [Float](repeating: 0, count: 6 * spatial)
        for i in 0..<(3 * spatial) { denIn[i] = zT[i] / 2.218 }
        for i in 0..<(3 * spatial) { denIn[3 * spatial + i] = lqFlat[i] }
        let denInArray = try MLMultiArray(shape: [1, 6, NSNumber(value: h), NSNumber(value: w)], dataType: .float32)
        memcpy(denInArray.dataPointer, denIn, denIn.count * 4)

        status = "Denoising…"
        let denFile = model.files.first { $0.name.contains("Denoiser") }?.name ?? model.files[1].name
        let denoiser = try await ModelLoader.load(for: model, named: denFile)
        let denOut = try await denoiser.prediction(from: MLDictionaryFeatureProvider(dictionary: ["input": denInArray]))
        guard let predRaw = denOut.featureValue(for: "predicted_latent")?.multiArrayValue else { return nil }
        var pred: [Float] = {
            let s = predRaw.shape.map { $0.intValue }; var r = [Float](repeating: 0, count: s.reduce(1, *))
            if s.count == 4 { var i = 0; for n in 0..<s[0] { for c in 0..<s[1] { for hh in 0..<s[2] { for ww in 0..<s[3] {
                r[i] = predRaw[[n,c,hh,ww] as [NSNumber]].floatValue; i += 1 } } } } } else {
                for i in 0..<r.count { r[i] = predRaw[i].floatValue } }; return r }()
        for i in 0..<pred.count { pred[i] = min(max(pred[i], -1), 1) }
        let decInArray = try MLMultiArray(shape: [1, 3, NSNumber(value: h), NSNumber(value: w)], dataType: .float32)
        memcpy(decInArray.dataPointer, pred, pred.count * 4)

        status = "Decoding…"
        let decFile = model.files.first { $0.name.contains("Decoder") }?.name ?? model.files[2].name
        let decoder = try await ModelLoader.load(for: model, named: decFile)
        let decOut = try await decoder.prediction(from: MLDictionaryFeatureProvider(dictionary: ["latent": decInArray]))
        guard let srRaw = decOut.featureValue(for: "image")?.multiArrayValue else { return nil }
        let sr: [Float] = {
            let s = srRaw.shape.map { $0.intValue }; var r = [Float](repeating: 0, count: s.reduce(1, *))
            if s.count == 4 { var i = 0; for n in 0..<s[0] { for c in 0..<s[1] { for hh in 0..<s[2] { for ww in 0..<s[3] {
                r[i] = srRaw[[n,c,hh,ww] as [NSNumber]].floatValue; i += 1 } } } } }; return r }()

        let outSpatial = upW * upH
        var pixels = [UInt8](repeating: 255, count: upW * upH * 4)
        for i in 0..<outSpatial {
            pixels[i*4] = UInt8(min(max(sr[i]*0.5+0.5, 0), 1) * 255)
            pixels[i*4+1] = UInt8(min(max(sr[outSpatial+i]*0.5+0.5, 0), 1) * 255)
            pixels[i*4+2] = UInt8(min(max(sr[2*outSpatial+i]*0.5+0.5, 0), 1) * 255)
        }
        return ImageUtils.makeRGBA(pixels: pixels, width: upW, height: upH)
    }

    // MARK: - LAB Color

    private func srgbToL(r: Float, g: Float, b: Float) -> Float {
        func lin(_ c: Float) -> Float { c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4) }
        let y = 0.2126729 * lin(r) + 0.7151522 * lin(g) + 0.0721750 * lin(b)
        return 116.0 * (y > 0.008856 ? pow(y, 1.0/3.0) : 7.787 * y + 16.0/116.0) - 16.0
    }

    private func labToSrgb(l: Float, a: Float, b: Float) -> (Float, Float, Float) {
        let fy = (l + 16) / 116, fx = a / 500 + fy, fz = fy - b / 200
        func invF(_ t: Float) -> Float { t > 0.206893 ? t*t*t : (t - 16.0/116.0) / 7.787 }
        let x = invF(fx) * 0.95047, y = invF(fy), z = invF(fz) * 1.08883
        let rl = 3.2404542*x - 1.5371385*y - 0.4985314*z
        let gl = -0.9692660*x + 1.8760108*y + 0.0415560*z
        let bl = 0.0556434*x - 0.2040259*y + 1.0572252*z
        func gamma(_ c: Float) -> Float { c <= 0.0031308 ? 12.92*c : 1.055*pow(c, 1/2.4) - 0.055 }
        return (max(0, min(1, gamma(rl))), max(0, min(1, gamma(gl))), max(0, min(1, gamma(bl))))
    }
}
