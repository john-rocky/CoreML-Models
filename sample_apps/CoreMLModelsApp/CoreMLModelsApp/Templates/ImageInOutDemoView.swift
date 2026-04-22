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
    // pixel_art: cache the raw (pre cell_size) model output so the preset
    // picker only re-runs the cheap NEAREST resample + palette mapping.
    @State private var pixelArtRaw: CGImage?
    @State private var pixelArtPresetId: String = PixelArtPreset.all[0].id
    // nil = use the preset's default cell size; non-nil = user dragged the
    // slider. Reset to nil whenever the preset changes.
    @State private var pixelArtCellSizeOverride: Double?
    // User-selected pre-blur target (px). 512 = no blur. Smaller = more
    // abstracted network input. nil = derive from cellSize.
    @State private var pixelArtBlurOverride: Int?
    @StateObject private var session = ModelSession<MLModel>()

    private var outputType: String { model.configString("output_type") ?? "image" }
    private var pixelArtPreset: PixelArtPreset {
        PixelArtPreset.all.first { $0.id == pixelArtPresetId } ?? PixelArtPreset.all[0]
    }
    private var pixelArtCellSize: Int {
        Int(pixelArtCellSizeOverride ?? Double(pixelArtPreset.cellSize))
    }

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
                    if outputType == "pixel_art" && !showOriginal {
                        Image(uiImage: img).resizable().interpolation(.none).aspectRatio(contentMode: .fit)
                    } else {
                        Image(uiImage: img).resizable().aspectRatio(contentMode: .fit)
                    }
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

                if outputType == "pixel_art" && pixelArtRaw != nil {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 8) {
                            ForEach(PixelArtPreset.all, id: \.id) { preset in
                                Button {
                                    pixelArtPresetId = preset.id
                                } label: {
                                    VStack(spacing: 2) {
                                        Image(systemName: preset.systemImage).font(.body)
                                        Text(preset.name).font(.caption2)
                                    }
                                    .padding(.vertical, 6).padding(.horizontal, 10)
                                    .background(
                                        pixelArtPresetId == preset.id
                                            ? Color.accentColor.opacity(0.25)
                                            : Color(.systemGray6)
                                    )
                                    .cornerRadius(8)
                                }
                                .buttonStyle(.plain)
                            }
                        }
                        .padding(.horizontal, 4)
                    }

                    HStack(spacing: 10) {
                        Image(systemName: "square.grid.3x3")
                            .font(.caption).foregroundStyle(.secondary)
                        Slider(
                            value: Binding(
                                get: { Double(pixelArtCellSize) },
                                set: { pixelArtCellSizeOverride = $0 }
                            ),
                            in: 4...10, step: 1
                        ) { Text("Cell size") }
                        Text("\(pixelArtCellSize)")
                            .font(.caption.monospacedDigit())
                            .frame(width: 22, alignment: .trailing)
                    }

                    // Pre-blur (photo shrink) picker. Off = full-res network
                    // input; smaller targets trade fine detail for cleaner,
                    // more iconic palette cells.
                    Picker("Abstraction", selection: Binding(
                        get: { pixelArtBlurOverride ?? 0 },   // 0 == auto (derive from cs)
                        set: { pixelArtBlurOverride = $0 == 0 ? nil : $0 }
                    )) {
                        Text("Auto").tag(0)
                        Text("Off").tag(512)
                        Text("256").tag(256)
                        Text("128").tag(128)
                        Text("64").tag(64)
                        Text("32").tag(32)
                    }
                    .pickerStyle(.segmented)
                    .onChange(of: pixelArtBlurOverride) {
                        if let img = inputImage { Task { await runInference(on: img) } }
                    }

                    .onChange(of: pixelArtPresetId) {
                        // New preset → reset override & re-run inference so the
                        // pre-blur matches the preset's default cellSize.
                        pixelArtCellSizeOverride = nil
                        if let img = inputImage {
                            Task { await runInference(on: img) }
                        } else if let raw = pixelArtRaw {
                            outputImage = pixelArtPostProcess(
                                raw, cellSize: pixelArtCellSize, palette: pixelArtPreset.palette)
                        }
                    }
                    .onChange(of: pixelArtCellSizeOverride) {
                        // During drag: cheap palette re-snap only. The network
                        // re-run happens on slider release (onEditingChanged).
                        if let raw = pixelArtRaw {
                            outputImage = pixelArtPostProcess(
                                raw, cellSize: pixelArtCellSize, palette: pixelArtPreset.palette)
                        }
                    }
                }

                HStack(spacing: 12) {
                    PhotosPicker(selection: $item, matching: .images) {
                        Label("Select Photo", systemImage: "photo.badge.plus")
                    }.buttonStyle(.bordered).disabled(isProcessing)

                    if let output = outputImage {
                        if outputType == "mask" {
                            ShareLink(item: Image(uiImage: output), preview: SharePreview("Result", image: Image(uiImage: output))) {
                                Image(systemName: "square.and.arrow.up")
                            }.buttonStyle(.bordered)
                        }

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
        .onChange(of: item) { _, _ in
            pixelArtRaw = nil
            pixelArtCellSizeOverride = nil
            loadAndRun()
        }
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
                // pixel_art: pre-downsample the source based on cell size so the
                // fixed-512 network effectively sees a lower-resolution image
                // and makes its semantic abstraction at the user's chosen
                // chunkiness. Mimics upstream test_pro.py's input resize.
                let sourceForBuffer: CGImage = {
                    guard outputType == "pixel_art" else { return cgImage }
                    let target = pixelArtBlurOverride
                        ?? pixelArtPreBlurTarget(cellSize: pixelArtCellSize, inputSize: inputSize)
                    return target < inputSize
                        ? (resizeCGImageBicubic(cgImage, to: target) ?? cgImage)
                        : cgImage
                }()
                guard let pb = ImageUtils.pixelBuffer(from: sourceForBuffer, width: inputSize, height: inputSize) else {
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
            case "pixel_art":
                let raw = extractRawCGImage(output: output)
                await MainActor.run { pixelArtRaw = raw }
                result = raw.flatMap {
                    pixelArtPostProcess($0, cellSize: pixelArtCellSize,
                                        palette: pixelArtPreset.palette)
                }
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

    // MARK: - Output: pixel_art (Pixelization)

    private func extractRawCGImage(output: MLFeatureProvider) -> CGImage? {
        for name in output.featureNames {
            if let pb = output.featureValue(for: name)?.imageBufferValue {
                let ci = CIImage(cvPixelBuffer: pb)
                if let cg = CIContext(options: [.useSoftwareRenderer: false])
                    .createCGImage(ci, from: ci.extent) { return cg }
            }
        }
        return nil
    }

    /// Clean pixel-art rendering:
    ///   1. Mean-sample one color per `cs`×`cs` cell.
    ///   2. Palette-snap each cell (optional).
    ///   3. NEAREST upscale by `cs` into the final image.
    ///
    /// We deliberately do NOT run a separate edge-detection overlay. Source-
    /// resolution gradient detection picks up per-cell colour jitter and
    /// texture noise, sprinkling stray dark lines across flat areas
    /// ('line picture 'ちょろちょろ出る') — the chunky cells + limited palette
    /// already give enough silhouette definition on their own.
    private func pixelArtPostProcess(_ cg: CGImage, cellSize: Int, palette: [UInt32]?) -> UIImage? {
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
        return ImageUtils.makeRGBA(pixels: pixels, width: outW, height: outH)
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

// MARK: - Pixel art presets

/// A named pixel-art style. `cellSize` controls the grid resolution (larger =
/// chunkier). `palette` is an optional list of 0xRRGGBB colors to snap every
/// cell to — nil means "keep the generator's own colors".
struct PixelArtPreset {
    let id: String
    let name: String
    let systemImage: String
    let cellSize: Int
    let palette: [UInt32]?

    // All presets default to the network's native cellSize (4) — the palette
    // is what differentiates them. Users dial chunkiness via the slider; at
    // cs=4 the pre-blur is skipped and the network's own pixelization shows
    // through cleanest, which is what tends to read best across photos.
    static let all: [PixelArtPreset] = [
        PixelArtPreset(id: "off",     name: "Off",      systemImage: "circle",                  cellSize: 4, palette: nil),
        PixelArtPreset(id: "gameboy", name: "Game Boy", systemImage: "gamecontroller",          cellSize: 4, palette: PixelArtPalettes.gameBoy),
        PixelArtPreset(id: "nes",     name: "NES",      systemImage: "gamecontroller.fill",     cellSize: 4, palette: PixelArtPalettes.nes),
        PixelArtPreset(id: "pico8",   name: "Pico-8",   systemImage: "square.stack.3d.up.fill", cellSize: 4, palette: PixelArtPalettes.pico8),
        PixelArtPreset(id: "c64",     name: "C64",      systemImage: "desktopcomputer",         cellSize: 4, palette: PixelArtPalettes.c64),
    ]
}

enum PixelArtPalettes {
    // Game Boy DMG: 4 shades of olive-green.
    static let gameBoy: [UInt32] = [
        0x9BBC0F, 0x8BAC0F, 0x306230, 0x0F380F,
    ]

    // Pico-8 fantasy console: 16 colors.
    static let pico8: [UInt32] = [
        0x000000, 0x1D2B53, 0x7E2553, 0x008751,
        0xAB5236, 0x5F574F, 0xC2C3C7, 0xFFF1E8,
        0xFF004D, 0xFFA300, 0xFFEC27, 0x00E436,
        0x29ADFF, 0x83769C, 0xFF77A8, 0xFFCCAA,
    ]

    // Commodore 64: 16 colors (Pepto's well-known sRGB approximation).
    static let c64: [UInt32] = [
        0x000000, 0xFFFFFF, 0x68372B, 0x70A4B2,
        0x6F3D86, 0x588D43, 0x352879, 0xB8C76F,
        0x6F4F25, 0x433900, 0x9A6759, 0x444444,
        0x6C6C6C, 0x9AD284, 0x6C5EB5, 0x959595,
    ]

    // NES 2C02 PPU (Nintendulator NTSC approximation), 54 usable colors.
    static let nes: [UInt32] = [
        0x7C7C7C, 0x0000FC, 0x0000BC, 0x4428BC,
        0x940084, 0xA80020, 0xA81000, 0x881400,
        0x503000, 0x007800, 0x006800, 0x005800,
        0x004058,
        0xBCBCBC, 0x0078F8, 0x0058F8, 0x6844FC,
        0xD800CC, 0xE40058, 0xF83800, 0xE45C10,
        0xAC7C00, 0x00B800, 0x00A800, 0x00A844,
        0x008888,
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

/// Replicate each 3-byte grid cell into a `cs`×`cs` block of the output
/// RGBA buffer. Where `applyEdges` is true and the edge mask at the output
/// pixel's coordinate is set, write the dark RGB override instead.
/// Map the user's `cellSize` to the target resolution the photo should be
/// downsampled to before feeding a fixed-`inputSize` Pixelization network.
/// The paper's `test_pro.py` resizes the whole network input by cell_size so
/// the (fully-convolutional) generator makes its abstraction at that scale.
/// Our CoreML model is fixed-size, so we emulate the effect by shrinking the
/// source and letting CGContext resize it back up — the network sees a
/// lower-resolution image and produces cleaner coarse cells.
///
/// Matches the upstream `test_pro.py` factor (`inputSize * 4 / cellSize`).
/// A previous 2× boost to this was too radical — at cs=16 it shrank the
/// input to 64 px, destroying readability. The useful range is cs=4-8 in
/// practice; in that window the upstream formula gives 256-512 target,
/// which blurs texture enough to clean up palette cells without wiping
/// the subject. cellSize <= 4 keeps native resolution.
func pixelArtPreBlurTarget(cellSize: Int, inputSize: Int) -> Int {
    if cellSize <= 4 { return inputSize }
    return max(96, min(inputSize, inputSize * 4 / cellSize))
}

/// Redraw `cg` into a square `size`×`size` CGImage using .high interpolation.
func resizeCGImageBicubic(_ cg: CGImage, to size: Int) -> CGImage? {
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

/// Mean-sample a `gridW`×`gridH` buffer of RGB triplets from the source
/// image by averaging each `cs`×`cs` region.
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

/// NEAREST replicate each 3-byte grid cell into a `cs`×`cs` RGBA block.
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

/// Snap every RGB triplet in `buf` to the nearest palette color (Euclidean in
/// RGB). `buf` is tightly-packed 3 bytes per pixel (RGB, no alpha). Runs
/// concurrently — for 512×512 / 54-color NES it's ~50 ms on A-series CPUs.
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
