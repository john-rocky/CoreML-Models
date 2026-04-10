import SwiftUI
import PhotosUI
import CoreML

/// Generic image → image demo template.
/// Used by: RMBG, DDColor, SinSR, EfficientAD.
///
/// Manifest config `output_type` controls post-processing:
///   "mask"       — alpha mask, min-max normalized, applied over original (RMBG)
///   "lab_ab"     — LAB AB channels, combined with original L channel (DDColor)
///   "image"      — direct RGB output (default)
struct ImageInOutDemoView: View {
    let model: ModelEntry

    @State private var inputImage: UIImage?
    @State private var outputImage: UIImage?
    @State private var isProcessing = false
    @State private var status = ""
    @State private var processingTime: Double?
    @State private var item: PhotosPickerItem?
    @State private var showOriginal = false

    var body: some View {
        VStack(spacing: 0) {
            ZStack {
                if let img = showOriginal ? inputImage : outputImage ?? inputImage {
                    Image(uiImage: img).resizable().aspectRatio(contentMode: .fit)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                } else {
                    placeholder
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .padding(.horizontal)
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { _ in showOriginal = true }
                    .onEnded { _ in showOriginal = false }
            )

            controls
        }
        .onChange(of: item) { _, _ in loadAndRun() }
    }

    @ViewBuilder
    private var placeholder: some View {
        VStack(spacing: 12) {
            Image(systemName: "photo.on.rectangle.angled").font(.system(size: 60)).foregroundStyle(.secondary)
            Text("Select a photo to process").foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    @ViewBuilder
    private var controls: some View {
        VStack(spacing: 12) {
            if outputImage != nil {
                Text("Hold to see original").font(.caption2).foregroundStyle(.tertiary)
            }
            HStack {
                if let t = processingTime {
                    Text(String(format: "%.2fs", t)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                }
                Spacer()
                if isProcessing { ProgressView().controlSize(.small) }
                Text(status).font(.caption).foregroundStyle(.secondary)
            }
            PhotosPicker(selection: $item, matching: .images) {
                Label("Select Photo", systemImage: "photo.badge.plus").frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .disabled(isProcessing)
        }
        .padding()
    }

    private func loadAndRun() {
        guard let item else { return }
        isProcessing = true; status = "Loading…"
        Task {
            guard let data = try? await item.loadTransferable(type: Data.self),
                  let img = UIImage(data: data) else {
                await MainActor.run { isProcessing = false; status = "Failed to load" }
                return
            }
            await MainActor.run { inputImage = img; outputImage = nil }
            await runInference(on: img)
        }
    }

    private func runInference(on image: UIImage) async {
        await MainActor.run { status = "Compiling model…" }
        do {
            // SinSR: 3-model pipeline, skip single-model path
            if model.configString("output_type") == "sinsr" {
                let start = CFAbsoluteTimeGetCurrent()
                let result = try await processSinSRPipeline(image: image, inputSize: model.configInt("input_size") ?? 256)
                let elapsed = CFAbsoluteTimeGetCurrent() - start
                await MainActor.run {
                    outputImage = result ?? image; processingTime = elapsed
                    isProcessing = false; status = ""
                }
                return
            }

            let mlModel = try await ModelLoader.loadPrimary(for: model)
            await MainActor.run { status = "Processing…" }

            let inputSize = model.configInt("input_size") ?? 512
            guard let cgImage = ImageUtils.normalizeOrientation(image) else {
                await MainActor.run { isProcessing = false; status = "Image prep failed" }
                return
            }
            let origW = cgImage.width, origH = cgImage.height

            // Build input based on model's expected type
            let inputDesc = mlModel.modelDescription.inputDescriptionsByName
            let imageInput = inputDesc.first { $0.value.type == .image }
            let arrayInput = inputDesc.first { $0.value.type == .multiArray }

            let inputDict: [String: Any]

            if let imageInput {
                guard let pb = ImageUtils.pixelBuffer(from: cgImage, width: inputSize, height: inputSize) else {
                    await MainActor.run { isProcessing = false; status = "Image prep failed" }; return
                }
                inputDict = [imageInput.key: pb]
            } else if let arrayInput {
                let constraint = arrayInput.value.multiArrayConstraint
                let shape = constraint?.shape.map { $0.intValue } ?? [1, 3, inputSize, inputSize]
                let h = shape.count >= 4 ? shape[2] : inputSize
                let w = shape.count >= 4 ? shape[3] : inputSize

                let outputType = model.configString("output_type") ?? "image"
                let arr = try buildMultiArrayInput(cgImage: cgImage, shape: shape, w: w, h: h, outputType: outputType)
                inputDict = [arrayInput.key: MLFeatureValue(multiArray: arr)]
            } else {
                await MainActor.run { isProcessing = false; status = "Unknown input type" }; return
            }

            let start = CFAbsoluteTimeGetCurrent()
            let input = try MLDictionaryFeatureProvider(dictionary: inputDict)
            let output = try await mlModel.prediction(from: input)
            let elapsed = CFAbsoluteTimeGetCurrent() - start

            // Post-process based on output_type config
            let outputType = model.configString("output_type") ?? "image"
            let result: UIImage?

            switch outputType {
            case "mask":
                result = processMaskOutput(output: output, originalImage: cgImage, origW: origW, origH: origH, modelSize: inputSize)
            case "lab_ab":
                result = processLABABOutput(output: output, originalImage: cgImage, origW: origW, origH: origH, modelSize: inputSize)
            case "sinsr":
                result = nil  // handled above
            default:
                result = processImageOutput(output: output)
            }

            await MainActor.run {
                outputImage = result ?? image
                processingTime = elapsed
                isProcessing = false; status = ""
            }
        } catch {
            await MainActor.run { isProcessing = false; status = "Error: \(error.localizedDescription)" }
        }
    }

    // MARK: - Input building

    private func buildMultiArrayInput(cgImage: CGImage, shape: [Int], w: Int, h: Int, outputType: String) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        guard let pb = ImageUtils.pixelBuffer(from: cgImage, width: w, height: h) else { return arr }

        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        let base = CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self)
        let bpr = CVPixelBufferGetBytesPerRow(pb)
        let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)

        if outputType == "lab_ab" {
            // DDColor: convert RGB → LAB L channel → grayscale RGB (L with a=0,b=0 → sRGB)
            for y in 0..<h {
                for x in 0..<w {
                    let px = base + y * bpr + x * 4
                    let r = Float(px[2]) / 255.0
                    let g = Float(px[1]) / 255.0
                    let b = Float(px[0]) / 255.0
                    let l = srgbToL(r: r, g: g, b: b)
                    let (gr, gg, gb) = labToSrgb(l: l, a: 0, b: 0)
                    ptr[0 * h * w + y * w + x] = gr
                    ptr[1 * h * w + y * w + x] = gg
                    ptr[2 * h * w + y * w + x] = gb
                }
            }
        } else {
            // Default: BGRA → RGB normalized [0,1] in CHW layout
            for y in 0..<h {
                for x in 0..<w {
                    let px = base + y * bpr + x * 4
                    ptr[0 * h * w + y * w + x] = Float(px[2]) / 255.0
                    ptr[1 * h * w + y * w + x] = Float(px[1]) / 255.0
                    ptr[2 * h * w + y * w + x] = Float(px[0]) / 255.0
                }
            }
        }
        return arr
    }

    // MARK: - Output: direct image

    private func processImageOutput(output: MLFeatureProvider) -> UIImage? {
        for name in output.featureNames {
            guard let fv = output.featureValue(for: name) else { continue }
            if let pixBuf = fv.imageBufferValue {
                let ci = CIImage(cvPixelBuffer: pixBuf)
                if let cg = CIContext().createCGImage(ci, from: ci.extent) { return UIImage(cgImage: cg) }
            }
            if let arr = fv.multiArrayValue {
                return ImageUtils.imageFromMultiArray(arr)
            }
        }
        return nil
    }

    // MARK: - Output: alpha mask (RMBG)

    private func processMaskOutput(output: MLFeatureProvider, originalImage: CGImage, origW: Int, origH: Int, modelSize: Int) -> UIImage? {
        // Find the mask array
        guard let maskArr = output.featureNames.compactMap({ output.featureValue(for: $0)?.multiArrayValue }).first else { return nil }

        // Extract floats with min-max normalization
        var raw = ImageUtils.extractFloats(maskArr)
        let mi = raw.min() ?? 0, ma = raw.max() ?? 1
        let range = ma - mi
        if range > 1e-6 { for i in raw.indices { raw[i] = (raw[i] - mi) / range } }

        // Apply alpha mask at original resolution with bilinear interpolation
        var pixelData = [UInt8](repeating: 0, count: origW * origH * 4)
        guard let ctx = CGContext(data: &pixelData, width: origW, height: origH,
                                  bitsPerComponent: 8, bytesPerRow: origW * 4,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.draw(originalImage, in: CGRect(x: 0, y: 0, width: origW, height: origH))

        var result = [UInt8](repeating: 0, count: origW * origH * 4)
        for y in 0..<origH {
            let srcY = Float(y) * Float(modelSize) / Float(origH)
            let y0 = min(Int(srcY), modelSize - 1)
            let y1 = min(y0 + 1, modelSize - 1)
            let fy = srcY - Float(y0)
            for x in 0..<origW {
                let srcX = Float(x) * Float(modelSize) / Float(origW)
                let x0 = min(Int(srcX), modelSize - 1)
                let x1 = min(x0 + 1, modelSize - 1)
                let fx = srcX - Float(x0)

                let v00 = raw[y0 * modelSize + x0]
                let v10 = raw[y0 * modelSize + x1]
                let v01 = raw[y1 * modelSize + x0]
                let v11 = raw[y1 * modelSize + x1]
                let alpha = v00 * (1-fx) * (1-fy) + v10 * fx * (1-fy) + v01 * (1-fx) * fy + v11 * fx * fy
                let a = UInt8(clamping: Int(alpha * 255))

                let idx = (y * origW + x) * 4
                let af = Float(a) / 255.0
                result[idx]   = UInt8(clamping: Int(Float(pixelData[idx]) * af))
                result[idx+1] = UInt8(clamping: Int(Float(pixelData[idx+1]) * af))
                result[idx+2] = UInt8(clamping: Int(Float(pixelData[idx+2]) * af))
                result[idx+3] = a
            }
        }

        guard let outCtx = CGContext(data: &result, width: origW, height: origH,
                                     bitsPerComponent: 8, bytesPerRow: origW * 4,
                                     space: CGColorSpaceCreateDeviceRGB(),
                                     bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
              let cgImg = outCtx.makeImage() else { return nil }
        return UIImage(cgImage: cgImg)
    }

    // MARK: - Output: LAB AB channels (DDColor)

    private func processLABABOutput(output: MLFeatureProvider, originalImage: CGImage, origW: Int, origH: Int, modelSize: Int) -> UIImage? {
        guard let abArr = output.featureNames.compactMap({ output.featureValue(for: $0)?.multiArrayValue }).first else { return nil }

        // Extract AB channels [1,2,H,W]
        let ab = ImageUtils.extractFloats(abArr)
        let chSize = modelSize * modelSize

        // Extract original L channel in LAB space
        var origPixels = [UInt8](repeating: 0, count: origW * origH * 4)
        guard let ctx = CGContext(data: &origPixels, width: origW, height: origH,
                                  bitsPerComponent: 8, bytesPerRow: origW * 4,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
        ctx.draw(originalImage, in: CGRect(x: 0, y: 0, width: origW, height: origH))

        var origL = [Float](repeating: 0, count: origW * origH)
        for i in 0..<(origW * origH) {
            let r = Float(origPixels[i * 4]) / 255.0
            let g = Float(origPixels[i * 4 + 1]) / 255.0
            let b = Float(origPixels[i * 4 + 2]) / 255.0
            origL[i] = srgbToL(r: r, g: g, b: b)
        }

        // Bilinear resize AB from modelSize to origW×origH
        var abOrig = [Float](repeating: 0, count: 2 * origW * origH)
        for ch in 0..<2 {
            let srcOff = ch * chSize
            let dstOff = ch * origW * origH
            for y in 0..<origH {
                let sy = Float(y) * Float(modelSize) / Float(origH)
                let y0 = min(Int(sy), modelSize - 1); let y1 = min(y0 + 1, modelSize - 1)
                let fy = sy - Float(y0)
                for x in 0..<origW {
                    let sx = Float(x) * Float(modelSize) / Float(origW)
                    let x0 = min(Int(sx), modelSize - 1); let x1 = min(x0 + 1, modelSize - 1)
                    let fx = sx - Float(x0)
                    let v = ab[srcOff + y0*modelSize + x0] * (1-fx)*(1-fy) +
                            ab[srcOff + y0*modelSize + x1] * fx*(1-fy) +
                            ab[srcOff + y1*modelSize + x0] * (1-fx)*fy +
                            ab[srcOff + y1*modelSize + x1] * fx*fy
                    abOrig[dstOff + y * origW + x] = v
                }
            }
        }

        // Combine L + AB → RGB
        var resultPixels = [UInt8](repeating: 255, count: origW * origH * 4)
        let count = origW * origH
        for i in 0..<count {
            let (r, g, b) = labToSrgb(l: origL[i], a: abOrig[i], b: abOrig[count + i])
            resultPixels[i * 4]   = UInt8(clamping: Int(r * 255))
            resultPixels[i * 4 + 1] = UInt8(clamping: Int(g * 255))
            resultPixels[i * 4 + 2] = UInt8(clamping: Int(b * 255))
        }

        guard let outCtx = CGContext(data: &resultPixels, width: origW, height: origH,
                                     bitsPerComponent: 8, bytesPerRow: origW * 4,
                                     space: CGColorSpaceCreateDeviceRGB(),
                                     bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue),
              let cgImg = outCtx.makeImage() else { return nil }
        return UIImage(cgImage: cgImg)
    }

    // MARK: - SinSR 3-model pipeline (Encoder → Denoiser → Decoder)

    private func processSinSRPipeline(image: UIImage, inputSize: Int) async throws -> UIImage? {
        let h = inputSize, w = inputSize
        let upH = h * 4, upW = w * 4

        guard let lqImage = ImageUtils.resize(image, to: CGSize(width: w, height: h)),
              let lqUpImage = ImageUtils.resize(lqImage, to: CGSize(width: upW, height: upH)) else { return nil }

        // Image → [-1,1] NCHW float32
        func imageToArray(_ img: UIImage, width: Int, height: Int) -> MLMultiArray {
            let cg = img.cgImage!
            var pixels = [UInt8](repeating: 0, count: width * height * 4)
            let ctx = CGContext(data: &pixels, width: width, height: height,
                                bitsPerComponent: 8, bytesPerRow: width * 4,
                                space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)!
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
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
                    ptr[idx] = r; ptr[spatial + idx] = g; ptr[2 * spatial + idx] = b
                }
            }
            return array
        }

        let lqArray = imageToArray(lqImage, width: w, height: h)
        let lqUpArray = imageToArray(lqUpImage, width: upW, height: upH)

        // Step 1: Encode
        await MainActor.run { status = "Encoding…" }
        let encoderFile = model.files.first { $0.name.contains("Encoder") }?.name ?? model.files[0].name
        let encoder = try await ModelLoader.load(for: model, named: encoderFile)
        let encOut = try await encoder.prediction(from: MLDictionaryFeatureProvider(dictionary: ["image": lqUpArray]))
        guard let zYRaw = encOut.featureValue(for: "latent")?.multiArrayValue else { return nil }

        // Read latent via subscript (stride-safe)
        let zYShape = zYRaw.shape.map { $0.intValue }
        let zYCount = zYShape.reduce(1, *)
        var zY = [Float](repeating: 0, count: zYCount)
        if zYShape.count == 4 {
            var idx = 0
            for n in 0..<zYShape[0] { for c in 0..<zYShape[1] { for hh in 0..<zYShape[2] { for ww in 0..<zYShape[3] {
                zY[idx] = zYRaw[[n, c, hh, ww] as [NSNumber]].floatValue; idx += 1
            } } } }
        } else {
            for i in 0..<zYCount { zY[i] = zYRaw[i].floatValue }
        }

        // Step 2: Add noise
        let kappa: Float = 2.0, sqrtEtaT: Float = 0.99, normalizeStd: Float = 2.218
        let spatial = h * w
        var zT = [Float](repeating: 0, count: 3 * spatial)
        let noiseScale = kappa * sqrtEtaT
        for i in 0..<zT.count {
            let u1 = Float.random(in: Float.ulpOfOne...1)
            let u2 = Float.random(in: 0...1)
            zT[i] = zY[i] + noiseScale * sqrt(-2 * log(u1)) * cos(2 * .pi * u2)
        }

        // Step 3: Build denoiser input [scaled_zT | lq] as [1,6,H,W]
        let lqFlat: [Float] = {
            let ptr = lqArray.dataPointer.assumingMemoryBound(to: Float.self)
            return Array(UnsafeBufferPointer(start: ptr, count: lqArray.count))
        }()
        let invStd: Float = 1.0 / normalizeStd
        var denIn = [Float](repeating: 0, count: 6 * spatial)
        for i in 0..<(3 * spatial) { denIn[i] = zT[i] * invStd }
        for i in 0..<(3 * spatial) { denIn[3 * spatial + i] = lqFlat[i] }
        let denInArray = try MLMultiArray(shape: [1, 6, NSNumber(value: h), NSNumber(value: w)], dataType: .float32)
        let denInPtr = denInArray.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<denIn.count { denInPtr[i] = denIn[i] }

        // Step 4: Denoise (cpuOnly for Swin FP16 overflow)
        await MainActor.run { status = "Denoising…" }
        let denoiserFile = model.files.first { $0.name.contains("Denoiser") }?.name ?? model.files[1].name
        let denoiser = try await ModelLoader.load(for: model, named: denoiserFile)
        let denOut = try await denoiser.prediction(from: MLDictionaryFeatureProvider(dictionary: ["input": denInArray]))
        guard let predRaw = denOut.featureValue(for: "predicted_latent")?.multiArrayValue else { return nil }

        var pred = [Float](repeating: 0, count: predRaw.count)
        if predRaw.shape.count == 4 {
            let s = predRaw.shape.map { $0.intValue }
            var idx = 0
            for n in 0..<s[0] { for c in 0..<s[1] { for hh in 0..<s[2] { for ww in 0..<s[3] {
                pred[idx] = predRaw[[n, c, hh, ww] as [NSNumber]].floatValue; idx += 1
            } } } }
        } else {
            for i in 0..<pred.count { pred[i] = predRaw[i].floatValue }
        }

        // Step 5: Clamp + scale for decoder
        var decIn = [Float](repeating: 0, count: pred.count)
        for i in 0..<pred.count { decIn[i] = min(max(pred[i], -1), 1) }
        let decInArray = try MLMultiArray(shape: [1, 3, NSNumber(value: h), NSNumber(value: w)], dataType: .float32)
        let decInPtr = decInArray.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<decIn.count { decInPtr[i] = decIn[i] }

        // Step 6: Decode
        await MainActor.run { status = "Decoding…" }
        let decoderFile = model.files.first { $0.name.contains("Decoder") }?.name ?? model.files[2].name
        let decoder = try await ModelLoader.load(for: model, named: decoderFile)
        let decOut = try await decoder.prediction(from: MLDictionaryFeatureProvider(dictionary: ["latent": decInArray]))
        guard let srRaw = decOut.featureValue(for: "image")?.multiArrayValue else { return nil }

        // Read via subscript
        let srShape = srRaw.shape.map { $0.intValue }
        var sr = [Float](repeating: 0, count: srShape.reduce(1, *))
        if srShape.count == 4 {
            var idx = 0
            for n in 0..<srShape[0] { for c in 0..<srShape[1] { for hh in 0..<srShape[2] { for ww in 0..<srShape[3] {
                sr[idx] = srRaw[[n, c, hh, ww] as [NSNumber]].floatValue; idx += 1
            } } } }
        }

        // Convert [-1,1] NCHW to UIImage
        let outSpatial = upW * upH
        var pixels = [UInt8](repeating: 255, count: upW * upH * 4)
        for y in 0..<upH {
            for x in 0..<upW {
                let idx = y * upW + x
                let r = UInt8(min(max(sr[idx] * 0.5 + 0.5, 0), 1) * 255)
                let g = UInt8(min(max(sr[outSpatial + idx] * 0.5 + 0.5, 0), 1) * 255)
                let b = UInt8(min(max(sr[2 * outSpatial + idx] * 0.5 + 0.5, 0), 1) * 255)
                let pIdx = (y * upW + x) * 4
                pixels[pIdx] = r; pixels[pIdx+1] = g; pixels[pIdx+2] = b
            }
        }
        return ImageUtils.makeRGBA(pixels: pixels, width: upW, height: upH)
    }

    // MARK: - LAB ↔ sRGB color conversion

    private func srgbToL(r: Float, g: Float, b: Float) -> Float {
        func lin(_ c: Float) -> Float { c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4) }
        let rl = lin(r), gl = lin(g), bl = lin(b)
        let y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl
        let fy = y > 0.008856 ? pow(y, 1.0/3.0) : (7.787 * y + 16.0/116.0)
        return 116.0 * fy - 16.0
    }

    private func labToSrgb(l: Float, a: Float, b: Float) -> (Float, Float, Float) {
        let fy = (l + 16.0) / 116.0
        let fx = a / 500.0 + fy
        let fz = fy - b / 200.0

        func invF(_ t: Float) -> Float { t > 0.206893 ? t * t * t : (t - 16.0/116.0) / 7.787 }
        let x = invF(fx) * 0.95047
        let y = invF(fy) * 1.00000
        let z = invF(fz) * 1.08883

        let rl =  3.2404542 * x - 1.5371385 * y - 0.4985314 * z
        let gl = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
        let bl =  0.0556434 * x - 0.2040259 * y + 1.0572252 * z

        func gamma(_ c: Float) -> Float { c <= 0.0031308 ? 12.92 * c : 1.055 * pow(c, 1.0/2.4) - 0.055 }
        return (max(0, min(1, gamma(rl))), max(0, min(1, gamma(gl))), max(0, min(1, gamma(bl))))
    }
}
