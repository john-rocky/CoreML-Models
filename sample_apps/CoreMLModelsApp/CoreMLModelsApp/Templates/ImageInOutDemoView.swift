import SwiftUI
import PhotosUI
import CoreML

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
                    if let t = processingTime {
                        Text(String(format: "%.2fs", t)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    }
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
                let result = try await processSinSRPipeline(image: image, inputSize: model.configInt("input_size") ?? 256)
                let elapsed = CFAbsoluteTimeGetCurrent() - start
                await MainActor.run { outputImage = result; processingTime = elapsed; isProcessing = false; status = "" }
                return
            }

            await MainActor.run { status = "Compiling model…" }
            let mlModel = try await ModelLoader.loadPrimary(for: model)
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
            let output = try await mlModel.prediction(from: input)
            let elapsed = CFAbsoluteTimeGetCurrent() - start

            let result: UIImage?
            switch outputType {
            case "mask":
                result = processMaskOutput(output: output, originalImage: cgImage, origW: origW, origH: origH, modelSize: inputSize)
            case "lab_ab":
                result = processLABABOutput(output: output, originalImage: cgImage, origW: origW, origH: origH, modelSize: inputSize)
            default:
                result = processImageOutput(output: output)
            }

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

        var pixelData = [UInt8](repeating: 0, count: origW * origH * 4)
        guard let ctx = CGContext(data: &pixelData, width: origW, height: origH, bitsPerComponent: 8,
                                  bytesPerRow: origW * 4, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.draw(originalImage, in: CGRect(x: 0, y: 0, width: origW, height: origH))

        var result = [UInt8](repeating: 0, count: origW * origH * 4)
        for y in 0..<origH {
            let srcY = Float(y) * Float(modelSize) / Float(origH)
            let y0 = min(Int(srcY), modelSize - 1), y1 = min(y0 + 1, modelSize - 1)
            let fy = srcY - Float(y0)
            for x in 0..<origW {
                let srcX = Float(x) * Float(modelSize) / Float(origW)
                let x0 = min(Int(srcX), modelSize - 1), x1 = min(x0 + 1, modelSize - 1)
                let fx = srcX - Float(x0)
                let alpha = raw[y0*modelSize+x0]*(1-fx)*(1-fy) + raw[y0*modelSize+x1]*fx*(1-fy)
                    + raw[y1*modelSize+x0]*(1-fx)*fy + raw[y1*modelSize+x1]*fx*fy
                let a = UInt8(clamping: Int(alpha * 255))
                let idx = (y * origW + x) * 4
                let af = Float(a) / 255.0
                result[idx] = UInt8(clamping: Int(Float(pixelData[idx]) * af))
                result[idx+1] = UInt8(clamping: Int(Float(pixelData[idx+1]) * af))
                result[idx+2] = UInt8(clamping: Int(Float(pixelData[idx+2]) * af))
                result[idx+3] = a
            }
        }
        guard let outCtx = CGContext(data: &result, width: origW, height: origH, bitsPerComponent: 8,
                                     bytesPerRow: origW * 4, space: CGColorSpaceCreateDeviceRGB(),
                                     bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
              let cgImg = outCtx.makeImage() else { return nil }
        return UIImage(cgImage: cgImg)
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

        var origL = [Float](repeating: 0, count: origW * origH)
        for i in 0..<(origW * origH) {
            origL[i] = srgbToL(r: Float(origPixels[i*4])/255, g: Float(origPixels[i*4+1])/255, b: Float(origPixels[i*4+2])/255)
        }

        var abOrig = [Float](repeating: 0, count: 2 * origW * origH)
        for ch in 0..<2 {
            let srcOff = ch * chSize, dstOff = ch * origW * origH
            for y in 0..<origH {
                let sy = Float(y) * Float(modelSize) / Float(origH)
                let y0 = min(Int(sy), modelSize-1), y1 = min(y0+1, modelSize-1), fy = sy - Float(y0)
                for x in 0..<origW {
                    let sx = Float(x) * Float(modelSize) / Float(origW)
                    let x0 = min(Int(sx), modelSize-1), x1 = min(x0+1, modelSize-1), fx = sx - Float(x0)
                    abOrig[dstOff + y*origW + x] = ab[srcOff+y0*modelSize+x0]*(1-fx)*(1-fy) + ab[srcOff+y0*modelSize+x1]*fx*(1-fy)
                        + ab[srcOff+y1*modelSize+x0]*(1-fx)*fy + ab[srcOff+y1*modelSize+x1]*fx*fy
                }
            }
        }

        let count = origW * origH
        var resultPixels = [UInt8](repeating: 255, count: count * 4)
        for i in 0..<count {
            let (r, g, b) = labToSrgb(l: origL[i], a: abOrig[i], b: abOrig[count + i])
            resultPixels[i*4] = UInt8(clamping: Int(r * 255))
            resultPixels[i*4+1] = UInt8(clamping: Int(g * 255))
            resultPixels[i*4+2] = UInt8(clamping: Int(b * 255))
        }
        guard let outCtx = CGContext(data: &resultPixels, width: origW, height: origH, bitsPerComponent: 8,
                                     bytesPerRow: origW * 4, space: CGColorSpaceCreateDeviceRGB(),
                                     bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue),
              let cgImg = outCtx.makeImage() else { return nil }
        return UIImage(cgImage: cgImg)
    }

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
