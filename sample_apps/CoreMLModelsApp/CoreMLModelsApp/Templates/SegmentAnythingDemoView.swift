import SwiftUI
import PhotosUI
import CoreML
import UniformTypeIdentifiers
import Accelerate

/// Segment Anything demo ported from SamKit.
/// Features: tap-to-segment, subject lifting (long press), glowing outline,
/// background dimming, lift context menu (copy/save/share), haptic feedback.
struct SegmentAnythingDemoView: View {
    let model: ModelEntry

    @State private var inputImage: UIImage?
    @State private var tapPoints: [CGPoint] = []  // in image coordinates
    @State private var isProcessing = false
    @State private var status = ""
    @State private var item: PhotosPickerItem?
    @State private var decoderModel: MLModel?
    @State private var imageEmbedding: MLMultiArray?
    @State private var promptEncoder: SAMPromptEncoder?
    @State private var isEncoderLoaded = false

    // Letterbox transform parameters
    @State private var transformScale: Float = 1.0
    @State private var transformPadX: Float = 0.0
    @State private var transformPadY: Float = 0.0

    // Mask state
    @State private var maskCGImage: CGImage?
    @State private var outlineImage: CGImage?

    // Lift state
    @State private var isLifted = false
    @State private var liftedImage: UIImage?
    @State private var liftDragOffset: CGSize = .zero
    @State private var liftStartTranslation: CGSize = .zero
    @State private var showLiftMenu = false
    @State private var toastMessage: String?
    @State private var showShareSheet = false

    // Gesture tracking
    @State private var gestureStartTime: Date?
    @State private var lastGestureTranslation: CGSize = .zero

    private var inputSize: Int { model.configInt("input_size") ?? 1024 }

    // SAM normalization constants (ImageNet in 0-255 range)
    private static let samMean: [Float] = [123.675, 116.28, 103.53]
    private static let samStd: [Float] = [58.395, 57.12, 57.375]

    var body: some View {
        VStack(spacing: 0) {
            GeometryReader { geo in
                ZStack {
                    if let img = inputImage {
                        let fitted = fitSize(imageSize: img.size, in: geo.size)

                        // Base image
                        Image(uiImage: img)
                            .resizable().scaledToFit()
                            .frame(width: fitted.width, height: fitted.height)
                            .contentShape(Rectangle())
                            .gesture(
                                DragGesture(minimumDistance: 0)
                                    .onChanged { value in
                                        if gestureStartTime == nil {
                                            gestureStartTime = Date()
                                            // Long press to lift
                                            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                                                guard gestureStartTime != nil, !isLifted, maskCGImage != nil else { return }
                                                let moved = hypot(lastGestureTranslation.width, lastGestureTranslation.height)
                                                guard moved < 15 else { return }
                                                liftStartTranslation = lastGestureTranslation
                                                handleLift()
                                            }
                                        }
                                        lastGestureTranslation = value.translation
                                        if isLifted {
                                            liftDragOffset = CGSize(
                                                width: value.translation.width - liftStartTranslation.width,
                                                height: value.translation.height - liftStartTranslation.height
                                            )
                                        }
                                    }
                                    .onEnded { value in
                                        let elapsed = Date().timeIntervalSince(gestureStartTime ?? Date())
                                        let moved = hypot(value.translation.width, value.translation.height)
                                        gestureStartTime = nil
                                        lastGestureTranslation = .zero

                                        if isLifted {
                                            withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                                                liftDragOffset = .zero
                                            }
                                            withAnimation(.spring(response: 0.25, dampingFraction: 0.8)) {
                                                showLiftMenu = true
                                            }
                                            return
                                        }
                                        // Short tap → add point
                                        if elapsed < 0.3 && moved < 15 && isEncoderLoaded {
                                            let imagePoint = viewToImageCoordinates(
                                                value.startLocation, viewSize: fitted, imageSize: img.size
                                            )
                                            tapPoints.append(imagePoint)
                                            runDecoder()
                                        }
                                    }
                            )

                        // Subject highlight: dim background + bright subject
                        if let mask = maskCGImage, !isLifted {
                            Color.black.opacity(0.25).allowsHitTesting(false)
                                .frame(width: fitted.width, height: fitted.height)

                            Image(uiImage: img)
                                .resizable().scaledToFit()
                                .frame(width: fitted.width, height: fitted.height)
                                .mask(
                                    Image(uiImage: UIImage(cgImage: mask))
                                        .resizable().scaledToFit()
                                        .frame(width: fitted.width, height: fitted.height)
                                )
                                .allowsHitTesting(false)
                        }

                        // Glowing outline
                        if !isLifted, let outline = outlineImage {
                            GlowingOutline(outline: outline, width: fitted.width, height: fitted.height)
                                .allowsHitTesting(false)
                        }

                        // Point markers
                        if !isLifted {
                            ForEach(tapPoints.indices, id: \.self) { i in
                                let viewPos = imageToViewCoordinates(
                                    tapPoints[i], viewSize: fitted, imageSize: img.size
                                )
                                Circle()
                                    .fill(.green)
                                    .frame(width: 14, height: 14)
                                    .overlay(Circle().stroke(.white, lineWidth: 2))
                                    .shadow(radius: 2)
                                    .position(viewPos)
                                    .allowsHitTesting(false)
                            }
                        }

                        // Lifted subject overlay
                        if isLifted {
                            Color.black.opacity(0.4)
                                .ignoresSafeArea()
                                .allowsHitTesting(showLiftMenu)
                                .contentShape(Rectangle())
                                .onTapGesture { dismissLift() }

                            if let mask = maskCGImage {
                                Image(uiImage: img)
                                    .resizable().scaledToFit()
                                    .frame(width: fitted.width, height: fitted.height)
                                    .mask(
                                        Image(uiImage: UIImage(cgImage: mask))
                                            .resizable().scaledToFit()
                                            .frame(width: fitted.width, height: fitted.height)
                                    )
                                    .shadow(color: .black.opacity(0.6), radius: 24, y: 12)
                                    .scaleEffect(showLiftMenu ? 1.0 : 1.05)
                                    .offset(liftDragOffset)
                                    .allowsHitTesting(false)
                                    .animation(.spring(response: 0.35, dampingFraction: 0.75), value: showLiftMenu)
                            }

                            if showLiftMenu {
                                liftContextMenu
                                    .transition(.scale(scale: 0.8).combined(with: .opacity))
                            }
                        }

                    } else {
                        VStack(spacing: 12) {
                            Image(systemName: "hand.tap")
                                .font(.system(size: 60)).foregroundStyle(.secondary)
                            Text("Select a photo, then tap to segment")
                                .foregroundStyle(.secondary)
                        }
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                    }

                    // Processing
                    if isProcessing {
                        Color.black.opacity(0.3).allowsHitTesting(false)
                        ProgressView().tint(.white).scaleEffect(1.5)
                    }
                }
            }

            // Controls
            VStack(spacing: 8) {
                HStack {
                    if !isEncoderLoaded && inputImage != nil && !isProcessing {
                        Label("Encoding image…", systemImage: "brain")
                            .font(.caption).foregroundStyle(.orange)
                    }
                    Spacer()
                    if isProcessing {
                        Text(status).font(.caption).foregroundStyle(.secondary)
                    }
                }

                HStack(spacing: 12) {
                    PhotosPicker(selection: $item, matching: .images) {
                        Label("Photo", systemImage: "photo.badge.plus")
                    }
                    .buttonStyle(.bordered)
                    .disabled(isProcessing)

                    if !tapPoints.isEmpty {
                        Button {
                            tapPoints.removeAll()
                            maskCGImage = nil; outlineImage = nil; liftedImage = nil
                        } label: {
                            Label("Clear", systemImage: "arrow.counterclockwise")
                        }
                        .buttonStyle(.bordered)
                    }

                    Spacer()

                    if let output = liftedImage {
                        Button {
                            UIImageWriteToSavedPhotosAlbum(output, nil, nil, nil)
                            showToast("Saved to Photos")
                        } label: {
                            Image(systemName: "arrow.down.to.line")
                        }
                        .buttonStyle(.bordered)
                    }
                }
            }
            .padding()
        }
        .overlay { toastOverlay }
        .sheet(isPresented: $showShareSheet) {
            if let lifted = liftedImage {
                ActivityView(items: [lifted])
            }
        }
        .onChange(of: item) { _, _ in loadPhoto() }
        .animation(.easeInOut(duration: 0.25), value: isLifted)
        .animation(.spring(response: 0.3, dampingFraction: 0.75), value: showLiftMenu)
    }

    // MARK: - Lift context menu

    @ViewBuilder
    private var liftContextMenu: some View {
        VStack(spacing: 0) {
            Button {
                if let img = liftedImage, let data = img.pngData() {
                    UIPasteboard.general.setData(data, forPasteboardType: UTType.png.identifier)
                    dismissLift(); showToast("Copied to clipboard")
                }
            } label: {
                Label("Copy", systemImage: "doc.on.doc")
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 16).padding(.vertical, 12)
            }
            Divider()
            Button {
                if let img = liftedImage {
                    UIImageWriteToSavedPhotosAlbum(img, nil, nil, nil)
                    dismissLift(); showToast("Saved to Photos")
                }
            } label: {
                Label("Save to Photos", systemImage: "square.and.arrow.down")
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 16).padding(.vertical, 12)
            }
            Divider()
            Button {
                showShareSheet = true; dismissLift()
            } label: {
                Label("Share...", systemImage: "square.and.arrow.up")
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 16).padding(.vertical, 12)
            }
        }
        .foregroundColor(.primary).font(.body)
        .frame(width: 220)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 14))
        .shadow(color: .black.opacity(0.2), radius: 20, y: 5)
    }

    // MARK: - Toast

    @ViewBuilder
    private var toastOverlay: some View {
        if let msg = toastMessage {
            VStack {
                Spacer()
                Text(msg)
                    .font(.subheadline.weight(.medium))
                    .foregroundColor(.white)
                    .padding(.horizontal, 16).padding(.vertical, 10)
                    .background(Capsule().fill(Color.black.opacity(0.75)))
                    .padding(.bottom, 60)
            }
            .transition(.move(edge: .bottom).combined(with: .opacity))
        }
    }

    private func showToast(_ msg: String) {
        withAnimation { toastMessage = msg }
        Task {
            try? await Task.sleep(for: .seconds(1.5))
            await MainActor.run { withAnimation { toastMessage = nil } }
        }
    }

    // MARK: - Lift

    private func handleLift() {
        guard let inputImage, let mask = maskCGImage,
              let cgImage = ImageUtils.normalizeOrientation(inputImage) else { return }

        let w = cgImage.width, h = cgImage.height
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))
        guard let imageData = ctx.data else { return }
        let pixels = imageData.bindMemory(to: UInt8.self, capacity: w * h * 4)

        guard let maskCtx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8,
                                      bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
        maskCtx.draw(mask, in: CGRect(x: 0, y: 0, width: w, height: h))
        guard let maskData = maskCtx.data else { return }
        let maskPixels = maskData.bindMemory(to: UInt8.self, capacity: w * h * 4)

        for i in 0..<(w * h) {
            let alpha = maskPixels[i * 4 + 3]
            if alpha < 128 {
                pixels[i * 4] = 0; pixels[i * 4 + 1] = 0
                pixels[i * 4 + 2] = 0; pixels[i * 4 + 3] = 0
            }
        }

        guard let extracted = ctx.makeImage() else { return }
        liftedImage = UIImage(cgImage: extracted)

        let generator = UIImpactFeedbackGenerator(style: .medium)
        generator.impactOccurred()

        withAnimation(.spring(response: 0.35, dampingFraction: 0.75)) {
            isLifted = true
        }
    }

    private func dismissLift() {
        withAnimation(.easeOut(duration: 0.2)) {
            showLiftMenu = false; isLifted = false
        }
        liftDragOffset = .zero; liftStartTranslation = .zero
    }

    // MARK: - Photo loading & encoding

    private func loadPhoto() {
        guard let item else { return }
        tapPoints.removeAll(); maskCGImage = nil; outlineImage = nil
        liftedImage = nil; imageEmbedding = nil; isEncoderLoaded = false
        isProcessing = true; status = "Loading…"

        Task {
            guard let data = try? await item.loadTransferable(type: Data.self),
                  let img = UIImage(data: data) else {
                await MainActor.run { isProcessing = false; status = "Failed" }; return
            }
            await MainActor.run { inputImage = img; status = "Loading encoder…" }
            await encodeImage(img)
        }
    }

    private func encodeImage(_ image: UIImage) async {
        do {
            let cu = ModelLoader.parseComputeUnits(model.files.first?.computeUnits)

            // Load encoder and decoder.
            // Config may specify actual model file names (e.g. "mobile_sam_encoder") or
            // an archive name (e.g. "MobileSAM.zip") when both models come from one zip.
            let rawEnc = model.configString("encoder") ?? ""
            let rawDec = model.configString("decoder") ?? ""
            let sameArchive = rawEnc == rawDec || rawEnc.hasSuffix(".zip")

            let enc: MLModel
            let dec: MLModel
            if sameArchive {
                // Both point to the same archive → search by substring pattern
                enc = try await ModelLoader.loadBySubstring(modelId: model.id, substring: "encoder", computeUnits: cu)
                dec = try await ModelLoader.loadBySubstring(modelId: model.id, substring: "decoder", computeUnits: cu)
            } else {
                enc = try await ModelLoader.load(modelId: model.id, fileName: rawEnc, computeUnits: cu)
                dec = try await ModelLoader.load(modelId: model.id, fileName: rawDec, computeUnits: cu)
            }

            // Load prompt encoder weights if available
            let weightsName = model.configString("prompt_weights") ?? "mobile_sam_prompt_encoder_weights.json"
            let weightsURL = ModelLoader.auxFileURL(modelId: model.id, fileName: weightsName)
            var pe: SAMPromptEncoder?
            if FileManager.default.fileExists(atPath: weightsURL.path) {
                pe = try SAMPromptEncoder(weightsURL: weightsURL)
            }

            await MainActor.run { status = "Encoding image…" }

            guard let cgImage = ImageUtils.normalizeOrientation(image) else {
                await MainActor.run { isProcessing = false; status = "Image error" }; return
            }

            // Compute letterbox transform
            let origW = cgImage.width, origH = cgImage.height
            let longSide = max(origW, origH)
            let scale = Float(inputSize) / Float(longSide)
            let scaledW = Int(Float(origW) * scale)
            let scaledH = Int(Float(origH) * scale)
            let padX = Float(inputSize - scaledW) / 2.0
            let padY = Float(inputSize - scaledH) / 2.0

            // Build [1, 3, inputSize, inputSize] MLMultiArray with SAM normalization
            let imageArray = try buildSAMInput(cgImage: cgImage, scaledW: scaledW, scaledH: scaledH,
                                               padX: Int(padX), padY: Int(padY))

            let encInput = try MLDictionaryFeatureProvider(dictionary: ["image": imageArray])
            let encOutput = try await enc.prediction(from: encInput)

            guard let emb = encOutput.featureValue(for: "image_embeddings")?.multiArrayValue else {
                // Fallback: try first multiarray output
                var fallbackEmb: MLMultiArray?
                for name in encOutput.featureNames {
                    if let arr = encOutput.featureValue(for: name)?.multiArrayValue {
                        fallbackEmb = arr; break
                    }
                }
                guard let emb2 = fallbackEmb else {
                    await MainActor.run { isProcessing = false; status = "Encoding failed" }; return
                }
                await MainActor.run {
                    decoderModel = dec; imageEmbedding = emb2; promptEncoder = pe
                    transformScale = scale; transformPadX = padX; transformPadY = padY
                    isEncoderLoaded = true; isProcessing = false; status = ""
                }
                return
            }

            await MainActor.run {
                decoderModel = dec; imageEmbedding = emb; promptEncoder = pe
                transformScale = scale; transformPadX = padX; transformPadY = padY
                isEncoderLoaded = true; isProcessing = false; status = ""
            }
        } catch {
            await MainActor.run { isProcessing = false; status = "Error: \(error.localizedDescription)" }
        }
    }

    /// Build a [1, 3, H, W] MLMultiArray from a CGImage with letterbox padding and SAM normalization.
    private func buildSAMInput(cgImage: CGImage, scaledW: Int, scaledH: Int,
                               padX: Int, padY: Int) throws -> MLMultiArray {
        let size = inputSize
        let array = try MLMultiArray(shape: [1, 3, size as NSNumber, size as NSNumber], dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float32.self, capacity: array.count)
        let channelSize = size * size

        // Initialize to zero (black padding)
        memset(ptr, 0, array.count * MemoryLayout<Float32>.size)

        // Draw the scaled image into a temporary pixel buffer
        let bytesPerRow = scaledW * 4
        var pixelData = [UInt8](repeating: 0, count: scaledH * bytesPerRow)
        guard let ctx = CGContext(
            data: &pixelData, width: scaledW, height: scaledH,
            bitsPerComponent: 8, bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else { throw NSError(domain: "SAM", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to create CGContext"]) }
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: scaledW, height: scaledH))

        // Write normalized pixel values into planar CHW array at the padded offset
        let mean = Self.samMean
        let std = Self.samStd
        for y in 0..<scaledH {
            let destY = y + padY
            guard destY < size else { continue }
            for x in 0..<scaledW {
                let destX = x + padX
                guard destX < size else { continue }
                let srcIdx = y * bytesPerRow + x * 4
                let dstOffset = destY * size + destX
                let r = Float(pixelData[srcIdx])
                let g = Float(pixelData[srcIdx + 1])
                let b = Float(pixelData[srcIdx + 2])
                ptr[0 * channelSize + dstOffset] = (r - mean[0]) / std[0]
                ptr[1 * channelSize + dstOffset] = (g - mean[1]) / std[1]
                ptr[2 * channelSize + dstOffset] = (b - mean[2]) / std[2]
            }
        }

        return array
    }

    // MARK: - Decode mask

    private func runDecoder() {
        guard let decoderModel, let imageEmbedding, let inputImage,
              !tapPoints.isEmpty else { return }
        isProcessing = true; status = "Segmenting…"

        Task {
            do {
                let decoderInput: MLDictionaryFeatureProvider

                if let pe = promptEncoder {
                    // MobileSAM path: use PromptEncoder to build sparse/dense embeddings
                    let modelPoints = tapPoints.map { p -> (x: Float, y: Float) in
                        (x: Float(p.x) * transformScale + transformPadX,
                         y: Float(p.y) * transformScale + transformPadY)
                    }
                    let labels = [Float](repeating: 1.0, count: tapPoints.count)
                    let (sparse, dense) = try pe.encode(points: modelPoints, labels: labels)

                    decoderInput = try MLDictionaryFeatureProvider(dictionary: [
                        "image_embeddings": imageEmbedding,
                        "sparse_embeddings": sparse,
                        "dense_embeddings": dense,
                    ])
                } else {
                    // Standard SAM path: pass point_coords/point_labels directly
                    let n = tapPoints.count
                    let coords = try MLMultiArray(shape: [1, NSNumber(value: n), 2], dataType: .float32)
                    let labels = try MLMultiArray(shape: [1, NSNumber(value: n)], dataType: .float32)
                    let cPtr = coords.dataPointer.assumingMemoryBound(to: Float.self)
                    let lPtr = labels.dataPointer.assumingMemoryBound(to: Float.self)
                    for i in 0..<n {
                        cPtr[i * 2] = Float(tapPoints[i].x) * transformScale + transformPadX
                        cPtr[i * 2 + 1] = Float(tapPoints[i].y) * transformScale + transformPadY
                        lPtr[i] = 1.0
                    }

                    let desc = decoderModel.modelDescription.inputDescriptionsByName
                    var inputDict: [String: MLFeatureValue] = [:]
                    for (name, fd) in desc {
                        if fd.type == .multiArray {
                            if name.contains("coord") || name.contains("point") {
                                inputDict[name] = MLFeatureValue(multiArray: coords)
                            } else if name.contains("label") {
                                inputDict[name] = MLFeatureValue(multiArray: labels)
                            } else if name.contains("mask_input") {
                                let maskIn = try MLMultiArray(shape: [1, 1, 256, 256], dataType: .float32)
                                inputDict[name] = MLFeatureValue(multiArray: maskIn)
                            } else if name.contains("has_mask") {
                                let hasMask = try MLMultiArray(shape: [1], dataType: .float32)
                                inputDict[name] = MLFeatureValue(multiArray: hasMask)
                            } else {
                                inputDict[name] = MLFeatureValue(multiArray: imageEmbedding)
                            }
                        }
                    }
                    decoderInput = try MLDictionaryFeatureProvider(dictionary: inputDict)
                }

                let output = try await decoderModel.prediction(from: decoderInput)

                // Extract masks and IoU predictions
                let maskArr = output.featureValue(for: "masks")?.multiArrayValue
                let iouArr = output.featureValue(for: "iou_predictions")?.multiArrayValue

                guard let masks = maskArr else {
                    // Fallback: take first multiarray output
                    var fallback: MLMultiArray?
                    for name in output.featureNames {
                        if let arr = output.featureValue(for: name)?.multiArrayValue {
                            fallback = arr; break
                        }
                    }
                    guard let m = fallback else {
                        await MainActor.run { isProcessing = false; status = "No mask" }; return
                    }
                    let bm = buildBinaryMask(from: m, maskIndex: 0,
                                             width: Int(inputImage.size.width),
                                             height: Int(inputImage.size.height))
                    let ol = bm.flatMap { generateOutline(from: $0) }
                    await MainActor.run {
                        maskCGImage = bm; outlineImage = ol
                        isProcessing = false; status = ""
                    }
                    return
                }

                // Select best mask by IoU score
                let bestIdx = selectBestMask(masks: masks, iou: iouArr)

                let binaryMask = buildBinaryMask(from: masks, maskIndex: bestIdx,
                                                  width: Int(inputImage.size.width),
                                                  height: Int(inputImage.size.height))
                let outline = binaryMask.flatMap { generateOutline(from: $0) }

                await MainActor.run {
                    maskCGImage = binaryMask
                    outlineImage = outline
                    isProcessing = false; status = ""
                }
            } catch {
                await MainActor.run { isProcessing = false; status = "Error: \(error.localizedDescription)" }
            }
        }
    }

    /// Select the mask index with the highest IoU prediction.
    private func selectBestMask(masks: MLMultiArray, iou: MLMultiArray?) -> Int {
        guard let iou else { return 0 }
        let numMasks = iou.shape.last?.intValue ?? 1
        var bestIdx = 0
        var bestScore: Float = -.greatestFiniteMagnitude
        for i in 0..<numMasks {
            let score = ImageUtils.readFloat(iou, at: i)
            if score > bestScore {
                bestScore = score; bestIdx = i
            }
        }
        return bestIdx
    }

    // MARK: - Mask processing

    private func buildBinaryMask(from arr: MLMultiArray, maskIndex: Int,
                                  width: Int, height: Int) -> CGImage? {
        let shape = arr.shape.map { $0.intValue }
        // Expected shape: [1, numMasks, mH, mW]
        let mH = shape[shape.count - 2]
        let mW = shape[shape.count - 1]

        // Offset to the selected mask plane
        let maskPlaneSize = mH * mW
        let offset: Int
        if shape.count == 4 {
            offset = maskIndex * maskPlaneSize
        } else {
            let total = arr.count
            offset = total - mH * mW
        }

        guard let ctx = CGContext(data: nil, width: width, height: height,
                                  bitsPerComponent: 8, bytesPerRow: width * 4,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        guard let data = ctx.data else { return nil }
        let pixels = data.bindMemory(to: UInt8.self, capacity: width * height * 4)

        // The mask is in model space (256x256 with letterbox padding).
        // Map each pixel in the output image back through the letterbox transform.
        let maskScaleX = Float(mW) / Float(inputSize)
        let maskScaleY = Float(mH) / Float(inputSize)

        for y in 0..<height {
            // Image coord → model coord → mask coord
            let modelY = Float(y) * transformScale + transformPadY
            let my = Int(modelY * maskScaleY)
            guard my >= 0 && my < mH else {
                // Outside letterbox → background
                let rowStart = y * width * 4
                memset(pixels + rowStart, 0, width * 4)
                continue
            }
            for x in 0..<width {
                let modelX = Float(x) * transformScale + transformPadX
                let mx = Int(modelX * maskScaleX)
                let o = (y * width + x) * 4
                if mx >= 0 && mx < mW {
                    let val = ImageUtils.readFloat(arr, at: offset + my * mW + mx)
                    if val > 0 {
                        pixels[o] = 255; pixels[o + 1] = 255; pixels[o + 2] = 255; pixels[o + 3] = 255
                    } else {
                        pixels[o] = 0; pixels[o + 1] = 0; pixels[o + 2] = 0; pixels[o + 3] = 0
                    }
                } else {
                    pixels[o] = 0; pixels[o + 1] = 0; pixels[o + 2] = 0; pixels[o + 3] = 0
                }
            }
        }
        return ctx.makeImage()
    }

    private func generateOutline(from mask: CGImage) -> CGImage? {
        let w = mask.width, h = mask.height
        let rect = CGRect(x: 0, y: 0, width: w, height: h)
        let glowRadius = CGFloat(min(30, max(4, min(w, h) / 100)))

        guard let silCtx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8,
                                     bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                                     bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        silCtx.draw(mask, in: rect)
        silCtx.setBlendMode(.sourceIn)
        silCtx.setFillColor(UIColor.white.cgColor)
        silCtx.fill(rect)
        guard let whiteSil = silCtx.makeImage() else { return nil }

        guard let outCtx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8,
                                     bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                                     bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        outCtx.setShadow(offset: .zero, blur: glowRadius, color: UIColor.white.cgColor)
        outCtx.draw(whiteSil, in: rect)
        outCtx.setShadow(offset: .zero, blur: 0, color: nil)
        outCtx.setBlendMode(.destinationOut)
        outCtx.draw(whiteSil, in: rect)
        return outCtx.makeImage()
    }

    // MARK: - Coordinate helpers

    private func viewToImageCoordinates(_ point: CGPoint, viewSize: CGSize, imageSize: CGSize) -> CGPoint {
        CGPoint(x: point.x / viewSize.width * imageSize.width,
                y: point.y / viewSize.height * imageSize.height)
    }

    private func imageToViewCoordinates(_ point: CGPoint, viewSize: CGSize, imageSize: CGSize) -> CGPoint {
        CGPoint(x: point.x / imageSize.width * viewSize.width,
                y: point.y / imageSize.height * viewSize.height)
    }

    private func fitSize(imageSize: CGSize, in containerSize: CGSize) -> CGSize {
        let scale = min(containerSize.width / imageSize.width, containerSize.height / imageSize.height)
        return CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
    }
}

// MARK: - Glowing outline animation

private struct GlowingOutline: View {
    let outline: CGImage
    let width: CGFloat
    let height: CGFloat

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 30)) { timeline in
            let t = timeline.date.timeIntervalSinceReferenceDate
            let phase = t.truncatingRemainder(dividingBy: 2.5) / 2.5

            ZStack {
                Image(uiImage: UIImage(cgImage: outline))
                    .resizable().scaledToFit()
                    .frame(width: width, height: height)
                    .colorMultiply(Color(red: 0.5, green: 0.85, blue: 1.0))
                    .blur(radius: 5).opacity(0.8)

                Image(uiImage: UIImage(cgImage: outline))
                    .resizable().scaledToFit()
                    .frame(width: width, height: height)
                    .colorMultiply(.white)

                Image(uiImage: UIImage(cgImage: outline))
                    .resizable().scaledToFit()
                    .frame(width: width, height: height)
                    .colorMultiply(.white)
                    .mask(
                        AngularGradient(
                            gradient: Gradient(colors: [
                                .white, .white.opacity(0.5), .clear, .clear,
                                .clear, .clear, .clear, .white.opacity(0.3)
                            ]),
                            center: .center,
                            startAngle: .degrees(phase * 360),
                            endAngle: .degrees(phase * 360 + 360)
                        )
                    )
                    .blur(radius: 2)
            }
        }
    }
}

// MARK: - Share sheet

private struct ActivityView: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}
