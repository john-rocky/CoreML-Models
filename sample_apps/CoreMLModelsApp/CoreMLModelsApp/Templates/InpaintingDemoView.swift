import SwiftUI
import PhotosUI
import CoreML

/// Inpainting demo ported from lama-cleaner-iOS.
/// Features: brush drawing, auto-run on stroke end, undo stack, draggable before/after compare slider.
struct InpaintingDemoView: View {
    let model: ModelEntry

    @State private var inputImage: UIImage?
    @State private var outputImage: UIImage?
    @State private var imageHistory: [UIImage] = []
    @State private var maskStrokes: [MaskStroke] = []
    @State private var currentStroke: MaskStroke?
    @State private var isProcessing = false
    @State private var status = ""
    @State private var processingTime: Double?
    @State private var item: PhotosPickerItem?
    @State private var brushSize: CGFloat = 40
    @State private var compareMode = false
    @State private var comparePosition: CGFloat = 0.5
    @State private var mlModel: MLModel?
    @State private var displaySize: CGSize = .zero

    private var inputSize: Int { model.configInt("input_size") ?? 800 }

    var body: some View {
        VStack(spacing: 0) {
            GeometryReader { geo in
                ZStack {
                    if let currentImage = outputImage ?? inputImage {
                        let fitted = fitSize(imageSize: currentImage.size, in: geo.size)

                        if compareMode, let original = imageHistory.first, let result = outputImage {
                            // Draggable compare slider
                            compareView(original: original, result: result, size: fitted)
                                .position(x: geo.size.width / 2, y: geo.size.height / 2)
                        } else {
                            ZStack {
                                Image(uiImage: currentImage)
                                    .resizable()
                                    .frame(width: fitted.width, height: fitted.height)

                                // Drawing overlay
                                if outputImage == nil && !isProcessing {
                                    drawingOverlay(size: fitted)
                                }
                            }
                            .position(x: geo.size.width / 2, y: geo.size.height / 2)
                        }
                    } else {
                        VStack(spacing: 12) {
                            Image(systemName: "eraser")
                                .font(.system(size: 60)).foregroundStyle(.secondary)
                            Text("Select a photo, then draw over areas to remove")
                                .foregroundStyle(.secondary).multilineTextAlignment(.center)
                        }
                        .padding(.horizontal, 32)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                    }

                    // Processing overlay
                    if isProcessing {
                        Color.black.opacity(0.3).allowsHitTesting(false)
                        VStack(spacing: 8) {
                            ProgressView().tint(.white)
                            Text(status).font(.caption).foregroundStyle(.white)
                        }
                    }
                }
                .onAppear { displaySize = geo.size }
                .onChange(of: geo.size) { _, new in displaySize = new }
            }

            controls
        }
        .onChange(of: item) { _, _ in loadPhoto() }
    }

    // MARK: - Drawing overlay

    @ViewBuilder
    private func drawingOverlay(size: CGSize) -> some View {
        Canvas { ctx, _ in
            for stroke in maskStrokes + [currentStroke].compactMap({ $0 }) {
                var path = Path()
                guard let first = stroke.points.first else { continue }
                path.move(to: first)
                for pt in stroke.points.dropFirst() { path.addLine(to: pt) }
                ctx.stroke(path, with: .color(.yellow.opacity(0.5)),
                           style: StrokeStyle(lineWidth: stroke.width, lineCap: .round, lineJoin: .round))
            }
        }
        .frame(width: size.width, height: size.height)
        .contentShape(Rectangle())
        .gesture(
            DragGesture(minimumDistance: 0)
                .onChanged { value in
                    let pt = value.location
                    if currentStroke == nil {
                        currentStroke = MaskStroke(points: [pt], width: brushSize)
                    } else {
                        currentStroke?.points.append(pt)
                    }
                }
                .onEnded { _ in
                    if let stroke = currentStroke {
                        maskStrokes.append(stroke)
                    }
                    currentStroke = nil
                    // Auto-run on stroke end
                    runInpainting()
                }
        )
    }

    // MARK: - Compare slider

    @ViewBuilder
    private func compareView(original: UIImage, result: UIImage, size: CGSize) -> some View {
        ZStack {
            // Result (full)
            Image(uiImage: result)
                .resizable()
                .frame(width: size.width, height: size.height)

            // Original (clipped to left portion)
            Image(uiImage: original)
                .resizable()
                .frame(width: size.width, height: size.height)
                .clipShape(
                    HorizontalClip(splitRatio: comparePosition)
                )

            // Separator line + knob
            let lineX = size.width * comparePosition - size.width / 2
            Rectangle()
                .fill(.yellow.opacity(0.6))
                .frame(width: 2, height: size.height)
                .offset(x: lineX)

            Circle()
                .fill(.ultraThinMaterial)
                .frame(width: 36, height: 36)
                .overlay(
                    Image(systemName: "arrow.left.and.right")
                        .font(.caption2.bold())
                        .foregroundStyle(.primary)
                )
                .shadow(radius: 4)
                .offset(x: lineX)
        }
        .frame(width: size.width, height: size.height)
        .contentShape(Rectangle())
        .gesture(
            DragGesture()
                .onChanged { value in
                    let ratio = value.location.x / size.width
                    comparePosition = min(max(ratio, 0.01), 0.99)
                }
        )
    }

    // MARK: - Controls

    @ViewBuilder
    private var controls: some View {
        VStack(spacing: 8) {
            HStack {
                if let t = processingTime {
                    Text(String(format: "%.2fs", t)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                }
                Spacer()
            }

            // Brush slider (only when drawing)
            if inputImage != nil && outputImage == nil && !isProcessing {
                HStack {
                    Image(systemName: "circle.fill").font(.system(size: 8)).foregroundStyle(.secondary)
                    Slider(value: $brushSize, in: 10...80)
                    Image(systemName: "circle.fill").font(.system(size: 20)).foregroundStyle(.secondary)
                }
            }

            HStack(spacing: 12) {
                PhotosPicker(selection: $item, matching: .images) {
                    Label("Photo", systemImage: "photo.badge.plus")
                }
                .buttonStyle(.bordered)
                .disabled(isProcessing)

                // Undo
                if imageHistory.count > 1 {
                    Button {
                        undoInpainting()
                    } label: {
                        Image(systemName: "arrow.uturn.backward")
                    }
                    .buttonStyle(.bordered)
                    .disabled(isProcessing)
                }

                // Compare toggle
                if outputImage != nil && imageHistory.count > 1 {
                    Button {
                        compareMode.toggle()
                    } label: {
                        Image(systemName: "slider.horizontal.below.rectangle")
                    }
                    .buttonStyle(.bordered)
                    .tint(compareMode ? .accentColor : nil)
                }

                Spacer()

                // Save
                if outputImage != nil {
                    Button {
                        if let img = outputImage {
                            UIImageWriteToSavedPhotosAlbum(img, nil, nil, nil)
                        }
                    } label: {
                        Image(systemName: "arrow.down.to.line")
                    }
                    .buttonStyle(.bordered)

                    ShareLink(item: Image(uiImage: outputImage!),
                              preview: SharePreview("Inpainted", image: Image(uiImage: outputImage!))) {
                        Image(systemName: "square.and.arrow.up")
                    }
                    .buttonStyle(.bordered)
                }
            }
        }
        .padding()
    }

    // MARK: - Photo loading

    private func loadPhoto() {
        guard let item else { return }
        outputImage = nil; maskStrokes.removeAll(); imageHistory.removeAll()
        compareMode = false; mlModel = nil; processingTime = nil
        Task {
            guard let data = try? await item.loadTransferable(type: Data.self),
                  let img = UIImage(data: data) else { return }
            await MainActor.run {
                inputImage = img
                imageHistory = [img]
            }
        }
    }

    // MARK: - Undo

    private func undoInpainting() {
        guard imageHistory.count > 1 else { return }
        imageHistory.removeLast()
        let previous = imageHistory.last!
        inputImage = previous
        outputImage = nil
        maskStrokes.removeAll()
        compareMode = false
    }

    // MARK: - Inpainting

    private func runInpainting() {
        guard let inputImage, !maskStrokes.isEmpty else { return }
        isProcessing = true; status = "Loading model…"

        Task {
            do {
                if mlModel == nil {
                    mlModel = try await ModelLoader.loadPrimary(for: model)
                }
                guard let mlModel else { return }
                await MainActor.run { status = "Processing…" }

                guard let cgImage = ImageUtils.normalizeOrientation(inputImage) else {
                    await MainActor.run { isProcessing = false; status = "Image error" }; return
                }

                let sz = inputSize
                guard let imgPB = ImageUtils.pixelBuffer(from: cgImage, width: sz, height: sz) else {
                    await MainActor.run { isProcessing = false; status = "Prep failed" }; return
                }

                // Render mask scaled from display coordinates to model size
                let fitted = fitSize(imageSize: inputImage.size, in: displaySize)
                let maskImage = renderMask(strokes: maskStrokes, displaySize: fitted, outputSize: sz)
                guard let maskPB = grayscalePixelBuffer(from: maskImage, width: sz, height: sz) else {
                    await MainActor.run { isProcessing = false; status = "Mask error" }; return
                }

                // Build input
                let desc = mlModel.modelDescription.inputDescriptionsByName
                var inputDict: [String: Any] = [:]
                for (name, fd) in desc {
                    if fd.type == .image {
                        if let ic = fd.imageConstraint,
                           ic.pixelFormatType == kCVPixelFormatType_OneComponent8 {
                            inputDict[name] = maskPB
                        } else {
                            inputDict[name] = imgPB
                        }
                    }
                }
                if inputDict.isEmpty {
                    inputDict["image"] = imgPB; inputDict["mask"] = maskPB
                }

                let start = CFAbsoluteTimeGetCurrent()
                let input = try MLDictionaryFeatureProvider(dictionary: inputDict)
                let output = try await mlModel.prediction(from: input)
                let elapsed = CFAbsoluteTimeGetCurrent() - start

                // Extract result and merge with original at original resolution
                var resultImage: UIImage?
                for name in output.featureNames {
                    if let pb = output.featureValue(for: name)?.imageBufferValue {
                        let ci = CIImage(cvPixelBuffer: pb)
                        if let cg = CIContext().createCGImage(ci, from: ci.extent) {
                            resultImage = UIImage(cgImage: cg)
                            break
                        }
                    }
                    if let arr = output.featureValue(for: name)?.multiArrayValue {
                        resultImage = ImageUtils.imageFromMultiArray(arr)
                        if resultImage != nil { break }
                    }
                }

                // Merge: paste inpainted region back onto original at full resolution
                let merged = mergeResult(result: resultImage, mask: maskImage,
                                         original: inputImage, modelSize: sz)

                await MainActor.run {
                    let final = merged ?? resultImage ?? inputImage
                    outputImage = final
                    self.inputImage = final
                    imageHistory.append(final)
                    maskStrokes.removeAll()
                    processingTime = elapsed
                    isProcessing = false; status = ""
                }
            } catch {
                await MainActor.run { isProcessing = false; status = "Error: \(error.localizedDescription)" }
            }
        }
    }

    // MARK: - Merge result with original (like lama-cleaner)

    private func mergeResult(result: UIImage?, mask: UIImage, original: UIImage, modelSize: Int) -> UIImage? {
        guard let result, let origCG = ImageUtils.normalizeOrientation(original),
              let resCG = result.cgImage, let maskCG = mask.cgImage else { return nil }
        let w = origCG.width, h = origCG.height
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: w, height: h))
        return renderer.image { ctx in
            // Draw original
            UIImage(cgImage: origCG).draw(in: CGRect(x: 0, y: 0, width: w, height: h))
            // Draw result resized to original, masked by the inpaint mask
            let resizedResult = UIImage(cgImage: resCG)
            let resizedMask = UIImage(cgImage: maskCG)
            // Create mask-clipped result
            UIGraphicsBeginImageContextWithOptions(CGSize(width: w, height: h), false, 1.0)
            if let maskCtx = UIGraphicsGetCurrentContext() {
                maskCtx.draw(maskCG, in: CGRect(x: 0, y: 0, width: w, height: h))
            }
            let maskResized = UIGraphicsGetImageFromCurrentImageContext()
            UIGraphicsEndImageContext()

            if let maskResized, let maskRef = maskResized.cgImage {
                ctx.cgContext.saveGState()
                ctx.cgContext.clip(to: CGRect(x: 0, y: 0, width: w, height: h), mask: maskRef)
                resizedResult.draw(in: CGRect(x: 0, y: 0, width: w, height: h))
                ctx.cgContext.restoreGState()
            }
        }
    }

    // MARK: - Mask rendering

    private func renderMask(strokes: [MaskStroke], displaySize: CGSize, outputSize: Int) -> UIImage {
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: outputSize, height: outputSize))
        return renderer.image { ctx in
            ctx.cgContext.setFillColor(UIColor.black.cgColor)
            ctx.cgContext.fill(CGRect(x: 0, y: 0, width: outputSize, height: outputSize))

            let scaleX = CGFloat(outputSize) / displaySize.width
            let scaleY = CGFloat(outputSize) / displaySize.height

            ctx.cgContext.setStrokeColor(UIColor.white.cgColor)
            ctx.cgContext.setLineCap(.round)
            ctx.cgContext.setLineJoin(.round)

            for stroke in strokes {
                ctx.cgContext.setLineWidth(stroke.width * scaleX)
                guard let first = stroke.points.first else { continue }
                ctx.cgContext.beginPath()
                ctx.cgContext.move(to: CGPoint(x: first.x * scaleX, y: first.y * scaleY))
                for pt in stroke.points.dropFirst() {
                    ctx.cgContext.addLine(to: CGPoint(x: pt.x * scaleX, y: pt.y * scaleY))
                }
                ctx.cgContext.strokePath()
            }
        }
    }

    private func grayscalePixelBuffer(from image: UIImage, width: Int, height: Int) -> CVPixelBuffer? {
        guard let cg = image.cgImage else { return nil }
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                            kCVPixelFormatType_OneComponent8, nil, &pb)
        guard let buf = pb else { return nil }
        CVPixelBufferLockBaseAddress(buf, [])
        defer { CVPixelBufferUnlockBaseAddress(buf, []) }
        guard let base = CVPixelBufferGetBaseAddress(buf) else { return nil }
        let bpr = CVPixelBufferGetBytesPerRow(buf)
        guard let ctx = CGContext(data: base, width: width, height: height,
                                  bitsPerComponent: 8, bytesPerRow: bpr,
                                  space: CGColorSpaceCreateDeviceGray(),
                                  bitmapInfo: 0) else { return nil }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
        return buf
    }

    private func fitSize(imageSize: CGSize, in containerSize: CGSize) -> CGSize {
        let scale = min(containerSize.width / imageSize.width, containerSize.height / imageSize.height)
        return CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
    }
}

// MARK: - Supporting types

struct MaskStroke {
    var points: [CGPoint]
    var width: CGFloat
}

struct HorizontalClip: Shape {
    var splitRatio: CGFloat
    func path(in rect: CGRect) -> Path {
        Path(CGRect(x: 0, y: 0, width: rect.width * splitRatio, height: rect.height))
    }
}
