import SwiftUI
import PhotosUI
import CoreML

/// Inpainting demo: pick a photo, draw a mask, and fill in the masked region.
struct InpaintingDemoView: View {
    let model: ModelEntry

    @State private var inputImage: UIImage?
    @State private var outputImage: UIImage?
    @State private var maskPaths: [MaskStroke] = []
    @State private var currentStroke: MaskStroke?
    @State private var isProcessing = false
    @State private var status = ""
    @State private var processingTime: Double?
    @State private var item: PhotosPickerItem?
    @State private var brushSize: CGFloat = 30
    @State private var showOriginal = false

    private var inputSize: Int { model.configInt("input_size") ?? 800 }

    var body: some View {
        VStack(spacing: 0) {
            // Canvas area
            ZStack {
                if let img = showOriginal ? inputImage : outputImage ?? inputImage {
                    GeometryReader { geo in
                        let size = fitSize(imageSize: img.size, in: geo.size)
                        let origin = CGPoint(x: (geo.size.width - size.width) / 2,
                                             y: (geo.size.height - size.height) / 2)
                        ZStack {
                            Image(uiImage: img)
                                .resizable()
                                .frame(width: size.width, height: size.height)

                            // Mask overlay (only when no output yet)
                            if outputImage == nil {
                                Canvas { ctx, _ in
                                    for stroke in maskPaths + [currentStroke].compactMap({ $0 }) {
                                        var path = Path()
                                        guard let first = stroke.points.first else { continue }
                                        path.move(to: first)
                                        for pt in stroke.points.dropFirst() { path.addLine(to: pt) }
                                        ctx.stroke(path, with: .color(.red.opacity(0.5)),
                                                   lineWidth: stroke.width)
                                    }
                                }
                                .frame(width: size.width, height: size.height)
                                .contentShape(Rectangle())
                                .gesture(
                                    DragGesture(minimumDistance: 0)
                                        .onChanged { value in
                                            let pt = CGPoint(x: value.location.x, y: value.location.y)
                                            if currentStroke == nil {
                                                currentStroke = MaskStroke(points: [pt], width: brushSize)
                                            } else {
                                                currentStroke?.points.append(pt)
                                            }
                                        }
                                        .onEnded { _ in
                                            if let stroke = currentStroke { maskPaths.append(stroke) }
                                            currentStroke = nil
                                        }
                                )
                            }
                        }
                        .position(x: geo.size.width / 2, y: geo.size.height / 2)
                    }
                } else {
                    VStack(spacing: 12) {
                        Image(systemName: "photo.on.rectangle.angled")
                            .font(.system(size: 60)).foregroundStyle(.secondary)
                        Text("Select a photo, then draw over areas to remove")
                            .foregroundStyle(.secondary).multilineTextAlignment(.center)
                    }
                    .padding(.horizontal, 32)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { _ in if outputImage != nil { showOriginal = true } }
                    .onEnded { _ in showOriginal = false }
            )

            // Controls
            VStack(spacing: 8) {
                if outputImage != nil {
                    Text("Hold to see original").font(.caption2).foregroundStyle(.tertiary)
                }

                HStack {
                    if let t = processingTime {
                        Text(String(format: "%.2fs", t)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    }
                    Spacer()
                    if isProcessing {
                        ProgressView().controlSize(.small)
                        Text(status).font(.caption).foregroundStyle(.secondary)
                    }
                }

                if inputImage != nil && outputImage == nil {
                    HStack {
                        Text("Brush").font(.caption).foregroundStyle(.secondary)
                        Slider(value: $brushSize, in: 10...80)
                        Text("\(Int(brushSize))").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                            .frame(width: 30)
                    }
                }

                HStack(spacing: 12) {
                    PhotosPicker(selection: $item, matching: .images) {
                        Label("Select Photo", systemImage: "photo.badge.plus")
                    }
                    .buttonStyle(.bordered)
                    .disabled(isProcessing)

                    if inputImage != nil && !maskPaths.isEmpty && outputImage == nil {
                        Button {
                            runInpainting()
                        } label: {
                            Label("Inpaint", systemImage: "wand.and.stars")
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(isProcessing)
                    }

                    if !maskPaths.isEmpty && outputImage == nil {
                        Button {
                            maskPaths.removeAll()
                        } label: {
                            Image(systemName: "arrow.uturn.backward")
                        }
                        .buttonStyle(.bordered)
                    }

                    if outputImage != nil {
                        Button {
                            outputImage = nil
                            maskPaths.removeAll()
                        } label: {
                            Label("Reset", systemImage: "arrow.counterclockwise")
                        }
                        .buttonStyle(.bordered)

                        Button {
                            if let img = outputImage {
                                UIImageWriteToSavedPhotosAlbum(img, nil, nil, nil)
                            }
                        } label: {
                            Image(systemName: "arrow.down.to.line")
                        }
                        .buttonStyle(.bordered)
                    }
                }
            }
            .padding()
        }
        .onChange(of: item) { _, _ in loadPhoto() }
    }

    // MARK: - Photo loading

    private func loadPhoto() {
        guard let item else { return }
        outputImage = nil; maskPaths.removeAll()
        Task {
            guard let data = try? await item.loadTransferable(type: Data.self),
                  let img = UIImage(data: data) else { return }
            await MainActor.run { inputImage = img }
        }
    }

    // MARK: - Inpainting

    private func runInpainting() {
        guard let inputImage else { return }
        isProcessing = true; status = "Loading model…"

        Task {
            do {
                let mlModel = try await ModelLoader.loadPrimary(for: model)
                await MainActor.run { status = "Processing…" }

                guard let cgImage = ImageUtils.normalizeOrientation(inputImage) else {
                    await MainActor.run { isProcessing = false; status = "Image error" }; return
                }

                let sz = inputSize
                // Prepare image input
                guard let imgPB = ImageUtils.pixelBuffer(from: cgImage, width: sz, height: sz) else {
                    await MainActor.run { isProcessing = false; status = "Prep failed" }; return
                }

                // Render mask: draw strokes as white on black, scaled to image display size → model size
                let imageDisplaySize = fitSize(imageSize: inputImage.size,
                                               in: CGSize(width: UIScreen.main.bounds.width - 32,
                                                          height: UIScreen.main.bounds.height * 0.6))
                let maskImage = renderMask(strokes: maskPaths, displaySize: imageDisplaySize, outputSize: sz)
                guard let maskPB = grayscalePixelBuffer(from: maskImage, width: sz, height: sz) else {
                    await MainActor.run { isProcessing = false; status = "Mask error" }; return
                }

                // Build input from model description
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
                // Fallback: if model expects named inputs
                if inputDict.isEmpty {
                    inputDict["image"] = imgPB; inputDict["mask"] = maskPB
                }

                let start = CFAbsoluteTimeGetCurrent()
                let input = try MLDictionaryFeatureProvider(dictionary: inputDict)
                let output = try await mlModel.prediction(from: input)
                let elapsed = CFAbsoluteTimeGetCurrent() - start

                // Extract result
                var result: UIImage?
                for name in output.featureNames {
                    if let pb = output.featureValue(for: name)?.imageBufferValue {
                        let ci = CIImage(cvPixelBuffer: pb)
                        if let cg = CIContext().createCGImage(ci, from: ci.extent) {
                            result = UIImage(cgImage: cg)
                            break
                        }
                    }
                    if let arr = output.featureValue(for: name)?.multiArrayValue {
                        result = ImageUtils.imageFromMultiArray(arr)
                        if result != nil { break }
                    }
                }

                await MainActor.run {
                    outputImage = result ?? inputImage
                    processingTime = elapsed; isProcessing = false; status = ""
                }
            } catch {
                await MainActor.run { isProcessing = false; status = "Error: \(error.localizedDescription)" }
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

    // MARK: - Helpers

    private func fitSize(imageSize: CGSize, in containerSize: CGSize) -> CGSize {
        let scale = min(containerSize.width / imageSize.width, containerSize.height / imageSize.height)
        return CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
    }
}

struct MaskStroke {
    var points: [CGPoint]
    var width: CGFloat
}
