import SwiftUI
import PhotosUI
import CoreML

/// Segment Anything demo: pick a photo, tap to place points, and generate masks.
struct SegmentAnythingDemoView: View {
    let model: ModelEntry

    @State private var inputImage: UIImage?
    @State private var outputImage: UIImage?
    @State private var tapPoints: [CGPoint] = []  // normalized 0..1
    @State private var isProcessing = false
    @State private var status = ""
    @State private var processingTime: Double?
    @State private var item: PhotosPickerItem?
    @State private var imageEmbedding: MLMultiArray?
    @State private var encoderModel: MLModel?
    @State private var decoderModel: MLModel?
    @State private var isEncoderLoaded = false

    private var inputSize: Int { model.configInt("input_size") ?? 1024 }

    var body: some View {
        VStack(spacing: 0) {
            // Image + tap area
            ZStack {
                if let img = outputImage ?? inputImage {
                    GeometryReader { geo in
                        let size = fitSize(imageSize: img.size, in: geo.size)
                        ZStack {
                            Image(uiImage: img)
                                .resizable()
                                .frame(width: size.width, height: size.height)
                                .contentShape(Rectangle())
                                .onTapGesture { location in
                                    guard isEncoderLoaded else { return }
                                    let normX = location.x / size.width
                                    let normY = location.y / size.height
                                    tapPoints.append(CGPoint(x: normX, y: normY))
                                    runDecoder()
                                }

                            // Point indicators
                            ForEach(tapPoints.indices, id: \.self) { i in
                                Circle()
                                    .fill(.blue)
                                    .frame(width: 14, height: 14)
                                    .overlay(Circle().stroke(.white, lineWidth: 2))
                                    .position(x: tapPoints[i].x * size.width,
                                              y: tapPoints[i].y * size.height)
                            }
                        }
                        .position(x: geo.size.width / 2, y: geo.size.height / 2)
                    }
                } else {
                    VStack(spacing: 12) {
                        Image(systemName: "hand.tap")
                            .font(.system(size: 60)).foregroundStyle(.secondary)
                        Text("Select a photo, then tap to segment")
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            // Status
            VStack(spacing: 8) {
                HStack {
                    if let t = processingTime {
                        Text(String(format: "%.2fs", t)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    }
                    Spacer()
                    if isProcessing {
                        ProgressView().controlSize(.small)
                        Text(status).font(.caption).foregroundStyle(.secondary)
                    }
                    if inputImage != nil && !isEncoderLoaded && !isProcessing {
                        Text("Encoding image…").font(.caption).foregroundStyle(.orange)
                    }
                }

                HStack(spacing: 12) {
                    PhotosPicker(selection: $item, matching: .images) {
                        Label("Select Photo", systemImage: "photo.badge.plus")
                    }
                    .buttonStyle(.bordered)
                    .disabled(isProcessing)

                    if !tapPoints.isEmpty {
                        Button {
                            tapPoints.removeAll()
                            outputImage = nil
                            processingTime = nil
                        } label: {
                            Label("Clear", systemImage: "arrow.counterclockwise")
                        }
                        .buttonStyle(.bordered)
                    }

                    if let output = outputImage {
                        Button {
                            UIImageWriteToSavedPhotosAlbum(output, nil, nil, nil)
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

    // MARK: - Load photo and encode

    private func loadPhoto() {
        guard let item else { return }
        outputImage = nil; tapPoints.removeAll(); imageEmbedding = nil; isEncoderLoaded = false
        isProcessing = true; status = "Loading…"

        Task {
            guard let data = try? await item.loadTransferable(type: Data.self),
                  let img = UIImage(data: data) else {
                await MainActor.run { isProcessing = false; status = "Failed" }; return
            }
            await MainActor.run { inputImage = img }
            await encodeImage(img)
        }
    }

    private func encodeImage(_ image: UIImage) async {
        do {
            status = "Loading encoder…"
            let encName = model.configString("encoder") ?? model.files.first?.name ?? ""
            let enc = try await ModelLoader.load(for: model, named: encName)
            let decName = model.configString("decoder")
                ?? model.files.dropFirst().first?.name ?? model.files.last?.name ?? ""
            let dec = try await ModelLoader.load(for: model, named: decName)

            status = "Encoding image…"
            guard let cgImage = ImageUtils.normalizeOrientation(image) else {
                await MainActor.run { isProcessing = false; status = "Image error" }; return
            }
            guard let pb = ImageUtils.pixelBuffer(from: cgImage, width: inputSize, height: inputSize) else {
                await MainActor.run { isProcessing = false; status = "Prep failed" }; return
            }

            // Run encoder
            let inputDesc = enc.modelDescription.inputDescriptionsByName
            let inputName = inputDesc.keys.first ?? "image"
            let input = try MLDictionaryFeatureProvider(dictionary: [inputName: pb])
            let encOutput = try await enc.prediction(from: input)

            // Get embedding (first MultiArray output)
            var embedding: MLMultiArray?
            for name in encOutput.featureNames {
                if let arr = encOutput.featureValue(for: name)?.multiArrayValue {
                    embedding = arr; break
                }
            }

            guard let emb = embedding else {
                await MainActor.run { isProcessing = false; status = "Encoding failed" }; return
            }

            await MainActor.run {
                encoderModel = enc
                decoderModel = dec
                imageEmbedding = emb
                isEncoderLoaded = true
                isProcessing = false; status = ""
            }
        } catch {
            await MainActor.run { isProcessing = false; status = "Error: \(error.localizedDescription)" }
        }
    }

    // MARK: - Decode mask

    private func runDecoder() {
        guard let decoderModel, let imageEmbedding, let inputImage,
              !tapPoints.isEmpty else { return }
        isProcessing = true; status = "Segmenting…"

        Task {
            do {
                let start = CFAbsoluteTimeGetCurrent()

                // Build point coordinates: [N, 2] and labels: [N]
                let n = tapPoints.count
                let coords = try MLMultiArray(shape: [1, NSNumber(value: n), 2], dataType: .float32)
                let labels = try MLMultiArray(shape: [1, NSNumber(value: n)], dataType: .float32)
                let ptr = coords.dataPointer.assumingMemoryBound(to: Float.self)
                let lptr = labels.dataPointer.assumingMemoryBound(to: Float.self)
                for i in 0..<n {
                    ptr[i * 2] = Float(tapPoints[i].x) * Float(inputSize)
                    ptr[i * 2 + 1] = Float(tapPoints[i].y) * Float(inputSize)
                    lptr[i] = 1.0  // foreground point
                }

                // Build decoder input from model description
                let desc = decoderModel.modelDescription.inputDescriptionsByName
                var inputDict: [String: MLFeatureValue] = [:]
                for (name, fd) in desc {
                    if fd.type == .multiArray {
                        let shape = fd.multiArrayConstraint?.shape.map { $0.intValue } ?? []
                        // Match by shape heuristic
                        if shape.last == 2 || name.lowercased().contains("point") || name.lowercased().contains("coord") {
                            inputDict[name] = MLFeatureValue(multiArray: coords)
                        } else if shape.count <= 2 && shape.last == n || name.lowercased().contains("label") {
                            inputDict[name] = MLFeatureValue(multiArray: labels)
                        } else {
                            // Assume it's the image embedding
                            inputDict[name] = MLFeatureValue(multiArray: imageEmbedding)
                        }
                    }
                }

                let input = try MLDictionaryFeatureProvider(dictionary: inputDict)
                let output = try decoderModel.prediction(from: input)
                let elapsed = CFAbsoluteTimeGetCurrent() - start

                // Find mask output
                var maskArr: MLMultiArray?
                for name in output.featureNames {
                    if let arr = output.featureValue(for: name)?.multiArrayValue {
                        maskArr = arr; break
                    }
                }

                guard let mask = maskArr else {
                    await MainActor.run { isProcessing = false; status = "No mask output" }; return
                }

                // Apply mask overlay
                guard let cgImage = ImageUtils.normalizeOrientation(inputImage) else {
                    await MainActor.run { isProcessing = false; status = "Image error" }; return
                }
                let result = applyMaskOverlay(mask: mask, on: cgImage)

                await MainActor.run {
                    outputImage = result
                    processingTime = elapsed; isProcessing = false; status = ""
                }
            } catch {
                await MainActor.run { isProcessing = false; status = "Error: \(error.localizedDescription)" }
            }
        }
    }

    // MARK: - Mask overlay

    private func applyMaskOverlay(mask: MLMultiArray, on cgImage: CGImage) -> UIImage? {
        let shape = mask.shape.map { $0.intValue }
        // Find spatial dims — take last two
        let mH: Int, mW: Int
        if shape.count >= 2 {
            mH = shape[shape.count - 2]; mW = shape[shape.count - 1]
        } else {
            return nil
        }

        // Extract and threshold mask values
        let total = mask.count
        var values = [Float](repeating: 0, count: total)
        let ptr = mask.dataPointer
        if mask.dataType == .float16 {
            let fp16 = ptr.assumingMemoryBound(to: Float16.self)
            for i in 0..<total { values[i] = Float(fp16[i]) }
        } else {
            let fp32 = ptr.assumingMemoryBound(to: Float.self)
            for i in 0..<total { values[i] = fp32[i] }
        }

        // Use last mH*mW values (skip batch/multi-mask dims)
        let offset = total - mH * mW
        let maskValues = Array(values[offset...])

        let w = cgImage.width, h = cgImage.height
        var origPixels = [UInt8](repeating: 0, count: w * h * 4)
        guard let ctx = CGContext(data: &origPixels, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))

        // Blend: highlight masked region in blue
        var result = origPixels
        for y in 0..<h {
            let my = min(Int(Float(y) * Float(mH) / Float(h)), mH - 1)
            for x in 0..<w {
                let mx = min(Int(Float(x) * Float(mW) / Float(w)), mW - 1)
                let mVal = maskValues[my * mW + mx]
                if mVal > 0 {
                    let idx = (y * w + x) * 4
                    result[idx]   = UInt8(min(255, Float(origPixels[idx]) * 0.5 + 50))
                    result[idx+1] = UInt8(min(255, Float(origPixels[idx+1]) * 0.5 + 80))
                    result[idx+2] = UInt8(min(255, Float(origPixels[idx+2]) * 0.5 + 180))
                }
            }
        }
        return ImageUtils.makeRGBA(pixels: result, width: w, height: h)
    }

    private func fitSize(imageSize: CGSize, in containerSize: CGSize) -> CGSize {
        let scale = min(containerSize.width / imageSize.width, containerSize.height / imageSize.height)
        return CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
    }
}
