import SwiftUI
import PhotosUI
import CoreML

/// Image captioning with Photo + Camera tabs.
/// Matches Florence2Demo: task picker (Caption / Detailed / More Detail / OCR),
/// real-time camera captioning, copy button.
struct ImageToTextDemoView: View {
    let model: ModelEntry

    enum Tab: String, CaseIterable { case photo = "Photo", camera = "Camera" }
    enum CaptionTask: String, CaseIterable, Identifiable {
        case caption = "Caption"
        case detailed = "Detailed"
        case moreDetailed = "More Detail"
        case ocr = "OCR"
        var id: String { rawValue }

        /// Florence-2 prompt tokens. Matches `Florence2Task.inputIDs` in the
        /// Florence2Demo sample app. Hardcoded (not read from the manifest)
        /// because the prompts are intrinsic to Florence-2.
        var inputIDs: [Int] {
            switch self {
            case .caption:
                return [0, 2264, 473, 5, 2274, 6190, 116, 2]
            case .detailed:
                return [0, 47066, 21700, 11, 4617, 99, 16, 2343, 11, 5, 2274, 4, 2]
            case .moreDetailed:
                return [0, 47066, 21700, 19, 10, 17818, 99, 16, 2343, 11, 5, 2274, 4, 2]
            case .ocr:
                return [0, 2264, 16, 5, 2788, 11, 5, 2274, 116, 2]
            }
        }
    }

    @State private var tab: Tab = .photo
    @State private var inputImage: UIImage?
    @State private var resultText = ""
    @State private var selectedTask: CaptionTask = .caption
    @State private var questionText = ""
    @State private var isProcessing = false
    @State private var status = ""
    @State private var processingTime: Double?
    @State private var item: PhotosPickerItem?

    // Camera
    @State private var liveCaptionText = ""
    @State private var isCameraProcessing = false

    // Models
    @State private var visionEncoder: MLModel?
    @State private var textEncoder: MLModel?
    @State private var decoder: MLModel?
    @State private var reverseVocab: [Int: String] = [:]
    @State private var isModelLoaded = false
    @StateObject private var session = ModelSession<Void>()

    var body: some View {
        VStack(spacing: 0) {
            Picker("Tab", selection: $tab) {
                ForEach(Tab.allCases, id: \.self) { Text($0.rawValue).tag($0) }
            }.pickerStyle(.segmented).padding(.horizontal).padding(.top, 4)

            switch tab {
            case .photo: photoTab
            case .camera: cameraTab
            }
        }
        .task { await loadModels() }
        .onChange(of: item) { _, _ in loadPhoto() }
        .onDisappear {
            visionEncoder = nil
            textEncoder = nil
            decoder = nil
            reverseVocab.removeAll()
        }
    }

    // MARK: - Photo Tab

    @ViewBuilder
    private var photoTab: some View {
        VStack(spacing: 0) {
            if let img = inputImage {
                Image(uiImage: img).resizable().aspectRatio(contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .frame(maxHeight: 300).padding(.horizontal)
            } else {
                VStack(spacing: 12) {
                    Image(systemName: "text.below.photo").font(.system(size: 60)).foregroundStyle(.secondary)
                    Text("Select a photo to caption").foregroundStyle(.secondary)
                }.frame(maxWidth: .infinity, maxHeight: .infinity)
            }

            if !resultText.isEmpty {
                ScrollView {
                    HStack {
                        Text(resultText).font(.body).textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        Button { UIPasteboard.general.string = resultText } label: {
                            Image(systemName: "doc.on.doc").font(.caption)
                        }
                    }.padding()
                }
                .frame(maxHeight: 200)
                .background(Color(.systemGray6)).clipShape(RoundedRectangle(cornerRadius: 8))
                .padding(.horizontal)
            }

            Spacer()

            VStack(spacing: 8) {
                Picker("Task", selection: $selectedTask) {
                    ForEach(CaptionTask.allCases) { Text($0.rawValue).tag($0) }
                }.pickerStyle(.segmented)

                HStack {
                    TimingsLabel(loadSec: session.loadTimeSec, inferSec: processingTime)
                    Spacer()
                    if isProcessing { ProgressView().controlSize(.small); Text(status).font(.caption) }
                    if !isModelLoaded { Text("Loading models…").font(.caption).foregroundStyle(.orange) }
                }

                HStack(spacing: 12) {
                    PhotosPicker(selection: $item, matching: .images) {
                        Label("Photo", systemImage: "photo.badge.plus")
                    }.buttonStyle(.bordered)

                    Button {
                        if let img = inputImage { Task { await runCaption(on: img) } }
                    } label: {
                        Label("Run", systemImage: "play.fill").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isProcessing || inputImage == nil || !isModelLoaded)
                }
            }.padding()
        }
    }

    // MARK: - Camera Tab

    @ViewBuilder
    private var cameraTab: some View {
        ZStack(alignment: .bottom) {
            CameraView(position: .back) { pixelBuffer in
                guard isModelLoaded, !isCameraProcessing else { return }
                captionFrame(pixelBuffer)
            }

            // Caption overlay
            if !liveCaptionText.isEmpty || isCameraProcessing {
                HStack {
                    if isCameraProcessing { ProgressView().controlSize(.small).tint(.white) }
                    Text(liveCaptionText.isEmpty ? "Processing…" : liveCaptionText)
                        .font(.callout).foregroundStyle(.white)
                        .lineLimit(3)
                }
                .padding(.horizontal, 16).padding(.vertical, 10)
                .frame(maxWidth: .infinity)
                .background(.ultraThinMaterial)
            }
        }
    }

    // MARK: - Model Loading

    private func loadModels() async {
        session.ensure {
            let veFile = model.configString("vision_encoder")
                ?? model.files.first { $0.name.lowercased().contains("vision") }?.name ?? model.files[0].name
            let ve = try await ModelLoader.load(for: model, named: veFile)

            let teFile = model.configString("text_encoder")
                ?? model.files.first { $0.name.lowercased().contains("textencoder") }?.name
            let te = try await { () async throws -> MLModel? in
                guard let teFile else { return nil }
                return try await ModelLoader.load(for: model, named: teFile)
            }()

            let decFile = model.configString("decoder")
                ?? model.files.first { $0.name.lowercased().contains("decoder") }?.name
            let dec = try await { () async throws -> MLModel? in
                guard let decFile else { return nil }
                return try await ModelLoader.load(for: model, named: decFile)
            }()

            // Build reverse vocab
            var vocab: [Int: String] = [:]
            let vocabFile = model.configString("vocab_file")
                ?? model.files.first { ($0.kind ?? "") == "vocab" }?.name
            if let vf = vocabFile {
                let url = ModelLoader.auxFileURL(modelId: model.id, fileName: vf)
                if let data = try? Data(contentsOf: url),
                   let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    for (k, v) in json {
                        if let id = v as? Int { vocab[id] = k }
                        else if let piece = v as? String, let id = Int(k) { vocab[id] = piece }
                    }
                }
            }

            await MainActor.run {
                visionEncoder = ve
                textEncoder = te
                decoder = dec
                reverseVocab = vocab
                isModelLoaded = true
                status = ""
            }
        }
        do {
            try await session.get()
        } catch {
            await MainActor.run { status = "Load failed: \(error.localizedDescription)" }
        }
    }

    // MARK: - Photo Caption

    private func loadPhoto() {
        guard let item else { return }
        Task {
            if let data = try? await item.loadTransferable(type: Data.self),
               let img = UIImage(data: data) {
                await MainActor.run { inputImage = img; resultText = "" }
            }
        }
    }

    private func runCaption(on image: UIImage) async {
        isProcessing = true; resultText = ""
        do {
            let text = try await runPipeline(image: image, task: selectedTask)
            let elapsed = processingTime  // set inside pipeline
            await MainActor.run { resultText = text; isProcessing = false }
        } catch {
            await MainActor.run { isProcessing = false; status = "Error: \(error.localizedDescription)" }
        }
    }

    // MARK: - Camera Caption

    @State private var cameraFrameSkip = 0

    private func captionFrame(_ pixelBuffer: CVPixelBuffer) {
        cameraFrameSkip += 1
        guard cameraFrameSkip % 30 == 0 else { return }  // ~1 FPS for captioning

        isCameraProcessing = true
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cgImage = CIContext().createCGImage(ciImage, from: ciImage.extent) else {
            isCameraProcessing = false; return
        }

        Task {
            do {
                let text = try await runPipeline(image: UIImage(cgImage: cgImage), task: .caption)
                await MainActor.run { liveCaptionText = text; isCameraProcessing = false }
            } catch {
                await MainActor.run { isCameraProcessing = false }
            }
        }
    }

    // MARK: - Inference Pipeline

    private func runPipeline(image: UIImage, task: CaptionTask) async throws -> String {
        guard let visionEncoder, let textEncoder, let decoder else {
            throw NSError(domain: "F2", code: 1)
        }

        let imageSize = model.configInt("image_size") ?? 768
        let maxTokens = model.configInt("max_tokens") ?? 256

        guard let pb = ImageUtils.pixelBuffer(from: image, width: imageSize, height: imageSize) else {
            throw NSError(domain: "F2", code: 2)
        }

        let start = CFAbsoluteTimeGetCurrent()

        // 1. Vision encode — copy output so the VE buffer is free to be
        //    reused when later Core ML calls specialize on new shapes.
        await MainActor.run { status = "Encoding image…" }
        let veOutput = try await visionEncoder.prediction(from:
            MLDictionaryFeatureProvider(dictionary: ["image": pb]))
        guard let rawFeatures = veOutput.featureValue(for: "image_features")?.multiArrayValue else {
            throw NSError(domain: "F2", code: 3)
        }
        let imageFeatures = try copyMultiArray(rawFeatures)

        // 2. Text encode — task-specific prompt tokens drive detailed/OCR output.
        await MainActor.run { status = "Encoding text…" }
        let taskIds = task.inputIDs
        let inputIds = try MLMultiArray(shape: [1, NSNumber(value: taskIds.count)], dataType: .int32)
        for (i, tok) in taskIds.enumerated() {
            inputIds[[0, NSNumber(value: i)] as [NSNumber]] = NSNumber(value: tok)
        }
        let teOutput = try await textEncoder.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "image_features": imageFeatures,
            "input_ids": inputIds
        ]))
        guard let rawHS = teOutput.featureValue(for: "encoder_hidden_states")?.multiArrayValue else {
            throw NSError(domain: "F2", code: 4)
        }
        let encoderHiddenStates = try copyMultiArray(rawHS)

        // 3. Decoder autoregressive loop.
        await MainActor.run { status = "Generating…" }
        let eosTokenId: Int32 = 2
        var generatedIds: [Int32] = [eosTokenId]  // BART decoder_start_token_id = 2

        for _ in 0..<maxTokens {
            let decInputIds = try MLMultiArray(shape: [1, NSNumber(value: generatedIds.count)], dataType: .int32)
            for (i, tok) in generatedIds.enumerated() {
                decInputIds[[0, NSNumber(value: i)] as [NSNumber]] = NSNumber(value: tok)
            }
            let decOutput = try await decoder.prediction(from: MLDictionaryFeatureProvider(dictionary: [
                "decoder_input_ids": decInputIds,
                "encoder_hidden_states": encoderHiddenStates
            ]))
            guard let logits = decOutput.featureValue(for: "logits")?.multiArrayValue else { break }

            let nextToken = argmaxLastToken(logits)
            if nextToken == eosTokenId { break }
            generatedIds.append(nextToken)
        }

        await MainActor.run { processingTime = CFAbsoluteTimeGetCurrent() - start }

        return generatedIds.dropFirst().compactMap { id -> String? in
            guard let piece = reverseVocab[Int(id)], ![0, 1, 2].contains(Int(id)) else { return nil }
            return piece
        }.joined()
            .replacingOccurrences(of: "\u{0120}", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Helpers

    private func copyMultiArray(_ src: MLMultiArray) throws -> MLMultiArray {
        let dst = try MLMultiArray(shape: src.shape, dataType: src.dataType)
        let byteCount: Int
        switch src.dataType {
        case .float16: byteCount = src.count * 2
        case .float32, .int32: byteCount = src.count * 4
        case .float64: byteCount = src.count * 8
        default: byteCount = src.count * 4
        }
        memcpy(dst.dataPointer, src.dataPointer, byteCount)
        return dst
    }

    private func argmaxLastToken(_ logits: MLMultiArray) -> Int32 {
        let vocabSize = logits.shape.last?.intValue ?? 0
        let offset = logits.count - vocabSize
        if logits.dataType == .float16 {
            let ptr = logits.dataPointer.assumingMemoryBound(to: Float16.self)
            var maxIdx = 0
            var maxVal = ptr[offset]
            for i in 1..<vocabSize {
                let val = ptr[offset + i]
                if val > maxVal { maxVal = val; maxIdx = i }
            }
            return Int32(maxIdx)
        } else {
            let ptr = logits.dataPointer.assumingMemoryBound(to: Float32.self)
            var maxIdx = 0
            var maxVal = ptr[offset]
            for i in 1..<vocabSize {
                let val = ptr[offset + i]
                if val > maxVal { maxVal = val; maxIdx = i }
            }
            return Int32(maxIdx)
        }
    }
}
