import SwiftUI
import PhotosUI
import CoreML

/// Image captioning with Photo + Camera tabs.
/// Matches Florence2Demo: task picker (Caption/Detailed/OCR), real-time camera captioning,
/// question input for VQA mode, copy button.
struct ImageToTextDemoView: View {
    let model: ModelEntry

    enum Tab: String, CaseIterable { case photo = "Photo", camera = "Camera" }
    enum CaptionTask: String, CaseIterable, Identifiable {
        case caption = "Caption"
        case detailed = "Detailed"
        case ocr = "OCR"
        var id: String { rawValue }
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
                    if let t = processingTime {
                        Text(String(format: "%.2fs", t)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    }
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
        status = "Loading models…"
        do {
            let veFile = model.configString("vision_encoder")
                ?? model.files.first { $0.name.lowercased().contains("vision") }?.name ?? model.files[0].name
            visionEncoder = try await ModelLoader.load(for: model, named: veFile)

            let teFile = model.configString("text_encoder")
                ?? model.files.first { $0.name.lowercased().contains("textencoder") }?.name
            if let teFile { textEncoder = try await ModelLoader.load(for: model, named: teFile) }

            let decFile = model.configString("decoder")
                ?? model.files.first { $0.name.lowercased().contains("decoder") }?.name
            if let decFile { decoder = try await ModelLoader.load(for: model, named: decFile) }

            // Load vocab
            let vocabFile = model.configString("vocab_file")
                ?? model.files.first { ($0.kind ?? "") == "vocab" }?.name
            if let vf = vocabFile {
                let url = ModelLoader.auxFileURL(modelId: model.id, fileName: vf)
                if let data = try? Data(contentsOf: url),
                   let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    for (k, v) in json {
                        if let id = v as? Int { reverseVocab[id] = k }
                        else if let piece = v as? String, let id = Int(k) { reverseVocab[id] = piece }
                    }
                }
            }

            await MainActor.run { isModelLoaded = true; status = "" }
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
        guard let visionEncoder else { throw NSError(domain: "F2", code: 1) }

        let imageSize = model.configInt("image_size") ?? 768
        let maxTokens = model.configInt("max_tokens") ?? 256

        guard let pb = ImageUtils.pixelBuffer(from: image, width: imageSize, height: imageSize) else {
            throw NSError(domain: "F2", code: 2)
        }

        let start = CFAbsoluteTimeGetCurrent()

        // Vision encode
        await MainActor.run { status = "Encoding image…" }
        let veInputName = visionEncoder.modelDescription.inputDescriptionsByName.first {
            $0.value.type == .image
        }?.key ?? "image"
        let veOutput = try await visionEncoder.prediction(from:
            MLDictionaryFeatureProvider(dictionary: [veInputName: pb]))
        let imageFeatures = veOutput.featureNames.compactMap {
            veOutput.featureValue(for: $0)?.multiArrayValue
        }.first

        // Task input IDs
        let taskIds: [Int]
        if let tasks = model.demo.config?["tasks"]?.value as? [String: Any],
           let ids = tasks[task.rawValue.lowercased()] as? [Int] {
            taskIds = ids
        } else if let tasks = model.demo.config?["tasks"]?.value as? [String: Any],
                  let ids = tasks[task.rawValue.lowercased().replacingOccurrences(of: " ", with: "_")] as? [Int] {
            taskIds = ids
        } else {
            taskIds = [0, 2]
        }

        // Text encode
        var encoderHiddenStates: MLMultiArray?
        if let textEncoder, let imgFeat = imageFeatures {
            await MainActor.run { status = "Encoding text…" }
            let inputIds = try MLMultiArray(shape: [1, NSNumber(value: taskIds.count)], dataType: .int32)
            for (i, tok) in taskIds.enumerated() { inputIds[i] = NSNumber(value: tok) }
            var teDict: [String: Any] = ["input_ids": inputIds]
            for (key, _) in textEncoder.modelDescription.inputDescriptionsByName {
                if key.contains("image") || key.contains("feature") { teDict[key] = imgFeat }
            }
            let teOutput = try await textEncoder.prediction(from: MLDictionaryFeatureProvider(dictionary: teDict))
            encoderHiddenStates = teOutput.featureNames.compactMap { teOutput.featureValue(for: $0)?.multiArrayValue }.first
        } else {
            encoderHiddenStates = imageFeatures
        }

        // Autoregressive decode
        guard let decoder, let encHS = encoderHiddenStates else {
            // Single model — try text output directly
            if let text = veOutput.featureNames.compactMap({ veOutput.featureValue(for: $0)?.stringValue }).first {
                await MainActor.run { processingTime = CFAbsoluteTimeGetCurrent() - start }
                return text
            }
            throw NSError(domain: "F2", code: 3)
        }

        await MainActor.run { status = "Generating…" }
        let eosTokenId: Int32 = 2
        var generatedIds: [Int32] = [2]

        for _ in 0..<maxTokens {
            let decInputIds = try MLMultiArray(shape: [1, NSNumber(value: generatedIds.count)], dataType: .int32)
            for (i, tok) in generatedIds.enumerated() { decInputIds[i] = NSNumber(value: tok) }

            var decDict: [String: Any] = [:]
            for (key, _) in decoder.modelDescription.inputDescriptionsByName {
                if key.contains("decoder") && key.contains("input") { decDict[key] = decInputIds }
                else if key.contains("encoder") || key.contains("hidden") { decDict[key] = encHS }
            }
            if decDict.isEmpty { decDict = ["decoder_input_ids": decInputIds, "encoder_hidden_states": encHS] }

            let decOutput = try await decoder.prediction(from: MLDictionaryFeatureProvider(dictionary: decDict))
            guard let logits = decOutput.featureNames.compactMap({ decOutput.featureValue(for: $0)?.multiArrayValue }).first else { break }

            let shape = logits.shape.map { $0.intValue }
            let vocabSize = shape.last ?? 0
            let seqLen = shape.count == 3 ? shape[1] : 1
            let lastOffset = (seqLen - 1) * vocabSize

            var maxVal: Float = -.greatestFiniteMagnitude; var maxIdx: Int32 = 0
            for v in 0..<vocabSize {
                let val = ImageUtils.readFloat(logits, at: lastOffset + v)
                if val > maxVal { maxVal = val; maxIdx = Int32(v) }
            }
            if maxIdx == eosTokenId { break }
            generatedIds.append(maxIdx)
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        await MainActor.run { processingTime = elapsed }

        return generatedIds.dropFirst().compactMap { id -> String? in
            guard let piece = reverseVocab[Int(id)], ![0,1,2].contains(Int(id)) else { return nil }
            return piece
        }.joined()
            .replacingOccurrences(of: "\u{0120}", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
