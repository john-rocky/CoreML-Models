import SwiftUI
import PhotosUI
import CoreML

/// Image captioning / VQA: photo → text via encoder-decoder.
/// Used by: Florence-2.
///
/// Expected manifest config:
/// ```
/// {
///   "vision_encoder": "Florence2VisionEncoder.mlpackage",
///   "text_encoder": "Florence2TextEncoder.mlpackage",
///   "decoder": "Florence2Decoder.mlpackage",
///   "vocab_file": "florence2_vocab.json",
///   "image_size": 768,
///   "max_tokens": 256,
///   "tasks": {
///     "caption": [0, 2264, 473, 5, 2274, 6190, 116, 2],
///     "detailed_caption": [0, 2264, 473, 5, 31962, 2274, 6190, 116, 2],
///     "ocr": [0, 2264, 473, 5, 71307, 116, 2]
///   }
/// }
/// ```
struct ImageToTextDemoView: View {
    let model: ModelEntry

    @State private var inputImage: UIImage?
    @State private var generatedText = ""
    @State private var selectedTask = "caption"
    @State private var isProcessing = false
    @State private var status = ""
    @State private var processingTime: Double?
    @State private var item: PhotosPickerItem?

    private var taskNames: [String] {
        if let tasks = model.demo.config?["tasks"]?.value as? [String: Any] {
            return Array(tasks.keys).sorted()
        }
        return ["caption"]
    }

    var body: some View {
        VStack(spacing: 0) {
            if let img = inputImage {
                Image(uiImage: img).resizable().aspectRatio(contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .frame(maxHeight: 300).padding(.horizontal)
            } else {
                VStack(spacing: 12) {
                    Image(systemName: "text.below.photo").font(.system(size: 60)).foregroundStyle(.secondary)
                    Text("Select a photo to generate a caption").foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }

            if !generatedText.isEmpty {
                ScrollView {
                    Text(generatedText)
                        .font(.body).textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                }
                .frame(maxHeight: 200)
                .background(Color(.systemGray6)).clipShape(RoundedRectangle(cornerRadius: 8))
                .padding(.horizontal)
            }

            Spacer()

            VStack(spacing: 12) {
                if taskNames.count > 1 {
                    Picker("Task", selection: $selectedTask) {
                        ForEach(taskNames, id: \.self) { task in
                            Text(task.replacingOccurrences(of: "_", with: " ").capitalized).tag(task)
                        }
                    }
                    .pickerStyle(.segmented)
                }

                HStack {
                    if let t = processingTime {
                        Text(String(format: "%.2fs", t)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    }
                    Spacer()
                    if isProcessing { ProgressView().controlSize(.small); Text(status).font(.caption) }
                }

                HStack(spacing: 12) {
                    PhotosPicker(selection: $item, matching: .images) {
                        Label("Photo", systemImage: "photo.badge.plus")
                    }.buttonStyle(.bordered)

                    Button {
                        if let img = inputImage { Task { await runCaption(on: img) } }
                    } label: {
                        Label("Caption", systemImage: "text.bubble").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isProcessing || inputImage == nil)
                }
            }
            .padding()
        }
        .onChange(of: item) { _, _ in loadPhoto() }
    }

    private func loadPhoto() {
        guard let item else { return }
        Task {
            if let data = try? await item.loadTransferable(type: Data.self),
               let img = UIImage(data: data) {
                await MainActor.run { inputImage = img; generatedText = "" }
            }
        }
    }

    private func runCaption(on image: UIImage) async {
        isProcessing = true; generatedText = ""
        do {
            // Load vocab
            status = "Loading vocab…"
            let vocabFile = model.configString("vocab_file")
                ?? model.files.first { ($0.kind ?? "") == "vocab" }?.name
            var reverseVocab: [Int: String] = [:]
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

            // Load models
            status = "Loading vision encoder…"
            let veFile = model.configString("vision_encoder")
                ?? model.files.first { $0.name.lowercased().contains("vision") }?.name ?? model.files[0].name
            let visionEncoder = try await ModelLoader.load(for: model, named: veFile)

            status = "Loading text encoder…"
            let teFile = model.configString("text_encoder")
                ?? model.files.first { $0.name.lowercased().contains("textencoder") }?.name
            let textEncoder = teFile != nil ? try await ModelLoader.load(for: model, named: teFile!) : nil

            status = "Loading decoder…"
            let decFile = model.configString("decoder")
                ?? model.files.first { $0.name.lowercased().contains("decoder") }?.name
            let decoder = decFile != nil ? try await ModelLoader.load(for: model, named: decFile!) : nil

            let imageSize = model.configInt("image_size") ?? 768
            let maxTokens = model.configInt("max_tokens") ?? 256

            guard let pb = ImageUtils.pixelBuffer(from: image, width: imageSize, height: imageSize) else {
                isProcessing = false; status = "Image prep failed"; return
            }

            let start = CFAbsoluteTimeGetCurrent()

            // 1. Vision encode
            status = "Encoding image…"
            let veInputName = visionEncoder.modelDescription.inputDescriptionsByName.first {
                $0.value.type == .image
            }?.key ?? "image"
            let veOutput = try await visionEncoder.prediction(from:
                MLDictionaryFeatureProvider(dictionary: [veInputName: pb]))
            let imageFeatures = veOutput.featureNames.compactMap {
                veOutput.featureValue(for: $0)?.multiArrayValue
            }.first

            // Get task input_ids from config
            let taskIds: [Int]
            if let tasks = model.demo.config?["tasks"]?.value as? [String: Any],
               let ids = tasks[selectedTask] as? [Int] {
                taskIds = ids
            } else {
                taskIds = [0, 2]  // BOS + EOS fallback
            }

            // 2. Text encode (if separate text encoder)
            var encoderHiddenStates: MLMultiArray?
            if let textEncoder, let imgFeat = imageFeatures {
                status = "Encoding text…"
                let inputIds = try MLMultiArray(shape: [1, NSNumber(value: taskIds.count)], dataType: .int32)
                for (i, tok) in taskIds.enumerated() { inputIds[i] = NSNumber(value: tok) }

                var teDict: [String: Any] = ["input_ids": inputIds]
                let teNames = textEncoder.modelDescription.inputDescriptionsByName
                for (key, _) in teNames {
                    if key.contains("image") || key.contains("feature") { teDict[key] = imgFeat }
                }

                let teOutput = try await textEncoder.prediction(from: MLDictionaryFeatureProvider(dictionary: teDict))
                encoderHiddenStates = teOutput.featureNames.compactMap {
                    teOutput.featureValue(for: $0)?.multiArrayValue
                }.first
            } else if let imgFeat = imageFeatures {
                encoderHiddenStates = imgFeat
            }

            // 3. Autoregressive decode
            guard let decoder, let encHS = encoderHiddenStates else {
                // Single model — try to get text output directly
                if let textOut = veOutput.featureNames.compactMap({
                    veOutput.featureValue(for: $0)?.stringValue
                }).first {
                    await MainActor.run {
                        generatedText = textOut
                        processingTime = CFAbsoluteTimeGetCurrent() - start
                        isProcessing = false; status = ""
                    }
                    return
                }
                isProcessing = false; status = "Missing decoder model"; return
            }

            status = "Generating text…"
            let decoderStartTokenId: Int32 = 2  // </s> for Florence-2
            let eosTokenId: Int32 = 2
            var generatedIds: [Int32] = [decoderStartTokenId]

            for _ in 0..<maxTokens {
                let decInputIds = try MLMultiArray(shape: [1, NSNumber(value: generatedIds.count)], dataType: .int32)
                for (i, tok) in generatedIds.enumerated() { decInputIds[i] = NSNumber(value: tok) }

                let decNames = decoder.modelDescription.inputDescriptionsByName
                var decDict: [String: Any] = [:]
                for (key, _) in decNames {
                    if key.contains("decoder") && key.contains("input") { decDict[key] = decInputIds }
                    else if key.contains("encoder") || key.contains("hidden") { decDict[key] = encHS }
                }
                if decDict.isEmpty { decDict = ["decoder_input_ids": decInputIds, "encoder_hidden_states": encHS] }

                let decOutput = try await decoder.prediction(from: MLDictionaryFeatureProvider(dictionary: decDict))

                // Argmax on last token logits
                guard let logits = decOutput.featureNames.compactMap({
                    decOutput.featureValue(for: $0)?.multiArrayValue
                }).first else { break }

                let shape = logits.shape.map { $0.intValue }
                let vocabSize = shape.last ?? 0
                let seqLen = shape.count == 3 ? shape[1] : 1
                let lastTokenOffset = (seqLen - 1) * vocabSize

                var maxVal: Float = -.greatestFiniteMagnitude
                var maxIdx: Int32 = 0
                for v in 0..<vocabSize {
                    let val = ImageUtils.readFloat(logits, at: lastTokenOffset + v)
                    if val > maxVal { maxVal = val; maxIdx = Int32(v) }
                }

                if maxIdx == eosTokenId { break }
                generatedIds.append(maxIdx)
            }

            // Decode tokens to text
            let text = generatedIds.dropFirst().compactMap { id -> String? in
                guard let piece = reverseVocab[Int(id)] else { return nil }
                if [0, 1, 2].contains(Int(id)) { return nil }  // Skip special tokens
                return piece
            }.joined()
                .replacingOccurrences(of: "\u{0120}", with: " ")  // GPT-2 space encoding
                .trimmingCharacters(in: .whitespacesAndNewlines)

            let elapsed = CFAbsoluteTimeGetCurrent() - start

            await MainActor.run {
                generatedText = text
                processingTime = elapsed
                isProcessing = false; status = ""
            }
        } catch {
            await MainActor.run { isProcessing = false; status = "Error: \(error.localizedDescription)" }
        }
    }
}
