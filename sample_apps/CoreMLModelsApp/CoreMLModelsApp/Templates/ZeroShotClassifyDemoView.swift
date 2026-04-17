import SwiftUI
import PhotosUI
import CoreML

/// Zero-shot image classification: photo + class list → per-class scores.
/// Used by: SigLIP.
///
/// Expected manifest config:
/// ```
/// {
///   "input_size": 224,
///   "image_encoder": "SigLIP_ImageEncoder.mlpackage",
///   "text_encoder": "SigLIP_TextEncoder.mlpackage",
///   "vocab_file": "siglip_vocab.json",
///   "prompt_template": "a photo of a {}",
///   "logit_scale": 117.33
/// }
/// ```
struct ZeroShotClassifyDemoView: View {
    let model: ModelEntry

    @State private var inputImage: UIImage?
    @State private var classText = "cat, dog, bird, car, tree, person"
    @State private var results: [(String, Float)] = []
    @State private var isProcessing = false
    @State private var status = ""
    @State private var processingTime: Double?
    @State private var item: PhotosPickerItem?
    @StateObject private var session = ModelSession<(img: MLModel, txt: MLModel?)>()

    var body: some View {
        VStack(spacing: 0) {
            if let img = inputImage {
                Image(uiImage: img).resizable().aspectRatio(contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .frame(maxHeight: 300)
                    .padding(.horizontal)
            } else {
                VStack(spacing: 12) {
                    Image(systemName: "tag").font(.system(size: 60)).foregroundStyle(.secondary)
                    Text("Select a photo and enter class names").foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }

            if !results.isEmpty {
                List {
                    ForEach(results, id: \.0) { (label, score) in
                        HStack {
                            Text(label).font(.body)
                            Spacer()
                            Text(String(format: "%.1f%%", score * 100))
                                .font(.body.monospacedDigit())
                                .foregroundStyle(score > 0.5 ? .primary : .secondary)
                            GeometryReader { geo in
                                RoundedRectangle(cornerRadius: 4)
                                    .fill(score > 0.5 ? Color.accentColor : Color(.systemGray4))
                                    .frame(width: geo.size.width * CGFloat(score))
                            }
                            .frame(width: 80, height: 8)
                        }
                    }
                }
                .listStyle(.plain)
            }

            VStack(spacing: 12) {
                TextField("Classes (comma-separated)", text: $classText)
                    .textFieldStyle(.roundedBorder).font(.callout)

                HStack {
                    TimingsLabel(loadSec: session.loadTimeSec, inferSec: processingTime)
                    Spacer()
                    if isProcessing { ProgressView().controlSize(.small); Text(status).font(.caption) }
                }

                HStack(spacing: 12) {
                    PhotosPicker(selection: $item, matching: .images) {
                        Label("Photo", systemImage: "photo.badge.plus")
                    }.buttonStyle(.bordered)

                    Button {
                        if let img = inputImage { Task { await runClassification(on: img) } }
                    } label: {
                        Label("Classify", systemImage: "sparkles").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isProcessing || inputImage == nil || classText.isEmpty)
                }
            }
            .padding()
        }
        .task {
            session.ensure {
                let imgEncoderName = model.configString("image_encoder")
                    ?? model.files.first { $0.name.lowercased().contains("image") }?.name
                    ?? model.files[0].name
                let txtEncoderName = model.configString("text_encoder")
                    ?? model.files.first { $0.name.lowercased().contains("text") }?.name
                let img = try await ModelLoader.load(for: model, named: imgEncoderName)
                let txt = try await { () async throws -> MLModel? in
                    guard let n = txtEncoderName else { return nil }
                    return try await ModelLoader.load(for: model, named: n)
                }()
                return (img: img, txt: txt)
            }
        }
        .onChange(of: item) { _, _ in loadPhoto() }
    }

    private func loadPhoto() {
        guard let item else { return }
        Task {
            if let data = try? await item.loadTransferable(type: Data.self),
               let img = UIImage(data: data) {
                await MainActor.run { inputImage = img; results = [] }
            }
        }
    }

    private func runClassification(on image: UIImage) async {
        let classes = classText.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
        guard !classes.isEmpty else { return }

        isProcessing = true
        status = session.loadTimeSec == nil ? "Loading models…" : "Encoding…"
        do {
            let (imgEncoder, txtEncoder) = try await session.get()

            // Encode image
            status = "Encoding image…"
            let inputSize = model.configInt("input_size") ?? 224
            guard let pb = ImageUtils.pixelBuffer(from: image, width: inputSize, height: inputSize) else {
                isProcessing = false; status = "Image prep failed"; return
            }

            let imgInputName = imgEncoder.modelDescription.inputDescriptionsByName.first {
                $0.value.type == .image
            }?.key ?? "image"

            let start = CFAbsoluteTimeGetCurrent()
            let imgInput = try MLDictionaryFeatureProvider(dictionary: [imgInputName: pb])
            let imgOutput = try await imgEncoder.prediction(from: imgInput)

            let imgEmbName = imgOutput.featureNames.first ?? "image_embedding"
            guard let imgEmb = imgOutput.featureValue(for: imgEmbName)?.multiArrayValue else {
                isProcessing = false; status = "No image embedding"; return
            }
            let imageVec = ImageUtils.extractFloats(imgEmb)

            // Encode text (if text encoder model exists)
            var classScores: [Float]
            if let txtEncoder {
                // Load vocab if available
                let vocabFile = model.configString("vocab_file")
                    ?? model.files.first { ($0.kind ?? "") == "vocab" }?.name
                let vocab = vocabFile.flatMap { loadVocab(modelId: model.id, fileName: $0) }

                // SigLIP was trained on caption-style text; a bare "{}" template
                // gives misaligned embeddings. Default to a full caption prompt.
                let rawTemplate = model.configString("prompt_template") ?? "a photo of a {}"
                let template = (rawTemplate == "{}" || rawTemplate.isEmpty) ? "a photo of a {}" : rawTemplate
                let logitScale = model.configDouble("logit_scale").map { Float($0) } ?? 100.0

                status = "Encoding text…"
                // Keep (label, logit) paired so a failed-to-encode class can't
                // shift the ranking of the classes after it.
                var logits: [(String, Float)] = []
                for cls in classes {
                    let prompt = template.replacingOccurrences(of: "{}", with: cls)
                    let tokenIds = tokenize(prompt, vocab: vocab)

                    let tokenArr = try MLMultiArray(shape: [1, NSNumber(value: tokenIds.count)], dataType: .int32)
                    for (i, tok) in tokenIds.enumerated() { tokenArr[i] = NSNumber(value: tok) }

                    let txtInputName = txtEncoder.modelDescription.inputDescriptionsByName.keys
                        .first { $0.contains("input") } ?? "input_ids"
                    let txtInput = try MLDictionaryFeatureProvider(dictionary: [txtInputName: tokenArr])
                    let txtOutput = try await txtEncoder.prediction(from: txtInput)

                    let txtEmbName = txtOutput.featureNames.first ?? "text_embedding"
                    guard let txtEmb = txtOutput.featureValue(for: txtEmbName)?.multiArrayValue else { continue }
                    let textVec = ImageUtils.extractFloats(txtEmb)
                    logits.append((cls, ImageUtils.cosineSimilarity(imageVec, textVec) * logitScale))
                }

                let maxLogit = logits.map(\.1).max() ?? 0
                let exps = logits.map { exp($0.1 - maxLogit) }
                let sumExp = exps.reduce(0, +)
                let pairs: [(String, Float)] = exps.enumerated().map { (i, e) in
                    (logits[i].0, sumExp > 0 ? e / sumExp : 0)
                }

                let elapsed = CFAbsoluteTimeGetCurrent() - start
                let ranked = pairs.sorted { $0.1 > $1.1 }
                await MainActor.run {
                    results = ranked
                    processingTime = elapsed
                    isProcessing = false; status = ""
                }
                return
            }

            // Single model with built-in text handling — use output directly
            classScores = Array(repeating: 1.0 / Float(classes.count), count: classes.count)

            let elapsed = CFAbsoluteTimeGetCurrent() - start

            let ranked = zip(classes, classScores).sorted { $0.1 > $1.1 }.map { ($0.0, $0.1) }
            await MainActor.run {
                results = ranked
                processingTime = elapsed
                isProcessing = false; status = ""
            }
        } catch {
            await MainActor.run { isProcessing = false; status = "Error: \(error.localizedDescription)" }
        }
    }

    // MARK: - Tokenizer

    private func loadVocab(modelId: String, fileName: String) -> [String: Int]? {
        let url = ModelLoader.auxFileURL(modelId: modelId, fileName: fileName)
        guard let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }

        // Handle both {piece: id} and {id: piece} formats
        var vocab: [String: Int] = [:]
        for (k, v) in json {
            if let id = v as? Int { vocab[k] = id }
            else if let piece = v as? String, let id = Int(k) { vocab[piece] = id }
        }
        return vocab
    }

    private func tokenize(_ text: String, vocab: [String: Int]?) -> [Int32] {
        guard let vocab else {
            // Fallback: simple character tokenization
            return text.unicodeScalars.map { Int32($0.value % 30000) }
        }

        // SentencePiece-style greedy longest match
        let processed = "\u{2581}" + text.lowercased().replacingOccurrences(of: " ", with: "\u{2581}")
        var tokens: [Int32] = []
        var i = processed.startIndex

        while i < processed.endIndex {
            var bestLen = 0
            var bestId: Int32 = 0
            for len in (1...min(20, processed.distance(from: i, to: processed.endIndex))).reversed() {
                let end = processed.index(i, offsetBy: len)
                let piece = String(processed[i..<end])
                if let id = vocab[piece] { bestLen = len; bestId = Int32(id); break }
            }
            if bestLen == 0 { i = processed.index(after: i) }
            else { tokens.append(bestId); i = processed.index(i, offsetBy: bestLen) }
        }
        // SigLIP / SentencePiece models require EOS at the end of the sequence;
        // without it the text embedding is offset and ranking is wrong.
        if let eosID = vocab["</s>"] { tokens.append(Int32(eosID)) }
        return tokens
    }

    private func softmax(_ logits: [Float]) -> [Float] {
        guard !logits.isEmpty else { return [] }
        let maxL = logits.max()!
        let exps = logits.map { exp($0 - maxL) }
        let sum = exps.reduce(0, +)
        return exps.map { $0 / sum }
    }
}
