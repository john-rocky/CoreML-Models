import SwiftUI
import CoreML

/// Text-to-image generation: prompt → image via diffusion pipeline.
/// Used by: Hyper-SD (SD1.5 + 1-step TCD).
///
/// Expected manifest config:
/// ```
/// {
///   "text_encoder": "HyperSDTextEncoder.mlpackage",
///   "unet_chunk1": "HyperSDUnetChunk1.mlpackage",
///   "unet_chunk2": "HyperSDUnetChunk2.mlpackage",
///   "vae_decoder": "HyperSDVAEDecoder.mlpackage",
///   "vocab_file": "vocab.json",
///   "merges_file": "merges.txt",
///   "image_size": 512,
///   "latent_channels": 4,
///   "latent_size": 64,
///   "steps": 1,
///   "guidance_scale": 1.0,
///   "scheduler": "tcd"
/// }
/// ```
struct TextToImageDemoView: View {
    let model: ModelEntry

    @State private var prompt = "A photo of a cat wearing sunglasses on a beach"
    @State private var resultImage: CGImage?
    @State private var seed: UInt32 = 42
    @State private var isGenerating = false
    @State private var status = ""
    @State private var generationTime: Double?

    var body: some View {
        VStack(spacing: 8) {
            ZStack {
                if let image = resultImage {
                    Image(decorative: image, scale: 1.0).resizable()
                        .interpolation(.high).aspectRatio(contentMode: .fit)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                } else {
                    RoundedRectangle(cornerRadius: 12).fill(Color(.systemGray6))
                        .overlay {
                            VStack(spacing: 8) {
                                Image(systemName: "wand.and.stars").font(.largeTitle).foregroundStyle(.tertiary)
                                Text("Enter a prompt to generate").font(.subheadline).foregroundStyle(.tertiary)
                            }
                        }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .padding(.horizontal)

            HStack(spacing: 8) {
                if isGenerating { ProgressView().controlSize(.small) }
                Text(status).font(.caption2).foregroundStyle(.secondary)
                Spacer()
                if let t = generationTime {
                    Text(String(format: "%.1fs", t)).font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
                }
            }
            .padding(.horizontal)

            TextField("Describe an image...", text: $prompt, axis: .vertical)
                .textFieldStyle(.roundedBorder).lineLimit(1...2).padding(.horizontal)

            HStack(spacing: 8) {
                Button {
                    Task { await generate() }
                } label: {
                    if isGenerating {
                        HStack { ProgressView().controlSize(.small); Text("Generating…") }.frame(maxWidth: .infinity)
                    } else {
                        Label("Generate", systemImage: "sparkles").frame(maxWidth: .infinity)
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(isGenerating || prompt.isEmpty)

                if let image = resultImage {
                    Button {
                        UIImageWriteToSavedPhotosAlbum(UIImage(cgImage: image), nil, nil, nil)
                    } label: {
                        Image(systemName: "square.and.arrow.down")
                    }.buttonStyle(.bordered)
                }

                Button { seed = UInt32.random(in: 0...99999) } label: {
                    Image(systemName: "dice")
                }.buttonStyle(.bordered)
            }
            .padding(.horizontal)
        }
        .padding(.vertical, 8)
    }

    // MARK: - Generation

    private func generate() async {
        isGenerating = true; resultImage = nil; generationTime = nil
        let currentPrompt = prompt
        let currentSeed = seed

        do {
            // Load vocab + merges
            status = "Loading tokenizer…"
            let vocabFile = model.configString("vocab_file") ?? "vocab.json"
            let mergesFile = model.configString("merges_file") ?? "merges.txt"
            let tokenizer = try BPETokenizerSimple(
                vocabURL: ModelLoader.auxFileURL(modelId: model.id, fileName: vocabFile),
                mergesURL: ModelLoader.auxFileURL(modelId: model.id, fileName: mergesFile)
            )

            // Load models
            status = "Loading text encoder…"
            let teFile = model.configString("text_encoder")
                ?? model.files.first { $0.name.lowercased().contains("textencoder") }?.name ?? model.files[0].name
            let textEncoder = try await ModelLoader.load(for: model, named: teFile)

            status = "Loading UNet…"
            let chunk1File = model.configString("unet_chunk1")
                ?? model.files.first { $0.name.lowercased().contains("chunk1") }?.name
            let chunk2File = model.configString("unet_chunk2")
                ?? model.files.first { $0.name.lowercased().contains("chunk2") }?.name

            let unetChunk1 = chunk1File != nil ? try await ModelLoader.load(for: model, named: chunk1File!) : nil
            let unetChunk2 = chunk2File != nil ? try await ModelLoader.load(for: model, named: chunk2File!) : nil

            status = "Loading VAE decoder…"
            let vaeFile = model.configString("vae_decoder")
                ?? model.files.first { $0.name.lowercased().contains("decoder") }?.name ?? model.files.last!.name
            let vaeDecoder = try await ModelLoader.load(for: model, named: vaeFile)

            let latentSize = model.configInt("latent_size") ?? 64
            let latentChannels = model.configInt("latent_channels") ?? 4
            let steps = model.configInt("steps") ?? 1
            let guidanceScale = model.configDouble("guidance_scale").map { Float($0) } ?? 1.0

            let start = CFAbsoluteTimeGetCurrent()

            // 1. Tokenize
            status = "Tokenizing…"
            let tokens = tokenizer.tokenize(currentPrompt, maxLength: 77)
            let uncondTokens = tokenizer.tokenize("", maxLength: 77)

            // 2. Text encoding (batch unconditional + conditional)
            status = "Encoding text…"
            let inputIds = try MLMultiArray(shape: [2, 77], dataType: .int32)
            for i in 0..<77 {
                inputIds[i] = NSNumber(value: uncondTokens[i])
                inputIds[77 + i] = NSNumber(value: tokens[i])
            }
            let teInput = try MLDictionaryFeatureProvider(dictionary: ["input_ids": inputIds])
            let teOutput = try await textEncoder.prediction(from: teInput)
            let hiddenStates = teOutput.featureValue(for: teOutput.featureNames.first ?? "encoder_hidden_states")?.multiArrayValue

            // 3. Generate noise (seeded)
            status = "Generating latent…"
            let latent = try MLMultiArray(shape: [1, NSNumber(value: latentChannels),
                                                  NSNumber(value: latentSize), NSNumber(value: latentSize)],
                                         dataType: .float16)
            var rng = SeededRNG(seed: UInt64(currentSeed))
            let count = latent.count
            let fp16Ptr = latent.dataPointer.assumingMemoryBound(to: Float16.self)
            for i in 0..<count {
                let u1 = Float.random(in: Float.leastNonzeroMagnitude...1, using: &rng)
                let u2 = Float.random(in: 0...1, using: &rng)
                fp16Ptr[i] = Float16(sqrt(-2 * log(u1)) * cos(2 * .pi * u2))
            }

            // 4. UNet denoising loop (TCD scheduler)
            for step in 0..<steps {
                status = "Denoising step \(step + 1)/\(steps)…"

                let tcdTimestep: Float = steps == 1 ? 999.0 : Float(999 - step * (999 / steps))
                let timestepArr = try MLMultiArray(shape: [1], dataType: .float16)
                timestepArr.dataPointer.assumingMemoryBound(to: Float16.self)[0] = Float16(tcdTimestep)

                // Build UNet input with text hidden states
                var noiseResidual: MLMultiArray?

                if let c1 = unetChunk1, let c2 = unetChunk2, let hs = hiddenStates {
                    // Chunked UNet
                    let c1Names = c1.modelDescription.inputDescriptionsByName
                    var c1Dict: [String: Any] = [:]
                    for (key, desc) in c1Names {
                        if key.contains("sample") || key.contains("latent") { c1Dict[key] = latent }
                        else if key.contains("timestep") || key.contains("t_emb") { c1Dict[key] = timestepArr }
                        else if key.contains("hidden") || key.contains("encoder") { c1Dict[key] = hs }
                    }
                    let c1Out = try await c1.prediction(from: MLDictionaryFeatureProvider(dictionary: c1Dict))

                    // Pass chunk1 outputs to chunk2
                    var c2Dict: [String: Any] = [:]
                    let c2Names = c2.modelDescription.inputDescriptionsByName
                    for (key, _) in c2Names {
                        if let fv = c1Out.featureValue(for: key) { c2Dict[key] = fv.multiArrayValue ?? fv }
                        else if key.contains("timestep") || key.contains("t_emb") { c2Dict[key] = timestepArr }
                        else if key.contains("hidden") || key.contains("encoder") { c2Dict[key] = hs }
                    }
                    let c2Out = try await c2.prediction(from: MLDictionaryFeatureProvider(dictionary: c2Dict))
                    noiseResidual = c2Out.featureNames.compactMap { c2Out.featureValue(for: $0)?.multiArrayValue }.first
                }

                // TCD scheduler step: x_pred = latent - sigma * noise
                if let noise = noiseResidual {
                    let nPtr = noise.dataPointer.assumingMemoryBound(to: Float16.self)
                    let lPtr = latent.dataPointer.assumingMemoryBound(to: Float16.self)

                    if guidanceScale != 1.0 {
                        // CFG: noise = uncond + scale * (cond - uncond)
                        // For 1-step Hyper-SD, guidance_scale = 1.0 so this is skipped
                        let half = count
                        for i in 0..<half {
                            let uncond = Float(nPtr[i])
                            let cond = Float(nPtr[half + i])
                            lPtr[i] = Float16(uncond + guidanceScale * (cond - uncond))
                        }
                    } else {
                        // Single step: latent = predicted x0
                        // TCD 1-step: x0 = latent - noise (simplified)
                        let beta: Float = 0.00085  // SD1.5 beta_start
                        let alpha: Float = 1.0 - beta
                        let sqrtAlpha = sqrt(alpha)
                        for i in 0..<count {
                            lPtr[i] = Float16((Float(lPtr[i]) - Float(nPtr[i]) * sqrt(1 - alpha)) / sqrtAlpha)
                        }
                    }
                }
            }

            // 5. VAE decode
            status = "Decoding image…"
            // Scale latent for VAE
            let scaleFactor: Float = 0.18215
            let lPtr = latent.dataPointer.assumingMemoryBound(to: Float16.self)
            for i in 0..<count { lPtr[i] = Float16(Float(lPtr[i]) / scaleFactor) }

            let vaeInputName = vaeDecoder.modelDescription.inputDescriptionsByName.keys.first ?? "z"
            let vaeInput = try MLDictionaryFeatureProvider(dictionary: [vaeInputName: latent])
            let vaeOutput = try await vaeDecoder.prediction(from: vaeInput)

            let elapsed = CFAbsoluteTimeGetCurrent() - start

            // Extract image from VAE output
            var resultCG: CGImage?
            for name in vaeOutput.featureNames {
                if let pb = vaeOutput.featureValue(for: name)?.imageBufferValue {
                    let ciImage = CIImage(cvPixelBuffer: pb)
                    resultCG = CIContext().createCGImage(ciImage, from: ciImage.extent)
                    break
                }
                if let arr = vaeOutput.featureValue(for: name)?.multiArrayValue {
                    if let img = ImageUtils.imageFromMultiArray(arr) {
                        resultCG = img.cgImage
                    }
                    break
                }
            }

            await MainActor.run {
                resultImage = resultCG
                generationTime = elapsed
                isGenerating = false; status = ""
            }
        } catch {
            await MainActor.run { isGenerating = false; status = "Error: \(error.localizedDescription)" }
        }
    }
}

// MARK: - Simple BPE Tokenizer

struct BPETokenizerSimple {
    private let vocab: [String: Int]
    private let merges: [(String, String)]
    private let bosToken: Int
    private let eosToken: Int

    init(vocabURL: URL, mergesURL: URL) throws {
        let vocabData = try Data(contentsOf: vocabURL)
        guard let vocabDict = try JSONSerialization.jsonObject(with: vocabData) as? [String: Int] else {
            throw NSError(domain: "BPE", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid vocab.json"])
        }
        self.vocab = vocabDict
        self.bosToken = vocabDict["<|startoftext|>"] ?? vocabDict["<s>"] ?? 49406
        self.eosToken = vocabDict["<|endoftext|>"] ?? vocabDict["</s>"] ?? 49407

        let mergesText = try String(contentsOf: mergesURL, encoding: .utf8)
        self.merges = mergesText.split(separator: "\n").dropFirst().compactMap { line in
            let parts = line.split(separator: " ")
            guard parts.count == 2 else { return nil }
            return (String(parts[0]), String(parts[1]))
        }
    }

    func tokenize(_ text: String, maxLength: Int) -> [Int32] {
        let cleaned = text.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        let words = cleaned.split(separator: " ").map { String($0) + "</w>" }

        var allTokens: [Int] = [bosToken]

        for word in words {
            var symbols = word.map { String($0) }

            // Apply BPE merges
            for (a, b) in merges {
                var i = 0
                while i < symbols.count - 1 {
                    if symbols[i] == a && symbols[i + 1] == b {
                        symbols[i] = a + b
                        symbols.remove(at: i + 1)
                    } else {
                        i += 1
                    }
                }
            }

            for sym in symbols {
                if let id = vocab[sym] { allTokens.append(id) }
            }
        }

        allTokens.append(eosToken)

        // Pad or truncate to maxLength
        var result = [Int32](repeating: 0, count: maxLength)
        for i in 0..<min(allTokens.count, maxLength) { result[i] = Int32(allTokens[i]) }
        return result
    }
}

// MARK: - Seeded RNG

struct SeededRNG: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return state
    }
}
