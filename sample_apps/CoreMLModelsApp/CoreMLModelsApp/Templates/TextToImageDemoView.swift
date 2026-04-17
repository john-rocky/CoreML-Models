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
    @StateObject private var session = ModelSession<DiffusionAssets>()

    private struct DiffusionAssets {
        let tokenizer: BPETokenizerSimple
        let textEncoder: MLModel
        let unetChunk1: MLModel?
        let unetChunk2: MLModel?
        let vaeDecoder: MLModel
    }

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
                TimingsLabel(loadSec: session.loadTimeSec, inferSec: generationTime)
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
        .task {
            session.ensure {
                let vocabFile = model.configString("vocab_file") ?? "vocab.json"
                let mergesFile = model.configString("merges_file") ?? "merges.txt"
                let tokenizer = try BPETokenizerSimple(
                    vocabURL: ModelLoader.auxFileURL(modelId: model.id, fileName: vocabFile),
                    mergesURL: ModelLoader.auxFileURL(modelId: model.id, fileName: mergesFile)
                )

                let teFile = model.configString("text_encoder")
                    ?? model.files.first { $0.name.lowercased().contains("textencoder") }?.name ?? model.files[0].name
                let te = try await ModelLoader.load(for: model, named: teFile)

                let chunk1File = model.configString("unet_chunk1")
                    ?? model.files.first { $0.name.lowercased().contains("chunk1") }?.name
                let chunk2File = model.configString("unet_chunk2")
                    ?? model.files.first { $0.name.lowercased().contains("chunk2") }?.name
                let c1 = try await { () async throws -> MLModel? in
                    guard let f = chunk1File else { return nil }
                    return try await ModelLoader.load(for: model, named: f)
                }()
                let c2 = try await { () async throws -> MLModel? in
                    guard let f = chunk2File else { return nil }
                    return try await ModelLoader.load(for: model, named: f)
                }()

                let vaeFile = model.configString("vae_decoder")
                    ?? model.files.first { $0.name.lowercased().contains("decoder") }?.name ?? model.files.last!.name
                let vae = try await ModelLoader.load(for: model, named: vaeFile)

                return DiffusionAssets(
                    tokenizer: tokenizer,
                    textEncoder: te,
                    unetChunk1: c1,
                    unetChunk2: c2,
                    vaeDecoder: vae
                )
            }
        }
    }

    // MARK: - Generation

    private func generate() async {
        isGenerating = true; resultImage = nil; generationTime = nil
        let currentPrompt = prompt
        let currentSeed = seed

        do {
            status = session.loadTimeSec == nil ? "Loading models…" : "Preparing…"
            let assets = try await session.get()
            let tokenizer = assets.tokenizer
            let textEncoder = assets.textEncoder
            let unetChunk1 = assets.unetChunk1
            let unetChunk2 = assets.unetChunk2
            let vaeDecoder = assets.vaeDecoder

            let latentSize = model.configInt("latent_size") ?? 64
            let latentChannels = model.configInt("latent_channels") ?? 4
            let steps = max(1, model.configInt("steps") ?? 1)
            let guidanceScale = model.configDouble("guidance_scale").map { Float($0) } ?? 1.0
            let scaleFactor: Float = Float(model.configDouble("vae_scale_factor") ?? 0.18215)

            // Precompute SD1.5 cumulative alpha products (scaled_linear beta schedule 0.00085 -> 0.012 over 1000 steps)
            // Matches TCDScheduler in HyperSDDemo.
            let trainSteps = 1000
            let betaStart: Float = 0.00085
            let betaEnd: Float = 0.012
            var alphasCumProd = [Float](repeating: 1.0, count: trainSteps)
            do {
                let sqrtStart = sqrt(betaStart)
                let sqrtEnd = sqrt(betaEnd)
                var running: Float = 1.0
                for i in 0..<trainSteps {
                    let frac = trainSteps > 1 ? Float(i) / Float(trainSteps - 1) : 0
                    let sqrtBeta = sqrtStart + (sqrtEnd - sqrtStart) * frac
                    let beta = sqrtBeta * sqrtBeta
                    running *= (1.0 - beta)
                    alphasCumProd[i] = running
                }
            }

            // TCD trailing timestep schedule (matches diffusers TCDScheduler).
            let stepRatio = Float(trainSteps) / Float(steps)
            var timesteps: [Int] = []
            var fi = Float(steps)
            while fi >= 1 {
                timesteps.append(Int((fi * stepRatio).rounded()) - 1)
                fi -= 1
            }

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

            // 3. Generate noise (seeded). Single-sample latent [1, C, H, W]; we duplicate into a
            // batched sample [2, C, H, W] at each UNet call so CFG can subtract uncond from cond.
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
            for (stepIdx, t) in timesteps.enumerated() {
                status = "Denoising step \(stepIdx + 1)/\(timesteps.count)…"

                // Build batched latent [2, C, H, W] by duplicating the current single-sample latent.
                let batchedLatent = try MLMultiArray(shape: [2, NSNumber(value: latentChannels),
                                                             NSNumber(value: latentSize), NSNumber(value: latentSize)],
                                                    dataType: .float16)
                let bPtr = batchedLatent.dataPointer.assumingMemoryBound(to: Float16.self)
                let lPtr = latent.dataPointer.assumingMemoryBound(to: Float16.self)
                for i in 0..<count { bPtr[i] = lPtr[i]; bPtr[count + i] = lPtr[i] }

                // Timestep tensor shape [2] for the batched UNet (matches HyperSDDemo Unet.swift).
                let timestepArr = try MLMultiArray(shape: [2], dataType: .float16)
                let tsPtr = timestepArr.dataPointer.assumingMemoryBound(to: Float16.self)
                tsPtr[0] = Float16(Float(t)); tsPtr[1] = Float16(Float(t))

                var noiseResidual: MLMultiArray?

                if let c1 = unetChunk1, let c2 = unetChunk2, let hs = hiddenStates {
                    // Chunked UNet
                    let c1Names = c1.modelDescription.inputDescriptionsByName
                    var c1Dict: [String: Any] = [:]
                    for (key, _) in c1Names {
                        if key.contains("sample") || key.contains("latent") { c1Dict[key] = batchedLatent }
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

                guard let noise = noiseResidual else { continue }

                // Apply CFG: guided = uncond + scale * (cond - uncond). Noise is [2, C, H, W] in
                // the order [uncond, cond] because that is how input_ids were batched above.
                let nPtr = noise.dataPointer.assumingMemoryBound(to: Float16.self)
                var guided = [Float](repeating: 0, count: count)
                for i in 0..<count {
                    let uncond = Float(nPtr[i])
                    let cond = Float(nPtr[count + i])
                    guided[i] = uncond + guidanceScale * (cond - uncond)
                }

                // TCD scheduler step (eta=1.0 / pred_original_sample form, matches TCDScheduler.swift):
                //   x0 = (sample - sqrt(1 - alphaProd_t) * noise) / sqrt(alphaProd_t)
                let clampedT = min(max(t, 0), alphasCumProd.count - 1)
                let alphaProdT = alphasCumProd[clampedT]
                let sqrtAlpha = sqrt(alphaProdT)
                let sqrtBeta = sqrt(max(0, 1 - alphaProdT))
                for i in 0..<count {
                    let predX0 = (Float(lPtr[i]) - sqrtBeta * guided[i]) / sqrtAlpha
                    lPtr[i] = Float16(predX0)
                }
            }

            // 5. VAE decode
            status = "Decoding image…"
            // Scale latent for VAE
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

        // Pad with EOS (matches HyperSDDemo BPETokenizer which uses "<|endoftext|>" as padToken)
        // or truncate to maxLength.
        var result = [Int32](repeating: Int32(eosToken), count: maxLength)
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
