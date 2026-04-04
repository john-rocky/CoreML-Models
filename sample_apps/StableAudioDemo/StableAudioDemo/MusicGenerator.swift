import CoreML
import AVFoundation

class MusicGenerator: ObservableObject {
    private var t5Encoder: MLModel?
    private var numberEmbedder: MLModel?
    private var dit: MLModel?
    private var vaeDecoder: MLModel?

    @Published var isReady = false
    @Published var status = "Loading models..."

    static let sampleRate: Double = 44100
    static let latentChannels = 64
    static let latentLength = 256        // 524288 / 2048
    static let audioLength = 524288      // ~11.9s at 44.1kHz
    static let maxTokenLength = 64
    static let secondsMax: Float = 256.0

    private var pieceToID: [String: Int32] = [:]
    private let eosTokenID: Int32 = 1
    private let padTokenID: Int32 = 0

    init() {
        loadVocabulary()
        loadModels()
    }

    // MARK: - Model Loading

    private func loadVocabulary() {
        guard let url = Bundle.main.url(forResource: "t5_vocab", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: String] else {
            return
        }
        for (idStr, piece) in dict {
            if let id = Int32(idStr) { pieceToID[piece] = id }
        }
    }

    private func loadModels() {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self, let resourcePath = Bundle.main.resourcePath else { return }
            let fm = FileManager.default
            guard let items = try? fm.contentsOfDirectory(atPath: resourcePath) else { return }

            let modelItems = items.filter { $0.hasSuffix(".mlmodelc") }

            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndGPU

            // DiT (FP32 compute) must run on CPU to avoid GPU background permission issues
            let ditConfig = MLModelConfiguration()
            ditConfig.computeUnits = .cpuOnly

            let modelDefs: [(String, String, MLModelConfiguration)] = [
                ("T5Encoder", "Loading T5 Encoder...", config),
                ("NumberEmbedder", "Loading NumberEmbedder...", config),
                ("DiT", "Loading DiT...", ditConfig),
                ("VAEDecoder", "Loading VAE Decoder...", config),
            ]

            for (key, message, cfg) in modelDefs {
                DispatchQueue.main.async { self.status = message }
                guard let item = modelItems.first(where: { $0.contains(key) }) else { continue }
                let url = URL(fileURLWithPath: (resourcePath as NSString).appendingPathComponent(item))
                guard let model = try? MLModel(contentsOf: url, configuration: cfg) else { continue }
                switch key {
                case "T5Encoder": self.t5Encoder = model
                case "NumberEmbedder": self.numberEmbedder = model
                case "DiT": self.dit = model
                case "VAEDecoder": self.vaeDecoder = model
                default: break
                }
            }

            let ready = self.t5Encoder != nil && self.numberEmbedder != nil
                && self.dit != nil && self.vaeDecoder != nil
            DispatchQueue.main.async {
                self.isReady = ready
                self.status = ready ? "Ready" : "Failed to load models"
            }
        }
    }

    // MARK: - T5 SentencePiece Tokenizer

    func tokenize(_ text: String) -> [Int32] {
        let input = "\u{2581}" + text.lowercased().replacingOccurrences(of: " ", with: "\u{2581}")
        var tokens: [Int32] = []
        var pos = input.startIndex

        while pos < input.endIndex {
            var matched = false
            for len in stride(from: min(20, input.distance(from: pos, to: input.endIndex)), through: 1, by: -1) {
                let end = input.index(pos, offsetBy: len, limitedBy: input.endIndex) ?? input.endIndex
                let sub = String(input[pos..<end])
                if let id = pieceToID[sub] {
                    tokens.append(id)
                    pos = end
                    matched = true
                    break
                }
            }
            if !matched { pos = input.index(after: pos) }
        }
        tokens.append(eosTokenID)

        while tokens.count < Self.maxTokenLength { tokens.append(padTokenID) }
        if tokens.count > Self.maxTokenLength {
            tokens = Array(tokens.prefix(Self.maxTokenLength - 1)) + [eosTokenID]
        }
        return tokens
    }

    // MARK: - Generation

    func generate(
        prompt: String,
        seconds: Float,
        steps: Int,
        seed: UInt64,
        progress: @escaping (Int, Int, String) -> Void
    ) async throws -> URL {
        guard let t5Encoder, let numberEmbedder, let dit, let vaeDecoder else {
            throw GeneratorError.modelNotLoaded
        }

        // Step 1: Tokenize and encode text
        progress(0, steps + 2, "Encoding text...")
        let tokens = tokenize(prompt)
        let inputIDs = try MLMultiArray(shape: [1, NSNumber(value: Self.maxTokenLength)], dataType: .int32)
        for i in 0..<Self.maxTokenLength { inputIDs[i] = NSNumber(value: tokens[i]) }

        let t5Input = try MLDictionaryFeatureProvider(dictionary: ["input_ids": inputIDs])
        let t5Output = try await t5Encoder.prediction(from: t5Input)
        guard let textEmbeddings = t5Output.featureValue(for: "text_embeddings")?.multiArrayValue else {
            throw GeneratorError.predictionFailed
        }

        // Sanitize T5 output (INT8 quantization can produce occasional NaN)
        // and zero out padding positions
        let attentionTokenCount = tokens.firstIndex(of: padTokenID) ?? Self.maxTokenLength
        for t in 0..<Self.maxTokenLength {
            for c in 0..<768 {
                let v = textEmbeddings[[0, t, c] as [NSNumber]].floatValue
                if v.isNaN || v.isInfinite || t >= attentionTokenCount {
                    textEmbeddings[[0, t, c] as [NSNumber]] = 0
                }
            }
        }

        // Step 2: Encode seconds_total
        let normalizedSeconds = min(max(seconds, 0), Self.secondsMax) / Self.secondsMax
        let secondsInput = try MLMultiArray(shape: [1], dataType: .float16)
        secondsInput[0] = NSNumber(value: normalizedSeconds)

        let numInput = try MLDictionaryFeatureProvider(dictionary: ["normalized_seconds": secondsInput])
        let numOutput = try await numberEmbedder.prediction(from: numInput)
        guard let secondsEmbedding = numOutput.featureValue(for: "seconds_embedding")?.multiArrayValue else {
            throw GeneratorError.predictionFailed
        }

        // Step 3: Build conditioning tensors
        // cross_attn_cond [1, 65, 768] = cat(textEmbeddings, secondsEmbedding)
        let crossAttn = try MLMultiArray(shape: [1, 65, 768], dataType: .float16)
        for t in 0..<64 {
            for c in 0..<768 {
                crossAttn[[0, t, c] as [NSNumber]] = textEmbeddings[[0, t, c] as [NSNumber]]
            }
        }
        for c in 0..<768 {
            crossAttn[[0, 64, c] as [NSNumber]] = secondsEmbedding[[0, c] as [NSNumber]]
        }

        // global_embed [1, 768] = seconds embedding
        let globalEmbed = try MLMultiArray(shape: [1, 768], dataType: .float16)
        for c in 0..<768 {
            globalEmbed[[0, c] as [NSNumber]] = secondsEmbedding[[0, c] as [NSNumber]]
        }

        // Step 4: Diffusion loop (rectified flow, euler sampler)
        progress(1, steps + 2, "Starting diffusion...")
        var latent = try createNoise(seed: seed)
        let schedule = makeSchedule(steps: steps)

        for i in 0..<steps {
            progress(i + 1, steps + 2, "Step \(i + 1)/\(steps)")

            let tCurr = schedule[i]
            let tNext = schedule[i + 1]
            let dt = tNext - tCurr

            let timestep = try MLMultiArray(shape: [1], dataType: .float16)
            timestep[0] = NSNumber(value: tCurr)

            let ditInput = try MLDictionaryFeatureProvider(dictionary: [
                "latent": latent, "timestep": timestep,
                "cross_attn_cond": crossAttn, "global_embed": globalEmbed,
            ])
            let ditOutput = try await dit.prediction(from: ditInput)
            guard let velocity = ditOutput.featureValue(for: "velocity")?.multiArrayValue else {
                throw GeneratorError.predictionFailed
            }

            latent = try eulerStep(latent: latent, velocity: velocity, dt: Float(dt))
        }

        // Step 5: Decode latent to audio
        progress(steps + 1, steps + 2, "Decoding audio...")
        let vaeInput = try MLDictionaryFeatureProvider(dictionary: ["latent": latent])
        let vaeOutput = try await vaeDecoder.prediction(from: vaeInput)
        guard let audio = vaeOutput.featureValue(for: "audio")?.multiArrayValue else {
            throw GeneratorError.predictionFailed
        }

        // Step 6: Save as WAV
        let outputURL = try saveWAV(audio: audio, seconds: seconds)
        progress(steps + 2, steps + 2, "Done!")
        return outputURL
    }

    // MARK: - Helpers

    private func createNoise(seed: UInt64) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: [1, NSNumber(value: Self.latentChannels), NSNumber(value: Self.latentLength)], dataType: .float16)
        let ptr = arr.dataPointer.assumingMemoryBound(to: Float16.self)
        let count = Self.latentChannels * Self.latentLength

        var rng = SeededRNG(seed: seed)
        for i in stride(from: 0, to: count - 1, by: 2) {
            let u1 = Float.random(in: Float.leastNormalMagnitude...1.0, using: &rng)
            let u2 = Float.random(in: 0...1, using: &rng)
            let r = sqrtf(-2.0 * logf(u1))
            let theta = 2.0 * Float.pi * u2
            ptr[i] = Float16(r * cosf(theta))
            if i + 1 < count { ptr[i + 1] = Float16(r * sinf(theta)) }
        }
        if count % 2 != 0 {
            let u1 = Float.random(in: Float.leastNormalMagnitude...1.0, using: &rng)
            let u2 = Float.random(in: 0...1, using: &rng)
            ptr[count - 1] = Float16(sqrtf(-2.0 * logf(u1)) * cosf(2.0 * Float.pi * u2))
        }
        return arr
    }

    private func makeSchedule(steps: Int) -> [Float] {
        var logsnr = [Float](repeating: 0, count: steps + 1)
        for i in 0...steps {
            logsnr[i] = -6.0 + Float(i) / Float(steps) * 8.0
        }
        var t = logsnr.map { 1.0 / (1.0 + expf($0)) }
        t[0] = 1.0
        t[steps] = 0
        return t
    }

    private func eulerStep(latent: MLMultiArray, velocity: MLMultiArray, dt: Float) throws -> MLMultiArray {
        let result = try MLMultiArray(shape: latent.shape, dataType: .float16)
        let count = Self.latentChannels * Self.latentLength
        let xPtr = latent.dataPointer.assumingMemoryBound(to: Float16.self)
        let rPtr = result.dataPointer.assumingMemoryBound(to: Float16.self)

        if velocity.dataType == .float32 {
            let vPtr = velocity.dataPointer.assumingMemoryBound(to: Float.self)
            for i in 0..<count { rPtr[i] = Float16(Float(xPtr[i]) + dt * vPtr[i]) }
        } else {
            let vPtr = velocity.dataPointer.assumingMemoryBound(to: Float16.self)
            for i in 0..<count { rPtr[i] = Float16(Float(xPtr[i]) + dt * Float(vPtr[i])) }
        }
        return result
    }

    private func saveWAV(audio: MLMultiArray, seconds: Float) throws -> URL {
        let totalSamples = Self.audioLength
        let trimmedSamples = min(Int(seconds * Float(Self.sampleRate)), totalSamples)

        let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Self.sampleRate,
            channels: 2,
            interleaved: false
        )!
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(trimmedSamples)) else {
            throw GeneratorError.audioSaveFailed
        }
        buffer.frameLength = AVAudioFrameCount(trimmedSamples)

        guard let ch0 = buffer.floatChannelData?[0],
              let ch1 = buffer.floatChannelData?[1] else {
            throw GeneratorError.audioSaveFailed
        }

        for i in 0..<trimmedSamples {
            ch0[i] = audio[[0, 0, i] as [NSNumber]].floatValue
            ch1[i] = audio[[0, 1, i] as [NSNumber]].floatValue
        }

        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("stable_audio_\(Int(Date().timeIntervalSince1970)).wav")
        let file = try AVAudioFile(forWriting: outputURL, settings: format.settings)
        try file.write(from: buffer)
        return outputURL
    }
}

// MARK: - Seeded RNG

struct SeededRNG: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }
}

enum GeneratorError: LocalizedError {
    case modelNotLoaded, predictionFailed, audioSaveFailed

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded: return "Models not loaded"
        case .predictionFailed: return "Model prediction failed"
        case .audioSaveFailed: return "Failed to save audio"
        }
    }
}
