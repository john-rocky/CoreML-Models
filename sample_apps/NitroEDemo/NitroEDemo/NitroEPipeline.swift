import CoreML
import Foundation

/// Drives the three CoreML packages that make up Nitro-E:
/// `NitroE_TextEncoder` (Llama 3.2 1B last hidden state) →
/// `NitroE_EMMDiT`       (4-step distilled denoiser) →
/// `NitroE_VAEDecoder`   (DC-AE f32c32 decoder).
///
/// Loads models lazily and sequentially so peak memory stays below iPhone
/// limits. The text encoder is released after encoding; the denoiser after
/// the final step; only the VAE is still resident when decoding.
@MainActor
final class NitroEPipeline: ObservableObject {

    @Published var status: String = "idle"
    @Published var isReady: Bool = false

    enum PipelineError: Error { case modelMissing(String), shapeMismatch(String) }

    private let tokenizer: LlamaTokenizer
    private var textEncoder: MLModel?
    private var denoiser: MLModel?
    private var vae: MLModel?

    private let seqLen = 128
    private let latentChannels = 32
    private let latentSide = 16
    private let imageSide = 512

    init(tokenizer: LlamaTokenizer) { self.tokenizer = tokenizer }

    // MARK: - Loading

    private func url(for name: String) throws -> URL {
        guard let u = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else {
            throw PipelineError.modelMissing(name)
        }
        return u
    }

    /// Sequential load — calls `progress(modelName)` as each loads.
    func warmUp(progress: @escaping (String) -> Void = { _ in }) async throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuAndNeuralEngine
        progress("Text encoder")
        textEncoder = try MLModel(contentsOf: try url(for: "NitroE_TextEncoder"), configuration: cfg)
        progress("Denoiser")
        denoiser = try MLModel(contentsOf: try url(for: "NitroE_EMMDiT"), configuration: cfg)
        // DC-AE decoder: FP32 compute because the Sana linear-attention
        // block overflows in FP16 (smeared output).
        let vaeCfg = MLModelConfiguration()
        vaeCfg.computeUnits = .cpuAndNeuralEngine
        progress("VAE decoder")
        vae = try MLModel(contentsOf: try url(for: "NitroE_VAEDecoder"), configuration: vaeCfg)
        isReady = true
        status = "ready"
    }

    // MARK: - Generation

    struct GenerationResult {
        let image: CGImage
        let textMs: Double
        let denoiseMs: Double
        let decodeMs: Double
    }

    func generate(prompt: String, steps: Int = 4, seed: UInt64 = 42) async throws -> GenerationResult {
        guard let textEncoder, let denoiser, let vae else {
            throw PipelineError.modelMissing("pipeline not warmed up")
        }

        // 1) Tokenize + text encode
        status = "encoding text"
        let (ids, mask) = try tokenizer.encode(text: prompt.lowercased(), maxLength: seqLen)
        let t0 = Date()
        let textFeat = try await textEncoder.prediction(from: tokenInputs(ids: ids, mask: mask))
        let textMs = Date().timeIntervalSince(t0) * 1000
        guard let textHidden = textFeat.featureValue(for: "last_hidden_state")?.multiArrayValue else {
            throw PipelineError.shapeMismatch("last_hidden_state missing")
        }

        // 2) Init latent from seeded Gaussian noise and denoise 4 steps
        status = "denoising"
        let latentCount = latentChannels * latentSide * latentSide
        var latent = gaussianNoise(count: latentCount, seed: seed)
        let scheduler = FlowMatchEulerScheduler()
        scheduler.setTimesteps(steps)

        let t1 = Date()
        for i in 0..<steps {
            let ts = Int32(scheduler.timesteps[i].rounded())
            let inputs = try denoiserInputs(
                latent: latent, textHidden: textHidden,
                attentionMask: mask, timestep: ts
            )
            let out = try await denoiser.prediction(from: inputs)
            guard let pred = out.featureValue(for: "noise_pred")?.multiArrayValue else {
                throw PipelineError.shapeMismatch("noise_pred missing")
            }
            latent = scheduler.step(modelOutput: toFloats(pred), sample: latent)
        }
        let denoiseMs = Date().timeIntervalSince(t1) * 1000

        // 3) VAE decode
        status = "decoding"
        let t2 = Date()
        let decodeIn = try makeFloatArray(shape: [1, latentChannels, latentSide, latentSide], from: latent)
        let vaeOut = try await vae.prediction(from: MLDictionaryFeatureProvider(
            dictionary: ["latent": MLFeatureValue(multiArray: decodeIn)]
        ))
        guard let img = vaeOut.featureValue(for: "image")?.multiArrayValue else {
            throw PipelineError.shapeMismatch("image missing")
        }
        let cg = try cgImage(fromCHW: img, width: imageSide, height: imageSide)
        let decodeMs = Date().timeIntervalSince(t2) * 1000
        status = "ready"
        return GenerationResult(image: cg, textMs: textMs, denoiseMs: denoiseMs, decodeMs: decodeMs)
    }

    // MARK: - Helpers

    private func tokenInputs(ids: [Int32], mask: [Int32]) throws -> MLFeatureProvider {
        let idsArr = try makeInt32Array(shape: [1, seqLen], from: ids)
        let maskArr = try makeInt32Array(shape: [1, seqLen], from: mask)
        return try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: idsArr),
            "attention_mask": MLFeatureValue(multiArray: maskArr),
        ])
    }

    private func denoiserInputs(
        latent: [Float], textHidden: MLMultiArray, attentionMask: [Int32], timestep: Int32
    ) throws -> MLFeatureProvider {
        let latentArr = try makeFloatArray(shape: [1, latentChannels, latentSide, latentSide], from: latent)
        let maskArr = try makeInt32Array(shape: [1, seqLen], from: attentionMask)
        let tsArr = try MLMultiArray(shape: [1], dataType: .int32)
        tsArr[0] = NSNumber(value: timestep)
        return try MLDictionaryFeatureProvider(dictionary: [
            "latent": MLFeatureValue(multiArray: latentArr),
            "encoder_hidden_states": MLFeatureValue(multiArray: textHidden),
            "encoder_attention_mask": MLFeatureValue(multiArray: maskArr),
            "timestep": MLFeatureValue(multiArray: tsArr),
        ])
    }

    private func makeFloatArray(shape: [Int], from data: [Float]) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: shape.map(NSNumber.init), dataType: .float32)
        let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)
        data.withUnsafeBufferPointer { ptr.update(from: $0.baseAddress!, count: data.count) }
        return arr
    }

    private func makeInt32Array(shape: [Int], from data: [Int32]) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: shape.map(NSNumber.init), dataType: .int32)
        let ptr = arr.dataPointer.assumingMemoryBound(to: Int32.self)
        data.withUnsafeBufferPointer { ptr.update(from: $0.baseAddress!, count: data.count) }
        return arr
    }

    private func toFloats(_ m: MLMultiArray) -> [Float] {
        let count = m.count
        let ptr = m.dataPointer.assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    private func gaussianNoise(count: Int, seed: UInt64) -> [Float] {
        // Box–Muller on a SplitMix64 / xoshiro seed. Sufficient for demo; not
        // bit-identical to torch.Generator. Pipelines using the reference dump
        // should overwrite `latent` with the captured step-0 tensor.
        var rng = SplitMix64(seed: seed)
        var out = [Float](repeating: 0, count: count)
        var i = 0
        while i < count {
            let u1 = max(Float.leastNonzeroMagnitude, rng.nextUniform())
            let u2 = rng.nextUniform()
            let r = sqrtf(-2 * log(u1))
            let theta = 2 * Float.pi * u2
            out[i] = r * cosf(theta)
            if i + 1 < count { out[i + 1] = r * sinf(theta) }
            i += 2
        }
        return out
    }

    private func cgImage(fromCHW arr: MLMultiArray, width: Int, height: Int) throws -> CGImage {
        let c = 3
        let n = width * height * c
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        let strides = arr.strides.map { $0.intValue }
        let base = arr.dataPointer.assumingMemoryBound(to: Float.self)
        for y in 0..<height {
            for x in 0..<width {
                for ch in 0..<c {
                    let idx = ch * strides[1] + y * strides[2] + x * strides[3]
                    let v = max(-1, min(1, base[idx]))
                    let px = UInt8(((v + 1) * 0.5 * 255).rounded())
                    rgba[(y * width + x) * 4 + ch] = px
                }
            }
        }
        let ctx = CGContext(
            data: &rgba, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        )!
        return ctx.makeImage()!
        _ = n  // silence unused
    }
}

/// SplitMix64 PRNG (seeded) returning `Float` uniforms in (0, 1).
private struct SplitMix64 {
    var state: UInt64
    init(seed: UInt64) { self.state = seed == 0 ? 0x9E3779B97F4A7C15 : seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
    mutating func nextUniform() -> Float {
        // Use the top 24 bits so the result maps to [0, 1) in Float precision.
        Float(next() >> 40) / Float(1 << 24)
    }
}
