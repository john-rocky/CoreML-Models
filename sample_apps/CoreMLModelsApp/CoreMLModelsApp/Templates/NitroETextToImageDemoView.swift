import SwiftUI
import CoreML
import UIKit

/// Nitro-E text-to-image: prompt → image via a 3-model pipeline.
/// Used by: Nitro-E (AMD 304M E-MMDiT, 4-step distilled).
///
/// Distinct from the `text_to_image` template because Nitro-E uses
/// Llama 3.2 1B (byte-level BPE) for text encoding and the FlowMatchEuler
/// scheduler instead of SD1.5 CLIP + TCD.
///
/// Expected manifest config:
/// ```
/// {
///   "text_encoder": "NitroE_TextEncoder.mlpackage.zip",
///   "denoiser":     "NitroE_EMMDiT.mlpackage.zip",
///   "vae_decoder":  "NitroE_VAEDecoder.mlpackage.zip",
///   "vocab_file":   "Llama3Vocab.json",
///   "merges_file":  "Llama3Merges.txt",
///   "image_size": 512,
///   "latent_channels": 32,
///   "latent_size": 16,
///   "steps": 4,
///   "guidance_scale": 0.0,
///   "max_sequence_length": 128
/// }
/// ```
struct NitroETextToImageDemoView: View {
    let model: ModelEntry

    @State private var prompt = "a hot air balloon in the shape of a heart, grand canyon"
    @State private var resultImage: CGImage?
    @State private var seed: UInt32 = 42
    @State private var isGenerating = false
    @State private var status = ""
    @State private var generationTime: Double?
    @StateObject private var session = ModelSession<NitroEAssets>()

    private struct NitroEAssets {
        let tokenizer: LlamaBPETokenizer
        let textEncoder: MLModel
        let denoiser: MLModel
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

            TextField("Describe an image…", text: $prompt, axis: .vertical)
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
        .contentShape(Rectangle())
        .onTapGesture {
            UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder),
                                            to: nil, from: nil, for: nil)
        }
        .task {
            session.ensure {
                let vocabFile = model.configString("vocab_file") ?? "Llama3Vocab.json"
                let mergesFile = model.configString("merges_file") ?? "Llama3Merges.txt"
                let tokenizer = try LlamaBPETokenizer(
                    vocabURL: ModelLoader.auxFileURL(modelId: model.id, fileName: vocabFile),
                    mergesURL: ModelLoader.auxFileURL(modelId: model.id, fileName: mergesFile)
                )
                let teFile = model.configString("text_encoder")
                    ?? model.files.first { $0.name.lowercased().contains("textencoder") }?.name
                    ?? model.files[0].name
                let dnFile = model.configString("denoiser")
                    ?? model.files.first { $0.name.lowercased().contains("emmdit") }?.name
                    ?? model.files[1].name
                let vaeFile = model.configString("vae_decoder")
                    ?? model.files.first { $0.name.lowercased().contains("decoder") }?.name
                    ?? model.files.last!.name
                let te = try await ModelLoader.load(for: model, named: teFile)
                let dn = try await ModelLoader.load(for: model, named: dnFile)
                let vae = try await ModelLoader.load(for: model, named: vaeFile)
                return NitroEAssets(tokenizer: tokenizer, textEncoder: te, denoiser: dn, vaeDecoder: vae)
            }
        }
    }

    // MARK: - Generation

    private func generate() async {
        isGenerating = true; resultImage = nil; generationTime = nil
        let currentPrompt = prompt
        let currentSeed = seed

        let imageSize = model.configInt("image_size") ?? 512
        let latentSize = model.configInt("latent_size") ?? 16
        let latentChannels = model.configInt("latent_channels") ?? 32
        let steps = max(1, model.configInt("steps") ?? 4)
        let seqLen = model.configInt("max_sequence_length") ?? 128

        do {
            status = session.loadTimeSec == nil ? "Loading models…" : "Preparing…"
            let assets = try await session.get()

            let start = CFAbsoluteTimeGetCurrent()

            // 1) Tokenize + text encode
            status = "Encoding text…"
            let (ids, mask) = try assets.tokenizer.encode(text: currentPrompt.lowercased(), maxLength: seqLen)
            let idsArr = try makeInt32Array(shape: [1, seqLen], from: ids)
            let maskArr = try makeInt32Array(shape: [1, seqLen], from: mask)
            let teOut = try await assets.textEncoder.prediction(from: MLDictionaryFeatureProvider(
                dictionary: ["input_ids": idsArr, "attention_mask": maskArr]
            ))
            guard let textHidden = teOut.featureValue(for: "last_hidden_state")?.multiArrayValue else {
                throw NSError(domain: "NitroE", code: 1, userInfo: [NSLocalizedDescriptionKey: "last_hidden_state missing"])
            }

            // 2) Initial Gaussian latent, seeded
            let latentCount = latentChannels * latentSize * latentSize
            var latent = gaussianNoise(count: latentCount, seed: UInt64(currentSeed))

            // 3) Denoise loop (FlowMatchEuler)
            let scheduler = NitroEFlowMatchScheduler()
            scheduler.setTimesteps(steps)
            for i in 0..<steps {
                status = "Denoising \(i + 1)/\(steps)…"
                let ts = Int32(scheduler.timesteps[i].rounded())
                let tsArr = try MLMultiArray(shape: [1], dataType: .int32)
                tsArr[0] = NSNumber(value: ts)
                let latentArr = try makeFloatArray(shape: [1, latentChannels, latentSize, latentSize], from: latent)
                let out = try await assets.denoiser.prediction(from: MLDictionaryFeatureProvider(dictionary: [
                    "latent": latentArr,
                    "encoder_hidden_states": textHidden,
                    "encoder_attention_mask": maskArr,
                    "timestep": tsArr,
                ]))
                guard let pred = out.featureValue(for: "noise_pred")?.multiArrayValue else {
                    throw NSError(domain: "NitroE", code: 2, userInfo: [NSLocalizedDescriptionKey: "noise_pred missing"])
                }
                latent = scheduler.step(modelOutput: toFloats(pred), sample: latent)
            }

            // 4) VAE decode
            status = "Decoding…"
            let decodeIn = try makeFloatArray(shape: [1, latentChannels, latentSize, latentSize], from: latent)
            let vaeOut = try await assets.vaeDecoder.prediction(from: MLDictionaryFeatureProvider(
                dictionary: ["latent": decodeIn]
            ))
            guard let img = vaeOut.featureValue(for: "image")?.multiArrayValue else {
                throw NSError(domain: "NitroE", code: 3, userInfo: [NSLocalizedDescriptionKey: "image missing"])
            }
            let cg = try cgImage(fromCHW: img, width: imageSize, height: imageSize)
            let elapsed = CFAbsoluteTimeGetCurrent() - start

            await MainActor.run {
                resultImage = cg
                generationTime = elapsed
                isGenerating = false; status = ""
            }
        } catch {
            await MainActor.run { isGenerating = false; status = "Error: \(error.localizedDescription)" }
        }
    }

    // MARK: - Helpers

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
        let ptr = m.dataPointer.assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: ptr, count: m.count))
    }

    private func gaussianNoise(count: Int, seed: UInt64) -> [Float] {
        var rng = NitroESplitMix64(seed: seed == 0 ? 0x9E3779B97F4A7C15 : seed)
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
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        let strides = arr.strides.map { $0.intValue }
        let base = arr.dataPointer.assumingMemoryBound(to: Float.self)
        for y in 0..<height {
            for x in 0..<width {
                for ch in 0..<3 {
                    let idx = ch * strides[1] + y * strides[2] + x * strides[3]
                    let v = max(-1, min(1, base[idx]))
                    rgba[(y * width + x) * 4 + ch] = UInt8(((v + 1) * 0.5 * 255).rounded())
                }
            }
        }
        guard let ctx = CGContext(
            data: &rgba, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ), let img = ctx.makeImage() else {
            throw NSError(domain: "NitroE", code: 4, userInfo: [NSLocalizedDescriptionKey: "CGContext failed"])
        }
        return img
    }
}

// MARK: - FlowMatchEuler scheduler (Nitro-E)

final class NitroEFlowMatchScheduler {
    private(set) var sigmas: [Float] = []
    private(set) var timesteps: [Float] = []
    private let numTrainTimesteps: Int
    private let shift: Float
    private var stepIndex: Int = 0

    init(numTrainTimesteps: Int = 1000, shift: Float = 1.0) {
        self.numTrainTimesteps = numTrainTimesteps
        self.shift = shift
    }

    func setTimesteps(_ numInferenceSteps: Int) {
        let n = Float(numInferenceSteps)
        var sig = [Float](repeating: 0, count: numInferenceSteps)
        for i in 0..<numInferenceSteps {
            let t = Float(i) / max(n - 1, 1)
            let s = 1.0 - t * (1.0 - 1.0 / Float(numTrainTimesteps))
            sig[i] = shift * s / (1.0 + (shift - 1.0) * s)
        }
        timesteps = sig.map { $0 * Float(numTrainTimesteps) }
        sigmas = sig + [0.0]
        stepIndex = 0
    }

    func step(modelOutput: [Float], sample: [Float]) -> [Float] {
        precondition(stepIndex < timesteps.count)
        let dt = sigmas[stepIndex + 1] - sigmas[stepIndex]
        var out = [Float](repeating: 0, count: sample.count)
        for i in 0..<sample.count { out[i] = sample[i] + dt * modelOutput[i] }
        stepIndex += 1
        return out
    }
}

// MARK: - Llama 3 byte-level BPE tokenizer

final class LlamaBPETokenizer {
    enum Error: Swift.Error { case malformed(String) }

    private let vocab: [String: Int32]
    private let bpeRanks: [Pair: Int]
    private let byteEncoder: [UInt8: Character]
    private let bosTokenID: Int32
    private let eosTokenID: Int32
    private let padTokenID: Int32
    private let pattern: NSRegularExpression

    struct Pair: Hashable { let a: String; let b: String }

    init(vocabURL: URL, mergesURL: URL) throws {
        let vocabData = try Data(contentsOf: vocabURL)
        guard let dict = try JSONSerialization.jsonObject(with: vocabData) as? [String: Int] else {
            throw Error.malformed("Llama3Vocab.json")
        }
        self.vocab = dict.mapValues { Int32($0) }

        let mergesText = try String(contentsOf: mergesURL, encoding: .utf8)
        var ranks: [Pair: Int] = [:]
        for (rank, line) in mergesText.split(separator: "\n").enumerated() {
            let parts = line.split(separator: " ", maxSplits: 1).map(String.init)
            if parts.count == 2 { ranks[Pair(a: parts[0], b: parts[1])] = rank }
        }
        self.bpeRanks = ranks
        self.byteEncoder = Self.makeByteEncoder()

        self.bosTokenID = dict["<|begin_of_text|>"].map(Int32.init) ?? 128000
        self.eosTokenID = dict["<|end_of_text|>"].map(Int32.init) ?? 128001
        self.padTokenID = self.eosTokenID

        let patt = #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
        self.pattern = try NSRegularExpression(pattern: patt)
    }

    func encode(text: String, maxLength: Int = 128) throws -> (ids: [Int32], mask: [Int32]) {
        var ids: [Int32] = [bosTokenID]
        let ns = text as NSString
        let matches = pattern.matches(in: text, range: NSRange(location: 0, length: ns.length))
        for m in matches {
            let piece = ns.substring(with: m.range)
            var s = ""
            for b in Array(piece.utf8) { s.append(byteEncoder[b] ?? Character(Unicode.Scalar(b))) }
            for tok in bpe(of: s) {
                if let id = vocab[tok] { ids.append(id) }
            }
            if ids.count >= maxLength { break }
        }
        if ids.count >= maxLength {
            ids = Array(ids.prefix(maxLength))
            ids[maxLength - 1] = eosTokenID
        } else {
            ids.append(contentsOf: Array(repeating: padTokenID, count: maxLength - ids.count))
        }
        var mask = [Int32](repeating: 0, count: maxLength)
        for i in 0..<maxLength where ids[i] != padTokenID || i == 0 { mask[i] = 1 }
        return (ids, mask)
    }

    private static func makeByteEncoder() -> [UInt8: Character] {
        var bs: [UInt8] = []
        for b in 33...126 { bs.append(UInt8(b)) }
        for b in 161...172 { bs.append(UInt8(b)) }
        for b in 174...255 { bs.append(UInt8(b)) }
        var cs = bs.map { Int($0) }
        var n = 0
        for b in 0..<256 {
            if !bs.contains(UInt8(b)) {
                bs.append(UInt8(b))
                cs.append(256 + n)
                n += 1
            }
        }
        var map: [UInt8: Character] = [:]
        for i in 0..<bs.count {
            if let scalar = Unicode.Scalar(cs[i]) { map[bs[i]] = Character(scalar) }
        }
        return map
    }

    private func bpe(of word: String) -> [String] {
        if word.isEmpty { return [] }
        var tokens = word.map { String($0) }
        while tokens.count > 1 {
            var bestRank = Int.max
            var bestIdx = -1
            for i in 0..<(tokens.count - 1) {
                let pair = Pair(a: tokens[i], b: tokens[i + 1])
                if let r = bpeRanks[pair], r < bestRank { bestRank = r; bestIdx = i }
            }
            if bestIdx < 0 { break }
            let merged = tokens[bestIdx] + tokens[bestIdx + 1]
            tokens.replaceSubrange(bestIdx...(bestIdx + 1), with: [merged])
        }
        return tokens
    }
}

// MARK: - SplitMix64 RNG (seeded)

private struct NitroESplitMix64 {
    var state: UInt64
    init(seed: UInt64) { self.state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
    mutating func nextUniform() -> Float {
        Float(next() >> 40) / Float(1 << 24)
    }
}
