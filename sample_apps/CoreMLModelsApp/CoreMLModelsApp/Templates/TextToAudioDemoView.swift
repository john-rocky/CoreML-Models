import SwiftUI
import CoreML
import AVFoundation
import Accelerate

/// Text-to-audio: TTS (Kokoro) or music generation (Stable Audio).
/// Dispatch via config `mode`: "tts" or "music".
struct TextToAudioDemoView: View {
    let model: ModelEntry

    @State private var inputText = ""
    @State private var selectedVoice: String = ""
    @State private var duration: Double = 5.0
    @State private var outputURL: URL?
    @State private var isProcessing = false
    @State private var status = ""
    @State private var processingTime: Double?
    @State private var player: AVAudioPlayer?
    @State private var isPlaying = false

    private var isTTS: Bool { model.configString("mode") == "tts" }
    private var voices: [String] { model.configStringArray("voices") ?? [] }
    private var maxDuration: Double { model.configDouble("max_duration") ?? 30.0 }

    var body: some View {
        VStack(spacing: 0) {
            if outputURL != nil {
                VStack(spacing: 16) {
                    Image(systemName: "waveform").font(.system(size: 60)).foregroundStyle(.tint)
                    HStack(spacing: 24) {
                        Button { togglePlayback() } label: {
                            Image(systemName: isPlaying ? "pause.circle.fill" : "play.circle.fill").font(.system(size: 44))
                        }
                        if let url = outputURL {
                            ShareLink(item: url) { Image(systemName: "square.and.arrow.up").font(.title2) }
                        }
                    }
                    if let t = processingTime {
                        Text(String(format: "Generated in %.1fs", t)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    }
                }.frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                VStack(spacing: 12) {
                    Image(systemName: isTTS ? "mouth.fill" : "music.note").font(.system(size: 60)).foregroundStyle(.secondary)
                    Text(isTTS ? "Enter text to synthesize speech" : "Enter a prompt to generate audio").foregroundStyle(.secondary)
                }.frame(maxWidth: .infinity, maxHeight: .infinity)
            }

            VStack(spacing: 12) {
                if isProcessing { ProgressView(status) }
                TextField(isTTS ? "Text to speak…" : "Describe the audio…", text: $inputText, axis: .vertical)
                    .textFieldStyle(.roundedBorder).lineLimit(2...4)

                if isTTS && !voices.isEmpty {
                    Picker("Voice", selection: $selectedVoice) {
                        ForEach(voices, id: \.self) { v in
                            Text(v.replacingOccurrences(of: "_", with: " ").capitalized).tag(v)
                        }
                    }.pickerStyle(.menu)
                }
                if !isTTS {
                    HStack {
                        Text("Duration").font(.caption2).foregroundStyle(.secondary)
                        Slider(value: $duration, in: 1...maxDuration)
                        Text(String(format: "%.0fs", duration)).font(.caption2.monospacedDigit()).foregroundStyle(.secondary).frame(width: 30)
                    }
                }
                Button {
                    Task { await generate() }
                } label: {
                    Label(isTTS ? "Speak" : "Generate", systemImage: isTTS ? "speaker.wave.3" : "wand.and.rays").frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(isProcessing || inputText.trimmingCharacters(in: .whitespaces).isEmpty)
            }.padding()
        }
        .onAppear { if selectedVoice.isEmpty { selectedVoice = voices.first ?? "" }; if inputText.isEmpty { inputText = isTTS ? "Hello, this is a test." : "Upbeat electronic dance music" } }
    }

    private func togglePlayback() {
        if isPlaying { player?.stop(); isPlaying = false; return }
        guard let url = outputURL else { return }
        player = try? AVAudioPlayer(contentsOf: url); player?.play(); isPlaying = true
    }

    private func generate() async {
        isProcessing = true; outputURL = nil; isPlaying = false
        do {
            let start = CFAbsoluteTimeGetCurrent()
            let url: URL
            if isTTS { url = try await generateKokoro() }
            else { url = try await generateStableAudio() }
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            await MainActor.run { outputURL = url; processingTime = elapsed; isProcessing = false; status = "" }
        } catch {
            await MainActor.run { isProcessing = false; status = "Error: \(error.localizedDescription)" }
        }
    }

    // MARK: - Kokoro TTS Pipeline

    private func generateKokoro() async throws -> URL {
        // Load vocab
        status = "Loading vocab…"
        let vocabFile = model.configString("vocab_file") ?? model.files.first { ($0.kind ?? "") == "vocab" }?.name ?? "kokoro_vocab.json"
        let vocabURL = ModelLoader.auxFileURL(modelId: model.id, fileName: vocabFile)
        var vocab: [String: Int32] = [:]
        if let data = try? Data(contentsOf: vocabURL),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            for (k, v) in json {
                if let id = v as? Int { vocab[k] = Int32(id) }
                else if let id = v as? Int32 { vocab[k] = id }
            }
        }

        // Simple G2P: convert text to phonemes (simplified)
        let phonemes = simpleG2P(inputText)

        // Tokenize phonemes
        var tokenIds: [Int32] = [0]  // BOS
        for ch in phonemes.unicodeScalars {
            if let id = vocab[String(ch)] { tokenIds.append(id) }
        }
        tokenIds.append(0)  // EOS
        if tokenIds.count > 256 { tokenIds = Array(tokenIds.prefix(255)) + [0] }
        let T = tokenIds.count

        // Load voice embedding
        var refS = [Float](repeating: 0, count: 256)
        if !selectedVoice.isEmpty {
            let voiceFile = model.files.first { $0.name.contains(selectedVoice) }?.name
                ?? "voice_\(selectedVoice).bin"
            let voiceURL = ModelLoader.auxFileURL(modelId: model.id, fileName: voiceFile)
            if let data = try? Data(contentsOf: voiceURL) {
                let floats = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
                let row = max(0, min(T - 1, 509))
                if floats.count >= (row + 1) * 256 {
                    refS = Array(floats[row * 256 ..< row * 256 + 256])
                }
            }
        }
        let refSStyle = Array(refS[128..<256])  // Last 128 dims

        // Predictor
        status = "Predicting duration…"
        let predictorFile = model.files.first { $0.name.lowercased().contains("predict") }?.name ?? model.files[0].name
        let predictor = try await ModelLoader.load(for: model, named: predictorFile)

        let inputIds = try MLMultiArray(shape: [1, NSNumber(value: T)], dataType: .int32)
        let idPtr = inputIds.dataPointer.assumingMemoryBound(to: Int32.self)
        for (i, tok) in tokenIds.enumerated() { idPtr[i] = tok }

        let refStyleArr = try MLMultiArray(shape: [1, 128], dataType: .float32)
        let rsPtr = refStyleArr.dataPointer.assumingMemoryBound(to: Float.self)
        for (i, v) in refSStyle.enumerated() { rsPtr[i] = v }

        let predInput = try MLDictionaryFeatureProvider(dictionary: ["input_ids": inputIds, "ref_s_style": refStyleArr])
        let predOutput = try await predictor.prediction(from: predInput)

        guard let durArr = predOutput.featureValue(for: "duration")?.multiArrayValue,
              let dForAlign = predOutput.featureValue(for: "d_for_align")?.multiArrayValue,
              let tEn = predOutput.featureValue(for: "t_en")?.multiArrayValue else {
            throw NSError(domain: "Kokoro", code: 1, userInfo: [NSLocalizedDescriptionKey: "Predictor output missing"])
        }

        // Duration → integer frames
        var predDur = [Int](repeating: 0, count: T)
        var totalFrames = 0
        for i in 0..<T {
            predDur[i] = max(1, Int(durArr[i].floatValue.rounded()))
            totalFrames += predDur[i]
        }

        // Bucket selection
        let buckets = [128, 256, 512]
        let bucket = buckets.first { $0 >= totalFrames } ?? buckets.last!

        // Repeat-interleave expansion
        status = "Expanding features…"
        let dHidden = 640, tHidden = 512
        let enAligned = try MLMultiArray(shape: [1, NSNumber(value: dHidden), NSNumber(value: bucket)], dataType: .float32)
        let asrAligned = try MLMultiArray(shape: [1, NSNumber(value: tHidden), NSNumber(value: bucket)], dataType: .float32)
        let enPtr = enAligned.dataPointer.assumingMemoryBound(to: Float.self)
        let asrPtr = asrAligned.dataPointer.assumingMemoryBound(to: Float.self)
        memset(enPtr, 0, dHidden * bucket * MemoryLayout<Float>.size)
        memset(asrPtr, 0, tHidden * bucket * MemoryLayout<Float>.size)

        let dPtr = dForAlign.dataPointer.assumingMemoryBound(to: Float.self)
        let tPtr = tEn.dataPointer.assumingMemoryBound(to: Float.self)
        let dStrides = dForAlign.strides.map { $0.intValue }
        let tStrides = tEn.strides.map { $0.intValue }

        var outIdx = 0
        for i in 0..<T {
            let rep = predDur[i]
            for _ in 0..<rep {
                guard outIdx < bucket else { break }
                for c in 0..<dHidden {
                    enPtr[c * bucket + outIdx] = ImageUtils.readFloat(dForAlign, at: c * dStrides[1] + i * dStrides[2])
                }
                for c in 0..<tHidden {
                    asrPtr[c * bucket + outIdx] = ImageUtils.readFloat(tEn, at: c * tStrides[1] + i * tStrides[2])
                }
                outIdx += 1
            }
        }

        // Decoder
        status = "Generating audio…"
        let decoderFile = model.files.first { $0.name.lowercased().contains("decoder") && $0.name.contains("\(bucket)") }?.name
            ?? model.files.first { $0.name.lowercased().contains("decoder") }?.name ?? model.files.last!.name
        let decoder = try await ModelLoader.load(for: model, named: decoderFile)

        let refSArr = try MLMultiArray(shape: [1, 256], dataType: .float32)
        let refSPtr = refSArr.dataPointer.assumingMemoryBound(to: Float.self)
        for (i, v) in refS.enumerated() { refSPtr[i] = v }

        let decInput = try MLDictionaryFeatureProvider(dictionary: [
            "en_aligned": enAligned, "asr_aligned": asrAligned, "ref_s": refSArr
        ])
        let decOutput = try await decoder.prediction(from: decInput)
        guard let audioArr = decOutput.featureNames.compactMap({ decOutput.featureValue(for: $0)?.multiArrayValue }).first else {
            throw NSError(domain: "Kokoro", code: 2, userInfo: [NSLocalizedDescriptionKey: "No audio output"])
        }

        // Trim and save
        let actualSamples = totalFrames * 600
        let trimLen = min(actualSamples, audioArr.count)
        var audio = [Float](repeating: 0, count: trimLen)
        let aPtr = audioArr.dataPointer
        if audioArr.dataType == .float16 {
            let fp16 = aPtr.assumingMemoryBound(to: Float16.self)
            for i in 0..<trimLen { audio[i] = Float(fp16[i]) }
        } else {
            let fp32 = aPtr.assumingMemoryBound(to: Float.self)
            for i in 0..<trimLen { audio[i] = fp32[i] }
        }

        return try writeWAV(samples: audio, sampleRate: 24000, channels: 1)
    }

    // MARK: - Stable Audio Pipeline

    private func generateStableAudio() async throws -> URL {
        // T5 tokenize
        status = "Tokenizing…"
        let vocabFile = model.files.first { ($0.kind ?? "") == "vocab" }?.name ?? "t5_vocab.json"
        let vocabURL = ModelLoader.auxFileURL(modelId: model.id, fileName: vocabFile)
        var pieceToID: [String: Int32] = [:]
        if let data = try? Data(contentsOf: vocabURL),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            for (k, v) in json {
                if let piece = v as? String, let id = Int32(k) { pieceToID[piece] = id }
                else if let id = v as? Int { pieceToID[k] = Int32(id) }
            }
        }

        let tokens = t5Tokenize(inputText, vocab: pieceToID, maxLen: 64)

        // Load T5 Encoder
        status = "Encoding text…"
        let t5File = model.files.first { $0.name.contains("T5") }?.name ?? model.files[0].name
        let t5 = try await ModelLoader.load(for: model, named: t5File)

        let inputIds = try MLMultiArray(shape: [1, 64], dataType: .int32)
        let idPtr = inputIds.dataPointer.assumingMemoryBound(to: Int32.self)
        for (i, tok) in tokens.enumerated() { idPtr[i] = tok }

        let t5Out = try await t5.prediction(from: MLDictionaryFeatureProvider(dictionary: ["input_ids": inputIds]))
        guard let textEmb = t5Out.featureValue(for: "text_embeddings")?.multiArrayValue else {
            throw NSError(domain: "StableAudio", code: 1, userInfo: [NSLocalizedDescriptionKey: "T5 output missing"])
        }

        // Sanitize NaN from INT8 quantization + zero out padding
        let attentionLen = tokens.firstIndex(of: 0) ?? 64
        for t in 0..<64 {
            for c in 0..<768 {
                let v = textEmb[[0, t, c] as [NSNumber]].floatValue
                if t >= attentionLen || v.isNaN || v.isInfinite {
                    textEmb[[0, t, c] as [NSNumber]] = 0
                }
            }
        }

        // Number Embedder
        status = "Embedding duration…"
        let neFile = model.files.first { $0.name.contains("NumberEmbedder") }?.name
        guard let neFile else { throw NSError(domain: "StableAudio", code: 2) }
        let ne = try await ModelLoader.load(for: model, named: neFile)

        let normSec = try MLMultiArray(shape: [1], dataType: .float16)
        normSec.dataPointer.assumingMemoryBound(to: Float16.self)[0] = Float16(min(max(Float(duration), 0), 256) / 256.0)

        let neOut = try await ne.prediction(from: MLDictionaryFeatureProvider(dictionary: ["normalized_seconds": normSec]))
        guard let secEmb = neOut.featureValue(for: "seconds_embedding")?.multiArrayValue else {
            throw NSError(domain: "StableAudio", code: 3)
        }

        // Build cross_attn_cond [1,65,768] and global_embed [1,768]
        let crossAttn = try MLMultiArray(shape: [1, 65, 768], dataType: .float16)
        let globalEmb = try MLMultiArray(shape: [1, 768], dataType: .float16)
        let caPtr = crossAttn.dataPointer.assumingMemoryBound(to: Float16.self)
        let gePtr = globalEmb.dataPointer.assumingMemoryBound(to: Float16.self)

        // Copy text embeddings [0:64]
        for t in 0..<64 {
            for c in 0..<768 {
                caPtr[t * 768 + c] = Float16(textEmb[[0, t, c] as [NSNumber]].floatValue)
            }
        }
        // Append seconds embedding at position 64
        for c in 0..<768 {
            let v = ImageUtils.readFloat(secEmb, at: c)
            caPtr[64 * 768 + c] = Float16(v)
            gePtr[c] = Float16(v)
        }

        // Create noise [1,64,256]
        status = "Generating…"
        let latent = try MLMultiArray(shape: [1, 64, 256], dataType: .float16)
        let lPtr = latent.dataPointer.assumingMemoryBound(to: Float16.self)
        let count = 64 * 256
        var rngState: UInt64 = UInt64(arc4random())
        for i in stride(from: 0, to: count, by: 2) {
            rngState = rngState &* 6364136223846793005 &+ 1442695040888963407
            let u1 = max(Float.ulpOfOne, Float(rngState >> 33) / Float(1 << 31))
            rngState = rngState &* 6364136223846793005 &+ 1442695040888963407
            let u2 = Float(rngState >> 33) / Float(1 << 31)
            let r = sqrt(-2 * log(u1))
            lPtr[i] = Float16(r * cos(2 * .pi * u2))
            if i + 1 < count { lPtr[i + 1] = Float16(r * sin(2 * .pi * u2)) }
        }

        // Load DiT
        let ditFile = model.files.first { $0.name.contains("DiT") && !$0.name.contains("FP32") }?.name ?? model.files[2].name
        let dit = try await ModelLoader.load(for: model, named: ditFile)

        // Diffusion loop
        let steps = 100
        let schedule = makeSchedule(steps: steps)

        for i in 0..<steps {
            if i % 10 == 0 { status = "Diffusion step \(i)/\(steps)…" }
            let tCurr = schedule[i], tNext = schedule[i + 1]
            let dt = tNext - tCurr

            let ts = try MLMultiArray(shape: [1], dataType: .float16)
            ts.dataPointer.assumingMemoryBound(to: Float16.self)[0] = Float16(tCurr)

            let ditInput = try MLDictionaryFeatureProvider(dictionary: [
                "latent": latent, "timestep": ts, "cross_attn_cond": crossAttn, "global_embed": globalEmb
            ])
            let ditOut = try await dit.prediction(from: ditInput)
            guard let velocity = ditOut.featureValue(for: "velocity")?.multiArrayValue else { continue }

            // Euler step: latent = latent + dt * velocity
            let vPtr = velocity.dataPointer
            if velocity.dataType == .float16 {
                let vfp16 = vPtr.assumingMemoryBound(to: Float16.self)
                for j in 0..<count { lPtr[j] = Float16(Float(lPtr[j]) + dt * Float(vfp16[j])) }
            } else {
                let vfp32 = vPtr.assumingMemoryBound(to: Float.self)
                for j in 0..<count { lPtr[j] = Float16(Float(lPtr[j]) + dt * vfp32[j]) }
            }
        }

        // VAE Decode
        status = "Decoding audio…"
        let vaeFile = model.files.first { $0.name.contains("VAEDecoder") }?.name ?? model.files.last!.name
        let vae = try await ModelLoader.load(for: model, named: vaeFile)
        let vaeOut = try await vae.prediction(from: MLDictionaryFeatureProvider(dictionary: ["latent": latent]))
        guard let audioArr = vaeOut.featureValue(for: "audio")?.multiArrayValue else {
            throw NSError(domain: "StableAudio", code: 4)
        }

        // Extract stereo audio, trim to duration
        let trimmed = min(Int(duration * 44100), 524288)
        var ch0 = [Float](repeating: 0, count: trimmed)
        var ch1 = [Float](repeating: 0, count: trimmed)
        for i in 0..<trimmed {
            ch0[i] = audioArr[[0, 0, i] as [NSNumber]].floatValue
            ch1[i] = audioArr[[0, 1, i] as [NSNumber]].floatValue
        }

        // Interleave for WAV
        var stereo = [Float](repeating: 0, count: trimmed * 2)
        for i in 0..<trimmed { stereo[i] = ch0[i]; stereo[trimmed + i] = ch1[i] }
        return try writeWAV(samples: stereo, sampleRate: 44100, channels: 2)
    }

    private func makeSchedule(steps: Int) -> [Float] {
        var schedule = [Float](repeating: 0, count: steps + 1)
        for i in 0...steps {
            let logsnr: Float = -6.0 + Float(i) / Float(steps) * 8.0
            schedule[i] = 1.0 / (1.0 + exp(logsnr))
        }
        schedule[0] = 1.0; schedule[steps] = 0.0
        return schedule
    }

    // MARK: - T5 SentencePiece Tokenizer

    private func t5Tokenize(_ text: String, vocab: [String: Int32], maxLen: Int) -> [Int32] {
        let processed = "\u{2581}" + text.lowercased().replacingOccurrences(of: " ", with: "\u{2581}")
        var tokens: [Int32] = []
        var i = processed.startIndex
        while i < processed.endIndex {
            var bestLen = 0; var bestId: Int32 = 0
            for len in (1...min(20, processed.distance(from: i, to: processed.endIndex))).reversed() {
                let end = processed.index(i, offsetBy: len)
                if let id = vocab[String(processed[i..<end])] { bestLen = len; bestId = id; break }
            }
            if bestLen == 0 { i = processed.index(after: i) }
            else { tokens.append(bestId); i = processed.index(i, offsetBy: bestLen) }
        }
        tokens.append(1)  // EOS
        if tokens.count > maxLen { tokens = Array(tokens.prefix(maxLen - 1)) + [1] }
        var result = [Int32](repeating: 0, count: maxLen)
        for (idx, tok) in tokens.enumerated() { result[idx] = tok }
        return result
    }

    // MARK: - Simple G2P (simplified English phonemization)

    private func simpleG2P(_ text: String) -> String {
        // Simplified: pass through text as-is for phoneme vocab lookup
        // Full G2P would use lexicon lookup + letter-to-phoneme rules
        return text.lowercased()
    }

    // MARK: - WAV Writer

    private func writeWAV(samples: [Float], sampleRate: Int, channels: Int) throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("generated_\(UUID().uuidString).wav")
        let format = AVAudioFormat(standardFormatWithSampleRate: Double(sampleRate), channels: AVAudioChannelCount(channels))!
        let samplesPerChannel = samples.count / channels
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samplesPerChannel))!
        buffer.frameLength = AVAudioFrameCount(samplesPerChannel)
        for ch in 0..<channels {
            for i in 0..<samplesPerChannel { buffer.floatChannelData![ch][i] = samples[ch * samplesPerChannel + i] }
        }
        let file = try AVAudioFile(forWriting: url, settings: format.settings)
        try file.write(from: buffer)
        return url
    }
}
