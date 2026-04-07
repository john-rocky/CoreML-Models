import CoreML
import AVFoundation

/// Kokoro TTS pipeline.
///
/// Architecture:
///   1. Predictor (flexible 1..256 phonemes) → duration, d_for_align, t_en
///   2. Expand features by repeat_interleave(pred_dur)
///   3. Pick smallest decoder bucket >= total_frames; pad with zeros
///   4. Decoder bucket → audio
///   5. Trim audio to actual_samples = total_frames * 600 (24kHz)
class KokoroTTS: ObservableObject {
    @Published var isReady = false
    @Published var status = "Loading models..."

    private var predictor: MLModel?
    private var decoders: [Int: MLModel] = [:]
    private var vocab: [String: Int32] = [:]
    private var voices: [String: [Float]] = [:]
    private var samples: [SampleEntry] = []
    private let englishG2P = EnglishG2P()
    private let japaneseG2P = JapaneseG2P()

    static let sampleRate: Double = 24000
    static let samplesPerFrame: Int = 600
    static let voicePhonemeStride: Int = 256
    static let buckets: [Int] = [128, 256, 512]
    static let maxPhonemes: Int = 256

    struct SampleEntry: Codable {
        let text: String
        let phonemes: String
        let input_ids: [Int32]
        let language: String?  // "en" or "ja"; defaults to "en" if missing
    }

    var availableVoices: [String] { voices.keys.sorted() }
    var availableSamples: [SampleEntry] { samples }

    init() {
        loadResources()
        loadModels()
    }

    // MARK: - Resources

    private func loadResources() {
        if let url = Bundle.main.url(forResource: "kokoro_vocab", withExtension: "json"),
           let data = try? Data(contentsOf: url),
           let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            for (k, v) in dict {
                if let n = v as? NSNumber { vocab[k] = n.int32Value }
            }
        }

        let voiceNames = [
            "af_heart", "af_bella", "am_michael", "bf_emma", "bm_george",
            "jf_alpha", "jf_gongitsune", "jm_kumo", "jf_nezumi", "jf_tebukuro",
        ]
        for name in voiceNames {
            guard let url = Bundle.main.url(forResource: "voice_\(name)", withExtension: "bin"),
                  let data = try? Data(contentsOf: url) else { continue }
            let count = data.count / MemoryLayout<Float>.stride
            var arr = [Float](repeating: 0, count: count)
            _ = arr.withUnsafeMutableBytes { data.copyBytes(to: $0) }
            voices[name] = arr
        }

        if let url = Bundle.main.url(forResource: "samples", withExtension: "json"),
           let data = try? Data(contentsOf: url),
           let parsed = try? JSONDecoder().decode([SampleEntry].self, from: data) {
            samples = parsed
        }
    }

    private func loadModels() {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }
            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndGPU

            DispatchQueue.main.async { self.status = "Loading Predictor..." }
            if let url = Bundle.main.url(forResource: "Kokoro_Predictor", withExtension: "mlmodelc"),
               let model = try? MLModel(contentsOf: url, configuration: config) {
                self.predictor = model
            }

            for bucket in Self.buckets {
                DispatchQueue.main.async { self.status = "Loading Decoder \(bucket)..." }
                if let url = Bundle.main.url(forResource: "Kokoro_Decoder_\(bucket)", withExtension: "mlmodelc"),
                   let model = try? MLModel(contentsOf: url, configuration: config) {
                    self.decoders[bucket] = model
                }
            }

            let ok = self.predictor != nil && self.decoders.count == Self.buckets.count
            DispatchQueue.main.async {
                self.status = ok ? "Ready" : "Model load failed"
                self.isReady = ok
            }
        }
    }

    // MARK: - G2P + tokenization

    /// Convert text → input_ids ready for the predictor.
    /// language: "en" (English G2P) — Japanese will be added in a future patch.
    func tokenize(text: String, language: String = "en") -> [Int32] {
        return tokenizePhonemes(previewPhonemes(text: text, language: language))
    }

    /// Run G2P only and return the phoneme string (for UI display).
    func previewPhonemes(text: String, language: String = "en") -> String {
        switch language {
        case "en": return englishG2P.phonemize(text)
        case "ja": return japaneseG2P.phonemize(text)
        default:   return text
        }
    }

    /// Convert a phoneme string to token IDs using kokoro_vocab.json.
    /// Adds BOS=0 and EOS=0 padding as Kokoro expects.
    func tokenizePhonemes(_ phonemes: String) -> [Int32] {
        var ids: [Int32] = [0]  // BOS
        for scalar in phonemes.unicodeScalars {
            let ch = String(scalar)
            if let id = vocab[ch] {
                ids.append(id)
            }
            // Skip phonemes not in vocab (silently)
        }
        ids.append(0)  // EOS
        // Truncate to max length, preserving BOS/EOS
        if ids.count > Self.maxPhonemes {
            ids = Array(ids.prefix(Self.maxPhonemes - 1)) + [0]
        }
        return ids
    }

    // MARK: - Inference

    /// Synthesize speech from raw text (English).
    func synthesize(text: String, voice: String, language: String = "en") throws -> [Float] {
        let ids = tokenize(text: text, language: language)
        return try synthesize(inputIDs: ids, voice: voice)
    }

    /// Synthesize speech from phoneme token IDs (already including BOS/EOS = 0).
    func synthesize(inputIDs: [Int32], voice: String) throws -> [Float] {
        guard let predictor, !inputIDs.isEmpty, let voiceVec = voices[voice] else {
            return []
        }
        let T = inputIDs.count
        guard T <= Self.maxPhonemes else {
            throw NSError(domain: "KokoroTTS", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Too many phonemes (\(T) > \(Self.maxPhonemes))"
            ])
        }

        // Voice ref_s: voices [510, 1, 256], pick row [T-1] (clamped)
        let row = max(0, min(T - 1, 509))
        let stride = Self.voicePhonemeStride
        let refS = Array(voiceVec[row * stride ..< row * stride + stride])
        let refSStyle = Array(refS[128..<256])

        // Predictor inputs
        let inputIdsArr = try MLMultiArray(shape: [1, NSNumber(value: T)], dataType: .int32)
        for i in 0..<T { inputIdsArr[i] = NSNumber(value: inputIDs[i]) }

        let refStyleArr = try MLMultiArray(shape: [1, 128], dataType: .float32)
        for i in 0..<128 { refStyleArr[i] = NSNumber(value: refSStyle[i]) }

        let predOut = try predictor.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "input_ids": inputIdsArr,
            "ref_s_style": refStyleArr,
        ]))
        guard let durationArr = predOut.featureValue(for: "duration")?.multiArrayValue,
              let dForAlignArr = predOut.featureValue(for: "d_for_align")?.multiArrayValue,
              let tEnArr = predOut.featureValue(for: "t_en")?.multiArrayValue else {
            return []
        }

        // Convert duration → integer frames
        var predDur = [Int](repeating: 0, count: T)
        var totalFrames = 0
        let durPtr = durationArr.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<T {
            let v = max(1, Int(durPtr[i].rounded()))
            predDur[i] = v
            totalFrames += v
        }

        // Pick smallest bucket that fits
        let bucket = Self.buckets.first { $0 >= totalFrames } ?? Self.buckets.last!
        guard let decoder = decoders[bucket] else { return [] }

        let dHidden = Int(truncating: dForAlignArr.shape[1])  // 640
        let tHidden = Int(truncating: tEnArr.shape[1])         // 512

        let enArr = try MLMultiArray(shape: [1, NSNumber(value: dHidden), NSNumber(value: bucket)], dataType: .float32)
        let asrArr = try MLMultiArray(shape: [1, NSNumber(value: tHidden), NSNumber(value: bucket)], dataType: .float32)
        memset(enArr.dataPointer, 0, enArr.count * MemoryLayout<Float>.size)
        memset(asrArr.dataPointer, 0, asrArr.count * MemoryLayout<Float>.size)

        let dPtr = dForAlignArr.dataPointer.assumingMemoryBound(to: Float.self)
        let tPtr = tEnArr.dataPointer.assumingMemoryBound(to: Float.self)
        let enPtr = enArr.dataPointer.assumingMemoryBound(to: Float.self)
        let asrPtr = asrArr.dataPointer.assumingMemoryBound(to: Float.self)

        // Repeat-interleave: for each phoneme i, copy column to predDur[i] output positions.
        // Layout for [1, C, L]: idx = c*L + col
        var outIdx = 0
        for i in 0..<T {
            let rep = predDur[i]
            for _ in 0..<rep {
                if outIdx >= bucket { break }
                for c in 0..<dHidden {
                    enPtr[c * bucket + outIdx] = dPtr[c * T + i]
                }
                for c in 0..<tHidden {
                    asrPtr[c * bucket + outIdx] = tPtr[c * T + i]
                }
                outIdx += 1
            }
            if outIdx >= bucket { break }
        }

        let refSArr = try MLMultiArray(shape: [1, 256], dataType: .float32)
        for i in 0..<256 { refSArr[i] = NSNumber(value: refS[i]) }

        let decOut = try decoder.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "en_aligned": enArr,
            "asr_aligned": asrArr,
            "ref_s": refSArr,
        ]))
        guard let audioArr = decOut.featureValue(for: "audio")?.multiArrayValue else { return [] }

        let actualSamples = totalFrames * Self.samplesPerFrame
        let totalAudio = audioArr.count
        let trimLen = min(actualSamples, totalAudio)
        var audio = [Float](repeating: 0, count: trimLen)
        let aPtr = audioArr.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<trimLen { audio[i] = aPtr[i] }
        return audio
    }
}

// MARK: - WAV writing for playback

extension KokoroTTS {
    static func writeWav(samples: [Float], to url: URL) throws {
        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                   sampleRate: sampleRate,
                                   channels: 1,
                                   interleaved: false)!
        let file = try AVAudioFile(forWriting: url, settings: format.settings,
                                   commonFormat: .pcmFormatFloat32, interleaved: false)
        let frameCount = AVAudioFrameCount(samples.count)
        guard let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else { return }
        buf.frameLength = frameCount
        if let chan = buf.floatChannelData {
            samples.withUnsafeBufferPointer { src in
                chan[0].update(from: src.baseAddress!, count: samples.count)
            }
        }
        try file.write(from: buf)
    }
}
