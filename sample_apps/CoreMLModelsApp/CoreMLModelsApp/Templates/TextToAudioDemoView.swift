import SwiftUI
import CoreML
import AVFoundation
import Accelerate

/// Text-to-audio: TTS (Kokoro) or music generation (Stable Audio).
/// Dispatch via config `mode`: "tts" or "music".
struct TextToAudioDemoView: View {
    let model: ModelEntry

    // MARK: - Shared State

    @State private var isProcessing = false
    @State private var status = ""
    @State private var outputURL: URL?
    @State private var player: AVAudioPlayer?
    @State private var isPlaying = false
    @State private var errorMessage: String?

    // MARK: - TTS State

    enum TTSLanguage: String, CaseIterable, Identifiable {
        case english = "English"
        case japanese = "Japanese"
        var id: String { rawValue }
        var code: String { self == .english ? "en" : "ja" }
    }

    enum TTSInputMode: String, CaseIterable, Identifiable {
        case type = "Type"
        case sample = "Sample"
        var id: String { rawValue }
    }

    @State private var ttsLanguage: TTSLanguage = .english
    @State private var ttsInputMode: TTSInputMode = .type
    @State private var ttsVoice: String = ""
    @State private var ttsFreeTextEN = "Hello, this is a speech synthesis demo."
    @State private var ttsFreeTextJA = "今日は、これは音声合成のデモです。"
    @State private var ttsSelectedSampleIndex = 0
    @State private var ttsPhonemes: String = ""
    @State private var ttsInferenceMs: Double = 0
    @State private var ttsAudioDurationSec: Double = 0
    @FocusState private var ttsTextFocused: Bool

    // MARK: - Music State

    @State private var musicPrompt = "A gentle piano melody with soft strings"
    @State private var musicDuration: Double = 8.0
    @State private var musicSteps: Int = 25
    @State private var musicSeed: String = ""
    @State private var musicProgressStep: Int = 0
    @State private var musicProgressTotal: Int = 0
    @State private var musicProgressMessage: String = ""
    @FocusState private var musicTextFocused: Bool

    private let musicPresets = [
        "A gentle piano melody with soft strings",
        "Drum breaks 174 BPM",
        "Glitchy bass design",
        "Synth pluck arp with reverb and delay, 128 BPM",
        "Birds singing in the forest",
        "A short beautiful piano riff in C minor",
    ]

    // MARK: - Derived

    private var isTTS: Bool { model.configString("mode") == "tts" }
    private var voices: [String] { model.configStringArray("voices") ?? [] }
    private var maxDuration: Double { model.configDouble("max_duration") ?? 11.9 }

    private var ttsVoicesForLanguage: [String] {
        let prefixes: [String] = ttsLanguage == .english ? ["a", "b"] : ["j"]
        return voices.filter { v in prefixes.contains(where: { v.hasPrefix($0) }) }
    }

    private var ttsSamples: [(index: Int, text: String)] {
        // Build sample texts from config if available, otherwise use defaults
        let allSamples: [(text: String, lang: String)]
        if let configSamples = model.demo.config?["samples"]?.value as? [[String: Any]] {
            allSamples = configSamples.map { dict in
                (text: dict["text"] as? String ?? "", lang: dict["language"] as? String ?? "en")
            }
        } else {
            allSamples = [
                (text: "Hello, this is a test of text to speech.", lang: "en"),
                (text: "The quick brown fox jumps over the lazy dog.", lang: "en"),
                (text: "She sells seashells by the seashore.", lang: "en"),
                (text: "今日はとてもいい天気です。", lang: "ja"),
                (text: "吾輩は猫である。名前はまだ無い。", lang: "ja"),
            ]
        }
        let code = ttsLanguage.code
        return allSamples.enumerated()
            .filter { $0.element.lang == code }
            .map { (index: $0.offset, text: $0.element.text) }
    }

    private var ttsCurrentText: String {
        if ttsInputMode == .sample {
            let samples = ttsSamples
            guard ttsSelectedSampleIndex < samples.count else { return "" }
            return samples[ttsSelectedSampleIndex].text
        }
        return ttsLanguage == .english ? ttsFreeTextEN : ttsFreeTextJA
    }

    // MARK: - Body

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                if isTTS {
                    ttsBody
                } else {
                    musicBody
                }
            }
            .padding(.vertical)
        }
        .toolbar {
            ToolbarItemGroup(placement: .keyboard) {
                Spacer()
                Button("Done") {
                    ttsTextFocused = false
                    musicTextFocused = false
                }
            }
        }
        .onAppear {
            if ttsVoice.isEmpty, let first = ttsVoicesForLanguage.first {
                ttsVoice = first
            }
        }
    }

    // MARK: - TTS Body

    private var ttsBody: some View {
        VStack(spacing: 18) {
            // Language picker
            VStack(alignment: .leading, spacing: 8) {
                Text("Language")
                    .font(.headline)
                Picker("Language", selection: $ttsLanguage) {
                    ForEach(TTSLanguage.allCases) { lang in
                        Text(lang.rawValue).tag(lang)
                    }
                }
                .pickerStyle(.segmented)
                .onChange(of: ttsLanguage) { _, _ in
                    let filtered = ttsVoicesForLanguage
                    if !filtered.contains(ttsVoice), let first = filtered.first {
                        ttsVoice = first
                    }
                    ttsSelectedSampleIndex = 0
                }
            }
            .padding(.horizontal)

            // Voice picker
            if !ttsVoicesForLanguage.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Voice")
                        .font(.headline)
                    Picker("Voice", selection: $ttsVoice) {
                        ForEach(ttsVoicesForLanguage, id: \.self) { v in
                            Text(v).tag(v)
                        }
                    }
                    .pickerStyle(.segmented)
                }
                .padding(.horizontal)
            }

            // Input mode picker
            VStack(alignment: .leading, spacing: 8) {
                Picker("Input mode", selection: $ttsInputMode) {
                    ForEach(TTSInputMode.allCases) { m in
                        Text(m.rawValue).tag(m)
                    }
                }
                .pickerStyle(.segmented)

                if ttsInputMode == .type {
                    TextEditor(text: ttsLanguage == .english ? $ttsFreeTextEN : $ttsFreeTextJA)
                        .focused($ttsTextFocused)
                        .frame(minHeight: 100, maxHeight: 200)
                        .padding(8)
                        .background(Color(.systemGray6))
                        .cornerRadius(10)
                        .overlay(
                            RoundedRectangle(cornerRadius: 10)
                                .stroke(Color(uiColor: .separator), lineWidth: 0.5)
                        )
                } else {
                    let samples = ttsSamples
                    if !samples.isEmpty {
                        Picker("Sample", selection: $ttsSelectedSampleIndex) {
                            ForEach(0..<samples.count, id: \.self) { i in
                                Text(samples[i].text)
                                    .lineLimit(1)
                                    .tag(i)
                            }
                        }
                        .pickerStyle(.menu)

                        if ttsSelectedSampleIndex < samples.count {
                            Text(samples[ttsSelectedSampleIndex].text)
                                .font(.body)
                                .padding(12)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .background(Color(.systemGray6))
                                .cornerRadius(10)
                        }
                    }
                }
            }
            .padding(.horizontal)

            // Generate button
            Button(action: { Task { await generateTTS() } }) {
                HStack {
                    if isProcessing {
                        ProgressView()
                            .progressViewStyle(.circular)
                            .tint(.white)
                    } else {
                        Image(systemName: "waveform")
                    }
                    Text(isProcessing ? "Synthesizing..." : "Generate Speech")
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background(!isProcessing && !ttsCurrentText.trimmingCharacters(in: .whitespaces).isEmpty ? Color.accentColor : Color.gray)
                .foregroundColor(.white)
                .cornerRadius(12)
            }
            .disabled(isProcessing || ttsCurrentText.trimmingCharacters(in: .whitespaces).isEmpty)
            .padding(.horizontal)

            // Processing status
            if isProcessing {
                Text(status)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .padding(.horizontal)
            }

            // Stats row
            if ttsInferenceMs > 0 {
                HStack {
                    VStack(alignment: .leading) {
                        Text("Inference")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text(String(format: "%.0f ms", ttsInferenceMs))
                            .font(.title3.monospacedDigit())
                    }
                    Spacer()
                    VStack(alignment: .trailing) {
                        Text("Duration")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text(String(format: "%.1f s", ttsAudioDurationSec))
                            .font(.title3.monospacedDigit())
                    }
                }
                .padding(12)
                .background(Color(.systemGray6))
                .cornerRadius(10)
                .padding(.horizontal)
            }

            // Phonemes display
            if !ttsPhonemes.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Phonemes")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(ttsPhonemes)
                        .font(.system(.caption, design: .monospaced))
                        .padding(8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color(.systemGray6))
                        .cornerRadius(8)
                }
                .padding(.horizontal)
            }

            // Playback controls
            if outputURL != nil {
                VStack(spacing: 12) {
                    Divider()
                    HStack(spacing: 20) {
                        Button { togglePlayback() } label: {
                            Image(systemName: isPlaying ? "stop.circle.fill" : "play.circle.fill")
                                .font(.system(size: 44))
                                .foregroundColor(.accentColor)
                        }
                        if let url = outputURL {
                            ShareLink(item: url) {
                                Image(systemName: "square.and.arrow.up")
                                    .font(.title2)
                            }
                        }
                    }
                }
                .padding(.horizontal)
            }

            // Error
            if let errorMessage {
                Text(errorMessage)
                    .font(.caption)
                    .foregroundColor(.red)
                    .padding(.horizontal)
            }
        }
    }

    // MARK: - Music Body

    private var musicBody: some View {
        VStack(spacing: 20) {
            // Prompt
            VStack(alignment: .leading, spacing: 8) {
                Label("Prompt", systemImage: "text.quote")
                    .font(.headline)

                TextEditor(text: $musicPrompt)
                    .focused($musicTextFocused)
                    .frame(minHeight: 60, maxHeight: 100)
                    .padding(8)
                    .background(Color(.systemGray6))
                    .cornerRadius(10)

                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(musicPresets, id: \.self) { preset in
                            Button(preset) {
                                musicPrompt = preset
                            }
                            .font(.caption)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 6)
                            .background(Color(uiColor: .systemGray5))
                            .cornerRadius(8)
                        }
                    }
                }
            }
            .padding(.horizontal)

            // Duration slider
            VStack(alignment: .leading, spacing: 8) {
                Label("Duration: \(String(format: "%.1f", musicDuration))s", systemImage: "clock")
                    .font(.headline)
                Slider(value: $musicDuration, in: 1.0...min(11.9, maxDuration), step: 0.5)
            }
            .padding(.horizontal)

            // Steps slider
            VStack(alignment: .leading, spacing: 8) {
                Label("Steps: \(musicSteps)", systemImage: "arrow.triangle.2.circlepath")
                    .font(.headline)
                Slider(value: Binding(
                    get: { Double(musicSteps) },
                    set: { musicSteps = Int($0) }
                ), in: 5...50, step: 1)
            }
            .padding(.horizontal)

            // Seed
            HStack {
                Label("Seed", systemImage: "dice")
                    .font(.headline)
                TextField("Random", text: $musicSeed)
                    .textFieldStyle(.roundedBorder)
                    .keyboardType(.numberPad)
                    .frame(width: 120)
                Spacer()
            }
            .padding(.horizontal)

            // Generate button
            Button {
                Task { await generateMusic() }
            } label: {
                HStack {
                    if isProcessing {
                        ProgressView()
                            .tint(.white)
                    }
                    Image(systemName: "waveform")
                    Text(isProcessing ? "Generating..." : "Generate Music")
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background(!isProcessing && !musicPrompt.trimmingCharacters(in: .whitespaces).isEmpty ? Color.accentColor : Color.gray)
                .foregroundColor(.white)
                .cornerRadius(12)
            }
            .disabled(isProcessing || musicPrompt.trimmingCharacters(in: .whitespaces).isEmpty)
            .padding(.horizontal)

            // Progress bar
            if isProcessing {
                VStack(spacing: 8) {
                    ProgressView(value: Double(musicProgressStep), total: Double(max(musicProgressTotal, 1)))
                    Text(musicProgressMessage.isEmpty ? status : musicProgressMessage)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.horizontal)
            }

            // Error
            if let errorMessage {
                Text(errorMessage)
                    .font(.caption)
                    .foregroundColor(.red)
                    .padding(.horizontal)
            }

            // Playback
            if outputURL != nil {
                VStack(spacing: 12) {
                    Divider()
                    Label("Generated Audio", systemImage: "music.note")
                        .font(.headline)

                    HStack(spacing: 20) {
                        Button { togglePlayback() } label: {
                            Image(systemName: isPlaying ? "stop.circle.fill" : "play.circle.fill")
                                .font(.system(size: 44))
                                .foregroundColor(.accentColor)
                        }
                        if let url = outputURL {
                            ShareLink(item: url) {
                                Image(systemName: "square.and.arrow.up")
                                    .font(.title2)
                            }
                        }
                    }
                }
                .padding(.horizontal)
            }
        }
    }

    // MARK: - Playback

    private func togglePlayback() {
        if isPlaying {
            player?.stop()
            isPlaying = false
            return
        }
        guard let url = outputURL else { return }
        do {
            player = try AVAudioPlayer(contentsOf: url)
            player?.play()
            isPlaying = true
        } catch {
            errorMessage = "Playback error: \(error.localizedDescription)"
        }
    }

    // MARK: - TTS Generation

    private func generateTTS() async {
        ttsTextFocused = false
        isProcessing = true
        outputURL = nil
        isPlaying = false
        player?.stop()
        errorMessage = nil
        ttsPhonemes = ""
        ttsInferenceMs = 0
        ttsAudioDurationSec = 0

        do {
            let start = CFAbsoluteTimeGetCurrent()
            let url = try await generateKokoro()
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000

            await MainActor.run {
                ttsInferenceMs = elapsed
                outputURL = url
                isProcessing = false
                status = ""
            }

            // Auto-play
            await MainActor.run {
                do {
                    player = try AVAudioPlayer(contentsOf: url)
                    player?.play()
                    isPlaying = true
                } catch {
                    errorMessage = "Playback error: \(error.localizedDescription)"
                }
            }
        } catch {
            await MainActor.run {
                isProcessing = false
                errorMessage = error.localizedDescription
                status = ""
            }
        }
    }

    // MARK: - Music Generation

    private func generateMusic() async {
        musicTextFocused = false
        isProcessing = true
        outputURL = nil
        isPlaying = false
        player?.stop()
        errorMessage = nil
        musicProgressStep = 0
        musicProgressTotal = 0
        musicProgressMessage = ""

        do {
            let url = try await generateStableAudio()
            await MainActor.run {
                outputURL = url
                isProcessing = false
                status = ""
                musicProgressMessage = ""
            }
        } catch {
            await MainActor.run {
                isProcessing = false
                errorMessage = error.localizedDescription
                status = ""
            }
        }
    }

    // MARK: - Kokoro TTS Pipeline

    private func generateKokoro() async throws -> URL {
        let text = ttsCurrentText

        // Load vocab
        status = "Loading vocab..."
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
        let phonemes = simpleG2P(text)
        await MainActor.run { ttsPhonemes = phonemes }

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
        if !ttsVoice.isEmpty {
            let voiceFile = model.files.first { $0.name.contains(ttsVoice) }?.name
                ?? "voice_\(ttsVoice).bin"
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
        status = "Predicting duration..."
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

        // Duration -> integer frames
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
        status = "Expanding features..."
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
        status = "Generating audio..."
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
        let sampleRate = model.configInt("sample_rate") ?? 24000
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

        let audioDuration = Double(trimLen) / Double(sampleRate)
        await MainActor.run { ttsAudioDurationSec = audioDuration }

        return try writeWAV(samples: audio, sampleRate: sampleRate, channels: 1)
    }

    // MARK: - Stable Audio Pipeline

    private func generateStableAudio() async throws -> URL {
        let seedValue: UInt64
        if let parsed = UInt64(musicSeed), !musicSeed.isEmpty {
            seedValue = parsed
        } else {
            seedValue = UInt64.random(in: 0...UInt64.max)
        }

        // T5 tokenize
        status = "Tokenizing..."
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

        let tokens = t5Tokenize(musicPrompt, vocab: pieceToID, maxLen: 64)

        // Load T5 Encoder
        status = "Encoding text..."
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
        status = "Embedding duration..."
        let neFile = model.files.first { $0.name.contains("NumberEmbedder") }?.name
        guard let neFile else { throw NSError(domain: "StableAudio", code: 2) }
        let ne = try await ModelLoader.load(for: model, named: neFile)

        let normSec = try MLMultiArray(shape: [1], dataType: .float16)
        normSec.dataPointer.assumingMemoryBound(to: Float16.self)[0] = Float16(min(max(Float(musicDuration), 0), 256) / 256.0)

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
        status = "Generating..."
        let latent = try MLMultiArray(shape: [1, 64, 256], dataType: .float16)
        let lPtr = latent.dataPointer.assumingMemoryBound(to: Float16.self)
        let count = 64 * 256
        var rngState: UInt64 = seedValue
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
        let steps = musicSteps
        let schedule = makeSchedule(steps: steps)

        await MainActor.run {
            musicProgressTotal = steps
            musicProgressStep = 0
        }

        for i in 0..<steps {
            await MainActor.run {
                musicProgressStep = i
                musicProgressMessage = "Diffusion step \(i + 1)/\(steps)"
            }

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

        await MainActor.run {
            musicProgressStep = steps
            musicProgressMessage = "Decoding audio..."
        }

        // VAE Decode
        status = "Decoding audio..."
        let vaeFile = model.files.first { $0.name.contains("VAEDecoder") }?.name ?? model.files.last!.name
        let vae = try await ModelLoader.load(for: model, named: vaeFile)
        let vaeOut = try await vae.prediction(from: MLDictionaryFeatureProvider(dictionary: ["latent": latent]))
        guard let audioArr = vaeOut.featureValue(for: "audio")?.multiArrayValue else {
            throw NSError(domain: "StableAudio", code: 4)
        }

        // Extract stereo audio, trim to duration
        let sampleRate = model.configInt("sample_rate") ?? 44100
        let trimmed = min(Int(musicDuration * Double(sampleRate)), 524288)
        var ch0 = [Float](repeating: 0, count: trimmed)
        var ch1 = [Float](repeating: 0, count: trimmed)
        for i in 0..<trimmed {
            ch0[i] = audioArr[[0, 0, i] as [NSNumber]].floatValue
            ch1[i] = audioArr[[0, 1, i] as [NSNumber]].floatValue
        }

        // Interleave for WAV
        var stereo = [Float](repeating: 0, count: trimmed * 2)
        for i in 0..<trimmed { stereo[i] = ch0[i]; stereo[trimmed + i] = ch1[i] }
        return try writeWAV(samples: stereo, sampleRate: sampleRate, channels: 2)
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
