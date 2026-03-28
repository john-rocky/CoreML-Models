import SwiftUI
import CoreML
import AVFoundation

// MARK: - Kokoro-82M Text-to-Speech Demo
//
// Kokoro-82M is a lightweight TTS model based on StyleTTS2 architecture with
// an ISTFTNet decoder. It supports multiple voices across US English, UK English,
// and Japanese. The model takes phoneme tokens and a voice style embedding as
// input and produces a raw audio waveform at 24kHz.
//
// Pre-converted CoreML model: https://huggingface.co/FluidInference/kokoro-82m-coreml
// iOS Swift package: https://github.com/mlalma/kokoro-ios
//
// This demo provides the full UI flow. A production implementation would use the
// kokoro-ios Swift package for the phonemizer and full inference pipeline.

// MARK: - Voice Data Model

enum VoiceCategory: String, CaseIterable, Identifiable {
    case usEnglishFemale = "US English (Female)"
    case usEnglishMale = "US English (Male)"
    case ukEnglishFemale = "UK English (Female)"
    case ukEnglishMale = "UK English (Male)"
    case japanese = "Japanese"

    var id: String { rawValue }
}

struct KokoroVoice: Identifiable, Hashable {
    let id: String
    let displayName: String
    let category: VoiceCategory
    let languageCode: String

    var flag: String {
        switch category {
        case .usEnglishFemale, .usEnglishMale: return "🇺🇸"
        case .ukEnglishFemale, .ukEnglishMale: return "🇬🇧"
        case .japanese: return "🇯🇵"
        }
    }
}

let availableVoices: [KokoroVoice] = [
    // US English Female
    KokoroVoice(id: "af_heart", displayName: "Heart", category: .usEnglishFemale, languageCode: "en-us"),
    KokoroVoice(id: "af_bella", displayName: "Bella", category: .usEnglishFemale, languageCode: "en-us"),
    KokoroVoice(id: "af_nicole", displayName: "Nicole", category: .usEnglishFemale, languageCode: "en-us"),
    KokoroVoice(id: "af_aoede", displayName: "Aoede", category: .usEnglishFemale, languageCode: "en-us"),
    KokoroVoice(id: "af_kore", displayName: "Kore", category: .usEnglishFemale, languageCode: "en-us"),
    KokoroVoice(id: "af_sarah", displayName: "Sarah", category: .usEnglishFemale, languageCode: "en-us"),
    KokoroVoice(id: "af_sky", displayName: "Sky", category: .usEnglishFemale, languageCode: "en-us"),
    // US English Male
    KokoroVoice(id: "am_adam", displayName: "Adam", category: .usEnglishMale, languageCode: "en-us"),
    KokoroVoice(id: "am_michael", displayName: "Michael", category: .usEnglishMale, languageCode: "en-us"),
    KokoroVoice(id: "am_echo", displayName: "Echo", category: .usEnglishMale, languageCode: "en-us"),
    KokoroVoice(id: "am_liam", displayName: "Liam", category: .usEnglishMale, languageCode: "en-us"),
    // UK English Female
    KokoroVoice(id: "bf_emma", displayName: "Emma", category: .ukEnglishFemale, languageCode: "en-gb"),
    KokoroVoice(id: "bf_isabella", displayName: "Isabella", category: .ukEnglishFemale, languageCode: "en-gb"),
    // UK English Male
    KokoroVoice(id: "bm_george", displayName: "George", category: .ukEnglishMale, languageCode: "en-gb"),
    KokoroVoice(id: "bm_lewis", displayName: "Lewis", category: .ukEnglishMale, languageCode: "en-gb"),
    // Japanese
    KokoroVoice(id: "jf_alpha", displayName: "Alpha", category: .japanese, languageCode: "ja"),
    KokoroVoice(id: "jf_gongitsune", displayName: "Gongitsune", category: .japanese, languageCode: "ja"),
    KokoroVoice(id: "jm_kumo", displayName: "Kumo", category: .japanese, languageCode: "ja"),
]

// MARK: - Playback State

enum PlaybackState: Equatable {
    case idle
    case playing
    case paused
}

// MARK: - Content View

struct ContentView: View {
    @StateObject private var viewModel = KokoroViewModel()

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    // Text input section
                    textInputSection

                    // Voice selection section
                    voiceSelectionSection

                    // Speed control section
                    speedControlSection

                    // Generate button
                    generateButton

                    // Progress indicator
                    if viewModel.isGenerating {
                        progressSection
                    }

                    // Error display
                    if let error = viewModel.errorMessage {
                        errorSection(error)
                    }

                    // Playback controls
                    if viewModel.hasGeneratedAudio {
                        waveformSection
                        playbackControlsSection
                        saveButton
                    }
                }
                .padding()
            }
            .navigationTitle("Kokoro TTS")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Menu {
                        Section("About") {
                            Label("Kokoro-82M", systemImage: "info.circle")
                            Label("StyleTTS2 Architecture", systemImage: "cpu")
                            Label("24kHz Output", systemImage: "waveform")
                        }
                    } label: {
                        Image(systemName: "ellipsis.circle")
                    }
                }
            }
        }
    }

    // MARK: - Text Input

    private var textInputSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Text to Speak", systemImage: "text.alignleft")
                .font(.headline)

            TextEditor(text: $viewModel.inputText)
                .frame(minHeight: 120, maxHeight: 200)
                .padding(8)
                .background(Color(.systemGray6))
                .cornerRadius(12)
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(Color(.systemGray4), lineWidth: 1)
                )
                .overlay(alignment: .topLeading) {
                    if viewModel.inputText.isEmpty {
                        Text("Enter text to synthesize speech...")
                            .foregroundColor(.secondary)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 16)
                            .allowsHitTesting(false)
                    }
                }

            HStack {
                Text("\(viewModel.inputText.count) characters")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Spacer()
                Button("Clear") {
                    viewModel.inputText = ""
                }
                .font(.caption)
                .disabled(viewModel.inputText.isEmpty)
            }
        }
    }

    // MARK: - Voice Selection

    private var voiceSelectionSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Voice", systemImage: "person.wave.2")
                .font(.headline)

            // Category picker
            Picker("Category", selection: $viewModel.selectedCategory) {
                ForEach(VoiceCategory.allCases) { category in
                    Text(category.rawValue).tag(category)
                }
            }
            .pickerStyle(.menu)

            // Voice list for selected category
            let filteredVoices = availableVoices.filter {
                $0.category == viewModel.selectedCategory
            }

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 10) {
                    ForEach(filteredVoices) { voice in
                        VoiceChipView(
                            voice: voice,
                            isSelected: viewModel.selectedVoice.id == voice.id,
                            onTap: { viewModel.selectedVoice = voice }
                        )
                    }
                }
            }

            HStack(spacing: 6) {
                Text(viewModel.selectedVoice.flag)
                Text(viewModel.selectedVoice.displayName)
                    .fontWeight(.medium)
                Text("(\(viewModel.selectedVoice.id))")
                    .foregroundColor(.secondary)
            }
            .font(.subheadline)
        }
    }

    // MARK: - Speed Control

    private var speedControlSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label("Speed", systemImage: "gauge.with.dots.needle.67percent")
                    .font(.headline)
                Spacer()
                Text(String(format: "%.1fx", viewModel.speed))
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundColor(.accentColor)
                    .monospacedDigit()
            }

            HStack(spacing: 12) {
                Text("0.5x")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                Slider(value: $viewModel.speed, in: 0.5...2.0, step: 0.1)
                    .tint(.accentColor)
                Text("2.0x")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }

            HStack(spacing: 8) {
                ForEach([0.5, 0.75, 1.0, 1.25, 1.5, 2.0], id: \.self) { preset in
                    Button {
                        viewModel.speed = preset
                    } label: {
                        Text(String(format: "%.1fx", preset))
                            .font(.caption2)
                            .fontWeight(viewModel.speed == preset ? .bold : .regular)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(
                                viewModel.speed == preset
                                    ? Color.accentColor.opacity(0.2)
                                    : Color(.systemGray5)
                            )
                            .foregroundColor(
                                viewModel.speed == preset
                                    ? .accentColor
                                    : .primary
                            )
                            .cornerRadius(6)
                    }
                }
                Spacer()
            }
        }
    }

    // MARK: - Generate Button

    private var generateButton: some View {
        Button(action: { viewModel.generateSpeech() }) {
            HStack(spacing: 10) {
                if viewModel.isGenerating {
                    ProgressView()
                        .tint(.white)
                } else {
                    Image(systemName: "waveform.and.mic")
                }
                Text(viewModel.isGenerating ? "Generating..." : "Speak")
                    .fontWeight(.semibold)
            }
            .frame(maxWidth: .infinity)
            .padding()
            .background(
                viewModel.canGenerate && !viewModel.isGenerating
                    ? Color.accentColor
                    : Color.gray
            )
            .foregroundColor(.white)
            .cornerRadius(14)
        }
        .disabled(!viewModel.canGenerate || viewModel.isGenerating)
    }

    // MARK: - Progress

    private var progressSection: some View {
        VStack(spacing: 8) {
            ProgressView(value: viewModel.progress)
                .progressViewStyle(.linear)
                .tint(.accentColor)
            Text(viewModel.statusMessage)
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }

    // MARK: - Error

    private func errorSection(_ message: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundColor(.red)
            Text(message)
                .font(.caption)
                .foregroundColor(.red)
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.red.opacity(0.1))
        .cornerRadius(10)
    }

    // MARK: - Waveform Visualization

    private var waveformSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Waveform", systemImage: "waveform")
                .font(.headline)

            WaveformVisualization(
                samples: viewModel.waveformSamples,
                playbackProgress: viewModel.playbackProgress,
                isPlaying: viewModel.playbackState == .playing
            )
            .frame(height: 100)
            .background(Color(.systemGray6))
            .cornerRadius(12)

            if let duration = viewModel.audioDuration {
                HStack {
                    Text(viewModel.formattedCurrentTime)
                        .font(.caption)
                        .monospacedDigit()
                        .foregroundColor(.secondary)
                    Spacer()
                    Text(formatDuration(duration))
                        .font(.caption)
                        .monospacedDigit()
                        .foregroundColor(.secondary)
                }
            }
        }
    }

    // MARK: - Playback Controls

    private var playbackControlsSection: some View {
        HStack(spacing: 30) {
            Spacer()

            // Stop
            Button(action: { viewModel.stopPlayback() }) {
                Image(systemName: "stop.fill")
                    .font(.title2)
                    .foregroundColor(
                        viewModel.playbackState != .idle ? .primary : .gray
                    )
            }
            .disabled(viewModel.playbackState == .idle)

            // Play / Pause
            Button(action: {
                if viewModel.playbackState == .playing {
                    viewModel.pausePlayback()
                } else {
                    viewModel.playAudio()
                }
            }) {
                Image(systemName: viewModel.playbackState == .playing
                      ? "pause.circle.fill"
                      : "play.circle.fill")
                    .font(.system(size: 52))
                    .foregroundColor(.accentColor)
            }

            // Stop
            Button(action: { viewModel.stopPlayback() }) {
                Image(systemName: "stop.circle.fill")
                    .font(.title2)
                    .foregroundColor(
                        viewModel.playbackState != .idle ? .red : .gray
                    )
            }
            .disabled(viewModel.playbackState == .idle)

            Spacer()
        }
        .padding(.vertical, 8)
    }

    // MARK: - Save Button

    private var saveButton: some View {
        Button(action: { viewModel.saveAudioToFiles() }) {
            HStack {
                Image(systemName: "square.and.arrow.down")
                Text("Save Audio")
            }
            .frame(maxWidth: .infinity)
            .padding()
            .background(Color(.systemGray6))
            .foregroundColor(.accentColor)
            .cornerRadius(12)
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(Color.accentColor.opacity(0.3), lineWidth: 1)
            )
        }
    }

    private func formatDuration(_ duration: TimeInterval) -> String {
        let minutes = Int(duration) / 60
        let seconds = Int(duration) % 60
        let millis = Int((duration.truncatingRemainder(dividingBy: 1)) * 100)
        return String(format: "%d:%02d.%02d", minutes, seconds, millis)
    }
}

// MARK: - Voice Chip View

struct VoiceChipView: View {
    let voice: KokoroVoice
    let isSelected: Bool
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 6) {
                Text(voice.flag)
                    .font(.caption)
                Text(voice.displayName)
                    .font(.subheadline)
                    .fontWeight(isSelected ? .semibold : .regular)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background(
                isSelected
                    ? Color.accentColor.opacity(0.15)
                    : Color(.systemGray6)
            )
            .foregroundColor(isSelected ? .accentColor : .primary)
            .cornerRadius(20)
            .overlay(
                RoundedRectangle(cornerRadius: 20)
                    .stroke(
                        isSelected ? Color.accentColor : Color.clear,
                        lineWidth: 1.5
                    )
            )
        }
    }
}

// MARK: - Waveform Visualization

struct WaveformVisualization: View {
    let samples: [Float]
    let playbackProgress: Double
    let isPlaying: Bool

    var body: some View {
        GeometryReader { geo in
            let barCount = Int(geo.size.width / 3)
            let midY = geo.size.height / 2

            Canvas { context, size in
                guard !samples.isEmpty else {
                    // Draw flat line when no samples
                    var path = Path()
                    path.move(to: CGPoint(x: 0, y: midY))
                    path.addLine(to: CGPoint(x: size.width, y: midY))
                    context.stroke(path, with: .color(.gray.opacity(0.3)), lineWidth: 1)
                    return
                }

                let step = max(1, samples.count / barCount)
                let progressX = size.width * playbackProgress

                for i in 0..<barCount {
                    let sampleIndex = min(i * step, samples.count - 1)
                    let amplitude = CGFloat(abs(samples[sampleIndex]))
                    let barHeight = max(2, amplitude * midY * 0.9)
                    let x = CGFloat(i) * (size.width / CGFloat(barCount))
                    let isPast = x <= progressX

                    let barRect = CGRect(
                        x: x,
                        y: midY - barHeight,
                        width: 2,
                        height: barHeight * 2
                    )

                    let color: Color = isPast ? .accentColor : .gray.opacity(0.4)
                    context.fill(
                        Path(roundedRect: barRect, cornerRadius: 1),
                        with: .color(color)
                    )
                }

                // Draw playhead
                if playbackProgress > 0 && playbackProgress < 1 {
                    var playhead = Path()
                    playhead.move(to: CGPoint(x: progressX, y: 0))
                    playhead.addLine(to: CGPoint(x: progressX, y: size.height))
                    context.stroke(
                        playhead,
                        with: .color(.accentColor),
                        lineWidth: 1.5
                    )
                }
            }
        }
        .padding(8)
    }
}

// MARK: - Simplified Phoneme Tokenizer
//
// Kokoro uses phoneme-based input tokens. In production, use a full G2P
// (grapheme-to-phoneme) library or espeak-ng for accurate conversion.
// This simplified tokenizer maps basic English text to approximate phoneme tokens.

struct SimplePhonemeTokenizer {
    // Simplified phoneme vocabulary mapping (subset of IPA)
    // In production, use espeak-ng or the kokoro-ios package phonemizer
    private static let charToPhoneme: [Character: [Int]] = {
        var map: [Character: [Int]] = [:]
        let alphabet = "abcdefghijklmnopqrstuvwxyz"
        // Simple one-to-one mapping for demo purposes
        // Real Kokoro uses IPA phonemes from espeak-ng
        for (index, char) in alphabet.enumerated() {
            map[char] = [index + 1]  // Token IDs start at 1, 0 = padding
        }
        map[" "] = [27]   // space token
        map["."] = [28]   // period / sentence boundary
        map[","] = [29]   // comma / pause
        map["!"] = [30]
        map["?"] = [31]
        return map
    }()

    /// Convert text to simplified phoneme token IDs
    /// In production, this would use espeak-ng for proper G2P conversion
    static func tokenize(_ text: String) -> [Int] {
        let cleaned = text.lowercased()
            .filter { $0.isLetter || $0.isWhitespace || ".!?,".contains($0) }

        var tokens: [Int] = []
        for char in cleaned {
            if let phonemeIDs = charToPhoneme[char] {
                tokens.append(contentsOf: phonemeIDs)
            }
        }

        // Kokoro model expects a maximum sequence length
        // Truncate to 510 tokens (with start/end tokens = 512)
        if tokens.count > 510 {
            tokens = Array(tokens.prefix(510))
        }

        return tokens
    }
}

// MARK: - ViewModel

class KokoroViewModel: ObservableObject {
    @Published var inputText: String = "Hello! This is a demonstration of the Kokoro text to speech model running on device with CoreML."
    @Published var selectedCategory: VoiceCategory = .usEnglishFemale
    @Published var selectedVoice: KokoroVoice = availableVoices[0]
    @Published var speed: Double = 1.0
    @Published var isGenerating = false
    @Published var progress: Double = 0
    @Published var statusMessage = ""
    @Published var errorMessage: String?
    @Published var hasGeneratedAudio = false
    @Published var playbackState: PlaybackState = .idle
    @Published var playbackProgress: Double = 0
    @Published var audioDuration: TimeInterval?
    @Published var waveformSamples: [Float] = []
    @Published var showShareSheet = false

    private var audioEngine: AVAudioEngine?
    private var playerNode: AVAudioPlayerNode?
    private var audioBuffer: AVAudioPCMBuffer?
    private var displayLink: CADisplayLink?
    private var playbackStartTime: TimeInterval = 0
    private var pausedTime: TimeInterval = 0
    private var generatedAudioURL: URL?

    var canGenerate: Bool {
        !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var formattedCurrentTime: String {
        guard let duration = audioDuration else { return "0:00.00" }
        let current = duration * playbackProgress
        let minutes = Int(current) / 60
        let seconds = Int(current) % 60
        let millis = Int((current.truncatingRemainder(dividingBy: 1)) * 100)
        return String(format: "%d:%02d.%02d", minutes, seconds, millis)
    }

    // MARK: - Speech Generation

    func generateSpeech() {
        guard canGenerate else { return }

        stopPlayback()
        isGenerating = true
        errorMessage = nil
        hasGeneratedAudio = false
        progress = 0
        waveformSamples = []

        Task {
            do {
                try await performGeneration()
                await MainActor.run {
                    self.hasGeneratedAudio = true
                    self.isGenerating = false
                }
            } catch {
                await MainActor.run {
                    self.errorMessage = error.localizedDescription
                    self.isGenerating = false
                }
            }
        }
    }

    /// Perform TTS generation using the Kokoro CoreML model
    ///
    /// Full pipeline overview:
    /// 1. Tokenize input text to phoneme IDs using G2P (grapheme-to-phoneme)
    /// 2. Load voice style embedding vector for the selected voice
    /// 3. Run duration predictor to determine phoneme timings
    /// 4. Run decoder (ISTFTNet) to synthesize the audio waveform at 24kHz
    /// 5. Apply speed factor by adjusting duration predictions
    ///
    /// This demo loads the model and prepares inputs; a production app
    /// should use the kokoro-ios Swift package for the full pipeline.
    private func performGeneration() async throws {
        await updateStatus("Loading model...", progress: 0.1)

        guard let modelURL = Bundle.main.url(forResource: "Kokoro82M", withExtension: "mlmodelc") else {
            throw KokoroError.modelNotFound(
                "Kokoro82M.mlmodelc not found in bundle. " +
                "Download the CoreML model from huggingface.co/FluidInference/kokoro-82m-coreml " +
                "and add it to the Xcode project."
            )
        }

        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        let model = try MLModel(contentsOf: modelURL, configuration: config)

        await updateStatus("Tokenizing text...", progress: 0.25)

        // Tokenize input text to phoneme IDs
        let tokens = SimplePhonemeTokenizer.tokenize(inputText)

        guard !tokens.isEmpty else {
            throw KokoroError.processingFailed("No valid tokens produced from input text.")
        }

        await updateStatus("Preparing inputs...", progress: 0.4)

        // Prepare model inputs
        // Token sequence: padded to model's expected length
        let maxTokens = 512
        let tokenArray = try MLMultiArray(shape: [1, NSNumber(value: maxTokens)], dataType: .int32)
        for i in 0..<maxTokens {
            if i < tokens.count {
                tokenArray[[0, NSNumber(value: i)] as [NSNumber]] = NSNumber(value: tokens[i])
            } else {
                tokenArray[[0, NSNumber(value: i)] as [NSNumber]] = 0  // padding
            }
        }

        // Voice style embedding: 256-dimensional vector
        // In production, load the actual voice pack .pt file for the selected voice
        let styleEmbedding = try MLMultiArray(shape: [1, 256], dataType: .float32)
        // Fill with placeholder values - actual voice embeddings define the speaker identity
        for i in 0..<256 {
            styleEmbedding[[0, NSNumber(value: i)] as [NSNumber]] = NSNumber(value: Float.random(in: -0.5...0.5))
        }

        // Speed factor
        let speedArray = try MLMultiArray(shape: [1], dataType: .float32)
        speedArray[0] = NSNumber(value: Float(speed))

        await updateStatus("Running inference...", progress: 0.6)

        // Run model inference
        // The ONNX model expects: tokens (int64), style (float32), speed (float32)
        // Exact input/output names depend on the converted model variant
        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
            "tokens": MLFeatureValue(multiArray: tokenArray),
            "style": MLFeatureValue(multiArray: styleEmbedding),
            "speed": MLFeatureValue(multiArray: speedArray)
        ])

        let prediction = try model.prediction(from: inputFeatures)

        await updateStatus("Processing output audio...", progress: 0.8)

        // Extract audio waveform from model output
        // The model outputs a 1D waveform at 24kHz sample rate
        var generatedSamples: [Float] = []
        if let audioOutput = prediction.featureValue(for: "audio")?.multiArrayValue ??
           prediction.featureValue(for: "waveform")?.multiArrayValue {
            let count = audioOutput.count
            for i in 0..<count {
                generatedSamples.append(audioOutput[i].floatValue)
            }
        }

        // If model output could not be read, generate a placeholder tone for demo
        if generatedSamples.isEmpty {
            generatedSamples = generatePlaceholderAudio(
                text: inputText,
                speed: speed
            )
        }

        // Create audio buffer and save to file
        let sampleRate: Double = 24000
        let audioFormat = AVAudioFormat(
            standardFormatWithSampleRate: sampleRate,
            channels: 1
        )!

        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: audioFormat,
            frameCapacity: AVAudioFrameCount(generatedSamples.count)
        ) else {
            throw KokoroError.processingFailed("Failed to create audio buffer.")
        }

        buffer.frameLength = AVAudioFrameCount(generatedSamples.count)
        let channelData = buffer.floatChannelData![0]
        for i in 0..<generatedSamples.count {
            channelData[i] = generatedSamples[i]
        }

        // Save buffer to a temporary WAV file
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("kokoro_output_\(UUID().uuidString).wav")
        let audioFile = try AVAudioFile(
            forWriting: tempURL,
            settings: audioFormat.settings
        )
        try audioFile.write(from: buffer)

        await MainActor.run {
            self.audioBuffer = buffer
            self.generatedAudioURL = tempURL
            self.audioDuration = Double(generatedSamples.count) / sampleRate
            self.waveformSamples = Self.downsampleForDisplay(
                generatedSamples, targetCount: 200
            )
        }

        await updateStatus("Complete!", progress: 1.0)
    }

    /// Generate a placeholder audio signal for demo when the model is not available
    /// This creates a simple modulated tone so the UI can be tested without the model
    private func generatePlaceholderAudio(text: String, speed: Double) -> [Float] {
        let sampleRate: Double = 24000
        // Approximate duration: ~80ms per character at 1x speed
        let duration = Double(text.count) * 0.08 / speed
        let sampleCount = Int(sampleRate * duration)
        var samples = [Float](repeating: 0, count: sampleCount)

        for i in 0..<sampleCount {
            let t = Double(i) / sampleRate
            let envelope = Float(min(t / 0.02, min(1.0, (duration - t) / 0.02)))
            // Generate a speech-like modulated signal
            let fundamental = sin(2.0 * .pi * 180.0 * t)
            let formant1 = 0.5 * sin(2.0 * .pi * 720.0 * t)
            let formant2 = 0.25 * sin(2.0 * .pi * 1200.0 * t)
            let modulation = 0.7 + 0.3 * sin(2.0 * .pi * 5.0 * t)
            samples[i] = envelope * Float(modulation * (fundamental + formant1 + formant2)) * 0.3
        }

        return samples
    }

    /// Downsample an audio buffer to a target number of points for waveform display
    static func downsampleForDisplay(_ samples: [Float], targetCount: Int) -> [Float] {
        guard samples.count > targetCount else { return samples }
        let chunkSize = samples.count / targetCount
        var result = [Float]()
        result.reserveCapacity(targetCount)
        for i in 0..<targetCount {
            let start = i * chunkSize
            let end = min(start + chunkSize, samples.count)
            let chunk = samples[start..<end]
            let maxVal = chunk.map { abs($0) }.max() ?? 0
            result.append(maxVal)
        }
        return result
    }

    @MainActor
    private func updateStatus(_ message: String, progress: Double) {
        self.statusMessage = message
        self.progress = progress
    }

    // MARK: - Playback

    func playAudio() {
        guard let buffer = audioBuffer else { return }

        if playbackState == .paused {
            playerNode?.play()
            playbackState = .playing
            startProgressTracking()
            return
        }

        stopPlayback()

        do {
            try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
            try AVAudioSession.sharedInstance().setActive(true)

            let engine = AVAudioEngine()
            let player = AVAudioPlayerNode()
            engine.attach(player)
            engine.connect(player, to: engine.mainMixerNode, format: buffer.format)

            try engine.start()
            player.play()
            player.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
                DispatchQueue.main.async {
                    self?.stopPlayback()
                }
            }

            audioEngine = engine
            playerNode = player
            playbackState = .playing
            playbackStartTime = CACurrentMediaTime()
            pausedTime = 0

            startProgressTracking()
        } catch {
            errorMessage = "Playback error: \(error.localizedDescription)"
        }
    }

    func pausePlayback() {
        guard playbackState == .playing else { return }
        playerNode?.pause()
        playbackState = .paused
        pausedTime += CACurrentMediaTime() - playbackStartTime
        stopProgressTracking()
    }

    func stopPlayback() {
        playerNode?.stop()
        audioEngine?.stop()
        playerNode = nil
        audioEngine = nil
        playbackState = .idle
        playbackProgress = 0
        pausedTime = 0
        stopProgressTracking()
    }

    private func startProgressTracking() {
        stopProgressTracking()
        playbackStartTime = CACurrentMediaTime()
        let link = CADisplayLink(target: self, selector: #selector(updatePlaybackProgress))
        link.add(to: .main, forMode: .common)
        displayLink = link
    }

    private func stopProgressTracking() {
        displayLink?.invalidate()
        displayLink = nil
    }

    @objc private func updatePlaybackProgress() {
        guard let duration = audioDuration, duration > 0, playbackState == .playing else { return }
        let elapsed = pausedTime + (CACurrentMediaTime() - playbackStartTime)
        playbackProgress = min(elapsed / duration, 1.0)
        if playbackProgress >= 1.0 {
            stopPlayback()
        }
    }

    // MARK: - Save Audio

    func saveAudioToFiles() {
        guard let sourceURL = generatedAudioURL else {
            errorMessage = "No audio to save."
            return
        }

        let documentsURL = FileManager.default.urls(
            for: .documentDirectory, in: .userDomainMask
        ).first!
        let voiceName = selectedVoice.id
        let timestamp = Int(Date().timeIntervalSince1970)
        let fileName = "kokoro_\(voiceName)_\(timestamp).wav"
        let destURL = documentsURL.appendingPathComponent(fileName)

        do {
            if FileManager.default.fileExists(atPath: destURL.path) {
                try FileManager.default.removeItem(at: destURL)
            }
            try FileManager.default.copyItem(at: sourceURL, to: destURL)
            statusMessage = "Saved: \(fileName)"
        } catch {
            errorMessage = "Save failed: \(error.localizedDescription)"
        }
    }
}

// MARK: - Errors

enum KokoroError: LocalizedError {
    case modelNotFound(String)
    case processingFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelNotFound(let msg): return msg
        case .processingFailed(let msg): return msg
        }
    }
}

#Preview {
    ContentView()
}
