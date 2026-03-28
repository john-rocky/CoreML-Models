import SwiftUI
import CoreML
import AVFoundation
import Accelerate

// MARK: - Whisper Tiny Speech Recognition Demo
//
// Whisper is a general-purpose speech recognition model by OpenAI.
// The encoder processes a mel spectrogram (80 bins x 3000 frames for 30s of audio)
// and produces hidden states that the decoder uses autoregressively to generate tokens.
//
// This demo records audio via the microphone, computes a log-mel spectrogram using
// the Accelerate framework (vDSP), runs the WhisperTiny encoder CoreML model, and
// displays transcription results. The decoder step is simplified for demonstration;
// a production app should use WhisperKit or a full encoder+decoder pipeline.

// MARK: - Supported Languages

enum WhisperLanguage: String, CaseIterable, Identifiable {
    case english = "English"
    case japanese = "Japanese"
    case spanish = "Spanish"
    case french = "French"
    case german = "German"
    case chinese = "Chinese"
    case korean = "Korean"
    case portuguese = "Portuguese"

    var id: String { rawValue }

    var code: String {
        switch self {
        case .english: return "en"
        case .japanese: return "ja"
        case .spanish: return "es"
        case .french: return "fr"
        case .german: return "de"
        case .chinese: return "zh"
        case .korean: return "ko"
        case .portuguese: return "pt"
        }
    }
}

// MARK: - Transcription Entry

struct TranscriptionEntry: Identifiable {
    let id = UUID()
    let text: String
    let language: WhisperLanguage
    let timestamp: Date
    let duration: TimeInterval
}

// MARK: - ContentView

struct ContentView: View {
    @StateObject private var viewModel = WhisperViewModel()

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Language picker
                HStack {
                    Text("Language")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    Spacer()
                    Picker("Language", selection: $viewModel.selectedLanguage) {
                        ForEach(WhisperLanguage.allCases) { lang in
                            Text(lang.rawValue).tag(lang)
                        }
                    }
                    .pickerStyle(.menu)
                }
                .padding(.horizontal)
                .padding(.top, 8)

                Divider()
                    .padding(.vertical, 8)

                // Waveform visualization
                WaveformVisualization(
                    samples: viewModel.audioSamples,
                    isRecording: viewModel.isRecording
                )
                .frame(height: 100)
                .padding(.horizontal)
                .padding(.bottom, 8)

                // Recording controls
                VStack(spacing: 12) {
                    RecordButton(
                        isRecording: viewModel.isRecording,
                        onTap: {
                            if viewModel.isRecording {
                                viewModel.stopRecording()
                            } else {
                                viewModel.startRecording()
                            }
                        }
                    )

                    Text(viewModel.isRecording ? "Tap to stop recording" : "Tap to start recording")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    if viewModel.isRecording {
                        Text(viewModel.formattedRecordingDuration)
                            .font(.system(.title3, design: .monospaced))
                            .foregroundColor(.red)
                    }
                }
                .padding(.vertical, 12)

                // Processing indicator
                if viewModel.isProcessing {
                    VStack(spacing: 8) {
                        ProgressView()
                            .scaleEffect(1.2)
                        Text(viewModel.processingStatus)
                            .font(.caption)
                            .foregroundColor(.secondary)
                        ProgressView(value: viewModel.processingProgress)
                            .progressViewStyle(.linear)
                            .padding(.horizontal, 40)
                    }
                    .padding()
                }

                // Error display
                if let error = viewModel.errorMessage {
                    HStack {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.red)
                        Text(error)
                            .font(.caption)
                            .foregroundColor(.red)
                    }
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(Color.red.opacity(0.1))
                    .cornerRadius(8)
                    .padding(.horizontal)
                }

                // Current transcription result
                if let current = viewModel.currentTranscription {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Transcription")
                                .font(.headline)
                            Spacer()
                            Button(action: { viewModel.copyToClipboard(current.text) }) {
                                Image(systemName: "doc.on.doc")
                                    .font(.body)
                            }
                        }
                        Text(current.text)
                            .font(.body)
                            .padding()
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(Color(.systemGray6))
                            .cornerRadius(10)
                        HStack {
                            Text(current.language.rawValue)
                                .font(.caption2)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 2)
                                .background(Color.accentColor.opacity(0.15))
                                .cornerRadius(4)
                            Text(formatDuration(current.duration))
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }
                    }
                    .padding(.horizontal)
                    .padding(.vertical, 8)
                }

                Divider()
                    .padding(.vertical, 4)

                // History list
                if viewModel.transcriptionHistory.isEmpty && viewModel.currentTranscription == nil {
                    Spacer()
                    VStack(spacing: 12) {
                        Image(systemName: "waveform.circle")
                            .font(.system(size: 48))
                            .foregroundColor(.secondary.opacity(0.5))
                        Text("Record audio to begin transcription")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    Spacer()
                } else {
                    ScrollView {
                        LazyVStack(spacing: 10) {
                            ForEach(viewModel.transcriptionHistory) { entry in
                                TranscriptionRow(
                                    entry: entry,
                                    onCopy: { viewModel.copyToClipboard(entry.text) }
                                )
                            }
                        }
                        .padding(.horizontal)
                        .padding(.vertical, 8)
                    }
                }
            }
            .navigationTitle("Whisper Transcribe")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    if !viewModel.transcriptionHistory.isEmpty {
                        Button("Clear") {
                            viewModel.clearHistory()
                        }
                    }
                }
            }
            .onAppear {
                viewModel.requestMicrophonePermission()
            }
        }
    }

    private func formatDuration(_ duration: TimeInterval) -> String {
        let seconds = Int(duration)
        let ms = Int((duration - Double(seconds)) * 10)
        return String(format: "%d.%ds", seconds, ms)
    }
}

// MARK: - Record Button

struct RecordButton: View {
    let isRecording: Bool
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            ZStack {
                Circle()
                    .fill(isRecording ? Color.red.opacity(0.15) : Color.accentColor.opacity(0.1))
                    .frame(width: 80, height: 80)

                Circle()
                    .fill(isRecording ? Color.red : Color.accentColor)
                    .frame(width: 60, height: 60)

                if isRecording {
                    RoundedRectangle(cornerRadius: 4)
                        .fill(Color.white)
                        .frame(width: 22, height: 22)
                } else {
                    Circle()
                        .fill(Color.white)
                        .frame(width: 24, height: 24)
                }
            }
        }
        .buttonStyle(.plain)
        .animation(.easeInOut(duration: 0.2), value: isRecording)
    }
}

// MARK: - Waveform Visualization

struct WaveformVisualization: View {
    let samples: [Float]
    let isRecording: Bool
    @State private var animationPhase: CGFloat = 0

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 30.0)) { timeline in
            Canvas { context, size in
                let midY = size.height / 2
                let barWidth: CGFloat = 3
                let gap: CGFloat = 2
                let totalBarWidth = barWidth + gap
                let barCount = Int(size.width / totalBarWidth)

                if isRecording && !samples.isEmpty {
                    let step = max(1, samples.count / barCount)
                    for i in 0..<barCount {
                        let sampleIndex = min(i * step, samples.count - 1)
                        let amplitude = CGFloat(abs(samples[sampleIndex]))
                        let normalizedHeight = min(amplitude * size.height * 4, size.height * 0.9)
                        let barHeight = max(normalizedHeight, 2)
                        let x = CGFloat(i) * totalBarWidth
                        let rect = CGRect(
                            x: x,
                            y: midY - barHeight / 2,
                            width: barWidth,
                            height: barHeight
                        )
                        let roundedPath = Path(roundedRect: rect, cornerRadius: 1.5)
                        let opacity = 0.5 + 0.5 * Double(amplitude * 5).clamped(to: 0...1)
                        context.fill(roundedPath, with: .color(Color.red.opacity(opacity)))
                    }
                } else {
                    let time = timeline.date.timeIntervalSinceReferenceDate
                    for i in 0..<barCount {
                        let x = CGFloat(i) * totalBarWidth
                        let normalizedX = Double(i) / Double(barCount)
                        let height: CGFloat
                        if isRecording {
                            height = 2
                        } else {
                            let wave = sin(normalizedX * .pi * 4 + time * 2) * 0.3 + 0.5
                            height = max(CGFloat(wave) * 8, 2)
                        }
                        let rect = CGRect(
                            x: x,
                            y: midY - height / 2,
                            width: barWidth,
                            height: height
                        )
                        let roundedPath = Path(roundedRect: rect, cornerRadius: 1.5)
                        context.fill(roundedPath, with: .color(Color(.systemGray3)))
                    }
                }
            }
        }
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(.systemGray6))
        )
    }
}

// MARK: - Transcription Row

struct TranscriptionRow: View {
    let entry: TranscriptionEntry
    let onCopy: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(entry.language.rawValue)
                    .font(.caption2)
                    .fontWeight(.medium)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Color.accentColor.opacity(0.12))
                    .cornerRadius(4)

                Text(entry.timestamp, style: .time)
                    .font(.caption2)
                    .foregroundColor(.secondary)

                Spacer()

                Button(action: onCopy) {
                    Image(systemName: "doc.on.doc")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }

            Text(entry.text)
                .font(.body)
                .lineLimit(4)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(10)
    }
}

// MARK: - Clamped Extension

private extension Double {
    func clamped(to range: ClosedRange<Double>) -> Double {
        return min(max(self, range.lowerBound), range.upperBound)
    }
}

// MARK: - WhisperViewModel

class WhisperViewModel: ObservableObject {
    @Published var selectedLanguage: WhisperLanguage = .english
    @Published var isRecording = false
    @Published var isProcessing = false
    @Published var processingStatus = ""
    @Published var processingProgress: Double = 0
    @Published var errorMessage: String?
    @Published var currentTranscription: TranscriptionEntry?
    @Published var transcriptionHistory: [TranscriptionEntry] = []
    @Published var audioSamples: [Float] = []
    @Published var recordingDuration: TimeInterval = 0

    private var audioRecorder: AVAudioRecorder?
    private var recordingURL: URL?
    private var recordingTimer: Timer?
    private var sampleTimer: Timer?
    private var recordingStartTime: Date?

    var formattedRecordingDuration: String {
        let minutes = Int(recordingDuration) / 60
        let seconds = Int(recordingDuration) % 60
        let tenths = Int((recordingDuration - floor(recordingDuration)) * 10)
        return String(format: "%d:%02d.%d", minutes, seconds, tenths)
    }

    // MARK: - Microphone Permission

    func requestMicrophonePermission() {
        AVAudioSession.sharedInstance().requestRecordPermission { [weak self] granted in
            DispatchQueue.main.async {
                if !granted {
                    self?.errorMessage = "Microphone access denied. Please enable it in Settings."
                }
            }
        }
    }

    // MARK: - Recording

    func startRecording() {
        errorMessage = nil
        currentTranscription = nil

        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker])
            try session.setActive(true)
        } catch {
            errorMessage = "Failed to configure audio session: \(error.localizedDescription)"
            return
        }

        let tempDir = FileManager.default.temporaryDirectory
        let fileName = "whisper_recording_\(UUID().uuidString).wav"
        let fileURL = tempDir.appendingPathComponent(fileName)
        recordingURL = fileURL

        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: 16000.0,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false
        ]

        do {
            audioRecorder = try AVAudioRecorder(url: fileURL, settings: settings)
            audioRecorder?.isMeteringEnabled = true
            audioRecorder?.record()
            isRecording = true
            recordingStartTime = Date()
            recordingDuration = 0
            audioSamples = []
            startTimers()
        } catch {
            errorMessage = "Failed to start recording: \(error.localizedDescription)"
        }
    }

    func stopRecording() {
        guard isRecording else { return }

        audioRecorder?.stop()
        isRecording = false
        stopTimers()

        let duration = recordingDuration

        guard let url = recordingURL else {
            errorMessage = "Recording file not found."
            return
        }

        processRecording(url: url, duration: duration)
    }

    private func startTimers() {
        recordingTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            guard let self = self, let start = self.recordingStartTime else { return }
            DispatchQueue.main.async {
                self.recordingDuration = Date().timeIntervalSince(start)
            }
        }

        sampleTimer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            self.audioRecorder?.updateMeters()
            let power = self.audioRecorder?.averagePower(forChannel: 0) ?? -160
            // Convert dB to linear amplitude (0..1)
            let linear = pow(10, power / 20)
            DispatchQueue.main.async {
                self.audioSamples.append(linear)
                // Keep a rolling window of samples for visualization
                if self.audioSamples.count > 400 {
                    self.audioSamples.removeFirst(self.audioSamples.count - 400)
                }
            }
        }
    }

    private func stopTimers() {
        recordingTimer?.invalidate()
        recordingTimer = nil
        sampleTimer?.invalidate()
        sampleTimer = nil
    }

    // MARK: - Audio Processing

    private func processRecording(url: URL, duration: TimeInterval) {
        isProcessing = true
        errorMessage = nil
        processingProgress = 0
        processingStatus = "Loading audio..."

        Task {
            do {
                let transcription = try await runWhisperPipeline(url: url, duration: duration)
                await MainActor.run {
                    let entry = TranscriptionEntry(
                        text: transcription,
                        language: self.selectedLanguage,
                        timestamp: Date(),
                        duration: duration
                    )
                    self.currentTranscription = entry
                    self.transcriptionHistory.insert(entry, at: 0)
                    self.isProcessing = false
                    self.processingProgress = 1.0
                }
            } catch {
                await MainActor.run {
                    self.errorMessage = error.localizedDescription
                    self.isProcessing = false
                }
            }

            // Clean up temp file
            try? FileManager.default.removeItem(at: url)
        }
    }

    /// Full Whisper pipeline: load audio -> compute mel spectrogram -> run encoder -> decode
    ///
    /// NOTE: The decoder step is simplified here. A full implementation would:
    /// 1. Feed encoder output into the decoder model autoregressively
    /// 2. Use greedy or beam search to generate token IDs
    /// 3. Decode token IDs using the Whisper tokenizer
    /// For production use, consider WhisperKit (github.com/argmaxinc/WhisperKit).
    private func runWhisperPipeline(url: URL, duration: TimeInterval) async throws -> String {
        // Step 1: Load audio samples from WAV file
        await updateProgress("Loading audio file...", progress: 0.1)

        let audioData = try loadAudioSamples(from: url)

        // Step 2: Compute log-mel spectrogram using Accelerate
        await updateProgress("Computing mel spectrogram...", progress: 0.3)

        let melSpectrogram = try computeMelSpectrogram(from: audioData)

        // Step 3: Load and run encoder model
        await updateProgress("Running Whisper encoder...", progress: 0.5)

        guard let modelURL = Bundle.main.url(forResource: "WhisperTinyEncoder", withExtension: "mlmodelc") else {
            throw WhisperError.modelNotFound(
                "WhisperTinyEncoder.mlmodelc not found in bundle. " +
                "Run convert_whisper.py to generate the model, then add the compiled " +
                "WhisperTinyEncoder.mlmodelc to the Xcode project."
            )
        }

        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        let model = try MLModel(contentsOf: modelURL, configuration: config)

        // Prepare mel input: shape (1, 80, 3000)
        let melInput = try MLMultiArray(shape: [1, 80, 3000], dataType: .float32)
        let melCount = min(melSpectrogram.count, 80 * 3000)
        for i in 0..<melCount {
            melInput[i] = NSNumber(value: melSpectrogram[i])
        }

        await updateProgress("Running inference...", progress: 0.7)

        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
            "mel_input": MLFeatureValue(multiArray: melInput)
        ])

        let _ = try model.prediction(from: inputFeatures)

        // Step 4: Decoder (simplified)
        // In a full implementation, the encoder_output would be passed to the decoder
        // model in an autoregressive loop. Each step generates one token until
        // the end-of-transcript token is emitted.
        await updateProgress("Decoding tokens...", progress: 0.9)

        // Placeholder: return a message indicating successful encoder pass
        // A real implementation needs the decoder model and tokenizer.
        let languageNote = selectedLanguage.rawValue
        let durationStr = String(format: "%.1f", duration)
        return "[Whisper encoder processed \(durationStr)s of \(languageNote) audio successfully. " +
               "Connect the decoder model and tokenizer for full transcription output.]"
    }

    /// Load 16kHz mono PCM samples from a WAV file
    private func loadAudioSamples(from url: URL) throws -> [Float] {
        let fileData = try Data(contentsOf: url)

        // WAV header is 44 bytes; PCM 16-bit mono samples follow
        guard fileData.count > 44 else {
            throw WhisperError.processingFailed("Audio file too short or corrupted.")
        }

        let sampleData = fileData.dropFirst(44)
        let sampleCount = sampleData.count / 2 // 16-bit = 2 bytes per sample

        var floatSamples = [Float](repeating: 0, count: sampleCount)
        sampleData.withUnsafeBytes { rawBuffer in
            guard let baseAddress = rawBuffer.baseAddress else { return }
            let int16Ptr = baseAddress.bindMemory(to: Int16.self, capacity: sampleCount)
            // Convert Int16 samples to Float32 normalized to [-1, 1]
            var source = UnsafePointer(int16Ptr)
            var destination = UnsafeMutablePointer(&floatSamples)
            // Use vDSP for efficient conversion
            vDSP_vflt16(source, 1, &floatSamples, 1, vDSP_Length(sampleCount))
            var scale: Float = 1.0 / 32768.0
            vDSP_vsmul(floatSamples, 1, &scale, &floatSamples, 1, vDSP_Length(sampleCount))
        }

        return floatSamples
    }

    /// Compute 80-bin log-mel spectrogram from audio samples
    ///
    /// Whisper expects: 80 mel bins, 3000 time frames (for 30s at 16kHz with hop=160).
    /// Parameters: FFT size = 400, hop length = 160, sample rate = 16000.
    ///
    /// This implementation uses Accelerate's vDSP for the FFT computation.
    private func computeMelSpectrogram(from samples: [Float]) throws -> [Float] {
        let fftSize = 400
        let hopLength = 160
        let numMelBins = 80
        let maxFrames = 3000
        let sampleRate: Float = 16000.0

        // Pad or truncate audio to 30 seconds (480000 samples)
        let targetLength = 480000
        var paddedSamples: [Float]
        if samples.count >= targetLength {
            paddedSamples = Array(samples.prefix(targetLength))
        } else {
            paddedSamples = samples + [Float](repeating: 0, count: targetLength - samples.count)
        }

        // Number of frames
        let numFrames = min((paddedSamples.count - fftSize) / hopLength + 1, maxFrames)

        // Create FFT setup
        let log2n = vDSP_Length(ceil(log2(Float(fftSize))))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            throw WhisperError.processingFailed("Failed to create FFT setup.")
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        let fftSizeAligned = Int(pow(2, ceil(log2(Float(fftSize)))))
        let halfFFT = fftSizeAligned / 2

        // Hann window
        var window = [Float](repeating: 0, count: fftSize)
        vDSP_hann_window(&window, vDSP_Length(fftSize), Int32(vDSP_HANN_NORM))

        // Compute mel filter bank (simplified triangular filters)
        let melFilters = createMelFilterBank(
            numMelBins: numMelBins,
            fftSize: fftSizeAligned,
            sampleRate: sampleRate,
            numFreqBins: halfFFT + 1
        )

        // Output: (numMelBins x numFrames) stored row-major
        var melSpectrogram = [Float](repeating: 0, count: numMelBins * maxFrames)

        // Process each frame
        for frame in 0..<numFrames {
            let start = frame * hopLength
            var windowedFrame = [Float](repeating: 0, count: fftSizeAligned)

            // Apply window
            for i in 0..<fftSize {
                windowedFrame[i] = paddedSamples[start + i] * window[i]
            }

            // Split into real and imaginary for vDSP FFT
            var realPart = [Float](repeating: 0, count: halfFFT)
            var imagPart = [Float](repeating: 0, count: halfFFT)
            // Pack into split complex
            for i in 0..<halfFFT {
                realPart[i] = windowedFrame[2 * i]
                imagPart[i] = windowedFrame[2 * i + 1]
            }

            var splitComplex = DSPSplitComplex(
                realp: &realPart,
                imagp: &imagPart
            )

            // Forward FFT
            vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFTDirection(kFFTDirection_Forward))

            // Compute power spectrum
            var powerSpectrum = [Float](repeating: 0, count: halfFFT + 1)
            vDSP_zvmags(&splitComplex, 1, &powerSpectrum, 1, vDSP_Length(halfFFT))

            // Apply mel filter bank and log transform
            for mel in 0..<numMelBins {
                var dotProduct: Float = 0
                let filterOffset = mel * (halfFFT + 1)
                vDSP_dotpr(
                    powerSpectrum, 1,
                    Array(melFilters[filterOffset..<filterOffset + halfFFT + 1]), 1,
                    &dotProduct,
                    vDSP_Length(halfFFT + 1)
                )
                // Log-mel: log(max(dotProduct, 1e-10))
                let logMel = log(max(dotProduct, 1e-10))
                melSpectrogram[mel * maxFrames + frame] = logMel
            }
        }

        vDSP_destroy_fftsetup(fftSetup)

        return melSpectrogram
    }

    /// Create triangular mel filter bank
    /// Returns a flat array of shape (numMelBins x numFreqBins)
    private func createMelFilterBank(
        numMelBins: Int,
        fftSize: Int,
        sampleRate: Float,
        numFreqBins: Int
    ) -> [Float] {
        func hzToMel(_ hz: Float) -> Float {
            return 2595.0 * log10(1.0 + hz / 700.0)
        }

        func melToHz(_ mel: Float) -> Float {
            return 700.0 * (pow(10.0, mel / 2595.0) - 1.0)
        }

        let lowFreq: Float = 0
        let highFreq = sampleRate / 2.0
        let lowMel = hzToMel(lowFreq)
        let highMel = hzToMel(highFreq)

        // Equally spaced mel points
        let numPoints = numMelBins + 2
        var melPoints = [Float](repeating: 0, count: numPoints)
        for i in 0..<numPoints {
            melPoints[i] = lowMel + Float(i) * (highMel - lowMel) / Float(numPoints - 1)
        }

        // Convert back to Hz and then to FFT bin indices
        var binIndices = [Int](repeating: 0, count: numPoints)
        for i in 0..<numPoints {
            let hz = melToHz(melPoints[i])
            binIndices[i] = Int((hz / (sampleRate / Float(fftSize))) + 0.5)
        }

        // Build triangular filters
        var filters = [Float](repeating: 0, count: numMelBins * numFreqBins)
        for m in 0..<numMelBins {
            let left = binIndices[m]
            let center = binIndices[m + 1]
            let right = binIndices[m + 2]

            for k in left..<center {
                if k < numFreqBins && center > left {
                    filters[m * numFreqBins + k] = Float(k - left) / Float(center - left)
                }
            }
            for k in center..<right {
                if k < numFreqBins && right > center {
                    filters[m * numFreqBins + k] = Float(right - k) / Float(right - center)
                }
            }
        }

        return filters
    }

    @MainActor
    private func updateProgress(_ status: String, progress: Double) {
        self.processingStatus = status
        self.processingProgress = progress
    }

    // MARK: - Clipboard

    func copyToClipboard(_ text: String) {
        UIPasteboard.general.string = text
    }

    // MARK: - History

    func clearHistory() {
        transcriptionHistory.removeAll()
        currentTranscription = nil
    }
}

// MARK: - Errors

enum WhisperError: LocalizedError {
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
