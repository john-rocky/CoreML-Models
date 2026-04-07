import SwiftUI
import AVFoundation

struct ContentView: View {
    enum InputMode: String, CaseIterable, Identifiable {
        case freeText = "Type"
        case sample = "Sample"
        var id: String { rawValue }
    }

    enum Language: String, CaseIterable, Identifiable {
        case english = "English"
        case japanese = "日本語"
        var id: String { rawValue }
        var code: String { self == .english ? "en" : "ja" }
    }

    @StateObject private var tts = KokoroTTS()
    @State private var inputMode: InputMode = .freeText
    @State private var language: Language = .english
    @State private var freeTextEN = "Hello, this is Kokoro running on Apple's CoreML."
    @State private var freeTextJA = "今日は、これはコアエムエルのデモです。"
    @State private var selectedSampleIndex = 0
    @State private var selectedVoice = "af_heart"
    @State private var isGenerating = false
    @State private var generationTimeMs: Double = 0
    @State private var audioDurationSec: Double = 0
    @State private var lastPhonemes: String = ""
    @State private var errorMessage: String?
    @State private var player: AVAudioPlayer?
    @FocusState private var textFieldFocused: Bool

    private var voicesForLanguage: [String] {
        let prefix = language == .english ? ["a", "b"] : ["j"]
        return tts.availableVoices.filter { v in
            prefix.contains(where: { v.hasPrefix($0) })
        }
    }

    private var samplesForLanguage: [(Int, KokoroTTS.SampleEntry)] {
        let lang = language.code
        return tts.availableSamples.enumerated().filter { (_, s) in
            (s.language ?? "en") == lang
        }
    }

    private var freeTextBinding: Binding<String> {
        language == .english ? $freeTextEN : $freeTextJA
    }

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 18) {
                    statusRow
                    languageSection
                    inputModeSection
                    voiceSection
                    generateButton
                    statsRow
                    if !lastPhonemes.isEmpty {
                        phonemesView
                    }
                    if let errorMessage {
                        Text(errorMessage)
                            .font(.caption)
                            .foregroundColor(.red)
                    }
                }
                .padding()
            }
            .navigationTitle("Kokoro TTS")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItemGroup(placement: .keyboard) {
                    Spacer()
                    Button("Done") { textFieldFocused = false }
                }
            }
            .onTapGesture { textFieldFocused = false }
        }
    }

    // MARK: - Sections

    private var statusRow: some View {
        HStack {
            Circle()
                .fill(tts.isReady ? Color.green : Color.orange)
                .frame(width: 10, height: 10)
            Text(tts.status)
                .font(.caption)
                .foregroundColor(.secondary)
            Spacer()
        }
    }

    private var languageSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Language")
                .font(.headline)
            Picker("Language", selection: $language) {
                ForEach(Language.allCases) { l in
                    Text(l.rawValue).tag(l)
                }
            }
            .pickerStyle(.segmented)
            .onChange(of: language) { _, newValue in
                // Switch to a default voice for the new language
                let voices = voicesForLanguage
                if !voices.contains(selectedVoice), let first = voices.first {
                    selectedVoice = first
                }
                selectedSampleIndex = 0
            }
        }
    }

    private var inputModeSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Picker("Input mode", selection: $inputMode) {
                ForEach(InputMode.allCases) { m in
                    Text(m.rawValue).tag(m)
                }
            }
            .pickerStyle(.segmented)

            if inputMode == .freeText {
                TextEditor(text: freeTextBinding)
                    .focused($textFieldFocused)
                    .frame(minHeight: 100, maxHeight: 200)
                    .padding(8)
                    .background(Color(uiColor: .secondarySystemBackground))
                    .cornerRadius(10)
                    .overlay(
                        RoundedRectangle(cornerRadius: 10)
                            .stroke(Color(uiColor: .separator), lineWidth: 0.5)
                    )
            } else if !tts.availableSamples.isEmpty {
                let samples = samplesForLanguage
                Picker("Sample", selection: $selectedSampleIndex) {
                    ForEach(0..<samples.count, id: \.self) { i in
                        Text(samples[i].1.text)
                            .lineLimit(1)
                            .tag(i)
                    }
                }
                .pickerStyle(.menu)
                if selectedSampleIndex < samples.count {
                    Text(samples[selectedSampleIndex].1.text)
                        .font(.body)
                        .padding(12)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color(uiColor: .secondarySystemBackground))
                        .cornerRadius(10)
                }
            }
        }
    }

    private var voiceSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Voice")
                .font(.headline)
            Picker("Voice", selection: $selectedVoice) {
                ForEach(voicesForLanguage, id: \.self) { v in
                    Text(v).tag(v)
                }
            }
            .pickerStyle(.segmented)
        }
    }

    private var generateButton: some View {
        Button(action: synthesize) {
            HStack {
                if isGenerating {
                    ProgressView()
                        .progressViewStyle(.circular)
                        .tint(.white)
                } else {
                    Image(systemName: "waveform")
                }
                Text(isGenerating ? "Synthesizing..." : "Generate Speech")
            }
            .frame(maxWidth: .infinity)
            .padding()
            .background(tts.isReady && !isGenerating ? Color.accentColor : Color.gray)
            .foregroundColor(.white)
            .cornerRadius(12)
        }
        .disabled(!tts.isReady || isGenerating)
    }

    @ViewBuilder
    private var statsRow: some View {
        if generationTimeMs > 0 {
            HStack {
                VStack(alignment: .leading) {
                    Text("Inference")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(String(format: "%.0f ms", generationTimeMs))
                        .font(.title3.monospacedDigit())
                }
                Spacer()
                VStack(alignment: .trailing) {
                    Text("Duration")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(String(format: "%.1f s", audioDurationSec))
                        .font(.title3.monospacedDigit())
                }
            }
            .padding(12)
            .background(Color(uiColor: .secondarySystemBackground))
            .cornerRadius(10)
        }
    }

    private var phonemesView: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Phonemes")
                .font(.caption)
                .foregroundColor(.secondary)
            Text(lastPhonemes)
                .font(.system(.caption, design: .monospaced))
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color(uiColor: .secondarySystemBackground))
                .cornerRadius(8)
        }
    }

    // MARK: - Action

    private func synthesize() {
        guard tts.isReady else { return }
        textFieldFocused = false

        let voice = selectedVoice
        let mode = inputMode
        let lang = language.code
        let text = language == .english ? freeTextEN : freeTextJA
        let samples = samplesForLanguage
        let sampleIDs: [Int32]?
        let samplePhonemes: String?
        if mode == .sample, selectedSampleIndex < samples.count {
            sampleIDs = samples[selectedSampleIndex].1.input_ids
            samplePhonemes = samples[selectedSampleIndex].1.phonemes
        } else {
            sampleIDs = nil
            samplePhonemes = nil
        }

        isGenerating = true
        errorMessage = nil

        DispatchQueue.global(qos: .userInitiated).async {
            let start = CFAbsoluteTimeGetCurrent()
            do {
                let audio: [Float]
                let phonemes: String
                if let sampleIDs {
                    audio = try tts.synthesize(inputIDs: sampleIDs, voice: voice)
                    phonemes = samplePhonemes ?? ""
                } else {
                    let g2p = tts.previewPhonemes(text: text, language: lang)
                    audio = try tts.synthesize(text: text, voice: voice, language: lang)
                    phonemes = g2p
                }
                let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000

                let url = FileManager.default.temporaryDirectory
                    .appendingPathComponent("kokoro_out.wav")
                try KokoroTTS.writeWav(samples: audio, to: url)
                let duration = Double(audio.count) / KokoroTTS.sampleRate

                DispatchQueue.main.async {
                    generationTimeMs = elapsed
                    audioDurationSec = duration
                    lastPhonemes = phonemes
                    isGenerating = false
                    do {
                        player = try AVAudioPlayer(contentsOf: url)
                        player?.play()
                    } catch {
                        errorMessage = "Playback error: \(error.localizedDescription)"
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    isGenerating = false
                    errorMessage = error.localizedDescription
                }
            }
        }
    }
}

#Preview {
    ContentView()
}
