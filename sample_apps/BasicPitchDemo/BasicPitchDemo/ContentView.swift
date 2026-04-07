import SwiftUI
import AVFoundation
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var viewModel = BasicPitchViewModel()

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    // Model status
                    HStack {
                        Circle()
                            .fill(viewModel.isModelReady ? .green : .red)
                            .frame(width: 10, height: 10)
                        Text(viewModel.isModelReady ? "Model loaded" : viewModel.statusMessage)
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Spacer()
                    }
                    .padding(.horizontal)

                    // Audio import
                    if let fileName = viewModel.audioFileName {
                        HStack {
                            Image(systemName: "music.note")
                                .font(.title2)
                                .foregroundColor(.accentColor)
                            VStack(alignment: .leading) {
                                Text(fileName)
                                    .font(.headline)
                                    .lineLimit(1)
                                if let duration = viewModel.audioDuration {
                                    Text(formatDuration(duration))
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                            }
                            Spacer()
                            Button("Change") {
                                viewModel.showFilePicker = true
                            }
                            .font(.caption)
                        }
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(12)
                        .padding(.horizontal)
                    } else {
                        Button(action: { viewModel.showFilePicker = true }) {
                            VStack(spacing: 12) {
                                Image(systemName: "square.and.arrow.down")
                                    .font(.system(size: 36))
                                    .foregroundColor(.secondary)
                                Text("Import Audio File")
                                    .foregroundColor(.secondary)
                                Text("WAV, MP3, M4A, AAC")
                                    .font(.caption2)
                                    .foregroundColor(.secondary.opacity(0.7))
                            }
                            .frame(maxWidth: .infinity)
                            .frame(height: 140)
                            .background(Color(.systemGray6))
                            .cornerRadius(12)
                        }
                        .padding(.horizontal)
                    }

                    // Threshold sliders
                    VStack(alignment: .leading, spacing: 8) {
                        Label("Onset Threshold: \(String(format: "%.2f", viewModel.onsetThreshold))", systemImage: "waveform.badge.plus")
                            .font(.headline)
                        Slider(value: $viewModel.onsetThreshold, in: 0.1...0.9, step: 0.05)

                        Label("Frame Threshold: \(String(format: "%.2f", viewModel.frameThreshold))", systemImage: "waveform")
                            .font(.headline)
                        Slider(value: $viewModel.frameThreshold, in: 0.1...0.9, step: 0.05)
                    }
                    .padding(.horizontal)

                    // Transcribe button
                    Button {
                        viewModel.transcribe()
                    } label: {
                        HStack {
                            if viewModel.isProcessing {
                                ProgressView()
                                    .tint(.white)
                            }
                            Image(systemName: "pianokeys")
                            Text(viewModel.isProcessing ? "Transcribing..." : "Transcribe to MIDI")
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(viewModel.canTranscribe ? Color.blue : Color.gray)
                        .foregroundColor(.white)
                        .cornerRadius(12)
                    }
                    .disabled(!viewModel.canTranscribe)
                    .padding(.horizontal)

                    // Progress
                    if viewModel.isProcessing {
                        Text(viewModel.statusMessage)
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .padding(.horizontal)
                    }

                    // Error
                    if let error = viewModel.errorMessage {
                        Text(error)
                            .font(.caption)
                            .foregroundColor(.red)
                            .padding(.horizontal)
                    }

                    // Results
                    if !viewModel.detectedNotes.isEmpty {
                        VStack(spacing: 12) {
                            Divider()

                            HStack {
                                Label("\(viewModel.detectedNotes.count) notes detected", systemImage: "music.note.list")
                                    .font(.headline)
                                Spacer()
                            }
                            .padding(.horizontal)

                            // Piano Roll
                            PianoRollView(
                                notes: viewModel.detectedNotes,
                                totalDuration: viewModel.audioDuration ?? 10
                            )
                            .frame(height: 250)
                            .background(Color(.systemGray6))
                            .cornerRadius(12)
                            .padding(.horizontal)

                            // Playback & Export
                            HStack(spacing: 24) {
                                // Play original audio
                                Button {
                                    viewModel.toggleOriginalPlayback()
                                } label: {
                                    VStack(spacing: 4) {
                                        Image(systemName: viewModel.isPlayingOriginal ? "stop.circle.fill" : "play.circle.fill")
                                            .font(.system(size: 36))
                                        Text("Original")
                                            .font(.caption2)
                                    }
                                }
                                .foregroundColor(.blue)

                                // Play MIDI
                                Button {
                                    viewModel.toggleMIDIPlayback()
                                } label: {
                                    VStack(spacing: 4) {
                                        Image(systemName: viewModel.isPlayingMIDI ? "stop.circle.fill" : "play.circle.fill")
                                            .font(.system(size: 36))
                                        Text("MIDI")
                                            .font(.caption2)
                                    }
                                }
                                .foregroundColor(.orange)
                                .disabled(viewModel.synthURL == nil)

                                // Export MIDI
                                if let midiURL = viewModel.midiURL {
                                    ShareLink(item: midiURL) {
                                        VStack(spacing: 4) {
                                            Image(systemName: "square.and.arrow.up")
                                                .font(.system(size: 36))
                                            Text("Export MIDI")
                                                .font(.caption2)
                                        }
                                    }
                                }
                            }
                            .padding(.horizontal)

                            Text("Listen to Original vs MIDI to verify accuracy")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                                .padding(.horizontal)

                            // Detected notes list (debug)
                            DisclosureGroup("Detected Notes") {
                                VStack(alignment: .leading, spacing: 2) {
                                    ForEach(Array(viewModel.detectedNotes.prefix(30).enumerated()), id: \.offset) { _, note in
                                        let startT = NoteCreation.frameToTime(note.startFrame)
                                        let endT = NoteCreation.frameToTime(note.endFrame)
                                        let names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
                                        let name = names[note.midiPitch % 12] + "\(note.midiPitch / 12 - 1)"
                                        Text("\(String(format: "%.2f", startT))-\(String(format: "%.2f", endT))s  \(name) (MIDI \(note.midiPitch))  amp=\(String(format: "%.2f", note.amplitude))")
                                            .font(.system(size: 10, design: .monospaced))
                                    }
                                    if viewModel.detectedNotes.count > 30 {
                                        Text("... and \(viewModel.detectedNotes.count - 30) more")
                                            .font(.caption2)
                                            .foregroundColor(.secondary)
                                    }
                                }
                            }
                            .font(.caption)
                            .padding(.horizontal)
                        }
                    }
                }
                .padding(.vertical)
            }
            .navigationTitle("Basic Pitch")
        }
        .fileImporter(
            isPresented: $viewModel.showFilePicker,
            allowedContentTypes: [.audio, .wav, .mp3, .mpeg4Audio, .aiff],
            allowsMultipleSelection: false
        ) { result in
            viewModel.handleFileImport(result)
        }
    }

    private func formatDuration(_ seconds: Double) -> String {
        let mins = Int(seconds) / 60
        let secs = Int(seconds) % 60
        return String(format: "%d:%02d", mins, secs)
    }
}

// MARK: - ViewModel

@MainActor
class BasicPitchViewModel: ObservableObject {
    @Published var isModelReady = false
    @Published var statusMessage = "Loading model..."
    @Published var audioFileName: String?
    @Published var audioDuration: Double?
    @Published var showFilePicker = false
    @Published var isProcessing = false
    @Published var errorMessage: String?
    @Published var detectedNotes: [NoteEvent] = []
    @Published var midiURL: URL?
    @Published var synthURL: URL?
    @Published var isPlayingOriginal = false
    @Published var isPlayingMIDI = false
    @Published var onsetThreshold: Float = 0.5
    @Published var frameThreshold: Float = 0.3

    private let inference = BasicPitchInference()
    private var audioURL: URL?
    private var audioPlayer: AVAudioPlayer?

    var canTranscribe: Bool {
        isModelReady && audioURL != nil && !isProcessing
    }

    init() {
        Task {
            do {
                try await inference.loadModel()
                isModelReady = true
                statusMessage = "Ready"
            } catch {
                statusMessage = "Failed to load model: \(error.localizedDescription)"
            }
        }
    }

    func handleFileImport(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            guard let url = urls.first else { return }
            guard url.startAccessingSecurityScopedResource() else { return }
            defer { url.stopAccessingSecurityScopedResource() }

            // Copy to temp
            let tempURL = FileManager.default.temporaryDirectory
                .appendingPathComponent(url.lastPathComponent)
            try? FileManager.default.removeItem(at: tempURL)
            do {
                try FileManager.default.copyItem(at: url, to: tempURL)
                audioURL = tempURL
                audioFileName = url.lastPathComponent

                let file = try AVAudioFile(forReading: tempURL)
                audioDuration = Double(file.length) / file.fileFormat.sampleRate

                // Reset results
                detectedNotes = []
                midiURL = nil
                synthURL = nil
                errorMessage = nil
            } catch {
                errorMessage = "Failed to import: \(error.localizedDescription)"
            }

        case .failure(let error):
            errorMessage = "File picker error: \(error.localizedDescription)"
        }
    }

    func transcribe() {
        guard let audioURL = audioURL else { return }
        isProcessing = true
        errorMessage = nil
        detectedNotes = []
        midiURL = nil
        synthURL = nil

        let onset = onsetThreshold
        let frame = frameThreshold
        Task.detached(priority: .userInitiated) {
            do {
                let output = try await self.inference.transcribe(audioURL: audioURL) { message in
                    Task { @MainActor in
                        self.statusMessage = message
                    }
                }

                await MainActor.run { self.statusMessage = "Detecting notes..." }

                // Log raw model output stats
                print("=== BasicPitch Debug ===")
                print("Frames: \(output.notes.count) x \(output.notes.first?.count ?? 0)")
                print("Onsets: \(output.onsets.count) x \(output.onsets.first?.count ?? 0)")
                print("Contours: \(output.contours.count) x \(output.contours.first?.count ?? 0)")
                if let firstRow = output.notes.first {
                    let maxVal = firstRow.max() ?? 0
                    let maxIdx = firstRow.firstIndex(of: maxVal) ?? 0
                    print("Notes[0] max=\(maxVal) at bin \(maxIdx) (MIDI \(maxIdx + 21))")
                }

                let notes = NoteCreation.modelOutputToNotes(
                    output: output,
                    onsetThreshold: onset,
                    frameThreshold: frame
                )

                // Log detected notes
                let noteNames = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
                print("=== Detected \(notes.count) notes ===")

                // Distribution by pitch
                var pitchCount: [Int: Int] = [:]
                for n in notes { pitchCount[n.midiPitch, default: 0] += 1 }
                print("Pitch distribution (top 20):")
                for (pitch, count) in pitchCount.sorted(by: { $0.value > $1.value }).prefix(20) {
                    let name = noteNames[pitch % 12] + "\(pitch / 12 - 1)"
                    print("  \(name)(\(pitch)): \(count) notes")
                }

                // Show only melody-range (MIDI 60-84 = C4-C6) timeline
                print("--- Melody-range timeline (C4-C6) ---")
                let melodyNotes = notes.filter { $0.midiPitch >= 60 && $0.midiPitch <= 84 }.sorted { $0.startFrame < $1.startFrame }
                for n in melodyNotes.prefix(40) {
                    let t0 = Double(n.startFrame) * 256.0 / 22050.0
                    let t1 = Double(n.endFrame) * 256.0 / 22050.0
                    let name = noteNames[n.midiPitch % 12] + "\(n.midiPitch / 12 - 1)"
                    print("  \(String(format: "%6.3f", t0))-\(String(format: "%6.3f", t1))s \(name) amp=\(String(format: "%.2f", n.amplitude))")
                }
                print("Melody notes total: \(melodyNotes.count)")
                print("========================")

                // Write MIDI
                var midiPath: URL?
                if !notes.isEmpty {
                    let path = FileManager.default.temporaryDirectory
                        .appendingPathComponent("basic_pitch_output.mid")
                    try MIDIWriter.writeMIDI(notes: notes, to: path)
                    midiPath = path
                }

                // Pre-render synth audio
                await MainActor.run { self.statusMessage = "Rendering audio..." }
                var synthPath: URL?
                if !notes.isEmpty {
                    synthPath = NoteSynthesizer.render(notes: notes)
                }

                await MainActor.run {
                    self.detectedNotes = notes
                    self.midiURL = midiPath
                    self.synthURL = synthPath
                    self.statusMessage = "Done — \(notes.count) notes"
                    self.isProcessing = false
                }
            } catch {
                await MainActor.run {
                    self.errorMessage = error.localizedDescription
                    self.statusMessage = "Error"
                    self.isProcessing = false
                }
            }
        }
    }

    func toggleOriginalPlayback() {
        if isPlayingOriginal {
            stopAllPlayback()
            return
        }
        stopAllPlayback()
        guard let url = audioURL else { return }
        do {
            audioPlayer = try AVAudioPlayer(contentsOf: url)
            audioPlayer?.play()
            isPlayingOriginal = true
        } catch {
            errorMessage = "Playback error: \(error.localizedDescription)"
        }
    }

    func toggleMIDIPlayback() {
        if isPlayingMIDI {
            stopAllPlayback()
            return
        }
        stopAllPlayback()
        guard let url = synthURL else { return }
        do {
            audioPlayer = try AVAudioPlayer(contentsOf: url)
            audioPlayer?.play()
            isPlayingMIDI = true
        } catch {
            errorMessage = "Playback error: \(error.localizedDescription)"
        }
    }

    private func stopAllPlayback() {
        audioPlayer?.stop()
        isPlayingOriginal = false
        isPlayingMIDI = false
    }
}

// MARK: - UTType extensions for file picker

extension UTType {
    static let wav = UTType(filenameExtension: "wav")!
    static let mp3 = UTType(filenameExtension: "mp3")!
}

#Preview {
    ContentView()
}
