import SwiftUI
import UIKit
import CoreML
import AVFoundation
import UniformTypeIdentifiers

// MARK: - HTDemucs Audio Source Separation Demo
//
// HTDemucs separates audio into 4 stems: Vocals, Drums, Bass, Other.
//
// IMPORTANT: The model operates in the frequency domain.
// In a production app, you must perform STFT (Short-Time Fourier Transform) on the input
// audio to produce the freq_input (1,8,2049,336) tensor, and also provide the raw
// time_input (1,2,343980) waveform. After inference, the frequency and time domain
// outputs must be combined via iSTFT (Inverse STFT) to reconstruct each stem's waveform.
//
// This demo uses simplified/placeholder audio processing to demonstrate the UI flow.
// A full implementation would require an STFT library (e.g., Accelerate vDSP).

enum Stem: String, CaseIterable, Identifiable {
    case vocals = "Vocals"
    case drums = "Drums"
    case bass = "Bass"
    case other = "Other"

    var id: String { rawValue }

    var icon: String {
        switch self {
        case .vocals: return "mic.fill"
        case .drums: return "drum.fill"
        case .bass: return "guitars.fill"
        case .other: return "waveform"
        }
    }

    var color: Color {
        switch self {
        case .vocals: return .purple
        case .drums: return .orange
        case .bass: return .blue
        case .other: return .green
        }
    }
}

struct ContentView: View {
    @StateObject private var viewModel = DemucsViewModel()

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Audio import section
                VStack(spacing: 16) {
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
                    }
                }
                .padding()

                // Separation button
                if viewModel.audioURL != nil && !viewModel.isSeparated {
                    Button(action: { viewModel.separate() }) {
                        HStack {
                            if viewModel.isProcessing {
                                ProgressView()
                                    .tint(.white)
                            } else {
                                Image(systemName: "scissors")
                            }
                            Text(viewModel.isProcessing ? "Separating..." : "Separate Stems")
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(viewModel.isProcessing ? Color.gray : Color.accentColor)
                        .foregroundColor(.white)
                        .cornerRadius(12)
                    }
                    .disabled(viewModel.isProcessing)
                    .padding(.horizontal)
                }

                // Progress
                if viewModel.isProcessing {
                    VStack(spacing: 8) {
                        ProgressView(value: viewModel.progress)
                            .progressViewStyle(.linear)
                        Text(viewModel.statusMessage)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding()
                }

                // Error
                if let error = viewModel.errorMessage {
                    Text(error)
                        .foregroundColor(.red)
                        .font(.caption)
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(Color.red.opacity(0.1))
                        .cornerRadius(8)
                        .padding(.horizontal)
                }

                // Stem controls
                if viewModel.isSeparated {
                    VStack(spacing: 12) {
                        Text("Separated Stems")
                            .font(.headline)
                            .frame(maxWidth: .infinity, alignment: .leading)

                        ForEach(Stem.allCases) { stem in
                            StemPlayerView(
                                stem: stem,
                                isPlaying: viewModel.playingStem == stem,
                                onPlay: { viewModel.playStem(stem) },
                                onStop: { viewModel.stopPlayback() }
                            )
                        }
                    }
                    .padding()
                }

                Spacer()

                // Waveform visualization placeholder
                if viewModel.isSeparated {
                    WaveformView(activeStem: viewModel.playingStem)
                        .frame(height: 80)
                        .padding()
                }
            }
            .navigationTitle("Demucs Separator")
            .sheet(isPresented: $viewModel.showFilePicker) {
                AudioFilePickerView(audioURL: $viewModel.audioURL)
            }
        }
    }

    private func formatDuration(_ duration: TimeInterval) -> String {
        let minutes = Int(duration) / 60
        let seconds = Int(duration) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }
}

// MARK: - Stem Player Row

struct StemPlayerView: View {
    let stem: Stem
    let isPlaying: Bool
    let onPlay: () -> Void
    let onStop: () -> Void

    var body: some View {
        HStack(spacing: 16) {
            Image(systemName: stem.icon)
                .font(.title3)
                .foregroundColor(stem.color)
                .frame(width: 30)

            Text(stem.rawValue)
                .font(.body)
                .fontWeight(.medium)

            Spacer()

            // Volume indicator
            HStack(spacing: 2) {
                ForEach(0..<5) { i in
                    RoundedRectangle(cornerRadius: 1)
                        .fill(isPlaying ? stem.color : Color(.systemGray4))
                        .frame(width: 3, height: CGFloat(8 + i * 4))
                }
            }

            Button(action: {
                if isPlaying {
                    onStop()
                } else {
                    onPlay()
                }
            }) {
                Image(systemName: isPlaying ? "stop.circle.fill" : "play.circle.fill")
                    .font(.title)
                    .foregroundColor(isPlaying ? .red : stem.color)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(isPlaying ? stem.color.opacity(0.1) : Color(.systemGray6))
        )
    }
}

// MARK: - Animated Waveform

struct WaveformView: View {
    let activeStem: Stem?
    @State private var phase: CGFloat = 0

    var body: some View {
        TimelineView(.animation) { timeline in
            Canvas { context, size in
                let color = activeStem?.color ?? .gray
                let midY = size.height / 2
                let amplitude = activeStem != nil ? size.height * 0.35 : size.height * 0.1
                let time = timeline.date.timeIntervalSinceReferenceDate

                var path = Path()
                path.move(to: CGPoint(x: 0, y: midY))
                for x in stride(from: 0, through: size.width, by: 2) {
                    let normalizedX = x / size.width
                    let y = midY + sin(normalizedX * .pi * 6 + time * 3) * amplitude *
                        (0.5 + 0.5 * sin(normalizedX * .pi * 2 + time * 1.5))
                    path.addLine(to: CGPoint(x: x, y: y))
                }

                context.stroke(path, with: .color(color.opacity(0.7)), lineWidth: 2)
            }
        }
    }
}

// MARK: - Audio File Picker

struct AudioFilePickerView: UIViewControllerRepresentable {
    @Binding var audioURL: URL?
    @Environment(\.dismiss) private var dismiss

    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        let types: [UTType] = [.audio, .mp3, .wav, .aiff, UTType("public.mpeg-4-audio") ?? .audio]
        let picker = UIDocumentPickerViewController(forOpeningContentTypes: types)
        picker.delegate = context.coordinator
        picker.allowsMultipleSelection = false
        return picker
    }

    func updateUIViewController(_ uiViewController: UIDocumentPickerViewController, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    class Coordinator: NSObject, UIDocumentPickerDelegate {
        let parent: AudioFilePickerView

        init(_ parent: AudioFilePickerView) {
            self.parent = parent
        }

        func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
            parent.audioURL = urls.first
            parent.dismiss()
        }

        func documentPickerWasCancelled(_ controller: UIDocumentPickerViewController) {
            parent.dismiss()
        }
    }
}

// MARK: - ViewModel

class DemucsViewModel: ObservableObject {
    @Published var audioURL: URL? {
        didSet { updateAudioInfo() }
    }
    @Published var audioFileName: String?
    @Published var audioDuration: TimeInterval?
    @Published var showFilePicker = false
    @Published var isProcessing = false
    @Published var isSeparated = false
    @Published var progress: Double = 0
    @Published var statusMessage = ""
    @Published var errorMessage: String?
    @Published var playingStem: Stem?

    private var audioPlayer: AVAudioPlayer?

    private func updateAudioInfo() {
        guard let url = audioURL else {
            audioFileName = nil
            audioDuration = nil
            isSeparated = false
            return
        }

        _ = url.startAccessingSecurityScopedResource()
        audioFileName = url.lastPathComponent
        isSeparated = false

        let asset = AVURLAsset(url: url)
        Task {
            let duration = try? await asset.load(.duration)
            await MainActor.run {
                self.audioDuration = duration?.seconds
            }
        }
    }

    func separate() {
        guard audioURL != nil else { return }
        isProcessing = true
        errorMessage = nil
        progress = 0

        Task {
            do {
                try await performSeparation()
                await MainActor.run {
                    self.isSeparated = true
                    self.isProcessing = false
                }
            } catch {
                await MainActor.run {
                    self.errorMessage = error.localizedDescription
                    self.isProcessing = false
                }
            }
        }
    }

    // Perform source separation using HTDemucs CoreML model
    // NOTE: Full implementation requires:
    // 1. Load audio waveform (stereo, ~344k samples at 44.1kHz ~ 7.8s segment)
    // 2. Compute STFT to get freq_input (1,8,2049,336) - 8 channels = real+imag for 4 encoder inputs
    // 3. Provide time_input (1,2,343980) - raw stereo waveform
    // 4. Run model inference
    // 5. Apply iSTFT on frequency outputs + combine with time outputs for each of 4 stems
    // 6. Overlap-add for segments longer than ~7.8s
    private func performSeparation() async throws {
        await updateStatus("Loading model...", progress: 0.1)

        // Check for model
        guard let modelURL = Bundle.main.url(forResource: "HTDemucs_SourceSeparation", withExtension: "mlmodelc") else {
            throw DemucsError.modelNotFound(
                "HTDemucs_SourceSeparation.mlmodelc not found in bundle. " +
                "Please compile and add the HTDemucs_SourceSeparation.mlpackage to the project."
            )
        }

        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        let model = try MLModel(contentsOf: modelURL, configuration: config)

        await updateStatus("Computing STFT...", progress: 0.3)

        // STFT placeholder: In production, use Accelerate's vDSP to compute
        // the Short-Time Fourier Transform of the input audio.
        // Window size = 4096, hop = 1024, producing 2049 frequency bins x 336 time frames
        // The 8 channels represent real and imaginary parts for the hybrid architecture.

        let freqInput = try MLMultiArray(shape: [1, 8, 2049, 336], dataType: .float32)
        let timeInput = try MLMultiArray(shape: [1, 2, 343980], dataType: .float32)

        // Fill with placeholder data (in production: actual STFT values and waveform)
        // ...zero-initialized by default

        await updateStatus("Running inference...", progress: 0.5)

        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
            "freq_input": MLFeatureValue(multiArray: freqInput),
            "time_input": MLFeatureValue(multiArray: timeInput)
        ])

        let _ = try model.prediction(from: inputFeatures)

        await updateStatus("Reconstructing stems (iSTFT)...", progress: 0.8)

        // iSTFT placeholder: In production, apply inverse STFT on each stem's
        // frequency output and combine with time-domain output.
        // Each stem produces separate freq and time outputs that are summed.
        // Use overlap-add for audio longer than one segment (~7.8s at 44.1kHz).

        await updateStatus("Complete!", progress: 1.0)
    }

    @MainActor
    private func updateStatus(_ message: String, progress: Double) {
        self.statusMessage = message
        self.progress = progress
    }

    func playStem(_ stem: Stem) {
        // In production, play the separated stem audio buffer
        // For demo, we play the original audio or show the concept
        stopPlayback()
        playingStem = stem

        guard let url = audioURL else { return }
        _ = url.startAccessingSecurityScopedResource()
        do {
            try AVAudioSession.sharedInstance().setCategory(.playback)
            try AVAudioSession.sharedInstance().setActive(true)
            audioPlayer = try AVAudioPlayer(contentsOf: url)
            audioPlayer?.play()
        } catch {
            errorMessage = "Playback error: \(error.localizedDescription)"
        }
    }

    func stopPlayback() {
        audioPlayer?.stop()
        audioPlayer = nil
        playingStem = nil
    }
}

enum DemucsError: LocalizedError {
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
