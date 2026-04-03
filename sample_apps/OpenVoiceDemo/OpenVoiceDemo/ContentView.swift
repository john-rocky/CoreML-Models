import SwiftUI
import AVFoundation
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var converter = VoiceConverter()
    @StateObject private var sourceRecorder = AudioRecorder()
    @StateObject private var targetRecorder = AudioRecorder()
    @State private var status = ""
    @State private var convertedURL: URL?
    @State private var isConverting = false
    @State private var player: AVAudioPlayer?
    @State private var showTargetImport = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 28) {
                    // Model status
                    HStack {
                        Circle().fill(converter.isReady ? .green : .red).frame(width: 10, height: 10)
                        Text(converter.isReady ? "Models loaded" : converter.status)
                            .font(.caption).foregroundColor(.secondary)
                        Spacer()
                    }
                    .padding(.horizontal)

                    // Source voice
                    RecordSection(
                        title: "Your Voice",
                        subtitle: "Record what you want to say",
                        icon: "person.wave.2",
                        color: .blue,
                        recorder: sourceRecorder,
                        enabled: converter.isReady,
                        onPlay: { if let u = sourceRecorder.recordedURL { playAudio(url: u) } }
                    )

                    // Target voice
                    VStack(spacing: 12) {
                        HStack {
                            Image(systemName: "person.2.wave.2")
                                .foregroundColor(.purple)
                            Text("Target Voice").font(.headline)
                            Spacer()
                        }
                        .padding(.horizontal)

                        Text("Record or import a sample of the voice you want to sound like")
                            .font(.caption).foregroundColor(.secondary)
                            .padding(.horizontal)

                        HStack(spacing: 16) {
                            // Record target
                            Button {
                                if targetRecorder.isRecording {
                                    targetRecorder.stop()
                                } else {
                                    targetRecorder.start()
                                }
                            } label: {
                                VStack(spacing: 6) {
                                    ZStack {
                                        Circle()
                                            .fill(targetRecorder.isRecording ? .red : .purple)
                                            .frame(width: 60, height: 60)
                                        Image(systemName: targetRecorder.isRecording ? "stop.fill" : "mic.fill")
                                            .font(.title3).foregroundColor(.white)
                                    }
                                    Text(targetRecorder.isRecording ? "Stop" : "Record")
                                        .font(.caption2)
                                }
                            }
                            .disabled(!converter.isReady)

                            // Import target
                            Button {
                                showTargetImport = true
                            } label: {
                                VStack(spacing: 6) {
                                    ZStack {
                                        Circle().fill(.purple.opacity(0.2)).frame(width: 60, height: 60)
                                        Image(systemName: "doc.badge.plus")
                                            .font(.title3).foregroundColor(.purple)
                                    }
                                    Text("Import").font(.caption2)
                                }
                            }

                            // Play target
                            if targetRecorder.recordedURL != nil {
                                Button {
                                    playAudio(url: targetRecorder.recordedURL!)
                                } label: {
                                    VStack(spacing: 6) {
                                        ZStack {
                                            Circle().fill(.purple.opacity(0.2)).frame(width: 60, height: 60)
                                            Image(systemName: "play.fill")
                                                .font(.title3).foregroundColor(.purple)
                                        }
                                        Text("Play").font(.caption2)
                                    }
                                }
                            }
                        }

                        if let dur = targetRecorder.duration {
                            Text(String(format: "Target: %.1f sec", dur))
                                .font(.caption2).foregroundColor(.secondary)
                        }
                    }
                    .padding()
                    .background(RoundedRectangle(cornerRadius: 16).fill(.purple.opacity(0.05)))
                    .padding(.horizontal)

                    // Convert button
                    Button {
                        convert()
                    } label: {
                        HStack {
                            if isConverting {
                                ProgressView().controlSize(.small).tint(.white)
                            }
                            Image(systemName: "waveform.path.ecg")
                            Text(isConverting ? "Converting..." : "Convert Voice")
                                .bold()
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 8)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.orange)
                    .disabled(sourceRecorder.recordedURL == nil ||
                              targetRecorder.recordedURL == nil ||
                              isConverting || !converter.isReady)
                    .padding(.horizontal)

                    // Status
                    if !status.isEmpty {
                        Text(status)
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)
                    }

                    // Play converted
                    if let url = convertedURL {
                        Button {
                            playAudio(url: url)
                        } label: {
                            Label("Play Converted Voice", systemImage: "play.circle.fill")
                                .font(.title3).bold()
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 8)
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(.green)
                        .padding(.horizontal)

                        // Share
                        ShareLink(item: url) {
                            Label("Share", systemImage: "square.and.arrow.up")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                        .padding(.horizontal)
                    }
                }
                .padding(.vertical)
            }
            .navigationTitle("Voice Conversion")
            .fileImporter(isPresented: $showTargetImport,
                          allowedContentTypes: [.audio, .wav, .mp3, .mpeg4Audio]) { result in
                if case .success(let url) = result {
                    targetRecorder.setImported(url: url)
                }
            }
        }
    }

    private func convert() {
        guard let sourceURL = sourceRecorder.recordedURL,
              let targetURL = targetRecorder.recordedURL else { return }
        isConverting = true
        convertedURL = nil

        Task {
            do {
                await MainActor.run { status = "Extracting your voice profile..." }
                let srcSE = try await converter.extractSpeakerEmbedding(audioURL: sourceURL)

                await MainActor.run { status = "Extracting target voice profile..." }
                let tgtSE = try await converter.extractSpeakerEmbedding(audioURL: targetURL)

                let result = try await converter.convert(
                    sourceURL: sourceURL,
                    sourceSE: srcSE,
                    targetSE: tgtSE
                ) { msg in
                    DispatchQueue.main.async { status = msg }
                }

                await MainActor.run {
                    convertedURL = result
                    status = "Done!"
                    isConverting = false
                }
            } catch {
                await MainActor.run {
                    status = "Error: \(error.localizedDescription)"
                    isConverting = false
                }
            }
        }
    }

    private func playAudio(url: URL) {
        do {
            try AVAudioSession.sharedInstance().setCategory(.playback)
            try AVAudioSession.sharedInstance().setActive(true)
            player = try AVAudioPlayer(contentsOf: url)
            player?.play()
        } catch {
            status = "Playback error: \(error.localizedDescription)"
        }
    }
}

// MARK: - Record Section

struct RecordSection: View {
    let title: String
    let subtitle: String
    let icon: String
    let color: Color
    let recorder: AudioRecorder
    let enabled: Bool
    let onPlay: () -> Void

    var body: some View {
        VStack(spacing: 12) {
            HStack {
                Image(systemName: icon).foregroundColor(color)
                Text(title).font(.headline)
                Spacer()
            }
            .padding(.horizontal)

            Text(subtitle)
                .font(.caption).foregroundColor(.secondary)
                .padding(.horizontal)

            HStack(spacing: 16) {
                Button {
                    if recorder.isRecording {
                        recorder.stop()
                    } else {
                        recorder.start()
                    }
                } label: {
                    VStack(spacing: 6) {
                        ZStack {
                            Circle()
                                .fill(recorder.isRecording ? .red : color)
                                .frame(width: 60, height: 60)
                            Image(systemName: recorder.isRecording ? "stop.fill" : "mic.fill")
                                .font(.title3).foregroundColor(.white)
                        }
                        Text(recorder.isRecording ? "Stop" : "Record").font(.caption2)
                    }
                }
                .disabled(!enabled)

                if recorder.recordedURL != nil {
                    Button(action: onPlay) {
                        VStack(spacing: 6) {
                            ZStack {
                                Circle().fill(color.opacity(0.2)).frame(width: 60, height: 60)
                                Image(systemName: "play.fill")
                                    .font(.title3).foregroundColor(color)
                            }
                            Text("Play").font(.caption2)
                        }
                    }
                }
            }

            if let dur = recorder.duration {
                Text(String(format: "%.1f sec", dur))
                    .font(.caption2).foregroundColor(.secondary)
            }
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 16).fill(color.opacity(0.05)))
        .padding(.horizontal)
    }
}

// MARK: - Audio Recorder

class AudioRecorder: ObservableObject {
    @Published var isRecording = false
    @Published var recordedURL: URL?
    @Published var duration: Double?

    private var audioRecorder: AVAudioRecorder?

    func start() {
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.record, mode: .default)
            try session.setActive(true)
        } catch { return }

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("rec_\(UUID().uuidString).wav")

        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: 22050.0,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
        ]

        guard let rec = try? AVAudioRecorder(url: url, settings: settings) else { return }
        audioRecorder = rec
        rec.record()
        isRecording = true
        recordedURL = nil
        duration = nil
    }

    func stop() {
        guard let rec = audioRecorder else { return }
        rec.stop()
        isRecording = false
        recordedURL = rec.url
        if let file = try? AVAudioFile(forReading: rec.url) {
            duration = Double(file.length) / file.fileFormat.sampleRate
        }
    }

    func setImported(url: URL) {
        // Copy to temp so we have read access
        let dest = FileManager.default.temporaryDirectory
            .appendingPathComponent("import_\(UUID().uuidString).\(url.pathExtension)")
        guard url.startAccessingSecurityScopedResource() else { return }
        defer { url.stopAccessingSecurityScopedResource() }
        try? FileManager.default.copyItem(at: url, to: dest)
        recordedURL = dest
        if let file = try? AVAudioFile(forReading: dest) {
            duration = Double(file.length) / file.fileFormat.sampleRate
        }
    }
}
