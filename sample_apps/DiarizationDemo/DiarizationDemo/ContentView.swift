import SwiftUI
import AVFoundation
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var diarizer = SpeakerDiarizer()
    @StateObject private var recorder = AudioRecorder()
    @State private var segments: [SpeakerSegment] = []
    @State private var isProcessing = false
    @State private var status = ""
    @State private var showImport = false
    @State private var audioDuration: Double = 0
    @State private var player: AVAudioPlayer?
    @State private var audioURL: URL?
    @State private var speakerTracks: [Int: URL] = [:]
    @State private var isTranscribing = false

    private let speakerColors: [Color] = [.blue, .orange, .green]
    private let speakerNames = ["Speaker A", "Speaker B", "Speaker C"]

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Status
                HStack {
                    Circle().fill(diarizer.isReady ? .green : .red).frame(width: 8, height: 8)
                    Text(diarizer.isReady ? "Ready" : "Loading...").font(.caption).foregroundColor(.secondary)
                    Spacer()
                }
                .padding(.horizontal)
                .padding(.vertical, 4)

                if segments.isEmpty && !isProcessing {
                    // Input controls
                    Spacer()

                    VStack(spacing: 20) {
                        // Record
                        Button {
                            if recorder.isRecording {
                                recorder.stop()
                                audioURL = recorder.recordedURL
                                if let url = audioURL {
                                    audioDuration = getAudioDuration(url: url)
                                }
                            } else {
                                segments = []
                                recorder.start()
                            }
                        } label: {
                            ZStack {
                                Circle()
                                    .fill(recorder.isRecording ? .red : .blue)
                                    .frame(width: 80, height: 80)
                                Image(systemName: recorder.isRecording ? "stop.fill" : "mic.fill")
                                    .font(.title).foregroundColor(.white)
                            }
                        }
                        .disabled(!diarizer.isReady)

                        Text(recorder.isRecording ? "Recording..." : "Record a conversation")
                            .font(.caption).foregroundColor(.secondary)

                        // Import
                        Button {
                            showImport = true
                        } label: {
                            Label("Import Audio File", systemImage: "doc.badge.plus")
                        }
                        .buttonStyle(.bordered)

                        // Analyze button
                        if audioURL != nil {
                            Text(String(format: "%.1f sec", audioDuration))
                                .font(.caption).foregroundColor(.secondary)

                            HStack(spacing: 12) {
                                Button {
                                    playAudio(url: audioURL!)
                                } label: {
                                    Label("Play", systemImage: "play.circle")
                                }
                                .buttonStyle(.bordered)

                                Button {
                                    analyze()
                                } label: {
                                    Label("Analyze Speakers", systemImage: "person.2.wave.2")
                                }
                                .buttonStyle(.borderedProminent)
                            }
                        }
                    }

                    Spacer()
                } else if isProcessing {
                    Spacer()
                    ProgressView(status)
                    Spacer()
                } else {
                    // Results
                    TimelineView(segments: segments, duration: audioDuration,
                                 colors: speakerColors, names: speakerNames)
                        .frame(height: 120)
                        .padding()

                    // Show conversation view if transcribed, otherwise segment list
                    if segments.contains(where: { $0.transcript != nil }) {
                        ConversationView(segments: segments, colors: speakerColors, names: speakerNames)
                    } else {
                        List {
                            ForEach(segments) { seg in
                                HStack(spacing: 8) {
                                    Circle().fill(speakerColors[seg.speaker]).frame(width: 12, height: 12)
                                    Text(speakerNames[seg.speaker]).font(.body).bold()
                                    Spacer()
                                    Text(formatTime(seg.startTime) + " - " + formatTime(seg.endTime))
                                        .font(.caption).foregroundColor(.secondary)
                                    Text(String(format: "%.1fs", seg.duration))
                                        .font(.caption2).foregroundColor(.secondary)
                                }
                            }
                        }
                    }

                    // Action buttons
                    HStack(spacing: 12) {
                        Button {
                            if let url = audioURL { playAudio(url: url) }
                        } label: {
                            Image(systemName: "play.circle")
                        }
                        .buttonStyle(.bordered)

                        Button {
                            transcribeSegments()
                        } label: {
                            HStack(spacing: 4) {
                                if isTranscribing { ProgressView().controlSize(.mini) }
                                Image(systemName: "text.bubble")
                                Text("Transcribe")
                            }
                        }
                        .buttonStyle(.bordered)
                        .disabled(isTranscribing)

                        Button {
                            exportTracks()
                        } label: {
                            HStack(spacing: 4) {
                                Image(systemName: "square.and.arrow.up")
                                Text("Tracks")
                            }
                        }
                        .buttonStyle(.bordered)

                        Spacer()

                        // Speaker summary
                        ForEach(0..<3, id: \.self) { spk in
                            let total = segments.filter { $0.speaker == spk }.reduce(0.0) { $0 + $1.duration }
                            if total > 0 {
                                HStack(spacing: 2) {
                                    Circle().fill(speakerColors[spk]).frame(width: 6, height: 6)
                                    Text(String(format: "%.0fs", total)).font(.caption2)
                                }
                            }
                        }
                    }
                    .padding(.horizontal)
                    .padding(.vertical, 8)

                    // Speaker track playback
                    if !speakerTracks.isEmpty {
                        HStack(spacing: 12) {
                            ForEach(Array(speakerTracks.keys.sorted()), id: \.self) { spk in
                                Button {
                                    if let url = speakerTracks[spk] { playAudio(url: url) }
                                } label: {
                                    HStack(spacing: 4) {
                                        Circle().fill(speakerColors[spk]).frame(width: 8, height: 8)
                                        Text("Play \(speakerNames[spk])")
                                    }
                                    .font(.caption)
                                }
                                .buttonStyle(.borderedProminent)
                                .tint(speakerColors[spk])
                            }
                        }
                        .padding(.horizontal)
                    }

                    HStack {
                        Spacer()
                        Button("Clear") {
                            segments = []
                            audioURL = nil
                            speakerTracks = [:]
                        }
                        .foregroundColor(.red)
                    }
                    .padding(.horizontal).padding(.bottom, 4)
                }

                if !status.isEmpty && !isProcessing {
                    Text(status).font(.caption).foregroundColor(.secondary).padding(.horizontal)
                }
            }
            .navigationTitle("Speaker Diarization")
            .fileImporter(isPresented: $showImport,
                          allowedContentTypes: [.audio, .wav, .mp3, .mpeg4Audio]) { result in
                if case .success(let url) = result {
                    let dest = FileManager.default.temporaryDirectory
                        .appendingPathComponent("import_\(UUID().uuidString).\(url.pathExtension)")
                    guard url.startAccessingSecurityScopedResource() else { return }
                    defer { url.stopAccessingSecurityScopedResource() }
                    try? FileManager.default.copyItem(at: url, to: dest)
                    audioURL = dest
                    audioDuration = getAudioDuration(url: dest)
                }
            }
        }
    }

    private func analyze() {
        guard let url = audioURL else { return }
        isProcessing = true
        segments = []
        Task {
            do {
                let result = try await diarizer.diarize(audioURL: url) { msg in
                    DispatchQueue.main.async { status = msg }
                }
                await MainActor.run {
                    segments = result
                    status = "\(segments.count) segments found"
                    isProcessing = false
                }
            } catch {
                await MainActor.run {
                    status = "Error: \(error.localizedDescription)"
                    isProcessing = false
                }
            }
        }
    }

    private func transcribeSegments() {
        guard let url = audioURL else { return }
        isTranscribing = true
        let segs = segments
        Task {
            do {
                let transcribed = try await diarizer.transcribe(audioURL: url, segments: segs) { msg in
                    DispatchQueue.main.async { self.status = msg }
                }
                await MainActor.run {
                    segments = transcribed
                    status = "Transcription complete"
                    isTranscribing = false
                }
            } catch {
                await MainActor.run {
                    status = "Transcription error: \(error.localizedDescription)"
                    isTranscribing = false
                }
            }
        }
    }

    private func exportTracks() {
        guard let url = audioURL else { return }
        do {
            speakerTracks = try diarizer.exportSpeakerTracks(audioURL: url, segments: segments)
            status = "\(speakerTracks.count) tracks exported"
        } catch {
            status = "Export error: \(error.localizedDescription)"
        }
    }

    private func playAudio(url: URL) {
        do {
            try AVAudioSession.sharedInstance().setCategory(.playback)
            try AVAudioSession.sharedInstance().setActive(true)
            player = try AVAudioPlayer(contentsOf: url)
            player?.play()
        } catch {
            status = "Playback error"
        }
    }

    private func getAudioDuration(url: URL) -> Double {
        guard let file = try? AVAudioFile(forReading: url) else { return 0 }
        return Double(file.length) / file.fileFormat.sampleRate
    }

    private func formatTime(_ t: Double) -> String {
        let m = Int(t) / 60
        let s = Int(t) % 60
        let ms = Int((t - Double(Int(t))) * 10)
        return String(format: "%d:%02d.%d", m, s, ms)
    }
}

// MARK: - Timeline Visualization

struct TimelineView: View {
    let segments: [SpeakerSegment]
    let duration: Double
    let colors: [Color]
    let names: [String]

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let trackH: CGFloat = 24
            let gap: CGFloat = 4

            VStack(alignment: .leading, spacing: gap) {
                ForEach(0..<3, id: \.self) { spk in
                    let spkSegs = segments.filter { $0.speaker == spk }
                    if !spkSegs.isEmpty {
                        ZStack(alignment: .leading) {
                            RoundedRectangle(cornerRadius: 4)
                                .fill(Color.gray.opacity(0.1))
                                .frame(height: trackH)

                            ForEach(spkSegs) { seg in
                                let x = duration > 0 ? CGFloat(seg.startTime / duration) * w : 0
                                let segW = duration > 0 ? CGFloat(seg.duration / duration) * w : 0
                                RoundedRectangle(cornerRadius: 3)
                                    .fill(colors[spk])
                                    .frame(width: max(2, segW), height: trackH - 4)
                                    .offset(x: x)
                            }

                            Text(names[spk])
                                .font(.system(size: 10, weight: .bold))
                                .foregroundColor(.white)
                                .padding(.leading, 4)
                        }
                    }
                }

                // Time axis
                HStack {
                    Text("0:00").font(.system(size: 9))
                    Spacer()
                    Text(formatDuration(duration)).font(.system(size: 9))
                }
                .foregroundColor(.secondary)
            }
        }
    }

    private func formatDuration(_ t: Double) -> String {
        let m = Int(t) / 60
        let s = Int(t) % 60
        return String(format: "%d:%02d", m, s)
    }
}

// MARK: - Conversation View

struct ConversationView: View {
    let segments: [SpeakerSegment]
    let colors: [Color]
    let names: [String]

    // Merge adjacent segments from same speaker
    private var mergedSegments: [(speaker: Int, startTime: Double, endTime: Double, text: String)] {
        var result: [(speaker: Int, startTime: Double, endTime: Double, text: String)] = []
        for seg in segments {
            let text = seg.transcript ?? ""
            if let last = result.last, last.speaker == seg.speaker {
                // Merge with previous
                let merged = (speaker: last.speaker, startTime: last.startTime,
                              endTime: seg.endTime,
                              text: last.text + (text.isEmpty ? "" : " " + text))
                result[result.count - 1] = merged
            } else {
                result.append((speaker: seg.speaker, startTime: seg.startTime,
                                endTime: seg.endTime, text: text))
            }
        }
        return result
    }

    var body: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 12) {
                ForEach(Array(mergedSegments.enumerated()), id: \.offset) { _, seg in
                    HStack(alignment: .top, spacing: 10) {
                        Circle()
                            .fill(colors[seg.speaker])
                            .frame(width: 32, height: 32)
                            .overlay(
                                Text(String(names[seg.speaker].last ?? "?"))
                                    .font(.system(size: 14, weight: .bold))
                                    .foregroundColor(.white)
                            )

                        VStack(alignment: .leading, spacing: 2) {
                            HStack {
                                Text(names[seg.speaker])
                                    .font(.caption).bold()
                                    .foregroundColor(colors[seg.speaker])
                                Text(formatTime(seg.startTime))
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                            }
                            if !seg.text.isEmpty {
                                Text(seg.text)
                                    .font(.body)
                            } else {
                                Text("(no transcript)")
                                    .font(.body)
                                    .foregroundColor(.secondary)
                                    .italic()
                            }
                        }
                        Spacer()
                    }
                    .padding(.horizontal)
                }
            }
            .padding(.vertical, 8)
        }
    }

    private func formatTime(_ t: Double) -> String {
        let m = Int(t) / 60
        let s = Int(t) % 60
        return String(format: "%d:%02d", m, s)
    }
}

// MARK: - Audio Recorder

class AudioRecorder: ObservableObject {
    @Published var isRecording = false
    @Published var recordedURL: URL?
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
            AVSampleRateKey: 16000.0,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
        ]

        guard let rec = try? AVAudioRecorder(url: url, settings: settings) else { return }
        audioRecorder = rec
        rec.record()
        isRecording = true
        recordedURL = nil
    }

    func stop() {
        guard let rec = audioRecorder else { return }
        rec.stop()
        isRecording = false
        recordedURL = rec.url
    }
}
