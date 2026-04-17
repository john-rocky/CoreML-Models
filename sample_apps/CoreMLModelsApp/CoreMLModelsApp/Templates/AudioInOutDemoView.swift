import SwiftUI
import CoreML
import AVFoundation
import Accelerate

/// Audio processing template. Routes by manifest config:
///   HTDemucs — source separation (4 stems)
///   OpenVoice — voice conversion (STFT + speaker encoder + converter)
///   Diarization — speaker segmentation (powerset class decoding)
struct AudioInOutDemoView: View {
    let model: ModelEntry

    @State private var inputURL: URL?
    @State private var outputURLs: [(String, URL)] = []
    @State private var speakerTimeline: [(Int, Double, Double)] = []  // (speaker, start, end)
    @State private var isProcessing = false
    @State private var status = ""
    @State private var progress: Float = 0
    @State private var processingTime: Double?
    @State private var showingFilePicker = false
    @State private var currentlyPlaying: String?
    @State private var player: AVAudioPlayer?

    // Diarization-specific
    @StateObject private var diarizationRecorder = AudioRecorder(sampleRate: 16000)
    @State private var audioDuration: Double = 0

    // OpenVoice-specific
    @StateObject private var sourceRecorder = AudioRecorder(sampleRate: 22050)
    @StateObject private var targetRecorder = AudioRecorder(sampleRate: 22050)
    @State private var showTargetImport = false
    @State private var convertedURL: URL?

    @StateObject private var session = ModelSession<Void>()

    private var stemNames: [String] { model.configStringArray("output_stems") ?? ["output"] }

    private var mode: AudioMode {
        let stems = stemNames
        if stems.contains("speaker_timeline") { return .diarization }
        if stems.contains("converted") { return .voiceConversion }
        return .sourceSeparation
    }

    enum AudioMode {
        case sourceSeparation
        case diarization
        case voiceConversion
    }

    var body: some View {
        Group {
            switch mode {
            case .sourceSeparation:
                sourceSeparationBody
            case .diarization:
                diarizationBody
            case .voiceConversion:
                voiceConversionBody
            }
        }
        .task {
            session.ensure { try await eagerPreload() }
        }
    }

    /// Pre-compile the main inference model(s) for the current mode while the
    /// user is picking / recording material. Subsequent `ModelLoader.load(...)`
    /// calls in the run path hit the mlmodelc cache.
    private func eagerPreload() async throws {
        switch mode {
        case .sourceSeparation:
            _ = try await ModelLoader.loadPrimary(for: model)
        case .diarization:
            _ = try await ModelLoader.loadPrimary(for: model)
        case .voiceConversion:
            for file in model.files where (file.kind ?? "model") == "model" {
                _ = try await ModelLoader.load(for: model, named: file.name)
            }
        }
    }

    // MARK: - HTDemucs Source Separation UI

    private var sourceSeparationBody: some View {
        VStack(spacing: 0) {
            if let url = inputURL {
                HStack {
                    Image(systemName: "music.note")
                        .font(.title2)
                        .foregroundColor(.accentColor)
                    VStack(alignment: .leading) {
                        Text(url.lastPathComponent)
                            .font(.headline)
                            .lineLimit(1)
                        if audioDuration > 0 {
                            Text(formatDuration(audioDuration))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                    Spacer()
                    Button { playAudio(url: url, label: "input") } label: {
                        Image(systemName: currentlyPlaying == "input" ? "stop.circle.fill" : "play.circle.fill")
                            .font(.title2)
                    }
                    Button("Change") { showingFilePicker = true }
                        .font(.caption)
                }
                .padding()
                .background(Color(.systemGray6))
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .padding(.horizontal)
            }

            if !outputURLs.isEmpty {
                // Stem player list
                ScrollView {
                    VStack(spacing: 12) {
                        ForEach(outputURLs, id: \.0) { (stem, url) in
                            HStack(spacing: 12) {
                                Image(systemName: stemIcon(stem))
                                    .font(.title3)
                                    .foregroundColor(stemColor(stem))
                                    .frame(width: 30)

                                Text(stem.capitalized)
                                    .font(.body)
                                    .fontWeight(.medium)

                                Spacer()

                                ShareLink(item: url) {
                                    Image(systemName: "square.and.arrow.up")
                                        .font(.body)
                                        .foregroundColor(.secondary)
                                }

                                Button {
                                    playAudio(url: url, label: stem)
                                } label: {
                                    Image(systemName: currentlyPlaying == stem ? "stop.circle.fill" : "play.circle.fill")
                                        .font(.title)
                                        .foregroundColor(currentlyPlaying == stem ? .red : stemColor(stem))
                                }
                            }
                            .padding()
                            .background(
                                RoundedRectangle(cornerRadius: 12)
                                    .fill(currentlyPlaying == stem ? stemColor(stem).opacity(0.1) : Color(.systemGray6))
                            )
                        }
                    }
                    .padding()
                }
            } else {
                VStack(spacing: 12) {
                    Image(systemName: "waveform.path.ecg")
                        .font(.system(size: 60))
                        .foregroundStyle(.secondary)
                    Text("Import an audio file to separate stems")
                        .foregroundStyle(.secondary)
                    Text("WAV, MP3, M4A")
                        .font(.caption2)
                        .foregroundStyle(.secondary.opacity(0.7))
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }

            VStack(spacing: 12) {
                if isProcessing {
                    ProgressView(value: Double(progress))
                        .progressViewStyle(.linear)
                    Text(status)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                TimingsLabel(loadSec: session.loadTimeSec, inferSec: processingTime)

                if inputURL == nil {
                    Button { showingFilePicker = true } label: {
                        VStack(spacing: 12) {
                            Image(systemName: "square.and.arrow.down")
                                .font(.system(size: 36))
                                .foregroundColor(.secondary)
                            Text("Import Audio File")
                                .foregroundColor(.secondary)
                        }
                        .frame(maxWidth: .infinity)
                        .frame(height: 100)
                        .background(Color(.systemGray6))
                        .cornerRadius(12)
                    }
                } else if outputURLs.isEmpty {
                    HStack(spacing: 10) {
                        Button {
                            Task { await runProcessing(fullTrack: false) }
                        } label: {
                            HStack {
                                if isProcessing {
                                    ProgressView().tint(.white)
                                } else {
                                    Image(systemName: "scissors")
                                }
                                Text(isProcessing ? status : "Preview (~8s)")
                            }
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(isProcessing ? Color.gray : Color.accentColor)
                            .foregroundColor(.white)
                            .cornerRadius(12)
                        }
                        .disabled(isProcessing)

                        Button {
                            Task { await runProcessing(fullTrack: true) }
                        } label: {
                            HStack {
                                if isProcessing {
                                    ProgressView().tint(.white)
                                } else {
                                    Image(systemName: "wand.and.stars")
                                }
                                Text(isProcessing ? status : "Full Track")
                            }
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(isProcessing ? Color.gray : Color(.systemIndigo))
                            .foregroundColor(.white)
                            .cornerRadius(12)
                        }
                        .disabled(isProcessing)
                    }
                }
            }
            .padding()
        }
        .sheet(isPresented: $showingFilePicker) {
            AudioPickerView { url in
                inputURL = url; outputURLs = []; speakerTimeline = []
                audioDuration = getAudioDuration(url: url)
            }
        }
    }

    // MARK: - Diarization UI

    private var diarizationBody: some View {
        VStack(spacing: 0) {
            if speakerTimeline.isEmpty && !isProcessing {
                // Input controls
                Spacer()

                VStack(spacing: 20) {
                    // Record button
                    Button {
                        if diarizationRecorder.isRecording {
                            diarizationRecorder.stop()
                            inputURL = diarizationRecorder.recordedURL
                            if let url = inputURL {
                                audioDuration = getAudioDuration(url: url)
                            }
                        } else {
                            speakerTimeline = []
                            diarizationRecorder.start()
                        }
                    } label: {
                        ZStack {
                            Circle()
                                .fill(diarizationRecorder.isRecording ? .red : .blue)
                                .frame(width: 80, height: 80)
                            Image(systemName: diarizationRecorder.isRecording ? "stop.fill" : "mic.fill")
                                .font(.title)
                                .foregroundColor(.white)
                        }
                    }

                    Text(diarizationRecorder.isRecording ? "Recording..." : "Record a conversation")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    // Import audio
                    Button { showingFilePicker = true } label: {
                        Label("Import Audio File", systemImage: "doc.badge.plus")
                    }
                    .buttonStyle(.bordered)

                    // Show controls when audio is available
                    if inputURL != nil {
                        Text(String(format: "%.1f sec", audioDuration))
                            .font(.caption)
                            .foregroundStyle(.secondary)

                        HStack(spacing: 12) {
                            Button {
                                if let url = inputURL { playAudio(url: url, label: "input") }
                            } label: {
                                Label("Play", systemImage: currentlyPlaying == "input" ? "stop.circle" : "play.circle")
                            }
                            .buttonStyle(.bordered)

                            Button {
                                Task { await runProcessing() }
                            } label: {
                                Label("Analyze Speakers", systemImage: "person.2.wave.2")
                            }
                            .buttonStyle(.borderedProminent)
                            .disabled(isProcessing)
                        }
                    }
                }

                Spacer()
            } else if isProcessing {
                Spacer()
                VStack(spacing: 12) {
                    ProgressView(value: Double(progress))
                        .progressViewStyle(.linear)
                        .padding(.horizontal, 40)
                    Text(status)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            } else {
                // Results: Timeline visualization
                DiarizationTimelineView(
                    segments: speakerTimeline,
                    duration: audioDuration,
                    colors: diarizationColors,
                    names: diarizationNames
                )
                .frame(height: 120)
                .padding()

                // Segment list
                List {
                    ForEach(Array(speakerTimeline.enumerated()), id: \.offset) { _, seg in
                        HStack(spacing: 8) {
                            Circle()
                                .fill(speakerColor(seg.0))
                                .frame(width: 12, height: 12)
                            Text(diarizationNames[seg.0])
                                .font(.body)
                                .bold()
                            Spacer()
                            Text(formatTime(seg.1) + " - " + formatTime(seg.2))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Text(String(format: "%.1fs", seg.2 - seg.1))
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                .listStyle(.plain)

                // Action bar + speaker summary
                HStack(spacing: 12) {
                    Button {
                        if let url = inputURL { playAudio(url: url, label: "input") }
                    } label: {
                        Image(systemName: currentlyPlaying == "input" ? "stop.circle" : "play.circle")
                    }
                    .buttonStyle(.bordered)

                    Spacer()

                    // Speaker time summary
                    ForEach(0..<3, id: \.self) { spk in
                        let total = speakerTimeline.filter { $0.0 == spk }.reduce(0.0) { $0 + ($1.2 - $1.1) }
                        if total > 0 {
                            HStack(spacing: 2) {
                                Circle().fill(speakerColor(spk)).frame(width: 6, height: 6)
                                Text(String(format: "%.0fs", total)).font(.caption2)
                            }
                        }
                    }
                }
                .padding(.horizontal)
                .padding(.vertical, 8)

                HStack {
                    Spacer()
                    Button("Clear") {
                        speakerTimeline = []
                        inputURL = nil
                    }
                    .foregroundColor(.red)
                }
                .padding(.horizontal)
                .padding(.bottom, 4)
            }

            if !status.isEmpty && !isProcessing {
                Text(status)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal)
            }
            TimingsLabel(loadSec: session.loadTimeSec, inferSec: processingTime)
                .padding(.horizontal)
        }
        .sheet(isPresented: $showingFilePicker) {
            AudioPickerView { url in
                inputURL = url; speakerTimeline = []
                audioDuration = getAudioDuration(url: url)
            }
        }
    }

    // MARK: - OpenVoice Voice Conversion UI

    private var voiceConversionBody: some View {
        ScrollView {
            VStack(spacing: 28) {
                // Source voice section
                VStack(spacing: 12) {
                    HStack {
                        Image(systemName: "person.wave.2")
                            .foregroundColor(.blue)
                        Text("Your Voice")
                            .font(.headline)
                        Spacer()
                    }
                    .padding(.horizontal)

                    Text("Record what you want to say")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .padding(.horizontal)

                    HStack(spacing: 16) {
                        // Record source
                        Button {
                            if sourceRecorder.isRecording {
                                sourceRecorder.stop()
                            } else {
                                sourceRecorder.start()
                            }
                        } label: {
                            VStack(spacing: 6) {
                                ZStack {
                                    Circle()
                                        .fill(sourceRecorder.isRecording ? .red : .blue)
                                        .frame(width: 60, height: 60)
                                    Image(systemName: sourceRecorder.isRecording ? "stop.fill" : "mic.fill")
                                        .font(.title3)
                                        .foregroundColor(.white)
                                }
                                Text(sourceRecorder.isRecording ? "Stop" : "Record")
                                    .font(.caption2)
                            }
                        }

                        // Play source
                        if sourceRecorder.recordedURL != nil {
                            Button {
                                if let url = sourceRecorder.recordedURL {
                                    playAudio(url: url, label: "source")
                                }
                            } label: {
                                VStack(spacing: 6) {
                                    ZStack {
                                        Circle()
                                            .fill(.blue.opacity(0.2))
                                            .frame(width: 60, height: 60)
                                        Image(systemName: "play.fill")
                                            .font(.title3)
                                            .foregroundColor(.blue)
                                    }
                                    Text("Play")
                                        .font(.caption2)
                                }
                            }
                        }
                    }

                    if let dur = sourceRecorder.duration {
                        Text(String(format: "%.1f sec", dur))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
                .padding()
                .background(RoundedRectangle(cornerRadius: 16).fill(.blue.opacity(0.05)))
                .padding(.horizontal)

                // Target voice section
                VStack(spacing: 12) {
                    HStack {
                        Image(systemName: "person.2.wave.2")
                            .foregroundColor(.purple)
                        Text("Target Voice")
                            .font(.headline)
                        Spacer()
                    }
                    .padding(.horizontal)

                    Text("Record or import a sample of the voice you want to sound like")
                        .font(.caption)
                        .foregroundStyle(.secondary)
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
                                        .font(.title3)
                                        .foregroundColor(.white)
                                }
                                Text(targetRecorder.isRecording ? "Stop" : "Record")
                                    .font(.caption2)
                            }
                        }

                        // Import target
                        Button { showTargetImport = true } label: {
                            VStack(spacing: 6) {
                                ZStack {
                                    Circle()
                                        .fill(.purple.opacity(0.2))
                                        .frame(width: 60, height: 60)
                                    Image(systemName: "doc.badge.plus")
                                        .font(.title3)
                                        .foregroundColor(.purple)
                                }
                                Text("Import")
                                    .font(.caption2)
                            }
                        }

                        // Play target
                        if targetRecorder.recordedURL != nil {
                            Button {
                                if let url = targetRecorder.recordedURL {
                                    playAudio(url: url, label: "target")
                                }
                            } label: {
                                VStack(spacing: 6) {
                                    ZStack {
                                        Circle()
                                            .fill(.purple.opacity(0.2))
                                            .frame(width: 60, height: 60)
                                        Image(systemName: "play.fill")
                                            .font(.title3)
                                            .foregroundColor(.purple)
                                    }
                                    Text("Play")
                                        .font(.caption2)
                                }
                            }
                        }
                    }

                    if let dur = targetRecorder.duration {
                        Text(String(format: "Target: %.1f sec", dur))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
                .padding()
                .background(RoundedRectangle(cornerRadius: 16).fill(.purple.opacity(0.05)))
                .padding(.horizontal)

                // Convert button
                Button {
                    Task { await runVoiceConversionFlow() }
                } label: {
                    HStack {
                        if isProcessing {
                            ProgressView().controlSize(.small).tint(.white)
                        }
                        Image(systemName: "waveform.path.ecg")
                        Text(isProcessing ? "Converting..." : "Convert Voice")
                            .bold()
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 8)
                }
                .buttonStyle(.borderedProminent)
                .tint(.orange)
                .disabled(sourceRecorder.recordedURL == nil ||
                          targetRecorder.recordedURL == nil ||
                          isProcessing)
                .padding(.horizontal)

                // Status
                if !status.isEmpty {
                    Text(status)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                }

                // Play converted result
                if let url = convertedURL {
                    Button {
                        playAudio(url: url, label: "converted")
                    } label: {
                        Label("Play Converted Voice", systemImage: "play.circle.fill")
                            .font(.title3)
                            .bold()
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 8)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.green)
                    .padding(.horizontal)

                    ShareLink(item: url) {
                        Label("Share", systemImage: "square.and.arrow.up")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .padding(.horizontal)
                }

                TimingsLabel(loadSec: session.loadTimeSec, inferSec: processingTime)
            }
            .padding(.vertical)
        }
        .sheet(isPresented: $showTargetImport) {
            AudioPickerView { url in targetRecorder.setImported(url: url) }
        }
    }

    // MARK: - Helpers

    private let diarizationColors: [Color] = [.blue, .orange, .green]
    private let diarizationNames = ["Speaker A", "Speaker B", "Speaker C"]

    private func stemIcon(_ name: String) -> String {
        switch name.lowercased() {
        case "vocals": return "mic.fill"
        case "drums": return "drum.fill"
        case "bass": return "guitars.fill"
        case "other": return "music.note"
        default: return "waveform"
        }
    }

    private func stemColor(_ name: String) -> Color {
        switch name.lowercased() {
        case "vocals": return .purple
        case "drums": return .orange
        case "bass": return .blue
        case "other": return .green
        default: return .accentColor
        }
    }

    private func speakerColor(_ idx: Int) -> Color {
        [Color.blue, .orange, .green, .purple, .red, .cyan, .pink][idx % 7]
    }

    private func playAudio(url: URL, label: String) {
        if currentlyPlaying == label { player?.stop(); currentlyPlaying = nil; return }
        do {
            try AVAudioSession.sharedInstance().setCategory(.playback)
            try AVAudioSession.sharedInstance().setActive(true)
            player = try AVAudioPlayer(contentsOf: url)
            player?.play()
            currentlyPlaying = label
        } catch {
            status = "Playback error"
        }
    }

    private func getAudioDuration(url: URL) -> Double {
        guard let file = try? AVAudioFile(forReading: url) else { return 0 }
        return Double(file.length) / file.fileFormat.sampleRate
    }

    private func formatDuration(_ duration: Double) -> String {
        let minutes = Int(duration) / 60
        let seconds = Int(duration) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }

    private func formatTime(_ t: Double) -> String {
        let m = Int(t) / 60
        let s = Int(t) % 60
        let ms = Int((t - Double(Int(t))) * 10)
        return String(format: "%d:%02d.%d", m, s, ms)
    }

    // MARK: - Processing Dispatch

    private func runProcessing(fullTrack: Bool = true) async {
        guard let inputURL else { return }
        isProcessing = true; progress = 0; outputURLs = []; speakerTimeline = []

        do {
            let sampleRate = model.configInt("sample_rate") ?? 44100
            let start = CFAbsoluteTimeGetCurrent()

            let stems = stemNames
            if stems.contains("speaker_timeline") {
                try await runDiarization(inputURL: inputURL)
            } else if stems.contains("converted") {
                try await runVoiceConversion(inputURL: inputURL, sampleRate: sampleRate)
            } else {
                try await runSourceSeparation(inputURL: inputURL, sampleRate: sampleRate, fullTrack: fullTrack)
            }

            let elapsed = CFAbsoluteTimeGetCurrent() - start
            await MainActor.run { processingTime = elapsed; isProcessing = false; status = "" }
        } catch {
            await MainActor.run { isProcessing = false; status = "Error: \(error.localizedDescription)" }
        }
    }

    private func runVoiceConversionFlow() async {
        guard let sourceURL = sourceRecorder.recordedURL,
              let targetURL = targetRecorder.recordedURL else { return }
        isProcessing = true; convertedURL = nil

        do {
            let sampleRate = model.configInt("sample_rate") ?? 22050
            let start = CFAbsoluteTimeGetCurrent()

            try await runVoiceConversionWithTarget(sourceURL: sourceURL, targetURL: targetURL, sampleRate: sampleRate)

            let elapsed = CFAbsoluteTimeGetCurrent() - start
            await MainActor.run { processingTime = elapsed; isProcessing = false; status = "Done!" }
        } catch {
            await MainActor.run { isProcessing = false; status = "Error: \(error.localizedDescription)" }
        }
    }

    // MARK: - HTDemucs Source Separation

    private func runSourceSeparation(inputURL: URL, sampleRate: Int, fullTrack: Bool = true) async throws {
        status = "Loading audio..."
        // Resample + channel-match to model's expected format (mirrors DemucsDemo).
        let buffer = try AudioUtils.loadResampled(url: inputURL, targetSampleRate: Double(sampleRate), targetChannels: 2)
        guard let floatData = buffer.floatChannelData else { throw NSError(domain: "Audio", code: 1) }
        let totalSamples = Int(buffer.frameLength)

        status = "Loading model..."
        let mlModel = try await ModelLoader.loadPrimary(for: model)

        status = "Processing..."
        let segmentLength = model.configInt("segment_length") ?? totalSamples
        let channels = min(Int(buffer.format.channelCount), 2)

        // Determine how many samples to process
        let samplesToProcess: Int
        if fullTrack {
            samplesToProcess = totalSamples
        } else {
            // Preview mode: ~8 seconds
            samplesToProcess = min(totalSamples, sampleRate * 8)
        }

        // Build input
        let desc = mlModel.modelDescription.inputDescriptionsByName
        let inputShape: [NSNumber]
        if let firstInput = desc.values.first(where: { $0.type == .multiArray }),
           let constraint = firstInput.multiArrayConstraint {
            inputShape = constraint.shape
        } else {
            inputShape = [1, NSNumber(value: channels), NSNumber(value: segmentLength)]
        }

        let inputArr = try MLMultiArray(shape: inputShape, dataType: .float32)
        let inPtr = inputArr.dataPointer.assumingMemoryBound(to: Float.self)
        let inputLen = min(samplesToProcess, segmentLength)
        for ch in 0..<min(channels, 2) {
            for i in 0..<inputLen { inPtr[ch * segmentLength + i] = floatData[ch][i] }
        }

        let inputName = desc.keys.first ?? "audio"
        let output = try await mlModel.prediction(from: MLDictionaryFeatureProvider(dictionary: [inputName: inputArr]))
        await MainActor.run { progress = 0.8 }

        // Extract output stems
        status = "Saving stems..."
        var results: [(String, URL)] = []
        for (idx, name) in output.featureNames.enumerated() {
            guard let arr = output.featureValue(for: name)?.multiArrayValue else { continue }
            let stemName = idx < stemNames.count ? stemNames[idx] : name
            let outFloats = ImageUtils.extractFloats(arr)
            let outSamples = outFloats.count / max(1, channels)
            let url = try saveWAV(samples: outFloats, sampleRate: sampleRate, channels: channels, samplesPerChannel: outSamples, name: stemName)
            results.append((stemName, url))
        }
        await MainActor.run { outputURLs = results; progress = 1.0 }
    }

    // MARK: - Diarization (Pyannote powerset decoding)

    private func runDiarization(inputURL: URL) async throws {
        status = "Loading audio..."
        // Pyannote expects 16 kHz mono. Use AVAudioConverter so rate/channel mismatches are handled.
        let buffer = try AudioUtils.loadResampled(url: inputURL, targetSampleRate: 16000, targetChannels: 1)
        guard let samples = buffer.floatChannelData?[0] else { throw NSError(domain: "Audio", code: 1) }
        let totalSamples = Int(buffer.frameLength)

        status = "Loading model..."
        let mlModel = try await ModelLoader.loadPrimary(for: model)

        let chunkSamples = 160000  // 10s at 16kHz
        let hopSamples = chunkSamples / 2  // 50% overlap
        let framesPerChunk = 589
        let numClasses = 7
        let maxSpeakers = 3

        // Powerset: class -> speakers
        let classToSpeakers: [[Int]] = [[], [0], [1], [2], [0,1], [0,2], [1,2]]

        let totalFrames = max(1, Int(ceil(Double(totalSamples) / Double(hopSamples))) * (framesPerChunk / 2))
        var speakerProbs = [[Float]](repeating: [Float](repeating: 0, count: maxSpeakers), count: totalFrames)
        var frameCounts = [Int](repeating: 0, count: totalFrames)

        let numChunks = max(1, (totalSamples - chunkSamples + hopSamples) / hopSamples + (totalSamples <= chunkSamples ? 0 : 1))

        for chunk in 0..<max(1, numChunks) {
            status = "Chunk \(chunk + 1)/\(max(1, numChunks))..."
            await MainActor.run { progress = Float(chunk) / Float(max(1, numChunks)) }

            let startSample = chunk * hopSamples
            let inputArr = try MLMultiArray(shape: [1, 1, NSNumber(value: chunkSamples)], dataType: .float32)
            let inPtr = inputArr.dataPointer.assumingMemoryBound(to: Float.self)
            memset(inPtr, 0, chunkSamples * MemoryLayout<Float>.size)
            for i in 0..<chunkSamples {
                let srcIdx = startSample + i
                if srcIdx < totalSamples { inPtr[i] = samples[srcIdx] }
            }

            let output = try await mlModel.prediction(from: MLDictionaryFeatureProvider(dictionary: ["audio": inputArr]))
            guard let logits = output.featureValue(for: "speaker_logits")?.multiArrayValue else { continue }

            // Decode powerset logits
            let globalFrameStart = chunk * (framesPerChunk / 2)
            for f in 0..<framesPerChunk {
                let gf = globalFrameStart + f
                guard gf < totalFrames else { break }

                // exp(logSoftmax) -> probabilities
                var probs = [Float](repeating: 0, count: numClasses)
                for c in 0..<numClasses {
                    probs[c] = exp(ImageUtils.readFloat(logits, at: f * numClasses + c))
                }

                // Accumulate per-speaker probs
                for c in 0..<numClasses {
                    for spk in classToSpeakers[c] {
                        speakerProbs[gf][spk] += probs[c]
                    }
                }
                frameCounts[gf] += 1
            }
        }

        // Normalize by overlap count
        for f in 0..<totalFrames {
            if frameCounts[f] > 1 {
                for s in 0..<maxSpeakers { speakerProbs[f][s] /= Float(frameCounts[f]) }
            }
        }

        // Build segments
        let frameRate = Double(framesPerChunk) / 10.0  // 58.9 fps
        let threshold: Float = 0.5
        let minDuration = 0.3

        var segments: [(Int, Double, Double)] = []
        for spk in 0..<maxSpeakers {
            var segStart: Double?
            for f in 0..<totalFrames {
                let active = speakerProbs[f][spk] > threshold
                let time = Double(f) / frameRate
                if active && segStart == nil { segStart = time }
                if !active, let start = segStart {
                    if time - start >= minDuration { segments.append((spk, start, time)) }
                    segStart = nil
                }
            }
            if let start = segStart {
                let endTime = Double(totalFrames) / frameRate
                if endTime - start >= minDuration { segments.append((spk, start, endTime)) }
            }
        }

        segments.sort { $0.1 < $1.1 }
        await MainActor.run { speakerTimeline = segments; progress = 1.0 }
    }

    // MARK: - OpenVoice Voice Conversion (single source = self-conversion)

    private func runVoiceConversion(inputURL: URL, sampleRate: Int) async throws {
        status = "Loading audio..."
        let rate = 22050
        // OpenVoice expects 22050 Hz mono. Resample if needed.
        let buffer = try AudioUtils.loadResampled(url: inputURL, targetSampleRate: Double(rate), targetChannels: 1)
        guard let samples = buffer.floatChannelData?[0] else { throw NSError(domain: "Audio", code: 1) }
        let totalSamples = Int(buffer.frameLength)

        // STFT
        status = "Computing STFT..."
        let nFFT = 1024, hopLen = 256, winLen = 1024
        let freqBins = nFFT / 2 + 1  // 513
        let spec = computeSTFT(samples: samples, count: totalSamples, nFFT: nFFT, hopLength: hopLen, winLength: winLen)
        let numFrames = spec.count / freqBins

        // Load speaker encoder
        status = "Extracting speaker..."
        let seFile = model.files.first { $0.name.lowercased().contains("speaker") || $0.name.lowercased().contains("encoder") }?.name ?? model.files[0].name
        let se = try await ModelLoader.load(for: model, named: seFile)

        // Speaker encoder input: [1, T, 513] (time x frequency)
        let seInput = try MLMultiArray(shape: [1, NSNumber(value: numFrames), NSNumber(value: freqBins)], dataType: .float32)
        let sePtr = seInput.dataPointer.assumingMemoryBound(to: Float.self)
        for t in 0..<numFrames {
            for f in 0..<freqBins { sePtr[t * freqBins + f] = spec[f * numFrames + t] }  // transpose
        }

        let seOut = try await se.prediction(from: MLDictionaryFeatureProvider(dictionary: ["spectrogram": seInput]))
        guard let speakerEmb = seOut.featureValue(for: "speaker_embedding")?.multiArrayValue else {
            throw NSError(domain: "OpenVoice", code: 1)
        }
        let srcEmb = ImageUtils.extractFloats(speakerEmb)

        // Voice converter
        status = "Converting voice..."
        let vcFile = model.files.first { $0.name.lowercased().contains("converter") }?.name ?? model.files.last!.name
        let vc = try await ModelLoader.load(for: model, named: vcFile)

        // Converter input: spectrogram [1,513,T], spec_lengths [1], source/target speaker [1,256,1]
        let specArr = try MLMultiArray(shape: [1, NSNumber(value: freqBins), NSNumber(value: numFrames)], dataType: .float32)
        let spPtr = specArr.dataPointer.assumingMemoryBound(to: Float.self)
        memcpy(spPtr, spec, freqBins * numFrames * MemoryLayout<Float>.size)

        let specLen = try MLMultiArray(shape: [1], dataType: .float32)
        specLen[0] = NSNumber(value: numFrames)

        let srcSpk = try MLMultiArray(shape: [1, 256, 1], dataType: .float32)
        let tgtSpk = try MLMultiArray(shape: [1, 256, 1], dataType: .float32)
        let srcP = srcSpk.dataPointer.assumingMemoryBound(to: Float.self)
        let tgtP = tgtSpk.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<min(256, srcEmb.count) { srcP[i] = srcEmb[i]; tgtP[i] = srcEmb[i] }  // self-conversion demo

        let vcOut = try await vc.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "spectrogram": specArr, "spec_lengths": specLen,
            "source_speaker": srcSpk, "target_speaker": tgtSpk
        ]))

        guard let audioArr = vcOut.featureNames.compactMap({ vcOut.featureValue(for: $0)?.multiArrayValue }).first else {
            throw NSError(domain: "OpenVoice", code: 2)
        }

        var audio = ImageUtils.extractFloats(audioArr)
        // Normalize to prevent clipping
        let maxVal = audio.map { abs($0) }.max() ?? 1.0
        if maxVal > 1.0 { let s = 0.95 / maxVal; for i in audio.indices { audio[i] *= s } }

        let url = try saveWAV(samples: audio, sampleRate: rate, channels: 1, samplesPerChannel: audio.count, name: "converted")
        await MainActor.run { outputURLs = [("converted", url)]; progress = 1.0 }
    }

    // MARK: - OpenVoice Voice Conversion (source + target)

    private func runVoiceConversionWithTarget(sourceURL: URL, targetURL: URL, sampleRate: Int) async throws {
        let rate = 22050
        let nFFT = 1024, hopLen = 256, winLen = 1024
        let freqBins = nFFT / 2 + 1  // 513

        // Load speaker encoder
        await MainActor.run { status = "Extracting your voice profile..." }
        let seFile = model.files.first { $0.name.lowercased().contains("speaker") || $0.name.lowercased().contains("encoder") }?.name ?? model.files[0].name
        let se = try await ModelLoader.load(for: model, named: seFile)

        // Extract source speaker embedding
        let srcEmb = try await extractSpeakerEmbedding(audioURL: sourceURL, rate: rate, nFFT: nFFT, hopLen: hopLen, winLen: winLen, freqBins: freqBins, encoder: se)

        // Extract target speaker embedding
        await MainActor.run { status = "Extracting target voice profile..." }
        let tgtEmb = try await extractSpeakerEmbedding(audioURL: targetURL, rate: rate, nFFT: nFFT, hopLen: hopLen, winLen: winLen, freqBins: freqBins, encoder: se)

        // Load source audio for conversion (resampled to OpenVoice rate).
        await MainActor.run { status = "Computing STFT..." }
        let buffer = try AudioUtils.loadResampled(url: sourceURL, targetSampleRate: Double(rate), targetChannels: 1)
        guard let samples = buffer.floatChannelData?[0] else { throw NSError(domain: "Audio", code: 1) }
        let totalSamples = Int(buffer.frameLength)

        let spec = computeSTFT(samples: samples, count: totalSamples, nFFT: nFFT, hopLength: hopLen, winLength: winLen)
        let numFrames = spec.count / freqBins

        // Voice converter
        await MainActor.run { status = "Converting voice..." }
        let vcFile = model.files.first { $0.name.lowercased().contains("converter") }?.name ?? model.files.last!.name
        let vc = try await ModelLoader.load(for: model, named: vcFile)

        let specArr = try MLMultiArray(shape: [1, NSNumber(value: freqBins), NSNumber(value: numFrames)], dataType: .float32)
        let spPtr = specArr.dataPointer.assumingMemoryBound(to: Float.self)
        memcpy(spPtr, spec, freqBins * numFrames * MemoryLayout<Float>.size)

        let specLen = try MLMultiArray(shape: [1], dataType: .float32)
        specLen[0] = NSNumber(value: numFrames)

        let srcSpk = try MLMultiArray(shape: [1, 256, 1], dataType: .float32)
        let tgtSpk = try MLMultiArray(shape: [1, 256, 1], dataType: .float32)
        let srcP = srcSpk.dataPointer.assumingMemoryBound(to: Float.self)
        let tgtP = tgtSpk.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<min(256, srcEmb.count) { srcP[i] = srcEmb[i] }
        for i in 0..<min(256, tgtEmb.count) { tgtP[i] = tgtEmb[i] }

        let vcOut = try await vc.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "spectrogram": specArr, "spec_lengths": specLen,
            "source_speaker": srcSpk, "target_speaker": tgtSpk
        ]))

        guard let audioArr = vcOut.featureNames.compactMap({ vcOut.featureValue(for: $0)?.multiArrayValue }).first else {
            throw NSError(domain: "OpenVoice", code: 2)
        }

        var audio = ImageUtils.extractFloats(audioArr)
        let maxVal = audio.map { abs($0) }.max() ?? 1.0
        if maxVal > 1.0 { let s = 0.95 / maxVal; for i in audio.indices { audio[i] *= s } }

        let url = try saveWAV(samples: audio, sampleRate: rate, channels: 1, samplesPerChannel: audio.count, name: "converted")
        await MainActor.run { convertedURL = url; progress = 1.0 }
    }

    private func extractSpeakerEmbedding(audioURL: URL, rate: Int, nFFT: Int, hopLen: Int, winLen: Int, freqBins: Int, encoder: MLModel) async throws -> [Float] {
        let buffer = try AudioUtils.loadResampled(url: audioURL, targetSampleRate: Double(rate), targetChannels: 1)
        guard let samples = buffer.floatChannelData?[0] else { throw NSError(domain: "Audio", code: 1) }
        let totalSamples = Int(buffer.frameLength)

        let spec = computeSTFT(samples: samples, count: totalSamples, nFFT: nFFT, hopLength: hopLen, winLength: winLen)
        let numFrames = spec.count / freqBins

        let seInput = try MLMultiArray(shape: [1, NSNumber(value: numFrames), NSNumber(value: freqBins)], dataType: .float32)
        let sePtr = seInput.dataPointer.assumingMemoryBound(to: Float.self)
        for t in 0..<numFrames {
            for f in 0..<freqBins { sePtr[t * freqBins + f] = spec[f * numFrames + t] }
        }

        let seOut = try await encoder.prediction(from: MLDictionaryFeatureProvider(dictionary: ["spectrogram": seInput]))
        guard let speakerEmb = seOut.featureValue(for: "speaker_embedding")?.multiArrayValue else {
            throw NSError(domain: "OpenVoice", code: 1)
        }
        return ImageUtils.extractFloats(speakerEmb)
    }

    // MARK: - STFT (magnitude spectrogram)

    private func computeSTFT(samples: UnsafePointer<Float>, count: Int, nFFT: Int, hopLength: Int, winLength: Int) -> [Float] {
        let freqBins = nFFT / 2 + 1
        let padAmount = (nFFT - hopLength) / 2

        // Reflect pad
        var padded = [Float](repeating: 0, count: count + 2 * padAmount)
        for i in 0..<padAmount { padded[i] = samples[min(padAmount - i, count - 1)] }
        memcpy(&padded[padAmount], samples, count * MemoryLayout<Float>.size)
        for i in 0..<padAmount { padded[count + padAmount + i] = samples[max(0, count - 2 - i)] }

        let numFrames = (padded.count - nFFT) / hopLength + 1

        // Hann window (periodic)
        var window = [Float](repeating: 0, count: winLength)
        for i in 0..<winLength { window[i] = 0.5 * (1 - cos(2 * .pi * Float(i) / Float(winLength))) }

        // FFT setup
        let log2n = vDSP_Length(log2(Float(nFFT)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else { return [] }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        var result = [Float](repeating: 0, count: freqBins * numFrames)
        var realp = [Float](repeating: 0, count: nFFT / 2)
        var imagp = [Float](repeating: 0, count: nFFT / 2)

        for frame in 0..<numFrames {
            let start = frame * hopLength
            var windowed = [Float](repeating: 0, count: nFFT)
            for i in 0..<winLength { windowed[i] = padded[start + i] * window[i] }

            realp.withUnsafeMutableBufferPointer { rBuf in
                imagp.withUnsafeMutableBufferPointer { iBuf in
                    var split = DSPSplitComplex(realp: rBuf.baseAddress!, imagp: iBuf.baseAddress!)
                    windowed.withUnsafeBufferPointer { src in
                        src.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: nFFT / 2) { ptr in
                            vDSP_ctoz(ptr, 2, &split, 1, vDSP_Length(nFFT / 2))
                        }
                    }
                    vDSP_fft_zrip(fftSetup, &split, 1, log2n, FFTDirection(kFFTDirection_Forward))

                    // Magnitude: DC and Nyquist are packed in realp[0] and imagp[0]
                    let dc = sqrt(split.realp[0] * split.realp[0] + 1e-6)
                    let ny = sqrt(split.imagp[0] * split.imagp[0] + 1e-6)
                    result[0 * numFrames + frame] = dc
                    for k in 1..<(nFFT / 2) {
                        let r = split.realp[k] / 2  // vDSP 2x scaling
                        let im = split.imagp[k] / 2
                        result[k * numFrames + frame] = sqrt(r * r + im * im + 1e-6)
                    }
                    result[(nFFT / 2) * numFrames + frame] = ny
                }
            }
        }
        return result
    }

    // MARK: - WAV Writer

    private func saveWAV(samples: [Float], sampleRate: Int, channels: Int, samplesPerChannel: Int, name: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(name)_\(UUID().uuidString).wav")
        let format = AVAudioFormat(standardFormatWithSampleRate: Double(sampleRate), channels: AVAudioChannelCount(channels))!
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

// MARK: - Audio Loading Utilities

/// Shared audio preprocessing helpers. Mirrors DemucsDemo's AVAudioConverter-based
/// loader so templates always feed the model the sample rate / channel count it expects
/// regardless of the source file.
enum AudioUtils {
    /// Decode an audio file and resample / remix it to the requested format.
    /// - Parameter targetSampleRate: sample rate the model expects (e.g. 44100 for HTDemucs, 16000 for pyannote, 22050 for OpenVoice / BasicPitch).
    /// - Parameter targetChannels: 1 for mono, 2 for stereo.
    /// Returns a Float32, non-interleaved AVAudioPCMBuffer whose frameLength is the resampled length.
    static func loadResampled(url: URL, targetSampleRate: Double, targetChannels: AVAudioChannelCount) throws -> AVAudioPCMBuffer {
        let accessing = url.startAccessingSecurityScopedResource()
        defer { if accessing { url.stopAccessingSecurityScopedResource() } }

        let sourceFile = try AVAudioFile(forReading: url)
        let sourceFormat = sourceFile.processingFormat
        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: targetSampleRate,
            channels: targetChannels,
            interleaved: false
        ) else {
            throw NSError(domain: "AudioUtils", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Unsupported target audio format"])
        }

        // Compute output capacity from the sample-rate ratio plus a small padding for the resampler tail.
        let ratio = targetSampleRate / sourceFormat.sampleRate
        let estimatedFrames = AVAudioFrameCount(ceil(Double(sourceFile.length) * ratio)) + 1024
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: max(estimatedFrames, 1)) else {
            throw NSError(domain: "AudioUtils", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "Failed to allocate output buffer"])
        }

        // Fast path: source already matches target format and channel count.
        if sourceFormat.sampleRate == targetSampleRate && sourceFormat.channelCount == targetChannels {
            try sourceFile.read(into: outputBuffer)
            return outputBuffer
        }

        guard let converter = AVAudioConverter(from: sourceFormat, to: targetFormat) else {
            throw NSError(domain: "AudioUtils", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "Cannot create AVAudioConverter"])
        }

        // Pull-model conversion handles any sample-rate and channel-layout mismatch.
        let readChunk: AVAudioFrameCount = 4096
        let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
            guard let readBuffer = AVAudioPCMBuffer(pcmFormat: sourceFormat, frameCapacity: readChunk) else {
                outStatus.pointee = .endOfStream
                return nil
            }
            do {
                try sourceFile.read(into: readBuffer)
            } catch {
                outStatus.pointee = .endOfStream
                return nil
            }
            if readBuffer.frameLength == 0 {
                outStatus.pointee = .endOfStream
                return nil
            }
            outStatus.pointee = .haveData
            return readBuffer
        }

        var convertError: NSError?
        let status = converter.convert(to: outputBuffer, error: &convertError, withInputFrom: inputBlock)
        if let convertError { throw convertError }
        if status == .error {
            throw NSError(domain: "AudioUtils", code: 4,
                          userInfo: [NSLocalizedDescriptionKey: "Audio conversion failed"])
        }
        return outputBuffer
    }
}

// MARK: - Diarization Timeline Visualization

struct DiarizationTimelineView: View {
    let segments: [(Int, Double, Double)]
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
                    let spkSegs = segments.filter { $0.0 == spk }
                    if !spkSegs.isEmpty {
                        ZStack(alignment: .leading) {
                            RoundedRectangle(cornerRadius: 4)
                                .fill(Color.gray.opacity(0.1))
                                .frame(height: trackH)

                            ForEach(Array(spkSegs.enumerated()), id: \.offset) { _, seg in
                                let x = duration > 0 ? CGFloat(seg.1 / duration) * w : 0
                                let segW = duration > 0 ? CGFloat((seg.2 - seg.1) / duration) * w : 0
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

// MARK: - Audio Recorder

class AudioRecorder: ObservableObject {
    @Published var isRecording = false
    @Published var recordedURL: URL?
    @Published var duration: Double?

    private var audioRecorder: AVAudioRecorder?
    private let sampleRate: Double

    init(sampleRate: Double = 22050) {
        self.sampleRate = sampleRate
    }

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
            AVSampleRateKey: sampleRate,
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
        let dest = FileManager.default.temporaryDirectory
            .appendingPathComponent("import_\(UUID().uuidString).\(url.pathExtension)")
        _ = url.startAccessingSecurityScopedResource()
        defer { url.stopAccessingSecurityScopedResource() }
        try? FileManager.default.copyItem(at: url, to: dest)
        recordedURL = dest
        if let file = try? AVAudioFile(forReading: dest) {
            duration = Double(file.length) / file.fileFormat.sampleRate
        }
    }
}

// MARK: - Audio File Picker

struct AudioPickerView: UIViewControllerRepresentable {
    let onPick: (URL) -> Void

    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        let picker = UIDocumentPickerViewController(forOpeningContentTypes: [.audio])
        picker.delegate = context.coordinator
        return picker
    }
    func updateUIViewController(_ vc: UIDocumentPickerViewController, context: Context) {}
    func makeCoordinator() -> Coordinator { Coordinator(onPick: onPick) }

    class Coordinator: NSObject, UIDocumentPickerDelegate {
        let onPick: (URL) -> Void
        init(onPick: @escaping (URL) -> Void) { self.onPick = onPick }
        func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
            if let url = urls.first { _ = url.startAccessingSecurityScopedResource(); onPick(url) }
        }
    }
}
