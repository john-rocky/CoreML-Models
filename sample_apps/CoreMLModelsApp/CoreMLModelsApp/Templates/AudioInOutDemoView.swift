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

    private var stemNames: [String] { model.configStringArray("output_stems") ?? ["output"] }

    var body: some View {
        VStack(spacing: 0) {
            if let url = inputURL {
                HStack {
                    Image(systemName: "waveform").foregroundStyle(.secondary)
                    Text(url.lastPathComponent).font(.caption).lineLimit(1)
                    Spacer()
                    Button { playAudio(url: url, label: "input") } label: {
                        Image(systemName: currentlyPlaying == "input" ? "stop.circle.fill" : "play.circle.fill").font(.title2)
                    }
                }
                .padding().background(Color(.systemGray6)).clipShape(RoundedRectangle(cornerRadius: 8)).padding(.horizontal)
            }

            if !outputURLs.isEmpty {
                List {
                    ForEach(outputURLs, id: \.0) { (stem, url) in
                        HStack {
                            Image(systemName: stemIcon(stem)).foregroundStyle(.tint)
                            Text(stem.capitalized)
                            Spacer()
                            Button { playAudio(url: url, label: stem) } label: {
                                Image(systemName: currentlyPlaying == stem ? "stop.circle.fill" : "play.circle.fill").font(.title2)
                            }
                            ShareLink(item: url) { Image(systemName: "square.and.arrow.up") }
                        }
                    }
                }.listStyle(.plain)
            } else if !speakerTimeline.isEmpty {
                // Diarization timeline
                List {
                    ForEach(Array(speakerTimeline.enumerated()), id: \.offset) { _, seg in
                        HStack {
                            Circle().fill(speakerColor(seg.0)).frame(width: 12, height: 12)
                            Text("Speaker \(seg.0 + 1)")
                            Spacer()
                            Text(String(format: "%.1f – %.1fs", seg.1, seg.2))
                                .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                        }
                    }
                }.listStyle(.plain)
            } else {
                VStack(spacing: 12) {
                    Image(systemName: "waveform.path.ecg").font(.system(size: 60)).foregroundStyle(.secondary)
                    Text("Select an audio file to process").foregroundStyle(.secondary)
                }.frame(maxWidth: .infinity, maxHeight: .infinity)
            }

            VStack(spacing: 12) {
                if isProcessing { ProgressView(value: Double(progress)); Text(status).font(.caption).foregroundStyle(.secondary) }
                if let t = processingTime { Text(String(format: "%.1fs", t)).font(.caption.monospacedDigit()).foregroundStyle(.secondary) }
                HStack(spacing: 12) {
                    Button { showingFilePicker = true } label: {
                        Label("Select Audio", systemImage: "doc.badge.plus").frame(maxWidth: .infinity)
                    }.buttonStyle(.bordered)
                    if inputURL != nil {
                        Button { Task { await runProcessing() } } label: {
                            Label("Process", systemImage: "wand.and.rays").frame(maxWidth: .infinity)
                        }.buttonStyle(.borderedProminent).disabled(isProcessing)
                    }
                }
            }.padding()
        }
        .sheet(isPresented: $showingFilePicker) {
            AudioPickerView { url in inputURL = url; outputURLs = []; speakerTimeline = [] }
        }
    }

    private func stemIcon(_ name: String) -> String {
        switch name.lowercased() {
        case "vocals": return "mic.fill"
        case "drums": return "drum.fill"
        case "bass": return "guitars.fill"
        default: return "waveform"
        }
    }

    private func speakerColor(_ idx: Int) -> Color {
        [Color.blue, .green, .orange, .purple, .red, .cyan, .pink][idx % 7]
    }

    private func playAudio(url: URL, label: String) {
        if currentlyPlaying == label { player?.stop(); currentlyPlaying = nil; return }
        player = try? AVAudioPlayer(contentsOf: url); player?.play(); currentlyPlaying = label
    }

    private func runProcessing() async {
        guard let inputURL else { return }
        isProcessing = true; progress = 0; outputURLs = []; speakerTimeline = []

        do {
            let sampleRate = model.configInt("sample_rate") ?? 44100
            let start = CFAbsoluteTimeGetCurrent()

            // Detect mode from stems config or model name
            let stems = stemNames
            if stems.contains("speaker_timeline") {
                try await runDiarization(inputURL: inputURL)
            } else if stems.contains("converted") {
                try await runVoiceConversion(inputURL: inputURL, sampleRate: sampleRate)
            } else {
                try await runSourceSeparation(inputURL: inputURL, sampleRate: sampleRate)
            }

            let elapsed = CFAbsoluteTimeGetCurrent() - start
            await MainActor.run { processingTime = elapsed; isProcessing = false; status = "" }
        } catch {
            await MainActor.run { isProcessing = false; status = "Error: \(error.localizedDescription)" }
        }
    }

    // MARK: - HTDemucs Source Separation

    private func runSourceSeparation(inputURL: URL, sampleRate: Int) async throws {
        status = "Loading audio…"
        let audioFile = try AVAudioFile(forReading: inputURL)
        let format = AVAudioFormat(standardFormatWithSampleRate: Double(sampleRate), channels: 2)!
        let frameCount = AVAudioFrameCount(audioFile.length)
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)!
        try audioFile.read(into: buffer)
        guard let floatData = buffer.floatChannelData else { throw NSError(domain: "Audio", code: 1) }
        let totalSamples = Int(buffer.frameLength)

        status = "Loading model…"
        let mlModel = try await ModelLoader.loadPrimary(for: model)

        status = "Processing…"
        let segmentLength = model.configInt("segment_length") ?? totalSamples
        let channels = min(Int(buffer.format.channelCount), 2)

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
        let samplesToProcess = min(totalSamples, segmentLength)
        for ch in 0..<min(channels, 2) {
            for i in 0..<samplesToProcess { inPtr[ch * segmentLength + i] = floatData[ch][i] }
        }

        let inputName = desc.keys.first ?? "audio"
        let output = try await mlModel.prediction(from: MLDictionaryFeatureProvider(dictionary: [inputName: inputArr]))
        await MainActor.run { progress = 0.8 }

        // Extract output stems
        status = "Saving stems…"
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
        status = "Loading audio…"
        let audioFile = try AVAudioFile(forReading: inputURL)
        let format = AVAudioFormat(standardFormatWithSampleRate: 16000, channels: 1)!
        let frameCount = AVAudioFrameCount(audioFile.length * 16000 / Int64(audioFile.fileFormat.sampleRate))
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: max(frameCount, 1))!
        try audioFile.read(into: buffer)
        guard let samples = buffer.floatChannelData?[0] else { throw NSError(domain: "Audio", code: 1) }
        let totalSamples = Int(buffer.frameLength)

        status = "Loading model…"
        let mlModel = try await ModelLoader.loadPrimary(for: model)

        let chunkSamples = 160000  // 10s at 16kHz
        let hopSamples = chunkSamples / 2  // 50% overlap
        let framesPerChunk = 589
        let numClasses = 7
        let maxSpeakers = 3

        // Powerset: class → speakers
        let classToSpeakers: [[Int]] = [[], [0], [1], [2], [0,1], [0,2], [1,2]]

        let totalFrames = max(1, Int(ceil(Double(totalSamples) / Double(hopSamples))) * (framesPerChunk / 2))
        var speakerProbs = [[Float]](repeating: [Float](repeating: 0, count: maxSpeakers), count: totalFrames)
        var frameCounts = [Int](repeating: 0, count: totalFrames)

        let numChunks = max(1, (totalSamples - chunkSamples + hopSamples) / hopSamples + (totalSamples <= chunkSamples ? 0 : 1))

        for chunk in 0..<max(1, numChunks) {
            status = "Chunk \(chunk + 1)/\(max(1, numChunks))…"
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

                // exp(logSoftmax) → probabilities
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

    // MARK: - OpenVoice Voice Conversion

    private func runVoiceConversion(inputURL: URL, sampleRate: Int) async throws {
        status = "Loading audio…"
        let rate = 22050
        let audioFile = try AVAudioFile(forReading: inputURL)
        let format = AVAudioFormat(standardFormatWithSampleRate: Double(rate), channels: 1)!
        let frameCount = AVAudioFrameCount(audioFile.length * Int64(rate) / Int64(audioFile.fileFormat.sampleRate))
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: max(frameCount, 1))!
        try audioFile.read(into: buffer)
        guard let samples = buffer.floatChannelData?[0] else { throw NSError(domain: "Audio", code: 1) }
        let totalSamples = Int(buffer.frameLength)

        // STFT
        status = "Computing STFT…"
        let nFFT = 1024, hopLen = 256, winLen = 1024
        let freqBins = nFFT / 2 + 1  // 513
        let spec = computeSTFT(samples: samples, count: totalSamples, nFFT: nFFT, hopLength: hopLen, winLength: winLen)
        let numFrames = spec.count / freqBins

        // Load speaker encoder
        status = "Extracting speaker…"
        let seFile = model.files.first { $0.name.lowercased().contains("speaker") || $0.name.lowercased().contains("encoder") }?.name ?? model.files[0].name
        let se = try await ModelLoader.load(for: model, named: seFile)

        // Speaker encoder input: [1, T, 513] (time × frequency)
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
        status = "Converting voice…"
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
