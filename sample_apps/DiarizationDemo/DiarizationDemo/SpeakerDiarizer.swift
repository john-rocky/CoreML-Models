import CoreML
import AVFoundation
import Speech

struct SpeakerSegment: Identifiable {
    let id = UUID()
    let speaker: Int
    let startTime: Double
    let endTime: Double
    var transcript: String?

    var duration: Double { endTime - startTime }
}

class SpeakerDiarizer: ObservableObject {
    private var mlModel: MLModel?
    @Published var isReady = false

    // pyannote segmentation-3.0 parameters
    static let sampleRate: Double = 16000
    static let chunkDuration: Double = 10.0  // 10 seconds per chunk
    static let chunkSamples = 160000          // 10s * 16kHz
    static let numFrames = 589                 // output frames per 10s chunk
    static let numClasses = 7                  // powerset classes

    // Powerset class mapping:
    // 0: no speaker, 1: spk1, 2: spk2, 3: spk3
    // 4: spk1+2, 5: spk1+3, 6: spk2+3
    private static let classToSpeakers: [[Int]] = [
        [],     // 0: none
        [0],    // 1: speaker 1
        [1],    // 2: speaker 2
        [2],    // 3: speaker 3
        [0, 1], // 4: speakers 1+2
        [0, 2], // 5: speakers 1+3
        [1, 2], // 6: speakers 2+3
    ]

    init() {
        loadModel()
    }

    private func loadModel() {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self, let resourcePath = Bundle.main.resourcePath else { return }
            let fm = FileManager.default
            guard let items = try? fm.contentsOfDirectory(atPath: resourcePath) else { return }
            let config = MLModelConfiguration()
            config.computeUnits = .all
            for item in items where item.hasSuffix(".mlmodelc") && item.contains("Speaker") {
                let url = URL(fileURLWithPath: (resourcePath as NSString).appendingPathComponent(item))
                if let model = try? MLModel(contentsOf: url, configuration: config) {
                    self.mlModel = model
                    DispatchQueue.main.async { self.isReady = true }
                    return
                }
            }
        }
    }

    // MARK: - Diarize

    func diarize(audioURL: URL, progress: @escaping (String) -> Void) async throws -> [SpeakerSegment] {
        progress("Loading audio...")
        let samples = try loadAudio(url: audioURL)
        let totalDuration = Double(samples.count) / Self.sampleRate

        progress("Analyzing speakers...")
        guard let model = mlModel else { throw DiarizationError.modelNotLoaded }

        // Process in 10s chunks with 5s overlap
        let hopSamples = Self.chunkSamples / 2  // 5s hop for overlap
        let numChunks = max(1, (samples.count - Self.chunkSamples) / hopSamples + 1)

        // Accumulate per-frame speaker probabilities
        let frameRate = Double(Self.numFrames) / Self.chunkDuration
        let totalFrames = Int(totalDuration * frameRate) + 1
        var speakerProbs = [[Float]](repeating: [Float](repeating: 0, count: 3), count: totalFrames)
        var frameCounts = [Float](repeating: 0, count: totalFrames)

        for chunk in 0..<numChunks {
            let startSample = chunk * hopSamples
            let endSample = min(startSample + Self.chunkSamples, samples.count)

            // Pad if needed
            var chunkSamples = [Float](repeating: 0, count: Self.chunkSamples)
            let copyLen = endSample - startSample
            for i in 0..<copyLen {
                chunkSamples[i] = samples[startSample + i]
            }

            // Run model
            let inputArray = try MLMultiArray(shape: [1, 1, NSNumber(value: Self.chunkSamples)], dataType: .float32)
            for i in 0..<Self.chunkSamples {
                inputArray[i] = NSNumber(value: chunkSamples[i])
            }
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "audio": MLFeatureValue(multiArray: inputArray)
            ])
            let output = try model.prediction(from: input)
            guard let logits = output.featureValue(for: "speaker_logits")?.multiArrayValue else {
                throw DiarizationError.predictionFailed
            }

            // Convert log-softmax output to speaker probabilities
            // Output is already LogSoftmax → use exp() to get probs
            let chunkStartTime = Double(startSample) / Self.sampleRate

            // Debug: print first chunk's logits range
            if chunk == 0 {
                var minV: Float = .greatestFiniteMagnitude
                var maxV: Float = -.greatestFiniteMagnitude
                for i in 0..<(Self.numFrames * Self.numClasses) {
                    let v = logits[i].floatValue
                    minV = min(minV, v)
                    maxV = max(maxV, v)
                }
                print("[Diarize] Logits range: [\(minV), \(maxV)], shape: \(logits.shape)")
            }

            for f in 0..<Self.numFrames {
                let frameTime = chunkStartTime + Double(f) / frameRate
                let globalFrame = Int(frameTime * frameRate)
                guard globalFrame < totalFrames else { continue }

                // exp(log_softmax) = softmax probabilities
                var probs = [Float](repeating: 0, count: Self.numClasses)
                for c in 0..<Self.numClasses {
                    probs[c] = exp(logits[f * Self.numClasses + c].floatValue)
                }

                // Convert powerset probs to per-speaker probs
                for c in 0..<Self.numClasses {
                    for spk in Self.classToSpeakers[c] {
                        speakerProbs[globalFrame][spk] += probs[c]
                    }
                }
                frameCounts[globalFrame] += 1
            }

            progress("Analyzing... \(Int(Float(chunk + 1) / Float(numChunks) * 100))%")
        }

        // Normalize by overlap count
        for f in 0..<totalFrames {
            if frameCounts[f] > 0 {
                for s in 0..<3 { speakerProbs[f][s] /= frameCounts[f] }
            }
        }

        // Convert to segments
        progress("Building segments...")
        let segments = buildSegments(probs: speakerProbs, frameRate: frameRate, threshold: 0.5)
        return segments
    }

    // MARK: - Segment building

    private func buildSegments(probs: [[Float]], frameRate: Double, threshold: Float) -> [SpeakerSegment] {
        var segments: [SpeakerSegment] = []

        for spk in 0..<3 {
            var segStart: Int?
            for f in 0..<probs.count {
                let active = probs[f][spk] > threshold
                if active && segStart == nil {
                    segStart = f
                } else if !active, let start = segStart {
                    let startTime = Double(start) / frameRate
                    let endTime = Double(f) / frameRate
                    if endTime - startTime > 0.3 {  // minimum 300ms
                        segments.append(SpeakerSegment(speaker: spk, startTime: startTime, endTime: endTime))
                    }
                    segStart = nil
                }
            }
            // Close final segment
            if let start = segStart {
                let startTime = Double(start) / frameRate
                let endTime = Double(probs.count) / frameRate
                if endTime - startTime > 0.3 {
                    segments.append(SpeakerSegment(speaker: spk, startTime: startTime, endTime: endTime))
                }
            }
        }

        return segments.sorted { $0.startTime < $1.startTime }
    }

    // MARK: - Audio loading

    private func loadAudio(url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let originalFormat = file.processingFormat
        let frameCount = AVAudioFrameCount(file.length)

        guard let buffer = AVAudioPCMBuffer(pcmFormat: originalFormat, frameCapacity: frameCount) else {
            throw DiarizationError.audioLoadFailed
        }
        try file.read(into: buffer)

        // Resample to 16kHz mono
        let targetFormat = AVAudioFormat(standardFormatWithSampleRate: Self.sampleRate, channels: 1)!
        guard let converter = AVAudioConverter(from: originalFormat, to: targetFormat) else {
            throw DiarizationError.audioLoadFailed
        }

        let ratio = Self.sampleRate / originalFormat.sampleRate
        let outputFrameCount = AVAudioFrameCount(Double(frameCount) * ratio)
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outputFrameCount) else {
            throw DiarizationError.audioLoadFailed
        }

        var isDone = false
        let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
            if isDone { outStatus.pointee = .noDataNow; return nil }
            isDone = true
            outStatus.pointee = .haveData
            return buffer
        }
        try converter.convert(to: outputBuffer, error: nil, withInputFrom: inputBlock)

        guard let data = outputBuffer.floatChannelData else { throw DiarizationError.audioLoadFailed }
        return Array(UnsafeBufferPointer(start: data[0], count: Int(outputBuffer.frameLength)))
    }

    // MARK: - Track splitting

    func exportSpeakerTracks(audioURL: URL, segments: [SpeakerSegment]) throws -> [Int: URL] {
        let file = try AVAudioFile(forReading: audioURL)
        let format = file.processingFormat
        let sampleRate = format.sampleRate
        var tracks: [Int: URL] = [:]

        let speakers = Set(segments.map { $0.speaker })
        for spk in speakers {
            let spkSegs = segments.filter { $0.speaker == spk }
            let totalSamples = spkSegs.reduce(0) { $0 + Int($1.duration * sampleRate) }
            guard totalSamples > 0 else { continue }

            let outURL = FileManager.default.temporaryDirectory
                .appendingPathComponent("speaker_\(spk)_\(UUID().uuidString).wav")
            let outFile = try AVAudioFile(forWriting: outURL, settings: format.settings)

            for seg in spkSegs {
                let startFrame = AVAudioFramePosition(seg.startTime * sampleRate)
                let frameCount = AVAudioFrameCount(seg.duration * sampleRate)
                guard frameCount > 0 else { continue }

                file.framePosition = startFrame
                guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else { continue }
                let readCount = min(frameCount, AVAudioFrameCount(file.length - startFrame))
                guard readCount > 0 else { continue }
                try file.read(into: buffer, frameCount: readCount)
                try outFile.write(from: buffer)
            }
            tracks[spk] = outURL
        }
        return tracks
    }

    // MARK: - Transcription per speaker

    func transcribe(audioURL: URL, segments: [SpeakerSegment],
                    progress: @escaping (String) -> Void) async throws -> [SpeakerSegment] {
        guard SFSpeechRecognizer.authorizationStatus() == .authorized else {
            SFSpeechRecognizer.requestAuthorization { _ in }
            throw DiarizationError.transcriptionDenied
        }

        guard let recognizer = SFSpeechRecognizer() else {
            throw DiarizationError.transcriptionFailed
        }

        var result = segments
        for i in 0..<result.count {
            progress("Transcribing \(i + 1)/\(result.count)...")

            let seg = result[i]
            let segURL = try extractSegmentAudio(from: audioURL, start: seg.startTime, duration: seg.duration)

            let request = SFSpeechURLRecognitionRequest(url: segURL)
            request.shouldReportPartialResults = false

            let recognition = try await withCheckedThrowingContinuation { (cont: CheckedContinuation<SFSpeechRecognitionResult, Error>) in
                recognizer.recognitionTask(with: request) { res, error in
                    if let error { cont.resume(throwing: error); return }
                    if let res, res.isFinal { cont.resume(returning: res) }
                }
            }
            result[i].transcript = recognition.bestTranscription.formattedString

            try? FileManager.default.removeItem(at: segURL)
        }
        return result
    }

    private func extractSegmentAudio(from url: URL, start: Double, duration: Double) throws -> URL {
        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        let sampleRate = format.sampleRate

        let startFrame = AVAudioFramePosition(start * sampleRate)
        let frameCount = AVAudioFrameCount(duration * sampleRate)
        guard frameCount > 0 else { throw DiarizationError.audioLoadFailed }

        file.framePosition = startFrame
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw DiarizationError.audioLoadFailed
        }
        let readCount = min(frameCount, AVAudioFrameCount(file.length - startFrame))
        try file.read(into: buffer, frameCount: readCount)

        let outURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("seg_\(UUID().uuidString).wav")
        let outFile = try AVAudioFile(forWriting: outURL, settings: format.settings)
        try outFile.write(from: buffer)
        return outURL
    }
}

enum DiarizationError: LocalizedError {
    case modelNotLoaded, predictionFailed, audioLoadFailed, transcriptionDenied, transcriptionFailed
    var errorDescription: String? {
        switch self {
        case .modelNotLoaded: return "Model not loaded"
        case .predictionFailed: return "Prediction failed"
        case .audioLoadFailed: return "Failed to load audio"
        case .transcriptionDenied: return "Speech recognition not authorized"
        case .transcriptionFailed: return "Transcription failed"
        }
    }
}
