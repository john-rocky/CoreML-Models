import CoreML
import AVFoundation
import Accelerate

class VoiceConverter: ObservableObject {
    private var encoderModel: MLModel?
    private var converterModel: MLModel?
    @Published var isReady = false
    @Published var status = "Loading models..."

    // OpenVoice V2 audio parameters
    static let sampleRate: Double = 22050
    static let nFFT = 1024
    static let hopLength = 256
    static let winLength = 1024

    init() {
        loadModels()
    }

    private func loadModels() {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self, let resourcePath = Bundle.main.resourcePath else { return }
            let fm = FileManager.default
            guard let items = try? fm.contentsOfDirectory(atPath: resourcePath) else { return }

            let config = MLModelConfiguration()
            config.computeUnits = .all

            for item in items where item.hasSuffix(".mlmodelc") {
                let url = URL(fileURLWithPath: (resourcePath as NSString).appendingPathComponent(item))
                guard let model = try? MLModel(contentsOf: url, configuration: config) else { continue }
                if item.contains("SpeakerEncoder") {
                    self.encoderModel = model
                } else if item.contains("VoiceConverter") {
                    self.converterModel = model
                }
            }

            let ready = self.encoderModel != nil && self.converterModel != nil
            DispatchQueue.main.async {
                self.isReady = ready
                self.status = ready ? "Ready" : "Failed to load models"
            }
        }
    }

    // MARK: - Public API

    func extractSpeakerEmbedding(audioURL: URL) async throws -> [Float] {
        let samples = try loadAudio(url: audioURL)
        let spec = stft(samples: samples)
        // spec is [freqBins, frames] = [513, T]
        // SpeakerEncoder expects [1, T, 513] (transposed)
        let T = spec.count / 513
        let transposed = transposeSpec(spec, freqBins: 513, frames: T)

        guard let encoder = encoderModel else { throw VoiceError.modelNotLoaded }
        let inputArray = try MLMultiArray(shape: [1, T as NSNumber, 513], dataType: .float32)
        for i in 0..<transposed.count {
            inputArray[i] = NSNumber(value: transposed[i])
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "spectrogram": MLFeatureValue(multiArray: inputArray)
        ])
        let output = try encoder.prediction(from: input)
        guard let se = output.featureValue(for: "speaker_embedding")?.multiArrayValue else {
            throw VoiceError.predictionFailed
        }

        return (0..<se.count).map { se[$0].floatValue }
    }

    func convert(sourceURL: URL, sourceSE: [Float], targetSE: [Float],
                 progress: @escaping (String) -> Void) async throws -> URL {
        progress("Loading audio...")
        let samples = try loadAudio(url: sourceURL)

        progress("Computing spectrogram...")
        let spec = stft(samples: samples)
        let T = spec.count / 513

        progress("Converting voice...")
        guard let converter = converterModel else { throw VoiceError.modelNotLoaded }

        let specArray = try MLMultiArray(shape: [1, 513, T as NSNumber], dataType: .float32)
        for i in 0..<spec.count { specArray[i] = NSNumber(value: spec[i]) }

        let specLengths = try MLMultiArray(shape: [1], dataType: .float32)
        specLengths[0] = NSNumber(value: T)

        let srcSEArray = try MLMultiArray(shape: [1, 256, 1], dataType: .float32)
        for i in 0..<256 { srcSEArray[i] = NSNumber(value: sourceSE[i]) }

        let tgtSEArray = try MLMultiArray(shape: [1, 256, 1], dataType: .float32)
        for i in 0..<256 { tgtSEArray[i] = NSNumber(value: targetSE[i]) }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "spectrogram": MLFeatureValue(multiArray: specArray),
            "spec_lengths": MLFeatureValue(multiArray: specLengths),
            "source_speaker": MLFeatureValue(multiArray: srcSEArray),
            "target_speaker": MLFeatureValue(multiArray: tgtSEArray),
        ])

        let output = try converter.prediction(from: input)
        guard let audioOut = output.featureValue(for: "audio")?.multiArrayValue else {
            throw VoiceError.predictionFailed
        }

        progress("Saving audio...")
        let audioSamples = (0..<audioOut.count).map { audioOut[$0].floatValue }
        let outputURL = try saveWAV(samples: audioSamples, sampleRate: Self.sampleRate)

        progress("Done!")
        return outputURL
    }

    // MARK: - Audio I/O

    private func loadAudio(url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let originalFormat = file.processingFormat
        let frameCount = AVAudioFrameCount(file.length)

        guard let buffer = AVAudioPCMBuffer(pcmFormat: originalFormat, frameCapacity: frameCount) else {
            throw VoiceError.audioLoadFailed
        }
        try file.read(into: buffer)

        // Resample to 22050 Hz mono
        let targetFormat = AVAudioFormat(standardFormatWithSampleRate: Self.sampleRate, channels: 1)!
        guard let converter = AVAudioConverter(from: originalFormat, to: targetFormat) else {
            throw VoiceError.audioLoadFailed
        }

        let ratio = Self.sampleRate / originalFormat.sampleRate
        let outputFrameCount = AVAudioFrameCount(Double(frameCount) * ratio)
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outputFrameCount) else {
            throw VoiceError.audioLoadFailed
        }

        var isDone = false
        let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
            if isDone {
                outStatus.pointee = .noDataNow
                return nil
            }
            isDone = true
            outStatus.pointee = .haveData
            return buffer
        }

        try converter.convert(to: outputBuffer, error: nil, withInputFrom: inputBlock)

        guard let channelData = outputBuffer.floatChannelData else {
            throw VoiceError.audioLoadFailed
        }
        let count = Int(outputBuffer.frameLength)
        return Array(UnsafeBufferPointer(start: channelData[0], count: count))
    }

    private func saveWAV(samples: [Float], sampleRate: Double) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString + ".wav")

        let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!
        let file = try AVAudioFile(forWriting: url, settings: format.settings)

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)) else {
            throw VoiceError.audioSaveFailed
        }
        buffer.frameLength = AVAudioFrameCount(samples.count)

        let dst = buffer.floatChannelData![0]
        // Normalize to prevent clipping
        let maxVal = samples.map { abs($0) }.max() ?? 1.0
        let scale = maxVal > 1.0 ? 0.95 / maxVal : 1.0
        for i in 0..<samples.count {
            dst[i] = samples[i] * scale
        }

        try file.write(from: buffer)
        return url
    }

    // MARK: - STFT (matches OpenVoice: n_fft=1024, hop=256, win=1024, center=False, reflect pad)

    private func stft(samples: [Float]) -> [Float] {
        let padAmount = (Self.nFFT - Self.hopLength) / 2  // 384
        var padded = [Float](repeating: 0, count: samples.count + padAmount * 2)
        // Reflect padding
        for i in 0..<padAmount {
            padded[padAmount - 1 - i] = samples[min(i + 1, samples.count - 1)]
        }
        for i in 0..<samples.count {
            padded[padAmount + i] = samples[i]
        }
        for i in 0..<padAmount {
            padded[padAmount + samples.count + i] = samples[max(samples.count - 2 - i, 0)]
        }

        let freqBins = Self.nFFT / 2 + 1  // 513
        let numFrames = (padded.count - Self.nFFT) / Self.hopLength + 1

        // Periodic Hann window (matches torch.hann_window(periodic=True))
        var window = [Float](repeating: 0, count: Self.nFFT)
        for i in 0..<Self.nFFT {
            window[i] = 0.5 * (1.0 - cos(2.0 * .pi * Float(i) / Float(Self.nFFT)))
        }

        let log2n = vDSP_Length(log2(Float(Self.nFFT)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            return []
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        var result = [Float](repeating: 0, count: freqBins * numFrames)
        let halfN = Self.nFFT / 2

        for frame in 0..<numFrames {
            let start = frame * Self.hopLength
            var windowed = [Float](repeating: 0, count: Self.nFFT)
            vDSP_vmul(Array(padded[start..<start + Self.nFFT]), 1, window, 1, &windowed, 1, vDSP_Length(Self.nFFT))

            var realp = [Float](repeating: 0, count: halfN)
            var imagp = [Float](repeating: 0, count: halfN)

            realp.withUnsafeMutableBufferPointer { rBuf in
                imagp.withUnsafeMutableBufferPointer { iBuf in
                    var split = DSPSplitComplex(realp: rBuf.baseAddress!, imagp: iBuf.baseAddress!)

                    windowed.withUnsafeBufferPointer { wBuf in
                        wBuf.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: halfN) { complex in
                            vDSP_ctoz(complex, 2, &split, 1, vDSP_Length(halfN))
                        }
                    }

                    vDSP_fft_zrip(fftSetup, &split, 1, log2n, FFTDirection(FFT_FORWARD))

                    // vDSP forward FFT scales non-DC/Nyquist bins by 2.
                    // torch.stft(normalized=False) has no extra scaling.
                    // DC (realp[0]) and Nyquist (imagp[0]) are NOT doubled by vDSP.
                    let dc = rBuf[0]
                    result[0 * numFrames + frame] = sqrt(dc * dc + 1e-6)

                    let ny = iBuf[0]
                    result[(freqBins - 1) * numFrames + frame] = sqrt(ny * ny + 1e-6)

                    for k in 1..<(freqBins - 1) {
                        let r = rBuf[k] / 2.0  // undo vDSP 2x scaling
                        let im = iBuf[k] / 2.0
                        result[k * numFrames + frame] = sqrt(r * r + im * im + 1e-6)
                    }
                }
            }
        }

        return result
    }

    private func transposeSpec(_ spec: [Float], freqBins: Int, frames: Int) -> [Float] {
        // [freqBins, frames] → [frames, freqBins]
        var transposed = [Float](repeating: 0, count: spec.count)
        for f in 0..<freqBins {
            for t in 0..<frames {
                transposed[t * freqBins + f] = spec[f * frames + t]
            }
        }
        return transposed
    }
}

enum VoiceError: LocalizedError {
    case modelNotLoaded, predictionFailed, audioLoadFailed, audioSaveFailed

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded: return "Model not loaded"
        case .predictionFailed: return "Prediction failed"
        case .audioLoadFailed: return "Failed to load audio"
        case .audioSaveFailed: return "Failed to save audio"
        }
    }
}
