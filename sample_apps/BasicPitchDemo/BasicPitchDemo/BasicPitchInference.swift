import Foundation
import CoreML
import AVFoundation
import Accelerate

// MARK: - Constants (matching basic_pitch/constants.py)

enum BasicPitchConstants {
    static let audioSampleRate: Double = 22050
    static let fftHop: Int = 256
    static let notesBinsPerSemitone: Int = 1
    static let contoursBinsPerSemitone: Int = 3
    static let annotationsBaseFrequency: Double = 27.5  // A0
    static let annotationsNSemitones: Int = 88
    static let nFreqBinsNotes: Int = 88
    static let nFreqBinsContours: Int = 264  // 88 * 3
    static let audioWindowLength: Int = 2  // seconds
    static let annotationsFPS: Int = 22050 / 256  // 86
    static let annotNFrames: Int = 172  // 86 * 2
    static let audioNSamples: Int = 22050 * 2 - 256  // 43844
    static let defaultOverlappingFrames: Int = 30
    static let midiOffset: Int = 21  // freq_idx 0 = MIDI note 21 (A0)
}

// MARK: - Model Output

struct BasicPitchOutput {
    let notes: [[Float]]      // (frames, 88)
    let onsets: [[Float]]     // (frames, 88)
    let contours: [[Float]]  // (frames, 264)
}

// MARK: - Inference Engine

class BasicPitchInference {
    private var model: MLModel?

    var isReady: Bool { model != nil }

    func loadModel() async throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all

        guard let modelURL = Bundle.main.url(forResource: "nmp", withExtension: "mlmodelc")
                ?? Bundle.main.url(forResource: "nmp", withExtension: "mlpackage") else {
            throw BasicPitchError.modelNotLoaded
        }

        let compiledURL: URL
        if modelURL.pathExtension == "mlmodelc" {
            compiledURL = modelURL
        } else {
            compiledURL = try await MLModel.compileModel(at: modelURL)
        }
        self.model = try MLModel(contentsOf: compiledURL, configuration: config)
    }

    /// Run the full pipeline off the main thread
    nonisolated func transcribe(audioURL: URL, progress: @escaping @Sendable (String) -> Void) async throws -> BasicPitchOutput {
        guard let model = model else {
            throw BasicPitchError.modelNotLoaded
        }

        return try await Task.detached(priority: .userInitiated) {
            progress("Loading audio...")
            let audioSamples = try self.loadAudio(url: audioURL)

            // Debug: audio stats
            let audioMin = audioSamples.min() ?? 0
            let audioMax = audioSamples.max() ?? 0
            let rms = sqrtf(audioSamples.reduce(0) { $0 + $1 * $1 } / Float(audioSamples.count))
            print("=== Audio Debug ===")
            print("Samples: \(audioSamples.count) (\(String(format: "%.1f", Double(audioSamples.count) / 22050.0))s)")
            print("Range: [\(audioMin), \(audioMax)]  RMS: \(rms)")
            print("First 10: \(audioSamples.prefix(10).map { String(format: "%.4f", $0) })")

            progress("Preparing windows...")
            let windows = self.windowAudio(samples: audioSamples)
            print("Windows: \(windows.count), each \(windows.first?.count ?? 0) samples")

            progress("Running inference (0/\(windows.count))...")
            var allNotes: [[[Float]]] = []
            var allOnsets: [[[Float]]] = []
            var allContours: [[[Float]]] = []

            for (i, window) in windows.enumerated() {
                progress("Running inference (\(i + 1)/\(windows.count))...")

                let input = try self.createMLMultiArray(from: window)
                let provider = try MLDictionaryFeatureProvider(dictionary: ["input_2": MLFeatureValue(multiArray: input)])
                let result = try model.prediction(from: provider)

                let noteArray = result.featureValue(for: "Identity_1")!.multiArrayValue!
                let onsetArray = result.featureValue(for: "Identity_2")!.multiArrayValue!
                let contourArray = result.featureValue(for: "Identity")!.multiArrayValue!

                if i == 0 {
                    print("=== MLMultiArray Debug ===")
                    print("Note shape: \(noteArray.shape), strides: \(noteArray.strides)")
                    print("Onset shape: \(onsetArray.shape), strides: \(onsetArray.strides)")
                    print("Contour shape: \(contourArray.shape), strides: \(contourArray.strides)")
                    print("Input strides: \(try self.createMLMultiArray(from: window).strides)")
                }

                allNotes.append(self.multiArrayTo2D(noteArray, rows: BasicPitchConstants.annotNFrames, cols: BasicPitchConstants.nFreqBinsNotes))
                allOnsets.append(self.multiArrayTo2D(onsetArray, rows: BasicPitchConstants.annotNFrames, cols: BasicPitchConstants.nFreqBinsNotes))
                allContours.append(self.multiArrayTo2D(contourArray, rows: BasicPitchConstants.annotNFrames, cols: BasicPitchConstants.nFreqBinsContours))
            }

            progress("Unwrapping output...")
            let nOlap = BasicPitchConstants.defaultOverlappingFrames / 2  // 15

            let unwrappedNotes = self.unwrapOutput(allNotes, nOlap: nOlap)
            let unwrappedOnsets = self.unwrapOutput(allOnsets, nOlap: nOlap)
            let unwrappedContours = self.unwrapOutput(allContours, nOlap: nOlap)

            progress("Detecting notes...")

            return BasicPitchOutput(notes: unwrappedNotes, onsets: unwrappedOnsets, contours: unwrappedContours)
        }.value
    }

    // MARK: - Audio Loading

    private func loadAudio(url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let targetFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: BasicPitchConstants.audioSampleRate, channels: 1, interleaved: false)!

        guard let converter = AVAudioConverter(from: file.processingFormat, to: targetFormat) else {
            throw BasicPitchError.audioConversionFailed
        }

        let frameCapacity = AVAudioFrameCount(Double(file.length) * BasicPitchConstants.audioSampleRate / file.fileFormat.sampleRate) + 1024
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: frameCapacity) else {
            throw BasicPitchError.audioConversionFailed
        }

        let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
            let readBuffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 4096)!
            do {
                try file.read(into: readBuffer)
                if readBuffer.frameLength == 0 {
                    outStatus.pointee = .endOfStream
                    return nil
                }
                outStatus.pointee = .haveData
                return readBuffer
            } catch {
                outStatus.pointee = .endOfStream
                return nil
            }
        }

        var conversionError: NSError?
        converter.convert(to: outputBuffer, error: &conversionError, withInputFrom: inputBlock)
        if let error = conversionError {
            throw error
        }

        let channelData = outputBuffer.floatChannelData![0]
        var samples = Array(UnsafeBufferPointer(start: channelData, count: Int(outputBuffer.frameLength)))

        // Peak normalize to match librosa's typical output (~0.98)
        // iOS Core Audio MP3 decoder produces slightly hotter samples than librosa,
        // which causes the basic_pitch model to detect different notes.
        var peak: Float = 0
        vDSP_maxmgv(samples, 1, &peak, vDSP_Length(samples.count))
        if peak > 0 {
            let targetPeak: Float = 0.98
            var scale = targetPeak / peak
            vDSP_vsmul(samples, 1, &scale, &samples, 1, vDSP_Length(samples.count))
        }

        return samples
    }

    // MARK: - Windowing

    private func windowAudio(samples: [Float]) -> [[Float]] {
        let overlapLen = BasicPitchConstants.defaultOverlappingFrames * BasicPitchConstants.fftHop  // 7680
        let hopSize = BasicPitchConstants.audioNSamples - overlapLen  // 36164

        // Prepend half overlap in zeros
        let paddedSamples = [Float](repeating: 0, count: overlapLen / 2) + samples

        var windows: [[Float]] = []
        var offset = 0
        while offset < paddedSamples.count {
            let end = min(offset + BasicPitchConstants.audioNSamples, paddedSamples.count)
            var window = Array(paddedSamples[offset..<end])
            if window.count < BasicPitchConstants.audioNSamples {
                window += [Float](repeating: 0, count: BasicPitchConstants.audioNSamples - window.count)
            }
            windows.append(window)
            offset += hopSize
        }

        return windows
    }

    // MARK: - Output Unwrapping

    private func unwrapOutput(_ batchedOutput: [[[Float]]], nOlap: Int) -> [[Float]] {
        var result: [[Float]] = []
        result.reserveCapacity(batchedOutput.count * (BasicPitchConstants.annotNFrames - 2 * nOlap))
        for window in batchedOutput {
            let trimmed = Array(window[nOlap..<(window.count - nOlap)])
            result.append(contentsOf: trimmed)
        }
        return result
    }

    // MARK: - Helpers

    private func createMLMultiArray(from window: [Float]) throws -> MLMultiArray {
        let input = try MLMultiArray(shape: [1, NSNumber(value: BasicPitchConstants.audioNSamples), 1], dataType: .float32)
        // Use strides for correct layout regardless of compute unit
        let strides = input.strides.map { $0.intValue }
        let ptr = input.dataPointer.bindMemory(to: Float.self, capacity: input.count)
        for j in 0..<window.count {
            ptr[j * strides[1]] = window[j]
        }
        return input
    }

    /// Read MLMultiArray using strides (safe for Neural Engine output)
    private func multiArrayTo2D(_ array: MLMultiArray, rows: Int, cols: Int) -> [[Float]] {
        let strides = array.strides.map { $0.intValue }
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
        // Shape is (1, rows, cols), strides are (s0, s1, s2)
        let s1 = strides.count >= 3 ? strides[1] : cols
        let s2 = strides.count >= 3 ? strides[2] : 1
        var result = [[Float]]()
        result.reserveCapacity(rows)
        for r in 0..<rows {
            var row = [Float](repeating: 0, count: cols)
            for c in 0..<cols {
                row[c] = ptr[r * s1 + c * s2]
            }
            result.append(row)
        }
        return result
    }
}

// MARK: - Errors

enum BasicPitchError: LocalizedError {
    case modelNotLoaded
    case audioConversionFailed
    case invalidOutput

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded: return "Model not loaded. Add nmp.mlpackage to the project."
        case .audioConversionFailed: return "Failed to convert audio to 22050Hz mono"
        case .invalidOutput: return "Invalid model output"
        }
    }
}
