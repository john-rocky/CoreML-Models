import Foundation
import AVFoundation
import Accelerate

/// Renders NoteEvents to a WAV file using sine wave synthesis
enum NoteSynthesizer {
    private static let sampleRate: Double = 44100

    static func render(notes: [NoteEvent]) -> URL {
        let noteNames = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
        print("=== Synth: rendering \(notes.count) notes ===")
        for (i, n) in notes.prefix(10).enumerated() {
            let name = noteNames[n.midiPitch % 12] + "\(n.midiPitch / 12 - 1)"
            let freq = midiToFreq(n.midiPitch)
            print("  [\(i)] \(name) \(String(format: "%.0f", freq))Hz amp=\(String(format: "%.2f", n.amplitude)) \(String(format: "%.2f", NoteCreation.frameToTime(n.startFrame)))-\(String(format: "%.2f", NoteCreation.frameToTime(n.endFrame)))s")
        }

        // Find total duration
        let maxEndTime = notes.map { NoteCreation.frameToTime($0.endFrame) }.max() ?? 0
        let totalSamples = Int(maxEndTime * sampleRate) + Int(sampleRate * 0.5)  // + 0.5s tail
        print("Total duration: \(String(format: "%.1f", maxEndTime))s, \(totalSamples) samples")

        var buffer = [Float](repeating: 0, count: totalSamples)

        // Harmonics for piano-like timbre: fundamental + partials with decreasing amplitude
        let harmonics: [(partial: Float, relAmp: Float)] = [
            (1, 1.0), (2, 0.4), (3, 0.15), (4, 0.08)
        ]
        let attackMs = 0.005    // 5ms attack
        let releaseMs = 0.25    // 250ms release tail (extends beyond note end)

        for note in notes {
            let startTime = NoteCreation.frameToTime(note.startFrame)
            let endTime = NoteCreation.frameToTime(note.endFrame)
            let freq = midiToFreq(note.midiPitch)

            // Compress dynamic range and boost melody (higher pitches)
            // sqrt compression: 0.2 → 0.45, 0.65 → 0.81 (closer together)
            let compressedAmp = sqrtf(note.amplitude)
            // Boost notes above C4 (MIDI 60) to make melody audible
            let melodyBoost: Float = note.midiPitch >= 60 ? 1.5 : 1.0
            let amp = compressedAmp * melodyBoost * 0.15

            let startSample = Int(startTime * sampleRate)
            let releaseSamples = Int(releaseMs * sampleRate)
            let sustainEnd = min(Int(endTime * sampleRate), totalSamples)
            let renderEnd = min(sustainEnd + releaseSamples, totalSamples)
            let renderCount = renderEnd - startSample
            guard renderCount > 0 else { continue }

            let attackLen = min(Int(attackMs * sampleRate), renderCount / 2)
            let sustainLen = sustainEnd - startSample

            for i in 0..<renderCount {
                let sampleIdx = startSample + i
                guard sampleIdx < totalSamples else { break }

                // Envelope: attack → sustain → exponential release
                var envelope: Float
                if i < attackLen {
                    envelope = Float(i) / Float(max(attackLen, 1))
                } else if i < sustainLen {
                    envelope = 1.0
                } else {
                    let releaseProgress = Float(i - sustainLen) / Float(max(releaseSamples, 1))
                    envelope = expf(-4.0 * releaseProgress)  // exponential decay
                }

                // Additive synthesis with harmonics
                var sample: Float = 0
                let basePhase = Float(freq) * Float(i) / Float(sampleRate)
                for h in harmonics {
                    let hFreq = basePhase * h.partial
                    // Skip harmonics above Nyquist
                    guard hFreq * Float(sampleRate) / Float(i > 0 ? i : 1) < Float(sampleRate) / 2 else { continue }
                    sample += sinf(2.0 * Float.pi * hFreq) * h.relAmp
                }

                buffer[sampleIdx] += sample * amp * envelope
            }
        }

        // Clamp to [-1, 1]
        for i in 0..<buffer.count {
            buffer[i] = max(-1.0, min(1.0, buffer[i]))
        }

        // Write WAV
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("basic_pitch_synth.wav")
        writeWAV(samples: buffer, sampleRate: Int(sampleRate), to: url)
        return url
    }

    private static func midiToFreq(_ midi: Int) -> Double {
        return 440.0 * pow(2.0, Double(midi - 69) / 12.0)
    }

    private static func writeWAV(samples: [Float], sampleRate: Int, to url: URL) {
        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 1, interleaved: false)!
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)) else { return }
        buffer.frameLength = AVAudioFrameCount(samples.count)

        let dst = buffer.floatChannelData![0]
        samples.withUnsafeBufferPointer { src in
            dst.update(from: src.baseAddress!, count: samples.count)
        }

        try? FileManager.default.removeItem(at: url)
        guard let file = try? AVAudioFile(forWriting: url, settings: format.settings) else { return }
        try? file.write(from: buffer)
    }
}
