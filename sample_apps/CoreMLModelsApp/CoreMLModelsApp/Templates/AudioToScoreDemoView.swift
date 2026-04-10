import SwiftUI
import CoreML
import AVFoundation

/// Audio transcription: audio file → piano roll / note onset visualization.
/// Used by: Basic Pitch.
///
/// Expected manifest config:
/// ```
/// {
///   "sample_rate": 22050,
///   "window_size": 43844,
///   "hop_size": 256,
///   "n_bins": 88,
///   "onset_threshold": 0.5,
///   "note_threshold": 0.5,
///   "min_note": 21
/// }
/// ```
struct AudioToScoreDemoView: View {
    let model: ModelEntry

    @State private var inputURL: URL?
    @State private var pianoRollImage: UIImage?
    @State private var noteCount = 0
    @State private var isProcessing = false
    @State private var status = ""
    @State private var progress: Float = 0
    @State private var processingTime: Double?
    @State private var showingFilePicker = false
    @State private var player: AVAudioPlayer?
    @State private var isPlaying = false

    var body: some View {
        VStack(spacing: 0) {
            if let img = pianoRollImage {
                ScrollView(.horizontal, showsIndicators: true) {
                    Image(uiImage: img).resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(height: 300)
                }
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .padding(.horizontal)
                .frame(maxHeight: .infinity)

                HStack {
                    Text("\(noteCount) note onsets detected")
                        .font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    if let t = processingTime {
                        Text(String(format: "%.2fs", t)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    }
                }
                .padding(.horizontal)
            } else {
                VStack(spacing: 12) {
                    Image(systemName: "pianokeys").font(.system(size: 60)).foregroundStyle(.secondary)
                    Text("Select an audio file to transcribe to notes")
                        .multilineTextAlignment(.center).foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }

            VStack(spacing: 12) {
                if isProcessing {
                    ProgressView(value: Double(progress))
                    Text(status).font(.caption).foregroundStyle(.secondary)
                }

                if let url = inputURL {
                    HStack {
                        Text(url.lastPathComponent).font(.caption).lineLimit(1).foregroundStyle(.secondary)
                        Spacer()
                        Button { togglePlayback() } label: {
                            Image(systemName: isPlaying ? "stop.circle" : "play.circle").font(.title3)
                        }
                    }
                }

                HStack(spacing: 12) {
                    Button { showingFilePicker = true } label: {
                        Label("Select Audio", systemImage: "doc.badge.plus")
                    }.buttonStyle(.bordered)

                    Button {
                        Task { await runTranscription() }
                    } label: {
                        Label("Transcribe", systemImage: "music.note.list").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isProcessing || inputURL == nil)
                }
            }
            .padding()
        }
        .sheet(isPresented: $showingFilePicker) {
            AudioPickerView { url in inputURL = url; pianoRollImage = nil; noteCount = 0 }
        }
    }

    private func togglePlayback() {
        if isPlaying { player?.stop(); isPlaying = false; return }
        guard let url = inputURL else { return }
        player = try? AVAudioPlayer(contentsOf: url)
        player?.play(); isPlaying = true
    }

    private func runTranscription() async {
        guard let inputURL else { return }
        isProcessing = true; progress = 0; pianoRollImage = nil

        do {
            status = "Loading audio…"
            let sampleRate = model.configInt("sample_rate") ?? 22050
            let windowSize = model.configInt("window_size") ?? 43844
            let nBins = model.configInt("n_bins") ?? 88
            let onsetThreshold = model.configDouble("onset_threshold").map { Float($0) } ?? 0.5
            let noteThreshold = model.configDouble("note_threshold").map { Float($0) } ?? 0.5

            // Read and resample audio
            let audioFile = try AVAudioFile(forReading: inputURL)
            let format = AVAudioFormat(standardFormatWithSampleRate: Double(sampleRate), channels: 1)!
            let frameCount = AVAudioFrameCount(audioFile.length)
            let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)!
            try audioFile.read(into: buffer)
            guard let samples = buffer.floatChannelData?[0] else {
                isProcessing = false; status = "Failed to read"; return
            }
            let totalSamples = Int(buffer.frameLength)

            // Peak normalize
            var peak: Float = 0
            for i in 0..<totalSamples { peak = max(peak, abs(samples[i])) }
            if peak > 0 { for i in 0..<totalSamples { samples[i] *= 0.98 / peak } }

            status = "Loading model…"
            let mlModel = try await ModelLoader.loadPrimary(for: model)

            let start = CFAbsoluteTimeGetCurrent()

            // Window and process
            let overlapFrames = 30
            let hopSize = model.configInt("hop_size") ?? 256
            let stride = windowSize - overlapFrames * hopSize * 2
            let numWindows = max(1, (totalSamples + stride - 1) / stride)

            var allNotes: [[Float]] = []  // [timeFrames][nBins]
            var allOnsets: [[Float]] = []

            for w in 0..<numWindows {
                let windowStart = w * stride
                status = "Processing window \(w + 1)/\(numWindows)…"
                await MainActor.run { progress = Float(w) / Float(numWindows) }

                let inputArr = try MLMultiArray(shape: [1, NSNumber(value: windowSize), 1], dataType: .float32)
                let inPtr = inputArr.dataPointer.assumingMemoryBound(to: Float.self)
                for i in 0..<windowSize {
                    let srcIdx = windowStart + i
                    inPtr[i] = srcIdx < totalSamples ? samples[srcIdx] : 0
                }

                let inputName = mlModel.modelDescription.inputDescriptionsByName.keys.first ?? "input_2"
                let input = try MLDictionaryFeatureProvider(dictionary: [inputName: inputArr])
                let output = try await mlModel.prediction(from: input)

                // Extract notes and onsets (Identity_1, Identity_2)
                let arrays = output.featureNames.compactMap { output.featureValue(for: $0)?.multiArrayValue }

                for arr in arrays {
                    let shape = arr.shape.map { $0.intValue }
                    guard shape.count == 3 && shape[2] >= nBins else { continue }
                    let frames = shape[1], bins = min(shape[2], nBins)
                    let strides = arr.strides.map { $0.intValue }

                    var frameData: [[Float]] = []
                    let trimStart = w > 0 ? overlapFrames : 0
                    let trimEnd = w < numWindows - 1 ? frames - overlapFrames : frames

                    for f in trimStart..<trimEnd {
                        var row = [Float](repeating: 0, count: bins)
                        for b in 0..<bins {
                            row[b] = ImageUtils.readFloat(arr, at: f * strides[1] + b * strides[2])
                        }
                        frameData.append(row)
                    }

                    if bins == nBins {
                        if allNotes.isEmpty { allNotes = frameData }
                        else { allOnsets = frameData }
                    }
                }
            }

            let elapsed = CFAbsoluteTimeGetCurrent() - start

            // Render piano roll
            let (image, count) = renderPianoRoll(
                notes: allNotes, onsets: allOnsets, nBins: nBins,
                noteThreshold: noteThreshold, onsetThreshold: onsetThreshold
            )

            await MainActor.run {
                pianoRollImage = image; noteCount = count
                processingTime = elapsed; progress = 1.0
                isProcessing = false; status = ""
            }
        } catch {
            await MainActor.run { isProcessing = false; status = "Error: \(error.localizedDescription)" }
        }
    }

    private func renderPianoRoll(notes: [[Float]], onsets: [[Float]], nBins: Int,
                                  noteThreshold: Float, onsetThreshold: Float) -> (UIImage?, Int) {
        let timeFrames = notes.count
        guard timeFrames > 0 else { return (nil, 0) }

        let scale = 2
        let w = timeFrames * scale
        let h = nBins * scale
        var pixels = [UInt8](repeating: 0, count: w * h * 4)  // Black background

        // Set alpha
        for i in stride(from: 3, to: pixels.count, by: 4) { pixels[i] = 255 }

        var onsetCount = 0

        for t in 0..<timeFrames {
            for b in 0..<nBins {
                let noteVal = t < notes.count && b < notes[t].count ? notes[t][b] : 0
                let onsetVal = t < onsets.count && b < onsets[t].count ? onsets[t][b] : 0

                let y = (nBins - 1 - b) * scale  // Flip vertically (low notes at bottom)

                if onsetVal > onsetThreshold {
                    onsetCount += 1
                    // Bright red for onsets
                    for dy in 0..<scale { for dx in 0..<scale {
                        let idx = ((y + dy) * w + t * scale + dx) * 4
                        guard idx + 3 < pixels.count else { continue }
                        pixels[idx] = 255; pixels[idx+1] = 60; pixels[idx+2] = 60
                    }}
                } else if noteVal > noteThreshold {
                    // Blue-green for sustained notes
                    let intensity = UInt8(min(255, noteVal * 255))
                    for dy in 0..<scale { for dx in 0..<scale {
                        let idx = ((y + dy) * w + t * scale + dx) * 4
                        guard idx + 3 < pixels.count else { continue }
                        pixels[idx] = 30; pixels[idx+1] = intensity; pixels[idx+2] = UInt8(min(255, intensity + 60))
                    }}
                }
            }
        }

        let image = ImageUtils.makeRGBA(pixels: pixels, width: w, height: h)
        return (image, onsetCount)
    }
}
