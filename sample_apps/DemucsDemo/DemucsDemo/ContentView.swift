import SwiftUI
import UIKit
import CoreML
import AVFoundation
import UniformTypeIdentifiers
import Accelerate

// MARK: - HTDemucs Audio Source Separation Demo

enum Stem: String, CaseIterable, Identifiable {
    case drums = "Drums"
    case bass = "Bass"
    case vocals = "Vocals"
    case other = "Other"

    var id: String { rawValue }

    // Index in model output — matches Python's model.sources: [drums, bass, other, vocals]
    var modelIndex: Int {
        switch self {
        case .drums: return 0
        case .bass: return 1
        case .other: return 2
        case .vocals: return 3
        }
    }

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

enum SeparationMode {
    case preview   // Single ~7.8s segment
    case full      // Entire track with overlap-add
}

struct ContentView: View {
    @StateObject private var viewModel = DemucsViewModel()

    var body: some View {
        NavigationStack {
            ScrollView {
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

                    // Separation buttons
                    if viewModel.audioURL != nil && !viewModel.isSeparated {
                        VStack(spacing: 10) {
                            Button(action: { viewModel.separate(mode: .preview) }) {
                                HStack {
                                    if viewModel.isProcessing {
                                        ProgressView().tint(.white)
                                    } else {
                                        Image(systemName: "scissors")
                                    }
                                    Text(viewModel.isProcessing ? viewModel.statusMessage : "Quick Preview (~8s)")
                                }
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(viewModel.isProcessing ? Color.gray : Color.accentColor)
                                .foregroundColor(.white)
                                .cornerRadius(12)
                            }
                            .disabled(viewModel.isProcessing)

                            Button(action: { viewModel.separate(mode: .full) }) {
                                HStack {
                                    if viewModel.isProcessing {
                                        ProgressView().tint(.white)
                                    } else {
                                        Image(systemName: "wand.and.stars")
                                    }
                                    Text(viewModel.isProcessing ? viewModel.statusMessage : "Separate Full Track")
                                }
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(viewModel.isProcessing ? Color.gray : Color(.systemIndigo))
                                .foregroundColor(.white)
                                .cornerRadius(12)
                            }
                            .disabled(viewModel.isProcessing)
                        }
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
                            // Original playback
                            HStack(spacing: 16) {
                                Image(systemName: "music.note.list")
                                    .font(.title3)
                                    .foregroundColor(.primary)
                                    .frame(width: 30)
                                Text("Original")
                                    .font(.body)
                                    .fontWeight(.medium)
                                Spacer()
                                Button(action: {
                                    if viewModel.isPlayingOriginal { viewModel.stopPlayback() }
                                    else { viewModel.playOriginal() }
                                }) {
                                    Image(systemName: viewModel.isPlayingOriginal ? "stop.circle.fill" : "play.circle.fill")
                                        .font(.title)
                                        .foregroundColor(viewModel.isPlayingOriginal ? .red : .primary)
                                }
                            }
                            .padding()
                            .background(
                                RoundedRectangle(cornerRadius: 12)
                                    .fill(viewModel.isPlayingOriginal ? Color.primary.opacity(0.1) : Color(.systemGray6))
                            )

                            Text("Separated Stems")
                                .font(.headline)
                                .frame(maxWidth: .infinity, alignment: .leading)

                            ForEach(Stem.allCases) { stem in
                                StemPlayerView(
                                    stem: stem,
                                    isPlaying: viewModel.playingStem == stem,
                                    onPlay: { viewModel.playStem(stem) },
                                    onStop: { viewModel.stopPlayback() },
                                    onShare: { viewModel.shareStem(stem) }
                                )
                            }
                        }
                        .padding()
                    }
                }
            }
            .navigationTitle("Demucs Separator")
            .sheet(isPresented: $viewModel.showFilePicker) {
                AudioFilePickerView(audioURL: $viewModel.audioURL)
            }
            .sheet(isPresented: $viewModel.showShareSheet) {
                if let url = viewModel.shareURL {
                    ShareSheet(items: [url])
                }
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
    let onShare: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: stem.icon)
                .font(.title3)
                .foregroundColor(stem.color)
                .frame(width: 30)

            Text(stem.rawValue)
                .font(.body)
                .fontWeight(.medium)

            Spacer()

            Button(action: onShare) {
                Image(systemName: "square.and.arrow.up")
                    .font(.body)
                    .foregroundColor(.secondary)
            }

            Button(action: {
                if isPlaying { onStop() } else { onPlay() }
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

// MARK: - Share Sheet

struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
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

// MARK: - STFT / iSTFT Signal Processing

private enum DSP {
    static let fftSize = 4096
    static let hopSize = 1024
    static let numBins = 2048   // fftSize / 2
    static let numFrames = 336
    static let segmentLength = 343980
    static let sampleRate: Double = 44100
    static let overlap: Double = 0.25

    // Periodic Hann window (matches PyTorch's hann_window with periodic=True)
    static let window: [Float] = (0..<fftSize).map {
        Float(0.5 * (1.0 - cos(2.0 * .pi * Double($0) / Double(fftSize))))
    }

    /// Reflect-pad a signal with asymmetric left/right amounts (matches PyTorch's F.pad with mode='reflect')
    static func reflectPad(signal: [Float], left: Int, right: Int) -> [Float] {
        let n = signal.count
        var padded = [Float](repeating: 0, count: n + left + right)
        for i in 0..<left {
            padded[i] = signal[left - i]
        }
        signal.withUnsafeBufferPointer { src in
            padded.withUnsafeMutableBufferPointer { dst in
                memcpy(dst.baseAddress! + left, src.baseAddress!, n * MemoryLayout<Float>.size)
            }
        }
        for i in 0..<right {
            padded[left + n + i] = signal[n - 2 - i]
        }
        return padded
    }

    /// Load entire audio file as stereo Float32 arrays, resampled to 44100 Hz
    static func loadFullAudio(url: URL) throws -> (left: [Float], right: [Float]) {
        _ = url.startAccessingSecurityScopedResource()
        defer { url.stopAccessingSecurityScopedResource() }

        let sourceFile = try AVAudioFile(forReading: url)
        let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 2,
            interleaved: false
        )!

        let totalFrames = AVAudioFrameCount(
            Double(sourceFile.length) * sampleRate / sourceFile.processingFormat.sampleRate
        )
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: totalFrames) else {
            throw DemucsError.processingFailed("Failed to create audio buffer")
        }

        if sourceFile.processingFormat.sampleRate == sampleRate && sourceFile.processingFormat.channelCount == 2 {
            try sourceFile.read(into: outputBuffer)
        } else {
            guard let converter = AVAudioConverter(from: sourceFile.processingFormat, to: targetFormat) else {
                throw DemucsError.processingFailed("Cannot convert audio format")
            }
            let srcBuffer = AVAudioPCMBuffer(pcmFormat: sourceFile.processingFormat, frameCapacity: AVAudioFrameCount(sourceFile.length))!
            try sourceFile.read(into: srcBuffer)
            var error: NSError?
            converter.convert(to: outputBuffer, error: &error) { _, outStatus in
                outStatus.pointee = .haveData
                return srcBuffer
            }
            if let error { throw error }
        }

        let count = Int(outputBuffer.frameLength)
        let leftPtr = outputBuffer.floatChannelData![0]
        let rightCh = outputBuffer.format.channelCount > 1 ? 1 : 0
        let rightPtr = outputBuffer.floatChannelData![rightCh]

        let left = Array(UnsafeBufferPointer(start: leftPtr, count: count))
        let right = Array(UnsafeBufferPointer(start: rightPtr, count: count))
        return (left, right)
    }

    /// Forward STFT using vDSP. Frame count derived from signal length.
    static func forwardSTFT(signal: [Float]) -> (real: [Float], imag: [Float], frames: Int) {
        let log2n = vDSP_Length(log2(Float(fftSize)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else { return ([], [], 0) }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        let halfN = fftSize / 2
        let totalFrames = max(1, (signal.count - fftSize) / hopSize + 1)
        var allReal = [Float](repeating: 0, count: numBins * totalFrames)
        var allImag = [Float](repeating: 0, count: numBins * totalFrames)
        var frame = [Float](repeating: 0, count: fftSize)
        var rp = [Float](repeating: 0, count: halfN)
        var ip = [Float](repeating: 0, count: halfN)

        for f in 0..<totalFrames {
            let start = f * hopSize
            let avail = min(fftSize, max(0, signal.count - start))
            vDSP_vclr(&frame, 1, vDSP_Length(fftSize))
            if avail > 0 {
                signal.withUnsafeBufferPointer { buf in
                    frame.withUnsafeMutableBufferPointer { dst in
                        memcpy(dst.baseAddress!, buf.baseAddress! + start, avail * MemoryLayout<Float>.size)
                    }
                }
            }

            vDSP_vmul(frame, 1, window, 1, &frame, 1, vDSP_Length(fftSize))

            frame.withUnsafeBufferPointer { src in
                src.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: halfN) { complexPtr in
                    rp.withUnsafeMutableBufferPointer { rpBuf in
                        ip.withUnsafeMutableBufferPointer { ipBuf in
                            var sc = DSPSplitComplex(realp: rpBuf.baseAddress!, imagp: ipBuf.baseAddress!)
                            vDSP_ctoz(complexPtr, 2, &sc, 1, vDSP_Length(halfN))
                        }
                    }
                }
            }

            rp.withUnsafeMutableBufferPointer { rpBuf in
                ip.withUnsafeMutableBufferPointer { ipBuf in
                    var sc = DSPSplitComplex(realp: rpBuf.baseAddress!, imagp: ipBuf.baseAddress!)
                    vDSP_fft_zrip(fftSetup, &sc, 1, log2n, FFTDirection(kFFTDirection_Forward))
                }
            }

            allReal[f] = rp[0] * 0.5
            allImag[f] = 0
            for k in 1..<numBins {
                let idx = k * totalFrames + f
                allReal[idx] = rp[k] * 0.5
                allImag[idx] = ip[k] * 0.5
            }
        }

        return (allReal, allImag, totalFrames)
    }

    /// Compute spectral input matching Python's _spec + _magnitude with cac=True.
    static func computeSpectralInput(left: [Float], right: [Float]) -> [Float] {
        let le = numFrames
        let specPadLeft = hopSize / 2 * 3
        let specPadRight = specPadLeft + le * hopSize - segmentLength

        let channelSize = numBins * le
        var result = [Float](repeating: 0, count: 4 * channelSize)

        for (ch, signal) in [left, right].enumerated() {
            let padded = reflectPad(signal: signal, left: specPadLeft, right: specPadRight)
            let (real, imag, _) = forwardSTFT(signal: padded)
            let realCh = ch * 2
            let imagCh = ch * 2 + 1
            result.withUnsafeMutableBufferPointer { dst in
                real.withUnsafeBufferPointer { src in
                    memcpy(dst.baseAddress! + realCh * channelSize, src.baseAddress!, channelSize * MemoryLayout<Float>.size)
                }
                imag.withUnsafeBufferPointer { src in
                    memcpy(dst.baseAddress! + imagCh * channelSize, src.baseAddress!, channelSize * MemoryLayout<Float>.size)
                }
            }
        }

        // Normalize to match Python's torch.stft(normalized=True)
        var normScale = Float(1.0 / sqrt(Double(fftSize)))
        vDSP_vsmul(result, 1, &normScale, &result, 1, vDSP_Length(result.count))

        return result
    }

    /// Inverse STFT with overlap-add. Frame count derived from input array size.
    static func inverseSTFT(real: [Float], imag: [Float], outputLength: Int) -> [Float] {
        let log2n = vDSP_Length(log2(Float(fftSize)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else { return [] }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        let halfN = fftSize / 2
        let totalFrames = real.count / numBins
        var output = [Float](repeating: 0, count: outputLength)
        var windowSum = [Float](repeating: 0, count: outputLength)
        var rp = [Float](repeating: 0, count: halfN)
        var ip = [Float](repeating: 0, count: halfN)
        var frame = [Float](repeating: 0, count: fftSize)

        for f in 0..<totalFrames {
            rp[0] = real[f] * 2.0
            ip[0] = 0
            for k in 1..<numBins {
                let idx = k * totalFrames + f
                rp[k] = real[idx] * 2.0
                ip[k] = imag[idx] * 2.0
            }

            rp.withUnsafeMutableBufferPointer { rpBuf in
                ip.withUnsafeMutableBufferPointer { ipBuf in
                    var sc = DSPSplitComplex(realp: rpBuf.baseAddress!, imagp: ipBuf.baseAddress!)
                    vDSP_fft_zrip(fftSetup, &sc, 1, log2n, FFTDirection(kFFTDirection_Inverse))
                }
            }

            rp.withUnsafeBufferPointer { rpBuf in
                ip.withUnsafeBufferPointer { ipBuf in
                    var sc = DSPSplitComplex(
                        realp: UnsafeMutablePointer(mutating: rpBuf.baseAddress!),
                        imagp: UnsafeMutablePointer(mutating: ipBuf.baseAddress!)
                    )
                    frame.withUnsafeMutableBufferPointer { dst in
                        dst.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: halfN) { complexPtr in
                            vDSP_ztoc(&sc, 1, complexPtr, 2, vDSP_Length(halfN))
                        }
                    }
                }
            }

            var scale: Float = 1.0 / Float(2 * fftSize)
            vDSP_vsmul(frame, 1, &scale, &frame, 1, vDSP_Length(fftSize))
            vDSP_vmul(frame, 1, window, 1, &frame, 1, vDSP_Length(fftSize))

            let start = f * hopSize
            for i in 0..<fftSize {
                let idx = start + i
                if idx < outputLength {
                    output[idx] += frame[i]
                    windowSum[idx] += window[i] * window[i]
                }
            }
        }

        for i in 0..<outputLength {
            if windowSum[i] > 1e-8 {
                output[i] /= windowSum[i]
            }
        }
        return output
    }

    /// Inverse STFT matching Python's _ispec for a single mono channel.
    static func inverseSpec(real: [Float], imag: [Float]) -> [Float] {
        let le = numFrames
        let totalFrames = le + 4
        let specPad = hopSize / 2 * 3
        let centerPad = fftSize / 2

        var paddedReal = [Float](repeating: 0, count: numBins * totalFrames)
        var paddedImag = [Float](repeating: 0, count: numBins * totalFrames)
        for bin in 0..<numBins {
            let srcOffset = bin * le
            let dstOffset = bin * totalFrames + 2
            real.withUnsafeBufferPointer { src in
                paddedReal.withUnsafeMutableBufferPointer { dst in
                    memcpy(dst.baseAddress! + dstOffset, src.baseAddress! + srcOffset, le * MemoryLayout<Float>.size)
                }
            }
            imag.withUnsafeBufferPointer { src in
                paddedImag.withUnsafeMutableBufferPointer { dst in
                    memcpy(dst.baseAddress! + dstOffset, src.baseAddress! + srcOffset, le * MemoryLayout<Float>.size)
                }
            }
        }

        let rawLen = (totalFrames - 1) * hopSize + fftSize
        let rawOutput = inverseSTFT(real: paddedReal, imag: paddedImag, outputLength: rawLen)
        let trimStart = centerPad + specPad
        return Array(rawOutput[trimStart..<trimStart + segmentLength])
    }

    /// Extract a 2D channel from a 4D MLMultiArray, respecting strides
    static func extractChannel(from array: MLMultiArray, batch: Int, channel: Int, height: Int, width: Int) -> [Float] {
        let strides = array.strides.map { $0.intValue }
        let baseOffset = batch * strides[0] + channel * strides[1]
        let hStride = strides[2]
        let wStride = strides[3]
        var result = [Float](repeating: 0, count: height * width)

        if array.dataType == .float32 {
            let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
            if wStride == 1 {
                result.withUnsafeMutableBufferPointer { dst in
                    for h in 0..<height {
                        memcpy(dst.baseAddress! + h * width, ptr + baseOffset + h * hStride, width * 4)
                    }
                }
            } else {
                for h in 0..<height {
                    for w in 0..<width {
                        result[h * width + w] = ptr[baseOffset + h * hStride + w * wStride]
                    }
                }
            }
        } else if array.dataType == .float16 {
            let ptr = array.dataPointer.assumingMemoryBound(to: UInt16.self)
            if wStride == 1 {
                for h in 0..<height {
                    let rowStart = baseOffset + h * hStride
                    var srcBuf = vImage_Buffer(
                        data: UnsafeMutablePointer(mutating: ptr + rowStart),
                        height: 1, width: vImagePixelCount(width), rowBytes: width * 2)
                    result.withUnsafeMutableBufferPointer { dst in
                        var dstBuf = vImage_Buffer(
                            data: dst.baseAddress! + h * width,
                            height: 1, width: vImagePixelCount(width), rowBytes: width * 4)
                        vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, 0)
                    }
                }
            } else {
                for h in 0..<height {
                    for w in 0..<width {
                        result[h * width + w] = array[[batch, channel, h, w] as [NSNumber]].floatValue
                    }
                }
            }
        }
        return result
    }

    /// Extract a 1D channel from a 3D MLMultiArray, respecting strides
    static func extractChannel1D(from array: MLMultiArray, batch: Int, channel: Int, width: Int) -> [Float] {
        let strides = array.strides.map { $0.intValue }
        let baseOffset = batch * strides[0] + channel * strides[1]
        let wStride = strides[2]
        var result = [Float](repeating: 0, count: width)

        if array.dataType == .float32 {
            let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
            if wStride == 1 {
                result.withUnsafeMutableBufferPointer { dst in
                    memcpy(dst.baseAddress!, ptr + baseOffset, width * MemoryLayout<Float>.size)
                }
            } else {
                for w in 0..<width { result[w] = ptr[baseOffset + w * wStride] }
            }
        } else if array.dataType == .float16 {
            let ptr = array.dataPointer.assumingMemoryBound(to: UInt16.self)
            if wStride == 1 {
                var srcBuf = vImage_Buffer(
                    data: UnsafeMutablePointer(mutating: ptr + baseOffset),
                    height: 1, width: vImagePixelCount(width), rowBytes: width * 2)
                result.withUnsafeMutableBufferPointer { dst in
                    var dstBuf = vImage_Buffer(
                        data: dst.baseAddress!, height: 1,
                        width: vImagePixelCount(width), rowBytes: width * 4)
                    vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, 0)
                }
            } else {
                for w in 0..<width { result[w] = array[[batch, channel, w] as [NSNumber]].floatValue }
            }
        }
        return result
    }

    /// Write stereo Float32 audio to a WAV file
    static func writeWAV(left: [Float], right: [Float], to url: URL) throws {
        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 2, interleaved: false)!
        let count = AVAudioFrameCount(min(left.count, right.count))
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: count) else {
            throw DemucsError.processingFailed("Failed to create output buffer")
        }
        buffer.frameLength = count
        left.withUnsafeBufferPointer { src in
            buffer.floatChannelData![0].update(from: src.baseAddress!, count: Int(count))
        }
        right.withUnsafeBufferPointer { src in
            buffer.floatChannelData![1].update(from: src.baseAddress!, count: Int(count))
        }
        let file = try AVAudioFile(forWriting: url, settings: format.settings)
        try file.write(from: buffer)
    }

    // MARK: - Segment Processing

    /// Process a single segment through the model, returning per-stem stereo audio.
    static func processSegment(left: [Float], right: [Float], model: MLModel) throws -> [(left: [Float], right: [Float])] {
        let spectralData = computeSpectralInput(left: left, right: right)

        let spectral = try MLMultiArray(shape: [1, 4, 2048, 336], dataType: .float32)
        let spectralCount = 4 * numBins * numFrames
        let spectralPtr = spectral.dataPointer.bindMemory(to: Float.self, capacity: spectralCount)
        let spectralStrides = spectral.strides.map { $0.intValue }
        if spectralStrides.last == 1 && spectralStrides[2] == numFrames {
            spectralData.withUnsafeBufferPointer { src in
                memcpy(spectralPtr, src.baseAddress!, spectralCount * MemoryLayout<Float>.size)
            }
        } else {
            for c in 0..<4 {
                for h in 0..<numBins {
                    for w in 0..<numFrames {
                        spectralPtr[c * spectralStrides[1] + h * spectralStrides[2] + w * spectralStrides[3]] =
                            spectralData[c * numBins * numFrames + h * numFrames + w]
                    }
                }
            }
        }

        let waveform = try MLMultiArray(shape: [1, 2, 343980], dataType: .float32)
        let wavePtr = waveform.dataPointer.bindMemory(to: Float.self, capacity: 2 * segmentLength)
        left.withUnsafeBufferPointer { src in memcpy(wavePtr, src.baseAddress!, segmentLength * 4) }
        right.withUnsafeBufferPointer { src in memcpy(wavePtr + segmentLength, src.baseAddress!, segmentLength * 4) }

        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
            "spectral_magnitude": MLFeatureValue(multiArray: spectral),
            "audio_waveform": MLFeatureValue(multiArray: waveform)
        ])
        let output = try model.prediction(from: inputFeatures)

        guard let freqOut = output.featureValue(for: "freq_output")?.multiArrayValue,
              let timeOut = output.featureValue(for: "time_output")?.multiArrayValue else {
            throw DemucsError.processingFailed("Missing model output")
        }

        let ispecScale = Float(sqrt(Double(fftSize)))
        var results: [(left: [Float], right: [Float])] = []

        for stem in Stem.allCases {
            let i = stem.modelIndex
            let freqLR = extractChannel(from: freqOut, batch: 0, channel: 4 * i,     height: numBins, width: numFrames)
            let freqLI = extractChannel(from: freqOut, batch: 0, channel: 4 * i + 1, height: numBins, width: numFrames)
            let freqRR = extractChannel(from: freqOut, batch: 0, channel: 4 * i + 2, height: numBins, width: numFrames)
            let freqRI = extractChannel(from: freqOut, batch: 0, channel: 4 * i + 3, height: numBins, width: numFrames)

            var freqLeft = inverseSpec(real: freqLR, imag: freqLI)
            var freqRight = inverseSpec(real: freqRR, imag: freqRI)
            var scale = ispecScale
            vDSP_vsmul(freqLeft, 1, &scale, &freqLeft, 1, vDSP_Length(freqLeft.count))
            vDSP_vsmul(freqRight, 1, &scale, &freqRight, 1, vDSP_Length(freqRight.count))

            let timeLeft = extractChannel1D(from: timeOut, batch: 0, channel: 2 * i, width: segmentLength)
            let timeRight = extractChannel1D(from: timeOut, batch: 0, channel: 2 * i + 1, width: segmentLength)

            var stemLeft = [Float](repeating: 0, count: segmentLength)
            var stemRight = [Float](repeating: 0, count: segmentLength)
            vDSP_vadd(freqLeft, 1, timeLeft, 1, &stemLeft, 1, vDSP_Length(segmentLength))
            vDSP_vadd(freqRight, 1, timeRight, 1, &stemRight, 1, vDSP_Length(segmentLength))
            results.append((left: stemLeft, right: stemRight))
        }
        return results
    }

    /// Build triangular overlap-add window (matches Python apply.py)
    static func buildWeight() -> [Float] {
        let half = segmentLength / 2
        var w = [Float](repeating: 0, count: segmentLength)
        for i in 0..<half { w[i] = Float(i + 1) }
        for i in half..<segmentLength { w[i] = Float(segmentLength - i) }
        var maxVal: Float = 0
        vDSP_maxv(w, 1, &maxVal, vDSP_Length(segmentLength))
        if maxVal > 0 { var inv = 1.0 / maxVal; vDSP_vsmul(w, 1, &inv, &w, 1, vDSP_Length(segmentLength)) }
        return w
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
    @Published var isPlayingOriginal = false
    @Published var showShareSheet = false
    @Published var shareURL: URL?

    private var audioPlayer: AVAudioPlayer?
    private var stemURLs: [Stem: URL] = [:]
    private var lastMode: SeparationMode = .preview

    private func updateAudioInfo() {
        guard let url = audioURL else {
            audioFileName = nil
            audioDuration = nil
            isSeparated = false
            stemURLs.removeAll()
            return
        }

        _ = url.startAccessingSecurityScopedResource()
        audioFileName = url.lastPathComponent
        isSeparated = false
        stemURLs.removeAll()

        let asset = AVURLAsset(url: url)
        Task {
            let duration = try? await asset.load(.duration)
            await MainActor.run { self.audioDuration = duration?.seconds }
        }
    }

    func separate(mode: SeparationMode) {
        guard audioURL != nil else { return }
        isProcessing = true
        errorMessage = nil
        progress = 0
        lastMode = mode

        Task {
            do {
                try await performSeparation(mode: mode)
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

    private func loadModel() throws -> MLModel {
        guard let modelURL = Bundle.main.url(forResource: "HTDemucs_SourceSeparation_F32", withExtension: "mlmodelc") else {
            throw DemucsError.modelNotFound(
                "HTDemucs_SourceSeparation_F32.mlmodelc not found in bundle."
            )
        }
        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly
        return try MLModel(contentsOf: modelURL, configuration: config)
    }

    private func performSeparation(mode: SeparationMode) async throws {
        guard let url = audioURL else { return }

        await updateStatus("Loading audio...", progress: 0.05)
        let (fullLeft, fullRight) = try DSP.loadFullAudio(url: url)

        await updateStatus("Loading model...", progress: 0.1)
        let model = try loadModel()

        let tempDir = FileManager.default.temporaryDirectory
        var newStemURLs: [Stem: URL] = [:]

        switch mode {
        case .preview:
            // Pick a segment from the middle
            let midOffset = max(0, (fullLeft.count - DSP.segmentLength) / 2)
            let endIdx = min(midOffset + DSP.segmentLength, fullLeft.count)
            var segL = Array(fullLeft[midOffset..<endIdx])
            var segR = Array(fullRight[midOffset..<endIdx])
            if segL.count < DSP.segmentLength {
                segL.append(contentsOf: [Float](repeating: 0, count: DSP.segmentLength - segL.count))
                segR.append(contentsOf: [Float](repeating: 0, count: DSP.segmentLength - segR.count))
            }

            await updateStatus("Processing...", progress: 0.3)
            let stems = try DSP.processSegment(left: segL, right: segR, model: model)

            for (idx, stem) in Stem.allCases.enumerated() {
                let stemURL = tempDir.appendingPathComponent("demucs_\(stem.rawValue).wav")
                try DSP.writeWAV(left: stems[idx].left, right: stems[idx].right, to: stemURL)
                newStemURLs[stem] = stemURL
            }

        case .full:
            let totalSamples = fullLeft.count
            let stride = Int(Double(DSP.segmentLength) * (1.0 - DSP.overlap))
            let numChunks = max(1, Int(ceil(Double(max(0, totalSamples - DSP.segmentLength)) / Double(stride))) + 1)
            let weight = DSP.buildWeight()

            // Accumulators per stem
            var accL = Stem.allCases.map { _ in [Float](repeating: 0, count: totalSamples) }
            var accR = Stem.allCases.map { _ in [Float](repeating: 0, count: totalSamples) }
            var weightSum = [Float](repeating: 0, count: totalSamples)

            for chunk in 0..<numChunks {
                let offset = min(chunk * stride, max(0, totalSamples - DSP.segmentLength))
                let chunkLen = min(DSP.segmentLength, totalSamples - offset)

                var segL = Array(fullLeft[offset..<offset + chunkLen])
                var segR = Array(fullRight[offset..<offset + chunkLen])
                if segL.count < DSP.segmentLength {
                    segL.append(contentsOf: [Float](repeating: 0, count: DSP.segmentLength - segL.count))
                    segR.append(contentsOf: [Float](repeating: 0, count: DSP.segmentLength - segR.count))
                }

                await updateStatus("Segment \(chunk + 1)/\(numChunks)...", progress: 0.1 + 0.8 * Double(chunk) / Double(numChunks))
                let stems = try DSP.processSegment(left: segL, right: segR, model: model)

                // Weighted overlap-add
                for (stemIdx, _) in Stem.allCases.enumerated() {
                    for i in 0..<chunkLen {
                        accL[stemIdx][offset + i] += stems[stemIdx].left[i] * weight[i]
                        accR[stemIdx][offset + i] += stems[stemIdx].right[i] * weight[i]
                    }
                }
                for i in 0..<chunkLen {
                    weightSum[offset + i] += weight[i]
                }
            }

            // Normalize and write
            await updateStatus("Writing files...", progress: 0.95)
            for (stemIdx, stem) in Stem.allCases.enumerated() {
                for i in 0..<totalSamples {
                    if weightSum[i] > 1e-8 {
                        accL[stemIdx][i] /= weightSum[i]
                        accR[stemIdx][i] /= weightSum[i]
                    }
                }
                let stemURL = tempDir.appendingPathComponent("demucs_\(stem.rawValue).wav")
                try DSP.writeWAV(left: accL[stemIdx], right: accR[stemIdx], to: stemURL)
                newStemURLs[stem] = stemURL
            }
        }

        await MainActor.run { self.stemURLs = newStemURLs }
        await updateStatus("Complete!", progress: 1.0)
    }

    @MainActor
    private func updateStatus(_ message: String, progress: Double) {
        self.statusMessage = message
        self.progress = progress
    }

    func playOriginal() {
        stopPlayback()
        guard let url = audioURL else { return }
        isPlayingOriginal = true
        do {
            try AVAudioSession.sharedInstance().setCategory(.playback)
            try AVAudioSession.sharedInstance().setActive(true)
            audioPlayer = try AVAudioPlayer(contentsOf: url)
            if lastMode == .preview {
                // Seek to the same segment that was processed (middle)
                let duration = audioPlayer?.duration ?? 0
                let segDuration = Double(DSP.segmentLength) / DSP.sampleRate
                audioPlayer?.currentTime = max(0, (duration - segDuration) / 2)
            }
            audioPlayer?.play()
        } catch {
            errorMessage = "Playback error: \(error.localizedDescription)"
        }
    }

    func playStem(_ stem: Stem) {
        stopPlayback()
        playingStem = stem
        guard let url = stemURLs[stem] else { return }
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
        isPlayingOriginal = false
    }

    func shareStem(_ stem: Stem) {
        guard let url = stemURLs[stem] else { return }
        shareURL = url
        showShareSheet = true
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
