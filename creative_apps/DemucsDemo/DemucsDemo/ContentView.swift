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

    // Index in model output (freq_output: 4 channels per stem, time_output: 2 channels per stem)
    var modelIndex: Int {
        switch self {
        case .drums: return 0
        case .bass: return 1
        case .vocals: return 2
        case .other: return 3
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

struct ContentView: View {
    @StateObject private var viewModel = DemucsViewModel()

    var body: some View {
        NavigationStack {
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

                // Separation button
                if viewModel.audioURL != nil && !viewModel.isSeparated {
                    Button(action: { viewModel.separate() }) {
                        HStack {
                            if viewModel.isProcessing {
                                ProgressView()
                                    .tint(.white)
                            } else {
                                Image(systemName: "scissors")
                            }
                            Text(viewModel.isProcessing ? "Separating..." : "Separate Stems")
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(viewModel.isProcessing ? Color.gray : Color.accentColor)
                        .foregroundColor(.white)
                        .cornerRadius(12)
                    }
                    .disabled(viewModel.isProcessing)
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
                        Text("Separated Stems")
                            .font(.headline)
                            .frame(maxWidth: .infinity, alignment: .leading)

                        ForEach(Stem.allCases) { stem in
                            StemPlayerView(
                                stem: stem,
                                isPlaying: viewModel.playingStem == stem,
                                onPlay: { viewModel.playStem(stem) },
                                onStop: { viewModel.stopPlayback() }
                            )
                        }
                    }
                    .padding()
                }

                Spacer()

                // Waveform visualization
                if viewModel.isSeparated {
                    WaveformView(activeStem: viewModel.playingStem)
                        .frame(height: 80)
                        .padding()
                }
            }
            .navigationTitle("Demucs Separator")
            .sheet(isPresented: $viewModel.showFilePicker) {
                AudioFilePickerView(audioURL: $viewModel.audioURL)
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

    var body: some View {
        HStack(spacing: 16) {
            Image(systemName: stem.icon)
                .font(.title3)
                .foregroundColor(stem.color)
                .frame(width: 30)

            Text(stem.rawValue)
                .font(.body)
                .fontWeight(.medium)

            Spacer()

            HStack(spacing: 2) {
                ForEach(0..<5) { i in
                    RoundedRectangle(cornerRadius: 1)
                        .fill(isPlaying ? stem.color : Color(.systemGray4))
                        .frame(width: 3, height: CGFloat(8 + i * 4))
                }
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

// MARK: - Animated Waveform

struct WaveformView: View {
    let activeStem: Stem?
    @State private var phase: CGFloat = 0

    var body: some View {
        TimelineView(.animation) { timeline in
            waveformCanvas(time: timeline.date.timeIntervalSinceReferenceDate)
        }
    }

    private func waveformCanvas(time: Double) -> some View {
        let color: Color = activeStem?.color ?? .gray
        let isActive: Bool = activeStem != nil
        return Canvas { context, size in
            drawWaveform(context: context, size: size, time: time, color: color, isActive: isActive)
        }
    }

    private func drawWaveform(context: GraphicsContext, size: CGSize, time: Double, color: Color, isActive: Bool) {
        let midY: CGFloat = size.height / 2
        let amplitude: CGFloat = isActive ? size.height * 0.35 : size.height * 0.1

        var path = Path()
        path.move(to: CGPoint(x: 0, y: midY))
        for x in stride(from: 0, through: size.width, by: 2) {
            let normalizedX: CGFloat = x / size.width
            let wave1: CGFloat = sin(normalizedX * .pi * 6 + time * 3)
            let wave2: CGFloat = sin(normalizedX * .pi * 2 + time * 1.5)
            let y: CGFloat = midY + wave1 * amplitude * (0.5 + 0.5 * wave2)
            path.addLine(to: CGPoint(x: x, y: y))
        }

        context.stroke(path, with: .color(color.opacity(0.7)), lineWidth: 2)
    }
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
    static let segmentOffset = 343980  // Skip first ~7.8s to reach section with all instruments
    static let sampleRate: Double = 44100

    // Periodic Hann window (matches PyTorch's hann_window with periodic=True)
    static let window: [Float] = (0..<fftSize).map {
        Float(0.5 * (1.0 - cos(2.0 * .pi * Double($0) / Double(fftSize))))
    }

    /// Reflect-pad a signal (matches PyTorch's F.pad with mode='reflect')
    static func reflectPad(signal: [Float], padSize: Int) -> [Float] {
        let n = signal.count
        var padded = [Float](repeating: 0, count: n + 2 * padSize)
        // Left: signal[padSize], signal[padSize-1], ..., signal[1]
        for i in 0..<padSize {
            padded[i] = signal[padSize - i]
        }
        // Center
        signal.withUnsafeBufferPointer { src in
            padded.withUnsafeMutableBufferPointer { dst in
                memcpy(dst.baseAddress! + padSize, src.baseAddress!, n * MemoryLayout<Float>.size)
            }
        }
        // Right: signal[n-2], signal[n-3], ..., signal[n-1-padSize]
        for i in 0..<padSize {
            padded[padSize + n + i] = signal[n - 2 - i]
        }
        return padded
    }

    /// Load audio file as stereo Float32 arrays, resampled to 44100 Hz
    static func loadAudio(url: URL) throws -> (left: [Float], right: [Float]) {
        _ = url.startAccessingSecurityScopedResource()
        defer { url.stopAccessingSecurityScopedResource() }

        let sourceFile = try AVAudioFile(forReading: url)
        let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 2,
            interleaved: false
        )!

        let totalNeeded = segmentOffset + segmentLength
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: AVAudioFrameCount(totalNeeded)) else {
            throw DemucsError.processingFailed("Failed to create audio buffer")
        }

        if sourceFile.processingFormat.sampleRate == sampleRate && sourceFile.processingFormat.channelCount == 2 {
            let count = min(AVAudioFrameCount(sourceFile.length), AVAudioFrameCount(totalNeeded))
            try sourceFile.read(into: outputBuffer, frameCount: count)
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

        // Skip segmentOffset samples, take segmentLength samples
        let offset = min(segmentOffset, max(0, count - segmentLength))
        let available = min(segmentLength, count - offset)
        var left = Array(UnsafeBufferPointer(start: leftPtr + offset, count: available))
        var right = Array(UnsafeBufferPointer(start: rightPtr + offset, count: available))

        // Pad or trim to segment length
        if left.count < segmentLength {
            left.append(contentsOf: [Float](repeating: 0, count: segmentLength - left.count))
            right.append(contentsOf: [Float](repeating: 0, count: segmentLength - right.count))
        } else if left.count > segmentLength {
            left = Array(left.prefix(segmentLength))
            right = Array(right.prefix(segmentLength))
        }

        return (left, right)
    }

    /// Forward STFT using vDSP.
    /// Returns (real, imag) arrays of size [numBins * numFrames], bin-major order.
    /// Values are true (unscaled) DFT coefficients matching PyTorch's torch.stft output.
    static func forwardSTFT(signal: [Float]) -> (real: [Float], imag: [Float]) {
        let log2n = vDSP_Length(log2(Float(fftSize)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else { return ([], []) }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        let halfN = fftSize / 2
        var allReal = [Float](repeating: 0, count: numBins * numFrames)
        var allImag = [Float](repeating: 0, count: numBins * numFrames)
        var frame = [Float](repeating: 0, count: fftSize)
        var rp = [Float](repeating: 0, count: halfN)
        var ip = [Float](repeating: 0, count: halfN)

        for f in 0..<numFrames {
            let start = f * hopSize

            // Extract frame with zero-padding
            let avail = min(fftSize, max(0, signal.count - start))
            vDSP_vclr(&frame, 1, vDSP_Length(fftSize))
            if avail > 0 {
                signal.withUnsafeBufferPointer { buf in
                    frame.withUnsafeMutableBufferPointer { dst in
                        memcpy(dst.baseAddress!, buf.baseAddress! + start, avail * MemoryLayout<Float>.size)
                    }
                }
            }

            // Apply analysis window
            vDSP_vmul(frame, 1, window, 1, &frame, 1, vDSP_Length(fftSize))

            // Pack as split complex: rp[i] = frame[2i], ip[i] = frame[2i+1]
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

            // Forward FFT (output is 2x true DFT)
            rp.withUnsafeMutableBufferPointer { rpBuf in
                ip.withUnsafeMutableBufferPointer { ipBuf in
                    var sc = DSPSplitComplex(realp: rpBuf.baseAddress!, imagp: ipBuf.baseAddress!)
                    vDSP_fft_zrip(fftSetup, &sc, 1, log2n, FFTDirection(kFFTDirection_Forward))
                }
            }

            // Store true DFT values (divide by 2)
            // Bin 0 (DC): rp[0]/2, imag = 0
            allReal[f] = rp[0] * 0.5
            allImag[f] = 0

            // Bins 1..numBins-1
            for k in 1..<numBins {
                let idx = k * numFrames + f
                allReal[idx] = rp[k] * 0.5
                allImag[idx] = ip[k] * 0.5
            }
        }

        return (allReal, allImag)
    }

    /// Inverse STFT with overlap-add.
    /// Input: (real, imag) arrays of size [numBins * numFrames], bin-major order (true DFT values).
    /// outputLength: length of output signal (use padded length if center padding was applied).
    static func inverseSTFT(real: [Float], imag: [Float], outputLength: Int) -> [Float] {
        let log2n = vDSP_Length(log2(Float(fftSize)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else { return [] }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        let halfN = fftSize / 2
        var output = [Float](repeating: 0, count: outputLength)
        var windowSum = [Float](repeating: 0, count: outputLength)
        var rp = [Float](repeating: 0, count: halfN)
        var ip = [Float](repeating: 0, count: halfN)
        var frame = [Float](repeating: 0, count: fftSize)

        for f in 0..<numFrames {
            // Pack into vDSP format (2x true DFT values)
            rp[0] = real[f] * 2.0           // DC
            ip[0] = 0                        // Nyquist (not available, set to 0)
            for k in 1..<numBins {
                let idx = k * numFrames + f
                rp[k] = real[idx] * 2.0
                ip[k] = imag[idx] * 2.0
            }

            // Inverse FFT
            rp.withUnsafeMutableBufferPointer { rpBuf in
                ip.withUnsafeMutableBufferPointer { ipBuf in
                    var sc = DSPSplitComplex(realp: rpBuf.baseAddress!, imagp: ipBuf.baseAddress!)
                    vDSP_fft_zrip(fftSetup, &sc, 1, log2n, FFTDirection(kFFTDirection_Inverse))
                }
            }

            // Unpack split complex to interleaved real signal
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

            // Scale: vDSP inverse of (2*DFT) gives 2*fftSize * signal
            var scale: Float = 1.0 / Float(2 * fftSize)
            vDSP_vsmul(frame, 1, &scale, &frame, 1, vDSP_Length(fftSize))

            // Apply synthesis window
            vDSP_vmul(frame, 1, window, 1, &frame, 1, vDSP_Length(fftSize))

            // Overlap-add
            let start = f * hopSize
            for i in 0..<fftSize {
                let idx = start + i
                if idx < outputLength {
                    output[idx] += frame[i]
                    windowSum[idx] += window[i] * window[i]
                }
            }
        }

        // COLA normalization
        for i in 0..<outputLength {
            if windowSum[i] > 1e-8 {
                output[i] /= windowSum[i]
            }
        }

        return output
    }

    /// Convert MLMultiArray to [Float], handling Float16/Float32 output types
    static func mlArrayToFloat32(_ array: MLMultiArray) -> [Float] {
        let count = array.count
        switch array.dataType {
        case .float32:
            let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: count)
            return Array(UnsafeBufferPointer(start: ptr, count: count))
        case .float16:
            var result = [Float](repeating: 0, count: count)
            let srcPtr = array.dataPointer
            result.withUnsafeMutableBufferPointer { dst in
                var srcBuf = vImage_Buffer(
                    data: UnsafeMutableRawPointer(mutating: srcPtr),
                    height: 1,
                    width: vImagePixelCount(count),
                    rowBytes: count * 2
                )
                var dstBuf = vImage_Buffer(
                    data: dst.baseAddress!,
                    height: 1,
                    width: vImagePixelCount(count),
                    rowBytes: count * MemoryLayout<Float>.size
                )
                vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, 0)
            }
            return result
        default:
            return (0..<count).map { array[$0].floatValue }
        }
    }

    /// Extract a 2D channel [height x width] from a 4D MLMultiArray, respecting strides and data type
    static func extractChannel(from array: MLMultiArray, batch: Int, channel: Int, height: Int, width: Int) -> [Float] {
        let strides = array.strides.map { $0.intValue }
        let baseOffset = batch * strides[0] + channel * strides[1]
        let hStride = strides[2]
        let wStride = strides[3]
        let count = height * width
        var result = [Float](repeating: 0, count: count)

        if array.dataType == .float32 {
            let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
            if wStride == 1 {
                // Row-contiguous: copy row by row (handles padding between rows)
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
                // Row-contiguous Float16: convert row by row with vImage
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

    /// Extract a 1D channel from a 3D MLMultiArray [batch, channel, width], respecting strides
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

    private var audioPlayer: AVAudioPlayer?
    private var stemURLs: [Stem: URL] = [:]

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
            await MainActor.run {
                self.audioDuration = duration?.seconds
            }
        }
    }

    func separate() {
        guard audioURL != nil else { return }
        isProcessing = true
        errorMessage = nil
        progress = 0

        Task {
            do {
                try await performSeparation()
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

    private func performSeparation() async throws {
        guard let url = audioURL else { return }

        // 1. Load audio
        await updateStatus("Loading audio...", progress: 0.05)
        let (rawLeft, rawRight) = try DSP.loadAudio(url: url)

        // Load model
        await updateStatus("Loading model...", progress: 0.1)
        guard let modelURL = Bundle.main.url(forResource: "HTDemucs_SourceSeparation_F32", withExtension: "mlmodelc") else {
            throw DemucsError.modelNotFound(
                "HTDemucs_SourceSeparation.mlmodelc not found in bundle. " +
                "Please compile and add the HTDemucs_SourceSeparation.mlpackage to the project."
            )
        }
        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly
        let model = try MLModel(contentsOf: modelURL, configuration: config)

        // Prepare model inputs
        await updateStatus("Preparing input...", progress: 0.2)

        // spectral_magnitude [1, 4, 2048, 336] - zeros (freq output is Float16, overflows for real STFT data)
        let spectral = try MLMultiArray(shape: [1, 4, 2048, 336], dataType: .float32)

        // audio_waveform [1, 2, 343980]
        let waveform = try MLMultiArray(shape: [1, 2, 343980], dataType: .float32)
        let wavePtr = waveform.dataPointer.bindMemory(to: Float.self, capacity: 2 * DSP.segmentLength)
        rawLeft.withUnsafeBufferPointer { src in memcpy(wavePtr, src.baseAddress!, DSP.segmentLength * 4) }
        rawRight.withUnsafeBufferPointer { src in memcpy(wavePtr + DSP.segmentLength, src.baseAddress!, DSP.segmentLength * 4) }

        // 6. Run inference
        await updateStatus("Running inference...", progress: 0.5)
        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
            "spectral_magnitude": MLFeatureValue(multiArray: spectral),
            "audio_waveform": MLFeatureValue(multiArray: waveform)
        ])
        let output = try model.prediction(from: inputFeatures)

        // Extract time-domain output (freq_output overflows Float16 for real STFT data)
        guard let timeOut = output.featureValue(for: "time_output")?.multiArrayValue else {
            throw DemucsError.processingFailed("Missing model output")
        }

        // Reconstruct each stem using time-domain output only
        // (freq_output overflows Float16 → ±inf, needs Float32 model reconversion)
        let tempDir = FileManager.default.temporaryDirectory
        var newStemURLs: [Stem: URL] = [:]

        for stem in Stem.allCases {
            let i = stem.modelIndex
            await updateStatus("Reconstructing \(stem.rawValue)...", progress: 0.65 + Double(i) * 0.08)

            // Time domain output only (freq_output overflows Float16 range for real STFT data)
            let stemLeft = DSP.extractChannel1D(from: timeOut, batch: 0, channel: 2 * i, width: DSP.segmentLength)
            let stemRight = DSP.extractChannel1D(from: timeOut, batch: 0, channel: 2 * i + 1, width: DSP.segmentLength)

            // Write WAV file
            let stemURL = tempDir.appendingPathComponent("demucs_\(stem.rawValue).wav")
            try DSP.writeWAV(left: stemLeft, right: stemRight, to: stemURL)
            newStemURLs[stem] = stemURL
        }

        await MainActor.run { self.stemURLs = newStemURLs }
        await updateStatus("Complete!", progress: 1.0)
    }

    @MainActor
    private func updateStatus(_ message: String, progress: Double) {
        self.statusMessage = message
        self.progress = progress
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
