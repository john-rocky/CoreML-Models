import SwiftUI
import UIKit
import CoreML
import Vision
import PhotosUI
import AVFoundation

// MARK: - Wav2Lip: Audio-Driven Talking Head Generation
//
// Wav2Lip takes a face image and a mel-spectrogram audio segment and generates
// a lip-synced face output.
//
// Model Input:
//   - audio_mel (1,1,80,16): Mel-spectrogram of ~200ms audio chunk (80 mel bins x 16 time steps)
//   - face_input (1,6,96,96): Concatenation of reference face (3ch) + masked lower-half face (3ch)
//
// Model Output:
//   - output_face (1,3,96,96): Lip-synced face region
//
// For a full video, you would:
// 1. Extract face crops for each video frame
// 2. Compute mel-spectrogram for the entire audio
// 3. For each frame, pick the corresponding mel window and run inference
// 4. Paste the 96x96 output back into the original frame

struct ContentView: View {
    @StateObject private var viewModel = Wav2LipViewModel()

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    // Face image picker
                    Section {
                        PhotosPicker(selection: $viewModel.selectedPhoto,
                                     matching: .images) {
                            if let image = viewModel.faceImage {
                                Image(uiImage: image)
                                    .resizable()
                                    .scaledToFit()
                                    .frame(maxHeight: 220)
                                    .cornerRadius(12)
                            } else {
                                placeholderView(
                                    title: "Select Portrait Photo",
                                    systemImage: "person.crop.square"
                                )
                            }
                        }
                    } header: {
                        sectionHeader("Face Image")
                    }

                    // Audio section
                    Section {
                        VStack(spacing: 12) {
                            // Audio recorder
                            HStack(spacing: 16) {
                                Button(action: { viewModel.toggleRecording() }) {
                                    VStack(spacing: 6) {
                                        Image(systemName: viewModel.isRecording ?
                                              "stop.circle.fill" : "mic.circle.fill")
                                            .font(.system(size: 44))
                                            .foregroundColor(viewModel.isRecording ? .red : .accentColor)
                                        Text(viewModel.isRecording ? "Stop" : "Record")
                                            .font(.caption)
                                            .foregroundColor(viewModel.isRecording ? .red : .accentColor)
                                    }
                                }

                                VStack(alignment: .leading, spacing: 4) {
                                    if viewModel.isRecording {
                                        HStack(spacing: 4) {
                                            Circle()
                                                .fill(.red)
                                                .frame(width: 8, height: 8)
                                            Text("Recording...")
                                                .font(.subheadline)
                                                .foregroundColor(.red)
                                        }
                                        Text(String(format: "%.1fs", viewModel.recordingDuration))
                                            .font(.caption)
                                            .foregroundColor(.secondary)
                                    } else if viewModel.audioURL != nil {
                                        HStack {
                                            Image(systemName: "checkmark.circle.fill")
                                                .foregroundColor(.green)
                                            Text("Audio recorded")
                                                .font(.subheadline)
                                        }
                                        Text(String(format: "Duration: %.1fs", viewModel.recordingDuration))
                                            .font(.caption)
                                            .foregroundColor(.secondary)
                                    } else {
                                        Text("Tap to record audio for lip sync")
                                            .font(.subheadline)
                                            .foregroundColor(.secondary)
                                    }
                                }

                                Spacer()

                                // Playback button
                                if viewModel.audioURL != nil && !viewModel.isRecording {
                                    Button(action: { viewModel.playRecordedAudio() }) {
                                        Image(systemName: viewModel.isPlayingAudio ?
                                              "speaker.wave.2.fill" : "play.circle")
                                            .font(.title2)
                                            .foregroundColor(.accentColor)
                                    }
                                }
                            }
                            .padding()
                            .background(Color(.systemGray6))
                            .cornerRadius(12)

                            // Audio waveform visualization
                            if viewModel.isRecording {
                                AudioLevelView(level: viewModel.audioLevel)
                                    .frame(height: 40)
                            }
                        }
                    } header: {
                        sectionHeader("Audio Input")
                    }

                    // Generate button
                    if viewModel.faceImage != nil && viewModel.audioURL != nil {
                        Button(action: { viewModel.generateLipSync() }) {
                            HStack {
                                if viewModel.isProcessing {
                                    ProgressView()
                                        .tint(.white)
                                } else {
                                    Image(systemName: "mouth.fill")
                                }
                                Text(viewModel.isProcessing ? "Generating..." : "Generate Lip Sync")
                            }
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(viewModel.isProcessing ? Color.gray : Color.accentColor)
                            .foregroundColor(.white)
                            .cornerRadius(12)
                        }
                        .disabled(viewModel.isProcessing)
                    }

                    // Processing status
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
                    }

                    // Result display
                    if let result = viewModel.resultImage {
                        Section {
                            VStack(spacing: 12) {
                                HStack(spacing: 16) {
                                    // Original face
                                    VStack {
                                        if let face = viewModel.faceImage {
                                            Image(uiImage: face)
                                                .resizable()
                                                .scaledToFill()
                                                .frame(width: 120, height: 120)
                                                .clipped()
                                                .cornerRadius(12)
                                        }
                                        Text("Original")
                                            .font(.caption)
                                            .foregroundColor(.secondary)
                                    }

                                    Image(systemName: "arrow.right")
                                        .font(.title3)
                                        .foregroundColor(.secondary)

                                    // Lip-synced face
                                    VStack {
                                        Image(uiImage: result)
                                            .resizable()
                                            .scaledToFill()
                                            .frame(width: 120, height: 120)
                                            .clipped()
                                            .cornerRadius(12)
                                        Text("Lip-Synced")
                                            .font(.caption)
                                            .foregroundColor(.secondary)
                                    }
                                }

                                Text("Face + Audio = Lip-synced result (single frame preview)")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)

                                // Mel spectrogram visualization placeholder
                                MelSpectrogramView()
                                    .frame(height: 60)
                                    .cornerRadius(8)
                            }
                        } header: {
                            sectionHeader("Result")
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("Wav2Lip")
        }
    }

    private func sectionHeader(_ title: String) -> some View {
        HStack {
            Text(title)
                .font(.headline)
            Spacer()
        }
    }

    private func placeholderView(title: String, systemImage: String) -> some View {
        VStack(spacing: 12) {
            Image(systemName: systemImage)
                .font(.system(size: 40))
                .foregroundColor(.secondary)
            Text(title)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
        .frame(height: 180)
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

// MARK: - Audio Level Visualization

struct AudioLevelView: View {
    let level: Float

    var body: some View {
        GeometryReader { geo in
            HStack(spacing: 2) {
                ForEach(0..<30, id: \.self) { i in
                    let barLevel = Float(i) / 30.0
                    RoundedRectangle(cornerRadius: 1)
                        .fill(barLevel < level ? Color.green : Color(.systemGray5))
                        .frame(width: (geo.size.width - 60) / 30)
                }
            }
            .frame(height: geo.size.height)
        }
    }
}

// MARK: - Mel Spectrogram Visualization

struct MelSpectrogramView: View {
    var body: some View {
        GeometryReader { geo in
            Canvas { context, size in
                // Draw a placeholder mel-spectrogram visualization
                let cols = 80
                let rows = 16
                let cellWidth = size.width / CGFloat(cols)
                let cellHeight = size.height / CGFloat(rows)

                for row in 0..<rows {
                    for col in 0..<cols {
                        // Generate a pattern that looks like a mel spectrogram
                        let energy = sin(Double(col) * 0.15) *
                            cos(Double(row) * 0.3) * 0.5 + 0.5
                        let color = Color(
                            hue: 0.6 - energy * 0.4,
                            saturation: 0.8,
                            brightness: energy * 0.8 + 0.2
                        )
                        let rect = CGRect(
                            x: CGFloat(col) * cellWidth,
                            y: CGFloat(row) * cellHeight,
                            width: cellWidth + 1,
                            height: cellHeight + 1
                        )
                        context.fill(Path(rect), with: .color(color))
                    }
                }
            }
        }
        .overlay(
            VStack {
                HStack {
                    Text("Mel Spectrogram (80 bins x 16 frames)")
                        .font(.system(size: 8))
                        .foregroundColor(.white)
                        .padding(3)
                        .background(Color.black.opacity(0.5))
                        .cornerRadius(3)
                    Spacer()
                }
                Spacer()
            }
            .padding(4)
        )
    }
}

// MARK: - ViewModel

class Wav2LipViewModel: ObservableObject {
    @Published var selectedPhoto: PhotosPickerItem? {
        didSet { loadFaceImage() }
    }
    @Published var faceImage: UIImage?
    @Published var resultImage: UIImage?
    @Published var audioURL: URL?

    @Published var isRecording = false
    @Published var isPlayingAudio = false
    @Published var isProcessing = false
    @Published var recordingDuration: TimeInterval = 0
    @Published var audioLevel: Float = 0
    @Published var progress: Double = 0
    @Published var statusMessage = ""
    @Published var errorMessage: String?

    private var audioRecorder: AVAudioRecorder?
    private var audioPlayer: AVAudioPlayer?
    private var recordingTimer: Timer?
    private var levelTimer: Timer?

    private func loadFaceImage() {
        guard let item = selectedPhoto else { return }
        Task {
            if let data = try? await item.loadTransferable(type: Data.self),
               let image = UIImage(data: data) {
                await MainActor.run {
                    self.faceImage = image
                    self.resultImage = nil
                    self.errorMessage = nil
                }
            }
        }
    }

    func toggleRecording() {
        if isRecording {
            stopRecording()
        } else {
            startRecording()
        }
    }

    private func startRecording() {
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.playAndRecord, mode: .default)
            try session.setActive(true)
        } catch {
            errorMessage = "Audio session error: \(error.localizedDescription)"
            return
        }

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("wav2lip_audio.wav")

        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: 16000.0,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsFloatKey: false
        ]

        do {
            audioRecorder = try AVAudioRecorder(url: url, settings: settings)
            audioRecorder?.isMeteringEnabled = true
            audioRecorder?.record()
            audioURL = url
            isRecording = true
            recordingDuration = 0
            resultImage = nil

            // Update duration and level
            recordingTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
                guard let self = self else { return }
                self.recordingDuration = self.audioRecorder?.currentTime ?? 0
                self.audioRecorder?.updateMeters()
                let power = self.audioRecorder?.averagePower(forChannel: 0) ?? -160
                // Normalize from dB (-160...0) to 0...1
                let normalized = max(0, (power + 60) / 60)
                self.audioLevel = normalized
            }
        } catch {
            errorMessage = "Recording error: \(error.localizedDescription)"
        }
    }

    private func stopRecording() {
        audioRecorder?.stop()
        isRecording = false
        recordingTimer?.invalidate()
        recordingTimer = nil
        audioLevel = 0
    }

    func playRecordedAudio() {
        guard let url = audioURL else { return }
        if isPlayingAudio {
            audioPlayer?.stop()
            isPlayingAudio = false
            return
        }

        do {
            try AVAudioSession.sharedInstance().setCategory(.playback)
            audioPlayer = try AVAudioPlayer(contentsOf: url)
            audioPlayer?.play()
            isPlayingAudio = true

            // Auto-stop indicator when playback finishes
            DispatchQueue.main.asyncAfter(deadline: .now() + (audioPlayer?.duration ?? 1.0)) {
                self.isPlayingAudio = false
            }
        } catch {
            errorMessage = "Playback error: \(error.localizedDescription)"
        }
    }

    func generateLipSync() {
        guard faceImage != nil, audioURL != nil else { return }
        isProcessing = true
        errorMessage = nil
        progress = 0

        Task {
            do {
                let result = try await performLipSync()
                await MainActor.run {
                    self.resultImage = result
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

    // Perform lip sync using Wav2Lip CoreML model
    // Input: audio_mel (1,1,80,16) + face_input (1,6,96,96)
    // Output: output_face (1,3,96,96)
    private func performLipSync() async throws -> UIImage {
        await updateStatus("Loading model...", progress: 0.1)

        guard let modelURL = Bundle.main.url(forResource: "Wav2Lip", withExtension: "mlmodelc") else {
            throw Wav2LipError.modelNotFound(
                "Wav2Lip.mlmodelc not found in bundle. " +
                "Please compile and add the Wav2Lip.mlpackage to the project."
            )
        }

        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        let model = try MLModel(contentsOf: modelURL, configuration: config)

        await updateStatus("Preparing face input...", progress: 0.3)

        // Prepare face input (1, 6, 96, 96)
        // Channels 0-2: reference face RGB, Channels 3-5: lower-half masked face RGB
        guard let face = faceImage,
              let resizedFace = face.resized(to: CGSize(width: 96, height: 96)),
              let cgFace = resizedFace.cgImage else {
            throw Wav2LipError.processingFailed("Failed to prepare face image")
        }

        let faceArray = try MLMultiArray(shape: [1, 6, 96, 96], dataType: .float32)
        fillFaceInput(cgFace, into: faceArray)

        await updateStatus("Computing mel spectrogram...", progress: 0.5)

        // Prepare audio mel spectrogram (1, 1, 80, 16)
        // In production: compute mel spectrogram from audio using Accelerate/vDSP
        // - Sample rate: 16kHz
        // - FFT size: 800, Hop: 200
        // - Mel bins: 80
        // - Time steps per chunk: 16 (~200ms of audio)
        let melArray = try MLMultiArray(shape: [1, 1, 80, 16], dataType: .float32)
        // Fill with placeholder mel values (in production: real mel spectrogram)
        try fillPlaceholderMel(melArray)

        await updateStatus("Running inference...", progress: 0.7)

        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
            "audio_mel": MLFeatureValue(multiArray: melArray),
            "face_input": MLFeatureValue(multiArray: faceArray)
        ])

        let prediction = try model.prediction(from: inputFeatures)

        await updateStatus("Extracting result...", progress: 0.9)

        guard let outputArray = prediction.featureValue(for: "output_face")?.multiArrayValue else {
            throw Wav2LipError.processingFailed("Failed to extract output face")
        }

        guard let resultImage = imageFromMultiArray(outputArray, width: 96, height: 96) else {
            throw Wav2LipError.processingFailed("Failed to convert output to image")
        }

        await updateStatus("Complete!", progress: 1.0)
        return resultImage
    }

    // Fill face_input MLMultiArray (1,6,96,96) from CGImage
    // Channels 0-2: full face, Channels 3-5: lower-half masked
    private func fillFaceInput(_ cgImage: CGImage, into array: MLMultiArray) {
        let width = 96
        let height = 96
        let bytesPerPixel = 4
        var pixelData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &pixelData, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: bytesPerPixel * width,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        for y in 0..<height {
            for x in 0..<width {
                let offset = (y * width + x) * bytesPerPixel
                let r = Float(pixelData[offset]) / 255.0
                let g = Float(pixelData[offset + 1]) / 255.0
                let b = Float(pixelData[offset + 2]) / 255.0

                // Channels 0-2: reference face
                array[[0, 0, y, x] as [NSNumber]] = NSNumber(value: r)
                array[[0, 1, y, x] as [NSNumber]] = NSNumber(value: g)
                array[[0, 2, y, x] as [NSNumber]] = NSNumber(value: b)

                // Channels 3-5: masked face (lower half zeroed out for lip region)
                let isMasked = y > height / 2
                array[[0, 3, y, x] as [NSNumber]] = NSNumber(value: isMasked ? 0.0 : r)
                array[[0, 4, y, x] as [NSNumber]] = NSNumber(value: isMasked ? 0.0 : g)
                array[[0, 5, y, x] as [NSNumber]] = NSNumber(value: isMasked ? 0.0 : b)
            }
        }
    }

    // Fill placeholder mel spectrogram data
    private func fillPlaceholderMel(_ array: MLMultiArray) throws {
        // In production, compute real mel spectrogram from the recorded audio:
        // 1. Load audio samples at 16kHz mono
        // 2. Apply STFT with window=800, hop=200
        // 3. Apply mel filterbank (80 bins)
        // 4. Take log magnitude
        // 5. Extract 16-frame windows for each video frame
        for mel in 0..<80 {
            for t in 0..<16 {
                let value = Float.random(in: -4.0...0.0) // Placeholder: log-mel range
                array[[0, 0, mel, t] as [NSNumber]] = NSNumber(value: value)
            }
        }
    }

    // Convert (1,3,96,96) MLMultiArray back to UIImage
    private func imageFromMultiArray(_ array: MLMultiArray, width: Int, height: Int) -> UIImage? {
        var pixelData = [UInt8](repeating: 255, count: width * height * 4)

        for y in 0..<height {
            for x in 0..<width {
                let r = min(max(array[[0, 0, y, x] as [NSNumber]].floatValue, 0), 1)
                let g = min(max(array[[0, 1, y, x] as [NSNumber]].floatValue, 0), 1)
                let b = min(max(array[[0, 2, y, x] as [NSNumber]].floatValue, 0), 1)
                let offset = (y * width + x) * 4
                pixelData[offset] = UInt8(r * 255)
                pixelData[offset + 1] = UInt8(g * 255)
                pixelData[offset + 2] = UInt8(b * 255)
                pixelData[offset + 3] = 255
            }
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &pixelData, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ), let cgImage = context.makeImage() else { return nil }

        return UIImage(cgImage: cgImage)
    }

    @MainActor
    private func updateStatus(_ message: String, progress: Double) {
        self.statusMessage = message
        self.progress = progress
    }
}

enum Wav2LipError: LocalizedError {
    case modelNotFound(String)
    case processingFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelNotFound(let msg): return msg
        case .processingFailed(let msg): return msg
        }
    }
}

// MARK: - UIImage Extension

extension UIImage {
    func resized(to targetSize: CGSize) -> UIImage? {
        let renderer = UIGraphicsImageRenderer(size: targetSize)
        return renderer.image { _ in
            self.draw(in: CGRect(origin: .zero, size: targetSize))
        }
    }
}

#Preview {
    ContentView()
}
