import SwiftUI
import CoreMLLLM
import PhotosUI
import AVFoundation

/// LLM chat demo template with streaming generation.
/// Used by every CoreML-LLM model in the manifest:
///   - Gemma 4 E2B (text + image + audio + video)
///   - Gemma 4 E4B, Qwen3.5 2B, Qwen3.5 0.8B, Qwen2.5 0.5B (text)
///   - Qwen3-VL 2B (text + image)
///
/// Uses CoreML-LLM package for ANE-optimized on-device inference.
/// Supports multimodal input (text + image + audio + video) when the loaded
/// model includes the respective encoders.
struct ChatDemoView: View {
    let model: ModelEntry

    @State private var llm: CoreMLLLM?
    @State private var messages: [ChatMessage] = []
    @State private var inputText = ""
    @State private var isLoading = true
    @State private var isGenerating = false
    @State private var status = ""
    @State private var selectedImage: UIImage?
    @State private var photoItem: PhotosPickerItem?
    @State private var tokensPerSecond: Double?
    @State private var hasVision = false
    @State private var hasAudio = false
    @State private var hasVideo = false
    @State private var selectedVideoURL: URL?
    @State private var videoPickerItem: PhotosPickerItem?
    @State private var videoThumbnail: UIImage?
    @State private var videoDuration: TimeInterval = 0
    @StateObject private var audioRecorder = ChatAudioRecorder()

    var body: some View {
        VStack(spacing: 0) {
            // Messages
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        ForEach(messages) { msg in
                            MessageBubbleView(message: msg)
                                .id(msg.id)
                        }
                    }
                    .padding()
                }
                .onChange(of: messages.count) { _, _ in
                    if let last = messages.last { proxy.scrollTo(last.id, anchor: .bottom) }
                }
            }

            // Status bar
            if !status.isEmpty || tokensPerSecond != nil {
                HStack {
                    if isLoading { ProgressView().controlSize(.small) }
                    Text(status).font(.caption2).foregroundStyle(.secondary)
                    Spacer()
                    if let tps = tokensPerSecond {
                        Text(String(format: "%.1f tok/s", tps)).font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
                    }
                }
                .padding(.horizontal).padding(.vertical, 4)
            }

            Divider()

            // Input area
            VStack(spacing: 8) {
                // Image preview
                if let img = selectedImage {
                    HStack {
                        Image(uiImage: img).resizable().aspectRatio(contentMode: .fill)
                            .frame(width: 60, height: 60).clipShape(RoundedRectangle(cornerRadius: 8))
                        Spacer()
                        Button { selectedImage = nil; photoItem = nil } label: {
                            Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary)
                        }
                    }
                    .padding(.horizontal)
                }

                // Video preview
                if selectedVideoURL != nil {
                    HStack {
                        ZStack {
                            if let thumb = videoThumbnail {
                                Image(uiImage: thumb).resizable().aspectRatio(contentMode: .fill)
                                    .frame(width: 60, height: 60).clipShape(RoundedRectangle(cornerRadius: 8))
                            } else {
                                RoundedRectangle(cornerRadius: 8).fill(.gray.opacity(0.25))
                                    .frame(width: 60, height: 60)
                            }
                            Image(systemName: "play.circle.fill")
                                .font(.title3).foregroundStyle(.white).shadow(radius: 2)
                        }
                        Text(videoDuration > 0 ? String(format: "Video · %.1fs", videoDuration) : "Video")
                            .font(.caption).foregroundStyle(.secondary)
                        Spacer()
                        Button { clearVideo() } label: {
                            Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary)
                        }
                    }
                    .padding(.horizontal)
                }

                // Audio preview (recording / ready-to-send)
                if hasAudio && (audioRecorder.isRecording || audioRecorder.recordedSamples != nil) {
                    HStack(spacing: 8) {
                        Image(systemName: "waveform").foregroundStyle(.purple)
                        if audioRecorder.isRecording {
                            Text(String(format: "Recording… %.1fs", audioRecorder.duration))
                                .font(.caption).foregroundStyle(.secondary)
                        } else if let count = audioRecorder.recordedSamples?.count {
                            Text(String(format: "Audio ready (%.1fs)", Double(count) / 16_000))
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        Spacer()
                        if !audioRecorder.isRecording {
                            Button { audioRecorder.clear() } label: {
                                Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary)
                            }
                        }
                    }
                    .padding(.horizontal)
                }

                HStack(spacing: 8) {
                    if hasVision {
                        PhotosPicker(selection: $photoItem, matching: .images) {
                            Image(systemName: "photo").font(.title3)
                        }
                        .onChange(of: photoItem) { _, item in loadImage(item) }
                        .disabled(isGenerating || selectedVideoURL != nil)
                    }

                    if hasVideo {
                        PhotosPicker(selection: $videoPickerItem, matching: .videos) {
                            Image(systemName: "video").font(.title3)
                        }
                        .onChange(of: videoPickerItem) { _, item in loadVideo(item) }
                        .disabled(isGenerating)
                    }

                    if hasAudio {
                        Button { toggleRecording() } label: {
                            Image(systemName: audioRecorder.isRecording ? "stop.circle.fill" : "mic")
                                .font(.title3)
                                .foregroundStyle(audioRecorder.isRecording ? .red : .accentColor)
                        }
                        .disabled(isGenerating || selectedVideoURL != nil)
                    }

                    TextField("Message…", text: $inputText, axis: .vertical)
                        .textFieldStyle(.roundedBorder)
                        .lineLimit(1...4)
                        .onSubmit { send() }

                    Button { send() } label: {
                        Image(systemName: "arrow.up.circle.fill").font(.title2)
                    }
                    .disabled(sendButtonDisabled)
                }
                .padding(.horizontal).padding(.bottom, 8)
            }
        }
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button { resetConversation() } label: {
                    Image(systemName: "arrow.counterclockwise")
                }.disabled(isGenerating)
            }
        }
        .task { await loadModel() }
        .onDisappear {
            audioRecorder.stop()
            audioRecorder.clear()
            clearVideo()
            llm?.reset()
            llm = nil
            messages.removeAll()
        }
    }

    private var sendButtonDisabled: Bool {
        guard !isGenerating, llm != nil else { return true }
        let hasText = !inputText.trimmingCharacters(in: .whitespaces).isEmpty
        let hasAudioClip = audioRecorder.recordedSamples != nil
        let hasVideoClip = selectedVideoURL != nil
        return !(hasText || hasAudioClip || hasVideoClip)
    }

    // MARK: - Model Loading

    private func loadModel() async {
        isLoading = true
        status = "Loading model…"

        do {
            let llmInfo = llmModelInfoForEntry()
            let loaded = try await CoreMLLLM.load(
                model: llmInfo,
                computeUnits: .cpuAndNeuralEngine
            ) { s in
                Task { @MainActor in status = s }
            }

            await MainActor.run {
                llm = loaded
                hasVision = loaded.supportsVision
                hasAudio = loaded.supportsAudio
                // CoreMLLLM 1.4 still ships a video encoder only for Gemma 4 E2B
                // (Qwen3-VL 2B is image-only). No public supportsVideo flag yet,
                // so gate on the single currently-capable manifest id.
                hasVideo = loaded.supportsVision && model.id == "gemma4_e2b"
                audioRecorder.maxDuration = loaded.maxAudioDuration
                isLoading = false
                status = ""
            }
        } catch {
            await MainActor.run {
                isLoading = false
                status = "Load failed: \(error.localizedDescription)"
            }
            print("[Chat] Load error: \(error)")
        }
    }

    /// Map this manifest model entry to a ModelDownloader.ModelInfo.
    private func llmModelInfoForEntry() -> ModelDownloader.ModelInfo {
        let normalized = model.id.replacingOccurrences(of: "_", with: "-")
        return ModelDownloader.ModelInfo.defaults.first {
            $0.folderName == normalized || $0.id == normalized
        } ?? .gemma4e2b
    }

    // MARK: - Chat

    private func send() {
        let text = inputText.trimmingCharacters(in: .whitespaces)
        let audio = audioRecorder.recordedSamples
        let videoURL = selectedVideoURL
        guard let llm, !isGenerating,
              !text.isEmpty || audio != nil || videoURL != nil else { return }

        let image = selectedImage?.cgImage
        let displayImage = selectedImage
        let displayVideoThumbnail = videoThumbnail
        inputText = ""
        selectedImage = nil
        photoItem = nil
        audioRecorder.clear()
        // Keep the temp file on disk until the stream finishes; just drop the
        // UI state so the user sees the composer clear immediately.
        selectedVideoURL = nil
        videoPickerItem = nil
        videoThumbnail = nil
        videoDuration = 0

        // Label audio/video-only sends so the transcript still shows something visible.
        let displayText: String
        if videoURL != nil {
            displayText = text.isEmpty ? "[Video]" : "[Video] " + text
        } else if audio != nil {
            displayText = text.isEmpty ? "[Audio]" : "[Audio] " + text
        } else {
            displayText = text
        }

        messages.append(ChatMessage(role: .user, content: displayText,
                                    image: displayImage ?? displayVideoThumbnail))
        messages.append(ChatMessage(role: .assistant, content: ""))

        isGenerating = true
        tokensPerSecond = nil

        Task {
            let start = CFAbsoluteTimeGetCurrent()
            var tokenCount = 0

            do {
                let stream: AsyncStream<String>
                if let videoURL {
                    stream = try await llm.stream(text, videoURL: videoURL,
                                                  videoOptions: .init(),
                                                  maxTokens: 1024)
                } else {
                    stream = try await llm.stream(text, image: image, audio: audio,
                                                  maxTokens: 1024)
                }
                for await token in stream {
                    tokenCount += 1
                    await MainActor.run {
                        if var last = messages.last, last.role == .assistant {
                            messages[messages.count - 1].content += token
                        }
                    }
                }
            } catch {
                await MainActor.run {
                    messages[messages.count - 1].content += "\n[Error: \(error.localizedDescription)]"
                }
            }

            if let videoURL {
                try? FileManager.default.removeItem(at: videoURL)
            }

            let elapsed = CFAbsoluteTimeGetCurrent() - start
            await MainActor.run {
                isGenerating = false
                tokensPerSecond = elapsed > 0 ? Double(tokenCount) / elapsed : nil
            }
        }
    }

    private func resetConversation() {
        llm?.reset()
        audioRecorder.clear()
        messages.removeAll()
        tokensPerSecond = nil
    }

    private func toggleRecording() {
        if audioRecorder.isRecording {
            audioRecorder.stop()
        } else {
            do {
                try audioRecorder.start()
            } catch {
                status = "Mic error: \(error.localizedDescription)"
            }
        }
    }

    private func loadImage(_ item: PhotosPickerItem?) {
        guard let item else { return }
        Task {
            if let data = try? await item.loadTransferable(type: Data.self),
               let img = UIImage(data: data) {
                await MainActor.run {
                    selectedImage = img
                    // Image and video are mutually exclusive — CoreMLLLM's
                    // video stream has no image parameter.
                    clearVideo()
                }
            }
        }
    }

    private func loadVideo(_ item: PhotosPickerItem?) {
        guard let item else { return }
        Task {
            guard let data = try? await item.loadTransferable(type: Data.self) else { return }
            let tmpURL = FileManager.default.temporaryDirectory
                .appendingPathComponent("chat_video_\(UUID().uuidString).mp4")
            do {
                try data.write(to: tmpURL)
            } catch {
                print("[Chat] Failed to stage video: \(error)")
                return
            }
            let asset = AVURLAsset(url: tmpURL)
            let duration = (try? await asset.load(.duration).seconds) ?? 0
            let thumb = await generateVideoThumbnail(asset: asset)
            await MainActor.run {
                // Clear other modalities — video stream API is exclusive.
                selectedImage = nil
                photoItem = nil
                audioRecorder.clear()
                selectedVideoURL = tmpURL
                videoThumbnail = thumb
                videoDuration = duration
            }
        }
    }

    private func generateVideoThumbnail(asset: AVURLAsset) async -> UIImage? {
        let gen = AVAssetImageGenerator(asset: asset)
        gen.appliesPreferredTrackTransform = true
        gen.maximumSize = CGSize(width: 240, height: 240)
        let t = CMTime(seconds: 0.1, preferredTimescale: 600)
        return await withCheckedContinuation { continuation in
            gen.generateCGImagesAsynchronously(forTimes: [NSValue(time: t)]) { _, cg, _, _, _ in
                if let cg {
                    continuation.resume(returning: UIImage(cgImage: cg))
                } else {
                    continuation.resume(returning: nil)
                }
            }
        }
    }

    private func clearVideo() {
        if let url = selectedVideoURL {
            try? FileManager.default.removeItem(at: url)
        }
        selectedVideoURL = nil
        videoPickerItem = nil
        videoThumbnail = nil
        videoDuration = 0
    }
}

// MARK: - Audio Recorder (mono 16kHz PCM float32 for Gemma's audio encoder)

/// Captures microphone audio into a `[Float]` buffer at 16 kHz mono, suitable
/// for `CoreMLLLM.stream(..., audio:)`. Named `ChatAudioRecorder` to avoid
/// colliding with the file-based `AudioRecorder` used by `AudioInOutDemoView`.
final class ChatAudioRecorder: ObservableObject {
    @Published var isRecording = false
    @Published var duration: TimeInterval = 0
    @Published var recordedSamples: [Float]?

    /// Maximum recording length in seconds (set from `CoreMLLLM.maxAudioDuration`).
    var maxDuration: TimeInterval = 10.0

    private var engine: AVAudioEngine?
    private var samples: [Float] = []
    private let sampleRate: Double = 16_000

    func start() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.record, mode: .measurement)
        try session.setActive(true)

        samples.removeAll(keepingCapacity: true)
        duration = 0
        recordedSamples = nil

        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)
        guard let targetFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                               sampleRate: sampleRate,
                                               channels: 1,
                                               interleaved: false),
              let converter = AVAudioConverter(from: inputFormat, to: targetFormat) else {
            throw NSError(domain: "ChatAudioRecorder", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Unable to build audio converter"])
        }

        inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { [weak self] buffer, _ in
            guard let self, self.isRecording else { return }
            let outFrames = AVAudioFrameCount(
                Double(buffer.frameLength) * self.sampleRate / inputFormat.sampleRate)
            guard outFrames > 0,
                  let converted = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outFrames) else { return }
            var err: NSError?
            let status = converter.convert(to: converted, error: &err) { _, out in
                out.pointee = .haveData
                return buffer
            }
            guard status != .error, let channel = converted.floatChannelData else { return }
            let count = Int(converted.frameLength)
            let chunk = Array(UnsafeBufferPointer(start: channel[0], count: count))
            DispatchQueue.main.async {
                self.samples.append(contentsOf: chunk)
                self.duration = Double(self.samples.count) / self.sampleRate
                if self.duration >= self.maxDuration { self.stop() }
            }
        }

        try engine.start()
        self.engine = engine
        isRecording = true
    }

    func stop() {
        guard isRecording else { return }
        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        engine = nil
        isRecording = false
        let cap = Int(maxDuration * sampleRate)
        recordedSamples = Array(samples.prefix(cap))
    }

    func clear() {
        recordedSamples = nil
        samples.removeAll(keepingCapacity: false)
        duration = 0
    }
}

// MARK: - Chat Message Model

struct ChatMessage: Identifiable {
    let id = UUID()
    let role: Role
    var content: String
    var image: UIImage?

    enum Role { case user, assistant }
}

// MARK: - Message Bubble View

struct MessageBubbleView: View {
    let message: ChatMessage

    var body: some View {
        HStack {
            if message.role == .user { Spacer(minLength: 60) }

            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 4) {
                if let image = message.image {
                    Image(uiImage: image).resizable().aspectRatio(contentMode: .fill)
                        .frame(maxWidth: 200, maxHeight: 200)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                }

                if !message.content.isEmpty {
                    Text(message.content)
                        .font(.body)
                        .textSelection(.enabled)
                        .padding(.horizontal, 12).padding(.vertical, 8)
                        .background(message.role == .user ? Color.blue : Color(.systemGray5))
                        .foregroundStyle(message.role == .user ? .white : .primary)
                        .clipShape(RoundedRectangle(cornerRadius: 16))
                }
            }

            if message.role == .assistant { Spacer(minLength: 60) }
        }
    }
}
