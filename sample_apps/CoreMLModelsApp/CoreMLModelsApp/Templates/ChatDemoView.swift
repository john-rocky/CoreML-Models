import SwiftUI
import CoreMLLLM
import PhotosUI

/// LLM chat demo template with streaming generation.
/// Used by: Gemma 4 E2B (multimodal), Qwen2.5-0.5B (text-only).
///
/// Uses CoreML-LLM package for ANE-optimized on-device inference.
/// Supports multimodal input (text + image) when the model includes a vision encoder.
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

                HStack(spacing: 8) {
                    PhotosPicker(selection: $photoItem, matching: .images) {
                        Image(systemName: "photo").font(.title3)
                    }
                    .onChange(of: photoItem) { _, item in loadImage(item) }

                    TextField("Message…", text: $inputText, axis: .vertical)
                        .textFieldStyle(.roundedBorder)
                        .lineLimit(1...4)
                        .onSubmit { send() }

                    Button { send() } label: {
                        Image(systemName: "arrow.up.circle.fill").font(.title2)
                    }
                    .disabled(inputText.trimmingCharacters(in: .whitespaces).isEmpty || isGenerating || llm == nil)
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

    /// Map this manifest model entry to a CoreMLLLM.ModelDownloader.ModelInfo.
    private func llmModelInfoForEntry() -> CoreMLLLM.ModelDownloader.ModelInfo {
        let normalized = model.id.replacingOccurrences(of: "_", with: "-")
        return CoreMLLLM.ModelDownloader.ModelInfo.defaults.first {
            $0.folderName == normalized || $0.id == normalized
        } ?? .gemma4e2b
    }

    // MARK: - Chat

    private func send() {
        let text = inputText.trimmingCharacters(in: .whitespaces)
        guard !text.isEmpty, let llm, !isGenerating else { return }

        let image = selectedImage?.cgImage
        let displayImage = selectedImage
        inputText = ""
        selectedImage = nil
        photoItem = nil

        messages.append(ChatMessage(role: .user, content: text, image: displayImage))
        messages.append(ChatMessage(role: .assistant, content: ""))

        isGenerating = true
        tokensPerSecond = nil

        Task {
            let start = CFAbsoluteTimeGetCurrent()
            var tokenCount = 0

            do {
                for await token in try await llm.stream(text, image: image, maxTokens: 1024) {
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

            let elapsed = CFAbsoluteTimeGetCurrent() - start
            await MainActor.run {
                isGenerating = false
                tokensPerSecond = elapsed > 0 ? Double(tokenCount) / elapsed : nil
            }
        }
    }

    private func resetConversation() {
        llm?.reset()
        messages.removeAll()
        tokensPerSecond = nil
    }

    private func loadImage(_ item: PhotosPickerItem?) {
        guard let item else { return }
        Task {
            if let data = try? await item.loadTransferable(type: Data.self),
               let img = UIImage(data: data) {
                await MainActor.run { selectedImage = img }
            }
        }
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
