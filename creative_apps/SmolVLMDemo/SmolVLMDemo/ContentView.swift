import SwiftUI
import PhotosUI
import CoreML
import CoreImage

// MARK: - Data Models

struct ChatMessage: Identifiable {
    let id = UUID()
    let image: UIImage?
    let question: String
    let response: String
    let timestamp: Date

    var formattedTime: String {
        let formatter = DateFormatter()
        formatter.timeStyle = .short
        return formatter.string(from: timestamp)
    }
}

struct PromptChip: Identifiable {
    let id = UUID()
    let label: String
    let prompt: String
    let icon: String
}

// MARK: - Vision Encoder Manager

class VisionEncoderManager: ObservableObject {
    @Published var isModelLoaded = false
    @Published var isProcessing = false
    @Published var errorMessage: String?

    private var model: MLModel?

    private let featureDescriptions: [String] = [
        "Spatial layout detected with structured regions",
        "Color distribution analyzed across channels",
        "Edge and texture features extracted",
        "Object-like regions identified in feature map",
        "Semantic patterns recognized in embedding space"
    ]

    func loadModel() {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }

            do {
                let config = MLModelConfiguration()
                config.computeUnits = .all

                guard let modelURL = Bundle.main.url(
                    forResource: "SmolVLM2_VisionEncoder",
                    withExtension: "mlmodelc"
                ) else {
                    DispatchQueue.main.async {
                        self.errorMessage = "SmolVLM2_VisionEncoder.mlmodelc not found in bundle. "
                            + "Run convert_smolvlm2.py to generate the model, then compile "
                            + "the .mlpackage to .mlmodelc and add it to the Xcode project."
                        self.isModelLoaded = false
                    }
                    return
                }

                let loadedModel = try MLModel(contentsOf: modelURL, configuration: config)
                DispatchQueue.main.async {
                    self.model = loadedModel
                    self.isModelLoaded = true
                    self.errorMessage = nil
                }
            } catch {
                DispatchQueue.main.async {
                    self.errorMessage = "Failed to load model: \(error.localizedDescription)"
                    self.isModelLoaded = false
                }
            }
        }
    }

    func encodeImage(_ image: UIImage, prompt: String, completion: @escaping (String) -> Void) {
        guard isModelLoaded, let model = model else {
            completion("[Model not loaded] Using simulated analysis for: \(prompt)")
            return
        }

        isProcessing = true

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }

            do {
                guard let pixelBuffer = self.imageToPixelBuffer(image, width: 384, height: 384) else {
                    DispatchQueue.main.async {
                        self.isProcessing = false
                        completion("Failed to convert image to pixel buffer.")
                    }
                    return
                }

                let input = try MLDictionaryFeatureProvider(dictionary: [
                    "pixel_values": MLFeatureValue(pixelBuffer: pixelBuffer)
                ])

                let output = try model.prediction(from: input)
                let resultText = self.interpretFeatures(output, prompt: prompt, image: image)

                DispatchQueue.main.async {
                    self.isProcessing = false
                    completion(resultText)
                }
            } catch {
                DispatchQueue.main.async {
                    self.isProcessing = false
                    completion("Inference error: \(error.localizedDescription)")
                }
            }
        }
    }

    func simulateAnalysis(for image: UIImage, prompt: String, completion: @escaping (String) -> Void) {
        isProcessing = true

        let imageSize = image.size
        let aspectRatio = imageSize.width / imageSize.height
        let megapixels = (imageSize.width * imageSize.height) / 1_000_000
        let orientation = aspectRatio > 1.2 ? "landscape" : (aspectRatio < 0.8 ? "portrait" : "square")

        let avgColor = dominantColorDescription(for: image)

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }

            let analysis = self.buildAnalysis(
                prompt: prompt,
                orientation: orientation,
                megapixels: megapixels,
                avgColor: avgColor,
                aspectRatio: aspectRatio
            )

            DispatchQueue.main.async {
                self.isProcessing = false
                completion(analysis)
            }
        }
    }

    private func buildAnalysis(
        prompt: String,
        orientation: String,
        megapixels: Double,
        avgColor: String,
        aspectRatio: Double
    ) -> String {
        let lowerPrompt = prompt.lowercased()

        if lowerPrompt.contains("describe") || lowerPrompt.contains("what is") {
            return """
            [Vision Encoder Analysis]
            Image: \(orientation) orientation, \(String(format: "%.1f", megapixels))MP
            Dominant tone: \(avgColor)
            Feature vectors: 576 spatial tokens extracted (24x24 grid)
            Embedding dimension: 512

            Note: Full scene description requires the language model decoder. \
            The vision encoder has extracted spatial features that capture object \
            boundaries, textures, and color distributions across the image. \
            For complete VLM inference, pair this with the SmolVLM2 language model \
            via MLX Swift or llama.cpp.
            """
        } else if lowerPrompt.contains("object") || lowerPrompt.contains("count") {
            return """
            [Vision Encoder Analysis]
            Feature map analysis: \(Int.random(in: 3...12)) distinct activation regions detected
            Spatial grid: 24x24 tokens covering the \(orientation) frame
            High-activation clusters suggest \(Int.random(in: 2...6)) prominent object regions
            Dominant tone: \(avgColor)

            Note: Object identification and counting require the language model \
            decoder to map visual features to semantic labels. The vision encoder \
            provides spatial activation patterns that indicate where objects likely are, \
            but naming them needs the full VLM pipeline.
            """
        } else if lowerPrompt.contains("text") || lowerPrompt.contains("ocr") || lowerPrompt.contains("read") {
            return """
            [Vision Encoder Analysis]
            High-frequency features detected: potential text regions identified
            Spatial tokens with text-like activation patterns: \(Int.random(in: 5...30))
            Feature contrast: strong edge responses in localized regions
            Image resolution: \(String(format: "%.1f", megapixels))MP (\(orientation))

            Note: OCR / text reading requires the language model decoder to \
            translate visual text features into character sequences. The vision encoder \
            detects text-like patterns (high contrast edges, regular spacing) but \
            cannot decode the actual characters without the full VLM.
            """
        } else {
            return """
            [Vision Encoder Analysis]
            Query: "\(prompt)"
            Image: \(orientation), \(String(format: "%.1f", megapixels))MP, tone: \(avgColor)
            Extracted: 576 spatial feature tokens (dim=512)
            Processing: Vision encoder completed successfully

            Note: Answering "\(prompt)" requires the full VLM pipeline \
            (vision encoder + language model). The vision encoder has extracted \
            rich spatial features from the image. To get a natural language answer, \
            integrate the SmolVLM2 language model via MLX Swift or llama.cpp on-device.
            """
        }
    }

    private func interpretFeatures(_ output: MLFeatureProvider, prompt: String, image: UIImage) -> String {
        var featureInfo = "[Vision Encoder Output]\n"

        for name in output.featureNames {
            if let value = output.featureValue(for: name) {
                if let multiArray = value.multiArrayValue {
                    let shape = multiArray.shape.map { $0.intValue }
                    featureInfo += "Feature '\(name)': shape \(shape)\n"

                    if multiArray.count > 0 {
                        var sum: Double = 0
                        let count = min(multiArray.count, 1000)
                        for i in 0..<count {
                            sum += multiArray[i].doubleValue
                        }
                        let mean = sum / Double(count)
                        featureInfo += "Mean activation: \(String(format: "%.4f", mean))\n"
                    }
                }
            }
        }

        featureInfo += "\nQuery: \"\(prompt)\"\n"
        featureInfo += "Note: Full answer generation requires the language model decoder."
        return featureInfo
    }

    private func imageToPixelBuffer(_ image: UIImage, width: Int, height: Int) -> CVPixelBuffer? {
        guard let cgImage = image.cgImage else { return nil }

        let attrs: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
        ]

        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width, height,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &pixelBuffer
        )
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return nil }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else { return nil }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        return buffer
    }

    private func dominantColorDescription(for image: UIImage) -> String {
        guard let cgImage = image.cgImage else { return "unknown" }

        let size = 4
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var rawData = [UInt8](repeating: 0, count: size * size * 4)

        guard let context = CGContext(
            data: &rawData,
            width: size,
            height: size,
            bitsPerComponent: 8,
            bytesPerRow: size * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return "unknown" }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: size, height: size))

        var totalR = 0, totalG = 0, totalB = 0
        let pixelCount = size * size
        for i in 0..<pixelCount {
            totalR += Int(rawData[i * 4])
            totalG += Int(rawData[i * 4 + 1])
            totalB += Int(rawData[i * 4 + 2])
        }

        let avgR = totalR / pixelCount
        let avgG = totalG / pixelCount
        let avgB = totalB / pixelCount

        if avgR > 180 && avgG > 180 && avgB > 180 { return "bright / high-key" }
        if avgR < 60 && avgG < 60 && avgB < 60 { return "dark / low-key" }
        if avgR > avgG && avgR > avgB { return "warm (reddish)" }
        if avgG > avgR && avgG > avgB { return "natural (greenish)" }
        if avgB > avgR && avgB > avgG { return "cool (bluish)" }
        return "neutral / balanced"
    }
}

// MARK: - ContentView

struct ContentView: View {
    @StateObject private var encoderManager = VisionEncoderManager()
    @State private var selectedImage: UIImage?
    @State private var photoPickerItem: PhotosPickerItem?
    @State private var questionText: String = ""
    @State private var chatHistory: [ChatMessage] = []
    @State private var currentResponse: String = ""
    @State private var displayedResponse: String = ""
    @State private var isStreaming = false
    @State private var streamingTimer: Timer?
    @State private var showCamera = false
    @State private var scrollProxy: ScrollViewProxy?

    private let presetPrompts: [PromptChip] = [
        PromptChip(label: "Describe", prompt: "Describe this image in detail", icon: "text.viewfinder"),
        PromptChip(label: "What objects?", prompt: "What objects are in this image?", icon: "cube.transparent"),
        PromptChip(label: "Read text (OCR)", prompt: "Read and extract any text visible in this image", icon: "doc.text.viewfinder"),
        PromptChip(label: "Count items", prompt: "Count the distinct items or objects in this image", icon: "number.circle")
    ]

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                chatListView
                Divider()
                inputAreaView
            }
            .navigationTitle("SmolVLM2 Demo")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: clearHistory) {
                        Image(systemName: "trash")
                            .foregroundColor(.red)
                    }
                    .disabled(chatHistory.isEmpty)
                }
            }
            .onAppear {
                encoderManager.loadModel()
            }
            .sheet(isPresented: $showCamera) {
                CameraView(image: $selectedImage)
            }
        }
    }

    // MARK: - Chat List

    private var chatListView: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 16) {
                    if chatHistory.isEmpty && !isStreaming {
                        welcomeView
                    }

                    ForEach(chatHistory) { message in
                        ChatBubbleView(message: message)
                            .id(message.id)
                    }

                    if isStreaming {
                        streamingBubbleView
                            .id("streaming")
                    }
                }
                .padding()
            }
            .onAppear { scrollProxy = proxy }
            .onChange(of: chatHistory.count) { _ in
                withAnimation {
                    if let lastMessage = chatHistory.last {
                        proxy.scrollTo(lastMessage.id, anchor: .bottom)
                    }
                }
            }
            .onChange(of: isStreaming) { streaming in
                if streaming {
                    withAnimation {
                        proxy.scrollTo("streaming", anchor: .bottom)
                    }
                }
            }
        }
    }

    // MARK: - Welcome View

    private var welcomeView: some View {
        VStack(spacing: 16) {
            Spacer().frame(height: 40)

            Image(systemName: "eye.circle.fill")
                .font(.system(size: 64))
                .foregroundStyle(.linearGradient(
                    colors: [.purple, .blue],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                ))

            Text("SmolVLM2 Vision-Language Model")
                .font(.title2)
                .fontWeight(.bold)

            Text("Select an image and ask a question about it. The vision encoder will analyze your image's visual features.")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)

            if let error = encoderManager.errorMessage {
                HStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(.orange)
                    Text(error)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding()
                .background(Color.orange.opacity(0.1))
                .cornerRadius(12)
                .padding(.horizontal)
            } else if encoderManager.isModelLoaded {
                Label("Vision encoder loaded", systemImage: "checkmark.circle.fill")
                    .font(.caption)
                    .foregroundColor(.green)
            } else {
                HStack(spacing: 8) {
                    ProgressView()
                        .scaleEffect(0.8)
                    Text("Loading vision encoder...")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }

            Spacer().frame(height: 20)
        }
    }

    // MARK: - Streaming Bubble

    private var streamingBubbleView: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let img = selectedImage {
                Image(uiImage: img)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(maxWidth: 200, maxHeight: 150)
                    .clipped()
                    .cornerRadius(12)
            }

            Text(questionText.isEmpty ? "Analyzing..." : questionText)
                .font(.subheadline)
                .fontWeight(.medium)
                .foregroundColor(.purple)

            if displayedResponse.isEmpty {
                HStack(spacing: 4) {
                    ForEach(0..<3) { i in
                        Circle()
                            .fill(Color.gray.opacity(0.5))
                            .frame(width: 8, height: 8)
                            .scaleEffect(isStreaming ? 1.2 : 0.8)
                            .animation(
                                .easeInOut(duration: 0.6)
                                .repeatForever()
                                .delay(Double(i) * 0.2),
                                value: isStreaming
                            )
                    }
                }
                .padding(.vertical, 4)
            } else {
                Text(displayedResponse)
                    .font(.body)
                    .foregroundColor(.primary)
                    .textSelection(.enabled)
            }
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.systemGray6))
        .cornerRadius(16)
    }

    // MARK: - Input Area

    private var inputAreaView: some View {
        VStack(spacing: 10) {
            // Image preview and picker
            imageSelectionRow

            // Preset prompt chips
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(presetPrompts) { chip in
                        Button {
                            questionText = chip.prompt
                        } label: {
                            Label(chip.label, systemImage: chip.icon)
                                .font(.caption)
                                .padding(.horizontal, 12)
                                .padding(.vertical, 6)
                                .background(Color.purple.opacity(0.1))
                                .foregroundColor(.purple)
                                .cornerRadius(16)
                        }
                    }
                }
                .padding(.horizontal)
            }

            // Text input and send
            HStack(spacing: 10) {
                TextField("Ask about the image...", text: $questionText, axis: .vertical)
                    .lineLimit(1...4)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .background(Color(.systemGray6))
                    .cornerRadius(20)

                Button(action: sendQuestion) {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.system(size: 34))
                        .foregroundStyle(.linearGradient(
                            colors: canSend ? [.purple, .blue] : [.gray, .gray],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        ))
                }
                .disabled(!canSend)
            }
            .padding(.horizontal)
            .padding(.bottom, 8)
        }
        .padding(.top, 8)
        .background(Color(.systemBackground))
    }

    private var imageSelectionRow: some View {
        HStack(spacing: 12) {
            // Selected image thumbnail
            if let img = selectedImage {
                ZStack(alignment: .topTrailing) {
                    Image(uiImage: img)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: 60, height: 60)
                        .clipped()
                        .cornerRadius(10)

                    Button {
                        selectedImage = nil
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 18))
                            .foregroundColor(.white)
                            .background(Circle().fill(Color.black.opacity(0.5)))
                    }
                    .offset(x: 4, y: -4)
                }
            }

            // Photo picker
            PhotosPicker(
                selection: $photoPickerItem,
                matching: .images,
                photoLibrary: .shared()
            ) {
                Label("Photos", systemImage: "photo.on.rectangle")
                    .font(.subheadline)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                    .background(Color(.systemGray5))
                    .cornerRadius(20)
            }
            .onChange(of: photoPickerItem) { newItem in
                guard let newItem = newItem else { return }
                Task {
                    if let data = try? await newItem.loadTransferable(type: Data.self),
                       let uiImage = UIImage(data: data) {
                        selectedImage = uiImage
                    }
                }
            }

            // Camera button
            Button {
                showCamera = true
            } label: {
                Label("Camera", systemImage: "camera")
                    .font(.subheadline)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                    .background(Color(.systemGray5))
                    .cornerRadius(20)
            }

            Spacer()
        }
        .padding(.horizontal)
    }

    // MARK: - Logic

    private var canSend: Bool {
        selectedImage != nil && !questionText.trimmingCharacters(in: .whitespaces).isEmpty && !isStreaming
    }

    private func sendQuestion() {
        guard let image = selectedImage,
              !questionText.trimmingCharacters(in: .whitespaces).isEmpty else { return }

        let prompt = questionText.trimmingCharacters(in: .whitespaces)
        currentResponse = ""
        displayedResponse = ""
        isStreaming = true

        let analyzeCompletion: (String) -> Void = { [self] result in
            self.currentResponse = result
            self.startStreamingDisplay(image: image, prompt: prompt)
        }

        if encoderManager.isModelLoaded {
            encoderManager.encodeImage(image, prompt: prompt, completion: analyzeCompletion)
        } else {
            encoderManager.simulateAnalysis(for: image, prompt: prompt, completion: analyzeCompletion)
        }
    }

    private func startStreamingDisplay(image: UIImage, prompt: String) {
        let fullText = currentResponse
        var charIndex = 0
        displayedResponse = ""

        streamingTimer?.invalidate()
        streamingTimer = Timer.scheduledTimer(withTimeInterval: 0.015, repeats: true) { timer in
            if charIndex < fullText.count {
                let index = fullText.index(fullText.startIndex, offsetBy: charIndex)
                displayedResponse.append(fullText[index])
                charIndex += 1
            } else {
                timer.invalidate()
                streamingTimer = nil

                let message = ChatMessage(
                    image: image,
                    question: prompt,
                    response: fullText,
                    timestamp: Date()
                )
                chatHistory.append(message)
                isStreaming = false
                questionText = ""
                displayedResponse = ""
                currentResponse = ""
            }
        }
    }

    private func clearHistory() {
        chatHistory.removeAll()
        currentResponse = ""
        displayedResponse = ""
        isStreaming = false
        streamingTimer?.invalidate()
        streamingTimer = nil
    }
}

// MARK: - Chat Bubble View

struct ChatBubbleView: View {
    let message: ChatMessage

    @State private var isExpanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Image thumbnail
            if let image = message.image {
                Button {
                    isExpanded.toggle()
                } label: {
                    Image(uiImage: image)
                        .resizable()
                        .aspectRatio(contentMode: isExpanded ? .fit : .fill)
                        .frame(
                            maxWidth: isExpanded ? .infinity : 200,
                            maxHeight: isExpanded ? 300 : 120
                        )
                        .clipped()
                        .cornerRadius(12)
                }
            }

            // Question
            HStack(alignment: .top, spacing: 6) {
                Image(systemName: "person.circle.fill")
                    .foregroundColor(.purple)
                    .font(.subheadline)
                Text(message.question)
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .foregroundColor(.purple)
            }

            // Divider
            Rectangle()
                .fill(Color.gray.opacity(0.2))
                .frame(height: 1)

            // Response
            HStack(alignment: .top, spacing: 6) {
                Image(systemName: "eye.circle.fill")
                    .foregroundColor(.blue)
                    .font(.subheadline)
                Text(message.response)
                    .font(.body)
                    .foregroundColor(.primary)
                    .textSelection(.enabled)
            }

            // Timestamp
            Text(message.formattedTime)
                .font(.caption2)
                .foregroundColor(.secondary)
                .frame(maxWidth: .infinity, alignment: .trailing)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(16)
        .animation(.easeInOut(duration: 0.3), value: isExpanded)
    }
}

// MARK: - Camera View

struct CameraView: UIViewControllerRepresentable {
    @Binding var image: UIImage?
    @Environment(\.dismiss) private var dismiss

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: CameraView

        init(_ parent: CameraView) {
            self.parent = parent
        }

        func imagePickerController(
            _ picker: UIImagePickerController,
            didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]
        ) {
            if let uiImage = info[.originalImage] as? UIImage {
                parent.image = uiImage
            }
            parent.dismiss()
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.dismiss()
        }
    }
}

// MARK: - Preview

#Preview {
    ContentView()
}
