import SwiftUI
import UIKit
import CoreML
import PhotosUI
import Accelerate

// MARK: - Data Types

/// Represents a single detected text region with its bounding box and recognized text
struct TextRegion: Identifiable {
    let id = UUID()
    let boundingBox: CGRect      // Normalized coordinates (0...1)
    let text: String
    let confidence: Float
    let color: Color
}

/// Processing state for the two-stage OCR pipeline
enum OCRProcessingStep: String {
    case idle = "Ready"
    case detecting = "Detecting text regions..."
    case recognizing = "Recognizing text..."
    case done = "Complete"
}

// MARK: - PP-OCRv5 Processor

/// Two-stage OCR pipeline: text detection followed by text recognition
class PPOCRProcessor: ObservableObject {
    @Published var inputImage: UIImage?
    @Published var textRegions: [TextRegion] = []
    @Published var fullText: String = ""
    @Published var isProcessing = false
    @Published var processingStep: OCRProcessingStep = .idle
    @Published var errorMessage: String?
    @Published var detectionTime: Double = 0
    @Published var recognitionTime: Double = 0
    @Published var detectedLanguage: String = "Unknown"

    private var detModel: MLModel?
    private var recModel: MLModel?

    private let detInputSize = 640
    private let recHeight = 48
    private let recWidth = 320
    private let detThreshold: Float = 0.3
    private let boxThreshold: Float = 0.5
    private let minBoxSize: Float = 3.0

    /// Character set for CTC decoding (simplified multilingual set)
    private let vocabulary: [Character] = {
        var chars: [Character] = [" "]  // Index 0 = blank for CTC
        // ASCII printable characters
        for i in 32...126 {
            chars.append(Character(UnicodeScalar(i)!))
        }
        // Common CJK characters (simplified subset)
        let cjkRanges: [ClosedRange<UInt32>] = [
            0x4E00...0x4E50,  // Common Chinese
            0x3041...0x3096,  // Hiragana
            0x30A1...0x30FA,  // Katakana
            0xAC00...0xAC50,  // Korean Hangul
        ]
        for range in cjkRanges {
            for codePoint in range {
                if let scalar = UnicodeScalar(codePoint) {
                    chars.append(Character(scalar))
                }
            }
        }
        return chars
    }()

    /// Box colors for different detected regions
    private let boxColors: [Color] = [
        .red, .blue, .green, .orange, .purple,
        .pink, .yellow, .cyan, .mint, .indigo,
        .teal, .brown
    ]

    init() {
        loadModels()
    }

    private func loadModels() {
        let config = MLModelConfiguration()
        config.computeUnits = .all

        // Load detection model
        if let detURL = Bundle.main.url(forResource: "PPOCRv5_Det", withExtension: "mlmodelc") {
            do {
                detModel = try MLModel(contentsOf: detURL, configuration: config)
            } catch {
                errorMessage = "Failed to load detection model: \(error.localizedDescription)"
            }
        } else {
            errorMessage = "Detection model not found. Please add PPOCRv5_Det.mlmodelc to the project bundle."
        }

        // Load recognition model
        if let recURL = Bundle.main.url(forResource: "PPOCRv5_Rec", withExtension: "mlmodelc") {
            do {
                recModel = try MLModel(contentsOf: recURL, configuration: config)
            } catch {
                let msg = "Failed to load recognition model: \(error.localizedDescription)"
                errorMessage = errorMessage == nil ? msg : errorMessage! + "\n" + msg
            }
        } else {
            let msg = "Recognition model not found. Please add PPOCRv5_Rec.mlmodelc to the project bundle."
            errorMessage = errorMessage == nil ? msg : errorMessage! + "\n" + msg
        }
    }

    // MARK: - Image Preprocessing

    /// Resize and normalize image to CHW float array for detection model
    private func preprocessForDetection(_ image: UIImage) -> [Float]? {
        guard let cgImage = image.cgImage else { return nil }
        let size = detInputSize
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var pixelData = [UInt8](repeating: 0, count: size * size * 4)

        guard let context = CGContext(
            data: &pixelData,
            width: size,
            height: size,
            bitsPerComponent: 8,
            bytesPerRow: size * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else { return nil }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: size, height: size))

        // ImageNet normalization: (pixel/255 - mean) / std
        let mean: [Float] = [0.485, 0.456, 0.406]
        let std: [Float] = [0.229, 0.224, 0.225]

        var floatData = [Float](repeating: 0, count: 3 * size * size)
        for y in 0..<size {
            for x in 0..<size {
                let pixelIndex = (y * size + x) * 4
                let spatialIndex = y * size + x
                for c in 0..<3 {
                    let normalized = (Float(pixelData[pixelIndex + c]) / 255.0 - mean[c]) / std[c]
                    floatData[c * size * size + spatialIndex] = normalized
                }
            }
        }
        return floatData
    }

    /// Crop and resize a text region for the recognition model
    private func preprocessForRecognition(_ image: UIImage, box: CGRect) -> [Float]? {
        guard let cgImage = image.cgImage else { return nil }

        let imgWidth = CGFloat(cgImage.width)
        let imgHeight = CGFloat(cgImage.height)

        // Convert normalized box to pixel coordinates with padding
        let padding: CGFloat = 2.0
        let cropX = max(0, box.origin.x * imgWidth - padding)
        let cropY = max(0, box.origin.y * imgHeight - padding)
        let cropW = min(imgWidth - cropX, box.width * imgWidth + 2 * padding)
        let cropH = min(imgHeight - cropY, box.height * imgHeight + 2 * padding)

        let cropRect = CGRect(x: cropX, y: cropY, width: cropW, height: cropH)
        guard cropW > 0, cropH > 0,
              let croppedCG = cgImage.cropping(to: cropRect) else { return nil }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let w = recWidth
        let h = recHeight
        var pixelData = [UInt8](repeating: 0, count: w * h * 4)

        guard let context = CGContext(
            data: &pixelData,
            width: w,
            height: h,
            bitsPerComponent: 8,
            bytesPerRow: w * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else { return nil }

        // Fill with white background, then draw the cropped text region
        context.setFillColor(UIColor.white.cgColor)
        context.fill(CGRect(x: 0, y: 0, width: w, height: h))

        // Maintain aspect ratio
        let scaleX = CGFloat(w) / CGFloat(croppedCG.width)
        let scaleY = CGFloat(h) / CGFloat(croppedCG.height)
        let scale = min(scaleX, scaleY)
        let drawW = CGFloat(croppedCG.width) * scale
        let drawH = CGFloat(croppedCG.height) * scale
        let drawX = (CGFloat(w) - drawW) / 2.0
        let drawY = (CGFloat(h) - drawH) / 2.0

        context.draw(croppedCG, in: CGRect(x: drawX, y: drawY, width: drawW, height: drawH))

        let mean: [Float] = [0.5, 0.5, 0.5]
        let std: [Float] = [0.5, 0.5, 0.5]

        var floatData = [Float](repeating: 0, count: 3 * w * h)
        for y in 0..<h {
            for x in 0..<w {
                let pixelIndex = (y * w + x) * 4
                let spatialIndex = y * w + x
                for c in 0..<3 {
                    let normalized = (Float(pixelData[pixelIndex + c]) / 255.0 - mean[c]) / std[c]
                    floatData[c * w * h + spatialIndex] = normalized
                }
            }
        }
        return floatData
    }

    // MARK: - Detection Post-processing

    /// Extract text bounding boxes from the detection heatmap using threshold + contour finding
    private func extractBoxes(from heatmap: [Float], width: Int, height: Int) -> [CGRect] {
        // Apply threshold to create binary mask
        var binaryMask = [UInt8](repeating: 0, count: width * height)
        for i in 0..<(width * height) {
            binaryMask[i] = heatmap[i] > detThreshold ? 255 : 0
        }

        // Connected component labeling to find text regions
        var labels = [Int](repeating: 0, count: width * height)
        var currentLabel = 0
        var labelBoxes: [Int: (minX: Int, minY: Int, maxX: Int, maxY: Int)] = [:]

        for y in 0..<height {
            for x in 0..<width {
                let idx = y * width + x
                if binaryMask[idx] == 255 && labels[idx] == 0 {
                    currentLabel += 1
                    // Flood fill from this pixel
                    var stack = [(x, y)]
                    var minX = x, minY = y, maxX = x, maxY = y

                    while !stack.isEmpty {
                        let (cx, cy) = stack.removeLast()
                        let cidx = cy * width + cx
                        guard cx >= 0, cx < width, cy >= 0, cy < height,
                              binaryMask[cidx] == 255, labels[cidx] == 0 else { continue }

                        labels[cidx] = currentLabel
                        minX = min(minX, cx)
                        minY = min(minY, cy)
                        maxX = max(maxX, cx)
                        maxY = max(maxY, cy)

                        // 4-connected neighbors
                        stack.append((cx + 1, cy))
                        stack.append((cx - 1, cy))
                        stack.append((cx, cy + 1))
                        stack.append((cx, cy - 1))
                    }
                    labelBoxes[currentLabel] = (minX, minY, maxX, maxY)
                }
            }
        }

        // Convert to normalized CGRect, filter by minimum size
        var boxes: [CGRect] = []
        let fw = Float(width)
        let fh = Float(height)

        for (_, box) in labelBoxes {
            let bw = Float(box.maxX - box.minX)
            let bh = Float(box.maxY - box.minY)

            guard bw >= minBoxSize, bh >= minBoxSize else { continue }

            let rect = CGRect(
                x: CGFloat(Float(box.minX) / fw),
                y: CGFloat(Float(box.minY) / fh),
                width: CGFloat(bw / fw),
                height: CGFloat(bh / fh)
            )
            boxes.append(rect)
        }

        // Sort boxes top-to-bottom, left-to-right
        boxes.sort { a, b in
            if abs(a.origin.y - b.origin.y) < 0.02 {
                return a.origin.x < b.origin.x
            }
            return a.origin.y < b.origin.y
        }

        return boxes
    }

    // MARK: - Recognition Post-processing (CTC Decoding)

    /// CTC greedy decode: pick the most probable character at each timestep, collapse repeats, remove blanks
    private func ctcDecode(probabilities: [Float], timesteps: Int, numClasses: Int) -> (String, Float) {
        var decoded: [Int] = []
        var totalConfidence: Float = 0
        var validSteps = 0

        for t in 0..<timesteps {
            var maxIdx = 0
            var maxVal: Float = -Float.infinity
            for c in 0..<numClasses {
                let val = probabilities[t * numClasses + c]
                if val > maxVal {
                    maxVal = val
                    maxIdx = c
                }
            }

            // Skip blank token (index 0)
            if maxIdx != 0 {
                // Collapse repeated characters
                if decoded.isEmpty || decoded.last != maxIdx {
                    decoded.append(maxIdx)
                    totalConfidence += maxVal
                    validSteps += 1
                }
            }
        }

        let avgConfidence = validSteps > 0 ? totalConfidence / Float(validSteps) : 0
        let text = String(decoded.compactMap { idx -> Character? in
            guard idx > 0, idx < vocabulary.count else { return nil }
            return vocabulary[idx]
        })

        return (text, avgConfidence)
    }

    // MARK: - Language Detection

    /// Simple heuristic language detection based on character ranges
    private func detectLanguage(in text: String) -> String {
        var hasChinese = false
        var hasJapanese = false
        var hasKorean = false
        var hasLatin = false

        for scalar in text.unicodeScalars {
            let value = scalar.value
            if (0x4E00...0x9FFF).contains(value) {
                hasChinese = true
            } else if (0x3040...0x309F).contains(value) || (0x30A0...0x30FF).contains(value) {
                hasJapanese = true
            } else if (0xAC00...0xD7AF).contains(value) {
                hasKorean = true
            } else if (0x0041...0x007A).contains(value) {
                hasLatin = true
            }
        }

        var languages: [String] = []
        if hasJapanese { languages.append("Japanese") }
        if hasChinese { languages.append("Chinese") }
        if hasKorean { languages.append("Korean") }
        if hasLatin { languages.append("English") }

        return languages.isEmpty ? "Unknown" : languages.joined(separator: ", ")
    }

    // MARK: - Main OCR Pipeline

    /// Run the full two-stage OCR pipeline: detection then recognition
    func runOCR(on image: UIImage) async {
        guard detModel != nil || recModel != nil else {
            await MainActor.run {
                errorMessage = "Models are not loaded. Please add PPOCRv5_Det.mlmodelc and PPOCRv5_Rec.mlmodelc to the bundle."
            }
            return
        }

        await MainActor.run {
            inputImage = image
            textRegions = []
            fullText = ""
            isProcessing = true
            processingStep = .detecting
            errorMessage = nil
            detectionTime = 0
            recognitionTime = 0
            detectedLanguage = "Unknown"
        }

        // Stage 1: Text Detection
        var detectedBoxes: [CGRect] = []

        if let detModel = detModel {
            do {
                guard let inputData = preprocessForDetection(image) else {
                    await MainActor.run {
                        errorMessage = "Failed to preprocess image for detection."
                        isProcessing = false
                        processingStep = .idle
                    }
                    return
                }

                let inputArray = try MLMultiArray(
                    shape: [1, 3, NSNumber(value: detInputSize), NSNumber(value: detInputSize)],
                    dataType: .float32
                )
                let ptr = inputArray.dataPointer.bindMemory(to: Float.self, capacity: inputData.count)
                for i in 0..<inputData.count { ptr[i] = inputData[i] }

                let inputFeatures = try MLDictionaryFeatureProvider(
                    dictionary: ["image": MLFeatureValue(multiArray: inputArray)]
                )

                let detStart = CFAbsoluteTimeGetCurrent()
                let detOutput = try detModel.prediction(from: inputFeatures)
                let detEnd = CFAbsoluteTimeGetCurrent()

                await MainActor.run { detectionTime = (detEnd - detStart) * 1000.0 }

                // Parse detection output heatmap
                if let outputNames = detOutput.featureNames as? Set<String>,
                   let firstOutput = outputNames.first,
                   let heatmapArray = detOutput.featureValue(for: firstOutput)?.multiArrayValue {

                    let totalElements = heatmapArray.count
                    let heatmapPtr = heatmapArray.dataPointer.bindMemory(to: Float.self, capacity: totalElements)
                    // The output is typically (1, 1, H, W) -- use the spatial dims
                    let outH = heatmapArray.shape.count >= 3 ? heatmapArray.shape[heatmapArray.shape.count - 2].intValue : detInputSize
                    let outW = heatmapArray.shape.count >= 2 ? heatmapArray.shape[heatmapArray.shape.count - 1].intValue : detInputSize
                    let spatialSize = outH * outW
                    let offset = totalElements > spatialSize ? totalElements - spatialSize : 0

                    var heatmapData = [Float](repeating: 0, count: spatialSize)
                    for i in 0..<spatialSize {
                        // Sigmoid activation if raw logits
                        let val = heatmapPtr[offset + i]
                        heatmapData[i] = 1.0 / (1.0 + exp(-val))
                    }

                    detectedBoxes = extractBoxes(from: heatmapData, width: outW, height: outH)
                }
            } catch {
                await MainActor.run {
                    errorMessage = "Detection error: \(error.localizedDescription)"
                    isProcessing = false
                    processingStep = .idle
                }
                return
            }
        }

        // If no boxes detected, create a single full-image box as fallback
        if detectedBoxes.isEmpty {
            detectedBoxes = [CGRect(x: 0.02, y: 0.02, width: 0.96, height: 0.96)]
        }

        // Stage 2: Text Recognition
        await MainActor.run { processingStep = .recognizing }

        var regions: [TextRegion] = []
        var allTexts: [String] = []
        let recStart = CFAbsoluteTimeGetCurrent()

        for (index, box) in detectedBoxes.enumerated() {
            if let recModel = recModel {
                do {
                    guard let recInput = preprocessForRecognition(image, box: box) else { continue }

                    let recArray = try MLMultiArray(
                        shape: [1, 3, NSNumber(value: recHeight), NSNumber(value: recWidth)],
                        dataType: .float32
                    )
                    let recPtr = recArray.dataPointer.bindMemory(to: Float.self, capacity: recInput.count)
                    for i in 0..<recInput.count { recPtr[i] = recInput[i] }

                    let recFeatures = try MLDictionaryFeatureProvider(
                        dictionary: ["image": MLFeatureValue(multiArray: recArray)]
                    )

                    let recOutput = try recModel.prediction(from: recFeatures)

                    // Parse recognition output: (1, timesteps, num_classes)
                    if let outputNames = recOutput.featureNames as? Set<String>,
                       let firstOutput = outputNames.first,
                       let probArray = recOutput.featureValue(for: firstOutput)?.multiArrayValue {

                        let totalCount = probArray.count
                        let probPtr = probArray.dataPointer.bindMemory(to: Float.self, capacity: totalCount)
                        var probData = [Float](repeating: 0, count: totalCount)
                        for i in 0..<totalCount { probData[i] = probPtr[i] }

                        // Determine timesteps and num_classes from shape
                        let numClasses = probArray.shape.last?.intValue ?? vocabulary.count
                        let timesteps = totalCount / numClasses

                        let (text, confidence) = ctcDecode(
                            probabilities: probData,
                            timesteps: timesteps,
                            numClasses: numClasses
                        )

                        if !text.isEmpty {
                            let color = boxColors[index % boxColors.count]
                            regions.append(TextRegion(
                                boundingBox: box,
                                text: text,
                                confidence: confidence,
                                color: color
                            ))
                            allTexts.append(text)
                        }
                    }
                } catch {
                    // Skip this region on error, continue with others
                    continue
                }
            } else {
                // No recognition model -- add placeholder regions
                let color = boxColors[index % boxColors.count]
                regions.append(TextRegion(
                    boundingBox: box,
                    text: "[Recognition model not loaded]",
                    confidence: 0,
                    color: color
                ))
            }
        }

        let recEnd = CFAbsoluteTimeGetCurrent()
        let combinedText = allTexts.joined(separator: "\n")
        let language = detectLanguage(in: combinedText)

        await MainActor.run {
            recognitionTime = (recEnd - recStart) * 1000.0
            textRegions = regions
            fullText = combinedText
            detectedLanguage = language
            processingStep = .done
            isProcessing = false
        }
    }
}

// MARK: - Image Picker

struct ImagePicker: UIViewControllerRepresentable {
    @Binding var image: UIImage?

    func makeUIViewController(context: Context) -> PHPickerViewController {
        var config = PHPickerConfiguration()
        config.filter = .images
        config.selectionLimit = 1
        let picker = PHPickerViewController(configuration: config)
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: PHPickerViewController, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    class Coordinator: NSObject, PHPickerViewControllerDelegate {
        let parent: ImagePicker

        init(_ parent: ImagePicker) {
            self.parent = parent
        }

        func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
            picker.dismiss(animated: true)
            guard let provider = results.first?.itemProvider,
                  provider.canLoadObject(ofClass: UIImage.self) else { return }
            provider.loadObject(ofClass: UIImage.self) { image, _ in
                DispatchQueue.main.async {
                    self.parent.image = image as? UIImage
                }
            }
        }
    }
}

// MARK: - Camera Capture View

struct CameraCaptureView: UIViewControllerRepresentable {
    @Binding var image: UIImage?
    @Environment(\.dismiss) var dismiss

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
        let parent: CameraCaptureView

        init(_ parent: CameraCaptureView) {
            self.parent = parent
        }

        func imagePickerController(_ picker: UIImagePickerController,
                                   didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
            if let image = info[.originalImage] as? UIImage {
                parent.image = image
            }
            parent.dismiss()
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.dismiss()
        }
    }
}

// MARK: - Text Box Overlay View

/// Draws colored bounding boxes on the input image showing detected text regions
struct TextBoxOverlayView: View {
    let image: UIImage
    let regions: [TextRegion]

    var body: some View {
        GeometryReader { geometry in
            let imageSize = image.size
            let viewSize = geometry.size
            let scaleX = viewSize.width / imageSize.width
            let scaleY = viewSize.height / imageSize.height
            let scale = min(scaleX, scaleY)
            let drawWidth = imageSize.width * scale
            let drawHeight = imageSize.height * scale
            let offsetX = (viewSize.width - drawWidth) / 2
            let offsetY = (viewSize.height - drawHeight) / 2

            ZStack(alignment: .topLeading) {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(width: viewSize.width, height: viewSize.height)

                ForEach(regions) { region in
                    let box = region.boundingBox
                    let x = offsetX + box.origin.x * drawWidth
                    let y = offsetY + box.origin.y * drawHeight
                    let w = box.width * drawWidth
                    let h = box.height * drawHeight

                    Rectangle()
                        .stroke(region.color, lineWidth: 2)
                        .background(region.color.opacity(0.1))
                        .frame(width: w, height: h)
                        .position(x: x + w / 2, y: y + h / 2)
                }
            }
        }
    }
}

// MARK: - Content View

struct ContentView: View {
    @StateObject private var processor = PPOCRProcessor()
    @State private var showImagePicker = false
    @State private var showCamera = false
    @State private var selectedImage: UIImage?
    @State private var showFullText = false
    @State private var copiedToClipboard = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    // Header
                    headerSection

                    // Error display
                    if let error = processor.errorMessage {
                        errorBanner(error)
                    }

                    // Input buttons
                    inputButtonsSection

                    // Processing indicator
                    if processor.isProcessing {
                        processingIndicator
                    }

                    // Timing info
                    if processor.detectionTime > 0 || processor.recognitionTime > 0 {
                        timingSection
                    }

                    // Image with text box overlay
                    if let image = processor.inputImage {
                        imageOverlaySection(image: image)
                    }

                    // Detected text regions list
                    if !processor.textRegions.isEmpty {
                        detectedRegionsSection
                    }

                    // Full text result
                    if !processor.fullText.isEmpty {
                        fullTextSection
                    }

                    Spacer(minLength: 40)
                }
                .padding(.vertical)
            }
            .navigationTitle("PP-OCRv5")
            .sheet(isPresented: $showImagePicker) {
                ImagePicker(image: $selectedImage)
            }
            .sheet(isPresented: $showCamera) {
                CameraCaptureView(image: $selectedImage)
            }
            .sheet(isPresented: $showFullText) {
                fullTextSheet
            }
            .onChange(of: selectedImage) { newValue in
                guard let image = newValue else { return }
                Task {
                    await processor.runOCR(on: image)
                }
            }
        }
    }

    // MARK: - Header

    private var headerSection: some View {
        VStack(spacing: 8) {
            Image(systemName: "doc.text.viewfinder")
                .font(.system(size: 50))
                .foregroundColor(.blue)
            Text("Multilingual OCR")
                .font(.title2.bold())
            Text("PP-OCRv5 text detection and recognition\nSupports English, Chinese, Japanese, Korean")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding()
    }

    // MARK: - Input Buttons

    private var inputButtonsSection: some View {
        HStack(spacing: 12) {
            Button {
                showImagePicker = true
            } label: {
                Label("Photo Library", systemImage: "photo.badge.plus")
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(12)
            }

            Button {
                showCamera = true
            } label: {
                Label("Camera", systemImage: "camera.fill")
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.green)
                    .foregroundColor(.white)
                    .cornerRadius(12)
            }
        }
        .padding(.horizontal)
    }

    // MARK: - Processing Indicator

    private var processingIndicator: some View {
        VStack(spacing: 12) {
            ProgressView()
                .scaleEffect(1.2)
            Text(processor.processingStep.rawValue)
                .font(.subheadline.bold())
                .foregroundColor(.blue)

            // Step indicators
            HStack(spacing: 16) {
                stepBadge(
                    title: "Detect",
                    icon: "rectangle.dashed",
                    isActive: processor.processingStep == .detecting,
                    isDone: processor.processingStep == .recognizing || processor.processingStep == .done
                )
                Image(systemName: "arrow.right")
                    .foregroundColor(.secondary)
                stepBadge(
                    title: "Recognize",
                    icon: "textformat.abc",
                    isActive: processor.processingStep == .recognizing,
                    isDone: processor.processingStep == .done
                )
            }
        }
        .padding()
        .background(Color.blue.opacity(0.05))
        .cornerRadius(12)
        .padding(.horizontal)
    }

    private func stepBadge(title: String, icon: String, isActive: Bool, isDone: Bool) -> some View {
        HStack(spacing: 4) {
            Image(systemName: isDone ? "checkmark.circle.fill" : icon)
                .foregroundColor(isDone ? .green : (isActive ? .blue : .gray))
            Text(title)
                .font(.caption.bold())
                .foregroundColor(isDone ? .green : (isActive ? .blue : .gray))
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(
            RoundedRectangle(cornerRadius: 6)
                .fill(isDone ? Color.green.opacity(0.1) : (isActive ? Color.blue.opacity(0.1) : Color.gray.opacity(0.1)))
        )
    }

    // MARK: - Timing Section

    private var timingSection: some View {
        HStack(spacing: 16) {
            if processor.detectionTime > 0 {
                HStack(spacing: 4) {
                    Image(systemName: "rectangle.dashed")
                        .foregroundColor(.orange)
                    Text(String(format: "Det: %.0f ms", processor.detectionTime))
                        .font(.caption.bold())
                        .foregroundColor(.orange)
                }
            }
            if processor.recognitionTime > 0 {
                HStack(spacing: 4) {
                    Image(systemName: "textformat.abc")
                        .foregroundColor(.purple)
                    Text(String(format: "Rec: %.0f ms", processor.recognitionTime))
                        .font(.caption.bold())
                        .foregroundColor(.purple)
                }
            }
            if processor.detectedLanguage != "Unknown" {
                HStack(spacing: 4) {
                    Image(systemName: "globe")
                        .foregroundColor(.teal)
                    Text(processor.detectedLanguage)
                        .font(.caption.bold())
                        .foregroundColor(.teal)
                }
            }
        }
        .padding(.horizontal)
    }

    // MARK: - Image Overlay

    private func imageOverlaySection(image: UIImage) -> some View {
        VStack(spacing: 8) {
            HStack {
                Text("Detected Text Regions")
                    .font(.headline)
                Spacer()
                if !processor.textRegions.isEmpty {
                    Text("\(processor.textRegions.count) regions")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 2)
                        .background(Color.secondary.opacity(0.1))
                        .cornerRadius(8)
                }
            }
            .padding(.horizontal)

            TextBoxOverlayView(image: image, regions: processor.textRegions)
                .frame(height: 300)
                .cornerRadius(12)
                .padding(.horizontal)
        }
    }

    // MARK: - Detected Regions List

    private var detectedRegionsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Recognized Text")
                .font(.headline)
                .padding(.horizontal)

            ForEach(processor.textRegions) { region in
                HStack(alignment: .top, spacing: 8) {
                    RoundedRectangle(cornerRadius: 3)
                        .fill(region.color)
                        .frame(width: 6, height: 6)
                        .padding(.top, 6)

                    VStack(alignment: .leading, spacing: 2) {
                        Text(region.text)
                            .font(.body)
                            .textSelection(.enabled)
                        Text(String(format: "Confidence: %.1f%%", region.confidence * 100))
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }

                    Spacer()
                }
                .padding(.horizontal)
                .padding(.vertical, 4)
                .background(region.color.opacity(0.05))
                .cornerRadius(8)
                .padding(.horizontal)
            }
        }
    }

    // MARK: - Full Text Section

    private var fullTextSection: some View {
        VStack(spacing: 12) {
            HStack {
                Text("Full Text Result")
                    .font(.headline)
                Spacer()

                Button {
                    UIPasteboard.general.string = processor.fullText
                    copiedToClipboard = true
                    DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                        copiedToClipboard = false
                    }
                } label: {
                    Label(
                        copiedToClipboard ? "Copied" : "Copy All",
                        systemImage: copiedToClipboard ? "checkmark.circle.fill" : "doc.on.doc"
                    )
                    .font(.caption.bold())
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(copiedToClipboard ? Color.green : Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(8)
                }

                Button {
                    showFullText = true
                } label: {
                    Image(systemName: "arrow.up.left.and.arrow.down.right")
                        .font(.caption.bold())
                        .padding(6)
                        .background(Color.secondary.opacity(0.1))
                        .cornerRadius(8)
                }
            }
            .padding(.horizontal)

            Text(processor.fullText)
                .font(.body)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(12)
                .padding(.horizontal)
        }
    }

    // MARK: - Full Text Sheet

    private var fullTextSheet: some View {
        NavigationStack {
            ScrollView {
                Text(processor.fullText)
                    .font(.body)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
            }
            .navigationTitle("Full OCR Text")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        showFullText = false
                    }
                }
                ToolbarItem(placement: .navigationBarLeading) {
                    Button {
                        UIPasteboard.general.string = processor.fullText
                    } label: {
                        Label("Copy", systemImage: "doc.on.doc")
                    }
                }
            }
        }
    }

    // MARK: - Error Banner

    private func errorBanner(_ message: String) -> some View {
        HStack(alignment: .top) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundColor(.yellow)
            Text(message)
                .font(.caption)
        }
        .padding()
        .background(Color.red.opacity(0.1))
        .cornerRadius(8)
        .padding(.horizontal)
    }
}

// MARK: - Preview

#Preview {
    ContentView()
}
