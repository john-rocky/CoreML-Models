import SwiftUI
import UIKit
import CoreML
import Vision
import PhotosUI
import AVFoundation

// MARK: - YOLOE Open-Vocabulary Detection & Segmentation Demo
//
// YOLOE: Real-Time Seeing Anything (ICCV 2025)
// https://github.com/THU-MIG/yoloe
//
// This app demonstrates open-vocabulary object detection and instance segmentation.
// Users can type any text prompt (e.g., "coffee mug", "red car") and the model
// detects matching objects with bounding boxes and segmentation masks.
//
// Model: YOLOE-S exported to CoreML (YOLOE_S.mlmodelc)
// Input: 640x640 RGB image
// Output: bounding boxes, class confidence scores, segmentation masks
// Post-processing: Non-Maximum Suppression (NMS), confidence filtering

// MARK: - Detection Mode

enum DetectionMode: String, CaseIterable, Identifiable {
    case detection = "Detection"
    case segmentation = "Segmentation"

    var id: String { rawValue }

    var icon: String {
        switch self {
        case .detection: return "rectangle.dashed"
        case .segmentation: return "paintbrush.pointed.fill"
        }
    }
}

// MARK: - Detection Result

struct DetectionResult: Identifiable {
    let id = UUID()
    let label: String
    let confidence: Float
    let boundingBox: CGRect
    let maskData: [Float]?
    let color: Color

    var confidencePercent: String {
        String(format: "%.1f%%", confidence * 100)
    }
}

// MARK: - Preset Prompt Chips

struct PromptChip: Identifiable {
    let id = UUID()
    let label: String
    let icon: String
}

let presetChips: [PromptChip] = [
    PromptChip(label: "person", icon: "person.fill"),
    PromptChip(label: "car", icon: "car.fill"),
    PromptChip(label: "dog", icon: "dog.fill"),
    PromptChip(label: "phone", icon: "iphone"),
    PromptChip(label: "food", icon: "fork.knife"),
    PromptChip(label: "text", icon: "textformat"),
]

// MARK: - Color Palette for Detection Classes

let detectionColors: [Color] = [
    .red, .blue, .green, .orange, .purple, .pink,
    .cyan, .yellow, .mint, .indigo, .teal, .brown
]

func colorForIndex(_ index: Int) -> Color {
    detectionColors[index % detectionColors.count]
}

// MARK: - ContentView

struct ContentView: View {
    @StateObject private var viewModel = YOLOEViewModel()
    @State private var showCamera = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    // Image input section
                    imageSection

                    // Text prompt section
                    promptSection

                    // Mode toggle
                    modeToggleSection

                    // Detect button
                    detectButton

                    // Progress indicator
                    if viewModel.isProcessing {
                        progressSection
                    }

                    // Error display
                    if let error = viewModel.errorMessage {
                        errorSection(error)
                    }

                    // Detection overlay on image
                    if !viewModel.detections.isEmpty, let image = viewModel.inputImage {
                        detectionOverlaySection(image: image)
                    }

                    // Results list
                    if !viewModel.detections.isEmpty {
                        resultsListSection
                    }
                }
                .padding()
            }
            .navigationTitle("YOLOE Detector")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Menu {
                        Button(action: { viewModel.showPhotoPicker = true }) {
                            Label("Photo Library", systemImage: "photo.on.rectangle")
                        }
                        Button(action: { showCamera = true }) {
                            Label("Camera", systemImage: "camera")
                        }
                    } label: {
                        Image(systemName: "plus.circle.fill")
                            .font(.title3)
                    }
                }
            }
            .photosPicker(isPresented: $viewModel.showPhotoPicker, selection: $viewModel.selectedPhoto, matching: .images)
            .onChange(of: viewModel.selectedPhoto) { _ in
                viewModel.loadSelectedPhoto()
            }
            .fullScreenCover(isPresented: $showCamera) {
                CameraPickerView(image: $viewModel.inputImage)
                    .ignoresSafeArea()
            }
        }
    }

    // MARK: - Image Section

    private var imageSection: some View {
        Group {
            if let image = viewModel.inputImage {
                ZStack(alignment: .topTrailing) {
                    Image(uiImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(maxHeight: 300)
                        .cornerRadius(12)

                    Button(action: { viewModel.clearImage() }) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.title2)
                            .foregroundColor(.white)
                            .shadow(radius: 2)
                    }
                    .padding(8)
                }
            } else {
                VStack(spacing: 12) {
                    Image(systemName: "viewfinder")
                        .font(.system(size: 48))
                        .foregroundColor(.secondary)
                    Text("Select an Image")
                        .font(.headline)
                        .foregroundColor(.secondary)
                    Text("Use the + button to pick from library or camera")
                        .font(.caption)
                        .foregroundColor(.secondary.opacity(0.7))
                        .multilineTextAlignment(.center)
                }
                .frame(maxWidth: .infinity)
                .frame(height: 200)
                .background(Color(.systemGray6))
                .cornerRadius(12)
            }
        }
    }

    // MARK: - Prompt Section

    private var promptSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("What to detect")
                .font(.headline)

            HStack {
                Image(systemName: "magnifyingglass")
                    .foregroundColor(.secondary)
                TextField("e.g. coffee mug, red car, person with hat", text: $viewModel.promptText)
                    .textFieldStyle(.plain)
                    .autocorrectionDisabled()
                if !viewModel.promptText.isEmpty {
                    Button(action: { viewModel.promptText = "" }) {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundColor(.secondary)
                    }
                }
            }
            .padding(12)
            .background(Color(.systemGray6))
            .cornerRadius(10)

            // Preset chips
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(presetChips) { chip in
                        Button(action: {
                            appendPrompt(chip.label)
                        }) {
                            HStack(spacing: 4) {
                                Image(systemName: chip.icon)
                                    .font(.caption)
                                Text(chip.label)
                                    .font(.caption)
                                    .fontWeight(.medium)
                            }
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .background(
                                viewModel.promptText.lowercased().contains(chip.label)
                                    ? Color.accentColor.opacity(0.2)
                                    : Color(.systemGray5)
                            )
                            .foregroundColor(
                                viewModel.promptText.lowercased().contains(chip.label)
                                    ? .accentColor
                                    : .primary
                            )
                            .cornerRadius(20)
                        }
                    }
                }
            }
        }
    }

    // MARK: - Mode Toggle

    private var modeToggleSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Mode")
                .font(.headline)

            Picker("Mode", selection: $viewModel.detectionMode) {
                ForEach(DetectionMode.allCases) { mode in
                    Label(mode.rawValue, systemImage: mode.icon)
                        .tag(mode)
                }
            }
            .pickerStyle(.segmented)
        }
    }

    // MARK: - Detect Button

    private var detectButton: some View {
        Button(action: { viewModel.runDetection() }) {
            HStack {
                if viewModel.isProcessing {
                    ProgressView()
                        .tint(.white)
                } else {
                    Image(systemName: "sparkle.magnifyingglass")
                }
                Text(viewModel.isProcessing ? "Detecting..." : "Detect Objects")
                    .fontWeight(.semibold)
            }
            .frame(maxWidth: .infinity)
            .padding()
            .background(
                (viewModel.inputImage != nil && !viewModel.promptText.isEmpty && !viewModel.isProcessing)
                    ? Color.accentColor
                    : Color.gray
            )
            .foregroundColor(.white)
            .cornerRadius(12)
        }
        .disabled(viewModel.inputImage == nil || viewModel.promptText.isEmpty || viewModel.isProcessing)
    }

    // MARK: - Progress Section

    private var progressSection: some View {
        VStack(spacing: 8) {
            ProgressView(value: viewModel.progress)
                .progressViewStyle(.linear)
            Text(viewModel.statusMessage)
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }

    // MARK: - Error Section

    private func errorSection(_ error: String) -> some View {
        HStack {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundColor(.red)
            Text(error)
                .font(.caption)
                .foregroundColor(.red)
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.red.opacity(0.1))
        .cornerRadius(8)
    }

    // MARK: - Detection Overlay Section

    private func detectionOverlaySection(image: UIImage) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Results")
                .font(.headline)

            GeometryReader { geometry in
                let aspectRatio = image.size.width / image.size.height
                let displayWidth = geometry.size.width
                let displayHeight = displayWidth / aspectRatio

                ZStack(alignment: .topLeading) {
                    Image(uiImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fit)

                    // Segmentation masks
                    if viewModel.detectionMode == .segmentation {
                        ForEach(viewModel.detections) { det in
                            if let maskData = det.maskData {
                                MaskOverlayView(
                                    maskData: maskData,
                                    color: det.color,
                                    displaySize: CGSize(width: displayWidth, height: displayHeight)
                                )
                            }
                        }
                    }

                    // Bounding boxes
                    ForEach(viewModel.detections) { det in
                        let rect = convertBoundingBox(
                            det.boundingBox,
                            toViewSize: CGSize(width: displayWidth, height: displayHeight)
                        )

                        Rectangle()
                            .stroke(det.color, lineWidth: 2)
                            .frame(width: rect.width, height: rect.height)
                            .overlay(alignment: .topLeading) {
                                Text("\(det.label) \(det.confidencePercent)")
                                    .font(.system(size: 10, weight: .bold))
                                    .foregroundColor(.white)
                                    .padding(.horizontal, 4)
                                    .padding(.vertical, 2)
                                    .background(det.color.opacity(0.85))
                                    .cornerRadius(4)
                                    .offset(y: -18)
                            }
                            .position(x: rect.midX, y: rect.midY)
                    }
                }
                .frame(width: displayWidth, height: displayHeight)
            }
            .aspectRatio(image.size.width / image.size.height, contentMode: .fit)
        }
    }

    // MARK: - Results List

    private var resultsListSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Detected Objects")
                    .font(.headline)
                Spacer()
                Text("\(viewModel.detections.count) found")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            ForEach(viewModel.detections) { detection in
                DetectionRowView(
                    detection: detection,
                    sourceImage: viewModel.inputImage
                )
            }
        }
    }

    // MARK: - Helpers

    private func appendPrompt(_ text: String) {
        if viewModel.promptText.isEmpty {
            viewModel.promptText = text
        } else if !viewModel.promptText.lowercased().contains(text.lowercased()) {
            viewModel.promptText += ", \(text)"
        }
    }

    private func convertBoundingBox(_ bbox: CGRect, toViewSize size: CGSize) -> CGRect {
        let x = bbox.origin.x * size.width
        let y = bbox.origin.y * size.height
        let w = bbox.size.width * size.width
        let h = bbox.size.height * size.height
        return CGRect(x: x, y: y, width: w, height: h)
    }
}

// MARK: - Detection Row View

struct DetectionRowView: View {
    let detection: DetectionResult
    let sourceImage: UIImage?

    var body: some View {
        HStack(spacing: 12) {
            // Cropped thumbnail
            if let thumb = croppedThumbnail() {
                Image(uiImage: thumb)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(width: 50, height: 50)
                    .cornerRadius(8)
                    .clipped()
            } else {
                RoundedRectangle(cornerRadius: 8)
                    .fill(detection.color.opacity(0.2))
                    .frame(width: 50, height: 50)
                    .overlay {
                        Image(systemName: "cube.box")
                            .foregroundColor(detection.color)
                    }
            }

            VStack(alignment: .leading, spacing: 4) {
                Text(detection.label)
                    .font(.body)
                    .fontWeight(.medium)

                HStack(spacing: 8) {
                    // Confidence bar
                    GeometryReader { geo in
                        ZStack(alignment: .leading) {
                            RoundedRectangle(cornerRadius: 2)
                                .fill(Color(.systemGray5))
                            RoundedRectangle(cornerRadius: 2)
                                .fill(detection.color)
                                .frame(width: geo.size.width * CGFloat(detection.confidence))
                        }
                    }
                    .frame(height: 6)

                    Text(detection.confidencePercent)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .frame(width: 44, alignment: .trailing)
                }
            }

            Spacer()

            Circle()
                .fill(detection.color)
                .frame(width: 12, height: 12)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(10)
    }

    private func croppedThumbnail() -> UIImage? {
        guard let source = sourceImage else { return nil }
        let bbox = detection.boundingBox
        let cropRect = CGRect(
            x: bbox.origin.x * source.size.width,
            y: bbox.origin.y * source.size.height,
            width: bbox.width * source.size.width,
            height: bbox.height * source.size.height
        )
        guard cropRect.width > 0, cropRect.height > 0 else { return nil }
        guard let cgImage = source.cgImage?.cropping(to: cropRect) else { return nil }
        return UIImage(cgImage: cgImage)
    }
}

// MARK: - Mask Overlay View

struct MaskOverlayView: View {
    let maskData: [Float]
    let color: Color
    let displaySize: CGSize

    var body: some View {
        Canvas { context, size in
            let maskWidth = 160
            let maskHeight = 160
            let scaleX = size.width / CGFloat(maskWidth)
            let scaleY = size.height / CGFloat(maskHeight)

            for y in 0..<maskHeight {
                for x in 0..<maskWidth {
                    let index = y * maskWidth + x
                    guard index < maskData.count else { continue }
                    let value = maskData[index]
                    if value > 0.5 {
                        let rect = CGRect(
                            x: CGFloat(x) * scaleX,
                            y: CGFloat(y) * scaleY,
                            width: scaleX + 0.5,
                            height: scaleY + 0.5
                        )
                        context.fill(Path(rect), with: .color(color.opacity(0.35)))
                    }
                }
            }
        }
        .frame(width: displaySize.width, height: displaySize.height)
        .allowsHitTesting(false)
    }
}

// MARK: - Camera Picker

struct CameraPickerView: UIViewControllerRepresentable {
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
        let parent: CameraPickerView

        init(_ parent: CameraPickerView) {
            self.parent = parent
        }

        func imagePickerController(_ picker: UIImagePickerController,
                                   didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
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

// MARK: - ViewModel

class YOLOEViewModel: ObservableObject {
    @Published var inputImage: UIImage?
    @Published var selectedPhoto: PhotosPickerItem?
    @Published var showPhotoPicker = false
    @Published var promptText = ""
    @Published var detectionMode: DetectionMode = .detection
    @Published var isProcessing = false
    @Published var progress: Double = 0
    @Published var statusMessage = ""
    @Published var errorMessage: String?
    @Published var detections: [DetectionResult] = []

    private var mlModel: MLModel?

    // MARK: - Load Photo from Picker

    func loadSelectedPhoto() {
        guard let item = selectedPhoto else { return }
        Task {
            if let data = try? await item.loadTransferable(type: Data.self),
               let uiImage = UIImage(data: data) {
                await MainActor.run {
                    self.inputImage = uiImage
                    self.detections = []
                    self.errorMessage = nil
                }
            }
        }
    }

    func clearImage() {
        inputImage = nil
        selectedPhoto = nil
        detections = []
        errorMessage = nil
    }

    // MARK: - Run Detection

    func runDetection() {
        guard let image = inputImage, !promptText.isEmpty else { return }
        isProcessing = true
        errorMessage = nil
        detections = []
        progress = 0

        Task {
            do {
                let results = try await performDetection(image: image, prompt: promptText)
                await MainActor.run {
                    self.detections = results
                    self.isProcessing = false
                    self.progress = 1.0
                    self.statusMessage = "Done"
                }
            } catch {
                await MainActor.run {
                    self.errorMessage = error.localizedDescription
                    self.isProcessing = false
                }
            }
        }
    }

    // MARK: - CoreML Inference Pipeline

    private func performDetection(image: UIImage, prompt: String) async throws -> [DetectionResult] {
        await updateStatus("Loading model...", progress: 0.1)

        // Load the YOLOE-S CoreML model
        guard let modelURL = Bundle.main.url(forResource: "YOLOE_S", withExtension: "mlmodelc") else {
            throw YOLOEError.modelNotFound(
                "YOLOE_S.mlmodelc not found in bundle. " +
                "Please run convert_yoloe.py to export the model and add the compiled " +
                "YOLOE_S.mlmodelc to the Xcode project."
            )
        }

        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        let model = try MLModel(contentsOf: modelURL, configuration: config)

        await updateStatus("Preprocessing image...", progress: 0.3)

        // Resize image to 640x640 for model input
        guard let resizedImage = resizeImage(image, to: CGSize(width: 640, height: 640)),
              let pixelBuffer = resizedImage.toPixelBuffer(width: 640, height: 640) else {
            throw YOLOEError.processingFailed("Failed to preprocess input image.")
        }

        await updateStatus("Running YOLOE inference...", progress: 0.5)

        // Parse prompt into individual class labels
        let classLabels = prompt
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }

        // Run model prediction
        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(pixelBuffer: pixelBuffer)
        ])
        let output = try model.prediction(from: inputFeatures)

        await updateStatus("Post-processing...", progress: 0.75)

        // Extract output tensors
        // YOLOE outputs: detection boxes, scores, class predictions, and optionally masks
        let results = try parseModelOutput(
            output: output,
            classLabels: classLabels,
            imageSize: image.size,
            confidenceThreshold: 0.25,
            iouThreshold: 0.45,
            includeMasks: detectionMode == .segmentation
        )

        await updateStatus("Complete!", progress: 1.0)
        return results
    }

    // MARK: - Parse Model Output

    private func parseModelOutput(
        output: MLFeatureProvider,
        classLabels: [String],
        imageSize: CGSize,
        confidenceThreshold: Float,
        iouThreshold: Float,
        includeMasks: Bool
    ) throws -> [DetectionResult] {
        // Attempt to read the primary output feature
        // YOLOE typically outputs a combined tensor with shape [1, num_detections, 4+num_classes+mask_dim]
        // or separate outputs for boxes, scores, and masks.
        //
        // The exact output format depends on the export configuration.
        // Common output names: "output0" (detection), "output1" (segmentation protos)

        var rawDetections: [(bbox: CGRect, confidence: Float, classIndex: Int, maskCoeffs: [Float]?)] = []

        // Try to access detection output
        let featureNames = output.featureNames
        guard let primaryName = featureNames.first,
              let primaryValue = output.featureValue(for: primaryName),
              let detArray = primaryValue.multiArrayValue else {
            throw YOLOEError.processingFailed("Could not read model output tensor.")
        }

        let shape = detArray.shape.map { $0.intValue }
        // Expected shape: [1, numPredictions, attributes] or [1, attributes, numPredictions]
        // attributes = 4 (bbox) + numClasses + maskDim

        guard shape.count >= 2 else {
            throw YOLOEError.processingFailed("Unexpected output shape: \(shape)")
        }

        let numClasses = classLabels.count
        let numPredictions: Int
        let attributeDim: Int

        // YOLO-style output is typically [1, 4+numClasses+maskDim, numPredictions] (transposed)
        if shape.count == 3 {
            attributeDim = shape[1]
            numPredictions = shape[2]
        } else {
            numPredictions = shape[0]
            attributeDim = shape[1]
        }

        let maskDim = max(0, attributeDim - 4 - numClasses)
        let pointer = detArray.dataPointer.assumingMemoryBound(to: Float.self)

        for i in 0..<numPredictions {
            // Read bbox center_x, center_y, width, height (normalized 0-640)
            let cx: Float
            let cy: Float
            let w: Float
            let h: Float

            if shape.count == 3 {
                cx = pointer[0 * numPredictions + i]
                cy = pointer[1 * numPredictions + i]
                w  = pointer[2 * numPredictions + i]
                h  = pointer[3 * numPredictions + i]
            } else {
                let base = i * attributeDim
                cx = pointer[base + 0]
                cy = pointer[base + 1]
                w  = pointer[base + 2]
                h  = pointer[base + 3]
            }

            // Find the best class score
            var bestScore: Float = 0
            var bestClassIdx = 0
            for c in 0..<numClasses {
                let score: Float
                if shape.count == 3 {
                    score = pointer[(4 + c) * numPredictions + i]
                } else {
                    score = pointer[i * attributeDim + 4 + c]
                }
                if score > bestScore {
                    bestScore = score
                    bestClassIdx = c
                }
            }

            guard bestScore >= confidenceThreshold else { continue }

            // Convert from center format to origin format, normalized to 0..1
            let normX = (cx - w / 2.0) / 640.0
            let normY = (cy - h / 2.0) / 640.0
            let normW = w / 640.0
            let normH = h / 640.0

            let bbox = CGRect(
                x: CGFloat(max(0, normX)),
                y: CGFloat(max(0, normY)),
                width: CGFloat(min(1.0 - max(0, normX), max(0, normW))),
                height: CGFloat(min(1.0 - max(0, normY), max(0, normH)))
            )

            // Extract mask coefficients if available
            var maskCoeffs: [Float]?
            if includeMasks && maskDim > 0 {
                maskCoeffs = (0..<maskDim).map { m in
                    if shape.count == 3 {
                        return pointer[(4 + numClasses + m) * numPredictions + i]
                    } else {
                        return pointer[i * attributeDim + 4 + numClasses + m]
                    }
                }
            }

            rawDetections.append((bbox: bbox, confidence: bestScore, classIndex: bestClassIdx, maskCoeffs: maskCoeffs))
        }

        // Apply Non-Maximum Suppression
        let nmsResults = applyNMS(detections: rawDetections, iouThreshold: iouThreshold)

        // Build segmentation masks if in segmentation mode
        // The mask protos are in a second output tensor; coefficients are multiplied by protos
        var protoData: [Float]?
        if includeMasks, featureNames.count > 1 {
            let sortedNames = featureNames.sorted()
            if let protoName = sortedNames.dropFirst().first,
               let protoValue = output.featureValue(for: protoName),
               let protoArray = protoValue.multiArrayValue {
                let count = protoArray.count
                protoData = Array(UnsafeBufferPointer(start: protoArray.dataPointer.assumingMemoryBound(to: Float.self), count: count))
            }
        }

        // Convert to DetectionResult
        let results: [DetectionResult] = nmsResults.enumerated().map { idx, det in
            let label = det.classIndex < classLabels.count ? classLabels[det.classIndex] : "object"
            let color = colorForIndex(det.classIndex)

            var maskPixels: [Float]?
            if includeMasks, let coeffs = det.maskCoeffs, let protos = protoData {
                maskPixels = generateMask(coefficients: coeffs, protos: protos, maskSize: 160)
            }

            return DetectionResult(
                label: label,
                confidence: det.confidence,
                boundingBox: det.bbox,
                maskData: maskPixels,
                color: color
            )
        }

        return results
    }

    // MARK: - Non-Maximum Suppression

    private func applyNMS(
        detections: [(bbox: CGRect, confidence: Float, classIndex: Int, maskCoeffs: [Float]?)],
        iouThreshold: Float
    ) -> [(bbox: CGRect, confidence: Float, classIndex: Int, maskCoeffs: [Float]?)] {
        let sorted = detections.sorted { $0.confidence > $1.confidence }
        var selected: [(bbox: CGRect, confidence: Float, classIndex: Int, maskCoeffs: [Float]?)] = []

        for det in sorted {
            var shouldSelect = true
            for sel in selected {
                if det.classIndex == sel.classIndex && computeIoU(det.bbox, sel.bbox) > iouThreshold {
                    shouldSelect = false
                    break
                }
            }
            if shouldSelect {
                selected.append(det)
            }
        }

        return selected
    }

    private func computeIoU(_ a: CGRect, _ b: CGRect) -> Float {
        let intersection = a.intersection(b)
        guard !intersection.isNull else { return 0 }
        let intersectionArea = intersection.width * intersection.height
        let unionArea = a.width * a.height + b.width * b.height - intersectionArea
        guard unionArea > 0 else { return 0 }
        return Float(intersectionArea / unionArea)
    }

    // MARK: - Generate Segmentation Mask

    private func generateMask(coefficients: [Float], protos: [Float], maskSize: Int) -> [Float] {
        // The mask is generated by: mask = sigmoid(coefficients . protos)
        // protos shape: [maskDim, maskSize, maskSize], coefficients shape: [maskDim]
        let totalPixels = maskSize * maskSize
        var mask = [Float](repeating: 0, count: totalPixels)

        let maskDim = coefficients.count
        for pixel in 0..<totalPixels {
            var sum: Float = 0
            for d in 0..<maskDim {
                let protoIdx = d * totalPixels + pixel
                if protoIdx < protos.count {
                    sum += coefficients[d] * protos[protoIdx]
                }
            }
            // Sigmoid activation
            mask[pixel] = 1.0 / (1.0 + exp(-sum))
        }

        return mask
    }

    // MARK: - Image Preprocessing

    private func resizeImage(_ image: UIImage, to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, true, 1.0)
        image.draw(in: CGRect(origin: .zero, size: size))
        let resized = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return resized
    }

    @MainActor
    private func updateStatus(_ message: String, progress: Double) {
        self.statusMessage = message
        self.progress = progress
    }
}

// MARK: - UIImage -> CVPixelBuffer

extension UIImage {
    func toPixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
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

        guard let cgImage = self.cgImage else { return nil }
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        return buffer
    }
}

// MARK: - Errors

enum YOLOEError: LocalizedError {
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
