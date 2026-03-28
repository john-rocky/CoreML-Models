import SwiftUI
import UIKit
import CoreML
import Vision
import PhotosUI
import UniformTypeIdentifiers

// MARK: - Drag-and-Drop Image Classifier
// Uses PoolFormer_S12 model (224x224 input, 1000-class ImageNet output)
// Output feature name: "var_646"

struct ContentView: View {
    @StateObject private var classifier = DropClassifier()

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                if let error = classifier.errorMessage {
                    HStack {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.yellow)
                        Text(error)
                            .font(.caption)
                    }
                    .padding()
                    .background(Color(.systemOrange).opacity(0.1))
                }

                if classifier.classifiedItems.isEmpty && classifier.droppedImage == nil {
                    // Drop zone
                    VStack(spacing: 0) {
                        Spacer()
                        DropZoneView(
                            isTargeted: $classifier.isDropTargeted,
                            classifier: classifier
                        )
                        .padding()

                        // Also allow photo picker as fallback
                        PhotosPicker(
                            selection: $classifier.selectedItem,
                            matching: .images
                        ) {
                            Label("Or pick from Photos", systemImage: "photo.on.rectangle")
                                .font(.subheadline)
                        }
                        .padding(.bottom, 8)

                        Text("On iPad, drag photos from Files or Safari onto the drop zone")
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)
                        Spacer()
                    }
                } else {
                    // Results view
                    ScrollView {
                        VStack(spacing: 16) {
                            // Current dropped image
                            if let image = classifier.droppedImage {
                                VStack(spacing: 12) {
                                    Image(uiImage: image)
                                        .resizable()
                                        .scaledToFit()
                                        .frame(maxHeight: 250)
                                        .cornerRadius(12)
                                        .shadow(radius: 4)

                                    if classifier.isProcessing {
                                        ProgressView("Classifying...")
                                    } else if !classifier.currentResults.isEmpty {
                                        ResultsCard(results: classifier.currentResults)
                                    }
                                }
                                .padding()
                            }

                            // Drop zone for adding more
                            DropZoneView(
                                isTargeted: $classifier.isDropTargeted,
                                classifier: classifier,
                                compact: true
                            )
                            .padding(.horizontal)

                            PhotosPicker(
                                selection: $classifier.selectedItem,
                                matching: .images
                            ) {
                                Label("Or pick from Photos", systemImage: "photo.on.rectangle")
                                    .font(.caption)
                            }

                            // History
                            if !classifier.classifiedItems.isEmpty {
                                VStack(alignment: .leading, spacing: 8) {
                                    HStack {
                                        Text("Classification History")
                                            .font(.headline)
                                        Spacer()
                                        Button("Clear") {
                                            classifier.clearHistory()
                                        }
                                        .font(.caption)
                                    }
                                    .padding(.horizontal)

                                    ForEach(classifier.classifiedItems) { item in
                                        HistoryRow(item: item)
                                            .padding(.horizontal)
                                    }
                                }
                            }
                        }
                        .padding(.vertical)
                    }
                }
            }
            .navigationTitle("PoolFormer")
            .navigationBarTitleDisplayMode(.large)
        }
    }
}

// MARK: - Drop Zone View
struct DropZoneView: View {
    @Binding var isTargeted: Bool
    let classifier: DropClassifier
    var compact: Bool = false

    var body: some View {
        let height: CGFloat = compact ? 100 : 250

        RoundedRectangle(cornerRadius: 16)
            .strokeBorder(
                isTargeted ? Color.accentColor : Color.secondary.opacity(0.3),
                style: StrokeStyle(lineWidth: 3, dash: [10])
            )
            .background(
                RoundedRectangle(cornerRadius: 16)
                    .fill(isTargeted ? Color.accentColor.opacity(0.1) : Color(.systemGray6))
            )
            .frame(height: height)
            .overlay {
                VStack(spacing: 8) {
                    Image(systemName: isTargeted ? "arrow.down.circle.fill" : "arrow.down.doc")
                        .font(compact ? .title3 : .largeTitle)
                        .foregroundColor(isTargeted ? .accentColor : .secondary)
                    if !compact {
                        Text("Drop an image here")
                            .font(.headline)
                            .foregroundColor(.secondary)
                        Text("Drag a photo onto this area to classify it")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    } else {
                        Text("Drop another image")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }
            .onDrop(of: [UTType.image], isTargeted: $isTargeted) { providers in
                classifier.handleDrop(providers: providers)
                return true
            }
    }
}

// MARK: - Results Card
struct ResultsCard: View {
    let results: [(label: String, score: Float)]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Classification Results")
                .font(.headline)

            ForEach(Array(results.enumerated()), id: \.offset) { index, result in
                HStack {
                    Text(result.label)
                        .font(.subheadline)
                    Spacer()
                    Text(String(format: "%.1f%%", result.score * 100))
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                ProgressView(value: result.score)
                    .tint(index == 0 ? .accentColor : .gray)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: .black.opacity(0.1), radius: 4)
    }
}

// MARK: - History Row
struct HistoryItem: Identifiable {
    let id = UUID()
    let image: UIImage
    let topLabel: String
    let confidence: Float
    let timestamp: Date
}

struct HistoryRow: View {
    let item: HistoryItem

    var body: some View {
        HStack(spacing: 12) {
            Image(uiImage: item.image)
                .resizable()
                .scaledToFill()
                .frame(width: 50, height: 50)
                .cornerRadius(8)
                .clipped()

            VStack(alignment: .leading) {
                Text(item.topLabel)
                    .font(.subheadline)
                    .fontWeight(.medium)
                Text(String(format: "%.1f%%", item.confidence * 100))
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Spacer()

            Text(item.timestamp, style: .time)
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .padding(8)
        .background(Color(.systemGray6))
        .cornerRadius(8)
    }
}

// MARK: - Classifier ViewModel
@MainActor
class DropClassifier: ObservableObject {
    @Published var droppedImage: UIImage?
    @Published var currentResults: [(label: String, score: Float)] = []
    @Published var classifiedItems: [HistoryItem] = []
    @Published var isProcessing = false
    @Published var isDropTargeted = false
    @Published var errorMessage: String?

    @Published var selectedItem: PhotosPickerItem? {
        didSet { Task { await loadFromPicker() } }
    }

    private var vnModel: VNCoreMLModel?

    init() {
        loadModel()
    }

    private func loadModel() {
        // PLACEHOLDER: Add PoolFormer_S12.mlpackage to the Xcode project.
        // The compiled model class will be generated automatically by Xcode.
        // Download from the converted_models directory and drag into the project navigator.
        do {
            guard let modelURL = Bundle.main.url(forResource: "PoolFormer_S12", withExtension: "mlmodelc") else {
                errorMessage = "Model not found. Add PoolFormer_S12.mlpackage to the project."
                return
            }
            let mlModel = try MLModel(contentsOf: modelURL)
            vnModel = try VNCoreMLModel(for: mlModel)
        } catch {
            errorMessage = "Failed to load model: \(error.localizedDescription)"
        }
    }

    private func loadFromPicker() async {
        guard let item = selectedItem,
              let data = try? await item.loadTransferable(type: Data.self),
              let uiImage = UIImage(data: data) else { return }
        await classifyImage(uiImage)
    }

    func handleDrop(providers: [NSItemProvider]) {
        guard let provider = providers.first else { return }

        if provider.canLoadObject(ofClass: UIImage.self) {
            provider.loadObject(ofClass: UIImage.self) { [weak self] image, error in
                guard let self = self, let uiImage = image as? UIImage else { return }
                Task { @MainActor in
                    await self.classifyImage(uiImage)
                }
            }
        } else {
            provider.loadDataRepresentation(forTypeIdentifier: UTType.image.identifier) { [weak self] data, error in
                guard let self = self, let data = data, let uiImage = UIImage(data: data) else { return }
                Task { @MainActor in
                    await self.classifyImage(uiImage)
                }
            }
        }
    }

    private func classifyImage(_ image: UIImage) async {
        // Save previous result to history
        if let prevImage = droppedImage, !currentResults.isEmpty {
            let historyItem = HistoryItem(
                image: prevImage,
                topLabel: currentResults.first?.label ?? "Unknown",
                confidence: currentResults.first?.score ?? 0,
                timestamp: Date()
            )
            classifiedItems.insert(historyItem, at: 0)
            if classifiedItems.count > 20 {
                classifiedItems = Array(classifiedItems.prefix(20))
            }
        }

        droppedImage = image
        currentResults = []
        isProcessing = true

        guard let vnModel = vnModel else {
            isProcessing = false
            return
        }

        guard let cgImage = image.cgImage else {
            isProcessing = false
            return
        }

        let request = VNCoreMLRequest(model: vnModel)
        request.imageCropAndScaleOption = .centerCrop

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])

        do {
            try handler.perform([request])

            if let results = request.results as? [VNCoreMLFeatureValueObservation],
               let multiArray = results.first?.featureValue.multiArrayValue {
                let count = multiArray.count
                var scores = [Float](repeating: 0, count: count)
                for i in 0..<count {
                    scores[i] = multiArray[i].floatValue
                }

                let softmaxScores = softmax(scores)
                let topResults = ImageNetLabels.topK(scores: softmaxScores, k: 5)
                currentResults = topResults.map { (label: $0.label, score: $0.score) }
            } else if let results = request.results as? [VNClassificationObservation] {
                currentResults = results.prefix(5).map { (label: $0.identifier, score: $0.confidence) }
            }
        } catch {
            errorMessage = "Classification failed: \(error.localizedDescription)"
        }

        isProcessing = false
    }

    func clearHistory() {
        classifiedItems = []
    }

    private func softmax(_ input: [Float]) -> [Float] {
        let maxVal = input.max() ?? 0
        let expValues = input.map { exp($0 - maxVal) }
        let sumExp = expValues.reduce(0, +)
        return expValues.map { $0 / sumExp }
    }
}

#Preview {
    ContentView()
}
