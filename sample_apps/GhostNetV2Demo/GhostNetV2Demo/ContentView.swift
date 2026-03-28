import SwiftUI
import UIKit
import CoreML
import Vision
import PhotosUI

// MARK: - Batch Photo Classifier
// Uses GhostNetV2_100 model (224x224 input, 1000-class ImageNet output)
// Output feature name: "var_2336"

struct ClassifiedImage: Identifiable {
    let id = UUID()
    let image: UIImage
    var topLabel: String = "Processing..."
    var confidence: Float = 0
    var topResults: [(label: String, score: Float)] = []
    var isProcessing: Bool = true
}

struct ContentView: View {
    @StateObject private var classifier = BatchClassifier()
    @State private var showingDetail: ClassifiedImage?

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

                if classifier.images.isEmpty {
                    // Empty state
                    Spacer()
                    VStack(spacing: 16) {
                        Image(systemName: "photo.stack")
                            .font(.system(size: 60))
                            .foregroundColor(.secondary)
                        Text("Select multiple photos to classify them all at once")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)

                        PhotosPicker(
                            selection: $classifier.selectedItems,
                            maxSelectionCount: 20,
                            matching: .images
                        ) {
                            Label("Select Photos", systemImage: "photo.on.rectangle.angled")
                                .font(.headline)
                                .padding()
                                .frame(maxWidth: 280)
                                .background(Color.accentColor)
                                .foregroundColor(.white)
                                .cornerRadius(12)
                        }
                    }
                    .padding()
                    Spacer()
                } else {
                    // Results grid
                    ScrollView {
                        // Summary bar
                        HStack {
                            Text("\(classifier.images.count) images")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                            Spacer()
                            let done = classifier.images.filter { !$0.isProcessing }.count
                            if done < classifier.images.count {
                                ProgressView()
                                    .scaleEffect(0.8)
                                Text("\(done)/\(classifier.images.count)")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            } else {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundColor(.green)
                                Text("All classified")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                        .padding(.horizontal)
                        .padding(.top, 8)

                        LazyVGrid(columns: [
                            GridItem(.flexible(), spacing: 8),
                            GridItem(.flexible(), spacing: 8),
                            GridItem(.flexible(), spacing: 8)
                        ], spacing: 8) {
                            ForEach(classifier.images) { item in
                                ClassifiedImageCell(item: item)
                                    .onTapGesture {
                                        if !item.isProcessing {
                                            showingDetail = item
                                        }
                                    }
                            }
                        }
                        .padding(.horizontal, 8)
                        .padding(.bottom, 16)
                    }
                }
            }
            .navigationTitle("GhostNetV2 Batch")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                if !classifier.images.isEmpty {
                    ToolbarItem(placement: .navigationBarLeading) {
                        Button("Clear") {
                            classifier.clearAll()
                        }
                    }
                    ToolbarItem(placement: .navigationBarTrailing) {
                        PhotosPicker(
                            selection: $classifier.selectedItems,
                            maxSelectionCount: 20,
                            matching: .images
                        ) {
                            Image(systemName: "plus.circle")
                        }
                    }
                }
            }
            .sheet(item: $showingDetail) { item in
                DetailSheet(item: item)
            }
        }
    }
}

// MARK: - Grid Cell
struct ClassifiedImageCell: View {
    let item: ClassifiedImage

    var body: some View {
        VStack(spacing: 4) {
            Image(uiImage: item.image)
                .resizable()
                .scaledToFill()
                .frame(height: 100)
                .clipped()
                .cornerRadius(8)

            if item.isProcessing {
                ProgressView()
                    .scaleEffect(0.6)
                    .frame(height: 30)
            } else {
                Text(item.topLabel)
                    .font(.caption2)
                    .fontWeight(.medium)
                    .lineLimit(1)
                Text(String(format: "%.1f%%", item.confidence * 100))
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
    }
}

// MARK: - Detail Sheet
struct DetailSheet: View {
    let item: ClassifiedImage
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    Image(uiImage: item.image)
                        .resizable()
                        .scaledToFit()
                        .frame(maxHeight: 300)
                        .cornerRadius(12)

                    VStack(alignment: .leading, spacing: 8) {
                        Text("Top Predictions")
                            .font(.headline)

                        ForEach(Array(item.topResults.enumerated()), id: \.offset) { index, result in
                            HStack {
                                Text("\(index + 1).")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                    .frame(width: 20)
                                Text(result.label)
                                    .font(.subheadline)
                                Spacer()
                                Text(String(format: "%.2f%%", result.score * 100))
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                            }
                            ProgressView(value: result.score)
                                .tint(.accentColor)
                        }
                    }
                    .padding()
                }
                .padding()
            }
            .navigationTitle("Classification Detail")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

// MARK: - Batch Classifier ViewModel
@MainActor
class BatchClassifier: ObservableObject {
    @Published var images: [ClassifiedImage] = []
    @Published var errorMessage: String?

    @Published var selectedItems: [PhotosPickerItem] = [] {
        didSet { Task { await loadImages() } }
    }

    private var vnModel: VNCoreMLModel?

    init() {
        loadModel()
    }

    private func loadModel() {
        // PLACEHOLDER: Add GhostNetV2_100.mlpackage to the Xcode project.
        // The compiled model class will be generated automatically by Xcode.
        // Download from the converted_models directory and drag into the project navigator.
        do {
            guard let modelURL = Bundle.main.url(forResource: "GhostNetV2_100", withExtension: "mlmodelc") else {
                errorMessage = "Model not found. Add GhostNetV2_100.mlpackage to the project."
                return
            }
            let mlModel = try MLModel(contentsOf: modelURL)
            vnModel = try VNCoreMLModel(for: mlModel)
        } catch {
            errorMessage = "Failed to load model: \(error.localizedDescription)"
        }
    }

    private func loadImages() async {
        var newImages: [ClassifiedImage] = []

        for item in selectedItems {
            if let data = try? await item.loadTransferable(type: Data.self),
               let uiImage = UIImage(data: data) {
                newImages.append(ClassifiedImage(image: uiImage))
            }
        }

        images = newImages

        // Classify all images concurrently
        for index in images.indices {
            Task {
                await classifyImage(at: index)
            }
        }
    }

    private func classifyImage(at index: Int) async {
        guard index < images.count else { return }
        guard let vnModel = vnModel else {
            if index < images.count {
                images[index].isProcessing = false
                images[index].topLabel = "No model"
            }
            return
        }

        let image = images[index].image
        guard let cgImage = image.cgImage else {
            images[index].isProcessing = false
            images[index].topLabel = "Invalid image"
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

                guard index < images.count else { return }
                images[index].topResults = topResults.map { (label: $0.label, score: $0.score) }
                images[index].topLabel = topResults.first?.label ?? "Unknown"
                images[index].confidence = topResults.first?.score ?? 0
                images[index].isProcessing = false
            } else if let results = request.results as? [VNClassificationObservation] {
                let topResults = results.prefix(5).map { (label: $0.identifier, score: $0.confidence) }
                guard index < images.count else { return }
                images[index].topResults = topResults
                images[index].topLabel = topResults.first?.label ?? "Unknown"
                images[index].confidence = topResults.first?.score ?? 0
                images[index].isProcessing = false
            }
        } catch {
            guard index < images.count else { return }
            images[index].isProcessing = false
            images[index].topLabel = "Error"
        }
    }

    func clearAll() {
        images = []
        selectedItems = []
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
