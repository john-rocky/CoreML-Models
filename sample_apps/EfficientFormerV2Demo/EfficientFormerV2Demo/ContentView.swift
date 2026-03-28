import SwiftUI
import UIKit
import CoreML
import Vision
import PhotosUI

// MARK: - Side-by-Side Comparison Classifier
// Uses EfficientFormerV2_S0 model (224x224 input, 1000-class ImageNet output)
// Output feature name: "var_1617"

struct ContentView: View {
    @StateObject private var classifier = SideBySideClassifier()

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    // Header
                    Text("Pick two photos and compare classification results side by side.")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)

                    if let error = classifier.errorMessage {
                        ErrorBanner(message: error)
                    }

                    // Side-by-side panels
                    HStack(spacing: 12) {
                        ImagePanel(
                            title: "Image A",
                            image: classifier.imageA,
                            results: classifier.resultsA,
                            isProcessing: classifier.isProcessingA,
                            selectedItem: $classifier.photoItemA
                        )

                        ImagePanel(
                            title: "Image B",
                            image: classifier.imageB,
                            results: classifier.resultsB,
                            isProcessing: classifier.isProcessingB,
                            selectedItem: $classifier.photoItemB
                        )
                    }
                    .padding(.horizontal)

                    // Clear button
                    if classifier.imageA != nil || classifier.imageB != nil {
                        Button(role: .destructive) {
                            classifier.clearAll()
                        } label: {
                            Label("Clear All", systemImage: "trash")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                        .padding(.horizontal)
                    }
                }
                .padding(.vertical)
            }
            .navigationTitle("EfficientFormerV2")
            .navigationBarTitleDisplayMode(.large)
        }
    }
}

// MARK: - Image Panel View
struct ImagePanel: View {
    let title: String
    let image: UIImage?
    let results: [(label: String, score: Float)]
    let isProcessing: Bool
    @Binding var selectedItem: PhotosPickerItem?

    var body: some View {
        VStack(spacing: 8) {
            Text(title)
                .font(.headline)

            // Photo picker area
            PhotosPicker(selection: $selectedItem, matching: .images) {
                Group {
                    if let image = image {
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFill()
                            .frame(height: 160)
                            .clipped()
                            .cornerRadius(10)
                    } else {
                        RoundedRectangle(cornerRadius: 10)
                            .fill(Color(.systemGray6))
                            .frame(height: 160)
                            .overlay {
                                VStack(spacing: 6) {
                                    Image(systemName: "photo.badge.plus")
                                        .font(.title2)
                                    Text("Select Photo")
                                        .font(.caption)
                                }
                                .foregroundColor(.secondary)
                            }
                    }
                }
            }

            // Results
            if isProcessing {
                ProgressView("Classifying...")
                    .font(.caption)
            } else if !results.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(Array(results.prefix(5).enumerated()), id: \.offset) { _, result in
                        HStack {
                            Text(result.label)
                                .font(.caption2)
                                .lineLimit(1)
                            Spacer()
                            Text(String(format: "%.1f%%", result.score * 100))
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }
                        // Confidence bar
                        GeometryReader { geo in
                            RoundedRectangle(cornerRadius: 2)
                                .fill(Color.accentColor.opacity(0.3))
                                .frame(width: geo.size.width * CGFloat(result.score))
                        }
                        .frame(height: 3)
                    }
                }
            }
        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - Error Banner
struct ErrorBanner: View {
    let message: String

    var body: some View {
        HStack {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundColor(.yellow)
            Text(message)
                .font(.caption)
        }
        .padding()
        .background(Color(.systemOrange).opacity(0.1))
        .cornerRadius(8)
        .padding(.horizontal)
    }
}

// MARK: - Classifier ViewModel
@MainActor
class SideBySideClassifier: ObservableObject {
    @Published var imageA: UIImage?
    @Published var imageB: UIImage?
    @Published var resultsA: [(label: String, score: Float)] = []
    @Published var resultsB: [(label: String, score: Float)] = []
    @Published var isProcessingA = false
    @Published var isProcessingB = false
    @Published var errorMessage: String?

    @Published var photoItemA: PhotosPickerItem? {
        didSet { Task { await loadImage(from: photoItemA, side: .a) } }
    }
    @Published var photoItemB: PhotosPickerItem? {
        didSet { Task { await loadImage(from: photoItemB, side: .b) } }
    }

    private var vnModel: VNCoreMLModel?

    enum Side { case a, b }

    init() {
        loadModel()
    }

    private func loadModel() {
        // PLACEHOLDER: Add EfficientFormerV2_S0.mlpackage to the Xcode project.
        // The compiled model class will be generated automatically by Xcode.
        // Download from the converted_models directory and drag into the project navigator.
        do {
            guard let modelURL = Bundle.main.url(forResource: "EfficientFormerV2_S0", withExtension: "mlmodelc") else {
                errorMessage = "Model not found. Add EfficientFormerV2_S0.mlpackage to the project."
                return
            }
            let mlModel = try MLModel(contentsOf: modelURL)
            vnModel = try VNCoreMLModel(for: mlModel)
        } catch {
            errorMessage = "Failed to load model: \(error.localizedDescription)"
        }
    }

    private func loadImage(from item: PhotosPickerItem?, side: Side) async {
        guard let item = item,
              let data = try? await item.loadTransferable(type: Data.self),
              let uiImage = UIImage(data: data) else { return }

        switch side {
        case .a:
            imageA = uiImage
            resultsA = []
            isProcessingA = true
        case .b:
            imageB = uiImage
            resultsB = []
            isProcessingB = true
        }

        await classify(image: uiImage, side: side)
    }

    private func classify(image: UIImage, side: Side) async {
        guard let vnModel = vnModel else {
            switch side {
            case .a: isProcessingA = false
            case .b: isProcessingB = false
            }
            return
        }

        guard let cgImage = image.cgImage else {
            switch side {
            case .a: isProcessingA = false
            case .b: isProcessingB = false
            }
            return
        }

        let request = VNCoreMLRequest(model: vnModel)
        request.imageCropAndScaleOption = .centerCrop

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])

        do {
            try handler.perform([request])

            if let results = request.results as? [VNCoreMLFeatureValueObservation],
               let multiArray = results.first?.featureValue.multiArrayValue {
                // Extract scores from the "var_1617" output
                let count = multiArray.count
                var scores = [Float](repeating: 0, count: count)
                for i in 0..<count {
                    scores[i] = multiArray[i].floatValue
                }

                // Apply softmax
                let softmaxScores = softmax(scores)
                let topResults = ImageNetLabels.topK(scores: softmaxScores, k: 5)
                    .map { (label: $0.label, score: $0.score) }

                switch side {
                case .a:
                    resultsA = topResults
                    isProcessingA = false
                case .b:
                    resultsB = topResults
                    isProcessingB = false
                }
            } else if let results = request.results as? [VNClassificationObservation] {
                let topResults = results.prefix(5).map { (label: $0.identifier, score: $0.confidence) }
                switch side {
                case .a:
                    resultsA = topResults
                    isProcessingA = false
                case .b:
                    resultsB = topResults
                    isProcessingB = false
                }
            }
        } catch {
            errorMessage = "Classification failed: \(error.localizedDescription)"
            switch side {
            case .a: isProcessingA = false
            case .b: isProcessingB = false
            }
        }
    }

    func clearAll() {
        imageA = nil
        imageB = nil
        resultsA = []
        resultsB = []
        photoItemA = nil
        photoItemB = nil
    }

    /// Softmax function to convert logits to probabilities
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
