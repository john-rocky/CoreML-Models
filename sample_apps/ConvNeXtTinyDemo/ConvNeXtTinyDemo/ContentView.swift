import SwiftUI
import UIKit
import PhotosUI
import CoreML
import Vision

// MARK: - Classifier

class ConvNeXtClassifier: ObservableObject {
    @Published var predictions: [(label: String, confidence: Float)] = []
    @Published var errorMessage: String?
    @Published var isProcessing = false

    private var vnModel: VNCoreMLModel?

    init() {
        loadModel()
    }

    private func loadModel() {
        // PLACEHOLDER: Add ConvNeXtTiny.mlpackage to the Xcode project.
        // The compiled .mlmodelc will be bundled automatically.
        // Download from the CoreML-Models repository and drag into Xcode.

        guard let modelURL = Bundle.main.url(forResource: "ConvNeXtTiny", withExtension: "mlmodelc") else {
            DispatchQueue.main.async {
                self.errorMessage = "Model not found. Please add ConvNeXtTiny.mlpackage to the Xcode project."
            }
            return
        }

        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            let mlModel = try MLModel(contentsOf: modelURL, configuration: config)
            vnModel = try VNCoreMLModel(for: mlModel)
        } catch {
            DispatchQueue.main.async {
                self.errorMessage = "Failed to load model: \(error.localizedDescription)"
            }
        }
    }

    func classify(image: UIImage) {
        guard let vnModel = vnModel else { return }
        guard let cgImage = image.cgImage else { return }

        DispatchQueue.main.async { self.isProcessing = true }

        let request = VNCoreMLRequest(model: vnModel) { [weak self] request, error in
            if let results = request.results as? [VNCoreMLFeatureValueObservation],
               let multiArray = results.first?.featureValue.multiArrayValue {
                self?.processResults(multiArray: multiArray)
            } else if let results = request.results as? [VNClassificationObservation] {
                let top5 = results.prefix(5).map { (label: $0.identifier, confidence: $0.confidence) }
                DispatchQueue.main.async {
                    self?.predictions = top5
                    self?.isProcessing = false
                }
            }
        }
        request.imageCropAndScaleOption = .centerCrop

        DispatchQueue.global(qos: .userInitiated).async {
            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            try? handler.perform([request])
        }
    }

    private func processResults(multiArray: MLMultiArray) {
        let count = multiArray.count
        var scores = [Float](repeating: 0, count: count)
        for i in 0..<count {
            scores[i] = multiArray[i].floatValue
        }

        // Apply softmax
        let maxScore = scores.max() ?? 0
        let expScores = scores.map { exp($0 - maxScore) }
        let sumExp = expScores.reduce(0, +)
        let probs = expScores.map { $0 / sumExp }

        let top5 = ImageNetLabels.topK(scores: probs, k: 5)
        DispatchQueue.main.async {
            self.predictions = top5.map { (label: $0.label, confidence: $0.score) }
            self.isProcessing = false
        }
    }
}

// MARK: - Content View

struct ContentView: View {
    @StateObject private var classifier = ConvNeXtClassifier()
    @State private var selectedItem: PhotosPickerItem?
    @State private var selectedImage: UIImage?

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Image display area
                ZStack {
                    Color(.systemGroupedBackground)

                    if let image = selectedImage {
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFit()
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                            .padding()
                    } else {
                        VStack(spacing: 16) {
                            Image(systemName: "photo.on.rectangle.angled")
                                .font(.system(size: 60))
                                .foregroundColor(.secondary)
                            Text("Select a photo to classify")
                                .font(.headline)
                                .foregroundColor(.secondary)
                        }
                    }

                    if classifier.isProcessing {
                        ProgressView("Classifying...")
                            .padding()
                            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
                    }
                }
                .frame(maxHeight: .infinity)

                // Error message
                if let error = classifier.errorMessage {
                    HStack {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.yellow)
                        Text(error)
                            .font(.caption)
                    }
                    .padding()
                    .background(Color.red.opacity(0.1))
                    .cornerRadius(8)
                    .padding(.horizontal)
                }

                // Results area
                if !classifier.predictions.isEmpty {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Top-5 Predictions")
                            .font(.headline)
                            .padding(.horizontal)
                            .padding(.top, 12)

                        ForEach(Array(classifier.predictions.enumerated()), id: \.offset) { index, prediction in
                            VStack(alignment: .leading, spacing: 4) {
                                HStack {
                                    Text("\(index + 1). \(prediction.label)")
                                        .font(.system(.body, design: .rounded))
                                        .fontWeight(index == 0 ? .bold : .regular)
                                    Spacer()
                                    Text(String(format: "%.1f%%", prediction.confidence * 100))
                                        .font(.system(.body, design: .monospaced))
                                        .foregroundColor(.secondary)
                                }

                                // Confidence bar
                                GeometryReader { geometry in
                                    ZStack(alignment: .leading) {
                                        RoundedRectangle(cornerRadius: 4)
                                            .fill(Color.secondary.opacity(0.2))
                                            .frame(height: 8)

                                        RoundedRectangle(cornerRadius: 4)
                                            .fill(barColor(for: index))
                                            .frame(width: geometry.size.width * CGFloat(prediction.confidence), height: 8)
                                    }
                                }
                                .frame(height: 8)
                            }
                            .padding(.horizontal)
                        }
                        .padding(.bottom, 8)
                    }
                    .background(Color(.systemBackground))
                }

                // Photo picker button
                PhotosPicker(selection: $selectedItem, matching: .images) {
                    Label("Choose Photo", systemImage: "photo.fill")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.accentColor)
                        .foregroundColor(.white)
                        .cornerRadius(12)
                }
                .padding()
                .onChange(of: selectedItem) { newItem in
                    Task {
                        if let data = try? await newItem?.loadTransferable(type: Data.self),
                           let uiImage = UIImage(data: data) {
                            selectedImage = uiImage
                            classifier.classify(image: uiImage)
                        }
                    }
                }
            }
            .navigationTitle("ConvNeXt Tiny")
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    private func barColor(for index: Int) -> Color {
        switch index {
        case 0: return .blue
        case 1: return .green
        case 2: return .orange
        case 3: return .purple
        case 4: return .pink
        default: return .gray
        }
    }
}

#Preview {
    ContentView()
}
