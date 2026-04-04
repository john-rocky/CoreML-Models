import SwiftUI
import PhotosUI

struct ContentView: View {
    @StateObject private var classifier = SigLIPClassifier()
    @State private var selectedItem: PhotosPickerItem?
    @State private var selectedImage: UIImage?
    @State private var labelsText: String = "cat, dog, car, person, food"
    @State private var results: [ClassificationResult] = []
    @State private var isProcessing = false
    @State private var processingTime: Double?

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                // Status
                HStack {
                    Circle()
                        .fill(classifier.isReady ? .green : .red)
                        .frame(width: 8, height: 8)
                    Text(classifier.isReady ? "Model Ready" : "Loading...")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    if let time = processingTime {
                        Text(String(format: "%.2fs", time))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(.horizontal)

                // Image
                GeometryReader { geo in
                    if let image = selectedImage {
                        Image(uiImage: image)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(maxWidth: geo.size.width, maxHeight: geo.size.height)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                    } else {
                        RoundedRectangle(cornerRadius: 12)
                            .fill(Color(.systemGray6))
                            .overlay {
                                VStack(spacing: 8) {
                                    Image(systemName: "photo")
                                        .font(.largeTitle)
                                        .foregroundStyle(.tertiary)
                                    Text("Select an image")
                                        .font(.subheadline)
                                        .foregroundStyle(.tertiary)
                                }
                            }
                    }
                }
                .padding(.horizontal)

                // Results
                if !results.isEmpty {
                    VStack(spacing: 8) {
                        ForEach(results) { result in
                            HStack {
                                Text(result.label)
                                    .font(.body)
                                Spacer()
                                Text(String(format: "%.1f%%", result.score * 100))
                                    .font(.body.monospacedDigit())
                                    .foregroundStyle(result.score > 0.5 ? .primary : .secondary)
                            }
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .background(
                                GeometryReader { geo in
                                    Rectangle()
                                        .fill(Color.accentColor.opacity(0.15))
                                        .frame(width: geo.size.width * CGFloat(result.score))
                                }
                            )
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                        }
                    }
                    .padding(.horizontal)
                }

                // Labels input
                TextField("Labels (comma separated)", text: $labelsText)
                    .textFieldStyle(.roundedBorder)
                    .padding(.horizontal)

                // Controls
                HStack(spacing: 12) {
                    PhotosPicker(selection: $selectedItem, matching: .images) {
                        Label("Photo", systemImage: "photo.on.rectangle")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)

                    Button {
                        runClassification()
                    } label: {
                        if isProcessing {
                            ProgressView()
                                .frame(maxWidth: .infinity)
                        } else {
                            Label("Classify", systemImage: "sparkle.magnifyingglass")
                                .frame(maxWidth: .infinity)
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!classifier.isReady || selectedImage == nil || isProcessing)
                }
                .padding(.horizontal)
            }
            .padding(.vertical)
            .navigationTitle("SigLIP")
            .onChange(of: selectedItem) {
                Task {
                    if let data = try? await selectedItem?.loadTransferable(type: Data.self),
                       let image = UIImage(data: data) {
                        selectedImage = image
                        results = []
                        processingTime = nil
                    }
                }
            }
        }
    }

    private func runClassification() {
        guard let image = selectedImage else { return }
        let labels = labelsText.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        guard !labels.isEmpty else { return }

        isProcessing = true
        results = []
        processingTime = nil

        Task {
            let start = CFAbsoluteTimeGetCurrent()
            do {
                let r = try await classifier.classify(image: image, labels: labels)
                let elapsed = CFAbsoluteTimeGetCurrent() - start
                await MainActor.run {
                    results = r
                    processingTime = elapsed
                    isProcessing = false
                }
            } catch {
                await MainActor.run {
                    results = [ClassificationResult(label: "Error: \(error.localizedDescription)", score: 0)]
                    isProcessing = false
                }
            }
        }
    }
}
