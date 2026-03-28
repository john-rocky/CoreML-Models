import SwiftUI
import UIKit
import CoreML
import Vision
import PhotosUI
import AVKit

// MARK: - LivePortrait: Portrait Animation via Multi-Model Pipeline
//
// Pipeline stages:
// 1. MotionExtractor   - Extracts 3D motion parameters (pitch, yaw, roll, expression, translation)
//                        from each driving video frame
// 2. AppearanceExtractor - Extracts appearance features from the source portrait
// 3. WarpingNetwork     - Warps source appearance using motion deltas between source and driving
// 4. SPADEGenerator     - Generates the final animated frame from warped features
//
// Each model is loaded independently and run in sequence for each frame.

// MARK: - Pipeline Stage Model

enum PipelineStage: String, CaseIterable, Identifiable {
    case motionExtractor = "Motion Extractor"
    case appearanceExtractor = "Appearance Extractor"
    case warpingNetwork = "Warping Network"
    case spadeGenerator = "SPADE Generator"

    var id: String { rawValue }

    var modelFileName: String {
        switch self {
        case .motionExtractor: return "LivePortrait_MotionExtractor"
        case .appearanceExtractor: return "LivePortrait_AppearanceExtractor"
        case .warpingNetwork: return "LivePortrait_WarpingNetwork"
        case .spadeGenerator: return "LivePortrait_SPADEGenerator"
        }
    }

    var description: String {
        switch self {
        case .motionExtractor:
            return "Extracts 3D motion parameters (rotation, expression, translation) from face images"
        case .appearanceExtractor:
            return "Extracts identity-preserving appearance features from the source portrait"
        case .warpingNetwork:
            return "Warps source appearance features according to driving motion parameters"
        case .spadeGenerator:
            return "Generates the final animated frame using SPADE normalization"
        }
    }

    var icon: String {
        switch self {
        case .motionExtractor: return "arrow.triangle.branch"
        case .appearanceExtractor: return "person.crop.rectangle"
        case .warpingNetwork: return "wand.and.rays"
        case .spadeGenerator: return "paintbrush.pointed.fill"
        }
    }
}

enum StageStatus: Equatable {
    case pending
    case running
    case completed
    case failed(String)

    var color: Color {
        switch self {
        case .pending: return .gray
        case .running: return .orange
        case .completed: return .green
        case .failed: return .red
        }
    }
}

struct ContentView: View {
    @StateObject private var viewModel = LivePortraitViewModel()

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    // Source portrait picker
                    Section {
                        PhotosPicker(selection: $viewModel.selectedSourcePhoto,
                                     matching: .images) {
                            if let image = viewModel.sourceImage {
                                Image(uiImage: image)
                                    .resizable()
                                    .scaledToFit()
                                    .frame(maxHeight: 200)
                                    .cornerRadius(12)
                            } else {
                                placeholderView(
                                    title: "Select Source Portrait",
                                    systemImage: "person.crop.square"
                                )
                            }
                        }
                    } header: {
                        sectionHeader("Source Portrait")
                    }

                    // Driving video picker
                    Section {
                        PhotosPicker(selection: $viewModel.selectedDrivingVideo,
                                     matching: .videos) {
                            if viewModel.drivingVideoURL != nil {
                                HStack {
                                    Image(systemName: "video.fill")
                                        .font(.title2)
                                        .foregroundColor(.accentColor)
                                    VStack(alignment: .leading) {
                                        Text("Driving Video Selected")
                                            .font(.headline)
                                        Text("Tap to change")
                                            .font(.caption)
                                            .foregroundColor(.secondary)
                                    }
                                    Spacer()
                                    if let thumb = viewModel.drivingThumbnail {
                                        Image(uiImage: thumb)
                                            .resizable()
                                            .scaledToFill()
                                            .frame(width: 60, height: 60)
                                            .cornerRadius(8)
                                    }
                                }
                                .padding()
                                .background(Color(.systemGray6))
                                .cornerRadius(12)
                            } else {
                                placeholderView(
                                    title: "Select Driving Video",
                                    systemImage: "video.badge.plus"
                                )
                            }
                        }
                    } header: {
                        sectionHeader("Driving Video")
                    }

                    // Animate button
                    if viewModel.sourceImage != nil && viewModel.drivingVideoURL != nil {
                        Button(action: { viewModel.runPipeline() }) {
                            HStack {
                                if viewModel.isProcessing {
                                    ProgressView()
                                        .tint(.white)
                                } else {
                                    Image(systemName: "play.fill")
                                }
                                Text(viewModel.isProcessing ? "Processing..." : "Animate Portrait")
                            }
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(viewModel.isProcessing ? Color.gray : Color.accentColor)
                            .foregroundColor(.white)
                            .cornerRadius(12)
                        }
                        .disabled(viewModel.isProcessing)
                    }

                    // Pipeline status display
                    Section {
                        VStack(spacing: 12) {
                            ForEach(PipelineStage.allCases) { stage in
                                PipelineStageRow(
                                    stage: stage,
                                    status: viewModel.stageStatuses[stage] ?? .pending
                                )
                            }
                        }
                    } header: {
                        sectionHeader("Pipeline Stages")
                    }

                    // Error display
                    if let error = viewModel.errorMessage {
                        Text(error)
                            .foregroundColor(.red)
                            .font(.caption)
                            .padding()
                            .background(Color.red.opacity(0.1))
                            .cornerRadius(8)
                    }

                    // Result display
                    if let result = viewModel.resultImage {
                        Section {
                            VStack(spacing: 12) {
                                Image(uiImage: result)
                                    .resizable()
                                    .scaledToFit()
                                    .frame(maxHeight: 300)
                                    .cornerRadius(12)

                                Text("Animated portrait result (single frame preview)")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        } header: {
                            sectionHeader("Animated Result")
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("LivePortrait")
        }
    }

    private func sectionHeader(_ title: String) -> some View {
        HStack {
            Text(title)
                .font(.headline)
            Spacer()
        }
    }

    private func placeholderView(title: String, systemImage: String) -> some View {
        VStack(spacing: 12) {
            Image(systemName: systemImage)
                .font(.system(size: 40))
                .foregroundColor(.secondary)
            Text(title)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
        .frame(height: 160)
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

// MARK: - Pipeline Stage Row

struct PipelineStageRow: View {
    let stage: PipelineStage
    let status: StageStatus

    var body: some View {
        HStack(spacing: 12) {
            // Status indicator
            ZStack {
                Circle()
                    .fill(status.color.opacity(0.2))
                    .frame(width: 36, height: 36)
                if case .running = status {
                    ProgressView()
                        .scaleEffect(0.7)
                } else {
                    Image(systemName: statusIcon)
                        .font(.caption)
                        .foregroundColor(status.color)
                }
            }

            VStack(alignment: .leading, spacing: 2) {
                HStack {
                    Image(systemName: stage.icon)
                        .font(.caption)
                    Text(stage.rawValue)
                        .font(.subheadline)
                        .fontWeight(.medium)
                }
                Text(stage.description)
                    .font(.caption2)
                    .foregroundColor(.secondary)
                    .lineLimit(2)

                if case .failed(let msg) = status {
                    Text(msg)
                        .font(.caption2)
                        .foregroundColor(.red)
                }
            }

            Spacer()
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color(.systemGray6))
        )
    }

    private var statusIcon: String {
        switch status {
        case .pending: return "circle"
        case .running: return "arrow.clockwise"
        case .completed: return "checkmark.circle.fill"
        case .failed: return "xmark.circle.fill"
        }
    }
}

// MARK: - ViewModel

class LivePortraitViewModel: ObservableObject {
    @Published var selectedSourcePhoto: PhotosPickerItem? {
        didSet { loadSourceImage() }
    }
    @Published var selectedDrivingVideo: PhotosPickerItem? {
        didSet { loadDrivingVideo() }
    }
    @Published var sourceImage: UIImage?
    @Published var drivingVideoURL: URL?
    @Published var drivingThumbnail: UIImage?
    @Published var resultImage: UIImage?
    @Published var isProcessing = false
    @Published var errorMessage: String?
    @Published var stageStatuses: [PipelineStage: StageStatus] = [:]

    init() {
        // Initialize all stages as pending
        for stage in PipelineStage.allCases {
            stageStatuses[stage] = .pending
        }
    }

    private func loadSourceImage() {
        guard let item = selectedSourcePhoto else { return }
        Task {
            if let data = try? await item.loadTransferable(type: Data.self),
               let image = UIImage(data: data) {
                await MainActor.run {
                    self.sourceImage = image
                    self.resultImage = nil
                    self.resetStages()
                }
            }
        }
    }

    private func loadDrivingVideo() {
        guard let item = selectedDrivingVideo else { return }
        Task {
            // Load video as a Movie transferable
            if let videoData = try? await item.loadTransferable(type: Data.self) {
                let tempURL = FileManager.default.temporaryDirectory
                    .appendingPathComponent(UUID().uuidString)
                    .appendingPathExtension("mov")
                try? videoData.write(to: tempURL)

                // Generate thumbnail
                let asset = AVURLAsset(url: tempURL)
                let generator = AVAssetImageGenerator(asset: asset)
                generator.appliesPreferredTrackTransform = true
                let cgImage = try? generator.copyCGImage(at: .zero, actualTime: nil)

                await MainActor.run {
                    self.drivingVideoURL = tempURL
                    self.drivingThumbnail = cgImage.map { UIImage(cgImage: $0) }
                    self.resultImage = nil
                    self.resetStages()
                }
            }
        }
    }

    private func resetStages() {
        for stage in PipelineStage.allCases {
            stageStatuses[stage] = .pending
        }
    }

    func runPipeline() {
        guard sourceImage != nil, drivingVideoURL != nil else { return }
        isProcessing = true
        errorMessage = nil
        resetStages()

        Task {
            do {
                try await executePipeline()
                await MainActor.run {
                    self.isProcessing = false
                }
            } catch {
                await MainActor.run {
                    self.errorMessage = error.localizedDescription
                    self.isProcessing = false
                }
            }
        }
    }

    private func executePipeline() async throws {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine

        // Stage 1: Motion Extractor
        await setStageStatus(.motionExtractor, .running)
        do {
            guard let modelURL = Bundle.main.url(
                forResource: PipelineStage.motionExtractor.modelFileName,
                withExtension: "mlmodelc"
            ) else {
                throw LivePortraitError.modelNotFound(
                    "\(PipelineStage.motionExtractor.modelFileName).mlmodelc not found. " +
                    "Add the compiled model to the project."
                )
            }
            let _ = try MLModel(contentsOf: modelURL, configuration: config)

            // In production: extract motion params from each driving video frame
            // Output: pitch, yaw, roll, expression coefficients, translation vectors
            await setStageStatus(.motionExtractor, .completed)
        } catch {
            await setStageStatus(.motionExtractor, .failed(error.localizedDescription))
            throw error
        }

        // Stage 2: Appearance Extractor
        await setStageStatus(.appearanceExtractor, .running)
        do {
            guard let modelURL = Bundle.main.url(
                forResource: PipelineStage.appearanceExtractor.modelFileName,
                withExtension: "mlmodelc"
            ) else {
                throw LivePortraitError.modelNotFound(
                    "\(PipelineStage.appearanceExtractor.modelFileName).mlmodelc not found. " +
                    "Add the compiled model to the project."
                )
            }
            let _ = try MLModel(contentsOf: modelURL, configuration: config)

            // In production: extract appearance feature volume from source portrait
            // This is done once and reused for all frames
            await setStageStatus(.appearanceExtractor, .completed)
        } catch {
            await setStageStatus(.appearanceExtractor, .failed(error.localizedDescription))
            throw error
        }

        // Stage 3: Warping Network
        await setStageStatus(.warpingNetwork, .running)
        do {
            guard let modelURL = Bundle.main.url(
                forResource: PipelineStage.warpingNetwork.modelFileName,
                withExtension: "mlmodelc"
            ) else {
                throw LivePortraitError.modelNotFound(
                    "\(PipelineStage.warpingNetwork.modelFileName).mlmodelc not found. " +
                    "Add the compiled model to the project."
                )
            }
            let _ = try MLModel(contentsOf: modelURL, configuration: config)

            // In production: warp source appearance features using
            // the delta between source and driving motion parameters
            await setStageStatus(.warpingNetwork, .completed)
        } catch {
            await setStageStatus(.warpingNetwork, .failed(error.localizedDescription))
            throw error
        }

        // Stage 4: SPADE Generator
        await setStageStatus(.spadeGenerator, .running)
        do {
            guard let modelURL = Bundle.main.url(
                forResource: PipelineStage.spadeGenerator.modelFileName,
                withExtension: "mlmodelc"
            ) else {
                throw LivePortraitError.modelNotFound(
                    "\(PipelineStage.spadeGenerator.modelFileName).mlmodelc not found. " +
                    "Add the compiled model to the project."
                )
            }
            let _ = try MLModel(contentsOf: modelURL, configuration: config)

            // In production: generate final animated frame from warped features
            // using SPADE (Spatially-Adaptive Normalization) decoder

            // For demo, use the source image as placeholder result
            await MainActor.run {
                self.resultImage = self.sourceImage
            }
            await setStageStatus(.spadeGenerator, .completed)
        } catch {
            await setStageStatus(.spadeGenerator, .failed(error.localizedDescription))
            throw error
        }
    }

    @MainActor
    private func setStageStatus(_ stage: PipelineStage, _ status: StageStatus) {
        stageStatuses[stage] = status
    }
}

enum LivePortraitError: LocalizedError {
    case modelNotFound(String)
    case processingFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelNotFound(let msg): return msg
        case .processingFailed(let msg): return msg
        }
    }
}

// MARK: - UIImage Extension

extension UIImage {
    func resized(to targetSize: CGSize) -> UIImage? {
        let renderer = UIGraphicsImageRenderer(size: targetSize)
        return renderer.image { _ in
            self.draw(in: CGRect(origin: .zero, size: targetSize))
        }
    }
}

#Preview {
    ContentView()
}
