import SwiftUI
import UIKit
import CoreML
import Vision
import PhotosUI

// MARK: - Image Harmonization using CDTNet
// CDTNet takes a composite image and a mask indicating the foreground region,
// then produces a harmonized image where the foreground blends naturally with the background.

struct ContentView: View {
    @StateObject private var viewModel = HarmonizationViewModel()

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    // Composite image picker
                    Section {
                        PhotosPicker(selection: $viewModel.selectedPhoto,
                                     matching: .images) {
                            if let image = viewModel.compositeImage {
                                Image(uiImage: image)
                                    .resizable()
                                    .scaledToFit()
                                    .frame(maxHeight: 250)
                                    .cornerRadius(12)
                            } else {
                                placeholderView(title: "Select Composite Image",
                                                systemImage: "photo.on.rectangle")
                            }
                        }
                    } header: {
                        sectionHeader("Composite Image")
                    }

                    // Mask region selector
                    if viewModel.compositeImage != nil {
                        Section {
                            VStack(spacing: 10) {
                                Text("Drag to select foreground region (mask)")
                                    .font(.caption)
                                    .foregroundColor(.secondary)

                                MaskSelectionView(
                                    image: viewModel.compositeImage!,
                                    maskRect: $viewModel.normalizedMaskRect
                                )
                                .frame(height: 250)
                                .cornerRadius(12)
                            }
                        } header: {
                            sectionHeader("Mask Selection")
                        }
                    }

                    // Harmonize button
                    if viewModel.compositeImage != nil {
                        Button(action: { viewModel.harmonize() }) {
                            HStack {
                                if viewModel.isProcessing {
                                    ProgressView()
                                        .tint(.white)
                                } else {
                                    Image(systemName: "wand.and.stars")
                                }
                                Text("Harmonize")
                            }
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(viewModel.isProcessing ? Color.gray : Color.accentColor)
                            .foregroundColor(.white)
                            .cornerRadius(12)
                        }
                        .disabled(viewModel.isProcessing)
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

                    // Before / After comparison
                    if viewModel.harmonizedImage != nil {
                        Section {
                            BeforeAfterView(
                                before: viewModel.compositeImage!,
                                after: viewModel.harmonizedImage!
                            )
                            .frame(height: 300)
                            .cornerRadius(12)
                        } header: {
                            sectionHeader("Result: Before / After")
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("CDTNet Harmonization")
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
        .frame(height: 180)
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

// MARK: - ViewModel

class HarmonizationViewModel: ObservableObject {
    @Published var selectedPhoto: PhotosPickerItem? {
        didSet { loadImage() }
    }
    @Published var compositeImage: UIImage?
    @Published var harmonizedImage: UIImage?
    @Published var normalizedMaskRect: CGRect = CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5)
    @Published var isProcessing = false
    @Published var errorMessage: String?

    private func loadImage() {
        guard let item = selectedPhoto else { return }
        Task {
            if let data = try? await item.loadTransferable(type: Data.self),
               let image = UIImage(data: data) {
                await MainActor.run {
                    self.compositeImage = image
                    self.harmonizedImage = nil
                    self.errorMessage = nil
                }
            }
        }
    }

    func harmonize() {
        guard let inputImage = compositeImage else { return }
        isProcessing = true
        errorMessage = nil

        Task {
            do {
                let result = try await performHarmonization(image: inputImage, maskRect: normalizedMaskRect)
                await MainActor.run {
                    self.harmonizedImage = result
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

    // Perform harmonization using CDTNet CoreML model
    // Input: composite_image (1,3,256,256) + mask (1,1,256,256) -> harmonized (1,3,256,256)
    private func performHarmonization(image: UIImage, maskRect: CGRect) async throws -> UIImage {
        // Load the CoreML model
        guard let modelURL = Bundle.main.url(forResource: "CDTNet_Harmonization", withExtension: "mlmodelc") else {
            throw HarmonizationError.modelNotFound(
                "CDTNet_Harmonization.mlmodelc not found in bundle. " +
                "Please compile and add the CDTNet_Harmonization.mlpackage to the project."
            )
        }

        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        let model = try MLModel(contentsOf: modelURL, configuration: config)

        // Prepare composite image input (1, 3, 256, 256)
        let targetSize = CGSize(width: 256, height: 256)
        guard let resizedCG = image.resized(to: targetSize)?.cgImage else {
            throw HarmonizationError.imageProcessingFailed("Failed to resize composite image")
        }

        let compositeArray = try MLMultiArray(shape: [1, 3, 256, 256], dataType: .float32)
        fillMultiArrayFromImage(resizedCG, into: compositeArray)

        // Prepare mask input (1, 1, 256, 256) from the rectangular selection
        let maskArray = try MLMultiArray(shape: [1, 1, 256, 256], dataType: .float32)
        fillMaskArray(maskArray, rect: maskRect)

        // Run inference
        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
            "composite_image": MLFeatureValue(multiArray: compositeArray),
            "mask": MLFeatureValue(multiArray: maskArray)
        ])
        let prediction = try model.prediction(from: inputFeatures)

        // Extract harmonized output (1, 3, 256, 256)
        guard let outputArray = prediction.featureValue(for: "harmonized_image")?.multiArrayValue else {
            throw HarmonizationError.imageProcessingFailed("Failed to extract harmonized output")
        }

        let resultImage = imageFromMultiArray(outputArray, width: 256, height: 256)
        guard let finalImage = resultImage else {
            throw HarmonizationError.imageProcessingFailed("Failed to convert output to UIImage")
        }
        return finalImage
    }

    // Fill MLMultiArray with pixel data from CGImage (RGB, normalized 0-1)
    private func fillMultiArrayFromImage(_ cgImage: CGImage, into array: MLMultiArray) {
        let width = 256
        let height = 256
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        var pixelData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &pixelData, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        for y in 0..<height {
            for x in 0..<width {
                let offset = (y * width + x) * bytesPerPixel
                let r = Float(pixelData[offset]) / 255.0
                let g = Float(pixelData[offset + 1]) / 255.0
                let b = Float(pixelData[offset + 2]) / 255.0

                array[[0, 0, y, x] as [NSNumber]] = NSNumber(value: r)
                array[[0, 1, y, x] as [NSNumber]] = NSNumber(value: g)
                array[[0, 2, y, x] as [NSNumber]] = NSNumber(value: b)
            }
        }
    }

    // Create a binary mask from a normalized rect (values 0 or 1)
    private func fillMaskArray(_ array: MLMultiArray, rect: CGRect) {
        let width = 256
        let height = 256
        let x0 = Int(rect.minX * CGFloat(width))
        let y0 = Int(rect.minY * CGFloat(height))
        let x1 = Int(rect.maxX * CGFloat(width))
        let y1 = Int(rect.maxY * CGFloat(height))

        for y in 0..<height {
            for x in 0..<width {
                let value: Float = (x >= x0 && x < x1 && y >= y0 && y < y1) ? 1.0 : 0.0
                array[[0, 0, y, x] as [NSNumber]] = NSNumber(value: value)
            }
        }
    }

    // Convert (1, 3, 256, 256) MLMultiArray back to UIImage
    private func imageFromMultiArray(_ array: MLMultiArray, width: Int, height: Int) -> UIImage? {
        var pixelData = [UInt8](repeating: 255, count: width * height * 4)

        for y in 0..<height {
            for x in 0..<width {
                let r = min(max(array[[0, 0, y, x] as [NSNumber]].floatValue, 0), 1)
                let g = min(max(array[[0, 1, y, x] as [NSNumber]].floatValue, 0), 1)
                let b = min(max(array[[0, 2, y, x] as [NSNumber]].floatValue, 0), 1)
                let offset = (y * width + x) * 4
                pixelData[offset] = UInt8(r * 255)
                pixelData[offset + 1] = UInt8(g * 255)
                pixelData[offset + 2] = UInt8(b * 255)
                pixelData[offset + 3] = 255
            }
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &pixelData, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ), let cgImage = context.makeImage() else { return nil }

        return UIImage(cgImage: cgImage)
    }
}

enum HarmonizationError: LocalizedError {
    case modelNotFound(String)
    case imageProcessingFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelNotFound(let msg): return msg
        case .imageProcessingFailed(let msg): return msg
        }
    }
}

// MARK: - Mask Selection View (drag rectangle)

struct MaskSelectionView: View {
    let image: UIImage
    @Binding var maskRect: CGRect

    @State private var dragStart: CGPoint = .zero
    @State private var dragCurrent: CGPoint = .zero
    @State private var isDragging = false

    var body: some View {
        GeometryReader { geo in
            ZStack {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(width: geo.size.width, height: geo.size.height)

                // Semi-transparent mask overlay
                Rectangle()
                    .fill(Color.blue.opacity(0.3))
                    .border(Color.blue, width: 2)
                    .frame(
                        width: abs(dragCurrent.x - dragStart.x),
                        height: abs(dragCurrent.y - dragStart.y)
                    )
                    .position(
                        x: (dragStart.x + dragCurrent.x) / 2,
                        y: (dragStart.y + dragCurrent.y) / 2
                    )
                    .opacity(isDragging ? 1 : 0)

                // Display current mask rect
                if !isDragging {
                    Rectangle()
                        .fill(Color.green.opacity(0.25))
                        .border(Color.green, width: 2)
                        .frame(
                            width: maskRect.width * geo.size.width,
                            height: maskRect.height * geo.size.height
                        )
                        .position(
                            x: (maskRect.minX + maskRect.width / 2) * geo.size.width,
                            y: (maskRect.minY + maskRect.height / 2) * geo.size.height
                        )
                }
            }
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 5)
                    .onChanged { value in
                        if !isDragging {
                            isDragging = true
                            dragStart = value.startLocation
                        }
                        dragCurrent = value.location
                    }
                    .onEnded { value in
                        isDragging = false
                        let minX = min(dragStart.x, value.location.x) / geo.size.width
                        let minY = min(dragStart.y, value.location.y) / geo.size.height
                        let maxX = max(dragStart.x, value.location.x) / geo.size.width
                        let maxY = max(dragStart.y, value.location.y) / geo.size.height
                        maskRect = CGRect(
                            x: max(0, minX), y: max(0, minY),
                            width: min(1, maxX) - max(0, minX),
                            height: min(1, maxY) - max(0, minY)
                        )
                    }
            )
        }
    }
}

// MARK: - Before/After Comparison

struct BeforeAfterView: View {
    let before: UIImage
    let after: UIImage
    @State private var sliderPosition: CGFloat = 0.5

    var body: some View {
        GeometryReader { geo in
            ZStack {
                Image(uiImage: after)
                    .resizable()
                    .scaledToFit()

                Image(uiImage: before)
                    .resizable()
                    .scaledToFit()
                    .mask(
                        HStack(spacing: 0) {
                            Rectangle()
                                .frame(width: geo.size.width * sliderPosition)
                            Spacer(minLength: 0)
                        }
                    )

                // Divider line
                Rectangle()
                    .fill(Color.white)
                    .frame(width: 3)
                    .position(x: geo.size.width * sliderPosition, y: geo.size.height / 2)
                    .shadow(radius: 2)

                // Labels
                VStack {
                    HStack {
                        Text("Before")
                            .font(.caption)
                            .padding(4)
                            .background(Color.black.opacity(0.6))
                            .foregroundColor(.white)
                            .cornerRadius(4)
                        Spacer()
                        Text("After")
                            .font(.caption)
                            .padding(4)
                            .background(Color.black.opacity(0.6))
                            .foregroundColor(.white)
                            .cornerRadius(4)
                    }
                    .padding(.horizontal, 8)
                    Spacer()
                }
                .padding(.top, 8)
            }
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        sliderPosition = max(0, min(1, value.location.x / geo.size.width))
                    }
            )
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
