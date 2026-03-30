import SwiftUI
import UIKit
import CoreML
import Vision
import PhotosUI
import Photos
import Accelerate

// MARK: - Background Removal using BiRefNet
// BiRefNet is a bilateral reference network for high-resolution dichotomous image segmentation.
// It takes an input image and produces a precise foreground mask, enabling clean background removal.

struct ContentView: View {
    @StateObject private var viewModel = BackgroundRemovalViewModel()

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    // Image picker section
                    Section {
                        PhotosPicker(selection: $viewModel.selectedPhoto,
                                     matching: .images) {
                            if let image = viewModel.inputImage {
                                Image(uiImage: image)
                                    .resizable()
                                    .scaledToFit()
                                    .frame(maxHeight: 250)
                                    .cornerRadius(12)
                            } else {
                                placeholderView(title: "Select an Image",
                                                systemImage: "photo.on.rectangle")
                            }
                        }
                    } header: {
                        sectionHeader("Input Image")
                    }

                    // Process button
                    if viewModel.inputImage != nil {
                        Button(action: { viewModel.removeBackground() }) {
                            HStack {
                                if viewModel.isProcessing {
                                    ProgressView()
                                        .tint(.white)
                                } else {
                                    Image(systemName: "scissors")
                                }
                                Text(viewModel.isProcessing ? "Processing..." : "Remove Background")
                            }
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(viewModel.isProcessing ? Color.gray : Color.accentColor)
                            .foregroundColor(.white)
                            .cornerRadius(12)
                        }
                        .disabled(viewModel.isProcessing)
                    }

                    // Progress indicator
                    if viewModel.isProcessing {
                        VStack(spacing: 8) {
                            ProgressView(value: viewModel.progress)
                                .progressViewStyle(.linear)
                            Text(viewModel.progressMessage)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        .padding(.horizontal)
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

                    // Display mode selector
                    if viewModel.maskImage != nil {
                        Section {
                            Picker("Display Mode", selection: $viewModel.displayMode) {
                                Text("Comparison").tag(DisplayMode.comparison)
                                Text("Mask").tag(DisplayMode.mask)
                                Text("Cutout").tag(DisplayMode.cutout)
                            }
                            .pickerStyle(.segmented)
                        } header: {
                            sectionHeader("View Mode")
                        }
                    }

                    // Before / After comparison
                    if viewModel.displayMode == .comparison,
                       let original = viewModel.inputImage,
                       let cutout = viewModel.cutoutImage {
                        Section {
                            BeforeAfterView(
                                before: original,
                                after: cutout
                            )
                            .frame(height: 300)
                            .cornerRadius(12)
                        } header: {
                            sectionHeader("Before / After")
                        }
                    }

                    // Mask view
                    if viewModel.displayMode == .mask,
                       let mask = viewModel.maskImage {
                        Section {
                            Image(uiImage: mask)
                                .resizable()
                                .scaledToFit()
                                .frame(maxHeight: 300)
                                .cornerRadius(12)
                        } header: {
                            sectionHeader("Segmentation Mask")
                        }
                    }

                    // Cutout result
                    if viewModel.displayMode == .cutout,
                       let cutout = viewModel.cutoutImage {
                        Section {
                            VStack(spacing: 12) {
                                // Background color selector
                                HStack(spacing: 12) {
                                    Text("Background:")
                                        .font(.subheadline)
                                    ForEach(BackgroundOption.allCases, id: \.self) { option in
                                        Button(action: {
                                            viewModel.backgroundOption = option
                                            viewModel.updateCutout()
                                        }) {
                                            Circle()
                                                .fill(option.color)
                                                .frame(width: 30, height: 30)
                                                .overlay(
                                                    Circle()
                                                        .stroke(viewModel.backgroundOption == option ? Color.accentColor : Color.clear, lineWidth: 3)
                                                )
                                                .overlay(
                                                    option == .transparent ?
                                                    Image(systemName: "checkerboard.rectangle")
                                                        .font(.caption2)
                                                        .foregroundColor(.gray) : nil
                                                )
                                        }
                                    }
                                    Spacer()
                                }

                                // Cutout image with checkerboard for transparent
                                ZStack {
                                    if viewModel.backgroundOption == .transparent {
                                        CheckerboardView()
                                            .frame(maxHeight: 300)
                                            .cornerRadius(12)
                                    }
                                    Image(uiImage: cutout)
                                        .resizable()
                                        .scaledToFit()
                                        .frame(maxHeight: 300)
                                        .cornerRadius(12)
                                }
                            }
                        } header: {
                            sectionHeader("Cutout Result")
                        }
                    }

                    // Save button
                    if viewModel.cutoutImage != nil {
                        Button(action: { viewModel.saveToPhotoLibrary() }) {
                            HStack {
                                Image(systemName: "square.and.arrow.down")
                                Text("Save to Photos")
                            }
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.green)
                            .foregroundColor(.white)
                            .cornerRadius(12)
                        }

                        if viewModel.savedSuccessfully {
                            Text("Saved to Photo Library!")
                                .foregroundColor(.green)
                                .font(.caption)
                                .transition(.opacity)
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("BiRefNet Background Removal")
            .navigationBarTitleDisplayMode(.inline)
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

// MARK: - Display Mode

enum DisplayMode {
    case comparison
    case mask
    case cutout
}

// MARK: - Background Options

enum BackgroundOption: CaseIterable {
    case transparent
    case white
    case black
    case green
    case blue

    var color: Color {
        switch self {
        case .transparent: return Color.clear
        case .white: return Color.white
        case .black: return Color.black
        case .green: return Color.green
        case .blue: return Color.blue
        }
    }

    var uiColor: UIColor? {
        switch self {
        case .transparent: return nil
        case .white: return .white
        case .black: return .black
        case .green: return UIColor(red: 0, green: 0.8, blue: 0, alpha: 1)
        case .blue: return UIColor(red: 0, green: 0.4, blue: 1, alpha: 1)
        }
    }
}

// MARK: - ViewModel

class BackgroundRemovalViewModel: ObservableObject {
    @Published var selectedPhoto: PhotosPickerItem? {
        didSet { loadImage() }
    }
    @Published var inputImage: UIImage?
    @Published var maskImage: UIImage?
    @Published var cutoutImage: UIImage?
    @Published var isProcessing = false
    @Published var progress: Double = 0.0
    @Published var progressMessage: String = ""
    @Published var errorMessage: String?
    @Published var displayMode: DisplayMode = .comparison
    @Published var backgroundOption: BackgroundOption = .transparent
    @Published var savedSuccessfully = false

    private var rawMaskData: [Float]?
    private var maskWidth: Int = 0
    private var maskHeight: Int = 0

    private func loadImage() {
        guard let item = selectedPhoto else { return }
        Task {
            if let data = try? await item.loadTransferable(type: Data.self),
               let image = UIImage(data: data) {
                await MainActor.run {
                    self.inputImage = image
                    self.maskImage = nil
                    self.cutoutImage = nil
                    self.errorMessage = nil
                    self.savedSuccessfully = false
                    self.rawMaskData = nil
                    self.displayMode = .comparison
                }
            }
        }
    }

    func removeBackground() {
        guard let inputImage = inputImage else { return }
        isProcessing = true
        errorMessage = nil
        progress = 0.0
        progressMessage = "Loading model..."

        Task {
            do {
                let result = try await performSegmentation(image: inputImage)
                await MainActor.run {
                    self.maskImage = result.mask
                    self.cutoutImage = result.cutout
                    self.isProcessing = false
                    self.progress = 1.0
                    self.progressMessage = "Complete!"
                    self.displayMode = .comparison
                }
            } catch {
                await MainActor.run {
                    self.errorMessage = error.localizedDescription
                    self.isProcessing = false
                    self.progress = 0.0
                    self.progressMessage = ""
                }
            }
        }
    }

    func updateCutout() {
        guard let inputImage = inputImage,
              let maskData = rawMaskData else { return }
        let w = maskWidth
        let h = maskHeight
        cutoutImage = applyMask(to: inputImage, maskData: maskData,
                                maskWidth: w, maskHeight: h,
                                background: backgroundOption.uiColor)
    }

    func saveToPhotoLibrary() {
        guard let image = cutoutImage else { return }
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            guard status == .authorized || status == .limited else {
                DispatchQueue.main.async {
                    self.errorMessage = "Photo library access denied."
                }
                return
            }
            guard let pngData = image.pngData() else {
                DispatchQueue.main.async {
                    self.errorMessage = "Failed to encode image."
                }
                return
            }
            PHPhotoLibrary.shared().performChanges {
                let request = PHAssetCreationRequest.forAsset()
                request.addResource(with: .photo, data: pngData, options: nil)
            } completionHandler: { success, error in
                DispatchQueue.main.async {
                    if success {
                        self.savedSuccessfully = true
                        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                            self.savedSuccessfully = false
                        }
                    } else {
                        self.errorMessage = error?.localizedDescription ?? "Failed to save."
                    }
                }
            }
        }
    }

    // MARK: - Core ML Inference

    private func performSegmentation(image: UIImage) async throws -> (mask: UIImage, cutout: UIImage) {
        // Load the CoreML model
        guard let modelURL = Bundle.main.url(forResource: "BiRefNet", withExtension: "mlmodelc") else {
            throw SegmentationError.modelNotFound(
                "BiRefNet.mlmodelc not found in bundle. " +
                "Please convert the BiRefNet model to CoreML format using convert_birefnet.py, " +
                "then compile the .mlpackage and add it to the Xcode project."
            )
        }

        await MainActor.run {
            self.progress = 0.1
            self.progressMessage = "Loading model..."
        }

        let config = MLModelConfiguration()
        // ANE compilation fails on this model. Use CPU+GPU.
        config.computeUnits = .cpuAndGPU
        let model = try MLModel(contentsOf: modelURL, configuration: config)

        await MainActor.run {
            self.progress = 0.3
            self.progressMessage = "Preparing image..."
        }

        // Prepare input image (1, 3, 512, 512)
        let targetSize = CGSize(width: 512, height: 512)
        guard let resizedCG = image.resized(to: targetSize)?.cgImage else {
            throw SegmentationError.imageProcessingFailed("Failed to resize input image")
        }

        let inputArray = try MLMultiArray(shape: [1, 3, 512, 512], dataType: .float16)
        fillMultiArrayFromImage(resizedCG, into: inputArray, size: 512)

        await MainActor.run {
            self.progress = 0.5
            self.progressMessage = "Running BiRefNet inference..."
        }

        // Run inference
        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(multiArray: inputArray)
        ])
        let prediction = try model.prediction(from: inputFeatures)

        await MainActor.run {
            self.progress = 0.8
            self.progressMessage = "Generating mask..."
        }

        // Extract mask output (1, 1, 512, 512), apply sigmoid
        guard let outputArray = prediction.featureValue(for: "mask")?.multiArrayValue else {
            throw SegmentationError.imageProcessingFailed("Failed to extract mask output from model")
        }

        let width = 512
        let height = 512
        let totalPixels = width * height
        var maskData = [Float](repeating: 0, count: totalPixels)

        // Output is Float16 - read as UInt16 and convert to Float32
        let fp16Ptr = outputArray.dataPointer.bindMemory(to: UInt16.self, capacity: totalPixels)
        var rawFloats = [Float](repeating: 0, count: totalPixels)
        var srcBuf = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: fp16Ptr), height: 1, width: vImagePixelCount(totalPixels), rowBytes: totalPixels * 2)
        rawFloats.withUnsafeMutableBufferPointer { dstBufPtr in
            var dstBuf = vImage_Buffer(data: dstBufPtr.baseAddress!, height: 1, width: vImagePixelCount(totalPixels), rowBytes: totalPixels * 4)
            vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, 0)
        }
        for i in 0..<totalPixels {
            let raw = rawFloats[i]
            maskData[i] = raw.isNaN ? 0 : 1.0 / (1.0 + exp(-raw)) // sigmoid
        }

        // Store raw mask for background option changes
        await MainActor.run {
            self.rawMaskData = maskData
            self.maskWidth = width
            self.maskHeight = height
        }

        // Generate mask visualization image
        let maskUIImage = maskToUIImage(maskData: maskData, width: width, height: height)
        guard let finalMask = maskUIImage else {
            throw SegmentationError.imageProcessingFailed("Failed to create mask image")
        }

        await MainActor.run {
            self.progress = 0.9
            self.progressMessage = "Applying mask to image..."
        }

        // Apply mask to original image for cutout
        let bgColor = await MainActor.run { self.backgroundOption.uiColor }
        let cutoutUIImage = applyMask(to: image, maskData: maskData,
                                       maskWidth: width, maskHeight: height,
                                       background: bgColor)
        guard let finalCutout = cutoutUIImage else {
            throw SegmentationError.imageProcessingFailed("Failed to apply mask to image")
        }

        return (mask: finalMask, cutout: finalCutout)
    }

    // MARK: - Image Processing Helpers

    /// Fill MLMultiArray with pixel data from CGImage (RGB, normalized 0-1)
    private func fillMultiArrayFromImage(_ cgImage: CGImage, into array: MLMultiArray, size: Int) {
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * size
        var pixelData = [UInt8](repeating: 0, count: size * size * bytesPerPixel)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &pixelData, width: size, height: size,
            bitsPerComponent: 8, bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: size, height: size))

        // Write as Float16 (model expects Float16 input)
        let fp16Ptr = array.dataPointer.bindMemory(to: UInt16.self, capacity: 3 * size * size)
        let channelStride = size * size

        for y in 0..<size {
            for x in 0..<size {
                let offset = (y * size + x) * bytesPerPixel
                let idx = y * size + x
                var r = Float(pixelData[offset]) / 255.0
                var g = Float(pixelData[offset + 1]) / 255.0
                var b = Float(pixelData[offset + 2]) / 255.0
                var rh: UInt16 = 0, gh: UInt16 = 0, bh: UInt16 = 0
                withUnsafePointer(to: &r) { src in
                    var srcBuf = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src), height: 1, width: 1, rowBytes: 4)
                    var dstBuf = vImage_Buffer(data: &rh, height: 1, width: 1, rowBytes: 2)
                    vImageConvert_PlanarFtoPlanar16F(&srcBuf, &dstBuf, 0)
                }
                withUnsafePointer(to: &g) { src in
                    var srcBuf = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src), height: 1, width: 1, rowBytes: 4)
                    var dstBuf = vImage_Buffer(data: &gh, height: 1, width: 1, rowBytes: 2)
                    vImageConvert_PlanarFtoPlanar16F(&srcBuf, &dstBuf, 0)
                }
                withUnsafePointer(to: &b) { src in
                    var srcBuf = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src), height: 1, width: 1, rowBytes: 4)
                    var dstBuf = vImage_Buffer(data: &bh, height: 1, width: 1, rowBytes: 2)
                    vImageConvert_PlanarFtoPlanar16F(&srcBuf, &dstBuf, 0)
                }
                fp16Ptr[0 * channelStride + idx] = rh
                fp16Ptr[1 * channelStride + idx] = gh
                fp16Ptr[2 * channelStride + idx] = bh
            }
        }
    }

    /// Convert float mask data to a grayscale UIImage
    private func maskToUIImage(maskData: [Float], width: Int, height: Int) -> UIImage? {
        var pixelData = [UInt8](repeating: 0, count: width * height * 4)

        for i in 0..<(width * height) {
            let v = maskData[i]
            let val = UInt8(v.isNaN ? 0 : min(max(v, 0), 1) * 255)
            pixelData[i * 4] = val
            pixelData[i * 4 + 1] = val
            pixelData[i * 4 + 2] = val
            pixelData[i * 4 + 3] = 255
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

    /// Apply the segmentation mask to the original image
    /// If background color is nil, the result has transparency (PNG-friendly).
    private func applyMask(to image: UIImage, maskData: [Float],
                           maskWidth: Int, maskHeight: Int,
                           background: UIColor?) -> UIImage? {
        // Normalize orientation first to avoid rotation mismatch
        let normalizedImage = normalizeOrientation(image)
        let origWidth = Int(normalizedImage.size.width)
        let origHeight = Int(normalizedImage.size.height)

        guard let cgImage = normalizedImage.cgImage else { return nil }

        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * origWidth
        var pixelData = [UInt8](repeating: 0, count: origWidth * origHeight * bytesPerPixel)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &pixelData, width: origWidth, height: origHeight,
            bitsPerComponent: 8, bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: origWidth, height: origHeight))

        // Determine background RGBA
        var bgR: UInt8 = 0, bgG: UInt8 = 0, bgB: UInt8 = 0, bgA: UInt8 = 0
        if let bg = background {
            var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
            bg.getRed(&r, green: &g, blue: &b, alpha: &a)
            bgR = UInt8(r * 255)
            bgG = UInt8(g * 255)
            bgB = UInt8(b * 255)
            bgA = UInt8(a * 255)
        }

        var outputData = [UInt8](repeating: 0, count: origWidth * origHeight * bytesPerPixel)

        for y in 0..<origHeight {
            for x in 0..<origWidth {
                // Map original pixel coordinates to mask coordinates
                let maskX = min(Int(Float(x) / Float(origWidth) * Float(maskWidth)), maskWidth - 1)
                let maskY = min(Int(Float(y) / Float(origHeight) * Float(maskHeight)), maskHeight - 1)
                let alpha = maskData[maskY * maskWidth + maskX]

                let srcIdx = (y * origWidth + x) * bytesPerPixel
                let dstIdx = srcIdx

                let srcR = Float(pixelData[srcIdx])
                let srcG = Float(pixelData[srcIdx + 1])
                let srcB = Float(pixelData[srcIdx + 2])

                if background != nil {
                    // Blend foreground with background color
                    outputData[dstIdx]     = UInt8(min(max(srcR * alpha + Float(bgR) * (1 - alpha), 0), 255))
                    outputData[dstIdx + 1] = UInt8(min(max(srcG * alpha + Float(bgG) * (1 - alpha), 0), 255))
                    outputData[dstIdx + 2] = UInt8(min(max(srcB * alpha + Float(bgB) * (1 - alpha), 0), 255))
                    outputData[dstIdx + 3] = 255
                } else {
                    // Transparent background
                    let a = UInt8(min(max(alpha * 255, 0), 255))
                    outputData[dstIdx]     = UInt8(min(max(srcR * alpha, 0), 255))
                    outputData[dstIdx + 1] = UInt8(min(max(srcG * alpha, 0), 255))
                    outputData[dstIdx + 2] = UInt8(min(max(srcB * alpha, 0), 255))
                    outputData[dstIdx + 3] = a
                }
            }
        }

        let alphaInfo: CGImageAlphaInfo = background != nil ? .premultipliedLast : .premultipliedLast
        guard let outputContext = CGContext(
            data: &outputData, width: origWidth, height: origHeight,
            bitsPerComponent: 8, bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: alphaInfo.rawValue
        ), let outputCG = outputContext.makeImage() else { return nil }

        return UIImage(cgImage: outputCG)
    }

    /// Redraw UIImage with .up orientation to strip rotation metadata
    private func normalizeOrientation(_ image: UIImage) -> UIImage {
        guard image.imageOrientation != .up else { return image }
        UIGraphicsBeginImageContextWithOptions(image.size, false, image.scale)
        image.draw(in: CGRect(origin: .zero, size: image.size))
        let normalized = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return normalized ?? image
    }
}

// MARK: - Errors

enum SegmentationError: LocalizedError {
    case modelNotFound(String)
    case imageProcessingFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelNotFound(let msg): return msg
        case .imageProcessingFailed(let msg): return msg
        }
    }
}

// MARK: - Before/After Comparison View

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
                    .frame(width: geo.size.width, height: geo.size.height)

                Image(uiImage: before)
                    .resizable()
                    .scaledToFit()
                    .frame(width: geo.size.width, height: geo.size.height)
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

                // Drag handle
                Circle()
                    .fill(Color.white)
                    .frame(width: 30, height: 30)
                    .shadow(radius: 3)
                    .overlay(
                        Image(systemName: "arrow.left.and.right")
                            .font(.caption)
                            .foregroundColor(.gray)
                    )
                    .position(x: geo.size.width * sliderPosition, y: geo.size.height / 2)

                // Labels
                VStack {
                    HStack {
                        Text("Original")
                            .font(.caption)
                            .padding(4)
                            .background(Color.black.opacity(0.6))
                            .foregroundColor(.white)
                            .cornerRadius(4)
                        Spacer()
                        Text("Removed")
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

// MARK: - Checkerboard View (for transparent background visualization)

struct CheckerboardView: View {
    let tileSize: CGFloat = 12

    var body: some View {
        GeometryReader { geo in
            Canvas { context, size in
                let rows = Int(ceil(size.height / tileSize))
                let cols = Int(ceil(size.width / tileSize))
                for row in 0..<rows {
                    for col in 0..<cols {
                        let isLight = (row + col) % 2 == 0
                        let rect = CGRect(
                            x: CGFloat(col) * tileSize,
                            y: CGFloat(row) * tileSize,
                            width: tileSize,
                            height: tileSize
                        )
                        context.fill(
                            Path(rect),
                            with: .color(isLight ? Color(white: 0.9) : Color(white: 0.75))
                        )
                    }
                }
            }
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
