import SwiftUI
import UIKit
import CoreML
import PhotosUI
import CoreMotion
import Accelerate

// MARK: - Apple Depth Pro - Metric Depth Estimation Demo
//
// Depth Pro produces metric (absolute) depth maps from a single image,
// along with an estimated focal length. Input: 1536x1536 (fixed).
// Outputs: depth map (meters) + focal length (pixels).
//
// NOTE: Requires iPhone 15 Pro or later (6GB+ RAM).
// The model is ~1.2GB and processes 1536x1536 input.
//
// Features:
// - PhotosPicker for image selection
// - Color-coded depth visualization (turbo colormap)
// - Tap to measure distance at any point
// - 3D parallax effect using CoreMotion
// - Before/After depth overlay slider
// - Focal length display
// - Save depth map as image

// MARK: - Turbo Colormap

struct TurboColormap {
    /// Maps a normalized value [0,1] to a turbo colormap RGB tuple.
    /// Blue = far (0.0), Red = near (1.0).
    static func color(for value: Float) -> (r: UInt8, g: UInt8, b: UInt8) {
        let t = max(0, min(1, value))
        let r = clampByte(34.61 + t * (1172.33 - t * (10793.56 - t * (33300.12 - t * (38394.49 - t * 14825.05)))))
        let g = clampByte(23.31 + t * (557.33 + t * (1225.33 - t * (3574.96 - t * (1073.77 + t * 707.56)))))
        let b = clampByte(27.2 + t * (3211.1 - t * (15327.97 - t * (27814.0 - t * (22569.18 - t * 6838.66)))))
        return (r, g, b)
    }

    private static func clampByte(_ v: Float) -> UInt8 {
        return UInt8(max(0, min(255, Int(v))))
    }

    /// Generates a UIImage depth visualization from a depth float buffer.
    static func depthMapImage(from depthValues: [Float], width: Int, height: Int, minDepth: Float, maxDepth: Float) -> UIImage? {
        let count = width * height
        guard depthValues.count >= count else { return nil }

        var pixelData = [UInt8](repeating: 255, count: count * 4)
        let range = maxDepth - minDepth
        let safeRange = range > 0 ? range : 1.0

        for i in 0..<count {
            // Invert so near = 1.0 (red), far = 0.0 (blue)
            let normalized = 1.0 - ((depthValues[i] - minDepth) / safeRange)
            let (r, g, b) = color(for: normalized)
            pixelData[i * 4] = r
            pixelData[i * 4 + 1] = g
            pixelData[i * 4 + 2] = b
            pixelData[i * 4 + 3] = 255
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ), let cgImage = context.makeImage() else { return nil }

        return UIImage(cgImage: cgImage)
    }
}

// MARK: - Motion Manager

class MotionManager: ObservableObject {
    private let motionManager = CMMotionManager()
    @Published var pitch: Double = 0
    @Published var roll: Double = 0

    func start() {
        guard motionManager.isDeviceMotionAvailable else { return }
        motionManager.deviceMotionUpdateInterval = 1.0 / 60.0
        motionManager.startDeviceMotionUpdates(to: .main) { [weak self] motion, _ in
            guard let motion = motion else { return }
            self?.pitch = motion.attitude.pitch
            self?.roll = motion.attitude.roll
        }
    }

    func stop() {
        motionManager.stopDeviceMotionUpdates()
    }
}

// MARK: - ContentView

struct ContentView: View {
    @StateObject private var viewModel = DepthProViewModel()
    @StateObject private var motionManager = MotionManager()

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    // Device compatibility warning
                    HStack(spacing: 8) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.orange)
                        Text("Requires iPhone 15 Pro+ (6GB RAM). May crash on older devices.")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding(10)
                    .background(Color.orange.opacity(0.1))
                    .cornerRadius(8)

                    imageSelectionSection
                    processSection
                    progressSection
                    errorSection
                    if viewModel.depthMapImage != nil {
                        resultsSection
                        depthOverlaySection
                        pointMeasurementSection
                        parallaxSection
                        saveSection
                    }
                    colormapLegendSection
                }
                .padding()
            }
            .navigationTitle("Depth Pro")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    if viewModel.isProcessed {
                        Button(action: { viewModel.reset() }) {
                            Image(systemName: "arrow.counterclockwise")
                        }
                    }
                }
            }
        }
        .onAppear { motionManager.start() }
        .onDisappear { motionManager.stop() }
    }

    // MARK: - Image Selection Section

    private var imageSelectionSection: some View {
        VStack(spacing: 12) {
            Text("Select Image")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)

            if let image = viewModel.selectedImage {
                ZStack(alignment: .topTrailing) {
                    Image(uiImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(maxHeight: 280)
                        .cornerRadius(12)

                    PhotosPicker(selection: $viewModel.photoItem, matching: .images) {
                        Image(systemName: "arrow.triangle.2.circlepath.camera.fill")
                            .font(.title3)
                            .padding(8)
                            .background(.ultraThinMaterial)
                            .clipShape(Circle())
                    }
                    .padding(8)
                }
            } else {
                PhotosPicker(selection: $viewModel.photoItem, matching: .images) {
                    VStack(spacing: 12) {
                        Image(systemName: "photo.on.rectangle.angled")
                            .font(.system(size: 40))
                            .foregroundColor(.secondary)
                        Text("Choose a Photo")
                            .foregroundColor(.secondary)
                        Text("Depth Pro estimates metric depth for any image")
                            .font(.caption2)
                            .foregroundColor(.secondary.opacity(0.7))
                    }
                    .frame(maxWidth: .infinity)
                    .frame(height: 160)
                    .background(Color(.systemGray6))
                    .cornerRadius(12)
                }
            }
        }
    }

    // MARK: - Process Section

    private var processSection: some View {
        Group {
            if viewModel.selectedImage != nil && !viewModel.isProcessed {
                Button(action: { viewModel.estimateDepth() }) {
                    HStack {
                        if viewModel.isProcessing {
                            ProgressView()
                                .tint(.white)
                        } else {
                            Image(systemName: "cube.transparent")
                        }
                        Text(viewModel.isProcessing ? "Estimating Depth..." : "Estimate Depth")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(viewModel.isProcessing ? Color.gray : Color.accentColor)
                    .foregroundColor(.white)
                    .cornerRadius(12)
                }
                .disabled(viewModel.isProcessing)
            }
        }
    }

    // MARK: - Progress Section

    private var progressSection: some View {
        Group {
            if viewModel.isProcessing {
                VStack(spacing: 8) {
                    ProgressView(value: viewModel.progress)
                        .progressViewStyle(.linear)
                    Text(viewModel.statusMessage)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
    }

    // MARK: - Error Section

    private var errorSection: some View {
        Group {
            if let error = viewModel.errorMessage {
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
        }
    }

    // MARK: - Results Section (Side by Side)

    private var resultsSection: some View {
        VStack(spacing: 12) {
            Text("Depth Estimation Results")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)

            // Focal length info
            if let focal = viewModel.estimatedFocalLength {
                HStack {
                    Image(systemName: "camera.aperture")
                        .foregroundColor(.orange)
                    Text("Estimated Focal Length:")
                        .font(.subheadline)
                    Spacer()
                    Text(String(format: "%.1f px", focal))
                        .font(.subheadline)
                        .fontWeight(.semibold)
                        .foregroundColor(.orange)
                }
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(10)
            }

            // Depth statistics
            if let stats = viewModel.depthStats {
                HStack(spacing: 16) {
                    depthStatView(label: "Min", value: stats.min, color: .red)
                    depthStatView(label: "Max", value: stats.max, color: .blue)
                    depthStatView(label: "Mean", value: stats.mean, color: .green)
                }
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(10)
            }

            // Side-by-side comparison
            HStack(spacing: 8) {
                VStack(spacing: 4) {
                    Text("Original")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    if let img = viewModel.selectedImage {
                        Image(uiImage: img)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .cornerRadius(8)
                    }
                }

                VStack(spacing: 4) {
                    Text("Depth Map")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    if let depthImg = viewModel.depthMapImage {
                        Image(uiImage: depthImg)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .cornerRadius(8)
                    }
                }
            }
        }
    }

    // MARK: - Depth Overlay Section (Before/After Slider)

    private var depthOverlaySection: some View {
        VStack(spacing: 12) {
            Text("Before / After Overlay")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)

            ZStack {
                if let original = viewModel.selectedImage {
                    Image(uiImage: original)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                }

                GeometryReader { geo in
                    if let depthImg = viewModel.depthMapImage {
                        Image(uiImage: depthImg)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(width: geo.size.width, height: geo.size.height)
                            .clipped()
                            .mask(
                                HStack(spacing: 0) {
                                    Spacer()
                                        .frame(width: geo.size.width * viewModel.overlaySlider)
                                    Color.black
                                }
                            )
                    }

                    // Slider line
                    Rectangle()
                        .fill(Color.white)
                        .frame(width: 3)
                        .position(x: geo.size.width * viewModel.overlaySlider, y: geo.size.height / 2)
                        .shadow(radius: 2)

                    // Drag handle
                    Circle()
                        .fill(Color.white)
                        .frame(width: 28, height: 28)
                        .shadow(radius: 3)
                        .overlay(
                            Image(systemName: "arrow.left.and.right")
                                .font(.caption2)
                                .foregroundColor(.gray)
                        )
                        .position(x: geo.size.width * viewModel.overlaySlider, y: geo.size.height / 2)
                        .gesture(
                            DragGesture()
                                .onChanged { value in
                                    let newVal = value.location.x / geo.size.width
                                    viewModel.overlaySlider = max(0, min(1, newVal))
                                }
                        )
                }
            }
            .cornerRadius(12)
            .clipped()

            HStack {
                Text("Original")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                Spacer()
                Text("Depth")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
    }

    // MARK: - Point Measurement Section

    private var pointMeasurementSection: some View {
        VStack(spacing: 12) {
            Text("Tap to Measure Distance")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)

            Text("Tap any point on the depth map to see its estimated distance.")
                .font(.caption)
                .foregroundColor(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)

            ZStack(alignment: .topLeading) {
                if let depthImg = viewModel.depthMapImage {
                    GeometryReader { geo in
                        Image(uiImage: depthImg)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(width: geo.size.width)
                            .cornerRadius(12)
                            .gesture(
                                DragGesture(minimumDistance: 0)
                                    .onEnded { value in
                                        let imgSize = depthImg.size
                                        let viewWidth = geo.size.width
                                        let viewHeight = viewWidth * (imgSize.height / imgSize.width)
                                        let normX = value.location.x / viewWidth
                                        let normY = value.location.y / viewHeight
                                        if normX >= 0 && normX <= 1 && normY >= 0 && normY <= 1 {
                                            viewModel.measureDepth(
                                                atNormalized: CGPoint(x: normX, y: normY),
                                                viewLocation: value.location
                                            )
                                        }
                                    }
                            )

                        // Measurement indicator
                        if let measurement = viewModel.pointMeasurement {
                            VStack(spacing: 2) {
                                Image(systemName: "mappin.circle.fill")
                                    .font(.title2)
                                    .foregroundColor(.white)
                                    .shadow(radius: 3)
                                Text(String(format: "%.2f m", measurement.depth))
                                    .font(.caption)
                                    .fontWeight(.bold)
                                    .foregroundColor(.white)
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 4)
                                    .background(Color.black.opacity(0.75))
                                    .cornerRadius(8)
                            }
                            .position(x: measurement.viewPoint.x,
                                      y: max(40, measurement.viewPoint.y - 30))
                        }
                    }
                    .aspectRatio(depthImg.size.width / depthImg.size.height, contentMode: .fit)
                }
            }
        }
    }

    // MARK: - 3D Parallax Section

    private var parallaxSection: some View {
        VStack(spacing: 12) {
            Text("3D Parallax Effect")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)

            Text("Tilt your device to see the depth-based parallax effect.")
                .font(.caption)
                .foregroundColor(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)

            ZStack {
                if let original = viewModel.selectedImage {
                    Image(uiImage: original)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .cornerRadius(12)
                        .offset(
                            x: CGFloat(motionManager.roll) * -8,
                            y: CGFloat(motionManager.pitch) * -8
                        )
                }

                if let depthImg = viewModel.depthMapImage {
                    Image(uiImage: depthImg)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .cornerRadius(12)
                        .opacity(0.45)
                        .offset(
                            x: CGFloat(motionManager.roll) * 15,
                            y: CGFloat(motionManager.pitch) * 15
                        )
                }
            }
            .clipped()
            .cornerRadius(12)
            .shadow(color: .black.opacity(0.15), radius: 8, y: 4)
        }
    }

    // MARK: - Save Section

    private var saveSection: some View {
        VStack(spacing: 12) {
            Button(action: { viewModel.saveDepthMap() }) {
                HStack {
                    Image(systemName: viewModel.didSave ? "checkmark.circle.fill" : "square.and.arrow.down")
                    Text(viewModel.didSave ? "Saved to Photos" : "Save Depth Map")
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background(viewModel.didSave ? Color.green : Color(.systemGray5))
                .foregroundColor(viewModel.didSave ? .white : .primary)
                .cornerRadius(12)
            }
            .disabled(viewModel.didSave)
        }
    }

    // MARK: - Colormap Legend

    private var colormapLegendSection: some View {
        VStack(spacing: 8) {
            Text("Depth Colormap Legend")
                .font(.caption)
                .foregroundColor(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)

            GeometryReader { geo in
                let width = geo.size.width
                HStack(spacing: 0) {
                    ForEach(0..<Int(width), id: \.self) { i in
                        let t = Float(i) / Float(width)
                        let (r, g, b) = TurboColormap.color(for: t)
                        Color(
                            red: Double(r) / 255.0,
                            green: Double(g) / 255.0,
                            blue: Double(b) / 255.0
                        )
                        .frame(width: 1)
                    }
                }
            }
            .frame(height: 16)
            .cornerRadius(4)

            HStack {
                Text("Far (blue)")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                Spacer()
                Text("Near (red)")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
        .padding(.top, 8)
    }

    // MARK: - Helpers

    private func depthStatView(label: String, value: Float, color: Color) -> some View {
        VStack(spacing: 4) {
            Text(label)
                .font(.caption2)
                .foregroundColor(.secondary)
            Text(String(format: "%.2f m", value))
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundColor(color)
        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - Point Measurement Data

struct PointMeasurement {
    let depth: Float
    let normalizedPoint: CGPoint
    let viewPoint: CGPoint
}

// MARK: - Depth Statistics

struct DepthStats {
    let min: Float
    let max: Float
    let mean: Float
}

// MARK: - DepthPro ViewModel

class DepthProViewModel: ObservableObject {
    @Published var photoItem: PhotosPickerItem? {
        didSet { loadImage() }
    }
    @Published var selectedImage: UIImage?
    @Published var depthMapImage: UIImage?
    @Published var depthValues: [Float] = []
    @Published var depthWidth: Int = 0
    @Published var depthHeight: Int = 0
    @Published var estimatedFocalLength: Float?
    @Published var depthStats: DepthStats?
    @Published var isProcessing = false
    @Published var isProcessed = false
    @Published var progress: Double = 0
    @Published var statusMessage = ""
    @Published var errorMessage: String?
    @Published var overlaySlider: Double = 0.5
    @Published var pointMeasurement: PointMeasurement?
    @Published var didSave = false

    private func loadImage() {
        guard let item = photoItem else { return }
        reset()
        Task {
            do {
                if let data = try await item.loadTransferable(type: Data.self),
                   let uiImage = UIImage(data: data) {
                    await MainActor.run {
                        self.selectedImage = uiImage
                    }
                }
            } catch {
                await MainActor.run {
                    self.errorMessage = "Failed to load image: \(error.localizedDescription)"
                }
            }
        }
    }

    func reset() {
        depthMapImage = nil
        depthValues = []
        depthWidth = 0
        depthHeight = 0
        estimatedFocalLength = nil
        depthStats = nil
        isProcessed = false
        isProcessing = false
        progress = 0
        statusMessage = ""
        errorMessage = nil
        pointMeasurement = nil
        didSave = false
    }

    func estimateDepth() {
        guard selectedImage != nil else { return }
        isProcessing = true
        errorMessage = nil
        progress = 0

        Task {
            do {
                try await performDepthEstimation()
                await MainActor.run {
                    self.isProcessed = true
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

    // MARK: - Core ML Inference

    private func performDepthEstimation() async throws {
        await updateStatus("Loading Depth Pro model...", progress: 0.1)

        guard let modelURL = Bundle.main.url(forResource: "DepthPro", withExtension: "mlmodelc") else {
            throw DepthProError.modelNotFound(
                "DepthPro.mlmodelc not found in bundle. " +
                "Please convert the model using convert_depth_pro.py, " +
                "compile the .mlpackage, and add DepthPro.mlmodelc to the project."
            )
        }

        let config = MLModelConfiguration()
        // ANE compilation fails on this large model. Use CPU+GPU instead.
        config.computeUnits = .cpuAndGPU

        await updateStatus("Loading model (requires 6GB+ RAM)...", progress: 0.2)
        let model = try MLModel(contentsOf: modelURL, configuration: config)

        await updateStatus("Preprocessing image...", progress: 0.3)

        guard let inputImage = selectedImage else {
            throw DepthProError.processingFailed("No image selected.")
        }

        // Depth Pro requires exactly 1536x1536 input (ViT patch architecture constraint).
        // Requires iPhone 15 Pro or later (6GB+ RAM).
        let targetSize = CGSize(width: 1536, height: 1536)
        guard let resizedImage = resizeImage(inputImage, to: targetSize),
              let pixelBuffer = pixelBufferFromImage(resizedImage, size: targetSize) else {
            throw DepthProError.processingFailed("Failed to preprocess image for model input.")
        }

        await updateStatus("Running depth estimation...", progress: 0.5)

        // Model expects MLMultiArray input (1,3,1536,1536), not CVPixelBuffer
        let inputArray = try MLMultiArray(shape: [1, 3, 1536, 1536], dataType: .float16)
        fillMultiArrayFromPixelBuffer(pixelBuffer, into: inputArray, width: 1536, height: 1536)

        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(multiArray: inputArray)
        ])

        let result = try model.prediction(from: inputFeatures)

        await updateStatus("Processing depth output...", progress: 0.8)

        // Output name is "var_4563" (auto-generated during conversion)
        guard let depthMultiArray = result.featureValue(for: "var_4563")?.multiArrayValue else {
            // Fallback: try first available output
            let names = result.featureNames
            guard let firstName = names.first,
                  let depthArray = result.featureValue(for: firstName)?.multiArrayValue else {
                throw DepthProError.processingFailed("Model did not produce a depth output.")
            }
            // Use this fallback path
            let shape2 = depthArray.shape.map { $0.intValue }
            let dH2 = shape2.count >= 3 ? shape2[shape2.count - 2] : 1536
            let dW2 = shape2.count >= 2 ? shape2[shape2.count - 1] : 1536
            let totalPixels2 = dH2 * dW2
            let ptr2 = depthArray.dataPointer.bindMemory(to: Float.self, capacity: totalPixels2)
            var depths2 = [Float](repeating: 0, count: totalPixels2)
            for i in 0..<totalPixels2 { depths2[i] = ptr2[i] }
            var minD2: Float = .greatestFiniteMagnitude, maxD2: Float = -.greatestFiniteMagnitude, sumD2: Float = 0
            for d in depths2 { if d < minD2 { minD2 = d }; if d > maxD2 { maxD2 = d }; sumD2 += d }
            let meanD2 = sumD2 / Float(totalPixels2)
            let depthImage2 = TurboColormap.depthMapImage(from: depths2, width: dW2, height: dH2, minDepth: minD2, maxDepth: maxD2)
            await MainActor.run {
                self.depthMapImage = depthImage2
                self.minDepth = minD2; self.maxDepth = maxD2; self.meanDepth = meanD2
                self.depthWidth = dW2; self.depthHeight = dH2; self.depthValues = depths2
                self.isProcessed = true; self.isProcessing = false
            }
            return
        }

        // Extract focal length if available
        var focalLength: Float? = nil
        if let focalArray = result.featureValue(for: "focallength")?.multiArrayValue {
            focalLength = focalArray[0].floatValue
        }

        // Parse depth map dimensions
        let shape = depthMultiArray.shape.map { $0.intValue }
        let dH: Int
        let dW: Int
        if shape.count == 4 {
            dH = shape[2]
            dW = shape[3]
        } else if shape.count == 3 {
            dH = shape[1]
            dW = shape[2]
        } else if shape.count == 2 {
            dH = shape[0]
            dW = shape[1]
        } else {
            dH = 1536
            dW = 1536
        }

        // Copy depth values (handle both Float32 and Float16 output)
        let totalPixels = dH * dW
        var depths = [Float](repeating: 0, count: totalPixels)
        if depthMultiArray.dataType == .float32 {
            let pointer = depthMultiArray.dataPointer.bindMemory(to: Float.self, capacity: totalPixels)
            for i in 0..<totalPixels { depths[i] = pointer[i] }
        } else {
            // Float16 output
            let fp16Ptr = depthMultiArray.dataPointer.bindMemory(to: UInt16.self, capacity: totalPixels)
            var srcBuffer = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: fp16Ptr), height: 1, width: vImagePixelCount(totalPixels), rowBytes: totalPixels * 2)
            depths.withUnsafeMutableBufferPointer { dstBuf in
                var dstBuffer = vImage_Buffer(data: dstBuf.baseAddress!, height: 1, width: vImagePixelCount(totalPixels), rowBytes: totalPixels * 4)
                vImageConvert_Planar16FtoPlanarF(&srcBuffer, &dstBuffer, 0)
            }
        }

        // Compute statistics
        var minD: Float = .greatestFiniteMagnitude
        var maxD: Float = -.greatestFiniteMagnitude
        var sumD: Float = 0
        for d in depths {
            if d < minD { minD = d }
            if d > maxD { maxD = d }
            sumD += d
        }
        let meanD = sumD / Float(totalPixels)

        // Generate colorized depth image
        let depthImage = TurboColormap.depthMapImage(from: depths, width: dW, height: dH, minDepth: minD, maxDepth: maxD)

        await updateStatus("Complete!", progress: 1.0)

        await MainActor.run {
            self.depthValues = depths
            self.depthWidth = dW
            self.depthHeight = dH
            self.depthMapImage = depthImage
            self.estimatedFocalLength = focalLength
            self.depthStats = DepthStats(min: minD, max: maxD, mean: meanD)
        }
    }

    // MARK: - Measure Depth at Point

    func measureDepth(atNormalized point: CGPoint, viewLocation: CGPoint) {
        guard !depthValues.isEmpty, depthWidth > 0, depthHeight > 0 else { return }

        let px = Int(point.x * CGFloat(depthWidth))
        let py = Int(point.y * CGFloat(depthHeight))
        let clampedX = max(0, min(depthWidth - 1, px))
        let clampedY = max(0, min(depthHeight - 1, py))
        let index = clampedY * depthWidth + clampedX

        guard index >= 0 && index < depthValues.count else { return }
        let depth = depthValues[index]

        pointMeasurement = PointMeasurement(
            depth: depth,
            normalizedPoint: point,
            viewPoint: viewLocation
        )
    }

    // MARK: - Save Depth Map

    func saveDepthMap() {
        guard let image = depthMapImage else { return }
        UIImageWriteToSavedPhotosAlbum(image, nil, nil, nil)
        didSave = true
    }

    // MARK: - Image Utilities

    private func resizeImage(_ image: UIImage, to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, true, 1.0)
        image.draw(in: CGRect(origin: .zero, size: size))
        let resized = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return resized
    }

    private func pixelBufferFromImage(_ image: UIImage, size: CGSize) -> CVPixelBuffer? {
        let width = Int(size.width)
        let height = Int(size.height)
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ]
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, width, height,
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
            bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
        ) else { return nil }

        guard let cgImage = image.cgImage else { return nil }
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        return buffer
    }

    // MARK: - PixelBuffer to MLMultiArray

    private func fillMultiArrayFromPixelBuffer(_ buffer: CVPixelBuffer, into array: MLMultiArray, width: Int, height: Int) {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(buffer) else { return }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        let ptr = baseAddress.assumingMemoryBound(to: UInt8.self)

        // BGRA → RGB normalized to [0,1], stored as Float16
        let fp16Ptr = array.dataPointer.bindMemory(to: UInt16.self, capacity: 3 * width * height)
        for y in 0..<height {
            for x in 0..<width {
                let offset = y * bytesPerRow + x * 4
                let b = Float(ptr[offset]) / 255.0
                let g = Float(ptr[offset + 1]) / 255.0
                let r = Float(ptr[offset + 2]) / 255.0
                // Channel-first layout: [1, 3, H, W]
                let idx = y * width + x
                fp16Ptr[0 * height * width + idx] = float32ToFloat16(r)
                fp16Ptr[1 * height * width + idx] = float32ToFloat16(g)
                fp16Ptr[2 * height * width + idx] = float32ToFloat16(b)
            }
        }
    }

    private func float32ToFloat16(_ value: Float) -> UInt16 {
        var f = value
        var h: UInt16 = 0
        withUnsafePointer(to: &f) { src in
            withUnsafeMutablePointer(to: &h) { dst in
                var bufferFloat32 = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src), height: 1, width: 1, rowBytes: 4)
                var bufferFloat16 = vImage_Buffer(data: UnsafeMutableRawPointer(dst), height: 1, width: 1, rowBytes: 2)
                vImageConvert_PlanarFtoPlanar16F(&bufferFloat32, &bufferFloat16, 0)
            }
        }
        return h
    }

    // MARK: - Status Updates

    @MainActor
    private func updateStatus(_ message: String, progress: Double) {
        self.statusMessage = message
        self.progress = progress
    }
}

// MARK: - Errors

enum DepthProError: LocalizedError {
    case modelNotFound(String)
    case processingFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelNotFound(let msg): return msg
        case .processingFailed(let msg): return msg
        }
    }
}

// MARK: - Preview

#Preview {
    ContentView()
}
