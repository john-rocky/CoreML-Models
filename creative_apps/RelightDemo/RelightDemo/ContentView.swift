import SwiftUI
import UIKit
import CoreML
import PhotosUI

// MARK: - Spherical Harmonics Lighting Presets

/// Preset SH lighting coefficients (9 coefficients for 2nd order SH)
struct SHLightingPreset: Identifiable, Equatable {
    let id = UUID()
    let name: String
    let icon: String
    let coefficients: [Float] // 9 SH coefficients

    static func == (lhs: SHLightingPreset, rhs: SHLightingPreset) -> Bool {
        lhs.id == rhs.id
    }

    /// Preset lighting directions using 2nd-order Spherical Harmonics
    /// SH basis: [Y00, Y1-1, Y10, Y11, Y2-2, Y2-1, Y20, Y21, Y22]
    static let front = SHLightingPreset(
        name: "Front",
        icon: "sun.max.fill",
        coefficients: [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )

    static let left = SHLightingPreset(
        name: "Left",
        icon: "arrow.left.circle.fill",
        coefficients: [0.5, 0.0, 0.0, -0.6, 0.0, 0.0, 0.0, 0.0, 0.3]
    )

    static let right = SHLightingPreset(
        name: "Right",
        icon: "arrow.right.circle.fill",
        coefficients: [0.5, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.3]
    )

    static let top = SHLightingPreset(
        name: "Top",
        icon: "arrow.up.circle.fill",
        coefficients: [0.5, 0.0, 0.6, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0]
    )

    static let bottom = SHLightingPreset(
        name: "Bottom",
        icon: "arrow.down.circle.fill",
        coefficients: [0.5, 0.0, -0.6, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0]
    )

    static let allPresets: [SHLightingPreset] = [front, left, right, top, bottom]
}

// MARK: - Relighting Processor

/// Processes portrait images through the DPR Relighting CoreML model
class RelightProcessor: ObservableObject {
    @Published var inputImage: UIImage?
    @Published var luminanceImage: UIImage?
    @Published var relitImage: UIImage?
    @Published var isProcessing = false
    @Published var errorMessage: String?
    @Published var selectedPreset: SHLightingPreset = .front
    @Published var customSH: [Float] = [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    private var model: MLModel?
    private let inputSize = 512

    init() {
        loadModel()
    }

    private func loadModel() {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all

            guard let modelURL = Bundle.main.url(forResource: "DPR_Relighting", withExtension: "mlmodelc") else {
                errorMessage = "Model not found. Please add DPR_Relighting.mlmodelc to the project bundle."
                return
            }
            model = try MLModel(contentsOf: modelURL, configuration: config)
        } catch {
            errorMessage = "Failed to load model: \(error.localizedDescription)"
        }
    }

    /// Convert color image to grayscale luminance
    private func convertToLuminance(_ image: UIImage) -> (UIImage?, [Float]?) {
        guard let cgImage = image.cgImage else { return (nil, nil) }

        let width = inputSize
        let height = inputSize
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var pixelData = [UInt8](repeating: 0, count: width * height * 4)

        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else { return (nil, nil) }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        // Compute luminance: Y = 0.299*R + 0.587*G + 0.114*B
        var luminanceData = [Float](repeating: 0, count: width * height)
        var grayPixels = [UInt8](repeating: 0, count: width * height * 4)

        for y in 0..<height {
            for x in 0..<width {
                let pixelIndex = (y * width + x) * 4
                let spatialIndex = y * width + x
                let r = Float(pixelData[pixelIndex]) / 255.0
                let g = Float(pixelData[pixelIndex + 1]) / 255.0
                let b = Float(pixelData[pixelIndex + 2]) / 255.0
                let lum = 0.299 * r + 0.587 * g + 0.114 * b
                luminanceData[spatialIndex] = lum

                let lumByte = UInt8(max(0, min(255, lum * 255.0)))
                grayPixels[spatialIndex * 4] = lumByte
                grayPixels[spatialIndex * 4 + 1] = lumByte
                grayPixels[spatialIndex * 4 + 2] = lumByte
                grayPixels[spatialIndex * 4 + 3] = 255
            }
        }

        // Create grayscale UIImage for display
        guard let grayContext = CGContext(
            data: &grayPixels,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ), let grayCGImage = grayContext.makeImage() else { return (nil, luminanceData) }

        return (UIImage(cgImage: grayCGImage), luminanceData)
    }

    /// Convert single-channel float array to grayscale UIImage
    private func luminanceToImage(_ data: [Float], width: Int, height: Int) -> UIImage? {
        var pixelData = [UInt8](repeating: 255, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let spatialIndex = y * width + x
                let pixelIndex = spatialIndex * 4
                let val = UInt8(max(0, min(255, data[spatialIndex] * 255.0)))
                pixelData[pixelIndex] = val
                pixelData[pixelIndex + 1] = val
                pixelData[pixelIndex + 2] = val
                pixelData[pixelIndex + 3] = 255
            }
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ), let cgImage = context.makeImage() else { return nil }

        return UIImage(cgImage: cgImage)
    }

    /// Run relighting with the given SH coefficients
    func relight(image: UIImage, shCoefficients: [Float]) async {
        guard let model = model else {
            await MainActor.run { errorMessage = "Model is not loaded." }
            return
        }

        await MainActor.run {
            inputImage = image
            relitImage = nil
            isProcessing = true
            errorMessage = nil
        }

        do {
            // Convert to luminance
            let (grayImage, luminanceData) = convertToLuminance(image)
            await MainActor.run { luminanceImage = grayImage }

            guard let luminanceData = luminanceData else {
                await MainActor.run {
                    errorMessage = "Failed to compute luminance."
                    isProcessing = false
                }
                return
            }

            // Create luminance input: (1, 1, 512, 512)
            let lumArray = try MLMultiArray(shape: [1, 1, 512, 512] as [NSNumber], dataType: .float32)
            let lumPtr = lumArray.dataPointer.bindMemory(to: Float.self, capacity: inputSize * inputSize)
            for i in 0..<(inputSize * inputSize) {
                lumPtr[i] = luminanceData[i]
            }

            // Create SH lighting input: (1, 9, 1, 1)
            let shArray = try MLMultiArray(shape: [1, 9, 1, 1] as [NSNumber], dataType: .float32)
            let shPtr = shArray.dataPointer.bindMemory(to: Float.self, capacity: 9)
            for i in 0..<9 {
                shPtr[i] = i < shCoefficients.count ? shCoefficients[i] : 0.0
            }

            let features = try MLDictionaryFeatureProvider(dictionary: [
                "luminance_image": MLFeatureValue(multiArray: lumArray),
                "target_sh_lighting": MLFeatureValue(multiArray: shArray)
            ])

            let output = try model.prediction(from: features)

            // Try common output names
            let outputNames = ["relit_image", "output", "relit"]
            var outputArray: MLMultiArray?
            for name in outputNames {
                if let arr = output.featureValue(for: name)?.multiArrayValue {
                    outputArray = arr
                    break
                }
            }

            // Fallback: try first feature
            if outputArray == nil {
                let featureNames = output.featureNames
                for name in featureNames {
                    if let arr = output.featureValue(for: name)?.multiArrayValue {
                        outputArray = arr
                        break
                    }
                }
            }

            guard let resultArray = outputArray else {
                await MainActor.run {
                    errorMessage = "Unexpected model output format."
                    isProcessing = false
                }
                return
            }

            let outputSize = inputSize * inputSize
            var outputData = [Float](repeating: 0, count: outputSize)
            let outPtr = resultArray.dataPointer.bindMemory(to: Float.self, capacity: outputSize)
            for i in 0..<outputSize {
                outputData[i] = outPtr[i]
            }

            let resultImage = luminanceToImage(outputData, width: inputSize, height: inputSize)

            await MainActor.run {
                relitImage = resultImage
                isProcessing = false
            }
        } catch {
            await MainActor.run {
                errorMessage = "Inference error: \(error.localizedDescription)"
                isProcessing = false
            }
        }
    }
}

// MARK: - Light Direction Sphere View

/// Interactive sphere for dragging light direction
struct LightDirectionSphere: View {
    @Binding var shCoefficients: [Float]
    @State private var lightPosition: CGPoint = CGPoint(x: 0.5, y: 0.5)

    var body: some View {
        GeometryReader { geometry in
            let size = min(geometry.size.width, geometry.size.height)
            let center = CGPoint(x: size / 2, y: size / 2)
            let radius = size / 2 - 10

            ZStack {
                // Sphere background
                Circle()
                    .fill(
                        RadialGradient(
                            gradient: Gradient(colors: [Color(.systemGray4), Color(.systemGray6)]),
                            center: UnitPoint(
                                x: lightPosition.x,
                                y: lightPosition.y
                            ),
                            startRadius: 0,
                            endRadius: radius
                        )
                    )
                    .frame(width: size - 20, height: size - 20)
                    .overlay(
                        Circle()
                            .stroke(Color(.systemGray2), lineWidth: 2)
                    )

                // Crosshair
                Path { path in
                    path.move(to: CGPoint(x: center.x - radius, y: center.y))
                    path.addLine(to: CGPoint(x: center.x + radius, y: center.y))
                }
                .stroke(Color(.systemGray3), lineWidth: 0.5)

                Path { path in
                    path.move(to: CGPoint(x: center.x, y: center.y - radius))
                    path.addLine(to: CGPoint(x: center.x, y: center.y + radius))
                }
                .stroke(Color(.systemGray3), lineWidth: 0.5)

                // Light indicator
                Circle()
                    .fill(Color.yellow)
                    .frame(width: 24, height: 24)
                    .shadow(color: .yellow.opacity(0.6), radius: 8)
                    .position(
                        x: center.x + (lightPosition.x - 0.5) * 2 * radius,
                        y: center.y + (lightPosition.y - 0.5) * 2 * radius
                    )
            }
            .frame(width: size, height: size)
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        // Constrain to sphere
                        let dx = (value.location.x - center.x) / radius
                        let dy = (value.location.y - center.y) / radius
                        let dist = sqrt(dx * dx + dy * dy)
                        let clampedDist = min(dist, 1.0)
                        let angle = atan2(dy, dx)
                        let clampedX = cos(angle) * clampedDist
                        let clampedY = sin(angle) * clampedDist

                        lightPosition = CGPoint(
                            x: 0.5 + Double(clampedX) / 2,
                            y: 0.5 + Double(clampedY) / 2
                        )

                        // Convert position to SH coefficients
                        updateSHFromPosition(x: Float(clampedX), y: Float(clampedY))
                    }
            )
        }
        .aspectRatio(1, contentMode: .fit)
    }

    /// Convert 2D position to approximate SH coefficients
    private func updateSHFromPosition(x: Float, y: Float) {
        let z = sqrt(max(0, 1.0 - x * x - y * y))

        // 2nd-order SH from direction (x, -y, z) mapped to light direction
        // Y00 = ambient
        // Y1-1 = y direction, Y10 = z direction, Y11 = x direction
        shCoefficients = [
            0.5,                          // Y00: ambient
            -y * 0.5,                     // Y1-1: vertical
            z * 0.4,                      // Y10: depth
            x * 0.5,                      // Y11: horizontal
            x * (-y) * 0.3,              // Y2-2
            (-y) * z * 0.3,              // Y2-1
            (3 * z * z - 1) * 0.1,       // Y20
            x * z * 0.3,                 // Y21
            (x * x - y * y) * 0.15       // Y22
        ]
    }
}

// MARK: - Image Picker

struct ImagePicker: UIViewControllerRepresentable {
    @Binding var image: UIImage?

    func makeUIViewController(context: Context) -> PHPickerViewController {
        var config = PHPickerConfiguration()
        config.filter = .images
        config.selectionLimit = 1
        let picker = PHPickerViewController(configuration: config)
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: PHPickerViewController, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    class Coordinator: NSObject, PHPickerViewControllerDelegate {
        let parent: ImagePicker

        init(_ parent: ImagePicker) {
            self.parent = parent
        }

        func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
            picker.dismiss(animated: true)
            guard let provider = results.first?.itemProvider,
                  provider.canLoadObject(ofClass: UIImage.self) else { return }
            provider.loadObject(ofClass: UIImage.self) { image, _ in
                DispatchQueue.main.async {
                    self.parent.image = image as? UIImage
                }
            }
        }
    }
}

// MARK: - Content View

struct ContentView: View {
    @StateObject private var processor = RelightProcessor()
    @State private var showImagePicker = false
    @State private var selectedImage: UIImage?
    @State private var useCustomLighting = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    // Header
                    headerSection

                    // Error display
                    if let error = processor.errorMessage {
                        errorBanner(error)
                    }

                    // Pick image button
                    Button {
                        showImagePicker = true
                    } label: {
                        Label("Select Portrait Photo", systemImage: "person.crop.rectangle")
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.orange)
                            .foregroundColor(.white)
                            .cornerRadius(12)
                    }
                    .padding(.horizontal)

                    // Lighting controls
                    lightingControlSection

                    // Apply button
                    if processor.inputImage != nil && !processor.isProcessing {
                        Button {
                            applyRelighting()
                        } label: {
                            Label("Apply Relighting", systemImage: "light.max")
                                .font(.headline)
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.yellow)
                                .foregroundColor(.black)
                                .cornerRadius(12)
                        }
                        .padding(.horizontal)
                    }

                    // Processing indicator
                    if processor.isProcessing {
                        ProgressView("Relighting portrait...")
                            .padding()
                    }

                    // Results
                    if processor.inputImage != nil || processor.relitImage != nil {
                        resultsSection
                    }

                    // SH coefficient display
                    shCoefficientDisplay

                    Spacer(minLength: 40)
                }
                .padding(.vertical)
            }
            .navigationTitle("Portrait Relight")
            .sheet(isPresented: $showImagePicker) {
                ImagePicker(image: $selectedImage)
            }
            .onChange(of: selectedImage) { newValue in
                if let image = newValue {
                    processor.inputImage = image
                }
            }
        }
    }

    // MARK: - Subviews

    private var headerSection: some View {
        VStack(spacing: 8) {
            Image(systemName: "light.beacon.max")
                .font(.system(size: 50))
                .foregroundColor(.orange)
            Text("Portrait Relighting")
                .font(.title2.bold())
            Text("Change lighting direction on portraits using DPR model with Spherical Harmonics")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding()
    }

    private func errorBanner(_ message: String) -> some View {
        HStack {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundColor(.yellow)
            Text(message)
                .font(.caption)
        }
        .padding()
        .background(Color.red.opacity(0.1))
        .cornerRadius(8)
        .padding(.horizontal)
    }

    private var lightingControlSection: some View {
        VStack(spacing: 16) {
            // Toggle between presets and custom
            Picker("Lighting Mode", selection: $useCustomLighting) {
                Text("Presets").tag(false)
                Text("Custom").tag(true)
            }
            .pickerStyle(.segmented)
            .padding(.horizontal)

            if useCustomLighting {
                // Interactive sphere
                VStack(spacing: 8) {
                    Text("Drag to Set Light Direction")
                        .font(.subheadline.bold())
                    LightDirectionSphere(shCoefficients: $processor.customSH)
                        .frame(height: 200)
                        .padding(.horizontal, 60)
                }
            } else {
                // Preset buttons
                VStack(spacing: 8) {
                    Text("Lighting Presets")
                        .font(.subheadline.bold())

                    HStack(spacing: 12) {
                        ForEach(SHLightingPreset.allPresets) { preset in
                            Button {
                                processor.selectedPreset = preset
                                processor.customSH = preset.coefficients
                            } label: {
                                VStack(spacing: 4) {
                                    Image(systemName: preset.icon)
                                        .font(.title2)
                                    Text(preset.name)
                                        .font(.caption2)
                                }
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 12)
                                .background(
                                    processor.selectedPreset == preset
                                        ? Color.orange.opacity(0.2)
                                        : Color(.systemGray6)
                                )
                                .cornerRadius(10)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 10)
                                        .stroke(
                                            processor.selectedPreset == preset
                                                ? Color.orange
                                                : Color.clear,
                                            lineWidth: 2
                                        )
                                )
                            }
                            .foregroundColor(.primary)
                        }
                    }
                    .padding(.horizontal)
                }
            }
        }
    }

    private var resultsSection: some View {
        VStack(spacing: 16) {
            // Original vs Relit comparison
            HStack(spacing: 12) {
                // Original
                VStack(spacing: 4) {
                    Text("Original")
                        .font(.caption.bold())
                        .foregroundColor(.secondary)
                    if let image = processor.inputImage {
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFit()
                            .cornerRadius(8)
                    }
                }

                // Relit
                VStack(spacing: 4) {
                    Text("Relit")
                        .font(.caption.bold())
                        .foregroundColor(.secondary)
                    if let image = processor.relitImage {
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFit()
                            .cornerRadius(8)
                    } else {
                        Rectangle()
                            .fill(Color(.systemGray5))
                            .overlay(
                                Text("Run relighting")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            )
                            .cornerRadius(8)
                    }
                }
            }
            .padding(.horizontal)

            // Luminance intermediate
            if let lumImage = processor.luminanceImage {
                VStack(spacing: 4) {
                    Text("Luminance Input (512x512)")
                        .font(.caption.bold())
                        .foregroundColor(.secondary)
                    Image(uiImage: lumImage)
                        .resizable()
                        .scaledToFit()
                        .frame(maxHeight: 200)
                        .cornerRadius(8)
                }
                .padding(.horizontal)
            }
        }
    }

    private var shCoefficientDisplay: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("SH Coefficients")
                .font(.headline)

            let labels = ["Y00 (ambient)", "Y1-1 (vertical)", "Y10 (depth)", "Y11 (horizontal)",
                          "Y2-2", "Y2-1", "Y20", "Y21", "Y22"]

            ForEach(0..<9, id: \.self) { i in
                HStack {
                    Text(labels[i])
                        .font(.system(size: 10, design: .monospaced))
                        .frame(width: 110, alignment: .leading)
                    Slider(
                        value: Binding(
                            get: { Double(processor.customSH[i]) },
                            set: { processor.customSH[i] = Float($0) }
                        ),
                        in: -1.0...1.0
                    )
                    .tint(.orange)
                    Text(String(format: "%.2f", processor.customSH[i]))
                        .font(.system(size: 10, design: .monospaced))
                        .frame(width: 40, alignment: .trailing)
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
        .padding(.horizontal)
    }

    // MARK: - Actions

    private func applyRelighting() {
        guard let image = processor.inputImage else { return }
        let sh = useCustomLighting ? processor.customSH : processor.selectedPreset.coefficients
        Task {
            await processor.relight(image: image, shCoefficients: sh)
        }
    }
}

#Preview {
    ContentView()
}
