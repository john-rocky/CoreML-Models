import SwiftUI
import UIKit
import AVFoundation
import CoreML
import Vision
import Accelerate

// MARK: - Segmentation Classes

struct SegmentationClass {
    let name: String
    let color: SIMD4<UInt8> // RGBA
}

let segmentationClasses: [SegmentationClass] = [
    SegmentationClass(name: "Background",    color: SIMD4(0, 0, 0, 0)),
    SegmentationClass(name: "Aeroplane",     color: SIMD4(128, 0, 0, 180)),
    SegmentationClass(name: "Bicycle",       color: SIMD4(0, 128, 0, 180)),
    SegmentationClass(name: "Bird",          color: SIMD4(128, 128, 0, 180)),
    SegmentationClass(name: "Boat",          color: SIMD4(0, 0, 128, 180)),
    SegmentationClass(name: "Bottle",        color: SIMD4(128, 0, 128, 180)),
    SegmentationClass(name: "Bus",           color: SIMD4(0, 128, 128, 180)),
    SegmentationClass(name: "Car",           color: SIMD4(128, 128, 128, 180)),
    SegmentationClass(name: "Cat",           color: SIMD4(64, 0, 0, 180)),
    SegmentationClass(name: "Chair",         color: SIMD4(192, 0, 0, 180)),
    SegmentationClass(name: "Cow",           color: SIMD4(64, 128, 0, 180)),
    SegmentationClass(name: "Dining Table",  color: SIMD4(192, 128, 0, 180)),
    SegmentationClass(name: "Dog",           color: SIMD4(64, 0, 128, 180)),
    SegmentationClass(name: "Horse",         color: SIMD4(192, 0, 128, 180)),
    SegmentationClass(name: "Motorbike",     color: SIMD4(64, 128, 128, 180)),
    SegmentationClass(name: "Person",        color: SIMD4(192, 128, 128, 180)),
    SegmentationClass(name: "Potted Plant",  color: SIMD4(0, 64, 0, 180)),
    SegmentationClass(name: "Sheep",         color: SIMD4(128, 64, 0, 180)),
    SegmentationClass(name: "Sofa",          color: SIMD4(0, 192, 0, 180)),
    SegmentationClass(name: "Train",         color: SIMD4(128, 192, 0, 180)),
    SegmentationClass(name: "TV/Monitor",    color: SIMD4(0, 64, 128, 180))
]

// MARK: - Camera Manager

class CameraManager: NSObject, ObservableObject {
    let session = AVCaptureSession()
    var onFrame: ((CMSampleBuffer) -> Void)?

    private let sessionQueue = DispatchQueue(label: "camera.session")

    func configure() {
        sessionQueue.async { [weak self] in
            self?.setupSession()
        }
    }

    private func setupSession() {
        session.beginConfiguration()
        session.sessionPreset = .high

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device) else {
            session.commitConfiguration()
            return
        }

        if session.canAddInput(input) {
            session.addInput(input)
        }

        let output = AVCaptureVideoDataOutput()
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "camera.frame"))
        output.alwaysDiscardsLateVideoFrames = true

        if session.canAddOutput(output) {
            session.addOutput(output)
        }

        session.commitConfiguration()
        session.startRunning()
    }

    func stop() {
        sessionQueue.async { [weak self] in
            self?.session.stopRunning()
        }
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        onFrame?(sampleBuffer)
    }
}

// MARK: - Camera Preview

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: .zero)
        let previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)
        context.coordinator.previewLayer = previewLayer
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {
        context.coordinator.previewLayer?.frame = uiView.bounds
    }

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    class Coordinator {
        var previewLayer: AVCaptureVideoPreviewLayer?
    }
}

// MARK: - Segmentation Engine

class SegmentationEngine: ObservableObject {
    @Published var overlayImage: UIImage?
    @Published var detectedClasses: [String] = []
    @Published var errorMessage: String?

    private var vnModel: VNCoreMLModel?
    private var isProcessing = false

    init() {
        loadModel()
    }

    private func loadModel() {
        // PLACEHOLDER: Add DeepLabV3MobileNetV3.mlpackage to the Xcode project.
        // The compiled .mlmodelc will be bundled automatically.
        // Download from the CoreML-Models repository and drag into Xcode.

        guard let modelURL = Bundle.main.url(forResource: "DeepLabV3MobileNetV3", withExtension: "mlmodelc") else {
            DispatchQueue.main.async {
                self.errorMessage = "Model not found. Please add DeepLabV3MobileNetV3.mlpackage to the Xcode project."
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

    func segment(sampleBuffer: CMSampleBuffer) {
        guard !isProcessing, let vnModel = vnModel else { return }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        isProcessing = true

        let request = VNCoreMLRequest(model: vnModel) { [weak self] request, error in
            defer { self?.isProcessing = false }

            if let results = request.results as? [VNCoreMLFeatureValueObservation],
               let multiArray = results.first?.featureValue.multiArrayValue {
                self?.processSegmentation(multiArray: multiArray)
            }
        }
        request.imageCropAndScaleOption = .scaleFill

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right, options: [:])
        try? handler.perform([request])
    }

    private func processSegmentation(multiArray: MLMultiArray) {
        // Output shape: 1 x 21 x 512 x 512
        let numClasses = 21
        let height = 512
        let width = 512

        let pointer = multiArray.dataPointer.bindMemory(to: Float.self, capacity: multiArray.count)

        // For each pixel, find the class with highest score (argmax across 21 classes)
        var pixelData = [UInt8](repeating: 0, count: width * height * 4) // RGBA
        var foundClasses = Set<Int>()

        for y in 0..<height {
            for x in 0..<width {
                var maxVal: Float = -Float.infinity
                var maxClass = 0

                for c in 0..<numClasses {
                    let index = c * height * width + y * width + x
                    let val = pointer[index]
                    if val > maxVal {
                        maxVal = val
                        maxClass = c
                    }
                }

                if maxClass != 0 {
                    foundClasses.insert(maxClass)
                }

                let color = segmentationClasses[maxClass].color
                let pixelIndex = (y * width + x) * 4
                pixelData[pixelIndex]     = color.x  // R
                pixelData[pixelIndex + 1] = color.y  // G
                pixelData[pixelIndex + 2] = color.z  // B
                pixelData[pixelIndex + 3] = color.w  // A
            }
        }

        // Create UIImage from pixel data
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)

        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ), let cgImage = context.makeImage() else { return }

        let image = UIImage(cgImage: cgImage)
        let classes = foundClasses.sorted().map { segmentationClasses[$0].name }

        DispatchQueue.main.async {
            self.overlayImage = image
            self.detectedClasses = classes
        }
    }
}

// MARK: - Content View

struct ContentView: View {
    @StateObject private var camera = CameraManager()
    @StateObject private var segEngine = SegmentationEngine()
    @State private var showLegend = false

    var body: some View {
        ZStack {
            // Camera feed
            CameraPreview(session: camera.session)
                .ignoresSafeArea()

            // Segmentation overlay
            if let overlay = segEngine.overlayImage {
                Image(uiImage: overlay)
                    .resizable()
                    .scaledToFill()
                    .ignoresSafeArea()
                    .allowsHitTesting(false)
            }

            VStack {
                // Top bar with title and legend toggle
                HStack {
                    Text("DeepLabV3 Segmentation")
                        .font(.headline)
                        .foregroundColor(.white)
                        .shadow(radius: 2)

                    Spacer()

                    Button(action: { showLegend.toggle() }) {
                        Image(systemName: "list.bullet")
                            .font(.title3)
                            .foregroundColor(.white)
                            .padding(8)
                            .background(.black.opacity(0.5), in: Circle())
                    }
                }
                .padding()

                Spacer()

                // Error message
                if let error = segEngine.errorMessage {
                    VStack(spacing: 8) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.largeTitle)
                            .foregroundColor(.yellow)
                        Text(error)
                            .font(.caption)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)
                    }
                    .padding()
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
                    .padding()
                }

                // Detected classes
                if !segEngine.detectedClasses.isEmpty {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 8) {
                            ForEach(segEngine.detectedClasses, id: \.self) { className in
                                Text(className)
                                    .font(.caption)
                                    .fontWeight(.medium)
                                    .padding(.horizontal, 10)
                                    .padding(.vertical, 4)
                                    .background(.black.opacity(0.6))
                                    .foregroundColor(.white)
                                    .cornerRadius(12)
                            }
                        }
                        .padding(.horizontal)
                    }
                    .padding(.bottom, 8)
                }
            }

            // Legend sheet
            if showLegend {
                VStack {
                    HStack {
                        Text("Class Legend")
                            .font(.headline)
                        Spacer()
                        Button("Done") { showLegend = false }
                    }
                    .padding()

                    ScrollView {
                        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 8) {
                            ForEach(1..<segmentationClasses.count, id: \.self) { i in
                                HStack(spacing: 8) {
                                    let sc = segmentationClasses[i]
                                    RoundedRectangle(cornerRadius: 4)
                                        .fill(Color(
                                            red: Double(sc.color.x) / 255,
                                            green: Double(sc.color.y) / 255,
                                            blue: Double(sc.color.z) / 255
                                        ))
                                        .frame(width: 20, height: 20)
                                    Text(sc.name)
                                        .font(.caption)
                                    Spacer()
                                }
                            }
                        }
                        .padding(.horizontal)
                    }
                }
                .background(.ultraThinMaterial)
                .cornerRadius(16)
                .padding()
            }
        }
        .onAppear {
            camera.onFrame = { [weak segEngine] buffer in
                segEngine?.segment(sampleBuffer: buffer)
            }
            camera.configure()
        }
        .onDisappear {
            camera.stop()
        }
    }
}

#Preview {
    ContentView()
}
