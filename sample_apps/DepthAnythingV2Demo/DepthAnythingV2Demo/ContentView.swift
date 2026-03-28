import SwiftUI
import UIKit
import AVFoundation
import CoreML
import Vision
import Accelerate

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

// MARK: - Depth Estimator

class DepthEstimator: ObservableObject {
    @Published var depthImage: UIImage?
    @Published var errorMessage: String?
    @Published var minDepth: Float = 0
    @Published var maxDepth: Float = 0

    private var vnModel: VNCoreMLModel?
    private var isProcessing = false

    /// Width and height of the model output depth map.
    private let depthSize = 518

    init() {
        loadModel()
    }

    private func loadModel() {
        // PLACEHOLDER: Add DepthAnythingV2Small.mlpackage to the Xcode project.
        // The compiled .mlmodelc will be bundled automatically.
        // Download from the CoreML-Models repository and drag into Xcode.

        guard let modelURL = Bundle.main.url(forResource: "DepthAnythingV2Small", withExtension: "mlmodelc") else {
            DispatchQueue.main.async {
                self.errorMessage = "Model not found. Please add DepthAnythingV2Small.mlpackage to the Xcode project."
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

    func estimateDepth(sampleBuffer: CMSampleBuffer) {
        guard !isProcessing, let vnModel = vnModel else { return }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        isProcessing = true

        let request = VNCoreMLRequest(model: vnModel) { [weak self] request, error in
            defer { self?.isProcessing = false }

            guard let self = self else { return }

            if let results = request.results as? [VNCoreMLFeatureValueObservation],
               let multiArray = results.first?.featureValue.multiArrayValue {
                self.processDepthOutput(multiArray: multiArray)
            }
        }
        request.imageCropAndScaleOption = .scaleFill

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right, options: [:])
        try? handler.perform([request])
    }

    private func processDepthOutput(multiArray: MLMultiArray) {
        let count = multiArray.count
        let size = depthSize

        // Extract raw depth values
        var depths = [Float](repeating: 0, count: count)
        let ptr = multiArray.dataPointer.bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            depths[i] = ptr[i]
        }

        // Find min and max for normalization
        var minVal: Float = Float.greatestFiniteMagnitude
        var maxVal: Float = -Float.greatestFiniteMagnitude
        vDSP_minv(depths, 1, &minVal, vDSP_Length(count))
        vDSP_maxv(depths, 1, &maxVal, vDSP_Length(count))

        let range = maxVal - minVal
        guard range > 0 else {
            return
        }

        // Create RGBA pixel data with a color gradient
        var pixelData = [UInt8](repeating: 255, count: size * size * 4)

        for i in 0..<count {
            let normalized = (depths[i] - minVal) / range  // 0.0 (near) to 1.0 (far)
            let (r, g, b) = depthToColor(value: normalized)
            let offset = i * 4
            pixelData[offset]     = r
            pixelData[offset + 1] = g
            pixelData[offset + 2] = b
            pixelData[offset + 3] = 200  // Semi-transparent for overlay mode
        }

        // Create CGImage from pixel data
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)

        guard let providerRef = CGDataProvider(data: Data(pixelData) as CFData),
              let cgImage = CGImage(
                width: size,
                height: size,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: size * 4,
                space: colorSpace,
                bitmapInfo: bitmapInfo,
                provider: providerRef,
                decode: nil,
                shouldInterpolate: true,
                intent: .defaultIntent
              ) else {
            return
        }

        let image = UIImage(cgImage: cgImage)

        DispatchQueue.main.async {
            self.depthImage = image
            self.minDepth = minVal
            self.maxDepth = maxVal
        }
    }

    /// Maps a normalized depth value (0 = near, 1 = far) to a color.
    /// Near objects are warm (red/yellow), far objects are cool (blue/cyan).
    private func depthToColor(value: Float) -> (UInt8, UInt8, UInt8) {
        // Turbo-inspired colormap: near = warm (red), far = cool (blue)
        let t = max(0, min(1, value))

        let r: Float
        let g: Float
        let b: Float

        if t < 0.25 {
            // Red -> Yellow
            let s = t / 0.25
            r = 1.0
            g = s
            b = 0.0
        } else if t < 0.5 {
            // Yellow -> Green
            let s = (t - 0.25) / 0.25
            r = 1.0 - s
            g = 1.0
            b = 0.0
        } else if t < 0.75 {
            // Green -> Cyan
            let s = (t - 0.5) / 0.25
            r = 0.0
            g = 1.0
            b = s
        } else {
            // Cyan -> Blue
            let s = (t - 0.75) / 0.25
            r = 0.0
            g = 1.0 - s
            b = 1.0
        }

        return (
            UInt8(r * 255),
            UInt8(g * 255),
            UInt8(b * 255)
        )
    }

    /// Creates a fully opaque version of the depth map for full-screen display.
    func opaqueDepthImage() -> UIImage? {
        guard let cgImage = depthImage?.cgImage else { return nil }
        let width = cgImage.width
        let height = cgImage.height

        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }

        // Draw black background then the image on top
        context.setFillColor(UIColor.black.cgColor)
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        guard let result = context.makeImage() else { return nil }
        return UIImage(cgImage: result)
    }
}

// MARK: - Content View

struct ContentView: View {
    @StateObject private var camera = CameraManager()
    @StateObject private var depthEstimator = DepthEstimator()
    @State private var showFullDepthMap = false

    var body: some View {
        ZStack {
            // Camera feed (hidden when full depth map is shown)
            if !showFullDepthMap {
                CameraPreview(session: camera.session)
                    .ignoresSafeArea()
            } else {
                Color.black
                    .ignoresSafeArea()
            }

            // Depth map overlay or full-screen depth map
            if let depthImg = depthEstimator.depthImage {
                if showFullDepthMap {
                    if let opaqueImg = depthEstimator.opaqueDepthImage() {
                        Image(uiImage: opaqueImg)
                            .resizable()
                            .aspectRatio(contentMode: .fill)
                            .ignoresSafeArea()
                    }
                } else {
                    Image(uiImage: depthImg)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .ignoresSafeArea()
                }
            }

            // UI controls
            VStack {
                // Top bar with title and toggle
                HStack {
                    Text("Depth Anything V2")
                        .font(.headline)
                        .foregroundColor(.white)

                    Spacer()

                    Button(action: {
                        withAnimation(.easeInOut(duration: 0.3)) {
                            showFullDepthMap.toggle()
                        }
                    }) {
                        HStack(spacing: 4) {
                            Image(systemName: showFullDepthMap ? "camera.fill" : "square.stack.3d.up.fill")
                                .font(.body)
                            Text(showFullDepthMap ? "Camera" : "Depth")
                                .font(.caption)
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(.ultraThinMaterial, in: Capsule())
                        .foregroundColor(.white)
                    }
                }
                .padding(.horizontal)
                .padding(.top, 8)

                Spacer()

                // Error message if model not loaded
                if let error = depthEstimator.errorMessage {
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

                // Depth info overlay
                if depthEstimator.depthImage != nil {
                    VStack(spacing: 8) {
                        // Color legend
                        HStack(spacing: 0) {
                            Text("Near")
                                .font(.caption2)
                                .foregroundColor(.white)
                            Spacer()

                            // Gradient bar
                            LinearGradient(
                                gradient: Gradient(colors: [.red, .yellow, .green, .cyan, .blue]),
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                            .frame(height: 8)
                            .cornerRadius(4)
                            .padding(.horizontal, 8)

                            Spacer()
                            Text("Far")
                                .font(.caption2)
                                .foregroundColor(.white)
                        }

                        // Depth statistics
                        HStack {
                            Label {
                                Text(String(format: "Min: %.2f", depthEstimator.minDepth))
                                    .font(.system(.caption2, design: .monospaced))
                            } icon: {
                                Image(systemName: "arrow.down.circle.fill")
                                    .foregroundColor(.red)
                                    .font(.caption2)
                            }

                            Spacer()

                            Label {
                                Text(String(format: "Max: %.2f", depthEstimator.maxDepth))
                                    .font(.system(.caption2, design: .monospaced))
                            } icon: {
                                Image(systemName: "arrow.up.circle.fill")
                                    .foregroundColor(.blue)
                                    .font(.caption2)
                            }
                        }
                        .foregroundColor(.white)
                    }
                    .padding()
                    .background(.black.opacity(0.7), in: RoundedRectangle(cornerRadius: 16))
                    .padding()
                }
            }
        }
        .onAppear {
            camera.onFrame = { [weak depthEstimator] buffer in
                depthEstimator?.estimateDepth(sampleBuffer: buffer)
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
