import SwiftUI
import AVFoundation

/// Reusable camera preview + frame capture component.
///
/// Usage:
/// ```
/// CameraView(position: .back) { pixelBuffer in
///     // Process each frame
/// }
/// ```
struct CameraView: UIViewRepresentable {
    let position: AVCaptureDevice.Position
    let onFrame: ((CVPixelBuffer) -> Void)?

    init(position: AVCaptureDevice.Position = .back, onFrame: ((CVPixelBuffer) -> Void)? = nil) {
        self.position = position
        self.onFrame = onFrame
    }

    func makeUIView(context: Context) -> CameraPreviewUIView {
        let view = CameraPreviewUIView()
        view.position = position
        view.onFrame = onFrame
        view.startSession()
        return view
    }

    func updateUIView(_ uiView: CameraPreviewUIView, context: Context) {}

    static func dismantleUIView(_ uiView: CameraPreviewUIView, coordinator: ()) {
        uiView.stopSession()
    }
}

/// UIKit view that hosts an AVCaptureSession with video preview.
class CameraPreviewUIView: UIView, AVCaptureVideoDataOutputSampleBufferDelegate {
    var position: AVCaptureDevice.Position = .back
    var onFrame: ((CVPixelBuffer) -> Void)?

    private let session = AVCaptureSession()
    private var previewLayer: AVCaptureVideoPreviewLayer?
    private let outputQueue = DispatchQueue(label: "cameraOutput", qos: .userInitiated)

    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }

    func startSession() {
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position),
              let input = try? AVCaptureDeviceInput(device: device) else { return }

        session.beginConfiguration()
        session.sessionPreset = .hd1280x720
        if session.canAddInput(input) { session.addInput(input) }

        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: outputQueue)
        if session.canAddOutput(output) { session.addOutput(output) }

        // Set video orientation
        if let connection = output.connection(with: .video) {
            connection.videoRotationAngle = 90  // Portrait
            if position == .front { connection.isVideoMirrored = true }
        }

        session.commitConfiguration()

        // Setup preview layer
        let preview = self.layer as! AVCaptureVideoPreviewLayer
        preview.session = session
        preview.videoGravity = .resizeAspectFill

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.session.startRunning()
        }
    }

    func stopSession() {
        session.stopRunning()
    }

    // MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        onFrame?(pixelBuffer)
    }
}

/// Overlay view for drawing bounding boxes on camera feed.
struct BoundingBoxOverlay: View {
    let detections: [DetectionBox]
    let colors: [Color] = [.red, .blue, .green, .orange, .purple, .cyan, .yellow, .pink]

    struct DetectionBox: Identifiable {
        let id = UUID()
        let label: String
        let confidence: Float
        let rect: CGRect  // normalized 0..1, origin top-left
        let classIndex: Int
    }

    var body: some View {
        GeometryReader { geo in
            ForEach(detections) { det in
                let color = colors[det.classIndex % colors.count]
                let rect = CGRect(
                    x: det.rect.origin.x * geo.size.width,
                    y: det.rect.origin.y * geo.size.height,
                    width: det.rect.width * geo.size.width,
                    height: det.rect.height * geo.size.height
                )

                Rectangle()
                    .stroke(color, lineWidth: 2)
                    .frame(width: rect.width, height: rect.height)
                    .position(x: rect.midX, y: rect.midY)

                Text("\(det.label) \(Int(det.confidence * 100))%")
                    .font(.caption2.bold())
                    .foregroundStyle(.white)
                    .padding(.horizontal, 4).padding(.vertical, 2)
                    .background(color.opacity(0.8))
                    .clipShape(RoundedRectangle(cornerRadius: 4))
                    .position(x: rect.midX, y: rect.minY - 10)
            }
        }
    }
}
