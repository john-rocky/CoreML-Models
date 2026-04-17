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
///
/// Supply `frameSize` (pixel buffer / image dimensions) and `contentMode` so
/// the overlay can match AVCaptureVideoPreviewLayer's .resizeAspectFill or a
/// SwiftUI Image's .fit — otherwise boxes drift when aspect ratios differ.
/// If `frameSize == .zero`, normalized coords are mapped straight onto the
/// view bounds (legacy behavior).
struct BoundingBoxOverlay: View {
    let detections: [DetectionBox]
    var frameSize: CGSize = .zero
    var contentMode: ContentMode = .fill
    let colors: [Color] = [.red, .blue, .green, .orange, .purple, .cyan, .yellow, .pink]

    struct DetectionBox: Identifiable {
        let id = UUID()
        let label: String
        let confidence: Float
        let rect: CGRect  // normalized 0..1, origin top-left
        let classIndex: Int
        var trackId: Int? = nil
        // Recent Kalman-center history (normalized top-left, oldest → newest),
        // populated by ByteTracker for motion trails; empty for raw detections.
        var trail: [CGPoint] = []
    }

    var body: some View {
        GeometryReader { geo in
            let t = transform(for: geo.size)

            // Motion trails for tracked objects, underneath the boxes.
            ForEach(detections) { det in
                if det.trail.count >= 2 {
                    let colorIdx = det.trackId ?? det.classIndex
                    let color = colors[colorIdx % colors.count]
                    Path { path in
                        let pts = det.trail.map { p in
                            CGPoint(x: t.offset.x + p.x * t.size.width,
                                    y: t.offset.y + p.y * t.size.height)
                        }
                        path.move(to: pts[0])
                        for p in pts.dropFirst() { path.addLine(to: p) }
                    }
                    .stroke(color.opacity(0.75),
                            style: StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round))
                }
            }

            ForEach(detections) { det in
                let colorIdx = det.trackId ?? det.classIndex
                let color = colors[colorIdx % colors.count]
                let rect = CGRect(
                    x: t.offset.x + det.rect.origin.x * t.size.width,
                    y: t.offset.y + det.rect.origin.y * t.size.height,
                    width: det.rect.width * t.size.width,
                    height: det.rect.height * t.size.height
                )
                let labelText: String = {
                    if let tid = det.trackId {
                        return "#\(tid) \(det.label) \(Int(det.confidence * 100))%"
                    }
                    return "\(det.label) \(Int(det.confidence * 100))%"
                }()

                Rectangle()
                    .stroke(color, lineWidth: 2)
                    .frame(width: rect.width, height: rect.height)
                    .position(x: rect.midX, y: rect.midY)

                Text(labelText)
                    .font(.caption2.bold())
                    .foregroundStyle(.white)
                    .padding(.horizontal, 4).padding(.vertical, 2)
                    .background(color.opacity(0.8))
                    .clipShape(RoundedRectangle(cornerRadius: 4))
                    .position(x: rect.midX, y: rect.minY - 10)
            }
        }
    }

    /// Mirrors AVCaptureVideoPreviewLayer / SwiftUI Image aspect transforms.
    private func transform(for viewSize: CGSize) -> (size: CGSize, offset: CGPoint) {
        guard frameSize.width > 0, frameSize.height > 0 else {
            return (viewSize, .zero)
        }
        let sx = viewSize.width / frameSize.width
        let sy = viewSize.height / frameSize.height
        let scale = (contentMode == .fill) ? max(sx, sy) : min(sx, sy)
        let w = frameSize.width * scale
        let h = frameSize.height * scale
        return (CGSize(width: w, height: h),
                CGPoint(x: (viewSize.width - w) / 2, y: (viewSize.height - h) / 2))
    }
}
