import SwiftUI
import PhotosUI
import AVFoundation

struct ContentView: View {
    @StateObject private var reconstructor = FaceReconstructor()

    var body: some View {
        TabView {
            PhotoTab(reconstructor: reconstructor)
                .tabItem { Label("Photo", systemImage: "photo") }
            VideoTab(reconstructor: reconstructor)
                .tabItem { Label("Video", systemImage: "video") }
            CameraTab(reconstructor: reconstructor)
                .tabItem { Label("Camera", systemImage: "camera") }
        }
    }
}

// MARK: - Photo Tab

struct PhotoTab: View {
    @ObservedObject var reconstructor: FaceReconstructor
    @State private var selectedItem: PhotosPickerItem?
    @State private var image: UIImage?
    @State private var results: [FacePoseResult] = []
    @State private var processing = false

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("3D Face Pose").font(.headline)
                Spacer()
                PhotosPicker(selection: $selectedItem, matching: .images) {
                    Image(systemName: "photo.badge.plus")
                        .font(.title2)
                }
            }
            .padding()

            GeometryReader { geo in
                if let image = image {
                    let imageSize = image.size
                    let fitSize = aspectFitSize(imageSize: imageSize, in: geo.size)
                    let offsetX = (geo.size.width - fitSize.width) / 2
                    let offsetY = (geo.size.height - fitSize.height) / 2

                    ZStack(alignment: .topLeading) {
                        Image(uiImage: image)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(width: geo.size.width, height: geo.size.height)

                        ForEach(Array(results.enumerated()), id: \.offset) { _, result in
                            PoseAxesView(
                                result: result,
                                imageSize: imageSize,
                                displaySize: fitSize,
                                offset: CGPoint(x: offsetX, y: offsetY)
                            )
                        }
                    }
                } else {
                    VStack {
                        Spacer()
                        Image(systemName: "face.smiling")
                            .font(.system(size: 60))
                            .foregroundColor(.secondary)
                        Text("Select a photo with faces")
                            .foregroundColor(.secondary)
                            .padding(.top, 8)
                        Spacer()
                    }
                    .frame(maxWidth: .infinity)
                }
            }

            if !results.isEmpty {
                AngleInfoBar(results: results)
            }

            if processing {
                ProgressView("Analyzing...")
                    .padding()
            }
        }
        .onChange(of: selectedItem) { _ in
            loadImage()
        }
    }

    private func loadImage() {
        guard let item = selectedItem else { return }
        processing = true
        results = []
        Task {
            if let data = try? await item.loadTransferable(type: Data.self),
               let uiImage = UIImage(data: data) {
                image = uiImage
                let detected = await reconstructor.detect(image: uiImage)
                await MainActor.run {
                    results = detected
                    processing = false
                }
            } else {
                await MainActor.run { processing = false }
            }
        }
    }
}

// MARK: - Pose Axes Overlay

struct PoseAxesView: View {
    let result: FacePoseResult
    let imageSize: CGSize
    let displaySize: CGSize
    let offset: CGPoint

    var body: some View {
        Canvas { ctx, _ in
            let scaleX = displaySize.width / imageSize.width
            let scaleY = displaySize.height / imageSize.height

            let cx = (result.faceRect.midX * imageSize.width) * scaleX + offset.x
            let cy = ((1.0 - result.faceRect.midY) * imageSize.height) * scaleY + offset.y
            let center = CGPoint(x: cx, y: cy)

            let axisLen = result.faceRect.width * imageSize.width * scaleX * 0.6

            let (xD, yD, zD) = poseAxesFromEuler(
                yaw: CGFloat(result.yaw), pitch: CGFloat(result.pitch),
                roll: CGFloat(result.roll), length: axisLen)
            drawAxis(ctx: ctx, from: center, to: CGPoint(x: cx + xD.x, y: cy + xD.y), color: .red)
            drawAxis(ctx: ctx, from: center, to: CGPoint(x: cx + yD.x, y: cy + yD.y), color: .green)
            drawAxis(ctx: ctx, from: center, to: CGPoint(x: cx + zD.x, y: cy + zD.y), color: .blue)

            // Face rect
            let rx = result.faceRect.origin.x * imageSize.width * scaleX + offset.x
            let ry = (1.0 - result.faceRect.origin.y - result.faceRect.height) * imageSize.height * scaleY + offset.y
            let rw = result.faceRect.width * imageSize.width * scaleX
            let rh = result.faceRect.height * imageSize.height * scaleY
            let faceRectPath = Path(CGRect(x: rx, y: ry, width: rw, height: rh))
            ctx.stroke(faceRectPath, with: .color(.yellow), lineWidth: 2)
        }
        .allowsHitTesting(false)
    }

    private func drawAxis(ctx: GraphicsContext, from: CGPoint, to: CGPoint, color: Color) {
        var path = Path()
        path.move(to: from)
        path.addLine(to: to)
        ctx.stroke(path, with: .color(color), style: StrokeStyle(lineWidth: 3, lineCap: .round))

        // Arrow head
        let dx = to.x - from.x
        let dy = to.y - from.y
        let angle = atan2(dy, dx)
        let headLen: CGFloat = 10
        var arrow = Path()
        arrow.move(to: to)
        arrow.addLine(to: CGPoint(
            x: to.x - headLen * cos(angle - .pi / 6),
            y: to.y - headLen * sin(angle - .pi / 6)
        ))
        arrow.move(to: to)
        arrow.addLine(to: CGPoint(
            x: to.x - headLen * cos(angle + .pi / 6),
            y: to.y - headLen * sin(angle + .pi / 6)
        ))
        ctx.stroke(arrow, with: .color(color), style: StrokeStyle(lineWidth: 3, lineCap: .round))
    }
}

// MARK: - Angle Info Bar

struct AngleInfoBar: View {
    let results: [FacePoseResult]

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 16) {
                ForEach(Array(results.enumerated()), id: \.offset) { i, r in
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Face \(i + 1)").font(.caption).bold()
                        HStack(spacing: 12) {
                            AngleLabel(name: "Yaw", value: r.yaw, color: .blue)
                            AngleLabel(name: "Pitch", value: r.pitch, color: .green)
                            AngleLabel(name: "Roll", value: r.roll, color: .red)
                        }
                    }
                    .padding(10)
                    .background(RoundedRectangle(cornerRadius: 8).fill(.ultraThinMaterial))
                }
            }
            .padding(.horizontal)
        }
        .padding(.vertical, 8)
    }
}

struct AngleLabel: View {
    let name: String
    let value: Float
    let color: Color

    var body: some View {
        VStack(spacing: 2) {
            Text(name).font(.caption2).foregroundColor(.secondary)
            Text(String(format: "%.1f°", value))
                .font(.system(.caption, design: .monospaced))
                .bold()
                .foregroundColor(color)
        }
    }
}

// MARK: - Video Tab

struct VideoTab: View {
    @ObservedObject var reconstructor: FaceReconstructor
    @State private var selectedItem: PhotosPickerItem?
    @State private var currentFrame: UIImage?
    @State private var results: [FacePoseResult] = []
    @State private var progress: Double = 0
    @State private var fps: Double = 0
    @State private var playbackTask: Task<Void, Never>?

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            if let currentFrame {
                GeometryReader { geo in
                    let fitSize = aspectFitSize(imageSize: currentFrame.size, in: geo.size)
                    let offsetX = (geo.size.width - fitSize.width) / 2
                    let offsetY = (geo.size.height - fitSize.height) / 2

                    ZStack(alignment: .topLeading) {
                        Image(uiImage: currentFrame)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(maxWidth: .infinity, maxHeight: .infinity)

                        ForEach(Array(results.enumerated()), id: \.offset) { _, result in
                            PoseAxesView(
                                result: result,
                                imageSize: currentFrame.size,
                                displaySize: fitSize,
                                offset: CGPoint(x: offsetX, y: offsetY)
                            )
                        }
                    }
                }
            } else {
                PhotosPicker(selection: $selectedItem, matching: .videos) {
                    VStack(spacing: 12) {
                        Image(systemName: "video.badge.plus")
                            .font(.system(size: 48))
                        Text("Tap to select a video")
                            .font(.subheadline)
                    }
                    .foregroundStyle(.secondary)
                }
            }

            VStack {
                Spacer()
                VStack(spacing: 8) {
                    if currentFrame != nil {
                        ProgressView(value: progress)
                            .tint(.blue)
                    }
                    HStack {
                        if !results.isEmpty {
                            Text("\(results.count) faces").font(.caption)
                        }
                        Spacer()
                        if fps > 0 {
                            Text(String(format: "%.1f FPS", fps))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        PhotosPicker(selection: $selectedItem, matching: .videos) {
                            Image(systemName: "video.badge.plus")
                                .font(.body)
                                .foregroundColor(.white)
                        }
                    }
                    .foregroundColor(.white)
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 10)
                .background(.ultraThinMaterial)
            }
        }
        .onChange(of: selectedItem) { _ in loadAndProcess() }
        .onDisappear { playbackTask?.cancel() }
    }

    private func loadAndProcess() {
        playbackTask?.cancel()
        guard let selectedItem else { return }
        Task {
            guard let videoData = try? await selectedItem.loadTransferable(type: VideoTransferable.self) else { return }
            playbackTask = Task.detached(priority: .userInitiated) {
                await processVideo(url: videoData.url)
            }
        }
    }

    private func processVideo(url: URL) async {
        let asset = AVURLAsset(url: url)
        guard let track = try? await asset.loadTracks(withMediaType: .video).first else { return }
        let duration = try? await asset.load(.duration)
        let totalSeconds = duration.map { CMTimeGetSeconds($0) } ?? 1
        let nominalFPS = (try? await track.load(.nominalFrameRate)) ?? 30
        let frameInterval = 1.0 / Double(nominalFPS)

        guard let reader = try? AVAssetReader(asset: asset) else { return }
        let outputSettings: [String: Any] = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        let trackOutput = AVAssetReaderTrackOutput(track: track, outputSettings: outputSettings)
        reader.add(trackOutput)
        reader.startReading()

        let ciContext = CIContext()

        while !Task.isCancelled, let sb = trackOutput.copyNextSampleBuffer() {
            let pts = CMSampleBufferGetPresentationTimeStamp(sb)
            let currentSec = CMTimeGetSeconds(pts)

            guard let pb = CMSampleBufferGetImageBuffer(sb) else { continue }
            let start = CFAbsoluteTimeGetCurrent()
            let detected = reconstructor.detect(pixelBuffer: pb)
            let elapsed = CFAbsoluteTimeGetCurrent() - start

            let ciImage = CIImage(cvPixelBuffer: pb)
            guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else { continue }
            let frame = UIImage(cgImage: cgImage)

            let currentFPS = 1.0 / max(elapsed, 0.001)

            await MainActor.run {
                currentFrame = frame
                results = detected
                progress = currentSec / totalSeconds
                fps = fps == 0 ? currentFPS : fps * 0.9 + currentFPS * 0.1
            }

            let sleepTime = max(frameInterval - elapsed, 0)
            if sleepTime > 0 {
                try? await Task.sleep(for: .seconds(sleepTime))
            }
        }

        await MainActor.run { progress = 1.0 }
    }
}

struct VideoTransferable: Transferable {
    let url: URL
    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { movie in
            SentTransferredFile(movie.url)
        } importing: { received in
            let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(
                UUID().uuidString + "." + received.file.pathExtension)
            try FileManager.default.copyItem(at: received.file, to: tmp)
            return Self(url: tmp)
        }
    }
}

// MARK: - Camera Tab

struct CameraTab: View {
    @ObservedObject var reconstructor: FaceReconstructor

    var body: some View {
        CameraViewWrapper(reconstructor: reconstructor)
            .ignoresSafeArea()
    }
}

struct CameraViewWrapper: UIViewControllerRepresentable {
    let reconstructor: FaceReconstructor

    func makeUIViewController(context: Context) -> CameraVC {
        CameraVC(reconstructor: reconstructor)
    }

    func updateUIViewController(_ vc: CameraVC, context: Context) {}
}

// MARK: - Camera VC

class CameraVC: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let reconstructor: FaceReconstructor
    private let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "session")
    private let inferenceQueue = DispatchQueue(label: "inference")
    private var previewLayer: AVCaptureVideoPreviewLayer!
    private var overlayLayer = CALayer()
    private var statsLayer = CATextLayer()
    private var isProcessing = false
    private var smoothedLatency: Double = 0

    // Temporal smoothing
    private let smoothingFactor: Float = 0.3
    private var smoothedResults: [SmoothedFace] = []

    private struct SmoothedFace {
        var faceRect: CGRect
        var yaw: Float
        var pitch: Float
        var roll: Float
        var rotationMatrix: [[Float]]
    }

    init(reconstructor: FaceReconstructor) {
        self.reconstructor = reconstructor
        super.init(nibName: nil, bundle: nil)
    }

    required init?(coder: NSCoder) { fatalError() }

    override func viewDidLoad() {
        super.viewDidLoad()
        setupCamera()
        setupOverlay()
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer?.frame = view.bounds
        overlayLayer.frame = view.bounds
        statsLayer.frame = CGRect(x: 16, y: view.safeAreaInsets.top + 8, width: 200, height: 50)
    }

    private func setupCamera() {
        previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)

        sessionQueue.async { [weak self] in
            guard let self else { return }
            session.beginConfiguration()
            session.sessionPreset = .high

            guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front),
                  let input = try? AVCaptureDeviceInput(device: device),
                  session.canAddInput(input) else { return }
            session.addInput(input)

            let output = AVCaptureVideoDataOutput()
            output.alwaysDiscardsLateVideoFrames = true
            output.setSampleBufferDelegate(self, queue: inferenceQueue)
            if session.canAddOutput(output) {
                session.addOutput(output)
            }
            if let conn = output.connection(with: .video) {
                conn.videoOrientation = .portrait
                conn.isVideoMirrored = true
            }
            session.commitConfiguration()
            session.startRunning()
        }
    }

    private func setupOverlay() {
        view.layer.addSublayer(overlayLayer)

        statsLayer.fontSize = 14
        statsLayer.foregroundColor = UIColor.white.cgColor
        statsLayer.backgroundColor = UIColor.black.withAlphaComponent(0.5).cgColor
        statsLayer.cornerRadius = 6
        statsLayer.contentsScale = UIScreen.main.scale
        statsLayer.alignmentMode = .left
        view.layer.addSublayer(statsLayer)
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard !isProcessing else { return }
        isProcessing = true

        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            isProcessing = false
            return
        }

        let start = CACurrentMediaTime()
        let rawResults = reconstructor.detect(pixelBuffer: pb)
        let latency = (CACurrentMediaTime() - start) * 1000
        smoothedLatency = smoothedLatency * 0.8 + latency * 0.2

        let bufW = CGFloat(CVPixelBufferGetWidth(pb))
        let bufH = CGFloat(CVPixelBufferGetHeight(pb))

        let displayResults = applySmoothing(rawResults)

        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.drawSmoothedResults(displayResults, bufferSize: CGSize(width: bufW, height: bufH))
            self.statsLayer.string = String(format: "  Latency: %.0f ms  FPS: %.0f", smoothedLatency, 1000.0 / max(smoothedLatency, 1))
            self.isProcessing = false
        }
    }

    private func applySmoothing(_ results: [FacePoseResult]) -> [SmoothedFace] {
        let a = smoothingFactor

        if results.isEmpty {
            smoothedResults = []
            return []
        }

        // Match new results to existing smoothed faces by closest center
        var newSmoothed: [SmoothedFace] = []
        for r in results {
            if let idx = closestSmoothedIndex(for: r.faceRect, in: smoothedResults) {
                let prev = smoothedResults[idx]
                let lerp = { (old: Float, new: Float) -> Float in old * (1 - a) + new * a }
                let lerpCG = { (old: CGFloat, new: CGFloat) -> CGFloat in old * CGFloat(1 - a) + new * CGFloat(a) }

                var sR = [[Float]](repeating: [Float](repeating: 0, count: 3), count: 3)
                for i in 0..<3 { for j in 0..<3 { sR[i][j] = lerp(prev.rotationMatrix[i][j], r.rotationMatrix[i][j]) } }

                newSmoothed.append(SmoothedFace(
                    faceRect: CGRect(
                        x: lerpCG(prev.faceRect.origin.x, r.faceRect.origin.x),
                        y: lerpCG(prev.faceRect.origin.y, r.faceRect.origin.y),
                        width: lerpCG(prev.faceRect.width, r.faceRect.width),
                        height: lerpCG(prev.faceRect.height, r.faceRect.height)
                    ),
                    yaw: lerp(prev.yaw, r.yaw),
                    pitch: lerp(prev.pitch, r.pitch),
                    roll: lerp(prev.roll, r.roll),
                    rotationMatrix: sR
                ))
            } else {
                newSmoothed.append(SmoothedFace(
                    faceRect: r.faceRect, yaw: r.yaw, pitch: r.pitch, roll: r.roll,
                    rotationMatrix: r.rotationMatrix
                ))
            }
        }
        smoothedResults = newSmoothed
        return newSmoothed
    }

    private func closestSmoothedIndex(for rect: CGRect, in faces: [SmoothedFace]) -> Int? {
        guard !faces.isEmpty else { return nil }
        let center = CGPoint(x: rect.midX, y: rect.midY)
        var bestIdx = 0
        var bestDist = CGFloat.greatestFiniteMagnitude
        for (i, f) in faces.enumerated() {
            let fc = CGPoint(x: f.faceRect.midX, y: f.faceRect.midY)
            let d = hypot(center.x - fc.x, center.y - fc.y)
            if d < bestDist { bestDist = d; bestIdx = i }
        }
        return bestDist < 0.3 ? bestIdx : nil
    }

    private func drawSmoothedResults(_ results: [SmoothedFace], bufferSize: CGSize) {
        overlayLayer.sublayers?.forEach { $0.removeFromSuperlayer() }

        let viewSize = view.bounds.size
        let scale = max(viewSize.width / bufferSize.width, viewSize.height / bufferSize.height)
        let scaledW = bufferSize.width * scale
        let scaledH = bufferSize.height * scale
        let offsetX = (viewSize.width - scaledW) / 2
        let offsetY = (viewSize.height - scaledH) / 2

        for r in results {
            let cx = r.faceRect.midX * scaledW + offsetX
            let cy = (1.0 - r.faceRect.midY) * scaledH + offsetY
            let axisLen = r.faceRect.width * scaledW * 0.6

            let (xD, yD, zD) = poseAxesFromEuler(
                yaw: CGFloat(r.yaw), pitch: CGFloat(r.pitch),
                roll: CGFloat(r.roll), length: axisLen)
            drawAxisLine(from: CGPoint(x: cx, y: cy), dx: xD.x, dy: xD.y, color: UIColor.red)
            drawAxisLine(from: CGPoint(x: cx, y: cy), dx: yD.x, dy: yD.y, color: UIColor.green)
            drawAxisLine(from: CGPoint(x: cx, y: cy), dx: zD.x, dy: zD.y, color: UIColor.blue)

            let rx = r.faceRect.origin.x * scaledW + offsetX
            let ry = (1.0 - r.faceRect.origin.y - r.faceRect.height) * scaledH + offsetY
            let rw = r.faceRect.width * scaledW
            let rh = r.faceRect.height * scaledH
            let rectLayer = CAShapeLayer()
            rectLayer.path = UIBezierPath(rect: CGRect(x: rx, y: ry, width: rw, height: rh)).cgPath
            rectLayer.strokeColor = UIColor.yellow.cgColor
            rectLayer.fillColor = UIColor.clear.cgColor
            rectLayer.lineWidth = 2
            overlayLayer.addSublayer(rectLayer)

            let label = CATextLayer()
            let font = UIFont.monospacedSystemFont(ofSize: 13, weight: .bold)
            let attrStr = NSMutableAttributedString()
            let parts: [(String, UIColor)] = [
                (String(format: "Y:%.0f° ", r.yaw), .systemBlue),
                (String(format: "P:%.0f° ", r.pitch), .systemGreen),
                (String(format: "R:%.0f°", r.roll), .systemRed),
            ]
            for (text, color) in parts {
                attrStr.append(NSAttributedString(string: text, attributes: [
                    .foregroundColor: color, .font: font
                ]))
            }
            label.string = attrStr
            label.backgroundColor = UIColor.black.withAlphaComponent(0.6).cgColor
            label.cornerRadius = 4
            label.contentsScale = UIScreen.main.scale
            label.alignmentMode = .center
            label.frame = CGRect(x: rx, y: ry - 24, width: rw, height: 22)
            overlayLayer.addSublayer(label)
        }
    }

    private func drawAxisLine(from: CGPoint, dx: CGFloat, dy: CGFloat, color: UIColor) {
        let to = CGPoint(x: from.x + dx, y: from.y + dy)
        let line = CAShapeLayer()
        let path = UIBezierPath()
        path.move(to: from)
        path.addLine(to: to)

        // Arrow head
        let angle = atan2(dy, dx)
        let headLen: CGFloat = 10
        path.move(to: to)
        path.addLine(to: CGPoint(
            x: to.x - headLen * cos(angle - .pi / 6),
            y: to.y - headLen * sin(angle - .pi / 6)
        ))
        path.move(to: to)
        path.addLine(to: CGPoint(
            x: to.x - headLen * cos(angle + .pi / 6),
            y: to.y - headLen * sin(angle + .pi / 6)
        ))

        line.path = path.cgPath
        line.strokeColor = color.cgColor
        line.fillColor = UIColor.clear.cgColor
        line.lineWidth = 3
        line.lineCap = .round
        overlayLayer.addSublayer(line)
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        sessionQueue.async { [weak self] in
            self?.session.stopRunning()
        }
    }
}

// MARK: - Helpers

func aspectFitSize(imageSize: CGSize, in containerSize: CGSize) -> CGSize {
    let scale = min(containerSize.width / imageSize.width, containerSize.height / imageSize.height)
    return CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
}

// Standard draw_axis from 3DDFA_V2 reference implementation.
// Returns (dx, dy) offsets for X, Y, Z axes in image coordinates.
func poseAxesFromEuler(yaw: CGFloat, pitch: CGFloat, roll: CGFloat, length: CGFloat) -> (x: CGPoint, y: CGPoint, z: CGPoint) {
    let yr = yaw * .pi / 180
    let pr = pitch * .pi / 180
    let rr = roll * .pi / 180

    // X axis (Red) - right
    let xDx = length * cos(yr) * cos(rr)
    let xDy = length * (cos(pr) * sin(rr) + cos(rr) * sin(pr) * sin(yr))

    // Y axis (Green) - up (negated from original which points down)
    let yDx = length * (cos(yr) * sin(rr))
    let yDy = length * (-cos(pr) * cos(rr) + sin(pr) * sin(yr) * sin(rr))

    // Z axis (Blue) - forward / nose direction
    let zDx = length * sin(yr)
    let zDy = length * (-cos(yr) * sin(pr))

    return (CGPoint(x: xDx, y: xDy), CGPoint(x: yDx, y: yDy), CGPoint(x: zDx, y: zDy))
}
