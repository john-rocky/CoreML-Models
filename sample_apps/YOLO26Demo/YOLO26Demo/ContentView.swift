import SwiftUI
import UIKit
import AVFoundation
import CoreML
import Vision
import PhotosUI

// MARK: - Main TabView

struct ContentView: View {
    @StateObject private var detector = Detector()
    @State private var selectedTab = 2

    var body: some View {
        ZStack {
            TabView(selection: $selectedTab) {
                PhotoDetectionView(detector: detector)
                    .tabItem { Label("Photo", systemImage: "photo") }
                    .tag(0)
                VideoDetectionView(detector: detector)
                    .tabItem { Label("Video", systemImage: "video") }
                    .tag(1)
                CameraDetectionView(detector: detector)
                    .tabItem { Label("Camera", systemImage: "camera") }
                    .tag(2)
            }

            if !detector.isReady {
                Color.black.ignoresSafeArea()
                VStack(spacing: 16) {
                    ProgressView()
                        .tint(.white)
                        .scaleEffect(1.2)
                    Text("Loading model...")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
            }
        }
        .tint(.white)
        .preferredColorScheme(.dark)
    }
}

// MARK: - Detection Overlay (shared for photo & video)

struct DetectionOverlay: View {
    let detections: [Detection]
    let imageSize: CGSize
    let displaySize: CGSize
    let colors: [UIColor]

    var body: some View {
        let transform = aspectFitTransform()
        ForEach(detections) { det in
            let r = scaledRect(det.normRect, transform: transform)
            // Color by trackId when present so each tracked object keeps
            // its own color across frames; otherwise color by class.
            let colorIdx = det.trackId ?? det.classIndex
            let color = Color(colors[colorIdx % colors.count])
            let labelText: String = {
                if let tid = det.trackId {
                    return "  #\(tid) \(det.label) \(Int(det.confidence * 100))%  "
                } else {
                    return "  \(det.label) \(Int(det.confidence * 100))%  "
                }
            }()

            RoundedRectangle(cornerRadius: 10)
                .stroke(color, lineWidth: 2)
                .background(RoundedRectangle(cornerRadius: 10).fill(color.opacity(0.08)))
                .frame(width: r.width, height: r.height)
                .position(x: r.midX, y: r.midY)

            Text(labelText)
                .font(.system(size: 11, weight: .semibold))
                .foregroundColor(.white)
                .padding(.horizontal, 4)
                .padding(.vertical, 2)
                .background(color.opacity(0.85))
                .cornerRadius(8)
                .position(x: r.midX, y: r.minY > 20 ? r.minY - 14 : r.maxY + 14)
        }
    }

    private struct FitTransform {
        let scale: CGFloat
        let offsetX: CGFloat
        let offsetY: CGFloat
    }

    private func aspectFitTransform() -> FitTransform {
        guard imageSize.width > 0, imageSize.height > 0 else {
            return FitTransform(scale: 1, offsetX: 0, offsetY: 0)
        }
        let scaleX = displaySize.width / imageSize.width
        let scaleY = displaySize.height / imageSize.height
        let scale = min(scaleX, scaleY)
        let scaledW = imageSize.width * scale
        let scaledH = imageSize.height * scale
        return FitTransform(scale: scale,
                            offsetX: (displaySize.width - scaledW) / 2,
                            offsetY: (displaySize.height - scaledH) / 2)
    }

    private func scaledRect(_ nr: CGRect, transform t: FitTransform) -> CGRect {
        let x = nr.minX * imageSize.width * t.scale + t.offsetX
        let y = nr.minY * imageSize.height * t.scale + t.offsetY
        let w = nr.width * imageSize.width * t.scale
        let h = nr.height * imageSize.height * t.scale
        return CGRect(x: x, y: y, width: w, height: h)
    }
}

// MARK: - Photo Detection

struct PhotoDetectionView: View {
    let detector: Detector
    @State private var selectedItem: PhotosPickerItem?
    @State private var image: UIImage?
    @State private var detections: [Detection] = []
    @State private var isProcessing = false
    @State private var inferenceTime: Double = 0

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            if let image {
                GeometryReader { geo in
                    Image(uiImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)

                    DetectionOverlay(detections: detections,
                                     imageSize: image.size,
                                     displaySize: geo.size,
                                     colors: detector.colors)
                }
            } else {
                PhotosPicker(selection: $selectedItem, matching: .images) {
                    VStack(spacing: 12) {
                        Image(systemName: "photo.on.rectangle.angled")
                            .font(.system(size: 48))
                        Text("Tap to select a photo")
                            .font(.subheadline)
                    }
                    .foregroundStyle(.secondary)
                }
            }

            // Bottom bar
            VStack {
                Spacer()
                HStack {
                    if !detections.isEmpty {
                        Text("\(detections.count) objects")
                            .font(.caption)
                    }
                    Spacer()
                    if inferenceTime > 0 {
                        Text(String(format: "%.0fms", inferenceTime))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    PhotosPicker(selection: $selectedItem, matching: .images) {
                        Image(systemName: "photo.badge.plus")
                            .font(.body)
                            .foregroundColor(.white)
                    }
                }
                .foregroundColor(.white)
                .padding(.horizontal, 20)
                .padding(.vertical, 10)
                .background(.ultraThinMaterial)
            }

            if isProcessing {
                ProgressView()
                    .tint(.white)
                    .scaleEffect(1.5)
                    .padding(24)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
            }
        }
        .onChange(of: selectedItem) { _ in loadAndDetect() }
    }

    private func loadAndDetect() {
        guard let selectedItem else { return }
        isProcessing = true
        Task {
            if let data = try? await selectedItem.loadTransferable(type: Data.self),
               let uiImage = UIImage(data: data) {
                let start = CFAbsoluteTimeGetCurrent()
                let dets = await detector.detect(image: uiImage)
                let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
                await MainActor.run {
                    image = uiImage
                    detections = dets
                    inferenceTime = elapsed
                    isProcessing = false
                }
            } else {
                await MainActor.run { isProcessing = false }
            }
        }
    }
}

// MARK: - Video Detection

struct VideoDetectionView: View {
    let detector: Detector
    @State private var selectedItem: PhotosPickerItem?
    @State private var currentFrame: UIImage?
    @State private var detections: [Detection] = []
    @State private var isPlaying = false
    @State private var progress: Double = 0
    @State private var fps: Double = 0
    @State private var playbackTask: Task<Void, Never>?
    @State private var trackingEnabled: Bool = true

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            if let currentFrame {
                GeometryReader { geo in
                    Image(uiImage: currentFrame)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)

                    DetectionOverlay(detections: detections,
                                     imageSize: currentFrame.size,
                                     displaySize: geo.size,
                                     colors: detector.colors)
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

            // Bottom bar
            VStack {
                Spacer()
                VStack(spacing: 8) {
                    if currentFrame != nil {
                        ProgressView(value: progress)
                            .tint(Color(detector.colors[0]))
                    }
                    HStack(spacing: 12) {
                        if !detections.isEmpty {
                            Text("\(detections.count) objects")
                                .font(.caption)
                        }
                        Spacer()
                        Button {
                            trackingEnabled.toggle()
                            loadAndProcess()
                        } label: {
                            HStack(spacing: 4) {
                                Image(systemName: trackingEnabled ? "scope" : "circle.dashed")
                                Text(trackingEnabled ? "Track" : "Raw")
                            }
                            .font(.caption.weight(.semibold))
                            .foregroundColor(trackingEnabled ? .black : .white)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(trackingEnabled ? Color.white : Color.white.opacity(0.15))
                            .cornerRadius(6)
                        }
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
            let url = videoData.url
            let tracking = trackingEnabled
            await MainActor.run { isPlaying = true }
            playbackTask = Task.detached(priority: .userInitiated) {
                await processVideo(url: url, tracking: tracking)
            }
        }
    }

    private func processVideo(url: URL, tracking: Bool) async {
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
        var frameCount = 0
        let tracker = ByteTracker()

        while !Task.isCancelled, let sb = trackOutput.copyNextSampleBuffer() {
            let pts = CMSampleBufferGetPresentationTimeStamp(sb)
            let currentSec = CMTimeGetSeconds(pts)

            guard let pb = CMSampleBufferGetImageBuffer(sb) else { continue }
            let start = CFAbsoluteTimeGetCurrent()
            let rawDets = detector.detect(pixelBuffer: pb)
            let dets = tracking ? tracker.update(detections: rawDets) : rawDets
            let elapsed = CFAbsoluteTimeGetCurrent() - start

            // Convert pixel buffer to UIImage
            let ciImage = CIImage(cvPixelBuffer: pb)
            guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else { continue }
            let frame = UIImage(cgImage: cgImage)

            frameCount += 1
            let currentFPS = 1.0 / max(elapsed, 0.001)

            await MainActor.run {
                currentFrame = frame
                detections = dets
                progress = currentSec / totalSeconds
                fps = fps == 0 ? currentFPS : fps * 0.9 + currentFPS * 0.1
            }

            let sleepTime = max(frameInterval - elapsed, 0)
            if sleepTime > 0 {
                try? await Task.sleep(for: .seconds(sleepTime))
            }
        }

        await MainActor.run {
            isPlaying = false
            progress = 1.0
        }
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

// MARK: - Camera Detection (UIKit wrapper)

struct CameraDetectionView: View {
    let detector: Detector
    @State private var trackingEnabled: Bool = true

    var body: some View {
        ZStack(alignment: .topTrailing) {
            CameraVCWrapper(detector: detector, trackingEnabled: trackingEnabled)
                .ignoresSafeArea()

            Button {
                trackingEnabled.toggle()
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: trackingEnabled ? "scope" : "circle.dashed")
                    Text(trackingEnabled ? "Track" : "Raw")
                }
                .font(.caption.weight(.semibold))
                .foregroundColor(trackingEnabled ? .black : .white)
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(trackingEnabled ? Color.white : Color.black.opacity(0.4))
                .cornerRadius(8)
            }
            .padding(.top, 60)
            .padding(.trailing, 16)
        }
    }
}

struct CameraVCWrapper: UIViewControllerRepresentable {
    let detector: Detector
    let trackingEnabled: Bool
    func makeUIViewController(context: Context) -> CameraVC {
        let vc = CameraVC(detector: detector)
        vc.setTracking(enabled: trackingEnabled)
        return vc
    }
    func updateUIViewController(_ vc: CameraVC, context: Context) {
        vc.setTracking(enabled: trackingEnabled)
    }
}

// MARK: - Camera ViewController

class CameraVC: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let detector: Detector
    private let tracker = ByteTracker()
    private var trackingEnabled: Bool = true
    private let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "session")
    private let inferenceQueue = DispatchQueue(label: "inference")
    private var previewLayer: AVCaptureVideoPreviewLayer!
    private var boxViews: [BoundingBoxView] = []
    private var isProcessing = false
    private var longSide: CGFloat = 1920
    private var shortSide: CGFloat = 1080
    private var frameSizeCaptured = false

    // Stats (EMA smoothed)
    private var smoothedMs: Double = 0
    private var smoothedFps: Double = 0
    private let statsLabel = CATextLayer()

    init(detector: Detector) {
        self.detector = detector
        super.init(nibName: nil, bundle: nil)
    }
    required init?(coder: NSCoder) { fatalError() }

    /// Called from SwiftUI when the Track toggle changes. Resets the
    /// tracker so IDs restart cleanly whenever tracking is toggled.
    func setTracking(enabled: Bool) {
        inferenceQueue.async { [weak self] in
            guard let self else { return }
            if self.trackingEnabled != enabled {
                self.trackingEnabled = enabled
                self.tracker.reset()
            }
        }
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)
        for _ in 0..<100 {
            let bv = BoundingBoxView()
            bv.addToLayer(previewLayer)
            boxViews.append(bv)
        }

        // Stats overlay
        statsLabel.fontSize = 13
        statsLabel.font = UIFont.monospacedSystemFont(ofSize: 13, weight: .medium)
        statsLabel.foregroundColor = UIColor.white.cgColor
        statsLabel.backgroundColor = UIColor.black.withAlphaComponent(0.5).cgColor
        statsLabel.cornerRadius = 8
        statsLabel.masksToBounds = true
        statsLabel.contentsScale = UIScreen.main.scale
        statsLabel.alignmentMode = .center
        statsLabel.frame = CGRect(x: 0, y: 0, width: 220, height: 28)
        view.layer.addSublayer(statsLabel)
        AVCaptureDevice.requestAccess(for: .video) { [weak self] ok in
            guard ok else { return }
            self?.sessionQueue.async { self?.setupCamera() }
        }
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer.frame = view.bounds
        statsLabel.frame = CGRect(x: (view.bounds.width - 160) / 2,
                                  y: view.safeAreaInsets.top + 8,
                                  width: 220, height: 28)
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        inferenceQueue.async { [weak self] in self?.tracker.reset() }
        sessionQueue.async { if !self.session.isRunning { self.session.startRunning() } }
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        sessionQueue.async { self.session.stopRunning() }
    }

    private func setupCamera() {
        session.beginConfiguration()
        session.sessionPreset = .high
        guard let dev = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: dev) else { session.commitConfiguration(); return }
        if session.canAddInput(input) { session.addInput(input) }
        let out = AVCaptureVideoDataOutput()
        out.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        out.alwaysDiscardsLateVideoFrames = true
        out.setSampleBufferDelegate(self, queue: inferenceQueue)
        if session.canAddOutput(out) { session.addOutput(out) }
        session.commitConfiguration()
        out.connection(with: .video)?.videoOrientation = .portrait
        previewLayer.connection?.videoOrientation = .portrait
        session.startRunning()
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sb: CMSampleBuffer, from conn: AVCaptureConnection) {
        guard !isProcessing else { return }
        guard let pb = CMSampleBufferGetImageBuffer(sb) else { return }
        if !frameSizeCaptured {
            let w = CGFloat(CVPixelBufferGetWidth(pb))
            let h = CGFloat(CVPixelBufferGetHeight(pb))
            longSide = max(w, h); shortSide = min(w, h)
            frameSizeCaptured = true
        }
        isProcessing = true
        let start = CACurrentMediaTime()
        let rawDets = detector.detect(pixelBuffer: pb)
        let dets = trackingEnabled ? tracker.update(detections: rawDets) : rawDets
        let ms = (CACurrentMediaTime() - start) * 1000
        isProcessing = false

        // EMA smoothing (alpha=0.2)
        smoothedMs = smoothedMs == 0 ? ms : smoothedMs * 0.8 + ms * 0.2
        smoothedFps = smoothedFps == 0 ? 1000/ms : smoothedFps * 0.8 + (1000/ms) * 0.2

        let visionDets = dets.map { d in
            (d.label, d.confidence, d.classIndex, d.trackId,
             CGRect(x: d.normRect.minX, y: 1 - d.normRect.maxY,
                    width: d.normRect.width, height: d.normRect.height))
        }
        let statsText = String(format: "  %.1f ms  |  %.1f FPS  ", smoothedMs, smoothedFps)
        DispatchQueue.main.async {
            self.showBoxes(visionDets)
            CATransaction.begin()
            CATransaction.setDisableActions(true)
            self.statsLabel.string = statsText
            CATransaction.commit()
        }
    }

    private func showBoxes(_ dets: [(String, Float, Int, Int?, CGRect)]) {
        let width = view.bounds.width
        let height = view.bounds.height
        let ratio = (height / width) / (longSide / shortSide)

        for i in 0..<boxViews.count {
            guard i < dets.count && i < 50 else { boxViews[i].hide(); continue }
            let (label, conf, cid, tid, nr) = dets[i]
            var displayRect = nr

            if ratio >= 1 {
                let offset = (1 - ratio) * (0.5 - displayRect.minX)
                let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: offset, y: -1)
                displayRect = displayRect.applying(transform)
                displayRect.size.width *= ratio
            } else {
                let offset = (ratio - 1) * (0.5 - displayRect.maxY)
                let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: offset - 1)
                displayRect = displayRect.applying(transform)
                let r2 = (height / width) / (shortSide / longSide)
                displayRect.size.height /= r2
            }

            let screenRect = VNImageRectForNormalizedRect(displayRect, Int(width), Int(height))
            let colorIdx = tid ?? cid
            let color = detector.colors[colorIdx % detector.colors.count]
            let text: String = tid.map { String(format: "#%d %@ %.0f%%", $0, label, conf * 100) }
                ?? String(format: "%@ %.0f%%", label, conf * 100)
            let alpha = CGFloat(max(conf - 0.2, 0.1) / 0.8 * 0.9)
            boxViews[i].show(frame: screenRect, label: text, color: color, alpha: alpha)
        }
    }
}

// MARK: - Bounding Box View (CALayer pool for camera)

class BoundingBoxView {
    let shapeLayer = CAShapeLayer()
    let fillLayer = CAShapeLayer()
    let textLayer = CATextLayer()

    init() {
        fillLayer.isHidden = true
        shapeLayer.fillColor = nil
        shapeLayer.lineWidth = 2
        shapeLayer.lineCap = .round
        shapeLayer.lineJoin = .round
        shapeLayer.isHidden = true
        textLayer.fontSize = 11
        textLayer.font = UIFont.systemFont(ofSize: 11, weight: .semibold)
        textLayer.foregroundColor = UIColor.white.cgColor
        textLayer.contentsScale = UIScreen.main.scale
        textLayer.isHidden = true
        textLayer.cornerRadius = 8
        textLayer.masksToBounds = true
        textLayer.alignmentMode = .center
    }

    func addToLayer(_ parent: CALayer) {
        parent.addSublayer(fillLayer)
        parent.addSublayer(shapeLayer)
        parent.addSublayer(textLayer)
    }

    func show(frame: CGRect, label: String, color: UIColor, alpha: CGFloat) {
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        let path = UIBezierPath(roundedRect: frame, cornerRadius: 10).cgPath
        shapeLayer.path = path
        shapeLayer.strokeColor = color.withAlphaComponent(alpha).cgColor
        shapeLayer.isHidden = false
        fillLayer.path = path
        fillLayer.fillColor = color.withAlphaComponent(0.08).cgColor
        fillLayer.isHidden = false
        textLayer.string = "  \(label)  "
        textLayer.backgroundColor = color.withAlphaComponent(min(alpha + 0.1, 0.9)).cgColor
        let tw = CGFloat(label.count) * 7 + 20
        let ty = frame.minY > 28 ? frame.minY - 24 : frame.maxY + 4
        textLayer.frame = CGRect(x: frame.minX, y: ty,
                                 width: min(tw, max(frame.width + 24, 64)), height: 20)
        textLayer.isHidden = false
        CATransaction.commit()
    }

    func hide() {
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        shapeLayer.isHidden = true
        fillLayer.isHidden = true
        textLayer.isHidden = true
        CATransaction.commit()
    }
}

#Preview { ContentView() }
