import SwiftUI
import UIKit
import AVFoundation
import CoreML
import Vision
import Accelerate
import PhotosUI
import AVKit

// MARK: - Main TabView

struct ContentView: View {
    @StateObject private var detector = TextGroundingDetector()
    @State private var selectedTab = 2
    @State private var threshold: Float = 0.15

    var body: some View {
        ZStack {
            TabView(selection: $selectedTab) {
                PhotoDetectionView(detector: detector, threshold: $threshold)
                    .tabItem { Label("Photo", systemImage: "photo") }
                    .tag(0)
                VideoDetectionView(detector: detector, threshold: $threshold)
                    .tabItem { Label("Video", systemImage: "video") }
                    .tag(1)
                CameraDetectionView(detector: detector, threshold: $threshold)
                    .tabItem { Label("Camera", systemImage: "camera") }
                    .tag(2)
            }

            if !detector.isModelLoaded {
                Color.black.ignoresSafeArea()
                VStack(spacing: 16) {
                    ProgressView().tint(.white).scaleEffect(1.2)
                    Text("Loading model...").font(.subheadline).foregroundColor(.secondary)
                }
            }
        }
        .tint(.white)
        .preferredColorScheme(.dark)
    }
}

// MARK: - Detection Result

struct Detection: Identifiable {
    let id = UUID()
    let label: String
    let confidence: Float
    let classIndex: Int
    let normRect: CGRect // Normalized [0,1], origin at top-left
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
            let color = Color(colors[det.classIndex % colors.count])

            RoundedRectangle(cornerRadius: 10)
                .stroke(color, lineWidth: 2)
                .background(RoundedRectangle(cornerRadius: 10).fill(color.opacity(0.08)))
                .frame(width: r.width, height: r.height)
                .position(x: r.midX, y: r.midY)

            Text("  \(det.label) \(Int(det.confidence * 100))%  ")
                .font(.system(size: 11, weight: .semibold))
                .foregroundColor(.white)
                .padding(.horizontal, 4)
                .padding(.vertical, 2)
                .background(color.opacity(0.85))
                .cornerRadius(8)
                .position(x: r.midX, y: r.minY > 20 ? r.minY - 14 : r.maxY + 14)
        }
    }

    private struct FitTransform { let scale: CGFloat; let offsetX: CGFloat; let offsetY: CGFloat }

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
        CGRect(x: nr.minX * imageSize.width * t.scale + t.offsetX,
               y: nr.minY * imageSize.height * t.scale + t.offsetY,
               width: nr.width * imageSize.width * t.scale,
               height: nr.height * imageSize.height * t.scale)
    }
}

// MARK: - Photo Detection

struct PhotoDetectionView: View {
    let detector: TextGroundingDetector
    @Binding var threshold: Float
    @State private var selectedItem: PhotosPickerItem?
    @State private var image: UIImage?
    @State private var allDetections: [Detection] = []
    @State private var isProcessing = false
    @State private var inferenceTime: Double = 0
    @State private var queryText = "person, dog, car"

    private var filteredDetections: [Detection] {
        allDetections.filter { $0.confidence >= threshold }
    }

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            if let image {
                GeometryReader { geo in
                    Image(uiImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)

                    DetectionOverlay(detections: filteredDetections,
                                     imageSize: image.size,
                                     displaySize: geo.size,
                                     colors: detector.colors)
                }
            } else {
                PhotosPicker(selection: $selectedItem, matching: .images) {
                    VStack(spacing: 12) {
                        Image(systemName: "photo.on.rectangle.angled")
                            .font(.system(size: 48))
                        Text("Tap to select a photo").font(.subheadline)
                    }
                    .foregroundStyle(.secondary)
                }
            }

            VStack {
                Spacer()
                VStack(spacing: 8) {
                    HStack(spacing: 8) {
                        TextField("Objects (comma-separated)", text: $queryText)
                            .textFieldStyle(.roundedBorder)
                            .onSubmit { runDetection() }
                            .submitLabel(.search)
                        Button { runDetection() } label: {
                            Image(systemName: "magnifyingglass")
                                .foregroundColor(.white)
                                .padding(6)
                                .background(Color.blue, in: Circle())
                        }
                    }
                    HStack(spacing: 4) {
                        Text(String(format: "%.0f%%", threshold * 100))
                            .font(.caption).monospacedDigit().frame(width: 36)
                        Slider(value: $threshold, in: 0.05...0.95, step: 0.05)
                            .onChange(of: threshold) { val in detector.confidenceThreshold = val }
                    }
                    HStack {
                        if !filteredDetections.isEmpty {
                            Text("\(filteredDetections.count) objects").font(.caption)
                        }
                        Spacer()
                        if inferenceTime > 0 {
                            Text(String(format: "%.0fms", inferenceTime))
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        PhotosPicker(selection: $selectedItem, matching: .images) {
                            Image(systemName: "photo.badge.plus")
                                .font(.body).foregroundColor(.white)
                        }
                    }
                    .foregroundColor(.white)
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 10)
                .background(.ultraThinMaterial)
            }

            if isProcessing {
                ProgressView().tint(.white).scaleEffect(1.5)
                    .padding(24)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
            }
        }
        .onChange(of: selectedItem) { _ in loadAndDetect() }
        .onTapGesture { UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil) }
    }

    private func loadAndDetect() {
        guard let selectedItem else { return }
        isProcessing = true
        Task {
            if let data = try? await selectedItem.loadTransferable(type: Data.self),
               let uiImage = UIImage(data: data) {
                detector.updateQueries(queryText)
                let start = CFAbsoluteTimeGetCurrent()
                let dets = detector.detectSync(image: uiImage)
                let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
                await MainActor.run {
                    image = uiImage
                    allDetections = dets
                    inferenceTime = elapsed
                    isProcessing = false
                }
            } else {
                await MainActor.run { isProcessing = false }
            }
        }
    }

    private func runDetection() {
        guard let image else { return }
        isProcessing = true
        detector.updateQueries(queryText)
        Task {
            let start = CFAbsoluteTimeGetCurrent()
            let dets = detector.detectSync(image: image)
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
            await MainActor.run {
                allDetections = dets
                inferenceTime = elapsed
                isProcessing = false
            }
        }
    }
}

// MARK: - Video Detection

struct VideoDetectionView: View {
    let detector: TextGroundingDetector
    @Binding var threshold: Float
    @State private var selectedItem: PhotosPickerItem?
    @State private var currentFrame: UIImage?
    @State private var detections: [Detection] = []
    @State private var progress: Double = 0
    @State private var fps: Double = 0
    @State private var playbackTask: Task<Void, Never>?
    @State private var queryText = "person, dog, car"

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
                        Text("Tap to select a video").font(.subheadline)
                    }
                    .foregroundStyle(.secondary)
                }
            }

            VStack {
                Spacer()
                VStack(spacing: 8) {
                    if currentFrame != nil {
                        ProgressView(value: progress).tint(.blue)
                    }
                    HStack(spacing: 8) {
                        TextField("Objects (comma-separated)", text: $queryText)
                            .textFieldStyle(.roundedBorder)
                            .onSubmit { detector.updateQueries(queryText) }
                            .submitLabel(.search)
                        Button { detector.updateQueries(queryText) } label: {
                            Image(systemName: "magnifyingglass")
                                .foregroundColor(.white)
                                .padding(6)
                                .background(Color.blue, in: Circle())
                        }
                    }
                    HStack(spacing: 4) {
                        Text(String(format: "%.0f%%", threshold * 100))
                            .font(.caption).monospacedDigit().frame(width: 36)
                        Slider(value: $threshold, in: 0.05...0.95, step: 0.05)
                            .onChange(of: threshold) { val in detector.confidenceThreshold = val }
                    }
                    HStack {
                        if !detections.isEmpty {
                            Text("\(detections.count) objects").font(.caption)
                        }
                        Spacer()
                        if fps > 0 {
                            Text(String(format: "%.1f FPS", fps))
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        PhotosPicker(selection: $selectedItem, matching: .videos) {
                            Image(systemName: "video.badge.plus")
                                .font(.body).foregroundColor(.white)
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
        .onTapGesture { UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil) }
    }

    private func loadAndProcess() {
        playbackTask?.cancel()
        guard let selectedItem else { return }
        detector.updateQueries(queryText)

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

        guard let reader = try? AVAssetReader(asset: asset) else { return }
        let outputSettings: [String: Any] = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        let trackOutput = AVAssetReaderTrackOutput(track: track, outputSettings: outputSettings)
        reader.add(trackOutput)
        reader.startReading()

        let ciContext = CIContext()
        let nominalFPS = (try? await track.load(.nominalFrameRate)) ?? 30
        let frameInterval = 1.0 / Double(nominalFPS)

        while !Task.isCancelled, let sb = trackOutput.copyNextSampleBuffer() {
            let pts = CMSampleBufferGetPresentationTimeStamp(sb)
            let currentSec = CMTimeGetSeconds(pts)

            guard let pb = CMSampleBufferGetImageBuffer(sb) else { continue }
            let ciImage = CIImage(cvPixelBuffer: pb)
            guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else { continue }
            let frame = UIImage(cgImage: cgImage)

            let start = CFAbsoluteTimeGetCurrent()
            let dets = detector.detectSync(image: frame)
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            let currentFPS = 1.0 / max(elapsed, 0.001)

            await MainActor.run {
                currentFrame = frame
                detections = dets
                progress = currentSec / totalSeconds
                fps = fps == 0 ? currentFPS : fps * 0.9 + currentFPS * 0.1
            }

            let sleepTime = max(frameInterval - elapsed, 0)
            if sleepTime > 0 { try? await Task.sleep(for: .seconds(sleepTime)) }
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
            let tmp = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString + "." + received.file.pathExtension)
            try FileManager.default.copyItem(at: received.file, to: tmp)
            return Self(url: tmp)
        }
    }
}

// MARK: - Camera Detection (UIKit — CALayer pool)

struct CameraDetectionView: View {
    let detector: TextGroundingDetector
    @Binding var threshold: Float
    var body: some View {
        CameraVCWrapper(detector: detector, threshold: $threshold)
            .ignoresSafeArea(edges: .bottom)
    }
}

struct CameraVCWrapper: UIViewControllerRepresentable {
    let detector: TextGroundingDetector
    @Binding var threshold: Float
    func makeUIViewController(context: Context) -> CameraVC { CameraVC(detector: detector) }
    func updateUIViewController(_ vc: CameraVC, context: Context) {
        detector.confidenceThreshold = threshold
    }
}

class CameraVC: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let detector: TextGroundingDetector
    private let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "session")
    private let inferenceQueue = DispatchQueue(label: "inference")
    private var previewLayer: AVCaptureVideoPreviewLayer!
    private var boxViews: [BoundingBoxView] = []
    private var isProcessing = false
    private var longSide: CGFloat = 1920
    private var shortSide: CGFloat = 1080
    private var frameSizeCaptured = false

    private var smoothedMs: Double = 0
    private var smoothedFps: Double = 0
    private let statsLabel = CATextLayer()

    // Query UI
    private let queryField = UITextField()
    private let queryBar = UIVisualEffectView(effect: UIBlurEffect(style: .dark))

    init(detector: TextGroundingDetector) {
        self.detector = detector
        super.init(nibName: nil, bundle: nil)
    }
    required init?(coder: NSCoder) { fatalError() }

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black

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
        view.layer.addSublayer(statsLabel)

        // Query bar
        queryBar.layer.cornerRadius = 12
        queryBar.clipsToBounds = true
        queryBar.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(queryBar)

        queryField.placeholder = "Objects (comma-separated)"
        queryField.text = "person, dog, car"
        queryField.borderStyle = .roundedRect
        queryField.font = .systemFont(ofSize: 14)
        queryField.returnKeyType = .search
        queryField.autocorrectionType = .no
        queryField.delegate = self
        queryField.translatesAutoresizingMaskIntoConstraints = false
        queryBar.contentView.addSubview(queryField)

        NSLayoutConstraint.activate([
            queryBar.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 12),
            queryBar.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -12),
            queryBar.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -8),
            queryBar.heightAnchor.constraint(equalToConstant: 48),
            queryField.leadingAnchor.constraint(equalTo: queryBar.contentView.leadingAnchor, constant: 8),
            queryField.trailingAnchor.constraint(equalTo: queryBar.contentView.trailingAnchor, constant: -8),
            queryField.centerYAnchor.constraint(equalTo: queryBar.contentView.centerYAnchor),
        ])

        detector.updateQueries(queryField.text ?? "")

        AVCaptureDevice.requestAccess(for: .video) { [weak self] ok in
            guard ok else { return }
            self?.sessionQueue.async { self?.setupCamera() }
        }
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer.frame = view.bounds
        statsLabel.frame = CGRect(
            x: (view.bounds.width - 220) / 2,
            y: view.safeAreaInsets.top + 8,
            width: 220, height: 28
        )
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
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
        let dets = detector.detectSync(pixelBuffer: pb)
        let ms = (CACurrentMediaTime() - start) * 1000
        isProcessing = false

        smoothedMs = smoothedMs == 0 ? ms : smoothedMs * 0.8 + ms * 0.2
        smoothedFps = smoothedFps == 0 ? 1000/ms : smoothedFps * 0.8 + (1000/ms) * 0.2

        // Convert to Vision-style rects for display
        let visionDets = dets.map { d in
            (d.label, d.confidence, d.classIndex,
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

    private func showBoxes(_ dets: [(String, Float, Int, CGRect)]) {
        let width = view.bounds.width
        let height = view.bounds.height
        let ratio = (height / width) / (longSide / shortSide)

        for i in 0..<boxViews.count {
            guard i < dets.count && i < 50 else { boxViews[i].hide(); continue }
            let (label, conf, cid, nr) = dets[i]
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
            let color = detector.colors[cid % detector.colors.count]
            let text = String(format: "%@ %.0f%%", label, conf * 100)
            let alpha = CGFloat(max(conf - 0.2, 0.1) / 0.8 * 0.9)
            boxViews[i].show(frame: screenRect, label: text, color: color, alpha: alpha)
        }
    }
}

extension CameraVC: UITextFieldDelegate {
    func textFieldShouldReturn(_ textField: UITextField) -> Bool {
        textField.resignFirstResponder()
        detector.updateQueries(textField.text ?? "")
        return true
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

// MARK: - Text Grounding Detector

class TextGroundingDetector: ObservableObject {
    @Published var isModelLoaded = false

    let colors: [UIColor] = [
        .systemRed, .systemGreen, .systemBlue, .systemOrange,
        .systemPurple, .systemYellow, .systemPink, .systemCyan,
    ]

    private var visualModel: MLModel?
    private var textEncoder: MLModel?
    private var tokenizer: CLIPTokenizer?

    private let maxClasses = 80
    private let inputSize = 640
    var confidenceThreshold: Float = 0.15
    private let nmsThreshold: Float = 0.5
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    private var imageArray: MLMultiArray?
    private var cachedQueryString = ""
    private var cachedQueries: [String] = []
    private var cachedTxtFeats: MLMultiArray?

    init() {
        loadModels()
    }

    private func loadModels() {
        do {
            guard let d = Bundle.main.url(forResource: "yoloworld_detector", withExtension: "mlmodelc"),
                  let e = Bundle.main.url(forResource: "clip_text_encoder", withExtension: "mlmodelc"),
                  let v = Bundle.main.url(forResource: "clip_vocab", withExtension: "json") else { return }
            let config = MLModelConfiguration()
            config.computeUnits = .all
            visualModel = try MLModel(contentsOf: d, configuration: config)
            textEncoder = try MLModel(contentsOf: e, configuration: config)
            tokenizer = try CLIPTokenizer(vocabularyURL: v)
            DispatchQueue.main.async { self.isModelLoaded = true }
        } catch {
            print("[TextGrounding] Model load failed: \(error)")
        }
    }

    // MARK: - Text Encoding

    func updateQueries(_ queryString: String) {
        guard queryString != cachedQueryString else { return }
        cachedQueryString = queryString

        let queries = queryString.split(separator: ",").map {
            $0.trimmingCharacters(in: .whitespaces)
        }.filter { !$0.isEmpty }

        guard !queries.isEmpty, let textEncoder = textEncoder, let tokenizer = tokenizer else {
            cachedQueries = []; cachedTxtFeats = nil; return
        }
        cachedQueries = queries

        do {
            let txtFeats = try MLMultiArray(shape: [1, maxClasses as NSNumber, 512], dataType: .float32)
            let ptr = txtFeats.dataPointer.bindMemory(to: Float32.self, capacity: maxClasses * 512)
            memset(ptr, 0, maxClasses * 512 * 4)

            for (i, query) in queries.prefix(maxClasses).enumerated() {
                let tokenArray = try MLMultiArray(
                    shape: [maxClasses as NSNumber, tokenizer.contextLength as NSNumber], dataType: .int32)
                let tokenPtr = tokenArray.dataPointer.bindMemory(to: Int32.self, capacity: maxClasses * tokenizer.contextLength)
                memset(tokenPtr, 0, maxClasses * tokenizer.contextLength * 4)
                let tokens = tokenizer.tokenize(query)
                for j in 0..<tokenizer.contextLength { tokenPtr[j] = Int32(tokens[j]) }

                let input = try MLDictionaryFeatureProvider(dictionary: ["text_tokens": tokenArray])
                let output = try textEncoder.prediction(from: input)
                guard let embeddings = output.featureValue(for: "text_embeddings")?.multiArrayValue else { continue }

                var embedding = [Float](repeating: 0, count: 512)
                let emb = readFloat(embeddings)
                for j in 0..<512 { embedding[j] = emb[j] }

                var norm: Float = 0
                vDSP_svesq(embedding, 1, &norm, vDSP_Length(512))
                norm = sqrt(norm)
                if norm > 1e-8 {
                    var s = 1.0 / norm
                    vDSP_vsmul(embedding, 1, &s, &embedding, 1, vDSP_Length(512))
                }
                for j in 0..<512 { ptr[i * 512 + j] = embedding[j] }
            }
            cachedTxtFeats = txtFeats
        } catch {
            print("[TextGrounding] Encode failed: \(error)")
        }
    }

    // MARK: - Sync Detection (for camera / photo / video)

    func detectSync(pixelBuffer: CVPixelBuffer) -> [Detection] {
        guard let cgImage = cgImageFromPixelBuffer(pixelBuffer) else { return [] }
        return runDetection(cgImage: cgImage)
    }

    func detectSync(image: UIImage) -> [Detection] {
        guard let cgImage = image.cgImage else { return [] }
        return runDetection(cgImage: cgImage)
    }

    private func cgImageFromPixelBuffer(_ pb: CVPixelBuffer) -> CGImage? {
        let ci = CIImage(cvPixelBuffer: pb)
        return ciContext.createCGImage(ci, from: ci.extent)
    }

    // MARK: - Core Detection

    private func runDetection(cgImage: CGImage) -> [Detection] {
        guard let visualModel = visualModel, let txtFeats = cachedTxtFeats, !cachedQueries.isEmpty else { return [] }

        do {
            let (tensor, imgW, imgH, padX, padY, scale) = try preprocessImage(cgImage)

            let input = try MLDictionaryFeatureProvider(dictionary: [
                "image": tensor, "txt_feats": txtFeats,
            ])
            let output = try visualModel.prediction(from: input)

            guard let boxesMA = output.featureValue(for: "boxes")?.multiArrayValue,
                  let scoresMA = output.featureValue(for: "scores")?.multiArrayValue else { return [] }

            let boxes = readFloat(boxesMA)
            let scores = readFloat(scoresMA)
            let shape = scoresMA.shape.map { $0.intValue }
            let numClasses = shape.count >= 2 ? shape[1] : maxClasses
            let numAnchors = shape.count >= 3 ? shape[2] : 8400

            var allDets: [(CGRect, Float, Int)] = []

            for qi in 0..<min(cachedQueries.count, numClasses) {
                let off = qi * numAnchors
                for a in 0..<numAnchors {
                    let score = scores[off + a]
                    guard score >= confidenceThreshold else { continue }

                    let cx = boxes[0 * numAnchors + a]
                    let cy = boxes[1 * numAnchors + a]
                    let bw = boxes[2 * numAnchors + a]
                    let bh = boxes[3 * numAnchors + a]

                    let nx = (cx - bw / 2 - padX) / (Float(imgW) * scale)
                    let ny = (cy - bh / 2 - padY) / (Float(imgH) * scale)
                    let nw = bw / (Float(imgW) * scale)
                    let nh = bh / (Float(imgH) * scale)

                    let rect = CGRect(
                        x: CGFloat(max(0, min(1, nx))),
                        y: CGFloat(max(0, min(1, ny))),
                        width: CGFloat(max(0, min(1, nw))),
                        height: CGFloat(max(0, min(1, nh)))
                    )
                    allDets.append((rect, score, qi))
                }
            }

            // NMS
            allDets.sort { $0.1 > $1.1 }
            var kept: [Int] = []
            for i in allDets.indices {
                var suppress = false
                for ki in kept {
                    if allDets[i].2 == allDets[ki].2 && iou(allDets[i].0, allDets[ki].0) > nmsThreshold {
                        suppress = true; break
                    }
                }
                if !suppress { kept.append(i) }
            }

            return kept.prefix(20).map { i in
                Detection(label: cachedQueries[allDets[i].2],
                          confidence: allDets[i].1,
                          classIndex: allDets[i].2,
                          normRect: allDets[i].0)
            }
        } catch {
            return []
        }
    }

    // MARK: - Preprocessing

    private func preprocessImage(_ cgImage: CGImage) throws
        -> (MLMultiArray, Int, Int, Float, Float, Float)
    {
        let imgW = cgImage.width, imgH = cgImage.height
        let scale = Float(inputSize) / Float(max(imgW, imgH))
        let scaledW = Int(Float(imgW) * scale)
        let scaledH = Int(Float(imgH) * scale)
        let padX = (inputSize - scaledW) / 2
        let padY = (inputSize - scaledH) / 2

        guard let ctx = CGContext(
            data: nil, width: inputSize, height: inputSize,
            bitsPerComponent: 8, bytesPerRow: inputSize * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else { throw NSError(domain: "Preprocess", code: 1) }

        ctx.setFillColor(gray: 0.5, alpha: 1.0)
        ctx.fill(CGRect(x: 0, y: 0, width: inputSize, height: inputSize))
        ctx.draw(cgImage, in: CGRect(x: padX, y: padY, width: scaledW, height: scaledH))

        guard let pixels = ctx.data else { throw NSError(domain: "Preprocess", code: 2) }

        if imageArray == nil {
            imageArray = try MLMultiArray(
                shape: [1, 3, inputSize as NSNumber, inputSize as NSNumber], dataType: .float32)
        }
        let dst = imageArray!.dataPointer.bindMemory(to: Float32.self, capacity: 3 * inputSize * inputSize)
        let src = pixels.bindMemory(to: UInt8.self, capacity: inputSize * inputSize * 4)
        let hw = inputSize * inputSize
        let inv: Float = 1.0 / 255.0
        for i in 0..<hw {
            dst[0 * hw + i] = Float(src[i * 4 + 0]) * inv
            dst[1 * hw + i] = Float(src[i * 4 + 1]) * inv
            dst[2 * hw + i] = Float(src[i * 4 + 2]) * inv
        }

        return (imageArray!, imgW, imgH, Float(padX), Float(padY), scale)
    }

    // MARK: - Helpers

    private func iou(_ a: CGRect, _ b: CGRect) -> Float {
        let interX = max(0, min(a.maxX, b.maxX) - max(a.minX, b.minX))
        let interY = max(0, min(a.maxY, b.maxY) - max(a.minY, b.minY))
        let inter = Float(interX * interY)
        let union = Float(a.width * a.height) + Float(b.width * b.height) - inter
        return union > 0 ? inter / union : 0
    }

    private func readFloat(_ array: MLMultiArray) -> [Float] {
        let count = array.count
        var result = [Float](repeating: 0, count: count)
        let ptr = array.dataPointer.assumingMemoryBound(to: Float32.self)
        for i in 0..<count { result[i] = ptr[i] }
        return result
    }
}

#Preview { ContentView() }
