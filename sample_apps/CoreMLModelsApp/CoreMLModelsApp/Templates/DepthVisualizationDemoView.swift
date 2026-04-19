import SwiftUI
import PhotosUI
import CoreML
import AVFoundation

struct DepthVisualizationDemoView: View {
    let model: ModelEntry

    enum Mode: String, CaseIterable, Identifiable {
        case camera = "Camera", video = "Video", photo = "Photo"
        var id: String { rawValue }
    }

    @State private var mode: Mode = .photo

    // Photo-mode state
    @State private var inputImage: UIImage?
    @State private var depthImage: UIImage?
    @State private var normalImage: UIImage?
    @State private var confidenceImage: UIImage?
    @State private var depthRange: (min: Float, max: Float) = (0, 0)
    @State private var processingTime: Double?
    @State private var viewMode: ViewMode = .depth
    @State private var isProcessing = false
    @State private var status = ""
    @State private var item: PhotosPickerItem?
    // Aspect ratio of the source image/frame. The model takes a square
    // stretch-resized input and emits square outputs; we replay the source
    // aspect at display time so the depth/normal/confidence views line up
    // with the original and neither preview shows black letterbox bands.
    @State private var sourceAspect: CGFloat = 1.0

    // Camera-mode state. `mlModel` is cached on load so the per-frame
    // callback can run prediction without awaiting the session.
    @State private var mlModel: MLModel?
    @State private var liveDepth: UIImage?
    @State private var liveFps: Double = 0
    @State private var isInferring = false
    @State private var frameSkip = 0

    // Video-mode state
    @State private var videoItem: PhotosPickerItem?
    @State private var videoProgress: Double = 0
    @State private var videoTask: Task<Void, Never>?

    @StateObject private var session = ModelSession<MLModel>()

    enum ViewMode: String, CaseIterable, Identifiable {
        case original, depth, normal, confidence
        var id: String { rawValue }
        var label: String { rawValue.capitalized }
    }

    private var availableModes: [ViewMode] {
        var modes: [ViewMode] = [.original, .depth]
        if normalImage != nil { modes.append(.normal) }
        if confidenceImage != nil { modes.append(.confidence) }
        return modes
    }

    private var currentDisplayImage: UIImage? {
        switch viewMode {
        case .original: return inputImage
        case .depth: return depthImage
        case .normal: return normalImage
        case .confidence: return confidenceImage
        }
    }

    // "meters", "relative", or nil. Controls the range label unit suffix.
    private var depthUnit: String {
        model.configString("depth_unit") ?? "meters"
    }

    var body: some View {
        VStack(spacing: 0) {
            Picker("Mode", selection: $mode) {
                ForEach(Mode.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .padding(.horizontal).padding(.top, 4)

            ZStack {
                switch mode {
                case .camera: cameraContent
                case .video: videoContent
                case .photo: photoContent
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            modeControls
        }
        .task { await loadModel() }
        .onChange(of: item) { _, _ in loadAndRun() }
        .onChange(of: videoItem) { _, _ in loadAndProcessVideo() }
        .onChange(of: mode) { _, newMode in
            liveDepth = nil
            liveFps = 0
            frameSkip = 0
            if newMode != .video { videoTask?.cancel() }
        }
        .onDisappear {
            videoTask?.cancel()
            mlModel = nil
        }
    }

    @ViewBuilder
    private var modeControls: some View {
        switch mode {
        case .photo: photoControls
        case .camera: cameraControls
        case .video: videoControls
        }
    }

    // MARK: - Photo mode

    @ViewBuilder
    private var photoContent: some View {
        VStack(spacing: 0) {
            displayArea.frame(maxHeight: .infinity)

            if depthImage != nil {
                Picker("View", selection: $viewMode) {
                    ForEach(availableModes) { Text($0.label).tag($0) }
                }
                .pickerStyle(.segmented)
                .padding(.horizontal)
            }
        }
    }

    @ViewBuilder
    private var displayArea: some View {
        if let img = currentDisplayImage {
            Image(uiImage: img).resizable()
                .aspectRatio(sourceAspect, contentMode: .fit)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            VStack(spacing: 12) {
                Image(systemName: "cube.transparent").font(.system(size: 60)).foregroundStyle(.secondary)
                Text("Select a photo to estimate depth + surface normals")
                    .multilineTextAlignment(.center).foregroundStyle(.secondary).padding(.horizontal, 24)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    @ViewBuilder
    private var photoControls: some View {
        VStack(spacing: 8) {
            HStack {
                if depthRange.max > 0 {
                    let format = depthUnit == "meters"
                        ? "Depth: %.2f – %.2f m"
                        : "Depth (relative): %.2f – %.2f"
                    Text(String(format: format, depthRange.min, depthRange.max))
                        .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                }
                Spacer()
                TimingsLabel(loadSec: session.loadTimeSec, inferSec: processingTime)
            }
            .padding(.horizontal)

            if isProcessing { ProgressView(status) }
            if !status.isEmpty { Text(status).font(.caption).foregroundStyle(.secondary) }

            HStack(spacing: 12) {
                PhotosPicker(selection: $item, matching: .images) {
                    Label("Select Photo", systemImage: "photo.badge.plus").frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)

                if viewMode != .original, let img = currentDisplayImage {
                    Button {
                        UIImageWriteToSavedPhotosAlbum(img, nil, nil, nil)
                    } label: {
                        Image(systemName: "arrow.down.to.line")
                    }
                    .buttonStyle(.bordered)
                }
            }
        }
        .padding()
    }

    // MARK: - Camera mode

    @ViewBuilder
    private var cameraContent: some View {
        ZStack {
            CameraView(position: .back) { pb in
                processCameraFrame(pb)
            }
            .opacity(liveDepth == nil ? 1 : 0.001)

            if let depth = liveDepth {
                Image(uiImage: depth).resizable()
                    .aspectRatio(sourceAspect, contentMode: .fit)
            }

            if mlModel == nil {
                ProgressView("Loading model…").tint(.white).padding(16)
                    .background(.ultraThinMaterial).clipShape(RoundedRectangle(cornerRadius: 12))
            }
        }
    }

    @ViewBuilder
    private var cameraControls: some View {
        HStack {
            Text(String(format: "%.1f FPS", liveFps))
                .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            Spacer()
            TimingsLabel(loadSec: session.loadTimeSec, inferSec: nil)
        }
        .padding(.horizontal).padding(.vertical, 8)
    }

    // MARK: - Video mode

    @ViewBuilder
    private var videoContent: some View {
        ZStack {
            if let depth = liveDepth {
                Image(uiImage: depth).resizable()
                    .aspectRatio(sourceAspect, contentMode: .fit)
            } else {
                VStack(spacing: 12) {
                    Image(systemName: "film").font(.system(size: 60)).foregroundStyle(.secondary)
                    Text("Select a video to run depth frame-by-frame").foregroundStyle(.secondary)
                }
            }
        }
    }

    @ViewBuilder
    private var videoControls: some View {
        VStack(spacing: 8) {
            HStack {
                Text(String(format: "%.1f FPS", liveFps))
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                Spacer()
                TimingsLabel(loadSec: session.loadTimeSec, inferSec: nil)
            }
            .padding(.horizontal)

            if liveDepth != nil {
                ProgressView(value: videoProgress).tint(.blue).padding(.horizontal)
            }

            PhotosPicker(selection: $videoItem, matching: .videos) {
                Label("Select Video", systemImage: "video.badge.plus").frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .padding(.horizontal)
        }
        .padding(.vertical, 8)
    }

    // MARK: - Model loading

    private func loadModel() async {
        session.ensure { try await ModelLoader.loadPrimary(for: model) }
        do {
            let loaded = try await session.get()
            await MainActor.run { mlModel = loaded }
        } catch {
            await MainActor.run { status = "Load failed: \(error.localizedDescription)" }
        }
    }

    // MARK: - Camera frame pipeline

    // Runs on the CameraView output queue. Skips when an inference is already
    // in flight so frames don't pipeline and blow memory.
    private func processCameraFrame(_ pb: CVPixelBuffer) {
        guard let mlModel, !isInferring else { return }
        frameSkip += 1
        guard frameSkip % 2 == 0 else { return }
        isInferring = true

        let inputSize = model.configInt("input_size") ?? 504
        let aspect = CGFloat(CVPixelBufferGetWidth(pb)) / CGFloat(max(CVPixelBufferGetHeight(pb), 1))

        Task.detached(priority: .userInitiated) {
            let start = CFAbsoluteTimeGetCurrent()
            let heatmap = runLiveDepth(pixelBuffer: pb, mlModel: mlModel, inputSize: inputSize)
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            await MainActor.run {
                if heatmap != nil { liveDepth = heatmap }
                sourceAspect = aspect
                let fps = 1.0 / max(elapsed, 0.001)
                liveFps = liveFps == 0 ? fps : liveFps * 0.9 + fps * 0.1
                isInferring = false
            }
        }
    }

    // MARK: - Photo inference

    private func loadAndRun() {
        guard let item else { return }
        isProcessing = true; status = "Loading photo…"
        Task {
            do {
                guard let data = try await item.loadTransferable(type: Data.self),
                      let img = UIImage(data: data) else {
                    await MainActor.run { isProcessing = false; status = "Invalid image data" }
                    return
                }
                await MainActor.run {
                    inputImage = img
                    // Pre-set so the freshly-shown input isn't briefly rendered
                    // with the previous photo's aspect ratio.
                    sourceAspect = img.size.width / max(img.size.height, 1)
                }
                await runDepth(on: img)
            } catch {
                await MainActor.run { isProcessing = false; status = "Load error: \(error.localizedDescription)" }
            }
        }
    }

    private func runDepth(on image: UIImage) async {
        await MainActor.run { status = session.isLoading ? "Loading model…" : "Running inference…" }
        do {
            let mlModel = try await session.get()
            await MainActor.run { status = "Running inference…" }

            let inputSize = model.configInt("input_size") ?? 504
            guard let cgImage = ImageUtils.normalizeOrientation(image) else {
                await MainActor.run { isProcessing = false; status = "Image prep failed" }
                return
            }

            guard let pb = ImageUtils.stretchResize(cgImage, size: inputSize) else {
                await MainActor.run { isProcessing = false; status = "Preprocess failed" }
                return
            }
            let aspect = CGFloat(cgImage.width) / CGFloat(max(cgImage.height, 1))
            await MainActor.run { sourceAspect = aspect }

            let inputName = mlModel.modelDescription.inputDescriptionsByName.first {
                $0.value.type == .image
            }?.key ?? "image"

            let start = CFAbsoluteTimeGetCurrent()
            let input = try MLDictionaryFeatureProvider(dictionary: [inputName: pb])
            let output = try await mlModel.prediction(from: input)
            let elapsed = CFAbsoluteTimeGetCurrent() - start

            let depthArr = output.featureValue(for: "depth")?.multiArrayValue
            let normalArr = output.featureValue(for: "normal")?.multiArrayValue
            let maskArr = output.featureValue(for: "mask")?.multiArrayValue
            let scaleArr = output.featureValue(for: "metric_scale")?.multiArrayValue
            let confArr = output.featureValue(for: "confidence")?.multiArrayValue

            let metricScale: Float = scaleArr.map { ImageUtils.readFloat($0, at: 0) } ?? 1.0

            var depthResult: UIImage?
            var dMin: Float = 0, dMax: Float = 0
            if let depthArr {
                let (heatmap, minV, maxV) = buildDepthHeatmap(
                    depthArr: depthArr, maskArr: maskArr, metricScale: metricScale
                )
                depthResult = heatmap
                dMin = minV; dMax = maxV
            }

            var normalResult: UIImage?
            if let normalArr { normalResult = ImageUtils.normalMapImage(normalArr) }

            var confResult: UIImage?
            if let confArr { confResult = buildConfidenceHeatmap(confArr) }

            await MainActor.run {
                depthImage = depthResult
                normalImage = normalResult
                confidenceImage = confResult
                depthRange = (dMin, dMax)
                processingTime = elapsed
                if viewMode == .normal && normalResult == nil { viewMode = .depth }
                if viewMode == .confidence && confResult == nil { viewMode = .depth }
                isProcessing = false; status = ""
            }
        } catch {
            await MainActor.run { isProcessing = false; status = "Error: \(error.localizedDescription)" }
        }
    }

    // MARK: - Video pipeline

    private func loadAndProcessVideo() {
        videoTask?.cancel()
        guard let videoItem else { return }
        liveDepth = nil; liveFps = 0; videoProgress = 0
        Task {
            guard let transferable = try? await videoItem.loadTransferable(type: DepthVideoTransferable.self) else {
                await MainActor.run { status = "Failed to load video" }
                return
            }
            let url = transferable.url
            let inputSize = model.configInt("input_size") ?? 504
            guard let mlModel else { return }
            videoTask = Task.detached(priority: .userInitiated) {
                await runDepthVideo(url: url, mlModel: mlModel, inputSize: inputSize) { frame, progress, fps, aspect in
                    Task { @MainActor in
                        liveDepth = frame
                        sourceAspect = aspect
                        videoProgress = progress
                        liveFps = liveFps == 0 ? fps : liveFps * 0.9 + fps * 0.1
                    }
                }
                await MainActor.run { videoProgress = 1.0 }
            }
        }
    }

    // MARK: - Heatmap helpers (shared between photo and camera)

    private func buildDepthHeatmap(
        depthArr: MLMultiArray, maskArr: MLMultiArray?, metricScale: Float
    ) -> (UIImage?, Float, Float) {
        let shape = depthArr.shape.map { $0.intValue }
        let strides = depthArr.strides.map { $0.intValue }
        let h = shape.count == 3 ? shape[1] : shape[2]
        let w = shape.count == 3 ? shape[2] : shape[3]
        let hS = shape.count == 3 ? strides[1] : strides[2]
        let wS = shape.count == 3 ? strides[2] : strides[3]

        var depthValues = [Float](repeating: 0, count: h * w)
        for y in 0..<h {
            for x in 0..<w {
                var v = ImageUtils.readFloat(depthArr, at: y * hS + x * wS)
                v *= metricScale
                if let maskArr {
                    let mv = ImageUtils.readFloat(maskArr, at: y * hS + x * wS)
                    if mv < 0.5 { v = 0 }
                }
                depthValues[y * w + x] = v
            }
        }
        let dMin = depthValues.filter { $0 > 0 }.min() ?? 0
        let dMax = depthValues.filter { $0 > 0 }.max() ?? 0
        let heatmap = ImageUtils.heatmapFromDepth(depthValues, width: w, height: h)
        return (heatmap, dMin, dMax)
    }

    private func buildConfidenceHeatmap(_ confArr: MLMultiArray) -> UIImage? {
        let shape = confArr.shape.map { $0.intValue }
        let strides = confArr.strides.map { $0.intValue }
        let h = shape.count == 3 ? shape[1] : shape[2]
        let w = shape.count == 3 ? shape[2] : shape[3]
        let hS = shape.count == 3 ? strides[1] : strides[2]
        let wS = shape.count == 3 ? strides[2] : strides[3]
        var vals = [Float](repeating: 0, count: h * w)
        for y in 0..<h {
            for x in 0..<w {
                vals[y * w + x] = ImageUtils.readFloat(confArr, at: y * hS + x * wS)
            }
        }
        return ImageUtils.heatmapFromDepth(vals, width: w, height: h)
    }
}

// MARK: - Video transferable

struct DepthVideoTransferable: Transferable {
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

// MARK: - Video depth (detached)

// Streams a picked video through the depth model. Cooperative cancellation via
// Task.isCancelled lets the view abort when the user switches modes or picks
// another clip; callback fires on every decoded frame for UI updates.
private func runDepthVideo(
    url: URL,
    mlModel: MLModel,
    inputSize: Int,
    onFrame: @escaping (UIImage, Double, Double, CGFloat) -> Void
) async {
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

    while !Task.isCancelled, let sampleBuffer = trackOutput.copyNextSampleBuffer() {
        let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        let currentSec = CMTimeGetSeconds(pts)
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { continue }

        let start = CFAbsoluteTimeGetCurrent()
        guard let heatmap = runLiveDepth(pixelBuffer: pixelBuffer, mlModel: mlModel, inputSize: inputSize) else { continue }
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let fps = 1.0 / max(elapsed, 0.001)
        let progress = min(currentSec / totalSeconds, 1.0)
        let aspect = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
            / CGFloat(max(CVPixelBufferGetHeight(pixelBuffer), 1))
        onFrame(heatmap, progress, fps, aspect)

        // Pace to source frame rate when inference is faster than playback,
        // so a 30fps clip shows at 30fps instead of tearing through in seconds.
        let sleepTime = max(frameInterval - elapsed, 0)
        if sleepTime > 0 {
            try? await Task.sleep(for: .seconds(sleepTime))
        }
    }
}

// MARK: - Live depth (detached)

// Runs off the main actor so the CameraView queue isn't blocked by CoreImage
// + the Swift heatmap loop. `mlModel.prediction` is thread-safe.
private func runLiveDepth(pixelBuffer: CVPixelBuffer, mlModel: MLModel, inputSize: Int) -> UIImage? {
    let ci = CIImage(cvPixelBuffer: pixelBuffer)
    let ctx = CIContext(options: [.useSoftwareRenderer: false])
    guard let cg = ctx.createCGImage(ci, from: ci.extent) else { return nil }
    guard let pb = ImageUtils.stretchResize(cg, size: inputSize) else { return nil }

    let inputName = mlModel.modelDescription.inputDescriptionsByName.first {
        $0.value.type == .image
    }?.key ?? "image"

    do {
        let input = try MLDictionaryFeatureProvider(dictionary: [inputName: pb])
        let output = try mlModel.prediction(from: input)

        guard let depthArr = output.featureValue(for: "depth")?.multiArrayValue else { return nil }
        let maskArr = output.featureValue(for: "mask")?.multiArrayValue

        let shape = depthArr.shape.map { $0.intValue }
        let strides = depthArr.strides.map { $0.intValue }
        let h = shape.count == 3 ? shape[1] : shape[2]
        let w = shape.count == 3 ? shape[2] : shape[3]
        let hS = shape.count == 3 ? strides[1] : strides[2]
        let wS = shape.count == 3 ? strides[2] : strides[3]

        var depthValues = [Float](repeating: 0, count: h * w)
        for y in 0..<h {
            for x in 0..<w {
                var v = ImageUtils.readFloat(depthArr, at: y * hS + x * wS)
                if let maskArr {
                    let mv = ImageUtils.readFloat(maskArr, at: y * hS + x * wS)
                    if mv < 0.5 { v = 0 }
                }
                depthValues[y * w + x] = v
            }
        }
        return ImageUtils.heatmapFromDepth(depthValues, width: w, height: h)
    } catch {
        return nil
    }
}
