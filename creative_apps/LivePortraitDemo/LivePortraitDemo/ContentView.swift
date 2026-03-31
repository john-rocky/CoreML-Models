import SwiftUI
import UIKit
import CoreML
import PhotosUI
import AVFoundation
import Accelerate

// MARK: - Pipeline Stage

enum PipelineStage: String, CaseIterable, Identifiable {
    case motionExtractor = "Motion Extractor"
    case appearanceExtractor = "Appearance Extractor"
    case warpingNetwork = "Warping Network"
    case spadeGenerator = "SPADE Generator"

    var id: String { rawValue }

    var modelFileName: String {
        switch self {
        case .motionExtractor: return "LivePortrait_MotionExtractor"
        case .appearanceExtractor: return "LivePortrait_AppearanceExtractor"
        case .warpingNetwork: return "LivePortrait_WarpingNetwork"
        case .spadeGenerator: return "LivePortrait_SPADEGenerator"
        }
    }

    var icon: String {
        switch self {
        case .motionExtractor: return "arrow.triangle.branch"
        case .appearanceExtractor: return "person.crop.rectangle"
        case .warpingNetwork: return "wand.and.rays"
        case .spadeGenerator: return "paintbrush.pointed.fill"
        }
    }
}

enum StageStatus: Equatable {
    case pending, running, completed, failed(String)
    var color: Color {
        switch self {
        case .pending: return .gray
        case .running: return .orange
        case .completed: return .green
        case .failed: return .red
        }
    }
}

// MARK: - Motion Parameters

struct MotionInfo {
    var kp: [Float]       // [63] canonical keypoints
    var exp: [Float]      // [63] expression
    var scale: Float
    var t: [Float]        // [3] translation
    var pitchBins: [Float]  // [66]
    var yawBins: [Float]    // [66]
    var rollBins: [Float]   // [66]

    var pitch: Float { headposePredToDegree(pitchBins) }
    var yaw: Float { headposePredToDegree(yawBins) }
    var roll: Float { headposePredToDegree(rollBins) }
    var rotMat: [[Float]] { getRotationMatrix(pitch: pitch, yaw: yaw, roll: roll) }
}

// MARK: - Math Helpers

func headposePredToDegree(_ pred: [Float]) -> Float {
    let maxVal = pred.max() ?? 0
    let exps = pred.map { exp($0 - maxVal) }
    let sum = exps.reduce(0, +)
    let probs = exps.map { $0 / sum }
    var degree: Float = 0
    for i in 0..<66 { degree += probs[i] * Float(i) }
    return degree * 3.0 - 97.5
}

func getRotationMatrix(pitch: Float, yaw: Float, roll: Float) -> [[Float]] {
    let p = pitch * .pi / 180, y = yaw * .pi / 180, r = roll * .pi / 180
    let rx: [[Float]] = [[1,0,0],[0,cos(p),-sin(p)],[0,sin(p),cos(p)]]
    let ry: [[Float]] = [[cos(y),0,sin(y)],[0,1,0],[-sin(y),0,cos(y)]]
    let rz: [[Float]] = [[cos(r),-sin(r),0],[sin(r),cos(r),0],[0,0,1]]
    let zy = matmul3x3(rz, ry)
    let zyx = matmul3x3(zy, rx)
    return transpose3x3(zyx)
}

func matmul3x3(_ a: [[Float]], _ b: [[Float]]) -> [[Float]] {
    var c = [[Float]](repeating: [Float](repeating: 0, count: 3), count: 3)
    for i in 0..<3 { for j in 0..<3 { for k in 0..<3 { c[i][j] += a[i][k] * b[k][j] } } }
    return c
}

func transpose3x3(_ m: [[Float]]) -> [[Float]] {
    var r = [[Float]](repeating: [Float](repeating: 0, count: 3), count: 3)
    for i in 0..<3 { for j in 0..<3 { r[i][j] = m[j][i] } }
    return r
}

/// kp_transformed = scale * (kp @ R + exp) + t (t.z = 0)
func transformKeypoint(kp: [Float], exp: [Float], scale: Float, t: [Float], rotMat: [[Float]]) -> [Float] {
    var result = [Float](repeating: 0, count: 63)
    for i in 0..<21 {
        var rotated = [Float](repeating: 0, count: 3)
        for j in 0..<3 {
            for k in 0..<3 { rotated[j] += kp[i*3+k] * rotMat[k][j] }
        }
        for j in 0..<3 {
            result[i*3+j] = scale * (rotated[j] + exp[i*3+j])
        }
        result[i*3+0] += t[0]
        result[i*3+1] += t[1]
    }
    return result
}

// MARK: - Image / MultiArray Helpers

func imageToMultiArray(_ image: UIImage, size: Int) -> MLMultiArray? {
    guard let resized = image.resized(to: CGSize(width: size, height: size)),
          let cgImage = resized.cgImage else { return nil }

    let bpr = size * 4
    guard let ctx = CGContext(data: nil, width: size, height: size, bitsPerComponent: 8,
                               bytesPerRow: bpr, space: CGColorSpaceCreateDeviceRGB(),
                               bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
    ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: size, height: size))
    guard let data = ctx.data else { return nil }
    let ptr = data.assumingMemoryBound(to: UInt8.self)

    guard let array = try? MLMultiArray(shape: [1, 3, NSNumber(value: size), NSNumber(value: size)], dataType: .float16) else { return nil }
    let count = size * size
    let f16 = array.dataPointer.bindMemory(to: UInt16.self, capacity: 3 * count)

    var rBuf = [Float](repeating: 0, count: count)
    var gBuf = [Float](repeating: 0, count: count)
    var bBuf = [Float](repeating: 0, count: count)
    let scale: Float = 1.0 / 255.0
    for y in 0..<size {
        for x in 0..<size {
            let off = y * bpr + x * 4
            let idx = y * size + x
            rBuf[idx] = Float(ptr[off]) * scale
            gBuf[idx] = Float(ptr[off+1]) * scale
            bBuf[idx] = Float(ptr[off+2]) * scale
        }
    }
    convertF32toF16(rBuf, to: f16, count: count)
    convertF32toF16(gBuf, to: f16.advanced(by: count), count: count)
    convertF32toF16(bBuf, to: f16.advanced(by: 2*count), count: count)
    return array
}

func convertF32toF16(_ src: [Float], to dst: UnsafeMutablePointer<UInt16>, count: Int) {
    src.withUnsafeBufferPointer { srcBuf in
        var srcV = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: srcBuf.baseAddress!),
                                  height: 1, width: vImagePixelCount(count), rowBytes: count * 4)
        var dstV = vImage_Buffer(data: UnsafeMutableRawPointer(dst),
                                  height: 1, width: vImagePixelCount(count), rowBytes: count * 2)
        vImageConvert_PlanarFtoPlanar16F(&srcV, &dstV, 0)
    }
}

func multiArrayToFloat(_ array: MLMultiArray, count: Int) -> [Float] {
    let fp16 = array.dataPointer.bindMemory(to: UInt16.self, capacity: count)
    var result = [Float](repeating: 0, count: count)
    result.withUnsafeMutableBufferPointer { dstBuf in
        var srcV = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: fp16),
                                  height: 1, width: vImagePixelCount(count), rowBytes: count * 2)
        var dstV = vImage_Buffer(data: dstBuf.baseAddress!,
                                  height: 1, width: vImagePixelCount(count), rowBytes: count * 4)
        vImageConvert_Planar16FtoPlanarF(&srcV, &dstV, 0)
    }
    return result
}

func generatedImageToUIImage(_ array: MLMultiArray) -> UIImage? {
    // [1, 3, 512, 512] Float16 → UIImage
    let size = 512
    let count = size * size
    let floats = multiArrayToFloat(array, count: 3 * count)

    var pixels = [UInt8](repeating: 255, count: count * 4)
    for i in 0..<count {
        pixels[i*4]   = UInt8(max(0, min(255, floats[i] * 255)))
        pixels[i*4+1] = UInt8(max(0, min(255, floats[count + i] * 255)))
        pixels[i*4+2] = UInt8(max(0, min(255, floats[2*count + i] * 255)))
        pixels[i*4+3] = 255
    }

    guard let provider = CGDataProvider(data: Data(pixels) as CFData),
          let cgImage = CGImage(width: size, height: size, bitsPerComponent: 8,
                                bitsPerPixel: 32, bytesPerRow: size * 4,
                                space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                                provider: provider, decode: nil, shouldInterpolate: true,
                                intent: .defaultIntent) else { return nil }
    return UIImage(cgImage: cgImage)
}

func flatToMLMultiArray(shape: [NSNumber], values: [Float], dataType: MLMultiArrayDataType = .float16) -> MLMultiArray? {
    guard let array = try? MLMultiArray(shape: shape, dataType: dataType) else { return nil }
    let count = values.count
    let dst = array.dataPointer.bindMemory(to: UInt16.self, capacity: count)
    convertF32toF16(values, to: dst, count: count)
    return array
}

// MARK: - Video Frame Extraction

func extractFrames(from url: URL, maxFrames: Int = 30) async -> [UIImage] {
    let asset = AVURLAsset(url: url)
    guard let track = try? await asset.loadTracks(withMediaType: .video).first,
          let duration = try? await asset.load(.duration) else { return [] }

    let fps = (try? await track.load(.nominalFrameRate)) ?? 30
    let totalSeconds = CMTimeGetSeconds(duration)
    let totalFrameCount = Int(totalSeconds * Double(fps))
    let step = max(1, totalFrameCount / maxFrames)
    let frameCount = min(maxFrames, totalFrameCount)

    let generator = AVAssetImageGenerator(asset: asset)
    generator.appliesPreferredTrackTransform = true
    generator.requestedTimeToleranceBefore = .zero
    generator.requestedTimeToleranceAfter = .zero

    var frames: [UIImage] = []
    for i in 0..<frameCount {
        let timeVal = CMTimeMake(value: Int64(i * step), timescale: Int32(fps))
        if let cgImage = try? generator.copyCGImage(at: timeVal, actualTime: nil) {
            frames.append(UIImage(cgImage: cgImage))
        }
    }
    return frames
}

// MARK: - ContentView

struct ContentView: View {
    @StateObject private var viewModel = LivePortraitViewModel()

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    // Source portrait
                    Section {
                        PhotosPicker(selection: $viewModel.selectedSourcePhoto, matching: .images) {
                            if let image = viewModel.sourceImage {
                                Image(uiImage: image).resizable().scaledToFit()
                                    .frame(maxHeight: 200).cornerRadius(12)
                            } else {
                                placeholderView(title: "Select Source Portrait", systemImage: "person.crop.square")
                            }
                        }
                    } header: { sectionHeader("Source Portrait") }

                    // Driving video
                    Section {
                        PhotosPicker(selection: $viewModel.selectedDrivingVideo, matching: .videos) {
                            if viewModel.drivingVideoURL != nil {
                                HStack {
                                    Image(systemName: "video.fill").font(.title2).foregroundColor(.accentColor)
                                    VStack(alignment: .leading) {
                                        Text("Driving Video Selected").font(.headline)
                                        Text("Tap to change").font(.caption).foregroundColor(.secondary)
                                    }
                                    Spacer()
                                    if let thumb = viewModel.drivingThumbnail {
                                        Image(uiImage: thumb).resizable().scaledToFill()
                                            .frame(width: 60, height: 60).cornerRadius(8)
                                    }
                                }
                                .padding().background(Color(.systemGray6)).cornerRadius(12)
                            } else {
                                placeholderView(title: "Select Driving Video", systemImage: "video.badge.plus")
                            }
                        }
                    } header: { sectionHeader("Driving Video") }

                    // Animate button
                    if viewModel.sourceImage != nil && viewModel.drivingVideoURL != nil {
                        Button(action: { viewModel.runPipeline() }) {
                            HStack {
                                if viewModel.isProcessing { ProgressView().tint(.white) }
                                else { Image(systemName: "play.fill") }
                                Text(viewModel.isProcessing ? "Processing..." : "Animate Portrait")
                            }
                            .frame(maxWidth: .infinity).padding()
                            .background(viewModel.isProcessing ? Color.gray : Color.accentColor)
                            .foregroundColor(.white).cornerRadius(12)
                        }
                        .disabled(viewModel.isProcessing)
                    }

                    // Pipeline stages
                    Section {
                        VStack(spacing: 8) {
                            ForEach(PipelineStage.allCases) { stage in
                                PipelineStageRow(stage: stage,
                                                 status: viewModel.stageStatuses[stage] ?? .pending)
                            }
                        }
                    } header: { sectionHeader("Pipeline") }

                    // Progress
                    if viewModel.isProcessing {
                        Text(viewModel.statusMessage).font(.caption).foregroundColor(.secondary)
                    }

                    if let error = viewModel.errorMessage {
                        Text(error).foregroundColor(.red).font(.caption).padding()
                            .background(Color.red.opacity(0.1)).cornerRadius(8)
                    }

                    // Result - animated playback
                    if !viewModel.resultFrames.isEmpty {
                        Section {
                            VStack(spacing: 12) {
                                Image(uiImage: viewModel.resultFrames[viewModel.currentFrameIndex])
                                    .resizable().scaledToFit()
                                    .frame(maxHeight: 350).cornerRadius(12)

                                HStack {
                                    Button(action: { viewModel.togglePlayback() }) {
                                        Image(systemName: viewModel.isPlaying ? "pause.fill" : "play.fill")
                                    }
                                    Slider(value: Binding(
                                        get: { Double(viewModel.currentFrameIndex) },
                                        set: { viewModel.currentFrameIndex = Int($0) }
                                    ), in: 0...Double(max(1, viewModel.resultFrames.count - 1)), step: 1)
                                    Text("\(viewModel.currentFrameIndex + 1)/\(viewModel.resultFrames.count)")
                                        .font(.caption).monospacedDigit()
                                }
                            }
                        } header: { sectionHeader("Animated Result") }
                    }
                }
                .padding()
            }
            .navigationTitle("LivePortrait")
        }
    }

    private func sectionHeader(_ title: String) -> some View {
        HStack { Text(title).font(.headline); Spacer() }
    }

    private func placeholderView(title: String, systemImage: String) -> some View {
        VStack(spacing: 12) {
            Image(systemName: systemImage).font(.system(size: 40)).foregroundColor(.secondary)
            Text(title).foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity).frame(height: 160)
        .background(Color(.systemGray6)).cornerRadius(12)
    }
}

// MARK: - Pipeline Stage Row

struct PipelineStageRow: View {
    let stage: PipelineStage
    let status: StageStatus
    var body: some View {
        HStack(spacing: 12) {
            ZStack {
                Circle().fill(status.color.opacity(0.2)).frame(width: 32, height: 32)
                if case .running = status { ProgressView().scaleEffect(0.6) }
                else { Image(systemName: statusIcon).font(.caption2).foregroundColor(status.color) }
            }
            VStack(alignment: .leading, spacing: 2) {
                HStack { Image(systemName: stage.icon).font(.caption2); Text(stage.rawValue).font(.caption).fontWeight(.medium) }
                if case .failed(let msg) = status { Text(msg).font(.caption2).foregroundColor(.red) }
            }
            Spacer()
        }
        .padding(8).background(RoundedRectangle(cornerRadius: 8).fill(Color(.systemGray6)))
    }
    private var statusIcon: String {
        switch status {
        case .pending: return "circle"
        case .running: return "arrow.clockwise"
        case .completed: return "checkmark.circle.fill"
        case .failed: return "xmark.circle.fill"
        }
    }
}

// MARK: - ViewModel

class LivePortraitViewModel: ObservableObject {
    @Published var selectedSourcePhoto: PhotosPickerItem? { didSet { loadSourceImage() } }
    @Published var selectedDrivingVideo: PhotosPickerItem? { didSet { loadDrivingVideo() } }
    @Published var sourceImage: UIImage?
    @Published var drivingVideoURL: URL?
    @Published var drivingThumbnail: UIImage?
    @Published var resultFrames: [UIImage] = []
    @Published var currentFrameIndex: Int = 0
    @Published var isPlaying = false
    @Published var isProcessing = false
    @Published var errorMessage: String?
    @Published var statusMessage = ""
    @Published var stageStatuses: [PipelineStage: StageStatus] = [:]

    private var playbackTimer: Timer?

    init() { resetStages() }

    func togglePlayback() {
        if isPlaying {
            playbackTimer?.invalidate()
            playbackTimer = nil
            isPlaying = false
        } else {
            isPlaying = true
            playbackTimer = Timer.scheduledTimer(withTimeInterval: 1.0/15.0, repeats: true) { [weak self] _ in
                guard let self else { return }
                DispatchQueue.main.async {
                    self.currentFrameIndex = (self.currentFrameIndex + 1) % max(1, self.resultFrames.count)
                }
            }
        }
    }

    private func loadSourceImage() {
        guard let item = selectedSourcePhoto else { return }
        Task {
            if let data = try? await item.loadTransferable(type: Data.self),
               let image = UIImage(data: data) {
                await MainActor.run { self.sourceImage = image; self.resultFrames = []; self.resetStages() }
            }
        }
    }

    private func loadDrivingVideo() {
        guard let item = selectedDrivingVideo else { return }
        Task {
            if let videoData = try? await item.loadTransferable(type: Data.self) {
                let tempURL = FileManager.default.temporaryDirectory
                    .appendingPathComponent(UUID().uuidString).appendingPathExtension("mov")
                try? videoData.write(to: tempURL)
                let asset = AVURLAsset(url: tempURL)
                let gen = AVAssetImageGenerator(asset: asset)
                gen.appliesPreferredTrackTransform = true
                let cg = try? gen.copyCGImage(at: .zero, actualTime: nil)
                await MainActor.run {
                    self.drivingVideoURL = tempURL
                    self.drivingThumbnail = cg.map { UIImage(cgImage: $0) }
                    self.resultFrames = []; self.resetStages()
                }
            }
        }
    }

    private func resetStages() {
        for stage in PipelineStage.allCases { stageStatuses[stage] = .pending }
    }

    func runPipeline() {
        guard sourceImage != nil, drivingVideoURL != nil else { return }
        isProcessing = true; errorMessage = nil; resultFrames = []; resetStages()
        Task {
            do {
                try await executePipeline()
                await MainActor.run { self.isProcessing = false; self.statusMessage = "Done" }
            } catch {
                await MainActor.run { self.errorMessage = error.localizedDescription; self.isProcessing = false }
            }
        }
    }

    // MARK: - Full Pipeline

    private func executePipeline() async throws {
        guard let sourceImage, let drivingVideoURL else { return }

        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU

        // Load all models
        func loadModel(_ stage: PipelineStage) throws -> MLModel {
            guard let url = Bundle.main.url(forResource: stage.modelFileName, withExtension: "mlmodelc") else {
                throw LivePortraitError.modelNotFound("\(stage.modelFileName) not found")
            }
            return try MLModel(contentsOf: url, configuration: config)
        }

        let motionModel = try loadModel(.motionExtractor)
        let appearanceModel = try loadModel(.appearanceExtractor)
        let warpingModel = try loadModel(.warpingNetwork)
        let spadeModel = try loadModel(.spadeGenerator)

        // Prepare source image (256x256)
        guard let srcArray = imageToMultiArray(sourceImage, size: 256) else {
            throw LivePortraitError.processingFailed("Failed to preprocess source image")
        }

        // Stage 1: Extract source motion
        await setStageStatus(.motionExtractor, .running)
        await setStatus("Extracting source motion...")

        let srcMotionInput = try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(multiArray: srcArray)])
        let srcMotionOut = try motionModel.prediction(from: srcMotionInput)
        let srcMotion = extractMotionInfo(srcMotionOut)
        let srcR = srcMotion.rotMat
        let kpSource = transformKeypoint(kp: srcMotion.kp, exp: srcMotion.exp, scale: srcMotion.scale,
                                          t: srcMotion.t, rotMat: srcR)

        // Extract driving video frames
        await setStatus("Extracting video frames...")
        let drivingFrames = await extractFrames(from: drivingVideoURL, maxFrames: 30)
        guard !drivingFrames.isEmpty else {
            throw LivePortraitError.processingFailed("Could not extract frames from driving video")
        }

        // Extract motion from first driving frame (reference)
        guard let drv0Array = imageToMultiArray(drivingFrames[0], size: 256) else {
            throw LivePortraitError.processingFailed("Failed to preprocess driving frame 0")
        }
        let drv0Input = try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(multiArray: drv0Array)])
        let drv0Out = try motionModel.prediction(from: drv0Input)
        let drv0Motion = extractMotionInfo(drv0Out)
        let drv0R = drv0Motion.rotMat

        await setStageStatus(.motionExtractor, .completed)

        // Stage 2: Extract source appearance (once)
        await setStageStatus(.appearanceExtractor, .running)
        await setStatus("Extracting appearance features...")

        let appInput = try MLDictionaryFeatureProvider(dictionary: ["source_image": MLFeatureValue(multiArray: srcArray)])
        let appOut = try appearanceModel.prediction(from: appInput)
        guard let feature3d = appOut.featureValue(for: "feature_3d")?.multiArrayValue else {
            throw LivePortraitError.processingFailed("Failed to extract feature_3d")
        }

        await setStageStatus(.appearanceExtractor, .completed)

        // Prepare kp_source as MLMultiArray [1, 21, 3]
        guard let kpSourceArray = flatToMLMultiArray(shape: [1, 21, 3], values: kpSource) else {
            throw LivePortraitError.processingFailed("Failed to create kp_source array")
        }

        // Stage 3 & 4: Process each driving frame
        await setStageStatus(.warpingNetwork, .running)
        await setStageStatus(.spadeGenerator, .running)

        var outputFrames: [UIImage] = []

        for (i, frame) in drivingFrames.enumerated() {
            await setStatus("Frame \(i+1)/\(drivingFrames.count)...")

            // Extract driving motion
            guard let drvArray = imageToMultiArray(frame, size: 256) else { continue }
            let drvInput = try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(multiArray: drvArray)])
            let drvOut = try motionModel.prediction(from: drvInput)
            let drvMotion = extractMotionInfo(drvOut)
            let drvR = drvMotion.rotMat

            // Relative motion: R_new = (R_drv_i @ R_drv_0^T) @ R_src
            let drv0RT = transpose3x3(drv0R)
            let deltaR = matmul3x3(drvR, drv0RT)
            let rNew = matmul3x3(deltaR, srcR)

            // Relative expression, scale, translation
            var expNew = [Float](repeating: 0, count: 63)
            for j in 0..<63 { expNew[j] = srcMotion.exp[j] + (drvMotion.exp[j] - drv0Motion.exp[j]) }
            let scaleNew = srcMotion.scale * (drvMotion.scale / drv0Motion.scale)
            var tNew = [Float](repeating: 0, count: 3)
            tNew[0] = srcMotion.t[0] + (drvMotion.t[0] - drv0Motion.t[0])
            tNew[1] = srcMotion.t[1] + (drvMotion.t[1] - drv0Motion.t[1])
            tNew[2] = 0

            let kpDriving = transformKeypoint(kp: srcMotion.kp, exp: expNew, scale: scaleNew,
                                               t: tNew, rotMat: rNew)

            guard let kpDrivingArray = flatToMLMultiArray(shape: [1, 21, 3], values: kpDriving) else { continue }

            // Warping
            let warpInput = try MLDictionaryFeatureProvider(dictionary: [
                "feature_3d": MLFeatureValue(multiArray: feature3d),
                "kp_driving": MLFeatureValue(multiArray: kpDrivingArray),
                "kp_source": MLFeatureValue(multiArray: kpSourceArray)
            ])
            let warpOut = try warpingModel.prediction(from: warpInput)
            guard let warpedFeature = warpOut.featureValue(for: "warped_feature")?.multiArrayValue else { continue }

            // SPADE Generator
            let spadeInput = try MLDictionaryFeatureProvider(dictionary: [
                "warped_feature": MLFeatureValue(multiArray: warpedFeature)
            ])
            let spadeOut = try spadeModel.prediction(from: spadeInput)
            guard let genImage = spadeOut.featureValue(for: "generated_image")?.multiArrayValue else { continue }

            if let uiImage = generatedImageToUIImage(genImage) {
                outputFrames.append(uiImage)
            }

            // Update UI periodically
            if i % 3 == 0 || i == drivingFrames.count - 1 {
                let frames = outputFrames
                await MainActor.run { self.resultFrames = frames }
            }
        }

        await setStageStatus(.warpingNetwork, .completed)
        await setStageStatus(.spadeGenerator, .completed)

        let finalFrames = outputFrames
        await MainActor.run {
            self.resultFrames = finalFrames
            self.currentFrameIndex = 0
        }
    }

    private func extractMotionInfo(_ output: MLFeatureProvider) -> MotionInfo {
        let pitchArr = output.featureValue(for: "pitch")!.multiArrayValue!
        let yawArr = output.featureValue(for: "yaw")!.multiArrayValue!
        let rollArr = output.featureValue(for: "roll")!.multiArrayValue!
        let tArr = output.featureValue(for: "t")!.multiArrayValue!
        let expArr = output.featureValue(for: "exp")!.multiArrayValue!
        let scaleArr = output.featureValue(for: "scale")!.multiArrayValue!
        let kpArr = output.featureValue(for: "kp")!.multiArrayValue!

        return MotionInfo(
            kp: multiArrayToFloat(kpArr, count: 63),
            exp: multiArrayToFloat(expArr, count: 63),
            scale: multiArrayToFloat(scaleArr, count: 1)[0],
            t: multiArrayToFloat(tArr, count: 3),
            pitchBins: multiArrayToFloat(pitchArr, count: 66),
            yawBins: multiArrayToFloat(yawArr, count: 66),
            rollBins: multiArrayToFloat(rollArr, count: 66)
        )
    }

    @MainActor
    private func setStageStatus(_ stage: PipelineStage, _ status: StageStatus) {
        stageStatuses[stage] = status
    }

    @MainActor
    private func setStatus(_ msg: String) {
        statusMessage = msg
    }
}

// MARK: - Errors

enum LivePortraitError: LocalizedError {
    case modelNotFound(String)
    case processingFailed(String)
    var errorDescription: String? {
        switch self {
        case .modelNotFound(let m): return m
        case .processingFailed(let m): return m
        }
    }
}

// MARK: - UIImage Extension

extension UIImage {
    func resized(to targetSize: CGSize) -> UIImage? {
        let renderer = UIGraphicsImageRenderer(size: targetSize)
        return renderer.image { _ in self.draw(in: CGRect(origin: .zero, size: targetSize)) }
    }
}

#Preview { ContentView() }
