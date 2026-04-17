import SwiftUI
import PhotosUI
import CoreML
import AVFoundation
import AVKit
import Accelerate
import CoreImage
import UIKit
import UniformTypeIdentifiers
import Vision

/// Video matting: video clip → alpha-composited video with an automatically
/// generated first-frame mask.
///
/// Used by: MatAnyone.
///
/// The first-frame mask is produced by `VNGeneratePersonSegmentationRequest`
/// (`.accurate`), binarised at 0.5, then dilated by radius 8 — the same seed
/// pipeline as the stand-alone MatAnyoneDemo. Users can optionally override
/// the mask with a manual image if their subject is non-human.
///
/// Expected manifest config:
/// ```
/// {
///   "encoder": "MatAnyone_encoder.mlpackage",
///   "mask_encoder": "MatAnyone_mask_encoder.mlpackage",
///   "read_first": "MatAnyone_read_first.mlpackage",
///   "read": "MatAnyone_read.mlpackage",
///   "decoder": "MatAnyone_decoder.mlpackage"
/// }
/// ```
///
/// The CoreML graph is locked to landscape 768x432; portrait sources are
/// rotated to landscape before the pipeline and rotated back afterwards.
struct VideoMattingDemoView: View {
    let model: ModelEntry

    @State private var inputVideoURL: URL?
    @State private var outputVideoURL: URL?
    @State private var thumbnail: UIImage?
    @State private var maskPreview: UIImage?
    @State private var overrideMask: UIImage?
    @State private var isProcessing = false
    @State private var progress: Double = 0
    @State private var status = ""
    @State private var processingTime: Double?
    @State private var videoPickerItem: PhotosPickerItem?
    @State private var overrideMaskItem: PhotosPickerItem?
    @State private var backgroundColorChoice: Int = 0
    @StateObject private var session = ModelSession<MatAnyoneHubEngine>()

    // Solid background colours the user can cycle through while previewing.
    private let bgChoices: [(name: String, color: CIColor)] = [
        ("Green", CIColor(red: 0, green: 1, blue: 0)),
        ("Blue",  CIColor(red: 0, green: 0, blue: 1)),
        ("White", CIColor(red: 1, green: 1, blue: 1)),
        ("Black", CIColor(red: 0, green: 0, blue: 0)),
        ("Red",   CIColor(red: 1, green: 0, blue: 0)),
    ]

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 16) {
                VStack(spacing: 4) {
                    if let thumb = thumbnail {
                        Image(uiImage: thumb).resizable().aspectRatio(contentMode: .fit)
                            .frame(maxWidth: 160, maxHeight: 160)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                    } else {
                        RoundedRectangle(cornerRadius: 8).fill(Color(.systemGray6))
                            .frame(width: 160, height: 120)
                            .overlay {
                                Image(systemName: "film").font(.title).foregroundStyle(.tertiary)
                            }
                    }
                    Text("Video").font(.caption2).foregroundStyle(.secondary)
                }

                VStack(spacing: 4) {
                    if let mask = maskPreview {
                        Image(uiImage: mask).resizable().aspectRatio(contentMode: .fit)
                            .frame(maxWidth: 160, maxHeight: 160)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                    } else {
                        RoundedRectangle(cornerRadius: 8).fill(Color(.systemGray6))
                            .frame(width: 160, height: 120)
                            .overlay {
                                Image(systemName: "person.fill.viewfinder").font(.title).foregroundStyle(.tertiary)
                            }
                    }
                    Text("Auto Mask").font(.caption2).foregroundStyle(.secondary)
                }
            }
            .padding()

            if isProcessing {
                VStack(spacing: 8) {
                    ProgressView(value: progress)
                    Text(status).font(.caption).foregroundStyle(.secondary)
                }
                .padding(.horizontal)
            }

            if let url = outputVideoURL {
                VideoPlayerView(url: url)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .padding(.horizontal)
            } else {
                VStack(spacing: 12) {
                    Image(systemName: "person.and.background.dotted")
                        .font(.system(size: 60)).foregroundStyle(.secondary)
                    Text("Pick a video — the subject mask is generated automatically.")
                        .multilineTextAlignment(.center).foregroundStyle(.secondary)
                        .padding(.horizontal, 24)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }

            VStack(spacing: 12) {
                TimingsLabel(loadSec: session.loadTimeSec, inferSec: processingTime)

                // Background colour picker.
                Picker("Background", selection: $backgroundColorChoice) {
                    ForEach(0..<bgChoices.count, id: \.self) { i in
                        Text(bgChoices[i].name).tag(i)
                    }
                }
                .pickerStyle(.segmented)

                HStack(spacing: 12) {
                    PhotosPicker(selection: $videoPickerItem, matching: .videos) {
                        Label("Video", systemImage: "film")
                    }.buttonStyle(.bordered)

                    PhotosPicker(selection: $overrideMaskItem, matching: .images) {
                        Label(overrideMask == nil ? "Override Mask" : "Mask Set", systemImage: "person.crop.rectangle")
                    }.buttonStyle(.bordered)

                    Button {
                        Task { await runMatting() }
                    } label: {
                        Label("Process", systemImage: "wand.and.rays").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isProcessing || inputVideoURL == nil)
                }
            }
            .padding()
        }
        .onChange(of: videoPickerItem) { _, _ in loadPickedVideo() }
        .onChange(of: overrideMaskItem) { _, _ in loadOverrideMask() }
        .task {
            session.ensure { try await MatAnyoneHubEngine(model: model) }
        }
    }

    private func loadPickedVideo() {
        guard let videoPickerItem else { return }
        Task {
            guard let movie = try? await videoPickerItem.loadTransferable(type: PickedMovie.self) else { return }
            await MainActor.run {
                inputVideoURL = movie.url
                outputVideoURL = nil
                processingTime = nil
                maskPreview = nil
                extractThumbnail(from: movie.url)
            }
            // Auto-run as soon as a video is picked — the whole point
            // of the Vision bootstrap is that the user should not need
            // to fiddle with an initial mask.
            await runMatting()
        }
    }

    private func loadOverrideMask() {
        guard let overrideMaskItem else { return }
        Task {
            if let data = try? await overrideMaskItem.loadTransferable(type: Data.self),
               let img = UIImage(data: data) {
                await MainActor.run { overrideMask = img }
            }
        }
    }

    private func extractThumbnail(from url: URL) {
        let asset = AVURLAsset(url: url)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 320, height: 320)
        Task {
            if let cgImage = try? generator.copyCGImage(at: .zero, actualTime: nil) {
                await MainActor.run { thumbnail = UIImage(cgImage: cgImage) }
            }
        }
    }

    private func runMatting() async {
        guard let videoURL = inputVideoURL else { return }
        await MainActor.run {
            isProcessing = true
            progress = 0
            status = "Loading models..."
            outputVideoURL = nil
            processingTime = nil
        }

        let start = CFAbsoluteTimeGetCurrent()
        do {
            let engine = try await session.get()
            await MainActor.run { status = "Running MatAnyone..." }

            let matter = MatAnyoneHubMatter(engine: engine)
            let bg = bgChoices[backgroundColorChoice].color
            let asset = AVURLAsset(url: videoURL)

            let result = try await matter.process(
                asset: asset,
                backgroundColor: bg,
                overrideMask: overrideMask,
                progress: { p in
                    Task { @MainActor in progress = p }
                }
            )

            let elapsed = CFAbsoluteTimeGetCurrent() - start
            await MainActor.run {
                outputVideoURL = result.videoURL
                maskPreview = result.firstFrameMask
                processingTime = elapsed
                isProcessing = false
                status = ""
            }
        } catch {
            await MainActor.run {
                isProcessing = false
                status = "Error: \(error.localizedDescription)"
            }
        }
    }
}

// MARK: - MatAnyone Engine (5-model state machine)

/// Per-frame state machine that drives the 5 split MatAnyone Core ML models.
/// Reproduces the Python `InferenceCore.step` loop in Swift — see
/// `sample_apps/MatAnyoneDemo/MatAnyoneDemo/MatAnyoneEngine.swift` for the
/// original reference. This is the same engine inlined into the hub app so
/// the user only has to pick a video to get a matted result.
final class MatAnyoneHubEngine: @unchecked Sendable {
    // Resolution & topology — must match the converter (locked to 768x432).
    static let inputHeight = 432
    static let inputWidth  = 768
    static let stride      = 16
    static let queryHeight = inputHeight / stride     // 27
    static let queryWidth  = inputWidth  / stride     // 48
    static let queryHW     = queryHeight * queryWidth // 1296
    static let memMaxFrames = 5
    static let memCapacity  = memMaxFrames * queryHW  // 6480
    static let memEvery    = 5
    static let valueDim    = 256
    static let keyDim      = 64
    static let sensoryDim  = 256
    static let querySlots  = 16
    static let summaryDim  = 257

    enum EngineError: LocalizedError {
        case missingModel(String)
        case missingFeature(String)
        var errorDescription: String? {
            switch self {
            case .missingModel(let n): return "Missing MatAnyone module: \(n)"
            case .missingFeature(let n): return "Output \"\(n)\" was not produced"
            }
        }
    }

    // MARK: Models

    private let encoder: MLModel
    private let maskEncoder: MLModel
    private let readFirst: MLModel
    private let read: MLModel
    private let decoder: MLModel

    // MARK: Persistent state

    private var sensory: MLMultiArray
    private var lastMask: MLMultiArray
    private var lastPixFeat: MLMultiArray
    private var lastMskValue: MLMultiArray
    private var objMemory: MLMultiArray
    private var memKey: MLMultiArray
    private var memShrinkage: MLMultiArray
    private var memMskValue: MLMultiArray
    private var memValid: MLMultiArray

    private var currentFrame: Int = -1
    private var lastMemFrame: Int = 0
    private var nextFifoSlot: Int = 1

    // MARK: Init

    init(model: ModelEntry) async throws {
        // encoder / mask_encoder / decoder run fine on .cpuAndGPU.
        // read / read_first ship .cpuOnly — the singleton num_objects slice
        // in those graphs crashes MPS on iOS GPU (see MatAnyoneDemo README).
        //
        // Prefer the manifest config keys, fall back to substring search in
        // the extracted model directory. Config values in the draft manifest
        // have inconsistent casing vs the actual filenames, so substring
        // search is the most robust fallback.
        encoder     = try await Self.loadModule(model: model, configKey: "encoder",
                                                substring: "encoder",
                                                excludeSubstring: "mask", units: .cpuAndGPU)
        maskEncoder = try await Self.loadModule(model: model, configKey: "mask_encoder",
                                                substring: "mask_encoder",
                                                excludeSubstring: nil, units: .cpuAndGPU)
        readFirst   = try await Self.loadModule(model: model, configKey: "read_first",
                                                substring: "read_first",
                                                excludeSubstring: nil, units: .cpuOnly)
        read        = try await Self.loadModule(model: model, configKey: "read",
                                                substring: "read",
                                                excludeSubstring: "first", units: .cpuOnly)
        decoder     = try await Self.loadModule(model: model, configKey: "decoder",
                                                substring: "decoder",
                                                excludeSubstring: nil, units: .cpuAndGPU)

        sensory      = try MLMultiArray(shape: [1, 1, NSNumber(value: Self.sensoryDim), NSNumber(value: Self.queryHeight), NSNumber(value: Self.queryWidth)], dataType: .float32)
        lastMask     = try MLMultiArray(shape: [1, 1, NSNumber(value: Self.inputHeight), NSNumber(value: Self.inputWidth)], dataType: .float32)
        lastPixFeat  = try MLMultiArray(shape: [1, NSNumber(value: Self.valueDim), NSNumber(value: Self.queryHeight), NSNumber(value: Self.queryWidth)], dataType: .float32)
        lastMskValue = try MLMultiArray(shape: [1, 1, NSNumber(value: Self.valueDim), NSNumber(value: Self.queryHeight), NSNumber(value: Self.queryWidth)], dataType: .float32)
        objMemory    = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: Self.querySlots), NSNumber(value: Self.summaryDim)], dataType: .float32)
        memKey       = try MLMultiArray(shape: [1, NSNumber(value: Self.keyDim), NSNumber(value: Self.memCapacity)], dataType: .float32)
        memShrinkage = try MLMultiArray(shape: [1, 1, NSNumber(value: Self.memCapacity)], dataType: .float32)
        memMskValue  = try MLMultiArray(shape: [1, NSNumber(value: Self.valueDim), NSNumber(value: Self.memCapacity)], dataType: .float32)
        memValid     = try MLMultiArray(shape: [1, NSNumber(value: Self.memCapacity)], dataType: .float32)

        zero(sensory); zero(lastMask); zero(lastPixFeat); zero(lastMskValue); zero(objMemory)
        zero(memKey); zero(memShrinkage); zero(memMskValue); zero(memValid)
    }

    private static func loadModule(model: ModelEntry, configKey: String,
                                   substring: String, excludeSubstring: String?,
                                   units: MLComputeUnits) async throws -> MLModel {
        // Try the manifest config first.
        if let explicit = model.configString(configKey) {
            if let loaded = try? await ModelLoader.load(
                modelId: model.id, fileName: explicit, computeUnits: units) {
                return loaded
            }
        }
        // Fall back to file entries whose name matches the substring.
        if let fileName = model.files.first(where: {
            let lower = $0.name.lowercased()
            if !lower.contains(substring) { return false }
            if let excl = excludeSubstring, lower.contains(excl) { return false }
            return true
        })?.name {
            if let loaded = try? await ModelLoader.load(
                modelId: model.id, fileName: fileName, computeUnits: units) {
                return loaded
            }
        }
        // Last resort: scan the extracted package directory.
        do {
            return try await ModelLoader.loadBySubstring(
                modelId: model.id, substring: substring, computeUnits: units)
        } catch {
            throw EngineError.missingModel(substring)
        }
    }

    // MARK: Public API

    func reset() {
        zero(sensory); zero(lastMask); zero(lastPixFeat); zero(lastMskValue); zero(objMemory)
        zero(memKey); zero(memShrinkage); zero(memMskValue); zero(memValid)
        currentFrame = -1
        lastMemFrame = 0
        nextFifoSlot = 1
    }

    /// Seed the engine with the first frame and its segmentation mask.
    /// The caller should follow up with one `step(firstFramePred: true)`.
    func initializeSeed(firstImage image: MLMultiArray, mask: MLMultiArray) async throws {
        reset()
        _ = try await step(image: image, providedMask: mask, firstFramePred: false)
    }

    /// Run a single matting step. Returns the alpha matte as a flat row-major
    /// `[Float]` of length H*W in `[0, 1]`.
    func step(image: MLMultiArray, providedMask: MLMultiArray? = nil, firstFramePred: Bool = false) async throws -> [Float] {
        let isMemFrame: Bool
        let needSegment: Bool
        if firstFramePred {
            currentFrame = 0
            lastMemFrame = 0
            isMemFrame = true
            needSegment = true
        } else {
            currentFrame += 1
            isMemFrame = ((currentFrame - lastMemFrame) >= Self.memEvery) || (providedMask != nil)
            needSegment = providedMask == nil
        }

        let encOut = try await encoder.prediction(from: dict(["image": image]))
        let f16 = try feature(encOut, "f16")
        let f8  = try feature(encOut, "f8")
        let f4  = try feature(encOut, "f4")
        let f2  = try feature(encOut, "f2")
        let f1  = try feature(encOut, "f1")
        let pixFeat = try feature(encOut, "pix_feat")
        let key       = try feature(encOut, "key")
        let shrinkage = try feature(encOut, "shrinkage")
        let selection = try feature(encOut, "selection")

        var alphaArray = [Float](repeating: 0, count: Self.inputHeight * Self.inputWidth)
        if needSegment {
            let memReadout: MLMultiArray
            if currentFrame == 0 {
                let rfOut = try await readFirst.prediction(from: dict([
                    "pix_feat": pixFeat,
                    "last_msk_value": lastMskValue,
                    "sensory": sensory,
                    "last_mask": lastMask,
                    "obj_memory": objMemory,
                ]))
                memReadout = try feature(rfOut, "mem_readout")
            } else {
                let rdOut = try await read.prediction(from: dict([
                    "query_key": key,
                    "query_selection": selection,
                    "pix_feat": pixFeat,
                    "sensory": sensory,
                    "last_mask": lastMask,
                    "last_pix_feat": lastPixFeat,
                    "last_msk_value": lastMskValue,
                    "mem_key": memKey,
                    "mem_shrinkage": memShrinkage,
                    "mem_msk_value": memMskValue,
                    "mem_valid": memValid,
                    "obj_memory": objMemory,
                ]))
                memReadout = try feature(rdOut, "mem_readout")
            }

            let decOut = try await decoder.prediction(from: dict([
                "f16": f16, "f8": f8, "f4": f4, "f2": f2, "f1": f1,
                "mem_readout": memReadout,
                "sensory": sensory,
            ]))
            let newSensory = try feature(decOut, "new_sensory")
            let alpha      = try feature(decOut, "alpha")
            copyArray(alpha, into: &alphaArray)
            copyArrayContents(from: newSensory, to: sensory)
            copyArrayContents(from: alpha, to: lastMask)
        } else if let m = providedMask {
            copyArrayContents(from: m, to: lastMask)
            copyArray(m, into: &alphaArray)
        }

        copyArrayContents(from: pixFeat, to: lastPixFeat)

        let meOut = try await maskEncoder.prediction(from: dict([
            "image": image,
            "pix_feat": pixFeat,
            "sensory": sensory,
            "mask": lastMask,
        ]))
        let mskValue = try feature(meOut, "mask_value")
        let newSensoryME = try feature(meOut, "new_sensory")
        let objSummary = try feature(meOut, "obj_summary")
        copyArrayContents(from: mskValue, to: lastMskValue)

        if isMemFrame {
            if firstFramePred {
                zero(memValid)
                zero(objMemory)
                nextFifoSlot = 1
            }

            let slot: Int
            if providedMask != nil || firstFramePred {
                slot = 0
            } else {
                slot = nextFifoSlot
                nextFifoSlot += 1
                if nextFifoSlot >= Self.memMaxFrames { nextFifoSlot = 1 }
            }
            writeMemorySlot(slot: slot, key: key, shrinkage: shrinkage, mskValue: mskValue)
            accumulateObjSummary(objSummary)

            copyArrayContents(from: newSensoryME, to: sensory)
            lastMemFrame = currentFrame
        }

        return alphaArray
    }

    // MARK: Memory bookkeeping

    private func writeMemorySlot(slot: Int, key: MLMultiArray, shrinkage: MLMultiArray, mskValue: MLMultiArray) {
        let HW = Self.queryHW
        let start = slot * HW

        let inK = Self.dense(key)
        let inS = Self.dense(shrinkage)
        let inM = Self.dense(mskValue)

        let kPtr = memKey.dataPointer.assumingMemoryBound(to: Float.self)
        for c in 0..<Self.keyDim {
            let dst = c * Self.memCapacity + start
            let src = c * HW
            for i in 0..<HW { kPtr[dst + i] = inK[src + i] }
        }

        let sPtr = memShrinkage.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<HW { sPtr[start + i] = inS[i] }

        let mPtr = memMskValue.dataPointer.assumingMemoryBound(to: Float.self)
        for c in 0..<Self.valueDim {
            let dst = c * Self.memCapacity + start
            let src = c * HW
            for i in 0..<HW { mPtr[dst + i] = inM[src + i] }
        }

        let vPtr = memValid.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<HW { vPtr[start + i] = 1.0 }
    }

    private func accumulateObjSummary(_ summary: MLMultiArray) {
        let src = Self.dense(summary)
        let dst = objMemory.dataPointer.assumingMemoryBound(to: Float.self)
        let n = Self.querySlots * Self.summaryDim
        var any: Float = 0
        for i in 0..<n { any += abs(dst[i]) }
        if any == 0 {
            for i in 0..<n { dst[i] = src[i] }
        } else {
            for i in 0..<n { dst[i] += src[i] }
        }
    }

    // MARK: Helpers

    private func dict(_ pairs: [String: MLMultiArray]) -> MLDictionaryFeatureProvider {
        var d = [String: MLFeatureValue]()
        for (k, v) in pairs { d[k] = MLFeatureValue(multiArray: v) }
        return try! MLDictionaryFeatureProvider(dictionary: d)
    }

    private func feature(_ provider: MLFeatureProvider, _ name: String) throws -> MLMultiArray {
        guard let v = provider.featureValue(for: name)?.multiArrayValue else {
            throw EngineError.missingFeature(name)
        }
        return v
    }

    private func copyArrayContents(from src: MLMultiArray, to dst: MLMultiArray) {
        precondition(src.count == dst.count, "shape mismatch \(src.count) vs \(dst.count)")
        let dPtr = dst.dataPointer.assumingMemoryBound(to: Float.self)
        Self.readFloats(src, into: dPtr, count: src.count)
    }

    private func copyArray(_ src: MLMultiArray, into dst: inout [Float]) {
        let n = min(src.count, dst.count)
        dst.withUnsafeMutableBufferPointer { buf in
            Self.readFloats(src, into: buf.baseAddress!, count: n)
        }
    }

    private func zero(_ a: MLMultiArray) {
        let p = a.dataPointer.assumingMemoryBound(to: Float.self)
        memset(p, 0, a.count * MemoryLayout<Float>.size)
    }

    /// Stride-safe read of an MLMultiArray into a contiguous Float buffer.
    /// CoreML may return non-dense strides on ANE / GPU; a naive memcpy
    /// silently scrambles the data.
    static func readFloats(_ src: MLMultiArray, into dst: UnsafeMutablePointer<Float>, count: Int) {
        let shape = src.shape.map { $0.intValue }
        let strides = src.strides.map { $0.intValue }
        let srcPtr = src.dataPointer.assumingMemoryBound(to: Float.self)

        var expected = 1
        var dense = true
        for i in (0..<shape.count).reversed() {
            if strides[i] != expected { dense = false; break }
            expected *= shape[i]
        }
        if dense {
            memcpy(dst, srcPtr, count * MemoryLayout<Float>.size)
            return
        }

        let n = shape.count
        var counters = [Int](repeating: 0, count: n)
        for i in 0..<count {
            var offset = 0
            for d in 0..<n { offset += counters[d] * strides[d] }
            dst[i] = srcPtr[offset]
            var d = n - 1
            while d >= 0 {
                counters[d] += 1
                if counters[d] < shape[d] { break }
                counters[d] = 0
                d -= 1
            }
        }
    }

    static func dense(_ src: MLMultiArray) -> [Float] {
        let count = src.count
        let buf = UnsafeMutablePointer<Float>.allocate(capacity: count)
        defer { buf.deallocate() }
        readFloats(src, into: buf, count: count)
        return Array(UnsafeBufferPointer(start: buf, count: count))
    }
}

// MARK: - Video Matter (Vision seed + compositor)

/// Reads frames from an AVAsset, runs `MatAnyoneHubEngine` on each, and
/// writes a composited MP4. The first-frame mask is generated automatically
/// via Vision person segmentation (`.accurate`, binarised at 0.5, dilated by
/// radius 8), matching the stand-alone `MatAnyoneDemo`.
///
/// `@unchecked Sendable` because the engine owns the per-frame ring buffer
/// and we only run one `process(...)` at a time.
final class MatAnyoneHubMatter: @unchecked Sendable {
    private let engine: MatAnyoneHubEngine
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    struct Output {
        let videoURL: URL
        let firstFrameMask: UIImage?
    }

    init(engine: MatAnyoneHubEngine) { self.engine = engine }

    func process(asset: AVAsset,
                 backgroundColor: CIColor,
                 overrideMask: UIImage?,
                 progress: @escaping (Double) -> Void) async throws -> Output {

        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw NSError(domain: "MatAnyoneHub", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "No video track"])
        }
        let nominalSize = try await track.load(.naturalSize)
        let transform = try await track.load(.preferredTransform)
        let displaySize = nominalSize.applying(transform)
        let displayW = abs(displaySize.width)
        let displayH = abs(displaySize.height)
        let isPortrait = displayH > displayW

        let engineW = MatAnyoneHubEngine.inputWidth
        let engineH = MatAnyoneHubEngine.inputHeight
        let outW = isPortrait ? engineH : engineW
        let outH = isPortrait ? engineW : engineH

        let nominalFps = (try? await track.load(.nominalFrameRate)) ?? 30.0
        let frameDuration = CMTime(value: 1, timescale: max(Int32(round(nominalFps)), 1))
        let durationCM = try await asset.load(.duration)
        let totalFrames = max(1, Int(round(CMTimeGetSeconds(durationCM) * Double(nominalFps))))

        let reader = try AVAssetReader(asset: asset)
        let readerSettings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        ]
        let readerOutput = AVAssetReaderTrackOutput(track: track, outputSettings: readerSettings)
        readerOutput.alwaysCopiesSampleData = false
        reader.add(readerOutput)
        guard reader.startReading() else {
            throw reader.error ?? NSError(domain: "MatAnyoneHub", code: 2,
                                          userInfo: [NSLocalizedDescriptionKey: "AVAssetReader failed"])
        }

        let outURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("matanyone_\(UUID().uuidString).mp4")
        try? FileManager.default.removeItem(at: outURL)
        let writer = try AVAssetWriter(outputURL: outURL, fileType: .mp4)
        let writerSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: outW,
            AVVideoHeightKey: outH,
        ]
        let writerInput = AVAssetWriterInput(mediaType: .video, outputSettings: writerSettings)
        writerInput.expectsMediaDataInRealTime = false
        let pbAdaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: writerInput,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: outW,
                kCVPixelBufferHeightKey as String: outH,
            ]
        )
        writer.add(writerInput)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)

        let imageArray = try MLMultiArray(
            shape: [1, 3, NSNumber(value: engineH), NSNumber(value: engineW)],
            dataType: .float32
        )

        // Scratch buffers reused across every frame.
        let bgraRowBytes = engineW * 4
        let bgraPlaneSize = engineH * bgraRowBytes
        let monoPlaneSize = engineW * engineH
        let inputBGRA  = UnsafeMutablePointer<UInt8>.allocate(capacity: bgraPlaneSize)
        defer { inputBGRA.deallocate() }
        let planeR = UnsafeMutablePointer<UInt8>.allocate(capacity: monoPlaneSize)
        let planeG = UnsafeMutablePointer<UInt8>.allocate(capacity: monoPlaneSize)
        let planeB = UnsafeMutablePointer<UInt8>.allocate(capacity: monoPlaneSize)
        defer { planeR.deallocate(); planeG.deallocate(); planeB.deallocate() }
        let alphaPlane = UnsafeMutablePointer<UInt8>.allocate(capacity: monoPlaneSize)
        defer { alphaPlane.deallocate() }

        // Pre-render the solid background colour once.
        let bgBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: bgraPlaneSize)
        defer { bgBuffer.deallocate() }
        if let bgFillCtx = CGContext(
            data: bgBuffer, width: engineW, height: engineH, bitsPerComponent: 8,
            bytesPerRow: bgraRowBytes,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) {
            let bgCG = ciContext.createCGImage(
                CIImage(color: backgroundColor)
                    .cropped(to: CGRect(x: 0, y: 0, width: engineW, height: engineH)),
                from: CGRect(x: 0, y: 0, width: engineW, height: engineH)
            )
            if let bgCG { bgFillCtx.draw(bgCG, in: CGRect(x: 0, y: 0, width: engineW, height: engineH)) }
        }
        var bgViBuf = vImage_Buffer(
            data: bgBuffer,
            height: vImagePixelCount(engineH),
            width: vImagePixelCount(engineW),
            rowBytes: bgraRowBytes
        )
        // vImage copyMask is interpreted in ARGB byte order. Our buffer is
        // BGRA so the alpha byte sits at index 3 (mask bit 0x01).
        vImageOverwriteChannelsWithScalar_ARGB8888(255, &bgViBuf, &bgViBuf, 0x01, vImage_Flags(kvImageNoFlags))

        var frameIndex = 0
        var seedMaskPreview: UIImage? = nil

        while reader.status == .reading {
            guard let sample = readerOutput.copyNextSampleBuffer() else { break }
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sample) else { continue }

            var ci = CIImage(cvPixelBuffer: pixelBuffer)
                .oriented(forExifOrientation: exifOrientation(for: transform))
            if isPortrait {
                ci = ci.oriented(.right)  // portrait -> landscape
            }
            let scaled = ci.transformed(by: CGAffineTransform(
                scaleX: CGFloat(engineW) / ci.extent.width,
                y: CGFloat(engineH) / ci.extent.height
            ))
            let renderRect = CGRect(x: 0, y: 0, width: engineW, height: engineH)
            guard let cg = ciContext.createCGImage(scaled, from: renderRect) else { continue }
            fillImageArray(
                cg: cg, dst: imageArray, scratchBGRA: inputBGRA, rowBytes: bgraRowBytes,
                planeR: planeR, planeG: planeG, planeB: planeB
            )

            let alpha: [Float]
            if frameIndex == 0 {
                let seedFloat: [Float]
                if let overrideMask {
                    // Manual mask override: binarise at 0.5, then dilate.
                    let raw = try maskToFloatArray(overrideMask, width: engineW, height: engineH,
                                                   rotateForLandscape: isPortrait)
                    seedFloat = binaryDilate(raw, width: engineW, height: engineH,
                                             threshold: 0.5, radius: 8)
                } else {
                    let visionAlpha = try generatePersonMaskFloat(
                        cgImage: cg, width: engineW, height: engineH)
                    seedFloat = binaryDilate(visionAlpha, width: engineW, height: engineH,
                                             threshold: 0.5, radius: 8)
                }
                let seed = try makeSeedMaskMLArray(seedFloat, width: engineW, height: engineH)
                try await engine.initializeSeed(firstImage: imageArray, mask: seed)
                alpha = try await engine.step(image: imageArray, providedMask: nil, firstFramePred: true)
                seedMaskPreview = makeMaskPreviewImage(
                    seedFloat, width: engineW, height: engineH, rotateLeft: isPortrait
                )
            } else {
                alpha = try await engine.step(image: imageArray)
            }

            let landscapePB = try makeOutputPixelBuffer(width: engineW, height: engineH)
            composite(
                input: cg, alpha: alpha,
                bgBuffer: bgBuffer, bgRowBytes: bgraRowBytes,
                alphaScratch: alphaPlane,
                into: landscapePB
            )
            let outPB: CVPixelBuffer
            if isPortrait {
                outPB = try rotateBuffer(landscapePB, by: .left, width: outW, height: outH)
            } else {
                outPB = landscapePB
            }

            let timestamp = CMTimeMultiply(frameDuration, multiplier: Int32(frameIndex))
            while !writerInput.isReadyForMoreMediaData {
                try await Task.sleep(nanoseconds: 5_000_000)
            }
            pbAdaptor.append(outPB, withPresentationTime: timestamp)

            frameIndex += 1
            progress(min(1.0, Double(frameIndex) / Double(totalFrames)))
        }

        writerInput.markAsFinished()
        await writer.finishWriting()
        if writer.status != .completed {
            throw writer.error ?? NSError(domain: "MatAnyoneHub", code: 3,
                                          userInfo: [NSLocalizedDescriptionKey: "AVAssetWriter failed"])
        }
        return Output(videoURL: outURL, firstFrameMask: seedMaskPreview)
    }

    // MARK: Vision person segmentation

    private func generatePersonMaskFloat(cgImage: CGImage, width: Int, height: Int) throws -> [Float] {
        let request = VNGeneratePersonSegmentationRequest()
        request.qualityLevel = .accurate
        request.outputPixelFormat = kCVPixelFormatType_OneComponent8
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try handler.perform([request])

        var out = [Float](repeating: 0, count: width * height)
        guard let result = request.results?.first,
              let pixelBuffer = result.pixelBuffer as CVPixelBuffer? else {
            // No person detected — fall back to a full-image mask so we at
            // least produce some output instead of crashing.
            for i in 0..<out.count { out[i] = 1.0 }
            return out
        }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        let mw = CVPixelBufferGetWidth(pixelBuffer)
        let mh = CVPixelBufferGetHeight(pixelBuffer)
        let bpr = CVPixelBufferGetBytesPerRow(pixelBuffer)
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            for i in 0..<out.count { out[i] = 1.0 }
            return out
        }
        let bytes = base.assumingMemoryBound(to: UInt8.self)
        for y in 0..<height {
            let sy = min(mh - 1, y * mh / height)
            for x in 0..<width {
                let sx = min(mw - 1, x * mw / width)
                out[y * width + x] = Float(bytes[sy * bpr + sx]) / 255.0
            }
        }
        return out
    }

    /// Rasterise a user-supplied UIImage into a `[Float]` alpha array. Uses
    /// the luminance of the RGB channels (standard 0.299/0.587/0.114 weights)
    /// so images where the mask is white-on-black still produce the expected
    /// foreground alpha.
    private func maskToFloatArray(_ image: UIImage, width: Int, height: Int,
                                  rotateForLandscape: Bool) throws -> [Float] {
        var ci = CIImage(image: image) ?? CIImage()
        if rotateForLandscape {
            ci = ci.oriented(.right)
        }
        let scaled = ci.transformed(by: CGAffineTransform(
            scaleX: CGFloat(width)  / max(ci.extent.width,  1),
            y: CGFloat(height) / max(ci.extent.height, 1)
        ))
        let rect = CGRect(x: 0, y: 0, width: width, height: height)
        guard let cg = ciContext.createCGImage(scaled, from: rect) else {
            throw NSError(domain: "MatAnyoneHub", code: 4,
                          userInfo: [NSLocalizedDescriptionKey: "Mask rasterisation failed"])
        }
        let bpr = width * 4
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        guard let ctx = CGContext(
            data: &pixels, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: bpr,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else {
            throw NSError(domain: "MatAnyoneHub", code: 5,
                          userInfo: [NSLocalizedDescriptionKey: "CGContext failed"])
        }
        ctx.draw(cg, in: rect)
        var out = [Float](repeating: 0, count: width * height)
        for i in 0..<(width * height) {
            let r = Float(pixels[i * 4])
            let g = Float(pixels[i * 4 + 1])
            let b = Float(pixels[i * 4 + 2])
            out[i] = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
        }
        return out
    }

    private func makeSeedMaskMLArray(_ alpha: [Float], width: Int, height: Int) throws -> MLMultiArray {
        let mask = try MLMultiArray(shape: [1, 1, NSNumber(value: height), NSNumber(value: width)],
                                    dataType: .float32)
        let dst = mask.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<(width * height) { dst[i] = alpha[i] }
        return mask
    }

    /// Binarise then dilate — approximates the official `gen_dilate` used by
    /// MatAnyone's reference pipeline.
    private func binaryDilate(_ src: [Float], width: Int, height: Int,
                              threshold: Float, radius: Int) -> [Float] {
        var bin = [Float](repeating: 0, count: width * height)
        for i in 0..<bin.count { bin[i] = src[i] > threshold ? 1 : 0 }
        guard radius > 0 else { return bin }
        return softDilate(bin, width: width, height: height, radius: radius)
    }

    private func softDilate(_ src: [Float], width: Int, height: Int, radius: Int) -> [Float] {
        guard radius > 0 else { return src }
        var horiz = [Float](repeating: 0, count: width * height)
        for y in 0..<height {
            let row = y * width
            for x in 0..<width {
                let lo = max(0, x - radius)
                let hi = min(width - 1, x + radius)
                var m: Float = 0
                for k in lo...hi { let v = src[row + k]; if v > m { m = v } }
                horiz[row + x] = m
            }
        }
        var out = [Float](repeating: 0, count: width * height)
        for x in 0..<width {
            for y in 0..<height {
                let lo = max(0, y - radius)
                let hi = min(height - 1, y + radius)
                var m: Float = 0
                for k in lo...hi { let v = horiz[k * width + x]; if v > m { m = v } }
                out[y * width + x] = m
            }
        }
        return out
    }

    private func makeMaskPreviewImage(_ alpha: [Float], width: Int, height: Int,
                                      rotateLeft: Bool) -> UIImage? {
        var pixels = [UInt8](repeating: 255, count: width * height * 4)
        for i in 0..<(width * height) {
            let v = UInt8(min(255, max(0, alpha[i] * 255)))
            pixels[i * 4]     = v
            pixels[i * 4 + 1] = v
            pixels[i * 4 + 2] = v
        }
        guard let provider = CGDataProvider(data: Data(pixels) as CFData),
              let cg = CGImage(
                width: width, height: height,
                bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
              ) else { return nil }
        if rotateLeft {
            let ci = CIImage(cgImage: cg).oriented(.left)
            if let rotated = ciContext.createCGImage(ci, from: ci.extent) {
                return UIImage(cgImage: rotated)
            }
        }
        return UIImage(cgImage: cg)
    }

    // MARK: Image <-> MLMultiArray

    private func fillImageArray(
        cg: CGImage,
        dst: MLMultiArray,
        scratchBGRA: UnsafeMutablePointer<UInt8>,
        rowBytes: Int,
        planeR: UnsafeMutablePointer<UInt8>,
        planeG: UnsafeMutablePointer<UInt8>,
        planeB: UnsafeMutablePointer<UInt8>
    ) {
        let w = MatAnyoneHubEngine.inputWidth
        let h = MatAnyoneHubEngine.inputHeight
        let spatial = w * h

        guard let ctx = CGContext(
            data: scratchBGRA, width: w, height: h, bitsPerComponent: 8, bytesPerRow: rowBytes,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else { return }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))

        let dstPtr = dst.dataPointer.assumingMemoryBound(to: Float.self)
        let rowBytesF = w * MemoryLayout<Float>.size
        var rDst = vImage_Buffer(data: dstPtr,                            height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: rowBytesF)
        var gDst = vImage_Buffer(data: dstPtr.advanced(by: spatial),      height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: rowBytesF)
        var bDst = vImage_Buffer(data: dstPtr.advanced(by: 2 * spatial),  height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: rowBytesF)

        var src = vImage_Buffer(data: scratchBGRA, height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: rowBytes)
        var rPlane = vImage_Buffer(data: planeR, height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: w)
        var gPlane = vImage_Buffer(data: planeG, height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: w)
        var bPlane = vImage_Buffer(data: planeB, height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: w)
        vImageExtractChannel_ARGB8888(&src, &bPlane, 0, vImage_Flags(kvImageNoFlags))
        vImageExtractChannel_ARGB8888(&src, &gPlane, 1, vImage_Flags(kvImageNoFlags))
        vImageExtractChannel_ARGB8888(&src, &rPlane, 2, vImage_Flags(kvImageNoFlags))
        vImageConvert_Planar8toPlanarF(&rPlane, &rDst, 1.0, 0.0, vImage_Flags(kvImageNoFlags))
        vImageConvert_Planar8toPlanarF(&gPlane, &gDst, 1.0, 0.0, vImage_Flags(kvImageNoFlags))
        vImageConvert_Planar8toPlanarF(&bPlane, &bDst, 1.0, 0.0, vImage_Flags(kvImageNoFlags))
    }

    private func makeOutputPixelBuffer(width: Int, height: Int) throws -> CVPixelBuffer {
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true,
        ]
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                         kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
        guard status == kCVReturnSuccess, let buf = pb else {
            throw NSError(domain: "MatAnyoneHub", code: 6,
                          userInfo: [NSLocalizedDescriptionKey: "CVPixelBufferCreate failed"])
        }
        return buf
    }

    private func composite(
        input: CGImage,
        alpha: [Float],
        bgBuffer: UnsafeMutablePointer<UInt8>,
        bgRowBytes: Int,
        alphaScratch: UnsafeMutablePointer<UInt8>,
        into pb: CVPixelBuffer
    ) {
        let w = CVPixelBufferGetWidth(pb)
        let h = CVPixelBufferGetHeight(pb)
        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return }
        let bpr = CVPixelBufferGetBytesPerRow(pb)

        // 1. Render the foreground frame into the output pixel buffer.
        guard let fgCtx = CGContext(
            data: base, width: w, height: h, bitsPerComponent: 8, bytesPerRow: bpr,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else { return }
        fgCtx.draw(input, in: CGRect(x: 0, y: 0, width: w, height: h))

        // 2. Float alpha → Planar8.
        alpha.withUnsafeBufferPointer { fp in
            var srcF = vImage_Buffer(
                data: UnsafeMutableRawPointer(mutating: fp.baseAddress),
                height: vImagePixelCount(h),
                width: vImagePixelCount(w),
                rowBytes: w * MemoryLayout<Float>.size
            )
            var dst8 = vImage_Buffer(
                data: alphaScratch,
                height: vImagePixelCount(h),
                width: vImagePixelCount(w),
                rowBytes: w
            )
            vImageConvert_PlanarFtoPlanar8(&srcF, &dst8, 1.0, 0.0, vImage_Flags(kvImageNoFlags))
        }

        // 3. Stamp the alpha plane into the foreground BGRA buffer's A channel.
        var alphaPlaneBuf = vImage_Buffer(
            data: alphaScratch,
            height: vImagePixelCount(h),
            width: vImagePixelCount(w),
            rowBytes: w
        )
        var fgBuf = vImage_Buffer(
            data: base,
            height: vImagePixelCount(h),
            width: vImagePixelCount(w),
            rowBytes: bpr
        )
        vImageOverwriteChannels_ARGB8888(&alphaPlaneBuf, &fgBuf, &fgBuf, 0x01, vImage_Flags(kvImageNoFlags))

        // 4. Premultiply BGR by alpha (vImage has no non-premultiplied BGRA blender).
        vImagePremultiplyData_RGBA8888(&fgBuf, &fgBuf, vImage_Flags(kvImageNoFlags))

        // 5. Premultiplied src-over blend.
        var bgBuf = vImage_Buffer(
            data: bgBuffer,
            height: vImagePixelCount(h),
            width: vImagePixelCount(w),
            rowBytes: bgRowBytes
        )
        vImagePremultipliedAlphaBlend_BGRA8888(&fgBuf, &bgBuf, &fgBuf, vImage_Flags(kvImageNoFlags))
    }

    private func rotateBuffer(_ src: CVPixelBuffer, by direction: CGImagePropertyOrientation,
                              width: Int, height: Int) throws -> CVPixelBuffer {
        let dst = try makeOutputPixelBuffer(width: width, height: height)
        let ci = CIImage(cvPixelBuffer: src).oriented(direction)
        ciContext.render(ci, to: dst)
        return dst
    }

    private func exifOrientation(for transform: CGAffineTransform) -> Int32 {
        let a = transform.a, b = transform.b, c = transform.c, d = transform.d
        if a == 0  && b == 1  && c == -1 && d == 0  { return 6 } // 90 CW
        if a == 0  && b == -1 && c == 1  && d == 0  { return 8 } // 90 CCW
        if a == -1 && b == 0  && c == 0  && d == -1 { return 3 } // 180
        return 1
    }
}

// MARK: - Photo library video transferable

/// PhotosPicker only hands back a short-lived file URL, so we copy the movie
/// into the app's temporary directory and keep the copy's URL around.
struct PickedMovie: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { movie in
            SentTransferredFile(movie.url)
        } importing: { received in
            let copy = FileManager.default.temporaryDirectory
                .appendingPathComponent("picked_\(UUID().uuidString).mov")
            try? FileManager.default.removeItem(at: copy)
            try FileManager.default.copyItem(at: received.file, to: copy)
            return Self(url: copy)
        }
    }
}

// MARK: - Simple Video Player

struct VideoPlayerView: UIViewControllerRepresentable {
    let url: URL

    func makeUIViewController(context: Context) -> AVPlayerViewController {
        let vc = AVPlayerViewController()
        vc.player = AVPlayer(url: url)
        return vc
    }

    func updateUIViewController(_ vc: AVPlayerViewController, context: Context) {}
}
