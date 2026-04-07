import CoreML
import Foundation
import Accelerate

/// Per-frame state machine that drives the 5 split MatAnyone Core ML models.
///
/// Reproduces the Python `InferenceCore.step` loop in Swift. The model split
/// pushes all "object/memory bookkeeping" up to this class so each Core ML
/// model is a pure function of its inputs.
@MainActor
final class MatAnyoneEngine {
    // MARK: Resolution & topology — must match the converter

    static let inputHeight = 432
    static let inputWidth  = 768
    static let stride      = 16
    static let queryHeight = inputHeight / stride   // 27
    static let queryWidth  = inputWidth  / stride   // 48
    static let queryHW     = queryHeight * queryWidth // 1296
    static let memMaxFrames = 5
    static let memCapacity  = memMaxFrames * queryHW  // 6480
    static let memEvery    = 5
    static let valueDim    = 256
    static let keyDim      = 64
    static let sensoryDim  = 256
    static let querySlots  = 16
    static let summaryDim  = 257   // embed_dim + 1 count slot

    // MARK: Models

    private let encoder: MLModel
    private let maskEncoder: MLModel
    private let readFirst: MLModel
    private let read: MLModel
    private let decoder: MLModel

    // MARK: Persistent state (carried between frames)

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

    init() throws {
        // Encoder/decoder/mask_encoder are pure conv stacks and run fine on
        // GPU. The two memory-readout modules contain reshapes and slices on
        // the singleton num_objects dimension that crash Metal Performance
        // Shaders ("subRange.start = -1 vs length 1"), so they have to stay
        // on the CPU until the converter is rewritten to drop those singleton
        // dims.
        let gpuCfg = MLModelConfiguration()
        gpuCfg.computeUnits = .cpuAndGPU
        let cpuCfg = MLModelConfiguration()
        cpuCfg.computeUnits = .cpuOnly

        encoder     = try Self.loadModel("MatAnyone_encoder", config: gpuCfg)
        maskEncoder = try Self.loadModel("MatAnyone_mask_encoder", config: gpuCfg)
        readFirst   = try Self.loadModel("MatAnyone_read_first", config: cpuCfg)
        read        = try Self.loadModel("MatAnyone_read", config: cpuCfg)
        decoder     = try Self.loadModel("MatAnyone_decoder", config: gpuCfg)

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

    private static func loadModel(_ name: String, config: MLModelConfiguration) throws -> MLModel {
        guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else {
            throw EngineError.modelMissing(name)
        }
        return try MLModel(contentsOf: url, configuration: config)
    }

    // MARK: Public API

    /// Reset memory state for a fresh clip.
    func reset() {
        zero(sensory); zero(lastMask); zero(lastPixFeat); zero(lastMskValue); zero(objMemory)
        zero(memKey); zero(memShrinkage); zero(memMskValue); zero(memValid)
        currentFrame = -1
        lastMemFrame = 0
        nextFifoSlot = 1
    }

    /// Initialize state from the very first frame and its segmentation mask.
    /// `image` should be (3, H, W) float in [0, 1] in CHW order.
    /// `mask` should be (H, W) float in [0, 1].
    /// `warmupPasses` matches the official `n_warmup` parameter (default 10):
    /// re-running the first frame this many times stabilises the memory before
    /// the actual sequence starts. Without it the alpha tends to erode.
    func initialize(firstImage image: MLMultiArray, mask: MLMultiArray, warmupPasses: Int = 10) async throws {
        reset()
        // 1) Seed slot 0 with the supplied mask (need_segment = false).
        _ = try await step(image: image, providedMask: mask, firstFramePred: false)
        // 2) Replicate the official `n_warmup` first-frame passes. Each pass
        //    clears working memory, runs read_first/decoder/mask_encoder again
        //    using the previous pass's mask_value, and refines the entry.
        for _ in 0..<warmupPasses {
            _ = try await step(image: image, providedMask: nil, firstFramePred: true)
        }
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

        // ---- encoder ----
        let encOut = try await encoder.prediction(from: dict([
            "image": image
        ]))
        let f16 = try feature(encOut, "f16")
        let f8  = try feature(encOut, "f8")
        let f4  = try feature(encOut, "f4")
        let f2  = try feature(encOut, "f2")
        let f1  = try feature(encOut, "f1")
        let pixFeat = try feature(encOut, "pix_feat")
        let key       = try feature(encOut, "key")
        let shrinkage = try feature(encOut, "shrinkage")
        let selection = try feature(encOut, "selection")

        // ---- segmentation if needed ----
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
            let alpha       = try feature(decOut, "alpha")  // (1,1,H,W)
            copyArray(alpha, into: &alphaArray)
            copyArrayContents(from: newSensory, to: sensory)
            // last_mask = alpha (clamped) — this is the foreground channel.
            copyArrayContents(from: alpha, to: lastMask)
        } else if let m = providedMask {
            // Use the provided mask as last_mask directly.
            copyArrayContents(from: m, to: lastMask)
            copyArray(m, into: &alphaArray)
        }

        // last_pix_feat is updated every frame we have an encoder output.
        copyArrayContents(from: pixFeat, to: lastPixFeat)

        // ---- mask encoder (always called; used for memory frames or as the
        // shallow update on non-memory frames so the next read sees the latest
        // last_msk_value).
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
                // Drop the temp working memory and obj memory; we'll reseed
                // with the warmup frame.
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
        // memKey shape (1, 64, capacity); key shape (1, 64, h, w) flattened.
        let kPtr = memKey.dataPointer.assumingMemoryBound(to: Float.self)
        let inK = key.dataPointer.assumingMemoryBound(to: Float.self)
        for c in 0..<Self.keyDim {
            let dst = c * Self.memCapacity + start
            let src = c * HW
            for i in 0..<HW { kPtr[dst + i] = inK[src + i] }
        }

        // memShrinkage (1, 1, capacity)
        let sPtr = memShrinkage.dataPointer.assumingMemoryBound(to: Float.self)
        let inS = shrinkage.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<HW { sPtr[start + i] = inS[i] }

        // memMskValue (1, 256, capacity); mskValue shape (1, 1, 256, h, w)
        let mPtr = memMskValue.dataPointer.assumingMemoryBound(to: Float.self)
        let inM = mskValue.dataPointer.assumingMemoryBound(to: Float.self)
        for c in 0..<Self.valueDim {
            let dst = c * Self.memCapacity + start
            let src = c * HW
            for i in 0..<HW { mPtr[dst + i] = inM[src + i] }
        }

        let vPtr = memValid.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<HW { vPtr[start + i] = 1.0 }
    }

    private func accumulateObjSummary(_ summary: MLMultiArray) {
        // summary: (1, 1, 16, 257). We add into objMemory channelwise; the
        // last value in each row is the count and is treated identically to
        // the embedding sums (matches the streaming average in MemoryManager).
        let dst = objMemory.dataPointer.assumingMemoryBound(to: Float.self)
        let src = summary.dataPointer.assumingMemoryBound(to: Float.self)
        let n = Self.querySlots * Self.summaryDim
        // If everything is still zero this is the first call → straight copy
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
        let s = src.dataPointer.assumingMemoryBound(to: Float.self)
        let d = dst.dataPointer.assumingMemoryBound(to: Float.self)
        memcpy(d, s, src.count * MemoryLayout<Float>.size)
    }

    private func copyArray(_ src: MLMultiArray, into dst: inout [Float]) {
        let s = src.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<min(src.count, dst.count) { dst[i] = s[i] }
    }

    private func zero(_ a: MLMultiArray) {
        let p = a.dataPointer.assumingMemoryBound(to: Float.self)
        memset(p, 0, a.count * MemoryLayout<Float>.size)
    }
}

enum EngineError: LocalizedError {
    case modelMissing(String)
    case missingFeature(String)
    var errorDescription: String? {
        switch self {
        case .modelMissing(let n): return "Missing CoreML model: \(n)"
        case .missingFeature(let n): return "Model output \"\(n)\" was not produced"
        }
    }
}
