import Foundation
import CoreML
import CoreGraphics
import Accelerate

/// Main SAM inference session (MobileSAM). Vendored from SAMKit.
/// The URL-based initializer expects pre-compiled `.mlmodelc` packages; use the
/// preloaded-MLModel initializer when the app already compiled/loaded the models
/// through its own model loader (which is how the hub app runs).
final class SamSession {

    private let encoder: MLModel
    private let decoder: MLModel
    private let preprocessor: Preprocessor
    private let postprocessor: Postprocessor
    private let promptEncoder: PromptEncoder?
    private let modelType: ModelType

    private var cachedEmbedding: MLMultiArray?
    private var transformParams: TransformParams?
    private let modelSize: Int

    // MARK: - Initialization

    init(model: SamModelRef, config: RuntimeConfig = .bestAvailable) throws {
        self.modelSize = model.inputSize
        self.modelType = model.modelType

        // Encoder: honor the caller's config (ANE-preferred gives the biggest
        // win on the image tower).
        let encoderConfig = MLModelConfiguration()
        encoderConfig.computeUnits = config.computeUnits.mlComputeUnits
        encoderConfig.allowLowPrecisionAccumulationOnGPU = true

        // Decoder: MobileSAM's mask decoder uses `conv_transpose` with an
        // explicit `output_shape`, which ANE cannot compile ("E5RT ... Could
        // not infer a new output shape for conv_transpose ... Error plan
        // build: -1"). CoreML sometimes fails to auto-fall-back after the
        // failed ANE plan, so for MobileSAM we pin the decoder to CPU+GPU
        // upfront. Other model types honor the caller's choice.
        let decoderConfig = MLModelConfiguration()
        if model.modelType == .mobileSam {
            decoderConfig.computeUnits = .cpuAndGPU
        } else {
            decoderConfig.computeUnits = config.computeUnits.mlComputeUnits
        }
        decoderConfig.allowLowPrecisionAccumulationOnGPU = true

        self.encoder = try MLModel(contentsOf: model.encoderURL, configuration: encoderConfig)
        self.decoder = try MLModel(contentsOf: model.decoderURL, configuration: decoderConfig)

        self.preprocessor = Preprocessor(modelSize: modelSize)
        self.postprocessor = Postprocessor(
            modelSize: modelSize,
            isHuggingFaceModel: model.modelType != .mobileSam
        )

        if model.modelType == .mobileSam,
           let weightsURL = model.promptEncoderWeightsURL {
            self.promptEncoder = try PromptEncoder(weightsURL: weightsURL)
        } else {
            self.promptEncoder = nil
        }
    }

    /// Alternate initializer that accepts already-loaded MLModels. Used by the hub
    /// app so that the shared `ModelLoader` handles compilation, caching, and
    /// compute-unit selection.
    init(
        encoder: MLModel,
        decoder: MLModel,
        modelSize: Int = 1024,
        modelType: ModelType = .mobileSam,
        promptEncoderWeightsURL: URL? = nil
    ) throws {
        self.encoder = encoder
        self.decoder = decoder
        self.modelSize = modelSize
        self.modelType = modelType
        self.preprocessor = Preprocessor(modelSize: modelSize)
        self.postprocessor = Postprocessor(
            modelSize: modelSize,
            isHuggingFaceModel: modelType != .mobileSam
        )
        if modelType == .mobileSam, let url = promptEncoderWeightsURL {
            self.promptEncoder = try PromptEncoder(weightsURL: url)
        } else {
            self.promptEncoder = nil
        }
    }

    // MARK: - Public API

    /// Run the image encoder once and cache the embedding.
    func setImage(_ image: CGImage) throws {
        let t0 = CFAbsoluteTimeGetCurrent()

        cachedEmbedding = nil
        transformParams = nil

        let (processedImage, transform) = try preprocessor.process(image)
        self.transformParams = transform

        let t1 = CFAbsoluteTimeGetCurrent()

        // Add batch dimension [3, H, W] → [1, 3, H, W] if the encoder expects it.
        let batchedImage: MLMultiArray
        if processedImage.shape.count == 3 {
            let s = processedImage.shape.map { $0.intValue }
            batchedImage = try MLMultiArray(
                shape: [1, s[0] as NSNumber, s[1] as NSNumber, s[2] as NSNumber],
                dataType: .float32
            )
            let src = processedImage.dataPointer.bindMemory(to: Float32.self, capacity: processedImage.count)
            let dst = batchedImage.dataPointer.bindMemory(to: Float32.self, capacity: batchedImage.count)
            memcpy(dst, src, processedImage.count * MemoryLayout<Float32>.size)
        } else {
            batchedImage = processedImage
        }

        let encInput = try MLDictionaryFeatureProvider(dictionary: ["image": batchedImage])
        let encOutput = try encoder.prediction(from: encInput)

        let t2 = CFAbsoluteTimeGetCurrent()

        guard let embedding = encOutput.featureValue(for: "image_embeddings")?.multiArrayValue else {
            throw SamError.invalidModelOutput("Missing image_embeddings from encoder")
        }
        self.cachedEmbedding = embedding
        print("[SAMKit] setImage: preprocess=\(Int((t1 - t0) * 1000))ms encoder=\(Int((t2 - t1) * 1000))ms")
    }

    /// Run mask prediction with the given prompts.
    func predict(
        points: [SamPoint] = [],
        box: SamBox? = nil,
        maskInput: SamMaskRef? = nil,
        options: SamOptions = SamOptions()
    ) throws -> SamResult {

        let t0 = CFAbsoluteTimeGetCurrent()

        guard let embedding = cachedEmbedding, let transform = transformParams else {
            throw SamError.imageNotSet
        }

        let decoderInput: MLDictionaryFeatureProvider

        if let pe = promptEncoder {
            let (sparse, dense) = try pe.encode(points: points, box: box, transform: transform)
            decoderInput = try MLDictionaryFeatureProvider(dictionary: [
                "image_embeddings": embedding,
                "sparse_embeddings": sparse,
                "dense_embeddings": dense,
            ])
        } else {
            var allPoints = points
            if let box = box {
                allPoints.append(SamPoint(x: CGFloat(box.x0), y: CGFloat(box.y0), label: .positive))
                allPoints.append(SamPoint(x: CGFloat(box.x1), y: CGFloat(box.y1), label: .positive))
            }
            let (coords, labels) = try preprocessor.encodePoints(allPoints, transform: transform)
            if box != nil {
                let lp = labels.dataPointer.bindMemory(to: Float32.self, capacity: labels.count)
                let start = allPoints.count - 2
                lp[start] = 2.0
                lp[start + 1] = 3.0
            }
            var inputs: [String: Any] = [
                "image_embeddings": embedding,
                "point_coords": coords,
                "point_labels": labels,
                "has_mask_input": MLMultiArray.scalar(Float(maskInput != nil ? 1.0 : 0.0))
            ]
            if let maskInput = maskInput {
                inputs["mask_input"] = try preprocessor.encodeMask(maskInput, transform: transform)
            } else {
                inputs["mask_input"] = try MLMultiArray.zeros(shape: [1, 1, 256, 256])
            }
            decoderInput = try MLDictionaryFeatureProvider(dictionary: inputs)
        }

        let t1 = CFAbsoluteTimeGetCurrent()
        let decoderOutput = try decoder.prediction(from: decoderInput)
        let t2 = CFAbsoluteTimeGetCurrent()

        guard let maskLogits = decoderOutput.featureValue(for: "masks")?.multiArrayValue,
              let iouPredictions = decoderOutput.featureValue(for: "iou_predictions")?.multiArrayValue else {
            throw SamError.invalidModelOutput("Missing masks or iou_predictions from decoder")
        }

        let result = try postprocessor.process(
            maskLogits: maskLogits,
            iouPredictions: iouPredictions,
            transform: transform,
            options: options
        )

        let t3 = CFAbsoluteTimeGetCurrent()
        print("[SAMKit] prompt=\(Int((t1-t0)*1000))ms decoder=\(Int((t2-t1)*1000))ms postprocess=\(Int((t3-t2)*1000))ms total=\(Int((t3-t0)*1000))ms")
        return result
    }

    func clear() {
        cachedEmbedding = nil
        transformParams = nil
    }
}

// MARK: - Supporting Types

struct SamModelRef {
    let encoderURL: URL
    let decoderURL: URL
    let inputSize: Int
    let modelType: ModelType
    let promptEncoderWeightsURL: URL?

    init(
        encoderURL: URL,
        decoderURL: URL,
        inputSize: Int = 1024,
        modelType: ModelType,
        promptEncoderWeightsURL: URL? = nil
    ) {
        self.encoderURL = encoderURL
        self.decoderURL = decoderURL
        self.inputSize = inputSize
        self.modelType = modelType
        self.promptEncoderWeightsURL = promptEncoderWeightsURL
    }
}

enum ModelType {
    case mobileSam

    var modelName: String { "mobile_sam" }
    var inputSize: Int { 1024 }
}

struct RuntimeConfig {
    enum ComputeUnits {
        case cpuOnly
        case gpuPreferred
        case neuralEnginePreferred
        case bestAvailable

        var mlComputeUnits: MLComputeUnits {
            switch self {
            case .cpuOnly: return .cpuOnly
            case .gpuPreferred: return .cpuAndGPU
            case .neuralEnginePreferred:
                if #available(iOS 16.0, *) { return .cpuAndNeuralEngine }
                return .cpuAndGPU
            case .bestAvailable: return .all
            }
        }
    }

    let computeUnits: ComputeUnits
    let enableFP16: Bool

    init(computeUnits: ComputeUnits = .bestAvailable, enableFP16: Bool = true) {
        self.computeUnits = computeUnits
        self.enableFP16 = enableFP16
    }

    static let bestAvailable = RuntimeConfig()
}

struct SamPoint {
    let x: CGFloat
    let y: CGFloat
    let label: SamPointLabel

    init(x: CGFloat, y: CGFloat, label: SamPointLabel) {
        self.x = x; self.y = y; self.label = label
    }
}

enum SamPointLabel: Int {
    case positive = 1
    case negative = 0
}

struct SamBox {
    let x0: Float
    let y0: Float
    let x1: Float
    let y1: Float

    init(x0: Float, y0: Float, x1: Float, y1: Float) {
        self.x0 = x0; self.y0 = y0; self.x1 = x1; self.y1 = y1
    }
}

struct SamOptions {
    let multimaskOutput: Bool
    let returnLogits: Bool
    let maskThreshold: Float
    let maxMasks: Int

    init(
        multimaskOutput: Bool = true,
        returnLogits: Bool = false,
        maskThreshold: Float = 0.0,
        maxMasks: Int = 3
    ) {
        self.multimaskOutput = multimaskOutput
        self.returnLogits = returnLogits
        self.maskThreshold = maskThreshold
        self.maxMasks = maxMasks
    }
}

struct SamMask {
    let width: Int
    let height: Int
    let logits: [Float]?
    let alpha: Data
    let score: Float
    let cgImage: CGImage

    init(width: Int, height: Int, logits: [Float]?, alpha: Data, score: Float, cgImage: CGImage) {
        self.width = width
        self.height = height
        self.logits = logits
        self.alpha = alpha
        self.score = score
        self.cgImage = cgImage
    }
}

typealias SamMaskRef = SamMask

struct SamResult {
    let masks: [SamMask]
    let scores: [Float]

    init(masks: [SamMask], scores: [Float]) {
        self.masks = masks
        self.scores = scores
    }
}

enum SamError: LocalizedError {
    case imageNotSet
    case modelNotFound
    case invalidModelOutput(String)
    case preprocessingFailed(String)
    case postprocessingFailed(String)

    var errorDescription: String? {
        switch self {
        case .imageNotSet: return "Image not set. Call setImage() first."
        case .modelNotFound: return "Model files not found in bundle"
        case .invalidModelOutput(let m): return "Invalid model output: \(m)"
        case .preprocessingFailed(let m): return "Preprocessing failed: \(m)"
        case .postprocessingFailed(let m): return "Postprocessing failed: \(m)"
        }
    }
}

// MARK: - MLMultiArray Helpers

extension MLMultiArray {
    static func zeros(shape: [Int]) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: shape as [NSNumber], dataType: .float32)
        let ptr = arr.dataPointer.bindMemory(to: Float32.self, capacity: arr.count)
        for i in 0..<arr.count { ptr[i] = 0 }
        return arr
    }

    static func scalar(_ value: Float) -> MLMultiArray {
        if let arr = try? MLMultiArray(shape: [1], dataType: .float32) {
            arr[0] = NSNumber(value: value)
            return arr
        }
        fatalError("Failed to create scalar MLMultiArray")
    }
}
