import CoreML
import UIKit

enum ClassifierError: LocalizedError {
    case modelNotLoaded
    case invalidImage
    case predictionFailed

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded: return "Model not loaded"
        case .invalidImage: return "Failed to process image"
        case .predictionFailed: return "Prediction failed"
        }
    }
}

struct ClassificationResult: Identifiable {
    let id = UUID()
    let label: String
    let score: Float
}

@MainActor
class SigLIPClassifier: ObservableObject {
    @Published var isReady = false

    // SigLIP constants (from training)
    private let logitScale: Float = 117.330765
    private let logitBias: Float = -12.932437
    private let eosTokenID: Int32 = 1

    // SentencePiece vocab: id → piece (for encoding we need piece → id)
    private var pieceToID: [String: Int32] = [:]
    private var pieces: [String] = []  // sorted by length descending for greedy matching

    init() {
        loadVocab()
        checkModelsExist()
    }

    private func checkModelsExist() {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self,
                  let _ = self.modelURL(containing: "ImageEncoder"),
                  let _ = self.modelURL(containing: "TextEncoder") else { return }
            DispatchQueue.main.async { self.isReady = true }
        }
    }

    private func modelURL(containing name: String) -> URL? {
        guard let resourcePath = Bundle.main.resourcePath,
              let items = try? FileManager.default.contentsOfDirectory(atPath: resourcePath) else { return nil }
        for item in items where item.hasSuffix(".mlmodelc") && item.contains(name) {
            return URL(fileURLWithPath: (resourcePath as NSString).appendingPathComponent(item))
        }
        return nil
    }

    private func loadModel(containing name: String) throws -> MLModel {
        guard let url = modelURL(containing: name) else { throw ClassifierError.modelNotLoaded }
        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly
        return try MLModel(contentsOf: url, configuration: config)
    }

    private func loadVocab() {
        guard let url = Bundle.main.url(forResource: "siglip_vocab", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: String] else { return }
        for (idStr, piece) in dict {
            if let id = Int32(idStr) {
                pieceToID[piece] = id
            }
        }
        // Sort pieces by length descending for greedy matching
        pieces = Array(pieceToID.keys).sorted { $0.count > $1.count }
    }

    // MARK: - Simple SentencePiece Tokenizer

    func tokenize(_ text: String) -> [Int32] {
        // SentencePiece prepends space and lowercases
        let input = "\u{2581}" + text.lowercased().replacingOccurrences(of: " ", with: "\u{2581}")
        var tokens: [Int32] = []
        var pos = input.startIndex

        while pos < input.endIndex {
            var matched = false
            // Greedy longest-match
            for len in stride(from: min(20, input.distance(from: pos, to: input.endIndex)), through: 1, by: -1) {
                let end = input.index(pos, offsetBy: len, limitedBy: input.endIndex) ?? input.endIndex
                let sub = String(input[pos..<end])
                if let id = pieceToID[sub] {
                    tokens.append(id)
                    pos = end
                    matched = true
                    break
                }
            }
            if !matched {
                // Unknown character, skip
                pos = input.index(after: pos)
            }
        }
        tokens.append(eosTokenID)
        return tokens
    }

    // MARK: - Classification

    func classify(image: UIImage, labels: [String]) async throws -> [ClassificationResult] {
        guard isReady else { throw ClassifierError.modelNotLoaded }

        guard let resized = resizeTo224(image),
              let pixelBuffer = createPixelBuffer(from: resized) else {
            throw ClassifierError.invalidImage
        }

        // 1. Image embedding (load, run, release)
        let imageEmbedding: [Float] = try await {
            let ie = try loadModel(containing: "ImageEncoder")
            let input = try MLDictionaryFeatureProvider(dictionary: ["image": pixelBuffer])
            let output = try await ie.prediction(from: input)
            guard let emb = output.featureValue(for: "image_embedding")?.multiArrayValue else {
                throw ClassifierError.predictionFailed
            }
            return extractFloats(emb)
        }()

        // 2. Text embeddings (load once, run for each label, release)
        let te = try loadModel(containing: "TextEncoder")
        var logits: [(String, Float)] = []

        for label in labels {
            let prompt = "a photo of a " + label
            let tokenIDs = tokenize(prompt)
            let idsArray = try MLMultiArray(shape: [1, NSNumber(value: tokenIDs.count)], dataType: .int32)
            for (i, id) in tokenIDs.enumerated() {
                idsArray[[0, NSNumber(value: i)] as [NSNumber]] = NSNumber(value: id)
            }
            let input = try MLDictionaryFeatureProvider(dictionary: ["input_ids": idsArray])
            let output = try await te.prediction(from: input)
            guard let emb = output.featureValue(for: "text_embedding")?.multiArrayValue else {
                throw ClassifierError.predictionFailed
            }
            let textEmbedding = extractFloats(emb)

            var dot: Float = 0
            for i in 0..<imageEmbedding.count {
                dot += imageEmbedding[i] * textEmbedding[i]
            }
            logits.append((label, dot * logitScale))
        }

        // Softmax over labels for relative comparison
        let maxLogit = logits.map(\.1).max() ?? 0
        let exps = logits.map { exp($0.1 - maxLogit) }
        let sumExp = exps.reduce(0, +)
        var results: [ClassificationResult] = []
        for (i, (label, _)) in logits.enumerated() {
            results.append(ClassificationResult(label: label, score: exps[i] / sumExp))
        }

        return results.sorted { $0.score > $1.score }
    }

    // MARK: - Utilities

    private func extractFloats(_ array: MLMultiArray) -> [Float] {
        let count = array.count
        if array.dataType == .float16 {
            let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)
            return (0..<count).map { Float(ptr[$0]) }
        } else {
            let ptr = array.dataPointer.assumingMemoryBound(to: Float32.self)
            return (0..<count).map { ptr[$0] }
        }
    }

    private func resizeTo224(_ image: UIImage) -> UIImage? {
        let size = CGSize(width: 224, height: 224)
        UIGraphicsBeginImageContextWithOptions(size, true, 1.0)
        image.draw(in: CGRect(origin: .zero, size: size))
        let resized = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return resized
    }

    private func createPixelBuffer(from image: UIImage) -> CVPixelBuffer? {
        guard let cgImage = image.cgImage else { return nil }
        let w = cgImage.width, h = cgImage.height
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, w, h, kCVPixelFormatType_32BGRA,
                           [kCVPixelBufferCGImageCompatibilityKey: true,
                            kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary, &pb)
        guard let buffer = pb else { return nil }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let ctx = CGContext(data: CVPixelBufferGetBaseAddress(buffer),
                                 width: w, height: h, bitsPerComponent: 8,
                                 bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                 space: CGColorSpaceCreateDeviceRGB(),
                                 bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
        else { return nil }
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))
        return buffer
    }
}
