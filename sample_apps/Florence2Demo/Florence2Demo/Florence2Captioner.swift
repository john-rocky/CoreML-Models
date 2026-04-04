import CoreML
import UIKit

enum Florence2Task: String, CaseIterable {
    case caption = "Caption"
    case detailedCaption = "Detailed Caption"
    case moreDetailedCaption = "More Detailed Caption"
    case ocr = "OCR"

    var inputIDs: [Int32] {
        switch self {
        case .caption:
            return [0, 2264, 473, 5, 2274, 6190, 116, 2]
        case .detailedCaption:
            return [0, 47066, 21700, 11, 4617, 99, 16, 2343, 11, 5, 2274, 4, 2]
        case .moreDetailedCaption:
            return [0, 47066, 21700, 19, 10, 17818, 99, 16, 2343, 11, 5, 2274, 4, 2]
        case .ocr:
            return [0, 2264, 16, 5, 2788, 11, 5, 2274, 116, 2]
        }
    }
}

enum CaptionerError: LocalizedError {
    case invalidImage
    case modelNotLoaded
    case predictionFailed

    var errorDescription: String? {
        switch self {
        case .invalidImage: return "Failed to process image"
        case .modelNotLoaded: return "Model not loaded"
        case .predictionFailed: return "Prediction failed"
        }
    }
}

@MainActor
class Florence2Captioner: ObservableObject {
    @Published var isReady = false

    private var reverseVocab: [Int: String] = [:]
    private var forwardVocab: [String: Int] = [:]

    private let eosTokenID: Int32 = 2
    private let decoderStartTokenID: Int32 = 2
    private let specialTokenIDs: Set<Int32> = [0, 1, 2]

    // GPT-2/RoBERTa byte-to-unicode mapping
    private static let byteEncoder: [UInt8: Character] = {
        var directBytes = Set<UInt8>()
        for b in 0x21...0x7E { directBytes.insert(UInt8(b)) }
        for b in 0xA1...0xAC { directBytes.insert(UInt8(b)) }
        for b in 0xAE...0xFF { directBytes.insert(UInt8(b)) }

        var encoder: [UInt8: Character] = [:]
        for b in directBytes {
            encoder[b] = Character(UnicodeScalar(b))
        }
        var n = 0
        for b: UInt8 in 0...255 {
            if !directBytes.contains(b) {
                encoder[b] = Character(UnicodeScalar(256 + n)!)
                n += 1
            }
        }
        return encoder
    }()

    init() {
        loadVocab()
        checkModelsExist()
    }

    private func checkModelsExist() {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self,
                  let _ = self.modelURL(containing: "VisionEncoder"),
                  let _ = self.modelURL(containing: "TextEncoder"),
                  let _ = self.modelURL(containing: "Decoder") else { return }
            DispatchQueue.main.async { self.isReady = true }
        }
    }

    private func modelURL(containing name: String) -> URL? {
        guard let resourcePath = Bundle.main.resourcePath,
              let items = try? FileManager.default.contentsOfDirectory(atPath: resourcePath) else {
            return nil
        }
        for item in items where item.hasSuffix(".mlmodelc") && item.contains(name) {
            return URL(fileURLWithPath: (resourcePath as NSString).appendingPathComponent(item))
        }
        return nil
    }

    private func loadModel(containing name: String) throws -> MLModel {
        guard let url = modelURL(containing: name) else {
            throw CaptionerError.modelNotLoaded
        }
        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly
        return try MLModel(contentsOf: url, configuration: config)
    }

    private func loadVocab() {
        guard let url = Bundle.main.url(forResource: "florence2_vocab", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: String] else {
            return
        }
        for (key, value) in dict {
            if let id = Int(key) {
                reverseVocab[id] = value
                forwardVocab[value] = id
            }
        }
    }

    // MARK: - Tokenizer

    func tokenize(_ text: String) -> [Int32] {
        var tokens: [Int32] = [0] // BOS <s>

        let pattern = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
        let regex = try! NSRegularExpression(pattern: pattern, options: [])
        let nsText = text as NSString
        let matches = regex.matches(in: text, range: NSRange(location: 0, length: nsText.length))

        for match in matches {
            let piece = nsText.substring(with: match.range)
            let encoded = byteEncode(piece)

            // Greedy longest match
            var i = encoded.startIndex
            while i < encoded.endIndex {
                var bestEnd = encoded.index(after: i)
                var bestID: Int?
                var j = encoded.index(after: i)
                while j <= encoded.endIndex {
                    let sub = String(encoded[i..<j])
                    if let id = forwardVocab[sub] {
                        bestEnd = j
                        bestID = id
                    }
                    if j == encoded.endIndex { break }
                    j = encoded.index(after: j)
                }
                if let id = bestID {
                    tokens.append(Int32(id))
                }
                i = bestEnd
            }
        }

        tokens.append(2) // EOS </s>
        return tokens
    }

    private func byteEncode(_ text: String) -> String {
        var result = ""
        for byte in Array(text.utf8) {
            if let ch = Self.byteEncoder[byte] {
                result.append(ch)
            }
        }
        return result
    }

    // MARK: - Inference

    func caption(image: UIImage, task: Florence2Task, maxTokens: Int = 256) async throws -> String {
        return try await infer(image: image, inputIDs: task.inputIDs, maxTokens: maxTokens)
    }

    func answer(image: UIImage, question: String, maxTokens: Int = 256) async throws -> String {
        let inputIDs = tokenize(question)
        return try await infer(image: image, inputIDs: inputIDs, maxTokens: maxTokens)
    }

    private func infer(image: UIImage, inputIDs: [Int32], maxTokens: Int) async throws -> String {
        guard isReady else { throw CaptionerError.modelNotLoaded }

        guard let resized = resizeTo768(image),
              let pixelBuffer = createPixelBuffer(from: resized) else {
            throw CaptionerError.invalidImage
        }

        // Step 1: VisionEncoder
        let imageFeatures: MLMultiArray = try await {
            let ve = try loadModel(containing: "VisionEncoder")
            let input = try MLDictionaryFeatureProvider(dictionary: ["image": pixelBuffer])
            let output = try await ve.prediction(from: input)
            guard let feat = output.featureValue(for: "image_features")?.multiArrayValue else {
                throw CaptionerError.predictionFailed
            }
            return try copyMultiArray(feat)
        }()

        // Step 2: TextEncoder
        let encoderHS: MLMultiArray = try await {
            let te = try loadModel(containing: "TextEncoder")
            let idsArray = try MLMultiArray(shape: [1, NSNumber(value: inputIDs.count)], dataType: .int32)
            for (i, id) in inputIDs.enumerated() {
                idsArray[[0, NSNumber(value: i)] as [NSNumber]] = NSNumber(value: id)
            }
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "image_features": imageFeatures,
                "input_ids": idsArray
            ])
            let output = try await te.prediction(from: input)
            guard let hs = output.featureValue(for: "encoder_hidden_states")?.multiArrayValue else {
                throw CaptionerError.predictionFailed
            }
            return try copyMultiArray(hs)
        }()

        // Step 3: Decoder autoregressive loop
        let dec = try loadModel(containing: "Decoder")
        var decoderIDs: [Int32] = [decoderStartTokenID]

        for _ in 0..<maxTokens {
            let decArray = try MLMultiArray(shape: [1, NSNumber(value: decoderIDs.count)], dataType: .int32)
            for (i, id) in decoderIDs.enumerated() {
                decArray[[0, NSNumber(value: i)] as [NSNumber]] = NSNumber(value: id)
            }
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "decoder_input_ids": decArray,
                "encoder_hidden_states": encoderHS
            ])
            let output = try await dec.prediction(from: input)
            guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
                throw CaptionerError.predictionFailed
            }
            let nextToken = argmax(logits)
            if nextToken == eosTokenID { break }
            decoderIDs.append(nextToken)
        }

        return decodeTokens(Array(decoderIDs.dropFirst()))
    }

    // MARK: - Helpers

    private func copyMultiArray(_ src: MLMultiArray) throws -> MLMultiArray {
        let dst = try MLMultiArray(shape: src.shape, dataType: src.dataType)
        let byteCount: Int
        switch src.dataType {
        case .float16: byteCount = src.count * 2
        case .float32: byteCount = src.count * 4
        case .int32:   byteCount = src.count * 4
        default:       byteCount = src.count * 4
        }
        memcpy(dst.dataPointer, src.dataPointer, byteCount)
        return dst
    }

    private func decodeTokens(_ tokenIDs: [Int32]) -> String {
        var pieces: [String] = []
        for tid in tokenIDs {
            if specialTokenIDs.contains(tid) { continue }
            if let piece = reverseVocab[Int(tid)] {
                pieces.append(piece)
            }
        }
        var text = pieces.joined()
        text = text.replacingOccurrences(of: "\u{0120}", with: " ")
        text = text.trimmingCharacters(in: .whitespaces)
        return text
    }

    private func argmax(_ array: MLMultiArray) -> Int32 {
        let vocabSize = array.shape.last!.intValue
        let offset = array.count - vocabSize

        if array.dataType == .float16 {
            let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)
            var maxIdx = 0
            var maxVal = ptr[offset]
            for i in 1..<vocabSize {
                let val = ptr[offset + i]
                if val > maxVal { maxVal = val; maxIdx = i }
            }
            return Int32(maxIdx)
        } else {
            let ptr = array.dataPointer.assumingMemoryBound(to: Float32.self)
            var maxIdx = 0
            var maxVal = ptr[offset]
            for i in 1..<vocabSize {
                let val = ptr[offset + i]
                if val > maxVal { maxVal = val; maxIdx = i }
            }
            return Int32(maxIdx)
        }
    }

    private func resizeTo768(_ image: UIImage) -> UIImage? {
        let size = CGSize(width: 768, height: 768)
        UIGraphicsBeginImageContextWithOptions(size, true, 1.0)
        image.draw(in: CGRect(origin: .zero, size: size))
        let resized = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return resized
    }

    private func createPixelBuffer(from image: UIImage) -> CVPixelBuffer? {
        guard let cgImage = image.cgImage else { return nil }
        let width = cgImage.width
        let height = cgImage.height
        var pixelBuffer: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, width, height,
            kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pixelBuffer
        )
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return nil }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else { return nil }
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        return buffer
    }
}
