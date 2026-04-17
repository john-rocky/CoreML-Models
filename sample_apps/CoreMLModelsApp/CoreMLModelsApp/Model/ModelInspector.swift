import CoreML
import Foundation

// MARK: - Inspection result types

struct TensorInfo: Identifiable {
    let id: String
    let name: String
    let featureType: String
    let shape: [Int]?
    let dataType: String
}

struct ModelFileInspection: Identifiable {
    let id: String
    let fileName: String
    let computeUnits: String
    let sizeOnDisk: Int64
    let inputs: [TensorInfo]
    let outputs: [TensorInfo]
    let metadata: [(key: String, value: String)]
    let downloadUrl: String?
    let archive: String?
}

// MARK: - Inspector

enum ModelInspector {

    static func inspect(model: ModelEntry) async throws -> [ModelFileInspection] {
        let modelFiles = model.files.filter { ($0.kind ?? "model") == "model" }
        guard !modelFiles.isEmpty else { throw InspectError.noModelFiles }

        return try await withThrowingTaskGroup(of: ModelFileInspection.self) { group in
            for file in modelFiles {
                group.addTask { try await inspectFile(model: model, file: file) }
            }
            var results: [ModelFileInspection] = []
            for try await inspection in group { results.append(inspection) }
            // Preserve manifest order
            return results.sorted { a, b in
                let ai = modelFiles.firstIndex(where: { $0.name == a.fileName }) ?? 0
                let bi = modelFiles.firstIndex(where: { $0.name == b.fileName }) ?? 0
                return ai < bi
            }
        }
    }

    static func inspectFile(model: ModelEntry, file: FileSpec) async throws -> ModelFileInspection {
        let mlModel = try await ModelLoader.load(
            for: model, named: file.name
        )
        let desc = mlModel.modelDescription

        let inputs = desc.inputDescriptionsByName
            .sorted(by: { $0.key < $1.key })
            .map { tensorInfo(name: $0.key, desc: $0.value) }

        let outputs = desc.outputDescriptionsByName
            .sorted(by: { $0.key < $1.key })
            .map { tensorInfo(name: $0.key, desc: $0.value) }

        let meta = extractMetadata(mlModel)

        let resolvedName = resolveFileName(file.name)
        let fileDir = Paths.modelDir(id: model.id)
        let size = directorySize(fileDir.appendingPathComponent(resolvedName))
            ?? Int64(file.sizeBytes)

        return ModelFileInspection(
            id: file.name,
            fileName: resolvedName,
            computeUnits: file.computeUnits ?? "all",
            sizeOnDisk: size,
            inputs: inputs,
            outputs: outputs,
            metadata: meta,
            downloadUrl: file.url,
            archive: file.archive
        )
    }

    static func diskSize(modelId: String) -> Int64 {
        directorySize(Paths.modelDir(id: modelId)) ?? 0
    }

    // MARK: - Private

    private static func tensorInfo(name: String, desc: MLFeatureDescription) -> TensorInfo {
        let kind: String
        var shape: [Int]?
        var dtype: String

        switch desc.type {
        case .multiArray:
            kind = "MultiArray"
            if let c = desc.multiArrayConstraint {
                shape = c.shape.map { $0.intValue }
                dtype = dataTypeName(c.dataType)
            } else {
                dtype = "Unknown"
            }
        case .image:
            kind = "Image"
            if let c = desc.imageConstraint {
                shape = [c.pixelsHigh, c.pixelsWide]
                dtype = pixelFormatName(c.pixelFormatType)
            } else {
                dtype = "Unknown"
            }
        case .int64:
            kind = "Int64"; dtype = "Int64"
        case .double:
            kind = "Double"; dtype = "Float64"
        case .string:
            kind = "String"; dtype = "String"
        case .dictionary:
            kind = "Dictionary"; dtype = "Dictionary"
        case .sequence:
            kind = "Sequence"; dtype = "Sequence"
        @unknown default:
            kind = "Unknown"; dtype = "Unknown"
        }

        return TensorInfo(id: name, name: name, featureType: kind, shape: shape, dataType: dtype)
    }

    private static func dataTypeName(_ dt: MLMultiArrayDataType) -> String {
        switch dt {
        case .float16: return "Float16"
        case .float32: return "Float32"
        case .float64: return "Float64"
        case .int32:   return "Int32"
        @unknown default: return "Unknown"
        }
    }

    private static func pixelFormatName(_ fmt: OSType) -> String {
        switch fmt {
        case kCVPixelFormatType_32BGRA: return "BGRA8"
        case kCVPixelFormatType_32RGBA: return "RGBA8"
        case kCVPixelFormatType_OneComponent8: return "Gray8"
        case kCVPixelFormatType_OneComponent16Half: return "Gray16Half"
        default: return String(format: "0x%08X", fmt)
        }
    }

    private static func extractMetadata(_ model: MLModel) -> [(key: String, value: String)] {
        let meta = model.modelDescription.metadata
        var result: [(key: String, value: String)] = []
        if let v = meta[.author] as? String, !v.isEmpty { result.append(("Author", v)) }
        if let v = meta[.description] as? String, !v.isEmpty { result.append(("Description", v)) }
        if let v = meta[.license] as? String, !v.isEmpty { result.append(("License", v)) }
        if let v = meta[.versionString] as? String, !v.isEmpty { result.append(("Version", v)) }
        if let dict = meta[.creatorDefinedKey] as? [String: String] {
            for (k, v) in dict.sorted(by: { $0.key < $1.key }) {
                result.append((k, v))
            }
        }
        return result
    }

    private static func resolveFileName(_ name: String) -> String {
        var resolved = name
        for ext in [".zip", ".tar.gz"] {
            if resolved.hasSuffix(ext) { resolved = String(resolved.dropLast(ext.count)) }
        }
        return resolved
    }

    private static func directorySize(_ url: URL) -> Int64? {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(
            at: url, includingPropertiesForKeys: [.fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else { return nil }

        var total: Int64 = 0
        for case let fileURL as URL in enumerator {
            if let size = try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                total += Int64(size)
            }
        }
        return total > 0 ? total : nil
    }

    // MARK: - Code snippet generation

    static func generateSnippet(for inspection: ModelFileInspection) -> String {
        var lines: [String] = []
        let hasImageInput = inspection.inputs.contains { $0.featureType == "Image" }
        let hasMultiArrayInput = inspection.inputs.contains { $0.featureType == "MultiArray" }
        let hasMultiArrayOutput = inspection.outputs.contains { $0.featureType == "MultiArray" }

        // Imports
        lines.append("import CoreML")
        if hasImageInput {
            lines.append("import CoreVideo")
            lines.append("import UIKit")
        }
        lines.append("")

        // Download-from-HF option (as comment)
        if let url = inspection.downloadUrl {
            lines.append("// MARK: - Load the model")
            lines.append("//")
            lines.append("// Option A · Download from Hugging Face at first launch:")
            lines.append("//   1. Download \(url)")
            if let ar = inspection.archive, !ar.isEmpty {
                lines.append("//   2. Unarchive (.\(ar)) to get \(inspection.fileName)")
            }
            lines.append("//   3. Pass the resulting \(inspection.fileName) URL to MLModel.compileModel(at:).")
            lines.append("//")
            lines.append("// Option B · Add \(inspection.fileName) to your Xcode target (shown below).")
            lines.append("")
        }

        // Load
        let resourceName = (inspection.fileName as NSString).deletingPathExtension
        let pathExt = (inspection.fileName as NSString).pathExtension
        lines.append("let modelURL = Bundle.main.url(forResource: \"\(resourceName)\",")
        lines.append("                               withExtension: \"\(pathExt)\")!")
        if pathExt == "mlpackage" {
            lines.append("let compiledURL = try MLModel.compileModel(at: modelURL)")
        } else {
            lines.append("let compiledURL = modelURL")
        }
        lines.append("")

        // Config
        let cu = computeUnitsSwift(inspection.computeUnits)
        lines.append("let config = MLModelConfiguration()")
        lines.append("config.computeUnits = \(cu)  // recommended for this model")
        lines.append("let model = try MLModel(contentsOf: compiledURL, configuration: config)")
        lines.append("")

        // Input preparation
        lines.append("// MARK: - Prepare inputs")
        for tensor in inspection.inputs {
            lines.append(contentsOf: inputPreparation(for: tensor))
        }
        lines.append("")

        // Build feature provider
        lines.append("let input = try MLDictionaryFeatureProvider(dictionary: [")
        for (i, tensor) in inspection.inputs.enumerated() {
            let comma = i < inspection.inputs.count - 1 ? "," : ""
            let value = inputFeatureValue(tensor)
            lines.append("    \"\(tensor.name)\": \(value)\(comma)")
        }
        lines.append("])")
        lines.append("")

        // Prediction
        lines.append("// MARK: - Run inference")
        lines.append("let output = try model.prediction(from: input)")
        lines.append("")

        // Outputs
        lines.append("// MARK: - Read outputs")
        for tensor in inspection.outputs {
            let accessor = outputAccessor(tensor)
            let comment = outputComment(tensor)
            let varName = swiftVarName(tensor.name)
            lines.append("let \(varName) = output.featureValue(for: \"\(tensor.name)\")!\(accessor)  // \(comment)")
        }

        // Output reading tip for MultiArray (stride-aware)
        if hasMultiArrayOutput, let sample = inspection.outputs.first(where: { $0.featureType == "MultiArray" }) {
            lines.append("")
            lines.append("// Read values safely via strides (ANE pads rows for SIMD alignment,")
            lines.append("// so `.dataPointer` is not C-contiguous on .all / .cpuAndNeuralEngine).")
            lines.append("let \(swiftVarName(sample.name))Strides = \(swiftVarName(sample.name)).strides.map { $0.intValue }")
            if let s = sample.shape, s.count >= 2 {
                let idx = (0..<s.count).map { _ in "0" }.joined(separator: ", ")
                lines.append("// let offset = \(swiftVarName(sample.name))Strides[0]*0 + ... at indices [\(idx)]")
            }
        }

        // Helpers
        if hasImageInput {
            lines.append("")
            lines.append("// MARK: - Helpers")
            lines.append(bgraHelperSource)
        }
        if hasMultiArrayInput {
            if !hasImageInput { lines.append("") ; lines.append("// MARK: - Helpers") }
            lines.append("")
            lines.append(multiArrayFillStub)
        }

        return lines.joined(separator: "\n")
    }

    /// Per-input preparation lines (pixel buffer, MLMultiArray init, primitive default).
    private static func inputPreparation(for tensor: TensorInfo) -> [String] {
        let varName = swiftVarName(tensor.name)
        switch tensor.featureType {
        case "Image":
            let h = tensor.shape?.first ?? 0
            let w = tensor.shape?.last ?? 0
            return [
                "// \"\(tensor.name)\": Image \(h) × \(w) (\(tensor.dataType))",
                "let image: UIImage = /* your image */",
                "guard let \(varName)Buffer = makeBGRAPixelBuffer(from: image, width: \(w), height: \(h)) else {",
                "    fatalError(\"Failed to create pixel buffer for \\\"\(tensor.name)\\\"\")",
                "}",
            ]
        case "MultiArray":
            let shape = tensor.shape ?? []
            let shapeLiteral = "[\(shape.map(String.init).joined(separator: ", "))]"
            let dtype = mlArrayDataTypeSwift(tensor.dataType)
            return [
                "// \"\(tensor.name)\": \(shape.map(String.init).joined(separator: " × ")) (\(tensor.dataType))",
                "let \(varName) = try MLMultiArray(shape: \(shapeLiteral), dataType: \(dtype))",
                "fill(\(varName))  // TODO: populate via \(varName).dataPointer",
            ]
        case "Int64":
            return ["let \(varName): Int64 = 0  // \"\(tensor.name)\""]
        case "Double":
            return ["let \(varName): Double = 0.0  // \"\(tensor.name)\""]
        case "String":
            return ["let \(varName): String = \"\"  // \"\(tensor.name)\""]
        default:
            return ["// \"\(tensor.name)\": \(tensor.featureType) (not yet scaffolded)"]
        }
    }

    private static func inputFeatureValue(_ t: TensorInfo) -> String {
        let v = swiftVarName(t.name)
        switch t.featureType {
        case "Image":      return "MLFeatureValue(pixelBuffer: \(v)Buffer)"
        case "MultiArray": return "MLFeatureValue(multiArray: \(v))"
        case "Int64":      return "MLFeatureValue(int64: \(v))"
        case "Double":     return "MLFeatureValue(double: \(v))"
        case "String":     return "MLFeatureValue(string: \(v))"
        default:           return "/* \(t.featureType) */"
        }
    }

    private static func mlArrayDataTypeSwift(_ name: String) -> String {
        switch name {
        case "Float16": return ".float16"
        case "Float32": return ".float32"
        case "Float64": return ".float64"
        case "Int32":   return ".int32"
        default:        return ".float32"
        }
    }

    private static let bgraHelperSource = """
    /// Create a BGRA CVPixelBuffer sized width × height from a UIImage.
    func makeBGRAPixelBuffer(from image: UIImage, width: Int, height: Int) -> CVPixelBuffer? {
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        ]
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                            kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
        guard let buffer = pb, let cg = image.cgImage else { return nil }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let ctx = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
                      | CGBitmapInfo.byteOrder32Little.rawValue
        ) else { return nil }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
        return buffer
    }
    """

    private static let multiArrayFillStub = """
    /// Replace with your real tensor population. Cast `dataPointer` to Float16/Float/Int32
    /// depending on `array.dataType` and write via strides for correctness on ANE.
    func fill(_ array: MLMultiArray) {
        // let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
        // for i in 0..<array.count { ptr[i] = 0 }
    }
    """

    private static func computeUnitsSwift(_ cu: String) -> String {
        switch cu {
        case "cpuOnly": return ".cpuOnly"
        case "cpuAndGPU": return ".cpuAndGPU"
        case "cpuAndNeuralEngine": return ".cpuAndNeuralEngine"
        default: return ".all"
        }
    }

    private static func inputComment(_ t: TensorInfo) -> String {
        if let shape = t.shape {
            let dims = shape.map(String.init).joined(separator: " \u{00D7} ")
            return "\(dims), \(t.dataType)"
        }
        return t.dataType
    }

    private static func inputPlaceholder(_ t: TensorInfo) -> String {
        switch t.featureType {
        case "Image":
            return "MLFeatureValue(pixelBuffer: pixelBuffer)"
        case "MultiArray":
            return "MLFeatureValue(multiArray: \(swiftVarName(t.name)))"
        case "Int64":
            return "MLFeatureValue(int64: 0)"
        case "Double":
            return "MLFeatureValue(double: 0.0)"
        case "String":
            return "MLFeatureValue(string: \"\")"
        default:
            return "/* \(t.featureType) */"
        }
    }

    private static func outputAccessor(_ t: TensorInfo) -> String {
        switch t.featureType {
        case "MultiArray": return ".multiArrayValue!"
        case "Image":      return ".imageBufferValue!"
        case "Int64":      return ".int64Value"
        case "Double":     return ".doubleValue"
        case "String":     return ".stringValue"
        case "Dictionary": return ".dictionaryValue"
        default:           return ""
        }
    }

    private static func outputComment(_ t: TensorInfo) -> String {
        inputComment(t)
    }

    private static func swiftVarName(_ tensorName: String) -> String {
        let cleaned = tensorName
            .replacingOccurrences(of: ".", with: "_")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "-", with: "_")
        // Ensure it starts with lowercase
        guard let first = cleaned.first else { return "value" }
        return first.lowercased() + cleaned.dropFirst()
    }

    enum InspectError: LocalizedError {
        case noModelFiles
        var errorDescription: String? { "No model files found in this entry" }
    }
}
