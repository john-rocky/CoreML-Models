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
            metadata: meta
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
        lines.append("import CoreML")
        lines.append("")

        // Config
        let cu = computeUnitsSwift(inspection.computeUnits)
        lines.append("let config = MLModelConfiguration()")
        lines.append("config.computeUnits = \(cu)")
        lines.append("")

        // Load
        lines.append("let model = try MLModel(")
        lines.append("    contentsOf: Bundle.main.url(forResource: \"\(inspection.fileName)\",")
        lines.append("                                withExtension: nil)!,")
        lines.append("    configuration: config")
        lines.append(")")
        lines.append("")

        // Input
        lines.append("let input = try MLDictionaryFeatureProvider(dictionary: [")
        for (i, tensor) in inspection.inputs.enumerated() {
            let comma = i < inspection.inputs.count - 1 ? "," : ""
            let comment = inputComment(tensor)
            let value = inputPlaceholder(tensor)
            lines.append("    \"\(tensor.name)\": \(value)\(comma)  // \(comment)")
        }
        lines.append("])")
        lines.append("")

        // Prediction
        lines.append("let output = try model.prediction(from: input)")

        // Outputs
        for tensor in inspection.outputs {
            let accessor = outputAccessor(tensor)
            let comment = outputComment(tensor)
            let varName = swiftVarName(tensor.name)
            lines.append("let \(varName) = output.featureValue(forName: \"\(tensor.name)\")!\(accessor)  // \(comment)")
        }

        return lines.joined(separator: "\n")
    }

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
