import CoreML
import Foundation

// MARK: - Result types

struct BenchmarkResult {
    let fileName: String
    let iterations: Int
    let warmupIterations: Int
    let latenciesMs: [Double]
    let averageMs: Double
    let medianMs: Double
    let minMs: Double
    let maxMs: Double
    let deviceName: String
    let chipName: String
    let memoryDeltaMB: Double?
}

enum BenchmarkPhase: Equatable {
    case idle
    case loadingModel
    case warmup(current: Int, total: Int)
    case measuring(current: Int, total: Int)
    case done
    case failed(String)

    var isRunning: Bool {
        switch self {
        case .loadingModel, .warmup, .measuring: return true
        default: return false
        }
    }
}

// MARK: - Runner

enum BenchmarkRunner {

    static func run(
        model: ModelEntry,
        file: FileSpec,
        iterations: Int = 10,
        warmupIterations: Int = 3,
        computeUnits: MLComputeUnits? = nil,
        onPhase: @MainActor (BenchmarkPhase) -> Void
    ) async throws -> BenchmarkResult {
        // Load
        await onPhase(.loadingModel)
        let units = computeUnits ?? ModelLoader.parseComputeUnits(file.computeUnits)
        let mlModel = try await ModelLoader.load(
            modelId: model.id, fileName: file.name, computeUnits: units
        )
        let dummyInput = try generateDummyInput(for: mlModel)

        // Warmup
        for i in 1...warmupIterations {
            try Task.checkCancellation()
            await onPhase(.warmup(current: i, total: warmupIterations))
            _ = try await mlModel.prediction(from: dummyInput)
        }

        // Measure
        let memBefore = currentMemoryMB()
        var latencies: [Double] = []
        latencies.reserveCapacity(iterations)

        let clock = ContinuousClock()
        for i in 1...iterations {
            try Task.checkCancellation()
            await onPhase(.measuring(current: i, total: iterations))
            let start = clock.now
            _ = try await mlModel.prediction(from: dummyInput)
            let elapsed = clock.now - start
            let ms = Double(elapsed.components.seconds) * 1000.0
                + Double(elapsed.components.attoseconds) / 1_000_000_000_000_000.0
            latencies.append(ms)
        }
        let memAfter = currentMemoryMB()

        // Stats
        let sorted = latencies.sorted()
        let avg = sorted.reduce(0, +) / Double(sorted.count)
        let median: Double = sorted.count % 2 == 0
            ? (sorted[sorted.count / 2 - 1] + sorted[sorted.count / 2]) / 2.0
            : sorted[sorted.count / 2]

        let (deviceName, chipName) = deviceInfo()

        let result = BenchmarkResult(
            fileName: file.name,
            iterations: iterations,
            warmupIterations: warmupIterations,
            latenciesMs: latencies,
            averageMs: avg,
            medianMs: median,
            minMs: sorted.first ?? 0,
            maxMs: sorted.last ?? 0,
            deviceName: deviceName,
            chipName: chipName,
            memoryDeltaMB: memAfter != nil && memBefore != nil
                ? memAfter! - memBefore! : nil
        )

        await onPhase(.done)
        return result
    }

    // MARK: - Dummy input

    static func generateDummyInput(for model: MLModel) throws -> MLDictionaryFeatureProvider {
        var dict: [String: MLFeatureValue] = [:]
        for (name, desc) in model.modelDescription.inputDescriptionsByName {
            dict[name] = try dummyValue(for: desc)
        }
        return try MLDictionaryFeatureProvider(dictionary: dict)
    }

    private static func dummyValue(for desc: MLFeatureDescription) throws -> MLFeatureValue {
        switch desc.type {
        case .multiArray:
            guard let c = desc.multiArrayConstraint else {
                throw BenchmarkError.unsupportedInput("MultiArray without constraint")
            }
            let array = try MLMultiArray(shape: c.shape, dataType: c.dataType)
            return MLFeatureValue(multiArray: array)

        case .image:
            guard let c = desc.imageConstraint else {
                throw BenchmarkError.unsupportedInput("Image without constraint")
            }
            guard let buf = grayPixelBuffer(width: c.pixelsWide, height: c.pixelsHigh,
                                            pixelFormat: c.pixelFormatType) else {
                throw BenchmarkError.unsupportedInput("Failed to create pixel buffer")
            }
            return MLFeatureValue(pixelBuffer: buf)

        case .int64:
            return MLFeatureValue(int64: 0)

        case .double:
            return MLFeatureValue(double: 0.0)

        case .string:
            return MLFeatureValue(string: "")

        default:
            throw BenchmarkError.unsupportedInput("Unsupported feature type: \(desc.type)")
        }
    }

    private static func grayPixelBuffer(width: Int, height: Int, pixelFormat: OSType) -> CVPixelBuffer? {
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
        ]
        CVPixelBufferCreate(kCFAllocatorDefault, width, height, pixelFormat, attrs as CFDictionary, &pb)
        guard let buf = pb else { return nil }
        CVPixelBufferLockBaseAddress(buf, [])
        if let base = CVPixelBufferGetBaseAddress(buf) {
            let size = CVPixelBufferGetDataSize(buf)
            memset(base, 0x80, size)
        }
        CVPixelBufferUnlockBaseAddress(buf, [])
        return buf
    }

    // MARK: - Device info

    static func deviceInfo() -> (model: String, chip: String) {
        let machine = currentMachineIdentifier()
        let model = deviceNames[machine] ?? machine
        // Exact match first, then longest prefix match. Fall back to the raw
        // identifier so we at least show something actionable.
        let chip: String
        if let exact = chipNames[machine] {
            chip = exact
        } else if let prefix = chipNames
            .filter({ machine.hasPrefix($0.key) })
            .max(by: { $0.key.count < $1.key.count })?.value {
            chip = prefix
        } else {
            chip = machine
        }
        return (model, chip)
    }

    private static func currentMachineIdentifier() -> String {
        // On the simulator `hw.machine` returns the host architecture
        // (arm64 / x86_64). The real device identifier is exposed via an
        // environment variable.
        if let simID = ProcessInfo.processInfo.environment["SIMULATOR_MODEL_IDENTIFIER"],
           !simID.isEmpty {
            return simID
        }
        return sysctlString("hw.machine") ?? "Unknown"
    }

    private static func sysctlString(_ name: String) -> String? {
        var size: Int = 0
        sysctlbyname(name, nil, &size, nil, 0)
        guard size > 0 else { return nil }
        var buf = [CChar](repeating: 0, count: size)
        sysctlbyname(name, &buf, &size, nil, 0)
        return String(cString: buf)
    }

    static func currentMemoryMB() -> Double? {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }
        return Double(info.resident_size) / (1024 * 1024)
    }

    // MARK: - Device name mappings

    private static let deviceNames: [String: String] = [
        // iPhone 17 / Air
        "iPhone18,1": "iPhone 17 Pro", "iPhone18,2": "iPhone 17 Pro Max",
        "iPhone18,3": "iPhone 17", "iPhone18,4": "iPhone Air",
        // iPhone 16
        "iPhone17,1": "iPhone 16 Pro", "iPhone17,2": "iPhone 16 Pro Max",
        "iPhone17,3": "iPhone 16", "iPhone17,4": "iPhone 16 Plus",
        "iPhone17,5": "iPhone 16e",
        // iPhone 15
        "iPhone16,1": "iPhone 15 Pro", "iPhone16,2": "iPhone 15 Pro Max",
        "iPhone15,4": "iPhone 15", "iPhone15,5": "iPhone 15 Plus",
        // iPhone 14
        "iPhone15,2": "iPhone 14 Pro", "iPhone15,3": "iPhone 14 Pro Max",
        "iPhone14,7": "iPhone 14", "iPhone14,8": "iPhone 14 Plus",
        // iPhone 13
        "iPhone14,2": "iPhone 13 Pro", "iPhone14,3": "iPhone 13 Pro Max",
        "iPhone14,5": "iPhone 13", "iPhone14,4": "iPhone 13 mini",
        // iPhone SE (2nd / 3rd gen)
        "iPhone12,8": "iPhone SE (2nd gen)", "iPhone14,6": "iPhone SE (3rd gen)",
        // iPhone 12
        "iPhone13,1": "iPhone 12 mini", "iPhone13,2": "iPhone 12",
        "iPhone13,3": "iPhone 12 Pro", "iPhone13,4": "iPhone 12 Pro Max",
        // iPhone 11
        "iPhone12,1": "iPhone 11",
        "iPhone12,3": "iPhone 11 Pro", "iPhone12,5": "iPhone 11 Pro Max",
        // iPad Pro M4
        "iPad16,3": "iPad Pro 11\" (M4)", "iPad16,4": "iPad Pro 11\" (M4)",
        "iPad16,5": "iPad Pro 13\" (M4)", "iPad16,6": "iPad Pro 13\" (M4)",
        // iPad Air M2/M3
        "iPad14,8": "iPad Air 11\" (M2)", "iPad14,9": "iPad Air 11\" (M2)",
        "iPad14,10": "iPad Air 13\" (M2)", "iPad14,11": "iPad Air 13\" (M2)",
        // iPad mini 7 (A17 Pro)
        "iPad16,1": "iPad mini (7th gen)", "iPad16,2": "iPad mini (7th gen)",
        // iPad (11th gen, A16) / iPad (10th gen, A14)
        "iPad15,7": "iPad (11th gen)", "iPad15,8": "iPad (11th gen)",
        "iPad13,18": "iPad (10th gen)", "iPad13,19": "iPad (10th gen)",
        // iPad Pro M2 (2022)
        "iPad14,3": "iPad Pro 11\" (M2)", "iPad14,4": "iPad Pro 11\" (M2)",
        "iPad14,5": "iPad Pro 12.9\" (M2)", "iPad14,6": "iPad Pro 12.9\" (M2)",
        // iPad Air 5 / iPad Pro M1
        "iPad13,16": "iPad Air (5th gen)", "iPad13,17": "iPad Air (5th gen)",
        "iPad13,4": "iPad Pro 11\" (M1)", "iPad13,5": "iPad Pro 11\" (M1)",
        "iPad13,6": "iPad Pro 11\" (M1)", "iPad13,7": "iPad Pro 11\" (M1)",
        "iPad13,8": "iPad Pro 12.9\" (M1)", "iPad13,9": "iPad Pro 12.9\" (M1)",
        "iPad13,10": "iPad Pro 12.9\" (M1)", "iPad13,11": "iPad Pro 12.9\" (M1)",
        // iPad mini 6 (A15)
        "iPad14,1": "iPad mini (6th gen)", "iPad14,2": "iPad mini (6th gen)",
    ]

    private static let chipNames: [String: String] = [
        // iPhone 17 / Air — A19 / A19 Pro
        "iPhone18,1": "A19 Pro", "iPhone18,2": "A19 Pro",
        "iPhone18,3": "A19", "iPhone18,4": "A19 Pro",
        // iPhone 16
        "iPhone17,1": "A18 Pro", "iPhone17,2": "A18 Pro",
        "iPhone17,3": "A18", "iPhone17,4": "A18", "iPhone17,5": "A18",
        // iPhone 15
        "iPhone16,1": "A17 Pro", "iPhone16,2": "A17 Pro",
        "iPhone15,4": "A16", "iPhone15,5": "A16",
        // iPhone 14
        "iPhone15,2": "A16", "iPhone15,3": "A16",
        "iPhone14,7": "A15", "iPhone14,8": "A15",
        // iPhone 13
        "iPhone14,2": "A15", "iPhone14,3": "A15",
        "iPhone14,4": "A15", "iPhone14,5": "A15",
        // iPhone SE 2 / 3
        "iPhone12,8": "A13", "iPhone14,6": "A15",
        // iPhone 12
        "iPhone13,1": "A14", "iPhone13,2": "A14",
        "iPhone13,3": "A14", "iPhone13,4": "A14",
        // iPhone 11
        "iPhone12,1": "A13", "iPhone12,3": "A13", "iPhone12,5": "A13",
        // iPad Pro M4
        "iPad16,3": "M4", "iPad16,4": "M4", "iPad16,5": "M4", "iPad16,6": "M4",
        // iPad mini 7
        "iPad16,1": "A17 Pro", "iPad16,2": "A17 Pro",
        // iPad 11 / 10
        "iPad15,7": "A16", "iPad15,8": "A16",
        "iPad13,18": "A14", "iPad13,19": "A14",
        // iPad Air M2
        "iPad14,8": "M2", "iPad14,9": "M2", "iPad14,10": "M2", "iPad14,11": "M2",
        // iPad Pro M2
        "iPad14,3": "M2", "iPad14,4": "M2", "iPad14,5": "M2", "iPad14,6": "M2",
        // iPad Air 5 / iPad Pro M1
        "iPad13,16": "M1", "iPad13,17": "M1",
        "iPad13,4": "M1", "iPad13,5": "M1", "iPad13,6": "M1", "iPad13,7": "M1",
        "iPad13,8": "M1", "iPad13,9": "M1", "iPad13,10": "M1", "iPad13,11": "M1",
        // iPad mini 6
        "iPad14,1": "A15", "iPad14,2": "A15",
    ]

    enum BenchmarkError: LocalizedError {
        case unsupportedInput(String)
        var errorDescription: String? {
            switch self {
            case .unsupportedInput(let msg): return "Unsupported input: \(msg)"
            }
        }
    }
}
