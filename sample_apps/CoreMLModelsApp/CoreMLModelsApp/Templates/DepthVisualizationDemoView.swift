import SwiftUI
import PhotosUI
import CoreML

struct DepthVisualizationDemoView: View {
    let model: ModelEntry

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

    // "meters", "relative", or nil. Controls the range label unit suffix.
    private var depthUnit: String {
        model.configString("depth_unit") ?? "meters"
    }

    var body: some View {
        VStack(spacing: 0) {
            displayArea.frame(maxHeight: .infinity)

            if depthImage != nil {
                Picker("View", selection: $viewMode) {
                    ForEach(availableModes) { Text($0.label).tag($0) }
                }
                .pickerStyle(.segmented)
                .padding(.horizontal)

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
            }

            VStack(spacing: 12) {
                if isProcessing { ProgressView(status) }
                Text(status).font(.caption).foregroundStyle(.secondary)
                PhotosPicker(selection: $item, matching: .images) {
                    Label("Select Photo", systemImage: "photo.badge.plus").frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
            }
            .padding()
        }
        .task {
            session.ensure { try await ModelLoader.loadPrimary(for: model) }
        }
        .onChange(of: item) { _, newItem in
            print("[Depth] onChange fired, newItem=\(String(describing: newItem))")
            loadAndRun()
        }
    }

    @ViewBuilder
    private var displayArea: some View {
        let img: UIImage? = {
            switch viewMode {
            case .original: return inputImage
            case .depth: return depthImage
            case .normal: return normalImage
            case .confidence: return confidenceImage
            }
        }()
        if let img {
            Image(uiImage: img).resizable().aspectRatio(contentMode: .fit)
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

    private func loadAndRun() {
        guard let item else {
            print("[Depth] loadAndRun: item is nil, returning")
            return
        }
        print("[Depth] loadAndRun: starting with item=\(item)")
        isProcessing = true; status = "Loading photo…"
        Task {
            do {
                guard let data = try await item.loadTransferable(type: Data.self) else {
                    print("[Depth] loadTransferable returned nil")
                    await MainActor.run { isProcessing = false; status = "Photo load returned nil" }
                    return
                }
                print("[Depth] Got data: \(data.count) bytes")
                guard let img = UIImage(data: data) else {
                    print("[Depth] UIImage(data:) failed")
                    await MainActor.run { isProcessing = false; status = "Invalid image data" }
                    return
                }
                print("[Depth] Image loaded: \(img.size)")
                await MainActor.run { inputImage = img }
                await runDepth(on: img)
            } catch {
                print("[Depth] loadTransferable error: \(error)")
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
            print("[Depth] inputSize=\(inputSize)")
            guard let cgImage = ImageUtils.normalizeOrientation(image) else {
                print("[Depth] normalizeOrientation failed")
                await MainActor.run { isProcessing = false; status = "Image prep failed" }
                return
            }
            print("[Depth] cgImage: \(cgImage.width)x\(cgImage.height)")

            guard let (pb, validRect) = ImageUtils.letterbox(cgImage, size: inputSize) else {
                print("[Depth] letterbox failed")
                await MainActor.run { isProcessing = false; status = "Letterbox failed" }
                return
            }
            print("[Depth] Letterbox done, validRect=\(validRect)")

            // Find image input name
            let inputName = mlModel.modelDescription.inputDescriptionsByName.first {
                $0.value.type == .image
            }?.key ?? "image"
            print("[Depth] Using input name: \(inputName)")

            let start = CFAbsoluteTimeGetCurrent()
            let input = try MLDictionaryFeatureProvider(dictionary: [inputName: pb])
            let output = try await mlModel.prediction(from: input)
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            print("[Depth] Inference done in \(elapsed)s, output keys: \(output.featureNames)")

            // Extract outputs
            for name in output.featureNames {
                if let arr = output.featureValue(for: name)?.multiArrayValue {
                    print("[Depth] output '\(name)': shape=\(arr.shape), dtype=\(arr.dataType.rawValue)")
                } else {
                    print("[Depth] output '\(name)': not a multiarray")
                }
            }

            let depthArr = output.featureValue(for: "depth")?.multiArrayValue
            let normalArr = output.featureValue(for: "normal")?.multiArrayValue
            let maskArr = output.featureValue(for: "mask")?.multiArrayValue
            let scaleArr = output.featureValue(for: "metric_scale")?.multiArrayValue
            let confArr = output.featureValue(for: "confidence")?.multiArrayValue

            print("[Depth] depthArr=\(depthArr != nil), normalArr=\(normalArr != nil), maskArr=\(maskArr != nil), scaleArr=\(scaleArr != nil), confArr=\(confArr != nil)")

            let metricScale: Float = scaleArr.map { ImageUtils.readFloat($0, at: 0) } ?? 1.0
            print("[Depth] metricScale=\(metricScale)")

            // Build depth heatmap
            var depthResult: UIImage?
            var dMin: Float = 0, dMax: Float = 0
            if let depthArr {
                let shape = depthArr.shape.map { $0.intValue }
                let strides = depthArr.strides.map { $0.intValue }
                print("[Depth] depth shape=\(shape), strides=\(strides)")
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
                dMin = depthValues.filter { $0 > 0 }.min() ?? 0
                dMax = depthValues.filter { $0 > 0 }.max() ?? 0
                print("[Depth] depth range: \(dMin) – \(dMax)")
                depthResult = ImageUtils.heatmapFromDepth(depthValues, width: w, height: h)
                print("[Depth] heatmap: \(depthResult != nil)")
            }

            // Build normal map
            var normalResult: UIImage?
            if let normalArr {
                normalResult = ImageUtils.normalMapImage(normalArr)
                print("[Depth] normalMap: \(normalResult != nil)")
            }

            // Build confidence heatmap (normalized to [0, 1] across the image).
            var confResult: UIImage?
            if let confArr {
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
                confResult = ImageUtils.heatmapFromDepth(vals, width: w, height: h)
                print("[Depth] confidence heatmap: \(confResult != nil)")
            }

            await MainActor.run {
                depthImage = depthResult
                normalImage = normalResult
                confidenceImage = confResult
                depthRange = (dMin, dMax)
                processingTime = elapsed
                if viewMode == .normal && normalResult == nil { viewMode = .depth }
                if viewMode == .confidence && confResult == nil { viewMode = .depth }
                processingTime = elapsed
                isProcessing = false; status = ""
                print("[Depth] UI updated. depthImage=\(depthResult != nil), normalImage=\(normalResult != nil), confidenceImage=\(confResult != nil)")
            }
        } catch {
            print("[Depth] ERROR: \(error)")
            await MainActor.run { isProcessing = false; status = "Error: \(error.localizedDescription)" }
        }
    }
}
