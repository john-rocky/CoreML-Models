import SwiftUI
import PhotosUI
import CoreML
import Accelerate

/// Open-vocabulary detection: photo + text query → boxes + labels.
/// Used by: YOLO-World (detector + CLIP text encoder).
///
/// Detector input: "image" [1,3,640,640] float32 + "txt_feats" [1,80,512] float32
/// Detector output: "boxes" [4,8400] + "scores" [numClasses,8400]
struct OpenVocabDetectionDemoView: View {
    let model: ModelEntry

    @State private var inputImage: UIImage?
    @State private var annotatedImage: UIImage?
    @State private var queryText = "person, car, dog"
    @State private var detections: [Detection] = []
    @State private var isProcessing = false
    @State private var status = ""
    @State private var processingTime: Double?
    @State private var item: PhotosPickerItem?
    @State private var confidenceThreshold: Float = 0.15
    @StateObject private var session = ModelSession<(detector: MLModel, textEncoder: MLModel?)>()

    struct Detection: Identifiable {
        let id = UUID()
        let label: String
        let confidence: Float
        let box: CGRect  // normalized 0..1
    }

    private let inputSize = 640
    private let maxClasses = 80
    private let embedDim = 512

    private static let tokenizer: CLIPTokenizer? = {
        guard let url = Bundle.main.url(forResource: "clip_vocab", withExtension: "json") else { return nil }
        return try? CLIPTokenizer(vocabularyURL: url)
    }()

    var body: some View {
        VStack(spacing: 0) {
            ZStack {
                if let img = annotatedImage ?? inputImage {
                    Image(uiImage: img).resizable().aspectRatio(contentMode: .fit)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                } else {
                    VStack(spacing: 12) {
                        Image(systemName: "text.viewfinder").font(.system(size: 60)).foregroundStyle(.secondary)
                        Text("Select a photo and enter class names").foregroundStyle(.secondary)
                    }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .padding(.horizontal)

            if !detections.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(detections) { d in
                            Text("\(d.label) \(Int(d.confidence * 100))%")
                                .font(.caption2).padding(.horizontal, 8).padding(.vertical, 4)
                                .background(.ultraThinMaterial).clipShape(Capsule())
                        }
                    }.padding(.horizontal)
                }.frame(height: 40)
            }

            VStack(spacing: 12) {
                TextField("Classes (comma-separated)", text: $queryText)
                    .textFieldStyle(.roundedBorder).font(.callout)

                HStack {
                    Text("Confidence").font(.caption2).foregroundStyle(.secondary)
                    Slider(value: $confidenceThreshold, in: 0.05...0.9)
                    Text(String(format: "%.0f%%", confidenceThreshold * 100))
                        .font(.caption2.monospacedDigit()).foregroundStyle(.secondary).frame(width: 36)
                }

                HStack {
                    TimingsLabel(loadSec: session.loadTimeSec, inferSec: processingTime)
                    if !detections.isEmpty {
                        Text("· \(detections.count)").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    }
                    Spacer()
                    if isProcessing { ProgressView().controlSize(.small) }
                }

                HStack(spacing: 12) {
                    PhotosPicker(selection: $item, matching: .images) {
                        Label("Photo", systemImage: "photo.badge.plus")
                    }.buttonStyle(.bordered)

                    Button {
                        if let img = inputImage { Task { await runDetection(on: img) } }
                    } label: {
                        Label("Detect", systemImage: "sparkle.magnifyingglass").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isProcessing || inputImage == nil || queryText.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            }
            .padding()
        }
        .task {
            session.ensure {
                let detectorFile = model.files.first {
                    $0.name.lowercased().contains("detector") || $0.name.lowercased().contains("yoloworld")
                }?.name ?? model.files[0].name
                let textEncFile = model.files.first {
                    $0.name.lowercased().contains("clip") || $0.name.lowercased().contains("text")
                }?.name
                let det = try await ModelLoader.load(for: model, named: detectorFile)
                let txt = try await { () async throws -> MLModel? in
                    guard let n = textEncFile else { return nil }
                    return try await ModelLoader.load(for: model, named: n)
                }()
                return (detector: det, textEncoder: txt)
            }
        }
        .onChange(of: item) { _, _ in loadPhoto() }
    }

    private func loadPhoto() {
        guard let item else { return }
        Task {
            if let data = try? await item.loadTransferable(type: Data.self),
               let img = UIImage(data: data) {
                await MainActor.run { inputImage = img; annotatedImage = nil; detections = [] }
            }
        }
    }

    private func runDetection(on image: UIImage) async {
        let classes = queryText.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        guard !classes.isEmpty else { return }

        isProcessing = true
        do {
            status = session.loadTimeSec == nil ? "Loading models…" : "Preparing…"
            let (detector, textEncoder) = try await session.get()

            guard let cgImage = ImageUtils.normalizeOrientation(image) else {
                isProcessing = false; status = "Image error"; return
            }

            let start = CFAbsoluteTimeGetCurrent()

            // 1. Encode text queries → txt_feats [1,80,512]
            status = "Encoding text…"
            let txtFeats = try encodeTextQueries(classes, textEncoder: textEncoder)

            // 2. Preprocess image → MLMultiArray [1,3,640,640] with letterbox
            status = "Detecting…"
            let imgW = cgImage.width, imgH = cgImage.height
            let scale = Float(inputSize) / Float(max(imgW, imgH))
            let scaledW = Int(Float(imgW) * scale)
            let scaledH = Int(Float(imgH) * scale)
            let padX = (inputSize - scaledW) / 2
            let padY = (inputSize - scaledH) / 2

            let imageTensor = try preprocessImage(cgImage, scaledW: scaledW, scaledH: scaledH, padX: padX, padY: padY)

            // 3. Run detector
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "image": imageTensor, "txt_feats": txtFeats
            ])
            let output = try await detector.prediction(from: input)
            let elapsed = CFAbsoluteTimeGetCurrent() - start

            // 4. Parse output: boxes [4,numAnchors], scores [numClasses,numAnchors]
            guard let boxesMA = output.featureValue(for: "boxes")?.multiArrayValue,
                  let scoresMA = output.featureValue(for: "scores")?.multiArrayValue else {
                isProcessing = false; status = "No detection output"; return
            }

            let boxes = ImageUtils.extractFloats(boxesMA)
            let scores = ImageUtils.extractFloats(scoresMA)
            let scShape = scoresMA.shape.map { $0.intValue }
            let numClasses = scShape.count >= 2 ? scShape[1] : maxClasses
            let numAnchors = scShape.count >= 3 ? scShape[2] : (scores.count / max(1, numClasses))

            var allDets: [(CGRect, Float, Int)] = []
            let threshold = confidenceThreshold

            for qi in 0..<min(classes.count, numClasses) {
                let off = qi * numAnchors
                for a in 0..<numAnchors {
                    let score = scores[off + a]
                    guard score >= threshold else { continue }

                    let cx = boxes[0 * numAnchors + a]
                    let cy = boxes[1 * numAnchors + a]
                    let bw = boxes[2 * numAnchors + a]
                    let bh = boxes[3 * numAnchors + a]

                    let nx = (cx - bw/2 - Float(padX)) / (Float(imgW) * scale)
                    let ny = (cy - bh/2 - Float(padY)) / (Float(imgH) * scale)
                    let nw = bw / (Float(imgW) * scale)
                    let nh = bh / (Float(imgH) * scale)

                    let rect = CGRect(
                        x: CGFloat(max(0, min(1, nx))), y: CGFloat(max(0, min(1, ny))),
                        width: CGFloat(max(0, min(1, nw))), height: CGFloat(max(0, min(1, nh)))
                    )
                    allDets.append((rect, score, qi))
                }
            }

            // NMS per class
            allDets.sort { $0.1 > $1.1 }
            var kept: [Int] = []
            for i in allDets.indices {
                var suppress = false
                for ki in kept {
                    if allDets[i].2 == allDets[ki].2 && iou(allDets[i].0, allDets[ki].0) > 0.5 {
                        suppress = true; break
                    }
                }
                if !suppress { kept.append(i) }
            }

            let dets = kept.prefix(30).map { i in
                Detection(label: classes[allDets[i].2], confidence: allDets[i].1, box: allDets[i].0)
            }

            let annotated = drawDetections(dets, on: image)
            await MainActor.run {
                detections = dets
                annotatedImage = annotated
                processingTime = elapsed
                isProcessing = false; status = ""
            }
        } catch {
            await MainActor.run { isProcessing = false; status = "Error: \(error.localizedDescription)" }
        }
    }

    // MARK: - CLIP Text Encoding

    private func encodeTextQueries(_ queries: [String], textEncoder: MLModel?) throws -> MLMultiArray {
        let txtFeats = try MLMultiArray(shape: [1, maxClasses as NSNumber, embedDim as NSNumber], dataType: .float32)
        let ptr = txtFeats.dataPointer.assumingMemoryBound(to: Float.self)
        memset(ptr, 0, maxClasses * embedDim * MemoryLayout<Float>.size)

        guard let textEncoder, let tokenizer = Self.tokenizer else { return txtFeats }
        let contextLength = tokenizer.contextLength

        for (i, query) in queries.prefix(maxClasses).enumerated() {
            let tokenArr = try MLMultiArray(shape: [maxClasses as NSNumber, contextLength as NSNumber], dataType: .int32)
            let tPtr = tokenArr.dataPointer.assumingMemoryBound(to: Int32.self)
            memset(tPtr, 0, maxClasses * contextLength * MemoryLayout<Int32>.size)
            let tokens = tokenizer.tokenize(query)
            for j in 0..<contextLength { tPtr[j] = Int32(tokens[j]) }

            let input = try MLDictionaryFeatureProvider(dictionary: ["text_tokens": tokenArr])
            let output = try textEncoder.prediction(from: input)

            guard let emb = output.featureValue(for: "text_embeddings")?.multiArrayValue else { continue }
            let embFloats = ImageUtils.extractFloats(emb)

            // L2 normalize
            var embedding = Array(embFloats.prefix(embedDim))
            if embedding.count < embedDim { embedding.append(contentsOf: [Float](repeating: 0, count: embedDim - embedding.count)) }
            var norm: Float = 0
            vDSP_svesq(embedding, 1, &norm, vDSP_Length(embedDim))
            norm = sqrt(norm)
            if norm > 1e-8 {
                var s = 1.0 / norm
                vDSP_vsmul(embedding, 1, &s, &embedding, 1, vDSP_Length(embedDim))
            }
            for j in 0..<embedDim { ptr[i * embedDim + j] = embedding[j] }
        }
        return txtFeats
    }

    // MARK: - Image Preprocessing (letterbox 0.5 gray)

    private func preprocessImage(_ cgImage: CGImage, scaledW: Int, scaledH: Int, padX: Int, padY: Int) throws -> MLMultiArray {
        guard let ctx = CGContext(
            data: nil, width: inputSize, height: inputSize,
            bitsPerComponent: 8, bytesPerRow: inputSize * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else { throw NSError(domain: "Preprocess", code: 1) }

        ctx.setFillColor(gray: 0.5, alpha: 1.0)
        ctx.fill(CGRect(x: 0, y: 0, width: inputSize, height: inputSize))
        ctx.draw(cgImage, in: CGRect(x: padX, y: padY, width: scaledW, height: scaledH))

        guard let pixels = ctx.data else { throw NSError(domain: "Preprocess", code: 2) }

        let arr = try MLMultiArray(shape: [1, 3, inputSize as NSNumber, inputSize as NSNumber], dataType: .float32)
        let dst = arr.dataPointer.assumingMemoryBound(to: Float.self)
        let src = pixels.assumingMemoryBound(to: UInt8.self)
        let hw = inputSize * inputSize
        let inv: Float = 1.0 / 255.0
        for i in 0..<hw {
            dst[0 * hw + i] = Float(src[i * 4 + 0]) * inv  // R
            dst[1 * hw + i] = Float(src[i * 4 + 1]) * inv  // G
            dst[2 * hw + i] = Float(src[i * 4 + 2]) * inv  // B
        }
        return arr
    }

    // MARK: - Helpers

    private func iou(_ a: CGRect, _ b: CGRect) -> Float {
        let interX = max(0, min(a.maxX, b.maxX) - max(a.minX, b.minX))
        let interY = max(0, min(a.maxY, b.maxY) - max(a.minY, b.minY))
        let inter = Float(interX * interY)
        let union = Float(a.width * a.height) + Float(b.width * b.height) - inter
        return union > 0 ? inter / union : 0
    }

    private func drawDetections(_ dets: [Detection], on image: UIImage) -> UIImage? {
        guard let cgImage = ImageUtils.normalizeOrientation(image) else { return image }
        let w = CGFloat(cgImage.width), h = CGFloat(cgImage.height)
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: w, height: h))
        return renderer.image { ctx in
            // UIGraphicsImageRenderer uses UIKit coordinates (origin top-left) —
            // cgContext.draw(_:in:) would flip the image, so draw via UIImage.
            UIImage(cgImage: cgImage).draw(in: CGRect(x: 0, y: 0, width: w, height: h))
            let colors: [UIColor] = [.systemRed, .systemBlue, .systemGreen, .systemOrange, .systemPurple]
            for (i, det) in dets.enumerated() {
                let color = colors[i % colors.count]
                let rect = CGRect(x: det.box.origin.x * w, y: det.box.origin.y * h,
                                  width: det.box.width * w, height: det.box.height * h)
                ctx.cgContext.setStrokeColor(color.cgColor)
                ctx.cgContext.setLineWidth(max(2, w / 300))
                ctx.cgContext.stroke(rect)
                let attrs: [NSAttributedString.Key: Any] = [
                    .font: UIFont.boldSystemFont(ofSize: max(12, w / 50)),
                    .foregroundColor: UIColor.white,
                    .backgroundColor: color.withAlphaComponent(0.7)
                ]
                ("\(det.label) \(Int(det.confidence * 100))%" as NSString)
                    .draw(at: CGPoint(x: rect.minX + 2, y: rect.minY + 2), withAttributes: attrs)
            }
        }
    }
}
