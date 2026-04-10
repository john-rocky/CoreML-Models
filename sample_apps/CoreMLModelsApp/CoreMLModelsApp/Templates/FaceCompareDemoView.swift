import SwiftUI
import PhotosUI
import CoreML
import Vision

/// Face comparison: two photos → embedding cosine similarity.
/// Used by: AdaFace.
///
/// Expected manifest config:
/// ```
/// { "input_size": 112, "embedding_dim": 512, "match_threshold": 0.6 }
/// ```
struct FaceCompareDemoView: View {
    let model: ModelEntry

    @State private var imageA: UIImage?
    @State private var imageB: UIImage?
    @State private var faceA: UIImage?
    @State private var faceB: UIImage?
    @State private var similarity: Float?
    @State private var isProcessing = false
    @State private var status = ""
    @State private var processingTime: Double?
    @State private var itemA: PhotosPickerItem?
    @State private var itemB: PhotosPickerItem?

    private var threshold: Float { Float(model.configDouble("match_threshold") ?? 0.6) }

    var body: some View {
        VStack(spacing: 16) {
            HStack(spacing: 16) {
                faceSlot(image: faceA ?? imageA, label: "Face A", item: $itemA)
                faceSlot(image: faceB ?? imageB, label: "Face B", item: $itemB)
            }
            .padding(.horizontal)

            if let sim = similarity {
                VStack(spacing: 8) {
                    let isMatch = sim >= threshold
                    HStack {
                        Image(systemName: isMatch ? "checkmark.circle.fill" : "xmark.circle.fill")
                            .foregroundStyle(isMatch ? .green : .red)
                            .font(.title)
                        VStack(alignment: .leading) {
                            Text(isMatch ? "Same Person" : "Different People")
                                .font(.headline)
                            Text(String(format: "Similarity: %.3f (threshold: %.2f)", sim, threshold))
                                .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                        }
                    }
                    ProgressView(value: Double(max(0, min(1, sim))))
                        .tint(isMatch ? .green : .red)
                        .padding(.horizontal)
                }
            }

            Spacer()

            VStack(spacing: 12) {
                if let t = processingTime {
                    Text(String(format: "%.2fs", t)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                }
                if isProcessing { ProgressView(status) }
                Button {
                    Task { await compare() }
                } label: {
                    Label("Compare Faces", systemImage: "person.2.circle").frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(isProcessing || imageA == nil || imageB == nil)
            }
            .padding()
        }
        .onChange(of: itemA) { _, _ in loadImage(from: itemA, into: { imageA = $0; faceA = nil; similarity = nil }) }
        .onChange(of: itemB) { _, _ in loadImage(from: itemB, into: { imageB = $0; faceB = nil; similarity = nil }) }
    }

    @ViewBuilder
    private func faceSlot(image: UIImage?, label: String, item: Binding<PhotosPickerItem?>) -> some View {
        PhotosPicker(selection: item, matching: .images) {
            ZStack {
                if let img = image {
                    Image(uiImage: img).resizable().aspectRatio(contentMode: .fill)
                        .frame(width: 150, height: 150)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                } else {
                    RoundedRectangle(cornerRadius: 12).fill(Color(.systemGray6))
                        .frame(width: 150, height: 150)
                        .overlay {
                            VStack(spacing: 4) {
                                Image(systemName: "person.crop.rectangle").font(.title2).foregroundStyle(.tertiary)
                                Text(label).font(.caption).foregroundStyle(.tertiary)
                            }
                        }
                }
            }
        }
    }

    private func loadImage(from item: PhotosPickerItem?, into setter: @escaping (UIImage?) -> Void) {
        guard let item else { return }
        Task {
            if let data = try? await item.loadTransferable(type: Data.self),
               let img = UIImage(data: data) {
                await MainActor.run { setter(img) }
            }
        }
    }

    private func compare() async {
        guard let imgA = imageA, let imgB = imageB else { return }
        isProcessing = true; status = "Loading model…"

        do {
            let mlModel = try await ModelLoader.loadPrimary(for: model)
            let inputSize = model.configInt("input_size") ?? 112

            status = "Detecting faces…"
            let alignedA = try await detectAndAlign(imgA, size: inputSize)
            let alignedB = try await detectAndAlign(imgB, size: inputSize)

            await MainActor.run {
                faceA = alignedA.thumbnail; faceB = alignedB.thumbnail
            }

            status = "Computing embeddings…"
            let start = CFAbsoluteTimeGetCurrent()

            let inputName = mlModel.modelDescription.inputDescriptionsByName.first {
                $0.value.type == .image
            }?.key ?? "face_image"

            let outA = try await mlModel.prediction(from:
                MLDictionaryFeatureProvider(dictionary: [inputName: alignedA.buffer]))
            let outB = try await mlModel.prediction(from:
                MLDictionaryFeatureProvider(dictionary: [inputName: alignedB.buffer]))

            let embName = outA.featureNames.first(where: { $0.contains("embed") }) ?? outA.featureNames.first ?? ""
            guard let embA = outA.featureValue(for: embName)?.multiArrayValue,
                  let embB = outB.featureValue(for: embName)?.multiArrayValue else {
                isProcessing = false; status = "No embedding output"; return
            }

            let vecA = ImageUtils.extractFloats(embA)
            let vecB = ImageUtils.extractFloats(embB)
            let sim = ImageUtils.cosineSimilarity(vecA, vecB)
            let elapsed = CFAbsoluteTimeGetCurrent() - start

            await MainActor.run {
                similarity = sim; processingTime = elapsed
                isProcessing = false; status = ""
            }
        } catch {
            await MainActor.run { isProcessing = false; status = "Error: \(error.localizedDescription)" }
        }
    }

    // MARK: - Face detection & alignment

    struct AlignedFace {
        let buffer: CVPixelBuffer
        let thumbnail: UIImage
    }

    private func detectAndAlign(_ image: UIImage, size: Int) async throws -> AlignedFace {
        guard let cgImage = ImageUtils.normalizeOrientation(image) else {
            throw NSError(domain: "FaceCompare", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid image"])
        }

        let faceRect = try await withCheckedThrowingContinuation { (cont: CheckedContinuation<CGRect, Error>) in
            let req = VNDetectFaceRectanglesRequest { req, err in
                if let err { cont.resume(throwing: err); return }
                if let face = (req.results as? [VNFaceObservation])?.first {
                    cont.resume(returning: face.boundingBox)
                } else {
                    // No face found — use center crop
                    cont.resume(returning: CGRect(x: 0.1, y: 0.1, width: 0.8, height: 0.8))
                }
            }
            try? VNImageRequestHandler(cgImage: cgImage, orientation: .up).perform([req])
        }

        // Expand face rect for alignment margin
        let w = CGFloat(cgImage.width), h = CGFloat(cgImage.height)
        let fx = faceRect.origin.x * w, fy = (1 - faceRect.origin.y - faceRect.height) * h
        let fw = faceRect.width * w, fh = faceRect.height * h
        let expand: CGFloat = 0.3
        let cropRect = CGRect(
            x: max(0, fx - fw * expand), y: max(0, fy - fh * expand),
            width: min(w, fw * (1 + 2 * expand)), height: min(h, fh * (1 + 2 * expand))
        ).intersection(CGRect(x: 0, y: 0, width: w, height: h))

        guard let cropped = cgImage.cropping(to: cropRect),
              let pb = ImageUtils.pixelBuffer(from: cropped, width: size, height: size) else {
            throw NSError(domain: "FaceCompare", code: 2, userInfo: [NSLocalizedDescriptionKey: "Face crop failed"])
        }

        let thumb = UIImage(cgImage: cropped)
        return AlignedFace(buffer: pb, thumbnail: thumb)
    }
}
