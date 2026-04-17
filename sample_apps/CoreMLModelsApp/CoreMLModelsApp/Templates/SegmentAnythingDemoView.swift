import SwiftUI
import PhotosUI
import CoreML

/// Thin wrapper around the vendored `SamView` (SAMKit/SamView.swift).
///
/// Responsibilities: pick a photo, resolve the model URLs from the hub's
/// download cache, and hand everything to `SamView` — which owns the full
/// tap-to-segment / subject-lift UX. We intentionally do *not* re-implement
/// the segmentation view; earlier attempts at a port introduced coordinate
/// bugs and slow inference. SAMKit's view is the canonical one.
struct SegmentAnythingDemoView: View {
    let model: ModelEntry

    @State private var item: PhotosPickerItem?
    @State private var inputImage: UIImage?
    @State private var samModel: SamModelRef?
    @State private var status: String = "Pick a photo to start"
    @State private var errorMessage: String?
    @State private var photoSession: Int = 0  // bumps on each new photo to rebuild SamView
    @StateObject private var session = ModelSession<SamModelRef>()

    var body: some View {
        VStack(spacing: 16) {
            if let inputImage, let samModel {
                SamView(
                    image: inputImage,
                    model: samModel,
                    config: RuntimeConfig(computeUnits: .bestAvailable, enableFP16: true)
                )
                .id(photoSession)  // force SamView to rebuild on each new photo
            } else if let inputImage {
                // Image picked, but model session still loading. Show the image
                // immediately with a progress overlay so the user isn't staring
                // at a blank placeholder.
                ZStack {
                    Image(uiImage: inputImage)
                        .resizable()
                        .scaledToFit()
                    Color.black.opacity(0.3)
                    VStack(spacing: 12) {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                            .scaleEffect(1.3)
                        Text("Preparing models…")
                            .font(.subheadline.weight(.medium))
                            .foregroundColor(.white)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 8)
                            .background(
                                RoundedRectangle(cornerRadius: 10)
                                    .fill(Color.black.opacity(0.55))
                            )
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                VStack(spacing: 12) {
                    Image(systemName: "hand.tap")
                        .font(.system(size: 60))
                        .foregroundStyle(.secondary)
                    if let errorMessage {
                        Text(errorMessage)
                            .font(.caption)
                            .foregroundColor(.red)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)
                    } else {
                        Text(status)
                            .foregroundStyle(.secondary)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }

            TimingsLabel(loadSec: session.loadTimeSec, inferSec: nil)

            PhotosPicker(selection: $item, matching: .images) {
                Label(inputImage == nil ? "Pick Photo" : "Change Photo",
                      systemImage: "photo.badge.plus")
            }
            .buttonStyle(.bordered)
            .padding(.bottom, 12)
        }
        .task {
            session.ensure { try await resolveSamModelRef() }
        }
        .onChange(of: item) { _, _ in loadPhoto() }
    }

    private func loadPhoto() {
        guard let item else { return }
        errorMessage = nil
        status = "Loading photo…"

        Task {
            guard let data = try? await item.loadTransferable(type: Data.self),
                  let img = UIImage(data: data) else {
                await MainActor.run {
                    errorMessage = "Failed to load selected photo"
                }
                return
            }

            // Show the picked image right away. SamView won't appear until
            // the model session is ready, so we render the image with a
            // "Preparing models…" overlay in the meantime.
            await MainActor.run {
                self.inputImage = img
                self.status = session.loadTimeSec == nil ? "Preparing models…" : ""
            }

            do {
                let ref = try await session.get()
                await MainActor.run {
                    self.samModel = ref
                    self.status = ""
                    self.photoSession &+= 1
                }
            } catch {
                await MainActor.run {
                    errorMessage = "Model load failed: \(error.localizedDescription)"
                }
            }
        }
    }

    private func resolveSamModelRef() async throws -> SamModelRef {
        let rawEnc = model.configString("encoder") ?? ""
        let rawDec = model.configString("decoder") ?? ""

        let sameArchive = rawEnc == rawDec || rawEnc.hasSuffix(".zip")

        let encoderURL: URL
        let decoderURL: URL
        if sameArchive {
            encoderURL = try await ModelLoader.compiledURL(
                modelId: model.id, substring: "encoder")
            decoderURL = try await ModelLoader.compiledURL(
                modelId: model.id, substring: "decoder")
        } else {
            // Rare path: encoder/decoder downloaded separately by exact name.
            _ = try await ModelLoader.load(modelId: model.id, fileName: rawEnc)
            _ = try await ModelLoader.load(modelId: model.id, fileName: rawDec)
            encoderURL = try await ModelLoader.compiledURL(
                modelId: model.id, substring: "encoder")
            decoderURL = try await ModelLoader.compiledURL(
                modelId: model.id, substring: "decoder")
        }

        let weightsName = model.configString("prompt_weights")
            ?? "mobile_sam_prompt_encoder_weights.json"
        let weightsURL = ModelLoader.auxFileURL(
            modelId: model.id, fileName: weightsName)
        let weightsOrNil = FileManager.default.fileExists(atPath: weightsURL.path)
            ? weightsURL
            : nil

        return SamModelRef(
            encoderURL: encoderURL,
            decoderURL: decoderURL,
            inputSize: model.configInt("input_size") ?? 1024,
            modelType: .mobileSam,
            promptEncoderWeightsURL: weightsOrNil
        )
    }
}
