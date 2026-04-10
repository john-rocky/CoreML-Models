import SwiftUI
import PhotosUI
import CoreML
import AVFoundation

/// Video matting: video clip + mask frame → alpha-composited video.
/// Used by: MatAnyone.
///
/// Expected manifest config:
/// ```
/// {
///   "encoder": "MatAnyone_Encoder.mlpackage",
///   "mask_encoder": "MatAnyone_MaskEncoder.mlpackage",
///   "read_first": "MatAnyone_ReadFirst.mlpackage",
///   "read": "MatAnyone_Read.mlpackage",
///   "decoder": "MatAnyone_Decoder.mlpackage",
///   "frame_size": 512,
///   "memory_frames": 5
/// }
/// ```
struct VideoMattingDemoView: View {
    let model: ModelEntry

    @State private var inputVideoURL: URL?
    @State private var outputVideoURL: URL?
    @State private var thumbnail: UIImage?
    @State private var maskFrame: UIImage?
    @State private var isProcessing = false
    @State private var progress: Float = 0
    @State private var status = ""
    @State private var processingTime: Double?
    @State private var showingVideoPicker = false
    @State private var showingMaskPicker = false
    @State private var maskItem: PhotosPickerItem?

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 16) {
                // Video thumbnail
                VStack(spacing: 4) {
                    if let thumb = thumbnail {
                        Image(uiImage: thumb).resizable().aspectRatio(contentMode: .fit)
                            .frame(maxWidth: 160, maxHeight: 160)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                    } else {
                        RoundedRectangle(cornerRadius: 8).fill(Color(.systemGray6))
                            .frame(width: 160, height: 120)
                            .overlay {
                                Image(systemName: "film").font(.title).foregroundStyle(.tertiary)
                            }
                    }
                    Text("Video").font(.caption2).foregroundStyle(.secondary)
                }

                // Mask
                VStack(spacing: 4) {
                    if let mask = maskFrame {
                        Image(uiImage: mask).resizable().aspectRatio(contentMode: .fit)
                            .frame(maxWidth: 160, maxHeight: 160)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                    } else {
                        RoundedRectangle(cornerRadius: 8).fill(Color(.systemGray6))
                            .frame(width: 160, height: 120)
                            .overlay {
                                Image(systemName: "person.fill.viewfinder").font(.title).foregroundStyle(.tertiary)
                            }
                    }
                    Text("Mask").font(.caption2).foregroundStyle(.secondary)
                }
            }
            .padding()

            if isProcessing {
                VStack(spacing: 8) {
                    ProgressView(value: Double(progress))
                    Text(status).font(.caption).foregroundStyle(.secondary)
                }
                .padding(.horizontal)
            }

            if let url = outputVideoURL {
                VideoPlayerView(url: url)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .padding(.horizontal)
            } else {
                VStack(spacing: 12) {
                    Image(systemName: "person.and.background.dotted")
                        .font(.system(size: 60)).foregroundStyle(.secondary)
                    Text("Select a video and a mask image for the foreground subject")
                        .multilineTextAlignment(.center).foregroundStyle(.secondary)
                        .padding(.horizontal, 24)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }

            VStack(spacing: 12) {
                if let t = processingTime {
                    Text(String(format: "%.1fs", t)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                }

                HStack(spacing: 12) {
                    Button {
                        showingVideoPicker = true
                    } label: {
                        Label("Video", systemImage: "film")
                    }.buttonStyle(.bordered)

                    PhotosPicker(selection: $maskItem, matching: .images) {
                        Label("Mask", systemImage: "person.crop.rectangle")
                    }.buttonStyle(.bordered)

                    Button {
                        Task { await runMatting() }
                    } label: {
                        Label("Process", systemImage: "wand.and.rays").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isProcessing || inputVideoURL == nil || maskFrame == nil)
                }
            }
            .padding()
        }
        .sheet(isPresented: $showingVideoPicker) {
            VideoPickerView { url in
                inputVideoURL = url
                extractThumbnail(from: url)
            }
        }
        .onChange(of: maskItem) { _, _ in loadMask() }
    }

    private func loadMask() {
        guard let maskItem else { return }
        Task {
            if let data = try? await maskItem.loadTransferable(type: Data.self),
               let img = UIImage(data: data) {
                await MainActor.run { maskFrame = img }
            }
        }
    }

    private func extractThumbnail(from url: URL) {
        let asset = AVURLAsset(url: url)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 320, height: 320)
        Task {
            if let cgImage = try? generator.copyCGImage(at: .zero, actualTime: nil) {
                await MainActor.run { thumbnail = UIImage(cgImage: cgImage) }
            }
        }
    }

    private func runMatting() async {
        guard let videoURL = inputVideoURL, let mask = maskFrame else { return }
        isProcessing = true; progress = 0; outputVideoURL = nil

        do {
            status = "Loading models…"

            // Load encoder (at minimum — full pipeline needs all 5 models)
            let encoderFile = model.configString("encoder")
                ?? model.files.first { $0.name.lowercased().contains("encoder") && !$0.name.lowercased().contains("mask") }?.name
            guard let encFile = encoderFile else {
                status = "Encoder model not found"; isProcessing = false; return
            }

            let encoder = try await ModelLoader.load(for: model, named: encFile)
            let frameSize = model.configInt("frame_size") ?? 512

            // Extract video frames
            status = "Extracting frames…"
            let asset = AVURLAsset(url: videoURL)
            let reader = try AVAssetReader(asset: asset)
            guard let videoTrack = try await asset.loadTracks(withMediaType: .video).first else {
                status = "No video track"; isProcessing = false; return
            }

            let outputSettings: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ]
            let output = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: outputSettings)
            reader.add(output)
            reader.startReading()

            // Process frames
            var processedFrames: [(CGImage, Double)] = []
            var frameCount = 0
            let totalFrames = max(1, Int(try await asset.load(.duration).seconds * 30))

            // Prepare mask pixel buffer
            guard let maskPB = ImageUtils.pixelBuffer(from: mask, width: frameSize, height: frameSize) else {
                status = "Mask prep failed"; isProcessing = false; return
            }

            let start = CFAbsoluteTimeGetCurrent()

            while let sampleBuffer = output.copyNextSampleBuffer() {
                guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { continue }

                // Resize frame
                let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
                let ctx = CIContext()
                guard let frameCG = ctx.createCGImage(ciImage, from: ciImage.extent),
                      let framePB = ImageUtils.pixelBuffer(from: frameCG, width: frameSize, height: frameSize) else {
                    continue
                }

                // Run encoder on frame + mask
                let encNames = encoder.modelDescription.inputDescriptionsByName
                var encDict: [String: Any] = [:]
                for (key, desc) in encNames {
                    if desc.type == .image {
                        if key.lowercased().contains("mask") { encDict[key] = maskPB }
                        else { encDict[key] = framePB }
                    }
                }
                if encDict.isEmpty { encDict = ["image": framePB] }

                let encOutput = try await encoder.prediction(from: MLDictionaryFeatureProvider(dictionary: encDict))

                // Extract output image or alpha
                for name in encOutput.featureNames {
                    if let pb = encOutput.featureValue(for: name)?.imageBufferValue {
                        let ci = CIImage(cvPixelBuffer: pb)
                        if let cg = ctx.createCGImage(ci, from: ci.extent) {
                            let time = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
                            processedFrames.append((cg, time))
                        }
                        break
                    }
                    if let arr = encOutput.featureValue(for: name)?.multiArrayValue,
                       let img = ImageUtils.imageFromMultiArray(arr)?.cgImage {
                        let time = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
                        processedFrames.append((img, time))
                        break
                    }
                }

                frameCount += 1
                await MainActor.run { progress = Float(frameCount) / Float(totalFrames) }
            }

            // Write output video
            status = "Writing video…"
            let outputURL = FileManager.default.temporaryDirectory.appendingPathComponent("matted_\(UUID().uuidString).mp4")

            if !processedFrames.isEmpty {
                try await writeFramesToVideo(processedFrames, to: outputURL, size: CGSize(width: frameSize, height: frameSize))
            }

            let elapsed = CFAbsoluteTimeGetCurrent() - start

            await MainActor.run {
                outputVideoURL = outputURL
                processingTime = elapsed
                isProcessing = false; status = ""
            }
        } catch {
            await MainActor.run { isProcessing = false; status = "Error: \(error.localizedDescription)" }
        }
    }

    private func writeFramesToVideo(_ frames: [(CGImage, Double)], to url: URL, size: CGSize) async throws {
        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: Int(size.width),
            AVVideoHeightKey: Int(size.height)
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: nil)
        writer.add(input)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)

        for (i, (frame, time)) in frames.enumerated() {
            while !input.isReadyForMoreMediaData { try await Task.sleep(nanoseconds: 10_000_000) }
            if let pb = ImageUtils.pixelBuffer(from: frame, width: Int(size.width), height: Int(size.height)) {
                adaptor.append(pb, withPresentationTime: CMTime(seconds: time, preferredTimescale: 600))
            }
        }

        input.markAsFinished()
        await writer.finishWriting()
    }
}

// MARK: - Video Picker (UIDocumentPicker wrapper)

struct VideoPickerView: UIViewControllerRepresentable {
    let onPick: (URL) -> Void

    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        let picker = UIDocumentPickerViewController(forOpeningContentTypes: [.movie, .video])
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ vc: UIDocumentPickerViewController, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(onPick: onPick) }

    class Coordinator: NSObject, UIDocumentPickerDelegate {
        let onPick: (URL) -> Void
        init(onPick: @escaping (URL) -> Void) { self.onPick = onPick }
        func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
            if let url = urls.first { onPick(url) }
        }
    }
}

// MARK: - Simple Video Player

struct VideoPlayerView: UIViewControllerRepresentable {
    let url: URL

    func makeUIViewController(context: Context) -> AVPlayerViewController {
        let vc = AVPlayerViewController()
        vc.player = AVPlayer(url: url)
        return vc
    }

    func updateUIViewController(_ vc: AVPlayerViewController, context: Context) {}
}

import AVKit
