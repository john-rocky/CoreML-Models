import SwiftUI

/// Routes from a downloaded model to its demo template view.
///
/// Templates are matched by `model.demo.template` string, declared in models.json.
/// Add a new case here when introducing a new template.
struct DemoLauncherView: View {
    let model: ModelEntry

    var body: some View {
        Group {
            switch model.demo.template {
            case "depth_visualization":
                DepthVisualizationDemoView(model: model)
            case "image_in_out":
                ImageInOutDemoView(model: model)
            case "image_detection":
                ImageDetectionDemoView(model: model)
            case "open_vocab_detection":
                OpenVocabDetectionDemoView(model: model)
            case "zero_shot_classify":
                ZeroShotClassifyDemoView(model: model)
            case "face_compare":
                FaceCompareDemoView(model: model)
            case "face_3d":
                Face3DDemoView(model: model)
            case "text_to_image":
                TextToImageDemoView(model: model)
            case "text_to_image_nitroe":
                NitroETextToImageDemoView(model: model)
            case "image_to_text":
                ImageToTextDemoView(model: model)
            case "video_matting":
                VideoMattingDemoView(model: model)
            case "audio_in_out":
                AudioInOutDemoView(model: model)
            case "text_to_audio":
                TextToAudioDemoView(model: model)
            case "audio_to_score":
                AudioToScoreDemoView(model: model)
            case "chat":
                ChatDemoView(model: model)
            case "inpainting":
                InpaintingDemoView(model: model)
            case "segment_anything":
                SegmentAnythingDemoView(model: model)
            default:
                unknownTemplate
            }
        }
        .navigationTitle(model.name)
        .navigationBarTitleDisplayMode(.inline)
    }

    @ViewBuilder
    private var unknownTemplate: some View {
        VStack(spacing: 12) {
            Image(systemName: "questionmark.square.dashed")
                .font(.system(size: 48))
                .foregroundStyle(.secondary)
            Text("Demo template not yet available")
                .font(.headline)
            Text("This model uses the '\(model.demo.template)' template, which hasn't been implemented in this version of the app.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
