import SwiftUI
import PhotosUI
import CoreML
import Vision
import SceneKit

/// 3D face reconstruction: photo → 3DMM parameters → mesh visualization.
/// Used by: 3DDFA_V2.
///
/// Expected manifest config:
/// ```
/// { "input_size": 120, "num_params": 62 }
/// ```
struct Face3DDemoView: View {
    let model: ModelEntry

    @State private var inputImage: UIImage?
    @State private var faceImage: UIImage?
    @State private var angles: (yaw: Float, pitch: Float, roll: Float)?
    @State private var isProcessing = false
    @State private var status = ""
    @State private var processingTime: Double?
    @State private var item: PhotosPickerItem?
    @State private var sceneNode: SCNNode?

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 16) {
                if let img = faceImage ?? inputImage {
                    Image(uiImage: img).resizable().aspectRatio(contentMode: .fit)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                        .frame(maxWidth: 180, maxHeight: 180)
                }

                if let node = sceneNode {
                    SceneView(
                        scene: makeScene(with: node),
                        options: [.allowsCameraControl, .autoenablesDefaultLighting]
                    )
                    .frame(maxWidth: .infinity, maxHeight: 240)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .background(Color(.systemGray6).clipShape(RoundedRectangle(cornerRadius: 8)))
                }
            }
            .padding(.horizontal)

            if let angles {
                HStack(spacing: 24) {
                    angleLabel("Yaw", value: angles.yaw)
                    angleLabel("Pitch", value: angles.pitch)
                    angleLabel("Roll", value: angles.roll)
                }
                .padding(.top, 8)
            }

            Spacer()

            VStack(spacing: 12) {
                if let t = processingTime {
                    Text(String(format: "%.2fs", t)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                }
                if isProcessing { ProgressView(status) }
                PhotosPicker(selection: $item, matching: .images) {
                    Label("Select Face Photo", systemImage: "person.crop.rectangle").frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered).disabled(isProcessing)
            }
            .padding()
        }
        .onChange(of: item) { _, _ in loadAndRun() }
    }

    @ViewBuilder
    private func angleLabel(_ name: String, value: Float) -> some View {
        VStack(spacing: 2) {
            Text(name).font(.caption2).foregroundStyle(.secondary)
            Text(String(format: "%.1f°", value)).font(.body.monospacedDigit())
        }
    }

    private func loadAndRun() {
        guard let item else { return }
        isProcessing = true; status = "Loading…"
        Task {
            guard let data = try? await item.loadTransferable(type: Data.self),
                  let img = UIImage(data: data) else {
                await MainActor.run { isProcessing = false; status = "Failed" }; return
            }
            await MainActor.run { inputImage = img; faceImage = nil; sceneNode = nil; angles = nil }
            await runReconstruction(on: img)
        }
    }

    private func runReconstruction(on image: UIImage) async {
        await MainActor.run { status = "Detecting face…" }
        do {
            guard let cgImage = ImageUtils.normalizeOrientation(image) else {
                await MainActor.run { isProcessing = false; status = "Image error" }; return
            }

            // Detect face
            let faceRect = try await withCheckedThrowingContinuation { (cont: CheckedContinuation<CGRect, Error>) in
                let req = VNDetectFaceRectanglesRequest { req, err in
                    if let err { cont.resume(throwing: err); return }
                    if let face = (req.results as? [VNFaceObservation])?.first {
                        cont.resume(returning: face.boundingBox)
                    } else {
                        cont.resume(returning: CGRect(x: 0.15, y: 0.15, width: 0.7, height: 0.7))
                    }
                }
                try? VNImageRequestHandler(cgImage: cgImage, orientation: .up).perform([req])
            }

            // Crop with expansion (3DDFA_V2 style)
            let w = CGFloat(cgImage.width), h = CGFloat(cgImage.height)
            let fx = faceRect.origin.x * w, fy = (1 - faceRect.origin.y - faceRect.height) * h
            let fw = faceRect.width * w, fh = faceRect.height * h
            let side = max(fw, fh) * 1.58
            let cx = fx + fw / 2, cy = fy + fh / 2 - fh * 0.14
            let cropRect = CGRect(x: cx - side / 2, y: cy - side / 2, width: side, height: side)
                .intersection(CGRect(x: 0, y: 0, width: w, height: h))
            guard let cropped = cgImage.cropping(to: cropRect) else {
                await MainActor.run { isProcessing = false; status = "Face crop failed" }; return
            }

            let inputSize = model.configInt("input_size") ?? 120
            guard let pb = ImageUtils.pixelBuffer(from: cropped, width: inputSize, height: inputSize) else {
                await MainActor.run { isProcessing = false; status = "Pixel buffer failed" }; return
            }

            await MainActor.run { status = "Compiling model…"; faceImage = UIImage(cgImage: cropped) }
            let mlModel = try await ModelLoader.loadPrimary(for: model)

            await MainActor.run { status = "Running inference…" }
            let start = CFAbsoluteTimeGetCurrent()

            let inputName = mlModel.modelDescription.inputDescriptionsByName.first {
                $0.value.type == .image
            }?.key ?? "face_image"
            let input = try MLDictionaryFeatureProvider(dictionary: [inputName: pb])
            let output = try await mlModel.prediction(from: input)
            let elapsed = CFAbsoluteTimeGetCurrent() - start

            // Extract params (typically [62] float values)
            guard let paramsArr = output.featureNames.compactMap({
                output.featureValue(for: $0)?.multiArrayValue
            }).first else {
                await MainActor.run { isProcessing = false; status = "No output" }; return
            }

            let params = ImageUtils.extractFloats(paramsArr)

            // Extract rotation matrix (first 12 values = 3x4 pose matrix)
            guard params.count >= 12 else {
                await MainActor.run { isProcessing = false; status = "Unexpected output shape" }; return
            }

            // 3x3 rotation from params[0:12]
            let R = (0..<3).map { r in (0..<3).map { c in params[r * 4 + c] } }

            // Normalize rotation matrix rows
            var Rn = R
            for r in 0..<3 {
                let norm = sqrt(Rn[r].reduce(0) { $0 + $1 * $1 })
                if norm > 0 { for c in 0..<3 { Rn[r][c] /= norm } }
            }

            // Euler angles
            let yaw = atan2(Rn[0][2], Rn[2][2]) * 180 / .pi
            let pitch = asin(max(-1, min(1, -Rn[1][2]))) * 180 / .pi
            let roll = atan2(Rn[1][0], Rn[1][1]) * 180 / .pi

            // Create SceneKit visualization
            let node = makeFaceNode(yaw: yaw, pitch: pitch, roll: roll)

            await MainActor.run {
                angles = (yaw, pitch, roll)
                sceneNode = node
                processingTime = elapsed
                isProcessing = false; status = ""
            }
        } catch {
            await MainActor.run { isProcessing = false; status = "Error: \(error.localizedDescription)" }
        }
    }

    private func makeFaceNode(yaw: Float, pitch: Float, roll: Float) -> SCNNode {
        let head = SCNSphere(radius: 0.5)
        head.firstMaterial?.diffuse.contents = UIColor.systemBlue.withAlphaComponent(0.6)
        head.firstMaterial?.transparency = 0.6

        let node = SCNNode(geometry: head)

        // Nose indicator
        let nose = SCNCone(topRadius: 0, bottomRadius: 0.08, height: 0.25)
        nose.firstMaterial?.diffuse.contents = UIColor.systemRed
        let noseNode = SCNNode(geometry: nose)
        noseNode.position = SCNVector3(0, 0, 0.5)
        noseNode.eulerAngles.x = -.pi / 2
        node.addChildNode(noseNode)

        // Eye indicators
        for x: Float in [-0.18, 0.18] {
            let eye = SCNSphere(radius: 0.06)
            eye.firstMaterial?.diffuse.contents = UIColor.white
            let eyeNode = SCNNode(geometry: eye)
            eyeNode.position = SCNVector3(x, 0.1, 0.42)
            node.addChildNode(eyeNode)
        }

        // Apply rotation
        node.eulerAngles = SCNVector3(
            pitch * .pi / 180,
            yaw * .pi / 180,
            roll * .pi / 180
        )

        return node
    }

    private func makeScene(with node: SCNNode) -> SCNScene {
        let scene = SCNScene()
        scene.rootNode.addChildNode(node)

        let camera = SCNCamera()
        let cameraNode = SCNNode()
        cameraNode.camera = camera
        cameraNode.position = SCNVector3(0, 0, 2.5)
        scene.rootNode.addChildNode(cameraNode)

        return scene
    }
}
