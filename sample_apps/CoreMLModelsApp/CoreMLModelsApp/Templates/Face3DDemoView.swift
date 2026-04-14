import SwiftUI
import PhotosUI
import CoreML
import Vision
import SceneKit

/// 3D face reconstruction with camera + photo support.
/// Matches Face3DDemo UX: live/photo modes, 3D pose axes overlay, yaw/pitch/roll display.
struct Face3DDemoView: View {
    let model: ModelEntry

    enum Mode: String, CaseIterable { case camera = "Camera", photo = "Photo" }

    @State private var mode: Mode = .photo
    @State private var inputImage: UIImage?
    @State private var faceImage: UIImage?
    @State private var angles: (yaw: Float, pitch: Float, roll: Float)?
    @State private var isProcessing = false
    @State private var status = ""
    @State private var processingTime: Double?
    @State private var item: PhotosPickerItem?
    @State private var sceneNode: SCNNode?
    @State private var mlModel: MLModel?
    @State private var isModelLoaded = false
    @State private var liveAngles: (yaw: Float, pitch: Float, roll: Float)?

    var body: some View {
        VStack(spacing: 0) {
            Picker("Mode", selection: $mode) {
                ForEach(Mode.allCases, id: \.self) { Text($0.rawValue).tag($0) }
            }.pickerStyle(.segmented).padding(.horizontal).padding(.top, 4)

            ZStack {
                switch mode {
                case .camera:
                    CameraView(position: .front) { pb in
                        if isModelLoaded { detectFaceLive(pb) }
                    }
                case .photo:
                    photoContent
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            // Angle display
            if let a = mode == .camera ? liveAngles : angles {
                HStack(spacing: 24) {
                    angleLabel("Yaw", value: a.yaw)
                    angleLabel("Pitch", value: a.pitch)
                    angleLabel("Roll", value: a.roll)
                }.padding(.vertical, 4)
            }

            if mode == .photo {
                VStack(spacing: 8) {
                    if let t = processingTime {
                        Text(String(format: "%.2fs", t)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    }
                    if isProcessing { ProgressView(status) }
                    PhotosPicker(selection: $item, matching: .images) {
                        Label("Select Face Photo", systemImage: "person.crop.rectangle").frame(maxWidth: .infinity)
                    }.buttonStyle(.bordered).disabled(isProcessing)
                }.padding()
            } else {
                HStack {
                    Text(isModelLoaded ? "Live" : "Loading model…").font(.caption).foregroundStyle(.secondary)
                    Spacer()
                }.padding(.horizontal).padding(.bottom, 8)
            }
        }
        .task { await loadModel() }
        .onChange(of: item) { _, _ in loadAndRun() }
        .onDisappear {
            mlModel = nil
            sceneNode = nil
        }
    }

    @ViewBuilder
    private var photoContent: some View {
        HStack(spacing: 16) {
            if let img = faceImage ?? inputImage {
                Image(uiImage: img).resizable().aspectRatio(contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 8)).frame(maxWidth: 180, maxHeight: 180)
            }
            if let node = sceneNode {
                SceneView(scene: makeScene(with: node), options: [.allowsCameraControl, .autoenablesDefaultLighting])
                    .frame(maxWidth: .infinity, maxHeight: 240)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .background(Color(.systemGray6).clipShape(RoundedRectangle(cornerRadius: 8)))
            }
        }.padding(.horizontal)
    }

    @ViewBuilder
    private func angleLabel(_ name: String, value: Float) -> some View {
        VStack(spacing: 2) {
            Text(name).font(.caption2).foregroundStyle(.secondary)
            Text(String(format: "%.1f°", value)).font(.body.monospacedDigit())
        }
    }

    // MARK: - Model Loading

    private func loadModel() async {
        status = "Compiling model…"
        do {
            let loaded = try await ModelLoader.loadPrimary(for: model)
            await MainActor.run { mlModel = loaded; isModelLoaded = true; status = "" }
        } catch {
            await MainActor.run { status = "Load failed: \(error.localizedDescription)" }
        }
    }

    // MARK: - Live Camera Detection

    @State private var isDetecting = false

    private func detectFaceLive(_ pixelBuffer: CVPixelBuffer) {
        guard !isDetecting, let mlModel else { return }
        isDetecting = true

        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let ctx = CIContext()
        guard let cgImage = ctx.createCGImage(ciImage, from: ciImage.extent) else {
            isDetecting = false; return
        }

        // Detect face
        let req = VNDetectFaceRectanglesRequest { req, _ in
            defer { isDetecting = false }
            guard let face = (req.results as? [VNFaceObservation])?.first else {
                DispatchQueue.main.async { liveAngles = nil }; return
            }
            if let result = runInference(cgImage: cgImage, faceRect: face.boundingBox, mlModel: mlModel) {
                DispatchQueue.main.async { liveAngles = result }
            }
        }
        try? VNImageRequestHandler(cgImage: cgImage, orientation: .up).perform([req])
    }

    // MARK: - Photo Mode

    private func loadAndRun() {
        guard let item else { return }
        isProcessing = true; status = "Loading…"
        Task {
            guard let data = try? await item.loadTransferable(type: Data.self),
                  let img = UIImage(data: data) else {
                await MainActor.run { isProcessing = false; status = "Failed" }; return
            }
            await MainActor.run { inputImage = img; faceImage = nil; sceneNode = nil; angles = nil }
            await runOnPhoto(img)
        }
    }

    private func runOnPhoto(_ image: UIImage) async {
        guard let cgImage = ImageUtils.normalizeOrientation(image), let mlModel else {
            isProcessing = false; status = "Error"; return
        }
        status = "Detecting face…"

        let faceRect: CGRect = await withCheckedContinuation { cont in
            let req = VNDetectFaceRectanglesRequest { req, _ in
                let rect = (req.results as? [VNFaceObservation])?.first?.boundingBox
                    ?? CGRect(x: 0.15, y: 0.15, width: 0.7, height: 0.7)
                cont.resume(returning: rect)
            }
            try? VNImageRequestHandler(cgImage: cgImage, orientation: .up).perform([req])
        }

        status = "Running inference…"
        let start = CFAbsoluteTimeGetCurrent()
        if let result = runInference(cgImage: cgImage, faceRect: faceRect, mlModel: mlModel) {
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            let node = makeFaceNode(yaw: result.yaw, pitch: result.pitch, roll: result.roll)
            await MainActor.run {
                angles = result; sceneNode = node; processingTime = elapsed
                isProcessing = false; status = ""
            }
        } else {
            await MainActor.run { isProcessing = false; status = "Inference failed" }
        }
    }

    // MARK: - Shared Inference

    private func runInference(cgImage: CGImage, faceRect: CGRect, mlModel: MLModel) -> (yaw: Float, pitch: Float, roll: Float)? {
        let w = CGFloat(cgImage.width), h = CGFloat(cgImage.height)
        // 3DDFA_V2 ROI: 1.58x expansion, 0.14 upward shift
        let fx = faceRect.origin.x * w, fy = (1 - faceRect.origin.y - faceRect.height) * h
        let fw = faceRect.width * w, fh = faceRect.height * h
        let side = max(fw, fh) * 1.58
        let cx = fx + fw / 2, cy = fy + fh / 2 - fh * 0.14
        let cropRect = CGRect(x: cx - side/2, y: cy - side/2, width: side, height: side)
            .intersection(CGRect(x: 0, y: 0, width: w, height: h))

        guard let cropped = cgImage.cropping(to: cropRect) else { return nil }

        let inputSize = model.configInt("input_size") ?? 120
        guard let pb = ImageUtils.pixelBuffer(from: cropped, width: inputSize, height: inputSize) else { return nil }

        DispatchQueue.main.async { faceImage = UIImage(cgImage: cropped) }

        let inputName = mlModel.modelDescription.inputDescriptionsByName.first {
            $0.value.type == .image
        }?.key ?? "face_image"

        guard let output = try? mlModel.prediction(from:
            MLDictionaryFeatureProvider(dictionary: [inputName: pb])) else { return nil }

        guard let paramsArr = output.featureNames.compactMap({
            output.featureValue(for: $0)?.multiArrayValue
        }).first else { return nil }

        let params = ImageUtils.extractFloats(paramsArr)
        guard params.count >= 12 else { return nil }

        // Extract 3x3 rotation, normalize rows
        var R = (0..<3).map { r in (0..<3).map { c in params[r * 4 + c] } }
        for r in 0..<3 {
            let norm = sqrt(R[r].reduce(0) { $0 + $1 * $1 })
            if norm > 0 { for c in 0..<3 { R[r][c] /= norm } }
        }

        let yaw = atan2(R[0][2], R[2][2]) * 180 / .pi
        let pitch = asin(max(-1, min(1, -R[1][2]))) * 180 / .pi
        let roll = atan2(R[1][0], R[1][1]) * 180 / .pi

        return (yaw, pitch, roll)
    }

    // MARK: - SceneKit

    private func makeFaceNode(yaw: Float, pitch: Float, roll: Float) -> SCNNode {
        let head = SCNSphere(radius: 0.5)
        head.firstMaterial?.diffuse.contents = UIColor.systemBlue.withAlphaComponent(0.6)
        let node = SCNNode(geometry: head)

        let nose = SCNCone(topRadius: 0, bottomRadius: 0.08, height: 0.25)
        nose.firstMaterial?.diffuse.contents = UIColor.systemRed
        let noseNode = SCNNode(geometry: nose)
        noseNode.position = SCNVector3(0, 0, 0.5); noseNode.eulerAngles.x = -.pi / 2
        node.addChildNode(noseNode)

        for x: Float in [-0.18, 0.18] {
            let eye = SCNSphere(radius: 0.06)
            eye.firstMaterial?.diffuse.contents = UIColor.white
            let eyeNode = SCNNode(geometry: eye)
            eyeNode.position = SCNVector3(x, 0.1, 0.42)
            node.addChildNode(eyeNode)
        }

        node.eulerAngles = SCNVector3(pitch * .pi / 180, yaw * .pi / 180, roll * .pi / 180)
        return node
    }

    private func makeScene(with node: SCNNode) -> SCNScene {
        let scene = SCNScene()
        scene.rootNode.addChildNode(node)
        let cam = SCNNode(); cam.camera = SCNCamera(); cam.position = SCNVector3(0, 0, 2.5)
        scene.rootNode.addChildNode(cam)
        return scene
    }
}
