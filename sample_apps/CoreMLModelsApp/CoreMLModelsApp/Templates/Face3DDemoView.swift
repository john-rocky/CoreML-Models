import SwiftUI
import PhotosUI
import CoreML
import Vision
import AVFoundation

/// 3D face pose estimation view. Ports the Face3DDemo sample app's 2D pose axes
/// overlay (X=red / Y=green / Z=blue arrows + yellow face rect) plus the
/// yaw/pitch/roll info bar. No SceneKit mesh — 3DDFA_V2's 62-param output only
/// carries pose + shape + expression coefficients, not vertex positions, so the
/// sample app visualizes orientation via axis arrows drawn on the image.
struct Face3DDemoView: View {
    let model: ModelEntry

    enum Mode: String, CaseIterable { case camera = "Camera", photo = "Photo" }

    // Photo mode state
    @State private var mode: Mode = .photo
    @State private var inputImage: UIImage?
    @State private var photoResults: [FacePoseResult] = []
    @State private var isProcessing = false
    @State private var status = ""
    @State private var processingTime: Double?
    @State private var item: PhotosPickerItem?
    @State private var mlModel: MLModel?
    @State private var isModelLoaded = false
    @StateObject private var session = ModelSession<MLModel>()

    // Camera mode state
    @State private var liveResults: [FacePoseResult] = []
    @State private var liveBufferSize: CGSize = .zero
    @State private var isDetecting = false

    var body: some View {
        VStack(spacing: 0) {
            Picker("Mode", selection: $mode) {
                ForEach(Mode.allCases, id: \.self) { Text($0.rawValue).tag($0) }
            }.pickerStyle(.segmented).padding(.horizontal).padding(.top, 4)

            ZStack {
                switch mode {
                case .camera:
                    cameraContent
                case .photo:
                    photoContent
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            // Angle info bar (shared UX between modes)
            let shown = mode == .camera ? liveResults : photoResults
            if !shown.isEmpty {
                AngleInfoBar(results: shown).padding(.vertical, 4)
            }

            if mode == .photo {
                VStack(spacing: 8) {
                    TimingsLabel(loadSec: session.loadTimeSec, inferSec: processingTime)
                    if isProcessing { ProgressView(status) }
                    PhotosPicker(selection: $item, matching: .images) {
                        Label("Select Face Photo", systemImage: "person.crop.rectangle")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .disabled(isProcessing)
                }
                .padding()
            } else {
                HStack {
                    Text(isModelLoaded ? "Live" : "Loading model…")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                }
                .padding(.horizontal)
                .padding(.bottom, 8)
            }
        }
        .task { await loadModel() }
        .onChange(of: item) { _, _ in loadAndRun() }
        .onDisappear {
            mlModel = nil
        }
    }

    // MARK: - Photo content

    @ViewBuilder
    private var photoContent: some View {
        GeometryReader { geo in
            if let img = inputImage {
                let fitSize = aspectFit(imageSize: img.size, in: geo.size)
                let offsetX = (geo.size.width - fitSize.width) / 2
                let offsetY = (geo.size.height - fitSize.height) / 2

                ZStack(alignment: .topLeading) {
                    Image(uiImage: img)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width: geo.size.width, height: geo.size.height)

                    ForEach(Array(photoResults.enumerated()), id: \.offset) { _, r in
                        PoseAxesCanvas(
                            result: r,
                            imageSize: img.size,
                            displaySize: fitSize,
                            offset: CGPoint(x: offsetX, y: offsetY)
                        )
                    }
                }
            } else {
                VStack {
                    Spacer()
                    Image(systemName: "face.smiling")
                        .font(.system(size: 60))
                        .foregroundStyle(.secondary)
                    Text("Select a face photo")
                        .foregroundStyle(.secondary)
                        .padding(.top, 8)
                    Spacer()
                }
                .frame(maxWidth: .infinity)
            }
        }
    }

    // MARK: - Camera content

    @ViewBuilder
    private var cameraContent: some View {
        GeometryReader { geo in
            ZStack {
                CameraView(position: .front) { pb in
                    if isModelLoaded { detectFaceLive(pb) }
                }

                // Overlay matches AVCaptureVideoPreviewLayer .resizeAspectFill behavior.
                if !liveResults.isEmpty, liveBufferSize.width > 0, liveBufferSize.height > 0 {
                    ForEach(Array(liveResults.enumerated()), id: \.offset) { _, r in
                        PoseAxesCanvas(
                            result: r,
                            imageSize: liveBufferSize,
                            displaySize: aspectFillSize(imageSize: liveBufferSize, in: geo.size),
                            offset: aspectFillOffset(imageSize: liveBufferSize, in: geo.size),
                            mirrorX: true
                        )
                    }
                }
            }
        }
    }

    // MARK: - Model loading

    private func loadModel() async {
        session.ensure { try await ModelLoader.loadPrimary(for: model) }
        do {
            let loaded = try await session.get()
            await MainActor.run {
                mlModel = loaded
                isModelLoaded = true
                status = ""
            }
        } catch {
            await MainActor.run { status = "Load failed: \(error.localizedDescription)" }
        }
    }

    // MARK: - Live camera detection

    private func detectFaceLive(_ pixelBuffer: CVPixelBuffer) {
        guard !isDetecting, let mlModel else { return }
        isDetecting = true

        let bufW = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let bufH = CGFloat(CVPixelBufferGetHeight(pixelBuffer))

        let req = VNDetectFaceRectanglesRequest { req, _ in
            defer { isDetecting = false }
            let faces = (req.results as? [VNFaceObservation]) ?? []
            let results = faces.compactMap { face -> FacePoseResult? in
                runInferencePixelBuffer(
                    pixelBuffer: pixelBuffer,
                    faceRect: face.boundingBox,
                    mlModel: mlModel
                )
            }
            DispatchQueue.main.async {
                liveResults = results
                liveBufferSize = CGSize(width: bufW, height: bufH)
            }
        }
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:]).perform([req])
    }

    // MARK: - Photo inference

    private func loadAndRun() {
        guard let item else { return }
        isProcessing = true
        status = "Loading…"
        Task {
            guard let data = try? await item.loadTransferable(type: Data.self),
                  let img = UIImage(data: data) else {
                await MainActor.run { isProcessing = false; status = "Failed" }
                return
            }
            await MainActor.run {
                inputImage = img
                photoResults = []
            }
            await runOnPhoto(img)
        }
    }

    private func runOnPhoto(_ image: UIImage) async {
        guard let cgImage = ImageUtils.normalizeOrientation(image), let mlModel else {
            await MainActor.run { isProcessing = false; status = "Error" }
            return
        }
        status = "Detecting faces…"

        let faceRects: [CGRect] = await withCheckedContinuation { cont in
            let req = VNDetectFaceRectanglesRequest { req, _ in
                let faces = (req.results as? [VNFaceObservation]) ?? []
                cont.resume(returning: faces.map { $0.boundingBox })
            }
            try? VNImageRequestHandler(cgImage: cgImage, orientation: .up).perform([req])
        }

        status = "Running inference…"
        let start = CFAbsoluteTimeGetCurrent()
        let results = faceRects.compactMap {
            runInferenceCGImage(cgImage: cgImage, faceRect: $0, mlModel: mlModel)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        await MainActor.run {
            photoResults = results
            processingTime = elapsed
            isProcessing = false
            status = ""
        }
    }

    // MARK: - Inference (CGImage path — photo)

    private func runInferenceCGImage(cgImage: CGImage, faceRect: CGRect, mlModel: MLModel) -> FacePoseResult? {
        let w = CGFloat(cgImage.width), h = CGFloat(cgImage.height)
        let cropRect = roiBox(faceRect: faceRect, imageWidth: w, imageHeight: h, flipY: true)
        guard cropRect.width > 0, cropRect.height > 0 else { return nil }
        guard let cropped = cgImage.cropping(to: cropRect) else { return nil }

        let inputSize = model.configInt("input_size") ?? 120
        guard let pb = ImageUtils.pixelBuffer(from: cropped, width: inputSize, height: inputSize) else { return nil }

        return predict(pixelBuffer: pb, faceRect: faceRect, mlModel: mlModel)
    }

    // MARK: - Inference (CVPixelBuffer path — camera)

    private func runInferencePixelBuffer(pixelBuffer: CVPixelBuffer, faceRect: CGRect, mlModel: MLModel) -> FacePoseResult? {
        let w = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let h = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        let cropRect = roiBox(faceRect: faceRect, imageWidth: w, imageHeight: h, flipY: false)
        guard cropRect.width > 0, cropRect.height > 0 else { return nil }

        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            .cropped(to: cropRect)
            .transformed(by: CGAffineTransform(translationX: -cropRect.origin.x, y: -cropRect.origin.y))

        let inputSize = model.configInt("input_size") ?? 120
        let sx = CGFloat(inputSize) / cropRect.width
        let sy = CGFloat(inputSize) / cropRect.height
        let scaled = ciImage.transformed(by: CGAffineTransform(scaleX: sx, y: sy))

        var out: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, inputSize, inputSize, kCVPixelFormatType_32BGRA, nil, &out)
        guard let outBuffer = out else { return nil }
        let ctx = CIContext(options: [.useSoftwareRenderer: false])
        ctx.render(scaled,
                   to: outBuffer,
                   bounds: CGRect(x: 0, y: 0, width: inputSize, height: inputSize),
                   colorSpace: CGColorSpaceCreateDeviceRGB())

        return predict(pixelBuffer: outBuffer, faceRect: faceRect, mlModel: mlModel)
    }

    // MARK: - Core prediction

    private func predict(pixelBuffer: CVPixelBuffer, faceRect: CGRect, mlModel: MLModel) -> FacePoseResult? {
        let inputName = mlModel.modelDescription.inputDescriptionsByName.first {
            $0.value.type == .image
        }?.key ?? "face_image"

        guard let provider = try? MLDictionaryFeatureProvider(dictionary: [inputName: pixelBuffer]),
              let output = try? mlModel.prediction(from: provider) else { return nil }

        guard let arr = output.featureNames.compactMap({
            output.featureValue(for: $0)?.multiArrayValue
        }).first else { return nil }

        let params = ImageUtils.extractFloats(arr)
        guard params.count >= 12 else { return nil }

        // Pose is a flattened 3x4 matrix [R | t] packed as 12 floats row-major.
        var R: [[Float]] = [
            [params[0], params[1], params[2]],
            [params[4], params[5], params[6]],
            [params[8], params[9], params[10]]
        ]
        // Remove isotropic/row scale so the rotation decomposition is well-defined.
        for i in 0..<3 {
            let norm = sqrt(R[i][0] * R[i][0] + R[i][1] * R[i][1] + R[i][2] * R[i][2])
            if norm > 1e-6 {
                R[i][0] /= norm; R[i][1] /= norm; R[i][2] /= norm
            }
        }

        let (yaw, pitch, roll) = eulerAngles(from: R)
        return FacePoseResult(faceRect: faceRect, yaw: yaw, pitch: pitch, roll: roll, rotationMatrix: R)
    }

    // 3DDFA_V2 ROI crop: square box, 1.58x expansion, 0.14 shift toward forehead.
    private func roiBox(faceRect: CGRect, imageWidth: CGFloat, imageHeight: CGFloat, flipY: Bool) -> CGRect {
        let left = faceRect.origin.x * imageWidth
        let right = (faceRect.origin.x + faceRect.width) * imageWidth
        let top: CGFloat
        let bottom: CGFloat
        if flipY {
            top = (1.0 - faceRect.origin.y - faceRect.height) * imageHeight
            bottom = (1.0 - faceRect.origin.y) * imageHeight
        } else {
            top = faceRect.origin.y * imageHeight
            bottom = (faceRect.origin.y + faceRect.height) * imageHeight
        }

        let oldSize = ((right - left) + (bottom - top)) / 2.0
        let centerX = (left + right) / 2.0
        let centerY: CGFloat = flipY
            ? (top + bottom) / 2.0 - oldSize * 0.14
            : (top + bottom) / 2.0 + oldSize * 0.14
        let size = oldSize * 1.58

        return CGRect(x: centerX - size / 2, y: centerY - size / 2, width: size, height: size)
            .intersection(CGRect(x: 0, y: 0, width: imageWidth, height: imageHeight))
    }

    private func eulerAngles(from R: [[Float]]) -> (yaw: Float, pitch: Float, roll: Float) {
        let sy = sqrt(R[0][0] * R[0][0] + R[1][0] * R[1][0])
        let singular = sy < 1e-6

        let pitch: Float
        let yaw: Float
        let roll: Float
        if !singular {
            pitch = atan2(R[2][1], R[2][2])
            yaw = atan2(-R[2][0], sy)
            roll = atan2(R[1][0], R[0][0])
        } else {
            pitch = atan2(-R[1][2], R[1][1])
            yaw = atan2(-R[2][0], sy)
            roll = 0
        }
        let toDeg: Float = 180.0 / .pi
        return (yaw * toDeg, -pitch * toDeg, -roll * toDeg)
    }
}

// MARK: - Supporting types

private struct FacePoseResult {
    let faceRect: CGRect      // Vision normalized rect (origin at bottom-left)
    let yaw: Float            // degrees
    let pitch: Float
    let roll: Float
    let rotationMatrix: [[Float]]
}

/// Canvas overlay that draws the 3DDFA_V2 pose axes (X red, Y green, Z blue)
/// plus a yellow face rect, matching the sample Face3DDemo app.
private struct PoseAxesCanvas: View {
    let result: FacePoseResult
    let imageSize: CGSize
    let displaySize: CGSize
    let offset: CGPoint
    var mirrorX: Bool = false   // true for mirrored front-camera preview

    var body: some View {
        Canvas { ctx, _ in
            let scaleX = displaySize.width / imageSize.width
            let scaleY = displaySize.height / imageSize.height

            // Face rect center in display coords. Vision uses bottom-left origin.
            var cx = result.faceRect.midX * imageSize.width * scaleX + offset.x
            let cy = (1.0 - result.faceRect.midY) * imageSize.height * scaleY + offset.y
            if mirrorX { cx = displaySize.width + 2 * offset.x - cx }
            let center = CGPoint(x: cx, y: cy)

            let axisLen = result.faceRect.width * imageSize.width * scaleX * 0.6
            let effectiveYaw: CGFloat = mirrorX ? -CGFloat(result.yaw) : CGFloat(result.yaw)
            let effectiveRoll: CGFloat = mirrorX ? -CGFloat(result.roll) : CGFloat(result.roll)

            let (xD, yD, zD) = poseAxesFromEuler(
                yaw: effectiveYaw,
                pitch: CGFloat(result.pitch),
                roll: effectiveRoll,
                length: axisLen
            )
            drawAxis(ctx: ctx, from: center, to: CGPoint(x: cx + xD.x, y: cy + xD.y), color: .red)
            drawAxis(ctx: ctx, from: center, to: CGPoint(x: cx + yD.x, y: cy + yD.y), color: .green)
            drawAxis(ctx: ctx, from: center, to: CGPoint(x: cx + zD.x, y: cy + zD.y), color: .blue)

            // Face bounding box.
            var rx = result.faceRect.origin.x * imageSize.width * scaleX + offset.x
            let ry = (1.0 - result.faceRect.origin.y - result.faceRect.height) * imageSize.height * scaleY + offset.y
            let rw = result.faceRect.width * imageSize.width * scaleX
            let rh = result.faceRect.height * imageSize.height * scaleY
            if mirrorX { rx = displaySize.width + 2 * offset.x - rx - rw }
            ctx.stroke(Path(CGRect(x: rx, y: ry, width: rw, height: rh)),
                       with: .color(.yellow), lineWidth: 2)
        }
        .allowsHitTesting(false)
    }

    private func drawAxis(ctx: GraphicsContext, from: CGPoint, to: CGPoint, color: Color) {
        var path = Path()
        path.move(to: from)
        path.addLine(to: to)
        ctx.stroke(path, with: .color(color), style: StrokeStyle(lineWidth: 3, lineCap: .round))

        let dx = to.x - from.x
        let dy = to.y - from.y
        let angle = atan2(dy, dx)
        let headLen: CGFloat = 10
        var arrow = Path()
        arrow.move(to: to)
        arrow.addLine(to: CGPoint(
            x: to.x - headLen * cos(angle - .pi / 6),
            y: to.y - headLen * sin(angle - .pi / 6)
        ))
        arrow.move(to: to)
        arrow.addLine(to: CGPoint(
            x: to.x - headLen * cos(angle + .pi / 6),
            y: to.y - headLen * sin(angle + .pi / 6)
        ))
        ctx.stroke(arrow, with: .color(color), style: StrokeStyle(lineWidth: 3, lineCap: .round))
    }
}

/// Compact info row showing yaw/pitch/roll per face. One-to-one with the
/// sample app's AngleInfoBar.
private struct AngleInfoBar: View {
    let results: [FacePoseResult]

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 16) {
                ForEach(Array(results.enumerated()), id: \.offset) { i, r in
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Face \(i + 1)").font(.caption).bold()
                        HStack(spacing: 12) {
                            angleLabel("Yaw", value: r.yaw, color: .blue)
                            angleLabel("Pitch", value: r.pitch, color: .green)
                            angleLabel("Roll", value: r.roll, color: .red)
                        }
                    }
                    .padding(10)
                    .background(RoundedRectangle(cornerRadius: 8).fill(.ultraThinMaterial))
                }
            }
            .padding(.horizontal)
        }
    }

    @ViewBuilder
    private func angleLabel(_ name: String, value: Float, color: Color) -> some View {
        VStack(spacing: 2) {
            Text(name).font(.caption2).foregroundStyle(.secondary)
            Text(String(format: "%.1f°", value))
                .font(.system(.caption, design: .monospaced))
                .bold()
                .foregroundStyle(color)
        }
    }
}

// MARK: - Layout helpers

private func aspectFit(imageSize: CGSize, in containerSize: CGSize) -> CGSize {
    let scale = min(containerSize.width / imageSize.width,
                    containerSize.height / imageSize.height)
    return CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
}

private func aspectFillSize(imageSize: CGSize, in containerSize: CGSize) -> CGSize {
    let scale = max(containerSize.width / imageSize.width,
                    containerSize.height / imageSize.height)
    return CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
}

private func aspectFillOffset(imageSize: CGSize, in containerSize: CGSize) -> CGPoint {
    let filled = aspectFillSize(imageSize: imageSize, in: containerSize)
    return CGPoint(x: (containerSize.width - filled.width) / 2,
                   y: (containerSize.height - filled.height) / 2)
}

/// Standard draw_axis from the 3DDFA_V2 reference implementation.
/// Returns (dx, dy) offsets for X, Y, Z axes in image coordinates.
private func poseAxesFromEuler(yaw: CGFloat, pitch: CGFloat, roll: CGFloat, length: CGFloat)
    -> (x: CGPoint, y: CGPoint, z: CGPoint) {
    let yr = yaw * .pi / 180
    let pr = pitch * .pi / 180
    let rr = roll * .pi / 180

    // X axis (Red) — right
    let xDx = length * cos(yr) * cos(rr)
    let xDy = length * (cos(pr) * sin(rr) + cos(rr) * sin(pr) * sin(yr))

    // Y axis (Green) — up
    let yDx = length * (cos(yr) * sin(rr))
    let yDy = length * (-cos(pr) * cos(rr) + sin(pr) * sin(yr) * sin(rr))

    // Z axis (Blue) — forward / nose direction
    let zDx = length * sin(yr)
    let zDy = length * (-cos(yr) * sin(pr))

    return (CGPoint(x: xDx, y: xDy), CGPoint(x: yDx, y: yDy), CGPoint(x: zDx, y: zDy))
}
