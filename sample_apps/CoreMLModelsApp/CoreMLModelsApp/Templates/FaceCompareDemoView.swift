import SwiftUI
import PhotosUI
import CoreML
import Vision

/// Face comparison with Register + Compare + Camera tabs.
/// Matches AdaFaceDemo: face registration, radial similarity visualization, live camera recognition.
struct FaceCompareDemoView: View {
    let model: ModelEntry

    enum Tab: String, CaseIterable { case register = "Register", compare = "Compare", camera = "Camera" }

    @State private var tab: Tab = .register
    @State private var mlModel: MLModel?
    @State private var isModelLoaded = false
    @State private var registeredFaces: [(String, UIImage, [Float])] = []  // (name, thumbnail, embedding)

    var body: some View {
        VStack(spacing: 0) {
            Picker("Tab", selection: $tab) {
                ForEach(Tab.allCases, id: \.self) { Text($0.rawValue).tag($0) }
            }.pickerStyle(.segmented).padding(.horizontal).padding(.top, 4)

            switch tab {
            case .register: registerTab
            case .compare: compareTab
            case .camera: cameraTab
            }
        }
        .task { await loadModel() }
        .onDisappear {
            mlModel = nil
            registeredFaces.removeAll()
        }
    }

    private var threshold: Float { Float(model.configDouble("match_threshold") ?? 0.6) }

    // MARK: - Register Tab

    @State private var registerItem: PhotosPickerItem?
    @State private var registerImage: UIImage?
    @State private var registerName = ""
    @State private var registerStatus = ""

    @ViewBuilder
    private var registerTab: some View {
        VStack(spacing: 16) {
            if let img = registerImage {
                Image(uiImage: img).resizable().aspectRatio(contentMode: .fill)
                    .frame(width: 120, height: 120).clipShape(Circle())
            } else {
                Circle().fill(Color(.systemGray5)).frame(width: 120, height: 120)
                    .overlay { Image(systemName: "person.crop.circle.badge.plus").font(.title).foregroundStyle(.secondary) }
            }

            PhotosPicker(selection: $registerItem, matching: .images) {
                Label("Select Face", systemImage: "photo")
            }.buttonStyle(.bordered)
            .onChange(of: registerItem) { _, item in
                guard let item else { return }
                Task {
                    if let data = try? await item.loadTransferable(type: Data.self),
                       let img = UIImage(data: data) { registerImage = img }
                }
            }

            TextField("Name", text: $registerName).textFieldStyle(.roundedBorder).frame(maxWidth: 200)

            Button {
                Task { await registerFace() }
            } label: {
                Label("Register", systemImage: "person.badge.plus").frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(registerImage == nil || registerName.isEmpty || !isModelLoaded)

            if !registerStatus.isEmpty {
                Text(registerStatus).font(.caption).foregroundStyle(registerStatus.contains("Error") ? .red : .green)
            }

            if !registeredFaces.isEmpty {
                List {
                    ForEach(registeredFaces.indices, id: \.self) { i in
                        HStack {
                            Image(uiImage: registeredFaces[i].1).resizable().aspectRatio(contentMode: .fill)
                                .frame(width: 40, height: 40).clipShape(Circle())
                            Text(registeredFaces[i].0).font(.body)
                        }
                    }
                    .onDelete { registeredFaces.remove(atOffsets: $0) }
                }.listStyle(.plain)
            }

            Spacer()
        }.padding()
    }

    // MARK: - Compare Tab

    @State private var compareItem: PhotosPickerItem?
    @State private var compareImage: UIImage?
    @State private var compareResult: (String, Float)?

    @ViewBuilder
    private var compareTab: some View {
        VStack(spacing: 16) {
            if let img = compareImage {
                Image(uiImage: img).resizable().aspectRatio(contentMode: .fill)
                    .frame(width: 150, height: 150).clipShape(RoundedRectangle(cornerRadius: 12))
            }

            if let (name, sim) = compareResult {
                let isMatch = sim >= threshold
                HStack {
                    Image(systemName: isMatch ? "checkmark.circle.fill" : "xmark.circle.fill")
                        .foregroundStyle(isMatch ? .green : .red).font(.title)
                    VStack(alignment: .leading) {
                        Text(isMatch ? "Match: \(name)" : "No match").font(.headline)
                        Text(String(format: "Similarity: %.3f", sim)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    }
                }
                ProgressView(value: Double(max(0, min(1, sim)))).tint(isMatch ? .green : .red).padding(.horizontal)
            }

            // Radial visualization of all registered faces
            if !registeredFaces.isEmpty && compareResult != nil {
                GeometryReader { geo in
                    let center = CGPoint(x: geo.size.width / 2, y: geo.size.height / 2)
                    let maxR = min(geo.size.width, geo.size.height) / 2 - 30

                    // Threshold circle
                    Circle().stroke(Color.green.opacity(0.3), lineWidth: 1)
                        .frame(width: maxR * 2 * CGFloat(threshold), height: maxR * 2 * CGFloat(threshold))
                        .position(center)

                    ForEach(registeredFaces.indices, id: \.self) { i in
                        let (_, thumb, emb) = registeredFaces[i]
                        let sim = compareResult.map { _ in
                            compareImage.flatMap { img -> Float? in
                                guard let queryEmb = extractEmbeddingSync(img) else { return nil }
                                return ImageUtils.cosineSimilarity(queryEmb, emb)
                            } ?? 0
                        } ?? Float(0)
                        let dist = (1 - CGFloat(sim)) * maxR
                        let angle = CGFloat(i) * 2 * .pi / CGFloat(registeredFaces.count)
                        let pos = CGPoint(x: center.x + dist * cos(angle), y: center.y + dist * sin(angle))

                        Image(uiImage: thumb).resizable().aspectRatio(contentMode: .fill)
                            .frame(width: 40, height: 40).clipShape(Circle())
                            .overlay(Circle().stroke(sim >= threshold ? Color.green : Color.red, lineWidth: 2))
                            .position(pos)
                    }
                }.frame(height: 200)
            }

            Spacer()

            PhotosPicker(selection: $compareItem, matching: .images) {
                Label("Select Photo to Compare", systemImage: "person.crop.rectangle").frame(maxWidth: .infinity)
            }.buttonStyle(.bordered)
            .onChange(of: compareItem) { _, item in
                guard let item else { return }
                Task {
                    if let data = try? await item.loadTransferable(type: Data.self),
                       let img = UIImage(data: data) {
                        compareImage = img
                        await compare(img)
                    }
                }
            }
        }.padding()
    }

    // MARK: - Camera Tab

    @State private var liveFaceLabel = ""
    @State private var liveSimilarity: Float = 0
    @State private var isCameraDetecting = false

    @ViewBuilder
    private var cameraTab: some View {
        ZStack(alignment: .bottom) {
            CameraView(position: .front) { pb in
                guard isModelLoaded, !registeredFaces.isEmpty, !isCameraDetecting else { return }
                detectLive(pb)
            }

            if !liveFaceLabel.isEmpty {
                HStack {
                    Circle().fill(liveSimilarity >= threshold ? Color.green : Color.red).frame(width: 12, height: 12)
                    Text(liveFaceLabel).font(.headline).foregroundStyle(.white)
                    Text(String(format: "%.0f%%", liveSimilarity * 100)).font(.caption.monospacedDigit()).foregroundStyle(.white)
                }
                .padding(.horizontal, 16).padding(.vertical, 10)
                .background(.ultraThinMaterial).clipShape(Capsule())
                .padding(.bottom, 20)
            }

            if registeredFaces.isEmpty {
                Text("Register faces first").font(.headline).foregroundStyle(.white)
                    .padding().background(.ultraThinMaterial).clipShape(Capsule())
            }
        }
    }

    // MARK: - Model Loading

    private func loadModel() async {
        do {
            mlModel = try await ModelLoader.loadPrimary(for: model)
            isModelLoaded = true
        } catch {
            registerStatus = "Error: \(error.localizedDescription)"
        }
    }

    // MARK: - Registration

    private func registerFace() async {
        guard let image = registerImage, let model = mlModel else { return }
        registerStatus = "Detecting face…"

        guard let embedding = await extractEmbedding(image) else {
            registerStatus = "Error: No face detected"; return
        }

        let thumb = ImageUtils.resize(image, to: CGSize(width: 80, height: 80)) ?? image
        registeredFaces.append((registerName, thumb, embedding))
        registerStatus = "\(registerName) registered!"
        registerName = ""; registerImage = nil; registerItem = nil
    }

    // MARK: - Comparison

    private func compare(_ image: UIImage) async {
        guard let queryEmb = await extractEmbedding(image) else {
            compareResult = ("No face", 0); return
        }
        var bestName = "Unknown"; var bestSim: Float = 0
        for (name, _, emb) in registeredFaces {
            let sim = ImageUtils.cosineSimilarity(queryEmb, emb)
            if sim > bestSim { bestSim = sim; bestName = name }
        }
        compareResult = (bestName, bestSim)
    }

    // MARK: - Live Camera

    private func detectLive(_ pixelBuffer: CVPixelBuffer) {
        isCameraDetecting = true
        let ci = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cgImage = CIContext().createCGImage(ci, from: ci.extent),
              let emb = extractEmbeddingSync(UIImage(cgImage: cgImage)) else {
            isCameraDetecting = false; return
        }

        var bestName = ""; var bestSim: Float = 0
        for (name, _, regEmb) in registeredFaces {
            let sim = ImageUtils.cosineSimilarity(emb, regEmb)
            if sim > bestSim { bestSim = sim; bestName = name }
        }

        DispatchQueue.main.async {
            liveFaceLabel = bestSim >= threshold ? bestName : "Unknown"
            liveSimilarity = bestSim
            isCameraDetecting = false
        }
    }

    // MARK: - Embedding Extraction

    private func extractEmbedding(_ image: UIImage) async -> [Float]? {
        guard let cgImage = ImageUtils.normalizeOrientation(image), let mlModel else { return nil }

        // Detect and crop face
        let faceRect: CGRect = await withCheckedContinuation { cont in
            let req = VNDetectFaceRectanglesRequest { req, _ in
                let rect = (req.results as? [VNFaceObservation])?.first?.boundingBox
                    ?? CGRect(x: 0.1, y: 0.1, width: 0.8, height: 0.8)
                cont.resume(returning: rect)
            }
            try? VNImageRequestHandler(cgImage: cgImage, orientation: .up).perform([req])
        }

        let w = CGFloat(cgImage.width), h = CGFloat(cgImage.height)
        let fx = faceRect.origin.x * w, fy = (1 - faceRect.origin.y - faceRect.height) * h
        let fw = faceRect.width * w, fh = faceRect.height * h
        let expand: CGFloat = 0.3
        let cropRect = CGRect(x: fx - fw*expand, y: fy - fh*expand, width: fw*(1+2*expand), height: fh*(1+2*expand))
            .intersection(CGRect(x: 0, y: 0, width: w, height: h))

        guard let cropped = cgImage.cropping(to: cropRect) else { return nil }
        let inputSize = model.configInt("input_size") ?? 112
        guard let pb = ImageUtils.pixelBuffer(from: cropped, width: inputSize, height: inputSize) else { return nil }

        let inputName = mlModel.modelDescription.inputDescriptionsByName.first { $0.value.type == .image }?.key ?? "face_image"
        guard let output = try? await mlModel.prediction(from: MLDictionaryFeatureProvider(dictionary: [inputName: pb])) else { return nil }
        let embName = output.featureNames.first(where: { $0.contains("embed") }) ?? output.featureNames.first ?? ""
        guard let embArr = output.featureValue(for: embName)?.multiArrayValue else { return nil }
        return ImageUtils.extractFloats(embArr)
    }

    private func extractEmbeddingSync(_ image: UIImage) -> [Float]? {
        guard let cgImage = ImageUtils.normalizeOrientation(image), let mlModel else { return nil }
        let inputSize = model.configInt("input_size") ?? 112
        guard let pb = ImageUtils.pixelBuffer(from: cgImage, width: inputSize, height: inputSize) else { return nil }
        let inputName = mlModel.modelDescription.inputDescriptionsByName.first { $0.value.type == .image }?.key ?? "face_image"
        guard let output = try? mlModel.prediction(from: MLDictionaryFeatureProvider(dictionary: [inputName: pb])) else { return nil }
        let embName = output.featureNames.first(where: { $0.contains("embed") }) ?? output.featureNames.first ?? ""
        guard let embArr = output.featureValue(for: embName)?.multiArrayValue else { return nil }
        return ImageUtils.extractFloats(embArr)
    }
}
