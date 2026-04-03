import SwiftUI
import PhotosUI
import AVFoundation

struct ContentView: View {
    @StateObject private var recognizer = FaceRecognizer()

    var body: some View {
        TabView {
            RegisterTab(recognizer: recognizer)
                .tabItem { Label("Register", systemImage: "person.badge.plus") }
            CompareTab(recognizer: recognizer)
                .tabItem { Label("Compare", systemImage: "person.2.circle") }
            CameraTab(recognizer: recognizer)
                .tabItem { Label("Live", systemImage: "camera") }
        }
    }
}

// MARK: - Register Tab

struct RegisterTab: View {
    @ObservedObject var recognizer: FaceRecognizer
    @State private var selectedItem: PhotosPickerItem?
    @State private var selectedImage: UIImage?
    @State private var name = ""
    @State private var registering = false
    @State private var message = ""
    @State private var showCamera = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                HStack(spacing: 12) {
                    // Face image preview
                    Group {
                        if let img = selectedImage {
                            Image(uiImage: img)
                                .resizable()
                                .scaledToFill()
                                .frame(width: 80, height: 80)
                                .clipShape(Circle())
                        } else {
                            ZStack {
                                Circle().fill(.quaternary).frame(width: 80, height: 80)
                                Image(systemName: "person.crop.circle.badge.plus")
                                    .font(.title)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                    .onTapGesture { showCamera = true }

                    VStack(alignment: .leading, spacing: 8) {
                        TextField("Name", text: $name)
                            .textFieldStyle(.roundedBorder)

                        HStack(spacing: 8) {
                            Button {
                                registerFace()
                            } label: {
                                HStack {
                                    if registering {
                                        ProgressView().controlSize(.small)
                                    }
                                    Text("Register")
                                }
                                .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.borderedProminent)
                            .disabled(selectedImage == nil || name.isEmpty || registering)

                            PhotosPicker(selection: $selectedItem, matching: .images) {
                                Image(systemName: "photo")
                                    .frame(width: 36, height: 36)
                            }
                            .buttonStyle(.bordered)

                            Button {
                                showCamera = true
                            } label: {
                                Image(systemName: "camera")
                                    .frame(width: 36, height: 36)
                            }
                            .buttonStyle(.bordered)
                        }
                    }
                }
                .padding(.horizontal)

                if !message.isEmpty {
                    Text(message)
                        .font(.caption)
                        .foregroundColor(message.contains("Error") ? .red : .green)
                        .padding(.horizontal)
                }

                List {
                    ForEach(recognizer.registered) { face in
                        HStack(spacing: 12) {
                            if let thumb = face.thumbnail {
                                Image(uiImage: thumb)
                                    .resizable()
                                    .scaledToFill()
                                    .frame(width: 44, height: 44)
                                    .clipShape(Circle())
                            } else {
                                Image(systemName: "person.circle.fill")
                                    .font(.title2)
                                    .foregroundColor(.blue)
                                    .frame(width: 44, height: 44)
                            }
                            VStack(alignment: .leading) {
                                Text(face.name).font(.body).bold()
                                Text(face.date, style: .date)
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                    .onDelete { offsets in
                        recognizer.deleteRegistered(at: offsets)
                    }

                    if recognizer.registered.isEmpty {
                        Text("No faces registered yet")
                            .foregroundColor(.secondary)
                    }
                }
            }
            .navigationTitle("Face Registration")
            .onTapGesture {
                UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder),
                                                to: nil, from: nil, for: nil)
            }
            .sheet(isPresented: $showCamera) {
                CameraCaptureView { image in
                    selectedImage = image
                    showCamera = false
                }
            }
        }
        .onChange(of: selectedItem) { _ in loadImage() }
    }

    private func loadImage() {
        guard let item = selectedItem else { return }
        Task {
            if let data = try? await item.loadTransferable(type: Data.self),
               let img = UIImage(data: data) {
                selectedImage = img
            }
        }
    }

    private func registerFace() {
        guard let img = selectedImage else { return }
        registering = true
        message = ""
        Task {
            let success = await recognizer.register(name: name, image: img)
            await MainActor.run {
                if success {
                    message = "Registered \(name)"
                    name = ""
                    selectedImage = nil
                    selectedItem = nil
                } else {
                    message = "Error: No face detected in image"
                }
                registering = false
            }
        }
    }
}

// MARK: - Compare Tab

struct CompareEntry: Identifiable {
    let id = UUID()
    let thumbnail: UIImage?
    let embedding: [Float]
    var similarity: Float
}

struct CompareTab: View {
    @ObservedObject var recognizer: FaceRecognizer
    @State private var selectedRefIndex = 0
    @State private var entries: [CompareEntry] = []
    @State private var selectedItem: PhotosPickerItem?
    @State private var processing = false

    private var reference: FaceEmbedding? {
        guard !recognizer.registered.isEmpty,
              selectedRefIndex < recognizer.registered.count else { return nil }
        return recognizer.registered[selectedRefIndex]
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                if recognizer.registered.isEmpty {
                    Spacer()
                    Text("Register a face first").foregroundColor(.secondary)
                    Spacer()
                } else {
                    // Reference selector
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 12) {
                            ForEach(Array(recognizer.registered.enumerated()), id: \.element.id) { i, face in
                                VStack(spacing: 4) {
                                    if let thumb = face.thumbnail {
                                        Image(uiImage: thumb)
                                            .resizable()
                                            .scaledToFill()
                                            .frame(width: 48, height: 48)
                                            .clipShape(Circle())
                                            .overlay(Circle().stroke(i == selectedRefIndex ? Color.blue : Color.clear, lineWidth: 3))
                                    } else {
                                        Circle().fill(.quaternary).frame(width: 48, height: 48)
                                            .overlay(Circle().stroke(i == selectedRefIndex ? Color.blue : Color.clear, lineWidth: 3))
                                    }
                                    Text(face.name).font(.caption2).lineLimit(1)
                                }
                                .onTapGesture {
                                    selectedRefIndex = i
                                    recalcSimilarities()
                                }
                            }
                        }
                        .padding(.horizontal)
                    }
                    .padding(.vertical, 8)

                    // Radial comparison view
                    GeometryReader { geo in
                        let center = CGPoint(x: geo.size.width / 2, y: geo.size.height / 2)
                        let maxRadius = min(geo.size.width, geo.size.height) / 2 - 40

                        ZStack {
                            // Threshold circles
                            ForEach([0.6, 0.4, 0.2], id: \.self) { threshold in
                                let r = maxRadius * CGFloat(1.0 - threshold)
                                Circle()
                                    .stroke(threshold == 0.6 ? Color.green.opacity(0.3) : Color.gray.opacity(0.15), lineWidth: 1)
                                    .frame(width: r * 2, height: r * 2)
                                    .position(center)
                            }

                            // "Same person" zone label
                            Text("Same Person")
                                .font(.caption2)
                                .foregroundColor(.green.opacity(0.5))
                                .position(x: center.x, y: center.y - maxRadius * 0.4 + 14)

                            // 60% label
                            Text("60%")
                                .font(.caption2)
                                .foregroundColor(.green.opacity(0.4))
                                .position(x: center.x + maxRadius * 0.4 + 16, y: center.y)

                            // Reference face at center
                            if let ref = reference, let thumb = ref.thumbnail {
                                Image(uiImage: thumb)
                                    .resizable()
                                    .scaledToFill()
                                    .frame(width: 56, height: 56)
                                    .clipShape(Circle())
                                    .overlay(Circle().stroke(Color.blue, lineWidth: 3))
                                    .position(center)
                            } else {
                                Circle().fill(.blue.opacity(0.3)).frame(width: 56, height: 56)
                                    .overlay(Text(reference?.name.prefix(1) ?? "?").font(.title2).foregroundColor(.white))
                                    .position(center)
                            }

                            // Comparison entries
                            ForEach(Array(entries.enumerated()), id: \.element.id) { i, entry in
                                let angle = entryAngle(index: i, total: entries.count)
                                let dist = maxRadius * CGFloat(1.0 - max(0, entry.similarity))
                                let x = center.x + dist * cos(angle)
                                let y = center.y + dist * sin(angle)
                                let isMatch = entry.similarity >= 0.6

                                VStack(spacing: 2) {
                                    if let thumb = entry.thumbnail {
                                        Image(uiImage: thumb)
                                            .resizable()
                                            .scaledToFill()
                                            .frame(width: 40, height: 40)
                                            .clipShape(Circle())
                                            .overlay(Circle().stroke(isMatch ? Color.green : Color.red, lineWidth: 2))
                                    } else {
                                        Circle().fill(.gray).frame(width: 40, height: 40)
                                    }
                                    Text(String(format: "%.0f%%", entry.similarity * 100))
                                        .font(.system(size: 10, weight: .bold, design: .monospaced))
                                        .foregroundColor(isMatch ? .green : .red)
                                }
                                .position(x: x, y: y)
                            }
                        }
                    }

                    // Bottom bar
                    HStack {
                        PhotosPicker(selection: $selectedItem, matching: .images) {
                            Label("Add Photo", systemImage: "plus.circle.fill")
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(reference == nil || processing)

                        if processing {
                            ProgressView().controlSize(.small)
                        }

                        Spacer()

                        if !entries.isEmpty {
                            Button("Clear") {
                                entries.removeAll()
                            }
                            .foregroundColor(.red)
                        }
                    }
                    .padding()
                }
            }
            .navigationTitle("Compare")
            .onChange(of: selectedItem) { _ in addCompareImage() }
        }
    }

    private func entryAngle(index: Int, total: Int) -> CGFloat {
        guard total > 0 else { return 0 }
        let start: CGFloat = -.pi / 2
        return start + CGFloat(index) * (2 * .pi / CGFloat(max(total, 1)))
    }

    private func addCompareImage() {
        guard let item = selectedItem, let ref = reference else { return }
        processing = true
        Task {
            if let data = try? await item.loadTransferable(type: Data.self),
               let img = UIImage(data: data),
               let result = await recognizer.extractEmbedding(image: img) {
                let sim = recognizer.similarity(ref.embedding, result.embedding)
                await MainActor.run {
                    entries.append(CompareEntry(thumbnail: result.thumbnail, embedding: result.embedding, similarity: sim))
                }
            }
            await MainActor.run {
                processing = false
                selectedItem = nil
            }
        }
    }

    private func recalcSimilarities() {
        guard let ref = reference else { return }
        for i in entries.indices {
            entries[i].similarity = recognizer.similarity(ref.embedding, entries[i].embedding)
        }
    }
}

// MARK: - Camera Capture for Registration

struct CameraCaptureView: UIViewControllerRepresentable {
    let onCapture: (UIImage) -> Void

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.cameraDevice = .front
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ vc: UIImagePickerController, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(onCapture: onCapture) }

    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let onCapture: (UIImage) -> Void
        init(onCapture: @escaping (UIImage) -> Void) { self.onCapture = onCapture }

        func imagePickerController(_ picker: UIImagePickerController,
                                   didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
            if let image = info[.originalImage] as? UIImage {
                onCapture(image)
            }
            picker.dismiss(animated: true)
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            picker.dismiss(animated: true)
        }
    }
}

// MARK: - Camera Tab

struct CameraTab: View {
    @ObservedObject var recognizer: FaceRecognizer

    var body: some View {
        ZStack {
            CameraViewWrapper(recognizer: recognizer)
                .ignoresSafeArea()

            if recognizer.registered.isEmpty {
                VStack {
                    Spacer()
                    Text("Register faces first to start recognition")
                        .font(.callout)
                        .padding()
                        .background(.ultraThinMaterial)
                        .cornerRadius(10)
                        .padding(.bottom, 100)
                }
            }
        }
    }
}

struct CameraViewWrapper: UIViewControllerRepresentable {
    let recognizer: FaceRecognizer
    func makeUIViewController(context: Context) -> CameraVC { CameraVC(recognizer: recognizer) }
    func updateUIViewController(_ vc: CameraVC, context: Context) {}
}

// MARK: - Camera VC

class CameraVC: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let recognizer: FaceRecognizer
    private let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "session")
    private let inferenceQueue = DispatchQueue(label: "inference")
    private var previewLayer: AVCaptureVideoPreviewLayer!
    private var overlayLayer = CALayer()
    private var isProcessing = false
    private var smoothedLatency: Double = 0

    init(recognizer: FaceRecognizer) {
        self.recognizer = recognizer
        super.init(nibName: nil, bundle: nil)
    }

    required init?(coder: NSCoder) { fatalError() }

    override func viewDidLoad() {
        super.viewDidLoad()
        setupCamera()
        setupOverlay()
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer?.frame = view.bounds
        overlayLayer.frame = view.bounds
    }

    private func setupCamera() {
        previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)

        sessionQueue.async { [weak self] in
            guard let self else { return }
            session.beginConfiguration()
            session.sessionPreset = .high

            guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front),
                  let input = try? AVCaptureDeviceInput(device: device),
                  session.canAddInput(input) else { return }
            session.addInput(input)

            let output = AVCaptureVideoDataOutput()
            output.alwaysDiscardsLateVideoFrames = true
            output.setSampleBufferDelegate(self, queue: inferenceQueue)
            if session.canAddOutput(output) {
                session.addOutput(output)
            }
            if let conn = output.connection(with: .video) {
                conn.videoOrientation = .portrait
                conn.isVideoMirrored = true
            }
            session.commitConfiguration()
            session.startRunning()
        }
    }

    private func setupOverlay() {
        view.layer.addSublayer(overlayLayer)
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard !isProcessing else { return }
        isProcessing = true

        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            isProcessing = false
            return
        }

        let start = CACurrentMediaTime()
        let results = recognizer.recognize(pixelBuffer: pb)
        let latency = (CACurrentMediaTime() - start) * 1000
        smoothedLatency = smoothedLatency * 0.8 + latency * 0.2

        let bufW = CGFloat(CVPixelBufferGetWidth(pb))
        let bufH = CGFloat(CVPixelBufferGetHeight(pb))

        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.drawResults(results, bufferSize: CGSize(width: bufW, height: bufH))
            self.isProcessing = false
        }
    }

    private func drawResults(_ results: [RecognitionResult], bufferSize: CGSize) {
        overlayLayer.sublayers?.forEach { $0.removeFromSuperlayer() }

        let viewSize = view.bounds.size
        let scale = max(viewSize.width / bufferSize.width, viewSize.height / bufferSize.height)
        let scaledW = bufferSize.width * scale
        let scaledH = bufferSize.height * scale
        let offsetX = (viewSize.width - scaledW) / 2
        let offsetY = (viewSize.height - scaledH) / 2

        for r in results {
            let rx = r.faceRect.origin.x * scaledW + offsetX
            let ry = (1.0 - r.faceRect.origin.y - r.faceRect.height) * scaledH + offsetY
            let rw = r.faceRect.width * scaledW
            let rh = r.faceRect.height * scaledH
            let faceFrame = CGRect(x: rx, y: ry, width: rw, height: rh)

            let matched = r.match != nil && r.similarity > 0.3
            let borderColor = matched ? UIColor.systemGreen : UIColor.systemRed

            let rectLayer = CAShapeLayer()
            rectLayer.path = UIBezierPath(roundedRect: faceFrame, cornerRadius: 8).cgPath
            rectLayer.strokeColor = borderColor.cgColor
            rectLayer.fillColor = UIColor.clear.cgColor
            rectLayer.lineWidth = 3
            overlayLayer.addSublayer(rectLayer)

            let label = CATextLayer()
            let nameText = r.match?.name ?? "Unknown"
            let simText = String(format: "%.0f%%", r.similarity * 100)
            label.string = "\(nameText)  \(simText)"
            label.fontSize = 16
            label.font = UIFont.boldSystemFont(ofSize: 16)
            label.foregroundColor = UIColor.white.cgColor
            label.backgroundColor = borderColor.withAlphaComponent(0.7).cgColor
            label.cornerRadius = 6
            label.contentsScale = UIScreen.main.scale
            label.alignmentMode = .center
            label.frame = CGRect(x: rx, y: ry - 30, width: rw, height: 26)
            overlayLayer.addSublayer(label)

            let barBg = CALayer()
            barBg.frame = CGRect(x: rx, y: ry + rh + 4, width: rw, height: 6)
            barBg.backgroundColor = UIColor.darkGray.cgColor
            barBg.cornerRadius = 3
            overlayLayer.addSublayer(barBg)

            let barFill = CALayer()
            let fillWidth = rw * CGFloat(max(0, min(1, r.similarity)))
            barFill.frame = CGRect(x: rx, y: ry + rh + 4, width: fillWidth, height: 6)
            barFill.backgroundColor = borderColor.cgColor
            barFill.cornerRadius = 3
            overlayLayer.addSublayer(barFill)
        }

        let statsLayer = CATextLayer()
        statsLayer.string = String(format: "  %.0f ms", smoothedLatency)
        statsLayer.fontSize = 14
        statsLayer.foregroundColor = UIColor.white.cgColor
        statsLayer.backgroundColor = UIColor.black.withAlphaComponent(0.5).cgColor
        statsLayer.cornerRadius = 6
        statsLayer.contentsScale = UIScreen.main.scale
        statsLayer.frame = CGRect(x: 16, y: view.safeAreaInsets.top + 8, width: 100, height: 30)
        overlayLayer.addSublayer(statsLayer)
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        sessionQueue.async { [weak self] in
            self?.session.stopRunning()
        }
    }
}
