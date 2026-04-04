import SwiftUI
import PhotosUI

struct ContentView: View {
    @StateObject private var captioner = Florence2Captioner()

    var body: some View {
        TabView {
            PhotoTab(captioner: captioner)
                .tabItem { Label("Photo", systemImage: "photo") }
            CameraTab(captioner: captioner)
                .tabItem { Label("Camera", systemImage: "camera") }
        }
    }
}

// MARK: - Photo Tab

struct PhotoTab: View {
    @ObservedObject var captioner: Florence2Captioner
    @State private var selectedItem: PhotosPickerItem?
    @State private var selectedImage: UIImage?
    @State private var resultText = ""
    @State private var isProcessing = false
    @State private var processingTime: Double?
    @State private var selectedTask: Florence2Task = .caption
    @State private var questionText = ""

    var body: some View {
        NavigationStack {
            VStack(spacing: 12) {
                // Status
                HStack {
                    Circle()
                        .fill(captioner.isReady ? .green : .red)
                        .frame(width: 8, height: 8)
                    Text(captioner.isReady ? "Model Ready" : "Loading Models...")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    if let time = processingTime {
                        Text(String(format: "%.1fs", time))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(.horizontal)

                // Image display
                GeometryReader { geo in
                    if let image = selectedImage {
                        Image(uiImage: image)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(maxWidth: geo.size.width, maxHeight: geo.size.height)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                    } else {
                        RoundedRectangle(cornerRadius: 12)
                            .fill(Color(.systemGray6))
                            .overlay {
                                VStack(spacing: 8) {
                                    Image(systemName: "photo")
                                        .font(.largeTitle)
                                        .foregroundStyle(.tertiary)
                                    Text("Select an image")
                                        .font(.subheadline)
                                        .foregroundStyle(.tertiary)
                                }
                            }
                    }
                }
                .padding(.horizontal)

                // Result
                if !resultText.isEmpty {
                    ScrollView {
                        Text(resultText)
                            .font(.body)
                            .padding()
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(Color(.systemGray6))
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                    }
                    .frame(maxHeight: 100)
                    .padding(.horizontal)
                }

                // Task picker
                Picker("Task", selection: $selectedTask) {
                    ForEach(Florence2Task.allCases, id: \.self) { task in
                        Text(task.rawValue).tag(task)
                    }
                }
                .pickerStyle(.segmented)
                .padding(.horizontal)

                // Controls
                HStack(spacing: 12) {
                    PhotosPicker(selection: $selectedItem, matching: .images) {
                        Label("Photo", systemImage: "photo.on.rectangle")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)

                    Button {
                        runCaption()
                    } label: {
                        if isProcessing {
                            ProgressView()
                                .frame(maxWidth: .infinity)
                        } else {
                            Label("Run", systemImage: "play.fill")
                                .frame(maxWidth: .infinity)
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!captioner.isReady || selectedImage == nil || isProcessing)
                }
                .padding(.horizontal)

                // Question input
                HStack(spacing: 8) {
                    TextField("Ask about this image...", text: $questionText)
                        .textFieldStyle(.roundedBorder)
                        .onSubmit { askQuestion() }
                    Button {
                        askQuestion()
                    } label: {
                        Image(systemName: "arrow.up.circle.fill")
                            .font(.title2)
                    }
                    .disabled(questionText.isEmpty || !captioner.isReady || selectedImage == nil || isProcessing)
                }
                .padding(.horizontal)

                // Copy button
                if !resultText.isEmpty {
                    Button {
                        UIPasteboard.general.string = resultText
                    } label: {
                        Label("Copy", systemImage: "doc.on.doc")
                    }
                    .buttonStyle(.bordered)
                }
            }
            .padding(.vertical)
            .navigationTitle("Florence-2")
            .onChange(of: selectedItem) {
                Task {
                    if let data = try? await selectedItem?.loadTransferable(type: Data.self),
                       let image = UIImage(data: data) {
                        selectedImage = image
                        resultText = ""
                        processingTime = nil
                    }
                }
            }
        }
    }

    private func runCaption() {
        guard let image = selectedImage else { return }
        isProcessing = true
        resultText = ""
        processingTime = nil

        Task {
            let start = CFAbsoluteTimeGetCurrent()
            do {
                let result = try await captioner.caption(image: image, task: selectedTask)
                let elapsed = CFAbsoluteTimeGetCurrent() - start
                await MainActor.run {
                    resultText = result
                    processingTime = elapsed
                    isProcessing = false
                }
            } catch {
                let elapsed = CFAbsoluteTimeGetCurrent() - start
                await MainActor.run {
                    resultText = "Error: \(error.localizedDescription)"
                    processingTime = elapsed
                    isProcessing = false
                }
            }
        }
    }

    private func askQuestion() {
        guard let image = selectedImage, !questionText.isEmpty else { return }
        isProcessing = true
        resultText = ""
        processingTime = nil

        Task {
            let start = CFAbsoluteTimeGetCurrent()
            do {
                let result = try await captioner.answer(image: image, question: questionText)
                let elapsed = CFAbsoluteTimeGetCurrent() - start
                await MainActor.run {
                    resultText = result
                    processingTime = elapsed
                    isProcessing = false
                }
            } catch {
                let elapsed = CFAbsoluteTimeGetCurrent() - start
                await MainActor.run {
                    resultText = "Error: \(error.localizedDescription)"
                    processingTime = elapsed
                    isProcessing = false
                }
            }
        }
    }
}

// MARK: - Camera Tab

struct CameraTab: View {
    @ObservedObject var captioner: Florence2Captioner
    @StateObject private var camera = CameraManager()
    @State private var captionText = ""
    @State private var isProcessing = false
    @State private var isActive = false

    var body: some View {
        NavigationStack {
            ZStack {
                if camera.isAuthorized {
                    CameraPreviewView(session: camera.session)
                        .ignoresSafeArea()

                    VStack {
                        Spacer()

                        HStack(spacing: 8) {
                            if isProcessing {
                                ProgressView()
                                    .tint(.white)
                            }
                            Text(captionText.isEmpty ? "Point camera at something..." : captionText)
                                .font(.body)
                                .foregroundColor(.white)
                                .multilineTextAlignment(.leading)
                        }
                        .padding()
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(.ultraThinMaterial)
                        .clipShape(RoundedRectangle(cornerRadius: 16))
                        .padding()
                    }
                } else {
                    VStack(spacing: 12) {
                        Image(systemName: "camera.fill")
                            .font(.largeTitle)
                            .foregroundStyle(.tertiary)
                        Text("Camera access required")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("Live Caption")
            .navigationBarTitleDisplayMode(.inline)
            .onAppear {
                isActive = true
                camera.start()
            }
            .onDisappear {
                isActive = false
                camera.stop()
            }
            .onChange(of: camera.frameID) {
                guard isActive, !isProcessing, captioner.isReady,
                      let frame = camera.latestFrame else { return }
                processFrame(frame)
            }
        }
    }

    private func processFrame(_ image: UIImage) {
        isProcessing = true
        Task {
            do {
                let result = try await captioner.caption(image: image, task: .caption)
                await MainActor.run {
                    captionText = result
                    isProcessing = false
                }
            } catch {
                await MainActor.run {
                    isProcessing = false
                }
            }
        }
    }
}
