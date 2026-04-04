import SwiftUI
import PhotosUI

struct ContentView: View {
    @StateObject private var captioner = Florence2Captioner()
    @State private var selectedItem: PhotosPickerItem?
    @State private var selectedImage: UIImage?
    @State private var captionText: String = ""
    @State private var isProcessing = false
    @State private var processingTime: Double?
    @State private var selectedTask: Florence2Task = .caption

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
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

                // Caption result
                if !captionText.isEmpty {
                    ScrollView {
                        Text(captionText)
                            .font(.body)
                            .padding()
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(Color(.systemGray6))
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                    }
                    .frame(maxHeight: 120)
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

                // Copy button
                if !captionText.isEmpty {
                    Button {
                        UIPasteboard.general.string = captionText
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
                        captionText = ""
                        processingTime = nil
                    }
                }
            }
        }
    }

    private func runCaption() {
        guard let image = selectedImage else { return }
        isProcessing = true
        captionText = ""
        processingTime = nil

        Task {
            let start = CFAbsoluteTimeGetCurrent()
            do {
                let result = try await captioner.caption(image: image, task: selectedTask)
                let elapsed = CFAbsoluteTimeGetCurrent() - start
                await MainActor.run {
                    captionText = result
                    processingTime = elapsed
                    isProcessing = false
                }
            } catch {
                let elapsed = CFAbsoluteTimeGetCurrent() - start
                await MainActor.run {
                    captionText = "Error: \(error.localizedDescription)"
                    processingTime = elapsed
                    isProcessing = false
                }
            }
        }
    }
}
