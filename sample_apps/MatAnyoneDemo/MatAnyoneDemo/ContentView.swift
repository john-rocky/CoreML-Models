import AVKit
import CoreImage
import PhotosUI
import SwiftUI

struct ContentView: View {
    @State private var selectedItem: PhotosPickerItem?
    @State private var sourceURL: URL?
    @State private var resultURL: URL?
    @State private var seedMask: UIImage?
    @State private var player: AVPlayer?
    @State private var isProcessing = false
    @State private var progress: Double = 0
    @State private var status: String = ""
    @State private var bgChoice: BackgroundChoice = .green
    /// Loaded once on app launch in the background and reused across runs.
    /// Holding `VideoMatter` here means the 5 mlpackage loads (~30-100MB
    /// each, several hundred ms in total) happen during the splash/idle
    /// time instead of when the user taps Remove BG.
    @State private var videoMatter: VideoMatter?
    @State private var modelLoadError: String?

    enum BackgroundChoice: String, CaseIterable, Identifiable {
        case green = "Green"
        case white = "White"
        case black = "Black"
        var id: String { rawValue }
        var ciColor: CIColor {
            switch self {
            case .green: return CIColor(red: 0.0, green: 0.78, blue: 0.31)
            case .white: return CIColor(red: 1, green: 1, blue: 1)
            case .black: return CIColor(red: 0, green: 0, blue: 0)
            }
        }
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 12) {
                // Player fills all available space; aspectRatio adapts to portrait/landscape.
                // The seed mask is overlaid as a thumbnail in the top-right
                // corner so the user can see what bootstrapped MatAnyone
                // without stealing space from the video.
                GeometryReader { _ in
                    ZStack(alignment: .topTrailing) {
                        Color.black
                        if let player {
                            VideoPlayer(player: player)
                        } else {
                            VStack(spacing: 8) {
                                Image(systemName: "video.slash")
                                    .font(.largeTitle)
                                    .foregroundStyle(.tertiary)
                                Text("Pick a video to remove the background")
                                    .font(.subheadline)
                                    .foregroundStyle(.tertiary)
                            }
                        }
                        if let seedMask {
                            VStack(spacing: 4) {
                                Image(uiImage: seedMask)
                                    .resizable()
                                    .interpolation(.high)
                                    .aspectRatio(contentMode: .fit)
                                    .frame(width: 90)
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 6)
                                            .stroke(Color.white.opacity(0.6), lineWidth: 1)
                                    )
                                Text("Seed mask")
                                    .font(.caption2.bold())
                                    .foregroundStyle(.white)
                                    .padding(.horizontal, 6).padding(.vertical, 2)
                                    .background(.ultraThinMaterial)
                                    .clipShape(Capsule())
                            }
                            .padding(10)
                        }
                    }
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                }
                .padding(.horizontal)

                if isProcessing {
                    VStack(alignment: .leading, spacing: 6) {
                        ProgressView(value: progress)
                        Text(String(format: "%.0f%%  %@", progress * 100, status))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.horizontal)
                } else if !status.isEmpty {
                    Text(status)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Picker("Background", selection: $bgChoice) {
                    ForEach(BackgroundChoice.allCases) { c in Text(c.rawValue).tag(c) }
                }
                .pickerStyle(.segmented)
                .padding(.horizontal)
                .disabled(isProcessing)

                HStack(spacing: 16) {
                    PhotosPicker(selection: $selectedItem, matching: .videos) {
                        Label("Video", systemImage: "video.badge.plus")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .disabled(isProcessing)

                    Button {
                        runMatting()
                    } label: {
                        if isProcessing {
                            ProgressView().frame(maxWidth: .infinity)
                        } else if videoMatter == nil && modelLoadError == nil {
                            // Models still loading from app launch.
                            HStack(spacing: 6) {
                                ProgressView().controlSize(.small)
                                Text("Loading models")
                            }
                            .frame(maxWidth: .infinity)
                        } else {
                            Label("Remove BG", systemImage: "person.crop.rectangle")
                                .frame(maxWidth: .infinity)
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(sourceURL == nil || isProcessing || videoMatter == nil)
                }
                .padding(.horizontal)

                if let url = resultURL {
                    HStack(spacing: 16) {
                        ShareLink(item: url) {
                            Label("Share", systemImage: "square.and.arrow.up")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)

                        Button {
                            saveToPhotos(url)
                        } label: {
                            Label("Save", systemImage: "square.and.arrow.down")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                    }
                    .padding(.horizontal)
                }
            }
            .padding(.vertical, 8)
            .navigationTitle("MatAnyone")
            .navigationBarTitleDisplayMode(.inline)
            .onChange(of: selectedItem) {
                Task { await loadVideo() }
            }
            .task {
                // Pre-load the 5 MatAnyone mlpackages on a background task
                // as soon as the app comes up, so the user doesn't sit on a
                // ~1-2s spinner the first time they tap Remove BG.
                await preloadModels()
            }
        }
    }

    private func preloadModels() async {
        guard videoMatter == nil, modelLoadError == nil else { return }
        if status.isEmpty { status = "Loading models…" }
        do {
            let matter = try await Task.detached(priority: .userInitiated) {
                try VideoMatter()
            }.value
            videoMatter = matter
            if status == "Loading models…" { status = "" }
        } catch {
            modelLoadError = error.localizedDescription
            status = "Model load failed: \(error.localizedDescription)"
        }
    }

    private func loadVideo() async {
        guard let item = selectedItem else { return }
        do {
            if let movie = try await item.loadTransferable(type: VideoFile.self) {
                sourceURL = movie.url
                resultURL = nil
                seedMask = nil
                status = ""
                progress = 0
                player = AVPlayer(url: movie.url)
            }
        } catch {
            status = "Error loading video: \(error.localizedDescription)"
        }
    }

    private func runMatting() {
        guard let url = sourceURL, let matter = videoMatter else { return }
        isProcessing = true
        progress = 0
        status = "Processing frames..."
        resultURL = nil

        Task {
            do {
                let asset = AVURLAsset(url: url)
                let result = try await matter.process(
                    asset: asset,
                    backgroundColor: bgChoice.ciColor
                ) { p in
                    progress = p
                }
                resultURL = result.videoURL
                seedMask = result.firstFrameMask
                player = AVPlayer(url: result.videoURL)
                status = "Done"
            } catch {
                status = "Error: \(error.localizedDescription)"
            }
            isProcessing = false
        }
    }

    private func saveToPhotos(_ url: URL) {
        UISaveVideoAtPathToSavedPhotosAlbum(url.path, nil, nil, nil)
        status = "Saved to Photos"
    }
}

/// Photos picker transfer wrapper that gives us a temp file URL.
struct VideoFile: Transferable {
    let url: URL
    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { file in
            SentTransferredFile(file.url)
        } importing: { received in
            let copy = FileManager.default.temporaryDirectory.appendingPathComponent("source_\(UUID().uuidString).mov")
            try? FileManager.default.removeItem(at: copy)
            try FileManager.default.copyItem(at: received.file, to: copy)
            return VideoFile(url: copy)
        }
    }
}
