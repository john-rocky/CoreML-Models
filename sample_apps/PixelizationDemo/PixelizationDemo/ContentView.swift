import SwiftUI
import PhotosUI
import UIKit

struct ContentView: View {
    @State private var selectedItem: PhotosPickerItem?
    @State private var originalImage: UIImage?
    @State private var networkOutput: CGImage?        // 512×512 raw model output
    @State private var displayImage: UIImage?         // after preset post-process
    @State private var presetId: String = PixelArtPreset.all[0].id
    @State private var cellSizeOverride: Double?
    @State private var blurOverride: Int?
    @State private var isProcessing = false
    @State private var status = ""
    @State private var processingTime: Double?

    private var preset: PixelArtPreset {
        PixelArtPreset.all.first { $0.id == presetId } ?? PixelArtPreset.all[0]
    }
    private var cellSize: Int {
        Int(cellSizeOverride ?? Double(preset.cellSize))
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                HStack {
                    Spacer()
                    if let t = processingTime {
                        Text(String(format: "%.1fs", t))
                            .font(.caption).foregroundColor(.secondary)
                    }
                }
                .padding(.horizontal)
                .padding(.vertical, 4)

                GeometryReader { geo in
                    Group {
                        if let display = displayImage {
                            Image(uiImage: display)
                                .resizable()
                                .interpolation(.none)
                                .aspectRatio(contentMode: .fit)
                        } else if let original = originalImage {
                            Image(uiImage: original)
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                        } else {
                            RoundedRectangle(cornerRadius: 12)
                                .fill(Color(.systemGray6))
                                .overlay {
                                    VStack(spacing: 8) {
                                        Image(systemName: "gamecontroller")
                                            .font(.largeTitle)
                                            .foregroundStyle(.tertiary)
                                        Text("Select a photo")
                                            .font(.subheadline)
                                            .foregroundStyle(.tertiary)
                                    }
                                }
                        }
                    }
                    .frame(maxWidth: geo.size.width, maxHeight: geo.size.height)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                }
                .padding(.horizontal)

                VStack(spacing: 12) {
                    if isProcessing {
                        ProgressView(status)
                    }

                    if networkOutput != nil {
                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 8) {
                                ForEach(PixelArtPreset.all, id: \.id) { p in
                                    Button {
                                        presetId = p.id
                                    } label: {
                                        VStack(spacing: 2) {
                                            Image(systemName: p.systemImage).font(.body)
                                            Text(p.name).font(.caption2)
                                        }
                                        .padding(.vertical, 6).padding(.horizontal, 10)
                                        .background(
                                            presetId == p.id
                                                ? Color.accentColor.opacity(0.25)
                                                : Color(.systemGray6)
                                        )
                                        .cornerRadius(8)
                                    }
                                    .buttonStyle(.plain)
                                }
                            }
                            .padding(.horizontal, 4)
                        }

                        HStack(spacing: 10) {
                            Image(systemName: "square.grid.3x3")
                                .font(.caption).foregroundStyle(.secondary)
                            Slider(
                                value: Binding(
                                    get: { Double(cellSize) },
                                    set: { cellSizeOverride = $0 }
                                ),
                                in: 4...10, step: 1
                            ) { Text("Cell size") }
                            Text("\(cellSize)")
                                .font(.caption.monospacedDigit())
                                .frame(width: 22, alignment: .trailing)
                        }

                        Picker("Abstraction", selection: Binding(
                            get: { blurOverride ?? 0 },
                            set: { blurOverride = $0 == 0 ? nil : $0 }
                        )) {
                            Text("Auto").tag(0)
                            Text("Off").tag(512)
                            Text("256").tag(256)
                            Text("128").tag(128)
                            Text("64").tag(64)
                            Text("32").tag(32)
                        }
                        .pickerStyle(.segmented)
                        .onChange(of: blurOverride) {
                            if originalImage != nil { pixelize() }
                        }
                    }

                    HStack(spacing: 16) {
                        PhotosPicker(selection: $selectedItem, matching: .images) {
                            Label("Photo", systemImage: "photo.on.rectangle")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)

                        Button {
                            pixelize()
                        } label: {
                            if isProcessing {
                                ProgressView().frame(maxWidth: .infinity)
                            } else {
                                Label("Pixelize", systemImage: "square.grid.3x3")
                                    .frame(maxWidth: .infinity)
                            }
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(originalImage == nil || isProcessing)
                    }

                    if let result = displayImage {
                        HStack(spacing: 16) {
                            ShareLink(
                                item: Image(uiImage: result),
                                preview: SharePreview("Pixelized",
                                                      image: Image(uiImage: result))
                            ) {
                                Label("Share", systemImage: "square.and.arrow.up")
                                    .frame(maxWidth: .infinity)
                            }.buttonStyle(.bordered)

                            Button {
                                savePNG(result)
                            } label: {
                                Label("Save PNG", systemImage: "square.and.arrow.down")
                                    .frame(maxWidth: .infinity)
                            }.buttonStyle(.bordered)
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("Pixelization")
            .onChange(of: selectedItem) {
                Task {
                    if let data = try? await selectedItem?.loadTransferable(type: Data.self),
                       let image = UIImage(data: data) {
                        originalImage = image
                        networkOutput = nil
                        displayImage = nil
                        processingTime = nil
                        status = ""
                        cellSizeOverride = nil
                    }
                }
            }
            .onChange(of: presetId) {
                cellSizeOverride = nil
                // Preset changes the cell size → re-run the network with the
                // new pre-blur. Cheap post-process alone can't match the
                // paper's scale-aware abstraction.
                if originalImage != nil {
                    pixelize()
                } else if let net = networkOutput {
                    displayImage = Pixelizer.postProcess(net, cellSize: cellSize, palette: preset.palette)
                }
            }
            .onChange(of: cellSizeOverride) {
                // During drag: cheap palette re-snap only.
                guard let net = networkOutput else { return }
                displayImage = Pixelizer.postProcess(net, cellSize: cellSize, palette: preset.palette)
            }
        }
    }

    private func pixelize() {
        guard let image = originalImage else { return }
        isProcessing = true
        status = "Pixelizing..."
        networkOutput = nil
        displayImage = nil
        processingTime = nil

        Task {
            let start = CFAbsoluteTimeGetCurrent()
            do {
                let blurTarget = blurOverride ?? Pixelizer.preBlurTargetSize(for: cellSize)
                let raw = try await Pixelizer.runModel(on: image, preBlurTarget: blurTarget)
                let ui = Pixelizer.postProcess(raw, cellSize: cellSize, palette: preset.palette)
                let elapsed = CFAbsoluteTimeGetCurrent() - start
                await MainActor.run {
                    networkOutput = raw
                    displayImage = ui
                    processingTime = elapsed
                    status = ""
                    isProcessing = false
                }
            } catch {
                await MainActor.run {
                    status = "Error: \(error.localizedDescription)"
                    isProcessing = false
                }
            }
        }
    }

    private func savePNG(_ image: UIImage) {
        guard let data = image.pngData(),
              let pngImage = UIImage(data: data) else { return }
        UIImageWriteToSavedPhotosAlbum(pngImage, nil, nil, nil)
        status = "Saved to Photos"
    }
}
