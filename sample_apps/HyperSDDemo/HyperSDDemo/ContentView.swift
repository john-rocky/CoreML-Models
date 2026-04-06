import SwiftUI
import CoreML

struct ContentView: View {
    @State private var prompt = "A photo of a cat wearing sunglasses on a beach"
    @State private var resultImage: CGImage?
    @State private var isGenerating = false
    @State private var generationTime: Double?
    @State private var seed: UInt32 = 42
    @State private var status = ""
    @State private var pipeline: StableDiffusionPipeline?

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                HStack {
                    Circle()
                        .fill(pipeline != nil ? .green : .red)
                        .frame(width: 8, height: 8)
                    Text(status.isEmpty ? (pipeline != nil ? "Ready" : "Loading...") : status)
                        .font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    if let t = generationTime {
                        Text(String(format: "%.1fs", t))
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                .padding(.horizontal)

                ZStack {
                    if let image = resultImage {
                        Image(decorative: image, scale: 1.0)
                            .resizable()
                            .interpolation(.high)
                            .aspectRatio(1, contentMode: .fit)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                    } else {
                        RoundedRectangle(cornerRadius: 12)
                            .fill(Color(.systemGray6))
                            .aspectRatio(1, contentMode: .fit)
                            .overlay {
                                VStack(spacing: 8) {
                                    Image(systemName: "wand.and.stars")
                                        .font(.largeTitle).foregroundStyle(.tertiary)
                                    Text("Enter a prompt to generate")
                                        .font(.subheadline).foregroundStyle(.tertiary)
                                }
                            }
                    }
                }
                .frame(maxWidth: .infinity)
                .padding(.horizontal)

                TextField("Describe an image...", text: $prompt, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .lineLimit(2...4)
                    .padding(.horizontal)

                HStack {
                    Text("Seed:").font(.caption).foregroundStyle(.secondary)
                    TextField("", value: $seed, format: .number)
                        .textFieldStyle(.roundedBorder).frame(width: 80)
                    Button { seed = UInt32.random(in: 0...99999) } label: {
                        Image(systemName: "dice")
                    }
                    Spacer()
                }
                .padding(.horizontal)

                Button {
                    generate()
                } label: {
                    if isGenerating {
                        HStack { ProgressView(); Text("Generating...") }
                            .frame(maxWidth: .infinity)
                    } else {
                        Label("Generate", systemImage: "sparkles")
                            .frame(maxWidth: .infinity)
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(pipeline == nil || isGenerating || prompt.isEmpty)
                .padding(.horizontal)

                if let image = resultImage {
                    Button {
                        let uiImage = UIImage(cgImage: image)
                        UIImageWriteToSavedPhotosAlbum(uiImage, nil, nil, nil)
                    } label: {
                        Label("Save", systemImage: "square.and.arrow.down")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .padding(.horizontal)
                }
            }
            .padding(.vertical)
            .navigationTitle("Hyper-SD")
            .task { await loadPipeline() }
        }
    }

    private func loadPipeline() async {
        status = "Loading pipeline..."
        do {
            guard let resourceURL = Bundle.main.resourceURL else {
                status = "Resources not found"
                return
            }
            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndNeuralEngine

            let pipe = try StableDiffusionPipeline(
                resourcesAt: resourceURL,
                controlNet: [],
                configuration: config,
                disableSafety: true,
                reduceMemory: true
            )
            pipeline = pipe
            status = ""
        } catch {
            status = "Error: \(error.localizedDescription)"
        }
    }

    private func generate() {
        guard let pipe = pipeline else { return }
        isGenerating = true
        resultImage = nil
        generationTime = nil
        status = "Generating..."

        Task.detached {
            let start = CFAbsoluteTimeGetCurrent()
            do {
                var config = StableDiffusionPipeline.Configuration(prompt: prompt)
                config.stepCount = 1
                config.seed = seed
                config.guidanceScale = 1.0  // No CFG amplification
                config.schedulerType = .tcdScheduler  // Hyper-SD requires TCD

                let images = try pipe.generateImages(configuration: config)
                let elapsed = CFAbsoluteTimeGetCurrent() - start

                await MainActor.run {
                    resultImage = images.first ?? nil
                    generationTime = elapsed
                    status = ""
                    isGenerating = false
                }
            } catch {
                let elapsed = CFAbsoluteTimeGetCurrent() - start
                await MainActor.run {
                    status = "Error: \(error.localizedDescription)"
                    generationTime = elapsed
                    isGenerating = false
                }
            }
        }
    }
}
