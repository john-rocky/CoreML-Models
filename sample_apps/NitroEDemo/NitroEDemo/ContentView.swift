import SwiftUI
import CoreML
#if canImport(UIKit)
import UIKit
#endif

struct ContentView: View {
    @StateObject private var pipeline = PipelineHolder()
    @State private var prompt = "a hot air balloon in the shape of a heart, grand canyon"
    @State private var image: CGImage?
    @State private var seed: UInt64 = 42
    @State private var status = "loading models…"
    @State private var timing = ""
    @State private var isRunning = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                header
                imageView
                TextField("Prompt", text: $prompt, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .lineLimit(2...4)
                    .padding(.horizontal)
                HStack {
                    Text("Seed:").font(.caption).foregroundStyle(.secondary)
                    TextField("", value: $seed, format: .number)
                        .textFieldStyle(.roundedBorder).frame(width: 80)
                    Button { seed = UInt64.random(in: 0...99_999) } label: { Image(systemName: "dice") }
                    Spacer()
                }
                .padding(.horizontal)
                generateButton
                if let image {
                    ShareLink(item: Image(decorative: image, scale: 1, orientation: .up), preview: SharePreview("Nitro-E output", image: Image(decorative: image, scale: 1, orientation: .up))) {
                        Label("Save / Share", systemImage: "square.and.arrow.up")
                    }
                }
            }
            .padding()
            .navigationTitle("Nitro-E Demo")
            .contentShape(Rectangle())
            .onTapGesture { dismissKeyboard() }
        }
        .task { await load() }
    }

    private func dismissKeyboard() {
        #if canImport(UIKit)
        UIApplication.shared.sendAction(
            #selector(UIResponder.resignFirstResponder),
            to: nil, from: nil, for: nil
        )
        #endif
    }

    private var header: some View {
        HStack {
            Circle().fill(pipeline.isReady ? .green : .orange).frame(width: 8, height: 8)
            Text(status).font(.caption).foregroundStyle(.secondary)
            Spacer()
            if !timing.isEmpty {
                Text(timing).font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private var imageView: some View {
        ZStack {
            if let image {
                Image(decorative: image, scale: 1, orientation: .up)
                    .resizable().interpolation(.high)
                    .aspectRatio(1, contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
            } else {
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color.gray.opacity(0.15))
                    .aspectRatio(1, contentMode: .fit)
                    .overlay {
                        VStack(spacing: 8) {
                            Image(systemName: "wand.and.stars").font(.largeTitle).foregroundStyle(.tertiary)
                            Text("Enter a prompt and tap Generate").font(.subheadline).foregroundStyle(.tertiary)
                        }
                    }
            }
        }
        .padding(.horizontal)
    }

    private var generateButton: some View {
        Button {
            Task { await run() }
        } label: {
            if isRunning {
                HStack { ProgressView(); Text("Generating…") }.frame(maxWidth: .infinity)
            } else {
                Label("Generate", systemImage: "sparkles").frame(maxWidth: .infinity)
            }
        }
        .buttonStyle(.borderedProminent)
        .disabled(!pipeline.isReady || isRunning)
        .padding(.horizontal)
    }

    private func load() async {
        do {
            try await pipeline.load { status = "loading \($0)…" }
            status = "ready"
        } catch {
            status = "load failed: \(error)"
        }
    }

    private func run() async {
        guard let pipe = pipeline.instance else { return }
        isRunning = true
        status = "running"
        timing = ""
        do {
            let result = try await pipe.generate(prompt: prompt, steps: 4, seed: seed)
            image = result.image
            timing = String(format: "text %.0f + denoise %.0f + vae %.0f = %.0f ms",
                            result.textMs, result.denoiseMs, result.decodeMs,
                            result.textMs + result.denoiseMs + result.decodeMs)
            status = "done"
        } catch {
            status = "error: \(error)"
        }
        isRunning = false
    }
}

@MainActor
final class PipelineHolder: ObservableObject {
    @Published var isReady = false
    private(set) var instance: NitroEPipeline?

    func load(progress: @escaping (String) -> Void) async throws {
        let tokenizer = try LlamaTokenizer()
        let pipe = NitroEPipeline(tokenizer: tokenizer)
        try await pipe.warmUp(progress: progress)
        self.instance = pipe
        self.isReady = pipe.isReady
    }
}
