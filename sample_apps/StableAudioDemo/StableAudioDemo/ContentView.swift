import SwiftUI
import AVFoundation

struct ContentView: View {
    @StateObject private var generator = MusicGenerator()
    @State private var prompt = "A gentle piano melody with soft strings"
    @State private var seconds: Float = 8.0
    @State private var steps = 25
    @State private var seed: String = ""
    @State private var isGenerating = false
    @State private var progressStep = 0
    @State private var progressTotal = 0
    @State private var progressMessage = ""
    @State private var outputURL: URL?
    @State private var player: AVAudioPlayer?
    @State private var isPlaying = false
    @State private var errorMessage: String?

    private let presets = [
        "A gentle piano melody with soft strings",
        "Drum breaks 174 BPM",
        "Glitchy bass design",
        "Synth pluck arp with reverb and delay, 128 BPM",
        "Birds singing in the forest",
        "A short beautiful piano riff in C minor",
    ]

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    // Model status
                    HStack {
                        Circle()
                            .fill(generator.isReady ? .green : .red)
                            .frame(width: 10, height: 10)
                        Text(generator.isReady ? "Models loaded" : generator.status)
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Spacer()
                    }
                    .padding(.horizontal)

                    // Prompt
                    VStack(alignment: .leading, spacing: 8) {
                        Label("Prompt", systemImage: "text.quote")
                            .font(.headline)

                        TextEditor(text: $prompt)
                            .frame(minHeight: 60, maxHeight: 100)
                            .padding(8)
                            .background(Color(.systemGray6))
                            .cornerRadius(10)

                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 8) {
                                ForEach(presets, id: \.self) { preset in
                                    Button(preset) {
                                        prompt = preset
                                    }
                                    .font(.caption)
                                    .padding(.horizontal, 10)
                                    .padding(.vertical, 6)
                                    .background(Color(.systemGray5))
                                    .cornerRadius(8)
                                }
                            }
                        }
                    }
                    .padding(.horizontal)

                    // Duration slider
                    VStack(alignment: .leading, spacing: 8) {
                        Label("Duration: \(String(format: "%.1f", seconds))s", systemImage: "clock")
                            .font(.headline)
                        Slider(value: $seconds, in: 1...11.9, step: 0.5)
                    }
                    .padding(.horizontal)

                    // Steps slider
                    VStack(alignment: .leading, spacing: 8) {
                        Label("Steps: \(steps)", systemImage: "arrow.triangle.2.circlepath")
                            .font(.headline)
                        Slider(value: Binding(
                            get: { Double(steps) },
                            set: { steps = Int($0) }
                        ), in: 5...50, step: 1)
                    }
                    .padding(.horizontal)

                    // Seed
                    HStack {
                        Label("Seed", systemImage: "dice")
                            .font(.headline)
                        TextField("Random", text: $seed)
                            .textFieldStyle(.roundedBorder)
                            .keyboardType(.numberPad)
                            .frame(width: 120)
                        Spacer()
                    }
                    .padding(.horizontal)

                    // Generate button
                    Button {
                        generateMusic()
                    } label: {
                        HStack {
                            if isGenerating {
                                ProgressView()
                                    .tint(.white)
                            }
                            Image(systemName: "waveform")
                            Text(isGenerating ? "Generating..." : "Generate Music")
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(generator.isReady && !isGenerating ? Color.blue : Color.gray)
                        .foregroundColor(.white)
                        .cornerRadius(12)
                    }
                    .disabled(!generator.isReady || isGenerating)
                    .padding(.horizontal)

                    // Progress
                    if isGenerating {
                        VStack(spacing: 8) {
                            ProgressView(value: Double(progressStep), total: Double(max(progressTotal, 1)))
                            Text(progressMessage)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        .padding(.horizontal)
                    }

                    // Error
                    if let errorMessage {
                        Text(errorMessage)
                            .font(.caption)
                            .foregroundColor(.red)
                            .padding(.horizontal)
                    }

                    // Playback
                    if outputURL != nil {
                        VStack(spacing: 12) {
                            Divider()
                            Label("Generated Audio", systemImage: "music.note")
                                .font(.headline)

                            HStack(spacing: 20) {
                                Button {
                                    togglePlayback()
                                } label: {
                                    Image(systemName: isPlaying ? "stop.circle.fill" : "play.circle.fill")
                                        .font(.system(size: 44))
                                        .foregroundColor(.blue)
                                }

                                if let url = outputURL {
                                    ShareLink(item: url) {
                                        Image(systemName: "square.and.arrow.up")
                                            .font(.title2)
                                    }
                                }
                            }
                        }
                        .padding(.horizontal)
                    }
                }
                .padding(.vertical)
            }
            .navigationTitle("Stable Audio")
        }
    }

    // MARK: - Actions

    private func generateMusic() {
        isGenerating = true
        errorMessage = nil
        outputURL = nil
        player?.stop()
        isPlaying = false

        let seedValue: UInt64
        if let parsed = UInt64(seed), !seed.isEmpty {
            seedValue = parsed
        } else {
            seedValue = UInt64.random(in: 0...UInt64.max)
        }

        Task {
            do {
                let url = try await generator.generate(
                    prompt: prompt,
                    seconds: seconds,
                    steps: steps,
                    seed: seedValue
                ) { step, total, message in
                    DispatchQueue.main.async {
                        progressStep = step
                        progressTotal = total
                        progressMessage = message
                    }
                }
                await MainActor.run {
                    outputURL = url
                    isGenerating = false
                }
            } catch {
                await MainActor.run {
                    errorMessage = error.localizedDescription
                    isGenerating = false
                }
            }
        }
    }

    private func togglePlayback() {
        if isPlaying {
            player?.stop()
            isPlaying = false
            return
        }
        guard let url = outputURL else { return }
        do {
            player = try AVAudioPlayer(contentsOf: url)
            player?.play()
            isPlaying = true
        } catch {
            errorMessage = "Playback error: \(error.localizedDescription)"
        }
    }

}

#Preview {
    ContentView()
}
