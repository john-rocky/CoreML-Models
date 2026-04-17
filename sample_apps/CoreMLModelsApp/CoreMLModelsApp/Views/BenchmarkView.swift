import CoreML
import SwiftUI

struct BenchmarkView: View {
    let model: ModelEntry

    @State private var selectedFile: FileSpec
    @State private var selectedComputeUnits: MLComputeUnits
    @State private var iterations: Int = 10
    @State private var phase: BenchmarkPhase = .idle
    @State private var result: BenchmarkResult?
    @State private var benchmarkTask: Task<Void, Never>?

    private var modelFiles: [FileSpec] {
        model.files.filter { ($0.kind ?? "model") == "model" }
    }

    private static let computeUnitOptions: [(MLComputeUnits, String)] = [
        (.all, "All"),
        (.cpuAndNeuralEngine, "CPU + Neural Engine"),
        (.cpuAndGPU, "CPU + GPU"),
        (.cpuOnly, "CPU Only"),
    ]

    init(model: ModelEntry) {
        self.model = model
        let primary = model.files.first { ($0.kind ?? "model") == "model" } ?? model.files[0]
        _selectedFile = State(initialValue: primary)
        _selectedComputeUnits = State(initialValue: ModelLoader.parseComputeUnits(primary.computeUnits))
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                configSection
                runButton
                progressSection
                if let result { resultsSection(result) }
                if case .failed(let msg) = phase {
                    VStack(alignment: .leading, spacing: 6) {
                        Label("Error", systemImage: "exclamationmark.triangle.fill")
                            .font(.subheadline.bold())
                            .foregroundStyle(.red)
                        Text(msg)
                            .font(.caption)
                            .foregroundStyle(.red)
                            .textSelection(.enabled)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .padding()
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.red.opacity(0.1),
                                in: RoundedRectangle(cornerRadius: 12))
                }
            }
            .padding()
        }
        .navigationTitle("Benchmark")
        .navigationBarTitleDisplayMode(.inline)
        .onDisappear { benchmarkTask?.cancel() }
    }

    // MARK: - Config

    @ViewBuilder
    private var configSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Configuration").font(.headline)

            if modelFiles.count > 1 {
                Picker("Model file", selection: $selectedFile) {
                    ForEach(modelFiles) { file in
                        Text(file.name).tag(file)
                    }
                }
                .onChange(of: selectedFile) { _, newValue in
                    selectedComputeUnits = ModelLoader.parseComputeUnits(newValue.computeUnits)
                }
            }

            Stepper("Iterations: \(iterations)", value: $iterations, in: 5...100, step: 5)
                .font(.subheadline)

            Picker("Compute units", selection: $selectedComputeUnits) {
                ForEach(Self.computeUnitOptions, id: \.0) { opt in
                    Text(opt.1).tag(opt.0)
                }
            }
            .font(.subheadline)

            if let recommended = selectedFile.computeUnits,
               ModelLoader.parseComputeUnits(recommended) != selectedComputeUnits {
                Text("Recommended: \(recommended)")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }
        .padding()
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    // MARK: - Run

    @ViewBuilder
    private var runButton: some View {
        Button {
            startBenchmark()
        } label: {
            Label(result != nil ? "Run Again" : "Run Benchmark", systemImage: "timer")
                .frame(maxWidth: .infinity)
        }
        .buttonStyle(.borderedProminent)
        .disabled(phase.isRunning)
    }

    // MARK: - Progress

    @ViewBuilder
    private var progressSection: some View {
        switch phase {
        case .loadingModel:
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text("Loading model…").font(.caption).foregroundStyle(.secondary)
            }
        case .warmup(let current, let total):
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text("Warmup \(current)/\(total)").font(.caption).foregroundStyle(.secondary)
            }
        case .measuring(let current, let total):
            VStack(alignment: .leading, spacing: 4) {
                ProgressView(value: Double(current), total: Double(total))
                Text("Measuring \(current)/\(total)")
                    .font(.caption).foregroundStyle(.secondary)
            }
        default:
            EmptyView()
        }
    }

    // MARK: - Results

    @ViewBuilder
    private func resultsSection(_ r: BenchmarkResult) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Results").font(.headline)

            resultRow("Average", String(format: "%.1f ms", r.averageMs))
            resultRow("Median", String(format: "%.1f ms", r.medianMs))
            resultRow("Min", String(format: "%.1f ms", r.minMs))
            resultRow("Max", String(format: "%.1f ms", r.maxMs))
            resultRow("Iterations", "\(r.iterations) (+\(r.warmupIterations) warmup)")

            if let mem = r.memoryDeltaMB {
                resultRow("Memory delta", String(format: "%+.1f MB", mem))
            }

            Divider()

            Text("Device").font(.subheadline.bold())
            resultRow("Model", r.deviceName)
            resultRow("Chip", r.chipName)
        }
        .padding()
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    @ViewBuilder
    private func resultRow(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label).foregroundStyle(.secondary)
            Spacer()
            Text(value).monospacedDigit()
        }
        .font(.subheadline)
    }

    // MARK: - Execution

    private func startBenchmark() {
        benchmarkTask?.cancel()
        result = nil
        phase = .idle

        benchmarkTask = Task {
            do {
                let r = try await BenchmarkRunner.run(
                    model: model,
                    file: selectedFile,
                    iterations: iterations,
                    computeUnits: selectedComputeUnits
                ) { newPhase in
                    phase = newPhase
                }
                result = r
            } catch is CancellationError {
                phase = .idle
            } catch {
                phase = .failed(error.localizedDescription)
            }
        }
    }
}
