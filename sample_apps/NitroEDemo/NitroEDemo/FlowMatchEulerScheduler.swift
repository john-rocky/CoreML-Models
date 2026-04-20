import Foundation

/// Swift port of `diffusers.FlowMatchEulerDiscreteScheduler`, matching the
/// defaults used by Nitro-E's `init_pipe` (num_train_timesteps=1000, shift=1.0,
/// no Karras / exponential / beta sigma remapping).
///
/// For a 4-step distilled run, `.setTimesteps(4)` produces
/// timesteps ≈ [1000, 667, 334, 1] and sigmas ≈ [1.0, 0.667, 0.334, 0.001, 0.0].
final class FlowMatchEulerScheduler {

    private(set) var sigmas: [Float] = []
    private(set) var timesteps: [Float] = []
    private let numTrainTimesteps: Int
    private let shift: Float
    private var stepIndex: Int = 0

    init(numTrainTimesteps: Int = 1000, shift: Float = 1.0) {
        self.numTrainTimesteps = numTrainTimesteps
        self.shift = shift
    }

    /// Build the inference schedule. Call before the denoise loop.
    func setTimesteps(_ numInferenceSteps: Int) {
        let n = Float(numInferenceSteps)
        var sig = [Float](repeating: 0, count: numInferenceSteps)
        for i in 0..<numInferenceSteps {
            // linspace(1.0, 1/T, num_inference_steps) where T = numTrainTimesteps
            let t = Float(i) / max(n - 1, 1)
            let s = 1.0 - t * (1.0 - 1.0 / Float(numTrainTimesteps))
            sig[i] = shift * s / (1.0 + (shift - 1.0) * s)
        }
        timesteps = sig.map { $0 * Float(numTrainTimesteps) }
        sigmas = sig + [0.0]  // append terminal 0
        stepIndex = 0
    }

    /// Euler update `x' = x + (σ_next − σ) · v_pred` for one denoise step.
    /// `sample` and `modelOutput` have identical shape. Returns a new array.
    func step(modelOutput: [Float], sample: [Float]) -> [Float] {
        precondition(stepIndex < timesteps.count, "step called past end of schedule")
        let sigma = sigmas[stepIndex]
        let sigmaNext = sigmas[stepIndex + 1]
        let dt = sigmaNext - sigma
        var out = [Float](repeating: 0, count: sample.count)
        for i in 0..<sample.count {
            out[i] = sample[i] + dt * modelOutput[i]
        }
        stepIndex += 1
        return out
    }
}
