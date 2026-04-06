import Accelerate
import CoreML

/// TCD (Trajectory Consistency Distillation) Scheduler for Hyper-SD 1-step inference.
/// Mirrors HuggingFace diffusers' TCDScheduler with eta=1.0.
@available(iOS 16.2, macOS 13.1, *)
public final class TCDScheduler: Scheduler {
    public let trainStepCount: Int
    public let inferenceStepCount: Int
    public let betas: [Float]
    public let alphas: [Float]
    public let alphasCumProd: [Float]
    public let timeSteps: [Int]
    public private(set) var modelOutputs: [MLShapedArray<Float32>] = []

    private let alphaProd: [Float]
    private let strength: Float = 0.0
    public var initNoiseSigma: Float { 1.0 }

    public init(
        stepCount: Int = 1,
        trainStepCount: Int = 1000,
        betaSchedule: BetaSchedule = .scaledLinear,
        betaStart: Float = 0.00085,
        betaEnd: Float = 0.012
    ) {
        self.trainStepCount = trainStepCount
        self.inferenceStepCount = stepCount

        switch betaSchedule {
        case .linear:
            self.betas = linspace(betaStart, betaEnd, trainStepCount)
        case .scaledLinear:
            self.betas = linspace(pow(betaStart, 0.5), pow(betaEnd, 0.5), trainStepCount).map { $0 * $0 }
        }
        self.alphas = betas.map { 1.0 - $0 }
        var cumProd = self.alphas
        for i in 1..<cumProd.count { cumProd[i] *= cumProd[i - 1] }
        self.alphasCumProd = cumProd
        self.alphaProd = cumProd

        // Trailing timesteps (matching diffusers TCDScheduler)
        var ts = [Int]()
        let stepRatio = Float(trainStepCount) / Float(stepCount)
        for i in stride(from: Float(stepCount), through: 1, by: -1) {
            ts.append(Int((i * stepRatio).rounded()) - 1)
        }
        self.timeSteps = ts
    }

    public func step(
        output: MLShapedArray<Float32>,
        timeStep t: Int,
        sample s: MLShapedArray<Float32>
    ) -> MLShapedArray<Float32> {
        // TCD with eta=1.0 (full stochasticity, equivalent to using prev_sample = pred_original_sample)
        // For 1-step: pred_original_sample is the final output
        // pred_original_sample = (sample - sqrt(1 - alpha_t) * model_output) / sqrt(alpha_t)
        let alphaProdT = alphaProd[t]
        let sqrtAlpha = sqrt(alphaProdT)
        let sqrtBeta = sqrt(1 - alphaProdT)
        let scalarCount = s.scalarCount

        return MLShapedArray(unsafeUninitializedShape: s.shape) { scalars, _ in
            s.withUnsafeShapedBufferPointer { sampleBuf, _, _ in
                output.withUnsafeShapedBufferPointer { outputBuf, _, _ in
                    for i in 0..<scalarCount {
                        let predX0 = (sampleBuf[i] - sqrtBeta * outputBuf[i]) / sqrtAlpha
                        scalars.initializeElement(at: i, to: predX0)
                    }
                }
            }
        }
    }
}
