import Foundation
import CoreML
import Accelerate

/// Swift prompt encoder for MobileSAM.
/// The CoreML decoder does not include prompt encoding (`index_put_` incompatibility),
/// so we run it in Swift using weights exported from PyTorch.
/// Vendored verbatim from SAMKit.
final class PromptEncoder {

    /// Upper bound on user points supported by the decoder model's enumerated
    /// sparse_embeddings shape. The model accepts up to (maxPoints + 1) tokens.
    static let maxPoints = 9

    private let embedDim: Int
    private let imageEmbeddingSize: (h: Int, w: Int)
    private let inputImageSize: (h: Int, w: Int)

    private let gaussianMatrix: [Float]
    private let numPosFeats: Int
    private let pointEmbeddings: [[Float]]
    private let notAPointEmbed: [Float]
    private let noMaskEmbed: [Float]

    private let cachedDenseEmbedding: MLMultiArray

    init(weightsURL: URL) throws {
        let data = try Data(contentsOf: weightsURL)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]

        self.embedDim = json["embed_dim"] as! Int
        let ies = json["image_embedding_size"] as! [Int]
        self.imageEmbeddingSize = (ies[0], ies[1])
        let iis = json["input_image_size"] as! [Int]
        self.inputImageSize = (iis[0], iis[1])

        let gm = json["gaussian_matrix"] as! [[Double]]
        self.numPosFeats = gm[0].count
        self.gaussianMatrix = gm.flatMap { $0.map { Float($0) } }

        let pe = json["point_embeddings"] as! [[Double]]
        self.pointEmbeddings = pe.map { $0.map { Float($0) } }

        self.notAPointEmbed = (json["not_a_point_embed"] as! [Double]).map { Float($0) }
        self.noMaskEmbed = (json["no_mask_embed"] as! [Double]).map { Float($0) }

        let h = ies[0]; let w = ies[1]; let ed = self.embedDim
        let dense = try MLMultiArray(shape: [1, ed as NSNumber, h as NSNumber, w as NSNumber],
                                     dataType: .float32)
        let dPtr = dense.dataPointer.bindMemory(to: Float32.self, capacity: dense.count)
        let spatial = vDSP_Length(h * w)
        for c in 0..<ed {
            var val = self.noMaskEmbed[c]
            vDSP_vfill(&val, dPtr + c * Int(spatial), 1, spatial)
        }
        self.cachedDenseEmbedding = dense
    }

    /// Encode point and box prompts into sparse embeddings; dense is cached (no-mask).
    func encode(
        points: [SamPoint],
        box: SamBox? = nil,
        transform: TransformParams
    ) throws -> (MLMultiArray, MLMultiArray) {

        let boxSlots = (box != nil) ? 2 : 0
        let maxUserPoints = Self.maxPoints - boxSlots
        let clampedPoints = points.count > maxUserPoints
            ? Array(points.suffix(maxUserPoints))
            : points

        var coords: [(Float, Float)] = []
        var labels: [Float] = []

        if clampedPoints.isEmpty && box == nil {
            coords.append((Float(inputImageSize.w / 2), Float(inputImageSize.h / 2)))
            labels.append(-1)
        } else {
            for p in clampedPoints {
                let mp = transform.toModel(p)
                coords.append((Float(mp.x), Float(mp.y)))
                labels.append(Float(p.label.rawValue))
            }
        }

        if let box = box {
            let tl = SamPoint(x: CGFloat(box.x0), y: CGFloat(box.y0), label: .positive)
            let br = SamPoint(x: CGFloat(box.x1), y: CGFloat(box.y1), label: .positive)
            let tlM = transform.toModel(tl); let brM = transform.toModel(br)
            coords.append((Float(tlM.x), Float(tlM.y))); labels.append(2)
            coords.append((Float(brM.x), Float(brM.y))); labels.append(3)
        }

        coords.append((0, 0)); labels.append(-1)

        let numTokens = coords.count
        var sparse = [Float](repeating: 0, count: numTokens * embedDim)

        for i in 0..<numTokens {
            let cx = coords[i].0 + 0.5
            let cy = coords[i].1 + 0.5
            let pe = positionalEncoding(x: cx, y: cy)
            let offset = i * embedDim
            for d in 0..<embedDim { sparse[offset + d] = pe[d] }

            let label = labels[i]
            if label < 0 {
                for d in 0..<embedDim { sparse[offset + d] = notAPointEmbed[d] }
            } else if label == 0 {
                for d in 0..<embedDim { sparse[offset + d] += pointEmbeddings[0][d] }
            } else if label == 1 {
                for d in 0..<embedDim { sparse[offset + d] += pointEmbeddings[1][d] }
            } else if label == 2 {
                for d in 0..<embedDim { sparse[offset + d] += pointEmbeddings[2][d] }
            } else if label == 3 {
                for d in 0..<embedDim { sparse[offset + d] += pointEmbeddings[3][d] }
            }
        }

        let sparseArray = try MLMultiArray(
            shape: [1, numTokens as NSNumber, embedDim as NSNumber],
            dataType: .float32
        )
        let sparsePtr = sparseArray.dataPointer.bindMemory(to: Float32.self, capacity: sparse.count)
        sparse.withUnsafeBufferPointer { buf in
            sparsePtr.update(from: buf.baseAddress!, count: sparse.count)
        }
        return (sparseArray, cachedDenseEmbedding)
    }

    private func positionalEncoding(x: Float, y: Float) -> [Float] {
        let nx = x / Float(inputImageSize.w)
        let ny = y / Float(inputImageSize.h)
        let sx = 2.0 * nx - 1.0
        let sy = 2.0 * ny - 1.0
        var result = [Float](repeating: 0, count: embedDim)
        for j in 0..<numPosFeats {
            let val = sx * gaussianMatrix[j] + sy * gaussianMatrix[numPosFeats + j]
            let angle = val * 2.0 * .pi
            result[j] = sin(angle)
            result[numPosFeats + j] = cos(angle)
        }
        return result
    }
}
