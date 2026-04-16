import Foundation
import CoreML
import Accelerate

/// Swift prompt encoder for MobileSAM.
///
/// The MobileSAM CoreML decoder does not include prompt encoding
/// (due to index_put_ incompatibility in coremltools), so encoding
/// is performed in Swift using weights exported from the PyTorch model.
/// Ported from SamKit (github.com/john-rocky/SamKit).
final class SAMPromptEncoder {

    static let maxPoints = 9

    private let embedDim: Int
    private let imageEmbeddingSize: (h: Int, w: Int)
    private let inputImageSize: (h: Int, w: Int)

    private let gaussianMatrix: [Float]   // [2 * numPosFeats]
    private let numPosFeats: Int
    private let pointEmbeddings: [[Float]] // 4 x [embedDim]
    private let notAPointEmbed: [Float]    // [embedDim]
    private let noMaskEmbed: [Float]       // [embedDim]

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

        // Pre-build cached dense embedding [1, embedDim, H, W]
        let h = ies[0], w = ies[1], ed = self.embedDim
        let dense = try MLMultiArray(shape: [1, ed as NSNumber, h as NSNumber, w as NSNumber], dataType: .float32)
        let dPtr = dense.dataPointer.bindMemory(to: Float32.self, capacity: dense.count)
        let spatialSize = vDSP_Length(h * w)
        for c in 0..<ed {
            var val = self.noMaskEmbed[c]
            vDSP_vfill(&val, dPtr + c * Int(spatialSize), 1, spatialSize)
        }
        self.cachedDenseEmbedding = dense
    }

    /// Encode point prompts into sparse and dense embeddings for the decoder.
    /// - Parameters:
    ///   - points: Tap points in model coordinates (after letterbox transform).
    ///   - labels: 1.0 for positive (foreground), 0.0 for negative (background).
    /// - Returns: (sparse_embeddings [1, N, embedDim], dense_embeddings [1, embedDim, H, W])
    func encode(points: [(x: Float, y: Float)], labels: [Float]) throws -> (MLMultiArray, MLMultiArray) {
        let clamped = Array(points.suffix(Self.maxPoints))
        let clampedLabels = Array(labels.suffix(Self.maxPoints))

        var coords: [(Float, Float)] = []
        var allLabels: [Float] = []

        if clamped.isEmpty {
            coords.append((Float(inputImageSize.w / 2), Float(inputImageSize.h / 2)))
            allLabels.append(-1)
        } else {
            for (i, p) in clamped.enumerated() {
                coords.append((p.x, p.y))
                allLabels.append(clampedLabels[i])
            }
        }

        // SAM always pads with one extra token
        coords.append((0, 0))
        allLabels.append(-1)

        let numTokens = coords.count
        var sparse = [Float](repeating: 0, count: numTokens * embedDim)

        for i in 0..<numTokens {
            let cx = coords[i].0 + 0.5
            let cy = coords[i].1 + 0.5
            let pe = positionalEncoding(x: cx, y: cy)
            let offset = i * embedDim

            for d in 0..<embedDim {
                sparse[offset + d] = pe[d]
            }

            let label = allLabels[i]
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
        let sPtr = sparseArray.dataPointer.bindMemory(to: Float32.self, capacity: sparse.count)
        sparse.withUnsafeBufferPointer { buf in
            sPtr.update(from: buf.baseAddress!, count: sparse.count)
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
