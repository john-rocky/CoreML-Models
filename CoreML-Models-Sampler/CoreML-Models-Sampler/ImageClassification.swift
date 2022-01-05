//
//  ImageClassification.swift
//  CoreML-Models
//
//  Created by Daisuke Majima on 2022/01/05.
//

import Foundation
import Vision
import AVKit

struct ImageClassification {
    var resuest: VNCoreMLRequest?

    
    init(mlModel: MLModel) {
        do {
            let coreMLRequest = VNCoreMLRequest(model: try VNCoreMLModel(for: mlModel))
            coreMLRequest.imageCropAndScaleOption = .scaleFill
            self.resuest = coreMLRequest
        } catch let error {
            print(error)
        }
    }
    
    func inference(ciImage: CIImage) -> VNClassificationObservation? {
        guard let resuest = resuest else { print("Model initializing failed."); return nil }
        let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])
        do {
            try handler.perform([resuest])
            guard let result = resuest.results?.first as? VNClassificationObservation else { print("No result"); return nil }
            return result
        } catch let error {
            print(error)
            return nil
        }
    }
}
