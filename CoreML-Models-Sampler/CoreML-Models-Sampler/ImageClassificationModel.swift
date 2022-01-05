//
//  ImageClassificationModel.swift
//  CoreML-Models-Sampler
//
//  Created by 間嶋大輔 on 2022/01/06.
//

import Foundation
import AVKit
import CoreML

struct ImageClassificationModel {
    var label: String?
    var confidence: Float?
    var image: CIImage?
    
    var imageClassification: ImageClassification
    
    init(mlModel: MLModel) {
        imageClassification = ImageClassification(mlModel: mlModel)
    }
    
    mutating func inference() {
        guard let image = image else {
            print("nil image is pathed")
            return
        }
        let out = imageClassification.inference(ciImage: image)
        let label = out?.identifier
        let confidence = out?.confidence
        self.label = label
        self.confidence = confidence
    }
}
