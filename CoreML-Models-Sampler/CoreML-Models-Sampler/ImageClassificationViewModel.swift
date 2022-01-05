//
//  ImageClassificationViewModel.swift
//  CoreML-Models
//
//  Created by Daisuke Majima on 2022/01/05.
//

import Foundation
import CoreML
import UIKit

class ImageClassificationViewModel: ObservableObject {
    
    @Published var imageClassificationModel: ImageClassificationModel
    let ciContext = CIContext()
    
    init(modelName: String) {
        guard let path = Bundle.main.path(forResource: modelName, ofType: "mlmodel", inDirectory: "models") else { fatalError("Model initializing failed.") }
        let url = URL(fileURLWithPath: path)
        guard let compiledURL = try? MLModel.compileModel(at: url),
              let  mlmodel = try? MLModel(contentsOf: compiledURL) else { fatalError("Model initializing failed.") }
        self.imageClassificationModel = ImageClassificationModel(mlModel: mlmodel)
    }
    
    var label:String {
        guard let label = imageClassificationModel.label else { return "Label" }
        return label
    }
        
    var confidence: String {
        guard let confidence = imageClassificationModel.confidence else { return "confidence:" }
        let integerConfidence = floor(confidence * 100)
        let confidenceString = "confidence: \(integerConfidence) %"
        return confidenceString
    }
    
    func inference(uiImage: UIImage) {
        let correctOrientationImage = getCorrectOrientationUIImage(uiImage: uiImage)
        guard let ciImage = CIImage(image: correctOrientationImage) else {print("Image failed"); return }
        imageClassificationModel.image = ciImage
        imageClassificationModel.inference()
    }
    
    func getCorrectOrientationUIImage(uiImage:UIImage) -> UIImage {
            var newImage = UIImage()
            switch uiImage.imageOrientation.rawValue {
            case 1:
                guard let orientedCIImage = CIImage(image: uiImage)?.oriented(CGImagePropertyOrientation.down),
                      let cgImage = ciContext.createCGImage(orientedCIImage, from: orientedCIImage.extent) else { return uiImage}
                
                newImage = UIImage(cgImage: cgImage)
            case 3:
                guard let orientedCIImage = CIImage(image: uiImage)?.oriented(CGImagePropertyOrientation.right),
                        let cgImage = ciContext.createCGImage(orientedCIImage, from: orientedCIImage.extent) else { return uiImage}
                newImage = UIImage(cgImage: cgImage)
            default:
                newImage = uiImage
            }
        return newImage
    }
}
