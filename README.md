# CoreML-Models
Converted CoreML Models

**Image Classifier**
| Google Drive Link | Size | Original Project |
| ------------- | ------------- | ------------- |
| [Efficientnetb0](https://drive.google.com/file/d/1mJq8SMuDaCQHW77ui3fAfe5o3Qu2GKMi/view?usp=sharing) | 22.7 MB | [TensorFlowHub](https://tfhub.dev/tensorflow/efficientnet/b0/classification/1)  |

**GAN**

| Google Drive Link | Size | Original Project |
| ------------- | ------------- | ------------- |
| [UGATIT_selfie2anime](https://drive.google.com/file/d/1cOB1comTnd5I22htZ4_OJ7tFQuQEI2ne/view?usp=sharing) | 1.12GB | [taki0112/UGATIT](https://github.com/taki0112/UGATIT)  |
| [AnimeGANv2_Hayao](https://drive.google.com/file/d/1i_kwj41BxA1xZNu2B7yX2VVqNF66atMN/view?usp=sharing)　| 8.7MB | [TachibanaYoshino/AnimeGANv2](https://github.com/TachibanaYoshino/AnimeGANv2)|
| [AnimeGANv2_Paprika](https://drive.google.com/file/d/1wuoaVoI8-HOOQ1kUiZkVJ9GWnPbtPnWF/view?usp=sharing)　| 8.7MB | [TachibanaYoshino/AnimeGANv2](https://github.com/TachibanaYoshino/AnimeGANv2)|
| [WarpGAN Caricature](https://drive.google.com/file/d/1QjfA6DSOu7Za1zY-b9ajX6tsIWe9-C16/view?usp=sharing)　| 35.5MB | [seasonSH/WarpGAN](https://github.com/seasonSH/WarpGAN)|
<img width="256" alt="スクリーンショット 2020-05-19 11 09 03" src="https://user-images.githubusercontent.com/23278992/85667417-881d7d00-b6f8-11ea-8d1c-0d66b2b72de9.png">

<!-- <img width="256" alt="スクリーンショット 2020-06-22 4 10 54" src="https://user-images.githubusercontent.com/23278992/85667453-91a6e500-b6f8-11ea-84bf-22853b0995dc.png"> -->

How to use in a xcode project.

```swift:

import Vision
lazy var coreMLRequest:VNCoreMLRequest = {
   let model = try! VNCoreMLModel(for: modelname().model)
   let request = VNCoreMLRequest(model: model, completionHandler: self.coreMLCompletionHandler)
   return request
   }()

let handler = VNImageRequestHandler(ciImage: ciimage,options: [:])
   DispatchQueue.global(qos: .userInitiated).async {
   try? handler.perform([coreMLRequest])
}
```

For visualizing multiArray as image, Mr. Hollance’s “CoreML Helpers” are very convenient.
[CoreML Helpers](https://github.com/hollance/CoreMLHelpers)

[Converting from MultiArray to Image with CoreML Helpers.](https://medium.com/@rockyshikoku/converting-from-multiarray-to-image-with-coreml-helpers-59fdc34d80d8)

```swift:
func coreMLCompletionHandler（request：VNRequest？、error：Error？）{
   let = coreMLRequest.results？.first as！VNCoreMLFeatureValueObservation
   let multiArray = result.featureValue.multiArrayValue
   let cgimage = multiArray？.cgImage（min：-1、max：1、channel：nil）
```


Apps made by Core ML models.
[AnimateU](https://apps.apple.com/us/app/animateu/id1513582287)
