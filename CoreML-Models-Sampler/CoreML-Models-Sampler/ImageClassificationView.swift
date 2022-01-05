//
//  ImageClassificationView.swift
//  CoreML-Models
//
//  Created by 間嶋大輔 on 2022/01/05.
//

import SwiftUI
import AVKit
import PhotosUI
import os

struct ImageClassificationView: View {
    @StateObject var viewModel:ImageClassificationViewModel
    @State private var images:[UIImage] = []
    @State private var showPHPicker:Bool = false
    static var config: PHPickerConfiguration {
        var config = PHPickerConfiguration()
        config.filter = .images
        return config
    }
    let logger = Logger(subsystem: "com.smalldesksoftware.PHPickerSample", category: "PHPickerSample")
    
    init(modelName:String){
        let viewModel = ImageClassificationViewModel(modelName: modelName)
        _viewModel = StateObject(wrappedValue: viewModel)
    }
    
    var body: some View {
        VStack {
            Image(uiImage: images.first ?? UIImage(systemName: "photo") ?? UIImage())
                .resizable()
                .scaledToFit()
            Text(viewModel.label)
            Text(viewModel.confidence)
            HStack {
                Spacer()
                Button(action: {
                    showPHPicker.toggle()
                }, label: {
                    Image(systemName: "photo")
                        .font(.largeTitle)
                })
            }
            .padding()
        }
        .sheet(isPresented: $showPHPicker) {
            SwiftUIPHPicker(configuration: ImageClassificationView.config) { results in
                for result in results {
                    let itemProvider = result.itemProvider
                    if itemProvider.canLoadObject(ofClass: UIImage.self) {
                        itemProvider.loadObject(ofClass: UIImage.self) { image, error in
                            if let image = image as? UIImage {
                                DispatchQueue.main.async {
                                    self.images = []
                                    self.images.append(image)
                                    self.viewModel.inference(uiImage: image)
                                }
                            }
                            if let error = error {
                                logger.error("\(error.localizedDescription)")
                            }
                        }
                    }
                }
            }
        }
    }
}



struct ImageClassificationView_Previews: PreviewProvider {
    static var previews: some View {
        ImageClassificationView(modelName: "")
    }
}
