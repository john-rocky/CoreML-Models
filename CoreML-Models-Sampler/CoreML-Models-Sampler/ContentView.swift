//
//  ContentView.swift
//  CoreML-Models-Sampler
//
//  Created by 間嶋大輔 on 2022/01/05.
//

import SwiftUI

struct ContentView: View {
    var body: some View {
        NavigationView {
            VStack {
                Text("CoreML-Models")
                    .font(.title)
                List {
                    NavigationLink(
                        destination: ImageClassificationView(modelName: "efficientnet"),
                        label: {
                            Text("EfficientNet")
                                .font(.headline)
                        })
                }
            }.navigationBarTitle("")
            .navigationBarHidden(true)
            .navigationViewStyle(StackNavigationViewStyle())
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
