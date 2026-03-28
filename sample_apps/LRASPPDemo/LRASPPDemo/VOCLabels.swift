import Foundation
import SwiftUI

// MARK: - Pascal VOC Segmentation Labels
// 21 classes used by LRASPP_MobileNetV3 semantic segmentation model

struct VOCLabels {
    struct SegmentationClass {
        let index: Int
        let name: String
        let color: Color
        let rgbColor: (UInt8, UInt8, UInt8)
    }

    static let classes: [SegmentationClass] = [
        SegmentationClass(index: 0,  name: "Background",    color: .black,              rgbColor: (0, 0, 0)),
        SegmentationClass(index: 1,  name: "Aeroplane",     color: .red,                rgbColor: (128, 0, 0)),
        SegmentationClass(index: 2,  name: "Bicycle",       color: .green,              rgbColor: (0, 128, 0)),
        SegmentationClass(index: 3,  name: "Bird",          color: .blue,               rgbColor: (128, 128, 0)),
        SegmentationClass(index: 4,  name: "Boat",          color: .yellow,             rgbColor: (0, 0, 128)),
        SegmentationClass(index: 5,  name: "Bottle",        color: .purple,             rgbColor: (128, 0, 128)),
        SegmentationClass(index: 6,  name: "Bus",           color: .orange,             rgbColor: (0, 128, 128)),
        SegmentationClass(index: 7,  name: "Car",           color: .cyan,               rgbColor: (128, 128, 128)),
        SegmentationClass(index: 8,  name: "Cat",           color: .mint,               rgbColor: (64, 0, 0)),
        SegmentationClass(index: 9,  name: "Chair",         color: .teal,               rgbColor: (192, 0, 0)),
        SegmentationClass(index: 10, name: "Cow",           color: .indigo,             rgbColor: (64, 128, 0)),
        SegmentationClass(index: 11, name: "Dining Table",  color: .brown,              rgbColor: (192, 128, 0)),
        SegmentationClass(index: 12, name: "Dog",           color: Color(red: 1.0, green: 0.4, blue: 0.4), rgbColor: (64, 0, 128)),
        SegmentationClass(index: 13, name: "Horse",         color: Color(red: 0.4, green: 1.0, blue: 0.4), rgbColor: (192, 0, 128)),
        SegmentationClass(index: 14, name: "Motorbike",     color: Color(red: 0.4, green: 0.4, blue: 1.0), rgbColor: (64, 128, 128)),
        SegmentationClass(index: 15, name: "Person",        color: Color(red: 1.0, green: 0.0, blue: 0.5), rgbColor: (192, 128, 128)),
        SegmentationClass(index: 16, name: "Potted Plant",  color: Color(red: 0.5, green: 1.0, blue: 0.0), rgbColor: (0, 64, 0)),
        SegmentationClass(index: 17, name: "Sheep",         color: Color(red: 0.0, green: 0.5, blue: 1.0), rgbColor: (128, 64, 0)),
        SegmentationClass(index: 18, name: "Sofa",          color: Color(red: 0.8, green: 0.8, blue: 0.0), rgbColor: (0, 192, 0)),
        SegmentationClass(index: 19, name: "Train",         color: Color(red: 0.0, green: 0.8, blue: 0.8), rgbColor: (128, 192, 0)),
        SegmentationClass(index: 20, name: "TV/Monitor",    color: Color(red: 0.8, green: 0.0, blue: 0.8), rgbColor: (0, 64, 128))
    ]

    /// Get name for a class index
    static func name(for index: Int) -> String {
        guard index >= 0 && index < classes.count else { return "Unknown" }
        return classes[index].name
    }

    /// Get color for a class index
    static func color(for index: Int) -> Color {
        guard index >= 0 && index < classes.count else { return .gray }
        return classes[index].color
    }

    /// Get RGB color tuple for a class index
    static func rgbColor(for index: Int) -> (UInt8, UInt8, UInt8) {
        guard index >= 0 && index < classes.count else { return (128, 128, 128) }
        return classes[index].rgbColor
    }
}
