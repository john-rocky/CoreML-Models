import Foundation

// MARK: - ImageNet Labels (Condensed Demo Version)
// This file contains a subset of 20 common ImageNet-1K labels for demo purposes.
// For the full 1000-class label list, download from:
// https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
// and replace this array with all 1000 entries.

struct ImageNetLabels {
    /// Full ImageNet-1K has 1000 labels. This is a condensed demo set.
    /// Index positions correspond to the model output indices.
    /// Replace with the full list for production use.
    static let labels: [Int: String] = [
        0: "tench",
        1: "goldfish",
        2: "great white shark",
        7: "cock",
        14: "indigo bunting",
        65: "sea snake",
        99: "goose",
        207: "golden retriever",
        208: "Labrador retriever",
        231: "collie",
        235: "German shepherd",
        258: "Samoyed",
        259: "Pomeranian",
        281: "tabby cat",
        282: "tiger cat",
        285: "Egyptian cat",
        291: "lion",
        340: "zebra",
        386: "African elephant",
        409: "analog clock",
        417: "balloon",
        430: "basketball",
        446: "bikini",
        457: "bow tie",
        468: "cab",
        504: "coffee mug",
        508: "computer keyboard",
        531: "digital watch",
        537: "dog sled",
        539: "drum",
        549: "envelope",
        555: "fire truck",
        569: "fountain",
        604: "golf ball",
        609: "grand piano",
        620: "hamburger",
        659: "mixing bowl",
        671: "mountain bike",
        673: "mouse",
        701: "parachute",
        717: "pickup truck",
        737: "pot",
        755: "redbone",
        779: "school bus",
        812: "space shuttle",
        817: "sports car",
        834: "sunglasses",
        849: "tennis ball",
        852: "thatch",
        859: "toaster",
        876: "tray",
        880: "umbrella",
        892: "wall clock",
        907: "wine bottle",
        920: "traffic light",
        934: "hot dog",
        945: "bell pepper",
        947: "mushroom",
        950: "orange",
        954: "banana",
        963: "pizza",
        965: "burrito",
        967: "espresso",
        985: "daisy",
        988: "sunflower",
        999: "toilet tissue"
    ]

    /// Get the label for a given class index.
    /// Returns "class_{index}" for indices not in the condensed set.
    static func label(for index: Int) -> String {
        return labels[index] ?? "class_\(index)"
    }

    /// Get top-K predictions from a probability/score array.
    static func topK(scores: [Float], k: Int = 5) -> [(index: Int, label: String, score: Float)] {
        let indexed = scores.enumerated().map { (index: $0.offset, score: $0.element) }
        let sorted = indexed.sorted { $0.score > $1.score }
        let topK = sorted.prefix(k)
        return topK.map { (index: $0.index, label: label(for: $0.index), score: $0.score) }
    }
}
