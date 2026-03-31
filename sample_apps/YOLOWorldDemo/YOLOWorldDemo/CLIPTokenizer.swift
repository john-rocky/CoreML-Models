import Foundation
import CoreML

/// CLIP BPE tokenizer implemented in pure Swift.
///
/// Loads vocabulary and merge rules from a JSON file exported by
/// `convert_models.py`. Produces token ID sequences compatible
/// with the CLIP text encoder CoreML model.
final class CLIPTokenizer {

    // MARK: - Properties

    private let encoder: [String: Int]
    private let decoder: [Int: String]
    private let bpeMerges: [(String, String)]
    private let bpeRanks: [String: Int]
    private let bosToken: Int
    private let eosToken: Int
    let contextLength: Int

    private var cache: [String: [Int]] = [:]

    // MARK: - Initialization

    init(vocabularyURL: URL) throws {
        let data = try Data(contentsOf: vocabularyURL)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NSError(domain: "CLIPTokenizer", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid vocabulary JSON"])
        }

        guard let encoderDict = json["encoder"] as? [String: Int] else {
            throw NSError(domain: "CLIPTokenizer", code: 2, userInfo: [NSLocalizedDescriptionKey: "Missing 'encoder'"])
        }

        guard let mergesList = json["merges"] as? [String] else {
            throw NSError(domain: "CLIPTokenizer", code: 3, userInfo: [NSLocalizedDescriptionKey: "Missing 'merges'"])
        }

        self.encoder = encoderDict
        self.decoder = Dictionary(uniqueKeysWithValues: encoderDict.map { ($1, $0) })

        self.bpeMerges = mergesList.compactMap { line -> (String, String)? in
            let parts = line.split(separator: " ", maxSplits: 1)
            guard parts.count == 2 else { return nil }
            return (String(parts[0]), String(parts[1]))
        }

        self.bpeRanks = Dictionary(uniqueKeysWithValues:
            bpeMerges.enumerated().map { ($0.element.0 + " " + $0.element.1, $0.offset) }
        )

        let bosStr = json["bos_token"] as? String ?? "<|startoftext|>"
        let eosStr = json["eos_token"] as? String ?? "<|endoftext|>"
        self.bosToken = encoderDict[bosStr] ?? 49406
        self.eosToken = encoderDict[eosStr] ?? 49407
        self.contextLength = json["context_length"] as? Int ?? 77
    }

    // MARK: - Tokenization

    func tokenize(_ text: String) -> [Int] {
        if let cached = cache[text] { return cached }

        let cleaned = text.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        let words = cleaned.split(separator: " ").map { String($0) }

        var tokens: [Int] = [bosToken]

        for word in words {
            let encoded = byteEncode(word + "</w>")
            let bpeTokens = bpe(encoded)
            for token in bpeTokens {
                if let id = encoder[token] {
                    tokens.append(id)
                }
            }
        }

        tokens.append(eosToken)

        if tokens.count > contextLength {
            tokens = Array(tokens.prefix(contextLength - 1)) + [eosToken]
        }

        while tokens.count < contextLength {
            tokens.append(0)
        }

        cache[text] = tokens
        return tokens
    }

    func clearCache() {
        cache.removeAll()
    }

    // MARK: - BPE

    private func byteEncode(_ text: String) -> [String] {
        text.utf8.map { byteToUnicode($0) }
    }

    private func byteToUnicode(_ byte: UInt8) -> String {
        let b = Int(byte)
        if (33...126).contains(b) || (161...172).contains(b) || (174...255).contains(b) {
            return String(Unicode.Scalar(b)!)
        }
        return String(Unicode.Scalar(256 + b)!)
    }

    private func bpe(_ tokens: [String]) -> [String] {
        if tokens.count <= 1 { return tokens }

        var word = tokens

        while true {
            var bestPair: (Int, String, String)?
            var bestRank = Int.max

            for i in 0..<(word.count - 1) {
                let pair = word[i] + " " + word[i + 1]
                if let rank = bpeRanks[pair], rank < bestRank {
                    bestRank = rank
                    bestPair = (i, word[i], word[i + 1])
                }
            }

            guard let (_, first, second) = bestPair else { break }

            var newWord: [String] = []
            var i = 0
            while i < word.count {
                if i < word.count - 1 && word[i] == first && word[i + 1] == second {
                    newWord.append(first + second)
                    i += 2
                } else {
                    newWord.append(word[i])
                    i += 1
                }
            }
            word = newWord

            if word.count == 1 { break }
        }

        return word
    }
}
