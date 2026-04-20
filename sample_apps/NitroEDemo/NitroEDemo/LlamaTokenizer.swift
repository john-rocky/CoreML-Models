import Foundation

/// Minimal Llama 3 BPE tokenizer for the Nitro-E text encoder path.
///
/// Reads `Llama3Vocab.json` and `Llama3Merges.txt` from the app bundle, both
/// exported from Hugging Face's `meta-llama/Llama-3.2-1B` tokenizer via
/// `tokenizer.json`. The implementation is the byte-level BPE described in
/// the GPT-2 paper, which Llama 3 inherits. Enough for prompt encoding with
/// `max_length=128`, truncation + padding — it is NOT a full Tokenizers port
/// (no added-token handling beyond `<|begin_of_text|>`).
final class LlamaTokenizer {

    enum TokenizerError: Error { case resourceMissing(String), malformed(String) }

    private let vocab: [String: Int32]
    private let bpeRanks: [Pair: Int]
    private let byteEncoder: [UInt8: Character]
    private let bosTokenID: Int32
    private let eosTokenID: Int32
    private let padTokenID: Int32
    private let pattern: NSRegularExpression

    struct Pair: Hashable { let a: String; let b: String }

    init() throws {
        // 1) vocab.json  {token: id}
        guard let vocabURL = Bundle.main.url(forResource: "Llama3Vocab", withExtension: "json"),
              let data = try? Data(contentsOf: vocabURL),
              let dict = try JSONSerialization.jsonObject(with: data) as? [String: Int] else {
            throw TokenizerError.resourceMissing("Llama3Vocab.json")
        }
        self.vocab = dict.mapValues { Int32($0) }

        // 2) merges.txt  (pair per line, ranked)
        guard let mergesURL = Bundle.main.url(forResource: "Llama3Merges", withExtension: "txt"),
              let text = try? String(contentsOf: mergesURL, encoding: .utf8) else {
            throw TokenizerError.resourceMissing("Llama3Merges.txt")
        }
        var ranks: [Pair: Int] = [:]
        for (rank, line) in text.split(separator: "\n").enumerated() {
            let parts = line.split(separator: " ", maxSplits: 1).map(String.init)
            if parts.count == 2 { ranks[Pair(a: parts[0], b: parts[1])] = rank }
        }
        self.bpeRanks = ranks

        // 3) GPT-2 byte ↔ printable-unicode mapping
        self.byteEncoder = Self.makeByteEncoder()

        // 4) Special tokens — Llama 3 uses `<|begin_of_text|>` (128000) and
        //    `<|end_of_text|>` (128001). We use <|end_of_text|> as pad.
        self.bosTokenID = dict["<|begin_of_text|>"].map(Int32.init) ?? 128000
        self.eosTokenID = dict["<|end_of_text|>"].map(Int32.init) ?? 128001
        self.padTokenID = self.eosTokenID

        // 5) Pre-tokenization regex (same as tiktoken `cl100k_base` / Llama 3)
        let patt = #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
        self.pattern = try NSRegularExpression(pattern: patt, options: [])
    }

    /// Encode to fixed-length (maxLength) ids + mask. Adds `<|begin_of_text|>`
    /// at position 0 and pads with `<|end_of_text|>`.
    func encode(text: String, maxLength: Int = 128) throws -> (ids: [Int32], mask: [Int32]) {
        var ids: [Int32] = [bosTokenID]
        let ns = text as NSString
        let matches = pattern.matches(in: text, range: NSRange(location: 0, length: ns.length))
        for m in matches {
            let piece = ns.substring(with: m.range)
            let bytes = Array(piece.utf8)
            // Map bytes to visible unicode chars, then BPE-merge and look up.
            var s = ""
            for b in bytes { s.append(byteEncoder[b] ?? Character(Unicode.Scalar(b))) }
            for tok in bpe(of: s) {
                if let id = vocab[tok] { ids.append(id) }
            }
            if ids.count >= maxLength { break }
        }
        // Truncate / pad to fixed length
        if ids.count >= maxLength {
            ids = Array(ids.prefix(maxLength))
            ids[maxLength - 1] = eosTokenID
        } else {
            let real = ids.count
            let pad = Array(repeating: padTokenID, count: maxLength - real)
            ids.append(contentsOf: pad)
        }
        var mask = [Int32](repeating: 0, count: maxLength)
        for i in 0..<maxLength where ids[i] != padTokenID || i == 0 { mask[i] = 1 }
        return (ids, mask)
    }

    // MARK: - Byte-level BPE internals

    private static func makeByteEncoder() -> [UInt8: Character] {
        // Build the exact printable-byte set GPT-2 uses (so our "bytes → string"
        // is reversible and matches the vocab's merge keys).
        var bs: [UInt8] = []
        for b in 33...126 { bs.append(UInt8(b)) }   // !..~
        for b in 161...172 { bs.append(UInt8(b)) }  // ¡..¬
        for b in 174...255 { bs.append(UInt8(b)) }  // ®..ÿ
        var cs = bs.map { Int($0) }
        var n = 0
        for b in 0..<256 {
            if !bs.contains(UInt8(b)) {
                bs.append(UInt8(b))
                cs.append(256 + n)
                n += 1
            }
        }
        var map: [UInt8: Character] = [:]
        for i in 0..<bs.count {
            if let scalar = Unicode.Scalar(cs[i]) { map[bs[i]] = Character(scalar) }
        }
        return map
    }

    private func bpe(of word: String) -> [String] {
        if word.isEmpty { return [] }
        var tokens = word.map { String($0) }
        while tokens.count > 1 {
            // Find the lowest-rank adjacent pair
            var bestRank = Int.max
            var bestIdx = -1
            for i in 0..<(tokens.count - 1) {
                let pair = Pair(a: tokens[i], b: tokens[i + 1])
                if let r = bpeRanks[pair], r < bestRank { bestRank = r; bestIdx = i }
            }
            if bestIdx < 0 { break }
            let merged = tokens[bestIdx] + tokens[bestIdx + 1]
            tokens.replaceSubrange(bestIdx...(bestIdx + 1), with: [merged])
        }
        return tokens
    }
}
