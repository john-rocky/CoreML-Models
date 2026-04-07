import Foundation
import NaturalLanguage

/// Minimal English G2P for Kokoro TTS.
///
/// Strategy:
///   1. Tokenize input text into words and punctuation
///   2. Spell out numbers via NumberFormatter
///   3. Lex lookup in us_gold then us_silver. Heteronyms: NLTagger POS → POS-keyed entry, fallback DEFAULT
///   4. OOV: simple letter-by-letter spell-out
///   5. Concatenate phonemes with spaces and trailing punctuation
///
/// This is a lightweight reimplementation of the lexicon path of `misaki/en.py`,
/// without the BART fallback (which would require MLX). Coverage is good for
/// common English text; rare/technical words may degrade gracefully via spell-out.
final class EnglishG2P {
    private var goldLexicon: [String: Any] = [:]
    private var silverLexicon: [String: Any] = [:]
    private let numberFormatter: NumberFormatter

    /// Single-letter to default phoneme mapping (rule-based fallback for OOV).
    /// Values approximate the most common English pronunciation per letter.
    private static let letterToPhoneme: [Character: String] = [
        "a": "æ", "b": "b", "c": "k", "d": "d", "e": "ɛ",
        "f": "f", "g": "ɡ", "h": "h", "i": "ɪ", "j": "ʤ",
        "k": "k", "l": "l", "m": "m", "n": "n", "o": "ɑ",
        "p": "p", "q": "k", "r": "ɹ", "s": "s", "t": "t",
        "u": "ʌ", "v": "v", "w": "w", "x": "ks", "y": "j",
        "z": "z",
    ]

    /// Common English digraphs handled before single-letter fallback.
    private static let digraphs: [String: String] = [
        "ph": "f", "th": "θ", "ch": "ʧ", "sh": "ʃ", "wh": "w",
        "ng": "ŋ", "ck": "k", "qu": "kw",
    ]

    /// Letters used to spell-out individual characters (e.g., for unknown acronyms).
    /// Values from misaki's US English alphabet pronunciation.
    private static let letterSpelled: [Character: String] = [
        "a": "ˈA", "b": "bˈi", "c": "sˈi", "d": "dˈi", "e": "ˈi",
        "f": "ˈɛf", "g": "ʤˈi", "h": "ˈAʧ", "i": "ˈI", "j": "ʤˈA",
        "k": "kˈA", "l": "ˈɛl", "m": "ˈɛm", "n": "ˈɛn", "o": "ˈO",
        "p": "pˈi", "q": "kjˈu", "r": "ˈɑɹ", "s": "ˈɛs", "t": "tˈi",
        "u": "jˈu", "v": "vˈi", "w": "dˈʌbᵊljˌu", "x": "ˈɛks",
        "y": "wˈI", "z": "zˈi",
    ]

    init() {
        numberFormatter = NumberFormatter()
        numberFormatter.numberStyle = .spellOut
        numberFormatter.locale = Locale(identifier: "en_US")
        loadLexicons()
    }

    private func loadLexicons() {
        if let url = Bundle.main.url(forResource: "us_gold", withExtension: "json"),
           let data = try? Data(contentsOf: url),
           let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            goldLexicon = dict
        }
        if let url = Bundle.main.url(forResource: "us_silver", withExtension: "json"),
           let data = try? Data(contentsOf: url),
           let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            silverLexicon = dict
        }
    }

    /// Convert English text to a phoneme string compatible with Kokoro.
    func phonemize(_ text: String) -> String {
        let normalized = expandNumbers(text)
        let tokens = tokenize(normalized)
        var posTagger: NLTagger?
        // POS tag the cleaned word stream (for heteronym disambiguation)
        if tokens.contains(where: { isWord($0) }) {
            posTagger = NLTagger(tagSchemes: [.lexicalClass])
            posTagger?.string = normalized
        }

        var output = ""
        for token in tokens {
            if isWord(token) {
                let pos = posTag(for: token, in: normalized, tagger: posTagger)
                let phon = lookup(word: token, pos: pos)
                if !output.isEmpty && !output.hasSuffix(" ") {
                    output += " "
                }
                output += phon
            } else {
                // Punctuation passes through (Kokoro tokenizer accepts it)
                output += token
            }
        }
        return output
    }

    // MARK: - Tokenization

    /// Split text into a sequence of words and standalone punctuation tokens.
    private func tokenize(_ text: String) -> [String] {
        var result: [String] = []
        var current = ""
        for ch in text {
            if ch.isLetter || ch == "'" {
                current.append(ch)
            } else {
                if !current.isEmpty {
                    result.append(current)
                    current = ""
                }
                if ch.isWhitespace {
                    continue
                }
                result.append(String(ch))
            }
        }
        if !current.isEmpty { result.append(current) }
        return result
    }

    private func isWord(_ token: String) -> Bool {
        return token.first?.isLetter == true
    }

    // MARK: - Number expansion

    /// Replace standalone integer/decimal sequences with their spell-out form.
    private func expandNumbers(_ text: String) -> String {
        let pattern = #"\b\d+(?:\.\d+)?\b"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return text }
        let range = NSRange(text.startIndex..., in: text)
        let matches = regex.matches(in: text, range: range).reversed()
        var result = text
        for m in matches {
            guard let r = Range(m.range, in: result) else { continue }
            let num = String(result[r])
            if let value = Double(num),
               let spelled = numberFormatter.string(from: NSNumber(value: value)) {
                result.replaceSubrange(r, with: spelled)
            }
        }
        return result
    }

    // MARK: - POS tagging

    private func posTag(for word: String, in text: String, tagger: NLTagger?) -> String? {
        guard let tagger else { return nil }
        guard let range = text.range(of: word) else { return nil }
        let (tag, _) = tagger.tag(at: range.lowerBound, unit: .word, scheme: .lexicalClass)
        return tag?.rawValue
    }

    /// Map Apple's NL lexical class to misaki/spaCy-style POS tag (subset).
    private func mapPOS(_ nlTag: String?) -> String {
        guard let nlTag else { return "DEFAULT" }
        switch nlTag {
        case "Verb":        return "VBP"  // approximate; misaki uses many verb tags
        case "Noun":        return "NN"
        case "Adjective":   return "ADJ"
        case "Adverb":      return "ADV"
        case "Pronoun":     return "PRON"
        case "Preposition": return "ADP"
        case "Conjunction": return "CONJ"
        case "Determiner":  return "DET"
        case "Particle":    return "PRT"
        default:            return "DEFAULT"
        }
    }

    // MARK: - Lexicon lookup

    private func lookup(word: String, pos: String?) -> String {
        // Possessive 's: split base + 's. e.g., "Apple's" → lookup("Apple") + lookup("'s")
        if word.hasSuffix("'s") || word.hasSuffix("’s") {
            let base = String(word.dropLast(2))
            if !base.isEmpty {
                let basePhon = lookupBase(word: base, pos: pos)
                let possPhon = lookupBase(word: "'s", pos: nil)
                return basePhon + possPhon
            }
        }
        return lookupBase(word: word, pos: pos)
    }

    private func lookupBase(word: String, pos: String?) -> String {
        let key = word
        let lowerKey = word.lowercased()
        let mappedPOS = mapPOS(pos)

        // Try exact case first, then lowercase, in gold then silver.
        for lexicon in [goldLexicon, silverLexicon] {
            for tryKey in [key, lowerKey] {
                if let entry = lexicon[tryKey] {
                    if let str = entry as? String {
                        return str
                    }
                    if let dict = entry as? [String: String] {
                        if let posSpecific = dict[mappedPOS] { return posSpecific }
                        if let def = dict["DEFAULT"] { return def }
                        if let any = dict.values.first { return any }
                    }
                }
            }
        }

        // OOV fallback: rule-based grapheme→phoneme.
        // Acronyms (all caps, ≤4 letters) get spelled out letter by letter.
        if word.count <= 4 && word == word.uppercased() && word.allSatisfy({ $0.isLetter }) {
            return word.lowercased().compactMap { Self.letterSpelled[$0] }.joined(separator: " ")
        }
        return graphemeToPhoneme(word.lowercased())
    }

    /// Naive grapheme-to-phoneme: applies common digraphs first, then per-letter mapping.
    /// Used only for words not in the lexicon. Quality is rough but understandable.
    private func graphemeToPhoneme(_ word: String) -> String {
        var out = ""
        let chars = Array(word)
        var i = 0
        while i < chars.count {
            let ch = chars[i]
            if i + 1 < chars.count {
                let pair = String(chars[i]) + String(chars[i+1])
                if let mapped = Self.digraphs[pair] {
                    out += mapped
                    i += 2
                    continue
                }
            }
            if let phon = Self.letterToPhoneme[ch] {
                out += phon
            }
            i += 1
        }
        return out
    }
}
