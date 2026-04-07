import Foundation

/// Japanese G2P for Kokoro TTS, using Apple's built-in CFStringTokenizer
/// (kanji → romaji) + Latin-Hiragana ICU transform, with no external
/// dependencies (no MeCab, no IPADic).
///
/// Strategy:
///   1. CFStringTokenizer with ja_JP locale + LatinTranscription attribute
///      → tokens with romaji readings (handles kanji disambiguation via CoreText)
///   2. Apple "Latin-Hiragana" transform → per-token hiragana
///   3. Particle correction (は/へ as particles → わ/え equivalents)
///   4. Apply HEPBURN IPA table + context rules (ported from misaki/cutlet.py)
///
/// Output matches the cutlet branch of misaki's JAG2P for typical input.
final class JapaneseG2P {

    /// Hiragana → IPA phoneme string. Ported verbatim from misaki/cutlet.py
    /// (164 entries, single + digraph hiragana). Kokoro's Japanese voices
    /// were trained on this exact phoneme set.
    private static let hepburn: [String: String] = {
        var m: [String: String] = [
            // Single hiragana (84 entries)
            "ぁ":"a","あ":"a","ぃ":"i","い":"i","ぅ":"ɯ","う":"ɯ","ぇ":"e","え":"e",
            "ぉ":"o","お":"o","か":"ka","が":"ɡa","き":"kʲi","ぎ":"ɡʲi","く":"kɯ",
            "ぐ":"ɡɯ","け":"ke","げ":"ɡe","こ":"ko","ご":"ɡo","さ":"sa","ざ":"ʣa",
            "し":"ɕi","じ":"ʥi","す":"sɨ","ず":"zɨ","せ":"se","ぜ":"ʣe","そ":"so",
            "ぞ":"ʣo","た":"ta","だ":"da","ち":"ʨi","ぢ":"ʥi","つ":"ʦɨ","づ":"zɨ",
            "て":"te","で":"de","と":"to","ど":"do","な":"na","に":"ɲi","ぬ":"nɯ",
            "ね":"ne","の":"no","は":"ha","ば":"ba","ぱ":"pa","ひ":"çi","び":"bʲi",
            "ぴ":"pʲi","ふ":"ɸɯ","ぶ":"bɯ","ぷ":"pɯ","へ":"he","べ":"be","ぺ":"pe",
            "ほ":"ho","ぼ":"bo","ぽ":"po","ま":"ma","み":"mʲi","む":"mɯ","め":"me",
            "も":"mo","ゃ":"ja","や":"ja","ゅ":"jɯ","ゆ":"jɯ","ょ":"jo","よ":"jo",
            "ら":"ɾa","り":"ɾʲi","る":"ɾɯ","れ":"ɾe","ろ":"ɾo","ゎ":"βa","わ":"βa",
            "ゐ":"i","ゑ":"e","を":"o","ゔ":"vɯ","ゕ":"ka","ゖ":"ke",
            // Digraphs (80 entries)
            "いぇ":"je","うぃ":"βi","うぇ":"βe","うぉ":"βo",
            "きぇ":"kʲe","きゃ":"kʲa","きゅ":"kʲɨ","きょ":"kʲo",
            "ぎゃ":"ɡʲa","ぎゅ":"ɡʲɨ","ぎょ":"ɡʲo",
            "くぁ":"kᵝa","くぃ":"kᵝi","くぇ":"kᵝe","くぉ":"kᵝo",
            "ぐぁ":"ɡᵝa","ぐぃ":"ɡᵝi","ぐぇ":"ɡᵝe","ぐぉ":"ɡᵝo",
            "しぇ":"ɕe","しゃ":"ɕa","しゅ":"ɕɨ","しょ":"ɕo",
            "じぇ":"ʥe","じゃ":"ʥa","じゅ":"ʥɨ","じょ":"ʥo",
            "ちぇ":"ʨe","ちゃ":"ʨa","ちゅ":"ʨɨ","ちょ":"ʨo",
            "ぢゃ":"ʥa","ぢゅ":"ʥɨ","ぢょ":"ʥo",
            "つぁ":"ʦa","つぃ":"ʦʲi","つぇ":"ʦe","つぉ":"ʦo",
            "てぃ":"tʲi","てゅ":"tʲɨ","でぃ":"dʲi","でゅ":"dʲɨ",
            "とぅ":"tɯ","どぅ":"dɯ",
            "にぇ":"ɲe","にゃ":"ɲa","にゅ":"ɲɨ","にょ":"ɲo",
            "ひぇ":"çe","ひゃ":"ça","ひゅ":"çɨ","ひょ":"ço",
            "びゃ":"bʲa","びゅ":"bʲɨ","びょ":"bʲo",
            "ぴゃ":"pʲa","ぴゅ":"pʲɨ","ぴょ":"pʲo",
            "ふぁ":"ɸa","ふぃ":"ɸʲi","ふぇ":"ɸe","ふぉ":"ɸo","ふゅ":"ɸʲɨ","ふょ":"ɸʲo",
            "みゃ":"mʲa","みゅ":"mʲɨ","みょ":"mʲo",
            "りゃ":"ɾʲa","りゅ":"ɾʲɨ","りょ":"ɾʲo",
            "ゔぁ":"va","ゔぃ":"vʲi","ゔぇ":"ve","ゔぉ":"vo","ゔゅ":"bʲɨ","ゔょ":"bʲo",
        ]
        // Punctuation passthrough/normalization
        let punct: [String: String] = [
            "。":".","、":",","？":"?","！":"!","「":"\u{201C}","」":"\u{201D}",
            "『":"\u{201C}","』":"\u{201D}","：":":","；":";","（":"(","）":")",
            "《":"(","》":")","【":"[","】":"]","・":" ","，":",","～":"—","〜":"—","—":"—",
        ]
        for (k, v) in punct { m[k] = v }
        return m
    }()

    private static let suteganaSet: Set<Character> = Set("ゃゅょぁぃぅぇぉ")

    /// Compound greetings whose final は is a particle (read "wa" not "ha"),
    /// but Apple's tokenizer treats them as single tokens.
    private static let particleHaCompounds: Set<String> = [
        "こんにちは", "こんばんは",
    ]

    init() {}

    /// Convert Japanese text to Kokoro-compatible IPA phoneme string.
    func phonemize(_ text: String) -> String {
        let tokens = tokenizeWithReadings(text)
        var pieces: [String] = []
        for (surface, hiraganaReading, isParticle) in tokens {
            let hira = hiraganaReading

            // Standalone particles: override pronunciation
            if isParticle {
                if surface == "は" { pieces.append("βa"); continue }
                if surface == "へ" { pieces.append("e"); continue }
                if surface == "を" { pieces.append("o"); continue }
            }

            // Special compound greetings: replace final は → わ before lookup
            var workHira = hira
            if Self.particleHaCompounds.contains(surface), workHira.hasSuffix("は") {
                workHira = String(workHira.dropLast()) + "わ"
            }

            let phon = romaji(hiragana: workHira)
            pieces.append(normalizeLongVowels(phon))
        }
        return pieces.joined(separator: " ")
    }

    /// Apple's CFStringTokenizer transcribes long vowels as "ou"/"uu"/"ei"
    /// (e.g., 今日 → "kyou"), which after Latin-Hiragana becomes きょう →
    /// kʲoɯ. Real Japanese pronunciation uses long vowels (kʲoː). MeCab+UniDic
    /// already returns "ー" for these cases, but Apple's transcription doesn't.
    /// We approximate by collapsing common long-vowel sequences.
    private func normalizeLongVowels(_ s: String) -> String {
        // Common Japanese long vowels in IPA after the cutlet HEPBURN table:
        //   "oɯ" (おう/おー) → "oː"
        //   "ɯɯ" (うう/うー) → "ɯː"
        //   "eː" stays
        //   "ei" can be a long vowel but is also a real diphthong → leave as-is
        var out = s
        out = out.replacingOccurrences(of: "oɯ", with: "oː")
        out = out.replacingOccurrences(of: "ɯɯ", with: "ɯː")
        return out
    }

    // MARK: - Tokenization

    /// Returns (surface, hiraganaReading, isParticleSingleton).
    /// Uses CFStringTokenizer to get romaji, then Latin-Hiragana to convert.
    private func tokenizeWithReadings(_ text: String) -> [(String, String, Bool)] {
        var result: [(String, String, Bool)] = []
        let cf = text as CFString
        let tokenizer = CFStringTokenizerCreate(
            kCFAllocatorDefault, cf,
            CFRangeMake(0, CFStringGetLength(cf)),
            kCFStringTokenizerUnitWordBoundary,
            Locale(identifier: "ja_JP") as CFLocale
        )
        var t = CFStringTokenizerGoToTokenAtIndex(tokenizer, 0)
        while t != [] {
            let range = CFStringTokenizerGetCurrentTokenRange(tokenizer)
            if range.length > 0 {
                let nsRange = NSRange(location: range.location, length: range.length)
                if let r = Range(nsRange, in: text) {
                    let surface = String(text[r])
                    let hira = readingHiragana(for: surface, tokenizer: tokenizer)
                    let isParticle = (surface.count == 1) &&
                        (surface == "は" || surface == "へ" || surface == "を")
                    result.append((surface, hira, isParticle))
                }
            }
            t = CFStringTokenizerAdvanceToNextToken(tokenizer)
        }
        return result
    }

    /// Get hiragana reading for a token. Falls back to:
    /// (a) the romaji transcription via Latin-Hiragana
    /// (b) the surface itself converted katakana→hiragana
    /// (c) the surface unchanged
    private func readingHiragana(for surface: String, tokenizer: CFStringTokenizer?) -> String {
        guard let tokenizer else { return surface }
        // If surface is already hiragana, use it directly
        if surface.unicodeScalars.allSatisfy({ $0.value >= 0x3040 && $0.value <= 0x309F }) {
            return surface
        }
        // Get romaji from CFStringTokenizer attribute
        if let romaji = CFStringTokenizerCopyCurrentTokenAttribute(
            tokenizer, kCFStringTokenizerAttributeLatinTranscription) as? String,
           !romaji.isEmpty {
            let m = NSMutableString(string: romaji)
            CFStringTransform(m, nil, "Latin-Hiragana" as CFString, false)
            let hira = m as String
            if !hira.isEmpty && hira != romaji { return hira }
        }
        // Fallback: katakana → hiragana
        let m = NSMutableString(string: surface)
        CFStringTransform(m, nil, kCFStringTransformHiraganaKatakana, true)
        return m as String
    }

    // MARK: - Hiragana → IPA (cutlet HEPBURN logic)

    /// Convert a hiragana string to IPA phonemes using HEPBURN + context rules.
    private func romaji(hiragana: String) -> String {
        var out = ""
        let chars = Array(hiragana)
        var i = 0
        while i < chars.count {
            let ch = chars[i]
            let kk = String(ch)
            let pk = i > 0 ? String(chars[i - 1]) : nil
            let nk = i + 1 < chars.count ? String(chars[i + 1]) : nil

            // Try digraph first: pk + kk (already handled below by skipping the kk char when it's a sutegana)
            if let prev = pk, let mapped = Self.hepburn[prev + kk] {
                // The previous iteration emitted the base; this digraph case shouldn't normally happen
                // because we look ahead via nk. Skip.
                _ = mapped
            }

            // Look ahead: kk + nk (digraph starting at kk)
            if let next = nk, Self.hepburn[kk + next] != nil {
                // Skip this char; the next iteration will emit the digraph as kk+nk via prev+kk path
                // Actually we need to emit it now and skip nk
                if let dg = Self.hepburn[kk + next] {
                    out += dg
                    i += 2
                    continue
                }
            }

            // sutegana that wasn't merged → skip
            if Self.suteganaSet.contains(ch) {
                i += 1
                continue
            }

            // 長音符 ー
            if kk == "ー" {
                out += "ː"
                i += 1
                continue
            }

            // 促音 っ — emit glottal stop placeholder (Kokoro phoneme is ʔ in cutlet)
            if kk == "っ" {
                out += "ʔ"
                i += 1
                continue
            }

            // ん — context-dependent (m before m/p/b, ŋ before k/g, ɲ before ɲ/ʨ/ʥ, n before n/t/d/r/z, ɴ otherwise)
            if kk == "ん" {
                let nextPhon = nk.flatMap { Self.hepburn[$0] } ?? ""
                let first = nextPhon.first
                if let f = first {
                    if "mpb".contains(f) { out += "m"; i += 1; continue }
                    if "kɡ".contains(f) { out += "ŋ"; i += 1; continue }
                    if nextPhon.hasPrefix("ɲ") || nextPhon.hasPrefix("ʨ") || nextPhon.hasPrefix("ʥ") {
                        out += "ɲ"; i += 1; continue
                    }
                    if "ntdɾz".contains(f) { out += "n"; i += 1; continue }
                }
                out += "ɴ"
                i += 1
                continue
            }

            // Plain single hiragana lookup
            if let phon = Self.hepburn[kk] {
                out += phon
            }
            i += 1
        }
        return out
    }
}
