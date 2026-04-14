import Foundation
import Compression

/// Minimal ZIP extractor for .mlpackage.zip archives.
///
/// Parses the ZIP central directory ourselves and uses Apple's Compression
/// framework (`compression_decode_buffer` with `COMPRESSION_ZLIB`, i.e. raw
/// deflate) for each deflate-compressed entry. Stored (method 0) entries are
/// copied through unchanged. Deflate (method 8) is the only other format
/// produced by standard zip tools on macOS.
enum ZipUnpacker {
    enum UnzipError: LocalizedError {
        case invalidArchive(String)
        case unsupported(String)
        case io(String)

        var errorDescription: String? {
            switch self {
            case .invalidArchive(let m): return "Invalid zip: \(m)"
            case .unsupported(let m): return "Unsupported zip feature: \(m)"
            case .io(let m): return "I/O error during unzip: \(m)"
            }
        }
    }

    static func unpack(archive: URL, to destination: URL) throws {
        let raw: Data
        do {
            raw = try Data(contentsOf: archive, options: [.mappedIfSafe])
        } catch {
            throw UnzipError.io(error.localizedDescription)
        }
        let bytes = [UInt8](raw)

        let entries = try parseCentralDirectory(bytes: bytes)
        let fm = FileManager.default
        for entry in entries {
            // Skip macOS metadata injected by Finder/zip
            if entry.name.hasPrefix("__MACOSX") || entry.name.contains("/._") { continue }
            let outURL = destination.appendingPathComponent(entry.name)
            if entry.isDirectory {
                try fm.createDirectory(at: outURL, withIntermediateDirectories: true)
                continue
            }
            try fm.createDirectory(at: outURL.deletingLastPathComponent(),
                                   withIntermediateDirectories: true)
            let compressed = try readLocalFile(bytes: bytes, entry: entry)
            let decompressed: Data
            switch entry.method {
            case 0:
                decompressed = Data(compressed)
            case 8:
                decompressed = try inflate(compressed, uncompressedSize: Int(entry.uncompressedSize))
            default:
                throw UnzipError.unsupported("compression method \(entry.method)")
            }
            try decompressed.write(to: outURL)
        }
    }

    // MARK: - ZIP parsing

    private struct Entry {
        let name: String
        let method: UInt16
        let compressedSize: UInt32
        let uncompressedSize: UInt32
        let localHeaderOffset: UInt32
        let isDirectory: Bool
    }

    private static func parseCentralDirectory(bytes: [UInt8]) throws -> [Entry] {
        guard bytes.count >= 22 else { throw UnzipError.invalidArchive("too small") }
        let maxComment = 65535
        let start = max(0, bytes.count - 22 - maxComment)
        var eocdOffset: Int? = nil
        var i = bytes.count - 22
        while i >= start {
            if bytes[i] == 0x50 && bytes[i + 1] == 0x4b
                && bytes[i + 2] == 0x05 && bytes[i + 3] == 0x06 {
                eocdOffset = i
                break
            }
            i -= 1
        }
        guard let eocd = eocdOffset else {
            throw UnzipError.invalidArchive("no EOCD")
        }

        let totalEntries = readU16(bytes, at: eocd + 10)
        let cdOffset = readU32(bytes, at: eocd + 16)

        var entries: [Entry] = []
        var p = Int(cdOffset)
        for _ in 0..<totalEntries {
            guard p + 46 <= bytes.count,
                  bytes[p] == 0x50 && bytes[p + 1] == 0x4b
                    && bytes[p + 2] == 0x01 && bytes[p + 3] == 0x02 else {
                throw UnzipError.invalidArchive("bad CDH signature at \(p)")
            }
            let method = readU16(bytes, at: p + 10)
            let compressed = readU32(bytes, at: p + 20)
            let uncompressed = readU32(bytes, at: p + 24)
            let nameLen = Int(readU16(bytes, at: p + 28))
            let extraLen = Int(readU16(bytes, at: p + 30))
            let commentLen = Int(readU16(bytes, at: p + 32))
            let localOffset = readU32(bytes, at: p + 42)
            let nameStart = p + 46
            let nameBytes = Array(bytes[nameStart..<nameStart + nameLen])
            let name = String(decoding: nameBytes, as: UTF8.self)
            let isDir = name.hasSuffix("/")
            entries.append(Entry(name: name,
                                 method: method,
                                 compressedSize: compressed,
                                 uncompressedSize: uncompressed,
                                 localHeaderOffset: localOffset,
                                 isDirectory: isDir))
            p = nameStart + nameLen + extraLen + commentLen
        }
        return entries
    }

    private static func readLocalFile(bytes: [UInt8], entry: Entry) throws -> [UInt8] {
        let lh = Int(entry.localHeaderOffset)
        guard lh + 30 <= bytes.count,
              bytes[lh] == 0x50 && bytes[lh + 1] == 0x4b
                && bytes[lh + 2] == 0x03 && bytes[lh + 3] == 0x04 else {
            throw UnzipError.invalidArchive("bad LFH at \(lh)")
        }
        let nameLen = Int(readU16(bytes, at: lh + 26))
        let extraLen = Int(readU16(bytes, at: lh + 28))
        let dataStart = lh + 30 + nameLen + extraLen
        let dataEnd = dataStart + Int(entry.compressedSize)
        guard dataEnd <= bytes.count else {
            throw UnzipError.invalidArchive("LFH overflow")
        }
        return Array(bytes[dataStart..<dataEnd])
    }

    // MARK: - Deflate

    private static func inflate(_ input: [UInt8], uncompressedSize: Int) throws -> Data {
        let outCapacity = max(uncompressedSize, 1)
        var output = Data(count: outCapacity)
        let written = output.withUnsafeMutableBytes { (outBuf: UnsafeMutableRawBufferPointer) -> Int in
            input.withUnsafeBufferPointer { (inBuf: UnsafeBufferPointer<UInt8>) -> Int in
                guard let srcPtr = inBuf.baseAddress,
                      let dstPtr = outBuf.bindMemory(to: UInt8.self).baseAddress else {
                    return 0
                }
                return compression_decode_buffer(dstPtr, outCapacity,
                                                 srcPtr, input.count,
                                                 nil,
                                                 COMPRESSION_ZLIB)
            }
        }
        if written == 0 {
            throw UnzipError.io("decompression failed")
        }
        return output.prefix(written)
    }

    // MARK: - Little-endian byte reads

    private static func readU16(_ bytes: [UInt8], at offset: Int) -> UInt16 {
        let b0: UInt16 = UInt16(bytes[offset])
        let b1: UInt16 = UInt16(bytes[offset + 1])
        return b0 | (b1 &<< 8)
    }

    private static func readU32(_ bytes: [UInt8], at offset: Int) -> UInt32 {
        let b0: UInt32 = UInt32(bytes[offset])
        let b1: UInt32 = UInt32(bytes[offset + 1])
        let b2: UInt32 = UInt32(bytes[offset + 2])
        let b3: UInt32 = UInt32(bytes[offset + 3])
        return b0 | (b1 &<< 8) | (b2 &<< 16) | (b3 &<< 24)
    }
}
