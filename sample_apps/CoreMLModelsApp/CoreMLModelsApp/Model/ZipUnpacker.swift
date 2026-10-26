import Foundation
import Compression

/// Minimal ZIP extractor for .mlpackage.zip archives.
///
/// Parses the ZIP central directory ourselves and streams each entry from
/// disk through Apple's Compression framework (`.zlib`, i.e. raw deflate) in
/// 1 MB chunks, so memory stays flat however large the archive is (the
/// Stable Audio DiT weights alone are 1.3 GB; reading the whole archive into
/// memory got the app killed on iPhone). Stored (method 0) entries are copied
/// through unchanged. Deflate (method 8) is the only other format produced by
/// standard zip tools on macOS. ZIP64 archives are not supported.
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

    private static let chunkSize = 1 << 20

    static func unpack(archive: URL, to destination: URL) throws {
        let handle: FileHandle
        do {
            handle = try FileHandle(forReadingFrom: archive)
        } catch {
            throw UnzipError.io(error.localizedDescription)
        }
        defer { try? handle.close() }

        let entries = try parseCentralDirectory(handle: handle)
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
            try extract(entry, from: handle, to: outURL)
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

    private static func parseCentralDirectory(handle: FileHandle) throws -> [Entry] {
        let fileSize = try handle.seekToEnd()
        guard fileSize >= 22 else { throw UnzipError.invalidArchive("too small") }
        // The end-of-central-directory record sits in the last 22 + comment bytes.
        let maxComment: UInt64 = 65535
        let tailLength = min(fileSize, 22 + maxComment)
        let tail = try read(handle, at: fileSize - tailLength, count: Int(tailLength))
        var eocdOffset: Int? = nil
        var i = tail.count - 22
        while i >= 0 {
            if tail[i] == 0x50 && tail[i + 1] == 0x4b
                && tail[i + 2] == 0x05 && tail[i + 3] == 0x06 {
                eocdOffset = i
                break
            }
            i -= 1
        }
        guard let eocd = eocdOffset else {
            throw UnzipError.invalidArchive("no EOCD")
        }

        let totalEntries = readU16(tail, at: eocd + 10)
        let cdSize = readU32(tail, at: eocd + 12)
        let cdOffset = readU32(tail, at: eocd + 16)
        if totalEntries == 0xFFFF || cdSize == 0xFFFF_FFFF || cdOffset == 0xFFFF_FFFF {
            throw UnzipError.unsupported("ZIP64")
        }
        guard UInt64(cdOffset) + UInt64(cdSize) <= fileSize else {
            throw UnzipError.invalidArchive("central directory out of range")
        }
        let bytes = try read(handle, at: UInt64(cdOffset), count: Int(cdSize))

        var entries: [Entry] = []
        var p = 0
        for _ in 0..<totalEntries {
            guard p + 46 <= bytes.count,
                  bytes[p] == 0x50 && bytes[p + 1] == 0x4b
                    && bytes[p + 2] == 0x01 && bytes[p + 3] == 0x02 else {
                throw UnzipError.invalidArchive("bad CDH signature at \(Int(cdOffset) + p)")
            }
            let method = readU16(bytes, at: p + 10)
            let compressed = readU32(bytes, at: p + 20)
            let uncompressed = readU32(bytes, at: p + 24)
            let nameLen = Int(readU16(bytes, at: p + 28))
            let extraLen = Int(readU16(bytes, at: p + 30))
            let commentLen = Int(readU16(bytes, at: p + 32))
            let localOffset = readU32(bytes, at: p + 42)
            if compressed == 0xFFFF_FFFF || uncompressed == 0xFFFF_FFFF || localOffset == 0xFFFF_FFFF {
                throw UnzipError.unsupported("ZIP64")
            }
            let nameStart = p + 46
            guard nameStart + nameLen <= bytes.count else {
                throw UnzipError.invalidArchive("CDH name overflow")
            }
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

    // MARK: - Streaming extract

    /// Copies (method 0) or inflates (method 8) one entry to `outURL`, 1 MB of
    /// input at a time, and checks the written size against the directory.
    private static func extract(_ entry: Entry, from handle: FileHandle, to outURL: URL) throws {
        guard entry.method == 0 || entry.method == 8 else {
            throw UnzipError.unsupported("compression method \(entry.method)")
        }
        // The local header's name/extra lengths can differ from the central directory's.
        let lh = UInt64(entry.localHeaderOffset)
        let header = try read(handle, at: lh, count: 30)
        guard header.count == 30,
              header[0] == 0x50 && header[1] == 0x4b
                && header[2] == 0x03 && header[3] == 0x04 else {
            throw UnzipError.invalidArchive("bad LFH at \(lh)")
        }
        let nameLen = UInt64(readU16(header, at: 26))
        let extraLen = UInt64(readU16(header, at: 28))
        try handle.seek(toOffset: lh + 30 + nameLen + extraLen)

        guard FileManager.default.createFile(atPath: outURL.path, contents: nil) else {
            throw UnzipError.io("cannot create \(entry.name)")
        }
        let out = try FileHandle(forWritingTo: outURL)
        defer { try? out.close() }

        var written: UInt64 = 0
        var filter: OutputFilter?
        if entry.method == 8 {
            filter = try OutputFilter(.decompress, using: .zlib) { chunk in
                guard let chunk else { return }
                try out.write(contentsOf: chunk)
                written += UInt64(chunk.count)
            }
        }
        var remaining = UInt64(entry.compressedSize)
        while remaining > 0 {
            try autoreleasepool {
                let chunk = try handle.read(upToCount: Int(min(remaining, UInt64(chunkSize)))) ?? Data()
                guard !chunk.isEmpty else {
                    throw UnzipError.invalidArchive("\(entry.name) is truncated")
                }
                remaining -= UInt64(chunk.count)
                if let filter {
                    try filter.write(chunk)
                } else {
                    try out.write(contentsOf: chunk)
                    written += UInt64(chunk.count)
                }
            }
        }
        try filter?.finalize()
        guard written == UInt64(entry.uncompressedSize) else {
            throw UnzipError.io("\(entry.name): wrote \(written) of \(entry.uncompressedSize) bytes")
        }
    }

    private static func read(_ handle: FileHandle, at offset: UInt64, count: Int) throws -> [UInt8] {
        try handle.seek(toOffset: offset)
        return [UInt8](try handle.read(upToCount: count) ?? Data())
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
