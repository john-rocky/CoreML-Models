import Foundation

// MARK: - Manifest root

struct Manifest: Codable {
    let manifestVersion: Int
    let updatedAt: String
    let minAppVersion: String
    let categories: [Category]
    let models: [ModelEntry]

    enum CodingKeys: String, CodingKey {
        case manifestVersion = "manifest_version"
        case updatedAt = "updated_at"
        case minAppVersion = "min_app_version"
        case categories
        case models
    }
}

// MARK: - Category

struct Category: Codable, Identifiable, Hashable {
    let id: String
    let name: String
    let icon: String
    let order: Int
}

// MARK: - Model entry

struct ModelEntry: Codable, Identifiable, Hashable {
    let id: String
    let name: String
    let subtitle: String?
    let categoryId: String
    let descriptionMd: String
    let thumbnailUrl: String?
    let demo: DemoSpec
    let files: [FileSpec]
    let requirements: Requirements
    let license: LicenseInfo
    let upstream: UpstreamInfo
    let creditsMd: String?

    enum CodingKeys: String, CodingKey {
        case id, name, subtitle
        case categoryId = "category_id"
        case descriptionMd = "description_md"
        case thumbnailUrl = "thumbnail_url"
        case demo, files, requirements, license, upstream
        case creditsMd = "credits_md"
    }

    /// Sum of all (non-optional) file sizes in bytes.
    var downloadSize: Int64 {
        files.reduce(0) { $0 + (($1.optional ?? false) ? 0 : Int64($1.sizeBytes)) }
    }

    static func == (lhs: ModelEntry, rhs: ModelEntry) -> Bool { lhs.id == rhs.id }
    func hash(into hasher: inout Hasher) { hasher.combine(id) }
}

// MARK: - Demo

struct DemoSpec: Codable, Hashable {
    let template: String
    let config: [String: AnyCodable]?
}

// MARK: - File

struct FileSpec: Codable, Hashable {
    let name: String
    let url: String
    let archive: String?
    let sizeBytes: Int
    let sha256: String
    let computeUnits: String?
    let optional: Bool?
    let kind: String?

    enum CodingKeys: String, CodingKey {
        case name, url, archive
        case sizeBytes = "size_bytes"
        case sha256
        case computeUnits = "compute_units"
        case optional, kind
    }
}

// MARK: - Requirements

struct Requirements: Codable, Hashable {
    let minIos: String
    let minRamMb: Int
    let deviceCapabilities: [String]?

    enum CodingKeys: String, CodingKey {
        case minIos = "min_ios"
        case minRamMb = "min_ram_mb"
        case deviceCapabilities = "device_capabilities"
    }
}

// MARK: - License

struct LicenseInfo: Codable, Hashable {
    let name: String
    let url: String
}

// MARK: - Upstream

struct UpstreamInfo: Codable, Hashable {
    let name: String
    let url: String
    let year: Int?
}

// MARK: - AnyCodable (for demo config values of mixed types)

struct AnyCodable: Codable, Hashable {
    let value: Any

    init(_ value: Any) { self.value = value }

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let v = try? c.decode(Bool.self) { value = v; return }
        if let v = try? c.decode(Int.self) { value = v; return }
        if let v = try? c.decode(Double.self) { value = v; return }
        if let v = try? c.decode(String.self) { value = v; return }
        if let v = try? c.decode([AnyCodable].self) { value = v.map { $0.value }; return }
        if let v = try? c.decode([String: AnyCodable].self) {
            value = v.mapValues { $0.value }; return
        }
        value = NSNull()
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch value {
        case let v as Bool: try c.encode(v)
        case let v as Int: try c.encode(v)
        case let v as Double: try c.encode(v)
        case let v as String: try c.encode(v)
        case let v as [Any]: try c.encode(v.map { AnyCodable($0) })
        case let v as [String: Any]: try c.encode(v.mapValues { AnyCodable($0) })
        default: try c.encodeNil()
        }
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(String(describing: value))
    }

    static func == (lhs: AnyCodable, rhs: AnyCodable) -> Bool {
        String(describing: lhs.value) == String(describing: rhs.value)
    }
}
