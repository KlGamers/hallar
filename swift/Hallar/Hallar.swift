import CHallar
import Foundation

public struct HallarError: Error, CustomStringConvertible, Equatable {
    public let message: String
    public var description: String { message }

    static func last() -> HallarError {
        HallarError(message: String(cString: hallar_last_error()))
    }
}

public enum SearchMode: Int32, Sendable, CaseIterable {
    case hybrid = 0
    case vector = 1
    case keyword = 2
}

public struct SearchResult: Sendable, Hashable {
    public let id: String
    public let text: String
    public let score: Float
}

public final class HallarModel: @unchecked Sendable {
    let ptr: OpaquePointer

    public init(path: String) throws {
        guard let p = hallar_model_open(path) else { throw HallarError.last() }
        ptr = p
    }

    public convenience init(url: URL) throws {
        try self.init(path: url.path)
    }

    deinit { hallar_model_free(ptr) }

    public var dimension: Int { hallar_model_dim(ptr) }
    public var vocabularySize: Int { hallar_model_vocab(ptr) }

    public func embed(_ text: String) throws -> [Float] {
        var out = [Float](repeating: 0, count: dimension)
        let rc = out.withUnsafeMutableBufferPointer { buf in hallar_embed(ptr, text, buf.baseAddress) }
        if rc != 0 { throw HallarError.last() }
        return out
    }

    public func tokenize(_ text: String) throws -> [Int32] {
        var n = 0
        if hallar_tokenize(ptr, text, nil, 0, &n) != 0 { throw HallarError.last() }
        var ids = [Int32](repeating: 0, count: n)
        let rc = ids.withUnsafeMutableBufferPointer { buf in hallar_tokenize(ptr, text, buf.baseAddress, n, &n) }
        if rc != 0 { throw HallarError.last() }
        return ids
    }
}

public final class HallarIndex: @unchecked Sendable {
    let ptr: OpaquePointer
    public let model: HallarModel

    public init(model: HallarModel, connections: Int = 16, buildEffort: Int = 200) throws {
        guard let p = hallar_engine_new(model.ptr, connections, buildEffort) else { throw HallarError.last() }
        ptr = p
        self.model = model
    }

    init(model: HallarModel, ptr: OpaquePointer) {
        self.model = model
        self.ptr = ptr
    }

    public static func load(model: HallarModel, from dir: URL) throws -> HallarIndex {
        guard let p = hallar_engine_load(model.ptr, dir.path) else { throw HallarError.last() }
        return HallarIndex(model: model, ptr: p)
    }

    deinit { hallar_engine_free(ptr) }

    public var count: Int { hallar_engine_size(ptr) }

    public func contains(id: String) -> Bool { hallar_engine_has(ptr, id) == 1 }

    public func setVectorWeight(_ w: Float) throws {
        if hallar_engine_set_weight(ptr, w) != 0 { throw HallarError.last() }
    }

    public func add(id: String, text: String) throws {
        if hallar_engine_add(ptr, id, text) != 0 { throw HallarError.last() }
    }

    public func add<S: Sequence>(_ docs: S) throws where S.Element == (id: String, text: String) {
        for d in docs { try add(id: d.id, text: d.text) }
    }

    public func search(_ query: String, limit: Int = 10, mode: SearchMode = .hybrid, effort: Int = 64) throws -> [SearchResult] {
        var hits = hallar_hits()
        if hallar_engine_search_mode(ptr, query, limit, effort, mode.rawValue, &hits) != 0 { throw HallarError.last() }
        defer { hallar_hits_free(&hits) }
        guard let base = hits.hits else { return [] }
        return (0..<hits.len).map { i in
            let h = base[i]
            return SearchResult(id: String(cString: h.id), text: String(cString: h.text), score: h.score)
        }
    }

    public func save(to dir: URL) throws {
        if hallar_engine_save(ptr, dir.path) != 0 { throw HallarError.last() }
    }
}

public enum Hallar {
    public static var version: String { String(cString: hallar_version()) }
}
