import Foundation
import XCTest
@testable import Hallar

final class HallarTests: XCTestCase {
    static let fixture = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("fixtures/tiny.hallar")

    let docs: [(id: String, text: String)] = [
        ("0", "el perro y el gato"),
        ("1", "cafe con leche caliente"),
        ("2", "receta de croquetas de jamon"),
        ("3", "boleto de tren para el viaje"),
    ]

    func model() throws -> HallarModel { try HallarModel(url: Self.fixture) }

    func testModel() throws {
        let m = try model()
        XCTAssertEqual(m.dimension, 8)
        XCTAssertEqual(m.vocabularySize, 32)
        XCTAssertEqual(try m.tokenize("Hola mundos!"), [2, 3, 4, 23])
        let v = try m.embed("hola mundo")
        XCTAssertEqual(Double(v.reduce(0) { $0 + $1 * $1 }), 1.0, accuracy: 1e-5)
        XCTAssertEqual(try m.embed("zzz"), [Float](repeating: 0, count: 8))
        XCTAssertFalse(Hallar.version.isEmpty)
    }

    func testOpenErrors() {
        XCTAssertThrowsError(try HallarModel(path: "/nope/x.hallar")) { e in
            XCTAssertTrue((e as? HallarError)?.message.contains("cannot open") ?? false)
        }
    }

    func testSearchModes() throws {
        let idx = try HallarIndex(model: model())
        try idx.add(docs)
        XCTAssertEqual(idx.count, 4)
        XCTAssertTrue(idx.contains(id: "2"))
        XCTAssertFalse(idx.contains(id: "9"))
        XCTAssertEqual(try idx.search("croquetas de jamón", limit: 1, mode: .keyword).first?.id, "2")
        XCTAssertEqual(try idx.search("boleto tren").first?.id, "3")
        XCTAssertEqual(try idx.search("perro", limit: 2, mode: .vector).count, 2)
        XCTAssertTrue(try idx.search("zzz", mode: .keyword).isEmpty)
        XCTAssertThrowsError(try idx.add(id: "0", text: "dup"))
        XCTAssertThrowsError(try idx.setVectorWeight(2))
        try idx.setVectorWeight(1)
        XCTAssertEqual(try idx.search("gato", limit: 1).first?.id, try idx.search("gato", limit: 1, mode: .vector).first?.id)
    }

    func testSaveLoad() throws {
        let m = try model()
        let idx = try HallarIndex(model: m)
        try idx.add(docs)
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("hallar-swift-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try idx.save(to: dir)
        let back = try HallarIndex.load(model: m, from: dir)
        XCTAssertEqual(back.count, 4)
        for q in ["perro", "cafe", "viaje"] {
            XCTAssertEqual(try back.search(q), try idx.search(q))
        }
    }

    func testConcurrentReads() throws {
        let idx = try HallarIndex(model: model())
        try idx.add(docs)
        let want = try idx.search("tren")
        DispatchQueue.concurrentPerform(iterations: 64) { _ in
            XCTAssertEqual(try? idx.search("tren"), want)
        }
    }
}
