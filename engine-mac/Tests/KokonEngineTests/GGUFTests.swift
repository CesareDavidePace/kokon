import Foundation
import XCTest
@testable import KokonEngine

/// Small GGUF files written byte by byte for the tests.
enum TestGGUF {
    /// A GGUF v3 with `metadata` and `tensors` (name, ggml dims, type, raw bytes), data aligned to 32.
    static func write(in directory: URL, name: String = "test-\(UUID().uuidString).gguf",
                      metadata: [(String, UInt32, [UInt8])] = [],
                      tensors: [(String, [Int], QuantType, [UInt8])]) throws -> String {
        var out: [UInt8] = []
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { out += $0 } }
        func u64(_ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { out += $0 } }
        func string(_ s: String) { u64(UInt64(s.utf8.count)); out += Array(s.utf8) }
        u32(0x4655_4747); u32(3); u64(UInt64(tensors.count)); u64(UInt64(metadata.count))
        for (key, type, value) in metadata { string(key); u32(type); out += value }
        var offset = 0
        var offsets: [Int] = []
        for (_, _, _, bytes) in tensors {
            offsets.append(offset)
            offset = (offset + bytes.count + 31) / 32 * 32
        }
        for (i, (name, dims, type, _)) in tensors.enumerated() {
            string(name); u32(UInt32(dims.count))
            for d in dims { u64(UInt64(d)) }
            u32(type.rawValue); u64(UInt64(offsets[i]))
        }
        while out.count % 32 != 0 { out.append(0) }
        let start = out.count
        for (i, (_, _, _, bytes)) in tensors.enumerated() {
            while out.count < start + offsets[i] { out.append(0) }
            out += bytes
        }
        let path = directory.appendingPathComponent(name).path
        try Data(out).write(to: URL(fileURLWithPath: path))
        return path
    }
}

final class GGUFTests: XCTestCase {
    private var temporary: URL!
    override func setUpWithError() throws {
        temporary = FileManager.default.temporaryDirectory.appendingPathComponent("kokon-engine-tests-\(UUID())")
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
    }
    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: temporary) }

    func testReadsMetadataAndTensors() throws {
        let floats: [Float] = [1, -2, 0.5, 3]
        let f32 = floats.withUnsafeBytes { Array($0) }
        let bf16: [UInt8] = [0x80, 0x3f, 0x00, 0xc0]   // 1.0, -2.0
        var arch: [UInt8] = []
        withUnsafeBytes(of: UInt64(10).littleEndian) { arch += $0 }
        arch += Array("qwen_image".utf8)
        let path = try TestGGUF.write(in: temporary, metadata: [("general.architecture", 8, arch)],
                                      tensors: [("a", [2, 2], .f32, f32), ("b", [2], .bf16, bf16)])
        let file = try GGUFFile(path: path)
        XCTAssertEqual(file.version, 3)
        XCTAssertEqual(file.metadata["general.architecture"]?.string, "qwen_image")
        XCTAssertEqual(file.order, ["a", "b"])
        let a = try file.tensor("a")
        XCTAssertEqual(a.rows, 2); XCTAssertEqual(a.columns, 2)
        XCTAssertEqual(try Dequant.array(file, "a"), floats)
        XCTAssertEqual(try Dequant.array(file, "b"), [1, -2])
        XCTAssertEqual(file.dataStart % 32, 0)
    }

    func testRefusesBadFiles() throws {
        let notGGUF = temporary.appendingPathComponent("x.gguf")
        try Data("hello world, not a model".utf8).write(to: notGGUF)
        XCTAssertThrowsError(try GGUFFile(path: notGGUF.path))
        // A tensor whose bytes run past the end of the file.
        let path = try TestGGUF.write(in: temporary, tensors: [("a", [256, 4], .q4_K, [UInt8](repeating: 0, count: 100))])
        XCTAssertThrowsError(try GGUFFile(path: path))
        XCTAssertThrowsError(try GGUFFile(path: temporary.appendingPathComponent("missing.gguf").path))
    }
}
