import Foundation
import Metal
import XCTest
@testable import KokonEngine

/// The unpacking of every GGUF type the engine reads, CPU and GPU, against gguf-py's values
/// (`Fixtures/quant.json`, written by `tools/make_quant_fixtures.py`): zero, −0, tiny, subnormal and
/// largest fp16 scales, all-zero and all-one quant bits, random content.
final class QuantTests: XCTestCase {
    struct Case { let type: QuantType; let bytes: [UInt8]; let values: [Float] }

    static let cases: [Case] = {
        let url = Bundle.module.url(forResource: "quant", withExtension: "json", subdirectory: "Fixtures")!
        let json = try! JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: [String: Any]]
        let types: [String: QuantType] = ["Q4_0": .q4_0, "Q8_0": .q8_0, "Q4_K": .q4_K, "Q5_K": .q5_K,
                                          "Q6_K": .q6_K, "F16": .f16, "BF16": .bf16]
        return json.keys.sorted().map { name in
            let entry = json[name]!
            let hex = Array((entry["bytes"] as! String).utf8)
            let bytes = stride(from: 0, to: hex.count, by: 2).map { UInt8(String(decoding: hex[$0..<$0 + 2], as: UTF8.self), radix: 16)! }
            return Case(type: types[name]!, bytes: bytes, values: (entry["values"] as! [NSNumber]).map(\.floatValue))
        }
    }()

    /// Equal, or within one rounding of fp32 (ggml and numpy may order `d · sc · q` differently).
    static func close(_ a: Float, _ b: Float) -> Bool {
        if a == b || (a.isNaN && b.isNaN) { return true }
        return abs(a - b) <= max(abs(a), abs(b)) * 2.4e-7
    }

    func testCPUMatchesGGUFPy() {
        XCTAssertEqual(Self.cases.count, 7)
        for c in Self.cases {
            var out = [Float](repeating: .nan, count: c.values.count)
            c.bytes.withUnsafeBytes { raw in
                out.withUnsafeMutableBufferPointer {
                    Dequant.row(raw.baseAddress!, type: c.type, count: c.values.count, into: $0.baseAddress!)
                }
            }
            let wrong = zip(out, c.values).enumerated().filter { !Self.close($0.element.0, $0.element.1) }
            XCTAssert(wrong.isEmpty, "\(c.type.name): \(wrong.count) values differ, first at \(wrong.first.map { "\($0.offset): \($0.element)" } ?? "")")
        }
    }

    /// The same blocks laid out as a small GGUF (several rows), unpacked on the GPU in both layouts and both
    /// precisions.
    func testGPUMatchesGGUFPy() throws {
        let context = try MetalContext()
        let gpu = GPUDequant(context: context)
        for c in Self.cases {
            let columns = c.type.blockValues == 1 ? 16 : c.type.blockValues * 3
            let rows = c.values.count / columns
            XCTAssertEqual(rows * columns, c.values.count)
            let path = try TestGGUF.write(in: temporary, tensors: [("w", [columns, rows], c.type, c.bytes)])
            let file = try GGUFFile(path: path)
            let tensor = try file.tensor("w")
            for output in [GPUDequant.Output.f32, .f16] {
                for transposed in [false, true] {
                    let out = try context.scratch(c.values.count * output.bytes, label: "test")
                    let commands = context.queue.makeCommandBuffer()!
                    try gpu.encode(tensor, of: file, into: out, transposed: transposed, output: output, commands: commands)
                    try MetalContext.finish(commands, "test")
                    var wrong = 0
                    for r in 0..<rows {
                        for col in 0..<columns {
                            let i = transposed ? col * rows + r : r * columns + col
                            let want = c.values[r * columns + col]
                            let got: Float = output == .f32 ? out.contents().assumingMemoryBound(to: Float.self)[i]
                                : Float(out.contents().assumingMemoryBound(to: Float16.self)[i])
                            let expected = output == .f32 ? want : Float(Float16(want))
                            if !Self.close(got, expected) { wrong += 1 }
                        }
                    }
                    XCTAssertEqual(wrong, 0, "\(c.type.name) \(output) transposed \(transposed)")
                }
            }
        }
    }

    func testBlockSizes() {
        XCTAssertEqual(QuantType.q4_K.bytes(count: 4096), 16 * 144)
        XCTAssertEqual(QuantType.q6_K.bytes(count: 256), 210)
        XCTAssertEqual(QuantType.bf16.bytes(count: 64), 128)
    }

    private var temporary: URL!
    override func setUpWithError() throws {
        temporary = FileManager.default.temporaryDirectory.appendingPathComponent("kokon-engine-tests-\(UUID())")
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
    }
    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: temporary) }
}
