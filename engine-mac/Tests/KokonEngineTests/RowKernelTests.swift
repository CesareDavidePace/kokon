import Metal
import XCTest
@testable import KokonEngine

/// `swiglu` and `layernorm_modulate` with and without the fp16 copy of the next product's input (`QwenDiT.HalfOut`):
/// the values, which rows get fp32, which fp16, and the padding rows of the copy zeroed. The text encoder calls
/// `swiglu` with every row in fp32 (it broke once when the kernel's arguments changed).
final class RowKernelTests: XCTestCase {
    let rows = 6, columns = 512

    func values(_ count: Int, seed: UInt64) -> [Float] {
        var s = seed
        return (0..<count).map { _ in
            s = s &* 6364136223846793005 &+ 1442695040888963407
            return (Float(s >> 40) / Float(1 << 24) - 0.5) * 6
        }
    }

    func testSwiglu() throws {
        let context = try MetalContext()
        let gate = values(rows * columns, seed: 1), proj = values(rows * columns, seed: 2)
        let expected = zip(gate, proj).map { g, p in (g / (1 + exp(-g))) * p }
        // (fp32 from row, fp16 rows, padding rows of the copy)
        for (fullFrom, halfRows, padded) in [(0, 0, 0), (4, 4, 0), (3, 3, 2)] {
            let g = try context.scratch(gate.count * 4, label: "gate"), p = try context.scratch(proj.count * 4, label: "proj")
            g.contents().copyMemory(from: gate, byteCount: gate.count * 4)
            p.contents().copyMemory(from: proj, byteCount: proj.count * 4)
            let total = halfRows + padded
            let half = try context.scratch(max(1, total) * columns * 2, label: "half")
            memset(half.contents(), 0x7f, half.length)   // not zeros, so the padding's zeros are the kernel's
            let commands = context.queue.makeCommandBuffer()!
            let e = commands.makeComputeCommandEncoder()!
            e.setComputePipelineState(try context.pipeline("swiglu"))
            e.setBuffer(g, offset: 0, index: 0); e.setBuffer(p, offset: 0, index: 1)
            let realRows = halfRows == 0 ? rows : halfRows   // with a copy, the product's real rows end where its padding starts
            var params = QwenDiT.SwigluParams(columns: UInt32(columns),
                                              half: halfRows == 0 ? QwenDiT.fullOnly(rows)
                                                  : QwenDiT.HalfOut(rows: UInt32(realRows), halfRows: UInt32(total), fullFrom: UInt32(fullFrom)))
            e.setBytes(&params, length: MemoryLayout<QwenDiT.SwigluParams>.stride, index: 2)
            e.setBuffer(half, offset: 0, index: 3)
            e.dispatchThreads(MTLSize(width: max(realRows, total) * columns, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            e.endEncoding()
            try MetalContext.finish(commands, "swiglu")
            let out = g.contents().assumingMemoryBound(to: Float.self)
            let h = half.contents().assumingMemoryBound(to: Float16.self)
            for r in 0..<realRows {
                for c in 0..<columns {
                    let i = r * columns + c
                    if r >= fullFrom { XCTAssertEqual(out[i], expected[i], accuracy: 1e-5 * max(1, abs(expected[i])), "fp32 row \(r)") }
                    else { XCTAssertEqual(out[i], gate[i], "row \(r) below fullFrom keeps its input") }
                    if r < halfRows { XCTAssertEqual(h[i], Float16(expected[i]), "fp16 row \(r)") }
                }
            }
            for r in stride(from: realRows, to: total, by: 1) { for c in 0..<columns { XCTAssertEqual(h[r * columns + c], 0, "padding row \(r)") } }
        }
    }

    func testLayerNormHalfCopy() throws {
        let context = try MetalContext()
        let x = values(rows * columns, seed: 3), mod = values(columns, seed: 4)
        let xb = try context.scratch(x.count * 4, label: "x"), out = try context.scratch(x.count * 4, label: "out")
        let mb = try context.scratch(columns * 4, label: "mod"), half = try context.scratch((rows + 2) * columns * 2, label: "half")
        xb.contents().copyMemory(from: x, byteCount: x.count * 4)
        mb.contents().copyMemory(from: mod, byteCount: mod.count * 4)
        memset(half.contents(), 0x7f, half.length)
        let commands = context.queue.makeCommandBuffer()!
        let e = commands.makeComputeCommandEncoder()!
        e.setComputePipelineState(try context.pipeline("layernorm_modulate"))
        e.setBuffer(xb, offset: 0, index: 0); e.setBuffer(out, offset: 0, index: 1)
        e.setBuffer(mb, offset: 0, index: 2); e.setBuffer(mb, offset: 0, index: 3)
        // Rows 0..<4 to fp16 only, 4..<6 to fp32 only, 6..<8 padding of the copy.
        var params = QwenDiT.NormParams(columns: UInt32(columns), targetStart: 0, eps: 1e-6,
                                        half: QwenDiT.HalfOut(rows: UInt32(rows), halfRows: 8, fullFrom: 4))
        e.setBytes(&params, length: MemoryLayout<QwenDiT.NormParams>.stride, index: 4)
        e.setBuffer(half, offset: 0, index: 5)
        e.dispatchThreadgroups(MTLSize(width: 8, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        e.endEncoding()
        try MetalContext.finish(commands, "layernorm")
        let o = out.contents().assumingMemoryBound(to: Float.self), h = half.contents().assumingMemoryBound(to: Float16.self)
        for r in 0..<rows {
            let row = Array(x[(r * columns)..<((r + 1) * columns)])
            let mean = row.reduce(0, +) / Float(columns)
            let variance = row.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(columns)
            for c in 0..<columns {
                let want = (row[c] - mean) / (variance + 1e-6).squareRoot() * mod[c]
                if r < 4 { XCTAssertEqual(Float(h[r * columns + c]), want, accuracy: 2e-3 * max(1, abs(want))) }
                else { XCTAssertEqual(o[r * columns + c], want, accuracy: 1e-4 * max(1, abs(want))) }
            }
        }
        for i in (rows * columns)..<(8 * columns) { XCTAssertEqual(h[i], 0) }
    }
}
