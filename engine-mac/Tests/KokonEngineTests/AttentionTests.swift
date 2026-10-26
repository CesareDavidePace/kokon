import Metal
import XCTest
@testable import KokonEngine

/// `flash_attention` against attention computed in double on the CPU: plain and causal (the block-causal mask's
/// text segments), sizes that are not multiples of the kernel's tiles, rows past the queries left untouched.
final class AttentionTests: XCTestCase {
    struct FlashParams { var queries, keys, causal, stride: UInt32; var scale: Float; var headBase: UInt32 }

    struct PackParams { var keys, paddedKeys, stride, heads: UInt32 }

    /// `kernel` on K and V packed by `attention_pack_kv` when `packed`.
    func run(queries: Int, keys: Int, causal: Bool, heads: Int = 2, magnitude: Float = 4,
             kernel: String = "flash_attention", packed: Bool = false) throws -> (worst: Double, untouched: Bool) {
        let context = try MetalContext()
        let stride = heads * 128, padding = 64
        let rows = max(queries, keys) + padding
        var seed: UInt64 = UInt64(queries * 1000 + keys)
        func random() -> Float {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return (Float(seed >> 40) / Float(1 << 24) - 0.5) * magnitude
        }
        let q = (0..<(rows * stride)).map { _ in random() }
        let k = (0..<(rows * stride)).map { _ in random() }
        let v = (0..<(rows * stride)).map { _ in random() }
        func buffer(_ a: [Float]) throws -> MTLBuffer {
            let b = try context.scratch(a.count * 4, label: "t")
            b.contents().copyMemory(from: a, byteCount: a.count * 4)
            return b
        }
        let (qb, kb, vb) = (try buffer(q), try buffer(k), try buffer(v))
        let ob = try buffer([Float](repeating: 7, count: rows * stride))
        let commands = context.queue.makeCommandBuffer()!
        let paddedKeys = (keys + 63) / 64 * 64
        let kp = try context.scratch(heads * paddedKeys * 128 * 2, label: "kp"), vp = try context.scratch(heads * paddedKeys * 128 * 2, label: "vp")
        if packed {
            let e = commands.makeComputeCommandEncoder()!
            e.setComputePipelineState(try context.pipeline("attention_pack_kv"))
            for (i, b) in [kb, vb, kp, vp].enumerated() { e.setBuffer(b, offset: 0, index: i) }
            var pp = PackParams(keys: UInt32(keys), paddedKeys: UInt32(paddedKeys), stride: UInt32(stride), heads: UInt32(heads))
            e.setBytes(&pp, length: MemoryLayout<PackParams>.stride, index: 4)
            e.dispatchThreads(MTLSize(width: 32, height: paddedKeys, depth: heads), threadsPerThreadgroup: MTLSize(width: 32, height: 8, depth: 1))
            e.endEncoding()
        }
        let e = commands.makeComputeCommandEncoder()!
        e.setComputePipelineState(try context.pipeline(kernel))
        for (i, b) in [qb, packed ? kp : kb, packed ? vp : vb, ob].enumerated() { e.setBuffer(b, offset: 0, index: i) }
        var padded = UInt32(paddedKeys)
        if packed { e.setBytes(&padded, length: 4, index: 5) }
        var p = FlashParams(queries: UInt32(queries), keys: UInt32(keys), causal: causal ? 1 : 0, stride: UInt32(stride),
                            scale: 1 / Float(128).squareRoot() * Float(M_LOG2E), headBase: 0)
        e.setBytes(&p, length: MemoryLayout<FlashParams>.stride, index: 4)
        e.dispatchThreadgroups(MTLSize(width: (queries + 63) / 64, height: heads, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        e.endEncoding()
        try MetalContext.finish(commands, "flash")
        let o = ob.contents().assumingMemoryBound(to: Float.self)
        var worst = 0.0
        for h in 0..<heads {
            for i in 0..<queries {
                let visible = causal ? keys - queries + i + 1 : keys
                var scores = [Double](repeating: 0, count: visible)
                for j in 0..<visible {
                    var s = 0.0
                    for d in 0..<128 { s += Double(q[i * stride + h * 128 + d]) * Double(k[j * stride + h * 128 + d]) }
                    scores[j] = s / 128.0.squareRoot()
                }
                let peak = scores.max()!
                let weights = scores.map { exp($0 - peak) }
                let total = weights.reduce(0, +)
                for d in 0..<128 {
                    var acc = 0.0
                    for j in 0..<visible { acc += weights[j] * Double(v[j * stride + h * 128 + d]) }
                    worst = max(worst, abs(acc / total - Double(o[i * stride + h * 128 + d])))
                }
            }
        }
        let untouched = (queries * stride..<(rows * stride)).allSatisfy { o[$0] == 7 }
        return (worst, untouched)
    }

    func testPlain() throws {
        for (q, k) in [(32, 32), (40, 70), (100, 130), (7, 300)] {
            let r = try run(queries: q, keys: k, causal: false)
            XCTAssertLessThan(r.worst, 1e-5, "\(q)×\(k)")
            XCTAssertTrue(r.untouched, "\(q)×\(k) wrote past its queries")
        }
    }

    /// Broad softmax (small scores) catches a missed barrier: it once failed 1 run in 4 at 0.017. Large scores
    /// (hundreds) leave fp32's relative error in the exponent, the limit of any fp32 attention: bounded loosely.
    func testMagnitudes() throws {
        for (magnitude, bound): (Float, Double) in [(4, 1e-5), (16, 1e-3), (40, 1e-2)] {
            for _ in 0..<3 {
                let r = try run(queries: 64, keys: 96, causal: false, magnitude: magnitude)
                XCTAssertLessThan(r.worst, bound, "magnitude \(magnitude)")
            }
        }
    }

    func testCausal() throws {
        for (q, k) in [(24, 24), (24, 60), (45, 45), (70, 100)] {
            let r = try run(queries: q, keys: k, causal: true)
            XCTAssertLessThan(r.worst, 1e-5, "\(q)×\(k) causal")
            XCTAssertTrue(r.untouched, "\(q)×\(k) causal wrote past its queries")
        }
    }

    /// The fp16 kernels on packed K and V (the target's attention): within fp16's rounding of exact.
    func testPacked() throws {
        for kernel in ["flash_attention_packed", "flash_attention_direct"] {
            for (q, k) in [(32, 32), (40, 70), (100, 130), (7, 300)] {
                let r = try run(queries: q, keys: k, causal: false, kernel: kernel, packed: true)
                XCTAssertLessThan(r.worst, 5e-3, "\(kernel) \(q)×\(k)")
                XCTAssertTrue(r.untouched, "\(kernel) \(q)×\(k) wrote past its queries")
            }
        }
    }
}
