import Foundation

/// The ggml tensor types Kokon's files use: the DiT (Q4_K + BF16; the unsloth Q4_K_M adds Q5_K, Q6_K,
/// Q8_0, F32), the text encoder (Q4_K_M or Q8_0) and the vision projector (F16). Raw values are ggml's.
public enum QuantType: UInt32, Sendable, CaseIterable {
    case f32 = 0, f16 = 1, q4_0 = 2, q8_0 = 8, q4_K = 12, q5_K = 13, q6_K = 14, bf16 = 30

    /// Values per block along the contiguous dimension.
    public var blockValues: Int {
        switch self {
        case .f32, .f16, .bf16: return 1
        case .q4_0, .q8_0: return 32
        case .q4_K, .q5_K, .q6_K: return 256
        }
    }

    /// Bytes per block (`block_q4_K` and friends in ggml-common.h).
    public var blockBytes: Int {
        switch self {
        case .f32: return 4
        case .f16, .bf16: return 2
        case .q4_0: return 18     // d, qs[16]
        case .q8_0: return 34     // d, qs[32]
        case .q4_K: return 144    // d, dmin, scales[12], qs[128]
        case .q5_K: return 176    // d, dmin, scales[12], qh[32], qs[128]
        case .q6_K: return 210    // ql[128], qh[64], scales[16], d
        }
    }

    public func bytes(count: Int) -> Int { count / blockValues * blockBytes }

    public var name: String {
        switch self {
        case .f32: return "F32"
        case .f16: return "F16"
        case .bf16: return "BF16"
        case .q4_0: return "Q4_0"
        case .q8_0: return "Q8_0"
        case .q4_K: return "Q4_K"
        case .q5_K: return "Q5_K"
        case .q6_K: return "Q6_K"
        }
    }
}

/// The CPU reference: ggml's `dequantize_row_*`, value for value. The GPU kernels (`Kernels/dequant.metal`)
/// are tested against it, and it serves the small tensors (norm weights, modulation) the CPU reads.
public enum Dequant {
    /// `count` values (a whole number of blocks) of `type` from `source` into `out`, in fp32.
    public static func row(_ source: UnsafeRawPointer, type: QuantType, count: Int, into out: UnsafeMutablePointer<Float>) {
        let blocks = count / type.blockValues
        for b in 0..<blocks {
            block(source + b * type.blockBytes, type: type, into: out + b * type.blockValues)
        }
    }

    /// A whole tensor `[rows, columns]` (ggml rows, `columns` contiguous) into `out`, on every core.
    public static func tensor(_ file: GGUFFile, _ tensor: GGUFTensor, into out: UnsafeMutablePointer<Float>) throws {
        guard let type = tensor.type else {
            throw GGUFFile.Failure.unsupported("\(tensor.name): ggml type \(tensor.typeID)")
        }
        let rows = tensor.rows, columns = tensor.columns
        let rowBytes = type.bytes(count: columns)
        nonisolated(unsafe) let (src, dst) = (file.data(tensor), out)
        let chunks = min(rows, 64)
        DispatchQueue.concurrentPerform(iterations: chunks) { chunk in
            for r in (chunk * rows / chunks)..<((chunk + 1) * rows / chunks) {
                row(src + r * rowBytes, type: type, count: columns, into: dst + r * columns)
            }
        }
    }

    /// A whole tensor as a Swift array (small tensors, tests).
    public static func array(_ file: GGUFFile, _ name: String) throws -> [Float] {
        let t = try file.tensor(name)
        var out = [Float](repeating: 0, count: t.count)
        try out.withUnsafeMutableBufferPointer { try tensor(file, t, into: $0.baseAddress!) }
        return out
    }

    @inline(__always) static func half(_ p: UnsafeRawPointer, _ at: Int) -> Float {
        Float(Float16(bitPattern: p.loadUnaligned(fromByteOffset: at, as: UInt16.self)))
    }

    /// `get_scale_min_k4`: the 6-bit scale and min of sub-block `j` (0..<8) of a Q4_K/Q5_K block.
    @inline(__always) static func scaleMin(_ j: Int, _ q: UnsafePointer<UInt8>) -> (Float, Float) {
        if j < 4 { return (Float(q[j] & 63), Float(q[j + 4] & 63)) }
        return (Float((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)), Float((q[j + 4] >> 4) | ((q[j] >> 6) << 4)))
    }

    /// One block of `type` into `blockValues` floats.
    public static func block(_ x: UnsafeRawPointer, type: QuantType, into y: UnsafeMutablePointer<Float>) {
        let bytes = x.assumingMemoryBound(to: UInt8.self)
        switch type {
        case .f32:
            y[0] = x.loadUnaligned(as: Float.self)
        case .f16:
            y[0] = half(x, 0)
        case .bf16:
            y[0] = Float(bitPattern: UInt32(x.loadUnaligned(as: UInt16.self)) << 16)
        case .q4_0:
            let d = half(x, 0), qs = bytes + 2
            for j in 0..<16 {
                y[j] = Float(Int(qs[j] & 0xF) - 8) * d
                y[j + 16] = Float(Int(qs[j] >> 4) - 8) * d
            }
        case .q8_0:
            let d = half(x, 0)
            for j in 0..<32 { y[j] = Float(Int8(bitPattern: bytes[2 + j])) * d }
        case .q4_K:
            let d = half(x, 0), dmin = half(x, 2), scales = bytes + 4
            var q = bytes + 16, out = y
            for s in stride(from: 0, to: 8, by: 2) {
                let (sc1, m1) = scaleMin(s, scales), (sc2, m2) = scaleMin(s + 1, scales)
                let (d1, n1, d2, n2) = (d * sc1, dmin * m1, d * sc2, dmin * m2)
                for l in 0..<32 { out[l] = d1 * Float(q[l] & 0xF) - n1 }
                for l in 0..<32 { out[32 + l] = d2 * Float(q[l] >> 4) - n2 }
                q += 32; out += 64
            }
        case .q5_K:
            let d = half(x, 0), dmin = half(x, 2), scales = bytes + 4, qh = bytes + 16
            var ql = bytes + 48, out = y
            var u1: UInt8 = 1, u2: UInt8 = 2
            for s in stride(from: 0, to: 8, by: 2) {
                let (sc1, m1) = scaleMin(s, scales), (sc2, m2) = scaleMin(s + 1, scales)
                let (d1, n1, d2, n2) = (d * sc1, dmin * m1, d * sc2, dmin * m2)
                for l in 0..<32 { out[l] = d1 * Float(Int(ql[l] & 0xF) + (qh[l] & u1 != 0 ? 16 : 0)) - n1 }
                for l in 0..<32 { out[32 + l] = d2 * Float(Int(ql[l] >> 4) + (qh[l] & u2 != 0 ? 16 : 0)) - n2 }
                ql += 32; out += 64; u1 <<= 2; u2 <<= 2
            }
        case .q6_K:
            let d = half(x, 208)
            var ql = bytes, qh = bytes + 128, out = y
            var sc = UnsafeRawPointer(bytes + 192).assumingMemoryBound(to: Int8.self)
            for _ in 0..<2 {
                for l in 0..<32 {
                    let i = l / 16
                    let q1 = Int((ql[l] & 0xF) | ((qh[l] >> 0) & 3) << 4) - 32
                    let q2 = Int((ql[l + 32] & 0xF) | ((qh[l] >> 2) & 3) << 4) - 32
                    let q3 = Int((ql[l] >> 4) | ((qh[l] >> 4) & 3) << 4) - 32
                    let q4 = Int((ql[l + 32] >> 4) | ((qh[l] >> 6) & 3) << 4) - 32
                    out[l] = d * Float(sc[i]) * Float(q1)
                    out[l + 32] = d * Float(sc[i + 2]) * Float(q2)
                    out[l + 64] = d * Float(sc[i + 4]) * Float(q3)
                    out[l + 96] = d * Float(sc[i + 6]) * Float(q4)
                }
                out += 128; ql += 64; qh += 32; sc += 8
            }
        }
    }
}
