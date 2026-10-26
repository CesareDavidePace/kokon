import Accelerate
import Foundation

/// **Attention of some heads on the CPU** (the AMX for the two products, the cores for the softmax), fp32, on the
/// layout the products write: rows of `stride` floats, head h at columns `[128 h, 128 h + 128)`. No mask: the
/// target's queries see every key.
///
/// Work items are (head, block of `blockRows` queries), cut into one contiguous run per thread, so the bits do not
/// depend on which thread computed a block. Per item:
///
///     S = scale · Q_b · K_hᵀ            cblas_sgemm   [blockRows, keys]
///     P = exp(S − max S), l = Σ P      vDSP, vvexpf  per row
///     O_b = (P · V_h) / l               cblas_sgemm, then the rows scaled into `out`
///
/// The operands are packed first (Q_b, and K_h, V_h once per head of a run): read in place a head's rows are 16 KB
/// apart, a power of two, and land in the same cache sets (Siliconed measured a head 1.8× slower so).
enum CPUAttention {
    static let blockRows = 128, headDim = 128

    /// Heads `heads` of `queries` query rows against `keys` key rows; `q` points at the first query row.
    static func attend(q: UnsafePointer<Float>, k: UnsafePointer<Float>, v: UnsafePointer<Float>,
                       out: UnsafeMutablePointer<Float>, stride: Int, queries: Int, keys: Int, heads: Range<Int>,
                       scale: Float, threads: Int) {
        let blocks = (queries + blockRows - 1) / blockRows
        let items = heads.count * blocks
        guard items > 0 else { return }
        let threads = max(1, min(threads, items))
        nonisolated(unsafe) let (q, k, v, out) = (q, k, v, out)
        DispatchQueue.concurrentPerform(iterations: threads) { t in
            let first = items * t / threads, end = items * (t + 1) / threads
            let d = headDim
            let kh = UnsafeMutablePointer<Float>.allocate(capacity: keys * d)
            let vh = UnsafeMutablePointer<Float>.allocate(capacity: keys * d)
            let qb = UnsafeMutablePointer<Float>.allocate(capacity: blockRows * d)
            let s = UnsafeMutablePointer<Float>.allocate(capacity: blockRows * keys)
            let ob = UnsafeMutablePointer<Float>.allocate(capacity: blockRows * d)
            defer { kh.deallocate(); vh.deallocate(); qb.deallocate(); s.deallocate(); ob.deallocate() }
            var packed = -1
            for item in first..<end {
                let head = heads.lowerBound + item / blocks, block = item % blocks
                let column = head * d
                if head != packed {
                    for r in 0..<keys {
                        (kh + r * d).update(from: k + r * stride + column, count: d)
                        (vh + r * d).update(from: v + r * stride + column, count: d)
                    }
                    packed = head
                }
                let row0 = block * blockRows, rows = min(blockRows, queries - row0)
                for r in 0..<rows { (qb + r * d).update(from: q + (row0 + r) * stride + column, count: d) }
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Int32(rows), Int32(keys), Int32(d), scale,
                            qb, Int32(d), kh, Int32(d), 0, s, Int32(keys))
                var sums = [Float](repeating: 0, count: rows)
                var n = Int32(keys)
                for r in 0..<rows {
                    let row = s + r * keys
                    var peak: Float = 0
                    vDSP_maxv(row, 1, &peak, vDSP_Length(keys))
                    var shift = -peak
                    vDSP_vsadd(row, 1, &shift, row, 1, vDSP_Length(keys))
                    vvexpf(row, row, &n)
                    vDSP_sve(row, 1, &sums[r], vDSP_Length(keys))
                }
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(rows), Int32(d), Int32(keys), 1,
                            s, Int32(keys), vh, Int32(d), 0, ob, Int32(d))
                for r in 0..<rows {
                    var inverse = 1 / sums[r]
                    vDSP_vsmul(ob + r * d, 1, &inverse, out + (row0 + r) * stride + column, 1, vDSP_Length(d))
                }
            }
        }
    }
}
