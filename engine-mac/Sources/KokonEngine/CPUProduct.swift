import Accelerate

/// **Products on the CPU's matrix unit (AMX)**, row-major, contiguous rows: `C[m, n] = α · A[m, k] · B`, B `[k, n]`
/// or, `transposedB`, `[n, k]`; C fp32, A and B fp32 or fp16 (`halfA`, `halfB`).
///
/// BNNS rather than `cblas_sgemm`: on the DiT's AMX shares, both fp32 and exact to 4e-6, BNNS runs 1.3–1.9× faster
/// (M2 Pro, macOS 27, 2026-10-25: 448 × 4096 × 4096 8.0 ms against 14.8 ms, 448 × 4096 × 12288 23.7 against ~40,
/// 320 × 12288 × 4096 20.2 against 25.8). `BNNSMatMul` is deprecated since macOS 15 for BNNSGraph, which compiles
/// a whole graph ahead; for one product with run-time operands this call is still the direct way.
enum CPUProduct {
    static func multiply(_ a: UnsafeRawPointer, _ b: UnsafeRawPointer, into c: UnsafeMutablePointer<Float>,
                         m: Int, k: Int, n: Int, alpha: Float = 1, transposedB: Bool = false, halfA: Bool = false, halfB: Bool = false,
                         threads: Int? = nil) -> Bool {
        func matrix(_ p: UnsafeRawPointer, rows: Int, columns: Int, type: BNNSDataType = .float) -> BNNSNDArrayDescriptor {
            var d = BNNSNDArrayDescriptor()
            d.layout = BNNSDataLayoutRowMajorMatrix
            d.size = (columns, rows, 0, 0, 0, 0, 0, 0)
            d.data = UnsafeMutableRawPointer(mutating: p)
            d.data_type = type
            d.data_scale = 1
            return d
        }
        var da = matrix(a, rows: m, columns: k, type: halfA ? .float16 : .float)
        let typeB: BNNSDataType = halfB ? .float16 : .float
        var db = transposedB ? matrix(b, rows: n, columns: k, type: typeB) : matrix(b, rows: k, columns: n, type: typeB)
        var dc = matrix(c, rows: m, columns: n)
        var parameters = BNNSFilterParameters()
        parameters.n_threads = threads ?? 0
        let size = BNNSMatMulWorkspaceSize(false, transposedB, alpha, &da, &db, &dc, &parameters)
        guard size >= 0 else { return false }
        let workspace = size > 0 ? UnsafeMutableRawPointer.allocate(byteCount: size, alignment: 64) : nil
        defer { workspace?.deallocate() }
        return BNNSMatMul(false, transposedB, alpha, &da, &db, &dc, workspace, &parameters) == 0
    }
}
