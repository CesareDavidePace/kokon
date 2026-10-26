import Foundation
import Metal

/// **The GPU's rows of a product on its neural accelerators** (`Kernels/metal4/na_matmul.metal`: Metal 4's tensor
/// operations, MetalPerformancePrimitives' `matmul2d`). On the M5 (Apple10 GPUs) they run on the matrix units each GPU
/// core gained; on older Metal 4 GPUs the same source runs on the shader cores, so its results can be checked on any
/// Mac with macOS 26 or later (`swift test`, `check-na`), only not its speed.
///
/// Selected for the split products' GPU rows wherever Metal 4 runs (fp16 A and weights, the defaults), MPS when the
/// source does not compile or with `KOKON_NA=0`. On the M2 Pro's shader cores it beats MPS too: 6.3 against 5.6
/// TFLOP/s at 1536 × 4096 × 4096, bit for bit the same result (`check-na`), and per evaluation at 1024², 3 alternated
/// rounds of 4, 7.57 → 7.48 s (2026-10-25). Not yet run on an M5, M1, M3 or M4; the attention stays on the engine's
/// own kernels.
public final class NeuralAccelerators: @unchecked Sendable {
    let pipeline: MTLComputePipelineState
    /// Why `make` returned nil, for the bench's report.
    nonisolated(unsafe) public private(set) static var unavailable: String?

    struct Params { var m, n, k: UInt32 }

    /// Compiles the kernel; nil without Metal 4 or when the compiler refuses the source.
    public static func make(device: MTLDevice) -> NeuralAccelerators? {
        guard #available(macOS 26, *), device.supportsFamily(.metal4) else {
            unavailable = "no Metal 4 on this GPU or macOS"
            return nil
        }
        do {
            guard let url = EngineBundle.kernels?
                .appendingPathComponent("metal4/na_matmul.metal") else { throw MetalContext.Failure.noKernelSource }
            let options = MTLCompileOptions()
            options.languageVersion = .version4_0
            options.mathMode = .safe
            let library = try device.makeLibrary(source: try String(contentsOf: url, encoding: .utf8), options: options)
            guard let function = library.makeFunction(name: "na_matmul") else { throw MetalContext.Failure.kernel("na_matmul not found") }
            return NeuralAccelerators(pipeline: try device.makeComputePipelineState(function: function))
        } catch {
            unavailable = "the kernel did not compile: \(error.localizedDescription)"
            return nil
        }
    }

    init(pipeline: MTLComputePipelineState) { self.pipeline = pipeline }

    /// Whether the split products' GPU rows go through them on this device.
    public static func selected(device: MTLDevice) -> Bool {
        switch ProcessInfo.processInfo.environment["KOKON_NA"] {
        case "0": return false
        case "1": return true
        default: if #available(macOS 26, *) { return device.supportsFamily(.metal4) } else { return false }
        }
    }

    /// Whether a product fits the kernel (no edge tiles).
    static func fits(m: Int, n: Int, k: Int) -> Bool { m % 64 == 0 && n % 64 == 0 && m > 0 && n > 0 && k > 0 }

    /// Encodes `C[m, n] = A[m, k] · B[k, n]`, A and B fp16, C fp32, dense rows.
    func encode(a: (MTLBuffer, Int), b: (MTLBuffer, Int), c: (MTLBuffer, Int), m: Int, n: Int, k: Int,
                commands: MTLCommandBuffer) throws {
        guard NeuralAccelerators.fits(m: m, n: n, k: k), let e = commands.makeComputeCommandEncoder() else {
            throw MetalContext.Failure.kernel("na_matmul \(m)×\(n)×\(k)")
        }
        e.label = "na_matmul"
        e.setComputePipelineState(pipeline)
        e.setBuffer(a.0, offset: a.1, index: 0)
        e.setBuffer(b.0, offset: b.1, index: 1)
        e.setBuffer(c.0, offset: c.1, index: 2)
        var p = Params(m: UInt32(m), n: UInt32(n), k: UInt32(k))
        e.setBytes(&p, length: MemoryLayout<Params>.stride, index: 3)
        e.dispatchThreadgroups(MTLSize(width: n / 64, height: m / 64, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: pipeline.threadExecutionWidth * 4, height: 1, depth: 1))
        e.endEncoding()
    }
}
