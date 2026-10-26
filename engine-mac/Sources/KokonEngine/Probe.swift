import Accelerate
import Foundation
import Metal
import MetalPerformanceShaders

/// **What this Mac's three matrix units can do on Qwen-Image 2.1's products**, alone and together.
///
/// A DiT step at 1024² is ~53 TFLOP, almost all in three product shapes at m = 4 096 image tokens:
/// `k × n` = 4 096 × 4 096 (q, k, v, out), 4 096 × 12 288 (MLP gate, proj), 12 288 × 4 096 (MLP out).
/// The probe times each unit on them (best of a few runs, warm), then one product cut by rows across the
/// GPU, the AMX and the Neural Engine at once. Its numbers are the ceiling the scheduler aims for; it takes
/// seconds and needs no model.
public enum Probe {
    public static let tokens = 4096
    public static let shapes: [(k: Int, n: Int, what: String)] = [
        (4096, 4096, "attention q/k/v/out"), (4096, 12288, "MLP gate/proj"), (12288, 4096, "MLP out"),
    ]

    static func flops(_ m: Int, _ k: Int, _ n: Int) -> Double { 2 * Double(m) * Double(k) * Double(n) }
    static func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) * 1e-9 }
    static func best(_ runs: Int, _ body: () throws -> Double) rethrows -> Double {
        var b = Double.infinity
        for _ in 0..<runs { b = min(b, try body()) }
        return b
    }

    public static func run() throws {
        let context = try MetalContext()
        print("device: \(context.device.name), \(ProcessInfo.processInfo.activeProcessorCount) CPU cores, "
              + "\(ProcessInfo.processInfo.physicalMemory >> 30) GB")
        print("m = \(tokens) rows (1024² image tokens) unless said; TFLOP/s, best of 5\n")

        var gpuRate: [String: Double] = [:], amxRate: [String: Double] = [:], aneRate: [String: Double] = [:]
        for (k, n, what) in shapes {
            let key = "\(k)×\(n)"
            let g32 = try gpu(context, m: tokens, k: k, n: n, type: .float32)
            let g16 = try gpu(context, m: tokens, k: k, n: n, type: .float16)
            let a = amx(m: 1024, k: k, n: n)
            let e = try? ane(rows: 2048, k: k, n: n)
            gpuRate[key] = g32; amxRate[key] = a; aneRate[key] = e ?? 0
            print(String(format: "%-30@ GPU fp32 %5.2f   GPU fp16 %5.2f   AMX fp32 %5.2f (m 1024)   ANE fp16 %@ (m 2048)",
                         "\(key) \(what)" as NSString, g32, g16, a, e.map { String(format: "%5.2f", $0) } ?? "  —  "))
        }
        print("\nNeural Engine by rows, 4096×4096 tiles:", try [512, 1024, 2048, 4096].map {
            String(format: "%d %.2f", $0, try ane(rows: $0, k: 4096, n: 4096)) }.joined(separator: ", "))

        print("\nTogether: each product cut by rows in proportion to the rates above (multiples of 256):")
        var stepAlone = 0.0, stepTogether = 0.0
        for (k, n, what) in shapes {
            let key = "\(k)×\(n)"
            let total = gpuRate[key]! + amxRate[key]! + aneRate[key]!
            func rows(_ rate: Double) -> Int { Int((Double(tokens) * rate / total / 256).rounded()) * 256 }
            let aneRows = aneRate[key]! > 0 ? max(256, rows(aneRate[key]!)) : 0
            let amxRows = max(256, rows(amxRate[key]!))
            let gpuRows = tokens - aneRows - amxRows
            let t = try coexecute(context, gpuRows: gpuRows, amxRows: amxRows, aneRows: aneRows, k: k, n: n)
            let rate = flops(tokens, k, n) / t.wall / 1e12
            print(String(format: "  %-30@ GPU %4d / AMX %4d / ANE %4d rows: %6.1f ms, %5.2f TFLOP/s (GPU alone %.2f, sum %.2f); busy GPU %.0f%%, AMX %.0f%%, ANE %.0f%%",
                         "\(key) \(what)" as NSString, gpuRows, amxRows, aneRows, t.wall * 1e3, rate, gpuRate[key]!, total,
                         100 * t.gpu / t.wall, 100 * t.amx / t.wall, 100 * t.ane / t.wall))
            let count = 32.0 * (k == n ? 4 : n > k ? 2 : 1)   // per layer: q, k, v, out; gate, proj; MLP out
            stepAlone += count * flops(tokens, k, n) / (gpuRate[key]! * 1e12)
            stepTogether += count * flops(tokens, k, n) / (rate * 1e12)
        }
        print(String(format: "\nDiT step at 1024², products only (attention adds ~9 TFLOP): %.1f s on the GPU alone, %.1f s together",
                     stepAlone, stepTogether))
    }

    // MARK: - the GPU (MPS)

    static func randomBuffer(_ context: MetalContext, count: Int, type: MPSDataType, scale: Float = 1) throws -> MTLBuffer {
        let b = try context.scratch(count * (type == .float32 ? 4 : 2), label: "probe")
        var seed: UInt64 = 0x9E37_79B9_7F4A_7C15
        for i in 0..<count {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            let v = (Float(seed >> 40) / Float(1 << 24) - 0.5) * scale
            if type == .float32 { b.contents().assumingMemoryBound(to: Float.self)[i] = v }
            else { b.contents().assumingMemoryBound(to: Float16.self)[i] = Float16(v) }
        }
        return b
    }

    static func gpuProduct(_ context: MetalContext, m: Int, k: Int, n: Int, type: MPSDataType,
                           a: MTLBuffer, b: MTLBuffer, c: MTLBuffer) -> (MTLCommandBuffer) -> Void {
        let size = type == .float32 ? 4 : 2
        let multiply = MPSMatrixMultiplication(device: context.device, transposeLeft: false, transposeRight: false,
                                               resultRows: m, resultColumns: n, interiorColumns: k, alpha: 1, beta: 0)
        let da = MPSMatrixDescriptor(rows: m, columns: k, rowBytes: k * size, dataType: type)
        let db = MPSMatrixDescriptor(rows: k, columns: n, rowBytes: n * size, dataType: type)
        let dc = MPSMatrixDescriptor(rows: m, columns: n, rowBytes: n * size, dataType: type)
        let (ma, mb, mc) = (MPSMatrix(buffer: a, descriptor: da), MPSMatrix(buffer: b, descriptor: db),
                            MPSMatrix(buffer: c, descriptor: dc))
        return { multiply.encode(commandBuffer: $0, leftMatrix: ma, rightMatrix: mb, resultMatrix: mc) }
    }

    static func gpu(_ context: MetalContext, m: Int, k: Int, n: Int, type: MPSDataType) throws -> Double {
        let a = try randomBuffer(context, count: m * k, type: type)
        let b = try randomBuffer(context, count: k * n, type: type, scale: 0.02)
        let c = try context.scratch(m * n * (type == .float32 ? 4 : 2), label: "probe c")
        let encode = gpuProduct(context, m: m, k: k, n: n, type: type, a: a, b: b, c: c)
        let seconds = try best(5) {
            let commands = context.queue.makeCommandBuffer()!
            encode(commands)
            return try MetalContext.finish(commands, "probe GEMM")
        }
        return flops(m, k, n) / seconds / 1e12
    }

    // MARK: - the AMX (Accelerate)

    static func amx(m: Int, k: Int, n: Int) -> Double {
        var a = [Float](repeating: 0, count: m * k), b = [Float](repeating: 0, count: k * n)
        var c = [Float](repeating: 0, count: m * n)
        for i in a.indices { a[i] = Float(i % 7) * 0.1 - 0.3 }
        for i in b.indices { b[i] = Float(i % 5) * 0.01 - 0.02 }
        let seconds = best(5) { () -> Double in
            let t = now()
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(m), Int32(n), Int32(k), 1, a, Int32(k),
                        b, Int32(n), 0, &c, Int32(n))
            return now() - t
        }
        return flops(m, k, n) / seconds / 1e12
    }

    // MARK: - the Neural Engine (Core ML)

    /// The product in tiles of at most 4 096 × 4 096 (the rate falls past ~4 096 of K or ~48 MB of weight):
    /// one program of the tile's shape, run once per tile.
    static func aneTiles(rows: Int, k: Int, n: Int) throws -> (program: NeuralEngineProduct, tiles: Int) {
        let (kParts, nParts) = ((k + 4095) / 4096, (n + 4095) / 4096)
        let p = try NeuralEngineProduct(rows: rows, k: k / kParts, n: n / nParts)
        p.write(p.x) { x in for i in 0..<(rows * p.k) { x[i] = Float16(Float(i % 7) * 0.1 - 0.3) } }
        p.write(p.w) { w in for i in 0..<(p.k * p.n) { w[i] = Float16(Float(i % 5) * 0.01 - 0.02) } }
        try p.run()   // compile and warm
        return (p, kParts * nParts)
    }

    static func ane(rows: Int, k: Int, n: Int) throws -> Double {
        let (p, tiles) = try aneTiles(rows: rows, k: k, n: n)
        let seconds = try best(5) { () -> Double in
            let t = now()
            for _ in 0..<tiles { try p.run() }
            return now() - t
        }
        return flops(rows, k, n) / seconds / 1e12
    }

    // MARK: - all three at once

    /// One `[m, k] × [k, n]` product, rows `[0, gpu)` on the GPU, the next `amx` on the AMX, the last `ane`
    /// on the Neural Engine, started together. fp32 on the GPU and the AMX, fp16 on the Neural Engine.
    static func coexecute(_ context: MetalContext, gpuRows: Int, amxRows: Int, aneRows: Int, k: Int, n: Int) throws
        -> (wall: Double, gpu: Double, amx: Double, ane: Double) {
        let a = try randomBuffer(context, count: tokens * k, type: .float32)
        let w = try randomBuffer(context, count: k * n, type: .float32, scale: 0.02)
        let c = try context.scratch(tokens * n * 4, label: "probe c")
        let encode = gpuProduct(context, m: gpuRows, k: k, n: n, type: .float32, a: a, b: w, c: c)
        let tiled = aneRows > 0 ? try aneTiles(rows: aneRows, k: k, n: n) : nil
        nonisolated(unsafe) let (pa, pw, pc) = (a.contents().assumingMemoryBound(to: Float.self),
                                                w.contents().assumingMemoryBound(to: Float.self),
                                                c.contents().assumingMemoryBound(to: Float.self))
        var result = (wall: Double.infinity, gpu: 0.0, amx: 0.0, ane: 0.0)
        for _ in 0..<3 {
            nonisolated(unsafe) var amxTime = 0.0, aneTime = 0.0
            let group = DispatchGroup()
            let start = now()
            DispatchQueue.global(qos: .userInitiated).async(group: group) {
                let t = now()
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(amxRows), Int32(n), Int32(k), 1,
                            pa + gpuRows * k, Int32(k), pw, Int32(n), 0, pc + gpuRows * n, Int32(n))
                amxTime = now() - t
            }
            if let tiled {
                DispatchQueue.global(qos: .userInitiated).async(group: group) {
                    let t = now()
                    for _ in 0..<tiled.tiles { try? tiled.program.run() }
                    aneTime = now() - t
                }
            }
            let commands = context.queue.makeCommandBuffer()!
            encode(commands)
            let t = now()
            try MetalContext.finish(commands, "probe co-execution")
            let gpuTime = now() - t
            group.wait()
            let wall = now() - start
            if wall < result.wall { result = (wall, gpuTime, amxTime, aneTime) }
        }
        return result
    }
}
