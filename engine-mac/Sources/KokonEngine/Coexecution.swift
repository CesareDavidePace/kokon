import Accelerate
import Foundation
import Metal

/// **One product on the GPU, the AMX and the Neural Engine at the same time**, cut by rows:
///
///     C[m, n] = A[m, k] · W[k, n]       W already unpacked to fp32 [k, n] (the reserve) by the GPU
///       rows [0, g)            GPU, MPS fp32
///       rows [g, g + a)        AMX, cblas_sgemm fp32 on the same reserve (unified memory)
///       rows [g + a, m)        Neural Engine, fp16 in tiles of 4 096 × 4 096 (`ane.metal` packs and unpacks)
///
/// GPU timeline per product p: unpack W → pack the Neural Engine's tiles → signal `ready` = p → its own rows →
/// wait `amxDone` ≥ p and `aneDone` ≥ p → unpack the Neural Engine's rows. The two CPU workers wait on `ready`,
/// compute, signal. Each output row is computed by exactly one unit, so the split never reorders a sum.
///
/// Nothing waits on the CPU while the GPU runs: the GPU encodes ahead, the workers follow the events. A reserve
/// or a surface is rewritten only after the GPU has waited for the product that read it.
final class Coexecutor {
    struct Split: CustomStringConvertible { var gpu: Int, amx: Int, ane: Int
        var description: String { "GPU \(gpu) / AMX \(amx) / ANE \(ane) rows" }
    }

    static let tile = 4096
    let context: MetalContext
    let aneRows: Int
    private let slots: [(product: NeuralEngineProduct, x: MTLBuffer, w: MTLBuffer, y: MTLBuffer)]
    private let ready: MTLSharedEvent, amxDone: MTLSharedEvent, aneDone: MTLSharedEvent
    private var counter: UInt64 = 0
    private let amxWorker = Worker(name: "kokon.amx"), aneWorker = Worker(name: "kokon.ane")
    /// λ (12 288), lifted norms (12 288), L, σ (aneRows): the Neural Engine's scales.
    private let scales: MTLBuffer
    /// The share of rows each unit takes; `KOKON_SPLIT=gpu,amx,ane` (fractions) overrides.
    var amxShare: Double, aneEnabled: Bool, amxEnabled: Bool
    /// Seconds each worker spent computing (not waiting), and failures (a failure leaves rows unwritten).
    private(set) var amxSeconds = 0.0, aneSeconds = 0.0
    private(set) var failures: [String] = []
    private let lock = NSLock()

    /// Defaults measured on the M2 Pro 16 GB, 1024² + 32 text rows (`time-dit`, 2026-10-25): AMX 512 rows 12.8 s per
    /// evaluation (the AMX the slowest, ~35 % under its solo rate beside the others), 256 rows 11.3 s, none 11.8 s;
    /// GPU alone 16.2 s. A balancing loop per shape replaces the fixed share later.
    init(context: MetalContext, aneRows: Int = 2048, amxShare: Double = 0.0625, ane: Bool = true, amx: Bool = true) throws {
        self.context = context
        self.aneRows = aneRows
        self.amxShare = amxShare
        self.aneEnabled = ane
        self.amxEnabled = amx
        let device = context.device
        guard let r = device.makeSharedEvent(), let a = device.makeSharedEvent(), let e = device.makeSharedEvent() else {
            throw MetalContext.Failure.buffer("shared events")
        }
        (ready, amxDone, aneDone) = (r, a, e)
        scales = try context.scratch((2 * 12288 + 16 + aneRows) * 4, label: "ane scales")
        slots = ane ? try (0..<3).map { _ in
            let p = try NeuralEngineProduct(rows: aneRows, k: Coexecutor.tile, n: Coexecutor.tile)
            try p.run()   // compile and warm
            return (p, try p.gpuBuffer(p.x, device: device), try p.gpuBuffer(p.w, device: device), try p.gpuBuffer(p.y, device: device))
        } : []
    }

    /// Whether a product takes the split: the Neural Engine's tiles need K and N in multiples of 4 096 with at
    /// most 3 tiles; below 1 024 rows the split costs more than it brings.
    func applies(m: Int, k: Int, n: Int) -> Bool {
        m >= 1024 && k % Coexecutor.tile == 0 && n % Coexecutor.tile == 0 && (k / Coexecutor.tile) * (n / Coexecutor.tile) <= 3
    }

    func split(m: Int) -> Split {
        let ane = aneEnabled && m >= aneRows + 512 ? aneRows : 0
        let amx = amxEnabled ? Int((Double(m) * amxShare / 256).rounded()) * 256 : 0
        return Split(gpu: m - ane - amx, amx: amx, ane: ane)
    }

    struct Shape { var k, n, k0, kPart, n0, nPart, row0, rows, kParts: UInt32 }

    /// Encodes the split product. `gpuRows` encodes the GPU's own rows `[0, g)` into the command buffer.
    func encode(a: (buffer: MTLBuffer, offset: Int, pointer: UnsafeMutablePointer<Float>),
                w: (buffer: MTLBuffer, offset: Int, pointer: UnsafeMutablePointer<Float>),
                c: (buffer: MTLBuffer, offset: Int, pointer: UnsafeMutablePointer<Float>),
                m: Int, k: Int, n: Int, commands: MTLCommandBuffer, gpuRows: (Int) throws -> Void) throws {
        let s = split(m: m)
        counter += 1
        let p = counter
        let (kParts, nParts) = (k / Coexecutor.tile, n / Coexecutor.tile)
        let lambda = 0, norm = 12288 * 4, state = 2 * 12288 * 4, sigma = 2 * 12288 * 4 + 64
        func shape(kp: Int = 0, np: Int = 0) -> Shape {
            Shape(k: UInt32(k), n: UInt32(n), k0: UInt32(kp * Coexecutor.tile), kPart: UInt32(Coexecutor.tile),
                  n0: UInt32(np * Coexecutor.tile), nPart: UInt32(Coexecutor.tile), row0: UInt32(s.gpu + s.amx),
                  rows: UInt32(s.ane), kParts: UInt32(kParts))
        }
        func kernel(_ name: String, _ body: (MTLComputeCommandEncoder) throws -> Void) throws {
            let e = commands.makeComputeCommandEncoder()!
            e.label = name
            e.setComputePipelineState(try context.pipeline(name))
            try body(e)
            e.endEncoding()
        }
        if s.ane > 0 {
            var sh = shape()
            try kernel("ane_columns") { e in
                e.setBuffer(w.buffer, offset: w.offset, index: 0)
                e.setBuffer(scales, offset: lambda, index: 1); e.setBuffer(scales, offset: norm, index: 2)
                e.setBytes(&sh, length: MemoryLayout<Shape>.stride, index: 3)
                e.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            }
            try kernel("ane_norm") { e in
                e.setBuffer(scales, offset: norm, index: 0); e.setBuffer(scales, offset: state, index: 1)
                e.setBytes(&sh, length: MemoryLayout<Shape>.stride, index: 2)
                e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
            }
            try kernel("ane_row_scales") { e in
                e.setBuffer(a.buffer, offset: a.offset, index: 0); e.setBuffer(scales, offset: state, index: 1)
                e.setBuffer(scales, offset: sigma, index: 2)
                e.setBytes(&sh, length: MemoryLayout<Shape>.stride, index: 3)
                e.dispatchThreadgroups(MTLSize(width: s.ane, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
            for kp in 0..<kParts {
                for np in 0..<nParts {
                    let slot = slots[kp * nParts + np]
                    var t = shape(kp: kp, np: np)
                    try kernel("ane_pack_weight") { e in
                        e.setBuffer(w.buffer, offset: w.offset, index: 0); e.setBuffer(scales, offset: lambda, index: 1)
                        e.setBuffer(slot.w, offset: 0, index: 2)
                        e.setBytes(&t, length: MemoryLayout<Shape>.stride, index: 3)
                        e.dispatchThreads(MTLSize(width: Coexecutor.tile, height: Coexecutor.tile, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 64, height: 4, depth: 1))
                    }
                    try kernel("ane_pack_rows") { e in
                        e.setBuffer(a.buffer, offset: a.offset, index: 0); e.setBuffer(scales, offset: sigma, index: 1)
                        e.setBuffer(slot.x, offset: 0, index: 2)
                        e.setBytes(&t, length: MemoryLayout<Shape>.stride, index: 3)
                        e.dispatchThreads(MTLSize(width: Coexecutor.tile, height: s.ane, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: 64, height: 4, depth: 1))
                    }
                }
            }
        }
        if s.amx > 0 || s.ane > 0 { commands.encodeSignalEvent(ready, value: p) }
        if s.amx > 0 {
            nonisolated(unsafe) let (pa, pw, pc) = (a.pointer + s.gpu * k, w.pointer, c.pointer + s.gpu * n)
            let rows = s.amx
            amxWorker.submit { [self] in
                defer { amxDone.signaledValue = p }
                guard wait(ready, p, "AMX") else { return }
                let t = DispatchTime.now().uptimeNanoseconds
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(rows), Int32(n), Int32(k), 1,
                            pa, Int32(k), pw, Int32(n), 0, pc, Int32(n))
                lock.withLock { amxSeconds += Double(DispatchTime.now().uptimeNanoseconds - t) * 1e-9 }
            }
        }
        if s.ane > 0 {
            let tiles = Array(slots.prefix(kParts * nParts).map(\.product))
            aneWorker.submit { [self] in
                defer { aneDone.signaledValue = p }
                guard wait(ready, p, "ANE") else { return }
                let t = DispatchTime.now().uptimeNanoseconds
                do { for tile in tiles { try tile.run() } } catch { fail("ANE: \(error)") }
                lock.withLock { aneSeconds += Double(DispatchTime.now().uptimeNanoseconds - t) * 1e-9 }
            }
        }
        try gpuRows(s.gpu)
        if s.amx > 0 { commands.encodeWaitForEvent(amxDone, value: p) }
        if s.ane > 0 {
            commands.encodeWaitForEvent(aneDone, value: p)
            for np in 0..<nParts {
                var t = shape(np: np)
                let ys = (0..<3).map { kp in slots[min(kp, kParts - 1) * nParts + np].y }
                try kernel("ane_unpack") { e in
                    for (i, y) in ys.enumerated() { e.setBuffer(y, offset: 0, index: i) }
                    e.setBuffer(scales, offset: lambda, index: 3); e.setBuffer(scales, offset: sigma, index: 4)
                    e.setBuffer(c.buffer, offset: c.offset, index: 5)
                    e.setBytes(&t, length: MemoryLayout<Shape>.stride, index: 6)
                    e.dispatchThreads(MTLSize(width: Coexecutor.tile, height: s.ane, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 64, height: 4, depth: 1))
                }
            }
        }
    }

    private func fail(_ message: String) { lock.withLock { failures.append(message) } }

    /// Waits for the GPU to reach `value`; false (and a failure) after a minute, so a broken command buffer never
    /// hangs a worker for good.
    private func wait(_ event: MTLSharedEvent, _ value: UInt64, _ who: String) -> Bool {
        if event.wait(untilSignaledValue: value, timeoutMS: 60_000) { return true }
        fail("\(who): the GPU never reached product \(value)")
        return false
    }

    func resetTimings() { lock.withLock { amxSeconds = 0; aneSeconds = 0 } }
}

/// A thread running submitted jobs in order.
final class Worker: @unchecked Sendable {
    private var jobs: [() -> Void] = []
    private let lock = NSLock(), available = DispatchSemaphore(value: 0)

    init(name: String) {
        let thread = Thread { [unowned self] in
            while true {
                available.wait()
                let job = lock.withLock { jobs.removeFirst() }
                job()
            }
        }
        thread.name = name
        thread.qualityOfService = .userInteractive
        thread.start()
    }

    func submit(_ job: @escaping () -> Void) {
        lock.withLock { jobs.append(job) }
        available.signal()
    }
}
