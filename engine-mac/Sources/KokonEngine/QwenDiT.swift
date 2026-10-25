import Accelerate
import Foundation
import Metal
import MetalPerformanceShaders

/// **The Qwen-Image 2.1 DiT** (`QwenImage21Transformer2DModel`, diffusers 0.42) on the GPU, fp32 activations, the
/// GGUF weights unpacked on the GPU just before each product.
///
///     encoder rows ─ txt_in (zero-centred RMSNorm, Linear, GELU tanh, Linear) ─┐
///     latent tokens ─ img_in ─────────────────────────────────────────────────┴ joint sequence x
///     σ ─ sinusoid [cos | sin] ─ timestep MLP ─ temb ─ SiLU ─ modulation.1 → [scale₁, gate₁, scale₂, gate₂]
///                                                    └ SiLU ─ norm_out.linear → final scale
///     32 × { x += tanh g₁ · to_out(SDPA(RoPE(RMS(q)), RoPE(RMS(k)), v))   on LN(x)·(1 + s₁)
///            x += tanh g₂ · out(SiLU(gate) ⊙ proj)                         on LN(x)·(1 + s₂) }
///     LN(x)·(1 + s_final) ─ proj_out → velocity of the target rows
///
/// Text and condition images read the modulation of t = 0, the target the one of σ (`causal_condition`).
/// Attention is block-causal: each prefix segment sees the keys before its end (a text segment with a causal
/// triangle), the target sees every key.
///
/// This first version runs the full pass every evaluation (no K/V cache of the prefix yet) on the GPU alone;
/// the AMX and the Neural Engine join in the products later.
public final class QwenDiT {
    public static let dim = 4096, heads = 32, headDim = 128, hidden = 12288, layers = 32, channels = 64
    static let eps: Float = 1e-6
    static let prefix = "model.diffusion_model."

    /// **Every product is submitted on a multiple of 64 rows, at least 64.** One MPS fp32 product of 8 to 32
    /// rows slows every later MPS product of the process: a 1024² evaluation with 32 text rows took 18.5 s of
    /// products against 11.4–11.6 s with 0 or 64 (M2 Pro, macOS 27, `time-dit`, 2026-10-25; Siliconed found the
    /// same on an M1 Pro). The padding rows stay zero: row kernels cover the real rows only, and the product of a
    /// zero row is zero. Slices therefore hold `padded(rows) + 64` rows.
    static func padded(_ rows: Int) -> Int { max(64, (rows + 63) / 64 * 64) }

    public let file: GGUFFile
    let context: MetalContext
    let dequant: GPUDequant
    public let sequence: QwenSequence
    let arena: Arena

    // Arena slices, rows of the joint sequence (N = prefix + target).
    let x, normed, q, k, v, attended, o, h1, h2: Arena.Slice
    let reserves: [Arena.Slice]
    let rope, modulation, qkNorms, textNorm, textIn, textOut, latent, output: Arena.Slice
    private var nextReserve = 0
    private var products: [String: MPSMatrixMultiplication] = [:]
    /// The AMX and the Neural Engine beside the GPU on the large products and the target's attention
    /// (`Coexecution.swift`, `AttentionSplit.swift`); nil: GPU alone. `KOKON_COEXEC=0` turns it off, `KOKON_ANE=0` /
    /// `KOKON_AMX=0` one unit, `KOKON_ANE_ATTENTION=1` heads on the Neural Engine too, `KOKON_BALANCE=0` the balancing loop.
    let coexecutor: Coexecutor?
    private var attentions: [String: GraphAttention] = [:]
    /// Seconds per phase of the last evaluation.
    public private(set) var timings: [String: Double] = [:]

    public init(file: GGUFFile, context: MetalContext, sequence: QwenSequence) throws {
        self.file = file
        self.context = context
        self.dequant = GPUDequant(context: context)
        self.sequence = sequence
        let d = QwenDiT.dim
        let n = QwenDiT.padded(sequence.count) + 64, s = QwenDiT.padded(sequence.target) + 64
        let t = QwenDiT.padded(sequence.textRows.count)
        let sizes = [n * d, n * d, n * d, n * d, n * d, n * d, n * d, n * QwenDiT.hidden, n * QwenDiT.hidden,
                     d * QwenDiT.hidden, d * QwenDiT.hidden,
                     n * 128, 2 * 5 * d, QwenDiT.layers * 2 * QwenDiT.headDim, d, t * d, t * d,
                     s * QwenDiT.channels, s * QwenDiT.channels]
        let arena = try Arena(device: context.device, bytes: Arena.bytes(sizes.map { $0 * 4 }))
        self.arena = arena
        var slices = try sizes.map { try arena.slice(floats: $0) }.makeIterator()
        x = slices.next()!; normed = slices.next()!; q = slices.next()!; k = slices.next()!; v = slices.next()!
        attended = slices.next()!; o = slices.next()!; h1 = slices.next()!; h2 = slices.next()!
        reserves = [slices.next()!, slices.next()!]
        rope = slices.next()!; modulation = slices.next()!; qkNorms = slices.next()!; textNorm = slices.next()!
        textIn = slices.next()!; textOut = slices.next()!; latent = slices.next()!; output = slices.next()!

        let environment = ProcessInfo.processInfo.environment
        if environment["KOKON_COEXEC"] != "0", sequence.count >= 1024 {
            coexecutor = try Coexecutor(context: context, rows: QwenDiT.padded(sequence.count),
                                        ane: environment["KOKON_ANE"] != "0", amx: environment["KOKON_AMX"] != "0",
                                        aneAttention: environment["KOKON_ANE_ATTENTION"] == "1")
        } else {
            coexecutor = nil
        }
        if let coexecutor, environment["KOKON_BALANCE"] != "0" { try coexecutor.recall() }
        if let coexecutor, coexecutor.attentionApplies(queries: sequence.target) {
            try coexecutor.prepareAttention(queries: sequence.target, keys: sequence.count)
        }
        let table = sequence.ropeTable()
        arena.pointer(rope).update(from: table, count: table.count)
        // Small weights unpacked once on the CPU: the QK norms of every layer, txt_in's norm as (w + 1).
        for layer in 0..<QwenDiT.layers {
            for (i, which) in ["norm_q", "norm_k"].enumerated() {
                let w = try Dequant.array(file, QwenDiT.prefix + "transformer_blocks.\(layer).attn.\(which).weight")
                arena.pointer(qkNorms, floatOffset: (layer * 2 + i) * QwenDiT.headDim).update(from: w, count: QwenDiT.headDim)
            }
        }
        let tn = try Dequant.array(file, QwenDiT.prefix + "txt_in.text_norm.weight")
        for j in 0..<d { arena.pointer(textNorm)[j] = tn[j] + 1 }
    }

    // MARK: - modulation (CPU, fp32)

    /// `[cos | sin]` of `1000 · t · ω_j`, `ω_j = exp(−ln(10⁴) · j / 128)`: QwenImage21TemporalTimesteps in fp32.
    static func sinusoid(_ t: Float) -> [Float] {
        let half = 128
        var out = [Float](repeating: 0, count: 2 * half)
        let scaled = 1000 * t
        for j in 0..<half {
            let omega = expf(-Float(log(10000.0)) * Float(j) / Float(half))
            let angle = scaled * omega
            out[j] = cosf(angle)
            out[half + j] = sinf(angle)
        }
        return out
    }

    static func silu(_ x: [Float]) -> [Float] { x.map { $0 / (1 + expf(-$0)) } }

    /// `y = W x` for each `x`, W a Linear weight `[out, in]` of the file unpacked once on the CPU.
    func linearCPU(_ name: String, _ xs: [[Float]]) throws -> [[Float]] {
        let t = try file.tensor(QwenDiT.prefix + name)
        let w = try Dequant.array(file, t.name)
        return xs.map { x in
            var y = [Float](repeating: 0, count: t.rows)
            cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(t.rows), Int32(t.columns), 1, w, Int32(t.columns), x, 1, 0, &y, 1)
            return y
        }
    }

    /// The stored modulation rows of each timestep seen (`modulationRows`): unpacking the four weights takes ~0.1 s
    /// (`modulation.1` is 268 MB in fp32), so the pipeline prepares all its timesteps at once.
    private var modulationCache: [Float: [Float]] = [:]

    /// Computes and keeps the modulation rows of `timesteps` not yet seen.
    public func prepareModulation(_ timesteps: [Float]) throws {
        let missing = Array(Set(timesteps.filter { modulationCache[$0] == nil }))
        guard !missing.isEmpty else { return }
        for (t, row) in zip(missing, try modulationRows(timesteps: missing).stored) { modulationCache[t] = row }
    }

    /// The temb, block modulation and final scale of each timestep, as stored for the kernels:
    /// `[1 + s₁, tanh g₁, 1 + s₂, tanh g₂, 1 + s_final]`, `dim` floats each.
    public func modulationRows(timesteps: [Float]) throws -> (temb: [[Float]], raw: [[Float]], stored: [[Float]]) {
        let d = QwenDiT.dim
        var raws: [[Float]] = [], stored: [[Float]] = []
        let h = try linearCPU("time_text_embed.timestep_embedder.linear_1.weight", timesteps.map(QwenDiT.sinusoid)).map(QwenDiT.silu)
        let tembs = try linearCPU("time_text_embed.timestep_embedder.linear_2.weight", h)
        let active = tembs.map(QwenDiT.silu)
        let ms = try linearCPU("modulation.1.weight", active), fs = try linearCPU("norm_out.linear.weight", active)
        for (m, f) in zip(ms, fs) {
            var row = [Float](repeating: 0, count: 5 * d)
            for j in 0..<d {
                row[j] = 1 + m[j]
                row[d + j] = tanhf(m[d + j])
                row[2 * d + j] = 1 + m[2 * d + j]
                row[3 * d + j] = tanhf(m[3 * d + j])
                row[4 * d + j] = 1 + f[j]
            }
            raws.append(m); stored.append(row)
        }
        return (tembs, raws, stored)
    }

    // MARK: - one evaluation

    /// Called with a stage's name and its rows (CPU pointer, floats) when `probe` is set; the pass then waits on
    /// the GPU at each stage. Stages: joint_in, l0_* (inside layer 0), layer0_out … layer31_out, norm_out, model_out.
    public typealias Probe = (String, UnsafePointer<Float>, Int) -> Void

    /// One evaluation: `encoder` the encoder's last layer `[T, 4096]` (all positions, image slots included),
    /// `target` the target latent `[S, 64]` as packed tokens, `sigma` in [0, 1]. Returns the velocity `[S, 64]`.
    public func forward(encoder: [Float], target: [Float], sigma: Float, probe: Probe? = nil) throws -> [Float] {
        let d = QwenDiT.dim, n = sequence.count, p = sequence.prefix, s = sequence.target
        let started = DispatchTime.now().uptimeNanoseconds
        timings = [:]
        try prepareModulation([0, sigma])
        arena.pointer(modulation).update(from: modulationCache[0]! + modulationCache[sigma]!, count: 10 * d)
        timings["modulation"] = elapsed(since: started)

        // Inputs: the text rows gathered in encoder order, the target tokens.
        for (i, row) in sequence.textRows.enumerated() {
            arena.pointer(textIn, floatOffset: i * d).update(from: Array(encoder[row.encoder * d..<(row.encoder + 1) * d]), count: d)
        }
        arena.pointer(latent).update(from: target, count: s * QwenDiT.channels)

        var commands = try begin()
        let textCount = sequence.textRows.count
        if textCount > 0 {
            try rmsNorm(textIn, into: textOut, rows: textCount, weight: textNorm, commands: commands)
            try linear("txt_in.in_layer.weight", a: textOut, c: textIn, m: textCount, commands: commands)
            try elementwise("gelu_tanh", [(textIn, 0)], count: textCount * d, commands: commands)
            try linear("txt_in.out_layer.weight", a: textIn, c: textOut, m: textCount, commands: commands)
            // Scatter to the joint rows, by runs.
            let blit = commands.makeBlitCommandEncoder()!
            var i = 0
            while i < textCount {
                var j = i + 1
                while j < textCount, sequence.textRows[j].joint == sequence.textRows[j - 1].joint + 1 { j += 1 }
                blit.copy(from: arena.buffer, sourceOffset: textOut.offset + i * d * 4, to: arena.buffer,
                          destinationOffset: x.offset + sequence.textRows[i].joint * d * 4, size: (j - i) * d * 4)
                i = j
            }
            blit.endEncoding()
        }
        try linear("img_in.weight", a: latent, c: x, m: s, cRow: p, commands: commands)
        if let probe { try finish(commands); probe("joint_in", arena.pointer(x), n * d); commands = try begin() }

        for layer in 0..<QwenDiT.layers {
            var stage: Stage?
            if let probe, layer == 0 {
                stage = { name, slice, current in
                    try self.finish(current)
                    probe("l0_" + name, self.arena.pointer(slice), n * d)
                    return try self.begin()
                }
            }
            try encodeLayer(layer, commands: &commands, stage: stage)
            if let probe {
                try finish(commands); probe("layer\(layer)_out", arena.pointer(x), n * d); commands = try begin()
            } else {
                commands.commit()
                commands = try begin()
            }
        }
        try layerNorm(x, into: normed, rows: n, prefixMod: 4 * d, targetMod: 5 * d + 4 * d, commands: commands)
        if let probe { try finish(commands); probe("norm_out", arena.pointer(normed), n * d); commands = try begin() }
        try linear("proj_out.weight", a: normed, aRow: p, c: output, m: s, commands: commands)
        try finish(commands)
        if let coexecutor, !profiling, balancing { try coexecutor.rebalance() }
        let out = Array(UnsafeBufferPointer(start: arena.pointer(output), count: s * QwenDiT.channels))
        probe?("model_out", arena.pointer(output), s * QwenDiT.channels)
        timings["total"] = elapsed(since: started)
        return out
    }

    /// One block on all rows of `x`.
    /// A probe point inside a layer: finishes the command buffer, reads, returns the next one.
    typealias Stage = (String, Arena.Slice, MTLCommandBuffer) throws -> MTLCommandBuffer

    func encodeLayer(_ layer: Int, commands: inout MTLCommandBuffer, stage: Stage? = nil) throws {
        func mark(_ name: String, _ slice: Arena.Slice) throws {
            if let stage { commands = try stage(name, slice, commands) }
        }
        let d = QwenDiT.dim, n = sequence.count, p = sequence.prefix
        let block = "transformer_blocks.\(layer)."
        let band = 5 * d   // the target's modulation row follows the prefix's
        try layerNorm(x, into: normed, rows: n, prefixMod: 0, targetMod: band, commands: commands)
        try linear(block + "attn.to_q.weight", a: normed, c: q, m: n, next: block + "attn.to_k.weight", commands: commands)
        try linear(block + "attn.to_k.weight", a: normed, c: k, m: n, next: block + "attn.to_v.weight", commands: commands)
        try mark("norm1", normed)
        try linear(block + "attn.to_v.weight", a: normed, c: v, m: n, next: block + "attn.to_out.0.weight", commands: commands)
        try headNormRope(q, weightOffset: (layer * 2) * QwenDiT.headDim, rows: n, commands: commands)
        try headNormRope(k, weightOffset: (layer * 2 + 1) * QwenDiT.headDim, rows: n, commands: commands)
        try mark("q", q); try mark("k", k); try mark("v", v)
        for segment in sequence.segments {
            try attend(queries: segment.start, count: segment.end - segment.start, keys: segment.end,
                       causal: segment.isText, commands: commands)
        }
        try attend(queries: p, count: n - p, keys: n, causal: false, commands: commands)
        try linear(block + "attn.to_out.0.weight", a: attended, c: o, m: n, next: block + "img_mlp.gate_layer.weight", commands: commands)
        try mark("attn_out", o)
        try residual(gate: d, rows: n, commands: commands)
        try mark("res1", x)
        try layerNorm(x, into: normed, rows: n, prefixMod: 2 * d, targetMod: band + 2 * d, commands: commands)
        try mark("norm2", normed)
        try linear(block + "img_mlp.gate_layer.weight", a: normed, c: h1, m: n, next: block + "img_mlp.proj.weight", commands: commands)
        try linear(block + "img_mlp.proj.weight", a: normed, c: h2, m: n, next: block + "img_mlp.out.weight", commands: commands)
        try elementwise("swiglu", [(h1, 0), (h2, 0)], count: n * QwenDiT.hidden, commands: commands)
        try linear(block + "img_mlp.out.weight", a: h1, c: o, m: n,
                   next: layer + 1 < QwenDiT.layers ? "transformer_blocks.\(layer + 1).attn.to_q.weight" : nil, commands: commands)
        try mark("mlp", o)
        try residual(gate: 3 * d, rows: n, commands: commands)
    }

    // MARK: - encoders

    /// The balancing loop after each evaluation (`Balancer.swift`); `KOKON_BALANCE=0` keeps the starting shares.
    public var balancing = ProcessInfo.processInfo.environment["KOKON_BALANCE"] != "0"

    /// When set, every operation runs in a command buffer of its own and its GPU time is added to `timings`
    /// under its kind (slower overall: the GPU idles between operations).
    public var profiling = false

    /// Encodes `body` into `commands`, or, when profiling, into its own command buffer, timed.
    func gpu(_ kind: String, _ commands: MTLCommandBuffer, _ body: (MTLCommandBuffer) throws -> Void) throws {
        guard profiling else { return try body(commands) }
        let own = try begin()
        try body(own)
        timings["gpu " + kind, default: 0] += try MetalContext.finish(own, kind)
    }

    func begin() throws -> MTLCommandBuffer {
        MPSCommandBuffer(from: context.queue)
    }

    func finish(_ commands: MTLCommandBuffer) throws {
        try MetalContext.finish(commands, "DiT")
    }

    func elapsed(since start: UInt64) -> Double { Double(DispatchTime.now().uptimeNanoseconds - start) * 1e-9 }

    /// The file's Linear weight `name` (without the prefix). Files in ComfyUI's layout (unsloth's GGUFs) fuse each
    /// block's `img_mlp.gate_layer` and `img_mlp.proj` into one `img_mlp.gate_up` `[2 · 12288, 4096]`, gate rows
    /// first (diffusers' single_file_utils and sd.cpp's qwen_image_2_1.hpp split it the same way): each half is
    /// then a view of the fused tensor's rows, a quantized row being contiguous, so nothing is copied.
    func weight(_ name: String) throws -> GGUFTensor {
        if let t = file.tensors[QwenDiT.prefix + name] { return t }
        for (part, half) in [("img_mlp.gate_layer.weight", 0), ("img_mlp.proj.weight", 1)] where name.hasSuffix(part) {
            let fused = try file.tensor(QwenDiT.prefix + name.dropLast(part.count) + "img_mlp.gate_up.weight")
            guard let type = fused.type, fused.dims.count == 2, fused.rows % 2 == 0 else {
                throw GGUFFile.Failure.unsupported("\(fused.name): \(fused.dims)")
            }
            let rows = fused.rows / 2
            return GGUFTensor(name: QwenDiT.prefix + name, dims: [fused.columns, rows], typeID: fused.typeID,
                              offset: fused.offset + half * rows * type.bytes(count: fused.columns))
        }
        return try file.tensor(QwenDiT.prefix + name)
    }

    /// The weight unpacked (and packed for the Neural Engine) during the product before, and its reserve.
    private var unpackedAhead: (name: String, reserve: Int)?

    /// `C[m, n] = A[m, k] · W` with W the file's Linear `[n, k]`, unpacked to `[k, n]` fp32 in a reserve first.
    /// `aRow`, `cRow`: first row of A and C in their slices. `next`: the weight of the product that follows, unpacked
    /// into the other reserve and packed for the Neural Engine while the AMX and the Neural Engine run this one (the
    /// GPU's ~1.8 ms per 4096 × 4096 weight otherwise ran alone between products).
    func linear(_ name: String, a: Arena.Slice, aRow: Int = 0, c: Arena.Slice, m: Int, cRow: Int = 0,
                next: String? = nil, commands: MTLCommandBuffer) throws {
        let tensor = try weight(name)
        let (kk, nn) = (tensor.columns, tensor.rows)
        let m = QwenDiT.padded(m)
        guard (aRow + m) * kk * 4 <= a.bytes, (cRow + m) * nn * 4 <= c.bytes else {
            throw MetalContext.Failure.buffer("\(name): no room for \(m) padded rows")
        }
        let index: Int
        if let ahead = unpackedAhead, ahead.name == name {
            index = ahead.reserve
        } else {
            index = nextReserve
            try gpu("dequant", commands) {
                try dequant.encode(tensor, of: file, into: arena.buffer, outOffset: reserves[index].offset, transposed: true, commands: $0)
            }
        }
        unpackedAhead = nil
        nextReserve = (index + 1) % reserves.count
        let reserve = reserves[index]
        // The GPU's rows: all of them, or its share of a split product.
        func gpuRows(_ rows: Int) throws {
            let key = "\(rows)×\(kk)×\(nn)"
            let product: MPSMatrixMultiplication
            if let existing = products[key] { product = existing } else {
                product = MPSMatrixMultiplication(device: context.device, transposeLeft: false, transposeRight: false,
                                                  resultRows: rows, resultColumns: nn, interiorColumns: kk, alpha: 1, beta: 0)
                products[key] = product
            }
            let left = MPSMatrix(buffer: arena.buffer, offset: a.offset + aRow * kk * 4,
                                 descriptor: MPSMatrixDescriptor(rows: rows, columns: kk, rowBytes: kk * 4, dataType: .float32))
            let right = MPSMatrix(buffer: arena.buffer, offset: reserve.offset,
                                  descriptor: MPSMatrixDescriptor(rows: kk, columns: nn, rowBytes: nn * 4, dataType: .float32))
            let result = MPSMatrix(buffer: arena.buffer, offset: c.offset + cRow * nn * 4,
                                   descriptor: MPSMatrixDescriptor(rows: rows, columns: nn, rowBytes: nn * 4, dataType: .float32))
            try gpu(m < 512 ? "products (few rows)" : "products", commands) {
                product.encode(commandBuffer: $0, leftMatrix: left, rightMatrix: right, resultMatrix: result)
            }
        }
        if let coexecutor, !profiling, coexecutor.applies(m: m, k: kk, n: nn) {
            func operand(_ slice: Arena.Slice, _ row: Int, _ width: Int) -> (MTLBuffer, Int, UnsafeMutablePointer<Float>) {
                (arena.buffer, slice.offset + row * width * 4, arena.pointer(slice, floatOffset: row * width))
            }
            var ahead: ((MTLCommandBuffer) throws -> Void)?
            if let next {
                let t = try weight(next), other = nextReserve
                ahead = { [self] commands in
                    try dequant.encode(t, of: file, into: arena.buffer, outOffset: reserves[other].offset, transposed: true, commands: commands)
                    try coexecutor.prepareWeight(next, w: operand(reserves[other], 0, t.rows), k: t.columns, n: t.rows, commands: commands)
                    unpackedAhead = (next, other)
                }
            }
            try coexecutor.encode(weight: name, a: operand(a, aRow, kk), w: operand(reserve, 0, nn), c: operand(c, cRow, nn),
                                  m: m, k: kk, n: nn, commands: commands, ahead: ahead, gpuRows: gpuRows)
        } else {
            try gpuRows(m)
        }
    }

    func compute(_ name: String, commands: MTLCommandBuffer, _ body: (MTLComputeCommandEncoder) throws -> Void) throws {
        try gpu(name, commands) { commands in
            let encoder = commands.makeComputeCommandEncoder()!
            encoder.label = name
            encoder.setComputePipelineState(try context.pipeline(name))
            try body(encoder)
            encoder.endEncoding()
        }
    }

    struct RowParams { var columns: UInt32; var targetStart: UInt32; var eps: Float }
    struct HeadParams { var heads: UInt32; var rows: UInt32; var eps: Float }
    struct ResidualParams { var columns: UInt32; var targetStart: UInt32; var count: UInt32 }

    /// `out = LN(x) · mod[band]`, `prefixMod` / `targetMod` floats into the modulation slice.
    func layerNorm(_ input: Arena.Slice, into out: Arena.Slice, rows: Int, prefixMod: Int, targetMod: Int,
                   commands: MTLCommandBuffer) throws {
        try compute("layernorm_modulate", commands: commands) { e in
            e.setBuffer(arena.buffer, offset: input.offset, index: 0)
            e.setBuffer(arena.buffer, offset: out.offset, index: 1)
            e.setBuffer(arena.buffer, offset: modulation.offset + prefixMod * 4, index: 2)
            e.setBuffer(arena.buffer, offset: modulation.offset + targetMod * 4, index: 3)
            var params = RowParams(columns: UInt32(QwenDiT.dim), targetStart: UInt32(sequence.prefix), eps: QwenDiT.eps)
            e.setBytes(&params, length: MemoryLayout<RowParams>.stride, index: 4)
            e.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
    }

    func rmsNorm(_ input: Arena.Slice, into out: Arena.Slice, rows: Int, weight: Arena.Slice, commands: MTLCommandBuffer) throws {
        try compute("rmsnorm_rows", commands: commands) { e in
            e.setBuffer(arena.buffer, offset: input.offset, index: 0)
            e.setBuffer(arena.buffer, offset: out.offset, index: 1)
            e.setBuffer(arena.buffer, offset: weight.offset, index: 2)
            var params = RowParams(columns: UInt32(QwenDiT.dim), targetStart: 0, eps: QwenDiT.eps)
            e.setBytes(&params, length: MemoryLayout<RowParams>.stride, index: 3)
            e.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
    }

    func headNormRope(_ tensor: Arena.Slice, weightOffset: Int, rows: Int, commands: MTLCommandBuffer) throws {
        try compute("head_rmsnorm_rope", commands: commands) { e in
            e.setBuffer(arena.buffer, offset: tensor.offset, index: 0)
            e.setBuffer(arena.buffer, offset: qkNorms.offset + weightOffset * 4, index: 1)
            e.setBuffer(arena.buffer, offset: rope.offset, index: 2)
            var params = HeadParams(heads: UInt32(QwenDiT.heads), rows: UInt32(rows), eps: QwenDiT.eps)
            e.setBytes(&params, length: MemoryLayout<HeadParams>.stride, index: 3)
            let units = rows * QwenDiT.heads   // one SIMD group each, 8 per threadgroup
            e.dispatchThreadgroups(MTLSize(width: (units + 7) / 8, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
    }

    /// `x += tanh(gate) · o`, the gate `gate` floats into each band's modulation row.
    func residual(gate: Int, rows: Int, commands: MTLCommandBuffer) throws {
        let d = QwenDiT.dim
        try compute("gated_residual", commands: commands) { e in
            e.setBuffer(arena.buffer, offset: x.offset, index: 0)
            e.setBuffer(arena.buffer, offset: o.offset, index: 1)
            e.setBuffer(arena.buffer, offset: modulation.offset + gate * 4, index: 2)
            e.setBuffer(arena.buffer, offset: modulation.offset + (5 * d + gate) * 4, index: 3)
            var params = ResidualParams(columns: UInt32(d), targetStart: UInt32(sequence.prefix), count: UInt32(rows * d))
            e.setBytes(&params, length: MemoryLayout<ResidualParams>.stride, index: 4)
            e.dispatchThreads(MTLSize(width: rows * d, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
    }

    func elementwise(_ name: String, _ buffers: [(Arena.Slice, Int)], count: Int, commands: MTLCommandBuffer) throws {
        try compute(name, commands: commands) { e in
            for (i, (slice, offset)) in buffers.enumerated() { e.setBuffer(arena.buffer, offset: slice.offset + offset * 4, index: i) }
            var c = UInt32(count)
            e.setBytes(&c, length: 4, index: buffers.count)
            e.dispatchThreads(MTLSize(width: count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
    }

    /// Attention through `Kernels/attention.metal` or MPSGraph (`KOKON_ATTENTION=graph`, the check). The kernel takes
    /// fp16 operands by default (Gabriele 10-25: 76.0 s against 79.9 s per 1024² image, the velocity within 6e-4 of
    /// fp64 instead of 1e-5); `KOKON_ATTENTION=fp32` keeps the fp32 kernel (the accuracy checks).
    let graphAttention = ProcessInfo.processInfo.environment["KOKON_ATTENTION"] == "graph"
    let flashKernel = ProcessInfo.processInfo.environment["KOKON_ATTENTION"] == "fp32" ? "flash_attention" : "flash_attention_half"
    struct FlashParams { var queries, keys, causal, stride: UInt32; var scale: Float; var headBase: UInt32 }

    /// Queries `[queries, queries + count)` against keys `[0, keys)`, into the same rows of `attended`. The target's
    /// queries are cut by heads with the Neural Engine and the CPU when the co-executor runs (`AttentionSplit.swift`).
    func attend(queries: Int, count: Int, keys: Int, causal: Bool, commands: MTLCommandBuffer) throws {
        let row = QwenDiT.dim * 4
        if let coexecutor, !profiling, !graphAttention, !causal, coexecutor.attentionApplies(queries: count) {
            func operand(_ slice: Arena.Slice, _ first: Int) -> Coexecutor.Operand {
                (arena.buffer, slice.offset + first * row, arena.pointer(slice, floatOffset: first * QwenDiT.dim))
            }
            try coexecutor.encodeAttention(q: operand(q, queries), k: operand(k, 0), v: operand(v, 0), out: operand(attended, queries),
                                           queries: count, keys: keys, stride: QwenDiT.dim, commands: commands) { gpuHeads in
                try flash(queries: queries, count: count, keys: keys, causal: false, heads: gpuHeads, commands: commands)
            }
            return
        }
        if !graphAttention {
            try flash(queries: queries, count: count, keys: keys, causal: causal, heads: QwenDiT.heads, commands: commands)
            return
        }
        let key = "\(count)×\(keys)\(causal ? "c" : "")"
        let attention: GraphAttention
        if let a = attentions[key] { attention = a } else {
            attention = try GraphAttention(device: context.device, queries: count, keys: keys, heads: QwenDiT.heads,
                                           headDim: QwenDiT.headDim, causal: causal)
            attentions[key] = attention
        }
        let b = arena.buffer
        try gpu("attention", commands) {
            attention.encode(commands: $0, q: (b, q.offset + queries * row), k: (b, k.offset), v: (b, v.offset),
                             out: (b, attended.offset + queries * row))
        }
    }

    /// `flash_attention` on heads `[0, heads)`.
    func flash(queries: Int, count: Int, keys: Int, causal: Bool, heads: Int, commands: MTLCommandBuffer) throws {
        guard heads > 0 else { return }
        let row = QwenDiT.dim * 4
        try compute(flashKernel, commands: commands) { e in
            e.setBuffer(arena.buffer, offset: q.offset + queries * row, index: 0)
            e.setBuffer(arena.buffer, offset: k.offset, index: 1)
            e.setBuffer(arena.buffer, offset: v.offset, index: 2)
            e.setBuffer(arena.buffer, offset: attended.offset + queries * row, index: 3)
            var params = FlashParams(queries: UInt32(count), keys: UInt32(keys), causal: causal ? 1 : 0,
                                     stride: UInt32(QwenDiT.dim), scale: 1 / Float(QwenDiT.headDim).squareRoot() * Float(M_LOG2E),
                                     headBase: 0)
            e.setBytes(&params, length: MemoryLayout<FlashParams>.stride, index: 4)
            e.dispatchThreadgroups(MTLSize(width: (count + 63) / 64, height: heads, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
    }
}
