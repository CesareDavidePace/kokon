import Foundation
import Metal
import MetalPerformanceShaders

/// **Qwen-Image 2.1's text encoder**: the text part of Qwen3-VL-8B-Instruct (`Qwen3VLTextModel`, transformers 5) on
/// the GPU, fp32 activations, the GGUF weights unpacked on the GPU just before each product, as the DiT does.
///
///     prompt ─ template ─ tokenizer ─ token_embd rows (CPU)
///     36 × { x += o_proj(causal SDPA(RoPE(RMS(q)), RoPE(RMS(k)), v))   on RMS(x)·w   (32 query heads, 8 key/value)
///            x += down(SiLU(gate) ⊙ up)                                  on RMS(x)·w }
///     conditioning = x without the system message's rows (no final norm)
///
/// What diffusers' QwenImage21Pipeline does (0.42, `_get_qwen_prompt_embeds`): the raw template below, the hidden
/// state of the last decoder layer *before* the model's final RMSNorm (the pipeline hooks the norm away), and the
/// leading tokens of the system message dropped. Text only: the three M-RoPE axes carry the same position, so the
/// interleaved M-RoPE is plain RoPE (θ = 5·10⁶) and the vision tower's deepstack inputs never apply.
public final class QwenTextEncoder {
    public static let dim = 4096, heads = 32, kvHeads = 8, headDim = 128, hidden = 12288, layers = 36
    static let eps: Float = 1e-6, theta: Float = 5_000_000
    public static let systemPrompt = "Comprehend and analyze the provided prompt."

    /// `prompt_template_t2i`: the prompt is inserted as is, the processor tokenizes the whole string.
    public static func textToImage(_ prompt: String) -> String {
        // Qwen has no bos token, so an empty prompt leaves the encoder nothing to read: diffusers sends " ".
        "<|im_start|>system\n\(systemPrompt)<|im_end|>\n<|im_start|>user\n\(prompt.isEmpty ? " " : prompt)<|im_end|>\n<|im_start|>assistant\n"
    }

    public let file: GGUFFile
    public let tokenizer: QwenTokenizer
    let context: MetalContext
    let dequant: GPUDequant
    /// Tokens of the system message as the chat template writes it alone (`_drop_idx`, 14).
    public let dropCount: Int
    private var products: [String: MPSMatrixMultiplication] = [:]
    /// Seconds per phase of the last forward.
    public private(set) var timings: [String: Double] = [:]

    public init(file: GGUFFile, context: MetalContext) throws {
        guard file.metadata["general.architecture"]?.string == "qwen3vl",
              file.metadata["qwen3vl.block_count"]?.int == QwenTextEncoder.layers,
              file.metadata["qwen3vl.embedding_length"]?.int == QwenTextEncoder.dim else {
            throw GGUFFile.Failure.unsupported("\(file.path): not the Qwen3-VL-8B text encoder")
        }
        self.file = file
        self.context = context
        dequant = GPUDequant(context: context)
        tokenizer = try QwenTokenizer(file: file)
        dropCount = tokenizer.encode("<|im_start|>system\n\(QwenTextEncoder.systemPrompt)<|im_end|>\n").count
    }

    /// Token ids of the text-to-image prompt.
    public func tokens(prompt: String) -> [Int] { tokenizer.encode(QwenTextEncoder.textToImage(prompt)) }

    /// The DiT's encoder rows for `prompt`: `[T, 4096]`, T = tokens − the system message's.
    public func conditioning(prompt: String) throws -> (rows: [Float], count: Int) {
        let ids = tokens(prompt: prompt)
        let all = try forward(ids: ids)
        let d = QwenTextEncoder.dim
        return (Array(all[(dropCount * d)...]), ids.count - dropCount)
    }

    /// Drops the GPU windows on the file (the caller drops the file to unmap it).
    public func release() { context.release(file) }

    // MARK: - forward

    /// Called with a stage's name and its rows (CPU pointer, floats); the pass then waits on the GPU at each stage.
    /// Stages: embeddings, layer0_out, layer1_out, layer17_out, layer35_out.
    public typealias Probe = (String, UnsafePointer<Float>, Int) -> Void

    struct Buffers {
        let arena: Arena
        let x, normed, q, k, v, kx, vx, attended, o, h1, h2, rope, norms: Arena.Slice
        let reserves: [Arena.Slice]
    }
    private var nextReserve = 0

    /// The last decoder layer's output for every token, `[ids.count, 4096]`. `cache`: each layer's K (after its norm
    /// and RoPE) and V are kept there too, for text generation (`QwenTextGenerator`).
    public func forward(ids: [Int], probe: Probe? = nil, cache: KVCache? = nil) throws -> [Float] {
        let d = QwenTextEncoder.dim, t = ids.count, kvWidth = QwenTextEncoder.kvHeads * QwenTextEncoder.headDim
        let started = DispatchTime.now().uptimeNanoseconds
        timings = [:]
        // Rows padded as the DiT's (64, see QwenDiT.padded), plus 64 for the attention kernel's overrun.
        let n = QwenDiT.padded(t) + 64
        let sizes = [n * d, n * d, n * d, n * kvWidth, n * kvWidth, n * d, n * d, n * d, n * d,
                     n * QwenTextEncoder.hidden, n * QwenTextEncoder.hidden, n * 128,
                     QwenTextEncoder.layers * 4 * d, d * QwenTextEncoder.hidden, d * QwenTextEncoder.hidden]
        let arena = try Arena(device: context.device, bytes: Arena.bytes(sizes.map { $0 * 4 }))
        var s = try sizes.map { try arena.slice(floats: $0) }.makeIterator()
        let b = Buffers(arena: arena, x: s.next()!, normed: s.next()!, q: s.next()!, k: s.next()!, v: s.next()!,
                        kx: s.next()!, vx: s.next()!, attended: s.next()!, o: s.next()!, h1: s.next()!, h2: s.next()!,
                        rope: s.next()!, norms: s.next()!, reserves: [s.next()!, s.next()!])

        // Norm weights of every layer (F32, small), laid out per layer: input, post-attention, q (128), k (128).
        for layer in 0..<QwenTextEncoder.layers {
            let base = layer * 4 * d
            for (i, name) in ["attn_norm", "ffn_norm"].enumerated() {
                let w = try Dequant.array(file, "blk.\(layer).\(name).weight")
                arena.pointer(b.norms, floatOffset: base + i * d).update(from: w, count: d)
            }
            for (i, name) in ["attn_q_norm", "attn_k_norm"].enumerated() {
                let w = try Dequant.array(file, "blk.\(layer).\(name).weight")
                arena.pointer(b.norms, floatOffset: base + 2 * d + i * QwenTextEncoder.headDim)
                    .update(from: w, count: QwenTextEncoder.headDim)
            }
        }
        let table = QwenTextEncoder.ropeTable(rows: t)
        arena.pointer(b.rope).update(from: table, count: table.count)
        // Embedding rows, unpacked on the CPU.
        let embedding = try file.tensor("token_embd.weight")
        guard let type = embedding.type, embedding.columns == d else {
            throw GGUFFile.Failure.unsupported("token_embd.weight: \(embedding.dims)")
        }
        let rowBytes = type.bytes(count: d)
        for (row, id) in ids.enumerated() {
            guard id >= 0, id < embedding.rows else { throw GGUFFile.Failure.unsupported("token id \(id)") }
            Dequant.row(file.data(embedding) + id * rowBytes, type: type, count: d, into: arena.pointer(b.x, floatOffset: row * d))
        }
        probe?("embeddings", arena.pointer(b.x), t * d)

        var commands = try begin()
        for layer in 0..<QwenTextEncoder.layers {
            try encodeLayer(layer, rows: t, b, commands: commands, cache: cache)
            if let probe, [0, 1, 17, QwenTextEncoder.layers - 1].contains(layer) {
                try MetalContext.finish(commands, "text encoder")
                probe("layer\(layer)_out", arena.pointer(b.x), t * d)
            } else {
                commands.commit()
            }
            commands = try begin()
        }
        try MetalContext.finish(commands, "text encoder")
        timings["total"] = Double(DispatchTime.now().uptimeNanoseconds - started) * 1e-9
        return Array(UnsafeBufferPointer(start: arena.pointer(b.x), count: t * d))
    }

    /// `[rows, 64, (cos, sin)]` of `position · ω_i`, `ω_i = 1 / θ^(2i / 128)`, fp32 as Qwen3VLTextRotaryEmbedding.
    static func ropeTable(rows: Int) -> [Float] {
        let half = QwenTextEncoder.headDim / 2
        let omegas = (0..<half).map { i in 1 / powf(theta, Float(2 * i) / Float(QwenTextEncoder.headDim)) }
        var out = [Float](repeating: 0, count: rows * half * 2)
        for row in 0..<rows {
            for (i, w) in omegas.enumerated() {
                let angle = Float(row) * w
                out[(row * half + i) * 2] = cosf(angle)
                out[(row * half + i) * 2 + 1] = sinf(angle)
            }
        }
        return out
    }

    func encodeLayer(_ layer: Int, rows t: Int, _ b: Buffers, commands: MTLCommandBuffer, cache: KVCache? = nil) throws {
        let d = QwenTextEncoder.dim, block = "blk.\(layer)."
        let norms = layer * 4 * d
        try rmsNorm(b, b.x, into: b.normed, rows: t, weight: norms, commands: commands)
        try linear(block + "attn_q.weight", b, a: b.normed, c: b.q, m: t, commands: commands)
        try linear(block + "attn_k.weight", b, a: b.normed, c: b.k, m: t, commands: commands)
        try linear(block + "attn_v.weight", b, a: b.normed, c: b.v, m: t, commands: commands)
        try headNormRope(b, b.q, heads: QwenTextEncoder.heads, weight: norms + 2 * d, rows: t, commands: commands)
        try headNormRope(b, b.k, heads: QwenTextEncoder.kvHeads, weight: norms + 2 * d + QwenTextEncoder.headDim,
                         rows: t, commands: commands)
        try cache?.store(layer: layer, k: (b.arena.buffer, b.k.offset), v: (b.arena.buffer, b.v.offset), rows: t, position: 0,
                         context: context, commands: commands)
        try repeatKV(b, b.k, into: b.kx, rows: t, commands: commands)
        try repeatKV(b, b.v, into: b.vx, rows: t, commands: commands)
        try attend(b, rows: t, commands: commands)
        try linear(block + "attn_output.weight", b, a: b.attended, c: b.o, m: t, commands: commands)
        try add(b, b.o, count: t * d, commands: commands)
        try rmsNorm(b, b.x, into: b.normed, rows: t, weight: norms + d, commands: commands)
        try linear(block + "ffn_gate.weight", b, a: b.normed, c: b.h1, m: t, commands: commands)
        try linear(block + "ffn_up.weight", b, a: b.normed, c: b.h2, m: t, commands: commands)
        try compute("swiglu", commands: commands) { e in
            e.setBuffer(b.arena.buffer, offset: b.h1.offset, index: 0)
            e.setBuffer(b.arena.buffer, offset: b.h2.offset, index: 1)
            // The DiT's kernel: every row in fp32, no fp16 copy (buffer 3 is never written then).
            var params = QwenDiT.SwigluParams(columns: UInt32(QwenTextEncoder.hidden), half: QwenDiT.fullOnly(t))
            e.setBytes(&params, length: MemoryLayout<QwenDiT.SwigluParams>.stride, index: 2)
            e.setBuffer(b.arena.buffer, offset: b.h1.offset, index: 3)
            e.dispatchThreads(MTLSize(width: t * QwenTextEncoder.hidden, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        try linear(block + "ffn_down.weight", b, a: b.h1, c: b.o, m: t, commands: commands)
        try add(b, b.o, count: t * d, commands: commands)
    }

    // MARK: - encoders

    func begin() throws -> MTLCommandBuffer { MPSCommandBuffer(from: context.queue) }

    func compute(_ name: String, commands: MTLCommandBuffer, _ body: (MTLComputeCommandEncoder) throws -> Void) throws {
        let encoder = commands.makeComputeCommandEncoder()!
        encoder.label = name
        encoder.setComputePipelineState(try context.pipeline(name))
        try body(encoder)
        encoder.endEncoding()
    }

    /// `C[m, n] = A[m, k] · W`, W the file's Linear `[n, k]` unpacked to `[k, n]` fp32 in a reserve first; m padded
    /// to 64 (QwenDiT.padded: small MPS products slow every later one).
    func linear(_ name: String, _ b: Buffers, a: Arena.Slice, c: Arena.Slice, m: Int, commands: MTLCommandBuffer) throws {
        let tensor = try file.tensor(name)
        let (kk, nn) = (tensor.columns, tensor.rows)
        let m = QwenDiT.padded(m)
        guard m * kk * 4 <= a.bytes, m * nn * 4 <= c.bytes else {
            throw MetalContext.Failure.buffer("\(name): no room for \(m) padded rows")
        }
        let reserve = b.reserves[nextReserve]
        nextReserve = (nextReserve + 1) % b.reserves.count
        try dequant.encode(tensor, of: file, into: b.arena.buffer, outOffset: reserve.offset, transposed: true, commands: commands)
        let key = "\(m)×\(kk)×\(nn)"
        let product = products[key] ?? MPSMatrixMultiplication(device: context.device, transposeLeft: false, transposeRight: false,
                                                               resultRows: m, resultColumns: nn, interiorColumns: kk, alpha: 1, beta: 0)
        products[key] = product
        func matrix(_ offset: Int, _ rows: Int, _ columns: Int) -> MPSMatrix {
            MPSMatrix(buffer: b.arena.buffer, offset: offset,
                      descriptor: MPSMatrixDescriptor(rows: rows, columns: columns, rowBytes: columns * 4, dataType: .float32))
        }
        product.encode(commandBuffer: commands, leftMatrix: matrix(a.offset, m, kk), rightMatrix: matrix(reserve.offset, kk, nn),
                       resultMatrix: matrix(c.offset, m, nn))
    }

    func rmsNorm(_ b: Buffers, _ input: Arena.Slice, into out: Arena.Slice, rows: Int, weight: Int, commands: MTLCommandBuffer) throws {
        try compute("rmsnorm_rows", commands: commands) { e in
            e.setBuffer(b.arena.buffer, offset: input.offset, index: 0)
            e.setBuffer(b.arena.buffer, offset: out.offset, index: 1)
            e.setBuffer(b.arena.buffer, offset: b.norms.offset + weight * 4, index: 2)
            var params = QwenDiT.RowParams(columns: UInt32(QwenTextEncoder.dim), targetStart: 0, eps: QwenTextEncoder.eps)
            e.setBytes(&params, length: MemoryLayout<QwenDiT.RowParams>.stride, index: 3)
            e.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
    }

    func headNormRope(_ b: Buffers, _ tensor: Arena.Slice, heads: Int, weight: Int, rows: Int, commands: MTLCommandBuffer) throws {
        try compute("head_rmsnorm_rope_neox", commands: commands) { e in
            e.setBuffer(b.arena.buffer, offset: tensor.offset, index: 0)
            e.setBuffer(b.arena.buffer, offset: b.norms.offset + weight * 4, index: 1)
            e.setBuffer(b.arena.buffer, offset: b.rope.offset, index: 2)
            var params = QwenDiT.HeadParams(heads: UInt32(heads), rows: UInt32(rows), eps: QwenTextEncoder.eps)
            e.setBytes(&params, length: MemoryLayout<QwenDiT.HeadParams>.stride, index: 3)
            let units = rows * heads   // one SIMD group each, 8 per threadgroup
            e.dispatchThreadgroups(MTLSize(width: (units + 7) / 8, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
    }

    struct RepeatParams { var count, heads, groups: UInt32 }

    func repeatKV(_ b: Buffers, _ input: Arena.Slice, into out: Arena.Slice, rows: Int, commands: MTLCommandBuffer) throws {
        try compute("repeat_kv", commands: commands) { e in
            e.setBuffer(b.arena.buffer, offset: input.offset, index: 0)
            e.setBuffer(b.arena.buffer, offset: out.offset, index: 1)
            var params = RepeatParams(count: UInt32(rows * QwenTextEncoder.dim), heads: UInt32(QwenTextEncoder.heads),
                                      groups: UInt32(QwenTextEncoder.heads / QwenTextEncoder.kvHeads))
            e.setBytes(&params, length: MemoryLayout<RepeatParams>.stride, index: 2)
            e.dispatchThreads(MTLSize(width: Int(params.count), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
    }

    func add(_ b: Buffers, _ y: Arena.Slice, count: Int, commands: MTLCommandBuffer) throws {
        try compute("add_inplace", commands: commands) { e in
            e.setBuffer(b.arena.buffer, offset: b.x.offset, index: 0)
            e.setBuffer(b.arena.buffer, offset: y.offset, index: 1)
            var c = UInt32(count)
            e.setBytes(&c, length: 4, index: 2)
            e.dispatchThreads(MTLSize(width: count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
    }

    /// Causal attention of every row on the DiT's flash kernel (`Kernels/attention.metal`), keys and values widened
    /// to the 32 query heads first.
    func attend(_ b: Buffers, rows: Int, commands: MTLCommandBuffer) throws {
        try compute("flash_attention", commands: commands) { e in
            e.setBuffer(b.arena.buffer, offset: b.q.offset, index: 0)
            e.setBuffer(b.arena.buffer, offset: b.kx.offset, index: 1)
            e.setBuffer(b.arena.buffer, offset: b.vx.offset, index: 2)
            e.setBuffer(b.arena.buffer, offset: b.attended.offset, index: 3)
            var params = QwenDiT.FlashParams(queries: UInt32(rows), keys: UInt32(rows), causal: 1, stride: UInt32(QwenTextEncoder.dim),
                                             scale: 1 / Float(QwenTextEncoder.headDim).squareRoot() * Float(M_LOG2E), headBase: 0)
            e.setBytes(&params, length: MemoryLayout<QwenDiT.FlashParams>.stride, index: 4)
            e.dispatchThreadgroups(MTLSize(width: (rows + 63) / 64, height: QwenTextEncoder.heads, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
    }
}
