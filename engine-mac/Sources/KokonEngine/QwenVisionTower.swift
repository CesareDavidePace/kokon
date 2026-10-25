import Foundation
import Metal
import MetalPerformanceShaders

/// **Qwen3-VL-8B's vision tower** (`Qwen3VLVisionModel`, transformers 5) on the GPU, fp32, from Kokon's mmproj GGUF
/// (llama.cpp's layout, F16 weights unpacked just before each product): what an edit's reference image becomes in the
/// text encoder's `<|image_pad|>` slots.
///
///     image ─ 16 × 16 patches, in 2 × 2 merge blocks ─ patch_embed (Conv3d over the image twice) + bilinear position table
///     27 × { x += proj(SDPA(rope2d(q), rope2d(k), v))   on LN1(x)       (16 heads of 72, every patch sees every patch)
///            x += fc2(GELU_tanh(fc1(LN2(x))))           }
///     merged = fc2(GELU(fc1(LN(x) as [N/4, 4608])))              [N/4, 4096], one row per text encoder slot
///     deepstack j = the same after block 8 / 16 / 24 (LN after the 2 × 2 merge)   added to the text encoder's layers 0–2
///
/// The heads (72 dimensions) run on the DiT's flash kernel zero-padded to 128 (`vision_rope_pad`): the scores and the
/// output are the 72-dimension ones. The processor's input is the image with alpha composited over white (as the
/// pipeline's PIL paste), scaled to [−1, 1], each frame of the Conv3d's two the same image.
public final class QwenVisionTower {
    public static let dim = 1152, heads = 16, headDim = 72, hidden = 4304, layers = 27, outDim = 4096
    static let patch = 16, merge = 2, positionSide = 48, eps: Float = 1e-6, theta: Float = 10_000
    /// Blocks whose output also goes through a deepstack merger (`deepstack_visual_indexes`).
    public static let deepstackLayers = [8, 16, 24]
    /// The processor's `smart_resize` bounds (`size.shortest_edge`, `longest_edge`): pixels of the image it encodes.
    static let minPixels = 65_536, maxPixels = 16_777_216

    public let file: GGUFFile
    let context: MetalContext
    let dequant: GPUDequant
    private var products: [String: MPSMatrixMultiplication] = [:]
    /// Seconds of the last encode.
    public private(set) var seconds: Double = 0

    public init(file: GGUFFile, context: MetalContext) throws {
        guard file.metadata["clip.projector_type"]?.string == "qwen3vl_merger",
              file.metadata["clip.vision.block_count"]?.int == QwenVisionTower.layers,
              file.metadata["clip.vision.embedding_length"]?.int == QwenVisionTower.dim else {
            throw GGUFFile.Failure.unsupported("\(file.path): not Qwen3-VL-8B's vision tower (mmproj)")
        }
        self.file = file
        self.context = context
        dequant = GPUDequant(context: context)
    }

    public func release() { context.release(file) }

    /// What the tower makes of one image: `merged` `[slots, 4096]` and one deepstack `[slots, 4096]` per entry of
    /// `deepstackLayers`, slots = (height / 32) · (width / 32) in merge-block order (row-major over 32 × 32 cells).
    public struct Output {
        public let merged: [Float]
        public let deepstack: [[Float]]
        public let height, width: Int
        public var slots: Int { (height / 32) * (width / 32) }
    }

    // MARK: - the processor

    /// Qwen2VLImageProcessor's `smart_resize` for sides already multiples of 32 (as the pipeline makes them): the size
    /// the tower sees, which differs only below 65 536 or above 16 777 216 pixels.
    public static func visionSize(width: Int, height: Int) -> (width: Int, height: Int) {
        let factor = Double(patch * merge)
        var h = Int((Double(height) / factor).rounded(.toNearestOrEven) * factor)
        var w = Int((Double(width) / factor).rounded(.toNearestOrEven) * factor)
        if h * w > maxPixels {
            let beta = (Double(height * width) / Double(maxPixels)).squareRoot()
            h = max(Int(factor), Int((Double(height) / beta / factor).rounded(.down) * factor))
            w = max(Int(factor), Int((Double(width) / beta / factor).rounded(.down) * factor))
        } else if h * w < minPixels {
            let beta = (Double(minPixels) / Double(height * width)).squareRoot()
            h = Int((Double(height) * beta / factor).rounded(.up) * factor)
            w = Int((Double(width) * beta / factor).rounded(.up) * factor)
        }
        return (w, h)
    }

    /// Interleaved 8-bit RGBA → the tower's input: alpha composited over white as PIL's `paste` with a mask does
    /// (the pipeline's `white.paste(img, mask=alpha)`), resized when `visionSize` says so (bicubic), then planar RGB
    /// `[3, H, W]` scaled to [−1, 1] (`rescale 1/255`, `normalize 0.5, 0.5`).
    public static func pixels(rgba: [UInt8], width: Int, height: Int) -> (rgb: [Float], width: Int, height: Int) {
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for p in 0..<(width * height) {
            let a = Int(rgba[p * 4 + 3])
            for c in 0..<3 {
                // ImagingPaste's BLEND: in·a + 255·(255 − a), divided by 255 with rounding (PIL's MULDIV255).
                let t = Int(rgba[p * 4 + c]) * a + 255 * (255 - a) + 128
                rgb[p * 3 + c] = UInt8((t + (t >> 8)) >> 8)
            }
        }
        let (w, h) = visionSize(width: width, height: height)
        if (w, h) != (width, height) { rgb = resizeBicubic(rgb, width: width, height: height, to: w, h) }
        var out = [Float](repeating: 0, count: 3 * w * h)
        for p in 0..<(w * h) {
            for c in 0..<3 { out[c * w * h + p] = (Float(rgb[p * 3 + c]) * (1 / 255.0) - 0.5) / 0.5 }
        }
        return (out, w, h)
    }

    /// Bicubic resize of interleaved 8-bit RGB exactly as torchvision's antialiased resize of a uint8 image (the
    /// processor's path; PIL's algorithm with torch's integer arithmetic): a = −0.5, the kernel stretched by the scale
    /// when shrinking, each output's weights normalized then quantized to int16 with as many fraction bits as the
    /// largest fits in, the horizontal pass rounded to 8 bits before the vertical one. Bit for bit against torchvision
    /// 0.26 on a 256 × 192 → 320 × 224 image (2026-10-25). Only images outside the processor's pixel bounds go through it.
    static func resizeBicubic(_ src: [UInt8], width: Int, height: Int, to w: Int, _ h: Int) -> [UInt8] {
        func taps(_ out: Int, _ inSize: Int) -> (taps: [(Int, [Int])], bits: Int) {
            let scale = Double(inSize) / Double(out), support = 2 * max(scale, 1), invScale = 1 / max(scale, 1)
            func cubic(_ x: Double) -> Double {
                let a = -0.5, x = abs(x)
                if x <= 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
                if x < 2 { return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a }
                return 0
            }
            let real = (0..<out).map { i -> (Int, [Double]) in
                let center = scale * (Double(i) + 0.5)
                let lo = max(Int(center - support + 0.5), 0), hi = min(Int(center + support + 0.5), inSize)
                let weights = (lo..<hi).map { cubic((Double($0) - center + 0.5) * invScale) }
                let total = weights.reduce(0, +)
                return (lo, weights.map { $0 / total })
            }
            let largest = real.flatMap { $0.1 }.map(abs).max() ?? 1
            var bits = 0
            while bits < 22, Int(0.5 + largest * Double(1 << (bits + 1))) < 1 << 15 { bits += 1 }
            let scaleBits = Double(1 << bits)
            return (real.map { ($0.0, $0.1.map { v in Int(v >= 0 ? (v * scaleBits + 0.5).rounded(.down) : (v * scaleBits - 0.5).rounded(.up)) }) }, bits)
        }
        func clamp(_ v: Int, _ bits: Int) -> UInt8 { UInt8(min(max((v + (1 << (bits - 1))) >> bits, 0), 255)) }
        let (tx, bx) = taps(w, width), (ty, by) = taps(h, height)
        var horizontal = [UInt8](repeating: 0, count: w * height * 3)
        for y in 0..<height {
            for x in 0..<w {
                let (lo, weights) = tx[x]
                for c in 0..<3 {
                    var s = 0
                    for (j, wt) in weights.enumerated() { s += wt * Int(src[(y * width + lo + j) * 3 + c]) }
                    horizontal[(y * w + x) * 3 + c] = clamp(s, bx)
                }
            }
        }
        var out = [UInt8](repeating: 0, count: w * h * 3)
        for y in 0..<h {
            let (lo, weights) = ty[y]
            for x in 0..<w {
                for c in 0..<3 {
                    var s = 0
                    for (j, wt) in weights.enumerated() { s += wt * Int(horizontal[((lo + j) * w + x) * 3 + c]) }
                    out[(y * w + x) * 3 + c] = clamp(s, by)
                }
            }
        }
        return out
    }

    // MARK: - forward

    /// Called with a stage's name and its rows (CPU pointer, floats); the pass then waits on the GPU at each stage.
    /// Stages: patches, block0, block8, block26 (the processor's order: 2 × 2 merge blocks).
    public typealias Probe = (String, UnsafePointer<Float>, Int) -> Void

    struct Buffers {
        let arena: Arena
        let x, normed, qkv, q, k, v, attended, a72, h, mergeIn, mergeHidden, smalls, rope: Arena.Slice
        let outputs: [Arena.Slice]
        let reserves: [Arena.Slice]
    }
    private var nextReserve = 0
    /// Float offsets of the small F32 tensors (norms, biases) in `smalls`.
    private var small: [String: Int] = [:]

    /// The tower on planar RGB `[3, height, width]` in [−1, 1] (`pixels`), sides multiples of 32. In its own
    /// autorelease pool.
    public func encode(rgb: [Float], width: Int, height: Int, probe: Probe? = nil) throws -> Output {
        try autoreleasepool { try forward(rgb: rgb, width: width, height: height, probe: probe) }
    }

    private func forward(rgb: [Float], width: Int, height: Int, probe: Probe?) throws -> Output {
        guard width % 32 == 0, height % 32 == 0, rgb.count == 3 * width * height else {
            throw QwenSequence.Failure.layout("vision tower: \(rgb.count) values for 3 × \(height) × \(width) (sides multiples of 32)")
        }
        let started = DispatchTime.now().uptimeNanoseconds
        let d = QwenVisionTower.dim, (gh, gw) = (height / 16, width / 16), n = gh * gw, slots = n / 4
        let merged = 4 * d   // 4608
        // Rows padded as the DiT's (64), plus 64 for the attention kernel's overrun; x is also read as [slots, 4608].
        let rows = max(QwenDiT.padded(n) + 64, 4 * QwenDiT.padded(slots))
        let mrows = QwenDiT.padded(slots)
        let smallNames = smallTensors()
        let smallCount = try smallNames.reduce(0) { $0 + (try file.tensor($1)).count } + n * d
        let sizes = [rows * d, rows * d, rows * 3 * d, rows * 2048, rows * 2048, rows * 2048, rows * 2048, rows * d,
                     rows * QwenVisionTower.hidden, mrows * merged, mrows * merged, smallCount, n * 72,
                     slots * 4096, slots * 4096, slots * 4096, slots * 4096, merged * merged, merged * merged]
        let arena = try Arena(device: context.device, bytes: Arena.bytes(sizes.map { $0 * 4 }))
        var s = try sizes.map { try arena.slice(floats: $0) }.makeIterator()
        let b = Buffers(arena: arena, x: s.next()!, normed: s.next()!, qkv: s.next()!, q: s.next()!, k: s.next()!,
                        v: s.next()!, attended: s.next()!, a72: s.next()!, h: s.next()!, mergeIn: s.next()!,
                        mergeHidden: s.next()!, smalls: s.next()!, rope: s.next()!,
                        outputs: [s.next()!, s.next()!, s.next()!, s.next()!], reserves: [s.next()!, s.next()!])

        // Norms and biases (F32), then the position table after them.
        small = [:]
        var at = 0
        for name in smallNames {
            let values = try Dequant.array(file, name)
            arena.pointer(b.smalls, floatOffset: at).update(from: values, count: values.count)
            small[name] = at
            at += values.count
        }
        small["positions"] = at
        let table = try positions(gridHeight: gh, gridWidth: gw)
        arena.pointer(b.smalls, floatOffset: at).update(from: table, count: table.count)

        // Patches [n, 768] in merge-block order (each row: channel, then 16 × 16 pixels): the Conv3d's two frames see
        // the same image, so its two kernels are summed (`v.patch_embd.weight` + `.weight.1`) into one [768, 1152].
        let patchRows = arena.pointer(b.normed)
        var index = 0
        for by in 0..<(gh / 2) { for bx in 0..<(gw / 2) { for iy in 0..<2 { for ix in 0..<2 {
            let (py, px) = ((by * 2 + iy) * 16, (bx * 2 + ix) * 16)
            for c in 0..<3 { for y in 0..<16 { for x in 0..<16 {
                patchRows[index * 768 + c * 256 + y * 16 + x] = rgb[(c * height + py + y) * width + px + x]
            } } }
            index += 1
        } } } }
        let w0 = try Dequant.array(file, "v.patch_embd.weight"), w1 = try Dequant.array(file, "v.patch_embd.weight.1")
        let kernel = arena.pointer(b.reserves[0])
        for o in 0..<d { for i in 0..<768 { kernel[i * d + o] = w0[o * 768 + i] + w1[o * 768 + i] } }

        var commands = MPSCommandBuffer(from: context.queue) as MTLCommandBuffer
        try product(b, a: b.normed, w: b.reserves[0], c: b.x, m: n, k: 768, n: d, commands: commands)
        try bias(b, b.x, "v.patch_embd.bias", rows: n, columns: d, commands: commands)
        if let probe {
            try MetalContext.finish(commands, "vision tower")
            probe("patches", arena.pointer(b.x), n * d)
            commands = MPSCommandBuffer(from: context.queue)
        }
        try add(b, b.x, b.smalls, offset: small["positions"]!, count: n * d, commands: commands)
        try ropeTable(b, gridHeight: gh, gridWidth: gw)

        for layer in 0..<QwenVisionTower.layers {
            try encodeBlock(layer, rows: n, b, commands: commands)
            if let j = QwenVisionTower.deepstackLayers.firstIndex(of: layer) {
                // Deepstack merger: the block's output as [slots, 4608], then its LayerNorm over 4608.
                try layerNorm(b, b.x, into: b.mergeIn, prefix: "v.deepstack.\(layer).norm", rows: slots, columns: merged,
                              commands: commands)
                try mergerMLP(b, prefix: "v.deepstack.\(layer)", fc1: "fc1", fc2: "fc2", out: b.outputs[1 + j], slots: slots,
                              commands: commands)
            }
            if let probe, [0, 8, 26].contains(layer) {
                try MetalContext.finish(commands, "vision tower")
                probe("block\(layer)", arena.pointer(b.x), n * d)
            } else {
                commands.commit()
            }
            commands = MPSCommandBuffer(from: context.queue)
        }
        // Merger: post_ln on each patch, then the 2 × 2 groups as rows of 4608.
        try layerNorm(b, b.x, into: b.mergeIn, prefix: "v.post_ln", rows: n, columns: d, commands: commands)
        try mergerMLP(b, prefix: "mm", fc1: "0", fc2: "2", out: b.outputs[0], slots: slots, commands: commands)
        try MetalContext.finish(commands, "vision tower")
        seconds = Double(DispatchTime.now().uptimeNanoseconds - started) * 1e-9
        func read(_ slice: Arena.Slice) -> [Float] { Array(UnsafeBufferPointer(start: arena.pointer(slice), count: slots * 4096)) }
        return Output(merged: read(b.outputs[0]), deepstack: (1...3).map { read(b.outputs[$0]) }, height: height, width: width)
    }

    func smallTensors() -> [String] {
        var names = ["v.patch_embd.bias", "v.post_ln.weight", "v.post_ln.bias", "mm.0.bias", "mm.2.bias"]
        for i in 0..<QwenVisionTower.layers {
            for t in ["ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias", "attn_qkv.bias", "attn_out.bias", "ffn_up.bias", "ffn_down.bias"] {
                names.append("v.blk.\(i).\(t)")
            }
        }
        for l in QwenVisionTower.deepstackLayers {
            for t in ["norm.weight", "norm.bias", "fc1.bias", "fc2.bias"] { names.append("v.deepstack.\(l).\(t)") }
        }
        return names
    }

    /// `pos_embed` (48 × 48 learned rows) resampled bilinearly, corners aligned, to each patch (merge-block order), as
    /// `get_vision_interpolation_indices_and_weights`: the four taps' rows weighted and summed in fp32.
    func positions(gridHeight gh: Int, gridWidth gw: Int) throws -> [Float] {
        let d = QwenVisionTower.dim, side = QwenVisionTower.positionSide
        let table = try Dequant.array(file, "v.position_embd.weight")   // [2304, 1152]
        func taps(_ index: Int, _ size: Int) -> [(Int, Float)] {
            let src = Float(index) * Float(side - 1) / Float(max(size - 1, 1))
            let floor = src.rounded(.down)
            return (0..<2).map { o in
                let tap = min(max(Int(floor) + o, 0), side - 1)
                return (tap, max(1 - abs(src - floor - Float(o)), 0))
            }
        }
        var out = [Float](repeating: 0, count: gh * gw * d)
        var index = 0
        for by in 0..<(gh / 2) { for bx in 0..<(gw / 2) { for iy in 0..<2 { for ix in 0..<2 {
            let (row, col) = (by * 2 + iy, bx * 2 + ix)
            let hs = taps(row, gh), ws = taps(col, gw)
            var acc = [Float](repeating: 0, count: d)
            for (ht, hw) in hs { for (wt, ww) in ws {
                let weight = hw * ww, base = (ht * side + wt) * d
                for c in 0..<d { acc[c] += table[base + c] * weight }
            } }
            for c in 0..<d { out[index * d + c] = acc[c] }
            index += 1
        } } } }
        return out
    }

    /// `[rows, 36, (cos, sin)]` of the axial rotary embedding: dimensions 0–17 rotate by the patch's row, 18–35 by its
    /// column (`inv_freq = 1 / 10000^(2i / 36)`), and dimension i + 36 shares dimension i's angle.
    func ropeTable(_ b: Buffers, gridHeight gh: Int, gridWidth gw: Int) throws {
        let inv = (0..<18).map { i in 1 / powf(QwenVisionTower.theta, Float(2 * i) / 36) }
        let p = b.arena.pointer(b.rope)
        var index = 0
        for by in 0..<(gh / 2) { for bx in 0..<(gw / 2) { for iy in 0..<2 { for ix in 0..<2 {
            let (row, col) = (Float(by * 2 + iy), Float(bx * 2 + ix))
            for i in 0..<36 {
                let angle = i < 18 ? row * inv[i] : col * inv[i - 18]
                p[(index * 36 + i) * 2] = cosf(angle)
                p[(index * 36 + i) * 2 + 1] = sinf(angle)
            }
            index += 1
        } } } }
    }

    func encodeBlock(_ layer: Int, rows n: Int, _ b: Buffers, commands: MTLCommandBuffer) throws {
        let d = QwenVisionTower.dim, block = "v.blk.\(layer)."
        try layerNorm(b, b.x, into: b.normed, prefix: block + "ln1", rows: n, columns: d, commands: commands)
        try linear(block + "attn_qkv", b, a: b.normed, c: b.qkv, m: n, commands: commands)
        try compute("vision_rope_pad", commands: commands) { e in
            e.setBuffer(b.arena.buffer, offset: b.qkv.offset, index: 0)
            e.setBuffer(b.arena.buffer, offset: b.q.offset, index: 1)
            e.setBuffer(b.arena.buffer, offset: b.k.offset, index: 2)
            e.setBuffer(b.arena.buffer, offset: b.v.offset, index: 3)
            e.setBuffer(b.arena.buffer, offset: b.rope.offset, index: 4)
            var r = UInt32(n)
            e.setBytes(&r, length: 4, index: 5)
            e.dispatchThreads(MTLSize(width: n * 1024, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        try compute("flash_attention", commands: commands) { e in
            e.setBuffer(b.arena.buffer, offset: b.q.offset, index: 0)
            e.setBuffer(b.arena.buffer, offset: b.k.offset, index: 1)
            e.setBuffer(b.arena.buffer, offset: b.v.offset, index: 2)
            e.setBuffer(b.arena.buffer, offset: b.attended.offset, index: 3)
            var params = QwenDiT.FlashParams(queries: UInt32(n), keys: UInt32(n), causal: 0, stride: 2048,
                                             scale: 1 / Float(QwenVisionTower.headDim).squareRoot() * Float(M_LOG2E), headBase: 0)
            e.setBytes(&params, length: MemoryLayout<QwenDiT.FlashParams>.stride, index: 4)
            e.dispatchThreadgroups(MTLSize(width: (n + 63) / 64, height: QwenVisionTower.heads, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        try compute("vision_unpad", commands: commands) { e in
            e.setBuffer(b.arena.buffer, offset: b.attended.offset, index: 0)
            e.setBuffer(b.arena.buffer, offset: b.a72.offset, index: 1)
            var r = UInt32(n)
            e.setBytes(&r, length: 4, index: 2)
            e.dispatchThreads(MTLSize(width: n * d, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        try linear(block + "attn_out", b, a: b.a72, c: b.normed, m: n, commands: commands)
        try add(b, b.x, b.normed, offset: 0, count: n * d, commands: commands)
        try layerNorm(b, b.x, into: b.normed, prefix: block + "ln2", rows: n, columns: d, commands: commands)
        try linear(block + "ffn_up", b, a: b.normed, c: b.h, m: n, commands: commands)
        try compute("gelu_tanh", commands: commands) { e in
            e.setBuffer(b.arena.buffer, offset: b.h.offset, index: 0)
            var c = UInt32(n * QwenVisionTower.hidden)
            e.setBytes(&c, length: 4, index: 1)
            e.dispatchThreads(MTLSize(width: Int(c), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        try linear(block + "ffn_down", b, a: b.h, c: b.normed, m: n, commands: commands)
        try add(b, b.x, b.normed, offset: 0, count: n * d, commands: commands)
    }

    /// `linear_fc2(GELU(linear_fc1(mergeIn)))` into `out` `[slots, 4096]`.
    func mergerMLP(_ b: Buffers, prefix: String, fc1: String, fc2: String, out: Arena.Slice, slots: Int,
                   commands: MTLCommandBuffer) throws {
        try linear("\(prefix).\(fc1)", b, a: b.mergeIn, c: b.mergeHidden, m: slots, commands: commands)
        try compute("gelu_erf", commands: commands) { e in
            e.setBuffer(b.arena.buffer, offset: b.mergeHidden.offset, index: 0)
            var c = UInt32(slots * 4 * QwenVisionTower.dim)
            e.setBytes(&c, length: 4, index: 1)
            e.dispatchThreads(MTLSize(width: Int(c), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        // The product writes padded rows; `out` holds only the real ones, so it goes through mergeIn.
        try linear("\(prefix).\(fc2)", b, a: b.mergeHidden, c: b.mergeIn, m: slots, commands: commands)
        let blit = commands.makeBlitCommandEncoder()!
        blit.copy(from: b.arena.buffer, sourceOffset: b.mergeIn.offset, to: b.arena.buffer, destinationOffset: out.offset,
                  size: slots * QwenVisionTower.outDim * 4)
        blit.endEncoding()
    }

    // MARK: - encoders

    func compute(_ name: String, commands: MTLCommandBuffer, _ body: (MTLComputeCommandEncoder) throws -> Void) throws {
        let encoder = commands.makeComputeCommandEncoder()!
        encoder.label = name
        encoder.setComputePipelineState(try context.pipeline(name))
        try body(encoder)
        encoder.endEncoding()
    }

    /// `C[m, n] = A[m, k] · W + bias`, W the file's Linear `[n, k]` unpacked to `[k, n]` fp32 in a reserve first; m
    /// padded to 64 as the text encoder's products.
    func linear(_ name: String, _ b: Buffers, a: Arena.Slice, c: Arena.Slice, m: Int, commands: MTLCommandBuffer) throws {
        let tensor = try file.tensor(name + ".weight")
        let (kk, nn) = (tensor.columns, tensor.rows)
        let reserve = b.reserves[nextReserve]
        nextReserve = (nextReserve + 1) % b.reserves.count
        try dequant.encode(tensor, of: file, into: b.arena.buffer, outOffset: reserve.offset, transposed: true, commands: commands)
        try product(b, a: a, w: reserve, c: c, m: m, k: kk, n: nn, commands: commands)
        try bias(b, c, name + ".bias", rows: m, columns: nn, commands: commands)
    }

    func product(_ b: Buffers, a: Arena.Slice, w: Arena.Slice, c: Arena.Slice, m: Int, k: Int, n: Int,
                 commands: MTLCommandBuffer) throws {
        let m = QwenDiT.padded(m)
        guard m * k * 4 <= a.bytes, m * n * 4 <= c.bytes, k * n * 4 <= w.bytes else {
            throw MetalContext.Failure.buffer("vision product \(m)×\(k)×\(n): no room")
        }
        let key = "\(m)×\(k)×\(n)"
        let product = products[key] ?? MPSMatrixMultiplication(device: context.device, transposeLeft: false, transposeRight: false,
                                                               resultRows: m, resultColumns: n, interiorColumns: k, alpha: 1, beta: 0)
        products[key] = product
        func matrix(_ offset: Int, _ rows: Int, _ columns: Int) -> MPSMatrix {
            MPSMatrix(buffer: b.arena.buffer, offset: offset,
                      descriptor: MPSMatrixDescriptor(rows: rows, columns: columns, rowBytes: columns * 4, dataType: .float32))
        }
        product.encode(commandBuffer: commands, leftMatrix: matrix(a.offset, m, k), rightMatrix: matrix(w.offset, k, n),
                       resultMatrix: matrix(c.offset, m, n))
    }

    func bias(_ b: Buffers, _ x: Arena.Slice, _ name: String, rows: Int, columns: Int, commands: MTLCommandBuffer) throws {
        guard let at = small[name] else { throw GGUFFile.Failure.unsupported("no tensor \(name)") }
        try compute("bias_rows", commands: commands) { e in
            e.setBuffer(b.arena.buffer, offset: x.offset, index: 0)
            e.setBuffer(b.arena.buffer, offset: b.smalls.offset + at * 4, index: 1)
            var p = SIMD2<UInt32>(UInt32(rows * columns), UInt32(columns))
            e.setBytes(&p, length: 8, index: 2)
            e.dispatchThreads(MTLSize(width: rows * columns, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
    }

    struct AffineParams { var columns: UInt32; var eps: Float }

    func layerNorm(_ b: Buffers, _ input: Arena.Slice, into out: Arena.Slice, prefix: String, rows: Int, columns: Int,
                   commands: MTLCommandBuffer) throws {
        guard let w = small[prefix + ".weight"], let bias = small[prefix + ".bias"] else {
            throw GGUFFile.Failure.unsupported("no tensor \(prefix)")
        }
        try compute("layernorm_affine", commands: commands) { e in
            e.setBuffer(b.arena.buffer, offset: input.offset, index: 0)
            e.setBuffer(b.arena.buffer, offset: out.offset, index: 1)
            e.setBuffer(b.arena.buffer, offset: b.smalls.offset + w * 4, index: 2)
            e.setBuffer(b.arena.buffer, offset: b.smalls.offset + bias * 4, index: 3)
            var p = AffineParams(columns: UInt32(columns), eps: QwenVisionTower.eps)
            e.setBytes(&p, length: MemoryLayout<AffineParams>.stride, index: 4)
            e.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
    }

    /// x += y (y at `offset` floats into its slice).
    func add(_ b: Buffers, _ x: Arena.Slice, _ y: Arena.Slice, offset: Int, count: Int, commands: MTLCommandBuffer) throws {
        try compute("add_inplace", commands: commands) { e in
            e.setBuffer(b.arena.buffer, offset: x.offset, index: 0)
            e.setBuffer(b.arena.buffer, offset: y.offset + offset * 4, index: 1)
            var c = UInt32(count)
            e.setBytes(&c, length: 4, index: 2)
            e.dispatchThreads(MTLSize(width: count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
    }
}
