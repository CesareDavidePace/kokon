import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

/// **The Qwen-Image 2.1 VAE decoder** (`AutoencoderKLQwenImage21.decode`, diffusers 0.42), fp32: the latent's
/// conversion and the mid block as one MPSGraph on the whole latent, the up blocks as single operations (`ops`), the
/// last two in tiles (`split`, `tiles`); `build` writes the whole decoder as one graph, the reference. For a single
/// image every causal 3D convolution of the Wan 2.2 design is a 2D one and the temporal convolutions never run (first
/// chunk), so the decoder is a plain 2D network:
///
///     latent · std + mean ─ post_quant_conv (1×1) ─ conv_in (3×3, 64 → 1152)
///     mid: residual ─ single-head attention over all pixels ─ residual
///     5 up blocks (1152, 1152, 1152 → 576, 576 → 288, 288 → 144): 3 residual blocks, then (the first four)
///         nearest 2× + conv 3×3, plus the block input duplicated into the new pixels (DupUp3D)
///     RMS norm ─ SiLU ─ conv_out (3×3, 144 → 4: RGBA) ─ clamp [−1, 1]
///
/// residual block: x + conv2(SiLU(norm2(conv1(SiLU(norm1(x)))))), the skip through a 1×1 conv when widths differ.
/// norm: F.normalize over channels × √C × γ. Weights are read by their Wan 2.2 names (Kokon's file); diffusers'
/// names in the comments (`tools/reference_vae.py` maps between them).
public final class QwenVAE {
    public static let latentChannels = 64
    static let latentsMean: [Float] = [0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722, 1.8873, -1.7177,
        -0.651, 0.2732, 0.7562, -0.6163, -1.0277, 3.8363, 2.021, 0.0472, 0.932, 2.0087, 2.4954, -0.1391, -1.4249, 1.8464,
        -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955, -0.0342, -1.0265, -0.7636, 3.0555, 0.0746,
        -3.0751, -0.1076, 1.7376, -1.0914, -1.9435, -0.2784, -1.368, 0.4809, -0.4433, 0.3764, 0.5729, -2.0595, 1.096,
        -1.326, -2.0211, -5.0179, 0.5275, 4.0162, 1.8505, 0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304]
    static let latentsStd: [Float] = [3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.791, 3.0785, 3.65, 3.9308,
        3.0904, 2.8778, 3.7675, 3.732, 5.0756, 3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489, 3.4896,
        3.8513, 3.9323, 3.4719, 3.7498, 4.283, 3.5694, 4.2467, 3.9037, 3.2947, 5.077, 3.5075, 3.27, 3.4767, 2.8063, 5.1125,
        3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875, 3.9624, 4.0168, 3.5643, 4.055, 5.5614, 4.2963,
        4.408, 3.4959, 3.8747, 3.7608, 3.5735, 3.149, 3.7662, 3.6746, 3.4563, 3.8161]
    /// Channels per up block (diffusers' `dims`: base 144 × dim_mult [1, 2, 4, 8, 8] reversed, the first repeated).
    static let dims = [1152, 1152, 1152, 576, 288, 144]
    /// Temporal factor of each up block's DupUp3D shortcut (`temperal_upsample`); the last block does not upsample.
    static let temporal = [2, 2, 2, 1]

    let context: MetalContext
    let file: SafetensorsFile
    /// Compiled graphs: the mid stage per latent size, each convolution per input size. The whole decoder as one graph,
    /// compiled to the end (`waitForCompilationCompletion`): 2.7 s at 1024² (M2
    /// Pro), mostly the shader compiler, which otherwise ran inside the first decode (6.4 s against 2.75 s; with it
    /// up front the first decode is 4.15 s). Beside the DiT it slows the steps far more than it saves: the pipeline
    /// keeps its VAE between images, and an idle engine can prepare ahead. fp16 up blocks (norms and the mid block in
    /// fp32) are wrong (2.7e-1) and 3–11× slower only at optimization level 1, where MPSGraph places them on the Neural
    /// Engine; at level 0 they stay on the GPU and are right, but save little: all five in fp16 2.43 s against 2.77 at
    /// 1024², the first two 2.65, for 2.2–2.7e-3 of diffusers instead of 1.1e-5 (M2 Pro, `check-vae`, 2026-10-25). The
    /// VAE stays fp32.
    var executables: [String: MPSGraphExecutable] = [:]
    private var feedOrders: [String: [Int]] = [:]
    let lock = NSLock()

    public init(path: String, context: MetalContext) throws {
        file = try SafetensorsFile(path: path)
        self.context = context
        guard file.tensors["decoder.conv1.weight"] != nil, file.tensors["conv2.weight"] != nil else {
            throw GGUFFile.Failure.unsupported("\(path): not a Qwen-Image 2.1 VAE (Wan 2.2 names expected)")
        }
    }

    /// Where the decoder is cut into graphs, as up block numbers (0: after the mid block; `KOKON_VAE_SPLIT`, e.g.
    /// `0,1,2,3`), and the side of each stage's tiles at its input (0: the whole input; `KOKON_VAE_TILES`, one per
    /// stage). The first stage runs on the whole latent (the mid block's attention sees every pixel); the others may run
    /// on tiles that read `halo` pixels beyond their share, each a graph compiled once for every image size.
    /// By default up blocks 0–2 whole and 3–4 with the head in tiles of 288 (2 × 2 at 1024²): M2 Pro 1024²,
    /// alternated rounds, 2026-10-25: 3.0 s a decode (first 3.25), 1.67 GB kept after it, peak 4.6 GB; one graph 2.8 s
    /// (first 4.5–5.0), 8.5 GB kept, peak 9.7; the up blocks untiled 2.7 s, 2.25 GB kept, but a peak of 6.9 GB that
    /// grows with the image; tiles of 144 3.9 s, peak 3.85.
    public var split = QwenVAE.numbers("KOKON_VAE_SPLIT") ?? [0, 3]
    public var tiles = QwenVAE.numbers("KOKON_VAE_TILES") ?? [0, 0, 288]
    static func numbers(_ key: String) -> [Int]? {
        ProcessInfo.processInfo.environment[key].map { $0.split(separator: ",").compactMap { Int($0) } }
    }

    /// A decoder stage: up blocks `first..<end`, after the latent's conversion and the mid block when `first` is −1,
    /// with the head when `end` is 5.
    public struct Stage: Hashable, Sendable, CustomStringConvertible {
        public let first, end: Int
        public static let whole = Stage(first: -1, end: 5)
        /// Channels at the input of up block `b` (5: the image's 4) and that size relative to the latent.
        static func channels(_ b: Int) -> Int { b < 0 ? 64 : b < 5 ? QwenVAE.dims[b] : 4 }
        static func scale(_ b: Int) -> Int { b < 0 ? 1 : 1 << min(b, 4) }
        /// Output pixels per input pixel along a side.
        var up: Int { Stage.scale(end) / Stage.scale(first) }
        /// How far the stage's 3×3 convolutions reach, in its input's pixels: 6 in each up block at its size, the
        /// upsampler's at twice that, the head's at the image's; a tile needs this much beyond its share to give what
        /// the whole graph computes (up block 3 to the end: 6 + ½ + 3 + ½ = 10).
        var halo: Int {
            let at = { (b: Int) in Double(Stage.scale(b)) / Double(Stage.scale(first)) }
            var reach = 0.0
            for i in max(first, 0)..<end { reach += 6 / at(i) + (i < 4 ? 1 / at(i + 1) : 0) }
            if end == 5 { reach += 1 / at(5) }
            return Int(reach.rounded(.up))
        }
        public var description: String { "\(first)..<\(end)" }
    }

    /// The stages in order, with their tile sides.
    public var stages: [(stage: Stage, tile: Int)] {
        let bounds = [-1] + split.filter { $0 >= 0 && $0 < 5 } + [5]
        return zip(bounds, bounds.dropFirst()).enumerated().map { i, b in
            (Stage(first: b.0, end: b.1), b.0 >= 0 && i < tiles.count ? tiles[i] : 0)
        }
    }

    /// Up blocks as single operations on the engine's buffers (`KOKON_VAE_OPS=0`: as MPSGraph graphs). A compiled graph
    /// keeps every intermediate tensor after it runs: the whole decoder held 8.5 GB after a 1024² decode, its up blocks 1
    /// and 2 alone 2.4 and 4.1 GB; a graph that is one convolution (its bias, skip and shortcut fused) keeps nothing
    /// beyond the buffers it is given (M2 Pro, 2026-10-25). So each convolution is a graph, with weights as inputs
    /// (shared across blocks and sizes, as fast as constants), and the norms, the 2× upsampling and the DupUp shortcut
    /// are kernels (`vae.metal`), all on buffers from a pool that lives for one decode.
    public var ops = ProcessInfo.processInfo.environment["KOKON_VAE_OPS"] != "0"

    /// Decodes a normalized latent `[64, h, w]` (the DiT's packed tokens transposed) into RGBA `[4, 16h, 16w]` in
    /// [−1, 1]. Every stage and tile goes in one command buffer.
    public func decode(_ latent: [Float], height h: Int, width w: Int) throws -> [Float] {
        var commands = MPSCommandBuffer(from: context.queue)
        let pool = BufferPool(context: context)
        let memory = ProcessInfo.processInfo.environment["KOKON_VAE_MEMORY"] != nil   // waits after each stage to tell
        var current = try pool.take(latent.count)
        current.contents().copyMemory(from: latent, byteCount: 4 * latent.count)
        for (stage, tile) in stages {
            let (H, W) = (Stage.scale(stage.first) * h, Stage.scale(stage.first) * w)
            let next = try pool.take(Stage.channels(stage.end) * stage.up * H * stage.up * W)
            if tile > 2 * stage.halo, H > tile || W > tile {
                try encode(stage, tiles: tile, from: current, to: next, height: H, width: W, pool: pool, commands: commands)
            } else {
                try encode(stage, from: current, to: next, height: H, width: W, pool: pool, commands: commands)
            }
            pool.give(current)
            current = next
            if memory {
                commands.commit()
                commands.waitUntilCompleted()
                commands = MPSCommandBuffer(from: context.queue)
                print(String(format: "  after %@: footprint %.2f GB, pool %.2f GB", stage.description, MemoryUsage.now().footprint,
                             Double(pool.bytes) / Double(1 << 30)))
            }
        }
        commands.commit()
        commands.waitUntilCompleted()
        if let error = commands.error { throw MetalContext.Failure.kernel("VAE decode: \(error.localizedDescription)") }
        var out = [Float](repeating: 0, count: 4 * 16 * h * 16 * w)
        out.withUnsafeMutableBytes { $0.copyMemory(from: UnsafeRawBufferPointer(start: current.contents(), count: $0.count)) }
        return out
    }

    /// Buffers for one decode, handed out again once given back: the command buffer runs in order, so a buffer's next
    /// user waits for its last one.
    final class BufferPool {
        let context: MetalContext
        private var free: [MTLBuffer] = []
        private(set) var bytes = 0
        init(context: MetalContext) { self.context = context }
        /// A buffer of at least `floats` floats: the smallest free one that fits, else a new one.
        func take(_ floats: Int) throws -> MTLBuffer {
            if let i = free.indices.filter({ free[$0].length >= 4 * floats }).min(by: { free[$0].length < free[$1].length }) {
                return free.remove(at: i)
            }
            bytes += 4 * floats
            return try context.scratch(4 * floats, label: "VAE")
        }
        func give(_ buffer: MTLBuffer) { free.append(buffer) }
    }

    func data(_ buffer: MTLBuffer, _ shape: [Int]) -> MPSGraphTensorData {
        MPSGraphTensorData(buffer, shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
    }

    /// `stage` on an `H` × `W` input, whole.
    private func encode(_ stage: Stage, from input: MTLBuffer, to output: MTLBuffer, height H: Int, width W: Int,
                        pool: BufferPool, commands: MPSCommandBuffer) throws {
        if ops && stage.first >= 0 {
            return try encodeOperations(stage, from: input, to: output, height: H, width: W, pool: pool, commands: commands)
        }
        _ = try prepare(stage, height: H, width: W)
            .encode(to: commands, inputs: [data(input, [1, Stage.channels(stage.first), H, W])],
                    results: [data(output, [1, Stage.channels(stage.end), stage.up * H, stage.up * W])], executionDescriptor: nil)
    }

    /// `stage` on `H` × `W` in tiles of side `tile` (the same graphs for all of them), each tile's share written to
    /// `output`.
    private func encode(_ stage: Stage, tiles tile: Int, from input: MTLBuffer, to output: MTLBuffer, height H: Int, width W: Int,
                        pool: BufferPool, commands: MPSCommandBuffer) throws {
        let (ci, co, up) = (Stage.channels(stage.first), Stage.channels(stage.end), stage.up)
        let (th, tw) = (min(tile, H), min(tile, W))
        let tileIn = try pool.take(ci * th * tw), tileOut = try pool.take(co * up * th * up * tw)
        for y in QwenVAE.spans(H, tile: th, halo: stage.halo) {
            for x in QwenVAE.spans(W, tile: tw, halo: stage.halo) {
                try window(input, (H, W, y.start, x.start), tileIn, (th, tw, 0, 0), channels: ci, rows: th, cols: tw, commands: commands)
                try encode(stage, from: tileIn, to: tileOut, height: th, width: tw, pool: pool, commands: commands)
                try window(tileOut, (up * th, up * tw, up * (y.core - y.start), up * (x.core - x.start)), output,
                           (up * H, up * W, up * y.core, up * x.core), channels: co, rows: up * y.length, cols: up * x.length,
                           commands: commands)
            }
        }
        pool.give(tileIn)
        pool.give(tileOut)
    }

    // MARK: - up blocks as operations

    /// Up blocks `stage.first..<stage.end` (and the head) on `[C, H, W]` in `input`, as `QwenVAE.build` writes them.
    private func encodeOperations(_ stage: Stage, from input: MTLBuffer, to output: MTLBuffer, height H: Int, width W: Int,
                                  pool: BufferPool, commands: MPSCommandBuffer) throws {
        var (x, c, hh, ww) = (input, Stage.channels(stage.first), H, W)
        /// Gives `b` back unless it is the stage's input (the caller's).
        func done(_ b: MTLBuffer) { if b !== input { pool.give(b) } }
        for i in stage.first..<stage.end {
            let (inC, outC) = (QwenVAE.dims[i], QwenVAE.dims[i + 1])
            let blockInput = x
            for j in 0..<3 {
                let name = "decoder.upsamples.\(i).upsamples.\(j)"
                let t = try pool.take(c * hh * ww)
                try normSilu(x, to: t, gamma: name + ".residual.0.gamma", channels: c, pixels: hh * ww, commands: commands)
                let u = try pool.take(outC * hh * ww)
                try convolution(t, name + ".residual.2", inC: c, outC: outC, k: 3, height: hh, width: ww, to: u, commands: commands)
                pool.give(t)
                try normSilu(u, to: u, gamma: name + ".residual.3.gamma", channels: outC, pixels: hh * ww, commands: commands)
                let y = try pool.take(outC * hh * ww)
                try convolution(u, name + ".residual.6", inC: outC, outC: outC, k: 3, height: hh, width: ww, to: y,
                                skip: (x, c == outC ? nil : name + ".shortcut", c), commands: commands)
                pool.give(u)
                if x !== blockInput { done(x) }
                (x, c) = (y, outC)
            }
            if i < 4 {
                let up = try pool.take(outC * 4 * hh * ww)
                try dispatch("vae_nearest_up", [x, up], [UInt32(outC), UInt32(hh), UInt32(ww)],
                             grid: MTLSize(width: 2 * ww, height: 2 * hh, depth: outC), commands: commands)
                done(x)
                let y = i == stage.end - 1 ? output : try pool.take(outC * 4 * hh * ww)
                try convolution(up, "decoder.upsamples.\(i).upsamples.3.resample.1", inC: outC, outC: outC, k: 3,
                                height: 2 * hh, width: 2 * ww, to: y, commands: commands)
                pool.give(up)
                try dispatch("vae_dup_up_add", [y, blockInput], [UInt32(inC), UInt32(outC), UInt32(QwenVAE.temporal[i]), UInt32(hh), UInt32(ww)],
                             grid: MTLSize(width: 2 * ww, height: 2 * hh, depth: outC), commands: commands)
                done(blockInput)
                (x, hh, ww) = (y, 2 * hh, 2 * ww)
            } else if blockInput !== x {
                done(blockInput)
            }
        }
        if stage.end == 5 {
            let t = try pool.take(c * hh * ww)
            try normSilu(x, to: t, gamma: "decoder.head.0.gamma", channels: c, pixels: hh * ww, commands: commands)
            done(x)
            try convolution(t, "decoder.head.2", inC: c, outC: 4, k: 3, height: hh, width: ww, to: output, clamp: true, commands: commands)
            pool.give(t)
        } else if x !== output {
            // A stage that ends before the head ends on an upsampler, which writes `output` itself.
            preconditionFailure("VAE stage \(stage) did not end in its output")
        }
    }

    /// Weights as fp32 buffers, each read from the file once.
    private var weightBuffers: [String: MTLBuffer] = [:]
    func weight(_ name: String) throws -> MTLBuffer {
        try lock.withLock {
            if let b = weightBuffers[name] { return b }
            let values = try file.floats(name)
            let b = try context.scratch(4 * values.count, label: name)
            b.contents().copyMemory(from: values, byteCount: 4 * values.count)
            weightBuffers[name] = b
            return b
        }
    }

    func dispatch(_ kernel: String, _ buffers: [MTLBuffer], _ parameters: [UInt32], grid: MTLSize,
                          commands: MPSCommandBuffer) throws {
        let pipeline = try context.pipeline(kernel)
        var p = parameters
        let e = commands.rootCommandBuffer.makeComputeCommandEncoder()!
        e.setComputePipelineState(pipeline)
        for (i, b) in buffers.enumerated() { e.setBuffer(b, offset: 0, index: i) }
        e.setBytes(&p, length: 4 * p.count, index: buffers.count)
        let width = min(64, grid.width)
        e.dispatchThreads(grid, threadsPerThreadgroup: MTLSize(width: width, height: max(1, min(grid.height, 256 / width)), depth: 1))
        e.endEncoding()
    }

    /// QwenImage21RMS_norm then SiLU, per pixel over `channels` planes (`x` and `out` may be the same).
    func normSilu(_ x: MTLBuffer, to out: MTLBuffer, gamma: String, channels: Int, pixels: Int, commands: MPSCommandBuffer) throws {
        try dispatch("vae_norm_silu", [x, out, try weight(gamma)], [UInt32(channels), UInt32(pixels)],
                     grid: MTLSize(width: pixels, height: 1, depth: 1), commands: commands)
    }

    /// `channels` planes: `rows` × `cols` from (`src`, its size, at y, x) to (`dst`, its size, at y, x).
    private func window(_ src: MTLBuffer, _ srcSize: (Int, Int, Int, Int), _ dst: MTLBuffer, _ dstSize: (Int, Int, Int, Int),
                        channels: Int, rows: Int, cols: Int, commands: MPSCommandBuffer) throws {
        try dispatch("window_copy", [src, dst], [channels, rows, cols, srcSize.0, srcSize.1, srcSize.2, srcSize.3,
                                                 dstSize.0, dstSize.1, dstSize.2, dstSize.3].map { UInt32($0) },
                     grid: MTLSize(width: cols, height: rows, depth: channels), commands: commands)
    }

    /// `name` (a k × k convolution and its bias) on `[inC, h, w]`, plus `skip` (the residual block's input, through
    /// its 1×1 shortcut when named), clamped to [−1, 1] for the head. `downsample`: stride 2 with one zero row and
    /// column after the input (the encoder's `ZeroPad2d((0, 1, 0, 1))` then `Conv2d(stride=2)`), `[outC, h/2, w/2]`.
    func convolution(_ x: MTLBuffer, _ name: String, inC: Int, outC: Int, k: Int, height h: Int, width w: Int,
                     to out: MTLBuffer, skip: (buffer: MTLBuffer, shortcut: String?, channels: Int)? = nil,
                     clamp: Bool = false, downsample: Bool = false, commands: MPSCommandBuffer) throws {
        let kind = skip == nil ? "" : skip!.shortcut == nil ? "+x" : "+1x1(\(skip!.channels))"
        let key = "conv \(inC)→\(outC) \(k)×\(k)\(kind)\(clamp ? " clamp" : "")\(downsample ? " /2" : "") \(h)×\(w)"
        let (oh, ow) = downsample ? (h / 2, w / 2) : (h, w)
        let (executable, order) = try lock.withLock { () throws -> (MPSGraphExecutable, [Int]) in
            if let e = executables[key], let order = feedOrders[key] { return (e, order) }
            let g = MPSGraph()
            func n(_ v: [Int]) -> [NSNumber] { v.map { NSNumber(value: $0) } }
            var feeds: [MPSGraphTensor: MPSGraphShapedType] = [:], placeholders: [MPSGraphTensor] = []
            func input(_ shape: [Int]) -> MPSGraphTensor {
                let t = g.placeholder(shape: n(shape), dataType: .float32, name: nil)
                feeds[t] = MPSGraphShapedType(shape: n(shape), dataType: .float32)
                placeholders.append(t)
                return t
            }
            func conv(_ x: MPSGraphTensor, _ inC: Int, _ k: Int) -> MPSGraphTensor {
                let (stride, before, after) = downsample ? (2, 0, 1) : (1, k / 2, k / 2)
                let d = MPSGraphConvolution2DOpDescriptor(strideInX: stride, strideInY: stride, dilationRateInX: 1, dilationRateInY: 1,
                                                          groups: 1, paddingLeft: before, paddingRight: after, paddingTop: before,
                                                          paddingBottom: after, paddingStyle: .explicit, dataLayout: .NCHW,
                                                          weightsLayout: .OIHW)!
                let y = g.convolution2D(x, weights: input([outC, inC, k, k]), descriptor: d, name: nil)
                return g.addition(y, input([1, outC, 1, 1]), name: nil)
            }
            // Inputs in order: x, weight, bias, then skip, and the shortcut's weight and bias.
            let x = input([1, inC, h, w])
            var y = conv(x, inC, k)
            if let skip {
                let s = input([1, skip.channels, oh, ow])
                y = g.addition(y, skip.shortcut == nil ? s : conv(s, skip.channels, 1), name: nil)
            }
            if clamp {
                y = g.clamp(y, min: g.constant(-1, dataType: .float32), max: g.constant(1, dataType: .float32), name: nil)
            }
            let descriptor = MPSGraphCompilationDescriptor()
            descriptor.waitForCompilationCompletion = true
            let e = g.compile(with: MPSGraphDevice(mtlDevice: context.device), feeds: feeds, targetTensors: [y],
                              targetOperations: nil, compilationDescriptor: descriptor)
            // The executable takes its inputs in its own order (`feedTensors`), not the placeholders'.
            let order = (e.feedTensors ?? placeholders).map { t in placeholders.firstIndex { $0 === t }! }
            executables[key] = e
            feedOrders[key] = order
            return (e, order)
        }
        var inputs = [data(x, [1, inC, h, w]), data(try weight(name + ".weight"), [outC, inC, k, k]),
                      data(try weight(name + ".bias"), [1, outC, 1, 1])]
        if let skip {
            inputs.append(data(skip.buffer, [1, skip.channels, oh, ow]))
            if let shortcut = skip.shortcut {
                inputs += [data(try weight(shortcut + ".weight"), [outC, skip.channels, 1, 1]), data(try weight(shortcut + ".bias"), [1, outC, 1, 1])]
            }
        }
        _ = executable.encode(to: commands, inputs: order.map { inputs[$0] },
                              results: [data(out, [1, outC, oh, ow])], executionDescriptor: nil)
    }

    /// Tiles of side `tile` over `n` pixels: each one's start, and the share it gives (`core`, `length`) at least
    /// `halo` pixels from its edges unless that edge is the image's.
    static func spans(_ n: Int, tile: Int, halo: Int) -> [(start: Int, core: Int, length: Int)] {
        if n <= tile { return [(0, 0, n)] }
        let k = (n + tile - 2 * halo - 1) / (tile - 2 * halo)
        return (0..<k).map { i in
            let (a, b) = (n * i / k, n * (i + 1) / k)
            return (min(max(a - halo, 0), n - tile), a, b - a)
        }
    }

    /// Builds and compiles a stage of the decoder for an `h` × `w` input, once.
    @discardableResult
    public func prepare(_ stage: Stage = .whole, height h: Int, width w: Int) throws -> MPSGraphExecutable {
        try lock.withLock {
            let key = "\(stage) \(h)×\(w)"
            if let e = executables[key] { return e }
            let built = try build(stage, height: h, width: w)
            let shape = MPSGraphShapedType(shape: built.input.shape, dataType: .float32)
            let descriptor = MPSGraphCompilationDescriptor()
            descriptor.waitForCompilationCompletion = true
            let e = built.graph.compile(with: MPSGraphDevice(mtlDevice: context.device), feeds: [built.input: shape],
                                        targetTensors: [built.output], targetOperations: nil, compilationDescriptor: descriptor)
            executables[key] = e
            return e
        }
    }

    // MARK: - the graph

    /// The graph of `stage` for an `h` × `w` input.
    private func build(_ stage: Stage, height h: Int, width w: Int) throws -> (graph: MPSGraph, input: MPSGraphTensor, output: MPSGraphTensor) {
        let g = MPSGraph()
        let parts = VAEGraph(g: g, file: file)
        let input: MPSGraphTensor
        var x: MPSGraphTensor
        let dims = QwenVAE.dims
        var (hh, ww) = (h, w)
        if stage.first >= 0 {
            input = g.placeholder(shape: parts.n([1, Stage.channels(stage.first), h, w]), dataType: .float32, name: "input")
            x = input
        } else {
            input = g.placeholder(shape: parts.n([1, 64, h, w]), dataType: .float32, name: "latent")
            x = g.addition(g.multiplication(input, parts.constant(QwenVAE.latentsStd, [1, 64, 1, 1]), name: nil),
                           parts.constant(QwenVAE.latentsMean, [1, 64, 1, 1]), name: nil)
            x = try parts.conv(x, "conv2", 64, 64, 1)                                    // post_quant_conv
            x = try parts.conv(x, "decoder.conv1", 64, dims[0], 3)                       // conv_in
            x = try parts.residual(x, "decoder.middle.0", dims[0], dims[0])              // mid_block.resnets.0
            x = try parts.attention(x, "decoder.middle.1", dims[0], h * w, h, w)        // mid_block.attentions.0
            x = try parts.residual(x, "decoder.middle.2", dims[0], dims[0])              // mid_block.resnets.1
        }
        for i in max(stage.first, 0)..<stage.end {                                                          // up_blocks.i
            let (inC, outC) = (dims[i], dims[i + 1])
            let blockInput = x
            var c = inC
            for j in 0..<3 {
                x = try parts.residual(x, "decoder.upsamples.\(i).upsamples.\(j)", c, outC)
                c = outC
            }
            if i < 4 {
                x = g.resize(x, size: parts.n([2 * hh, 2 * ww]), mode: .nearest, centerResult: true, alignCorners: false,
                             layout: .NCHW, name: nil)
                x = try parts.conv(x, "decoder.upsamples.\(i).upsamples.3.resample.1", outC, outC, 3)   // upsampler
                x = g.addition(x, parts.dupUp(blockInput, inC, outC, QwenVAE.temporal[i], hh, ww), name: nil)   // avg_shortcut
                (hh, ww) = (2 * hh, 2 * ww)
            }
        }
        if stage.end < 5 { return (g, input, x) }
        x = parts.silu(try parts.norm(x, "decoder.head.0.gamma", dims[5]))                    // norm_out
        x = try parts.conv(x, "decoder.head.2", dims[5], 4, 3)                           // conv_out
        x = g.clamp(x, min: g.constant(-1, dataType: .float32), max: g.constant(1, dataType: .float32), name: nil)
        return (g, input, x)
    }
}

/// The VAE's pieces as MPSGraph operations, the file's weights as constants: the decoder's graphs (`build`) and the
/// encoder's mid block and head (`QwenVAEEncoder.swift`).
struct VAEGraph {
    let g: MPSGraph
    let file: SafetensorsFile

    func n(_ values: [Int]) -> [NSNumber] { values.map { NSNumber(value: $0) } }
    func constant(_ values: [Float], _ shape: [Int]) -> MPSGraphTensor {
        g.constant(values.withUnsafeBytes { Data($0) }, shape: n(shape), dataType: .float32)
    }
    func weight(_ name: String, _ shape: [Int]) throws -> MPSGraphTensor {
        let values = try file.floats(name)
        guard values.count == shape.reduce(1, *) else {
            throw GGUFFile.Failure.unsupported("\(name): \(values.count) values, \(shape) expected")
        }
        return constant(values, shape)
    }
    /// Conv (OIHW weights, the file's 5D `[O, I, 1, k, k]` squeezed) + bias, padding k / 2.
    func conv(_ x: MPSGraphTensor, _ name: String, _ inC: Int, _ outC: Int, _ k: Int) throws -> MPSGraphTensor {
        let pad = k / 2
        let d = MPSGraphConvolution2DOpDescriptor(strideInX: 1, strideInY: 1, dilationRateInX: 1, dilationRateInY: 1,
                                                  groups: 1, paddingLeft: pad, paddingRight: pad, paddingTop: pad,
                                                  paddingBottom: pad, paddingStyle: .explicit, dataLayout: .NCHW,
                                                  weightsLayout: .OIHW)!
        let y = g.convolution2D(x, weights: try weight(name + ".weight", [outC, inC, k, k]), descriptor: d, name: nil)
        return g.addition(y, try weight(name + ".bias", [1, outC, 1, 1]), name: nil)
    }
    /// QwenImage21RMS_norm: F.normalize over channels (x / max(‖x‖, 1e-12)) · √C · γ.
    func norm(_ x: MPSGraphTensor, _ name: String, _ c: Int) throws -> MPSGraphTensor {
        let length = g.squareRoot(with: g.reductionSum(with: g.square(with: x, name: nil), axes: [1], name: nil), name: nil)
        let normalized = g.division(x, g.maximum(length, g.constant(1e-12, dataType: .float32), name: nil), name: nil)
        let scaled = g.multiplication(normalized, g.constant(Double(Float(c).squareRoot()), dataType: .float32), name: nil)
        return g.multiplication(scaled, try weight(name, [1, c, 1, 1]), name: nil)
    }
    func silu(_ x: MPSGraphTensor) -> MPSGraphTensor { g.multiplication(x, g.sigmoid(with: x, name: nil), name: nil) }
    /// QwenImage21ResidualBlock (Wan `residual.{0,2,3,6}`, `shortcut`).
    func residual(_ x: MPSGraphTensor, _ name: String, _ inC: Int, _ outC: Int) throws -> MPSGraphTensor {
        let skip = inC == outC ? x : try conv(x, name + ".shortcut", inC, outC, 1)
        var y = silu(try norm(x, name + ".residual.0.gamma", inC))
        y = try conv(y, name + ".residual.2", inC, outC, 3)
        y = silu(try norm(y, name + ".residual.3.gamma", outC))
        y = try conv(y, name + ".residual.6", outC, outC, 3)
        return g.addition(y, skip, name: nil)
    }
    /// QwenImage21AttentionBlock: one head over all `hw` pixels.
    func attention(_ x: MPSGraphTensor, _ name: String, _ c: Int, _ hw: Int, _ hh: Int, _ ww: Int) throws -> MPSGraphTensor {
        let qkv = try conv(try norm(x, name + ".norm.gamma", c), name + ".to_qkv", c, 3 * c, 1)   // [1, 3C, H, W]
        let rows = g.transpose(g.reshape(qkv, shape: n([3 * c, hw]), name: nil), permutation: n([1, 0]), name: nil)
        func part(_ i: Int) -> MPSGraphTensor {
            g.reshape(g.sliceTensor(rows, dimension: 1, start: i * c, length: c, name: nil), shape: n([1, 1, hw, c]), name: nil)
        }
        let o = g.scaledDotProductAttention(query: part(0), key: part(1), value: part(2), mask: nil,
                                            scale: 1 / Float(c).squareRoot(), name: nil)
        let back = g.reshape(g.transpose(g.reshape(o, shape: n([hw, c]), name: nil), permutation: n([1, 0]), name: nil),
                             shape: n([1, c, hh, ww]), name: nil)
        return g.addition(try conv(back, name + ".proj", c, c, 1), x, name: nil)
    }
    /// QwenImage21DupUp3D on a first chunk: each output channel's 2×2 pixels from the input channels it repeats.
    func dupUp(_ x: MPSGraphTensor, _ inC: Int, _ outC: Int, _ ft: Int, _ hh: Int, _ ww: Int) -> MPSGraphTensor {
        let repeats = outC * ft * 4 / inC
        var y = g.broadcast(g.reshape(x, shape: n([inC, 1, hh, ww]), name: nil), shape: n([inC, repeats, hh, ww]), name: nil)
        y = g.reshape(y, shape: n([outC, ft, 2, 2, hh, ww]), name: nil)
        y = g.sliceTensor(y, dimension: 1, start: ft - 1, length: 1, name: nil)
        y = g.reshape(y, shape: n([outC, 2, 2, hh, ww]), name: nil)
        y = g.transpose(y, permutation: n([0, 3, 1, 4, 2]), name: nil)   // [C, H, 2, W, 2]
        return g.reshape(y, shape: n([1, outC, 2 * hh, 2 * ww]), name: nil)
    }
}
