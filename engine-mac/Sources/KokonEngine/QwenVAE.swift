import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

/// **The Qwen-Image 2.1 VAE decoder** (`AutoencoderKLQwenImage21.decode`, diffusers 0.42), as one MPSGraph per
/// latent size, fp32. For a single image every causal 3D convolution of the Wan 2.2 design is a 2D one and the
/// temporal convolutions never run (first chunk), so the graph is a plain 2D network:
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
    /// One compiled graph per latent size. Building and compiling the 1024² one takes 0.3 s; its first run then costs
    /// ~3.7 s more than the next ones (M2 Pro), GPU time with ~1 s of CPU, so the pipeline keeps its VAE between
    /// images rather than building it beside the DiT. fp16 convolutions (norms and attention in fp32) were wrong
    /// (2.7e-1: an overflow) and 5.7× slower, MPSGraph trying to move them to the Neural Engine: the VAE stays fp32.
    private var executables: [String: MPSGraphExecutable] = [:]
    private let lock = NSLock()

    public init(path: String, context: MetalContext) throws {
        file = try SafetensorsFile(path: path)
        self.context = context
        guard file.tensors["decoder.conv1.weight"] != nil, file.tensors["conv2.weight"] != nil else {
            throw GGUFFile.Failure.unsupported("\(path): not a Qwen-Image 2.1 VAE (Wan 2.2 names expected)")
        }
    }

    /// Decodes a normalized latent `[64, h, w]` (the DiT's packed tokens transposed) into RGBA `[4, 16h, 16w]` in
    /// [−1, 1].
    public func decode(_ latent: [Float], height h: Int, width w: Int) throws -> [Float] {
        let executable = try prepare(height: h, width: w)
        let input = MPSGraphTensorData(device: MPSGraphDevice(mtlDevice: context.device),
                                       data: latent.withUnsafeBytes { Data($0) },
                                       shape: [1, 64, NSNumber(value: h), NSNumber(value: w)], dataType: .float32)
        guard let data = executable.run(with: context.queue, inputs: [input], results: nil, executionDescriptor: nil).first else {
            throw MetalContext.Failure.kernel("VAE decode")
        }
        var out = [Float](repeating: 0, count: 4 * 16 * h * 16 * w)
        out.withUnsafeMutableBytes { data.mpsndarray().readBytes($0.baseAddress!, strideBytes: nil) }
        return out
    }

    /// Builds and compiles the decoder for an `h` × `w` latent, once.
    @discardableResult
    public func prepare(height h: Int, width w: Int) throws -> MPSGraphExecutable {
        try lock.withLock {
            let key = "\(h)×\(w)"
            if let e = executables[key] { return e }
            let built = try build(height: h, width: w)
            let shape = MPSGraphShapedType(shape: [1, 64, NSNumber(value: h), NSNumber(value: w)], dataType: .float32)
            let e = built.graph.compile(with: MPSGraphDevice(mtlDevice: context.device), feeds: [built.input: shape],
                                        targetTensors: [built.output], targetOperations: nil, compilationDescriptor: nil)
            executables[key] = e
            return e
        }
    }

    // MARK: - the graph

    private func build(height h: Int, width w: Int) throws -> (graph: MPSGraph, input: MPSGraphTensor, output: MPSGraphTensor) {
        let g = MPSGraph()
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

        let input = g.placeholder(shape: n([1, 64, h, w]), dataType: .float32, name: "latent")
        var x = g.addition(g.multiplication(input, constant(QwenVAE.latentsStd, [1, 64, 1, 1]), name: nil),
                           constant(QwenVAE.latentsMean, [1, 64, 1, 1]), name: nil)
        x = try conv(x, "conv2", 64, 64, 1)                                        // post_quant_conv
        let dims = QwenVAE.dims
        x = try conv(x, "decoder.conv1", 64, dims[0], 3)                           // conv_in
        x = try residual(x, "decoder.middle.0", dims[0], dims[0])                  // mid_block.resnets.0
        x = try attention(x, "decoder.middle.1", dims[0], h * w, h, w)            // mid_block.attentions.0
        x = try residual(x, "decoder.middle.2", dims[0], dims[0])                  // mid_block.resnets.1
        var (hh, ww) = (h, w)
        for i in 0..<5 {                                                           // up_blocks.i
            let (inC, outC) = (dims[i], dims[i + 1])
            let blockInput = x
            var c = inC
            for j in 0..<3 {
                x = try residual(x, "decoder.upsamples.\(i).upsamples.\(j)", c, outC)
                c = outC
            }
            if i < 4 {
                x = g.resize(x, size: n([2 * hh, 2 * ww]), mode: .nearest, centerResult: true, alignCorners: false,
                             layout: .NCHW, name: nil)
                x = try conv(x, "decoder.upsamples.\(i).upsamples.3.resample.1", outC, outC, 3)   // upsampler
                x = g.addition(x, dupUp(blockInput, inC, outC, QwenVAE.temporal[i], hh, ww), name: nil)   // avg_shortcut
                (hh, ww) = (2 * hh, 2 * ww)
            }
        }
        x = silu(try norm(x, "decoder.head.0.gamma", dims[5]))                    // norm_out
        x = try conv(x, "decoder.head.2", dims[5], 4, 3)                           // conv_out
        x = g.clamp(x, min: g.constant(-1, dataType: .float32), max: g.constant(1, dataType: .float32), name: nil)
        return (g, input, x)
    }
}
