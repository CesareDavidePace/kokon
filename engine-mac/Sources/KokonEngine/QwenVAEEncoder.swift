import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

/// **The Qwen-Image 2.1 VAE encoder** (`AutoencoderKLQwenImage21.encode`, diffusers 0.42), fp32, for the reference
/// images of edits: the latent the edit pipeline uses (`_encode_vae_image`: the posterior's mode, `(z − mean) / std`).
/// For a single image the Wan 2.2 design is a 2D network, as the decoder (`QwenVAE`):
///
///     RGBA in [−1, 1] ─ conv_in (3×3, 4 → 96)
///     5 down blocks (96 → 96, 96 → 192, 192 → 384, 384 → 768, 768 → 768): 2 residual blocks, then (the first four)
///         zero pad right/bottom + conv 3×3 stride 2, plus the block input averaged down (AvgDown3D)
///     mid: residual ─ single-head attention over all pixels ─ residual
///     RMS norm ─ SiLU ─ conv_out (3×3, 768 → 128) ─ quant_conv (1×1) ─ the first 64 channels (the mode)
///
/// The down blocks run as single operations (the decoder's convolution graphs and `vae.metal` kernels, on pooled
/// buffers), the mid block and the head as one graph at 1/16 of the image (its attention sees every pixel).
extension QwenVAE {
    /// Encoder channels (diffusers' `dims`: base 96 × [1, 1, 2, 4, 8, 8]) and each down block's temporal factor
    /// (`temperal_downsample` [False, True, True, True]; the last block does not downsample).
    static let encoderDims = [96, 96, 192, 384, 768, 768]
    static let encoderTemporal = [1, 2, 2, 2, 1]

    /// The normalized latent `[64, H/16, W/16]` of an RGBA image `[4, H, W]` in [−1, 1] (sides multiples of 16).
    /// In its own autorelease pool: the command buffer and the pooled buffers go when it returns, whoever calls it (a
    /// loop on a thread without a pool kept 1.5 GB per 1024² encode, M2 Pro, 2026-10-25).
    public func encode(_ image: [Float], height H: Int, width W: Int) throws -> [Float] {
        try autoreleasepool { try encodeImage(image, height: H, width: W) }
    }

    private func encodeImage(_ image: [Float], height H: Int, width W: Int) throws -> [Float] {
        guard H % 16 == 0, W % 16 == 0, image.count == 4 * H * W else {
            throw QwenSequence.Failure.layout("VAE encoder: \(image.count) values for 4 × \(H) × \(W) (sides multiples of 16)")
        }
        guard file.tensors["encoder.conv1.weight"] != nil else { throw GGUFFile.Failure.unsupported("the VAE file has no encoder") }
        let commands = MPSCommandBuffer(from: context.queue)
        let pool = BufferPool(context: context)
        let dims = QwenVAE.encoderDims
        let input = try pool.take(image.count)
        input.contents().copyMemory(from: image, byteCount: 4 * image.count)
        var (hh, ww) = (H, W)
        var x = try pool.take(dims[0] * hh * ww)
        try convolution(input, "encoder.conv1", inC: 4, outC: dims[0], k: 3, height: hh, width: ww, to: x, commands: commands)
        pool.give(input)
        for i in 0..<5 {                                                                         // down_blocks.i
            let (inC, outC) = (dims[i], dims[i + 1])
            let blockInput = x
            var c = inC
            for j in 0..<2 {
                let name = "encoder.downsamples.\(i).downsamples.\(j)"
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
                if x !== blockInput { pool.give(x) }
                (x, c) = (y, outC)
            }
            let fs = i < 4 ? 2 : 1
            if i < 4 {                                                                           // downsampler
                let y = try pool.take(outC * (hh / 2) * (ww / 2))
                try convolution(x, "encoder.downsamples.\(i).downsamples.2.resample.1", inC: outC, outC: outC, k: 3,
                                height: hh, width: ww, to: y, downsample: true, commands: commands)
                pool.give(x)
                (x, hh, ww) = (y, hh / 2, ww / 2)
            }
            try dispatch("vae_avg_down_add", [x, blockInput],                                     // avg_shortcut
                         [UInt32(inC), UInt32(outC), UInt32(QwenVAE.encoderTemporal[i]), UInt32(fs), UInt32(hh), UInt32(ww)],
                         grid: MTLSize(width: ww, height: hh, depth: outC), commands: commands)
            pool.give(blockInput)
        }
        let latent = try pool.take(64 * hh * ww)
        _ = try prepareEncoderTail(height: hh, width: ww)
            .encode(to: commands, inputs: [data(x, [1, dims[5], hh, ww])], results: [data(latent, [1, 64, hh, ww])],
                    executionDescriptor: nil)
        commands.commit()
        commands.waitUntilCompleted()
        if let error = commands.error { throw MetalContext.Failure.kernel("VAE encode: \(error.localizedDescription)") }
        var out = [Float](repeating: 0, count: 64 * hh * ww)
        out.withUnsafeMutableBytes { $0.copyMemory(from: UnsafeRawBufferPointer(start: latent.contents(), count: $0.count)) }
        return out
    }

    /// The mid block, the head, quant_conv and the normalization as one graph for an `h` × `w` input, once per size.
    func prepareEncoderTail(height h: Int, width w: Int) throws -> MPSGraphExecutable {
        try lock.withLock {
            let key = "encoder tail \(h)×\(w)"
            if let e = executables[key] { return e }
            let g = MPSGraph()
            let parts = VAEGraph(g: g, file: file)
            let c = QwenVAE.encoderDims[5]
            let input = g.placeholder(shape: parts.n([1, c, h, w]), dataType: .float32, name: "input")
            var x = try parts.residual(input, "encoder.middle.0", c, c)                          // mid_block.resnets.0
            x = try parts.attention(x, "encoder.middle.1", c, h * w, h, w)                      // mid_block.attentions.0
            x = try parts.residual(x, "encoder.middle.2", c, c)                                 // mid_block.resnets.1
            x = parts.silu(try parts.norm(x, "encoder.head.0.gamma", c))                        // norm_out
            x = try parts.conv(x, "encoder.head.2", c, 128, 3)                                  // conv_out
            x = try parts.conv(x, "conv1", 128, 128, 1)                                         // quant_conv
            x = g.sliceTensor(x, dimension: 1, start: 0, length: 64, name: nil)                 // the mode: the mean
            x = g.division(g.subtraction(x, parts.constant(QwenVAE.latentsMean, [1, 64, 1, 1]), name: nil),
                           parts.constant(QwenVAE.latentsStd, [1, 64, 1, 1]), name: nil)
            let descriptor = MPSGraphCompilationDescriptor()
            descriptor.waitForCompilationCompletion = true
            let e = g.compile(with: MPSGraphDevice(mtlDevice: context.device),
                              feeds: [input: MPSGraphShapedType(shape: parts.n([1, c, h, w]), dataType: .float32)],
                              targetTensors: [x], targetOperations: nil, compilationDescriptor: descriptor)
            executables[key] = e
            return e
        }
    }
}
