import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// **Qwen-Image 2.1 Turbo, text to image**: `QwenImage21Pipeline.__call__` (diffusers 0.42) with the engine's parts.
///
///     prompt ─ QwenTextEncoder ─ conditioning [T, 4096]      (the encoder's file is unmapped before the DiT loads)
///     sd.cpp's noise for the seed [64, h, w] ─ packed [h·w, 64] ─ 8 × { v = DiT(x, σᵢ);  x += (σᵢ₊₁ − σᵢ) · v }   flow Euler
///     x ─ unpacked [64, h, w] ─ QwenVAE ─ RGBA [4, 16h, 16w] in [−1, 1] ─ 8-bit RGBA
///
/// The sigmas are Turbo's (`sample_sigmas` of the checkpoint, the ones Kokon already sends to sd.cpp); the
/// scheduler (shift 1, no dynamic shifting) uses them as they are and ends on 0. The DiT reads the timestep as σ.
public final class QwenPipeline {
    /// `planner::TURBO_SIGMAS` (crates/qls-core/src/planner.rs): the checkpoint's sample_sigmas, then 0.
    public static let turboSigmas: [Float] = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0]

    public let encoderPath, ditPath, vaePath: String
    let context: MetalContext
    /// Progress lines; stdout by default.
    public var log: (String) -> Void = { print($0) }
    /// Seconds per phase of the last run: encode, steps (each), dit load, vae, total.
    public private(set) var timings: [String: Double] = [:]
    /// Kept between images: its first decode of a size costs ~4 s more than the next ones at 1024² (M2 Pro). Compiled
    /// beside the DiT (from the second step) it slowed the steps by ~9 s to save 1.9: a long-lived engine pays it
    /// once per size, or ahead while idle (`QwenVAE.prepare`).
    private var vae: QwenVAE?

    public init(encoder: String, dit: String, vae: String) throws {
        encoderPath = encoder
        ditPath = dit
        vaePath = vae
        context = try MetalContext()
    }

    /// The output size diffusers works at: each side floored to a multiple of 32 (two latent cells).
    public static func workingSize(width: Int, height: Int) -> (width: Int, height: Int) {
        (width / 32 * 32, height / 32 * 32)
    }

    /// Text to image at `width` × `height` (multiples of 32). `noise`: a latent `[64, h, w]` to start from instead of
    /// the seed's; `conditioning`: the encoder's output `[T, 4096]` to use instead of encoding `prompt` (the checks);
    /// `step`: called after each step with its index and the packed latent `[h·w, 64]`.
    /// Returns the decoded image, RGBA `[4, height, width]` in [−1, 1].
    public func textToImage(prompt: String, width: Int, height: Int, seed: UInt64, noise: [Float]? = nil,
                            conditioning given: [Float]? = nil, sigmas: [Float] = QwenPipeline.turboSigmas,
                            step: ((Int, [Float]) -> Void)? = nil) throws -> [Float] {
        guard width % 32 == 0, height % 32 == 0, width >= 64, height >= 64 else {
            throw QwenSequence.Failure.layout("\(width)×\(height): each side must be a multiple of 32, at least 64")
        }
        let started = DispatchTime.now().uptimeNanoseconds
        timings = [:]
        func since(_ t: UInt64) -> Double { Double(DispatchTime.now().uptimeNanoseconds - t) * 1e-9 }
        let (h, w) = (height / 16, width / 16)
        let channels = QwenDiT.channels

        // 1. The prompt. The encoder and its 5 GB mapping are gone when this scope ends.
        var t = DispatchTime.now().uptimeNanoseconds
        let (conditioning, rows): ([Float], Int) = try {
            if let given { return (given, given.count / QwenDiT.dim) }
            let file = try GGUFFile(path: encoderPath)
            let encoder = try QwenTextEncoder(file: file, context: context)
            defer { encoder.release() }
            return try encoder.conditioning(prompt: prompt)
        }()
        timings["encode"] = since(t)
        log(String(format: "text encoder: %d tokens in %.2f s", rows, timings["encode"]!))

        // 2. Noise, packed: token y·w + x holds the 64 channels of cell (y, x).
        let start = noise ?? Noise.philox(seed: seed, count: channels * h * w)   // sd.cpp's, value for value
        guard start.count == channels * h * w else {
            throw QwenSequence.Failure.layout("noise of \(start.count) values for a \(channels)×\(h)×\(w) latent")
        }
        var x = QwenPipeline.pack(start, height: h, width: w)

        // 3. The steps.
        t = DispatchTime.now().uptimeNanoseconds
        let sequence = try QwenSequence.textToImage(text: rows, target: .init(height: h, width: w))
        try {
            let file = try GGUFFile(path: ditPath)
            defer { context.release(file) }
            let dit = try QwenDiT(file: file, context: context, sequence: sequence)
            try dit.prepareModulation([0] + sigmas.dropLast())
            timings["dit load"] = since(t)
            for i in 0..<(sigmas.count - 1) {
                let stepStart = DispatchTime.now().uptimeNanoseconds
                let velocity = try dit.forward(encoder: conditioning, target: x, sigma: sigmas[i])
                let dt = sigmas[i + 1] - sigmas[i]
                for j in x.indices { x[j] += dt * velocity[j] }
                timings["step \(i + 1)"] = since(stepStart)
                log(String(format: "step %d/%d (σ %.4f): %.2f s", i + 1, sigmas.count - 1, sigmas[i], timings["step \(i + 1)"]!))
                step?(i, x)
            }
        }()

        // 4. The image.
        t = DispatchTime.now().uptimeNanoseconds
        if vae == nil { vae = try QwenVAE(path: vaePath, context: context) }
        let image = try vae!.decode(QwenPipeline.unpack(x, height: h, width: w), height: h, width: w)
        timings["vae"] = since(t)
        timings["total"] = since(started)
        log(String(format: "VAE: %.2f s; total %.1f s", timings["vae"]!, timings["total"]!))
        return image
    }

    // MARK: - latents

    /// `[64, h, w]` → tokens `[h·w, 64]` (`_pack_latents`: a spatial flatten, then channels last).
    public static func pack(_ latent: [Float], height h: Int, width w: Int) -> [Float] {
        let c = latent.count / (h * w)
        var out = [Float](repeating: 0, count: latent.count)
        for ch in 0..<c { for p in 0..<(h * w) { out[p * c + ch] = latent[ch * h * w + p] } }
        return out
    }

    /// Tokens `[h·w, 64]` → `[64, h, w]` (`_unpack_latents`).
    public static func unpack(_ tokens: [Float], height h: Int, width w: Int) -> [Float] {
        let c = tokens.count / (h * w)
        var out = [Float](repeating: 0, count: tokens.count)
        for p in 0..<(h * w) { for ch in 0..<c { out[ch * h * w + p] = tokens[p * c + ch] } }
        return out
    }

    // MARK: - output

    /// RGBA `[4, H, W]` in [−1, 1] → interleaved 8-bit RGBA, as VaeImageProcessor.postprocess (`(x / 2 + 0.5)`
    /// clamped, `· 255`, rounded half to even).
    public static func pixels(_ image: [Float], width: Int, height: Int) -> [UInt8] {
        let plane = width * height, channels = image.count / plane
        var out = [UInt8](repeating: 255, count: plane * 4)
        for c in 0..<min(channels, 4) {
            for p in 0..<plane {
                let v = min(max(image[c * plane + p] / 2 + 0.5, 0), 1)
                out[p * 4 + c] = UInt8((v * 255).rounded(.toNearestOrEven))
            }
        }
        return out
    }

    /// Writes interleaved 8-bit RGBA (straight alpha) as a PNG.
    public static func writePNG(_ rgba: [UInt8], width: Int, height: Int, to path: String) throws {
        let data = Data(rgba) as CFData
        guard let provider = CGDataProvider(data: data),
              let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                                  space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                                  provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
              let destination = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL,
                                                                UTType.png.identifier as CFString, 1, nil) else {
            throw MetalContext.Failure.buffer("cannot write \(path)")
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw MetalContext.Failure.buffer("cannot write \(path)") }
    }
}
