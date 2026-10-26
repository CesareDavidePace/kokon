import Foundation

/// **The starting noise, as sd.cpp makes it**, so a seed gives the same latent as Kokon's sd.cpp engine.
///
/// sd.cpp's default generator (`CUDA_RNG`, `src/core/rng_philox.hpp` on the fork's `kokon` branch, a port of
/// AUTOMATIC1111's imitation of `torch.randn` on CUDA): value i of a draw is Philox 4×32-10 on the counter
/// (offset, 0, i, 0) with the key (seed low 32 bits, seed high 32 bits), its first two words through one
/// Box–Muller branch (`sin`), in fp32. The offset grows by one per draw; the first image of a seed is draw 0.
///
/// sd.cpp fills the latent `{W, H, C, 1}` (W contiguous), so value i lands at channel i / (h·w), cell i % (h·w):
/// the engine's `[64, h, w]` order. For a flow model the start is the noise itself (`latent · (1 − σ) + noise · σ`
/// with an empty latent and σ = 1).
public enum Noise {
    static let m0: UInt64 = 0xD251_1F53, m1: UInt64 = 0xCD9E_8D57
    static let w0: UInt32 = 0x9E37_79B9, w1: UInt32 = 0xBB67_AE85
    static let twoPow32Inv: Float = 2.3283064e-10
    static let twoPow32Inv2Pi: Float = twoPow32Inv * 6.2831855

    /// `PhiloxRNG(seed).randn(count)` after `offset` earlier draws.
    public static func philox(seed: UInt64, count: Int, offset: UInt32 = 0) -> [Float] {
        let key0 = UInt32(truncatingIfNeeded: seed), key1 = UInt32(truncatingIfNeeded: seed >> 32)
        var out = [Float](repeating: 0, count: count)
        out.withUnsafeMutableBufferPointer { buffer in
            nonisolated(unsafe) let out = buffer.baseAddress!
            let chunks = max(1, min(64, count / 4096))
            DispatchQueue.concurrentPerform(iterations: chunks) { chunk in
                for i in (chunk * count / chunks)..<((chunk + 1) * count / chunks) {
                    var (c0, c1, c2, c3) = (offset, UInt32(0), UInt32(truncatingIfNeeded: i), UInt32(0))
                    var (k0, k1) = (key0, key1)
                    for round in 0..<10 {
                        let v1 = UInt64(c0) &* m0, v2 = UInt64(c2) &* m1
                        c0 = UInt32(truncatingIfNeeded: v2 >> 32) ^ c1 ^ k0
                        c1 = UInt32(truncatingIfNeeded: v2)
                        c2 = UInt32(truncatingIfNeeded: v1 >> 32) ^ c3 ^ k1
                        c3 = UInt32(truncatingIfNeeded: v1)
                        if round < 9 { k0 &+= w0; k1 &+= w1 }
                    }
                    out[i] = boxMuller(Float(c0), Float(c1))
                }
            }
        }
        return out
    }

    /// sd.cpp's `box_muller`: `u`, `v` centred in their 2⁻³² bins, `sqrt(−2 ln u) · sin(v)`, fp32 throughout.
    /// `x · a + b` is one fused multiply-add, as Apple clang compiles it by default (`-ffp-contract=on`), and so
    /// Kokon's macOS engine (kokon.yml: CMake Release, no float flags): separate roundings change the last bit of
    /// some values. Checked bit for bit against rng_philox.hpp built that way (`tools/philox_reference.cpp`).
    @inline(__always) static func boxMuller(_ x: Float, _ y: Float) -> Float {
        let u = (twoPow32Inv / 2).addingProduct(x, twoPow32Inv)
        let v = (twoPow32Inv2Pi / 2).addingProduct(y, twoPow32Inv2Pi)
        return sqrtf(-2 * logf(u)) * sinf(v)
    }
}
