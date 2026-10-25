import XCTest
@testable import KokonEngine

/// The pipeline's glue, without a model: latent packing, the noise, the 8-bit conversion, the sizes.
final class QwenPipelineTests: XCTestCase {
    func testPackIsChannelsLastAndUnpackInvertsIt() {
        // [2 channels, 2 × 3]: value = 100 · channel + position.
        let (c, h, w) = (2, 2, 3)
        let latent = (0..<(c * h * w)).map { Float(($0 / (h * w)) * 100 + $0 % (h * w)) }
        let tokens = QwenPipeline.pack(latent, height: h, width: w)
        XCTAssertEqual(Array(tokens[0..<4]), [0, 100, 1, 101])          // token 0: both channels, then token 1
        XCTAssertEqual(Array(tokens[10..<12]), [5, 105])                // token 5 = cell (1, 2)
        XCTAssertEqual(QwenPipeline.unpack(tokens, height: h, width: w), latent)
    }

    func testNoiseIsSdCppsBitForBit() {
        // Values of sd.cpp's PhiloxRNG built as Kokon's macOS engine (tools/philox_reference.cpp): the first four
        // and the last of a 1024² latent's draw, then the first of the next draw.
        let cases: [(UInt64, [UInt32], UInt32)] = [
            (42, [0x3e46ace2, 0x400a53f3, 0xbe302e09, 0x3f595c00, 0x3fe8a248], 0x3e0e7a13),
            (711_449_798, [0x3f0c5c5e, 0x3e28992a, 0x3e0ac701, 0xbf825663, 0x3e9eeab0], 0xbfc42d4a),
        ]
        for (seed, want, next) in cases {
            let a = Noise.philox(seed: seed, count: 262_144)
            XCTAssertEqual([a[0], a[1], a[2], a[3], a[262_143]].map(\.bitPattern), want, "seed \(seed)")
            XCTAssertEqual(Noise.philox(seed: seed, count: 1, offset: 1)[0].bitPattern, next, "seed \(seed), next draw")
        }
    }

    func testNoiseIsStandardNormal() {
        let a = Noise.philox(seed: 7, count: 200_001)
        let mean = a.reduce(0, +) / Float(a.count)
        let variance = a.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Float(a.count)
        XCTAssertEqual(mean, 0, accuracy: 0.01)
        XCTAssertEqual(variance, 1, accuracy: 0.01)
        XCTAssert(a.allSatisfy(\.isFinite))
    }

    func testPixelsFollowVaeImageProcessor() {
        // One pixel, RGBA planes: −1 → 0, 1 → 255, beyond the range clamped, 0 → 127.5 rounded to even (128).
        let image: [Float] = [-1, 1, 3, 0]
        XCTAssertEqual(QwenPipeline.pixels(image, width: 1, height: 1), [0, 255, 255, 128])
        // Three channels: alpha stays opaque.
        XCTAssertEqual(QwenPipeline.pixels([-1, -1, -1], width: 1, height: 1), [0, 0, 0, 255])
    }

    func testWorkingSizeFloorsToTwoLatentCells() {
        XCTAssert(QwenPipeline.workingSize(width: 1024, height: 768) == (1024, 768))
        XCTAssert(QwenPipeline.workingSize(width: 1000, height: 1376) == (992, 1376))
    }

    func testTurboSigmasAreThePlannersAndDecrease() {
        XCTAssertEqual(QwenPipeline.turboSigmas.count, 9)
        XCTAssertEqual(QwenPipeline.turboSigmas.first, 1)
        XCTAssertEqual(QwenPipeline.turboSigmas.last, 0)
        XCTAssert(zip(QwenPipeline.turboSigmas, QwenPipeline.turboSigmas.dropFirst()).allSatisfy { $0 > $1 })
    }
}
