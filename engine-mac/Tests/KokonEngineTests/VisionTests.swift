import XCTest
@testable import KokonEngine

/// The vision tower's processor, without a model: the size it encodes at, the white composite, the resize.
final class VisionTests: XCTestCase {
    func testVisionSizeKeepsTheImageInsideTheProcessorsBounds() {
        XCTAssert(QwenVisionTower.visionSize(width: 1024, height: 768) == (1024, 768))
        XCTAssert(QwenVisionTower.visionSize(width: 320, height: 224) == (320, 224))       // 71 680 pixels: kept
        XCTAssert(QwenVisionTower.visionSize(width: 256, height: 192) == (320, 224))       // under 65 536: the processor's
        XCTAssert(QwenVisionTower.visionSize(width: 64, height: 64) == (256, 256))
    }

    func testAlphaIsCompositedOverWhiteAsPILPastes() {
        // Opaque, transparent and half-transparent pixels; PIL: white.paste(img, mask=alpha) on 8-bit values.
        let rgba: [UInt8] = [10, 200, 30, 255, 10, 200, 30, 0, 0, 100, 255, 128, 77, 77, 77, 1]
        let input = QwenVisionTower.pixels(rgba: rgba + [UInt8](repeating: 255, count: 4 * (256 * 256 - 4)), width: 256, height: 256)
        let plane = 256 * 256
        func level(_ c: Int, _ p: Int) -> Int { Int(((input.rgb[c * plane + p] * 0.5 + 0.5) * 255).rounded()) }
        XCTAssertEqual([level(0, 0), level(1, 0), level(2, 0)], [10, 200, 30])
        XCTAssertEqual([level(0, 1), level(1, 1), level(2, 1)], [255, 255, 255])
        // (v·128 + 255·127) / 255, rounded: 0 → 127, 100 → 177, 255 → 255.
        XCTAssertEqual([level(0, 2), level(1, 2), level(2, 2)], [127, 177, 255])
        XCTAssertEqual(level(0, 3), 254)
    }

    func testResizeKeepsFlatImagesFlatAndCoversTheOutput() {
        let flat = [UInt8](repeating: 90, count: 3 * 192 * 256)
        let up = QwenVisionTower.resizeBicubic(flat, width: 256, height: 192, to: 320, 224)
        XCTAssertEqual(up.count, 3 * 320 * 224)
        XCTAssert(up.allSatisfy { $0 == 90 })
        // A horizontal ramp stays monotonic (no ringing on a smooth ramp) and keeps its ends.
        var ramp = [UInt8](repeating: 0, count: 3 * 192 * 256)
        for y in 0..<192 { for x in 0..<256 { for c in 0..<3 { ramp[(y * 256 + x) * 3 + c] = UInt8(x) } } }
        let r = QwenVisionTower.resizeBicubic(ramp, width: 256, height: 192, to: 320, 224)
        let row = (0..<320).map { Int(r[(100 * 320 + $0) * 3]) }
        XCTAssertEqual(row.first, 0)
        XCTAssertEqual(row.last, 255)
        XCTAssert(zip(row, row.dropFirst()).allSatisfy { $0 <= $1 })
    }
}
