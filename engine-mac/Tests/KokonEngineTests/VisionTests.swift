import ImageIO
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

/// The server's edit requests, without a model: references decoded, sd.cpp's resize rule, what is refused.
final class EditRequestTests: XCTestCase {
    func png(_ width: Int, _ height: Int) throws -> String {
        let rgba = (0..<(width * height * 4)).map { UInt8(truncatingIfNeeded: $0 * 7) }
        return try QwenPipeline.pngData(rgba, width: width, height: height).base64EncodedString()
    }

    func body(_ extra: [String: Any]) throws -> Data {
        var json: [String: Any] = ["prompt": "make it blue", "width": 512, "height": 384, "seed": 3]
        json.merge(extra) { $1 }
        return try JSONSerialization.data(withJSONObject: json)
    }

    func testReferencesAreDecodedAndKeptAtTheirSizeWithoutResize() throws {
        let image = try png(320, 224)
        let (r, reason) = EngineServer.ImageRequest.parse(try body(["ref_images": ["data:image/png;base64," + image],
                                                                    "ref_image_args": "resize_before_vae=false"]))
        XCTAssertNil(reason)
        XCTAssertEqual(r?.references.count, 1)
        XCTAssertEqual(r.map { [$0.references[0].width, $0.references[0].height] }, [320, 224])
        XCTAssertEqual(r?.references[0].rgba[4 * 5 + 1], UInt8(truncatingIfNeeded: (4 * 5 + 1) * 7))   // straight RGBA as sent
    }

    func testResizeBeforeVAEScalesToTheOutputsArea() throws {
        // 512 × 384 output, a 320 × 224 reference: √(196 608 · 320 / 224) = 530 → 544; 530 · 224 / 320 = 371 → 384.
        let (r, reason) = EngineServer.ImageRequest.parse(try body(["ref_images": [try png(320, 224)]]))
        XCTAssertNil(reason)
        XCTAssertEqual(r.map { [$0.references[0].width, $0.references[0].height] }, [544, 384])
    }

    func testRefusals() throws {
        XCTAssertNotNil(EngineServer.ImageRequest.parse(try body(["ref_images": [try png(320, 224)]]), vision: false).1)
        XCTAssertNotNil(EngineServer.ImageRequest.parse(try body(["ref_images": [try png(320, 224)], "mask_image": "x"])).1)
        XCTAssertNotNil(EngineServer.ImageRequest.parse(try body(["ref_images": [try png(336, 224)],
                                                                  "ref_image_args": "resize_before_vae=false"])).1)   // 336 % 32
        XCTAssertNotNil(EngineServer.ImageRequest.parse(try body(["ref_images": ["not an image"]])).1)
    }

    /// 8-bit gray PNG, as qls-core sends masks (`gray_png_b64`).
    func grayPNG(_ gray: [UInt8], _ width: Int, _ height: Int) throws -> String {
        let data = NSMutableData()
        let provider = CGDataProvider(data: Data(gray) as CFData)!
        let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: width,
                            space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGBitmapInfo(rawValue: 0), provider: provider,
                            decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
        let destination = CGImageDestinationCreateWithData(data, "public.png" as CFString, 1, nil)!
        CGImageDestinationAddImage(destination, image, nil)
        XCTAssert(CGImageDestinationFinalize(destination))
        return (data as Data).base64EncodedString()
    }

    func testInitImageAndMaskAreReadAsSent() throws {
        let gray = (0..<(512 * 384)).map { UInt8(truncatingIfNeeded: $0 * 13) }
        let (r, reason) = EngineServer.ImageRequest.parse(try body(["init_image": try png(512, 384), "mask_image": try grayPNG(gray, 512, 384)]))
        XCTAssertNil(reason)
        XCTAssertEqual(r?.initImage?.mask, gray)   // gray values as stored, no color conversion
        XCTAssertEqual(r?.initImage?.strength, 0.75)   // sd-server's default
        XCTAssertEqual(r?.initImage?.image.rgba[4 * 5 + 1], UInt8(truncatingIfNeeded: (4 * 5 + 1) * 7))
        let plain = EngineServer.ImageRequest.parse(try body(["init_image": try png(512, 384), "strength": 1])).0
        XCTAssertNil(plain?.initImage?.mask)
        XCTAssertEqual(plain?.initImage?.strength, 1)
        XCTAssertNil(EngineServer.ImageRequest.parse(try body([:])).0?.initImage)
    }

    func testInitImageRefusals() throws {
        XCTAssertNotNil(EngineServer.ImageRequest.parse(try body(["init_image": try png(256, 384)])).1)   // not the output's size
        XCTAssertNotNil(EngineServer.ImageRequest.parse(try body(["mask_image": try png(512, 384)])).1)   // a mask alone
        XCTAssertNotNil(EngineServer.ImageRequest.parse(try body(["init_image": try png(512, 384), "mask_image": try png(512, 256)])).1)
        XCTAssertNotNil(EngineServer.ImageRequest.parse(try body(["init_image": try png(512, 384), "strength": 1.5])).1)
    }
}

/// The text encoder's side of images, without a model: Qwen3-VL's 3D positions and the templates.
final class ImagePromptTests: XCTestCase {
    /// get_rope_index, worked out by hand: 2 text tokens, a 2 × 3 grid of slots, 2 text tokens.
    func testPositionsOfAnImageAndTheTextAfterIt() throws {
        let pad = 9
        let ids = [1, 2] + [Int](repeating: pad, count: 6) + [3, 4]
        let p = try QwenTextEncoder.positions(ids: ids, imagePad: pad, grids: [(rows: 2, columns: 3)])
        let want: [[Int32]] = [[0, 0, 0], [1, 1, 1],
                               [2, 2, 2], [2, 2, 3], [2, 2, 4], [2, 3, 2], [2, 3, 3], [2, 3, 4],
                               [5, 5, 5], [6, 6, 6]]   // after the image: 2 + max(2, 3)
        XCTAssertEqual(p.map { [$0[0], $0[1], $0[2]] }, want)
        XCTAssertThrowsError(try QwenTextEncoder.positions(ids: ids, imagePad: pad, grids: [(rows: 2, columns: 2)]))
        XCTAssertEqual(try QwenTextEncoder.positions(ids: [1, 2, 3], imagePad: -1, grids: []).map { $0[1] }, [0, 1, 2])
    }

    /// The M-RoPE table: text rows equal the 1D table; an image row turns frequency 1 with its row, 2 with its column.
    func testMRopeInterleavesTheAxes() {
        let text = QwenTextEncoder.ropeTable(rows: 3)
        XCTAssertEqual(QwenTextEncoder.ropeTable(positions: (0..<3).map { SIMD3(repeating: Int32($0)) }), text)
        let image = QwenTextEncoder.ropeTable(positions: [SIMD3(4, 7, 9)])
        let omega = (0..<64).map { 1 / powf(5_000_000, Float(2 * $0) / 128) }
        for (i, axis) in [(0, 4), (1, 7), (2, 9), (3, 4), (60, 4), (61, 4), (62, 4), (63, 4)] {
            XCTAssertEqual(image[2 * i], cosf(Float(axis) * omega[i]), accuracy: 1e-7, "frequency \(i)")
        }
    }

    func testTemplates() {
        XCTAssertEqual(QwenTextEncoder.imageToImage("make it blue", images: 2),
                       "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n"
                       + "<image1><|vision_start|><|image_pad|><|vision_end|> <image2><|vision_start|><|image_pad|><|vision_end|>"
                       + "make it blue<|im_end|>\n<|im_start|>assistant\n")
        // sd.cpp's text generation: one image unlabelled, several labelled.
        XCTAssertEqual(QwenTextGenerator.template(system: "", prompt: "p", images: 1),
                       "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>p<|im_end|>\n<|im_start|>assistant\n")
        XCTAssertEqual(QwenTextGenerator.template(system: "s", prompt: "p", images: 2),
                       "<|im_start|>system\ns<|im_end|>\n<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|>"
                       + " <image2><|vision_start|><|image_pad|><|vision_end|>p<|im_end|>\n<|im_start|>assistant\n")
    }
}
