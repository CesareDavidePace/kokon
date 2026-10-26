import Foundation
import XCTest
@testable import KokonEngine

/// The tokenizer on a tiny vocabulary: the 256 byte characters, a few merges and one added token. The real
/// vocabulary is checked against the official processor by `check-tokenizer`.
final class QwenTokenizerTests: XCTestCase {
    private var temporary: URL!
    private var tokenizer: QwenTokenizer!
    private var file: GGUFFile!

    override func setUpWithError() throws {
        temporary = FileManager.default.temporaryDirectory.appendingPathComponent("kokon-tokenizer-tests-\(UUID())")
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        let bytes = QwenTokenizer.byteCharacters.map { String($0) }
        let merges = ["h e", "t he", "Ġ c", "Ġc a", "Ġca t", "' s"]
        let tokens = bytes + ["he", "the", "Ġc", "Ġca", "Ġcat", "'s", "<|im_start|>"]
        let types = [Int32](repeating: 1, count: tokens.count - 1) + [3]

        func string(_ s: String) -> [UInt8] {
            var out: [UInt8] = []
            withUnsafeBytes(of: UInt64(s.utf8.count).littleEndian) { out += $0 }
            return out + Array(s.utf8)
        }
        func array(_ itemType: UInt32, _ items: [[UInt8]]) -> [UInt8] {
            var out: [UInt8] = []
            withUnsafeBytes(of: itemType.littleEndian) { out += $0 }
            withUnsafeBytes(of: UInt64(items.count).littleEndian) { out += $0 }
            return out + items.flatMap { $0 }
        }
        let path = try TestGGUF.write(in: temporary, metadata: [
            ("tokenizer.ggml.model", 8, string("gpt2")),
            ("tokenizer.ggml.pre", 8, string("qwen2")),
            ("tokenizer.ggml.tokens", 9, array(8, tokens.map(string))),
            ("tokenizer.ggml.merges", 9, array(8, merges.map(string))),
            ("tokenizer.ggml.token_type", 9, array(5, types.map { t in withUnsafeBytes(of: t.littleEndian) { Array($0) } })),
        ], tensors: [])
        file = try GGUFFile(path: path)
        tokenizer = try QwenTokenizer(file: file)
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: temporary) }

    private func id(_ token: String) -> Int { tokenizer.tokens.firstIndex(of: token)! }
    private func byte(_ b: UInt8) -> Int { id(String(QwenTokenizer.byteCharacters[Int(b)])) }

    func testByteTableIsGPT2s() {
        XCTAssertEqual(QwenTokenizer.byteCharacters[Int(UInt8(ascii: "a"))], "a")
        XCTAssertEqual(QwenTokenizer.byteCharacters[32], "Ġ")     // space
        XCTAssertEqual(QwenTokenizer.byteCharacters[10], "Ċ")     // newline
        XCTAssertEqual(Set(QwenTokenizer.byteCharacters).count, 256)
    }

    func testMergesByRankAcrossSplitPieces() {
        // "the" merges h+e first (rank 0), then t+he; " cat" is its own piece; "'s" is the contraction alternative.
        XCTAssertEqual(tokenizer.encode("the cat's"), [id("the"), id("Ġcat"), id("'s")])
        // No merge for "ht": bytes stay single.
        XCTAssertEqual(tokenizer.encode("ht"), [byte(UInt8(ascii: "h")), byte(UInt8(ascii: "t"))])
    }

    func testAddedTokensAreSplitOutFirst() {
        XCTAssertEqual(tokenizer.encode("the<|im_start|>the"), [id("the"), id("<|im_start|>"), id("the")])
        XCTAssertEqual(tokenizer.encode("<|im_start|>"), [id("<|im_start|>")])
    }

    func testTextIsNFCNormalized() {
        // e + combining acute becomes é (C3 A9) before the bytes are taken.
        XCTAssertEqual(tokenizer.encode("e\u{301}"), [byte(0xC3), byte(0xA9)])
        XCTAssertEqual(tokenizer.decode(tokenizer.encode("the cat é")), "the cat é")
    }

    /// Text generation's stream: whole UTF-8 characters only, added tokens left out, invalid bytes as U+FFFD (sd.cpp's
    /// TokenStreamDecoder).
    func testStreamDecoderHoldsPartialCharacters() {
        var stream = TokenStreamDecoder(tokenizer: tokenizer)
        XCTAssertEqual(stream.push(id("the")), "the")
        XCTAssertEqual(stream.push(byte(0xC3)), "")          // the first byte of é
        XCTAssertEqual(stream.push(id("<|im_start|>")), "")  // an added token: nothing
        XCTAssertEqual(stream.push(byte(0xA9)), "é")
        XCTAssertEqual(stream.push(byte(0xFF)), "\u{FFFD}")  // never valid
        XCTAssertEqual(stream.push(byte(0xE2)), "")
        XCTAssertEqual(stream.push(byte(UInt8(ascii: "a"))), "\u{FFFD}a")   // a broken sequence, then the next character
        XCTAssertEqual(stream.push(byte(0xF0)), "")
        XCTAssertEqual(stream.flush(), "\u{FFFD}")           // unfinished at the end
        XCTAssertEqual(stream.flush(), "")
    }
}
