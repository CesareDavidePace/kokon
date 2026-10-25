import XCTest
@testable import KokonEngine

/// The joint sequence and its RoPE positions, worked out by hand from diffusers 0.42
/// (`build_token_metadata`, `_qwenimage21_prefix_segments`, `QwenImage21Rope.forward`).
final class QwenSequenceTests: XCTestCase {
    func testTextToImage() throws {
        let s = try QwenSequence.textToImage(text: 3, target: .init(height: 2, width: 4))
        XCTAssertEqual(s.count, 11)
        XCTAssertEqual(s.prefix, 3)
        XCTAssertEqual(s.target, 8)
        XCTAssertEqual(s.segments, [.init(start: 0, end: 3, isText: true)])
        XCTAssertEqual(s.blockStarts, [3])
        XCTAssertEqual(s.textRows.map(\.joint), [0, 1, 2])
        let expected: [(Int, Int, Int)] = [(0, 0, 0), (1, 1, 1), (2, 2, 2),
                                           (3, -1, -2), (3, -1, -1), (3, -1, 0), (3, -1, 1),
                                           (3, 0, -2), (3, 0, -1), (3, 0, 0), (3, 0, 1)]
        XCTAssertEqual(s.positions.map { [$0.0, $0.1, $0.2] }, expected.map { [$0.0, $0.1, $0.2] })
    }

    /// Two text positions, a 2×2 reference (one slot), one text position, the 2×2 target.
    func testEditLayout() throws {
        let s = try QwenSequence(slots: [false, false, true, false, true],
                                 images: [.init(height: 2, width: 2), .init(height: 2, width: 2)])
        XCTAssertEqual(s.count, 11)
        XCTAssertEqual(s.imageIds, [-1, -1, 0, 0, 0, 0, -1, 1, 1, 1, 1])
        XCTAssertEqual(s.blockStarts, [2, 7])
        XCTAssertEqual(s.prefix, 7)
        XCTAssertEqual(s.segments, [.init(start: 0, end: 2, isText: true), .init(start: 2, end: 6, isText: false),
                                    .init(start: 6, end: 7, isText: true)])
        XCTAssertEqual(s.textRows.map { [$0.joint, $0.encoder] }, [[0, 0], [1, 1], [6, 3]])
        let expected = [[0, 0, 0], [1, 1, 1], [2, -1, -1], [2, -1, 0], [2, 0, -1], [2, 0, 0], [4, 4, 4],
                        [5, -1, -1], [5, -1, 0], [5, 0, -1], [5, 0, 0]]
        XCTAssertEqual(s.positions.map { [$0.0, $0.1, $0.2] }, expected)
    }

    /// Two adjacent references stay two blocks (ids from token counts, not from runs of slots).
    func testAdjacentReferencesStayTwoBlocks() throws {
        let s = try QwenSequence(slots: [false, true, true, true],
                                 images: [.init(height: 2, width: 2), .init(height: 2, width: 2), .init(height: 2, width: 2)])
        XCTAssertEqual(s.segments.map { [$0.start, $0.end] }, [[0, 1], [1, 5], [5, 9]])
        XCTAssertEqual(s.blockStarts, [1, 5, 9])
    }

    func testRefusesBadLayouts() {
        XCTAssertThrowsError(try QwenSequence(slots: [false, true], images: [.init(height: 4, width: 4)]))
        XCTAssertThrowsError(try QwenSequence(slots: [true, false], images: [.init(height: 2, width: 2)]))
    }

    /// Row 0 at position 0 is (cos 0, sin 0) = (1, 0) on every pair; a row's pairs follow ω = 1/θ^(2i/dim).
    func testRopeTable() throws {
        let s = try QwenSequence.textToImage(text: 2, target: .init(height: 2, width: 2))
        let table = s.ropeTable()
        XCTAssertEqual(table.count, s.count * 128)
        for j in 0..<64 { XCTAssertEqual(table[2 * j], 1); XCTAssertEqual(table[2 * j + 1], 0) }
        // Row 1 is position (1, 1, 1): pair 1 of the frame axis (dim 16) has ω = 1/10000^(2/16).
        let omega = 1 / powf(10000, 2.0 / 16.0)
        XCTAssertEqual(table[128 + 2], cosf(omega), accuracy: 1e-7)
        XCTAssertEqual(table[128 + 3], sinf(omega), accuracy: 1e-7)
    }
}
