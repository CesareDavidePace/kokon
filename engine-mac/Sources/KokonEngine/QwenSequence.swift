import Foundation

/// **Qwen-Image 2.1's joint sequence**, as `QwenImage21Transformer2DModel.forward` builds it (diffusers 0.42): the
/// encoder's positions in order, each image slot standing for 2×2 latent tokens (so 4 rows), the target's slots
/// appended. Image tokens fill their rows row-major, conditions first, the target last.
public struct QwenSequence: Sendable, Equatable {
    public struct Grid: Sendable, Equatable { public let height, width: Int
        public init(height: Int, width: Int) { self.height = height; self.width = width }
        public var tokens: Int { height * width }
    }
    /// A run of prefix rows with the same image id: text (causal) or one condition image (bidirectional).
    public struct Segment: Sendable, Equatable { public let start, end: Int; public let isText: Bool }

    public let images: [Grid]
    /// Per joint row: -1 for text, else the image's index (`image_ids`).
    public let imageIds: [Int]
    /// The joint row of each text row, in encoder order, with its encoder row.
    public let textRows: [(joint: Int, encoder: Int)]
    /// First joint row of each image.
    public let blockStarts: [Int]
    public let segments: [Segment]
    /// (frame, height, width) RoPE positions per joint row.
    public let positions: [(Int, Int, Int)]

    public var count: Int { imageIds.count }
    public var target: Int { images.last!.tokens }
    public var prefix: Int { count - target }

    public static func == (a: QwenSequence, b: QwenSequence) -> Bool {
        a.images == b.images && a.imageIds == b.imageIds && a.blockStarts == b.blockStarts && a.segments == b.segments
    }

    public enum Failure: Error, CustomStringConvertible {
        case layout(String)
        public var description: String { if case .layout(let s) = self { return "sequence: \(s)" }; return "" }
    }

    /// Text to image: `text` encoder rows, then the target.
    public static func textToImage(text: Int, target: Grid) throws -> QwenSequence {
        try QwenSequence(slots: [Bool](repeating: false, count: text) + [Bool](repeating: true, count: target.tokens / 4),
                         images: [target])
    }

    /// `slots`: the encoder's image-slot mask with the target's slots appended; `images`: conditions, then target.
    public init(slots: [Bool], images: [Grid]) throws {
        guard let last = images.last else { throw Failure.layout("no target image") }
        guard images.allSatisfy({ $0.tokens % 4 == 0 }) else { throw Failure.layout("an image grid is not a multiple of 4 tokens") }
        guard 4 * slots.filter({ $0 }).count == images.reduce(0, { $0 + $1.tokens }) else {
            throw Failure.layout("\(slots.filter { $0 }.count) slots for \(images.reduce(0) { $0 + $1.tokens }) image tokens")
        }
        guard slots.suffix(last.tokens / 4).allSatisfy({ $0 }) else { throw Failure.layout("the target's slots are not last") }
        self.images = images
        var isImage: [Bool] = []
        var text: [(Int, Int)] = []
        for (position, slot) in slots.enumerated() {
            if slot { isImage += [true, true, true, true] } else { text.append((isImage.count, position)); isImage.append(false) }
        }
        textRows = text
        // Image ids from the token counts, not from runs of slots: two adjacent images stay two blocks.
        var ids = [Int](repeating: -1, count: isImage.count), starts: [Int] = []
        var block = 0, used = 0
        for row in isImage.indices where isImage[row] {
            if used == 0 { starts.append(row) }
            ids[row] = block
            used += 1
            if used == images[block].tokens { block += 1; used = 0 }
        }
        imageIds = ids
        blockStarts = starts
        for (i, s) in starts.enumerated() where !(s..<(s + images[i].tokens)).allSatisfy({ ids[$0] == i }) {
            throw Failure.layout("image \(i) is not contiguous")
        }
        let prefix = isImage.count - last.tokens
        var segments: [Segment] = []
        var start = 0
        if prefix > 0 {
            for index in 1...prefix where index == prefix || ids[index] != ids[start] {
                segments.append(Segment(start: start, end: index, isText: ids[start] < 0))
                start = index
            }
        }
        self.segments = segments
        // QwenImage21Rope.forward: text advances one position on all axes; an image freezes the frame at the
        // current position, centres (y, x) on zero, then advances it by max(h, w).
        var positions = [(Int, Int, Int)](repeating: (0, 0, 0), count: isImage.count)
        var cursor = 0, p = 0
        for (i, grid) in images.enumerated() {
            for row in cursor..<starts[i] { positions[row] = (p, p, p); p += 1 }
            let (h, w) = (grid.height, grid.width)
            for y in 0..<h {
                for x in 0..<w { positions[starts[i] + y * w + x] = (p, y - (h - h / 2), x - (w - w / 2)) }
            }
            cursor = starts[i] + h * w
            p += max(h, w)
        }
        for row in cursor..<isImage.count { positions[row] = (p, p, p); p += 1 }
        self.positions = positions
    }

    /// The RoPE table `[rows, 64 pairs, (cos, sin)]`: pairs `[frame 8 | height 28 | width 28]`,
    /// `ω_i = 1 / θ^(2i / dim)` and `angle = position · ω_i`, in fp32 as `QwenImage21Rope.rope_params` computes them.
    public func ropeTable(axes: [Int] = [16, 56, 56], theta: Float = 10000) -> [Float] {
        let omegas = axes.map { dim in (0..<(dim / 2)).map { i in 1 / powf(theta, Float(2 * i) / Float(dim)) } }
        let pairs = axes.reduce(0, +) / 2
        var out = [Float](repeating: 0, count: count * pairs * 2)
        for (row, position) in positions.enumerated() {
            let axisPositions = [position.0, position.1, position.2]
            var j = 0
            for (axis, omega) in omegas.enumerated() {
                for w in omega {
                    let angle = Float(axisPositions[axis]) * w
                    out[(row * pairs + j) * 2] = cosf(angle)
                    out[(row * pairs + j) * 2 + 1] = sinf(angle)
                    j += 1
                }
            }
        }
        return out
    }
}
