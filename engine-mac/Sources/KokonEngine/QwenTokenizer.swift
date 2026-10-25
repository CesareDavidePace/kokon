import Foundation

/// **Qwen's byte-level BPE tokenizer**, read from the GGUF metadata (`tokenizer.ggml.*`, written by llama.cpp from
/// the model's `tokenizer.json`), doing what the Hugging Face `tokenizers` pipeline of Qwen3-VL does:
///
///     added tokens (<|im_start|>, <|image_pad|>, …) split out first, matched longest-first, never normalized
///     each piece between them: NFC ─ the Qwen2 split regex ─ bytes as GPT-2's printable characters ─ BPE merges
///
/// Checked token for token against the processor of the official Qwen-Image 2.1 repository
/// (`tools/reference_tokenizer.py`, `check-tokenizer`).
public final class QwenTokenizer: @unchecked Sendable {
    /// `tokenizer.json`'s pre-tokenizer pattern (Split, behavior Isolated).
    static let pattern = #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#

    public let tokens: [String]
    let ids: [String: Int]
    /// "left right" → rank (the merge's index).
    let ranks: [String: Int]
    /// Added tokens (GGUF token types 3 control and 4 user-defined) by their UTF-8 bytes, longest first.
    let added: [(bytes: [UInt8], id: Int)]
    let regex: NSRegularExpression
    /// GPT-2's byte → printable character table.
    static let byteCharacters: [Character] = {
        let printable = Set(Array(33...126) + Array(161...172) + Array(174...255))
        var table = [Character](repeating: " ", count: 256)
        var next = 256
        for b in 0..<256 {
            if printable.contains(b) {
                table[b] = Character(Unicode.Scalar(UInt32(b))!)
            } else {
                table[b] = Character(Unicode.Scalar(UInt32(next))!)
                next += 1
            }
        }
        return table
    }()
    private var cache: [String: [Int]] = [:]
    private let lock = NSLock()

    public init(file: GGUFFile) throws {
        guard file.metadata["tokenizer.ggml.model"]?.string == "gpt2",
              file.metadata["tokenizer.ggml.pre"]?.string == "qwen2",
              case .array(let tokenValues)? = file.metadata["tokenizer.ggml.tokens"],
              case .array(let mergeValues)? = file.metadata["tokenizer.ggml.merges"],
              case .array(let typeValues)? = file.metadata["tokenizer.ggml.token_type"] else {
            throw GGUFFile.Failure.unsupported("\(file.path): no Qwen2 BPE tokenizer in the metadata")
        }
        tokens = tokenValues.map { $0.string ?? "" }
        var ids: [String: Int] = [:]
        ids.reserveCapacity(tokens.count)
        for (i, t) in tokens.enumerated() { ids[t] = i }
        self.ids = ids
        var ranks: [String: Int] = [:]
        ranks.reserveCapacity(mergeValues.count)
        for (i, m) in mergeValues.enumerated() { if let s = m.string { ranks[s] = i } }
        self.ranks = ranks
        let names = tokens
        added = typeValues.enumerated().compactMap { i, type in
            type.int == 3 || type.int == 4 ? (Array(names[i].utf8), i) : nil
        }.sorted { $0.bytes.count > $1.bytes.count }
        regex = try NSRegularExpression(pattern: QwenTokenizer.pattern)
    }

    /// The token id of an added token such as `<|im_start|>`.
    public func id(of token: String) -> Int? { ids[token] }

    public func encode(_ text: String) -> [Int] {
        var out: [Int] = []
        let bytes = Array(text.utf8)
        var start = 0, i = 0
        while i < bytes.count {
            if let match = added.first(where: { a in
                a.bytes.count <= bytes.count - i && a.bytes[0] == bytes[i] && bytes[i..<(i + a.bytes.count)].elementsEqual(a.bytes)
            }) {
                if start < i { encodeOrdinary(String(decoding: bytes[start..<i], as: UTF8.self), into: &out) }
                out.append(match.id)
                i += match.bytes.count
                start = i
            } else {
                i += 1
            }
        }
        if start < bytes.count { encodeOrdinary(String(decoding: bytes[start...], as: UTF8.self), into: &out) }
        return out
    }

    /// Text without added tokens: NFC, the split regex (unmatched gaps kept as pieces too), then BPE per piece.
    private func encodeOrdinary(_ text: String, into out: inout [Int]) {
        let normalized = text.precomposedStringWithCanonicalMapping as NSString
        var cursor = 0
        func piece(_ range: NSRange) {
            guard range.length > 0 else { return }
            out += bpe(normalized.substring(with: range))
        }
        for match in regex.matches(in: normalized as String, range: NSRange(location: 0, length: normalized.length)) {
            piece(NSRange(location: cursor, length: match.range.location - cursor))
            piece(match.range)
            cursor = match.range.location + match.range.length
        }
        piece(NSRange(location: cursor, length: normalized.length - cursor))
    }

    /// One piece: its bytes as characters, merged by rank (lowest first, leftmost first) until no merge applies.
    func bpe(_ piece: String) -> [Int] {
        lock.lock()
        if let hit = cache[piece] { lock.unlock(); return hit }
        lock.unlock()
        var parts = piece.utf8.map { String(QwenTokenizer.byteCharacters[Int($0)]) }
        while parts.count > 1 {
            var best = Int.max, at = -1
            for j in 0..<(parts.count - 1) {
                if let r = ranks[parts[j] + " " + parts[j + 1]], r < best { best = r; at = j }
            }
            guard at >= 0 else { break }
            parts[at] += parts[at + 1]
            parts.remove(at: at + 1)
        }
        // Every byte is a token of its own, so a part is always in the vocabulary.
        let result = parts.map { ids[$0] ?? 0 }
        lock.lock(); cache[piece] = result; lock.unlock()
        return result
    }

    static let byteValues: [Character: UInt8] = {
        var reverse: [Character: UInt8] = [:]
        for (b, c) in byteCharacters.enumerated() { reverse[c] = UInt8(b) }
        return reverse
    }()
    private lazy var addedIDs = Set(added.map(\.id))

    /// Token ids back to text (byte-level characters to bytes); for messages and tests.
    public func decode(_ ids: [Int]) -> String {
        var bytes: [UInt8] = []
        for id in ids where id >= 0 && id < tokens.count {
            bytes += self.bytes(of: id) ?? Array(tokens[id].utf8)
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    /// A token's bytes, nil for an added token (`<|im_end|>`, `<think>`, …: the ones text generation leaves out, as
    /// sd.cpp's Qwen2 special tokens).
    public func bytes(of id: Int) -> [UInt8]? {
        guard id >= 0, id < tokens.count else { return [] }
        lock.lock()
        let isAdded = addedIDs.contains(id)
        lock.unlock()
        if isAdded { return nil }
        return tokens[id].compactMap { QwenTokenizer.byteValues[$0] }
    }
}
