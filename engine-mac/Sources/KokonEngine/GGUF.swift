import Darwin
import Foundation

/// A GGUF file mapped read-only. The tensor data is never copied: the engine reads it in place, and the
/// GPU sees it through `bytesNoCopy` buffers (`MetalContext.buffer(for:in:)`).
///
/// Format (GGUF v2/v3, little-endian): magic `GGUF`, version u32, tensor count u64, key/value count u64,
/// the key/values, then per tensor its name, dims (ggml order: `ne[0]` is the contiguous one), type and
/// offset, then the data from the first multiple of `general.alignment` (default 32).
public final class GGUFFile: @unchecked Sendable {
    /// Unique per opened file: an object's address is reused once it is freed, this is not.
    public let id = UUID()
    public let path: String
    public let version: UInt32
    public let metadata: [String: GGUFValue]
    public let tensors: [String: GGUFTensor]
    /// Tensor names in file order.
    public let order: [String]
    public let alignment: Int
    /// Byte offset of the data section in the file.
    public let dataStart: Int
    public let fileSize: Int

    /// The whole file, mapped; `mappedLength` is the file size rounded up to the page.
    public let base: UnsafeRawPointer
    public let mappedLength: Int

    public enum Failure: Error, CustomStringConvertible {
        case open(String), notGGUF(String), truncated(String), unsupported(String)
        public var description: String {
            switch self {
            case .open(let s): return "cannot open \(s)"
            case .notGGUF(let s): return "not a GGUF file: \(s)"
            case .truncated(let s): return "truncated GGUF: \(s)"
            case .unsupported(let s): return "unsupported GGUF content: \(s)"
            }
        }
    }

    public init(path: String) throws {
        self.path = path
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw Failure.open(path) }
        defer { close(fd) }
        var info = stat()
        guard fstat(fd, &info) == 0 else { throw Failure.open(path) }
        fileSize = Int(info.st_size)
        let page = Int(getpagesize())
        mappedLength = max(page, (fileSize + page - 1) / page * page)
        guard let mapped = mmap(nil, mappedLength, PROT_READ, MAP_PRIVATE, fd, 0), mapped != MAP_FAILED else {
            throw Failure.open(path)
        }
        base = UnsafeRawPointer(mapped)

        var reader = Reader(base: base, size: fileSize, path: path)
        var raw: [(name: String, dims: [Int], type: UInt32, offset: Int)] = []
        do {
            guard try reader.u32() == 0x4655_4747 else { throw Failure.notGGUF(path) }   // "GGUF"
            version = try reader.u32()
            guard version == 2 || version == 3 else { throw Failure.unsupported("GGUF version \(version)") }
            let tensorCount = try reader.u64(), kvCount = try reader.u64()
            guard tensorCount < 1 << 20, kvCount < 1 << 20 else { throw Failure.truncated(path) }
            var metadata: [String: GGUFValue] = [:]
            for _ in 0..<kvCount {
                let key = try reader.string()
                metadata[key] = try reader.value(type: try reader.u32())
            }
            self.metadata = metadata
            alignment = metadata["general.alignment"]?.int ?? 32
            guard alignment > 0, alignment & (alignment - 1) == 0 else {
                throw Failure.unsupported("alignment \(alignment)")
            }
            for _ in 0..<tensorCount {
                let name = try reader.string()
                let rank = Int(try reader.u32())
                guard rank <= 4 else { throw Failure.unsupported("\(name): \(rank) dims") }
                var dims: [Int] = []
                for _ in 0..<rank { dims.append(Int(try reader.u64())) }
                raw.append((name, dims, try reader.u32(), Int(try reader.u64())))
            }
            dataStart = (reader.position + alignment - 1) / alignment * alignment
        } catch {
            munmap(UnsafeMutableRawPointer(mutating: base), mappedLength)
            throw error
        }
        var tensors: [String: GGUFTensor] = [:]
        var order: [String] = []
        for t in raw {
            let tensor = GGUFTensor(name: t.name, dims: t.dims, typeID: t.type, offset: dataStart + t.offset)
            if let type = tensor.type {
                let bytes = type.bytes(count: tensor.count)
                guard tensor.dims[0] % type.blockValues == 0, tensor.offset + bytes <= fileSize else {
                    munmap(UnsafeMutableRawPointer(mutating: base), mappedLength)
                    throw Failure.truncated("\(t.name) does not fit in the file")
                }
            }
            tensors[t.name] = tensor
            order.append(t.name)
        }
        self.tensors = tensors
        self.order = order
    }

    deinit { munmap(UnsafeMutableRawPointer(mutating: base), mappedLength) }

    /// The tensor's bytes in the mapping.
    public func data(_ tensor: GGUFTensor) -> UnsafeRawPointer { base + tensor.offset }

    public func tensor(_ name: String) throws -> GGUFTensor {
        guard let t = tensors[name] else { throw Failure.unsupported("no tensor \(name)") }
        return t
    }

    /// Bounds-checked little-endian reads over the mapped header.
    private struct Reader {
        let base: UnsafeRawPointer, size: Int, path: String
        var position = 0

        mutating func take(_ count: Int) throws -> UnsafeRawPointer {
            guard count >= 0, position + count <= size else { throw Failure.truncated(path) }
            defer { position += count }
            return base + position
        }
        mutating func load<T: FixedWidthInteger>(_: T.Type) throws -> T {
            T(littleEndian: try take(MemoryLayout<T>.size).loadUnaligned(as: T.self))
        }
        mutating func u32() throws -> UInt32 { try load(UInt32.self) }
        mutating func u64() throws -> UInt64 { try load(UInt64.self) }
        mutating func string() throws -> String {
            let n = Int(try u64())
            guard n < 1 << 24 else { throw Failure.truncated(path) }
            let p = try take(n)
            return String(decoding: UnsafeRawBufferPointer(start: p, count: n), as: UTF8.self)
        }
        mutating func value(type: UInt32) throws -> GGUFValue {
            switch type {
            case 0: return .int(Int(try load(UInt8.self)))
            case 1: return .int(Int(try load(Int8.self)))
            case 2: return .int(Int(try load(UInt16.self)))
            case 3: return .int(Int(try load(Int16.self)))
            case 4: return .int(Int(try load(UInt32.self)))
            case 5: return .int(Int(try load(Int32.self)))
            case 6: return .float(Double(Float(bitPattern: try load(UInt32.self))))
            case 7: return .bool(try load(UInt8.self) != 0)
            case 8: return .string(try string())
            case 9:
                let itemType = try u32(), n = Int(try u64())
                guard n < 1 << 26 else { throw Failure.truncated(path) }
                var items: [GGUFValue] = []
                items.reserveCapacity(n)
                for _ in 0..<n { items.append(try value(type: itemType)) }
                return .array(items)
            case 10: return .int(Int(truncatingIfNeeded: try load(UInt64.self)))
            case 11: return .int(Int(try load(Int64.self)))
            case 12: return .float(Double(bitPattern: try load(UInt64.self)))
            default: throw Failure.unsupported("metadata type \(type)")
            }
        }
    }
}

public enum GGUFValue: Sendable, Equatable {
    case int(Int), float(Double), bool(Bool), string(String), array([GGUFValue])
    public var int: Int? { if case .int(let v) = self { return v }; return nil }
    public var string: String? { if case .string(let v) = self { return v }; return nil }
}

public struct GGUFTensor: Sendable, Equatable {
    public let name: String
    /// ggml order: `dims[0]` is contiguous. A Linear weight `[out, in]` is stored as `dims = [in, out]`.
    public let dims: [Int]
    public let typeID: UInt32
    /// Absolute offset in the file.
    public let offset: Int

    public var type: QuantType? { QuantType(rawValue: typeID) }
    public var count: Int { dims.reduce(1, *) }
    /// Rows of `dims[0]` values each: for a Linear weight, its output features.
    public var rows: Int { count / dims[0] }
    public var columns: Int { dims[0] }
    public var bytes: Int { type?.bytes(count: count) ?? 0 }
}
