import Darwin
import Foundation

/// A safetensors file mapped read-only: an 8-byte header length, a JSON header (name → dtype, shape, byte range),
/// then the data.
public final class SafetensorsFile: @unchecked Sendable {
    public struct Tensor: Sendable {
        public let name: String, dtype: String, shape: [Int], start: Int, end: Int
        public var count: Int { shape.reduce(1, *) }
    }

    public let path: String
    public let tensors: [String: Tensor]
    public let metadata: [String: String]
    private let base: UnsafeRawPointer
    private let length: Int
    private let dataStart: Int

    public init(path: String) throws {
        self.path = path
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw GGUFFile.Failure.open(path) }
        defer { close(fd) }
        var info = stat()
        guard fstat(fd, &info) == 0, info.st_size > 8 else { throw GGUFFile.Failure.open(path) }
        length = Int(info.st_size)
        guard let p = mmap(nil, length, PROT_READ, MAP_PRIVATE, fd, 0), p != MAP_FAILED else { throw GGUFFile.Failure.open(path) }
        base = UnsafeRawPointer(p)
        let headerLength = Int(UInt64(littleEndian: base.loadUnaligned(as: UInt64.self)))
        guard headerLength > 0, 8 + headerLength <= length,
              let json = try? JSONSerialization.jsonObject(with: Data(bytes: base + 8, count: headerLength)) as? [String: Any] else {
            munmap(p, length)
            throw GGUFFile.Failure.notGGUF("\(path): not a safetensors file")
        }
        dataStart = 8 + headerLength
        var tensors: [String: Tensor] = [:]
        for (name, value) in json where name != "__metadata__" {
            guard let entry = value as? [String: Any], let dtype = entry["dtype"] as? String,
                  let shape = entry["shape"] as? [Int], let range = entry["data_offsets"] as? [Int], range.count == 2,
                  dataStart + range[1] <= length else {
                munmap(p, length)
                throw GGUFFile.Failure.truncated("\(path): \(name)")
            }
            tensors[name] = Tensor(name: name, dtype: dtype, shape: shape, start: range[0], end: range[1])
        }
        self.tensors = tensors
        metadata = (json["__metadata__"] as? [String: String]) ?? [:]
    }

    deinit { munmap(UnsafeMutableRawPointer(mutating: base), length) }

    /// A tensor as fp32 into `out` (its count of floats); BF16 widened 8 values at a time on every core (the VAE's
    /// ~260 M decoder weights, read again for each image below 32 GB of memory).
    public func copyFloats(_ name: String, into out: UnsafeMutablePointer<Float>) throws {
        guard let t = tensors[name] else { throw GGUFFile.Failure.unsupported("no tensor \(name) in \(path)") }
        guard t.dtype == "BF16" else {
            let values = try floats(name)
            out.update(from: values, count: values.count)
            return
        }
        let p = base + dataStart + t.start, count = t.count
        let chunk = 1 << 20, chunks = (count + chunk - 1) / chunk
        nonisolated(unsafe) let (source, target) = (p, UnsafeMutableRawPointer(out))
        DispatchQueue.concurrentPerform(iterations: chunks) { c in
            let start = c * chunk, end = min(count, start + chunk)
            var i = start
            while i + 8 <= end {
                let v = SIMD8<UInt32>(truncatingIfNeeded: source.loadUnaligned(fromByteOffset: 2 * i, as: SIMD8<UInt16>.self)) &<< 16
                target.storeBytes(of: v, toByteOffset: 4 * i, as: SIMD8<UInt32>.self)
                i += 8
            }
            while i < end {
                target.storeBytes(of: UInt32(source.loadUnaligned(fromByteOffset: 2 * i, as: UInt16.self)) << 16, toByteOffset: 4 * i, as: UInt32.self)
                i += 1
            }
        }
    }

    /// A tensor as fp32 (BF16, F16 or F32 in the file).
    public func floats(_ name: String) throws -> [Float] {
        guard let t = tensors[name] else { throw GGUFFile.Failure.unsupported("no tensor \(name) in \(path)") }
        let p = base + dataStart + t.start
        var out = [Float](repeating: 0, count: t.count)
        switch t.dtype {
        case "BF16":
            for i in 0..<t.count { out[i] = Float(bitPattern: UInt32(p.loadUnaligned(fromByteOffset: 2 * i, as: UInt16.self)) << 16) }
        case "F16":
            for i in 0..<t.count { out[i] = Float(Float16(bitPattern: p.loadUnaligned(fromByteOffset: 2 * i, as: UInt16.self))) }
        case "F32":
            for i in 0..<t.count { out[i] = p.loadUnaligned(fromByteOffset: 4 * i, as: Float.self) }
        default:
            throw GGUFFile.Failure.unsupported("\(name): dtype \(t.dtype)")
        }
        return out
    }
}
