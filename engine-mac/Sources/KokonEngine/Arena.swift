import Darwin
import Metal

/// The engine's working memory: one page-aligned anonymous mapping, seen by the CPU as pointers and by the GPU
/// as one shared buffer (no copies, unified memory). Slices start on a page, and a DiT row (4 096 floats) is
/// exactly one 16 KB page, so any row range of a slice is page-aligned too: `view` wraps it as a buffer of its
/// own for APIs that take no offset (MPSGraph).
public final class Arena: @unchecked Sendable {
    public let base: UnsafeMutableRawPointer
    public let capacity: Int
    public let buffer: MTLBuffer
    private let device: MTLDevice
    private var used = 0
    private var views: [Int: MTLBuffer] = [:]
    public static let page = Int(getpagesize())

    public struct Slice: Sendable {
        public let offset: Int, bytes: Int
        public var floats: Int { bytes / 4 }
    }

    public init(device: MTLDevice, bytes: Int) throws {
        capacity = max(Arena.page, (bytes + Arena.page - 1) / Arena.page * Arena.page)
        guard let p = mmap(nil, capacity, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0), p != MAP_FAILED else {
            throw MetalContext.Failure.buffer("arena of \(capacity) bytes")
        }
        base = p
        guard let b = device.makeBuffer(bytesNoCopy: p, length: capacity, options: .storageModeShared, deallocator: nil) else {
            munmap(p, capacity)
            throw MetalContext.Failure.buffer("arena of \(capacity) bytes (over the GPU's buffer limit?)")
        }
        b.label = "arena"
        buffer = b
        self.device = device
    }

    deinit { munmap(base, capacity) }

    /// Bytes a set of slices needs, each rounded to the page.
    public static func bytes(_ sizes: [Int]) -> Int { sizes.reduce(0) { $0 + ($1 + page - 1) / page * page } }

    public func slice(floats: Int) throws -> Slice {
        let bytes = max(4, floats * 4)
        let rounded = (bytes + Arena.page - 1) / Arena.page * Arena.page
        guard used + rounded <= capacity else { throw MetalContext.Failure.buffer("arena full (\(capacity) bytes)") }
        defer { used += rounded }
        return Slice(offset: used, bytes: bytes)
    }

    public func pointer(_ s: Slice, floatOffset: Int = 0) -> UnsafeMutablePointer<Float> {
        (base + s.offset).assumingMemoryBound(to: Float.self) + floatOffset
    }

    /// A buffer starting at `s.offset + byteOffset` (page-aligned), to the end of the slice.
    public func view(_ s: Slice, byteOffset: Int = 0) throws -> MTLBuffer {
        let start = s.offset + byteOffset
        guard start % Arena.page == 0 else { throw MetalContext.Failure.buffer("view at \(start) is not page-aligned") }
        let length = (s.offset + s.bytes - start + Arena.page - 1) / Arena.page * Arena.page
        if let v = views[start], v.length >= length { return v }
        guard let v = device.makeBuffer(bytesNoCopy: base + start, length: length, options: .storageModeShared,
                                        deallocator: nil) else {
            throw MetalContext.Failure.buffer("view at \(start)")
        }
        views[start] = v
        return v
    }
}
