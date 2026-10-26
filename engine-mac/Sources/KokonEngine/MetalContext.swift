import Foundation
import Metal

/// The GPU, its queue and the engine's kernels.
///
/// Kernels are compiled from the source bundled with the package (`Kernels/*.metal`) when the context is
/// made: `swift build` does not compile `.metal` files. A release build can ship a precompiled
/// `.metallib` instead (`xcrun metal`); the kernel names stay the same.
public final class MetalContext: @unchecked Sendable {
    public let device: MTLDevice
    public let queue: MTLCommandQueue
    public let library: MTLLibrary
    private var pipelines: [String: MTLComputePipelineState] = [:]
    private var windows: [WindowKey: (buffer: MTLBuffer, file: GGUFFile)] = [:]
    private let lock = NSLock()

    public enum Failure: Error, CustomStringConvertible {
        case noDevice, noKernelSource, kernel(String), buffer(String)
        public var description: String {
            switch self {
            case .noDevice: return "no Metal device"
            case .noKernelSource: return "the engine's kernel sources are missing from the bundle"
            case .kernel(let s): return "kernel \(s)"
            case .buffer(let s): return "buffer \(s)"
            }
        }
    }

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw Failure.noDevice
        }
        self.device = device
        self.queue = queue
        guard let directory = EngineBundle.kernels else {
            throw Failure.noKernelSource
        }
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "metal" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else { throw Failure.noKernelSource }
        // One translation unit: each file only adds kernels, includes are the standard library's.
        let source = try files.map { try String(contentsOf: $0, encoding: .utf8) }.joined(separator: "\n")
        let options = MTLCompileOptions()
        options.mathMode = .safe   // no fast-math reassociation: the kernels must match the CPU reference
        options.languageVersion = .version3_1
        library = try device.makeLibrary(source: source, options: options)
    }

    public func pipeline(_ name: String) throws -> MTLComputePipelineState {
        lock.lock(); defer { lock.unlock() }
        if let p = pipelines[name] { return p }
        guard let function = library.makeFunction(name: name) else { throw Failure.kernel("\(name) not found") }
        let p = try device.makeComputePipelineState(function: function)
        pipelines[name] = p
        return p
    }

    private struct WindowKey: Hashable { let file: UUID; let start: Int }

    /// A GPU view of `tensor`'s bytes in the mapped file, without a copy: a buffer over the page-aligned
    /// window that holds it, and the tensor's offset in that buffer. Windows are kept per file and start,
    /// and each keeps its file (hence the mapping under it) alive until `release`.
    public func buffer(for tensor: GGUFTensor, in file: GGUFFile) throws -> (buffer: MTLBuffer, offset: Int) {
        let page = Int(getpagesize())
        let start = tensor.offset / page * page
        let end = min(file.mappedLength, (tensor.offset + tensor.bytes + page - 1) / page * page)
        let key = WindowKey(file: file.id, start: start)
        lock.lock(); defer { lock.unlock() }
        if let w = windows[key], w.buffer.length >= end - start { return (w.buffer, tensor.offset - start) }
        guard let b = device.makeBuffer(bytesNoCopy: UnsafeMutableRawPointer(mutating: file.base + start),
                                        length: end - start, options: .storageModeShared, deallocator: nil) else {
            throw Failure.buffer("cannot map \(tensor.name) (\(end - start) bytes)")
        }
        b.label = tensor.name
        windows[key] = (b, file)
        return (b, tensor.offset - start)
    }

    /// Drops the GPU windows of `file`, and the context's hold on it.
    public func release(_ file: GGUFFile) {
        lock.lock(); defer { lock.unlock() }
        windows = windows.filter { $0.key.file != file.id }
    }

    /// A shared buffer of `bytes`, zeroed by the kernel.
    public func scratch(_ bytes: Int, label: String) throws -> MTLBuffer {
        guard let b = device.makeBuffer(length: max(bytes, 16), options: .storageModeShared) else {
            throw Failure.buffer("cannot allocate \(label) (\(bytes) bytes)")
        }
        b.label = label
        return b
    }

    /// Commits and waits; a failed command buffer is an error, never a silent stale output.
    @discardableResult
    public static func finish(_ commands: MTLCommandBuffer, _ what: @autoclosure () -> String) throws -> Double {
        commands.commit()
        commands.waitUntilCompleted()
        if let error = commands.error { throw Failure.kernel("\(what()): \(error.localizedDescription)") }
        return commands.gpuEndTime - commands.gpuStartTime
    }
}
