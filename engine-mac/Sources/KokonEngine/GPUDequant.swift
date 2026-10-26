import Metal

/// GGUF weights unpacked on the GPU (`Kernels/dequant.metal`), encoded into the caller's command buffer so
/// the unpacking of the next weight queues behind the current product instead of waiting on the CPU.
public struct GPUDequant {
    public enum Output: Sendable { case f32, f16
        var bytes: Int { self == .f32 ? 4 : 2 }
        var suffix: String { self == .f32 ? "f32" : "f16" }
    }

    public let context: MetalContext
    public init(context: MetalContext) { self.context = context }

    static func kernelName(_ type: QuantType, _ output: Output) -> String {
        let t: String
        switch type {
        case .f32: t = "f32"
        case .f16: t = "f16"
        case .bf16: t = "bf16"
        case .q4_0: t = "q4_0"
        case .q8_0: t = "q8_0"
        case .q4_K: t = "q4_K"
        case .q5_K: t = "q5_K"
        case .q6_K: t = "q6_K"
        }
        return "dequant_\(t)_\(output.suffix)"
    }

    /// Encodes `tensor` unpacked into `out` at `outOffset`: `[rows, columns]`, or `[columns, rows]` when
    /// `transposed` (a Linear weight as `[in, out]`, the `[k, n]` a product reads without transposing).
    public func encode(_ tensor: GGUFTensor, of file: GGUFFile, into out: MTLBuffer, outOffset: Int = 0,
                       transposed: Bool, output: Output = .f32, commands: MTLCommandBuffer) throws {
        guard let type = tensor.type else {
            throw GGUFFile.Failure.unsupported("\(tensor.name): ggml type \(tensor.typeID)")
        }
        guard out.length - outOffset >= tensor.count * output.bytes else {
            throw MetalContext.Failure.buffer("\(tensor.name): output holds \(out.length - outOffset) bytes")
        }
        let (source, offset) = try context.buffer(for: tensor, in: file)
        let pipeline = try context.pipeline(GPUDequant.kernelName(type, output))
        let columns = tensor.columns, rows = tensor.rows
        var params = DequantParams(rows: UInt32(rows), columns: UInt32(columns),
                                   blocksPerRow: UInt32(columns / type.blockValues),
                                   rowBytes: UInt32(type.bytes(count: columns)), transposed: transposed ? 1 : 0)
        guard let encoder = commands.makeComputeCommandEncoder() else {
            throw MetalContext.Failure.kernel("no encoder for \(tensor.name)")
        }
        encoder.label = "dequant \(tensor.name)"
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(source, offset: offset, index: 0)
        encoder.setBuffer(out, offset: outOffset, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<DequantParams>.stride, index: 2)
        let threads = rows * columns / type.blockValues
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
    }
}

/// Mirrors `DequantParams` in `dequant.metal`.
struct DequantParams {
    var rows: UInt32
    var columns: UInt32
    var blocksPerRow: UInt32
    var rowBytes: UInt32
    var transposed: UInt32
}
