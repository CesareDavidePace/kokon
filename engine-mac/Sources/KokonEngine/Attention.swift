import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

/// Scaled dot-product attention through MPSGraph, fp32, on the layout the products write: `q` `[queries, H·D]`,
/// `k`, `v` `[keys, H·D]`, out `[queries, H·D]`. The head transposes happen inside the graph.
///
/// `causal`: the queries are the last `queries` of the keys and query `i` sees keys `0 …< keys − queries + i + 1`
/// (a text segment of Qwen-Image 2.1's block-causal mask: everything before it, plus its own triangle).
///
/// MPSGraph builds the whole `[H, queries, keys]` score matrix: 2.2 GB at 1024². A flash-attention kernel
/// replaces this later (memory, and the split by heads with the CPU).
final class GraphAttention {
    let queries: Int, keys: Int, heads: Int, headDim: Int, causal: Bool
    private let graph: MPSGraph
    private let q: MPSGraphTensor, k: MPSGraphTensor, v: MPSGraphTensor, out: MPSGraphTensor
    private let mask: (tensor: MPSGraphTensor, data: MPSGraphTensorData)?

    init(device: MTLDevice, queries: Int, keys: Int, heads: Int, headDim: Int, causal: Bool) throws {
        self.queries = queries; self.keys = keys; self.heads = heads; self.headDim = headDim; self.causal = causal
        func n(_ x: Int) -> NSNumber { NSNumber(value: x) }
        let graph = MPSGraph()
        self.graph = graph
        let q = graph.placeholder(shape: [1, n(queries), n(heads), n(headDim)], dataType: .float32, name: "q")
        let k = graph.placeholder(shape: [1, n(keys), n(heads), n(headDim)], dataType: .float32, name: "k")
        let v = graph.placeholder(shape: [1, n(keys), n(heads), n(headDim)], dataType: .float32, name: "v")
        self.q = q; self.k = k; self.v = v
        func headsFirst(_ t: MPSGraphTensor) -> MPSGraphTensor { graph.transposeTensor(t, dimension: 1, withDimension: 2, name: nil) }
        var maskTensor: MPSGraphTensor?
        if causal {
            let shape: [NSNumber] = [1, 1, n(queries), n(keys)]
            guard let buffer = device.makeBuffer(length: queries * keys * 4, options: .storageModeShared) else {
                throw MetalContext.Failure.buffer("causal mask \(queries)×\(keys)")
            }
            let m = buffer.contents().assumingMemoryBound(to: Float.self)
            let offset = keys - queries
            for i in 0..<queries {
                for j in 0..<keys { m[i * keys + j] = j <= offset + i ? 0 : -.infinity }
            }
            let t = graph.placeholder(shape: shape, dataType: .float32, name: "mask")
            mask = (t, MPSGraphTensorData(buffer, shape: shape, dataType: .float32))
            maskTensor = t
        } else {
            mask = nil
        }
        let attended = graph.scaledDotProductAttention(query: headsFirst(q), key: headsFirst(k), value: headsFirst(v),
                                                       mask: maskTensor, scale: 1 / Float(headDim).squareRoot(), name: nil)
        out = graph.transposeTensor(attended, dimension: 1, withDimension: 2, name: nil)
    }

    /// Encodes into `commands`. Each operand is `(buffer, byte offset of its first row)`: MPSNDArrays over the
    /// caller's own buffer, so Metal orders this work with the kernels around it. (Separate buffers aliasing
    /// the same memory are not ordered: attention then raced the RoPE before it and the product after it.)
    func encode(commands: MTLCommandBuffer, q qb: (MTLBuffer, Int), k kb: (MTLBuffer, Int), v vb: (MTLBuffer, Int),
                out ob: (MTLBuffer, Int)) {
        func data(_ b: (MTLBuffer, Int), _ rows: Int) -> MPSGraphTensorData {
            let shape = [1, rows, heads, headDim].map { NSNumber(value: $0) }
            let array = MPSNDArray(buffer: b.0, offset: b.1,
                                   descriptor: MPSNDArrayDescriptor(dataType: .float32, shape: shape))
            return MPSGraphTensorData(array)
        }
        var feeds = [q: data(qb, queries), k: data(kb, keys), v: data(vb, keys)]
        if let mask { feeds[mask.tensor] = mask.data }
        graph.encode(to: commands as! MPSCommandBuffer, feeds: feeds, targetOperations: nil,
                     resultsDictionary: [out: data(ob, queries)], executionDescriptor: nil)
    }
}
