import Foundation
import Metal

/// Developer checks on real files, printed (the unit tests use small synthetic files instead).
public enum Checks {
    /// Each tensor (or one of each type when `names` is empty) unpacked on the GPU, in both layouts and
    /// both precisions, against the CPU reference.
    public static func dequant(path: String, names: [String]) throws {
        let file = try GGUFFile(path: path)
        let context = try MetalContext()
        let gpu = GPUDequant(context: context)
        var chosen = names
        if chosen.isEmpty {
            var seen = Set<UInt32>()
            for name in file.order {
                let t = file.tensors[name]!
                if t.type != nil, t.count >= 256, seen.insert(t.typeID).inserted { chosen.append(name) }
            }
        }
        for name in chosen {
            let t = try file.tensor(name)
            var reference = [Float](repeating: 0, count: t.count)
            try reference.withUnsafeMutableBufferPointer { try Dequant.tensor(file, t, into: $0.baseAddress!) }
            for output in [GPUDequant.Output.f32, .f16] {
                for transposed in [false, true] {
                    let out = try context.scratch(t.count * output.bytes, label: "check")
                    let commands = context.queue.makeCommandBuffer()!
                    try gpu.encode(t, of: file, into: out, transposed: transposed, output: output, commands: commands)
                    let seconds = try MetalContext.finish(commands, "dequant \(name)")
                    var worst: Float = 0
                    let (rows, columns) = (t.rows, t.columns)
                    for r in 0..<rows {
                        for c in 0..<columns {
                            let i = transposed ? c * rows + r : r * columns + c
                            let got: Float = output == .f32
                                ? out.contents().assumingMemoryBound(to: Float.self)[i]
                                : Float(out.contents().assumingMemoryBound(to: Float16.self)[i])
                            let want = output == .f32 ? reference[r * columns + c] : Float(Float16(reference[r * columns + c]))
                            worst = max(worst, abs(got - want))
                        }
                    }
                    let gbps = Double(t.count * output.bytes + t.bytes) / seconds / 1e9
                    print(String(format: "%@ %@ %@ %@: max |GPU - CPU| = %g, %.2f ms (%.0f GB/s)",
                                 name, t.type!.name, output == .f32 ? "f32" : "f16",
                                 transposed ? "[k,n]" : "[n,k]", worst, seconds * 1e3, gbps))
                }
            }
        }
    }
}
