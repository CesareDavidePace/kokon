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

extension Checks {
    /// Reads a reference tensor written by `tools/reference_dit.py`.
    static func reference(_ directory: String, _ name: String) throws -> [Float] {
        let data = try Data(contentsOf: URL(fileURLWithPath: directory).appendingPathComponent(name + ".f32"))
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    /// ‖a − b‖ / ‖b‖ and the largest |a − b|.
    static func compare(_ a: UnsafePointer<Float>, _ b: [Float]) -> (relative: Double, maxAbs: Float, finite: Bool) {
        var diff = 0.0, norm = 0.0, worst: Float = 0, finite = true
        for i in b.indices {
            let e = a[i] - b[i]
            if !a[i].isFinite { finite = false }
            diff += Double(e) * Double(e); norm += Double(b[i]) * Double(b[i]); worst = max(worst, abs(e))
        }
        return ((diff / max(norm, 1e-30)).squareRoot(), worst, finite)
    }

    /// One DiT evaluation against the diffusers reference in `directory` (`tools/reference_dit.py`).
    public static func dit(path: String, reference directory: String) throws {
        let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: directory)
            .appendingPathComponent("meta.json"))) as! [String: Any]
        let grid = meta["grid"] as! [Int], text = meta["text"] as! Int, sigma = Float(meta["sigma"] as! Double)
        let file = try GGUFFile(path: path)
        let context = try MetalContext()
        let sequence = try QwenSequence.textToImage(text: text, target: .init(height: grid[0], width: grid[1]))
        let dit = try QwenDiT(file: file, context: context, sequence: sequence)
        print("sequence: \(sequence.count) rows (\(sequence.prefix) prefix, \(sequence.target) target), σ = \(sigma)")

        func report(_ name: String, _ got: UnsafePointer<Float>, _ want: [Float]) {
            let r = compare(got, want)
            print(String(format: "  %-12@ rel %.2e   max |Δ| %.3g%@", name as NSString, r.relative, r.maxAbs, r.finite ? "" : "   NON-FINITE"))
        }
        let rope = sequence.ropeTable()
        rope.withUnsafeBufferPointer { report("rope", $0.baseAddress!, try! reference(directory, "rope")) }
        let m = try dit.modulationRows(timesteps: [sigma, 0])
        (m.temb[0] + m.temb[1]).withUnsafeBufferPointer { report("temb", $0.baseAddress!, try! reference(directory, "temb")) }
        (m.raw[0] + m.raw[1]).withUnsafeBufferPointer { report("modulation", $0.baseAddress!, try! reference(directory, "modulation")) }

        let encoder = try reference(directory, "in_encoder"), latent = try reference(directory, "in_latent")
        _ = try dit.forward(encoder: encoder, target: latent, sigma: sigma) { name, pointer, count in
            guard let want = try? reference(directory, name) else { return }
            precondition(want.count == count, "\(name): \(count) floats, reference \(want.count)")
            report(name, pointer, want)
        }
        // Timing without the stage waits.
        let started = DispatchTime.now().uptimeNanoseconds
        _ = try dit.forward(encoder: encoder, target: latent, sigma: sigma)
        print(String(format: "evaluation: %.3f s (modulation on the CPU %.3f s)",
                     Double(DispatchTime.now().uptimeNanoseconds - started) * 1e-9, dit.timings["modulation"] ?? 0))
    }
}

extension Checks {
    /// Times DiT evaluations on random inputs at a `grid` × `grid` target (64 = 1024²) after `text` text rows.
    public static func timeDiT(path: String, grid: Int, text: Int, runs: Int) throws {
        let file = try GGUFFile(path: path)
        let context = try MetalContext()
        let sequence = try QwenSequence.textToImage(text: text, target: .init(height: grid, width: grid))
        let dit = try QwenDiT(file: file, context: context, sequence: sequence)
        var seed: UInt64 = 1
        func random(_ n: Int, _ scale: Float) -> [Float] {
            (0..<n).map { _ in
                seed = seed &* 6364136223846793005 &+ 1442695040888963407
                return (Float(seed >> 40) / Float(1 << 24) - 0.5) * scale
            }
        }
        let encoder = random(text * QwenDiT.dim, 8), latent = random(sequence.target * QwenDiT.channels, 2)
        print("sequence: \(sequence.count) rows (\(sequence.prefix) text, \(sequence.target) target)")
        for run in 0..<runs {
            let started = DispatchTime.now().uptimeNanoseconds
            let out = try dit.forward(encoder: encoder, target: latent, sigma: 0.5)
            let seconds = Double(DispatchTime.now().uptimeNanoseconds - started) * 1e-9
            print(String(format: "evaluation %d: %.2f s%@", run + 1, seconds, out.allSatisfy(\.isFinite) ? "" : "  NON-FINITE OUTPUT"))
        }
        dit.profiling = true
        _ = try dit.forward(encoder: encoder, target: latent, sigma: 0.5)
        print("profile (each operation alone, GPU seconds):")
        for (kind, seconds) in dit.timings.sorted(by: { $0.value > $1.value }) {
            print(String(format: "  %-24@ %6.2f s", kind as NSString, seconds))
        }
    }
}
