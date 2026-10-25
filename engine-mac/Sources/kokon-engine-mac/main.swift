import Foundation
import KokonEngine

let usage = """
kokon-engine-mac: Kokon's Apple Silicon engine (in development)

  inspect <file.gguf>                      tensor types and shapes
  check-dequant <file.gguf> [name...]      GPU unpacking against the CPU reference
  probe                                    this Mac's matrix throughput: GPU, AMX, Neural Engine, together
"""

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(1)
}

let arguments = Array(CommandLine.arguments.dropFirst())
guard let command = arguments.first else { print(usage); exit(0) }

do {
    switch command {
    case "inspect":
        guard arguments.count == 2 else { fail(usage) }
        let file = try GGUFFile(path: arguments[1])
        var types: [String: Int] = [:]
        for name in file.order {
            let t = file.tensors[name]!
            types[t.type?.name ?? "ggml type \(t.typeID)", default: 0] += 1
        }
        print("GGUF v\(file.version), \(file.order.count) tensors, data at \(file.dataStart), alignment \(file.alignment)")
        print(types.sorted { $0.key < $1.key }.map { "\($0.key) \($0.value)" }.joined(separator: ", "))
        for name in file.order where !name.contains("blocks.") || name.contains("blocks.0.") {
            let t = file.tensors[name]!
            print("  \(name)  \(t.type?.name ?? "?")  \(t.dims)")
        }
    case "check-dequant":
        guard arguments.count >= 2 else { fail(usage) }
        try Checks.dequant(path: arguments[1], names: Array(arguments.dropFirst(2)))
    case "probe":
        try Probe.run()
    default:
        fail(usage)
    }
} catch {
    fail("error: \(error)")
}
