import Foundation
import KokonEngine

let usage = """
kokon-engine-mac: Kokon's Apple Silicon engine (in development)

  inspect <file.gguf>                      tensor types and shapes
  check-dequant <file.gguf> [name...]      GPU unpacking against the CPU reference
  check-dit <dit.gguf> <reference dir>     one DiT evaluation against tools/reference_dit.py
  time-dit <dit.gguf> [grid] [text] [runs] DiT evaluations on random inputs (grid 64 = 1024²)
  check-vae <vae.safetensors> <reference dir> [grid]   the VAE decoder against tools/reference_vae.py, tiles against the whole graph
  time-vae <vae.safetensors> [grid]                    the VAE decoder's time and memory (64 = 1024²; KOKON_VAE_TILE)
  check-tokenizer <encoder.gguf> <fixture.json>       the tokenizer against tools/reference_tokenizer.py
  check-text <encoder.gguf> <reference dir>           the text encoder against tools/reference_text_encoder.py
  check-generate <encoder.gguf> <dit.gguf> <vae.safetensors> <reference dir> [text reference dir]
                                           text to image against tools/reference_generate.py, from its noise
                                           (and from the text reference's conditioning when given)
  check-textgen <encoder.gguf> <dir> [tokens]       text generation (Prompt+), greedy, against tools/reference_text_generation.py
  time-matvec <encoder.gguf>                        the decode's products on the encoder's weights: ms and GB/s
  check-steps <dit.gguf> <generate reference dir> <text reference dir>
                                           each step alone against tools/reference_generate.py (its latents in)
  check-noise <philox.bin>                 the seed's noise against sd.cpp's generator (tools/philox_reference.cpp)
  generate <encoder.gguf> <dit.gguf> <vae.safetensors> "<prompt>" [--size WxH] [--seed N] [--out file.png]
                                           text to image, Turbo's 8 steps (default 1024x1024, seed 42, out.png)
  serve --diffusion-model <dit.gguf> --vae <vae.safetensors> --llm <encoder.gguf> [--listen-ip 127.0.0.1] [--listen-port 1234]
                                           the engine as a server speaking sd-server's /sdcpp/v1 API (text to image)
  bench [--quick] [--out report.md] [--encoder P --dit P --vae P]
                                           a few minutes: this Mac's paths, speed at 1024², accuracy against the fp32 path;
                                           prints a report to paste in a GitHub issue (Kokon's model files by default)
  check-ane8                               the M3/M4 path (int8 Neural Engine program) against the fp16 one
  check-na                                 the M5 path (Metal 4 tensor operations) against MPS on one product
  probe                                    this Mac's matrix throughput: GPU, AMX, Neural Engine, together
  probe-attention                          the target's attention at 1024² on the GPU, the Neural Engine, the CPU
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
    case "check-dit":
        guard arguments.count == 3 else { fail(usage) }
        try Checks.dit(path: arguments[1], reference: arguments[2])
    case "time-dit":
        guard arguments.count >= 2 else { fail(usage) }
        let numbers = arguments.dropFirst(2).compactMap { Int($0) }
        try Checks.timeDiT(path: arguments[1], grid: numbers.first ?? 64, text: numbers.dropFirst().first ?? 32,
                           runs: numbers.dropFirst(2).first ?? 3)
    case "check-textgen":
        guard arguments.count >= 3 else { fail(usage) }
        try Checks.textGeneration(path: arguments[1], directory: arguments[2], tokens: arguments.count > 3 ? Int(arguments[3]) ?? 48 : 48)
    case "time-matvec":
        guard arguments.count == 2 else { fail(usage) }
        try Checks.timeMatvec(path: arguments[1])
    case "check-ane":
        try Checks.aneSurfaces()
    case "check-vae":
        guard arguments.count >= 3 else { fail(usage) }
        try Checks.vae(path: arguments[1], reference: arguments[2], timeGrid: arguments.count > 3 ? Int(arguments[3]) : nil)
    case "time-vae":
        guard arguments.count >= 2 else { fail(usage) }
        try Checks.timeVAE(path: arguments[1], grid: arguments.count > 2 ? Int(arguments[2]) ?? 64 : 64)
    case "check-tokenizer":
        guard arguments.count == 3 else { fail(usage) }
        try Checks.tokenizer(path: arguments[1], fixture: arguments[2])
    case "check-text":
        guard arguments.count == 3 else { fail(usage) }
        try Checks.textEncoder(path: arguments[1], reference: arguments[2])
    case "check-generate":
        guard arguments.count == 5 || arguments.count == 6 else { fail(usage) }
        try Checks.generate(encoder: arguments[1], dit: arguments[2], vae: arguments[3], reference: arguments[4],
                            text: arguments.count == 6 ? arguments[5] : nil)
    case "generate":
        guard arguments.count >= 5 else { fail(usage) }
        var (width, height, seed, out) = (1024, 1024, UInt64(42), "out.png")
        var options = arguments.dropFirst(5).makeIterator()
        while let option = options.next() {
            guard let value = options.next() else { fail(usage) }
            switch option {
            case "--size":
                let parts = value.split(separator: "x").compactMap { Int($0) }
                guard parts.count == 2 else { fail("--size takes WxH, e.g. 1024x768") }
                (width, height) = (parts[0], parts[1])
            case "--seed": seed = UInt64(value) ?? seed
            case "--out": out = value
            default: fail(usage)
            }
        }
        let size = QwenPipeline.workingSize(width: width, height: height)
        if size != (width, height) { print("size \(width)×\(height) → \(size.width)×\(size.height) (multiples of 32)") }
        let pipeline = try QwenPipeline(encoder: arguments[1], dit: arguments[2], vae: arguments[3])
        let image = try pipeline.textToImage(prompt: arguments[4], width: size.width, height: size.height, seed: seed)
        try QwenPipeline.writePNG(QwenPipeline.pixels(image, width: size.width, height: size.height),
                                  width: size.width, height: size.height, to: out)
        print("wrote \(out) (\(size.width)×\(size.height), seed \(seed))")
    case "check-steps":
        guard arguments.count == 4 else { fail(usage) }
        try Checks.steps(dit: arguments[1], reference: arguments[2], text: arguments[3])
    case "check-noise":
        guard arguments.count == 2 else { fail(usage) }
        try Checks.noise(dump: arguments[1])
    case "serve":
        var (ip, port, dit, vae, llm) = ("127.0.0.1", UInt16(1234), "", "", "")
        var options = arguments.dropFirst().makeIterator()
        while let option = options.next() {
            guard let value = options.next() else { fail("\(option) needs a value\n\n" + usage) }
            switch option {
            case "--listen-ip": ip = value
            case "--listen-port": port = UInt16(value) ?? port
            case "--diffusion-model": dit = value
            case "--vae": vae = value
            case "--llm": llm = value
            case "--llm_vision": break   // edits, later
            default: fail("unknown option \(option)\n\n" + usage)
            }
        }
        guard !dit.isEmpty, !vae.isEmpty, !llm.isEmpty else { fail("serve needs --diffusion-model, --vae and --llm\n\n" + usage) }
        setvbuf(stdout, nil, _IOLBF, 0)   // qls-core follows the log file stdout goes to
        print("kokon-engine-mac (experimental), Qwen-Image 2.1 Turbo")
        let engine = try EngineServer(paths: .init(diffusionModel: dit, vae: vae, textEncoder: llm))
        let server = try HTTPServer(host: ip, port: port) { engine.handle($0) }
        print("[INFO   ] engine-mac - listening on: http://\(ip):\(server.port)")
        server.run()
    case "bench":
        // bench [--encoder P] [--dit P] [--vae P] [--quick] [--out report.md]; Kokon's files by default.
        let models = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/KokonData/sdcpp/models/qwen21").path
        func existing(_ names: [String]) -> String { names.map { models + "/" + $0 }.first { FileManager.default.fileExists(atPath: $0) } ?? models + "/" + names[0] }
        var encoder = existing(["Qwen3VL-8B-Instruct-Q4_K_M.gguf"])
        var dit = existing(["qwen-image-2.1-turbo-Q4_K_M.gguf", "qwen-image-2.1-turbo-Q4_K.gguf"])
        var vae = existing(["texture_fix_vae_for_qwen_image_2.1_bf16.safetensors", "qwen_image_2.1_vae_bf16.safetensors"])
        var (quick, out): (Bool, String?) = (false, nil)
        var options = arguments.dropFirst().makeIterator()
        while let option = options.next() {
            switch option {
            case "--quick": quick = true
            case "--encoder": encoder = options.next() ?? encoder
            case "--dit": dit = options.next() ?? dit
            case "--vae": vae = options.next() ?? vae
            case "--out": out = options.next()
            default: fail("unknown option \(option)\n\n" + usage)
            }
        }
        for path in [encoder, dit, vae] where !FileManager.default.fileExists(atPath: path) {
            fail("\(path) not found: install Kokon's models, or pass --encoder, --dit and --vae")
        }
        try Bench.run(encoder: encoder, dit: dit, vae: vae, quick: quick, out: out)
    case "bench-run":
        guard arguments.count >= 6 else { fail(usage) }
        try Bench.child(mode: arguments[1], encoder: arguments[2], dit: arguments[3], vae: arguments[4], prefix: arguments[5],
                        quick: arguments.contains("--quick"))
    case "check-ane8":
        try Checks.aneInt8()
    case "check-na":
        try Checks.neuralAccelerators()
    case "probe":
        try Probe.run()
    case "probe-attention":
        try Probe.attention()
    default:
        fail(usage)
    }
} catch {
    fail("error: \(error)")
}
