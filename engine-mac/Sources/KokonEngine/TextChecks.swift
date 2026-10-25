import Foundation

extension Checks {
    /// The tokenizer against `tools/reference_tokenizer.py` (the official processor): every case token for token.
    public static func tokenizer(path: String, fixture: String) throws {
        let file = try GGUFFile(path: path)
        var started = DispatchTime.now().uptimeNanoseconds
        let tokenizer = try QwenTokenizer(file: file)
        print(String(format: "tokenizer: %d tokens, read in %.2f s", tokenizer.tokens.count,
                     Double(DispatchTime.now().uptimeNanoseconds - started) * 1e-9))
        let cases = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: fixture))) as! [[String: Any]]
        var failed = 0
        started = DispatchTime.now().uptimeNanoseconds
        for c in cases {
            let text = c["text"] as! String, want = c["ids"] as! [Int]
            let got = tokenizer.encode(text)
            if got != want {
                failed += 1
                let at = (0..<min(got.count, want.count)).first { got[$0] != want[$0] } ?? min(got.count, want.count)
                print("MISMATCH \(text.debugDescription.prefix(80))\n  at \(at): got \(Array(got[at...].prefix(8))), want \(Array(want[at...].prefix(8)))")
            }
        }
        print(String(format: "%d / %d cases identical (%.1f ms)", cases.count - failed, cases.count,
                     Double(DispatchTime.now().uptimeNanoseconds - started) * 1e-6))
        if failed > 0 { throw GGUFFile.Failure.unsupported("tokenizer differs from the reference") }
    }

    /// The text encoder against `tools/reference_text_encoder.py`: the token ids, each saved stage, the conditioning.
    /// A reference with images (`--image`) needs the vision tower (`mmproj`): then also the positions, the image slots
    /// and the tower's outputs on the processor's input.
    public static func textEncoder(path: String, reference directory: String, mmproj: String? = nil) throws {
        let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: directory)
            .appendingPathComponent("meta.json"))) as! [String: Any]
        let prompt = meta["prompt"] as! String, wantIds = meta["ids"] as! [Int], wantDrop = meta["drop"] as! Int
        let context = try MetalContext()
        var images: [QwenVisionTower.Output] = []
        if let sizes = meta["images"] as? [[Int]], !sizes.isEmpty {
            guard let mmproj else { throw GGUFFile.Failure.unsupported("this reference has images: pass the mmproj GGUF") }
            let tower = try QwenVisionTower(file: try GGUFFile(path: mmproj), context: context)
            for i in sizes.indices {
                let png = try QwenPipeline.decodeImage(Data(contentsOf: URL(fileURLWithPath: directory).appendingPathComponent("image\(i + 1).png")))
                let input = QwenVisionTower.pixels(rgba: png.rgba, width: png.width, height: png.height)
                images.append(try tower.encode(rgb: input.rgb, width: input.width, height: input.height))
                print(String(format: "image %d: %d×%d, %d slots, vision tower %.2f s", i + 1, input.width, input.height,
                             images[i].slots, tower.seconds))
            }
            let merged = images.flatMap(\.merged)
            merged.withUnsafeBufferPointer {
                let r = compare($0.baseAddress!, try! reference(directory, "vision_merged"))
                print(String(format: "  vision merged rel %.2e", r.relative))
            }
            tower.release()
        }
        let file = try GGUFFile(path: path)
        let encoder = try QwenTextEncoder(file: file, context: context)
        let ids = encoder.tokens(prompt: prompt, images: images)
        print("tokens: \(ids.count), \(ids == wantIds ? "identical" : "DIFFERENT") ids; drop \(encoder.dropCount)"
              + (encoder.dropCount == wantDrop ? "" : " (reference \(wantDrop))"))
        guard ids == wantIds, encoder.dropCount == wantDrop else { throw GGUFFile.Failure.unsupported("token ids differ") }
        if !images.isEmpty {
            let positions = try QwenTextEncoder.positions(ids: ids, imagePad: encoder.imagePad,
                                                          grids: images.map { ($0.height / 32, $0.width / 32) })
            let want = try reference(directory, "positions")   // [3, T]
            let same = positions.indices.allSatisfy { i in (0..<3).allSatisfy { Float(positions[i][$0]) == want[$0 * ids.count + i] } }
            print("positions: \(same ? "identical" : "DIFFERENT")")
        }

        func report(_ name: String, _ got: UnsafePointer<Float>, _ want: [Float]) {
            let r = compare(got, want)
            print(String(format: "  %-12@ rel %.2e   max |Δ| %.3g%@", name as NSString, r.relative, r.maxAbs, r.finite ? "" : "   NON-FINITE"))
        }
        _ = try encoder.forward(ids: ids, probe: { name, pointer, count in
            guard let want = try? reference(directory, name) else { return }
            precondition(want.count == count, "\(name): \(count) floats, reference \(want.count)")
            report(name, pointer, want)
        }, images: images)
        if !images.isEmpty {
            // The text encoder alone: the reference's own vision rows in the slots.
            var at = 0
            let merged = try reference(directory, "vision_merged"), deep = try (0..<3).map { try reference(directory, "vision_deepstack\($0)") }
            let given = images.map { image -> QwenVisionTower.Output in
                defer { at += image.slots * QwenTextEncoder.dim }
                let range = at..<(at + image.slots * QwenTextEncoder.dim)
                return QwenVisionTower.Output(merged: Array(merged[range]), deepstack: deep.map { Array($0[range]) },
                                              height: image.height, width: image.width)
            }
            let (rows, _, _) = try encoder.conditioning(prompt: prompt, images: given)
            rows.withUnsafeBufferPointer { report("conditioning, the reference's vision rows", $0.baseAddress!, try! reference(directory, "conditioning")) }
        }
        for run in 1...2 {
            let started = DispatchTime.now().uptimeNanoseconds
            let (rows, count, slots) = try encoder.conditioning(prompt: prompt, images: images)
            let seconds = Double(DispatchTime.now().uptimeNanoseconds - started) * 1e-9
            if run == 1, let want = meta["image_pad_mask"] as? [Int] {
                print("image slots: \(slots.map { $0 ? 1 : 0 } == want ? "identical" : "DIFFERENT") (\(slots.filter { $0 }.count))")
            }
            if run == 1 { rows.withUnsafeBufferPointer { report("conditioning", $0.baseAddress!, try! reference(directory, "conditioning")) } }
            print(String(format: "encode %d (%d rows): %.3f s", run, count, seconds))
        }
    }
}

extension Checks {
    /// Text to image against `tools/reference_generate.py`, from its noise: each step's latent, the decoded image,
    /// the 8-bit pixels.
    /// `text`: a `reference_text_encoder.py` directory whose conditioning replaces the engine's encoder, to measure
    /// the DiT and the sampler alone.
    public static func generate(encoder: String, dit: String, vae: String, reference directory: String, text: String? = nil) throws {
        let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: directory)
            .appendingPathComponent("meta.json"))) as! [String: Any]
        let prompt = meta["prompt"] as! String, grid = meta["grid"] as! [Int]
        let (h, w) = (grid[0], grid[1])
        let pipeline = try QwenPipeline(encoder: encoder, dit: dit, vae: vae)
        func report(_ name: String, _ got: UnsafePointer<Float>, _ want: [Float]) {
            let r = compare(got, want)
            print(String(format: "  %-12@ rel %.2e   max |Δ| %.3g%@", name as NSString, r.relative, r.maxAbs, r.finite ? "" : "   NON-FINITE"))
        }
        let image = try pipeline.textToImage(prompt: prompt, width: w * 16, height: h * 16, seed: 0,
                                             noise: try reference(directory, "noise"),
                                             conditioning: try text.map { try reference($0, "conditioning") }) { i, latent in
            latent.withUnsafeBufferPointer { report("latent\(i)", $0.baseAddress!, try! reference(directory, "latent\(i)")) }
        }
        image.withUnsafeBufferPointer { report("image", $0.baseAddress!, try! reference(directory, "image")) }
        let got = QwenPipeline.pixels(image, width: w * 16, height: h * 16)
        let want = [UInt8](try Data(contentsOf: URL(fileURLWithPath: directory).appendingPathComponent("pixels.u8")))
        let channels = want.count / (h * 16 * w * 16)
        var differ = 0, worst = 0
        for p in 0..<(h * 16 * w * 16) {
            for c in 0..<channels {
                let d = abs(Int(got[p * 4 + c]) - Int(want[p * channels + c]))
                if d > 0 { differ += 1; worst = max(worst, d) }
            }
        }
        print("pixels: \(differ) of \(want.count) 8-bit values differ, by at most \(worst)")
        let png = URL(fileURLWithPath: directory).appendingPathComponent("engine.png").path
        try QwenPipeline.writePNG(got, width: w * 16, height: h * 16, to: png)
        print("wrote \(png)")
    }
}

extension Checks {
    /// Each step of a `reference_generate.py` run on its own: the DiT on the reference's latent before the step (its
    /// noise for the first), with the text reference's conditioning, against the reference's velocity. A trajectory
    /// amplifies rounding from step to step; this measures one evaluation's error on real inputs.
    public static func steps(dit path: String, reference directory: String, text: String) throws {
        let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: directory)
            .appendingPathComponent("meta.json"))) as! [String: Any]
        let grid = meta["grid"] as! [Int], sigmas = (meta["sigmas"] as! [Double]).map(Float.init)
        let conditioning = try reference(text, "conditioning")
        let file = try GGUFFile(path: path)
        let sequence = try QwenSequence.textToImage(text: conditioning.count / QwenDiT.dim,
                                                    target: .init(height: grid[0], width: grid[1]))
        let dit = try QwenDiT(file: file, context: try MetalContext(), sequence: sequence)
        for i in 0..<(sigmas.count - 1) {
            let input = i == 0 ? QwenPipeline.pack(try reference(directory, "noise"), height: grid[0], width: grid[1])
                               : try reference(directory, "latent\(i - 1)")
            let velocity = try dit.forward(encoder: conditioning, target: input, sigma: sigmas[i])
            let r = velocity.withUnsafeBufferPointer { compare($0.baseAddress!, try! reference(directory, "velocity\(i)")) }
            print(String(format: "  step %d (σ %.4f): velocity rel %.2e   max |Δ| %.3g", i + 1, sigmas[i], r.relative, r.maxAbs))
        }
    }

    /// The seed's noise against sd.cpp's own generator (`tools/philox_reference.cpp`): every value, bit for bit.
    public static func noise(dump: String) throws {
        let data = try Data(contentsOf: URL(fileURLWithPath: dump))
        let want = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let seeds: [UInt64] = [0, 1, 42, 711_449_798, 4001, 0xFFFF_FFFF, 0x0123_4567_89AB_CDEF, .max]
        var offset = 0, differ = 0, worst: Float = 0
        for seed in seeds {
            let got = Noise.philox(seed: seed, count: 262_144) + Noise.philox(seed: seed, count: 1000, offset: 1)
            for (i, g) in got.enumerated() where g.bitPattern != want[offset + i].bitPattern {
                differ += 1; worst = max(worst, abs(g - want[offset + i]))
            }
            offset += got.count
        }
        print("noise: \(offset - differ) of \(offset) values bit-identical to sd.cpp's" + (differ > 0 ? ", largest |Δ| \(worst)" : ""))
        if differ > 0 || offset != want.count { throw GGUFFile.Failure.unsupported("noise differs from sd.cpp's") }
    }
}

extension Checks {
    /// Text generation (Prompt+) against `tools/reference_text_generation.py`: the engine writes greedily (its tokens to
    /// `<dir>/engine.json`, its logits beside); once the reference has run on those tokens, each step's logits are
    /// compared with transformers' (relative error, the greedy choice, the margin between the two best) and the
    /// engine's speed is reported. `system` / `prompt` from `<dir>/engine.json` when it exists.
    public static func textGeneration(path: String, directory: String, tokens: Int = 48) throws {
        let dir = URL(fileURLWithPath: directory)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let previous = (try? JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("engine.json")))) as? [String: Any]
        let system = previous?["system"] as? String
            ?? "You write prompts for an image generation model. Expand the user's request into one detailed English paragraph that describes the finished image: subject, setting, composition, lighting and style. Output only the prompt."
        let prompt = previous?["prompt"] as? String ?? "un gatto rosso che dorme su un davanzale, sera"
        let file = try GGUFFile(path: path)
        let context = try MetalContext()
        let encoder = try QwenTextEncoder(file: file, context: context)
        let generator = QwenTextGenerator(encoder: encoder)
        let vocabulary = QwenTextGenerator.vocabulary
        var logits = [Float](repeating: 0, count: tokens * vocabulary)
        let result = try generator.generate(system: system, prompt: prompt, maxTokens: tokens, sampling: .greedy, seed: 0, logits: { step, l in
            logits.withUnsafeMutableBufferPointer { ($0.baseAddress! + step * vocabulary).update(from: l, count: vocabulary) }
        })
        let steps = result.tokens.count
        let promptIDs = encoder.tokenizer.encode(QwenTextGenerator.template(system: system, prompt: prompt))
        print(String(format: "%d prompt tokens, %d generated (%@): prefill %.2f s, decode %.2f s, %.1f tokens/s",
                     result.promptTokens, result.generatedTokens, result.finish.rawValue, result.prefill, result.decode,
                     Double(steps - 1) / max(result.decode, 1e-9)))
        print("text: \(result.text)")
        if let previous, let generated = previous["generated"] as? [Int], generated != result.tokens {
            print("NOTE: not the tokens of the engine's previous run (\(generated.count) then, \(steps) now): the reference must be run again")
        }
        let meta: [String: Any] = ["system": system, "prompt": prompt, "prompt_ids": promptIDs, "generated": result.tokens, "text": result.text]
        try JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .sortedKeys]).write(to: dir.appendingPathComponent("engine.json"))
        try logits.prefix(steps * vocabulary).withUnsafeBufferPointer { Data(buffer: $0) }.write(to: dir.appendingPathComponent("engine_logits.f32"))

        guard let want = try? reference(directory, "reference_logits"), want.count == steps * vocabulary,
              (previous?["generated"] as? [Int]) == result.tokens else {
            print("no reference for these tokens yet: run tools/reference_text_generation.py on \(directory), then this again")
            return
        }
        var worst = 0.0, agree = 0, ties = 0
        for s in 0..<steps {
            let r = logits.withUnsafeBufferPointer { compare($0.baseAddress! + s * vocabulary, Array(want[(s * vocabulary)..<((s + 1) * vocabulary)])) }
            worst = max(worst, r.relative)
            let row = want[(s * vocabulary)..<((s + 1) * vocabulary)]
            let sorted = row.enumerated().sorted { $0.element > $1.element }
            let (best, second) = (sorted[0], sorted[1])
            if best.offset == result.tokens[s] { agree += 1 } else if best.element - second.element < 0.05 { ties += 1 }
            if s < 4 || s == steps - 1 || best.offset != result.tokens[s] {
                print(String(format: "  step %2d  rel %.2e   max |Δ| %.3g   engine %6d  reference %6d  (margin %.3f)", s, r.relative, r.maxAbs,
                             result.tokens[s], best.offset, best.element - second.element))
            }
        }
        print(String(format: "logits within %.2e relative at every step; the greedy choice agrees at %d of %d (%d near-ties)", worst, agree, steps, ties))
    }
}

extension Checks {
    /// The decode's matrix-vector products on the encoder's real weights, alone: milliseconds and GB/s per shape (the
    /// decode reads every weight once a token, so these rates bound its speed), then one whole token's GPU time.
    public static func timeMatvec(path: String) throws {
        let file = try GGUFFile(path: path)
        let context = try MetalContext()
        let encoder = try QwenTextEncoder(file: file, context: context)
        let generator = QwenTextGenerator(encoder: encoder)
        let cache = try KVCache(context: context, layers: QwenTextEncoder.layers, kvHeads: QwenTextEncoder.kvHeads, capacity: 1024)
        let decoder = try QwenTextGenerator.Decoder(generator: generator, cache: cache)
        let x = try context.scratch(QwenTextEncoder.hidden * 4, label: "x"), y = try context.scratch(QwenTextGenerator.vocabulary * 4, label: "y")
        for i in 0..<QwenTextEncoder.hidden { x.contents().assumingMemoryBound(to: Float.self)[i] = Float(i % 7) * 0.1 - 0.3 }
        for name in ["blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_down.weight", "output.weight"] {
            let tensor = try file.tensor(name)
            let runs = 20
            // Warm (the file's pages), then timed: one command buffer of `runs` products.
            for timed in [false, true] {
                let commands = try encoder.begin()
                for _ in 0..<(timed ? runs : 2) { try decoder.matvec(tensor, x: (x, 0), y: (y, 0), commands) }
                let seconds = try MetalContext.finish(commands, name) / Double(timed ? runs : 2)
                if timed {
                    print(String(format: "  %-24@ %@ %6d × %5d   %.3f ms   %.0f GB/s", name as NSString, (tensor.type?.name ?? "?") as NSString,
                                 tensor.rows, tensor.columns, seconds * 1e3, Double(tensor.bytes) / seconds / 1e9))
                }
            }
        }
        for run in 0..<5 {
            let t = DispatchTime.now().uptimeNanoseconds
            _ = try decoder.token(1000 + run, position: run)
            print(String(format: "token %d: %.1f ms (GPU %.1f ms)", run, Double(DispatchTime.now().uptimeNanoseconds - t) * 1e-6, decoder.lastGPU * 1e3))
        }
    }
}

extension Checks {
    /// The vision tower against `tools/reference_vision.py` (transformers fp32): its input pixels (the processor's), the
    /// patches, blocks 0, 8 and 26, the merged rows and the three deepstack outputs; then the time of a 1024² encode.
    public static func vision(path: String, reference directory: String) throws {
        let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: directory)
            .appendingPathComponent("meta.json"))) as! [String: Any]
        let (w, h) = (meta["width"] as! Int, meta["height"] as! Int)
        let png = try QwenPipeline.decodeImage(Data(contentsOf: URL(fileURLWithPath: directory).appendingPathComponent("image.png")))
        let input = QwenVisionTower.pixels(rgba: png.rgba, width: png.width, height: png.height)
        let grid = meta["grid"] as! [Int]
        print("image \(png.width)×\(png.height) → \(input.width)×\(input.height)"
              + (grid == [1, input.height / 16, input.width / 16] ? "" : " (reference grid \(grid))") + ", reference \(w)×\(h)")
        // The processor's pixel rows: 2 × 2 merge blocks, each patch's 3 × 2 frames × 16 × 16 values.
        let wantPixels = try reference(directory, "pixels")
        var pixels = [Float](), (gh, gw) = (input.height / 16, input.width / 16)
        for by in 0..<(gh / 2) { for bx in 0..<(gw / 2) { for iy in 0..<2 { for ix in 0..<2 {
            for c in 0..<3 { for _ in 0..<2 { for y in 0..<16 { for x in 0..<16 {
                pixels.append(input.rgb[(c * input.height + (by * 2 + iy) * 16 + y) * input.width + (bx * 2 + ix) * 16 + x])
            } } } }
        } } } }
        if pixels.count == wantPixels.count {
            let r = pixels.withUnsafeBufferPointer { compare($0.baseAddress!, wantPixels) }
            print(String(format: "  pixels: rel %.2e, max |Δ| %.3g", r.relative, r.maxAbs))
        } else {
            print("  pixels: \(pixels.count) values, reference \(wantPixels.count)")
        }
        let context = try MetalContext()
        let tower = try QwenVisionTower(file: try GGUFFile(path: path), context: context)
        var stages: [(String, [Float])] = []
        let out = try tower.encode(rgb: input.rgb, width: input.width, height: input.height) { name, p, count in
            stages.append((name, Array(UnsafeBufferPointer(start: p, count: count))))
        }
        var worst = 0.0
        func report(_ name: String, _ got: [Float]) throws {
            let want = try reference(directory, name)
            guard want.count == got.count else { print("  \(name): \(got.count) values, reference \(want.count)"); worst = .infinity; return }
            let r = got.withUnsafeBufferPointer { compare($0.baseAddress!, want) }
            worst = max(worst, r.relative)
            print(String(format: "  %@: rel %.2e, max |Δ| %.3g%@", name, r.relative, r.maxAbs, r.finite ? "" : " NON-FINITE"))
        }
        for (name, values) in stages { try report(name, values) }
        try report("merged", out.merged)
        for (j, d) in out.deepstack.enumerated() { try report("deepstack\(j)", d) }
        print(String(format: "worst %.2e; encode %.2f s with the probes", worst, tower.seconds))
        let random = (0..<(3 * 1024 * 1024)).map { i -> Float in
            let (y, x) = ((i / 1024) % 1024, i % 1024)
            return sinf(Float(x) * 0.013 + Float(i / (1024 * 1024))) * cosf(Float(y) * 0.021) * 0.9
        }
        for run in 1...2 {
            _ = try tower.encode(rgb: random, width: 1024, height: 1024)
            Thread.sleep(forTimeInterval: 1.5)
            let m = MemoryUsage.now()
            print(String(format: "encode %d at 1024²: %.2f s, footprint after %.2f GB (peak %.2f GB)", run, tower.seconds, m.footprint, m.footprintPeak))
        }
    }
}
