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
    public static func textEncoder(path: String, reference directory: String) throws {
        let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: directory)
            .appendingPathComponent("meta.json"))) as! [String: Any]
        let prompt = meta["prompt"] as! String, wantIds = meta["ids"] as! [Int], wantDrop = meta["drop"] as! Int
        let file = try GGUFFile(path: path)
        let context = try MetalContext()
        let encoder = try QwenTextEncoder(file: file, context: context)
        let ids = encoder.tokens(prompt: prompt)
        print("tokens: \(ids.count), \(ids == wantIds ? "identical" : "DIFFERENT") ids; drop \(encoder.dropCount)"
              + (encoder.dropCount == wantDrop ? "" : " (reference \(wantDrop))"))
        guard ids == wantIds, encoder.dropCount == wantDrop else { throw GGUFFile.Failure.unsupported("token ids differ") }

        func report(_ name: String, _ got: UnsafePointer<Float>, _ want: [Float]) {
            let r = compare(got, want)
            print(String(format: "  %-12@ rel %.2e   max |Δ| %.3g%@", name as NSString, r.relative, r.maxAbs, r.finite ? "" : "   NON-FINITE"))
        }
        _ = try encoder.forward(ids: ids) { name, pointer, count in
            guard let want = try? reference(directory, name) else { return }
            precondition(want.count == count, "\(name): \(count) floats, reference \(want.count)")
            report(name, pointer, want)
        }
        for run in 1...2 {
            let started = DispatchTime.now().uptimeNanoseconds
            let (rows, count) = try encoder.conditioning(prompt: prompt)
            let seconds = Double(DispatchTime.now().uptimeNanoseconds - started) * 1e-9
            if run == 1 { rows.withUnsafeBufferPointer { report("conditioning", $0.baseAddress!, try! reference(directory, "conditioning")) } }
            print(String(format: "encode %d (%d rows): %.3f s", run, count, seconds))
        }
    }
}

extension Checks {
    /// Text to image against `tools/reference_generate.py`, from its noise: each step's latent, the decoded image,
    /// the 8-bit pixels.
    public static func generate(encoder: String, dit: String, vae: String, reference directory: String) throws {
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
                                             noise: try reference(directory, "noise")) { i, latent in
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
