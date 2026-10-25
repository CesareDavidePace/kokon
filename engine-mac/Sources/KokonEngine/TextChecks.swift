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
