import XCTest
@testable import KokonEngine

/// `Prefetch` reads files in the background and a newer call stops an older one; a missing file is skipped.
final class PrefetchTests: XCTestCase {
    func testReadsAndStops() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("kokon-prefetch-\(UUID())")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = dir.appendingPathComponent("model.bin")
        try Data(count: 40 << 20).write(to: file)
        Prefetch.files([dir.appendingPathComponent("missing").path, file.path])
        Prefetch.files([file.path])
        Prefetch.cancel()
        Thread.sleep(forTimeInterval: 0.2)   // the threads end on their own; nothing to wait for but no crash
    }
}
