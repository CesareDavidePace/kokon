import Foundation

/// **Reads the model files back into macOS's file cache after an image**, in the background, while the user looks at
/// it. Below 32 GB the models are not kept between images and the VAE decode's peak (8.6–8.9 GB) pushes their pages
/// out: the next image read them from the SSD while the user waited. M2 Pro 16 GB, Benchmark button 2026-10-26,
/// image 2 against image 1: text encoder +2.5–3.3 s, step 1 +0.5–2.1 s, VAE +0.3–0.7 s; steps 2–8 the same.
///
/// The pages are file cache (shared with the models' read-only mappings), not wired memory: macOS takes them back
/// when another app needs them, and the worst case is a few GB read for nothing. `KOKON_PREFETCH=0` turns it off.
public enum Prefetch {
    public static var enabled: Bool { ProcessInfo.processInfo.environment["KOKON_PREFETCH"] != "0" }

    /// Bumped by every `files` call: a newer prefetch stops an older one.
    private static let generation = Counter()

    /// Reads `paths` in order (the first one the next image needs first), on a background thread.
    public static func files(_ paths: [String]) {
        let mine = generation.increment()
        Thread.detachNewThread {
            Thread.current.qualityOfService = .utility
            let chunk = 16 << 20
            let buffer = UnsafeMutableRawPointer.allocate(byteCount: chunk, alignment: 16_384)
            defer { buffer.deallocate() }
            for path in paths {
                let fd = open(path, O_RDONLY)
                guard fd >= 0 else { continue }
                defer { close(fd) }
                var offset: off_t = 0
                while generation.value == mine {
                    let n = pread(fd, buffer, chunk, offset)
                    if n <= 0 { break }
                    offset += off_t(n)
                }
            }
        }
    }

    /// Stops the prefetch running now.
    public static func cancel() { _ = generation.increment() }
}

/// A counter shared between threads.
final class Counter: @unchecked Sendable {
    private var count = 0
    private let lock = NSLock()
    var value: Int { lock.lock(); defer { lock.unlock() }; return count }
    func increment() -> Int { lock.lock(); defer { lock.unlock() }; count += 1; return count }
}
