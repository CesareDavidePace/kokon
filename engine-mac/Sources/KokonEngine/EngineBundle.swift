import Foundation

/// **Where the kernels' source is**: SwiftPM's resource bundle, found by its folder rather than `Bundle.module`, which
/// looks at the root of an app bundle when the engine runs from `Kokon.app/Contents/MacOS` (and stops the process when
/// nothing is there). Places, in order: next to the executable (`swift build`, `install-dev.sh`), `engine-mac/` next
/// to it (`tauri dev` copies the app's resources there), `../Resources/engine-mac/` (inside Kokon.app).
public enum EngineBundle {
    static let name = "KokonEngineMac_KokonEngine.bundle"

    static let resources: Bundle = {
        let executable = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
        let candidates = places(executable: Bundle.main.executableURL ?? executable)
        if let found = candidates.lazy.compactMap({ FileManager.default.fileExists(atPath: $0.path) ? Bundle(url: $0) : nil }).first {
            return found
        }
        return Bundle.module
    }()

    /// The bundle's possible places for an engine at `executable`.
    static func places(executable: URL) -> [URL] {
        let dir = executable.deletingLastPathComponent()
        return [dir.appendingPathComponent(name), dir.appendingPathComponent("engine-mac/" + name),
                dir.deletingLastPathComponent().appendingPathComponent("Resources/engine-mac/" + name)]
    }

    /// The `Kernels` folder of `resources`.
    public static var kernels: URL? { resources.url(forResource: "Kernels", withExtension: nil) }
}
