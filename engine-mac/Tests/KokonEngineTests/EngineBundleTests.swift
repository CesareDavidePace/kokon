import XCTest
@testable import KokonEngine

/// The kernels are found from `swift build`'s folder, and the places include Kokon.app's layout.
final class EngineBundleTests: XCTestCase {
    func testFindsTheKernels() throws {
        let kernels = try XCTUnwrap(EngineBundle.kernels)
        XCTAssertTrue(FileManager.default.fileExists(atPath: kernels.appendingPathComponent("dit.metal").path))
    }

    func testPlacesInsideTheApp() {
        let places = EngineBundle.places(executable: URL(fileURLWithPath: "/Applications/Kokon.app/Contents/MacOS/kokon-engine-mac"))
        XCTAssertEqual(places.map(\.path), [
            "/Applications/Kokon.app/Contents/MacOS/KokonEngineMac_KokonEngine.bundle",
            "/Applications/Kokon.app/Contents/MacOS/engine-mac/KokonEngineMac_KokonEngine.bundle",
            "/Applications/Kokon.app/Contents/Resources/engine-mac/KokonEngineMac_KokonEngine.bundle",
        ])
    }
}
