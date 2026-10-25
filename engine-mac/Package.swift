// swift-tools-version:6.0
import PackageDescription

// Kokon's own engine for Apple Silicon: Qwen-Image 2.1 Turbo on the GPU, the CPU's matrix unit (AMX) and
// the Neural Engine at the same time. It reads the same GGUF files Kokon downloads for sd.cpp and will
// answer the same /sdcpp/v1 API, so the service, the app and MCP do not change.
let package = Package(
    name: "KokonEngineMac",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "kokon-engine-mac", targets: ["kokon-engine-mac"]),
    ],
    targets: [
        .target(
            name: "KokonEngine",
            resources: [.copy("Kernels")],
            cSettings: [.define("ACCELERATE_NEW_LAPACK")],
            swiftSettings: [.define("ACCELERATE_NEW_LAPACK")],
            linkerSettings: [.linkedFramework("Accelerate"), .linkedFramework("Metal"),
                             .linkedFramework("MetalPerformanceShaders"), .linkedFramework("CoreML")]
        ),
        .executableTarget(name: "kokon-engine-mac", dependencies: ["KokonEngine"]),
        .testTarget(name: "KokonEngineTests", dependencies: ["KokonEngine"], resources: [.copy("Fixtures")]),
    ]
)
