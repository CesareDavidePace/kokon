import CoreML
import Metal
import CoreVideo
import Foundation

/// **A matrix product on the Neural Engine**: `Y[rows, n] = X[rows, k] · W[k, n]`, all fp16.
///
/// Core ML is the only public door to the Neural Engine. Instead of shipping compiled models, the engine
/// writes one tiny program per shape at run time (`model.mil` + `coremldata.bin`, the two files a compiled
/// `.mlmodelc` needs) whose **weight is an input**: one program serves every weight of its shape, so no
/// model file is converted or downloaded. The operands live in IOSurface-backed buffers bound once, which
/// Core ML reads and writes in place.
///
/// Technique and the `coremldata.bin` layout from Siliconed (github.com/gwenn-ha-dev/Siliconed,
/// Apache-2.0, `NeuralEngine.swift`); written again here for Kokon.
///
/// What the caller must know (measured by Siliconed on an M1 Pro, to re-measure here with `probe`):
///   - fp16 flushes values under 6e-5 and overflows over 65 504: the caller scales weight columns and
///     activation rows by powers of two before, and undoes it in fp32 after;
///   - past K = 4 096 or a 48 MB weight the rate falls several times: cut K and sum the parts in fp32;
///   - rows in units of 256.
public final class NeuralEngineProduct: @unchecked Sendable {
    public let rows: Int, k: Int, n: Int
    private let model: MLModel
    private let features: MLFeatureProvider
    private let options: MLPredictionOptions
    public let x: CVPixelBuffer, w: CVPixelBuffer, y: CVPixelBuffer
    private let directory: URL

    public enum Failure: Error, CustomStringConvertible {
        case surface(String), program(String)
        public var description: String {
            switch self {
            case .surface(let s): return "Neural Engine surface: \(s)"
            case .program(let s): return "Neural Engine program: \(s)"
            }
        }
    }

    public init(rows: Int, k: Int, n: Int) throws {
        self.rows = rows; self.k = k; self.n = n
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("kokon-ane-\(ProcessInfo.processInfo.processIdentifier)-\(rows)x\(k)x\(n)-\(UUID().uuidString)")
            .appendingPathExtension("mlmodelc")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try NeuralEngineProduct.program(rows: rows, k: k, n: n)
            .write(to: directory.appendingPathComponent("model.mil"), atomically: true, encoding: .utf8)
        try NeuralEngineProduct.description(rows: rows, k: k, n: n)
            .write(to: directory.appendingPathComponent("coremldata.bin"))
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        do {
            model = try MLModel(contentsOf: directory, configuration: configuration)
        } catch {
            throw Failure.program("\(rows)×\(k)×\(n): \(error.localizedDescription)")
        }
        x = try NeuralEngineProduct.surface(rows: rows, columns: k)
        w = try NeuralEngineProduct.surface(rows: k, columns: n)
        y = try NeuralEngineProduct.surface(rows: rows, columns: n)
        func shape(_ dims: Int...) -> [NSNumber] { dims.map { NSNumber(value: $0) } }
        options = MLPredictionOptions()
        options.outputBackings = ["y": MLMultiArray(pixelBuffer: y, shape: shape(rows, n))]
        features = try MLDictionaryFeatureProvider(dictionary: [
            "x": MLFeatureValue(multiArray: MLMultiArray(pixelBuffer: x, shape: shape(rows, k))),
            "w": MLFeatureValue(multiArray: MLMultiArray(pixelBuffer: w, shape: shape(k, n))),
        ])
    }

    deinit { try? FileManager.default.removeItem(at: directory) }

    /// Runs the product on what the surfaces hold.
    public func run() throws {
        _ = try model.prediction(from: features, options: options)
    }

    /// Fills a surface through `body`, with its base address as fp16, locked for writing.
    public func write(_ surface: CVPixelBuffer, _ body: (UnsafeMutablePointer<Float16>) -> Void) {
        CVPixelBufferLockBaseAddress(surface, [])
        body(CVPixelBufferGetBaseAddress(surface)!.assumingMemoryBound(to: Float16.self))
        CVPixelBufferUnlockBaseAddress(surface, [])
    }

    /// Reads the output surface through `body`.
    public func read(_ body: (UnsafePointer<Float16>) -> Void) {
        CVPixelBufferLockBaseAddress(y, .readOnly)
        body(CVPixelBufferGetBaseAddress(y)!.assumingMemoryBound(to: Float16.self))
        CVPixelBufferUnlockBaseAddress(y, .readOnly)
    }

    /// A dense fp16 surface `rows × columns` (no row padding: the program reads it dense).
    static func surface(rows: Int, columns: Int) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let attributes = [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary
        guard CVPixelBufferCreate(kCFAllocatorDefault, columns, rows, kCVPixelFormatType_OneComponent16Half,
                                  attributes, &buffer) == kCVReturnSuccess, let buffer,
              CVPixelBufferGetBytesPerRow(buffer) == columns * 2 else {
            throw Failure.surface("\(rows)×\(columns) is not dense")
        }
        return buffer
    }

    /// The program: MIL text as coremltools writes it for `matmul(x, w)`.
    static func program(rows: Int, k: Int, n: Int) -> String {
        """
        program(1.3)
        [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3600.16.1"}, {"coremlc-version", "3600.25.2"}})]
        {
            func main<ios18>(tensor<fp16, [\(k), \(n)]> w, tensor<fp16, [\(rows), \(k)]> x) {
                    bool tx = const()[name = string("tx"), val = bool(false)];
                    bool ty = const()[name = string("ty"), val = bool(false)];
                    tensor<fp16, [\(rows), \(n)]> y = matmul(transpose_x = tx, transpose_y = ty, x = x, y = w)[name = string("y")];
                } -> (y);
        }
        """
    }

    /// `coremldata.bin`: a fixed header with the description's length (little-endian u64 at 0x4b), the
    /// model description as protobuf (inputs `x`, `w`, output `y`, fp16 multi-arrays), a fixed trailer.
    static func description(rows: Int, k: Int, n: Int) -> Data {
        func varint(_ value: Int) -> [UInt8] {
            var v = UInt64(value), out: [UInt8] = []
            repeat {
                let low = UInt8(v & 0x7f)
                v >>= 7
                out.append(v == 0 ? low : low | 0x80)
            } while v != 0
            return out
        }
        func field(_ tag: UInt8, _ body: [UInt8]) -> [UInt8] { [tag] + varint(body.count) + body }
        // A feature: name, then a multi-array type of `dims` in FLOAT16 (ArrayDataType 65 552).
        func feature(_ tag: UInt8, _ name: String, _ dims: [Int]) -> [UInt8] {
            let shape = field(0x0a, dims.flatMap(varint))
            let array = field(0x2a, shape + [0x10] + varint(65_552))
            return field(tag, field(0x0a, Array(name.utf8)) + field(0x1a, array))
        }
        let payload = feature(0x0a, "x", [rows, k]) + feature(0x0a, "w", [k, n]) + feature(0x52, "y", [rows, n])
            + [0xa2, 0x06, 0x00]
        var header = [UInt8](repeating: 0, count: 0x53)
        header[0] = 0xf6; header[1] = 0x01; header[4] = 0x09; header[0x1c] = 0x07
        for (i, byte) in "generic".utf8.enumerated() { header[0x24 + i] = byte }
        header[0x2b] = 0x09
        withUnsafeBytes(of: UInt64(payload.count).littleEndian) { for (i, b) in $0.enumerated() { header[0x4b + i] = b } }
        var trailer = [UInt8](repeating: 0, count: 16)
        trailer[0] = 0xf6; trailer[1] = 0x01
        return Data(header + payload + trailer)
    }
}

extension NeuralEngineProduct {
    /// A GPU buffer over a surface's memory, no copy: the GPU writes the Neural Engine's operands in place.
    /// The surface stays locked for the product's life (unified memory: the lock only pins the address).
    public func gpuBuffer(_ surface: CVPixelBuffer, device: MTLDevice) throws -> MTLBuffer {
        CVPixelBufferLockBaseAddress(surface, [])
        guard let io = CVPixelBufferGetIOSurface(surface)?.takeUnretainedValue(),
              let base = CVPixelBufferGetBaseAddress(surface) else { throw Failure.surface("no IOSurface") }
        // The surface reports its exact size (data + 64 bytes); its memory is whole pages.
        let page = Int(getpagesize())
        let length = (IOSurfaceGetAllocSize(io) + page - 1) / page * page
        guard Int(bitPattern: base) % page == 0,
              let buffer = device.makeBuffer(bytesNoCopy: base, length: length, options: .storageModeShared, deallocator: nil) else {
            throw Failure.surface("surface memory at \(base) is not page-aligned (\(length) bytes)")
        }
        return buffer
    }
}
