import Foundation

/// **A minimal HTTP/1.1 server on a loopback port**, enough for qls-core's client (reqwest): one request per
/// connection (`Connection: close`), a `Content-Length` body, JSON in and out, or a body written as it is made
/// (`Response.stream`, chunked: `text_gen`'s Server-Sent Events). POSIX sockets, a thread per connection: the engine
/// serves one local client, and the slow part of every request is the GPU, not the socket.
public final class HTTPServer: @unchecked Sendable {
    public struct Request: Sendable {
        public let method: String
        public let path: String
        public let body: Data
    }

    public struct Response: Sendable {
        public var status: Int
        public var body: Data
        public var contentType = "application/json"
        /// A body written piece by piece (chunked transfer): `stream` gets a writer, which returns false once the client
        /// has gone (the connection closed).
        public var stream: (@Sendable (_ write: (Data) -> Bool) -> Void)?

        public init(status: Int, body: Data, contentType: String = "application/json") {
            self.status = status
            self.body = body
            self.contentType = contentType
        }

        public static func streaming(contentType: String, _ stream: @escaping @Sendable (_ write: (Data) -> Bool) -> Void) -> Response {
            var r = Response(status: 200, body: Data(), contentType: contentType)
            r.stream = stream
            return r
        }

        public static func json(_ status: Int, _ object: Any) -> Response {
            let data = (try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])) ?? Data("{}".utf8)
            return Response(status: status, body: data)
        }

        /// sd-server's error body: `{"error": message}`.
        public static func error(_ status: Int, _ message: String) -> Response { json(status, ["error": message]) }
    }

    public enum Failure: Error, CustomStringConvertible {
        case socket(String)
        public var description: String {
            switch self { case .socket(let s): return "cannot listen: \(s)" }
        }
    }

    /// Requests larger than this are refused (reference images later: a few base64 PNGs).
    static let maxBody = 256 << 20
    public let port: UInt16
    private let socket: Int32
    private let handler: @Sendable (Request) -> Response

    /// Binds `host:port` (port 0: any free one, see `port`).
    public init(host: String = "127.0.0.1", port: UInt16, handler: @escaping @Sendable (Request) -> Response) throws {
        self.handler = handler
        let fd = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { throw Failure.socket("socket: \(String(cString: strerror(errno)))") }
        var yes: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))
        var address = sockaddr_in()
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = port.bigEndian
        guard inet_pton(AF_INET, host, &address.sin_addr) == 1 else {
            close(fd)
            throw Failure.socket("bad address \(host)")
        }
        let bound = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
        }
        guard bound == 0, listen(fd, 16) == 0 else {
            let message = String(cString: strerror(errno))
            close(fd)
            throw Failure.socket("\(host):\(port): \(message)")
        }
        var actual = sockaddr_in(), length = socklen_t(MemoryLayout<sockaddr_in>.size)
        _ = withUnsafeMutablePointer(to: &actual) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(fd, $0, &length) }
        }
        self.port = UInt16(bigEndian: actual.sin_port)
        socket = fd
    }

    /// Accepts connections until the process ends.
    public func run() -> Never {
        while true {
            let client = accept(socket, nil, nil)
            guard client >= 0 else { continue }
            var noSigpipe: Int32 = 1
            setsockopt(client, SOL_SOCKET, SO_NOSIGPIPE, &noSigpipe, socklen_t(MemoryLayout<Int32>.size))
            let thread = Thread { [self] in serve(client) }
            thread.name = "kokon.http"
            thread.start()
        }
    }

    private func serve(_ fd: Int32) {
        defer { close(fd) }
        let response: Response
        switch HTTPServer.read(fd) {
        case .success(let request): response = handler(request)
        case .failure(let message): response = .error(400, message)
        }
        HTTPServer.write(fd, response)
    }

    enum ReadResult { case success(Request), failure(String) }

    static func read(_ fd: Int32) -> ReadResult {
        var data = Data()
        var buffer = [UInt8](repeating: 0, count: 64 * 1024)
        let separator = Data("\r\n\r\n".utf8)
        var headerEnd: Range<Data.Index>?
        while headerEnd == nil {
            let n = recv(fd, &buffer, buffer.count, 0)
            guard n > 0 else { return .failure("connection closed before the request ended") }
            data.append(buffer, count: n)
            headerEnd = data.range(of: separator)
            if headerEnd == nil, data.count > 64 * 1024 { return .failure("request headers too long") }
        }
        let head = String(decoding: data[..<headerEnd!.lowerBound], as: UTF8.self)
        let lines = head.components(separatedBy: "\r\n")
        let parts = lines[0].split(separator: " ")
        guard parts.count >= 2 else { return .failure("bad request line") }
        var length = 0
        for line in lines.dropFirst() {
            let pair = line.split(separator: ":", maxSplits: 1)
            if pair.count == 2, pair[0].trimmingCharacters(in: .whitespaces).lowercased() == "content-length" {
                length = Int(pair[1].trimmingCharacters(in: .whitespaces)) ?? 0
            }
        }
        guard length <= maxBody else { return .failure("request body too large") }
        var body = Data(data[headerEnd!.upperBound...])
        while body.count < length {
            let n = recv(fd, &buffer, min(buffer.count, length - body.count), 0)
            guard n > 0 else { return .failure("connection closed inside the body") }
            body.append(buffer, count: n)
        }
        // Paths without the query string.
        let path = String(parts[1].split(separator: "?", maxSplits: 1).first ?? "")
        return .success(Request(method: String(parts[0]), path: path, body: body))
    }

    static func write(_ fd: Int32, _ response: Response) {
        let reason = [200: "OK", 202: "Accepted", 400: "Bad Request", 404: "Not Found", 409: "Conflict",
                      429: "Too Many Requests", 500: "Internal Server Error"][response.status] ?? "Status"
        if let stream = response.stream {
            guard send(fd, Data("HTTP/1.1 \(response.status) \(reason)\r\nContent-Type: \(response.contentType)\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n".utf8)) else { return }
            var open = true
            stream { piece in
                guard open, !piece.isEmpty else { return open }
                var chunk = Data((String(piece.count, radix: 16) + "\r\n").utf8)
                chunk.append(piece)
                chunk.append(contentsOf: "\r\n".utf8)
                open = send(fd, chunk)
                return open
            }
            if open { _ = send(fd, Data("0\r\n\r\n".utf8)) }
            return
        }
        var out = Data("HTTP/1.1 \(response.status) \(reason)\r\nContent-Type: \(response.contentType)\r\nContent-Length: \(response.body.count)\r\nConnection: close\r\n\r\n".utf8)
        out.append(response.body)
        _ = send(fd, out)
    }

    /// All of `data`; false when the connection is gone.
    static func send(_ fd: Int32, _ data: Data) -> Bool {
        data.withUnsafeBytes { raw in
            var sent = 0
            while sent < raw.count {
                let n = Darwin.send(fd, raw.baseAddress! + sent, raw.count - sent, 0)
                if n <= 0 { return false }
                sent += n
            }
            return true
        }
    }
}
