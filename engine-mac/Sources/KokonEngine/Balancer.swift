import Foundation

/// **The balancing loop.** After each evaluation the timeline gives each unit's rate on each kind of operation,
/// measured beside the others (the AMX runs at about half its solo rate there), and the next evaluation's shares are
/// set so the units finish together:
///
///   - products: the Neural Engine's rows are one count for every product (its programs are compiled per count, in
///     steps of 256), chosen to minimize the predicted time of all products; the AMX's rows per kind of product (steps
///     of 64) split the rest with the GPU in proportion to their rates;
///   - the target's attention: every (Neural Engine in groups of 4, CPU, GPU) cut of the 32 heads, the one whose
///     slowest unit is fastest.
///
/// A new cut is taken only when it is predicted 1.5 % faster, so the shares settle instead of flickering. Rates are
/// smoothed over evaluations and start from the M2 Pro's, so a unit without a share keeps an estimate. Moving a
/// share moves which rows go through the Neural Engine's fp16: images stay within its accuracy, not bit for bit.
struct Balancer {
    /// Rows (heads, for attention) per millisecond.
    struct Rates { var gpu: Double, amx: Double, ane: Double }
    var rates: [String: Rates]
    /// The AMX's measured rate at each row count tried, per kind: it is not linear in rows (MLP out on the M2 Pro:
    /// 320 rows 31 ms, 384 rows 43 ms), so a measured count beats the extrapolated rate.
    var amxAt: [String: [Int: Double]] = [:]
    let aneAttention: Bool
    static let smoothing = 0.5, margin = 0.985

    /// Seeds from `time-dit` at 1024² (1856 / 256 / 2048 rows; attention 19 GPU / 13 CPU heads) and `probe-attention`
    /// (the Neural Engine's heads), M2 Pro, 2026-10-25.
    init(aneAttention: Bool) {
        self.aneAttention = aneAttention
        rates = ["4096×4096": Rates(gpu: 149, amx: 28, ane: 172), "4096×12288": Rates(gpu: 48, amx: 9.8, ane: 58),
                 "12288×4096": Rates(gpu: 43.5, amx: 8.8, ane: 57.5), "attention": Rates(gpu: 0.224, amx: 0.142, ane: 0.195)]
    }

    /// Folds an evaluation's timeline into the rates: per kind, each unit's rows over its seconds.
    mutating func observe(_ entries: [Timeline.Entry]) {
        var sums: [String: (rows: Rates, ms: Rates)] = [:]
        for e in entries {
            var s = sums[e.kind] ?? (Rates(gpu: 0, amx: 0, ane: 0), Rates(gpu: 0, amx: 0, ane: 0))
            s.rows.gpu += Double(e.shares.gpu); s.ms.gpu += (e.gpu - e.released) * 1e3
            s.rows.amx += Double(e.shares.amx); s.ms.amx += e.amx.seconds * 1e3
            s.rows.ane += Double(e.shares.ane); s.ms.ane += e.ane.seconds * 1e3
            sums[e.kind] = s
        }
        for (kind, s) in sums {
            var r = rates[kind] ?? Rates(gpu: s.rows.gpu / max(s.ms.gpu, 1e-3), amx: 0, ane: 0)
            func fold(_ old: Double, _ rows: Double, _ ms: Double) -> Double {
                rows > 0 && ms > 0 ? Balancer.smoothing * old + (1 - Balancer.smoothing) * rows / ms : old
            }
            r.gpu = fold(r.gpu, s.rows.gpu, s.ms.gpu)
            r.amx = fold(r.amx, s.rows.amx, s.ms.amx)
            r.ane = fold(r.ane, s.rows.ane, s.ms.ane)
            rates[kind] = r
            if kind != "attention", s.rows.amx > 0, s.ms.amx > 0, let first = entries.first(where: { $0.kind == kind }) {
                amxAt[kind, default: [:]][first.shares.amx] = s.rows.amx / s.ms.amx
            }
        }
    }

    /// Predicted milliseconds of `counts` products of each kind with `ane` rows of `rows` on the Neural Engine and the
    /// rest cut ideally between the GPU and (when `amx`) the AMX.
    func productCost(rows: Int, ane: Int, amx: Bool, counts: [String: Int]) -> Double {
        counts.reduce(0) { sum, kc in
            guard let r = rates[kc.key] else { return sum }
            let rest = Double(rows - ane) / (r.gpu + (amx ? r.amx : 0))
            return sum + Double(kc.value) * max(ane > 0 ? Double(ane) / r.ane : 0, rest)
        }
    }

    /// Predicted milliseconds of one cut attention.
    func attentionCost(_ h: HeadSplit) -> Double {
        guard let r = rates["attention"] else { return .infinity }
        return max(Double(h.gpu) / r.gpu, h.ane > 0 ? Double(h.ane) / r.ane : 0, h.cpu > 0 ? Double(h.cpu) / r.amx : 0)
    }
}

extension Coexecutor {
    /// Reads the last evaluation's timeline and sets the next one's shares. Between evaluations only.
    func rebalance() throws {
        lastTimeline = timeline.drain()
        balancer.observe(lastTimeline)

        var counts: [String: Int] = [:]
        for e in lastTimeline where e.kind != "attention" { counts[e.kind, default: 0] += 1 }
        if !counts.isEmpty {
            if aneEnabled {
                func cost(_ e: Int) -> Double { balancer.productCost(rows: rows, ane: e, amx: amxEnabled, counts: counts) }
                let best = stride(from: 1024, through: rows - 512, by: 256).min { cost($0) < cost($1) } ?? aneRows
                if cost(best) < cost(aneRows) * Balancer.margin { try setAneRows(best) }
            }
            for kind in counts.keys {
                guard amxEnabled, let r = balancer.rates[kind] else { continue }
                // Of the two multiples of 64 around the ideal cut, the one whose slower unit is faster.
                let rest = rows - aneRows
                let ideal = Double(rest) * r.amx / (r.gpu + r.amx)
                let low = Int(ideal / 64) * 64
                func time(_ a: Int) -> Double {
                    max(a > 0 ? Double(a) / (balancer.amxAt[kind]?[a] ?? r.amx) : 0, Double(rest - a) / r.gpu)
                }
                amxRows[kind] = [low - 64, low, low + 64].filter { $0 >= 0 }.min { time($0) < time($1) }!
            }
        }

        if lastTimeline.contains(where: { $0.kind == "attention" }), HeadSplit.fromEnvironment() == nil {
            // The GPU keeps at least one head (its text segments run there anyway).
            let maxAne = aneEnabled && balancer.aneAttention ? QwenDiT.heads - Coexecutor.aneHeadGroup : 0
            var best = heads
            for ane in stride(from: 0, through: maxAne, by: Coexecutor.aneHeadGroup) {
                for cpu in 0...(amxEnabled ? QwenDiT.heads - 1 - ane : 0) {
                    let h = HeadSplit(gpu: QwenDiT.heads - ane - cpu, ane: ane, cpu: cpu)
                    if balancer.attentionCost(h) < balancer.attentionCost(best) { best = h }
                }
            }
            if balancer.attentionCost(best) < balancer.attentionCost(heads) * Balancer.margin {
                heads = best
                try prepareAttention(queries: attentionShape.queries, keys: attentionShape.keys)
            }
        }
    }

    /// The shares the next evaluation takes.
    var shares: String {
        let amx = amxRows.sorted { $0.key < $1.key }.map { "\($0.key) \($0.value)" }.joined(separator: ", ")
        return "products ANE \(aneRows) rows, AMX \(amx.isEmpty ? "default" : amx); attention \(heads)"
    }
}
