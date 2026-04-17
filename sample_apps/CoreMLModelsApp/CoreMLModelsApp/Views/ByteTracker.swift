import Foundation
import CoreGraphics

// ByteTrack multi-object tracker for the hub-app's BoundingBoxOverlay.
//
// Same algorithm as YOLO26Demo's Tracker.swift: a per-track 8D
// constant-velocity Kalman filter plus ByteTrack's two-stage IoU
// association (high-conf detections matched first, then low-conf
// detections are used to rescue tracks about to be lost). No
// appearance / ReID network, so it layers on top of any detector
// output without extra CoreML work.
//
// Reference: Zhang et al., "ByteTrack: Multi-Object Tracking by
// Associating Every Detection Box", ECCV 2022 (arxiv 2110.06864).

fileprivate typealias Det = BoundingBoxOverlay.DetectionBox

final class ByteTracker {

    struct Config {
        var trackHighThresh: Float = 0.5
        var trackLowThresh:  Float = 0.1
        var newTrackThresh:  Float = 0.6
        var trackBuffer:     Int   = 30
        var iouThreshFirst:       Float = 0.2
        var iouThreshSecond:      Float = 0.5
        var iouThreshUnconfirmed: Float = 0.3
        var perClass: Bool = true
        var trailMaxLen: Int = 60
    }

    private let config: Config
    private var trackedTracks: [STrack] = []
    private var lostTracks:    [STrack] = []
    private var nextId: Int = 1
    private var frameId: Int = 0

    init(config: Config = Config()) { self.config = config }

    func reset() {
        trackedTracks.removeAll()
        lostTracks.removeAll()
        nextId = 1
        frameId = 0
    }

    /// Run one tracking step on the current frame's detections. Returns
    /// detections with `trackId` and `trail` populated for confirmed
    /// tracks only.
    func update(detections dets: [Det]) -> [Det] {
        frameId += 1

        let high = dets.filter { $0.confidence >= config.trackHighThresh }
        let low  = dets.filter { $0.confidence >= config.trackLowThresh &&
                                 $0.confidence <  config.trackHighThresh }

        var unconfirmed: [STrack] = []
        var activated:   [STrack] = []
        for t in trackedTracks {
            if t.isActivated { activated.append(t) } else { unconfirmed.append(t) }
        }

        let pool = activated + lostTracks
        for t in pool { t.predict() }

        // Stage 1: activated + lost vs high-conf
        let (m1, uT1, uD1) = associate(tracks: pool, dets: high,
                                       iouThresh: config.iouThreshFirst)
        for (ti, di) in m1 {
            let t = pool[ti], d = high[di]
            if t.state == .tracked { t.update(det: d, frameId: frameId) }
            else                   { t.reActivate(det: d, frameId: frameId) }
        }

        // Stage 2: rescue un-matched still-tracked tracks with low-conf
        let rescuePool: [STrack] = uT1.compactMap { idx in
            let t = pool[idx]
            return t.state == .tracked ? t : nil
        }
        let (m2, uT2, _) = associate(tracks: rescuePool, dets: low,
                                     iouThresh: config.iouThreshSecond)
        for (ti, di) in m2 { rescuePool[ti].update(det: low[di], frameId: frameId) }
        for ti in uT2 {
            let t = rescuePool[ti]
            if t.state != .lost { t.markLost() }
        }

        // Stage 3: unconfirmed tracks vs leftover high-conf
        let leftoverHigh = uD1.map { high[$0] }
        let (m3, uU3, uD3) = associate(tracks: unconfirmed, dets: leftoverHigh,
                                       iouThresh: config.iouThreshUnconfirmed)
        for (ti, di) in m3 { unconfirmed[ti].update(det: leftoverHigh[di], frameId: frameId) }
        for ti in uU3 { unconfirmed[ti].markRemoved() }

        // Stage 4: spawn new tentative tracks from remaining strong dets
        for di in uD3 {
            let d = leftoverHigh[di]
            guard d.confidence >= config.newTrackThresh else { continue }
            let t = STrack(det: d, id: nextId, frameId: frameId,
                           firstFrame: frameId == 1,
                           trailMaxLen: config.trailMaxLen)
            nextId += 1
            trackedTracks.append(t)
        }

        // Re-partition and age out lost tracks beyond buffer
        var newTracked: [STrack] = []
        var newLost:    [STrack] = []
        var seen = Set<ObjectIdentifier>()
        for t in trackedTracks + lostTracks {
            let oid = ObjectIdentifier(t)
            if seen.contains(oid) { continue }
            seen.insert(oid)
            switch t.state {
            case .tracked: newTracked.append(t)
            case .lost:
                if frameId - t.frameId <= config.trackBuffer { newLost.append(t) }
            case .removed: break
            }
        }
        trackedTracks = newTracked
        lostTracks    = newLost

        return trackedTracks.filter { $0.isActivated }.map { $0.asDetection() }
    }

    private func associate(tracks: [STrack], dets: [Det], iouThresh: Float)
        -> (matches: [(Int, Int)], unmT: [Int], unmD: [Int])
    {
        let T = tracks.count, D = dets.count
        if T == 0 { return ([], [], Array(0..<D)) }
        if D == 0 { return ([], Array(0..<T), []) }

        var iouM: [[Float]] = Array(repeating: Array(repeating: 0, count: D), count: T)
        for i in 0..<T {
            let r = tracks[i].predictedRect()
            let cls = tracks[i].classIndex
            for j in 0..<D {
                let d = dets[j]
                if config.perClass && cls != d.classIndex { continue }
                iouM[i][j] = ByteTracker.iou(r, d.rect)
            }
        }

        var matches: [(Int, Int)] = []
        var usedT = Set<Int>(), usedD = Set<Int>()
        while true {
            var bestScore: Float = iouThresh
            var bi = -1, bj = -1
            for i in 0..<T where !usedT.contains(i) {
                for j in 0..<D where !usedD.contains(j) {
                    if iouM[i][j] > bestScore { bestScore = iouM[i][j]; bi = i; bj = j }
                }
            }
            if bi < 0 { break }
            matches.append((bi, bj))
            usedT.insert(bi); usedD.insert(bj)
        }
        let unmT = (0..<T).filter { !usedT.contains($0) }
        let unmD = (0..<D).filter { !usedD.contains($0) }
        return (matches, unmT, unmD)
    }

    private static func iou(_ a: CGRect, _ b: CGRect) -> Float {
        let inter = a.intersection(b)
        if inter.isNull || inter.isEmpty { return 0 }
        let ai = Float(inter.width * inter.height)
        let au = Float(a.width * a.height + b.width * b.height) - ai
        return au > 0 ? ai / au : 0
    }
}

// MARK: - Single Track

fileprivate final class STrack {
    enum State { case tracked, lost, removed }

    let id: Int
    var state: State = .tracked
    var classIndex: Int
    var label: String
    var confidence: Float
    private(set) var frameId: Int
    private(set) var startFrame: Int
    private(set) var hits: Int = 1
    private(set) var isActivated: Bool
    private let kf = KalmanBoxFilter()
    private let trailMaxLen: Int
    private var trailPoints: [CGPoint] = []

    init(det: Det, id: Int, frameId: Int, firstFrame: Bool, trailMaxLen: Int) {
        self.id = id
        self.classIndex = det.classIndex
        self.label = det.label
        self.confidence = det.confidence
        self.frameId = frameId
        self.startFrame = frameId
        self.trailMaxLen = trailMaxLen
        self.isActivated = firstFrame
        kf.initiate(measurement: STrack.xyah(det.rect))
        appendTrail()
    }

    func predict() {
        if state != .tracked { kf.mean[7] = 0 }
        kf.predict()
    }

    func update(det: Det, frameId: Int) {
        self.frameId = frameId
        self.confidence = det.confidence
        self.classIndex = det.classIndex
        self.label = det.label
        self.hits += 1
        kf.update(measurement: STrack.xyah(det.rect))
        state = .tracked
        isActivated = true
        appendTrail()
    }

    func reActivate(det: Det, frameId: Int) {
        kf.update(measurement: STrack.xyah(det.rect))
        self.frameId = frameId
        self.confidence = det.confidence
        state = .tracked
        isActivated = true
        appendTrail()
    }

    func markLost()    { state = .lost }
    func markRemoved() { state = .removed }

    func predictedRect() -> CGRect { STrack.rectFromXYAH(kf.mean) }

    func asDetection() -> Det {
        var d = Det(label: label, confidence: confidence,
                    rect: predictedRect(), classIndex: classIndex)
        d.trackId = id
        d.trail = trailPoints
        return d
    }

    private func appendTrail() {
        trailPoints.append(CGPoint(x: kf.mean[0], y: kf.mean[1]))
        if trailPoints.count > trailMaxLen {
            trailPoints.removeFirst(trailPoints.count - trailMaxLen)
        }
    }

    private static func xyah(_ r: CGRect) -> [Double] {
        let h = max(Double(r.height), 1e-6)
        let a = Double(r.width) / h
        return [Double(r.midX), Double(r.midY), a, h]
    }

    private static func rectFromXYAH(_ m: [Double]) -> CGRect {
        let cx = m[0], cy = m[1], a = m[2], h = m[3]
        let w = a * h
        return CGRect(x: cx - w / 2, y: cy - h / 2, width: w, height: h)
    }
}

// MARK: - 8D constant-velocity Kalman filter (ByteTrack specification)

fileprivate final class KalmanBoxFilter {
    var mean: [Double] = Array(repeating: 0, count: 8)
    var cov:  [[Double]]
    private let stdPos: Double = 1.0 / 20.0
    private let stdVel: Double = 1.0 / 160.0

    init() { cov = KalmanBoxFilter.identity(8, scale: 10) }

    func initiate(measurement: [Double]) {
        mean = measurement + [0, 0, 0, 0]
        let h = measurement[3]
        let std: [Double] = [
            2 * stdPos * h, 2 * stdPos * h, 1e-2, 2 * stdPos * h,
            10 * stdVel * h, 10 * stdVel * h, 1e-5, 10 * stdVel * h
        ]
        cov = KalmanBoxFilter.diag(std.map { $0 * $0 })
    }

    func predict() {
        for i in 0..<4 { mean[i] += mean[i + 4] }

        // F P F^T for F = [[I4, I4], [0, I4]] block-wise.
        var P = cov
        for i in 0..<4 {
            for j in 0..<4 {
                let A  = P[i][j]
                let B  = P[i][j + 4]
                let Bt = P[i + 4][j]
                let D  = P[i + 4][j + 4]
                P[i][j]         = A + B + Bt + D
                P[i][j + 4]     = B + D
                P[i + 4][j]     = Bt + D
            }
        }
        let h2 = max(mean[3], 1e-6)
        let std: [Double] = [
            stdPos * h2, stdPos * h2, 1e-2, stdPos * h2,
            stdVel * h2, stdVel * h2, 1e-5, stdVel * h2
        ]
        for i in 0..<8 { P[i][i] += std[i] * std[i] }
        cov = P
    }

    func update(measurement: [Double]) {
        let h2 = max(mean[3], 1e-6)
        let stdR: [Double] = [stdPos * h2, stdPos * h2, 1e-1, stdPos * h2]

        var S: [[Double]] = Array(repeating: Array(repeating: 0, count: 4), count: 4)
        for i in 0..<4 { for j in 0..<4 { S[i][j] = cov[i][j] } }
        for i in 0..<4 { S[i][i] += stdR[i] * stdR[i] }

        guard let Sinv = KalmanBoxFilter.invert4x4(S) else { return }

        var K: [[Double]] = Array(repeating: Array(repeating: 0, count: 4), count: 8)
        for i in 0..<8 {
            for j in 0..<4 {
                var s = 0.0
                for k in 0..<4 { s += cov[i][k] * Sinv[k][j] }
                K[i][j] = s
            }
        }

        var y: [Double] = Array(repeating: 0, count: 4)
        for i in 0..<4 { y[i] = measurement[i] - mean[i] }

        for i in 0..<8 {
            var s = 0.0
            for j in 0..<4 { s += K[i][j] * y[j] }
            mean[i] += s
        }

        var KHP: [[Double]] = Array(repeating: Array(repeating: 0, count: 8), count: 8)
        for i in 0..<8 {
            for j in 0..<8 {
                var s = 0.0
                for k in 0..<4 { s += K[i][k] * cov[k][j] }
                KHP[i][j] = s
            }
        }
        for i in 0..<8 { for j in 0..<8 { cov[i][j] -= KHP[i][j] } }
    }

    static func identity(_ n: Int, scale: Double) -> [[Double]] {
        var m = Array(repeating: Array(repeating: 0.0, count: n), count: n)
        for i in 0..<n { m[i][i] = scale }
        return m
    }

    static func diag(_ v: [Double]) -> [[Double]] {
        let n = v.count
        var m = Array(repeating: Array(repeating: 0.0, count: n), count: n)
        for i in 0..<n { m[i][i] = v[i] }
        return m
    }

    static func invert4x4(_ m: [[Double]]) -> [[Double]]? {
        let n = 4
        var a = m
        var b: [[Double]] = identity(n, scale: 1)
        for i in 0..<n {
            var piv = i
            var maxV = abs(a[i][i])
            for r in (i + 1)..<n where abs(a[r][i]) > maxV {
                maxV = abs(a[r][i]); piv = r
            }
            if maxV < 1e-12 { return nil }
            if piv != i { a.swapAt(i, piv); b.swapAt(i, piv) }
            let inv = 1.0 / a[i][i]
            for j in 0..<n { a[i][j] *= inv; b[i][j] *= inv }
            for r in 0..<n where r != i {
                let f = a[r][i]
                if f == 0 { continue }
                for j in 0..<n {
                    a[r][j] -= f * a[i][j]
                    b[r][j] -= f * b[i][j]
                }
            }
        }
        return b
    }
}
