import Foundation
import CoreGraphics

// ByteTrack multi-object tracker.
//
// Mobile sweet-spot: no appearance / ReID network, just a per-track
// constant-velocity Kalman filter plus ByteTrack's two-stage association
// (high-conf detections matched first, then low-conf detections are used
// to rescue tracks about to be lost). That trick is what keeps identities
// stable through motion blur and brief occlusions without paying for a
// second neural network on every frame.
//
// Reference: Zhang et al., "ByteTrack: Multi-Object Tracking by
// Associating Every Detection Box", ECCV 2022 (arxiv 2110.06864).

final class ByteTracker {

    struct Config {
        var trackHighThresh: Float = 0.5      // stage-1 dets must exceed this
        var trackLowThresh:  Float = 0.1      // below this, detections are ignored
        var newTrackThresh:  Float = 0.6      // unmatched high-conf spawns a new track only above this
        var trackBuffer:     Int   = 30       // frames a lost track survives before being dropped
        var iouThreshFirst:       Float = 0.2 // min IoU for stage-1 (activated/lost vs high-conf)
        var iouThreshSecond:      Float = 0.5 // min IoU for stage-2 (rescue with low-conf)
        var iouThreshUnconfirmed: Float = 0.3 // min IoU for stage-3 (tentative track confirmation)
        var perClass: Bool = true             // only match tracks and detections of the same class
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

    /// Run one tracking step. Returns the currently confirmed tracks,
    /// each expressed as a `Detection` whose `normRect` is the Kalman-
    /// smoothed position and whose `trackId` is the persistent ID.
    func update(detections dets: [Detection]) -> [Detection] {
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

        // Stage 1: activated + lost tracks vs. high-confidence detections.
        let (m1, uT1, uD1) = associate(tracks: pool, dets: high,
                                       iouThresh: config.iouThreshFirst)
        for (ti, di) in m1 {
            let t = pool[ti], d = high[di]
            if t.state == .tracked { t.update(det: d, frameId: frameId) }
            else                   { t.reActivate(det: d, frameId: frameId) }
        }

        // Stage 2: tracks that missed stage 1 but were Tracked last frame
        // get one more chance against the low-confidence pool. This is
        // the "BYTE" part -- low-conf dets of an occluded object can
        // still re-associate before we lose the identity.
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

        // Stage 3: unconfirmed (one-frame-old) tracks try to claim the
        // leftover high-conf detections.
        let leftoverHigh = uD1.map { high[$0] }
        let (m3, uU3, uD3) = associate(tracks: unconfirmed, dets: leftoverHigh,
                                       iouThresh: config.iouThreshUnconfirmed)
        for (ti, di) in m3 { unconfirmed[ti].update(det: leftoverHigh[di], frameId: frameId) }
        for ti in uU3 { unconfirmed[ti].markRemoved() }

        // Stage 4: remaining strong detections spawn new tentative tracks.
        for di in uD3 {
            let d = leftoverHigh[di]
            guard d.confidence >= config.newTrackThresh else { continue }
            let t = STrack(det: d, id: nextId, frameId: frameId,
                           firstFrame: frameId == 1)
            nextId += 1
            trackedTracks.append(t)
        }

        // Re-partition tracks by state, age out lost tracks beyond buffer.
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

    // Greedy IoU association, class-aware. For typical mobile scenes
    // (<50 objects) this is within noise of Hungarian at a fraction of
    // the code.
    private func associate(tracks: [STrack], dets: [Detection], iouThresh: Float)
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
                iouM[i][j] = ByteTracker.iou(r, d.normRect)
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

    init(det: Detection, id: Int, frameId: Int, firstFrame: Bool) {
        self.id = id
        self.classIndex = det.classIndex
        self.label = det.label
        self.confidence = det.confidence
        self.frameId = frameId
        self.startFrame = frameId
        // ByteTrack convention: only the very first frame's tracks are
        // considered confirmed immediately; later tracks need a second
        // frame match before they're drawn.
        self.isActivated = firstFrame
        kf.initiate(measurement: STrack.xyah(det.normRect))
    }

    func predict() {
        if state != .tracked { kf.mean[7] = 0 }  // freeze height velocity while lost
        kf.predict()
    }

    func update(det: Detection, frameId: Int) {
        self.frameId = frameId
        self.confidence = det.confidence
        self.classIndex = det.classIndex
        self.label = det.label
        self.hits += 1
        kf.update(measurement: STrack.xyah(det.normRect))
        state = .tracked
        isActivated = true
    }

    func reActivate(det: Detection, frameId: Int) {
        kf.update(measurement: STrack.xyah(det.normRect))
        self.frameId = frameId
        self.confidence = det.confidence
        state = .tracked
        isActivated = true
    }

    func markLost()    { state = .lost }
    func markRemoved() { state = .removed }

    func predictedRect() -> CGRect { STrack.rectFromXYAH(kf.mean) }

    func asDetection() -> Detection {
        var d = Detection(label: label, confidence: confidence,
                          classIndex: classIndex, normRect: predictedRect())
        d.trackId = id
        return d
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

// MARK: - 8D constant-velocity Kalman filter (ByteTrack specification).
//
// State is [cx, cy, a, h, vcx, vcy, va, vh] in normalized image coords,
// where a = w / h. Process/measurement noise scale with h, so the same
// tracker works for near and far objects.

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
        // x' = F x, F = I8 with +1 at the velocity offsets.
        for i in 0..<4 { mean[i] += mean[i + 4] }

        // P' = F P F^T computed block-wise:
        //   F = [[I4, I4], [0, I4]], P = [[A, B], [B^T, D]]
        //   => P' = [[A + B + B^T + D, B + D], [B^T + D, D]]
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
                // P[i+4][j+4] stays as D
            }
        }

        // Add process noise Q along the diagonal.
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

        // S = H P H^T + R = P[0..<4, 0..<4] + diag(stdR^2)
        var S: [[Double]] = Array(repeating: Array(repeating: 0, count: 4), count: 4)
        for i in 0..<4 { for j in 0..<4 { S[i][j] = cov[i][j] } }
        for i in 0..<4 { S[i][i] += stdR[i] * stdR[i] }

        guard let Sinv = KalmanBoxFilter.invert4x4(S) else { return }

        // K = P H^T S^-1 = cov[:, 0..<4] * Sinv, shape (8, 4)
        var K: [[Double]] = Array(repeating: Array(repeating: 0, count: 4), count: 8)
        for i in 0..<8 {
            for j in 0..<4 {
                var s = 0.0
                for k in 0..<4 { s += cov[i][k] * Sinv[k][j] }
                K[i][j] = s
            }
        }

        // Innovation y = z - H x
        var y: [Double] = Array(repeating: 0, count: 4)
        for i in 0..<4 { y[i] = measurement[i] - mean[i] }

        // x = x + K y
        for i in 0..<8 {
            var s = 0.0
            for j in 0..<4 { s += K[i][j] * y[j] }
            mean[i] += s
        }

        // P = P - K H P, where K H P = K * cov[0..<4, :]
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

    // Gauss-Jordan inverse for a 4x4 matrix. Returns nil on singularity.
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
