import Foundation
import Accelerate

// MARK: - Note Event

struct NoteEvent {
    let startFrame: Int
    let endFrame: Int
    let midiPitch: Int      // MIDI note number (21 = A0, 108 = C8)
    let amplitude: Float
    var pitchBends: [Float]? // per-frame pitch bend in 1/3 semitone units
}

// MARK: - Note Creation (port of basic_pitch/note_creation.py)

enum NoteCreation {

    static let defaultOnsetThreshold: Float = 0.5
    static let defaultFrameThreshold: Float = 0.3
    static let defaultMinNoteLenFrames: Int = 11  // ~127.7ms (Python default)
    static let energyTolerance: Int = 11           // Python default
    static let midiOffset: Int = 21

    // MARK: - Main Entry Point

    static func modelOutputToNotes(
        output: BasicPitchOutput,
        onsetThreshold: Float = defaultOnsetThreshold,
        frameThreshold: Float = defaultFrameThreshold,
        minNoteLen: Int = defaultMinNoteLenFrames,
        inferOnsets: Bool = true,
        melodiaTrick: Bool = true,
        includePitchBends: Bool = true
    ) -> [NoteEvent] {
        let frames = output.notes
        var onsets = output.onsets
        let contours = output.contours

        guard !frames.isEmpty else { return [] }

        let nFrames = frames.count
        let nFreqs = frames[0].count

        // Infer additional onsets from frame energy differences
        if inferOnsets {
            onsets = getInferredOnsets(onsets: onsets, frames: frames)
        }

        // Detect note events
        var noteEvents = outputToNotesPolyphonic(
            frames: frames,
            onsets: onsets,
            onsetThreshold: onsetThreshold,
            frameThreshold: frameThreshold,
            minNoteLen: minNoteLen,
            melodiaTrick: melodiaTrick
        )

        // (No merging — preserves separate notes of the same pitch)

        // Add pitch bends
        if includePitchBends {
            for i in 0..<noteEvents.count {
                noteEvents[i].pitchBends = getPitchBends(
                    contours: contours,
                    note: noteEvents[i]
                )
            }
        }

        return noteEvents
    }

    // MARK: - Onset Inference

    private static func getInferredOnsets(onsets: [[Float]], frames: [[Float]], nDiff: Int = 2) -> [[Float]] {
        let nFrames = frames.count
        let nFreqs = frames[0].count

        // 1. Compute frame diffs for each order, take element-wise MINIMUM across orders
        // Initialize with large values
        var frameDiff = [[Float]](repeating: [Float](repeating: Float.greatestFiniteMagnitude, count: nFreqs), count: nFrames)

        for diffOrder in 1...nDiff {
            for t in diffOrder..<nFrames {
                for f in 0..<nFreqs {
                    let diff = frames[t][f] - frames[t - diffOrder][f]
                    frameDiff[t][f] = min(frameDiff[t][f], diff)
                }
            }
        }

        // 2. Clip negatives to 0, zero out first nDiff frames
        for t in 0..<nFrames {
            for f in 0..<nFreqs {
                if t < nDiff || frameDiff[t][f] < 0 {
                    frameDiff[t][f] = 0
                }
            }
        }

        // 3. Rescale frameDiff to match the max of original onsets
        let maxOnset = onsets.flatMap { $0 }.max() ?? 1.0
        let maxFrameDiff = frameDiff.flatMap { $0 }.max() ?? 0.0
        if maxFrameDiff > 0 {
            let scale = maxOnset / maxFrameDiff
            for t in 0..<nFrames {
                for f in 0..<nFreqs {
                    frameDiff[t][f] *= scale
                }
            }
        }

        // 4. Return element-wise MAX of original onsets and rescaled frameDiff
        var result = onsets
        for t in 0..<nFrames {
            for f in 0..<nFreqs {
                result[t][f] = max(onsets[t][f], frameDiff[t][f])
            }
        }

        return result
    }

    // MARK: - Polyphonic Note Detection

    private static func outputToNotesPolyphonic(
        frames: [[Float]],
        onsets: [[Float]],
        onsetThreshold: Float,
        frameThreshold: Float,
        minNoteLen: Int,
        melodiaTrick: Bool
    ) -> [NoteEvent] {
        let nFrames = frames.count
        let nFreqs = frames[0].count
        let maxFreqIdx = nFreqs - 1  // 87 for 88 bins

        // Find onset peaks using local maxima detection (matches scipy.signal.argrelmax)
        // np.where returns (row, col) sorted by row then col, then we reverse [::-1]
        var onsetPeaks: [(time: Int, freq: Int)] = []
        for t in 1..<(nFrames - 1) {
            for f in 0..<nFreqs {
                if onsets[t][f] > onsets[t - 1][f] &&
                   onsets[t][f] > onsets[t + 1][f] &&
                   onsets[t][f] >= onsetThreshold {
                    onsetPeaks.append((time: t, freq: f))
                }
            }
        }
        // Reverse to process in descending (time, freq) order, matching Python's [::-1]
        onsetPeaks.reverse()

        // Remaining energy matrix (mutable copy of frames)
        var remainingEnergy = frames

        var noteEvents: [NoteEvent] = []

        // Greedy note tracking from each onset
        for peak in onsetPeaks {
            let noteStartIdx = peak.time
            let freqIdx = peak.freq

            // Skip if too close to end
            if noteStartIdx >= nFrames - 1 { continue }

            // Track forward until energy drops below threshold for energyTolerance frames
            var i = noteStartIdx + 1
            var k = 0
            while i < nFrames - 1 && k < energyTolerance {
                if remainingEnergy[i][freqIdx] < frameThreshold {
                    k += 1
                } else {
                    k = 0
                }
                i += 1
            }
            i -= k  // go back to last frame above threshold

            // if note too short, skip (Python uses <=)
            if i - noteStartIdx <= minNoteLen { continue }

            // Zero out energy for this note (and adjacent freq bins)
            for t in noteStartIdx..<i {
                remainingEnergy[t][freqIdx] = 0
                if freqIdx < maxFreqIdx {
                    remainingEnergy[t][freqIdx + 1] = 0
                }
                if freqIdx > 0 {
                    remainingEnergy[t][freqIdx - 1] = 0
                }
            }

            // Compute amplitude
            var ampSum: Float = 0
            for t in noteStartIdx..<i {
                ampSum += frames[t][freqIdx]
            }
            let amplitude = ampSum / Float(i - noteStartIdx)

            noteEvents.append(NoteEvent(
                startFrame: noteStartIdx,
                endFrame: i,
                midiPitch: freqIdx + midiOffset,
                amplitude: min(amplitude, 1.0),
                pitchBends: nil
            ))
        }

        // Melodia trick: find remaining energy peaks
        if melodiaTrick {
            let melodiaNotes = melodiaTrickNotes(
                remainingEnergy: &remainingEnergy,
                frames: frames,
                frameThreshold: frameThreshold,
                minNoteLen: minNoteLen
            )
            noteEvents.append(contentsOf: melodiaNotes)
        }

        // Sort by start time
        noteEvents.sort { $0.startFrame < $1.startFrame }

        return noteEvents
    }

    // MARK: - Melodia Trick

    private static let maxMelodiaIterations = 5000  // safety cap

    private static func melodiaTrickNotes(
        remainingEnergy: inout [[Float]],
        frames: [[Float]],
        frameThreshold: Float,
        minNoteLen: Int
    ) -> [NoteEvent] {
        let nFrames = remainingEnergy.count
        let nFreqs = remainingEnergy[0].count
        let maxFreqIdx = nFreqs - 1
        var notes: [NoteEvent] = []

        for _ in 0..<maxMelodiaIterations {
            // Find global maximum using vDSP for speed
            var maxVal: Float = 0
            var iMid = 0
            var freqIdx = 0
            for t in 0..<nFrames {
                remainingEnergy[t].withUnsafeBufferPointer { buf in
                    var localMax: Float = 0
                    var localIdx: vDSP_Length = 0
                    vDSP_maxvi(buf.baseAddress!, 1, &localMax, &localIdx, vDSP_Length(nFreqs))
                    if localMax > maxVal {
                        maxVal = localMax
                        iMid = t
                        freqIdx = Int(localIdx)
                    }
                }
            }

            if maxVal <= frameThreshold { break }
            remainingEnergy[iMid][freqIdx] = 0

            // Forward pass — zero energy as we walk
            var i = iMid + 1
            var k = 0
            while i < nFrames - 1 && k < energyTolerance {
                if remainingEnergy[i][freqIdx] < frameThreshold {
                    k += 1
                } else {
                    k = 0
                }
                remainingEnergy[i][freqIdx] = 0
                if freqIdx < maxFreqIdx {
                    remainingEnergy[i][freqIdx + 1] = 0
                }
                if freqIdx > 0 {
                    remainingEnergy[i][freqIdx - 1] = 0
                }
                i += 1
            }
            let iEnd = i - 1 - k

            // Backward pass — zero energy as we walk
            i = iMid - 1
            k = 0
            while i > 0 && k < energyTolerance {
                if remainingEnergy[i][freqIdx] < frameThreshold {
                    k += 1
                } else {
                    k = 0
                }
                remainingEnergy[i][freqIdx] = 0
                if freqIdx < maxFreqIdx {
                    remainingEnergy[i][freqIdx + 1] = 0
                }
                if freqIdx > 0 {
                    remainingEnergy[i][freqIdx - 1] = 0
                }
                i -= 1
            }
            let iStart = i + 1 + k

            if iEnd - iStart <= minNoteLen { continue }
            guard iStart >= 0 && iEnd <= nFrames && iEnd > iStart else { continue }

            var ampSum: Float = 0
            for t in iStart..<iEnd {
                ampSum += frames[t][freqIdx]
            }
            let amplitude = ampSum / Float(iEnd - iStart)

            notes.append(NoteEvent(
                startFrame: iStart,
                endFrame: iEnd,
                midiPitch: freqIdx + midiOffset,
                amplitude: min(amplitude, 1.0),
                pitchBends: nil
            ))
        }

        return notes
    }

    // MARK: - Pitch Bend Extraction

    private static func getPitchBends(contours: [[Float]], note: NoteEvent) -> [Float] {
        let nContourBins = contours[0].count  // 264
        let binsPerSemitone = BasicPitchConstants.contoursBinsPerSemitone  // 3

        // Convert MIDI pitch to contour bin center
        let freqIdx = note.midiPitch - midiOffset
        let centerBin = freqIdx * binsPerSemitone + binsPerSemitone / 2

        let windowHalf = 25
        let windowSize = windowHalf * 2 + 1  // 51

        // Gaussian window (std=5)
        var gaussian = [Float](repeating: 0, count: windowSize)
        let std: Float = 5.0
        for i in 0..<windowSize {
            let x = Float(i - windowHalf)
            gaussian[i] = exp(-x * x / (2 * std * std))
        }

        var bends = [Float](repeating: 0, count: note.endFrame - note.startFrame)

        for t in note.startFrame..<note.endFrame {
            let frameIdx = t - note.startFrame
            guard t < contours.count else {
                bends[frameIdx] = 0
                continue
            }

            // Extract contour values around center bin with Gaussian weighting
            var bestBin = centerBin
            var bestVal: Float = -1
            for i in 0..<windowSize {
                let bin = centerBin - windowHalf + i
                guard bin >= 0 && bin < nContourBins else { continue }
                let weighted = contours[t][bin] * gaussian[i]
                if weighted > bestVal {
                    bestVal = weighted
                    bestBin = bin
                }
            }

            // Pitch bend in units of 1/3 semitone
            bends[frameIdx] = Float(bestBin - centerBin)
        }

        return bends
    }

    // MARK: - Merge Adjacent Notes

    /// Merge notes of the same pitch that are separated by a small gap
    private static let maxMergedLenFrames = 344  // ~4 seconds max per merged note

    private static func mergeAdjacentNotes(_ notes: [NoteEvent], maxGapFrames: Int, frames: [[Float]]) -> [NoteEvent] {
        guard !notes.isEmpty else { return [] }

        // Group by pitch
        var byPitch: [Int: [NoteEvent]] = [:]
        for note in notes {
            byPitch[note.midiPitch, default: []].append(note)
        }

        var merged: [NoteEvent] = []
        for (pitch, pitchNotes) in byPitch {
            let sorted = pitchNotes.sorted { $0.startFrame < $1.startFrame }
            var current = sorted[0]

            for i in 1..<sorted.count {
                let next = sorted[i]
                let gap = next.startFrame - current.endFrame
                if gap <= maxGapFrames && (next.endFrame - current.startFrame) < maxMergedLenFrames {
                    // Merge: extend current note to cover next
                    current = NoteEvent(
                        startFrame: current.startFrame,
                        endFrame: next.endFrame,
                        midiPitch: pitch,
                        amplitude: (current.amplitude + next.amplitude) / 2,
                        pitchBends: nil
                    )
                } else {
                    merged.append(current)
                    current = next
                }
            }
            merged.append(current)
        }

        return merged.sorted { $0.startFrame < $1.startFrame }
    }

    // MARK: - Frame to Time Conversion

    static func frameToTime(_ frame: Int) -> Double {
        return Double(frame) * Double(BasicPitchConstants.fftHop) / BasicPitchConstants.audioSampleRate
    }
}
