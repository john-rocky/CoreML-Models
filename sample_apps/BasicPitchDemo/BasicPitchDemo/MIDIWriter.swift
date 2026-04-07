import Foundation

// MARK: - Standard MIDI File Writer

enum MIDIWriter {

    static func writeMIDI(notes: [NoteEvent], tempo: Double = 120.0, to url: URL) throws {
        let ticksPerBeat: UInt16 = 480
        let microsecondsPerBeat = UInt32(60_000_000 / tempo)

        var track = Data()

        // Tempo meta event
        track.append(deltaTime(0))
        track.append(contentsOf: [0xFF, 0x51, 0x03])
        track.append(UInt8((microsecondsPerBeat >> 16) & 0xFF))
        track.append(UInt8((microsecondsPerBeat >> 8) & 0xFF))
        track.append(UInt8(microsecondsPerBeat & 0xFF))

        // Program change: Electric Piano 1 (program 4)
        track.append(deltaTime(0))
        track.append(contentsOf: [0xC0, 4])

        // Collect all events with absolute tick times
        var events: [(tick: UInt32, data: [UInt8])] = []

        for note in notes {
            let startTime = NoteCreation.frameToTime(note.startFrame)
            let endTime = NoteCreation.frameToTime(note.endFrame)

            let startTick = timeToTick(startTime, tempo: tempo, ticksPerBeat: ticksPerBeat)
            let endTick = timeToTick(endTime, tempo: tempo, ticksPerBeat: ticksPerBeat)

            let velocity = UInt8(max(1, min(127, Int(roundf(127.0 * note.amplitude)))))
            let pitch = UInt8(max(0, min(127, note.midiPitch)))

            // Note On
            events.append((tick: startTick, data: [0x90, pitch, velocity]))
            // Note Off
            events.append((tick: endTick, data: [0x80, pitch, 0]))

            // Pitch bends
            if let bends = note.pitchBends, !bends.isEmpty {
                let noteDuration = endTick - startTick
                for (i, bend) in bends.enumerated() {
                    let bendTick = startTick + UInt32(Double(i) / Double(bends.count) * Double(noteDuration))
                    // Convert from 1/3 semitone units to MIDI pitch bend
                    // MIDI pitch bend: 8192 = center, range is +/- 2 semitones (8192 per 2 semitones)
                    let bendValue = Int(roundf(bend * 4096.0 / 3.0))
                    let clampedBend = max(-8192, min(8191, bendValue)) + 8192
                    let lsb = UInt8(clampedBend & 0x7F)
                    let msb = UInt8((clampedBend >> 7) & 0x7F)
                    events.append((tick: bendTick, data: [0xE0, lsb, msb]))
                }
            }
        }

        // Sort by tick time
        events.sort { $0.tick < $1.tick }

        // Write events with delta times
        var lastTick: UInt32 = 0
        for event in events {
            let delta = event.tick - lastTick
            track.append(deltaTime(delta))
            track.append(contentsOf: event.data)
            lastTick = event.tick
        }

        // End of track
        track.append(deltaTime(0))
        track.append(contentsOf: [0xFF, 0x2F, 0x00])

        // Build file
        var fileData = Data()

        // Header chunk: MThd
        fileData.append(contentsOf: [0x4D, 0x54, 0x68, 0x64])  // "MThd"
        fileData.append(uint32BE(6))             // header length
        fileData.append(uint16BE(0))             // format 0
        fileData.append(uint16BE(1))             // 1 track
        fileData.append(uint16BE(ticksPerBeat))  // ticks per beat

        // Track chunk: MTrk
        fileData.append(contentsOf: [0x4D, 0x54, 0x72, 0x6B])  // "MTrk"
        fileData.append(uint32BE(UInt32(track.count)))
        fileData.append(track)

        try fileData.write(to: url)
    }

    // MARK: - Helpers

    private static func timeToTick(_ time: Double, tempo: Double, ticksPerBeat: UInt16) -> UInt32 {
        return UInt32(time * tempo / 60.0 * Double(ticksPerBeat))
    }

    private static func deltaTime(_ ticks: UInt32) -> Data {
        var value = ticks
        var bytes: [UInt8] = []

        bytes.append(UInt8(value & 0x7F))
        value >>= 7

        while value > 0 {
            bytes.append(UInt8((value & 0x7F) | 0x80))
            value >>= 7
        }

        return Data(bytes.reversed())
    }

    private static func uint32BE(_ value: UInt32) -> Data {
        var data = Data(count: 4)
        data[0] = UInt8((value >> 24) & 0xFF)
        data[1] = UInt8((value >> 16) & 0xFF)
        data[2] = UInt8((value >> 8) & 0xFF)
        data[3] = UInt8(value & 0xFF)
        return data
    }

    private static func uint16BE(_ value: UInt16) -> Data {
        var data = Data(count: 2)
        data[0] = UInt8((value >> 8) & 0xFF)
        data[1] = UInt8(value & 0xFF)
        return data
    }
}
