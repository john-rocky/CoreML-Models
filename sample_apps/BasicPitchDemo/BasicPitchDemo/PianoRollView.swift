import SwiftUI

struct PianoRollView: View {
    let notes: [NoteEvent]
    let totalDuration: Double  // seconds

    private let noteNames = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    private let pianoKeyHeight: CGFloat = 10
    private let pixelsPerSecond: CGFloat = 80

    private var midiRange: ClosedRange<Int> {
        guard !notes.isEmpty else { return 60...72 }
        let pitches = notes.map(\.midiPitch)
        let lo = max(21, (pitches.min() ?? 60) - 2)
        let hi = min(108, (pitches.max() ?? 72) + 2)
        return lo...hi
    }

    var body: some View {
        let range = midiRange
        let totalWidth = max(pixelsPerSecond * CGFloat(totalDuration), 300)
        let totalHeight = CGFloat(range.count) * pianoKeyHeight

        ScrollView([.horizontal, .vertical]) {
            ZStack(alignment: .topLeading) {
                // Grid background
                Canvas { context, size in
                    // Horizontal lines for each pitch
                    for i in 0...range.count {
                        let y = CGFloat(i) * pianoKeyHeight
                        let pitch = range.upperBound - i
                        let isBlackKey = [1, 3, 6, 8, 10].contains(pitch % 12)

                        if isBlackKey {
                            context.fill(
                                Path(CGRect(x: 0, y: y, width: size.width, height: pianoKeyHeight)),
                                with: .color(.gray.opacity(0.1))
                            )
                        }

                        // C note line
                        if pitch % 12 == 0 {
                            var line = Path()
                            line.move(to: CGPoint(x: 0, y: y))
                            line.addLine(to: CGPoint(x: size.width, y: y))
                            context.stroke(line, with: .color(.gray.opacity(0.3)), lineWidth: 0.5)
                        }
                    }

                    // Beat lines
                    let beatInterval = pixelsPerSecond * 60.0 / 120.0  // at 120 BPM
                    var x: CGFloat = 0
                    while x < size.width {
                        var line = Path()
                        line.move(to: CGPoint(x: x, y: 0))
                        line.addLine(to: CGPoint(x: x, y: size.height))
                        context.stroke(line, with: .color(.gray.opacity(0.15)), lineWidth: 0.5)
                        x += beatInterval
                    }
                }
                .frame(width: totalWidth, height: totalHeight)

                // Note rectangles
                ForEach(Array(notes.enumerated()), id: \.offset) { _, note in
                    let startTime = NoteCreation.frameToTime(note.startFrame)
                    let endTime = NoteCreation.frameToTime(note.endFrame)
                    let x = CGFloat(startTime) * pixelsPerSecond
                    let width = CGFloat(endTime - startTime) * pixelsPerSecond
                    let y = CGFloat(range.upperBound - note.midiPitch) * pianoKeyHeight

                    RoundedRectangle(cornerRadius: 2)
                        .fill(noteColor(midi: note.midiPitch, amplitude: note.amplitude))
                        .frame(width: max(width, 2), height: pianoKeyHeight - 1)
                        .offset(x: x, y: y)
                }
            }
            .frame(width: totalWidth, height: totalHeight)
            .padding(.leading, 40)
            .overlay(alignment: .leading) {
                // Piano key labels
                VStack(spacing: 0) {
                    ForEach((range).reversed(), id: \.self) { pitch in
                        Text(noteName(pitch))
                            .font(.system(size: 7, design: .monospaced))
                            .frame(width: 36, height: pianoKeyHeight, alignment: .trailing)
                            .foregroundColor(pitch % 12 == 0 ? .primary : .secondary)
                    }
                }
            }
        }
    }

    private func noteName(_ midi: Int) -> String {
        let name = noteNames[midi % 12]
        let octave = midi / 12 - 1
        return "\(name)\(octave)"
    }

    private func noteColor(midi: Int, amplitude: Float) -> Color {
        let hue = Double(midi % 12) / 12.0
        return Color(hue: hue, saturation: 0.7, brightness: Double(0.5 + amplitude * 0.5))
    }
}
