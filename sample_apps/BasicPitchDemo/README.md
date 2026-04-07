# BasicPitchDemo

Sample iOS app for [spotify/basic-pitch](https://github.com/spotify/basic-pitch) — polyphonic Automatic Music Transcription. Converts any audio (any instrument, any voice) into MIDI notes with pitch bend detection. Just **17 K parameters / 272 KB** — runs in real time on iPhone with full ANE acceleration.

The first open-source iOS implementation. Loads any audio file, runs the CoreML model in 2-second sliding windows, then runs the full Python `note_creation.py` pipeline natively in Swift (onset inference, greedy backwards-in-time tracking, melodia trick, pitch bend extraction). Detected notes are visualised as a piano roll, exported as a Standard MIDI File, and played back through a built-in additive sine synth so you can A/B compare with the original audio.

<video src="https://github.com/user-attachments/assets/d4e96b51-680f-471c-93d1-7546d5890cd7" width="400"></video>

## Models

| Model | Size | Input | Output |
|-------|------|-------|--------|
| [BasicPitch_nmp.mlpackage.zip](https://github.com/john-rocky/CoreML-Models/releases/download/basic-pitch-v1/BasicPitch_nmp.mlpackage.zip) | 272 KB | audio waveform [1, 43844, 1] @ 22050 Hz mono | note [1, 172, 88] + onset [1, 172, 88] + contour [1, 172, 264] |

## Setup

1. Download the `.mlpackage.zip` above
2. Unzip and drag the `.mlpackage` into the Xcode project
3. Build and run on a physical device (iOS 17+)

## Pipeline

```
audio file ──→ AVAudioFile decode ──→ peak-normalize to 0.98 ──→ resample to 22050 Hz mono
                                                                           │
                                                          2-second sliding windows
                                                                           │
                                                                       BasicPitch
                                                                           │
                                                          (note, onset, contour)
                                                                           │
                                                              note_creation pipeline (Swift)
                                                                           │
                                                       MIDI notes + pitch bends
                                                                           │
                                                       piano-roll visual + .mid export + sine synth playback
```

## Implementation Notes (iOS-specific)

- **MLMultiArray strides matter on ANE.** The Neural Engine returns the `(1, 172, 88)` output with stride `[16512, 96, 1]` — rows are padded from 88 to 96 columns for alignment. Reading `dataPointer` linearly gives garbage; use `array.strides` to skip the padding.
- **MP3 decoder mismatch.** iOS Core Audio's MP3 decoder produces ~7% louder samples than librosa (sometimes exceeding ±1.0). The same MP3 fed to Python and to the iOS app yields different note detections from the same model. Fix: peak-normalise the loaded audio to 0.98 with `vDSP_maxmgv` before windowing.
- **Algorithm port is exact.** Onset inference uses element-wise min across diff orders, the greedy tracker uses Python's `i -= k` rollback with `i < n_frames - 1` boundary, the melodia trick zeroes energy in-place during forward/backward passes including the `± 1` freq neighbours. With matching audio input the Swift output matches the Python reference note-for-note.

A deeper dive into the post-processing port (greedy tracker rollback, melodia trick in-place zeroing, `min_note_len` boundary conditions) is in [`docs/coreml_conversion_notes.md`](../../docs/coreml_conversion_notes.md#music-transcription-basic-pitch--ios-pitfalls).
