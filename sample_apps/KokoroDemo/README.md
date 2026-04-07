# KokoroDemo

Sample iOS app for [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) — open-weight 82M-parameter TTS using a style-conditioned StyleTTS2 architecture (BERT + duration predictor + iSTFTNet vocoder) producing 24 kHz speech.

The first CoreML port with **on-device bilingual (English + Japanese) free-text input** — no MLX, no MeCab, no IPADic, no Python G2P at runtime.

<video src="https://github.com/user-attachments/assets/56eb2ffc-f915-4f8b-b6d3-1021f3d490ca" width="400"></video>

## Models

2 CoreML models: a flexible-length **Predictor** (BERT + LSTM duration head + text encoder) and **3 fixed-shape Decoder buckets** (128 / 256 / 512 frames). The Swift pipeline picks the smallest bucket that fits the predicted duration, pads input features with zeros, and trims the output audio.

| Model | Size | Input | Output |
|-------|------|-------|--------|
| [Kokoro_Predictor.mlpackage.zip](https://github.com/john-rocky/CoreML-Models/releases/download/kokoro-v1/Kokoro_Predictor.mlpackage.zip) | 75 MB | input_ids [1, T≤256] (int32) + ref_s_style [1, 128] | duration [1, T] + d_for_align [1, 640, T] + t_en [1, 512, T] |
| [Kokoro_Decoder_128.mlpackage.zip](https://github.com/john-rocky/CoreML-Models/releases/download/kokoro-v1/Kokoro_Decoder_128.mlpackage.zip) | 238 MB | en_aligned [1, 640, 128] + asr_aligned [1, 512, 128] + ref_s [1, 256] | audio [1, 76800] @ 24 kHz |
| [Kokoro_Decoder_256.mlpackage.zip](https://github.com/john-rocky/CoreML-Models/releases/download/kokoro-v1/Kokoro_Decoder_256.mlpackage.zip) | 241 MB | en_aligned [1, 640, 256] + asr_aligned [1, 512, 256] + ref_s [1, 256] | audio [1, 153600] @ 24 kHz |
| [Kokoro_Decoder_512.mlpackage.zip](https://github.com/john-rocky/CoreML-Models/releases/download/kokoro-v1/Kokoro_Decoder_512.mlpackage.zip) | 246 MB | en_aligned [1, 640, 512] + asr_aligned [1, 512, 512] + ref_s [1, 256] | audio [1, 307200] @ 24 kHz |

## Setup

1. Download the predictor and the three decoder buckets above
2. Unzip and drag them into the Xcode project
3. Build and run on a physical device (iOS 17+)

Voices ship with the project (510×256 style tensors, ~512 KB each):

- English (5): `af_heart`, `af_bella`, `am_michael`, `bf_emma`, `bm_george`
- Japanese (5): `jf_alpha`, `jf_gongitsune`, `jm_kumo`, `jf_nezumi`, `jf_tebukuro`

## On-Device G2P (no external dependencies)

- **English** — lexicon-based: `us_gold` (~90k entries) + `us_silver` fallback, with possessive splitting, acronym detection, and a rule-based grapheme→phoneme fallback for OOV words. ~6 MB of bundled JSON. No MLX, no Python.
- **Japanese** — Apple's `CFStringTokenizer` (`ja_JP` locale) emits romaji per token; `Latin-Hiragana` ICU transform converts to hiragana; a ported subset of misaki's `cutlet.HEPBURN` IPA table plus context rules (long vowels, ん assimilation, particles は→βa / へ→e) emits Kokoro-compatible phonemes. **No MeCab, no IPADic, ~zero overhead.**

## Conversion Notes

- **`RangeDim` for the predictor** — flexible phoneme length 1–256 (incl. BOS/EOS) so the bidirectional LSTM never sees padding tokens.
- **Fixed-shape decoder buckets** — the iSTFTNet vocoder + InstanceNorm + bidirectional LSTM stack is length-sensitive, so we ship 3 buckets (128 / 256 / 512). Padding inside one bucket only causes a phase shift (spec corr 0.93 vs unpadded reference) which is perceptually inaudible. The smaller padding ratio you can pick, the cleaner the output — hence multiple buckets.
- **Critical bug**: CoreML's `mod` op silently produces wrong values for `(float / scalar) % 1` (e.g. inside the SineGen of iSTFTNet). Output spec correlation drops from 0.996 to 0.67 vs PyTorch with this op in the graph. Replacing `(f0 / sr) % 1` with `(f0 / sr) - floor(f0 / sr)` brings it back to **0.996**. See [`conversion_scripts/convert_kokoro.py`](../../conversion_scripts/convert_kokoro.py) and the patched `kokoro/istftnet.py`.
- **Bypass `pack_padded_sequence` / `pad_packed_sequence`** in the predictor's `TextEncoder` and `DurationEncoder` LSTMs — coremltools can't trace them. The LSTM runs on the unpadded tensor directly via `RangeDim`.
- **Decoder ships at FP32** (`compute_precision=ct.precision.FLOAT32`); FP16 corrupts audio quality.
- **Deterministic noise** — the random noise generators (`torch.randn_like` in `SineGen` and `SourceModuleHnNSF`) are zeroed before tracing so the CoreML model is bit-for-bit reproducible vs PyTorch.
- **ANE-friendly decoder** — first vocoder inference ~700 ms, subsequent inferences ~200 ms on iPhone (warm cache).

## Conversion Script

[`conversion_scripts/convert_kokoro.py`](../../conversion_scripts/convert_kokoro.py)
