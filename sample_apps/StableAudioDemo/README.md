# StableAudioDemo

Sample iOS app for [stabilityai/stable-audio-open-small](https://huggingface.co/stabilityai/stable-audio-open-small) — text-to-music generation (497M params). Generates up to 11.9 seconds of stereo 44.1 kHz audio from text prompts using rectified flow diffusion.

<video src="https://github.com/user-attachments/assets/ea448e41-d5ae-407e-84a6-8312c1108cfd" width="400"></video>

## Models

4 CoreML models: T5 text encoder, NumberEmbedder (seconds conditioning), DiT (diffusion transformer), and VAE decoder (Oobleck).

| Model | Size | Input | Output |
|-------|------|-------|--------|
| [StableAudioT5Encoder.mlpackage.zip](https://github.com/john-rocky/CoreML-Models/releases/download/stable-audio-v1/StableAudioT5Encoder.mlpackage.zip) | 105 MB | input_ids [1, 64] | text_embeddings [1, 64, 768] |
| [StableAudioNumberEmbedder.mlpackage.zip](https://github.com/john-rocky/CoreML-Models/releases/download/stable-audio-v1/StableAudioNumberEmbedder.mlpackage.zip) | 396 KB | normalized_seconds [1] | seconds_embedding [1, 768] |
| [StableAudioDiT.mlpackage.zip](https://github.com/john-rocky/CoreML-Models/releases/download/stable-audio-v1/StableAudioDiT.mlpackage.zip) | 326 MB (INT8) | latent [1,64,256] + timestep + conditioning | velocity [1,64,256] |
| [StableAudioDiT_FP32.mlpackage.zip](https://github.com/john-rocky/CoreML-Models/releases/download/stable-audio-v1/StableAudioDiT_FP32.mlpackage.zip) | 1.3 GB (FP32) | latent [1,64,256] + timestep + conditioning | velocity [1,64,256] |
| [StableAudioVAEDecoder.mlpackage.zip](https://github.com/john-rocky/CoreML-Models/releases/download/stable-audio-v1/StableAudioVAEDecoder.mlpackage.zip) | 149 MB | latent [1, 64, 256] | stereo audio [1, 2, 524288] @ 44.1 kHz |

## Setup

1. Download the four `.mlpackage.zip` files (pick **either** `StableAudioDiT` **or** `StableAudioDiT_FP32`)
2. Unzip and drag them into the Xcode project
3. Build and run on a physical device (iOS 17+)

## DiT Variant Selection

| Variant | Compute units | Quality | Speed | Memory |
|---------|---------------|---------|-------|--------|
| `StableAudioDiT` (INT8) | `.cpuAndGPU` | Slight quantization loss | Fastest | 326 MB |
| `StableAudioDiT_FP32` | `.cpuOnly` | Best | Slower | 1.3 GB |

FP16 weights overflow inside the DiT attention on iOS GPU, so the high-quality path requires FP32 compute on CPU. The INT8 GPU path is the practical default for most devices.

## Conversion Notes

- **DiT INT8 + `.cpuAndGPU`** — fastest path, slight quality loss from quantization. Runs cleanly on the GPU because INT8 dequantizes to FP16/FP32 element-wise without the overflow issue.
- **DiT FP32 + `.cpuOnly`** — best quality, larger and slower. Required because the FP16 attention path overflows on the iOS GPU. CPU FP32 is the fallback that gives correct results.
- **T5 INT8 may emit occasional NaN values** — the sample app sanitizes the embeddings (replaces NaN with 0) before passing them to the DiT.
- **VAE Decoder FP16** — `weight_norm` must be removed before tracing because of the Snake activation; see the `remove_weight_norm()` call in the conversion script.

## Conversion Script

[`conversion_scripts/convert_stable_audio.py`](../../conversion_scripts/convert_stable_audio.py)
