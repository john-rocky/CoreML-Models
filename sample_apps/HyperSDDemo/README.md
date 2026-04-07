# HyperSDDemo

Sample iOS app for [ByteDance/Hyper-SD](https://huggingface.co/ByteDance/Hyper-SD) — single-step text-to-image generation distilled from SD 1.5 via Trajectory Segmented Consistency Distillation. ByteDance reports 2× user preference vs. SD-Turbo at 1 step.

<video src="https://github.com/user-attachments/assets/dd456c13-d778-4a84-8bb2-9dfd78de3070" width="400"></video>

<img width="400" src="hypersd_demo.png">

*1-step generations on iPhone, 512×512. Prompts: cat with sunglasses, cyberpunk city, japanese garden, astronaut on horse.*

## Models

4 CoreML models, ~947 MB total. CLIP text encoder + Swin-style chunked UNet (6-bit palettized) + VAE decoder. The TCD scheduler (custom Swift implementation) drives single-step inference.

| Model | Size | Input | Output |
|-------|------|-------|--------|
| [HyperSDTextEncoder.mlpackage.zip](https://github.com/john-rocky/CoreML-Models/releases/download/hypersd-v1/HyperSDTextEncoder.mlpackage.zip) | 235 MB | input_ids [1,77] | encoder_hidden_states [1,77,768] |
| [HyperSDUnetChunk1.mlpackage.zip](https://github.com/john-rocky/CoreML-Models/releases/download/hypersd-v1/HyperSDUnetChunk1.mlpackage.zip) | 318 MB | latent + encoder_hs + timestep | first half intermediates |
| [HyperSDUnetChunk2.mlpackage.zip](https://github.com/john-rocky/CoreML-Models/releases/download/hypersd-v1/HyperSDUnetChunk2.mlpackage.zip) | 299 MB | first half outputs + skip connections | noise_pred [2,4,64,64] |
| [HyperSDVAEDecoder.mlpackage.zip](https://github.com/john-rocky/CoreML-Models/releases/download/hypersd-v1/HyperSDVAEDecoder.mlpackage.zip) | 95 MB | latent [1,4,64,64] | image [1,3,512,512] |

## Setup

1. Download the four `.mlpackage.zip` files above
2. Unzip and drag them into the Xcode project
3. Build and run on iPhone 15 or newer (the chunked UNet expects ANE)

## Conversion Notes

- **LoRA fusion before conversion**. The Hyper-SD 1-step LoRA is fused into the SD 1.5 base model with `pipe.fuse_lora()` before handing the unified model to Apple's `ml-stable-diffusion`.
- **Apple's `torch2coreml` toolchain** is invoked with `--attention-implementation SPLIT_EINSUM` (Neural Engine path) and `--chunk-unet` (memory-efficient inference). UNet is split across two mlpackages so each chunk fits ANE memory.
- **6-bit kmeans palettization on UNet only**. The CLIP text encoder's FP16 weights contain `inf` values that break kmeans, so the text encoder ships at FP16 instead.
- **Quantize after chunking, not before**. Apple's tool palettizes the unchunked model; once chunks are emitted, each chunk has to be re-palettized separately.
- **coremltools 9.0 patches required**:
  - Custom `int` op converter for multi-dim tensor shape casts
  - `list(block.operations)` workaround in `chunk_mlprogram.py` for the new `CacheDoublyLinkedList` API
- **Inference scheduler**. A custom Swift TCD scheduler implementation drives single-step inference with `guidance_scale=1.0` (no CFG amplification at 1 step).

## Conversion Script

[`conversion_scripts/convert_hypersd.py`](../../conversion_scripts/convert_hypersd.py)
