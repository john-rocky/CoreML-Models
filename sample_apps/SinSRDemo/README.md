# SinSRDemo

Sample iOS app for [SinSR](https://github.com/wyf0912/SinSR) — single-step diffusion-based super-resolution (CVPR 2024). 4× upscaling in one denoiser pass via a Swin Transformer UNet on a VQ-VAE latent space (~113M params).

<img width="512" src="sinsr_demo.png">

*Left: bicubic 4× upscale, Right: SinSR single-step diffusion SR (128×128 → 512×512)*

## Models

| Model | Size | Input | Output |
|-------|------|-------|--------|
| [SinSR_Encoder.mlpackage.zip](https://github.com/john-rocky/CoreML-Models/releases/download/sinsr-v1/SinSR_Encoder.mlpackage.zip) | 39 MB | image [1,3,1024,1024] | latent [1,3,256,256] |
| [SinSR_Denoiser.mlpackage.zip](https://github.com/john-rocky/CoreML-Models/releases/download/sinsr-v1/SinSR_Denoiser.mlpackage.zip) | 420 MB | input [1,6,256,256] | predicted_latent [1,3,256,256] |
| [SinSR_Decoder.mlpackage.zip](https://github.com/john-rocky/CoreML-Models/releases/download/sinsr-v1/SinSR_Decoder.mlpackage.zip) | 58 MB | latent [1,3,256,256] | image [1,3,1024,1024] |

## Setup

1. Download the three `.mlpackage.zip` files above
2. Unzip and drag them into the Xcode project
3. Build and run on a physical device (iOS 17+)

## Inference Pipeline

```
LQ image → resize 256² → Encoder → latent [1,3,256,256]
                                          │
              add noise (κ=2.0, η_T=0.99) ▼
                                  noisy_latent
                                          │
               concat with LQ → [1,6,256,256]
                                          │
                                       Denoiser  (single step, t=14 baked in)
                                          │
                                  predicted_latent
                                          │
                       clamp [-1, 1] → Decoder → image [1,3,1024,1024]
```

The denoiser runs **once** (not iteratively) — that's the SinSR distillation. Swift handles noise injection, scaling, and latent space marshalling.

## Conversion Notes

- **Swin Transformer patches required for tracing**:
  - Pre-compute relative position bias as `register_buffer`
  - Replace `torch.roll` with `slice + concat`
  - Rewrite attention-mask creation to avoid `__setitem__`
  - Patch the coremltools `int` op converter to handle multi-dim tensor shape casts
- **VQ-VAE decoder ships with vector quantization inside the CoreML model** — 8192-entry codebook with `argmin` nearest-neighbor lookup runs on-device.
- **Denoiser input** is a 6-channel concat of `[scaled_noisy_latent, lq_image]` with the timestep baked in (always `t=14` for the single-step distillation).
- **Denoiser must use FP32 precision**. FP16 causes a pinkish color shift via overflow inside the Swin attention layers. Encoder/Decoder are fine in FP16.
- **Use `.cpuOnly` for the denoiser** for best accuracy. Encoder/Decoder run happily on `.cpuAndGPU` or `.all`.
- The output has a slight, consistent color shift relative to the input — that's inherent to the SinSR distilled architecture, not a conversion artifact.

## Conversion Script

[`conversion_scripts/convert_sinsr.py`](../../conversion_scripts/convert_sinsr.py)
