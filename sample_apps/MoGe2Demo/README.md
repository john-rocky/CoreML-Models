# MoGe2Demo

Sample iOS app for [microsoft/MoGe](https://github.com/microsoft/MoGe) (CVPR 2025 Oral) — open-domain monocular 3D geometry estimation. Given a single photo it predicts a metric depth map, surface normals, and a confidence mask in one forward pass on a DINOv2 ViT-B backbone with three task heads.

This demo uses the **MoGe-2 ViT-B + normal** variant (104 M parameters) at a fixed 504 × 504 input. The CoreML model is one self-contained mlpackage (~200 MB at FP16) that returns five tensors: `points`, `depth`, `normal`, `mask`, `metric_scale`.

## Models

| Module | Inputs | Outputs |
|---|---|---|
| `MoGe2_ViTB_Normal_504` | image [1, 3, 504, 504] in [0, 1] | points [1, 504, 504, 3], depth [1, 504, 504], normal [1, 504, 504, 3], mask [1, 504, 504], metric_scale [1] |

## Setup

1. Download [MoGe2_ViTB_Normal_504.mlpackage.zip](https://github.com/john-rocky/CoreML-Models/releases/download/moge2-v1/MoGe2_ViTB_Normal_504.mlpackage.zip) (or build from [`conversion_scripts/convert_moge2.py`](../../conversion_scripts/convert_moge2.py))
2. Unzip and drop `MoGe2_ViTB_Normal_504.mlpackage` into the Xcode project
3. Build and run on a physical device (iOS 17+)

## Conversion Notes

- **DINOv2 backbone with a frozen pos_embed.** The stock DINOv2 `interpolate_pos_encoding` does a bicubic + antialias resize of the pretrained positional embedding every forward call. coremltools cannot trace bicubic + antialias cleanly. The converter pre-computes the interpolated pos_embed once for the fixed 36 × 36 token grid and replaces the method with a constant lookup, so the traced graph never hits bicubic interpolation.
- **`onnx_compatible_mode = True`** disables the antialias path in `DINOv2Encoder.forward` as well, leaving only bilinear `F.interpolate` calls that coremltools handles natively.
- **Aspect-ratio-aware num_tokens path is collapsed.** MoGeModel computes `base_h, base_w` at runtime from `(num_tokens, aspect_ratio)`. The wrapper hard-codes `base_h = base_w = 36` (= 504 / 14) so the trace sees Python ints all the way through.
- **Pyramid features and UV grids are pre-computed**. `normalized_view_plane_uv` for each of the 5 levels is registered as a non-persistent buffer at wrapper construction time, so the converted graph contains no `linspace` / `meshgrid` ops.
- **`int` op patch for multi-dim shape casts** is required (same as SinSR / Swin Transformer). DINOv2's positional indexing emits int casts on a 2-element shape tensor that the stock coremltools converter assumes are scalars. The patched op accepts both.
- **Outputs match MoGeModel.forward with `remap_output='exp'`** baked in (the wrapper inlines the `xy * z, z = exp(z)` remap so Swift just receives the final `(B, H, W, 3)` point map plus a depth slice).
- **Focal / shift / intrinsics recovery is left to the Swift driver.** The CoreML model returns the affine point map plus `metric_scale`; the demo app applies the scale and ignores any focal-shift refinement (good enough for visualization). For metric SLAM-style use you would port `recover_focal_shift` to Swift.
- **FP16 throughout.** DINOv2 ViT attention does not overflow at this resolution; FP16 + `.cpuAndNeuralEngine` runs comfortably on iPhone 15 / 17. Real-image parity vs the PyTorch reference is ~1 % relative on depth and < 6° on surface normals.

## Inference Pipeline (Swift)

1. Center-crop the photo to a square, resize to 504 × 504.
2. Wrap in a BGRA `CVPixelBuffer` (the model's `ImageType` input applies `scale = 1/255` automatically).
3. Run `MLModel.prediction`. Read `depth`, `normal`, `mask`, `metric_scale` with **stride-aware** access (the ANE returns non-contiguous strides — see Basic Pitch in `docs/coreml_conversion_notes.md`).
4. Mask out background pixels and multiply depth by `metric_scale` to get meters.
5. Render: turbo colormap for depth (near = warm), surface-normal RGB for normals (`nx, -ny, nz` → `r, g, b`).

## Conversion Script

[`conversion_scripts/convert_moge2.py`](../../conversion_scripts/convert_moge2.py)

```bash
python convert_moge2.py                                    # ViT-B normal, 504×504
python convert_moge2.py --variant vits-normal --size 392   # smaller / faster
python convert_moge2.py --variant vitl-normal --size 504   # higher quality, ~660 MB
```
