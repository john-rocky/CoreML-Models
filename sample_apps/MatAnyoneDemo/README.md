# MatAnyoneDemo

Sample iOS app for [pq-yang/MatAnyone](https://github.com/pq-yang/MatAnyone) (CVPR 2025) — temporally consistent video matting with object-level memory propagation. Given a first-frame mask the network tracks and refines an alpha matte across the whole clip, holding sharp edges (hair, semitransparent regions) much better than per-frame matting baselines. Built on the Cutie video object segmentation backbone with a dedicated mask decoder for matting.

The sample app uses Vision's `VNGeneratePersonSegmentationRequest` to bootstrap the first-frame mask automatically — pick a video, tap "Remove BG", and it composites the foreground over the chosen background colour.

End-to-end alpha matte parity vs the official PyTorch reference: **MAE < 2e-4, correlation 0.9999+** across 18 frames including 3 memory cycles.

## Models

5 mlpackages, ~111 MB total at FP16. The CoreML graph is locked to landscape **768 × 432**; portrait sources are rotated to landscape before the pipeline and rotated back afterwards.

| Module | Inputs | Outputs |
|---|---|---|
| `encoder` | image [1, 3, 432, 768] | f16 / f8 / f4 / f2 / f1 multi-scale features + pix_feat + key + shrinkage + selection |
| `mask_encoder` | image, pix_feat, sensory, mask | mask_value, new_sensory, obj_summary |
| `read_first` | pix_feat, last_msk_value, sensory, last_mask, obj_memory | mem_readout (no memory attention, first-frame readout) |
| `read` | query_key, query_selection, pix_feat, sensory, last_mask, last_pix_feat, last_msk_value, mem_key, mem_shrinkage, mem_msk_value, mem_valid, obj_memory | mem_readout (memory attention readout over a fixed-T ring buffer) |
| `decoder` | f16 / f8 / f4 / f2 / f1, mem_readout, sensory | new_sensory, alpha matte [1, 1, 432, 768] |

## Setup

1. Build the mlpackages from [`conversion_scripts/convert_matanyone.py`](../../conversion_scripts/convert_matanyone.py) (release builds will be uploaded later)
2. Drop the 5 `.mlpackage` directories into the Xcode project
3. Build and run on a physical device (iOS 17+)

## Per-Frame State Machine (Swift)

The split puts all the bookkeeping in Swift so each CoreML module is a pure function of its inputs. State carried between frames:

- `sensory` — `(1, 1, 256, h, w)`
- `last_mask` — `(1, 1, H, W)`
- `last_pix_feat` — `(1, 256, h, w)`
- `last_msk_value` — `(1, 1, 256, h, w)`
- `obj_memory` — `(1, 1, 1, 16, 257)` (streaming sum + count)
- Working memory ring buffer at fixed `T_max = 5`:
  - `mem_key` — `(1, 64, T_max·h·w)`
  - `mem_shrinkage` — `(1, 1, T_max·h·w)`
  - `mem_msk_value` — `(1, 256, T_max·h·w)`
  - `mem_valid` — `(1, T_max·h·w)` per-slot validity mask
- `current_frame`, `last_mem_frame`, `next_fifo_slot`

Per-frame loop: encoder → (read_first or read) → decoder → mask_encoder → (if memory frame) update sensory + write the FIFO slot + accumulate `obj_memory`.

## Conversion Notes

- **Network split into 5 stateless mlpackages** so the per-frame memory state machine can live in Swift while CoreML handles the heavy compute.
- **`single_object=False` but hard-coded `num_objects=1`** lets the chunk loops collapse to the fast path. `flip_aug=False`, `use_long_term=False`, `chunk_size=-1` match the official matting config.
- **`torch.prod(1 - mask, dim=1)` in `aggregate` is monkey-patched to `1 - mask`** (identity for `num_objects=1`) since `prod` isn't supported by coremltools.
- **Memory tensors are pre-flattened to rank 3** (`[1, C, T·h·w]`) to stay within Core ML's rank-5 limit.
- **Variable-length working memory is handled by adding `(1 - valid) * -6e4`** to the similarity before top-k softmax (FP16-safe `-inf` substitute).
- **Resolution fixed at 768 × 432** (mobile 16:9, divisible by 16). For other aspect ratios, re-run the converter with `--height/--width`.
- **`read` and `read_first` ship CPU-only by default**, encoder / mask_encoder / decoder run on `.cpuAndGPU`. The CPU-only restriction comes from `PixelFeatureFuser.forward` slicing `[:, i:i+chunk_size]` on the singleton `num_objects` dim and the `visual_readout[:, 0] - last_msk_value[:, 0]` diff in the read wrapper — both trip Metal Performance Shaders with `subRange.start = -1 vs length 1` on iOS GPU. The converter has been patched to bypass both (single-object fast path in `PixelFeatureFuser.forward`, `.squeeze(1)` instead of `[:, 0]`), but you have to re-run `convert_matanyone.py` to rebuild the read / read_first mlpackages and then flip the corresponding `cpuCfg` lines to `gpuCfg` in `MatAnyoneEngine.init`. The old mlpackages have the singleton-dim slices baked into the graph and will crash on `.cpuAndGPU`.
- **No warmup loop**. The official `process_video` runs the first frame 10 times with `first_frame_pred=true` to stabilise memory, but at FP16 those repeated feedback iterations drift the alpha to zero. The single seed step + one `first_frame_pred` pass already matches the Python reference within MAE < 2e-4.

## First-Frame Mask Bootstrap (Vision)

The Vision soft mask is binarised at 0.5 and dilated by radius 8 (separable max filter) before being handed to MatAnyone. The model can shrink the alpha but cannot recover from a mask that misses parts of the foreground, so the seed is intentionally generous.

## Conversion Script

[`conversion_scripts/convert_matanyone.py`](../../conversion_scripts/convert_matanyone.py)
