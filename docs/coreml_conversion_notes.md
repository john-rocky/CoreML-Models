# CoreML Conversion Notes

Lessons learned from converting PyTorch models to CoreML for on-device iOS inference.

---

## coremltools Version

**coremltools 8.x has a critical MPS tile bug for transformer models.**

When saving models that contain `expand(-1, ...)` patterns (attention masks, positional embeddings), coremltools 8.x crashes with:

```
'mps.tile' op negative `multiplier` value -1 at index 1
```

The MPS Graph backend doesn't handle -1 as "keep dimension" in tile/expand ops.

**Fix:** Use coremltools >= 9.0. If stuck on 8.x, try ONNX as an intermediary (`PyTorch → ONNX → CoreML`).

---

## ANE (Neural Engine) Memory Limits

**Vision transformers with 768x768+ input crash the ANE.**

The error looks like:

```
Failed to add operation to E5 stream.
E5RT: No memory object bound to port. (2)
```

ANE (E5) pre-allocates fixed-size buffers for intermediate computations. DaViT's dual attention (spatial + channel) on 768x768 creates tensors that exceed these limits.

**Fix:** Use `.cpuOnly` compute units for large vision transformers. CPU inference is still practical — Florence-2 runs in 2-3s on iPhone.

```swift
let config = MLModelConfiguration()
config.computeUnits = .cpuOnly
```

Smaller models (512x512 DDColor, etc.) work fine with `.all` or `.cpuAndNeuralEngine`.

---

## Multi-Model Memory Management

**Loading 2+ CoreML models simultaneously causes OOM on iPhone.**

Each `MLModel` allocates weight memory + computation buffers at load time. Two 100MB models can consume 500MB+ total. `RangeDim` inputs also cause pre-allocation for the max size.

**Fix:** Sequential load/unload — only one model in memory at a time:

```swift
let result: MLMultiArray = try await {
    let model = try MLModel(contentsOf: url, configuration: config)
    let output = try await model.prediction(from: input)
    // Copy data before model goes out of scope
    return try copyMultiArray(output.featureValue(for: "key")!.multiArrayValue!)
    // model released here
}()
// Now safe to load the next model
```

Key: copy output data with `memcpy` before releasing the model, since `MLMultiArray` from predictions may reference model-internal buffers.

---

## INT8 Quantization Quality

**INT8 preserves quality for short-generation tasks.**

Using `coremltools.optimize.coreml.linear_quantize_weights` with `linear_symmetric` mode:

| Output length | Quality |
|---|---|
| Short (~10 tokens, CAPTION) | 100% match with FP32 |
| Medium (~50 tokens, DETAILED_CAPTION) | First ~15 tokens match, then diverge |
| Long (~100+ tokens) | Diverges earlier, but semantically correct |

Model size reduction: ~1/4 of FP32 (e.g. 360MB → 88MB).

The divergence is inherent to autoregressive generation — a single different logit cascades into a completely different continuation. The initial context is always preserved, so semantic meaning is maintained.

**Recommendation:** INT8 for generative models (Florence-2, RMBG). **FP16 required for contrastive/embedding models** (SigLIP, CLIP, AdaFace) — INT8 drops cosine similarity from 0.999 to 0.86, making similarity-based scoring unreliable.

---

## Seq2Seq Model Splitting for CoreML

**Split encoder-decoder models into 3 separate CoreML models, avoid KV cache.**

For BART-style seq2seq (Florence-2, etc.):

1. **VisionEncoder** — image → features
2. **TextEncoder** — features + input_ids → encoder_hidden_states
3. **Decoder** — decoder_input_ids + encoder_hidden_states → logits (no KV cache)

### Why not KV cache?

KV-cache requires Prefill + Step = 2 decoder models with duplicated weights. Issues encountered:

- Double memory for decoder weights
- coremltools 8.x MPS tile bug on Step model (fixed in 9.0, but adds complexity)
- 24 KV cache tensors to manage in Swift (6 layers × 4 per layer)
- `RangeDim` for growing self-attention cache dimensions

### No-cache tradeoff

The no-cache decoder re-runs the full sequence each step (O(n²) vs O(n) with cache). For short outputs (~30 tokens), this is negligible. Florence-2 runs in 2-3s total on iPhone without KV cache.

**Rule of thumb:** Start without KV cache. Only add it if generation exceeds ~100 tokens and speed is unacceptable.

### Variable-length dimensions

Encoder output length varies by task prompt (577 image tokens + N text tokens). Use `RangeDim` for both decoder_input_ids and encoder_hidden_states:

```python
enc_seq_dim = ct.RangeDim(lower_bound=580, upper_bound=600, default=585)
dec_seq_dim = ct.RangeDim(lower_bound=1, upper_bound=512, default=1)
```

---

## ImageNet Normalization in Vision Models

**Bake mean/std normalization into the model — CoreML ImageType can't do per-channel std.**

CoreML `ImageType(scale=1/255)` converts pixels to 0-1 range, but cannot apply per-channel normalization `(x - mean) / std` because it only supports a single `scale` value (not per-channel).

Without normalization: model produces wrong results (e.g. "brown" instead of "gray").

**Fix:** Add normalization as the first operation in the PyTorch wrapper:

```python
class VisionEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        # ... other layers

    def forward(self, pixel_values):
        pixel_values = (pixel_values - self.mean) / self.std
        # ... rest of forward
```

Then use `ImageType(scale=1/255)` which feeds 0-1 range into the model's built-in normalization.

---

## Model-Specific Preprocessing Gotchas

**Always check the original model's exact preprocessing — don't assume ImageNet.**

| Model | Normalization | Common Mistake |
|-------|--------------|----------------|
| Florence-2 (DaViT) | ImageNet mean/std | Forgetting to bake it in (ImageType can't do per-channel std) |
| SigLIP | mean=0.5, std=0.5 | Using ImageNet instead |
| RMBG-1.4 | mean=0.5, std=1.0 | Using ImageNet instead |

**RMBG-1.4 also requires min-max normalization after sigmoid.** The raw sigmoid output is in a narrow range like [0.5, 0.73]. The official post-processing stretches it:
```
mi, ma = output.min(), output.max()
output = (output - mi) / (ma - mi)
```
Without this, the mask has almost no contrast and everything looks like foreground.

---

## Unsupported Operations in coremltools

Some PyTorch operations are not supported by coremltools and cause conversion failures:

| Op | Example | Workaround |
|----|---------|------------|
| `\|` / `\|=` (bool OR) | GroundingDINO mask generation | Use float arithmetic: `(a.float() + b.float()).clamp(0,1)` |
| `~` (bitwise NOT) | Attention mask inversion | Pre-invert masks, pass as float input |
| `torch.eye` (dynamic size) | Attention mask init | Pre-compute outside model |
| In-place tensor assign | `tensor[..., :n] = val` | Rewrite with `torch.where` or scatter |
| `torch.nonzero` | Data-dependent indexing | Pre-compute outside model |
| `torchvision::deform_conv2d` | BiRefNet, deformable attention | No workaround — use model variant without it |
| `torch.prod` | MatAnyone `aggregate(prob, dim=1)` | For a singleton dim, replace with identity. General case: lower to a sequence of multiplications. |
| `pack_padded_sequence` / `pad_packed_sequence` | Kokoro Predictor LSTMs | Run the LSTM on the unpadded tensor directly via `RangeDim`; the model never sees padding tokens. |
| `tensor.float() % 1` (`mod` on float / scalar) | Kokoro iSTFTNet `SineGen` | CoreML's `mod` silently produces wrong values. Replace with `x - torch.floor(x)`. Spec corr 0.67 → 0.996. |
| `torch.randn_like` etc. | Kokoro `SourceModuleHnNSF`, any "regularisation" noise | Replace with `zeros_like` before tracing — otherwise the converted model is non-deterministic and parity tests are meaningless. |
| `torch.roll` | Swin Transformer shifted windows | Replace with `slice + concat`. |
| In-place attention-mask construction | Swin Transformer | Build the mask functionally (e.g. via `torch.where` / arange comparisons); avoid `mask[a:b, c:d] = val`. |
| `nn.utils.weight_norm` | Stable Audio Snake activations, etc. | Call `remove_weight_norm()` on the module before tracing. coremltools sees a much cleaner graph. |

---

## General Conversion Workflow

1. **Load model, inspect architecture** — understand which forward methods are actually used
2. **Create wrapper modules** — isolate sub-components, remove unused layers (e.g. `del model.norms`)
3. **Test wrapper output matches original** — verify with `(wrapper_out - original_out).abs().max()`
4. **`torch.jit.trace`** — watch for data-dependent control flow that trace can't capture
5. **`ct.convert` with FP16** — start here, not FP32
6. **INT8 quantize** — `linear_quantize_weights` for production
7. **Verify end-to-end** — compare CoreML output against PyTorch reference
8. **Test on device** — ANE issues and memory issues only surface on real hardware

---

## Anomaly Detection (EfficientAD) Conversion

**Multi-network models need a single wrapper for clean CoreML export.**

EfficientAD uses 3 separate networks (teacher, student, autoencoder) that must run together at inference time. The anomaly map is computed from the *difference* between their outputs, not from any single network.

**Architecture:**

```
Input (256x256 RGB)
  ├── Teacher (PDN-Small, frozen) → [1, 384, 56, 56]
  ├── Student (PDN-Small)         → [1, 768, 56, 56]
  └── Autoencoder                 → [1, 384, 56, 56]

map_st = mean((teacher - student[:,:384])², dim=channel)  → [1,1,56,56]
map_ae = mean((autoencoder - student[:,384:])², dim=channel) → [1,1,56,56]
anomaly_map = 0.5 * normalize(map_st) + 0.5 * normalize(map_ae)
→ upsample to [1,1,256,256]
```

**Key lessons:**

1. **Wrap all 3 networks + postprocessing into one `nn.Module`** — don't export 3 separate CoreML models. The anomaly map computation (MSE, quantile normalization, combination, upsample) should all be inside the wrapper so the CoreML model is self-contained.

2. **Pretrained weight completeness varies** — EfficientAD needs 4 extra parameters beyond the network weights: `teacher_mean`, `teacher_std` (channel-wise normalization), `q_st_start/end`, `q_ae_start/end` (quantile normalization). These are computed during training over the dataset. Without them, the anomaly map is uncalibrated. Always check that pretrained weights include these.

3. **Dropout layers are harmless** — the autoencoder has `Dropout(p=0.2)` layers. In `model.eval()` mode these become identity ops. `torch.jit.trace` correctly handles this; no need to remove them.

4. **Clamp output to [0, 1]** — raw anomaly maps can go negative (normal regions) or exceed 1 (severe anomalies) after quantile normalization. Clamping gives a clean probability-like output for downstream use.

5. **PatchCore is not suitable for CoreML** — it requires a nearest-neighbor search against a memory bank at inference time, which is not a standard neural network operation. EfficientAD is pure feed-forward CNN, making it directly convertible.

---

## Music Transcription (Basic Pitch) — iOS Pitfalls

Two non-obvious issues bit me when porting Spotify's Basic Pitch from Python to a Swift iOS app. The model conversion itself was trivial (Spotify ships an `.mlpackage` in the pip package), but reproducing Python's results on-device required fixing both of these.

### 1. Neural Engine pads MLMultiArray rows for alignment

The Basic Pitch model outputs three tensors of shape `(1, 172, 88)` and `(1, 172, 264)`. On iOS with `.all` compute units the Neural Engine returns these with **non-contiguous strides**:

```
Note shape: [1, 172, 88], strides: [16512, 96, 1]
Contour shape: [1, 172, 264], strides: [46784, 272, 1]
```

The row stride is **96 instead of 88** (and 272 instead of 264). The ANE pads each row to a multiple of 8 (or some hardware-specific alignment) for SIMD access. If you read `dataPointer` linearly assuming row stride = `cols`, you get garbage interleaved with real activations — the symptom in our case was that note detection produced a perfect 8-semitone pattern (D-F#-A#-D...) regardless of the input audio.

**Fix:** always use `array.strides` to compute the correct offset, never assume the data is C-contiguous:

```swift
let strides = array.strides.map { $0.intValue }
let s1 = strides[1]  // row stride (96, not 88)
let s2 = strides[2]  // 1
for r in 0..<rows {
    for c in 0..<cols {
        row[c] = ptr[r * s1 + c * s2]
    }
}
```

This is harmless on CPU/GPU (where strides usually match cols) but essential when ANE is involved.

### 2. iOS Core Audio MP3 decoder is hotter than librosa

Python `basic_pitch` loads audio with `librosa.load(path, sr=22050, mono=True)`, which goes through audioread → ffmpeg/soundfile and produces normalized samples in `[-0.978, 0.984]` for our test track (RMS 0.210). Loading the same MP3 through `AVAudioFile` + `AVAudioConverter` on iOS produces samples in `[-0.999, 1.0004]` with **RMS 0.225 — about 7% hotter** — and some peaks slightly exceed ±1.0.

This 7% gain difference is enough to push activations across the basic_pitch detection thresholds (0.5 onset, 0.3 frame), so the same MP3 yields different MIDI on Python vs Swift even with byte-identical algorithms downstream. The symptom was the Swift app missing every other melody note.

**Fix:** peak-normalize after loading, before windowing:

```swift
var peak: Float = 0
vDSP_maxmgv(samples, 1, &peak, vDSP_Length(samples.count))
if peak > 0 {
    var scale: Float = 0.98 / peak
    vDSP_vsmul(samples, 1, &scale, &samples, 1, vDSP_Length(samples.count))
}
```

After this fix, loading `Morning.mp3` directly into the iOS app gave a melody timeline matching Python's reference. WAV files (which both decoders pass through unchanged) were unaffected.

### 3. Algorithmic gotchas when porting `note_creation.py`

Even with correct audio and correct MLMultiArray reads, the post-processing port had several off-by-ones that quietly halved the detected note count:

- The greedy tracker uses `i -= k` (where `k` is the trailing below-threshold count from the Python loop), **not** `i -= energy_tol`. The two only agree when the loop exited via the `k >= energy_tol` condition; for notes that hit the audio boundary the difference is significant.
- The boundary check is `i < n_frames - 1`, not `i < n_frames`.
- `min_note_len` uses `<=` for skip (notes of length exactly `min_note_len` are rejected), not `<`.
- The melodia trick must zero out `remaining_energy[i, freq_idx]` and the `freq_idx ± 1` neighbors **inside** the forward/backward walks, not just at the end. Python's outer `while np.max > frame_thresh` loop relies on this in-place erasure to converge — without it the outer loop never makes progress and you end up needing an iteration cap.

After porting these exactly, the Swift algorithm matches Python `output_to_notes_polyphonic` note-for-note when given identical model output.

---

## Tensor Rank Limits

**Core ML supports tensor rank ≤ 5.** Anything higher fails at conversion time with an error like:

```
ValueError: Core ML only supports tensors with rank <= 5.
Layer "cast_X", with type "cast", outputs a rank 6 tensor.
```

This typically bites video / multi-object models that carry both an "objects" and a "time" dimension on top of `(B, C, H, W)`.

**Fix:** flatten one of the singleton dims at the IO boundary. For MatAnyone the working memory is stored as `(1, num_obj, C, T, h, w)` in PyTorch (rank 6). With `num_obj=1` hard-coded we can collapse it to `(1, C, T·h·w)` (rank 3) before passing it as a Core ML input, and reshape back inside the wrapper if needed.

The general rule: pre-flatten any singleton dim and reshape back inside the model.

---

## FP16 Attention Overflow

**Several attention architectures overflow when run at FP16, even on Apple Silicon GPU.**

Symptoms:
- **Swin Transformer (SinSR Denoiser)** — pinkish / wrong colour cast on the entire output.
- **Stable Audio DiT** — NaNs in the velocity prediction.
- Most softmax-based attention with large logits ranges.

Diagnosis: convert with `compute_precision=ct.precision.FLOAT16`, run on the device, look at the first inference. If the output is wildly off but PyTorch is fine, FP16 overflow is the most likely culprit.

Fixes (in order of preference):

1. **Convert at FP32** with `compute_precision=ct.precision.FLOAT32` and run on `.cpuOnly`. Larger model (~2× FP16), slower than ANE / GPU, but correct.
2. **INT8 quantize** the offending block and run it on `.cpuAndGPU`. Often works because INT8 dequantizes element-wise without the giant intermediate softmax tensor in FP16.
3. **Per-block precision** — keep most of the model FP16 and override the precision for the specific submodule. Rarely worth the complexity.

Don't try to "scale the logits down" inside the model — by the time you're running inference the conversion is already done.

---

## FP16-Safe Masked Softmax

**`-1e9` overflows in FP16.** Anywhere you do `logits + (1 - valid_mask) * -1e9` to mask out positions before a softmax, the converted model will turn the masked positions into `-inf` and then `nan` after the softmax.

Use `-6e4` (or even `-3e4`) instead. It's still smaller than any realistic similarity value but stays representable in FP16.

```python
masked = logits + (1.0 - valid) * -6.0e4
attn = masked.softmax(dim=-1)
```

This applies to MatAnyone's working-memory attention, but the same trick is needed for any masked attention you convert.

---

## FP16 Feedback Loops Drift

**Don't iterate the same input through a recurrent / feedback model more times than the Python reference does.**

MatAnyone's official `process_video` runs the first frame 10 times with `first_frame_pred=True` to refine the memory. In the FP16 CoreML port, those 10 repeated passes drift the alpha matte to zero — the foreground disappears entirely.

The PyTorch reference is FP32, which has enough headroom to absorb the iterative drift. A FP16 CoreML graph does not. **Replicate the official iteration count only when you've verified that each iteration is bit-stable in FP16.** Otherwise stick to the minimum number of passes (1 seed + 1 `first_frame_pred` is usually enough; the Python parity test should confirm).

---

## Swin Transformer Conversion Checklist

Several models in this repo (SinSR, etc.) wrap a Swin Transformer that needs the same set of patches before it traces cleanly:

1. **Pre-compute the relative-position bias as a `register_buffer`**, instead of building it from `arange` + `meshgrid` inside `forward`.
2. **Replace `torch.roll`** with `slice + concat` along H and W (the shifted-window mechanism).
3. **Rewrite the attention mask construction** to avoid `__setitem__` (no `mask[a:b, c:d] = val`). Build it functionally with `torch.where` / boolean arithmetic.
4. **Patch the coremltools `int` op converter** to handle multi-dim tensor shape casts. The default converter only handles scalar shapes; Swin's window partitioning produces a 2-D shape that needs the patched op.

A reference implementation is in [`conversion_scripts/convert_sinsr.py`](../conversion_scripts/convert_sinsr.py).

---

## Apple `ml-stable-diffusion` (`torch2coreml`) Gotchas

When using Apple's `ml-stable-diffusion` toolchain (Hyper-SD, SD 1.5, etc.):

- **Quantize after chunking, not before.** `torch2coreml` palettizes the unchunked model; once chunks are emitted by `chunk_mlprogram.py`, each chunk has to be re-palettized separately.
- **6-bit kmeans palettization breaks on weights containing `inf`.** This is most common in CLIP-style text encoders. Quantize the UNet only and ship the text encoder at FP16.
- **coremltools 9.0 patches** required for the toolchain itself:
  - Custom `int` op converter for multi-dim tensor shape casts.
  - `list(block.operations)` workaround in `chunk_mlprogram.py` for the new `CacheDoublyLinkedList` API.

---

## NaN Sanitisation Around INT8 Models

Some INT8-quantized text/vision encoders (notably Stable Audio's T5) intermittently emit `NaN` values for specific token positions. The downstream model then poisons everything from that point on.

**Mitigation:** sanitise on the Swift side before passing the embeddings to the next model.

```swift
for i in 0..<count {
    let v = embeddings[i].floatValue
    embeddings[i] = NSNumber(value: v.isNaN ? 0 : v)
}
```

It's not a fix for the underlying issue (FP32 conversion of the same encoder is clean), but it keeps the pipeline running on devices where the INT8 path is the only one that fits in memory.

---

## DINOv2 Backbone Conversion Checklist

DINOv2 ViT-S/B/L backbones (MoGe-2, etc.) are otherwise straightforward to convert, but the official implementation has two pieces of dynamic logic that you have to neutralise before tracing.

1. **Freeze the interpolated positional embedding.** `interpolate_pos_encoding` does a **bicubic + antialias** resize of the pretrained pos_embed every forward call. coremltools cannot trace bicubic + antialias cleanly. Since the result depends only on the input (h, w), and you are converting at a fixed resolution anyway, pre-compute the interpolated pos_embed once at conversion time and replace the method with a constant lookup:

   ```python
   def freeze_pos_embed(model, base_h, base_w):
       backbone = model.encoder.backbone
       img_h, img_w = base_h * backbone.patch_size, base_w * backbone.patch_size
       dummy = torch.zeros(1, 3, img_h, img_w)
       tokens = backbone.patch_embed(dummy)
       cls = backbone.cls_token.expand(1, -1, -1)
       x = torch.cat([cls, tokens], dim=1)
       with torch.no_grad():
           pos = backbone.interpolate_pos_encoding(x, img_h, img_w)
       backbone.register_buffer("_frozen_pos_embed", pos.detach().clone(), persistent=False)
       def _frozen_interp(self, x, h, w):
           return self._frozen_pos_embed
       backbone.interpolate_pos_encoding = _frozen_interp.__get__(backbone, type(backbone))
   ```

2. **Set `onnx_compatible_mode = True`** on the encoder. This disables the antialias path inside the rest of the encoder forward (not just `interpolate_pos_encoding`) so the only `F.interpolate` calls left are bilinear with explicit `size`.

3. **Apply the same `int` op patch as Swin / SinSR.** DINOv2's positional indexing emits int casts on a 2-element shape tensor that the stock coremltools converter assumes are scalars.

4. **Call `model.enable_pytorch_native_sdpa()`** so attention lowers to `F.scaled_dot_product_attention`, which coremltools handles natively. The custom `MemEffAttention` path otherwise emits an op pattern coremltools cannot fold cleanly.

A reference implementation is in [`conversion_scripts/convert_moge2.py`](../conversion_scripts/convert_moge2.py).

---

## MPS Slice on Singleton Dimension Crashes

**Some Core ML graphs crash on iOS GPU with:**

```
[MPSNDArrayDescriptor sliceDimension:withSubrange:] error:
subRange.start (18446744073709551615) is not less than length of dimension[1] (1)
```

`18446744073709551615` is `UINT64_MAX` — the result of casting `-1` to unsigned. It happens when an op slices a length-1 dimension with a negative index, and Metal Performance Shaders does not handle that case.

This bites models with reshapes/slices over a singleton "num_objects" or "T" dimension (MatAnyone's `read_first` / `read` modules hit it because the QueryTransformer reshapes obj memory across the size-1 num_objects dim).

**Fix:** load the offending model with `.cpuOnly`. The CPU path handles the slice correctly. The "real" fix is to drop the singleton dimension at the wrapper level before tracing, but that's a bigger refactor.

---

## coremltools 9.0 `_cast` with (1,1,...,1) numpy arrays

coremltools 9.0 `frontend/torch/ops.py::_cast` calls `dtype(x.val)` to fold a torch `aten::Int` / `aten::Bool` into a MIL constant. When traced with torch 2.11, shape-derived scalar values sometimes arrive as `np.ndarray` with shape `(1,)` or `(1,1)` rather than a true 0-d scalar, producing:

```
TypeError: only 0-dimensional arrays can be converted to Python scalars
```

**Fix:** monkey-patch `_cast` to call `.item()` on numpy arrays before `dtype(...)`:

```python
import coremltools.converters.mil.frontend.torch.ops as _ops
from coremltools.converters.mil import Builder as mb

def _cast(context, node, dtype, dtype_name):
    x = _ops._get_inputs(context, node, expected=1)[0]
    # ... unchanged guard on shape ...
    if x.can_be_folded_to_const() and not isinstance(x.val, dtype):
        val = x.val.item() if hasattr(x.val, "item") else x.val
        res = mb.const(val=dtype(val), name=node.name)
    # ... rest unchanged ...

_ops._cast = _cast
```

Seen while converting DC-AE (Sana VAE), E-MMDiT, and Llama 3.2 under torch 2.11 + coremltools 9.0.

---

## `torch.Tensor.movedim` unsupported by coremltools 9.0

coremltools 9.0 has no frontend op for `movedim`. Diffusers' DC-AE and Sana attention call `hidden.movedim(1, -1)` / `movedim(-1, 1)` on 4-D tensors. Equivalent permutes trace cleanly:

```python
_orig = torch.Tensor.movedim
def movedim(self, src, dst):
    if self.dim() == 4 and src == 1 and dst == -1:
        return self.permute(0, 2, 3, 1)
    if self.dim() == 4 and src == -1 and dst == 1:
        return self.permute(0, 3, 1, 2)
    return _orig(self, src, dst)
torch.Tensor.movedim = movedim
```

---

## DC-AE (Sana VAE) FP16 linear-attention overflow

The DC-AE `SanaMultiscaleLinearAttention` uses a linear-attention normalization `hidden / (hidden_sum + eps)`. FP16 converges to NaN / saturation, producing grossly smeared output images (parity diff ~2 against an output range of ~±1.3).

**Options:**
1. Convert the decoder FP32 (~600 MB for f32c32) and accept the size.
2. Mixed precision with `ct.transform.FP16ComputePrecision(op_selector=...)` to keep only the linear-attention block in FP32. Not yet validated.
3. Replace the processor with Sana's quadratic-attention branch (`use_linear_attention=False`) at decoder build-time, but that changes output slightly.

FP32 is what Nitro-E ships for now.

Additional DC-AE decoder monkey-patches required for trace: drop `output_size=` from the `repeat_interleave(..., dim=1)` call in `Decoder.forward` (it emits an `aten::Int` on a multi-dim tensor) and statically unpack `hidden_states.size()` in `SanaMultiscaleAttnProcessor2_0.__call__` (its `list(hidden_states.size())` unpack produces dynamic int casts that coremltools cannot fold).

---

## Llama for text-encoding requires transformers == 4.49.x

transformers 5.x rewrote `create_causal_mask` to index into `q_length.shape` / `q_length[0]` — shape assumptions that fail under `torch.jit.trace` (`IndexError: tuple index out of range`). Downgrade to transformers 4.49.0 (the version AMD Nitro-E pins) to convert Llama 3.2 1B cleanly. Keep the wrapper limited to `model.model` (drop the LM head) and return `last_hidden_state` for seq_len=128.

---
