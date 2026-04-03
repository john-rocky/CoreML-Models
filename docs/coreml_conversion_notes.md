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
