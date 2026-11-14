# YOLOEDemo

Open-vocabulary detection **and** instance segmentation on iOS using [YOLOE](https://github.com/THU-MIG/yoloe) + MobileCLIP.

Type any text — "person", "forklift", "coffee cup" — and detect/segment it in real-time camera, photos, or videos. No fixed class list.

## Architecture — region embeddings, similarity in Swift

Unlike YOLOWorldDemo (which bakes the text prompt into the class head and outputs
finished `scores`), YOLOEDemo keeps the open-vocabulary property fully decoupled:
the detector emits the **per-anchor region embedding** *before* the class logits,
and the region–text similarity is computed in Swift against cached text
embeddings. The image branch never sees the text, so **changing the query never
re-runs the detector** — only a cheap matmul.

```
Text query ─→ MobileCLIP text encoder ─→ reprta (RepRTA MLP) ─→ L2-norm, append 1.0
                                                                      │
                                                            text'  [N, 513]   (cached in Swift)
                                                                      │  similarity (BLAS)
Camera/Image ─→ YOLOE detector ─→ region_embeddings [1, 513, 8400] ───┤  logit = text' · region'
                              ├─→ boxes        [1, 4,  8400]           │  score = sigmoid(logit)
                              ├─→ mask_coeffs  [1, 32, 8400]           │
                              └─→ mask_protos  [1, 32, 160, 160]       └─→ NMS → boxes + masks
```

### Why 513 dimensions?

YOLOE's class head is a per-scale `BNContrastiveHead`:

```
logit = (BN(region) · normalize(text)) · exp(logit_scale) + bias
```

The exported detector folds the per-scale `exp(logit_scale)` into the first 512
embedding channels and appends the per-scale `bias` as channel 512. The cached
text gets a trailing constant `1.0`. The dot product then reproduces the exact
logit with a single matmul and **zero knowledge of the anchor layout** in Swift.
Verified parity vs the PyTorch head: `max |sigmoid(text'·region') − scores| ≈ 2e-11`,
and through the FP16 CoreML model the top detection's anchor/class match exactly.

## Models

| Model | Size | Description |
|-------|------|-------------|
| `yoloe_detector.mlpackage` | ~20 MB (FP16) | YOLOE-11s-seg region-embedding detector + seg |
| `reprta.mlpackage` | ~6 MB (FP32) | YOLOE RepRTA text-refinement MLP (`raw_tpe → tpe`) |
| `mobileclip_blt_text.mlpackage` | ~121 MB | Apple MobileCLIP B-LT text encoder (`text → final_emb_1`) |
| `visual_prompt_encoder.mlpackage` | ~20 MB (FP16) | YOLOE SAVPE (`image + mask[80²] → vpe`) for visual prompts |
| `mobile_sam_encoder.mlpackage` | ~38 MB (FP16) | MobileSAM image encoder (`image → image_embeddings`) for tap-to-segment |
| `mobile_sam_decoder.mlpackage` | ~8 MB (FP16) | MobileSAM mask decoder (`embeddings + prompts → masks`), runs `.cpuAndGPU` |
| `mobile_sam_prompt_encoder_weights.json` | 38 KB | MobileSAM prompt-encoder weights for the Swift `PromptEncoder` |
| `clip_vocab.json` | 1.6 MB | CLIP BPE vocabulary for the Swift tokenizer |

`mobileclip_blt_text.mlpackage` is Apple's official Core ML export from
[apple/ml-mobileclip](https://github.com/apple/ml-mobileclip) and is used as-is.

The app bundles the **S** detector. A higher-accuracy **L** variant
(`yoloe_detector_l` ~54 MB + `reprta_l`) is in the release — to use it, download
both, rename to `yoloe_detector.mlpackage` / `reprta.mlpackage`, and replace the
bundled files (Swift needs no changes: embed=512 is shared across S/M/L).

## Features

- **Camera**: real-time open-vocabulary detection + live instance masks
- **Photo**: pick from library, detect + colored instance masks, re-threshold without re-running
- **Video**: pick a video, detect + masks frame-by-frame with overlay
- **Visual**: capture or pick a reference photo, **tap an object** (auto-segmented) — or drag a box — then detect that object by example in the live camera; no text needed
- **Open-vocabulary**: up to 80 simultaneous queries, any text; switching queries is free

Live masks (camera/video) use the [yolo-ios-app](https://github.com/ultralytics/yolo-ios-app)
fast path: one BLAS matmul `coeffs[N,32] x protos[32,160²]` builds every instance mask at
proto resolution, composited into a single small RGBA image that Core Animation scales onto
a `maskLayer` — no per-pixel CGContext draw, no SwiftUI churn per frame.

**Visual prompts** (the Visual tab) use YOLOE's SAVPE module: the reference region is encoded
into a 512-d embedding in the *same* space as the text query, so `encodeVisualMask` just caches
`[vpe, 1.0]` and the per-frame detection path is unchanged. SAVPE takes an **80×80 mask** (640
input, stride 8) — not just a box — so a **tap** segments the object on the reference and feeds
that object-shaped mask straight to SAVPE; a tighter mask than a box gives a cleaner embedding
(no background). The tap is segmented by **MobileSAM** (a point prompt at the tapped pixel —
crisp and class-agnostic); if MobileSAM is unavailable it falls back to YOLOE's own seg head
(`maskFromTap`: the instance whose mask is strongest at the tapped point). Dragging a box still
works as a fallback. An **L** encoder is in the release too.

## Requirements

- iOS 16.0+, Xcode 15.0+
- Physical device (camera + Neural Engine)

## Quick Start

1. Open `YOLOEDemo.xcodeproj` in Xcode
2. Select your development team
3. Build and run on a physical device

## Re-converting Models (Optional)

```bash
pip install ultralytics coremltools clip-anytorch
python convert_models.py                # yoloe-11s-seg
python convert_models.py --size 11m-seg
```

This regenerates `yoloe_detector.mlpackage`, `reprta.mlpackage`, and
`clip_vocab.json`. The MobileCLIP text encoder is reused from Apple's export.

## Usage

1. Enter comma-separated object names (e.g. `person, dog, car`)
2. Tap the search button or press return
3. Switch Camera / Photo / Video with the bottom tabs
4. Adjust the confidence slider; in Photo mode masks re-render instantly

## License

LICENSE-AGPL-3.0.txt in this folder applies to convert_models.py, which contains code adapted from ultralytics/ultralytics. For the other files, see NOTICE at the repository root.
