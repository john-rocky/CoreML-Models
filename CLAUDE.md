# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This is a model zoo of PyTorch → Core ML conversions plus matching iOS sample apps. Each model has two halves that should usually be edited together:

- A Python conversion script in `conversion_scripts/convert_*.py` that produces one or more `.mlpackage` directories.
- A SwiftUI demo app in `sample_apps/<Name>Demo/` (or `creative_apps/` for unreleased / experimental ones) that consumes those mlpackages.

The README is the public model index — the long markdown lists were intentionally moved out of per-model docs into the README, with conversion notes living in `docs/coreml_conversion_notes.md` and per-app READMEs.

## Repo Layout

- `conversion_scripts/` — one `convert_<model>.py` per model. Most scripts vendor the upstream repo as a sibling subdirectory (e.g. `conversion_scripts/MatAnyone/`) and `sys.path.insert` it before importing. They typically build a small `nn.Module` wrapper, `torch.jit.trace` it, and call `coremltools.convert`.
- `sample_apps/` — published SwiftUI demos (one Xcode project per model). Each app's `.mlpackage`s live inside the app's source directory so they get bundled at build time.
- `creative_apps/` and `creative_models/` — experimental / unreleased apps and the mlpackages they depend on.
- `converted_models/` — standalone mlpackages without a dedicated demo app.
- `docs/coreml_conversion_notes.md` — **READ THIS BEFORE STARTING ANY NEW CONVERSION.** It is the institutional memory for every non-obvious bug we have hit (FP16 overflow, ANE memory limits, MPS slice on singleton dim, rank-5 limit, ImageNet normalization, Swin patches, INT8 NaN sanitisation, etc.). New conversion lessons should be appended here.
- `README.md` — the public-facing model zoo index. Keep per-model entries terse; long conversion notes go into the per-app README and `docs/coreml_conversion_notes.md`, not here.

## Common Commands

There is no project-level build system. Each conversion script is standalone Python; each iOS app is a standalone Xcode project.

```bash
# Run a conversion (from repo root). Most scripts take --help.
python conversion_scripts/convert_<model>.py --help

# Open an iOS sample app
open sample_apps/<Name>Demo/<Name>Demo.xcodeproj
```

Per CLAUDE global rules: **do not run iOS builds locally — building/testing happens on the user's device.** Do not commit `.mlpackage` files, `xcuserdata/`, or other build artifacts.

## Architecture Patterns You Will See Repeatedly

These patterns recur across nearly every conversion script and sample app — knowing them up front saves hours.

### Conversion script structure
1. Vendor the upstream repo, monkey-patch any ops coremltools cannot trace (`torch.prod`, `torch.roll`, in-place mask assignment, `nn.utils.weight_norm`, etc.) — see the table in `docs/coreml_conversion_notes.md` under "Unsupported Operations in coremltools".
2. Build a small `nn.Module` wrapper that exposes the *minimum* sub-graph the app actually needs, and bake any preprocessing (e.g. ImageNet mean/std normalization) into the wrapper because `ImageType(scale=...)` cannot do per-channel normalization.
3. `torch.jit.trace` → `coremltools.convert` with `compute_precision=ct.precision.FLOAT16` as the default. Drop to FP32 only when FP16 attention overflow is confirmed (Swin/SinSR, Stable Audio DiT).
4. Verify parity against the PyTorch reference (`(coreml_out - torch_out).abs().max()` is the bare minimum) before declaring victory.

### Multi-model state machines live in Swift, mlpackages stay stateless
For seq2seq, video, and feedback-loop models we deliberately split the network into several stateless mlpackages and keep all bookkeeping in Swift. Examples:

- **Florence-2**: VisionEncoder + TextEncoder + Decoder, no KV cache (seq2seq). The decoder re-runs the full sequence each step because adding KV cache doubles weight memory and complicates the Swift driver — only worth it past ~100 output tokens.
- **MatAnyone**: encoder + mask_encoder + read_first + read + decoder. The per-frame ring buffer (`mem_key`, `mem_shrinkage`, `mem_msk_value`, `mem_valid`, `sensory`, `obj_memory`, etc.) is owned by Swift; each mlpackage is a pure function of its inputs. See `sample_apps/MatAnyoneDemo/README.md` for the exact tensor shapes.
- **Hyper-SD**: text encoder + chunked UNet (2 mlpackages from Apple's `chunk_mlprogram`) + VAE decoder, driven by a Swift TCD scheduler.

When extending or debugging one of these pipelines, treat the Swift driver as the source of truth for shapes and ordering — the mlpackages are intentionally dumb.

### Multi-model memory management on iPhone
Loading two 100MB CoreML models simultaneously can OOM on real devices. The pattern is **sequential load → predict → copy output → release** (see `docs/coreml_conversion_notes.md` "Multi-Model Memory Management"). Always `memcpy` MLMultiArray outputs out of model-internal buffers before the model goes out of scope.

### Compute unit selection is per-model and load-bearing
- Vision transformers ≥ 768×768 must use `.cpuOnly` (ANE E5 buffer limits — Florence-2, etc.).
- FP16 Swin / DiT attention overflows on GPU/ANE → either FP32 + `.cpuOnly` or INT8 + `.cpuAndGPU`.
- Some Core ML graphs that slice a singleton dimension crash on iOS GPU with `subRange.start = -1 vs length 1` (MPS slice bug). MatAnyone's `read` / `read_first` ship `.cpuOnly` for this reason. Do not flip them to GPU without re-running the converter to drop the singleton dim.
- Most other models default to `.cpuAndNeuralEngine` or `.all`.

### Strides matter when reading MLMultiArray on ANE
ANE pads rows for SIMD alignment, so `MLMultiArray.dataPointer` is **not** C-contiguous on `.all` / `.cpuAndNeuralEngine`. Always read via `array.strides`. Basic Pitch's broken note detection (`docs/coreml_conversion_notes.md` "Music Transcription") is the canonical example.

### Model-specific preprocessing is never ImageNet-by-default
Florence-2 uses ImageNet, **SigLIP uses (0.5, 0.5)**, **RMBG uses (0.5, 1.0)**, and RMBG additionally requires post-sigmoid min-max stretch. Always check the original repo's preprocessing and bake it into the wrapper.

## Conventions

- **Code comments and UI strings are English only.** (Scripts and READMEs frequently reference Japanese / Chinese upstream repos, but our code stays in English.)
- **Do not commit `.mlpackage` files, build artifacts, `xcuserdata/`, or anything else that bloats the repo.** Models are distributed via GitHub Releases / Google Drive links from the README.
- **Do not include `claude` in commit messages or as committer.** Do not add Claude co-author trailers.
- **Build testing is done on the user's physical device** — do not attempt to run `xcodebuild` or simulator builds as part of a task.
- When adding a new conversion lesson, append it to `docs/coreml_conversion_notes.md` rather than scattering it across per-app READMEs.
