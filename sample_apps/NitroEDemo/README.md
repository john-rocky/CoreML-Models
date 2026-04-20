# NitroEDemo

SwiftUI sample that runs AMD's **Nitro-E** (304M E-MMDiT) text-to-image model on an iPhone/iPad using Core ML. It combines three mlpackages — Llama 3.2 1B text encoder, the E-MMDiT denoiser, and the DC-AE VAE decoder — with a Swift port of `FlowMatchEulerDiscreteScheduler` and a minimal Llama 3 BPE tokenizer.

![demo](./demo.png)

## Pipeline

| Stage | Model | Output |
| --- | --- | --- |
| Tokenize | `LlamaTokenizer` (byte-level BPE, reads `Llama3Vocab.json` + `Llama3Merges.txt`) | `input_ids [1,128]`, `attention_mask [1,128]` |
| Text encode | `NitroE_TextEncoder.mlpackage` (Llama 3.2 1B, last hidden state) | `[1,128,2048]` |
| Denoise ×4 | `NitroE_EMMDiT.mlpackage` (304M, 1 step, `guidance_scale=0`) | `noise_pred [1,32,16,16]` |
| Scheduler | `FlowMatchEulerScheduler` (Swift port, `num_train_timesteps=1000`, `shift=1.0`) | Updated latent |
| Decode | `NitroE_VAEDecoder.mlpackage` (DC-AE f32c32) | `image [1,3,512,512]` in `[-1,1]` |

## Precision notes

- The E-MMDiT and Llama models are FP16 with optional 8-bit / 4-bit palettization.
- **The DC-AE decoder must stay in FP32** for the linear-attention normalization step. If you convert it with `compute_precision=FLOAT16` the output is smeared. Weight-only palettization (8-bit per-grouped-channel) is safe and cuts the decoder from 608 MB to ~160 MB.
- See `docs/coreml_conversion_notes.md` in the repo root for the monkey-patches required during conversion.

## Getting the models

The mlpackages are not checked in. Convert them yourself or download the releases.

```bash
# from repo root
python3.12 conversion_scripts/convert_nitro_e_text_encoder.py
python3.12 conversion_scripts/convert_nitro_e_emmdit.py
python3.12 conversion_scripts/convert_nitro_e_vae_decoder.py --precision fp32 \
    --out conversion_scripts/NitroE_VAEDecoder_FP32.mlpackage
# optional: palettize to shrink
python3.12 conversion_scripts/palettize_nitro_e_text_encoder.py  # 2.3GB -> ~0.6GB
python3.12 conversion_scripts/palettize_nitro_e_emmdit.py        # 578MB -> ~150MB
python3.12 conversion_scripts/palettize_nitro_e_vae_decoder.py   # 608MB -> ~160MB
```

Copy the resulting `.mlpackage` directories into `sample_apps/NitroEDemo/NitroEDemo/` with the names the Xcode project expects:

- `NitroE_TextEncoder.mlpackage`
- `NitroE_EMMDiT.mlpackage`
- `NitroE_VAEDecoder.mlpackage`

Also generate the tokenizer files (already done on first export):

```bash
python3.12 conversion_scripts/export_llama_tokenizer_files.py
```

This writes `Llama3Vocab.json` and `Llama3Merges.txt` into `NitroEDemo/`.

## Requirements

- iOS 18.0+ (palettization uses `per_grouped_channel` which requires iOS 18).
- iPhone 15 Pro / iPhone 16 series or iPad with M-chip recommended — total model footprint is ~900 MB with INT4 + INT8 + INT8 palettization.
- Xcode 15 or newer.

## Credits

- AMD Nitro-E — https://huggingface.co/amd/Nitro-E
- Llama 3.2 1B — https://huggingface.co/meta-llama/Llama-3.2-1B (gated)
- DC-AE — https://huggingface.co/mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers
