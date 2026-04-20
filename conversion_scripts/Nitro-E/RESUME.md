# Nitro-E conversion — resume notes

Blocked on: `meta-llama/Llama-3.2-1B` HF gated access (request submitted 2026-04-20).

## What's already done
- Vendored upstream into `core/` (patched `cuda`-hardcoding in `emmdit_pipeline.py` / `inference_pipe.py`).
- `test_transformer_only.py`: E-MMDiT 302.4M params, weights load (missing=0, unexpected=0), dummy forward 0.15s/step (CPU FP32).
- `test_vae_only.py`: DC-AE decoder 159M, 6.3s/decode, scaling_factor=0.41407.
- `run_pytorch_reference.py`: end-to-end reference entrypoint (needs Llama access).

## Next steps once Llama access is granted

```bash
cd /Users/majimadaisuke/Downloads/CoreML-Models/conversion_scripts/Nitro-E

# 1. End-to-end reference on CPU (FP32 is safest; MPS+bf16 known-flaky for DiT)
python3 run_pytorch_reference.py --device cpu --out reference_out.png

# 2. Augment run_pytorch_reference.py to dump:
#    - prompt_embeds [1,128,2048]
#    - initial latent [1,32,16,16]
#    - per-step latents
#    - final decoded image tensor
#    as .pt files for parity checks during CoreML conversion.

# 3. Write the three conversion scripts:
#    - convert_nitro_e_text_encoder.py   (Llama 3.2 1B, max_seq=128, INT4 block-wise quant)
#    - convert_nitro_e_vae_decoder.py    (DC-AE decoder, 16x16x32 -> 512x512x3, FP16)
#    - convert_nitro_e_emmdit.py         (E-MMDiT, 1 denoise step, FP16 first; FP32 cpuOnly fallback)
```

## Known conversion risks
- `einops.rearrange` with ASA pattern `'b (l s n) c -> (b s) (l n) c'` — must rewrite with reshape+permute for CoreML tracing.
- `JointAttnProcessor2_0` uses SDPA → may need `qk_norm='rms_norm'` hand-rewrite.
- Llama 3.2 1B FP16 is ~2.5GB; need INT4 block-wise quantization to fit iPhone budget (target ~0.62GB).
- FlowMatchEulerDiscreteScheduler must be ported to Swift (simple: sigma = t / num_train_timesteps, x_{t-1} = x_t + (sigma_{t-1} - sigma_t) * v_pred).

## Files modified vs upstream
- `core/tools/inference_pipe.py`: whitespace fix (tab→spaces in else branch).
- `core/models/emmdit_pipeline.py`: removed two `device = torch.device('cuda')` hardcodings → `self._execution_device`.
