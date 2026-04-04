"""
Convert stabilityai/stable-audio-open-small to CoreML (4 models).

Architecture (497M total):
  T5TextEncoder (110M):  text → embeddings [1, 64, 768]
  NumberEmbedder (0.2M): seconds_total → embedding [1, 768]
  DiT (341M):            latent + conditioning + timestep → denoised latent
  VAEDecoder (156M):     latent [1, 64, 256] → stereo audio [1, 2, 524288]

Generation flow:
  1. T5 encodes text → [1, 64, 768], NumberEmbedder encodes seconds → [1, 768]
  2. Cross-attn cond = cat(T5 [1,64,768], seconds [1,1,768]) → [1, 65, 768]
  3. DiT runs N diffusion steps (rectified flow euler) on latent [1, 64, 256]
  4. VAE decoder converts latent → stereo audio [1, 2, 524288] at 44.1kHz

Requirements:
  pip install stable-audio-tools torch coremltools>=9.0 transformers

Usage:
  python convert_stable_audio.py
"""

import torch
import torch.nn as nn
import coremltools as ct
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig,
    OptimizationConfig,
    linear_quantize_weights,
)


# ============================================================
# Wrapper modules
# ============================================================

class T5TextEncoder(nn.Module):
    """text input_ids → embeddings [1, 64, 768]."""

    def __init__(self, t5_model):
        super().__init__()
        self.t5 = t5_model

    def forward(self, input_ids):
        return self.t5(input_ids=input_ids).last_hidden_state


class NumberEmbedderWrapper(nn.Module):
    """Normalized float → embedding [1, 768]."""

    def __init__(self, embedder):
        super().__init__()
        self.embedder = embedder

    def forward(self, x):
        # x: [1] normalized seconds_total (0..1)
        return self.embedder(x)  # [1, 768]


class DiTDenoiser(nn.Module):
    """Single denoising step: latent + cond + timestep → denoised velocity."""

    def __init__(self, dit):
        super().__init__()
        self.dit = dit

    def forward(self, x, t, cross_attn_cond, global_embed):
        # x: [1, 64, 256], t: [1], cross_attn_cond: [1, 65, 768], global_embed: [1, 768]
        return self.dit._forward(
            x, t,
            cross_attn_cond=cross_attn_cond,
            global_embed=global_embed,
            use_checkpointing=False,
        )


class VAEDecoder(nn.Module):
    """latent [1, 64, 256] → stereo audio [1, 2, 524288]."""

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, latent):
        return self.decoder(latent)


# ============================================================
# Conversion
# ============================================================

def main():
    from stable_audio_tools import get_pretrained_model
    from torch.nn.utils import remove_weight_norm

    print("Loading stabilityai/stable-audio-open-small ...")
    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-small")
    model.eval()

    out_dir = "."
    quant_config = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
    )

    # --- 1. T5 Text Encoder ---
    print("\n[1/4] Converting T5TextEncoder ...")
    t5_cond = model.conditioner.conditioners["prompt"]
    t5_enc = T5TextEncoder(t5_cond.model.float()).eval()

    dummy_ids = torch.randint(0, 32128, (1, 64))
    with torch.no_grad():
        traced_t5 = torch.jit.trace(t5_enc, (dummy_ids,))

    ml_t5 = ct.convert(
        traced_t5,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, 64), dtype=int),
        ],
        outputs=[ct.TensorType(name="text_embeddings")],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
    )
    ml_t5 = linear_quantize_weights(ml_t5, quant_config)
    ml_t5.author = "CoreML-Models"
    ml_t5.short_description = (
        "Stable Audio Open Small — T5-base text encoder. "
        "Input: input_ids [1,64]. Output: text_embeddings [1,64,768]. INT8."
    )
    ml_t5.save(f"{out_dir}/StableAudioT5Encoder.mlpackage")
    print("  Saved StableAudioT5Encoder.mlpackage")

    # --- 2. NumberEmbedder ---
    print("\n[2/4] Converting NumberEmbedder ...")
    num_cond = model.conditioner.conditioners["seconds_total"]
    num_emb = NumberEmbedderWrapper(num_cond.embedder.float()).eval()

    dummy_seconds = torch.tensor([0.04])  # 10s / 256
    with torch.no_grad():
        traced_num = torch.jit.trace(num_emb, (dummy_seconds,))

    ml_num = ct.convert(
        traced_num,
        inputs=[
            ct.TensorType(name="normalized_seconds", shape=(1,)),
        ],
        outputs=[ct.TensorType(name="seconds_embedding")],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
    )
    ml_num.author = "CoreML-Models"
    ml_num.short_description = (
        "Stable Audio Open Small — NumberEmbedder for seconds_total. "
        "Input: normalized seconds (0..1). Output: embedding [1,768]."
    )
    ml_num.save(f"{out_dir}/StableAudioNumberEmbedder.mlpackage")
    print("  Saved StableAudioNumberEmbedder.mlpackage")

    # --- 3. DiT Denoiser ---
    print("\n[3/4] Converting DiT ...")
    dit = model.model.model.float().eval()
    dit_wrapper = DiTDenoiser(dit).eval()

    dummy_x = torch.randn(1, 64, 256)
    dummy_t = torch.tensor([0.5])
    dummy_cross = torch.randn(1, 65, 768)
    dummy_global = torch.randn(1, 768)

    with torch.no_grad():
        traced_dit = torch.jit.trace(
            dit_wrapper,
            (dummy_x, dummy_t, dummy_cross, dummy_global),
            check_trace=False,
        )

    ml_dit = ct.convert(
        traced_dit,
        inputs=[
            ct.TensorType(name="latent", shape=(1, 64, 256)),
            ct.TensorType(name="timestep", shape=(1,)),
            ct.TensorType(name="cross_attn_cond", shape=(1, 65, 768)),
            ct.TensorType(name="global_embed", shape=(1, 768)),
        ],
        outputs=[ct.TensorType(name="velocity")],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
    )
    ml_dit = linear_quantize_weights(ml_dit, quant_config)
    ml_dit.author = "CoreML-Models"
    ml_dit.short_description = (
        "Stable Audio Open Small — DiT denoiser (341M). "
        "Single diffusion step. Rectified flow. INT8."
    )
    ml_dit.save(f"{out_dir}/StableAudioDiT.mlpackage")
    print("  Saved StableAudioDiT.mlpackage")

    # --- 4. VAE Decoder ---
    print("\n[4/4] Converting VAEDecoder ...")
    decoder = model.pretransform.model.decoder.float().eval()
    # Remove weight_norm for clean tracing
    for m in decoder.modules():
        try:
            remove_weight_norm(m)
        except ValueError:
            pass

    vae_wrapper = VAEDecoder(decoder).eval()
    dummy_latent = torch.randn(1, 64, 256)

    with torch.no_grad():
        traced_vae = torch.jit.trace(vae_wrapper, (dummy_latent,))

    ml_vae = ct.convert(
        traced_vae,
        inputs=[
            ct.TensorType(name="latent", shape=(1, 64, 256)),
        ],
        outputs=[ct.TensorType(name="audio")],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
    )
    ml_vae.author = "CoreML-Models"
    ml_vae.short_description = (
        "Stable Audio Open Small — VAE decoder (Oobleck, 156M). "
        "Input: latent [1,64,256]. Output: stereo audio [1,2,524288] at 44.1kHz. FP16."
    )
    ml_vae.save(f"{out_dir}/StableAudioVAEDecoder.mlpackage")
    print("  Saved StableAudioVAEDecoder.mlpackage")

    # --- Optional: DiT with FP32 compute (better quality, 1.3GB, cpuOnly required) ---
    print("\n[Optional] Converting DiT with FP32 compute ...")
    ml_dit_fp32 = ct.convert(
        traced_dit,
        inputs=[
            ct.TensorType(name="latent", shape=(1, 64, 256)),
            ct.TensorType(name="timestep", shape=(1,)),
            ct.TensorType(name="cross_attn_cond", shape=(1, 65, 768)),
            ct.TensorType(name="global_embed", shape=(1, 768)),
        ],
        outputs=[ct.TensorType(name="velocity")],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32,
    )
    ml_dit_fp32.author = "CoreML-Models"
    ml_dit_fp32.short_description = (
        "Stable Audio Open Small — DiT denoiser (341M). "
        "Single diffusion step. Rectified flow. FP32 compute (cpuOnly)."
    )
    ml_dit_fp32.save(f"{out_dir}/StableAudioDiT_FP32.mlpackage")
    print("  Saved StableAudioDiT_FP32.mlpackage")

    print("\nDone! Converted models:")
    print("  - StableAudioT5Encoder.mlpackage (T5-base, INT8)")
    print("  - StableAudioNumberEmbedder.mlpackage (seconds conditioner, FP16)")
    print("  - StableAudioDiT.mlpackage (DiT, INT8 — use with cpuAndGPU)")
    print("  - StableAudioDiT_FP32.mlpackage (DiT, FP32 — use with cpuOnly, better quality)")
    print("  - StableAudioVAEDecoder.mlpackage (VAE decoder, FP16)")


if __name__ == "__main__":
    main()
