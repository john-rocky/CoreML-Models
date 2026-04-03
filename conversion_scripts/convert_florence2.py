"""
Convert Microsoft Florence-2-base to CoreML (4 models).

Architecture:
  VisionEncoder: DaViT → pos_embed → temporal_embed → image_projection → LayerNorm
  TextEncoder:   Embedding(input_ids) + image_features → BART Encoder (6 layers)
  DecoderPrefill: BART Decoder (6 layers) + lm_head → logits + initial KV cache
  DecoderStep:    single token + KV cache → logits + updated self-attn KV cache

Requirements:
  pip install transformers==4.47.1 torch coremltools>=9.0

Usage:
  python convert_florence2.py
"""

import torch
import torch.nn as nn
import coremltools as ct
from transformers import AutoModelForCausalLM


# ============================================================
# Wrapper modules
# ============================================================

class VisionEncoder(nn.Module):
    """Image → projected features [1, 577, 768]."""

    def __init__(self, model):
        super().__init__()
        self.vision_tower = model.vision_tower
        self.image_pos_embed = model.image_pos_embed
        self.visual_temporal_embed = model.visual_temporal_embed
        self.image_projection = model.image_projection
        self.image_proj_norm = model.image_proj_norm
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, pixel_values):
        pixel_values = (pixel_values - self.mean) / self.std
        batch_size = pixel_values.shape[0]
        T = 1
        x = self.vision_tower.forward_features_unpool(pixel_values)
        num_tokens = x.shape[1]
        h = int(num_tokens ** 0.5)
        w = h
        x = x.view(batch_size * T, h, w, x.shape[-1])
        pos_embed = self.image_pos_embed(x)
        x = x + pos_embed
        x = x.view(batch_size, T * h * w, x.shape[-1])
        visual_temporal_embed = self.visual_temporal_embed(
            x.view(batch_size, T, -1, x.shape[-1])[:, :, 0]
        )
        x = (
            x.view(batch_size, T, -1, x.shape[-1])
            + visual_temporal_embed.view(1, T, 1, x.shape[-1])
        )
        spatial_avg_pool_x = x.mean(dim=2)
        temporal_avg_pool_x = x.mean(dim=1)
        x = torch.cat([spatial_avg_pool_x, temporal_avg_pool_x], dim=1)
        x = x @ self.image_projection
        x = self.image_proj_norm(x)
        return x


class TextEncoder(nn.Module):
    """image_features + input_ids → encoder_hidden_states [1, 585, 768]."""

    def __init__(self, model):
        super().__init__()
        self.embed_tokens = model.language_model.model.shared
        self.encoder = model.language_model.model.encoder

    def forward(self, image_features, input_ids):
        input_embeds = self.embed_tokens(input_ids)
        inputs_embeds = torch.cat([image_features, input_embeds], dim=1)
        batch_size = inputs_embeds.shape[0]
        seq_len = inputs_embeds.shape[1]
        attention_mask = torch.ones(
            batch_size, seq_len, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        encoder_out = self.encoder(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask
        )
        return encoder_out.last_hidden_state


class DecoderPrefill(nn.Module):
    """First decode step: no KV cache → logits + full KV cache."""

    def __init__(self, model):
        super().__init__()
        self.decoder = model.language_model.model.decoder
        self.lm_head = model.language_model.lm_head

    def forward(self, decoder_input_ids, encoder_hidden_states):
        encoder_attention_mask = torch.ones(
            1,
            encoder_hidden_states.shape[1],
            dtype=encoder_hidden_states.dtype,
            device=encoder_hidden_states.device,
        )
        decoder_out = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
        )
        logits = self.lm_head(decoder_out.last_hidden_state)
        pkv = decoder_out.past_key_values
        # logits, sa_kv (6×2), ca_kv (6×2)
        return (
            logits,
            pkv[0][0], pkv[0][1], pkv[1][0], pkv[1][1],
            pkv[2][0], pkv[2][1], pkv[3][0], pkv[3][1],
            pkv[4][0], pkv[4][1], pkv[5][0], pkv[5][1],
            pkv[0][2], pkv[0][3], pkv[1][2], pkv[1][3],
            pkv[2][2], pkv[2][3], pkv[3][2], pkv[3][3],
            pkv[4][2], pkv[4][3], pkv[5][2], pkv[5][3],
        )


class DecoderStep(nn.Module):
    """Subsequent decode step with KV cache → logits + updated self-attn KV."""

    def __init__(self, model):
        super().__init__()
        self.decoder = model.language_model.model.decoder
        self.lm_head = model.language_model.lm_head

    def forward(
        self,
        decoder_input_ids,
        encoder_hidden_states,
        sa_k_0, sa_v_0, sa_k_1, sa_v_1, sa_k_2, sa_v_2,
        sa_k_3, sa_v_3, sa_k_4, sa_v_4, sa_k_5, sa_v_5,
        ca_k_0, ca_v_0, ca_k_1, ca_v_1, ca_k_2, ca_v_2,
        ca_k_3, ca_v_3, ca_k_4, ca_v_4, ca_k_5, ca_v_5,
    ):
        past_key_values = tuple(
            (sa_k, sa_v, ca_k, ca_v)
            for sa_k, sa_v, ca_k, ca_v in zip(
                [sa_k_0, sa_k_1, sa_k_2, sa_k_3, sa_k_4, sa_k_5],
                [sa_v_0, sa_v_1, sa_v_2, sa_v_3, sa_v_4, sa_v_5],
                [ca_k_0, ca_k_1, ca_k_2, ca_k_3, ca_k_4, ca_k_5],
                [ca_v_0, ca_v_1, ca_v_2, ca_v_3, ca_v_4, ca_v_5],
            )
        )
        encoder_attention_mask = torch.ones(
            1,
            encoder_hidden_states.shape[1],
            dtype=encoder_hidden_states.dtype,
            device=encoder_hidden_states.device,
        )
        decoder_out = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = self.lm_head(decoder_out.last_hidden_state)
        pkv = decoder_out.past_key_values
        # Only return updated self-attention KV (cross-attn is unchanged)
        return (
            logits,
            pkv[0][0], pkv[0][1], pkv[1][0], pkv[1][1],
            pkv[2][0], pkv[2][1], pkv[3][0], pkv[3][1],
            pkv[4][0], pkv[4][1], pkv[5][0], pkv[5][1],
        )


# ============================================================
# Conversion
# ============================================================

def main():
    print("Loading microsoft/Florence-2-base ...")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-base", trust_remote_code=True
    )
    model.eval()

    out_dir = "."

    # --- 1. VisionEncoder ---
    print("\n[1/4] Converting VisionEncoder ...")
    ve = VisionEncoder(model)
    ve.eval()
    dummy_img = torch.randn(1, 3, 768, 768).clamp(0, 1)
    with torch.no_grad():
        traced_ve = torch.jit.trace(ve, dummy_img)
    ml_ve = ct.convert(
        traced_ve,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, 768, 768),
                scale=1.0 / 255.0,
                color_layout=ct.colorlayout.RGB,
            )
        ],
        outputs=[ct.TensorType(name="image_features")],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32,
    )
    ml_ve.author = "CoreML-Models"
    ml_ve.short_description = (
        "Florence-2 Vision Encoder (DaViT). "
        "Input: 768x768 RGB image. Output: image features [1, 577, 768]."
    )
    ml_ve.license = "MIT"
    ml_ve.save(f"{out_dir}/Florence2VisionEncoder.mlpackage")
    print("  Saved Florence2VisionEncoder.mlpackage")

    # --- 2. TextEncoder ---
    print("\n[2/4] Converting TextEncoder ...")
    te = TextEncoder(model)
    te.eval()
    dummy_feat = torch.randn(1, 577, 768)
    dummy_ids = torch.tensor([[0, 2264, 473, 5, 2274, 6190, 116, 2]])
    with torch.no_grad():
        traced_te = torch.jit.trace(te, (dummy_feat, dummy_ids))
    ml_te = ct.convert(
        traced_te,
        inputs=[
            ct.TensorType(name="image_features", shape=(1, 577, 768)),
            ct.TensorType(
                name="input_ids",
                shape=(1, ct.RangeDim(lower_bound=1, upper_bound=64, default=8)),
                dtype=int,
            ),
        ],
        outputs=[ct.TensorType(name="encoder_hidden_states")],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32,
    )
    ml_te.author = "CoreML-Models"
    ml_te.short_description = (
        "Florence-2 Text Encoder (BART). "
        "Input: image features [1,577,768] + input_ids. "
        "Output: encoder_hidden_states."
    )
    ml_te.license = "MIT"
    ml_te.save(f"{out_dir}/Florence2TextEncoder.mlpackage")
    print("  Saved Florence2TextEncoder.mlpackage")

    # --- 3. DecoderPrefill ---
    print("\n[3/4] Converting DecoderPrefill ...")
    dp = DecoderPrefill(model)
    dp.eval()
    dummy_enc_hs = torch.randn(1, 585, 768)
    decoder_start = torch.tensor([[2]])
    with torch.no_grad():
        traced_dp = torch.jit.trace(dp, (decoder_start, dummy_enc_hs))
    prefill_outputs = [ct.TensorType(name="logits")]
    for i in range(6):
        prefill_outputs.append(ct.TensorType(name=f"sa_key_{i}"))
        prefill_outputs.append(ct.TensorType(name=f"sa_value_{i}"))
    for i in range(6):
        prefill_outputs.append(ct.TensorType(name=f"ca_key_{i}"))
        prefill_outputs.append(ct.TensorType(name=f"ca_value_{i}"))
    ml_dp = ct.convert(
        traced_dp,
        inputs=[
            ct.TensorType(name="decoder_input_ids", shape=(1, 1), dtype=int),
            ct.TensorType(name="encoder_hidden_states", shape=(1, 585, 768)),
        ],
        outputs=prefill_outputs,
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32,
    )
    ml_dp.author = "CoreML-Models"
    ml_dp.short_description = (
        "Florence-2 Decoder Prefill. First decode step producing initial KV cache."
    )
    ml_dp.license = "MIT"
    ml_dp.save(f"{out_dir}/Florence2DecoderPrefill.mlpackage")
    print("  Saved Florence2DecoderPrefill.mlpackage")

    # --- 4. DecoderStep ---
    print("\n[4/4] Converting DecoderStep ...")
    ds = DecoderStep(model)
    ds.eval()
    with torch.no_grad():
        prefill_out = dp(decoder_start, dummy_enc_hs)
    pkv_flat = prefill_out[1:]  # skip logits
    trace_inputs = [torch.tensor([[0]]), dummy_enc_hs] + list(pkv_flat)
    with torch.no_grad():
        traced_ds = torch.jit.trace(ds, trace_inputs)
    sa_seq_dim = ct.RangeDim(lower_bound=1, upper_bound=1024, default=1)
    step_inputs = [
        ct.TensorType(name="decoder_input_ids", shape=(1, 1), dtype=int),
        ct.TensorType(name="encoder_hidden_states", shape=(1, 585, 768)),
    ]
    for i in range(6):
        step_inputs.append(
            ct.TensorType(name=f"sa_key_{i}", shape=(1, 12, sa_seq_dim, 64))
        )
        step_inputs.append(
            ct.TensorType(name=f"sa_value_{i}", shape=(1, 12, sa_seq_dim, 64))
        )
    for i in range(6):
        step_inputs.append(
            ct.TensorType(name=f"ca_key_{i}", shape=(1, 12, 585, 64))
        )
        step_inputs.append(
            ct.TensorType(name=f"ca_value_{i}", shape=(1, 12, 585, 64))
        )
    step_outputs = [ct.TensorType(name="logits")]
    for i in range(6):
        step_outputs.append(ct.TensorType(name=f"new_sa_key_{i}"))
        step_outputs.append(ct.TensorType(name=f"new_sa_value_{i}"))
    ml_ds = ct.convert(
        traced_ds,
        inputs=step_inputs,
        outputs=step_outputs,
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32,
    )
    ml_ds.author = "CoreML-Models"
    ml_ds.short_description = (
        "Florence-2 Decoder Step. Single token decode with KV cache. "
        "Outputs updated self-attn KV only (cross-attn is unchanged)."
    )
    ml_ds.license = "MIT"
    ml_ds.save(f"{out_dir}/Florence2DecoderStep.mlpackage")
    print("  Saved Florence2DecoderStep.mlpackage")

    print("\nDone! All 4 models converted.")


if __name__ == "__main__":
    main()
