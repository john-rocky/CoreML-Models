"""
Convert Microsoft Florence-2-base to CoreML (3 models, INT8).

Architecture:
  VisionEncoder: DaViT → pos_embed → temporal_embed → image_projection → LayerNorm
  TextEncoder:   Embedding(input_ids) + image_features → BART Encoder (6 layers)
  Decoder:       BART Decoder (6 layers) + lm_head (no KV cache, full sequence)

Requirements:
  pip install transformers==4.47.1 torch coremltools>=9.0

Usage:
  python convert_florence2.py
"""

import torch
import torch.nn as nn
import coremltools as ct
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig,
    OptimizationConfig,
    linear_quantize_weights,
)
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
        h = int(x.shape[1] ** 0.5)
        x = x.view(batch_size, h, h, x.shape[-1])
        x = x + self.image_pos_embed(x)
        x = x.view(batch_size, h * h, x.shape[-1])
        te = self.visual_temporal_embed(
            x.view(batch_size, T, -1, x.shape[-1])[:, :, 0]
        )
        x = (
            x.view(batch_size, T, -1, x.shape[-1])
            + te.view(1, T, 1, x.shape[-1])
        )
        spatial_avg = x.mean(dim=2)
        temporal_avg = x.mean(dim=1)
        x = torch.cat([spatial_avg, temporal_avg], dim=1)
        return self.image_proj_norm(x @ self.image_projection)


class TextEncoder(nn.Module):
    """image_features + input_ids → encoder_hidden_states."""

    def __init__(self, model):
        super().__init__()
        self.embed_tokens = model.language_model.model.shared
        self.encoder = model.language_model.model.encoder

    def forward(self, image_features, input_ids):
        text_embeds = self.embed_tokens(input_ids)
        inputs_embeds = torch.cat([image_features, text_embeds], dim=1)
        bs, sl = inputs_embeds.shape[:2]
        mask = torch.ones(bs, sl, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        return self.encoder(inputs_embeds=inputs_embeds, attention_mask=mask).last_hidden_state


class Decoder(nn.Module):
    """decoder_input_ids + encoder_hidden_states → logits (no KV cache)."""

    def __init__(self, model):
        super().__init__()
        self.decoder = model.language_model.model.decoder
        self.lm_head = model.language_model.lm_head

    def forward(self, decoder_input_ids, encoder_hidden_states):
        mask = torch.ones(
            1, encoder_hidden_states.shape[1],
            dtype=encoder_hidden_states.dtype, device=encoder_hidden_states.device
        )
        out = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=mask,
            use_cache=False,
        )
        return self.lm_head(out.last_hidden_state)


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
    quant_config = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
    )

    # --- 1. VisionEncoder ---
    print("\n[1/3] Converting VisionEncoder ...")
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
        compute_precision=ct.precision.FLOAT16,
    )
    ml_ve = linear_quantize_weights(ml_ve, quant_config)
    ml_ve.author = "CoreML-Models"
    ml_ve.short_description = (
        "Florence-2 Vision Encoder (DaViT). "
        "Input: 768x768 RGB image. Output: image features [1, 577, 768]. INT8."
    )
    ml_ve.license = "MIT"
    ml_ve.save(f"{out_dir}/Florence2VisionEncoder.mlpackage")
    print("  Saved Florence2VisionEncoder.mlpackage")

    # --- 2. TextEncoder ---
    print("\n[2/3] Converting TextEncoder ...")
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
        compute_precision=ct.precision.FLOAT16,
    )
    ml_te = linear_quantize_weights(ml_te, quant_config)
    ml_te.author = "CoreML-Models"
    ml_te.short_description = (
        "Florence-2 Text Encoder (BART). "
        "Input: image features [1,577,768] + input_ids. "
        "Output: encoder_hidden_states. INT8."
    )
    ml_te.license = "MIT"
    ml_te.save(f"{out_dir}/Florence2TextEncoder.mlpackage")
    print("  Saved Florence2TextEncoder.mlpackage")

    # --- 3. Decoder ---
    print("\n[3/3] Converting Decoder ...")
    dec = Decoder(model)
    dec.eval()
    with torch.no_grad():
        traced_dec = torch.jit.trace(
            dec, (torch.tensor([[2, 0]]), torch.randn(1, 585, 768))
        )
    enc_seq_dim = ct.RangeDim(lower_bound=580, upper_bound=600, default=585)
    dec_seq_dim = ct.RangeDim(lower_bound=1, upper_bound=512, default=1)
    ml_dec = ct.convert(
        traced_dec,
        inputs=[
            ct.TensorType(name="decoder_input_ids", shape=(1, dec_seq_dim), dtype=int),
            ct.TensorType(
                name="encoder_hidden_states", shape=(1, enc_seq_dim, 768)
            ),
        ],
        outputs=[ct.TensorType(name="logits")],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
    )
    ml_dec = linear_quantize_weights(ml_dec, quant_config)
    ml_dec.author = "CoreML-Models"
    ml_dec.short_description = (
        "Florence-2 Decoder (BART + LM Head). "
        "Input: decoder_input_ids + encoder_hidden_states. "
        "Output: logits. INT8."
    )
    ml_dec.license = "MIT"
    ml_dec.save(f"{out_dir}/Florence2Decoder.mlpackage")
    print("  Saved Florence2Decoder.mlpackage")

    print("\nDone! All 3 models converted.")


if __name__ == "__main__":
    main()
