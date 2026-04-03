"""
Convert Google SigLIP ViT-B/16-224 to CoreML (2 models, INT8).

Architecture:
  ImageEncoder: SiglipVisionTransformer → L2-normalize → 768-dim embedding
  TextEncoder:  SiglipTextTransformer → L2-normalize → 768-dim embedding

Similarity: sigmoid(image_emb @ text_emb.T * logit_scale + logit_bias)
  logit_scale (exp) = 117.3308
  logit_bias = -12.9324

Requirements:
  pip install transformers torch coremltools>=9.0 sentencepiece

Usage:
  python convert_siglip.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig,
    OptimizationConfig,
    linear_quantize_weights,
)
from transformers import AutoModel, AutoProcessor


class ImageEncoder(nn.Module):
    """224x224 RGB → L2-normalized 768-dim embedding."""

    def __init__(self, model):
        super().__init__()
        self.vision = model.vision_model

    def forward(self, pixel_values):
        return F.normalize(self.vision(pixel_values).pooler_output, dim=-1)


class TextEncoder(nn.Module):
    """Token IDs → L2-normalized 768-dim embedding."""

    def __init__(self, model):
        super().__init__()
        self.text = model.text_model

    def forward(self, input_ids):
        return F.normalize(self.text(input_ids).pooler_output, dim=-1)


def main():
    print("Loading google/siglip-base-patch16-224 ...")
    model_id = "google/siglip-base-patch16-224"
    model = AutoModel.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()

    quant_config = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
    )
    out_dir = "."

    # --- 1. ImageEncoder ---
    print("\n[1/2] Converting ImageEncoder ...")
    ie = ImageEncoder(model)
    ie.eval()
    dummy_img = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        traced_ie = torch.jit.trace(ie, dummy_img)

    # SigLIP preprocessing: mean=0.5, std=0.5 → scale=1/(255*0.5), bias=-1
    ml_ie = ct.convert(
        traced_ie,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, 224, 224),
                scale=1.0 / (255.0 * 0.5),
                bias=[-1, -1, -1],
                color_layout=ct.colorlayout.RGB,
            )
        ],
        outputs=[ct.TensorType(name="image_embedding")],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
    )
    ml_ie = linear_quantize_weights(ml_ie, quant_config)
    ml_ie.author = "CoreML-Models"
    ml_ie.short_description = (
        "SigLIP ViT-B/16 Image Encoder. "
        "224x224 RGB → L2-normalized 768-dim embedding. INT8."
    )
    ml_ie.license = "Apache-2.0"
    ml_ie.save(f"{out_dir}/SigLIP_ImageEncoder.mlpackage")
    print("  Saved SigLIP_ImageEncoder.mlpackage")

    # --- 2. TextEncoder ---
    print("\n[2/2] Converting TextEncoder ...")
    te = TextEncoder(model)
    te.eval()
    dummy_ids = torch.tensor([[0, 1, 2, 3]])
    with torch.no_grad():
        traced_te = torch.jit.trace(te, dummy_ids)

    text_seq = ct.RangeDim(lower_bound=1, upper_bound=64, default=4)
    ml_te = ct.convert(
        traced_te,
        inputs=[
            ct.TensorType(
                name="input_ids", shape=(1, text_seq), dtype=int
            )
        ],
        outputs=[ct.TensorType(name="text_embedding")],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
    )
    ml_te = linear_quantize_weights(ml_te, quant_config)
    ml_te.author = "CoreML-Models"
    ml_te.short_description = (
        "SigLIP ViT-B/16 Text Encoder. "
        "Token IDs → L2-normalized 768-dim embedding. INT8."
    )
    ml_te.license = "Apache-2.0"
    ml_te.save(f"{out_dir}/SigLIP_TextEncoder.mlpackage")
    print("  Saved SigLIP_TextEncoder.mlpackage")

    # Print constants needed for Swift
    scale = model.logit_scale.exp().item()
    bias = model.logit_bias.item()
    print(f"\n=== Constants for Swift ===")
    print(f"logit_scale = {scale:.6f}")
    print(f"logit_bias = {bias:.6f}")
    print(f"similarity = sigmoid(image_emb · text_emb * scale + bias)")

    print("\nDone! 2 models converted.")


if __name__ == "__main__":
    main()
