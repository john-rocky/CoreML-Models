"""
Convert BRIA RMBG-1.4 background removal to CoreML (1 model, INT8).

Architecture:
  IS-Net based U2Net variant with ImageNet normalization baked in.
  Input: 1024x1024 RGB image → Output: alpha mask [1, 1, 1024, 1024]

Requirements:
  pip install transformers torch coremltools>=9.0

Usage:
  python convert_rmbg.py
"""

import torch
import torch.nn as nn
import coremltools as ct
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig,
    OptimizationConfig,
    linear_quantize_weights,
)
from transformers import AutoModelForImageSegmentation


class RMBGWrapper(nn.Module):
    """Adds ImageNet normalization and sigmoid to raw model output."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x):
        x = (x - self.mean) / self.std
        return torch.sigmoid(self.model(x)[0][0])


def main():
    print("Loading briaai/RMBG-1.4 ...")
    model = AutoModelForImageSegmentation.from_pretrained(
        "briaai/RMBG-1.4", trust_remote_code=True
    )
    model.eval()

    wrapper = RMBGWrapper(model)
    wrapper.eval()

    dummy = torch.randn(1, 3, 1024, 1024).clamp(0, 1)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy)

    print("Converting to CoreML FP16 ...")
    ml = ct.convert(
        traced,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, 1024, 1024),
                scale=1.0 / 255.0,
                color_layout=ct.colorlayout.RGB,
            )
        ],
        outputs=[ct.TensorType(name="alpha_mask")],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
    )

    print("Quantizing to INT8 ...")
    quant_config = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
    )
    ml = linear_quantize_weights(ml, quant_config)

    ml.author = "CoreML-Models"
    ml.short_description = (
        "RMBG-1.4 background removal. "
        "1024x1024 RGB → alpha mask [1, 1, 1024, 1024]. INT8."
    )
    ml.license = "Apache-2.0"

    ml.save("RMBG_1_4.mlpackage")
    print("Saved RMBG_1_4.mlpackage")


if __name__ == "__main__":
    main()
