"""
Convert Hyper-SD (SD1.5 + 1-step LoRA) to CoreML using Apple's ml-stable-diffusion.

Pipeline:
  1. Download SD1.5 base model
  2. Apply ByteDance Hyper-SD 1-step LoRA and fuse
  3. Save merged pipeline
  4. Convert with Apple's torch2coreml (Split Einsum + chunked UNet + 6-bit palettization)
  5. Quantize chunks separately

Output (~947MB total, fits iPhone 15+):
  - HyperSDTextEncoder.mlpackage   (235 MB, FP16)
  - HyperSDUnetChunk1.mlpackage    (318 MB, 6-bit palettized)
  - HyperSDUnetChunk2.mlpackage    (299 MB, 6-bit palettized)
  - HyperSDVAEDecoder.mlpackage    ( 95 MB, FP16)

Prerequisites:
  pip install diffusers==0.30.2 transformers==4.44.2 huggingface-hub==0.25.2 \
              peft accelerate coremltools torch
  git clone https://github.com/apple/ml-stable-diffusion ~/Downloads/ml-stable-diffusion

Apple ml-stable-diffusion patches required (for coremltools 9.0 compatibility):
  - chunk_mlprogram.py: replace `block.operations.index(op)` with
    `list(block.operations).index(op)` (3 places)
  - chunk_mlprogram.py: replace `main_block.operations[op_idx]` with
    `list(main_block.operations)[op_idx]` (2 places)
  - torch2coreml.py: skip text_encoder quantization (FP16 has inf values)
    Change quantize_weights model list from
      ["text_encoder", "text_encoder_2", "unet", "refiner", "control-unet"]
    to
      ["unet", "refiner", "control-unet"]

Usage:
  python convert_hypersd.py
"""

import sys
import os
import numpy as np
import torch

# ---------- Step 1: Merge LoRA into SD1.5 ----------
def merge_lora():
    from diffusers import StableDiffusionPipeline

    out_dir = os.path.expanduser("~/Downloads/hyper-sd15-merged")
    if os.path.exists(out_dir):
        print(f"Merged pipeline exists at {out_dir}")
        return out_dir

    print("Loading SD1.5...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,
        safety_checker=None,
    )
    print("Loading Hyper-SD 1-step LoRA...")
    pipe.load_lora_weights("ByteDance/Hyper-SD", weight_name="Hyper-SD15-1step-lora.safetensors")
    pipe.fuse_lora(lora_scale=1.0)
    pipe.unload_lora_weights()

    print(f"Saving merged pipeline to {out_dir}...")
    pipe.save_pretrained(out_dir, safe_serialization=True)
    return out_dir


# ---------- Step 2: Patch coremltools int op converter ----------
import coremltools as ct
from coremltools.converters.mil.frontend.torch import ops as _ct_ops
from coremltools.converters.mil import Builder as mb

def _patched_int(context, node):
    inputs = _ct_ops._get_inputs(context, node)
    x = inputs[0]
    if x.val is not None:
        val = x.val
        if isinstance(val, np.ndarray):
            val = int(val.item()) if val.ndim == 0 else int(val.flat[0])
        else:
            val = int(val)
        res = mb.const(val=np.int32(val), name=node.name)
    else:
        res = mb.cast(x=x, dtype="int32", name=node.name)
    context.add(res)

_ct_ops._TORCH_OPS_REGISTRY.register_func(_patched_int, torch_alias=["int"], override=True)


# ---------- Step 3: Convert with Apple's tool ----------
def convert(merged_dir, output_dir):
    sys.path.insert(0, os.path.expanduser("~/Downloads/ml-stable-diffusion"))
    from python_coreml_stable_diffusion.torch2coreml import main, parser_spec

    parser = parser_spec()
    args = parser.parse_args([
        "--model-version", merged_dir,
        "--convert-text-encoder",
        "--convert-unet",
        "--convert-vae-decoder",
        "--attention-implementation", "SPLIT_EINSUM",
        "--compute-unit", "CPU_AND_NE",
        "--quantize-nbits", "6",
        "--chunk-unet",
        "-o", output_dir,
    ])
    main(args)


# ---------- Step 4: Quantize UNet chunks separately ----------
def quantize_chunks(output_dir, prefix):
    for chunk in ["chunk1", "chunk2"]:
        path = f"{output_dir}/{prefix}_unet_{chunk}.mlpackage"
        print(f"Quantizing {chunk}...")
        model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY)
        config = ct.optimize.coreml.OptimizationConfig(
            global_config=ct.optimize.coreml.OpPalettizerConfig(mode="kmeans", nbits=6)
        )
        ct.optimize.coreml.palettize_weights(model, config=config).save(path)


def main():
    output_dir = os.path.expanduser("~/Downloads/CoreML-Models/conversion_scripts/hypersd_apple")
    merged_dir = merge_lora()
    convert(merged_dir, output_dir)

    # Apple's tool generates files prefixed with the model path; quantize the chunks
    prefix = f"Stable_Diffusion_version_{merged_dir.replace('/', '_')}"
    quantize_chunks(output_dir, prefix)
    print("\nDone!")


if __name__ == "__main__":
    main()
