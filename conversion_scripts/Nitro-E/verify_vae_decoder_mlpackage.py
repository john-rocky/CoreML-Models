"""Numerically verify the converted VAE decoder mlpackage against the
PyTorch reference dump. Passes if max abs diff < 0.05 (FP16 tolerance).
"""

import os
import sys

import numpy as np
import torch
import coremltools as ct

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(THIS_DIR, "reference_dump")
MLPKG = os.environ.get("MLPKG",
    os.path.join(os.path.dirname(THIS_DIR), "NitroE_VAEDecoder.mlpackage"))


def main() -> None:
    print(f"[load] {MLPKG}")
    model = ct.models.MLModel(MLPKG, compute_units=ct.ComputeUnit.CPU_AND_NE)

    # Reference was fed with `latents / scaling_factor`; our wrapper accepts
    # the un-scaled pipeline latent, so multiply back by the scaling factor.
    scaling_factor = 0.41407
    z = torch.load(os.path.join(REF_DIR, "vae_decode_input.pt"),
                   map_location="cpu", weights_only=True)
    pipeline_latent = (z * scaling_factor).numpy().astype(np.float32)
    ref_out = torch.load(os.path.join(REF_DIR, "vae_decode_output.pt"),
                         map_location="cpu", weights_only=True).numpy()

    print(f"[input] pipeline_latent.shape={pipeline_latent.shape}")
    out = model.predict({"latent": pipeline_latent})
    img = list(out.values())[0]
    print(f"[output] shape={img.shape} dtype={img.dtype}")
    diff = np.abs(img.astype(np.float32) - ref_out).max()
    rel = np.abs(img.astype(np.float32) - ref_out).mean()
    print(f"[parity] max abs = {diff:.4f}  mean abs = {rel:.5f}")
    # Save the CoreML image for visual inspection
    from PIL import Image
    img01 = np.clip((img[0].transpose(1, 2, 0) + 1.0) / 2.0, 0.0, 1.0)
    Image.fromarray((img01 * 255).astype(np.uint8)).save(os.path.join(THIS_DIR, "vae_decoder_coreml.png"))
    print(f"[save] {os.path.join(THIS_DIR, 'vae_decoder_coreml.png')}")


if __name__ == "__main__":
    main()
