"""Check DC-AE f32c32 decode on dummy latent. Also prints param count."""

import os
import sys
import time

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from diffusers import AutoencoderDC  # noqa: E402


def main() -> None:
    device = torch.device("cpu")
    dtype = torch.float32
    print("[download] mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers")
    vae = AutoencoderDC.from_pretrained(
        "mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers",
        torch_dtype=dtype,
    )
    vae = vae.to(device).eval()
    print(f"[info] VAE scaling_factor={vae.scaling_factor}")
    total = sum(p.numel() for p in vae.parameters())
    dec = sum(p.numel() for p in vae.decoder.parameters())
    enc = sum(p.numel() for p in vae.encoder.parameters())
    print(f"[info] params total={total/1e6:.1f}M  encoder={enc/1e6:.1f}M  decoder={dec/1e6:.1f}M")

    latent = torch.randn(1, 32, 16, 16, dtype=dtype, device=device)
    print("[decode] latent.shape=", tuple(latent.shape))
    t0 = time.time()
    with torch.no_grad():
        out = vae.decode(latent / vae.scaling_factor)
    dt = time.time() - t0
    img = out.sample if hasattr(out, "sample") else out
    print(f"[decode] ok, out.shape={tuple(img.shape)}, {dt:.2f}s")
    print(f"[decode] stats: min={img.min().item():.3f} max={img.max().item():.3f} mean={img.mean().item():.3f}")


if __name__ == "__main__":
    main()
