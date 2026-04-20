"""Load just the E-MMDiT transformer + checkpoint, run a single dummy forward.

Used to verify the architecture compiles and the weights load before we spend
time downloading Llama 3.2 1B.
"""

import os
import sys
import time

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from safetensors.torch import load_file  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from core.models.transformer_emmdit import EMMDiTTransformer  # noqa: E402


def main() -> None:
    device = torch.device("cpu")
    dtype = torch.float32

    print("[build] EMMDiTTransformer(sample_size=16, use_sub_attn=True)")
    transformer = EMMDiTTransformer(sample_size=16, use_sub_attn=True)
    transformer.eval()
    n_params = sum(p.numel() for p in transformer.parameters())
    print(f"[info] params = {n_params/1e6:.1f} M")

    print("[download] Nitro-E-512px-dist.safetensors")
    ckpt_path = hf_hub_download("amd/Nitro-E", "Nitro-E-512px-dist.safetensors")
    print(f"[download] cached at {ckpt_path}")

    sd = load_file(ckpt_path)
    missing, unexpected = transformer.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print("  first missing:", missing[:5])
    if unexpected:
        print("  first unexpected:", unexpected[:5])

    transformer = transformer.to(device).to(dtype)

    # Dummy inputs:
    #   latent: [1, 32, 16, 16]  (DC-AE 32x compression at 512px)
    #   encoder_hidden_states: [1, 128, 2048]  (Llama 3.2 1B last-layer, max_seq=128)
    #   encoder_attention_mask: [1, 128]
    #   timestep: scalar → expanded to [1]
    latent = torch.randn(1, 32, 16, 16, dtype=dtype, device=device)
    text = torch.randn(1, 128, 2048, dtype=dtype, device=device)
    mask = torch.ones(1, 128, dtype=torch.long, device=device)
    t = torch.tensor([500], dtype=torch.long, device=device)

    print("[forward] running...")
    t0 = time.time()
    with torch.no_grad():
        out = transformer(
            latent,
            encoder_hidden_states=text,
            encoder_attention_mask=mask,
            timestep=t,
            return_dict=False,
        )[0]
    dt = time.time() - t0
    print(f"[forward] ok, out.shape={tuple(out.shape)}, dtype={out.dtype}, {dt:.2f}s")
    print(f"[forward] out stats: mean={out.mean().item():.4f} std={out.std().item():.4f}")


if __name__ == "__main__":
    main()
