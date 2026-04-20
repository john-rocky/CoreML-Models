"""End-to-end sanity check for the size-optimized Nitro-E pipeline.

Uses: NitroE_TextEncoder (FP16 or INT4), NitroE_EMMDiT (FP16 or INT8),
NitroE_VAEDecoder (INT8 palettized). Compares against PyTorch reference
image (reference_out.png).
"""

import os
import time

import numpy as np
import torch
import coremltools as ct
from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer
from PIL import Image

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(THIS_DIR, "reference_dump")
ROOT = os.path.dirname(THIS_DIR)


def main() -> None:
    # Allow overrides via env so we can swap precision tiers easily
    text_pkg = os.environ.get("TEXT_PKG",
        os.path.join(ROOT, "NitroE_TextEncoder.mlpackage"))
    emmdit_pkg = os.environ.get("EMMDIT_PKG",
        os.path.join(ROOT, "NitroE_EMMDiT.mlpackage"))
    vae_pkg = os.environ.get("VAE_PKG",
        os.path.join(ROOT, "NitroE_VAEDecoder_INT8.mlpackage"))

    print(f"[pkg] text   = {text_pkg}")
    print(f"[pkg] emmdit = {emmdit_pkg}")
    print(f"[pkg] vae    = {vae_pkg}")

    prompt = "a hot air balloon in the shape of a heart, grand canyon"

    t_load = time.time()
    te = ct.models.MLModel(text_pkg, compute_units=ct.ComputeUnit.CPU_AND_NE)
    emmdit = ct.models.MLModel(emmdit_pkg, compute_units=ct.ComputeUnit.CPU_AND_NE)
    vae = ct.models.MLModel(vae_pkg, compute_units=ct.ComputeUnit.CPU_AND_NE)
    print(f"[load] {time.time() - t_load:.1f}s")

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token
    tok = tokenizer(prompt, padding="max_length", max_length=128, truncation=True,
                    add_special_tokens=True, return_tensors="np")
    input_ids = tok["input_ids"].astype(np.int32)
    attention_mask = tok["attention_mask"].astype(np.int32)

    t0 = time.time()
    text = list(te.predict({"input_ids": input_ids, "attention_mask": attention_mask}).values())[0].astype(np.float32)
    t_te = time.time() - t0

    latent = torch.load(os.path.join(REF_DIR, "latent_in_step0.pt"),
                        map_location="cpu", weights_only=True)
    scheduler = FlowMatchEulerDiscreteScheduler(1000)
    scheduler.set_timesteps(4, device=torch.device("cpu"))

    t1 = time.time()
    for i, t in enumerate(scheduler.timesteps):
        ts = np.array([int(t.item())], dtype=np.int32)
        out = emmdit.predict({
            "latent": latent.numpy().astype(np.float32),
            "encoder_hidden_states": text,
            "encoder_attention_mask": attention_mask,
            "timestep": ts,
        })
        noise_pred = torch.from_numpy(list(out.values())[0].astype(np.float32))
        latent = scheduler.step(noise_pred, t, latent, return_dict=False)[0]
    t_dn = time.time() - t1

    t2 = time.time()
    img = list(vae.predict({"latent": latent.numpy().astype(np.float32)}).values())[0]
    t_vae = time.time() - t2

    img01 = np.clip((img[0].transpose(1, 2, 0) + 1.0) / 2.0, 0.0, 1.0)
    tag = os.path.basename(text_pkg).replace(".mlpackage", "") + "_" \
        + os.path.basename(emmdit_pkg).replace(".mlpackage", "") + "_" \
        + os.path.basename(vae_pkg).replace(".mlpackage", "")
    out_path = os.path.join(THIS_DIR, f"opt_{tag}.png")
    Image.fromarray((img01 * 255).astype(np.uint8)).save(out_path)

    ref = np.array(Image.open(os.path.join(THIS_DIR, "reference_out.png"))).astype(np.float32)
    ours = np.array(Image.open(out_path)).astype(np.float32)
    pixel_diff = np.abs(ref - ours).mean()

    print(f"[time] text {t_te*1000:.0f}ms + denoise {t_dn*1000:.0f}ms + vae {t_vae*1000:.0f}ms = {(t_te+t_dn+t_vae)*1000:.0f}ms")
    print(f"[pixel] mean abs diff vs reference = {pixel_diff:.2f} (0..255)")
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
