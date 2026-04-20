"""Run the entire Nitro-E 4-step distilled pipeline on CoreML only (no PyTorch
model in the loop). Scheduler runs in Python (FlowMatchEuler).
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
TEXT_PKG = os.path.join(ROOT, "NitroE_TextEncoder.mlpackage")
EMMDIT_PKG = os.path.join(ROOT, "NitroE_EMMDiT.mlpackage")
VAE_PKG = os.path.join(ROOT, "NitroE_VAEDecoder_FP32.mlpackage")


def main() -> None:
    prompt = "a hot air balloon in the shape of a heart, grand canyon"

    t_load = time.time()
    te = ct.models.MLModel(TEXT_PKG, compute_units=ct.ComputeUnit.CPU_AND_NE)
    emmdit = ct.models.MLModel(EMMDIT_PKG, compute_units=ct.ComputeUnit.CPU_AND_NE)
    vae = ct.models.MLModel(VAE_PKG, compute_units=ct.ComputeUnit.CPU_AND_NE)
    print(f"[load] all 3 models, {time.time() - t_load:.1f}s")

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token
    tok = tokenizer(
        prompt,
        padding="max_length",
        max_length=128,
        truncation=True,
        add_special_tokens=True,
        return_tensors="np",
    )
    input_ids = tok["input_ids"].astype(np.int32)
    attention_mask = tok["attention_mask"].astype(np.int32)

    t0 = time.time()
    te_out = te.predict({"input_ids": input_ids, "attention_mask": attention_mask})
    text = list(te_out.values())[0].astype(np.float32)
    t_te = time.time() - t0
    print(f"[text] {t_te*1000:.0f}ms, text.shape={text.shape}")

    # Deterministic initial latent — use the reference's starting latent so the
    # image matches reference bit-for-bit (aside from FP16 drift).
    latent = torch.load(os.path.join(REF_DIR, "latent_in_step0.pt"),
                        map_location="cpu", weights_only=True)

    scheduler = FlowMatchEulerDiscreteScheduler(1000)
    scheduler.set_timesteps(4, device=torch.device("cpu"))
    timesteps = scheduler.timesteps

    t1 = time.time()
    for i, t in enumerate(timesteps):
        ts = torch.tensor([int(t.item())], dtype=torch.int32).numpy()
        out = emmdit.predict({
            "latent": latent.numpy().astype(np.float32),
            "encoder_hidden_states": text,
            "encoder_attention_mask": attention_mask,
            "timestep": ts,
        })
        noise_pred = torch.from_numpy(list(out.values())[0].astype(np.float32))
        latent = scheduler.step(noise_pred, t, latent, return_dict=False)[0]
    t_dn = time.time() - t1
    print(f"[denoise] 4 steps = {t_dn*1000:.0f}ms")

    t2 = time.time()
    vae_out = vae.predict({"latent": latent.numpy().astype(np.float32)})
    img = list(vae_out.values())[0]
    t_vae = time.time() - t2
    print(f"[vae] {t_vae*1000:.0f}ms")

    img01 = np.clip((img[0].transpose(1, 2, 0) + 1.0) / 2.0, 0.0, 1.0)
    out_path = os.path.join(THIS_DIR, "coreml_full_pipeline.png")
    Image.fromarray((img01 * 255).astype(np.uint8)).save(out_path)
    print(f"[save] {out_path}")
    print(f"[total] end-to-end = {(t_te + t_dn + t_vae)*1000:.0f}ms")

    # Compare with PyTorch reference
    ref = np.array(Image.open(os.path.join(THIS_DIR, "reference_out.png"))).astype(np.float32)
    ours = np.array(Image.open(out_path)).astype(np.float32)
    pixel_diff = np.abs(ref - ours).mean()
    print(f"[pixel] mean abs diff vs reference_out.png = {pixel_diff:.2f} (0..255)")


if __name__ == "__main__":
    main()
