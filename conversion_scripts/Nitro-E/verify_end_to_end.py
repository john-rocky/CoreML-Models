"""End-to-end CoreML verification:
- Load E-MMDiT mlpackage + VAE decoder mlpackage
- Use PyTorch scheduler + reference prompt_embeds/initial_latent
- Run 4 denoise steps on CoreML + decode on CoreML
- Compare final image with reference_out PyTorch image
"""

import os
import time

import numpy as np
import torch
import coremltools as ct
from diffusers import FlowMatchEulerDiscreteScheduler
from PIL import Image

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(THIS_DIR, "reference_dump")
ROOT = os.path.dirname(THIS_DIR)
EMMDIT_PKG = os.path.join(ROOT, "NitroE_EMMDiT.mlpackage")
VAE_PKG = os.path.join(ROOT, "NitroE_VAEDecoder_FP32.mlpackage")


def main() -> None:
    emmdit = ct.models.MLModel(EMMDIT_PKG, compute_units=ct.ComputeUnit.CPU_AND_NE)
    vae = ct.models.MLModel(VAE_PKG, compute_units=ct.ComputeUnit.CPU_AND_NE)
    print("[load] both mlpackages loaded")

    text = torch.load(os.path.join(REF_DIR, "transformer_encoder_hidden_states.pt"),
                      map_location="cpu", weights_only=True).numpy().astype(np.float32)
    mask = torch.load(os.path.join(REF_DIR, "prompt_attention_mask.pt"),
                      map_location="cpu", weights_only=True).numpy().astype(np.int32)
    latent = torch.load(os.path.join(REF_DIR, "latent_in_step0.pt"),
                        map_location="cpu", weights_only=True)

    # Reproduce the pipeline's scheduler
    scheduler = FlowMatchEulerDiscreteScheduler(1000)
    scheduler.set_timesteps(4, device=torch.device("cpu"))
    timesteps = scheduler.timesteps
    print(f"[sched] timesteps = {timesteps.tolist()}")

    t0 = time.time()
    for i, t in enumerate(timesteps):
        ts = torch.tensor([int(t.item())], dtype=torch.int32).numpy()
        out = emmdit.predict({
            "latent": latent.numpy().astype(np.float32),
            "encoder_hidden_states": text,
            "encoder_attention_mask": mask,
            "timestep": ts,
        })
        noise_pred = torch.from_numpy(list(out.values())[0].astype(np.float32))
        latent = scheduler.step(noise_pred, t, latent, return_dict=False)[0]
    denoise_dt = time.time() - t0
    print(f"[denoise] 4 steps total = {denoise_dt:.2f}s")

    # Compare pre-decode latent with reference
    # (reference dump does not have post-scheduler latent directly, but
    # vae_decode_input was `latents / scaling_factor`)
    vae_input = torch.load(os.path.join(REF_DIR, "vae_decode_input.pt"),
                           map_location="cpu", weights_only=True)
    scaling = 0.41407
    ref_final_latent = vae_input * scaling
    diff = (latent - ref_final_latent).abs().max().item()
    print(f"[parity] final latent max abs diff vs reference = {diff:.5f}")

    t1 = time.time()
    vae_out = vae.predict({"latent": latent.numpy().astype(np.float32)})
    img = list(vae_out.values())[0]
    decode_dt = time.time() - t1
    print(f"[decode] {decode_dt:.2f}s, img.shape={img.shape}")

    img01 = np.clip((img[0].transpose(1, 2, 0) + 1.0) / 2.0, 0.0, 1.0)
    out_path = os.path.join(THIS_DIR, "coreml_end_to_end.png")
    Image.fromarray((img01 * 255).astype(np.uint8)).save(out_path)
    print(f"[save] {out_path}")

    # compare with reference image
    ref = np.array(Image.open(os.path.join(THIS_DIR, "reference_out.png"))).astype(np.float32)
    ours = np.array(Image.open(out_path)).astype(np.float32)
    pixel_diff = np.abs(ref - ours).mean()
    print(f"[pixel] mean abs diff vs reference_out.png = {pixel_diff:.2f} (0..255)")


if __name__ == "__main__":
    main()
