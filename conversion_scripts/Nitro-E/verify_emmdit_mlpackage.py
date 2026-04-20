"""Verify converted E-MMDiT mlpackage against the PyTorch reference dump.

Runs all 4 steps (with the PyTorch-captured inputs) and compares noise_pred.
"""

import os
import time

import numpy as np
import torch
import coremltools as ct

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(THIS_DIR, "reference_dump")
MLPKG = os.environ.get("MLPKG",
    os.path.join(os.path.dirname(THIS_DIR), "NitroE_EMMDiT.mlpackage"))


def main() -> None:
    print(f"[load] {MLPKG}")
    units = os.environ.get("UNITS", "CPU_AND_NE")
    cu_map = {
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "ALL": ct.ComputeUnit.ALL,
    }
    model = ct.models.MLModel(MLPKG, compute_units=cu_map[units])
    print(f"[load] compute_units={units}")

    text = torch.load(os.path.join(REF_DIR, "transformer_encoder_hidden_states.pt"),
                      map_location="cpu", weights_only=True).numpy().astype(np.float32)
    mask = torch.load(os.path.join(REF_DIR, "prompt_attention_mask.pt"),
                      map_location="cpu", weights_only=True).numpy().astype(np.int32)

    for step in range(4):
        latent = torch.load(os.path.join(REF_DIR, f"latent_in_step{step}.pt"),
                            map_location="cpu", weights_only=True).numpy().astype(np.float32)
        ts = torch.load(os.path.join(REF_DIR, f"timestep_step{step}.pt"),
                        map_location="cpu", weights_only=True).numpy().astype(np.int32)
        ref = torch.load(os.path.join(REF_DIR, f"noise_pred_step{step}.pt"),
                         map_location="cpu", weights_only=True).numpy()
        t0 = time.time()
        out = model.predict({
            "latent": latent,
            "encoder_hidden_states": text,
            "encoder_attention_mask": mask,
            "timestep": ts,
        })
        dt = time.time() - t0
        pred = list(out.values())[0]
        diff = np.abs(pred.astype(np.float32) - ref).max()
        mean = np.abs(pred.astype(np.float32) - ref).mean()
        print(f"[step{step}] {dt*1000:.0f} ms, max abs={diff:.4f} mean abs={mean:.5f}")


if __name__ == "__main__":
    main()
