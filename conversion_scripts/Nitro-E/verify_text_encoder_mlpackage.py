"""Verify converted Llama text encoder mlpackage against PyTorch reference."""

import os
import time

import numpy as np
import torch
import coremltools as ct
from transformers import AutoTokenizer

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(THIS_DIR, "reference_dump")
MLPKG = os.environ.get("MLPKG",
    os.path.join(os.path.dirname(THIS_DIR), "NitroE_TextEncoder.mlpackage"))


def main() -> None:
    units = os.environ.get("UNITS", "CPU_AND_NE")
    cu_map = {
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "ALL": ct.ComputeUnit.ALL,
    }
    print(f"[load] {MLPKG} on {units}")
    model = ct.models.MLModel(MLPKG, compute_units=cu_map[units])

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token

    prompt = "a hot air balloon in the shape of a heart, grand canyon"
    tokens = tokenizer(
        prompt,
        padding="max_length",
        max_length=128,
        truncation=True,
        add_special_tokens=True,
        return_tensors="np",
    )
    input_ids = tokens["input_ids"].astype(np.int32)
    attention_mask = tokens["attention_mask"].astype(np.int32)

    t0 = time.time()
    out = model.predict({"input_ids": input_ids, "attention_mask": attention_mask})
    dt = time.time() - t0
    pred = list(out.values())[0]
    print(f"[predict] {dt*1000:.0f}ms, shape={pred.shape} dtype={pred.dtype}")

    ref = torch.load(os.path.join(REF_DIR, "prompt_embeds.pt"),
                     map_location="cpu", weights_only=True).numpy()
    diff = np.abs(pred.astype(np.float32) - ref).max()
    mean = np.abs(pred.astype(np.float32) - ref).mean()
    print(f"[parity] vs reference prompt_embeds: max abs={diff:.4f} mean abs={mean:.5f}")


if __name__ == "__main__":
    main()
