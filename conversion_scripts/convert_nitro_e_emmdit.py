"""Convert the E-MMDiT denoiser (Nitro-E-512px-dist) to CoreML.

Target: 1 denoise step, batch=1 (distilled variant uses guidance_scale=0 so
no CFG). FP16 first, with the same coremltools/diffusers patches that the
VAE conversion needed.

Parity target: conversion_scripts/Nitro-E/reference_dump/latent_in_step*.pt,
               conversion_scripts/Nitro-E/reference_dump/noise_pred_step*.pt
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

import coremltools as ct
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
NITRO_E_DIR = os.path.join(THIS_DIR, "Nitro-E")
REF_DIR = os.path.join(NITRO_E_DIR, "reference_dump")
sys.path.insert(0, NITRO_E_DIR)

from core.models.transformer_emmdit import EMMDiTTransformer  # noqa: E402


def _patch_coremltools_cast():
    import numpy as np
    import coremltools.converters.mil.frontend.torch.ops as _ops
    from coremltools.converters.mil import Builder as mb

    _get_inputs = _ops._get_inputs

    def _cast(context, node, dtype, dtype_name):
        inputs = _get_inputs(context, node, expected=1)
        x = inputs[0]
        if not (len(x.shape) == 0 or np.all([d == 1 for d in x.shape])):
            raise ValueError("input to cast must be either a scalar or a length 1 tensor")
        if x.can_be_folded_to_const():
            if not isinstance(x.val, dtype):
                val = x.val
                if hasattr(val, "item"):
                    val = val.item()
                res = mb.const(val=dtype(val), name=node.name)
            else:
                res = x
        elif len(x.shape) > 0:
            x = mb.squeeze(x=x, name=node.name + "_item")
            res = mb.cast(x=x, dtype=dtype_name, name=node.name)
        else:
            res = mb.cast(x=x, dtype=dtype_name, name=node.name)
        context.add(res, node.name)

    _ops._cast = _cast


def _patch_torch_movedim():
    _orig = torch.Tensor.movedim

    def movedim(self, source, destination):
        if self.dim() == 4 and source == 1 and destination == -1:
            return self.permute(0, 2, 3, 1)
        if self.dim() == 4 and source == -1 and destination == 1:
            return self.permute(0, 3, 1, 2)
        return _orig(self, source, destination)

    torch.Tensor.movedim = movedim


class EMMDiTWrapper(nn.Module):
    """One denoise-step wrapper with a pure-tensor signature."""

    def __init__(self, transformer: EMMDiTTransformer):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        latent: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        # encoder_attention_mask is accepted but not consumed by the upstream
        # joint-attention processor; we still keep it as an input so the Swift
        # driver has a single call signature.
        _ = encoder_attention_mask
        return self.transformer(
            latent,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            timestep=timestep,
            return_dict=False,
        )[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(THIS_DIR, "NitroE_EMMDiT.mlpackage"))
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    args = ap.parse_args()

    latent_h = latent_w = args.resolution // 32

    _patch_coremltools_cast()
    _patch_torch_movedim()

    print("[build] EMMDiTTransformer(sample_size=16, use_sub_attn=True)")
    transformer = EMMDiTTransformer(sample_size=latent_h, use_sub_attn=True)
    transformer.eval()
    print("[download] Nitro-E-512px-dist.safetensors")
    ckpt_path = hf_hub_download("amd/Nitro-E", "Nitro-E-512px-dist.safetensors")
    sd = load_file(ckpt_path)
    miss, unex = transformer.load_state_dict(sd, strict=False)
    assert len(miss) == 0 and len(unex) == 0, f"weight mismatch: missing={len(miss)} unexpected={len(unex)}"

    transformer = transformer.to(torch.float32)
    wrapper = EMMDiTWrapper(transformer).eval()

    # Sample input shapes mirror the reference dump.
    latent = torch.randn(1, 32, latent_h, latent_w, dtype=torch.float32)
    text = torch.randn(1, args.seq_len, 2048, dtype=torch.float32)
    mask = torch.ones(1, args.seq_len, dtype=torch.long)
    t = torch.tensor([500], dtype=torch.long)

    # Parity sanity check vs reference dump (step 0 only)
    if os.path.exists(os.path.join(REF_DIR, "latent_in_step0.pt")):
        ref_latent = torch.load(os.path.join(REF_DIR, "latent_in_step0.pt"),
                                map_location="cpu", weights_only=True)
        ref_text = torch.load(os.path.join(REF_DIR, "transformer_encoder_hidden_states.pt"),
                              map_location="cpu", weights_only=True)
        ref_mask = torch.load(os.path.join(REF_DIR, "prompt_attention_mask.pt"),
                              map_location="cpu", weights_only=True)
        ref_ts = torch.load(os.path.join(REF_DIR, "timestep_step0.pt"),
                            map_location="cpu", weights_only=True)
        ref_noise = torch.load(os.path.join(REF_DIR, "noise_pred_step0.pt"),
                               map_location="cpu", weights_only=True)
        with torch.no_grad():
            our = wrapper(ref_latent.float(), ref_text.float(), ref_mask.long(), ref_ts.long())
        diff = (our - ref_noise).abs().max().item()
        print(f"[parity] wrapper vs reference noise_pred step0 max abs = {diff:.6f}")
        assert diff < 1e-3, "Wrapper forward does not match reference"
        # Use actual reference inputs for tracing so constants are sensible
        latent, text, mask, t = ref_latent.float(), ref_text.float(), ref_mask.long(), ref_ts.long()

    print("[trace]")
    traced = torch.jit.trace(wrapper, (latent, text, mask, t))
    traced = torch.jit.freeze(traced.eval())

    print(f"[convert] precision={args.precision}")
    ct_precision = ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="latent", shape=(1, 32, latent_h, latent_w), dtype=np.float32),
            ct.TensorType(name="encoder_hidden_states", shape=(1, args.seq_len, 2048), dtype=np.float32),
            ct.TensorType(name="encoder_attention_mask", shape=(1, args.seq_len), dtype=np.int32),
            ct.TensorType(name="timestep", shape=(1,), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="noise_pred", dtype=np.float32)],
        compute_precision=ct_precision,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
    )
    mlmodel.short_description = (
        f"Nitro-E E-MMDiT denoiser (1 step), latent [1,32,{latent_h},{latent_w}], "
        f"encoder_hidden_states [1,{args.seq_len},2048], timestep [1]. 4-step distilled variant."
    )
    print(f"[save] -> {args.out}")
    mlmodel.save(args.out)


if __name__ == "__main__":
    main()
