"""Convert the DC-AE f32c32 decoder used by Nitro-E to CoreML (FP16).

Input latent shape for 512px is [1, 32, 16, 16].
The Nitro-E pipeline calls ``vae.decode(latents / vae.config.scaling_factor)``
so we bake the scale division into the wrapper.

Parity target: conversion_scripts/Nitro-E/reference_dump/vae_decode_input.pt /
               conversion_scripts/Nitro-E/reference_dump/vae_decode_output.pt
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

import coremltools as ct
from diffusers import AutoencoderDC
from diffusers.models.autoencoders import autoencoder_dc as _dcae_mod


def _patch_coremltools_cast():
    """coremltools 9.0 `_cast` calls ``dtype(x.val)`` which fails on
    (1,1,...,1)-shape numpy arrays even though they're trivially scalar-like.
    Use ``.item()`` to extract the single value first.
    """
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


def _patch_sana_attn_processor():
    """The stock SanaMultiscaleAttnProcessor2_0 unpacks shapes via
    ``list(hidden_states.size())`` which traces to aten::Int on multi-dim
    tensors and breaks coremltools. Replace with a static-shape version — OK
    because our conversion uses fixed input shape.
    """
    import torch.nn.functional as F
    from diffusers.models.attention_processor import SanaMultiscaleAttnProcessor2_0, SanaMultiscaleLinearAttention

    def __call__(self, attn, hidden_states):
        shp = hidden_states.shape
        B, _, H, W = int(shp[0]), int(shp[1]), int(shp[2]), int(shp[3])
        HW = H * W
        use_linear_attention = HW > attn.attention_head_dim
        residual = hidden_states
        original_dtype = hidden_states.dtype

        hidden_states = hidden_states.movedim(1, -1)
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        hidden_states = torch.cat([query, key, value], dim=3)
        hidden_states = hidden_states.movedim(-1, 1)

        multi_scale_qkv = [hidden_states]
        for block in attn.to_qkv_multiscale:
            multi_scale_qkv.append(block(hidden_states))
        hidden_states = torch.cat(multi_scale_qkv, dim=1)

        if use_linear_attention:
            hidden_states = hidden_states.to(dtype=torch.float32)

        hidden_states = hidden_states.reshape(B, -1, 3 * attn.attention_head_dim, HW)
        query, key, value = hidden_states.chunk(3, dim=2)
        query = attn.nonlinearity(query)
        key = attn.nonlinearity(key)

        if use_linear_attention:
            hidden_states = attn.apply_linear_attention(query, key, value)
            hidden_states = hidden_states.to(dtype=original_dtype)
        else:
            hidden_states = attn.apply_quadratic_attention(query, key, value)

        hidden_states = hidden_states.reshape(B, -1, H, W)
        hidden_states = attn.to_out(hidden_states.movedim(1, -1)).movedim(-1, 1)

        if attn.norm_type == "rms_norm":
            hidden_states = attn.norm_out(hidden_states.movedim(1, -1)).movedim(-1, 1)
        else:
            hidden_states = attn.norm_out(hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states

    SanaMultiscaleAttnProcessor2_0.__call__ = __call__


def _patch_torch_movedim():
    """coremltools 9.0 has no torch `movedim` op. Replace (1,-1) / (-1,1) on 4D
    tensors with the equivalent permute, which traces cleanly."""
    _orig = torch.Tensor.movedim

    def movedim(self, source, destination):
        if self.dim() == 4 and source == 1 and destination == -1:
            return self.permute(0, 2, 3, 1)
        if self.dim() == 4 and source == -1 and destination == 1:
            return self.permute(0, 3, 1, 2)
        return _orig(self, source, destination)

    torch.Tensor.movedim = movedim


def _patch_decoder_forward():
    """Drop ``output_size=`` from the decoder's repeat_interleave. That kwarg
    produces an ``aten::Int`` node on a multi-dim tensor that coremltools
    cannot cast."""
    Decoder = _dcae_mod.Decoder

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.in_shortcut:
            x = hidden_states.repeat_interleave(self.in_shortcut_repeats, dim=1)
            hidden_states = self.conv_in(hidden_states) + x
        else:
            hidden_states = self.conv_in(hidden_states)
        for up_block in reversed(self.up_blocks):
            hidden_states = up_block(hidden_states)
        hidden_states = self.norm_out(hidden_states.movedim(1, -1)).movedim(-1, 1)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        return hidden_states

    Decoder.forward = forward

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(THIS_DIR, "Nitro-E", "reference_dump")


class DCAEDecoderWrapper(nn.Module):
    """Wrap DC-AE so CoreML input matches the pipeline-facing latent.

    The pipeline stores latents that need ``/ scaling_factor`` before decode,
    so we accept that un-scaled latent and divide internally. That keeps the
    Swift driver simple: it just forwards whatever latent the transformer
    produced.
    """

    def __init__(self, vae: AutoencoderDC):
        super().__init__()
        self.decoder = vae.decoder
        self.scaling_factor = float(vae.config.scaling_factor)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        z = latent / self.scaling_factor
        image = self.decoder(z)
        return image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(THIS_DIR, "NitroE_VAEDecoder.mlpackage"))
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    args = ap.parse_args()

    latent_h = args.resolution // 32
    latent_w = args.resolution // 32

    _patch_coremltools_cast()
    _patch_torch_movedim()
    _patch_decoder_forward()
    _patch_sana_attn_processor()

    print(f"[load] AutoencoderDC (dc-ae-f32c32-sana-1.0-diffusers)")
    vae = AutoencoderDC.from_pretrained(
        "mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers",
        torch_dtype=torch.float32,
    )
    vae.eval()
    wrapper = DCAEDecoderWrapper(vae).eval()

    # Parity check against the dumped reference before we trace.
    if os.path.exists(os.path.join(REF_DIR, "vae_decode_input.pt")):
        ref_in = torch.load(os.path.join(REF_DIR, "vae_decode_input.pt"),
                            map_location="cpu", weights_only=True)
        ref_out = torch.load(os.path.join(REF_DIR, "vae_decode_output.pt"),
                             map_location="cpu", weights_only=True)
        # vae_decode_input is z = latent / scaling_factor; undo so the wrapper's
        # division matches.
        pipeline_latent = ref_in * wrapper.scaling_factor
        with torch.no_grad():
            our_out = wrapper(pipeline_latent)
        diff = (our_out - ref_out).abs().max().item()
        print(f"[parity] wrapper vs pipeline decode max abs diff = {diff:.6f}")
        assert diff < 1e-4, "Wrapper decode does not match reference"
    else:
        print("[parity] no reference dump found, skipping parity check")
        pipeline_latent = torch.randn(1, 32, latent_h, latent_w)

    print("[trace]")
    traced = torch.jit.trace(wrapper, pipeline_latent)
    traced = torch.jit.freeze(traced.eval())

    print(f"[convert] precision={args.precision}")
    ct_precision = ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="latent",
                shape=(1, 32, latent_h, latent_w),
                dtype=np.float32,
            ),
        ],
        outputs=[ct.TensorType(name="image", dtype=np.float32)],
        compute_precision=ct_precision,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
    )
    mlmodel.short_description = (
        f"Nitro-E DC-AE f32c32 decoder — latent [1,32,{latent_h},{latent_w}] -> image [1,3,{args.resolution},{args.resolution}]. "
        f"Division by scaling_factor {wrapper.scaling_factor} is baked in."
    )
    print(f"[save] -> {args.out}")
    mlmodel.save(args.out)


if __name__ == "__main__":
    main()
