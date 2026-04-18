"""
Convert Depth Anything 3 (ByteDance-Seed, ICLR'26 oral) to Core ML.

This script targets the monocular depth subgraph of the DA3 Main Series.
Default variant is DA3-Small (Apache 2.0, 0.08B params, DINOv2 ViT-S/14 +
DualDPT head). The multi-view, camera, ray, sky and Gaussian branches are
intentionally dropped — we only need single-image relative depth for iOS.

Wrapper:
  - Takes (1, 3, H, W) RGB in [0, 255]; ImageNet normalization is baked in.
  - Unsqueezes to (1, 1, 3, H, W) so the upstream B, S, 3, H, W signature
    keeps working (S=1 for monocular).
  - Freezes the bicubic-interpolated DINOv2 positional embedding for the
    fixed token grid (same pattern as convert_moge2.py).
  - Replaces the in-place camera-token write inside the backbone
    (`x[:, :, 0] = cam_token`) with a torch.cat equivalent so torch.jit.trace
    captures a clean graph.
  - Returns (depth, conf) with shape (1, H, W) each, squeezed of the S=1
    batch dimension.

Usage:
  python convert_depth_anything_v3.py                    # DA3-Small, 504x504
  python convert_depth_anything_v3.py --size 518
  python convert_depth_anything_v3.py --output DA3_Small_504.mlpackage
"""
import argparse
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import coremltools as ct
from coremltools.converters.mil.frontend.torch import ops as _ct_ops
from coremltools.converters.mil import Builder as mb

REPO = Path(__file__).resolve().parent / "DepthAnythingV3" / "src"
sys.path.insert(0, str(REPO))

# Bypass depth_anything_3.api entirely — it pulls in moviepy / pycolmap / evo
# / trimesh / gsplat for optional GLB / 3DGS / COLMAP export, none of which
# are needed to trace the model. Load the net directly from config + HF
# safetensors.
from depth_anything_3.cfg import create_object, load_config  # noqa: E402
from depth_anything_3.registry import MODEL_REGISTRY  # noqa: E402
from depth_anything_3.model.da3 import DepthAnything3Net  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from safetensors.torch import load_file as load_safetensors  # noqa: E402


# ============================================================
# coremltools 9.0 patch: `int` op for multi-dim shape casts.
# Same patch as convert_moge2.py / convert_sinsr.py.
# ============================================================

def _patched_int(context, node):
    inputs = _ct_ops._get_inputs(context, node)
    x = inputs[0]
    if x.val is not None:
        val = x.val
        if isinstance(val, np.ndarray):
            val = int(val.item()) if val.ndim == 0 else int(val.flat[0])
        else:
            val = int(val)
        res = mb.const(val=np.int32(val), name=node.name)
    else:
        res = mb.cast(x=x, dtype="int32", name=node.name)
    context.add(res)


_ct_ops._TORCH_OPS_REGISTRY.register_func(_patched_int, torch_alias=["int"], override=True)


# ============================================================
# Replace torch.meshgrid-based UV grid (coremltools trips on it) with
# an explicit broadcast+stack that is trace-friendly.
# ============================================================

def _coreml_safe_create_uv_grid(
    width: int,
    height: int,
    aspect_ratio: float = None,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> torch.Tensor:
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)
    diag_factor = (aspect_ratio ** 2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor
    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height
    x_coords = torch.linspace(left_x, right_x, steps=width, dtype=dtype, device=device)
    y_coords = torch.linspace(top_y, bottom_y, steps=height, dtype=dtype, device=device)
    # Equivalent to torch.meshgrid(x_coords, y_coords, indexing="xy") +
    # stack(dim=-1): output is (height, width, 2), [..., 0] = x, [..., 1] = y.
    uu = x_coords.view(1, width).expand(height, width)
    vv = y_coords.view(height, 1).expand(height, width)
    return torch.stack((uu, vv), dim=-1)


def patch_head_utils():
    """Swap the meshgrid-based create_uv_grid with the trace-friendly one.
    DualDPT._add_pos_embed imports it directly, so we replace the symbol on
    the module it was imported into as well."""
    from depth_anything_3.model.utils import head_utils as _hu
    from depth_anything_3.model import dualdpt as _dualdpt
    _hu.create_uv_grid = _coreml_safe_create_uv_grid
    _dualdpt.create_uv_grid = _coreml_safe_create_uv_grid


# ============================================================
# Freeze DINOv2 positional embedding
# ============================================================

def freeze_pos_embed(backbone, base_h: int, base_w: int) -> None:
    """Bake the bicubic pos_embed interpolation as a constant buffer."""
    img_h, img_w = base_h * backbone.patch_size, base_w * backbone.patch_size
    dummy = torch.zeros(1, 3, img_h, img_w)
    tokens = backbone.patch_embed(dummy)
    cls = backbone.cls_token.expand(tokens.shape[0], -1, -1)
    x = torch.cat([cls, tokens], dim=1)
    with torch.no_grad():
        pos = backbone.interpolate_pos_encoding(x, img_h, img_w)
    backbone.register_buffer("_frozen_pos_embed", pos.detach().clone(), persistent=False)

    def _frozen_interp(self, x, h, w):  # noqa: ARG001
        return self._frozen_pos_embed

    backbone.interpolate_pos_encoding = types.MethodType(_frozen_interp, backbone)


# ============================================================
# Freeze RoPE 2D position grid
# ============================================================

def freeze_rope_positions(backbone, base_h: int, base_w: int) -> None:
    """Pre-compute the RoPE position grid (uses `torch.cartesian_prod` which
    coremltools does not implement) and stash it as a buffer, then replace
    `_prepare_rope` with a constant lookup."""
    if backbone.rope is None:
        return

    y = torch.arange(base_h)
    x = torch.arange(base_w)
    # (base_h * base_w, 2) y-x coordinates, matching PositionGetter.
    positions = torch.stack(torch.meshgrid(y, x, indexing="ij"), dim=-1).reshape(-1, 2)
    # Add the cls / camera / register token slot at the front.
    # Shape (1, N + patch_start_idx, 2). Values: special tokens all zeros,
    # patch tokens indexed from 1 (matches vision_transformer _prepare_rope).
    patch_start_idx = backbone.patch_start_idx
    pos_patches = (positions + 1).unsqueeze(0)  # (1, N, 2)
    pos_special = torch.zeros(1, patch_start_idx, 2, dtype=positions.dtype)
    pos_full = torch.cat([pos_special, pos_patches], dim=1)  # (1, N + pat, 2)
    pos_nodiff_full = torch.cat(
        [pos_special, torch.ones(1, positions.shape[0], 2, dtype=positions.dtype)], dim=1
    )

    # (B, S, N + pat, 2) — for our monocular wrapper B=S=1 so we can drop
    # the extra dims and broadcast.
    pos_full = pos_full.unsqueeze(1)  # (1, 1, N+pat, 2)
    pos_nodiff_full = pos_nodiff_full.unsqueeze(1)

    backbone.register_buffer("_frozen_rope_pos", pos_full.detach().clone(), persistent=False)
    backbone.register_buffer(
        "_frozen_rope_pos_nodiff", pos_nodiff_full.detach().clone(), persistent=False
    )

    def _frozen_prepare_rope(self, B, S, H, W, device):  # noqa: ARG001
        return self._frozen_rope_pos, self._frozen_rope_pos_nodiff

    backbone._prepare_rope = types.MethodType(_frozen_prepare_rope, backbone)


# ============================================================
# Replace in-place camera-token write with torch.cat
# ============================================================

def patch_backbone_forward(backbone):
    """Monkey-patch `_get_intermediate_layers_not_chunked` to avoid the
    in-place slice assignment at alt_start, which traces poorly.

    Also hard-codes the S=1 path — reference-view reordering never fires for
    S<3 and the tail `'b_idx' in locals()` check is unreachable.
    """
    import torch as _torch

    def _patched(self, x, n=1, export_feat_layers=[], **kwargs):
        B, S, _, H, W = x.shape
        x = self.prepare_tokens_with_masks(x)
        output, total_block_len, aux_output = [], len(self.blocks), []
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )
        pos, pos_nodiff = self._prepare_rope(B, S, H, W, x.device)

        local_x = x
        for i, blk in enumerate(self.blocks):
            if i < self.rope_start or self.rope is None:
                g_pos, l_pos = None, None
            else:
                g_pos = pos_nodiff
                l_pos = pos

            if self.alt_start != -1 and i == self.alt_start:
                # Build camera token — for S=1 just take the reference slot.
                if kwargs.get("cam_token", None) is not None:
                    cam_token = kwargs.get("cam_token")
                else:
                    ref_token = self.camera_token[:, :1].expand(B, -1, -1)
                    if S > 1:
                        src_token = self.camera_token[:, 1:].expand(B, S - 1, -1)
                        cam_token = _torch.cat([ref_token, src_token], dim=1)
                    else:
                        cam_token = ref_token
                # Replace the in-place `x[:, :, 0] = cam_token` with cat.
                # x: (B, S, N, C), cam_token: (B, S, C)
                cam_token = cam_token.unsqueeze(2)  # (B, S, 1, C)
                x = _torch.cat([cam_token, x[:, :, 1:]], dim=2)

            if self.alt_start != -1 and i >= self.alt_start and i % 2 == 1:
                x = self.process_attention(
                    x, blk, "global", pos=g_pos, attn_mask=kwargs.get("attn_mask", None)
                )
            else:
                x = self.process_attention(x, blk, "local", pos=l_pos)
                local_x = x

            if i in blocks_to_take:
                out_x = _torch.cat([local_x, x], dim=-1) if self.cat_token else x
                output.append((out_x[:, :, 0], out_x))
            if i in export_feat_layers:
                aux_output.append(x)
        return output, aux_output

    backbone._get_intermediate_layers_not_chunked = types.MethodType(_patched, backbone)


# ============================================================
# Wrapper
# ============================================================

# ImageNet normalization constants (from DA3 input_processor).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class MonoDepthWrapper(nn.Module):
    """Stateless wrapper around DepthAnything3Net that only exposes the
    monocular depth head. Input is RGB in [0, 1]; ImageNet normalization is
    applied inside the wrapper so Core ML can use a simple ImageType(scale).
    """

    def __init__(self, net: DepthAnything3Net, size: int):
        super().__init__()
        assert size % 14 == 0, f"size must be a multiple of 14, got {size}"
        self.net = net
        self.size = size

        mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def forward(self, image: torch.Tensor):
        # image: (1, 3, H, W) in [0, 1]
        x = (image - self.mean) / self.std
        x = x.unsqueeze(1)  # (B, 1, 3, H, W) — S=1 monocular

        # Backbone. Pass cam_token=None so the monkey-patched branch builds
        # it from the learnt `camera_token` parameter.
        feats, _aux = self.net.backbone(
            x, cam_token=None, export_feat_layers=[], ref_view_strategy="first"
        )

        # DualDPT head — returns dict with depth + depth_conf (+ aux ray+conf).
        out = self.net.head(feats, self.size, self.size, patch_start_idx=0)

        # depth / conf come back as (B=1, S=1, H, W). Squeeze S.
        depth = out.depth.squeeze(1)
        conf = out.depth_conf.squeeze(1)
        return depth, conf


# ============================================================
# Main
# ============================================================

VARIANTS = {
    # variant key -> (HF repo id, config name in MODEL_REGISTRY)
    "small": ("depth-anything/DA3-SMALL", "da3-small"),
    "base": ("depth-anything/DA3-BASE", "da3-base"),
    "large": ("depth-anything/DA3-LARGE-1.1", "da3-large"),
    "mono-large": ("depth-anything/DA3MONO-LARGE", "da3mono-large"),
}


def load_net(repo_id: str, config_name: str) -> DepthAnything3Net:
    """Build DepthAnything3Net from its yaml config and load safetensors
    weights directly from HF. Avoids importing depth_anything_3.api."""
    cfg_path = MODEL_REGISTRY[config_name]
    print(f"        config: {cfg_path}")
    net = create_object(load_config(cfg_path))
    net.eval()

    ckpt_path = hf_hub_download(repo_id, filename="model.safetensors")
    print(f"        checkpoint: {ckpt_path}")
    state = load_safetensors(ckpt_path)

    # DepthAnything3 stores the inner net under self.model.*, so strip that
    # prefix if present.
    if any(k.startswith("model.") for k in state):
        state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        print(f"        missing keys: {len(missing)} (first 3: {missing[:3]})")
    if unexpected:
        print(f"        unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")
    return net


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="small", choices=list(VARIANTS.keys()))
    p.add_argument("--size", type=int, default=504, help="square input size (multiple of 14)")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--quantize", action="store_true", help="apply INT8 weight quant")
    return p.parse_args()


def main():
    args = parse_args()
    repo_id, cfg_name = VARIANTS[args.variant]
    print(f"[1/5] Loading DepthAnything3 from {repo_id} (config: {cfg_name})")
    net = load_net(repo_id, cfg_name)
    backbone = net.backbone.pretrained  # DinoVisionTransformer

    assert args.size % backbone.patch_size == 0, (
        f"size={args.size} must be a multiple of patch={backbone.patch_size}"
    )
    base = args.size // backbone.patch_size

    print("[2-/5] Swapping create_uv_grid to avoid torch.meshgrid in trace")
    patch_head_utils()

    print(f"[2/5] Freezing pos_embed for {args.size}x{args.size} ({base}x{base} tokens)")
    freeze_pos_embed(backbone, base, base)

    print("[2b/5] Freezing RoPE position grid (cartesian_prod is unsupported in CoreML)")
    freeze_rope_positions(backbone, base, base)

    print("[2c/5] Patching backbone forward to remove in-place camera-token write")
    patch_backbone_forward(backbone)

    print("[3/5] Building wrapper and tracing")
    wrapper = MonoDepthWrapper(net, args.size).eval()
    example = torch.rand(1, 3, args.size, args.size)
    with torch.no_grad():
        ref_depth, ref_conf = wrapper(example)
        print(f"        eager depth: {ref_depth.shape}, conf: {ref_conf.shape}")
        print(f"        depth range: [{ref_depth.min().item():.3e}, {ref_depth.max().item():.3e}]")

        traced = torch.jit.trace(wrapper, example, strict=False)
        t_depth, t_conf = traced(example)
        for name, ref, got in [("depth", ref_depth, t_depth), ("conf", ref_conf, t_conf)]:
            err = (ref - got).abs().max().item()
            print(f"        trace parity {name:6s}: max abs err = {err:.3e}")
            assert err < 1e-3, f"trace parity broke for {name}: {err}"

    print("[4/5] Converting to Core ML")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, args.size, args.size),
                scale=1.0 / 255.0,
                color_layout=ct.colorlayout.RGB,
            )
        ],
        outputs=[
            ct.TensorType(name="depth"),
            ct.TensorType(name="confidence"),
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
    )

    if args.quantize:
        print("        applying INT8 weight quantization")
        from coremltools.optimize.coreml import (
            OpLinearQuantizerConfig,
            OptimizationConfig,
            linear_quantize_weights,
        )
        cfg = OptimizationConfig(
            global_config=OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
        )
        mlmodel = linear_quantize_weights(mlmodel, cfg)

    out_path = (
        args.output
        or f"DepthAnythingV3_{args.variant.replace('-', '_')}_{args.size}.mlpackage"
    )
    print(f"[5/5] Saving to {out_path}")
    mlmodel.save(out_path)
    print("Done.")


if __name__ == "__main__":
    main()
