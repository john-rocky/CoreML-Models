"""
Convert MoGe-2 (Microsoft, CVPR'25) to Core ML.

Default variant: `Ruicheng/moge-2-vitb-normal` (104M, DINOv2 ViT-B/14 + normal +
metric scale heads). Resolution is locked to a fixed square so the
aspect-ratio-aware num_tokens path collapses to constants.

The wrapper:
  - Takes a (1, 3, H, W) image in [0, 1] range. ImageNet normalization is
    already inside DINOv2Encoder, so we feed [0, 1] directly.
  - Pre-computes the bicubic-interpolated DINOv2 positional embedding for the
    fixed token grid and replaces `interpolate_pos_encoding` with a constant
    lookup, so the traced graph never hits bicubic + antialias.
  - Hard-codes base_h = base_w = H // 14 (= 36 at 504x504).
  - Returns a flat tuple: (points, depth, normal, mask, metric_scale).
    `points` is (1, H, W, 3); `depth` is points[..., 2] cloned out so the
    Swift side does not have to slice a 4-D tensor.

Recovery of focal / shift / camera intrinsics is left to the Swift driver.

Usage:
  python convert_moge2.py                              # ViT-B normal, 504x504
  python convert_moge2.py --variant vits-normal --size 392
  python convert_moge2.py --output MoGe2_ViTB_504.mlpackage
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import coremltools as ct
from coremltools.converters.mil.frontend.torch import ops as _ct_ops
from coremltools.converters.mil import Builder as mb

REPO = Path(__file__).resolve().parent / "MoGe"
sys.path.insert(0, str(REPO))

from moge.model.v2 import MoGeModel  # noqa: E402
from moge.utils.geometry_torch import normalized_view_plane_uv  # noqa: E402


# ============================================================
# coremltools 9.0 patch: `int` op for multi-dim shape casts.
# DINOv2 emits int casts on a 2-element shape tensor (h, w) when
# building positional indices; the stock converter assumes scalars.
# Same patch as conversion_scripts/convert_sinsr.py.
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
# Pre-compute and freeze DINOv2 positional embedding
# ============================================================

def freeze_pos_embed(model: MoGeModel, base_h: int, base_w: int) -> None:
    """Replace `interpolate_pos_encoding` with a constant lookup.

    DINOv2's stock implementation does a bicubic + antialias resize of the
    pretrained pos_embed every forward call. coremltools cannot trace bicubic
    + antialias cleanly, but the result depends only on the (h, w) we already
    fixed at conversion time, so we just bake it once and return it.
    """
    backbone = model.encoder.backbone
    img_h, img_w = base_h * backbone.patch_size, base_w * backbone.patch_size
    # Build a dummy input that has the correct (h, w); only the *shape* matters
    # because interpolate_pos_encoding only reads x.shape[1] and the (h, w)
    # arguments.
    dummy = torch.zeros(1, 3, img_h, img_w)
    tokens = backbone.patch_embed(dummy)
    # Mimic the cls-token concat that happens in prepare_tokens_with_masks so
    # interpolate_pos_encoding sees the correct npatch.
    cls = backbone.cls_token.expand(tokens.shape[0], -1, -1)
    x = torch.cat([cls, tokens], dim=1)
    with torch.no_grad():
        pos = backbone.interpolate_pos_encoding(x, img_h, img_w)
    backbone.register_buffer("_frozen_pos_embed", pos.detach().clone(), persistent=False)

    def _frozen_interp(self, x, h, w):  # noqa: ARG001
        return self._frozen_pos_embed

    backbone.interpolate_pos_encoding = _frozen_interp.__get__(backbone, type(backbone))


# ============================================================
# Wrapper
# ============================================================

class MoGe2Wrapper(nn.Module):
    """Stateless wrapper exposing MoGe-2 as a single CoreML model.

    Mirrors MoGeModel.forward but with a fixed square resolution, hard-coded
    `base_h`/`base_w`, no dict outputs and no Python conditionals.
    """

    def __init__(self, model: MoGeModel, size: int):
        super().__init__()
        assert size % 14 == 0, f"size must be a multiple of 14 (DINOv2 patch); got {size}"
        self.model = model
        self.size = size
        self.base = size // 14  # 504 -> 36

        # Pre-compute UV grids for all 5 pyramid levels (depend only on shape).
        for level in range(5):
            uv = normalized_view_plane_uv(
                width=self.base * 2 ** level,
                height=self.base * 2 ** level,
                aspect_ratio=1.0,
                dtype=torch.float32,
            )
            uv = uv.permute(2, 0, 1).unsqueeze(0).contiguous()  # (1, 2, h, w)
            self.register_buffer(f"uv_{level}", uv, persistent=False)

    def forward(self, image: torch.Tensor):
        # Encoder. Outputs (B, dim_out, base, base) feature map plus cls token.
        features_l0, cls_token = self.model.encoder(
            image, self.base, self.base, return_class_token=True
        )

        # Build the 5-level feature pyramid: only level 0 has encoder output;
        # levels 1..4 are pure UV until the neck mixes them in.
        levels = [features_l0, None, None, None, None]
        for level in range(5):
            uv = getattr(self, f"uv_{level}").expand(image.shape[0], -1, -1, -1)
            if levels[level] is None:
                levels[level] = uv
            else:
                levels[level] = torch.cat([levels[level], uv], dim=1)

        features = self.model.neck(levels)

        points = self.model.points_head(features)[-1]
        normal = self.model.normal_head(features)[-1]
        mask = self.model.mask_head(features)[-1]
        metric_scale = self.model.scale_head(cls_token)

        # Resize back to input resolution.
        points = F.interpolate(points, (self.size, self.size), mode="bilinear", align_corners=False)
        normal = F.interpolate(normal, (self.size, self.size), mode="bilinear", align_corners=False)
        mask = F.interpolate(mask, (self.size, self.size), mode="bilinear", align_corners=False)

        # Match MoGeModel.forward postprocessing for remap='exp'.
        points = points.permute(0, 2, 3, 1)  # (B, H, W, 3)
        xy, z = points.split([2, 1], dim=-1)
        z = torch.exp(z)
        points = torch.cat([xy * z, z], dim=-1)

        # Pull depth out before downstream so Swift does not have to slice
        # a 4-D tensor on its own.
        depth = points[..., 2]  # (B, H, W)

        normal = normal.permute(0, 2, 3, 1)  # (B, H, W, 3)
        normal = F.normalize(normal, dim=-1)

        mask = mask.squeeze(1).sigmoid()  # (B, H, W)
        metric_scale = metric_scale.squeeze(1).exp()  # (B,)

        return points, depth, normal, mask, metric_scale


# ============================================================
# Main
# ============================================================

VARIANTS = {
    "vits-normal": "Ruicheng/moge-2-vits-normal",
    "vitb-normal": "Ruicheng/moge-2-vitb-normal",
    "vitl-normal": "Ruicheng/moge-2-vitl-normal",
    "vitl": "Ruicheng/moge-2-vitl",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="vitb-normal", choices=list(VARIANTS.keys()))
    p.add_argument("--size", type=int, default=504, help="square input size (must be multiple of 14)")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--quantize", action="store_true", help="apply linear FP16 -> INT8 weight quant")
    return p.parse_args()


def main():
    args = parse_args()
    repo_id = VARIANTS[args.variant]
    print(f"[1/5] Loading MoGe-2 from {repo_id}")
    model = MoGeModel.from_pretrained(repo_id)
    model.eval()
    # Use PyTorch native SDPA so the conversion sees an op coremltools knows.
    model.enable_pytorch_native_sdpa()
    # Disable the bicubic+antialias fast path inside DINOv2.
    model.onnx_compatible_mode = True

    base = args.size // 14
    print(f"[2/5] Freezing pos_embed for {args.size}x{args.size} ({base}x{base} tokens)")
    freeze_pos_embed(model, base, base)

    print("[3/5] Building wrapper and tracing")
    wrapper = MoGe2Wrapper(model, args.size).eval()
    example = torch.rand(1, 3, args.size, args.size)
    with torch.no_grad():
        ref_points, ref_depth, ref_normal, ref_mask, ref_scale = wrapper(example)
        traced = torch.jit.trace(wrapper, example, strict=False)
        # Sanity check the traced module against eager mode.
        t_points, t_depth, t_normal, t_mask, t_scale = traced(example)
        for name, ref, got in [
            ("points", ref_points, t_points),
            ("depth", ref_depth, t_depth),
            ("normal", ref_normal, t_normal),
            ("mask", ref_mask, t_mask),
            ("scale", ref_scale, t_scale),
        ]:
            err = (ref - got).abs().max().item()
            print(f"        trace parity {name:8s}: max abs err = {err:.3e}")
            assert err < 1e-4, f"trace parity broke for {name}"

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
            ct.TensorType(name="points"),
            ct.TensorType(name="depth"),
            ct.TensorType(name="normal"),
            ct.TensorType(name="mask"),
            ct.TensorType(name="metric_scale"),
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

    out_path = args.output or f"MoGe2_{args.variant.replace('-', '_')}_{args.size}.mlpackage"
    print(f"[5/5] Saving to {out_path}")
    mlmodel.save(out_path)
    print("Done.")


if __name__ == "__main__":
    main()
