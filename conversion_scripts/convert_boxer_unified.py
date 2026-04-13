#!/usr/bin/env python3
"""
Convert Boxer to a single CoreML model matching the ONNX reference interface.

4 inputs:  image (1,3,960,960), sdp_patches (1,1,60,60),
           bb2d (1,M,4), ray_encoding (1,3600,6)
4 outputs: center (M,3), size (M,3), yaw (M,), confidence (M,)

All outputs are post-processed (meters, radians, probability).
"""

import os, sys, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BOXER_DIR = Path(__file__).parent / "boxer"
sys.path.insert(0, str(BOXER_DIR))
import utils.demo_utils
utils.demo_utils.CKPT_PATH = str(BOXER_DIR / "ckpts")

# Patch LinearKMaskedBias before import
from boxernet.dinov3_wrapper import LinearKMaskedBias
def _pf(self, input):
    if self.bias is not None and hasattr(self, 'bias_mask'):
        m = self.bias_mask.clone()
        m = torch.where(torch.isnan(m), torch.zeros_like(m), m)
        return F.linear(input, self.weight, self.bias * m)
    return F.linear(input, self.weight, self.bias)
LinearKMaskedBias.forward = _pf


def bake_masked_bias(model):
    from boxernet.dinov3_wrapper import LinearKMaskedBias
    reps = []
    for name, mod in model.named_modules():
        if isinstance(mod, LinearKMaskedBias):
            reps.append(name)
    for name in reps:
        parts = name.split('.')
        parent = model
        for p in parts[:-1]: parent = getattr(parent, p)
        old = getattr(parent, parts[-1])
        new_lin = nn.Linear(old.in_features, old.out_features, bias=old.bias is not None)
        new_lin.weight.data.copy_(old.weight.data)
        if old.bias is not None and hasattr(old, 'bias_mask'):
            m = old.bias_mask.clone()
            m = torch.where(torch.isnan(m), torch.zeros_like(m), m)
            new_lin.bias.data.copy_(old.bias.data * m)
        elif old.bias is not None:
            new_lin.bias.data.copy_(old.bias.data)
        setattr(parent, parts[-1], new_lin)
    print(f"  Replaced {len(reps)} LinearKMaskedBias")


class BoxerUnified(nn.Module):
    """Single model: image + sdp_patches + bb2d + ray_encoding → center, size, yaw, confidence."""

    def __init__(self, boxernet):
        super().__init__()
        self.dino_wrapper = boxernet.dino
        self.input2emb = boxernet.input2emb
        self.query2emb = boxernet.query2emb
        self.self_attn = boxernet.self_attn
        self.cross_attn = boxernet.cross_attn
        self.head_net = boxernet.head.net
        self.head_mean = boxernet.head.mean_head
        self.head_logvar = boxernet.head.logvar_head
        self.bbox_min = boxernet.head.bbox_min
        self.bbox_max = boxernet.head.bbox_max
        self.patch_size = 16
        # Bake normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image, sdp_patches, bb2d, ray_encoding):
        """
        Args:
            image: (1, 3, 960, 960) float [0, 1]
            sdp_patches: (1, 1, 60, 60) float — median depth per patch, -1 for invalid
            bb2d: (1, M, 4) float — normalized [xmin, xmax, ymin, ymax] in [0, 1]
            ray_encoding: (1, 3600, 6) float — Plucker rays in voxel frame
        Returns:
            center: (1, M, 3) — center in voxel frame (meters)
            size: (1, M, 3) — dimensions (meters)
            yaw: (1, M) — radians
            confidence: (1, M) — probability
        """
        B = image.shape[0]

        # 1. Run DINOv3
        img_norm = (image - self.mean) / self.std
        x, (H, W) = self.dino_wrapper.model.prepare_tokens_with_masks(img_norm)
        if self.dino_wrapper.model.rope_embed is not None:
            rope_sincos = self.dino_wrapper.model.rope_embed(H=H, W=W)
        else:
            rope_sincos = None
        for blk in self.dino_wrapper.model.blocks:
            x_n = blk.norm1(x)
            x_a = blk.attn(x_n, rope=rope_sincos)
            x = x + blk.ls1(x_a)
            x = x + blk.ls2(blk.mlp(blk.norm2(x)))
        if self.dino_wrapper.model.untie_cls_and_patch_norms:
            n_prefix = self.dino_wrapper.model.n_storage_tokens + 1
            x = torch.cat((self.dino_wrapper.model.cls_norm(x[:, :n_prefix]),
                           self.dino_wrapper.model.norm(x[:, n_prefix:])), dim=1)
        else:
            x = self.dino_wrapper.model.norm(x)
        dino_feat = x[:, self.dino_wrapper.model.n_storage_tokens + 1:]  # (B, 3600, 384)

        # 2. Concatenate features: dino + depth + rays
        sdp_flat = sdp_patches.reshape(B, 1, -1).permute(0, 2, 1)  # (B, 3600, 1)
        features = torch.cat([dino_feat, sdp_flat, ray_encoding], dim=-1)  # (B, 3600, 391)

        # 3. Encode
        input_enc = self.input2emb(features)
        input_enc = self.self_attn(input_enc)

        # 4. Query
        query = self.query2emb(bb2d)
        query = self.cross_attn(query, input_enc)

        # 5. Predict + decode
        h = self.head_net(query)
        mu = self.head_mean(h)
        logvar = torch.clamp(self.head_logvar(h), -10, 3)

        center = mu[..., :3]
        size = torch.sigmoid(mu[..., 3:6]) * (self.bbox_max - self.bbox_min) + self.bbox_min
        size = size.clamp(min=0.05)
        yaw = (math.pi / 2) * torch.tanh(mu[..., 6])
        confidence = 1.0 / (1.0 + torch.exp(logvar.squeeze(-1)))

        return center, size, yaw, confidence


def main():
    import coremltools as ct

    output_dir = Path("converted_models")
    output_dir.mkdir(exist_ok=True)
    max_boxes = 20

    print("Loading BoxerNet...")
    from boxernet.boxernet import BoxerNet
    boxernet = BoxerNet.load_from_checkpoint(
        str(BOXER_DIR / "ckpts" / "boxernet_hw960in4x6d768-wssxpf9p.ckpt"), device="cpu")
    boxernet.eval()
    bake_masked_bias(boxernet)

    print("Creating unified model...")
    model = BoxerUnified(boxernet)
    model.eval()

    # Test
    dummy_img = torch.rand(1, 3, 960, 960)
    dummy_sdp = -torch.ones(1, 1, 60, 60)
    dummy_bb2d = torch.rand(1, max_boxes, 4)
    dummy_rays = torch.randn(1, 3600, 6)
    with torch.no_grad():
        c, s, y, conf = model(dummy_img, dummy_sdp, dummy_bb2d, dummy_rays)
    print(f"  center: {c[0, 0].numpy()}")
    print(f"  size:   {s[0, 0].numpy()}")
    print(f"  yaw:    {y[0, 0].item():.4f}")
    print(f"  conf:   {conf[0, 0].item():.4f}")

    print("Exporting with torch.export...")
    exported = torch.export.export(
        model, (dummy_img, dummy_sdp, dummy_bb2d, dummy_rays), strict=False)
    exported = exported.run_decompositions({})

    print("Converting to CoreML (FP16)...")
    mlmodel = ct.convert(
        exported,
        inputs=[
            ct.TensorType(name="image", shape=(1, 3, 960, 960)),
            ct.TensorType(name="sdp_patches", shape=(1, 1, 60, 60)),
            ct.TensorType(name="bb2d", shape=(1, max_boxes, 4)),
            ct.TensorType(name="ray_encoding", shape=(1, 3600, 6)),
        ],
        outputs=[
            ct.TensorType(name="center"),
            ct.TensorType(name="size"),
            ct.TensorType(name="yaw"),
            ct.TensorType(name="confidence"),
        ],
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS18,
    )

    out_path = output_dir / "Boxer.mlpackage"
    mlmodel.save(str(out_path))
    size_mb = sum(f.stat().st_size for f in out_path.rglob('*') if f.is_file()) / 1e6
    print(f"Saved: {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
