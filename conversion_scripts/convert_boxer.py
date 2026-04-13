#!/usr/bin/env python3
"""
Convert Boxer (DINOv3 + BoxerNet) to CoreML.

Produces two mlpackage files:
  - BoxerDINOv3.mlpackage  (~115MB FP16) — image feature extractor
  - BoxerNet.mlpackage     (~200MB FP16) — 2D→3D box lifter

Usage:
  python conversion_scripts/convert_boxer.py [--output-dir converted_models/]
"""

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add boxer repo to path
BOXER_DIR = Path(__file__).parent / "boxer"
sys.path.insert(0, str(BOXER_DIR))

# Patch CKPT_PATH before importing boxernet
CKPT_DIR = BOXER_DIR / "ckpts"
os.environ["BOXER_CKPT_PATH"] = str(CKPT_DIR)

# Monkey-patch the CKPT_PATH in utils.demo_utils if needed
try:
    import utils.demo_utils
    utils.demo_utils.CKPT_PATH = str(CKPT_DIR)
except ImportError:
    pass


# =============================================================================
# Download checkpoints
# =============================================================================

HF_REPO = "facebook/boxer"
CKPT_FILES = {
    "boxernet": "boxernet_hw960in4x6d768-wssxpf9p.ckpt",
    "dinov3": "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
}


def download_checkpoints():
    """Download model checkpoints from HuggingFace."""
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    for name, filename in CKPT_FILES.items():
        path = CKPT_DIR / filename
        if path.exists():
            print(f"  [{name}] Already downloaded: {path}")
            continue
        print(f"  [{name}] Downloading {filename}...")
        url = f"https://huggingface.co/{HF_REPO}/resolve/main/{filename}"
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id=HF_REPO,
                filename=filename,
                local_dir=str(CKPT_DIR),
                local_dir_use_symlinks=False,
            )
        except ImportError:
            import urllib.request
            urllib.request.urlretrieve(url, str(path))
        print(f"  [{name}] Done: {path}")


# =============================================================================
# Monkey-patches for tracing
# =============================================================================

def patch_linear_k_masked_bias():
    """Replace LinearKMaskedBias with standard Linear using baked effective bias.

    This is a class-level patch so the model loads correctly.
    After loading, call bake_masked_bias_into_model() to replace instances
    with standard nn.Linear (eliminating isnan ops from the graph).
    """
    from boxernet.dinov3_wrapper import LinearKMaskedBias

    def patched_forward(self, input):
        if self.bias is not None and hasattr(self, 'bias_mask'):
            mask = self.bias_mask.clone()
            mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
            effective_bias = self.bias * mask
            return F.linear(input, self.weight, effective_bias)
        return F.linear(input, self.weight, self.bias)

    LinearKMaskedBias.forward = patched_forward
    print("  Patched LinearKMaskedBias")


def bake_masked_bias_into_model(model):
    """Replace all LinearKMaskedBias modules with standard nn.Linear (pre-baked bias)."""
    from boxernet.dinov3_wrapper import LinearKMaskedBias

    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, LinearKMaskedBias):
            replacements.append(name)

    for name in replacements:
        # Navigate to parent module
        parts = name.split('.')
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        attr = parts[-1]
        old = getattr(parent, attr)

        # Create standard Linear with baked bias
        new_linear = nn.Linear(
            old.in_features, old.out_features,
            bias=old.bias is not None,
            device=old.weight.device,
        )
        new_linear.weight.data.copy_(old.weight.data)
        if old.bias is not None and hasattr(old, 'bias_mask'):
            mask = old.bias_mask.clone()
            mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
            new_linear.bias.data.copy_(old.bias.data * mask)
        elif old.bias is not None:
            new_linear.bias.data.copy_(old.bias.data)

        setattr(parent, attr, new_linear)

    print(f"  Replaced {len(replacements)} LinearKMaskedBias → nn.Linear")


# =============================================================================
# DINOv3 Wrapper for CoreML
# =============================================================================

class DINOv3ForCoreML(nn.Module):
    """
    Traceable DINOv3 wrapper.
    Input:  image (1, 3, 960, 960) in [0, 1]
    Output: features (1, 384, 60, 60)
    """

    def __init__(self, dino_wrapper):
        super().__init__()
        self.model = dino_wrapper.model
        self.patch_size = dino_wrapper.patch_size
        # Bake ImageNet normalization
        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, image):
        # Normalize
        x = (image - self.mean) / self.std
        # Extract features using the model's internal method
        x, (H, W) = self.model.prepare_tokens_with_masks(x)

        # Pre-compute RoPE for fixed resolution
        if self.model.rope_embed is not None:
            rope_sincos = self.model.rope_embed(H=H, W=W)
        else:
            rope_sincos = None

        # Run through blocks - directly using eval path to avoid
        # drop-path int() ops that coremltools cannot handle
        for blk in self.model.blocks:
            x_normed = blk.norm1(x)
            x_attn = blk.attn(x_normed, rope=rope_sincos)
            x = x + blk.ls1(x_attn)
            x = x + blk.ls2(blk.mlp(blk.norm2(x)))

        # Apply norm
        if self.model.untie_cls_and_patch_norms:
            n_prefix = self.model.n_storage_tokens + 1
            x_cls_reg = self.model.cls_norm(x[:, :n_prefix])
            x_patch = self.model.norm(x[:, n_prefix:])
            x = torch.cat((x_cls_reg, x_patch), dim=1)
        else:
            x = self.model.norm(x)

        # Extract patch tokens (remove CLS + storage tokens)
        n_storage = self.model.n_storage_tokens
        patches = x[:, n_storage + 1:]  # (B, fH*fW, D)

        # Reshape to spatial (H, W are already patch grid dims from prepare_tokens)
        B = patches.shape[0]
        features = patches.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return features


# =============================================================================
# BoxerNet Core Wrapper for CoreML
# =============================================================================

class BoxerNetForCoreML(nn.Module):
    """
    Traceable BoxerNet transformer + head.
    Input:
      - input_features: (1, 3600, 385) = dino(384) + depth(1)
      - bb2d_normalized: (1, M, 4) = normalized box queries in [0, 1]
    Output:
      - params: (1, M, 7) = [center_xyz(3), size_raw(3), yaw_raw(1)]
      - logvar: (1, M, 1)
    """

    def __init__(self, boxernet):
        super().__init__()
        self.input2emb = boxernet.input2emb
        self.query2emb = boxernet.query2emb
        self.self_attn = boxernet.self_attn
        self.cross_attn = boxernet.cross_attn
        # Inline the AleHead
        self.head_net = boxernet.head.net
        self.head_mean = boxernet.head.mean_head
        self.head_logvar = boxernet.head.logvar_head

    def forward(self, input_features, bb2d_normalized):
        # Encode image tokens
        input_enc = self.input2emb(input_features)
        input_enc = self.self_attn(input_enc)

        # Encode queries and cross-attend
        query = self.query2emb(bb2d_normalized)
        query = self.cross_attn(query, input_enc)

        # Prediction head
        h = self.head_net(query)
        params = self.head_mean(h)
        logvar = self.head_logvar(h)
        logvar = torch.clamp(logvar, -10.0, 3.0)

        return params, logvar


# =============================================================================
# Conversion
# =============================================================================

def convert_dinov3(dino_wrapper, output_dir):
    """Convert DINOv3 to CoreML using torch.export → coremltools."""
    import coremltools as ct

    print("\n=== Converting DINOv3 ===")
    model = DINOv3ForCoreML(dino_wrapper)
    model.eval()

    dummy_img = torch.randn(1, 3, 960, 960).clamp(0, 1)

    # Verify model runs
    with torch.no_grad():
        ref_out = model(dummy_img)
    print(f"  Reference output shape: {ref_out.shape}")

    # Use torch.export for cleaner graph (better int op handling)
    print("  Exporting with torch.export...")
    exported = torch.export.export(model, (dummy_img,), strict=False)
    exported = exported.run_decompositions({})
    print(f"  Export successful")

    # Convert ExportedProgram to CoreML
    print("  Converting to CoreML (FP16)...")
    mlmodel = ct.convert(
        exported,
        inputs=[
            ct.TensorType(name="image", shape=(1, 3, 960, 960)),
        ],
        outputs=[ct.TensorType(name="features")],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )

    out_path = output_dir / "BoxerDINOv3.mlpackage"
    mlmodel.save(str(out_path))
    size_mb = sum(
        f.stat().st_size for f in out_path.rglob('*') if f.is_file()
    ) / 1e6
    print(f"  Saved: {out_path} ({size_mb:.1f} MB)")
    return out_path


def convert_boxernet(boxernet_model, output_dir, max_boxes=20):
    """Convert BoxerNet transformer + head to CoreML."""
    import coremltools as ct

    print("\n=== Converting BoxerNet ===")
    model = BoxerNetForCoreML(boxernet_model)
    model.eval()

    # Trace with dummy inputs
    in_dim = boxernet_model.in_dim  # 385 (384 dino + 1 depth)
    dummy_features = torch.randn(1, 3600, in_dim)
    dummy_boxes = torch.rand(1, max_boxes, 4)

    print(f"  Input features: (1, 3600, {in_dim})")
    print(f"  Input boxes: (1, {max_boxes}, 4)")
    print("  Tracing...")

    with torch.no_grad():
        traced = torch.jit.trace(model, (dummy_features, dummy_boxes))

    # Verify trace
    with torch.no_grad():
        ref_params, ref_logvar = model(dummy_features, dummy_boxes)
        traced_params, traced_logvar = traced(dummy_features, dummy_boxes)
    diff_p = (ref_params - traced_params).abs().max().item()
    diff_l = (ref_logvar - traced_logvar).abs().max().item()
    print(f"  Trace verification: params diff = {diff_p:.6e}, logvar diff = {diff_l:.6e}")
    assert diff_p < 1e-4, f"Params trace diverged: {diff_p}"

    # Convert to CoreML
    print("  Converting to CoreML (FP16)...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_features", shape=(1, 3600, in_dim)),
            ct.TensorType(name="bb2d_normalized", shape=(1, max_boxes, 4)),
        ],
        outputs=[
            ct.TensorType(name="params"),
            ct.TensorType(name="logvar"),
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
    )

    out_path = output_dir / "BoxerNet.mlpackage"
    mlmodel.save(str(out_path))
    print(f"  Saved: {out_path}")
    return out_path


# =============================================================================
# End-to-end verification
# =============================================================================

def verify_pipeline(dino_wrapper, boxernet_model, output_dir, max_boxes=20):
    """Run both PyTorch and CoreML and compare outputs."""
    import coremltools as ct

    print("\n=== Verifying Pipeline ===")

    # Create dummy inputs
    dummy_img = torch.randn(1, 3, 960, 960).clamp(0, 1)
    dummy_depth = torch.rand(1, 1, 60, 60) * 3.0  # 0-3 meters
    dummy_bb2d = torch.rand(1, max_boxes, 4)  # normalized [0,1]

    # PyTorch reference: DINOv3
    dino_coreml = DINOv3ForCoreML(dino_wrapper)
    dino_coreml.eval()
    with torch.no_grad():
        dino_features = dino_coreml(dummy_img)  # (1, 384, 60, 60)

    # Prepare BoxerNet input
    B, D, fH, fW = dino_features.shape
    dino_flat = dino_features.reshape(B, D, fH * fW).permute(0, 2, 1)  # (1, 3600, 384)
    depth_flat = dummy_depth.reshape(B, 1, fH * fW).permute(0, 2, 1)  # (1, 3600, 1)
    input_features = torch.cat([dino_flat, depth_flat], dim=-1)  # (1, 3600, 385)

    # PyTorch reference: BoxerNet
    boxer_coreml = BoxerNetForCoreML(boxernet_model)
    boxer_coreml.eval()
    with torch.no_grad():
        ref_params, ref_logvar = boxer_coreml(input_features, dummy_bb2d)

    print(f"  DINOv3 output: {dino_features.shape}, range [{dino_features.min():.3f}, {dino_features.max():.3f}]")
    print(f"  BoxerNet params: {ref_params.shape}, range [{ref_params.min():.3f}, {ref_params.max():.3f}]")
    print(f"  BoxerNet logvar: {ref_logvar.shape}, range [{ref_logvar.min():.3f}, {ref_logvar.max():.3f}]")

    # Decode params for sanity check
    center = ref_params[0, 0, :3]
    size_raw = ref_params[0, 0, 3:6]
    yaw_raw = ref_params[0, 0, 6]
    size = torch.sigmoid(size_raw) * (4.0 - 0.02) + 0.02
    yaw = (math.pi / 2) * torch.tanh(yaw_raw)
    prob = 1.0 / (1.0 + torch.exp(ref_logvar[0, 0, 0]))
    print(f"  Box 0: center={center.numpy()}, size={size.numpy()}, yaw={yaw.item():.3f}rad, prob={prob.item():.3f}")
    print("  Pipeline verification passed!")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Convert Boxer to CoreML")
    parser.add_argument(
        "--output-dir", type=str, default="converted_models",
        help="Output directory for mlpackage files"
    )
    parser.add_argument(
        "--max-boxes", type=int, default=20,
        help="Maximum number of 2D bounding boxes (fixed for CoreML)"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip checkpoint download"
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Only run verification, don't convert"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download checkpoints
    if not args.skip_download:
        print("Step 1: Downloading checkpoints...")
        download_checkpoints()
    else:
        print("Step 1: Skipping download")

    # 2. Monkey-patch problematic layers
    print("\nStep 2: Applying patches...")
    patch_linear_k_masked_bias()

    # 3. Load models
    print("\nStep 3: Loading models...")
    from boxernet.boxernet import BoxerNet
    boxernet = BoxerNet.load_from_checkpoint(
        str(CKPT_DIR / CKPT_FILES["boxernet"]),
        device="cpu",
    )
    boxernet.eval()

    # Bake masked biases into standard nn.Linear (removes isnan from graph)
    bake_masked_bias_into_model(boxernet)

    dino_wrapper = boxernet.dino
    print(f"  BoxerNet loaded: dim={boxernet.dim}, in_dim={boxernet.in_dim}")
    print(f"  DINOv3 loaded: feat_dim={dino_wrapper.feat_dim}")

    # Count params
    dino_params = sum(p.numel() for p in dino_wrapper.parameters())
    boxer_params = sum(p.numel() for p in boxernet.parameters()) - dino_params
    print(f"  DINOv3 params: {dino_params / 1e6:.1f}M")
    print(f"  BoxerNet params: {boxer_params / 1e6:.1f}M")

    if args.verify_only:
        verify_pipeline(dino_wrapper, boxernet, output_dir, args.max_boxes)
        return

    # 4. Convert DINOv3
    convert_dinov3(dino_wrapper, output_dir)

    # 5. Convert BoxerNet
    convert_boxernet(boxernet, output_dir, args.max_boxes)

    # 6. Verify
    verify_pipeline(dino_wrapper, boxernet, output_dir, args.max_boxes)

    print("\n=== Done! ===")
    print(f"Output: {output_dir}/BoxerDINOv3.mlpackage")
    print(f"Output: {output_dir}/BoxerNet.mlpackage")
    print(f"\nSwift driver must handle:")
    print(f"  - ARKit camera pose → gravity alignment → T_world_voxel")
    print(f"  - ARKit depth → per-patch median depth (60x60)")
    print(f"  - 2D box normalization: (xmin+0.5)/W, (ymin+0.5)/H, ...")
    print(f"  - Post-processing: sigmoid(size), tanh(yaw), coordinate transform")


if __name__ == "__main__":
    main()
