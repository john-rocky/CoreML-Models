"""
Convert SinSR (Single-Step Diffusion Super-Resolution) to CoreML.

Produces 3 models:
  1. SinSR_Encoder.mlpackage   — VQ-VAE encoder (image → latent)
  2. SinSR_Denoiser.mlpackage  — UNet single-step denoiser (Swin Transformer)
  3. SinSR_Decoder.mlpackage   — VQ-VAE decoder with vector quantization (latent → image)

Prerequisites:
  - Clone SinSR repo:  git clone https://github.com/wyf0912/SinSR ~/Downloads/SinSR
  - Download weights into ~/Downloads/SinSR/weights/:
      SinSR_v1.pth          (from SinSR releases)
      autoencoder_vq_f4.pth (from ResShift releases)
  - pip install omegaconf einops loguru timm coremltools torch

Source patches applied (to ~/Downloads/SinSR/models/swin_transformer.py):
  1. WindowAttention: pre-compute relative position bias as buffer (avoids dynamic indexing)
  2. window_reverse: replace int() with integer division (avoids aten::Int trace op)
  3. SwinTransformerBlock: replace torch.roll with _safe_roll (slice+cat, avoids tensor_assign)
  4. calculate_mask: label-based implementation (avoids __setitem__ tensor_assign)

Source patches applied (to ~/Downloads/SinSR/ldm/modules/diffusionmodules/model.py):
  5. AttnBlock.forward: use flatten() + reshape(x.shape) instead of int(c) scaling

coremltools patches (at runtime):
  6. Register custom 'int' op converter to handle multi-dimensional tensor shape casts
"""

import sys
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SINSR_ROOT = os.path.expanduser("~/Downloads/SinSR")
sys.path.insert(0, SINSR_ROOT)

from omegaconf import OmegaConf
from utils import util_common, util_net

import coremltools as ct

# ---------- coremltools patch: int op converter ----------
# The default converter crashes on multi-dim tensor values from shape ops.
from coremltools.converters.mil.frontend.torch import ops as _ct_ops
from coremltools.converters.mil import Builder as mb

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


# ---------- Configuration ----------
LATENT_H, LATENT_W = 256, 256  # latent spatial size (= LQ image size, output = 1024x1024)


# ---------- Pre-compute Swin biases for tracing ----------
def _precompute_swin_biases(model):
    """Bake relative position biases into buffers so tracing avoids dynamic indexing."""
    from models.swin_transformer import WindowAttention
    for m in model.modules():
        if isinstance(m, WindowAttention):
            m.precompute_bias()


# ---------- Load models ----------
def load_models():
    configs = OmegaConf.load(os.path.join(SINSR_ROOT, "configs/SinSR.yaml"))
    configs.model.ckpt_path = os.path.join(SINSR_ROOT, "weights/SinSR_v1.pth")
    configs.autoencoder.ckpt_path = os.path.join(SINSR_ROOT, "weights/autoencoder_vq_f4.pth")

    print("[1/3] Loading UNet...")
    model = util_common.instantiate_from_config(configs.model)
    state = torch.load(configs.model.ckpt_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    util_net.reload_model(model, state)
    model.eval().float()

    print("[2/3] Loading VQ-VAE...")
    autoencoder = util_common.instantiate_from_config(configs.autoencoder)
    ae_state = torch.load(configs.autoencoder.ckpt_path, map_location="cpu")
    if "state_dict" in ae_state:
        ae_state = ae_state["state_dict"]
    util_net.reload_model(autoencoder, ae_state)
    autoencoder.eval().float()

    print("[3/3] Building diffusion schedule...")
    diffusion = util_common.instantiate_from_config(configs.diffusion)

    return model, autoencoder, diffusion, configs


# ---------- Extract diffusion constants ----------
def extract_schedule_constants(diffusion):
    """Extract constants needed for single-step inference in Swift."""
    T = diffusion.num_timesteps
    t_idx = T - 1

    etas = np.array(diffusion.etas)
    sqrt_etas = np.array(diffusion.sqrt_etas)
    kappa = float(diffusion.kappa)
    scale_factor = float(diffusion.scale_factor)

    eta_T = float(etas[t_idx])
    sqrt_eta_T = float(sqrt_etas[t_idx])
    normalize_std = math.sqrt(eta_T * kappa ** 2 + 1)

    print(f"  num_timesteps={T}, t_idx={t_idx}")
    print(f"  kappa={kappa}, sqrtEtaT={sqrt_eta_T:.4f}, normalizeStd={normalize_std:.4f}")

    return {
        "t_idx": t_idx,
        "sqrt_eta_T": sqrt_eta_T,
        "kappa": kappa,
        "scale_factor": scale_factor,
        "normalize_std": normalize_std,
    }


# ---------- Wrapper modules ----------
class EncoderWrapper(nn.Module):
    """VQ-VAE encoder: bicubic-upsampled image → latent."""
    def __init__(self, autoencoder, scale_factor):
        super().__init__()
        self.encoder = autoencoder.encoder
        self.quant_conv = autoencoder.quant_conv
        self.scale_factor = scale_factor

    def forward(self, x):
        h = self.encoder(x)
        z = self.quant_conv(h)
        return z * self.scale_factor


class DecoderWrapper(nn.Module):
    """VQ-VAE decoder with vector quantization: latent → RGB image."""
    def __init__(self, autoencoder):
        super().__init__()
        self.embedding_weight = autoencoder.quantize.embedding.weight
        self.post_quant_conv = autoencoder.post_quant_conv
        self.decoder = autoencoder.decoder

    def forward(self, z):
        B, C, H, W = z.shape
        z_perm = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z_perm.reshape(-1, C)

        # Nearest codebook entry via L2 distance
        d = (torch.sum(z_flat ** 2, dim=1, keepdim=True)
             + torch.sum(self.embedding_weight ** 2, dim=1)
             - 2 * torch.matmul(z_flat, self.embedding_weight.t()))
        indices = torch.argmin(d, dim=1)
        z_q = torch.index_select(self.embedding_weight, 0, indices)
        z_q = z_q.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        quant = self.post_quant_conv(z_q)
        return self.decoder(quant)


class DenoiserWrapper(nn.Module):
    """UNet denoiser with baked-in timestep for single-step inference."""
    def __init__(self, model, t_idx):
        super().__init__()
        self.model = model
        self.register_buffer("timestep", torch.tensor([t_idx], dtype=torch.long))

    def forward(self, x_concat):
        # Input: [1, 6, H, W] = concat(scaled_noisy_latent, lq)
        x = x_concat[:, :3, :, :]
        lq = x_concat[:, 3:, :, :]
        return self.model(x, self.timestep, lq=lq)


# ---------- Conversion functions ----------
def convert_encoder(autoencoder, scale_factor, output_dir):
    print("\n=== Converting Encoder ===")
    wrapper = EncoderWrapper(autoencoder, scale_factor)
    wrapper.eval()

    inp_h, inp_w = LATENT_H * 4, LATENT_W * 4
    example = torch.randn(1, 3, inp_h, inp_w)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example)

    out = traced(example)
    print(f"  Input:  [1, 3, {inp_h}, {inp_w}] → Output: {list(out.shape)}")

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image", shape=example.shape)],
        outputs=[ct.TensorType(name="latent")],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
    )

    path = os.path.join(output_dir, "SinSR_Encoder.mlpackage")
    mlmodel.save(path)
    print(f"  Saved: {path}")


def convert_denoiser(model, t_idx, output_dir):
    print("\n=== Converting Denoiser ===")
    wrapper = DenoiserWrapper(model, t_idx)
    wrapper.eval()

    example = torch.randn(1, 6, LATENT_H, LATENT_W)

    with torch.no_grad():
        out = wrapper(example)
    print(f"  Input:  [1, 6, {LATENT_H}, {LATENT_W}] → Output: {list(out.shape)}")

    print("  Pre-computing attention biases...")
    _precompute_swin_biases(wrapper.model)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=example.shape)],
        outputs=[ct.TensorType(name="predicted_latent")],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32,
    )

    path = os.path.join(output_dir, "SinSR_Denoiser.mlpackage")
    mlmodel.save(path)
    print(f"  Saved: {path}")


def convert_decoder(autoencoder, output_dir):
    print("\n=== Converting Decoder ===")
    wrapper = DecoderWrapper(autoencoder)
    wrapper.eval()

    example = torch.randn(1, 3, LATENT_H, LATENT_W)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example)

    out = traced(example)
    print(f"  Input:  [1, 3, {LATENT_H}, {LATENT_W}] → Output: {list(out.shape)}")

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="latent", shape=example.shape)],
        outputs=[ct.TensorType(name="image")],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
    )

    path = os.path.join(output_dir, "SinSR_Decoder.mlpackage")
    mlmodel.save(path)
    print(f"  Saved: {path}")


# ---------- Verification ----------
def verify_pytorch(model, autoencoder, diffusion, constants):
    """Quick PyTorch inference to verify the pipeline before conversion."""
    print("\n=== Verification: PyTorch inference ===")
    lq = torch.randn(1, 3, LATENT_H, LATENT_W)

    with torch.no_grad():
        lq_up = F.interpolate(lq, scale_factor=4, mode="bicubic")
        z_y = autoencoder.encode(lq_up) * constants["scale_factor"]
        noise = torch.randn_like(z_y)
        z_T = z_y + constants["kappa"] * constants["sqrt_eta_T"] * noise
        z_T_scaled = z_T / constants["normalize_std"]
        t = torch.tensor([constants["t_idx"]], dtype=torch.long)
        pred_z0 = model(z_T_scaled, t, lq=lq).clamp(-1, 1)
        sr = autoencoder.decode(pred_z0 / constants["scale_factor"]).clamp(-1, 1)
        print(f"  {list(lq.shape)} → {list(sr.shape)}, range [{sr.min():.3f}, {sr.max():.3f}]")

    print("  Verification passed!")


def main():
    output_dir = os.path.expanduser("~/Downloads/CoreML-Models/conversion_scripts")
    os.makedirs(output_dir, exist_ok=True)

    model, autoencoder, diffusion, configs = load_models()
    constants = extract_schedule_constants(diffusion)
    verify_pytorch(model, autoencoder, diffusion, constants)

    convert_encoder(autoencoder, constants["scale_factor"], output_dir)
    convert_denoiser(model, constants["t_idx"], output_dir)
    convert_decoder(autoencoder, output_dir)

    print("\n=== Swift Constants ===")
    print(f"let kappa: Float = {constants['kappa']}")
    print(f"let sqrtEtaT: Float = {constants['sqrt_eta_T']}")
    print(f"let normalizeStd: Float = {constants['normalize_std']}")
    print(f"let scaleFactor: Float = {constants['scale_factor']}")
    print("\nDone!")


if __name__ == "__main__":
    main()
