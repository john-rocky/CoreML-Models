"""
Convert Pixelization (SIGGRAPH Asia 2022) to CoreML.

Repo: https://github.com/WuZongWei6/Pixelization
Weights mirror: https://huggingface.co/ashleykleynhans/pixelization

Architecture:
  G_A (C2PGen): RGBEnc -> RGBDec(modulated by cellcode) -> tanh image
    where cellcode = MLP(fixed_256d_vector) is a precomputed [1, 2048] style code.
  alias_net: AliasRGBEncoder -> AliasRGBDecoder -> tanh anti-aliased image.

Pipeline (baked into a single mlpackage):
  input[0,1] RGB
  -> x = 2x-1              (normalize to [-1,1])
  -> feature = RGBEnc(x)
  -> y = RGBDec(feature, cellcode)
  -> y = alias_net(y)
  -> y = (y+1)/2 clamped   (denorm to [0,1])
  -> output RGB image

Post-processing (done in Swift, not in the model):
  nearest-neighbor downscale by 4 -> logical pixel grid
  nearest-neighbor upscale by cell_size -> display size

Usage:
  python convert_pixelization.py --size 512
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvmodels

import coremltools as ct

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Pixelization")
sys.path.insert(0, REPO_DIR)


# Precomputed MLP_code constant from upstream test_pro.py (reshape to [1, 256, 1, 1]).
MLP_CODE = [
    233356.8125, -27387.5918, -32866.8008, 126575.0312, -181590.0156,
    -31543.1289, 50374.1289, 99631.4062, -188897.3750, 138322.7031,
    -107266.2266, 125778.5781, 42416.1836, 139710.8594, -39614.6250,
    -69972.6875, -21886.4141, 86938.4766, 31457.6270, -98892.2344,
    -1191.5887, -61662.1719, -180121.9062, -32931.0859, 43109.0391,
    21490.1328, -153485.3281, 94259.1797, 43103.1992, -231953.8125,
    52496.7422, 142697.4062, -34882.7852, -98740.0625, 34458.5078,
    -135436.3438, 11420.5488, -18895.8984, -71195.4141, 176947.2344,
    -52747.5742, 109054.6562, -28124.9473, -17736.6152, -41327.1562,
    69853.3906, 79046.2656, -3923.7344, -5644.5229, 96586.7578,
    -89315.2656, -146578.0156, -61862.1484, -83956.4375, 87574.5703,
    -75055.0469, 19571.8203, 79358.7891, -16501.5000, -147169.2188,
    -97861.6797, 60442.1797, 40156.9023, 223136.3906, -81118.0547,
    -221443.6406, 54911.6914, 54735.9258, -58805.7305, -168884.4844,
    40865.9609, -28627.9043, -18604.7227, 120274.6172, 49712.2383,
    164402.7031, -53165.0820, -60664.0469, -97956.1484, -121468.4062,
    -69926.1484, -4889.0151, 127367.7344, 200241.0781, -85817.7578,
    -143190.0625, -74049.5312, 137980.5781, -150788.7656, -115719.6719,
    -189250.1250, -153069.7344, -127429.7891, -187588.2500, 125264.7422,
    -79082.3438, -114144.5781, 36033.5039, -57502.2188, 80488.1562,
    36501.4570, -138817.5938, -22189.6523, -222146.9688, -73292.3984,
    127717.2422, -183836.3750, -105907.0859, 145422.8750, 66981.2031,
    -9596.6699, 78099.4922, 70226.3359, 35841.8789, -116117.6016,
    -150986.0156, 81622.4922, 113575.0625, 154419.4844, 53586.4141,
    118494.8750, 131625.4375, -19763.1094, 75581.1172, -42750.5039,
    97934.8281, 6706.7949, -101179.0078, 83519.6172, -83054.8359,
    -56749.2578, -30683.6992, 54615.9492, 84061.1406, -229136.7188,
    -60554.0000, 8120.2622, -106468.7891, -28316.3418, -166351.3125,
    47797.3984, 96013.4141, 71482.9453, -101429.9297, 209063.3594,
    -3033.6882, -38952.5352, -84920.6719, -5895.1543, -18641.8105,
    47884.3633, -14620.0273, -132898.6719, -40903.5859, 197217.3750,
    -128599.1328, -115397.8906, -22670.7676, -78569.9688, -54559.7070,
    -106855.2031, 40703.1484, 55568.3164, 60202.9844, -64757.9375,
    -32068.8652, 160663.3438, 72187.0703, -148519.5469, 162952.8906,
    -128048.2031, -136153.8906, -15270.3730, -52766.3281, -52517.4531,
    18652.1992, 195354.2188, -136657.3750, -8034.2622, -92699.6016,
    -129169.1406, 188479.9844, 46003.7500, -93383.0781, -67831.6484,
    -66710.5469, 104338.5234, 85878.8438, -73165.2031, 95857.3203,
    71213.1250, 94603.1094, -30359.8125, -107989.2578, 99822.1719,
    184626.3594, 79238.4531, -272978.9375, -137948.5781, -145245.8125,
    75359.2031, 26652.7930, 50421.4141, 60784.4102, -18286.3398,
    -182851.9531, -87178.7969, -13131.7539, 195674.8906, 59951.7852,
    124353.7422, -36709.1758, -54575.4766, 77822.6953, 43697.4102,
    -64394.3438, 113281.1797, -93987.0703, 221989.7188, 132902.5000,
    -9538.8574, -14594.1338, 65084.9453, -12501.7227, 130330.6875,
    -115123.4766, 20823.0898, 75512.4922, -75255.7422, -41936.7656,
    -186678.8281, -166799.9375, 138770.6250, -78969.9531, 124516.8047,
    -85558.5781, -69272.4375, -115539.1094, 228774.4844, -76529.3281,
    -107735.8906, -76798.8906, -194335.2812, 56530.5742, -9397.7529,
    132985.8281, 163929.8438, -188517.7969, -141155.6406, 45071.0391,
    207788.3125, -125826.1172, 8965.3320, -159584.8438, 95842.4609,
    -76929.4688,
]


def _prepare_dummy_vgg_weights():
    """C2PGen.__init__ insists on loading ./pixelart_vgg19.pth (cwd-relative).
    The VGG branch (PixelBlockEncoder) is only used during training and is
    unreachable at inference, but we still need the file to exist so
    construction succeeds. Write a dummy with matching structure; the real
    weights get overwritten when we load 160_net_G_A.pth anyway."""
    path = "./pixelart_vgg19.pth"
    if os.path.exists(path):
        return
    vgg = tvmodels.vgg.vgg19(weights=None)
    vgg.classifier._modules["6"] = nn.Linear(4096, 7, bias=True)
    torch.save(vgg.state_dict(), path)


def _swap_layernorm_with_groupnorm(module):
    """Replace the upstream custom LayerNorm (global mean/std + per-channel
    affine) with the mathematically equivalent nn.GroupNorm(1, C). The manual
    expansion (`x.view(-1).std()` over ~8M elements) diverges badly in FP16 —
    coremltools' native group_norm op handles it correctly."""
    from models.basic_layer import LayerNorm as UpstreamLN
    for name, ch in list(module.named_children()):
        if isinstance(ch, UpstreamLN):
            gn = nn.GroupNorm(1, ch.num_features, eps=ch.eps)
            with torch.no_grad():
                gn.weight.data.copy_(ch.gamma.data)
                gn.bias.data.copy_(ch.beta.data)
            setattr(module, name, gn)
        else:
            _swap_layernorm_with_groupnorm(ch)


def build_pytorch_model():
    # Run from REPO_DIR so relative paths in the vendored code resolve.
    os.chdir(REPO_DIR)
    _prepare_dummy_vgg_weights()
    from models.networks import define_G

    g_a = define_G(3, 3, 64, "c2pGen", "instance", False, "normal", 0.02, [])
    alias = define_G(3, 3, 64, "antialias", "instance", False, "normal", 0.02, [])

    g_a_sd = torch.load(
        "checkpoints/pixelize/160_net_G_A.pth", map_location="cpu"
    )
    alias_sd = torch.load("alias_net.pth", map_location="cpu")
    g_a.load_state_dict(g_a_sd)
    alias.load_state_dict(alias_sd)
    g_a.eval()
    alias.eval()

    _swap_layernorm_with_groupnorm(g_a)
    _swap_layernorm_with_groupnorm(alias)

    with torch.no_grad():
        code = torch.tensor(MLP_CODE).reshape(1, 256, 1, 1)
        cellcode = g_a.MLP(code).detach()  # [1, 2048]
    return g_a, alias, cellcode


class BakedModConv(nn.Module):
    """ModulationConvBlock with the (fixed) cellcode folded into the conv
    weights. The original op computes (W*c)/norm(W*c) at every forward; since c
    is constant we precompute that in FP32 and store it as a plain Conv2d
    weight, which keeps FP16 inference safe (W*c alone overflows FP16 because
    cellcode magnitudes reach 1e8)."""

    def __init__(self, orig, code_chunk):
        super().__init__()
        import torch.nn.functional as F
        self.F = F
        in_c = orig.in_c
        out_c = orig.out_c
        k = orig.ksize
        with torch.no_grad():
            w = orig.weight * orig.wscale  # (out_c, in_c, k, k)
            # Match the original view/permute sequence exactly (no semantic
            # transpose — this is the upstream convention).
            _w = w.view(1, k, k, in_c, out_c)
            _w = _w * code_chunk.view(1, 1, 1, in_c, 1)
            norm = torch.sqrt((_w ** 2).sum(dim=[1, 2, 3]) + orig.eps)
            _w = _w / norm.view(1, 1, 1, 1, out_c)
            w_perm = _w.permute(1, 2, 3, 0, 4).reshape(k, k, in_c, out_c)
            w_final = w_perm.permute(3, 2, 0, 1).contiguous()  # (out_c, in_c, k, k)
        self.register_buffer("weight", w_final)
        self.bias = nn.Parameter(orig.bias.detach().clone())
        self.padding = k // 2

    def forward(self, x):
        x = self.F.conv2d(x, self.weight, bias=None, padding=self.padding)
        x = x + self.bias.view(1, -1, 1, 1)
        x = self.F.leaky_relu(x, 0.2, inplace=False) * (2.0 ** 0.5)
        return x


class BakedRGBDec(nn.Module):
    """RGBDec with cellcode folded in, replacing the 8 modulation convs.
    Upstream reuses `mod_conv_2` for 7 of the 8 calls (mod_conv_3..8 are
    defined but unused); we preserve that behavior exactly."""

    def __init__(self, orig, cellcode):
        super().__init__()
        baked = []
        for i in range(8):
            src = orig.mod_conv_1 if i == 0 else orig.mod_conv_2
            chunk = cellcode[:, i * 256 : (i + 1) * 256]
            baked.append(BakedModConv(src, chunk))
        self.baked = nn.ModuleList(baked)
        self.upsample_block1 = orig.upsample_block1
        self.conv_1 = orig.conv_1
        self.upsample_block2 = orig.upsample_block2
        self.conv_2 = orig.conv_2
        self.conv_3 = orig.conv_3

    def forward(self, x):
        residual = x
        x = self.baked[0](x); x = self.baked[1](x); x = x + residual
        residual = x
        x = self.baked[2](x); x = self.baked[3](x); x = x + residual
        residual = x
        x = self.baked[4](x); x = self.baked[5](x); x = x + residual
        residual = x
        x = self.baked[6](x); x = self.baked[7](x); x = x + residual
        x = self.upsample_block1(x)
        x = self.conv_1(x)
        x = self.upsample_block2(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x


class PixelizationWrapper(nn.Module):
    """Input: RGB image in [0, 1], NCHW. Output: pixelized RGB in [0, 255]."""

    def __init__(self, g_a, alias, cellcode):
        super().__init__()
        self.rgb_enc = g_a.RGBEnc
        self.rgb_dec = BakedRGBDec(g_a.RGBDec, cellcode)
        self.alias = alias

    def forward(self, image):
        # `image` is in [0, 1] because ImageType sets scale=1/255.
        x = image * 2.0 - 1.0
        feature = self.rgb_enc(x)
        y = self.rgb_dec(feature)
        y = self.alias(y)
        # Scale to [0, 255] for ImageType output.
        y = (y + 1.0) * 127.5
        return torch.clamp(y, 0.0, 255.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=512,
                        help="Input H=W (must be multiple of 4).")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.dirname(os.path.abspath(__file__)))
    args = parser.parse_args()
    assert args.size % 4 == 0, "size must be a multiple of 4"
    args.output_dir = os.path.abspath(args.output_dir)

    print("Loading PyTorch weights...")
    g_a, alias, cellcode = build_pytorch_model()
    wrapper = PixelizationWrapper(g_a, alias, cellcode).eval()

    dummy = torch.rand(1, 3, args.size, args.size)
    with torch.no_grad():
        torch_out = wrapper(dummy)
    print(f"PyTorch output shape={tuple(torch_out.shape)}, "
          f"min={torch_out.min():.3f}, max={torch_out.max():.3f} (range [0,255])")

    print("Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy)

    precision = (ct.precision.FLOAT16 if args.precision == "fp16"
                 else ct.precision.FLOAT32)
    print(f"Converting to CoreML {args.precision.upper()}...")
    ml = ct.convert(
        traced,
        inputs=[ct.ImageType(
            name="image",
            shape=(1, 3, args.size, args.size),
            scale=1.0 / 255.0,
            color_layout=ct.colorlayout.RGB,
        )],
        outputs=[ct.ImageType(
            name="pixelized",
            color_layout=ct.colorlayout.RGB,
        )],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=precision,
        convert_to="mlprogram",
    )
    ml.author = "WuZongWei6 (paper) / CoreML-Models (conversion)"
    ml.short_description = (
        f"Pixelization (SIGGRAPH Asia 2022). "
        f"{args.size}x{args.size} RGB -> pixelized RGB (same size). "
        "Non-commercial research use only."
    )
    ml.license = "Non-commercial research (see upstream LICENSE.md)"

    suffix = "" if args.precision == "fp16" else "_FP32"
    out_path = os.path.join(args.output_dir,
                            f"Pixelization_{args.size}{suffix}.mlpackage")
    ml.save(out_path)
    print(f"Saved: {out_path}")

    print("Parity check on example image...")
    try:
        import PIL.Image as Image
        example = os.path.join(REPO_DIR, "examples", "2_1.png")
        if os.path.exists(example):
            pil = Image.open(example).convert("RGB").resize(
                (args.size, args.size), Image.BICUBIC)
            src = np.array(pil)
        else:
            src = (dummy[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            pil = Image.fromarray(src)

        ml_loaded = ct.models.MLModel(
            out_path, compute_units=ct.ComputeUnit.CPU_ONLY)
        ml_out = ml_loaded.predict({"image": pil})["pixelized"]
        ml_out.convert("RGB").save(
            os.path.join(args.output_dir,
                         f"pixelization_sample_{args.precision}.png"))

        t = torch.from_numpy(src).permute(2, 0, 1).float()[None] / 255.0
        with torch.no_grad():
            pt_img = wrapper(t)[0].permute(1, 2, 0).numpy()
        ml_arr = np.array(ml_out.convert("RGB")).astype(np.float32)
        diff = np.abs(ml_arr - pt_img)
        print(f"  max abs diff (0-255): {diff.max():.3f}")
        print(f"  mean abs diff (0-255): {diff.mean():.3f}")
        print(f"  sample saved: pixelization_sample_{args.precision}.png")
    except Exception as e:
        print(f"  parity check skipped: {e}")


if __name__ == "__main__":
    main()
