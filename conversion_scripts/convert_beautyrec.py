"""
Convert BeautyREC makeup transfer to CoreML (2 models: FaceParser + MakeupTransfer).

Architecture:
  1. FaceParser (BiSeNet): 256x256 RGB → 3-channel mask (lip/skin/eye)
  2. MakeupTransfer (BeautyREC): source + reference + 2 masks → makeup-transferred face

Pretrained weights:
  learningyan/BeautyREC on GitHub (in-repo checkpoints)

Requirements:
  pip install torch torchvision coremltools

Usage:
  # First clone BeautyREC repo:
  #   git clone https://github.com/learningyan/BeautyREC.git /tmp/BeautyREC
  # Then run from the CoreML-Models directory:
  python convert_beautyrec.py
"""

import sys
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct

REPO_PATH = "/tmp/BeautyREC"


def patch_imports():
    """Patch missing dependencies so we can import model code."""
    sys.modules["faceutils.faceplusplus"] = types.ModuleType("x")
    sys.modules["dlib"] = types.ModuleType("x")
    # Prevent network/__init__.py from auto-discovering models
    with open(f"{REPO_PATH}/network/__init__.py", "w") as f:
        f.write("# patched\n")
    with open(f"{REPO_PATH}/faceutils/__init__.py", "w") as f:
        f.write("from . import mask\n")
    sys.path.insert(0, REPO_PATH)


# ── Wrappers ──────────────────────────────────────────────────────────────────

class FaceParserWrapper(nn.Module):
    """BiSeNet face parser → 3-channel mask (lip, skin, eye).

    Bakes in ImageNet normalization and label remapping so the CoreML model
    takes a raw [0,1] image and outputs a ready-to-use mask.
    """

    MAPPER = [0, 1, 2, 3, 4, 5, 0, 11, 12, 0, 6, 8, 7, 9, 13, 0, 0, 10, 0]

    # Labels after remapping that correspond to each component
    LIP_LABELS = (9, 13)
    SKIN_LABELS = (4, 8, 10)
    EYE_LABELS = (1, 6)

    def __init__(self, bisenet):
        super().__init__()
        self.net = bisenet
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.register_buffer("mapper", torch.tensor(self.MAPPER, dtype=torch.long))

    def forward(self, x):
        # ImageNet normalize
        x = (x - self.mean) / self.std
        # Parse
        out = self.net(x)[0]  # [1, 19, H, W]
        parsing = out.argmax(dim=1)  # [1, H, W]
        # Remap labels
        remapped = self.mapper[parsing]  # [1, H, W]
        # Build 3-channel mask (use float add instead of bool OR for CoreML)
        lip = ((remapped == 9).float() + (remapped == 13).float()).clamp(0, 1)
        skin = ((remapped == 4).float() + (remapped == 8).float()
                + (remapped == 10).float()).clamp(0, 1)
        eye = ((remapped == 1).float() + (remapped == 6).float()).clamp(0, 1)
        mask = torch.stack([lip, skin, eye], dim=1)  # [1, 3, H, W]
        return mask


class MakeupTransferWrapper(nn.Module):
    """BeautyREC with normalization baked in.

    Input: source [0,1], reference [0,1], ref_mask [0,1], src_mask [0,1]
    Output: result [0,1]
    """

    def __init__(self, beautyrec):
        super().__init__()
        self.model = beautyrec

    def forward(self, source, reference, ref_mask, src_mask):
        # Normalize to [-1, 1]
        src = source * 2.0 - 1.0
        ref = reference * 2.0 - 1.0
        out = self.model(src, ref, ref_mask, src_mask)
        # Back to [0, 1]
        return (out + 1.0) / 2.0


# ── Conversion ────────────────────────────────────────────────────────────────

def convert_face_parser():
    from faceutils.mask.model import BiSeNet

    print("Loading BiSeNet face parser...")
    bisenet = BiSeNet(n_classes=19)
    bisenet.load_state_dict(torch.load(
        f"{REPO_PATH}/faceutils/mask/resnet.pth", map_location="cpu"))
    bisenet.eval()

    wrapper = FaceParserWrapper(bisenet)
    wrapper.eval()

    print("Tracing FaceParser...")
    dummy = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy)

    print("Converting FaceParser to CoreML...")
    ml = ct.convert(
        traced,
        inputs=[ct.ImageType(
            name="image", shape=(1, 3, 256, 256),
            scale=1.0 / 255.0, color_layout=ct.colorlayout.RGB,
        )],
        outputs=[ct.TensorType(name="mask")],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
    )

    ml.author = "CoreML-Models"
    ml.short_description = (
        "BiSeNet face parser for makeup transfer. "
        "256x256 RGB → 3-channel mask [lip, skin, eye]."
    )
    ml.license = "MIT"

    path = "BeautyREC_FaceParser.mlpackage"
    ml.save(path)
    print(f"Saved {path}")
    return path


def convert_makeup_transfer():
    from network.REC.REC import BeautyREC

    print("Loading BeautyREC...")
    params = {
        "dim": 48, "style_dim": 48, "activ": "relu",
        "n_downsample": 2, "n_res": 3, "pad_type": "reflect",
    }
    model = BeautyREC(params)
    ckpt = torch.load(
        f"{REPO_PATH}/checkpoints/BeautyREC.pt",
        map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["cleaner"])
    model.eval()

    wrapper = MakeupTransferWrapper(model)
    wrapper.eval()

    print("Tracing BeautyREC...")
    src = torch.rand(1, 3, 256, 256)
    ref = torch.rand(1, 3, 256, 256)
    ref_mask = torch.rand(1, 3, 256, 256)
    src_mask = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (src, ref, ref_mask, src_mask))

    print("Converting BeautyREC to CoreML...")
    ml = ct.convert(
        traced,
        inputs=[
            ct.ImageType(name="source", shape=(1, 3, 256, 256),
                         scale=1.0 / 255.0, color_layout=ct.colorlayout.RGB),
            ct.ImageType(name="reference", shape=(1, 3, 256, 256),
                         scale=1.0 / 255.0, color_layout=ct.colorlayout.RGB),
            ct.TensorType(name="ref_mask", shape=(1, 3, 256, 256)),
            ct.TensorType(name="src_mask", shape=(1, 3, 256, 256)),
        ],
        outputs=[ct.TensorType(name="result")],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
    )

    ml.author = "CoreML-Models"
    ml.short_description = (
        "BeautyREC makeup transfer. "
        "Source face + reference makeup + masks → makeup-transferred face 256x256."
    )
    ml.license = "BSD-3-Clause (non-commercial)"

    path = "BeautyREC_MakeupTransfer.mlpackage"
    ml.save(path)
    print(f"Saved {path}")
    return path


def main():
    patch_imports()
    convert_face_parser()
    convert_makeup_transfer()
    print("\nDone! Two models generated:")
    print("  1. BeautyREC_FaceParser.mlpackage")
    print("  2. BeautyREC_MakeupTransfer.mlpackage")


if __name__ == "__main__":
    main()
