"""
Convert CSD-MT makeup transfer to CoreML (2 models: FaceParser + MakeupTransfer).

Architecture:
  1. FaceParser (BiSeNet): 256x256 RGB → 10-channel one-hot parse + 3-channel face mask
  2. MakeupTransfer (CSD-MT Generator): source + reference + parses + masks → transferred face

Pretrained weights:
  Snowfallingplum/CSD-MT on GitHub (Google Drive download)

Requirements:
  pip install torch torchvision coremltools

Usage:
  # First clone CSD-MT and download weights:
  #   git clone https://github.com/Snowfallingplum/CSD-MT.git /tmp/CSD-MT
  #   Download CSD_MT.pth and 79999_iter.pth from Google Drive (see README)
  # Then run:
  python convert_csdmt.py
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct

REPO_PATH = "/tmp/CSD-MT/quick_start"


def patch_imports():
    """Patch missing dependencies."""
    import types
    sys.modules.setdefault("dlib", types.ModuleType("dlib"))
    with open(f"{REPO_PATH}/faceutils/__init__.py", "w") as f:
        f.write("from .face_parsing.model import BiSeNet\n")
    sys.path.insert(0, REPO_PATH)


# ── Label mapping (19-class BiSeNet → 10-channel one-hot) ─────────────────────

# BiSeNet class → CSD-MT channel
PARSE_MAP = {
    0: 0, 16: 0, 17: 0, 18: 0, 9: 0,  # background/hair/hat/earring
    1: 1, 6: 1,                          # skin + ear(glass)
    2: 2, 3: 2,                          # brows
    4: 3, 5: 3,                          # eyes
    7: 4, 8: 4,                          # ears
    10: 5,                               # nose
    11: 6,                               # inner mouth
    12: 7,                               # upper lip
    13: 8,                               # lower lip
    14: 9, 15: 9,                        # neck/necklace
}


# ── Wrappers ──────────────────────────────────────────────────────────────────

class FaceParserWrapper(nn.Module):
    """BiSeNet → 10-channel one-hot parse map + 3-channel face mask.

    Bakes in ImageNet normalization, argmax, label remapping, one-hot encoding,
    and face mask computation.
    """

    def __init__(self, bisenet):
        super().__init__()
        self.net = bisenet
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Build a lookup table: bisenet_class → csd_channel
        lut = torch.zeros(19, dtype=torch.long)
        for orig, ch in PARSE_MAP.items():
            lut[orig] = ch
        self.register_buffer("lut", lut)

    def forward(self, x):
        # ImageNet normalize
        x = (x - self.mean) / self.std
        # Parse at input resolution (no resize to 512 — baked into the model)
        out = self.net(x)[0]  # [1, 19, H, W]
        labels = out.argmax(dim=1)  # [1, H, W]
        # Remap to 10-channel scheme
        remapped = self.lut[labels]  # [1, H, W]
        # One-hot encode using float comparisons (CoreML-friendly)
        parse = torch.zeros(1, 10, x.shape[2], x.shape[3], device=x.device)
        for ch in range(10):
            parse[0, ch] = (remapped[0] == ch).float()
        # Face mask: everything except background(ch0), eyes(ch3), mouth(ch6)
        face_mask = 1.0 - parse[:, 0:1] - parse[:, 3:4] - parse[:, 6:7]
        face_mask = face_mask.clamp(0, 1)
        face_mask = torch.cat([face_mask, face_mask, face_mask], dim=1)
        return parse, face_mask


class MakeupTransferWrapper(nn.Module):
    """CSD-MT Generator with normalization baked in.

    Input: source [0,1], ref [0,1], source_parse, source_mask, ref_parse, ref_mask
    Output: result [0,1]
    """

    def __init__(self, generator):
        super().__init__()
        self.gen = generator

    def forward(self, source, reference, source_parse, source_mask, ref_parse, ref_mask):
        # Normalize to [-1, 1]
        src = source * 2.0 - 1.0
        ref = reference * 2.0 - 1.0
        output = self.gen(
            source_img=src, source_parse=source_parse, source_all_mask=source_mask,
            ref_img=ref, ref_parse=ref_parse, ref_all_mask=ref_mask,
        )
        # Only return transfer_img, normalized to [0, 1]
        return (output["transfer_img"][:, :3] + 1.0) / 2.0


# ── Conversion ────────────────────────────────────────────────────────────────

def convert_face_parser():
    from faceutils.face_parsing.model import BiSeNet

    print("Loading BiSeNet face parser...")
    bisenet = BiSeNet(n_classes=19)
    bisenet.load_state_dict(torch.load(
        f"{REPO_PATH}/faceutils/face_parsing/res/cp/79999_iter.pth", map_location="cpu"))
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
        outputs=[
            ct.TensorType(name="parse"),
            ct.TensorType(name="face_mask"),
        ],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
    )

    ml.author = "CoreML-Models"
    ml.short_description = (
        "BiSeNet face parser for CSD-MT makeup transfer. "
        "256x256 RGB → 10-channel parse + 3-channel face mask."
    )
    ml.license = "MIT"

    path = "CSDMT_FaceParser.mlpackage"
    ml.save(path)
    print(f"Saved {path}")
    return path


def convert_makeup_transfer():
    from CSD_MT.modules import Generator
    from CSD_MT.utils import init_net

    device = torch.device("cpu")
    print("Loading CSD-MT Generator...")
    gen = init_net(Generator(input_dim=3, parse_dim=10, ngf=16, device=device),
                   device, init_type="normal", gain=0.02)
    ckpt = torch.load(f"{REPO_PATH}/CSD_MT/weights/CSD_MT.pth", map_location="cpu")
    gen.load_state_dict(ckpt["gen"])
    gen.eval()

    wrapper = MakeupTransferWrapper(gen)
    wrapper.eval()

    print("Tracing CSD-MT Generator...")
    src = torch.rand(1, 3, 256, 256)
    ref = torch.rand(1, 3, 256, 256)
    sp = torch.rand(1, 10, 256, 256)
    sm = torch.rand(1, 3, 256, 256)
    rp = torch.rand(1, 10, 256, 256)
    rm = torch.rand(1, 3, 256, 256)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (src, ref, sp, sm, rp, rm), strict=False)

    print("Converting CSD-MT to CoreML...")
    ml = ct.convert(
        traced,
        inputs=[
            ct.ImageType(name="source", shape=(1, 3, 256, 256),
                         scale=1.0 / 255.0, color_layout=ct.colorlayout.RGB),
            ct.ImageType(name="reference", shape=(1, 3, 256, 256),
                         scale=1.0 / 255.0, color_layout=ct.colorlayout.RGB),
            ct.TensorType(name="source_parse", shape=(1, 10, 256, 256)),
            ct.TensorType(name="source_mask", shape=(1, 3, 256, 256)),
            ct.TensorType(name="ref_parse", shape=(1, 10, 256, 256)),
            ct.TensorType(name="ref_mask", shape=(1, 3, 256, 256)),
        ],
        outputs=[ct.TensorType(name="result")],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
    )

    ml.author = "CoreML-Models"
    ml.short_description = (
        "CSD-MT makeup transfer (CVPR 2024). "
        "Source + reference faces with parse maps → makeup-transferred face 256x256."
    )
    ml.license = "CC BY-NC-SA 4.0"

    path = "CSDMT_MakeupTransfer.mlpackage"
    ml.save(path)
    print(f"Saved {path}")
    return path


def main():
    patch_imports()
    convert_face_parser()
    convert_makeup_transfer()
    print("\nDone! Two models generated:")
    print("  1. CSDMT_FaceParser.mlpackage")
    print("  2. CSDMT_MakeupTransfer.mlpackage")


if __name__ == "__main__":
    main()
