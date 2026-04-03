"""
Convert AdaFace IR-18 face recognition model to CoreML.

Usage:
    python3 convert_adaface.py

Output:
    creative_apps/AdaFaceDemo/AdaFaceDemo/AdaFace_IR18.mlpackage

Model: AdaFace (CVPR 2022) via CVLface
  - Input: 112x112 RGB face image
  - Output: 512-dim L2-normalized face embedding
  - License: MIT
  - Repo: https://github.com/mk-minchul/AdaFace
"""

import torch
import torch.nn as nn
import coremltools as ct
import os
import sys

# === iResNet-18 architecture (from CVLface, fvcore dependency removed) ===

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class BasicBlockIR(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth))
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.BatchNorm2d(depth),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

def get_block(in_channel, depth, num_units, stride=2):
    return [BasicBlockIR(in_channel, depth, stride)] + \
           [BasicBlockIR(depth, depth, 1) for _ in range(num_units - 1)]

# IR-18 block config
BLOCKS_IR18 = [
    (64, 64, 2, 2),    # in_ch, depth, num_units, stride
    (64, 128, 2, 2),
    (128, 256, 2, 2),
    (256, 512, 2, 2),
]

class Backbone(nn.Module):
    def __init__(self, input_size=(112, 112), output_dim=512):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64))

        modules = []
        for in_ch, depth, num_units, stride in BLOCKS_IR18:
            modules += get_block(in_ch, depth, num_units, stride)
        self.body = nn.Sequential(*modules)

        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(0.4),
            Flatten(),
            nn.Linear(512 * 7 * 7, output_dim),
            nn.BatchNorm1d(output_dim, affine=False))

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return x

class AdaFaceModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        emb = self.backbone(x)
        emb = nn.functional.normalize(emb, p=2, dim=1)
        return emb

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load pretrained weights from HuggingFace cache
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub/models--minchul--cvlface_adaface_ir18_webface4m")
    snapshot_dir = None
    snapshots = os.path.join(hf_cache, "snapshots")
    if os.path.exists(snapshots):
        dirs = os.listdir(snapshots)
        if dirs:
            snapshot_dir = os.path.join(snapshots, dirs[0])

    if snapshot_dir is None:
        print("Please download the model first:")
        print("  python3 -c \"from huggingface_hub import hf_hub_download; hf_hub_download('minchul/cvlface_adaface_ir18_webface4m', 'pretrained_model/model.pt')\"")
        sys.exit(1)

    weights_path = os.path.join(snapshot_dir, "pretrained_model", "model.pt")
    print(f"Loading weights from {weights_path}")

    # Build model
    backbone = Backbone(input_size=(112, 112), output_dim=512)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)

    # Keys have "net." prefix
    new_state = {}
    for k, v in state_dict.items():
        new_key = k.replace("net.", "backbone.", 1) if k.startswith("net.") else k
        new_state[new_key] = v

    model = AdaFaceModel(backbone)
    model.load_state_dict(new_state, strict=False)
    model.eval()

    # Verify
    dummy = torch.randn(1, 3, 112, 112)
    with torch.no_grad():
        out = model(dummy)
    print(f"Output shape: {out.shape}, norm: {torch.norm(out, dim=1).item():.4f}")

    # Trace
    print("Tracing model...")
    traced = torch.jit.trace(model, dummy)

    # Convert to CoreML
    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.ImageType(
                name="face_image",
                shape=(1, 3, 112, 112),
                scale=1.0 / (0.5 * 255.0),
                bias=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
                color_layout=ct.colorlayout.BGR,
            )
        ],
        outputs=[ct.TensorType(name="embedding")],
        minimum_deployment_target=ct.target.iOS16,
    )

    mlmodel.author = "CoreML-Models"
    mlmodel.short_description = "AdaFace IR-18: Face recognition embedding (512-dim). Input: 112x112 face image."
    mlmodel.license = "MIT"

    output_path = os.path.join(script_dir, "..", "creative_apps", "AdaFaceDemo", "AdaFaceDemo", "AdaFace_IR18.mlpackage")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mlmodel.save(output_path)
    print(f"\nSaved to {output_path}")

    size_mb = sum(os.path.getsize(os.path.join(dp, f))
                  for dp, _, fns in os.walk(output_path) for f in fns) / 1e6
    print(f"Model size: {size_mb:.1f} MB")

if __name__ == "__main__":
    main()
