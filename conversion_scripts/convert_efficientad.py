"""
Convert EfficientAD (Anomaly Detection) to CoreML.

Architecture:
  PDN-Small teacher/student + autoencoder. Computes anomaly heatmap from
  the difference between teacher-student and autoencoder-student outputs.
  Input: 256x256 RGB image -> Output: anomaly_map [1, 1, 256, 256] + anomaly_score scalar

Pretrained weights:
  MSherbinii/efficientad-bottle on HuggingFace (MVTec AD bottle category)

Requirements:
  pip install torch coremltools huggingface_hub

Usage:
  python convert_efficientad.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
from huggingface_hub import hf_hub_download
import os


class EfficientAD(nn.Module):
    """Wraps teacher + student + autoencoder into a single inference model.

    Bakes in ImageNet normalization, anomaly map computation, quantile
    normalization, and upsampling so the CoreML model is self-contained.
    """

    def __init__(self, teacher, student, autoencoder,
                 q_st_start, q_st_end, q_ae_start, q_ae_end):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.autoencoder = autoencoder

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.q_st_start = q_st_start
        self.q_st_end = q_st_end
        self.q_ae_start = q_ae_start
        self.q_ae_end = q_ae_end

    def forward(self, x):
        # ImageNet normalization (input is [0, 1] range)
        x = (x - self.mean) / self.std

        # Forward all three networks
        t_out = self.teacher(x)             # [1, 384, 56, 56]
        s_out = self.student(x)             # [1, 768, 56, 56]
        ae_out = self.autoencoder(x)        # [1, 384, 56, 56]

        # Anomaly maps: MSE between teacher-student and autoencoder-student
        map_st = torch.mean((t_out - s_out[:, :384]) ** 2, dim=1, keepdim=True)
        map_ae = torch.mean((ae_out - s_out[:, 384:]) ** 2, dim=1, keepdim=True)

        # Quantile normalization
        map_st = 0.1 * (map_st - self.q_st_start) / (self.q_st_end - self.q_st_start)
        map_ae = 0.1 * (map_ae - self.q_ae_start) / (self.q_ae_end - self.q_ae_start)

        # Combine and upsample
        combined = 0.5 * map_st + 0.5 * map_ae
        anomaly_map = F.interpolate(combined, size=(256, 256),
                                    mode="bilinear", align_corners=False)

        # Clamp to [0, 1] for cleaner output
        anomaly_map = anomaly_map.clamp(0.0, 1.0)

        # Image-level anomaly score (max of the map)
        anomaly_score = anomaly_map.max().unsqueeze(0)

        return anomaly_map, anomaly_score


def download_weights(dest_dir="_weights/efficientad-bottle"):
    """Download pretrained weights from HuggingFace."""
    repo_id = "MSherbinii/efficientad-bottle"
    os.makedirs(dest_dir, exist_ok=True)
    files = ["teacher_final.pth", "student_final.pth",
             "autoencoder_final.pth", "normalization.pth"]
    for f in files:
        hf_hub_download(repo_id, f, local_dir=dest_dir)
    print(f"Downloaded weights to {dest_dir}")
    return dest_dir


def main():
    weight_dir = download_weights()

    print("Loading EfficientAD weights ...")
    teacher = torch.load(f"{weight_dir}/teacher_final.pth",
                         map_location="cpu", weights_only=False)
    student = torch.load(f"{weight_dir}/student_final.pth",
                         map_location="cpu", weights_only=False)
    autoencoder = torch.load(f"{weight_dir}/autoencoder_final.pth",
                             map_location="cpu", weights_only=False)
    norm = torch.load(f"{weight_dir}/normalization.pth",
                      map_location="cpu", weights_only=False)

    model = EfficientAD(
        teacher, student, autoencoder,
        q_st_start=norm["q_st_start"].item(),
        q_st_end=norm["q_st_end"].item(),
        q_ae_start=norm["q_ae_start"].item(),
        q_ae_end=norm["q_ae_end"].item(),
    )
    model.eval()

    print("Tracing model ...")
    dummy = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)

    print("Converting to CoreML FP16 ...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, 256, 256),
                scale=1.0 / 255.0,
                color_layout=ct.colorlayout.RGB,
            )
        ],
        outputs=[
            ct.TensorType(name="anomaly_map"),
            ct.TensorType(name="anomaly_score"),
        ],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
    )

    mlmodel.author = "CoreML-Models"
    mlmodel.short_description = (
        "EfficientAD anomaly detection (bottle category). "
        "256x256 RGB -> anomaly heatmap [1,1,256,256] + score [0-1]."
    )
    mlmodel.license = "MIT"

    out_path = "EfficientAD_Bottle.mlpackage"
    mlmodel.save(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
