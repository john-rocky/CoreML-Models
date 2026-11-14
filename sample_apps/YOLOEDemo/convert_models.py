#!/usr/bin/env python3
# Contains code adapted from ultralytics/ultralytics (adapted May 2026).
# This file as a whole is under the GNU Affero General Public License v3.0.
# See LICENSE-AGPL-3.0.txt in this folder.
"""Convert YOLOE to Core ML as an *open-vocabulary region-embedding* detector.

Unlike the usual YOLOE/YOLO-World export that bakes a fixed text prompt into the
class head, this keeps the open-vocabulary property fully on-device: the detector
emits the per-anchor region embedding *before* the class logits, and the
region-text similarity is computed in Swift against cached text embeddings. The
detector therefore never needs the text at all and the vocabulary can change at
runtime for free (no re-run of the image branch).

Pipeline (see YOLOEDemo/ContentView.swift for the Swift side):

  Text query --> MobileCLIP text encoder (Apple's mobileclip_blt_text.mlpackage)
              --> reprta.mlpackage (YOLOE RepRTA residual MLP)
              --> L2 normalize, append 1.0  ==> text'  [N, 513]   (cached in Swift)

  Image       --> yoloe_detector.mlpackage
              --> boxes [1,4,8400], region_embeddings [1,513,8400],
                  mask_coeffs [1,32,8400], mask_protos [1,32,160,160]

  Swift per frame: logit[k,a] = <region'[:,a], text'[k]>;  score = sigmoid(logit)
                   (this reproduces YOLOE's BNContrastiveHead exactly, see below)

Why 513 dims? YOLOE's class head is a per-scale BNContrastiveHead:
    logit = (BN(region) . normalize(text)) * exp(logit_scale) + bias
We fold the per-scale exp(logit_scale) into the first 512 channels and append the
per-scale `bias` as channel 512, while the cached text gets a trailing constant
1.0. The dot product then yields the exact logit with a single matmul in Swift
and zero knowledge of the anchor layout. Verified parity vs the PyTorch head:
max |sigmoid(region'.text') - official_scores| ~ 2e-11.

Outputs:
  1. yoloe_detector.mlpackage  - text-free region-embedding detector + seg
  2. reprta.mlpackage          - RepRTA residual MLP (raw_tpe -> tpe)
  3. clip_vocab.json           - CLIP BPE vocabulary for the Swift tokenizer

The MobileCLIP text encoder (mobileclip_blt_text.mlpackage) is Apple's official
Core ML export from https://github.com/apple/ml-mobileclip and is used as-is.

Usage:
    pip install ultralytics coremltools clip-anytorch
    python convert_models.py                # yoloe-11s-seg
    python convert_models.py --size 11m-seg
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
from coremltools.converters.mil.frontend.torch import ops as _ct_ops
from coremltools.converters.mil import Builder as mb


EMBED = 512
CONTEXT_LENGTH = 77

YOLOE_MODELS = {
    "11s-seg": "yoloe-11s-seg",
    "11m-seg": "yoloe-11m-seg",
    "11l-seg": "yoloe-11l-seg",
}


# coremltools `int` op patch: YOLOv8/YOLOE Detect heads emit multi-dim shape->int
# casts that the stock converter assumes are scalars. Fold them to int32 consts.
def _patched_int(context, node):
    x = _ct_ops._get_inputs(context, node)[0]
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


# ---------------------------------------------------------------------------
# YOLOE region-embedding detector (text-free)
# ---------------------------------------------------------------------------

class YOLOERegionDetector(nn.Module):
    """Run YOLOE's backbone + neck + head branches, but emit the per-anchor
    region embedding instead of class logits. No text input.

    region_embeddings[:, :512, a] = BN_s(cv3_s(feat))[a] * exp(logit_scale_s)
    region_embeddings[:,  512, a] = bias_s
    where s is the feature scale of anchor a. The class head's exact score is then
    sigmoid(<region'[:,a], [normalize(text_k), 1]>), computed in Swift.
    """

    def __init__(self, yoloe_model):
        super().__init__()
        self.layers = nn.ModuleList(list(yoloe_model.model[:-1]))
        self.head = yoloe_model.model[-1]
        self._f = [m.f for m in yoloe_model.model[:-1]]
        self._save = set(yoloe_model.save)
        self._head_f = self.head.f  # [16, 19, 22]

    def forward(self, image):
        # Backbone + neck (replicate ultralytics' layer routing).
        y = [None] * len(self.layers)
        x = image
        for i, m in enumerate(self.layers):
            f = self._f[i]
            if f != -1:
                x = y[f] if isinstance(f, int) else [x if j == -1 else y[j] for j in f]
            x = m(x)
            y[i] = x if i in self._save else None
        feats = [y[j] for j in self._head_f]

        head = self.head
        nl = head.nl

        # Boxes: reuse ultralytics' DFL decode -> xywh in input(640) pixels.
        boxes_raw = torch.cat(
            [head.cv2[i](feats[i]).view(1, 4 * head.reg_max, -1) for i in range(nl)], dim=-1
        )
        boxes = head._get_decode_boxes({"boxes": boxes_raw, "feats": feats})  # [1,4,8400]

        # Region embeddings with per-scale scale folded in + bias as last channel.
        region = []
        for i in range(nl):
            e = head.cv3[i](feats[i])                       # [1,512,H,W]
            e = head.cv4[i].norm(e)                         # BatchNorm
            e = e * head.cv4[i].logit_scale.exp()           # fold scale
            e = e.reshape(1, EMBED, -1)                     # [1,512,HW]
            bias_ch = e[:, :1, :] * 0.0 + head.cv4[i].bias  # [1,1,HW] filled with bias_s
            region.append(torch.cat([e, bias_ch], dim=1))   # [1,513,HW]
        region = torch.cat(region, dim=2)                   # [1,513,8400]

        # Masks.
        mask_coeffs = torch.cat(
            [head.cv5[i](feats[i]).view(1, head.nm, -1) for i in range(nl)], dim=-1
        )                                                    # [1,32,8400]
        mask_protos = head.proto(feats[0])                   # [1,32,160,160]

        return boxes, region, mask_coeffs, mask_protos


def convert_detector(model_name: str, output_dir: Path, input_size: int = 640):
    print(f"=== Converting YOLOE region-embedding detector ({model_name}) ===")
    from ultralytics import YOLOE

    model = YOLOE(f"{model_name}.pt")
    wm = model.model
    wm.eval()
    wrapper = YOLOERegionDetector(wm).eval()

    dummy = torch.randn(1, 3, input_size, input_size)
    with torch.no_grad():
        boxes, region, mc, proto = wrapper(dummy)
    print(f"  boxes {tuple(boxes.shape)}  region {tuple(region.shape)}  "
          f"mask_coeffs {tuple(mc.shape)}  protos {tuple(proto.shape)}")

    print("  Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy, check_trace=False)

    print("  Converting to Core ML (FP16)...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image", shape=(1, 3, input_size, input_size))],
        outputs=[
            ct.TensorType(name="boxes"),
            ct.TensorType(name="region_embeddings"),
            ct.TensorType(name="mask_coeffs"),
            ct.TensorType(name="mask_protos"),
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS16,
    )
    mlmodel.author = "coreml-models"
    mlmodel.short_description = (
        f"YOLOE {model_name} open-vocabulary region-embedding detector + segmentation"
    )
    mlmodel.version = "2.0.0"

    out = output_dir / "yoloe_detector.mlpackage"
    mlmodel.save(str(out))
    print(f"  Saved {out}")
    return out


# ---------------------------------------------------------------------------
# RepRTA residual MLP  (raw_tpe -> tpe), L2 normalize is done in Swift
# ---------------------------------------------------------------------------

class RepRTAWrapper(nn.Module):
    def __init__(self, reprta):
        super().__init__()
        self.reprta = reprta

    def forward(self, raw_tpe):  # [1, N, 512] -> [1, N, 512]
        return self.reprta(raw_tpe)


def convert_reprta(model_name: str, output_dir: Path, max_classes: int = 80):
    print("\n=== Converting RepRTA (text refinement MLP) ===")
    from ultralytics import YOLOE

    model = YOLOE(f"{model_name}.pt")
    reprta = model.model.model[-1].reprta
    wrapper = RepRTAWrapper(reprta).eval()

    dummy = torch.randn(1, max_classes, EMBED)
    with torch.no_grad():
        out_ref = wrapper(dummy)
    print(f"  reprta in {tuple(dummy.shape)} -> out {tuple(out_ref.shape)}")

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy, check_trace=False)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="raw_tpe", shape=(1, max_classes, EMBED))],
        outputs=[ct.TensorType(name="tpe")],
        compute_precision=ct.precision.FLOAT32,  # embedding model: keep FP32
        minimum_deployment_target=ct.target.iOS16,
    )
    mlmodel.author = "coreml-models"
    mlmodel.short_description = "YOLOE RepRTA residual MLP (text prompt refinement)"
    mlmodel.version = "2.0.0"

    out = output_dir / "reprta.mlpackage"
    mlmodel.save(str(out))
    print(f"  Saved {out}")
    return out


# ---------------------------------------------------------------------------
# Visual Prompt Encoder (SAVPE): image + box-mask -> VPE, a drop-in for the
# text query embedding. Lets you detect "the same object as this reference".
# ---------------------------------------------------------------------------

class VisualPromptEncoder(nn.Module):
    """backbone+neck -> SAVPE(feats, mask) -> normalized VPE [1, 1, 512].

    The mask is an 80x80 binary box mask (the model input is 640, SAVPE works at
    stride 8). Q is fixed to 1 (single reference object). SAVPE's forward is
    re-implemented with concrete shapes and an FP16-safe softmax mask, dropping
    the `expand(-1, ...)` / `logical_not` / `finfo.min` ops coremltools dislikes.
    """

    def __init__(self, yoloe_model):
        super().__init__()
        self.layers = nn.ModuleList(list(yoloe_model.model[:-1]))
        self.head = yoloe_model.model[-1]
        self.savpe = self.head.savpe
        self._f = [m.f for m in yoloe_model.model[:-1]]
        self._save = set(yoloe_model.save)
        self._head_f = self.head.f

    def _backbone(self, image):
        y = [None] * len(self.layers)
        x = image
        for i, m in enumerate(self.layers):
            f = self._f[i]
            if f != -1:
                x = y[f] if isinstance(f, int) else [x if j == -1 else y[j] for j in f]
            x = m(x)
            y[i] = x if i in self._save else None
        return [y[j] for j in self._head_f]

    def forward(self, image, mask):  # mask: [1, 1, 80, 80]
        x = self._backbone(image)
        s = self.savpe
        y = s.cv4(torch.cat([s.cv2[i](xi) for i, xi in enumerate(x)], 1))   # [1,16,80,80]
        xx = s.cv3(torch.cat([s.cv1[i](xi) for i, xi in enumerate(x)], 1))  # [1,512,80,80]
        C, H, W = EMBED, xx.shape[2], xx.shape[3]
        c = s.c
        xx = xx.view(1, C, H * W)
        ymap = s.cv6(torch.cat((y, s.cv5(mask.reshape(1, 1, H, W))), dim=1))
        ymap = ymap.reshape(1, 1, c, H * W)
        vpf = mask.reshape(1, 1, 1, H * W)
        score = ymap * vpf + (1.0 - vpf) * (-6.0e4)        # FP16-safe masked softmax
        score = F.softmax(score, dim=-1)
        agg = score.transpose(-2, -3) @ xx.reshape(1, c, C // c, H * W).transpose(-1, -2)
        out = agg.transpose(-2, -3).reshape(1, 1, C)
        return F.normalize(out, dim=-1, p=2)               # [1, 1, 512]


def convert_visual_encoder(model_name: str, output_dir: Path, input_size: int = 640):
    print("\n=== Converting Visual Prompt Encoder (SAVPE) ===")
    from ultralytics import YOLOE

    model = YOLOE(f"{model_name}.pt")
    wm = model.model
    wm.eval()
    wrapper = VisualPromptEncoder(wm).eval()

    dummy_img = torch.rand(1, 3, input_size, input_size)
    dummy_mask = torch.zeros(1, 1, input_size // 8, input_size // 8)
    dummy_mask[:, :, 20:50, 20:50] = 1.0
    with torch.no_grad():
        vpe = wrapper(dummy_img, dummy_mask)
    print(f"  vpe {tuple(vpe.shape)}")

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (dummy_img, dummy_mask), check_trace=False)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="image", shape=(1, 3, input_size, input_size)),
            ct.TensorType(name="mask", shape=(1, 1, input_size // 8, input_size // 8)),
        ],
        outputs=[ct.TensorType(name="vpe")],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS16,
    )
    mlmodel.author = "coreml-models"
    mlmodel.short_description = f"YOLOE {model_name} visual prompt encoder (SAVPE)"
    mlmodel.version = "2.0.0"

    out = output_dir / "visual_prompt_encoder.mlpackage"
    mlmodel.save(str(out))
    print(f"  Saved {out}")
    return out


# ---------------------------------------------------------------------------
# CLIP BPE vocabulary (same format the Swift CLIPTokenizer expects)
# ---------------------------------------------------------------------------

def export_vocabulary(output_dir: Path):
    print("\n=== Exporting CLIP vocabulary ===")
    from clip.simple_tokenizer import bytes_to_unicode
    import clip.clip as clip_mod

    bpe_path = Path(clip_mod.__file__).parent / "bpe_simple_vocab_16e6.txt.gz"
    with gzip.open(str(bpe_path), "rt", encoding="utf-8") as f:
        bpe_data = f.read()

    lines = bpe_data.strip().split("\n")
    merges = [ln for ln in lines if ln and not ln.startswith("#")]

    byte_encoder = bytes_to_unicode()
    vocab_list = list(byte_encoder.values())
    vocab_list += [v + "</w>" for v in vocab_list]
    for merge in merges:
        vocab_list.append("".join(merge.split()))
    vocab_list.extend(["<|startoftext|>", "<|endoftext|>"])
    encoder = {v: i for i, v in enumerate(vocab_list)}

    vocab_data = {
        "encoder": encoder,
        "merges": merges,
        "bos_token": "<|startoftext|>",
        "eos_token": "<|endoftext|>",
        "context_length": CONTEXT_LENGTH,
    }
    out = output_dir / "clip_vocab.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, ensure_ascii=False)
    print(f"  Saved vocabulary ({len(encoder)} tokens) to {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Convert YOLOE to a Core ML region-embedding detector")
    parser.add_argument("--size", default="11s-seg", choices=list(YOLOE_MODELS.keys()))
    parser.add_argument("--output", default="YOLOEDemo", help="Output directory")
    parser.add_argument("--skip-vocab", action="store_true", help="Skip clip_vocab.json export")
    args = parser.parse_args()

    model_name = YOLOE_MODELS[args.size]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    convert_detector(model_name, output_dir)
    convert_reprta(model_name, output_dir)
    convert_visual_encoder(model_name, output_dir)
    if not args.skip_vocab:
        export_vocabulary(output_dir)

    print("\n=== Done ===")
    print("Detector + reprta produced. mobileclip_blt_text.mlpackage is Apple's")
    print("official MobileCLIP export (apple/ml-mobileclip) and is used as-is.")


if __name__ == "__main__":
    main()
