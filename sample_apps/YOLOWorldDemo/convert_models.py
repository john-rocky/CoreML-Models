#!/usr/bin/env python3
"""
Convert YOLO-World + CLIP models to Core ML for open-vocabulary object detection.

Produces three outputs:
  1. yoloworld_detector.mlpackage - Visual detector (boxes + scores outputs)
  2. clip_text_encoder.mlpackage  - CLIP text encoder (any text -> embeddings)
  3. clip_vocab.json              - BPE vocabulary for Swift tokenizer

Architecture:
  The CoreML detector includes the full BNContrastiveHead scoring pipeline
  internally, so output scores are already sigmoid-calibrated confidence values.
  The model outputs separate "boxes" [1,4,8400] and "scores" [1,NC,8400] tensors.

Usage:
    pip install ultralytics open_clip_torch coremltools
    python convert_models.py
    python convert_models.py --size s
"""

import argparse
import json
import gzip
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import coremltools as ct


MAX_CLASSES = 80
CONTEXT_LENGTH = 77

YOLO_WORLD_MODELS = {
    "s": "yolov8s-worldv2",
    "m": "yolov8m-worldv2",
    "l": "yolov8l-worldv2",
    "x": "yolov8x-worldv2",
}


# ---------------------------------------------------------------------------
# YOLO-World Visual Detector
# ---------------------------------------------------------------------------

class YOLOWorldDetectorWrapper(nn.Module):
    """Wraps YOLO-World to output boxes + sigmoid-calibrated scores."""

    def __init__(self, world_model):
        super().__init__()
        self.model = world_model

    def forward(self, image, txt_feats):
        self.model.txt_feats = txt_feats
        out = self.model(image)
        pred = out[0]              # [1, 4+NC, 8400]
        boxes = pred[:, :4, :]     # [1, 4, 8400]
        scores = pred[:, 4:, :]    # [1, NC, 8400]
        return boxes, scores


def convert_detector(model_name: str, output_dir: Path, input_size: int = 640):
    print(f"=== Converting Visual Detector ({model_name}) ===")

    from ultralytics import YOLO
    model = YOLO(model_name)
    wm = model.model
    wm.eval()

    wrapper = YOLOWorldDetectorWrapper(wm)
    wrapper.eval()

    dummy_img = torch.randn(1, 3, input_size, input_size)
    dummy_txt = torch.randn(1, MAX_CLASSES, 512)

    with torch.no_grad():
        _ = wrapper(dummy_img, dummy_txt)
        _ = wrapper(dummy_img, dummy_txt)

    print("  Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (dummy_img, dummy_txt), check_trace=False)

    print("  Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="image", shape=(1, 3, input_size, input_size)),
            ct.TensorType(name="txt_feats", shape=(1, MAX_CLASSES, 512)),
        ],
        outputs=[
            ct.TensorType(name="boxes"),
            ct.TensorType(name="scores"),
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS15,
    )

    mlmodel.author = "coreml-models"
    mlmodel.short_description = f"{model_name} Visual Detector (open-vocabulary)"
    mlmodel.version = "1.0.0"

    out_path = output_dir / "yoloworld_detector.mlpackage"
    mlmodel.save(str(out_path))
    print(f"  Saved to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLIP Text Encoder
# ---------------------------------------------------------------------------

class CLIPTextEncoderWrapper(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def forward(self, text_tokens):
        x = self.token_embedding(text_tokens)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)]
        x = x @ self.text_projection
        return x


def convert_text_encoder(output_dir: Path):
    print("\n=== Converting CLIP Text Encoder ===")

    import open_clip

    # Patch MultiheadAttention for CoreML compatibility
    original_mha = torch.nn.MultiheadAttention.forward
    def patched_mha(self, query, key, value, key_padding_mask=None,
                    need_weights=True, attn_mask=None,
                    average_attn_weights=True, is_causal=False):
        return F.multi_head_attention_forward(
            query, key, value,
            self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
        )
    torch.nn.MultiheadAttention.forward = patched_mha

    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    clip_model.eval()

    wrapper = CLIPTextEncoderWrapper(clip_model)
    wrapper.eval()

    dummy_tokens = open_clip.tokenize(["placeholder"] * MAX_CLASSES)

    with torch.no_grad():
        _ = wrapper(dummy_tokens)

    print("  Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy_tokens, check_trace=False)

    print("  Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="text_tokens",
                shape=(MAX_CLASSES, CONTEXT_LENGTH),
                dtype=np.int32,
            )
        ],
        outputs=[
            ct.TensorType(name="text_embeddings"),
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS15,
    )

    mlmodel.author = "coreml-models"
    mlmodel.short_description = "CLIP ViT-B/32 Text Encoder"
    mlmodel.version = "1.0.0"

    out_path = output_dir / "clip_text_encoder.mlpackage"
    mlmodel.save(str(out_path))
    print(f"  Saved to {out_path}")

    torch.nn.MultiheadAttention.forward = original_mha
    return out_path


# ---------------------------------------------------------------------------
# CLIP Vocabulary
# ---------------------------------------------------------------------------

def export_vocabulary(output_dir: Path):
    print("\n=== Exporting CLIP Vocabulary ===")

    from open_clip import tokenizer as clip_tok

    bpe_path = Path(clip_tok.__file__).parent / "bpe_simple_vocab_16e6.txt.gz"
    with gzip.open(str(bpe_path), "rt", encoding="utf-8") as f:
        bpe_data = f.read()

    lines = bpe_data.strip().split("\n")
    merges = [l for l in lines if l and not l.startswith("#")]

    byte_encoder = clip_tok.bytes_to_unicode()
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

    out_path = output_dir / "clip_vocab.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, ensure_ascii=False)

    print(f"  Saved vocabulary ({len(encoder)} tokens) to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLO-World + CLIP to CoreML"
    )
    parser.add_argument(
        "--size", type=str, default="s", choices=["s", "m", "l", "x"],
        help="Model size: s(mall), m(edium), l(arge), x(tra-large) (default: s)",
    )
    parser.add_argument(
        "--output", type=str, default="YOLOWorldDemo",
        help="Output directory (default: YOLOWorldDemo)",
    )
    args = parser.parse_args()

    model_name = YOLO_WORLD_MODELS[args.size]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    convert_detector(model_name, output_dir)
    convert_text_encoder(output_dir)
    export_vocabulary(output_dir)

    print(f"\n=== Done ===")
    print(f"Add the .mlpackage files and clip_vocab.json to your Xcode project.")


if __name__ == "__main__":
    main()
